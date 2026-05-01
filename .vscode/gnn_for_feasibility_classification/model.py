# model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool

# axis order (너희 코드와 동일하게 유지)
AXES: List[str] = ["latency", "jitter", "loss", "bandwidth", "tsn", "frer"]


# -------------------------
# small utils
# -------------------------
def _as_2d_q(q: torch.Tensor) -> torch.Tensor:
    """
    q: (B, 6) or (B, 1, 6) -> (B, 6)
    """
    if q.dim() == 3:
        return q.squeeze(1)
    return q


def _ensure_1d_graph_feat(x: Any, batch_num_graphs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    path_hops / loss_budget_e2e 같은 그래프 단위 scalar를 (B,1) 텐서로 정규화.
    - scalar / python number / shape=(1,) / shape=(B,) 모두 처리
    """
    if x is None:
        return torch.zeros((batch_num_graphs, 1), device=device, dtype=dtype)

    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.numel() == 1:
            return t.view(1, 1).repeat(batch_num_graphs, 1)
        if t.dim() == 0:
            return t.view(1, 1).repeat(batch_num_graphs, 1)
        if t.dim() == 1:
            # (B,)
            if t.size(0) == batch_num_graphs:
                return t.view(batch_num_graphs, 1)
            # (1,) 등
            if t.size(0) == 1:
                return t.view(1, 1).repeat(batch_num_graphs, 1)
        if t.dim() == 2 and t.size(1) == 1 and t.size(0) == batch_num_graphs:
            return t
        # 그 외는 최대한 flatten -> (B,1) 시도
        t = t.view(-1)
        if t.numel() == batch_num_graphs:
            return t.view(batch_num_graphs, 1)
        if t.numel() == 1:
            return t.view(1, 1).repeat(batch_num_graphs, 1)
        raise ValueError(f"graph scalar has unexpected shape: {tuple(x.shape)}")

    # python scalar
    v = float(x)
    return torch.full((batch_num_graphs, 1), v, device=device, dtype=dtype)


def _edge_batch_from_node_batch(edge_index: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
    """
    PyG Batch에는 edge_batch가 기본 제공되지 않으므로,
    edge의 src 노드가 속한 그래프 id를 edge_batch로 사용.
    (intra-graph edge만 있다는 가정 하에 안전)
    """
    src = edge_index[0]
    return node_batch[src]


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    torch_scatter 없이도 동작하는 scatter_add 대체.
    src: (E, D) or (E,)
    index: (E,)
    return: (dim_size, D) or (dim_size,)
    """
    if src.dim() == 1:
        out = torch.zeros((dim_size,), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out
    else:
        out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# B안: 경로 힌트 없이 (NoPathStats)
# -------------------------
class Layer2GINENoPathStats(nn.Module):
    """
    ✅ 입력: x, edge_index, edge_attr, q, tclass, batch
    ❌ 사용 안 함: path_edge_mask, path_hops, loss_budget_e2e

    출력:
      y_logit: (B,)
      axis_logit: (B, 6)
    """

    def __init__(
        self,
        x_dim: int,
        edge_dim: int,
        q_dim: int = 6,
        hidden: int = 64,
        num_layers: int = 3,
        tclass_vocab: int = 8,
        t_emb: int = 16,
        drop: float = 0.1,
    ):
        super().__init__()

        self.t_emb = nn.Embedding(tclass_vocab, t_emb)

        self.q_mlp = nn.Sequential(
            nn.Linear(q_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.x_in = nn.Linear(x_dim, hidden)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn_edge, edge_dim=edge_dim))

        fused = hidden + hidden + t_emb
        self.y_head = MLP(fused, hidden, 1, drop=drop)
        self.axis_head = MLP(fused, hidden, len(AXES), drop=drop)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        q = _as_2d_q(data.q)
        tclass = data.tclass.view(-1)

        x = self.x_in(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        g = global_mean_pool(x, batch)   # (B, hidden)
        qh = self.q_mlp(q)               # (B, hidden)
        th = self.t_emb(tclass)          # (B, t_emb)

        z = torch.cat([g, qh, th], dim=-1)
        y_logit = self.y_head(z).view(-1)         # (B,)
        axis_logit = self.axis_head(z)            # (B, 6)
        return y_logit, axis_logit


# -------------------------
# A안: 경로 힌트 사용 (PlusPathStats)
# -------------------------
class Layer2GINEPlusPathStats(nn.Module):
    """
    ✅ 입력: x, edge_index, edge_attr, q, tclass, batch
    ✅ 추가 입력(경로 힌트):
       - path_edge_mask (E,) bool : "해당 flow path에 포함된 edge 마스크"
       - path_hops (B,) or scalar : hop 수
       - loss_budget_e2e (B,) or scalar : e2e 손실 예산 (실험 정의에 따라)
       - num_graphs : Batch.num_graphs 참조(집계 dim_size용)

    출력:
      y_logit: (B,)
      axis_logit: (B, 6)
    """

    def __init__(
        self,
        x_dim: int,
        edge_dim: int,
        q_dim: int = 6,
        hidden: int = 64,
        num_layers: int = 3,
        tclass_vocab: int = 8,
        t_emb: int = 16,
        drop: float = 0.1,
        # path stats 옵션
        use_path_mean: bool = True,
        use_path_count: bool = True,
    ):
        super().__init__()

        self.edge_dim = edge_dim
        self.use_path_mean = use_path_mean
        self.use_path_count = use_path_count

        self.t_emb = nn.Embedding(tclass_vocab, t_emb)

        self.q_mlp = nn.Sequential(
            nn.Linear(q_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.x_in = nn.Linear(x_dim, hidden)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn_edge, edge_dim=edge_dim))

        # path_stats 입력 차원 계산
        path_in = 0
        if self.use_path_mean:
            path_in += edge_dim
        if self.use_path_count:
            path_in += 1
        # path_hops + loss_budget_e2e는 항상 1씩
        path_in += 1  # path_hops
        path_in += 1  # loss_budget_e2e

        self.path_mlp = nn.Sequential(
            nn.Linear(path_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        fused = hidden + hidden + t_emb + hidden
        self.y_head = MLP(fused, hidden, 1, drop=drop)
        self.axis_head = MLP(fused, hidden, len(AXES), drop=drop)

    def _compute_path_stats(self, data) -> torch.Tensor:
        """
        path_edge_mask로 edge_attr를 그래프별 집계해서 path_stats 생성.
        return: (B, path_in)
        """
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        node_batch = data.batch

        B = int(data.num_graphs)  # detect_model_forward_fields에 잡히는 필드
        device = edge_attr.device
        dtype = edge_attr.dtype

        # edge_batch (E,)
        edge_batch = _edge_batch_from_node_batch(edge_index, node_batch)

        # path_edge_mask: (E,)
        pem = data.path_edge_mask
        if pem is None:
            # 학습 데이터에는 있다고 했으니 원래는 에러가 맞음.
            # 다만 안전장치로 0으로 두고 진행.
            pem = torch.zeros((edge_attr.size(0),), device=device, dtype=torch.bool)
        else:
            pem = pem.to(device=device)

        pem_f = pem.float().view(-1, 1)  # (E,1)

        feats: List[torch.Tensor] = []

        if self.use_path_mean:
            # sum over path edges
            sum_path = _scatter_add(edge_attr * pem_f, edge_batch, dim_size=B)  # (B, edge_dim)
            cnt = _scatter_add(pem.float(), edge_batch, dim_size=B).clamp_min(1.0).view(B, 1)  # (B,1)
            mean_path = sum_path / cnt
            feats.append(mean_path)

        if self.use_path_count:
            cnt2 = _scatter_add(pem.float(), edge_batch, dim_size=B).view(B, 1)
            feats.append(cnt2)

        # path_hops, loss_budget_e2e
        ph = _ensure_1d_graph_feat(getattr(data, "path_hops", None), B, device, dtype)
        lb = _ensure_1d_graph_feat(getattr(data, "loss_budget_e2e", None), B, device, dtype)

        feats.append(ph)
        feats.append(lb)

        return torch.cat(feats, dim=-1)  # (B, path_in)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        q = _as_2d_q(data.q)
        tclass = data.tclass.view(-1)

        # GNN encode
        x = self.x_in(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        g = global_mean_pool(x, batch)   # (B, hidden)
        qh = self.q_mlp(q)               # (B, hidden)
        th = self.t_emb(tclass)          # (B, t_emb)

        # path stats
        ps = self._compute_path_stats(data)   # (B, path_in)
        psh = self.path_mlp(ps)               # (B, hidden)

        z = torch.cat([g, qh, th, psh], dim=-1)
        y_logit = self.y_head(z).view(-1)      # (B,)
        axis_logit = self.axis_head(z)         # (B, 6)
        return y_logit, axis_logit


# -------------------------
# Factory
# -------------------------
def build_layer2_model(
    arch: str,
    x_dim: int,
    edge_dim: int,
    q_dim: int = 6,
    **kwargs,
) -> nn.Module:
    """
    너희 ckpt 로더 / eval 코드가 요구하는 인터페이스:
      build_layer2_model(arch=..., x_dim=..., edge_dim=..., q_dim=..., **model_kwargs)
    """
    arch = str(arch)

    # alias 허용
    if arch in ["plusstats", "Layer2GINEPlusPathStats", "Layer2GINE_PlusPathStats"]:
        return Layer2GINEPlusPathStats(x_dim=x_dim, edge_dim=edge_dim, q_dim=q_dim, **kwargs)

    if arch in ["nopathtats", "no_path_stats", "Layer2GINENoPathStats", "Layer2GINE_NoPathStats"]:
        return Layer2GINENoPathStats(x_dim=x_dim, edge_dim=edge_dim, q_dim=q_dim, **kwargs)

    raise ValueError(
        f"Unknown arch='{arch}'. "
        f"Supported: Layer2GINEPlusPathStats / Layer2GINENoPathStats (aliases: plusstats / no_path_stats)"
    )