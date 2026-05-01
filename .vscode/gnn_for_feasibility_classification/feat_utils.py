# feat_utils.py
import torch

# edge_attr = [bw, lat, jit, loss, queue, interf, tsn, frer]
BW, LAT, JIT, LOSS, QUEUE, INTERF, TSN, FRER = range(8)
EPS = 1e-9


def safe_log10(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return torch.log10(torch.clamp(x, min=eps))


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / torch.clamp(std, min=1e-6)


# -------------------------
# torch-only scatter helpers
# -------------------------
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    src: [E] or [E,C]
    index: [E] (0..dim_size-1)
    out: [dim_size] or [dim_size,C]
    """
    if src.dim() == 1:
        out = src.new_zeros((dim_size,))
        out.index_add_(0, index, src)
        return out
    else:
        out = src.new_zeros((dim_size, src.size(1)))
        out.index_add_(0, index, src)
        return out


def scatter_amin(src: torch.Tensor, index: torch.Tensor, dim_size: int, fill_value: float = 1e9) -> torch.Tensor:
    """
    src: [E] or [E,C]
    index: [E]
    out: [dim_size] or [dim_size,C]
    - torch >= 1.12면 scatter_reduce_를 사용(빠름)
    - 아니면 파이썬 루프 fallback(느리지만 동작)
    """
    if src.dim() == 1:
        out = src.new_full((dim_size,), fill_value)
        if hasattr(out, "scatter_reduce_"):
            out.scatter_reduce_(0, index, src, reduce="amin", include_self=True)
        else:
            for i in range(src.size(0)):
                j = int(index[i].item())
                out[j] = torch.minimum(out[j], src[i])
        return out

    # 2D
    out = src.new_full((dim_size, src.size(1)), fill_value)
    if hasattr(out, "scatter_reduce_"):
        idx = index.view(-1, 1).expand(-1, src.size(1))
        out.scatter_reduce_(0, idx, src, reduce="amin", include_self=True)
    else:
        for i in range(src.size(0)):
            j = int(index[i].item())
            out[j] = torch.minimum(out[j], src[i])
    return out


# -------------------------
# edge transform (for GNN)
# -------------------------
def edge_cont_transform(raw_edge_attr: torch.Tensor) -> torch.Tensor:
    """
    raw_edge_attr: [E,8]
    return: [E,6] (bw,lat,jit,loss -> log10, queue/interf -> 그대로)
    """
    bw = safe_log10(raw_edge_attr[:, BW])
    lat = safe_log10(torch.clamp(raw_edge_attr[:, LAT], min=EPS))
    jit = safe_log10(torch.clamp(raw_edge_attr[:, JIT], min=EPS))

    loss = torch.clamp(raw_edge_attr[:, LOSS], min=EPS, max=1.0 - 1e-6)
    loss = safe_log10(loss)

    queue = raw_edge_attr[:, QUEUE]
    interf = raw_edge_attr[:, INTERF]
    return torch.stack([bw, lat, jit, loss, queue, interf], dim=-1)


def make_edge_attr_for_gnn(raw_edge_attr: torch.Tensor,
                           edge_mean: torch.Tensor,
                           edge_std: torch.Tensor) -> torch.Tensor:
    """
    return: [E,8] (cont 6개는 정규화, tsn/frer은 그대로)
    """
    cont = edge_cont_transform(raw_edge_attr)
    cont_n = normalize(cont, edge_mean, edge_std)
    tsn = raw_edge_attr[:, TSN:TSN+1]
    frer = raw_edge_attr[:, FRER:FRER+1]
    return torch.cat([cont_n, tsn, frer], dim=-1)


# -------------------------
# q transform (for model)
# -------------------------
def q_cont_transform(q_raw: torch.Tensor, loss_budget_e2e: torch.Tensor) -> torch.Tensor:
    """
    q_raw: [B,6] = [lat_ms, jit_ms, loss_per_link(or e2e), bw_mbps, pri, basis]
    loss_budget_e2e: [B,1] or [B]
    return: [B,4] (lat,jit,loss_budget,bw -> log10)
    """
    q_lat = safe_log10(torch.clamp(q_raw[:, 0], min=EPS))
    q_jit = safe_log10(torch.clamp(q_raw[:, 1], min=EPS))

    lb = loss_budget_e2e.view(-1)
    lb = torch.clamp(lb, min=EPS, max=1.0 - 1e-6)
    q_loss = safe_log10(lb)

    q_bw = safe_log10(torch.clamp(q_raw[:, 3], min=EPS))
    return torch.stack([q_lat, q_jit, q_loss, q_bw], dim=-1)


def make_q_for_model(q_raw: torch.Tensor,
                     loss_budget_e2e: torch.Tensor,
                     q_mean: torch.Tensor,
                     q_std: torch.Tensor) -> torch.Tensor:
    """
    return: [B,6] (cont 4개 정규화 + pri/basis 그대로)
    """
    cont = q_cont_transform(q_raw, loss_budget_e2e)
    cont_n = normalize(cont, q_mean, q_std)
    pri_basis = q_raw[:, 4:6].float()
    return torch.cat([cont_n, pri_basis], dim=-1)


# -------------------------
# path stats from mask
# -------------------------
@torch.no_grad()
def compute_path_stats(raw_edge_attr: torch.Tensor,
                       edge_index: torch.Tensor,
                       batch: torch.Tensor,
                       path_edge_mask: torch.Tensor,
                       num_graphs: int):
    """
    raw_edge_attr: [E,8]  (Batch에서 합쳐진 edge)
    edge_index: [2,E]
    batch: [N] node->graph id
    path_edge_mask: [E] bool (directed path edges)

    return:
      stats_raw: [B,8] = [lat_sum, jit_sum, loss_e2e, bw_min, queue_mean, interf_mean, tsn_all, frer_all]
      hops_cnt: [B,1]  = number of directed edges on path
    """
    mask = path_edge_mask.view(-1).bool()
    u = edge_index[0]
    edge_batch = batch[u]  # [E]

    ones = mask.float()
    cnt = scatter_sum(ones, edge_batch, dim_size=num_graphs).clamp_min(1.0)          # [B]
    hops_cnt = scatter_sum(ones, edge_batch, dim_size=num_graphs)                    # [B]

    # sums
    lat_sum = scatter_sum(raw_edge_attr[:, LAT] * ones, edge_batch, dim_size=num_graphs)
    jit_sum = scatter_sum(raw_edge_attr[:, JIT] * ones, edge_batch, dim_size=num_graphs)

    # bw min (mask 아닌 곳은 +inf로)
    bw = raw_edge_attr[:, BW]
    big = torch.full_like(bw, 1e9)
    bw_masked = torch.where(mask, bw, big)
    bw_min = scatter_amin(bw_masked, edge_batch, dim_size=num_graphs, fill_value=1e9)
    bw_min = torch.where(bw_min > 1e8, torch.zeros_like(bw_min), bw_min)

    # loss_e2e: 1 - Π(1-loss) = 1 - exp(sum(log(1-loss)))
    loss = torch.clamp(raw_edge_attr[:, LOSS], min=0.0, max=1.0 - 1e-6)
    log_success = torch.log1p(-loss)  # log(1-loss)
    log_success_sum = scatter_sum(log_success * ones, edge_batch, dim_size=num_graphs)
    success = torch.exp(log_success_sum)
    loss_e2e = 1.0 - success

    # queue/interf mean
    queue_sum = scatter_sum(raw_edge_attr[:, QUEUE] * ones, edge_batch, dim_size=num_graphs)
    interf_sum = scatter_sum(raw_edge_attr[:, INTERF] * ones, edge_batch, dim_size=num_graphs)
    queue_mean = queue_sum / cnt
    interf_mean = interf_sum / cnt

    # tsn/frer all: min over path (mask 아닌 곳은 1로 두면 AND처럼 됨)
    tsn = raw_edge_attr[:, TSN]
    frer = raw_edge_attr[:, FRER]
    ones01 = torch.ones_like(tsn)
    tsn_masked = torch.where(mask, tsn, ones01)
    frer_masked = torch.where(mask, frer, ones01)
    tsn_all = scatter_amin(tsn_masked, edge_batch, dim_size=num_graphs, fill_value=1.0)
    frer_all = scatter_amin(frer_masked, edge_batch, dim_size=num_graphs, fill_value=1.0)

    stats_raw = torch.stack(
        [lat_sum, jit_sum, loss_e2e, bw_min, queue_mean, interf_mean, tsn_all, frer_all],
        dim=-1
    )
    return stats_raw, hops_cnt.view(-1, 1)


def path_cont_transform(stats_raw: torch.Tensor) -> torch.Tensor:
    """
    stats_raw: [B,8] (lat_sum, jit_sum, loss_e2e, bw_min, queue_mean, interf_mean, tsn_all, frer_all)
    return: [B,6] (lat_sum,jit_sum,loss_e2e,bw_min -> log10, queue/interf -> 그대로)
    """
    lat = safe_log10(torch.clamp(stats_raw[:, 0], min=EPS))
    jit = safe_log10(torch.clamp(stats_raw[:, 1], min=EPS))
    loss = safe_log10(torch.clamp(stats_raw[:, 2], min=EPS, max=1.0 - 1e-6))
    bw = safe_log10(torch.clamp(stats_raw[:, 3], min=EPS))
    queue = stats_raw[:, 4]
    interf = stats_raw[:, 5]
    return torch.stack([lat, jit, loss, bw, queue, interf], dim=-1)


def make_ratios(q_raw: torch.Tensor, loss_budget_e2e: torch.Tensor, stats_raw: torch.Tensor) -> torch.Tensor:
    """
    ratio features (log scale):
      lat_sum / q_lat
      jit_sum / q_jit
      loss_e2e / loss_budget_e2e
      q_bw / bw_min
    return: [B,4]
    """
    q_lat = torch.clamp(q_raw[:, 0], min=EPS)
    q_jit = torch.clamp(q_raw[:, 1], min=EPS)
    q_bw = torch.clamp(q_raw[:, 3], min=EPS)
    lb = torch.clamp(loss_budget_e2e.view(-1), min=EPS, max=1.0)

    lat_sum = torch.clamp(stats_raw[:, 0], min=EPS)
    jit_sum = torch.clamp(stats_raw[:, 1], min=EPS)
    loss_e2e = torch.clamp(stats_raw[:, 2], min=EPS, max=1.0)
    bw_min = torch.clamp(stats_raw[:, 3], min=EPS)

    r_lat = safe_log10(lat_sum / q_lat)
    r_jit = safe_log10(jit_sum / q_jit)
    r_loss = safe_log10(loss_e2e / lb)
    r_bw = safe_log10(q_bw / bw_min)
    return torch.stack([r_lat, r_jit, r_loss, r_bw], dim=-1)


def require_flags_from_q_tclass(q_raw: torch.Tensor, tclass: torch.Tensor):
    """
    require_tsn: tclass in (0,1)
    require_frer: priority == critical(=1.0 근처)
    """
    tc = tclass.view(-1).long()
    require_tsn = ((tc == 0) | (tc == 1)).float().view(-1, 1)

    pri = q_raw[:, 4]
    require_frer = (pri > 0.90).float().view(-1, 1)
    return require_tsn, require_frer