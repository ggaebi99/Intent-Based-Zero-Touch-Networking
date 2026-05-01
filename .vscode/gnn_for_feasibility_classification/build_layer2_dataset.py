# build_layer2_dataset.py
# graph_pool.pt -> Layer2 학습 샘플(Data) 생성 + bottleneck axis + path_edge_mask
#
# 기본(기존 동작과 동일):
#   python build_layer2_dataset.py --graphs graph_pool.pt --out layer2_pairs_pl_with_bneck.pt --loss_mode per_link --k_per_graph 200
#
# src/dst 다양화(데이터 분포 바뀜, 성능 상한 올리고 싶을 때 권장):
#   python build_layer2_dataset.py --graphs graph_pool.pt --out layer2_pairs_pl_with_bneck_randep.pt --loss_mode per_link --k_per_graph 200 --random_endpoints

import argparse
import math
import random
from typing import Dict, List, Tuple, Optional

import torch
import networkx as nx
from torch_geometric.data import Data

from topology_templates import NODE_TYPES

BW, LAT, JIT, LOSS, QUEUE, INTERF, TSN, FRER = range(8)

AXES = ["latency", "jitter", "loss", "bandwidth", "tsn", "frer"]
AXIS_ID = {k: i for i, k in enumerate(AXES)}

BIG_BIN = 10.0
EPS = 1e-9

PRI_MAP = {"low": 0.0, "medium": 0.33, "high": 0.66, "critical": 1.0}
BASIS_MAP = {"target": 0.0, "soft": 0.0, "hard": 1.0}

TCLASS_ID = {
    "control_loop": 0,
    "safety_io": 1,
    "pubsub_telemetry": 2,
    "video_stream": 3,
    "best_effort_misc": 4,
}

TPL_T1 = "T1_cell_tsn"
TPL_T2 = "T2_hybrid_wifi"
TPL_T3 = "T3_zone_backbone"
TPL_T4 = "T4_multizone_edgecloud"

HOPS_EST = {TPL_T1: 2.0, TPL_T2: 3.0, TPL_T3: 4.0, TPL_T4: 5.0}


def e2e_to_perlink_loss(loss_e2e: float, hops_est: float) -> float:
    h = max(1.0, float(hops_est))
    loss_e2e = min(max(float(loss_e2e), 0.0), 1.0)
    return 1.0 - (1.0 - loss_e2e) ** (1.0 / h)

def perlink_to_e2e_budget(loss_perlink: float, hops: float) -> float:
    h = max(1.0, float(hops))
    p = min(max(float(loss_perlink), 0.0), 1.0)
    return 1.0 - (1.0 - p) ** h


def unique_undirected_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[Tuple[int, int], torch.Tensor]:
    edge_map: Dict[Tuple[int, int], torch.Tensor] = {}
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for i, (u, v) in enumerate(zip(src, dst)):
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in edge_map:
            edge_map[(a, b)] = edge_attr[i].detach()
    return edge_map

def build_sp_graph(edge_map: Dict[Tuple[int, int], torch.Tensor]) -> nx.Graph:
    G = nx.Graph()
    for (u, v), attr in edge_map.items():
        bw, lat, jit, loss, queue, interf, tsn, frer = attr.tolist()
        w = float(lat) + 2.0 * float(queue) + 3.0 * float(interf) + 50.0 * float(loss)
        G.add_edge(int(u), int(v), weight=w)
    return G

def aggregate_path_metrics(path_nodes: List[int], edge_map: Dict[Tuple[int, int], torch.Tensor]) -> Optional[Dict[str, float]]:
    if len(path_nodes) < 2:
        return None

    total_lat = 0.0
    total_jit = 0.0
    min_bw = float("inf")
    success = 1.0
    tsn_all = True
    frer_all = True

    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        u, v = (a, b) if a < b else (b, a)
        bw, lat, jit, loss, queue, interf, tsn, frer = edge_map[(u, v)].tolist()

        total_lat += float(lat)
        total_jit += float(jit)
        min_bw = min(min_bw, float(bw))
        success *= (1.0 - float(loss))
        tsn_all = tsn_all and (float(tsn) > 0.5)
        frer_all = frer_all and (float(frer) > 0.5)

    hops = float(len(path_nodes) - 1)
    return dict(
        latency=float(total_lat),
        jitter=float(total_jit),
        loss=float(1.0 - success),
        bandwidth=float(min_bw),
        hops=hops,
        tsn_all=1.0 if tsn_all else 0.0,
        frer_all=1.0 if frer_all else 0.0,
    )


def get_priority_str(d: Data) -> str:
    if hasattr(d, "q_dict") and isinstance(d.q_dict, dict) and "priority" in d.q_dict:
        return str(d.q_dict["priority"])
    try:
        v = float(d.q.view(-1)[4].item())
    except Exception:
        return "high"
    cand = [("low", 0.0), ("medium", 0.33), ("high", 0.66), ("critical", 1.0)]
    return min(cand, key=lambda kv: abs(v - kv[1]))[0]

def require_flags(d: Data) -> Tuple[bool, bool]:
    tc = int(d.tclass.view(-1)[0].item()) if hasattr(d, "tclass") else 0
    require_tsn = (tc in (0, 1))
    require_frer = (get_priority_str(d) == "critical")
    return require_tsn, require_frer


TPL_MIX = {
    TPL_T1: {
        "traffic_class": {
            "control_loop": 0.50, "safety_io": 0.30, "pubsub_telemetry": 0.10, "video_stream": 0.07, "best_effort_misc": 0.03
        },
        "priority": {
            "control_loop": {"critical": 0.70, "high": 0.30},
            "safety_io": {"critical": 0.80, "high": 0.20},
            "pubsub_telemetry": {"high": 0.80, "medium": 0.20},
            "video_stream": {"high": 0.90, "medium": 0.10},
            "best_effort_misc": {"medium": 0.70, "low": 0.30},
        },
        "basis": {"hard": 0.85, "target": 0.15},
    },
    TPL_T2: {
        "traffic_class": {
            "control_loop": 0.25, "safety_io": 0.10, "pubsub_telemetry": 0.25, "video_stream": 0.35, "best_effort_misc": 0.05
        },
        "priority": {
            "control_loop": {"critical": 0.50, "high": 0.50},
            "safety_io": {"critical": 0.60, "high": 0.40},
            "pubsub_telemetry": {"high": 0.85, "medium": 0.15},
            "video_stream": {"high": 0.92, "medium": 0.08},
            "best_effort_misc": {"medium": 0.70, "low": 0.30},
        },
        "basis": {"hard": 0.70, "target": 0.30},
    },
    TPL_T3: {
        "traffic_class": {
            "control_loop": 0.35, "safety_io": 0.15, "pubsub_telemetry": 0.20, "video_stream": 0.25, "best_effort_misc": 0.05
        },
        "priority": {
            "control_loop": {"critical": 0.60, "high": 0.40},
            "safety_io": {"critical": 0.70, "high": 0.30},
            "pubsub_telemetry": {"high": 0.80, "medium": 0.20},
            "video_stream": {"high": 0.90, "medium": 0.10},
            "best_effort_misc": {"medium": 0.70, "low": 0.30},
        },
        "basis": {"hard": 0.80, "target": 0.20},
    },
    TPL_T4: {
        "traffic_class": {
            "control_loop": 0.20, "safety_io": 0.10, "pubsub_telemetry": 0.25, "video_stream": 0.40, "best_effort_misc": 0.05
        },
        "priority": {
            "control_loop": {"critical": 0.55, "high": 0.45},
            "safety_io": {"critical": 0.65, "high": 0.35},
            "pubsub_telemetry": {"high": 0.85, "medium": 0.15},
            "video_stream": {"high": 0.92, "medium": 0.08},
            "best_effort_misc": {"medium": 0.70, "low": 0.30},
        },
        "basis": {"hard": 0.75, "target": 0.25},
    },
}

def _choices(rng: random.Random, items: List[str], weights: List[float]) -> str:
    return rng.choices(items, weights=weights, k=1)[0]

def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 1.0001)
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))

def q_to_tensor(q: Dict) -> torch.Tensor:
    return torch.tensor(
        [
            float(q["latency_ms"]),
            float(q["jitter_ms"]),
            float(q["loss_rate"]),
            float(q["bandwidth_mbps"]),
            PRI_MAP.get(q.get("priority", "high"), 0.66),
            BASIS_MAP.get(q.get("basis", "hard"), 1.0),
        ],
        dtype=torch.float,
    )

def tclass_id(q: Dict) -> int:
    return int(TCLASS_ID.get(q.get("traffic_class", "control_loop"), 0))

def clone_graph_only(g: Data) -> Data:
    d = Data(x=g.x.clone(), edge_index=g.edge_index.clone(), edge_attr=g.edge_attr.clone())
    for k in ["ntype", "tpl"]:
        if hasattr(g, k):
            v = getattr(g, k)
            setattr(d, k, v.clone() if torch.is_tensor(v) else v)
    return d


def _rand_pick(rng: random.Random, arr: List[int], fallback: int) -> int:
    if arr:
        return rng.choice(arr)
    return fallback

def pick_src_dst_from_tclass(ntype: torch.Tensor, tcid: int, rng: Optional[random.Random] = None, random_endpoints: bool = False) -> Tuple[int, int]:
    """
    random_endpoints=False: 기존처럼 첫 후보로 고정 (데이터 유지)
    random_endpoints=True : 후보들 중 랜덤 (데이터 분포 바뀜, 학습 상한↑)
    """
    types = [NODE_TYPES[int(i)] for i in ntype.view(-1).tolist()]
    robots = [i for i, t in enumerate(types) if t == "robot"]
    agvs = [i for i, t in enumerate(types) if t == "agv"]
    plcs = [i for i, t in enumerate(types) if t == "plc"]
    visions = [i for i, t in enumerate(types) if t == "vision"]
    aps = [i for i, t in enumerate(types) if t == "ap"]
    tsns = [i for i, t in enumerate(types) if t == "tsn"]

    # 기존 고정 동작
    if not random_endpoints or rng is None:
        if tcid in (0, 1):
            src_candidates = robots + agvs
            if src_candidates and plcs:
                return src_candidates[0], plcs[0]

        if tcid == 2:
            src_candidates = robots + agvs + plcs
            if src_candidates and tsns:
                core_candidates = [i for i in tsns if i in [0, 1]]
                dst = core_candidates[0] if core_candidates else tsns[0]
                return src_candidates[0], dst
            if src_candidates and aps:
                return src_candidates[0], aps[0]

        if tcid == 3:
            if visions and tsns:
                core_candidates = [i for i in tsns if i in [0, 1]]
                dst = core_candidates[0] if core_candidates else tsns[0]
                return visions[0], dst
            if visions and aps:
                return visions[0], aps[0]

        if tcid == 4:
            if aps and tsns:
                return aps[0], tsns[0]
            if len(types) >= 2:
                return 0, 1

        src = 0
        dst = len(types) - 1
        if src == dst and len(types) > 1:
            dst = 1
        return src, dst

    # 랜덤 동작
    if tcid in (0, 1):
        src_candidates = robots + agvs
        if src_candidates and plcs:
            return _rand_pick(rng, src_candidates, 0), _rand_pick(rng, plcs, 1)

    if tcid == 2:
        src_candidates = robots + agvs + plcs
        if src_candidates and tsns:
            core_candidates = [i for i in tsns if i in [0, 1]] or tsns
            return _rand_pick(rng, src_candidates, 0), _rand_pick(rng, core_candidates, 1)
        if src_candidates and aps:
            return _rand_pick(rng, src_candidates, 0), _rand_pick(rng, aps, 1)

    if tcid == 3:
        if visions and tsns:
            core_candidates = [i for i in tsns if i in [0, 1]] or tsns
            return _rand_pick(rng, visions, 0), _rand_pick(rng, core_candidates, 1)
        if visions and aps:
            return _rand_pick(rng, visions, 0), _rand_pick(rng, aps, 1)

    if tcid == 4:
        if aps and tsns:
            return _rand_pick(rng, aps, 0), _rand_pick(rng, tsns, 1)
        if len(types) >= 2:
            return 0, 1

    src = 0
    dst = len(types) - 1
    if src == dst and len(types) > 1:
        dst = 1
    return src, dst


def sample_q_for_tpl(rng: random.Random, tpl_name: str, loss_mode: str = "per_link") -> Dict:
    mix = TPL_MIX.get(tpl_name, TPL_MIX[TPL_T3])

    tc_items = list(mix["traffic_class"].keys())
    tc_w = [mix["traffic_class"][k] for k in tc_items]
    tclass = _choices(rng, tc_items, tc_w)

    pr_items = list(mix["priority"][tclass].keys())
    pr_w = [mix["priority"][tclass][k] for k in pr_items]
    priority = _choices(rng, pr_items, pr_w)

    if priority == "critical":
        basis = "hard"
    else:
        b_items = list(mix["basis"].keys())
        b_w = [mix["basis"][k] for k in b_items]
        basis = _choices(rng, b_items, b_w)
        if tclass == "best_effort_misc" and basis == "hard":
            basis = "target"

    scale = {TPL_T1: 1.0, TPL_T2: 1.2, TPL_T3: 1.5, TPL_T4: 2.0}.get(tpl_name, 1.5)
    hops_est = HOPS_EST.get(tpl_name, 4.0)

    if tclass == "control_loop":
        latency_ms = rng.uniform(0.5, 5.0) * scale
        jitter_ms = rng.uniform(0.1, 1.0) * scale
        loss_e2e = _log_uniform(rng, 1e-7, 1e-4)
        bandwidth_mbps = _log_uniform(rng, 0.2, 5.0)
    elif tclass == "safety_io":
        latency_ms = rng.uniform(2.0, 20.0) * scale
        jitter_ms = rng.uniform(0.3, 3.0) * scale
        loss_e2e = _log_uniform(rng, 1e-6, 5e-3)
        bandwidth_mbps = _log_uniform(rng, 0.1, 3.0)
    elif tclass == "pubsub_telemetry":
        latency_ms = rng.uniform(10.0, 200.0) * scale
        jitter_ms = rng.uniform(1.0, 20.0) * scale
        loss_e2e = _log_uniform(rng, 1e-6, 1e-2)
        bandwidth_mbps = _log_uniform(rng, 0.05, 20.0)
    elif tclass == "video_stream":
        latency_ms = rng.uniform(10.0, 80.0) * scale
        jitter_ms = rng.uniform(1.0, 10.0) * scale
        loss_e2e = _log_uniform(rng, 1e-5, 2e-2)
        bw_hi = 150.0 if tpl_name in (TPL_T2, TPL_T4) else 80.0
        bandwidth_mbps = _log_uniform(rng, 1.0, bw_hi)
    else:
        latency_ms = rng.uniform(50.0, 500.0) * scale
        jitter_ms = rng.uniform(5.0, 50.0) * scale
        loss_e2e = _log_uniform(rng, 1e-4, 5e-2)
        bandwidth_mbps = _log_uniform(rng, 0.01, 10.0)

    if loss_mode == "per_link":
        loss_rate = e2e_to_perlink_loss(loss_e2e, hops_est)
    else:
        loss_rate = loss_e2e

    return dict(
        basis=basis,
        traffic_class=tclass,
        priority=priority,
        latency_ms=float(latency_ms),
        jitter_ms=float(jitter_ms),
        loss_rate=float(loss_rate),
        bandwidth_mbps=float(bandwidth_mbps),
        loss_mode=loss_mode,
    )


def oracle_label(d: Data, q: Dict, loss_mode: str, rng: random.Random, random_endpoints: bool) -> Tuple[int, Dict]:
    q_lat = float(q["latency_ms"])
    q_jit = float(q["jitter_ms"])
    q_loss_raw = float(q["loss_rate"])
    q_bw = float(q["bandwidth_mbps"])
    priority = q.get("priority", "high")
    tclass = q.get("traffic_class", "control_loop")

    require_frer = (priority == "critical")
    require_tsn = (tclass in ["control_loop", "safety_io"])

    tcid = tclass_id(q)
    src, dst = pick_src_dst_from_tclass(d.ntype, tcid, rng=rng, random_endpoints=random_endpoints)

    edge_map = unique_undirected_edges(d.edge_index, d.edge_attr)
    G = build_sp_graph(edge_map)

    try:
        path_nodes = nx.shortest_path(G, source=src, target=dst, weight="weight")
    except nx.NetworkXNoPath:
        return 0, {"reason": "no_path", "src": src, "dst": dst, "path_hops": -1, "violated": ["no_path"]}

    m = aggregate_path_metrics(path_nodes, edge_map)
    if m is None:
        return 0, {"reason": "bad_path", "src": src, "dst": dst, "path_hops": -1, "violated": ["bad_path"]}

    hops = float(m["hops"])
    if loss_mode == "per_link":
        q_loss_budget = perlink_to_e2e_budget(q_loss_raw, hops)
    else:
        q_loss_budget = q_loss_raw

    violated = []
    if m["latency"] > q_lat: violated.append("latency")
    if m["jitter"] > q_jit: violated.append("jitter")
    if m["loss"] > q_loss_budget: violated.append("loss")
    if m["bandwidth"] < q_bw: violated.append("bandwidth")
    if require_tsn and (m["tsn_all"] < 0.5): violated.append("tsn")
    if require_frer and (m["frer_all"] < 0.5): violated.append("frer")

    y = 1 if len(violated) == 0 else 0
    reason = "ok" if y == 1 else "violated_" + "_".join(violated)

    return y, {
        "reason": reason,
        "violated": violated,
        "src": src,
        "dst": dst,
        "path_hops": int(m["hops"]),
        "metrics": m,
        "loss_budget_e2e": float(q_loss_budget),
        "loss_mode": loss_mode,
        "path_nodes": path_nodes,
    }


def compute_slacks(d: Data, m: Dict[str, float]) -> torch.Tensor:
    q = d.q.squeeze(0) if d.q.dim() == 2 else d.q.view(-1)
    q_lat  = float(q[0].item())
    q_jit  = float(q[1].item())
    q_loss = float(q[2].item())
    q_bw   = float(q[3].item())

    hops = float(m.get("hops", 1.0))

    loss_mode = "e2e"
    if hasattr(d, "loss_mode"):
        loss_mode = str(d.loss_mode)
    elif hasattr(d, "q_dict") and isinstance(d.q_dict, dict) and "loss_mode" in d.q_dict:
        loss_mode = str(d.q_dict["loss_mode"])

    if loss_mode == "per_link":
        q_loss_budget = 1.0 - (1.0 - q_loss) ** max(1.0, hops)
    else:
        q_loss_budget = q_loss

    s_lat = (q_lat - float(m["latency"])) / max(q_lat, EPS)
    s_jit = (q_jit - float(m["jitter"])) / max(q_jit, EPS)
    s_loss = math.log10((q_loss_budget + EPS) / (float(m["loss"]) + EPS))
    s_bw   = math.log10((float(m["bandwidth"]) + EPS) / (q_bw + EPS))

    require_tsn, require_frer = require_flags(d)
    s_tsn  = BIG_BIN if (not require_tsn  or float(m["tsn_all"])  > 0.5) else -BIG_BIN
    s_frer = BIG_BIN if (not require_frer or float(m["frer_all"]) > 0.5) else -BIG_BIN

    return torch.tensor([s_lat, s_jit, s_loss, s_bw, s_tsn, s_frer], dtype=torch.float)

def choose_bottleneck_axis(slacks: torch.Tensor) -> Tuple[int, str]:
    idx = int(torch.argmin(slacks).item())
    return idx, AXES[idx]


def build_path_edge_mask(d: Data, path_nodes: List[int]) -> torch.Tensor:
    E = int(d.edge_index.size(1))
    mask = torch.zeros(E, dtype=torch.bool)
    if not path_nodes or len(path_nodes) < 2:
        return mask

    u_list = d.edge_index[0].tolist()
    v_list = d.edge_index[1].tolist()
    dir_map = {(u, v): i for i, (u, v) in enumerate(zip(u_list, v_list))}

    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        idx = dir_map.get((int(a), int(b)), None)
        if idx is not None:
            mask[idx] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, default="graph_pool.pt")
    ap.add_argument("--out", type=str, default="layer2_pairs_with_bneck.pt")
    ap.add_argument("--k_per_graph", type=int, default=200)
    ap.add_argument("--k_by_tpl", type=int, nargs=4, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss_mode", type=str, default="per_link", choices=["per_link", "e2e"])
    ap.add_argument("--limit_graphs", type=int, default=-1)
    ap.add_argument("--random_endpoints", action="store_true", help="src/dst 후보에서 랜덤 선택(데이터 분포 바뀜)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    graphs: List[Data] = torch.load(args.graphs, weights_only=False)
    if args.limit_graphs > 0:
        graphs = graphs[: args.limit_graphs]

    def k_for_graph(tpl: str) -> int:
        if args.k_by_tpl is None:
            return args.k_per_graph
        if tpl == TPL_T1: return int(args.k_by_tpl[0])
        if tpl == TPL_T2: return int(args.k_by_tpl[1])
        if tpl == TPL_T3: return int(args.k_by_tpl[2])
        if tpl == TPL_T4: return int(args.k_by_tpl[3])
        return args.k_per_graph

    out_list: List[Data] = []
    pos = neg = 0
    axis_hist = {k: 0 for k in AXES}
    extra_hist = {"no_path": 0, "bad_path": 0}

    for g in graphs:
        tpl = getattr(g, "tpl", "OTHER")
        kk = k_for_graph(tpl)

        for _ in range(kk):
            qd = sample_q_for_tpl(rng, tpl, loss_mode=args.loss_mode)

            d = clone_graph_only(g)
            d.q = q_to_tensor(qd).view(1, -1)
            d.tclass = torch.tensor([tclass_id(qd)], dtype=torch.long)

            y, info = oracle_label(d, qd, loss_mode=args.loss_mode, rng=rng, random_endpoints=args.random_endpoints)

            d.y = torch.tensor([float(y)], dtype=torch.float)
            d.src = torch.tensor([int(info["src"])], dtype=torch.long)
            d.dst = torch.tensor([int(info["dst"])], dtype=torch.long)
            d.path_hops = torch.tensor([int(info["path_hops"])], dtype=torch.long)
            d.reason = info["reason"]

            d.loss_mode = info.get("loss_mode", args.loss_mode)
            d.loss_budget_e2e = torch.tensor([float(info.get("loss_budget_e2e", qd["loss_rate"]))], dtype=torch.float)

            d.q_dict = qd  # 디버깅용

            if "path_nodes" in info and isinstance(info["path_nodes"], list):
                d.path_edge_mask = build_path_edge_mask(d, info["path_nodes"])
            else:
                d.path_edge_mask = torch.zeros(d.edge_index.size(1), dtype=torch.bool)

            if info["reason"] == "no_path":
                slacks = torch.tensor([-BIG_BIN] * 6, dtype=torch.float)
                bidx, bname = AXIS_ID["loss"], "no_path"
                extra_hist["no_path"] += 1
            elif info["reason"] == "bad_path":
                slacks = torch.tensor([-BIG_BIN] * 6, dtype=torch.float)
                bidx, bname = AXIS_ID["loss"], "bad_path"
                extra_hist["bad_path"] += 1
            else:
                m = info["metrics"]
                slacks = compute_slacks(d, m)
                bidx, bname = choose_bottleneck_axis(slacks)
                axis_hist[bname] += 1

            d.bneck_slack = slacks.view(1, -1)
            d.bneck_axis_id = torch.tensor([int(bidx)], dtype=torch.long)
            d.bneck_axis_name = str(bname)

            out_list.append(d)
            pos += int(y == 1)
            neg += int(y == 0)

    total = len(out_list)
    print("[build_layer2_dataset] loss_mode:", args.loss_mode)
    print("[build_layer2_dataset] random_endpoints:", args.random_endpoints)
    print("[build_layer2_dataset] samples:", total, "pos:", pos, "neg:", neg, "pos_ratio:", pos / max(1, total))
    print("[build_layer2_dataset] axis_hist:", axis_hist)
    print("[build_layer2_dataset] extra_hist:", extra_hist)

    torch.save(out_list, args.out)
    print("[build_layer2_dataset] saved:", args.out)

if __name__ == "__main__":
    main()