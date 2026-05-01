# make_graph_pool.py
import argparse
import random
from typing import List, Sequence, Optional, Dict

import torch
from torch_geometric.data import Data

from topology_templates import TopologyTemplate, NODE_TYPES, build_T1, build_T2, build_T3, build_T4
from edge_profiles import ScenarioParams, sample_edge_attr

# edge_attr = [bw, lat, jit, loss, queue, interf, tsn, frer]
BW, LAT, JIT, LOSS, QUEUE, INTERF, TSN, FRER = range(8)


def node_onehot(node_type: str) -> List[float]:
    return [1.0 if node_type == t else 0.0 for t in NODE_TYPES]


def node_type_ids(node_types: List[str]) -> torch.Tensor:
    return torch.tensor([NODE_TYPES.index(t) for t in node_types], dtype=torch.long)


def sample_edge_attr_compat(src_type: str, dst_type: str, kind: str, scenario: ScenarioParams):
    """
    sample_edge_attr 시그니처가 환경마다 달라질 수 있어 호환 처리.
    - (src, dst, scenario)
    - (src, dst, kind, scenario)
    - (src, dst, is_wireless, scenario)
    """
    try:
        return sample_edge_attr(src_type, dst_type, scenario)
    except TypeError:
        pass
    try:
        return sample_edge_attr(src_type, dst_type, kind, scenario)
    except TypeError:
        pass
    return sample_edge_attr(src_type, dst_type, (kind == "wireless"), scenario)


def rand_scenario(rng: random.Random) -> ScenarioParams:
    return ScenarioParams(
        congestion_level=rng.uniform(0.0, 1.0),
        interference_level=rng.uniform(0.0, 1.0),
        failure_mode=rng.choice(["none", "none", "none", "disable_frer", "disable_tsn"]),
    )


def build_graph_only(template: TopologyTemplate, scenario: ScenarioParams) -> Data:
    """
    ✅ 그래프만 생성 (q/y 없음)
    """
    x = torch.tensor([node_onehot(t) for t in template.node_types], dtype=torch.float)
    ntype = node_type_ids(template.node_types)

    edge_pairs = []
    edge_attr_list = []

    for (u, v, kind) in template.edges:
        attr_uv = sample_edge_attr_compat(template.node_types[u], template.node_types[v], kind, scenario)

        # wireless면 interference +, wired면 -
        if kind == "wireless":
            attr_uv[INTERF] = min(1.0, float(attr_uv[INTERF]) + 0.2 * scenario.interference_level)
        else:
            attr_uv[INTERF] = max(0.0, float(attr_uv[INTERF]) - 0.1 * scenario.interference_level)

        edge_pairs.append([u, v]); edge_attr_list.append(attr_uv)
        edge_pairs.append([v, u]); edge_attr_list.append(attr_uv)

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.ntype = ntype
    data.tpl = template.name
    return data


def template_mix_stats(graphs: List[Data]) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for g in graphs:
        name = getattr(g, "tpl", "UNKNOWN")
        cnt[name] = cnt.get(name, 0) + 1
    return cnt


def generate_graph_pool(
    templates: Sequence[TopologyTemplate],
    n_graphs: int,
    tpl_probs: Optional[List[float]],
    seed: int,
) -> List[Data]:
    rng = random.Random(seed)

    if tpl_probs is None:
        tpl_probs = [1.0 / len(templates)] * len(templates)
    s = sum(tpl_probs)
    tpl_probs = [p / s for p in tpl_probs]

    out: List[Data] = []
    for _ in range(n_graphs):
        tpl = rng.choices(list(templates), weights=tpl_probs, k=1)[0]
        out.append(build_graph_only(tpl, rand_scenario(rng)))
    return out


def generate_graphs_fixed_counts(
    templates: Sequence[TopologyTemplate],
    counts: List[int],
    seed: int,
) -> List[Data]:
    """
    ✅ 정확히 [T1,T2,T3,T4] 개수만큼 생성 (예: 10 10 10 5)
    """
    rng = random.Random(seed)
    assert len(counts) == 4

    out: List[Data] = []
    for tpl, k in zip(templates, counts):
        for _ in range(k):
            out.append(build_graph_only(tpl, rand_scenario(rng)))

    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="graph_pool.pt")
    ap.add_argument("--seed", type=int, default=42)

    # 둘 중 하나만 쓰면 됨
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--tpl_probs", type=float, nargs=4, default=None)
    ap.add_argument("--fixed_tpl_counts", type=int, nargs=4, default=None,
                    help="EXACT counts for [T1,T2,T3,T4], e.g. 10 10 10 5")

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    templates = [
        build_T1(8),
        build_T2(12),
        build_T3(24, cells=4),
        build_T4(48, zones=4),
    ]

    if args.fixed_tpl_counts is not None:
        graphs = generate_graphs_fixed_counts(templates, args.fixed_tpl_counts, seed=args.seed)
    else:
        graphs = generate_graph_pool(templates, args.n, args.tpl_probs, seed=args.seed)

    print("[make_graph_pool] generated:", len(graphs))
    mix = template_mix_stats(graphs)
    print("[make_graph_pool] template mix:")
    for k in sorted(mix.keys()):
        print(f"  {k}: {mix[k]}")

    d0 = graphs[0]
    print("[make_graph_pool] first graph:",
          "x", tuple(d0.x.shape),
          "ntype", tuple(d0.ntype.shape),
          "edge_index", tuple(d0.edge_index.shape),
          "edge_attr", tuple(d0.edge_attr.shape),
          "tpl", getattr(d0, "tpl", None))

    torch.save(graphs, args.out)
    print("[make_graph_pool] saved:", args.out)


if __name__ == "__main__":
    main()