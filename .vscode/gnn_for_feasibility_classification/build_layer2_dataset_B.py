# build_layer2_dataset_B.py
import argparse
import random
from typing import List

import torch
from torch_geometric.data import Data

from build_layer2_dataset import (
    sample_q_for_tpl, q_to_tensor, tclass_id,
    clone_graph_only, oracle_label,
    pick_src_dst_from_tclass,
    compute_slacks, choose_bottleneck_axis,
)

AXES = ["latency", "jitter", "loss", "bandwidth", "tsn", "frer"]

FIXED_INTENTS = [
    dict(basis="hard", traffic_class="video_stream", priority="high",
         latency_ms=50.0, jitter_ms=5.0, loss_rate=0.001, bandwidth_mbps=40.0),
    dict(basis="hard", traffic_class="control_loop", priority="critical",
         latency_ms=1.0, jitter_ms=0.4, loss_rate=1e-5, bandwidth_mbps=1.0),
    dict(basis="hard", traffic_class="video_stream", priority="high",
         latency_ms=2.0, jitter_ms=1.0, loss_rate=0.002, bandwidth_mbps=0.5),
    dict(basis="hard", traffic_class="control_loop", priority="critical",
         latency_ms=2.0, jitter_ms=0.5, loss_rate=1e-5, bandwidth_mbps=0.5),
    dict(basis="hard", traffic_class="safety_io", priority="critical",
         latency_ms=12.0, jitter_ms=3.0, loss_rate=0.01, bandwidth_mbps=2.0),
]


def pack_minimal_input(d_full: Data, y: int, axis_id: int) -> Data:
    """
    ✅ B안 입력만 남김:
      x, edge_index, edge_attr, q, tclass + labels(y, bneck_axis_id)
    """
    out = Data(
        x=d_full.x,
        edge_index=d_full.edge_index,
        edge_attr=d_full.edge_attr,
    )
    out.q = d_full.q
    out.tclass = d_full.tclass
    out.y = torch.tensor([float(y)], dtype=torch.float32)
    out.bneck_axis_id = torch.tensor([int(axis_id)], dtype=torch.long)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--loss_mode", type=str, default="per_link", choices=["per_link", "e2e"])
    ap.add_argument("--k_per_graph", type=int, default=200)
    ap.add_argument("--max_graphs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--intent_mode", type=str, default="random", choices=["random", "fixed"])
    ap.add_argument("--random_endpoints", action="store_true",
                    help="oracle이 src/dst를 랜덤으로 뽑게 할지 (B안 기본은 False 추천)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    graphs: List[Data] = torch.load(args.graphs, weights_only=False)
    if args.max_graphs > 0:
        graphs = graphs[: args.max_graphs]
    print(f"[graphs] loaded={len(graphs)} from {args.graphs}")

    samples: List[Data] = []

    for gi, g in enumerate(graphs):
        tpl = getattr(g, "tpl", "OTHER")

        # intent 생성
        intents = []
        if args.intent_mode == "fixed":
            for ii in range(args.k_per_graph):
                qd = FIXED_INTENTS[ii % len(FIXED_INTENTS)].copy()
                qd["loss_mode"] = args.loss_mode
                intents.append(qd)
        else:
            for _ in range(args.k_per_graph):
                qd = sample_q_for_tpl(rng, tpl, loss_mode=args.loss_mode)
                intents.append(qd)

        for qd in intents:
            d = clone_graph_only(g)

            # ✅ (모델이 받을 것) : q, tclass
            d.q = q_to_tensor(qd).view(1, -1)
            tcid = tclass_id(qd)
            d.tclass = torch.tensor([tcid], dtype=torch.long)

            # oracle 라벨 계산용 src/dst (B안에서는 "내부 정책"으로 고정이라고 해석)
            src, dst = pick_src_dst_from_tclass(d.ntype, tcid)
            d.src = torch.tensor([src], dtype=torch.long)
            d.dst = torch.tensor([dst], dtype=torch.long)

            # ✅ oracle_label 시그니처(너희 코드) 반영: rng, random_endpoints 필요
            y_true, info = oracle_label(d, qd, args.loss_mode, rng, args.random_endpoints)

            # bottleneck axis 계산 (oracle info + slack 기반)
            if info.get("reason", "") not in ("no_path", "bad_path"):
                m = info["metrics"]
                slacks = compute_slacks(d, m)
                axis_id, _ = choose_bottleneck_axis(slacks)
            else:
                axis_id = 2  # fallback: loss

            samples.append(pack_minimal_input(d, int(y_true), int(axis_id)))

    print(f"[buildB] samples={len(samples)} (graphs={len(graphs)} x k_per_graph={args.k_per_graph})")
    torch.save(samples, args.out)
    print(f"[save] -> {args.out}")


if __name__ == "__main__":
    main()