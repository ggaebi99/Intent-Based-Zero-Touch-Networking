# dump_layer2_outputs.py
import argparse
import torch
from torch_geometric.loader import DataLoader
from model import build_layer2_model  # 너가 준 model.py 기준

def apply_norm(batch, norm):
    if norm is None:
        return batch
    em, es = norm["edge_mean"].to(batch.edge_attr.device), norm["edge_std"].to(batch.edge_attr.device)
    qm, qs = norm["q_mean"].to(batch.q.device), norm["q_std"].to(batch.q.device)
    batch.edge_attr = (batch.edge_attr - em) / (es + 1e-12)

    q = batch.q
    if q.dim() == 3:
        q = q.squeeze(1)
        q = (q - qm) / (qs + 1e-12)
        batch.q = q.unsqueeze(1)
    else:
        batch.q = (q - qm) / (qs + 1e-12)
    return batch

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)   # layer2_val_B.pt 같은 것
    ap.add_argument("--ckpt", required=True)   # layer2 checkpoint .pt
    ap.add_argument("--out", required=True)    # dump output .pt
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--use_cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    ds = torch.load(args.data, weights_only=False)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt["meta"]
    model = build_layer2_model(
        arch=ckpt["arch"],
        x_dim=int(meta["x_dim"]),
        edge_dim=int(meta["edge_dim"]),
        q_dim=int(meta["q_dim"]),
        **ckpt.get("model_kwargs", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    norm = ckpt.get("norm", None)
    thr = float(ckpt.get("best_thr", 0.5))
    print("arch:", ckpt["arch"], "thr:", thr, "device:", device)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    dumps = []
    for batch in loader:
        batch = batch.to(device)
        batch = apply_norm(batch, norm)
        y_logit, axis_logit = model(batch)

        p = torch.sigmoid(y_logit)  # (B,)
        y_hat = (p >= thr).float()

        topk = min(args.topk, axis_logit.size(-1))
        axis_topk = torch.topk(axis_logit, k=topk, dim=-1).indices  # (B,topk)

        dumps.append({
            "p_feasible": p.detach().cpu(),
            "y_hat": y_hat.detach().cpu(),
            "axis_logit": axis_logit.detach().cpu(),
            "axis_topk": axis_topk.detach().cpu(),
            "y_true": batch.y.view(-1).detach().cpu(),
            "axis_true": batch.bneck_axis_id.view(-1).detach().cpu(),
        })

    torch.save({
        "data_path": args.data,
        "ckpt_path": args.ckpt,
        "thr": thr,
        "chunks": dumps,
    }, args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()