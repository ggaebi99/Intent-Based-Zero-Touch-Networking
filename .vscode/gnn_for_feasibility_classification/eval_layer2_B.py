# eval_layer2_B.py
import argparse
import torch
from torch_geometric.loader import DataLoader
from model import build_layer2_model

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
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--use_cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print("[device]", device)

    ds = torch.load(args.data, weights_only=False)
    print(f"[load] {args.data} | samples={len(ds)}")

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
    print(f"[thr] {thr:.2f}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    total = 0
    correct = 0
    neg_total = 0
    axis_correct = 0

    for batch in loader:
        batch = batch.to(device)
        batch = apply_norm(batch, norm)
        y_logit, axis_logit = model(batch)

        prob = torch.sigmoid(y_logit).detach().cpu()
        y_true = batch.y.view(-1).detach().cpu()
        y_hat = (prob >= thr).float()

        total += y_true.numel()
        correct += (y_hat == y_true).sum().item()

        neg_mask = (y_true < 0.5)
        if neg_mask.any():
            axis_true = batch.bneck_axis_id.view(-1).detach().cpu()
            axis_pred = torch.argmax(axis_logit.detach().cpu(), dim=-1)
            neg_total += neg_mask.sum().item()
            axis_correct += (axis_pred[neg_mask] == axis_true[neg_mask]).sum().item()

    y_acc = correct / max(1, total)
    axis_acc_neg = axis_correct / max(1, neg_total)

    print(f"[evalB] y_acc={y_acc:.4f} (total={total})")
    print(f"[evalB] axis_acc(neg)={axis_acc_neg:.4f} (neg={neg_total})")

if __name__ == "__main__":
    main()