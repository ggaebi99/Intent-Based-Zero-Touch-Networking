# train_layer2_B.py
import argparse
import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from model import build_layer2_model

AXES = ["latency", "jitter", "loss", "bandwidth", "tsn", "frer"]

def compute_norm(dataset: List, device="cpu") -> Dict[str, torch.Tensor]:
    # edge_attr / q mean,std
    edge_all = torch.cat([d.edge_attr for d in dataset], dim=0)
    q_all = torch.cat([d.q.squeeze(1) if d.q.dim()==3 else d.q for d in dataset], dim=0)
    norm = {
        "edge_mean": edge_all.mean(dim=0).to(device),
        "edge_std": edge_all.std(dim=0).to(device).clamp_min(1e-12),
        "q_mean": q_all.mean(dim=0).to(device),
        "q_std": q_all.std(dim=0).to(device).clamp_min(1e-12),
    }
    return norm

def apply_norm(batch, norm: Optional[Dict[str, torch.Tensor]]):
    if norm is None:
        return batch
    em, es = norm["edge_mean"], norm["edge_std"]
    qm, qs = norm["q_mean"], norm["q_std"]
    batch.edge_attr = (batch.edge_attr - em) / es
    q = batch.q
    if q.dim() == 3:
        q = q.squeeze(1)
        q = (q - qm) / qs
        batch.q = q.unsqueeze(1)
    else:
        batch.q = (q - qm) / qs
    return batch

@torch.no_grad()
def eval_epoch(model, loader, device, norm, thr=0.5):
    model.eval()
    total, correct = 0, 0
    neg_total, axis_correct = 0, 0

    all_prob = []
    all_y = []

    for batch in loader:
        batch = batch.to(device)
        batch = apply_norm(batch, norm)
        y_logit, axis_logit = model(batch)
        prob = torch.sigmoid(y_logit).detach().cpu()
        y_true = batch.y.view(-1).detach().cpu()
        y_hat = (prob >= thr).float()

        total += y_true.numel()
        correct += (y_hat == y_true).sum().item()

        # axis acc (neg only)
        neg_mask = (y_true < 0.5)
        if neg_mask.any():
            axis_true = batch.bneck_axis_id.view(-1).detach().cpu()
            axis_pred = torch.argmax(axis_logit.detach().cpu(), dim=-1)
            neg_total += neg_mask.sum().item()
            axis_correct += (axis_pred[neg_mask] == axis_true[neg_mask]).sum().item()

        all_prob.append(prob)
        all_y.append(y_true)

    y_acc = correct / max(1, total)
    axis_acc_neg = axis_correct / max(1, neg_total)
    return y_acc, axis_acc_neg

def find_best_thr(model, loader, device, norm):
    # 간단히 0.00~1.00 스윕(0.01 step)
    best_thr, best_acc = 0.5, -1.0
    for t in [i/100 for i in range(0, 101)]:
        y_acc, _ = eval_epoch(model, loader, device, norm, thr=t)
        if y_acc > best_acc:
            best_acc = y_acc
            best_thr = t
    return best_thr, best_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--val", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--arch", type=str, default="Layer2GINENoPathStats")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--tclass_vocab", type=int, default=8)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print("[device]", device)

    train_set: List = torch.load(args.train, weights_only=False)
    val_set: List = torch.load(args.val, weights_only=False)
    print(f"[load] train={len(train_set)} val={len(val_set)}")

    # dims
    g0 = train_set[0]
    x_dim = int(g0.x.size(-1))
    edge_dim = int(g0.edge_attr.size(-1))
    q_dim = 6

    model = build_layer2_model(
        arch=args.arch,
        x_dim=x_dim,
        edge_dim=edge_dim,
        q_dim=q_dim,
        hidden=args.hidden,
        num_layers=args.layers,
        tclass_vocab=args.tclass_vocab,
    ).to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # norm
    norm = compute_norm(train_set, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_ckpt = None

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch = apply_norm(batch, norm)

            y_logit, axis_logit = model(batch)
            y_true = batch.y.view(-1)

            y_loss = bce(y_logit, y_true)

            # axis는 neg-only로 학습 (Bottleneck은 infeasible에서 의미가 큼)
            neg_mask = (y_true < 0.5)
            if neg_mask.any():
                axis_true = batch.bneck_axis_id.view(-1)
                axis_loss = ce(axis_logit[neg_mask], axis_true[neg_mask])
            else:
                axis_loss = 0.0 * y_loss

            loss = y_loss + axis_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        # threshold 튜닝 + 평가
        thr, _ = find_best_thr(model, val_loader, device, norm)
        val_acc, val_axis_acc = eval_epoch(model, val_loader, device, norm, thr=thr)

        print(f"[ep {ep:03d}] loss={total_loss/max(1,steps):.4f} | val_y_acc={val_acc:.4f} val_axis_acc(neg)={val_axis_acc:.4f} thr={thr:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = {
                "arch": args.arch,
                "model_state_dict": model.state_dict(),
                "norm": {k: v.detach().cpu() for k, v in norm.items()},
                "best_thr": thr,
                "meta": {"x_dim": x_dim, "edge_dim": edge_dim, "q_dim": q_dim},
                "model_kwargs": {
                    "hidden": args.hidden,
                    "num_layers": args.layers,
                    "tclass_vocab": args.tclass_vocab,
                }
            }

    torch.save(best_ckpt, args.out)
    print(f"[save] best_val_acc={best_val_acc:.4f} -> {args.out}")

if __name__ == "__main__":
    main()