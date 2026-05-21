"""Train both encoders side-by-side and report per-epoch NLL + accuracy."""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import MultiSignalDataset
from .model import build_model, nll_loss


def evaluate(model, loader, device):
    model.eval()
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nll_loss(logits, y)
            pred = logits[:, :-1].argmax(-1)
            tgt = y[:, 1:]
            tot_correct += (pred == tgt).sum().item()
            tot_n += tgt.numel()
            tot_loss += loss.item() * tgt.numel()
    return tot_loss / tot_n, tot_correct / tot_n


def train_one(kind, train_loader, val_loader, args, device):
    model = build_model(
        kind=kind,
        n_channels=args.channels,
        d_model=args.d_model,
        n_bins=args.bins,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{kind}] params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.01,
        )
        if args.lr_schedule == "cosine" else None
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"[{kind}] ep{epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nll_loss(logits, y) + model.aux_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        if sched is not None:
            sched.step()
        train_loss = tot / n
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"[{kind}] ep{epoch:02d} train_nll={train_loss:.4f} "
            f"val_nll={val_loss:.4f} val_acc={val_acc:.3f} ({dt:.1f}s)"
        )
        history.append((epoch, train_loss, val_loss, val_acc))
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-series", type=int, default=512)
    p.add_argument("--T", type=int, default=192)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--encoders",
        type=str,
        default="fdm,sum",
        help="comma-separated subset of {fdm,sum}",
    )
    p.add_argument(
        "--lr-schedule", type=str, default="cosine",
        choices=["cosine", "constant"],
        help="LR schedule (cosine annealing to 1%% of lr, or constant)",
    )
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"device={device}")

    ds = MultiSignalDataset(
        n_series=args.n_series, T=args.T, C=args.channels, K=args.bins, seed=args.seed
    )
    n_val = max(32, len(ds) // 10)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"dataset: train={n_train} val={n_val} T={args.T} C={args.channels} K={args.bins}")

    results = {}
    for kind in [s.strip() for s in args.encoders.split(",") if s.strip()]:
        results[kind] = train_one(kind, train_loader, val_loader, args, device)

    print("\n=== Final comparison ===")
    for kind, hist in results.items():
        last = hist[-1]
        print(f"  {kind:>4}: val_nll={last[2]:.4f}  val_acc={last[3]:.3f}")


if __name__ == "__main__":
    main()
