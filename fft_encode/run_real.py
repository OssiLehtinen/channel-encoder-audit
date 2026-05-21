"""Run the main encoder comparison on ETTh1 (real multivariate time series).

ETTh1 has 7 variates. We use d_model=56 so that d_model mod C == 0 for
concat/fdm. Same architecture otherwise (3 layers, FFN=4*d_model, dropout
0.1, pre-LayerNorm). Cosine LR decay, 100 epochs, 5 seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import build_model, nll_loss
from .real_data import ETTh1Dataset
from .experiments import _make_scheduler


def train_one(encoder: str, seed: int, epochs: int, d_model: int,
              n_heads: int, n_layers: int, d_ff: int, bins: int,
              device: str, log_every: int = 10):
    torch.manual_seed(seed)
    tr = ETTh1Dataset("train", T=160, K=bins)
    va = ETTh1Dataset("val", T=160, K=bins)
    C = tr.C

    train_loader = DataLoader(tr, batch_size=32, shuffle=True)
    val_loader = DataLoader(va, batch_size=32)

    model = build_model(
        kind=encoder, n_channels=C, d_model=d_model, n_bins=bins,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # minimal RunCfg-shaped stub for scheduler
    from types import SimpleNamespace
    sched_cfg = SimpleNamespace(lr_schedule="cosine", lr=3e-4,
                                lr_min_frac=0.01, epochs=epochs)
    sched = _make_scheduler(opt, sched_cfg)

    trace = []
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = nll_loss(model(x), y) + model.aux_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched is not None:
            sched.step()
        if (ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1:
            model.eval()
            tot_loss, tot_correct, tot_n = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = nll_loss(logits, y)
                    pred = logits[:, :-1].argmax(-1)
                    tgt = y[:, 1:]
                    tot_correct += (pred == tgt).sum().item()
                    tot_n += tgt.numel()
                    tot_loss += loss.item() * tgt.numel()
            trace.append(dict(epoch=ep + 1,
                              val_nll=tot_loss / tot_n,
                              val_acc=tot_correct / tot_n))
    best = min(trace, key=lambda t: t["val_nll"])
    return dict(encoder=encoder, seed=seed, params=n_params,
                trace=trace,
                best_val_nll=best["val_nll"],
                best_val_acc=best["val_acc"],
                best_epoch=best["epoch"],
                final_val_nll=trace[-1]["val_nll"],
                final_val_acc=trace[-1]["val_acc"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_full/real.json")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--d-model", type=int, default=56,
                   help="must be divisible by number of channels (7)")
    p.add_argument("--encoders", nargs="+",
                   default=["sum", "sum-perch", "sum-ortho",
                            "concat", "fdm", "ci", "cat"])
    args = p.parse_args()

    d_model = args.d_model
    assert d_model % 7 == 0, "d_model must be divisible by 7 for ETTh1"
    n_heads = 7
    d_ff = 4 * d_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"d_model={d_model}, heads={n_heads}, d_ff={d_ff}")

    runs = []
    for enc in args.encoders:
        for seed in range(args.seeds):
            t0 = time.time()
            r = train_one(enc, seed, args.epochs, d_model, n_heads, 3,
                          d_ff, 32, device)
            runs.append(r)
            print(f"  [{enc:>10} s={seed}] "
                  f"best_nll={r['best_val_nll']:.4f}@ep{r['best_epoch']:>3} "
                  f"best_acc={r['best_val_acc']:.3f} "
                  f"params={r['params']:,} ({time.time()-t0:.1f}s)")

    # Aggregate
    print("\n=== ETTh1 aggregate (mean ± std over seeds) ===")
    encs = list(dict.fromkeys(r["encoder"] for r in runs))
    print(f"{'encoder':>10} {'params':>10} {'val_nll':>18} {'val_acc':>16}")
    for e in encs:
        rs = [r for r in runs if r["encoder"] == e]
        nll_m, nll_s = np.mean([r["best_val_nll"] for r in rs]), np.std([r["best_val_nll"] for r in rs])
        acc_m, acc_s = np.mean([r["best_val_acc"] for r in rs]), np.std([r["best_val_acc"] for r in rs])
        params = rs[0]["params"]
        print(f"{e:>10} {params:>10,} {nll_m:.4f} ± {nll_s:.4f}   {acc_m:.3f} ± {acc_s:.3f}")

    with open(args.out, "w") as f:
        json.dump(dict(runs=runs), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
