"""Grid search over fixed FDM carrier bands at d_model=128.

The question: do the default log-spaced carriers in [0.5, 8.0] rad/sample
just happen to be locally bad, or does no fixed frequency ladder close the
fdm-vs-concat gap that opens at d_model>=128?

We sweep (omega_min, omega_max) over a hand-picked 2D log grid covering:
- very low bands aligned with the actual signal content
  (signals live in ~[0.03, 0.50] rad/sample)
- the default
- very high bands
- narrow and wide bands

Reports mean +- std val NLL / acc per (omega_min, omega_max) over 3 seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .data import MultiSignalDataset
from .encodings import FDMChannelEncoding
from .experiments import RunCfg
from .model import SignalTransformer, nll_loss


def build_fdm_model(cfg: RunCfg, omega_min: float, omega_max: float):
    enc = FDMChannelEncoding(
        n_channels=cfg.C, d_model=cfg.d_model,
    )
    # override the default [0.5, 8.0] log-spaced carriers
    new_omegas = torch.logspace(
        math.log10(omega_min), math.log10(omega_max),
        steps=cfg.C, base=10.0,
    )
    with torch.no_grad():
        enc.omegas.copy_(new_omegas) if isinstance(enc.omegas, torch.nn.Parameter) \
            else enc.omegas.data.copy_(new_omegas)
    return SignalTransformer(
        enc, d_model=cfg.d_model, n_heads=cfg.heads, n_layers=cfg.layers,
        d_ff=cfg.d_ff, n_bins=cfg.bins, dropout=cfg.dropout,
    )


def run_one(cfg: RunCfg, omega_min: float, omega_max: float, device):
    torch.manual_seed(cfg.seed)
    ds = MultiSignalDataset(n_series=cfg.n_series, T=cfg.T, C=cfg.C,
                            K=cfg.bins, seed=cfg.seed)
    n_val = max(32, len(ds) // 10)
    tr, va = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=cfg.batch_size)

    model = build_fdm_model(cfg, omega_min, omega_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    from .experiments import _make_scheduler
    sched = _make_scheduler(opt, cfg)
    for _ in range(cfg.epochs):
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
    return tot_loss / tot_n, tot_correct / tot_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="carrier_grid.json")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--C", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # Hand-picked grid: spans signal band through 10x higher, narrow to wide.
    grid = [
        (0.03, 0.5),   # matches signal bandwidth
        (0.03, 2.0),
        (0.1, 1.0),    # narrow, low
        (0.1, 4.0),    # moderate-low
        (0.1, 16.0),   # wide, low-min
        (0.3, 3.0),    # narrow, mid
        (0.5, 8.0),    # default
        (0.5, 50.0),   # very wide
        (1.0, 16.0),   # mid-high
        (3.0, 30.0),   # high band
        (10.0, 100.0), # very high
    ]
    seeds = list(range(args.seeds))

    runs = []
    for (om_lo, om_hi) in grid:
        print(f"\n--- omega range [{om_lo}, {om_hi}] ---")
        for s in seeds:
            cfg = RunCfg(
                encoder="fdm", C=args.C, seed=s,
                epochs=args.epochs, d_model=args.d_model,
                d_ff=4 * args.d_model,
            )
            t0 = time.time()
            nll, acc = run_one(cfg, om_lo, om_hi, device)
            dt = time.time() - t0
            print(f"  seed={s} nll={nll:.4f} acc={acc:.3f} ({dt:.1f}s)")
            runs.append(dict(
                omega_min=om_lo, omega_max=om_hi,
                seed=s, val_nll=nll, val_acc=acc,
                d_model=args.d_model, C=args.C, epochs=args.epochs,
            ))

    # Aggregate
    agg = defaultdict(lambda: dict(val_nll=[], val_acc=[]))
    for r in runs:
        agg[(r["omega_min"], r["omega_max"])]["val_nll"].append(r["val_nll"])
        agg[(r["omega_min"], r["omega_max"])]["val_acc"].append(r["val_acc"])

    print("\n=== Aggregate (d_model=%d, C=%d, mean +- std over %d seeds) ==="
          % (args.d_model, args.C, args.seeds))
    print(f"{'omega_min':>10} {'omega_max':>10}  {'val_nll':>16}  {'val_acc':>14}")
    # Sort by mean NLL
    sorted_keys = sorted(
        agg.keys(), key=lambda k: float(np.mean(agg[k]["val_nll"]))
    )
    for k in sorted_keys:
        v = agg[k]
        nm, ns = np.mean(v["val_nll"]), np.std(v["val_nll"])
        am, a_s = np.mean(v["val_acc"]), np.std(v["val_acc"])
        print(f"{k[0]:>10} {k[1]:>10}  {nm:.4f} ± {ns:.4f}  {am:.3f} ± {a_s:.3f}")
    print("\nReferences at d_model=128: sum=3.265, concat=2.797, fdm(default)=2.963")

    with open(args.out, "w") as f:
        json.dump(dict(runs=runs), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
