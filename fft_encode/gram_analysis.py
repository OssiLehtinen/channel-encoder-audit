"""Measure how orthogonal the learned W_k actually become.

Trains sum-perch (no regulariser) and sum-ortho (with lambda=1e-2) on the
benchmark at C=4 and reports, per seed:
  - max and mean of |cos(W_i, W_j)| over i!=j,
  - the full (C, C) cosine matrix,
  - the W_k norms.

Writes gram_analysis.json so the paper can quote exact numbers.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .experiments import RunCfg
from .runner import train_and_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_full/gram.json")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--C", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    out = []
    for enc in ("sum-perch", "sum-ortho"):
        for seed in range(args.seeds):
            cfg = RunCfg(encoder=enc, C=args.C, seed=seed,
                         epochs=args.epochs, n_series=512)
            t0 = time.time()
            r, model, _ = train_and_trace(cfg, device, log_every=20,
                                          return_model=True)
            stats = model.encoder.gram_stats()
            out.append(dict(
                encoder=enc, seed=seed,
                best_val_nll=r["best_val_nll"],
                best_val_acc=r["best_val_acc"],
                flag=r["convergence_flag"],
                **stats,
            ))
            print(
                f"  [{enc} s={seed}] "
                f"nll={r['best_val_nll']:.4f} "
                f"max|cos|={stats['max_off_abs_cos']:.4f} "
                f"mean|cos|={stats['mean_off_abs_cos']:.4f} "
                f"norms={['%.3f' % x for x in stats['norms']]} "
                f"({time.time()-t0:.1f}s)"
            )

    # Aggregate
    print("\n=== mean over seeds ===")
    print(f"{'encoder':>10} {'best_nll':>10} {'max|cos|':>10} {'mean|cos|':>11}")
    for enc in ("sum-perch", "sum-ortho"):
        rows = [r for r in out if r["encoder"] == enc]
        nll = np.mean([r["best_val_nll"] for r in rows])
        mx = np.mean([r["max_off_abs_cos"] for r in rows])
        mn = np.mean([r["mean_off_abs_cos"] for r in rows])
        print(f"{enc:>10} {nll:>10.4f} {mx:>10.4f} {mn:>11.4f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
