"""Scale d_model: does fdm's edge survive at 128 and 256?

Sweeps d_model in {64, 128, 256} for the three encoders at C=4, 3 seeds.
Keeps d_ff = 4 * d_model and heads = 4 (so d_head grows with d_model).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np

import torch

from .experiments import RunCfg, train_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="scale_dmodel.json")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--C", type=int, default=4)
    p.add_argument("--d-models", type=str, default="64,128,256")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    encoders = ["sum", "concat", "fdm", "fdm-learn"]
    d_models = [int(x) for x in args.d_models.split(",")]
    runs = []
    for d_model in d_models:
        d_ff = 4 * d_model
        for enc in encoders:
            for s in range(args.seeds):
                cfg = RunCfg(
                    encoder=enc, C=args.C, seed=s,
                    epochs=args.epochs,
                    d_model=d_model, d_ff=d_ff,
                )
                t0 = time.time()
                r = train_run(cfg, device)
                print(
                    f"[d_model={d_model} {enc} s={s}] "
                    f"val_nll={r['val_nll']:.4f} val_acc={r['val_acc']:.3f} "
                    f"params={r['params']:,} ({time.time()-t0:.1f}s)"
                )
                runs.append(r)

    # Aggregate
    agg = defaultdict(lambda: dict(val_nll=[], val_acc=[], params=None))
    for r in runs:
        key = (r["cfg"]["d_model"], r["cfg"]["encoder"])
        agg[key]["val_nll"].append(r["val_nll"])
        agg[key]["val_acc"].append(r["val_acc"])
        agg[key]["params"] = r["params"]

    print("\n=== Aggregate (mean ± std over seeds) ===")
    print(f"{'d_model':>8} {'encoder':>7} {'params':>10} "
          f"{'val_nll':>16} {'val_acc':>14}")
    for d_model in d_models:
        for enc in encoders:
            v = agg[(d_model, enc)]
            nll_m, nll_s = np.mean(v["val_nll"]), np.std(v["val_nll"])
            acc_m, acc_s = np.mean(v["val_acc"]), np.std(v["val_acc"])
            print(
                f"{d_model:>8} {enc:>7} {v['params']:>10,} "
                f"{nll_m:.4f} ± {nll_s:.4f}  {acc_m:.3f} ± {acc_s:.3f}"
            )

    with open(args.out, "w") as f:
        json.dump(dict(runs=runs), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
