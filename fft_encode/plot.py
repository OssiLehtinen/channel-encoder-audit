"""Render paper figures from results.json."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def aggregate(runs):
    """runs -> {(encoder, C): dict(val_nll=[...], val_acc=[...])}"""
    g = defaultdict(lambda: dict(val_nll=[], val_acc=[]))
    for r in runs:
        key = (r["cfg"]["encoder"], r["cfg"]["C"])
        g[key]["val_nll"].append(r["val_nll"])
        g[key]["val_acc"].append(r["val_acc"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="paper/figures/scaling.pdf")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    g = aggregate(data["runs"])
    Cs = sorted({k[1] for k in g})
    encoders = ["sum", "concat", "fdm"]
    style = {
        "sum":    dict(color="#d1495b", marker="s", label="sum"),
        "concat": dict(color="#edae49", marker="o", label="concat (block, no carrier)"),
        "fdm":    dict(color="#00798c", marker="^", label="fdm (block + carrier)"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    for enc in encoders:
        xs, nll_m, nll_s, acc_m, acc_s = [], [], [], [], []
        for C in Cs:
            v = g[(enc, C)]
            xs.append(C)
            nll_m.append(np.mean(v["val_nll"]))
            nll_s.append(np.std(v["val_nll"]))
            acc_m.append(np.mean(v["val_acc"]))
            acc_s.append(np.std(v["val_acc"]))
        xs = np.array(xs)
        axes[0].errorbar(xs, nll_m, yerr=nll_s, linewidth=2,
                         capsize=3, **style[enc])
        axes[1].errorbar(xs, acc_m, yerr=acc_s, linewidth=2,
                         capsize=3, **style[enc])

    axes[0].axhline(np.log(32), color="gray", linestyle=":",
                    linewidth=1, label=r"uniform ($\ln 32$)")
    axes[1].axhline(1/32, color="gray", linestyle=":",
                    linewidth=1, label=r"random ($1/32$)")
    axes[0].set_xlabel(r"number of input channels $C$")
    axes[1].set_xlabel(r"number of input channels $C$")
    axes[0].set_ylabel("val NLL (lower is better)")
    axes[1].set_ylabel("val top-1 bin accuracy")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Cs)
        ax.set_xticklabels([str(c) for c in Cs])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
