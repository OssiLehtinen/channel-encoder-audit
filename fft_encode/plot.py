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


def plot_sweep(runs, x_field, xlabel, xticks_log2, out_path, random_nll, random_acc):
    g = defaultdict(lambda: dict(val_nll=[], val_acc=[]))
    for r in runs:
        key = (r["cfg"]["encoder"], r["cfg"][x_field])
        g[key]["val_nll"].append(r["val_nll"])
        g[key]["val_acc"].append(r["val_acc"])
    xs_all = sorted({k[1] for k in g})
    encoders = ["sum", "concat", "fdm"]
    if any(e.startswith("fdm-") for e in {k[0] for k in g}):
        encoders = ["sum", "concat", "fdm", "fdm-learn"]
    style = {
        "sum":       dict(color="#d1495b", marker="s", label="sum"),
        "concat":    dict(color="#edae49", marker="o", label="concat (block, no carrier)"),
        "fdm":       dict(color="#00798c", marker="^", label="fdm (block + fixed carrier)"),
        "fdm-learn": dict(color="#30638e", marker="D", label="fdm-learn (learnable $\\omega_k$)"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    for enc in encoders:
        if not any(k[0] == enc for k in g):
            continue
        xs, nll_m, nll_s, acc_m, acc_s = [], [], [], [], []
        for x in xs_all:
            v = g.get((enc, x))
            if v is None:
                continue
            xs.append(x)
            nll_m.append(np.mean(v["val_nll"]))
            nll_s.append(np.std(v["val_nll"]))
            acc_m.append(np.mean(v["val_acc"]))
            acc_s.append(np.std(v["val_acc"]))
        axes[0].errorbar(xs, nll_m, yerr=nll_s, linewidth=2, capsize=3, **style[enc])
        axes[1].errorbar(xs, acc_m, yerr=acc_s, linewidth=2, capsize=3, **style[enc])
    axes[0].axhline(random_nll, color="gray", linestyle=":", linewidth=1, label="uniform")
    axes[1].axhline(random_acc, color="gray", linestyle=":", linewidth=1, label="random")
    for ax in axes:
        ax.set_xlabel(xlabel)
        if xticks_log2:
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs_all)
            ax.set_xticklabels([str(c) for c in xs_all])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("val NLL (lower is better)")
    axes[1].set_ylabel("val top-1 bin accuracy")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="paper/figures/scaling.pdf")
    ap.add_argument("--dmodel-results", default="scale_dmodel.json")
    ap.add_argument("--dmodel-out", default="paper/figures/dmodel_scaling.pdf")
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

    import os
    if os.path.exists(args.dmodel_results):
        with open(args.dmodel_results) as f:
            d2 = json.load(f)
        plot_sweep(
            d2["runs"], "d_model",
            xlabel=r"$d_{\mathrm{model}}$",
            xticks_log2=True,
            out_path=args.dmodel_out,
            random_nll=np.log(32), random_acc=1/32,
        )


if __name__ == "__main__":
    main()
