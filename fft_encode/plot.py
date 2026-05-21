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
    """runs -> {(encoder, C): dict(val_nll=[...], val_acc=[...])}

    Accepts runs from results.json (flat val_nll) or results_full/main.json
    (best_val_nll inside run dicts). Uses best_val_nll when present.
    """
    g = defaultdict(lambda: dict(val_nll=[], val_acc=[]))
    for r in runs:
        if r.get("cfg", {}).get("encoder") == "cat" and r["cfg"]["C"] > 8:
            continue  # skipped in sweep, guard here too
        nll = r.get("best_val_nll", r.get("val_nll"))
        acc = r.get("best_val_acc", r.get("val_acc"))
        key = (r["cfg"]["encoder"], r["cfg"]["C"])
        g[key]["val_nll"].append(nll)
        g[key]["val_acc"].append(acc)
    return g


def plot_sweep(runs, x_field, xlabel, xticks_log2, out_path, random_nll, random_acc):
    g = defaultdict(lambda: dict(val_nll=[], val_acc=[]))
    for r in runs:
        nll = r.get("best_val_nll", r.get("val_nll"))
        acc = r.get("best_val_acc", r.get("val_acc"))
        key = (r["cfg"]["encoder"], r["cfg"][x_field])
        g[key]["val_nll"].append(nll)
        g[key]["val_acc"].append(acc)
    xs_all = sorted({k[1] for k in g})
    present = {k[0] for k in g}
    order = ["sum", "sum-ortho", "concat", "fdm", "fdm-learn", "ci", "cat"]
    encoders = [e for e in order if e in present]
    style = {
        "sum":       dict(color="#d1495b", marker="s", label="sum"),
        "sum-ortho": dict(color="#b5179e", marker="P", label="sum-ortho (soft orthogonality)"),
        "concat":    dict(color="#edae49", marker="o", label="concat (block, no carrier)"),
        "fdm":       dict(color="#00798c", marker="^", label="fdm (block + fixed carrier)"),
        "fdm-learn": dict(color="#30638e", marker="D", label="fdm-learn (learnable $\\omega_k$)"),
        "ci":        dict(color="#6a994e", marker="X", label="ci (channel-independent)"),
        "cat":       dict(color="#5f0f40", marker="*", label="cat (channel-as-token)"),
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
    ap.add_argument("--results", default="results_full/main.json")
    ap.add_argument("--out", default="paper/figures/scaling.pdf")
    ap.add_argument("--dmodel-results", default="results_full/dmodel.json")
    ap.add_argument("--dmodel-out", default="paper/figures/dmodel_scaling.pdf")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    g = aggregate(data["runs"])
    Cs = sorted({k[1] for k in g})
    present = {k[0] for k in g}
    order = ["sum", "sum-ortho", "concat", "fdm", "fdm-learn", "ci", "cat"]
    encoders = [e for e in order if e in present]
    style = {
        "sum":       dict(color="#d1495b", marker="s", label="sum"),
        "sum-ortho": dict(color="#b5179e", marker="P", label="sum-ortho"),
        "concat":    dict(color="#edae49", marker="o", label="concat"),
        "fdm":       dict(color="#00798c", marker="^", label="fdm"),
        "fdm-learn": dict(color="#30638e", marker="D", label="fdm-learn"),
        "ci":        dict(color="#6a994e", marker="X", label="ci"),
        "cat":       dict(color="#5f0f40", marker="*", label="cat"),
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
