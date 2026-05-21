"""Render paper figures from results JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLE = {
    "sum":        dict(color="#d1495b", marker="s", label="sum"),
    "linear":     dict(color="#2196f3", marker="D", label="linear"),
    "linear-ortho": dict(color="#b5179e", marker="P", label="linear-ortho"),
    "mlp":        dict(color="#ff9800", marker="v", label="mlp"),
    "linear-ppe": dict(color="#1b9e77", marker="^", label="linear-ppe"),
    "concat":     dict(color="#edae49", marker="o", label="concat"),
    "ci":         dict(color="#6a994e", marker="X", label="ci"),
    "cat":        dict(color="#5f0f40", marker="*", label="cat"),
}

# Back-compat aliases for results files produced before the
# sum-perch -> linear and sum-ortho -> linear-ortho renames.
# plot.py canonicalises by reading r["cfg"]["encoder"] through _canon.
_ENCODER_ALIAS = {"sum-perch": "linear", "sum-ortho": "linear-ortho"}


def _canon(enc: str) -> str:
    return _ENCODER_ALIAS.get(enc, enc)


ORDER = ["sum", "ci", "cat", "mlp", "concat", "linear", "linear-ortho",
         "linear-ppe"]

SKIP = {"fdm", "fdm-learn", "linear-lpe"}


def aggregate(runs):
    g = defaultdict(lambda: dict(val_nll=[], val_acc=[]))
    for r in runs:
        enc = _canon(r["cfg"]["encoder"])
        if enc in SKIP:
            continue
        C = r["cfg"]["C"]
        if enc == "cat" and C > 8:
            continue
        nll = r.get("best_val_nll", r.get("val_nll"))
        acc = r.get("best_val_acc", r.get("val_acc"))
        g[(enc, C)]["val_nll"].append(nll)
        g[(enc, C)]["val_acc"].append(acc)
    return g


def plot_sweep(runs, x_field, xlabel, xticks_log2, out_path,
               random_nll, random_acc):
    g = defaultdict(lambda: dict(val_nll=[], val_acc=[]))
    for r in runs:
        enc = _canon(r["cfg"]["encoder"])
        if enc in SKIP:
            continue
        nll = r.get("best_val_nll", r.get("val_nll"))
        acc = r.get("best_val_acc", r.get("val_acc"))
        g[(enc, r["cfg"][x_field])]["val_nll"].append(nll)
        g[(enc, r["cfg"][x_field])]["val_acc"].append(acc)

    xs_all = sorted({k[1] for k in g})
    present = {k[0] for k in g}
    encoders = [e for e in ORDER if e in present]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3),
                             constrained_layout=True)
    for enc in encoders:
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
        axes[0].errorbar(xs, nll_m, yerr=nll_s, linewidth=2,
                         capsize=3, **STYLE[enc])
        axes[1].errorbar(xs, acc_m, yerr=acc_s, linewidth=2,
                         capsize=3, **STYLE[enc])

    axes[0].axhline(random_nll, color="gray", linestyle=":",
                    linewidth=1, label="uniform")
    axes[1].axhline(random_acc, color="gray", linestyle=":",
                    linewidth=1, label="random")
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


def _load_runs(path):
    """Accept both the new reproduce.py shape (a list of run dicts) and the
    legacy ``{"runs": [...]}`` wrapper used by older result files."""
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data["runs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_paper/main.json")
    ap.add_argument("--out", default="paper/figures/scaling.pdf")
    ap.add_argument("--dmodel-results", default="results_paper/dmodel.json")
    ap.add_argument("--dmodel-out", default="paper/figures/dmodel_scaling.pdf")
    args = ap.parse_args()

    g = aggregate(_load_runs(args.results))
    Cs = sorted({k[1] for k in g})
    present = {k[0] for k in g}
    encoders = [e for e in ORDER if e in present]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3),
                             constrained_layout=True)
    n_enc = len(encoders)
    jitter = np.linspace(-0.06, 0.06, n_enc)
    for i, enc in enumerate(encoders):
        xs, nll_m, nll_s, acc_m, acc_s = [], [], [], [], []
        for C in Cs:
            v = g.get((enc, C))
            if v is None:
                continue
            xs.append(C * (2 ** jitter[i]))
            nll_m.append(np.mean(v["val_nll"]))
            nll_s.append(np.std(v["val_nll"]))
            acc_m.append(np.mean(v["val_acc"]))
            acc_s.append(np.std(v["val_acc"]))
        xs = np.array(xs)
        axes[0].errorbar(xs, nll_m, yerr=nll_s, linewidth=2,
                         capsize=3, **STYLE[enc])
        axes[1].errorbar(xs, acc_m, yerr=acc_s, linewidth=2,
                         capsize=3, **STYLE[enc])

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
        plot_sweep(
            _load_runs(args.dmodel_results), "d_model",
            xlabel=r"$d_{\mathrm{model}}$",
            xticks_log2=True,
            out_path=args.dmodel_out,
            random_nll=np.log(32), random_acc=1/32,
        )


if __name__ == "__main__":
    main()
