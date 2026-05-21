"""Aggregate canonical + extra_seeds outputs, rerun every paired test from
the paper at the new seed count, and produce bootstrap CIs.

Reads:
    <out>/main.json, etth1.json, main_largen.json, dmodel.json,
    geom_largen.json, geometry.json, pospro_geometry.json, bias.json
    plus every <out>/extra_seeds/{stage}_sNNN.json file.

Writes:
    <out>/analysis_<N>seeds.json   structured results
    <out>/analysis_<N>seeds.txt    human-readable summary

Usage:
    uv run python -m fft_encode.analyze --out results_paper
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _seed_from_filename(path: str) -> int:
    m = re.search(r"_s(\d+)\.json$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot extract seed from {path}")
    return int(m.group(1))


def load_all(out_dir: str) -> dict:
    """Merge canonical + extra_seeds outputs into per-stage flat lists of
    run dicts. For dict-shaped files (geometry, pospro_geometry) the
    aggregate preserves the dict structure with extended lists."""
    extra_dir = os.path.join(out_dir, "extra_seeds")
    data: dict = {}

    # List-shaped stages: main, etth1, main_largen, dmodel, geom_largen, bias
    for stage in ["main", "etth1", "main_largen", "dmodel",
                  "geom_largen", "bias"]:
        runs = _load_json(os.path.join(out_dir, f"{stage}.json")) or []
        for path in sorted(glob.glob(os.path.join(
                extra_dir, f"{stage}_s*.json"))):
            runs.extend(_load_json(path) or [])
        data[stage] = runs

    # Dict-shaped stages: geometry, pospro_geometry
    for stage in ["geometry", "pospro_geometry"]:
        canonical = _load_json(os.path.join(out_dir, f"{stage}.json")) or {}
        merged: dict = {k: list(v) for k, v in canonical.items()}
        for path in sorted(glob.glob(os.path.join(
                extra_dir, f"{stage}_s*.json"))):
            extra = _load_json(path) or {}
            for k, v in extra.items():
                merged.setdefault(k, []).extend(v)
        data[stage] = merged

    return data


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

_ENCODER_ALIAS = {"sum-perch": "linear", "sum-ortho": "linear-ortho"}


def _run_field(r: dict, key: str):
    """Read a config field that can live either at top level (etth1) or
    nested under ``cfg`` (main/dmodel/etc). Encoder names are canonicalised
    so the legacy ``sum-perch`` / ``sum-ortho`` runs match the renamed
    ``linear`` / ``linear-ortho`` references in comparisons."""
    if key in r:
        v = r[key]
    else:
        v = r.get("cfg", {}).get(key)
    if key == "encoder" and v in _ENCODER_ALIAS:
        return _ENCODER_ALIAS[v]
    return v


def paired_diffs(data, stage: str, A: str, B: str, **filters) -> dict:
    """Pull per-seed NLLs for two encoders matching the given filters,
    returning seed-aligned arrays plus paired diffs (B - A so positive
    means A is better). For list-shaped stages only."""
    runs = data[stage]
    def matches(r, enc):
        if _run_field(r, "encoder") != enc:
            return False
        for k, v in filters.items():
            if _run_field(r, k) != v:
                return False
        return True
    A_by_seed = {_run_field(r, "seed"): r for r in runs if matches(r, A)}
    B_by_seed = {_run_field(r, "seed"): r for r in runs if matches(r, B)}
    common = sorted(set(A_by_seed) & set(B_by_seed))
    if not common:
        return dict(seeds=[], nll_A=np.array([]), nll_B=np.array([]),
                    diff=np.array([]), per_example_A=[], per_example_B=[])
    nll_A = np.array([A_by_seed[s]["best_val_nll"] for s in common])
    nll_B = np.array([B_by_seed[s]["best_val_nll"] for s in common])
    per_example_A = [A_by_seed[s].get("best_per_example_nll") for s in common]
    per_example_B = [B_by_seed[s].get("best_per_example_nll") for s in common]
    return dict(
        seeds=common, nll_A=nll_A, nll_B=nll_B,
        diff=nll_B - nll_A,  # positive => A is better
        per_example_A=per_example_A,
        per_example_B=per_example_B,
    )


def paired_t(diffs: np.ndarray) -> dict:
    """Standard parametric paired t-test."""
    if len(diffs) < 2:
        return dict(t=None, p=None, mean=None, se=None, ci=(None, None))
    n = len(diffs)
    mean = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    t = mean / se if se > 0 else float("inf")
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci = (mean - t_crit * se, mean + t_crit * se)
    return dict(t=float(t), p=p, mean=mean, se=se, ci=(float(ci[0]), float(ci[1])))


def bootstrap_seed(diffs: np.ndarray, n_boot: int = 10000,
                   ci_level: float = 0.95, rng=None) -> dict:
    """Seed-level bootstrap CI on the mean diff."""
    if rng is None:
        rng = np.random.default_rng(0)
    if len(diffs) < 2:
        return dict(n=len(diffs), mean=None, ci=(None, None), p_one_sided=None)
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diffs[idx].mean()
    lo = float(np.percentile(means, 100 * (1 - ci_level) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci_level) / 2))
    # One-sided p-value: fraction of bootstrap means at or below zero.
    p_one = float((means <= 0).mean()) if np.mean(diffs) > 0 else \
            float((means >= 0).mean())
    return dict(n=n, mean=float(np.mean(diffs)),
                ci=(lo, hi), p_one_sided=p_one)


def bootstrap_cluster(per_seed_diffs: list, n_boot: int = 10000,
                      ci_level: float = 0.95, rng=None) -> dict:
    """Cluster bootstrap using per-example diffs.

    ``per_seed_diffs`` is a list of arrays (one per seed) containing
    per-example deltas (B - A) for that seed. We resample seeds with
    replacement and pool all their per-example deltas to compute the
    mean each iteration.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    per_seed_diffs = [np.asarray(d) for d in per_seed_diffs if d is not None
                      and len(d) > 0]
    if len(per_seed_diffs) < 2:
        return dict(n_seeds=len(per_seed_diffs), n_examples_total=0,
                    mean=None, ci=(None, None), p_one_sided=None)
    n_seeds = len(per_seed_diffs)
    n_examples_total = sum(len(d) for d in per_seed_diffs)
    all_means = np.empty(n_boot)
    seed_idx_pool = np.arange(n_seeds)
    for b in range(n_boot):
        sampled = rng.choice(seed_idx_pool, n_seeds, replace=True)
        pooled = np.concatenate([per_seed_diffs[i] for i in sampled])
        all_means[b] = pooled.mean()
    lo = float(np.percentile(all_means, 100 * (1 - ci_level) / 2))
    hi = float(np.percentile(all_means, 100 * (1 + ci_level) / 2))
    # Observed mean of pooled per-example diffs across all seeds.
    obs_mean = float(np.concatenate(per_seed_diffs).mean())
    p_one = float((all_means <= 0).mean()) if obs_mean > 0 else \
            float((all_means >= 0).mean())
    return dict(n_seeds=n_seeds, n_examples_total=n_examples_total,
                mean=obs_mean, ci=(lo, hi), p_one_sided=p_one)


def per_example_diffs(d: dict) -> list:
    """Compute paired per-example deltas (B - A) per seed where both
    encoders have per-example NLLs saved. Returns a list of np arrays
    (one per seed with available data)."""
    out = []
    for a, b in zip(d["per_example_A"], d["per_example_B"]):
        if a is None or b is None:
            continue
        a = np.asarray(a)
        b = np.asarray(b)
        if len(a) != len(b):
            continue  # different val sets shouldn't happen but be safe
        out.append(b - a)
    return out


# ---------------------------------------------------------------------------
# Comparisons defined for the paper
# ---------------------------------------------------------------------------

COMPARISONS = [
    # (stage, encoder_A_better_when_positive, encoder_B, filters, label)
    # Synthetic main sweep
    ("main", "linear-ppe", "linear",     {"C": 4},  "linear-ppe vs linear at C=4"),
    ("main", "linear-ppe", "linear",     {"C": 8},  "linear-ppe vs linear at C=8"),
    ("main", "linear-ppe", "linear",     {"C": 16}, "linear-ppe vs linear at C=16"),

    ("main", "mlp",        "linear",     {"C": 16}, "mlp vs linear at C=16"),
    ("main", "mlp",        "concat",     {"C": 16}, "mlp vs concat at C=16"),
    ("main", "mlp",        "linear-ortho",  {"C": 16}, "mlp vs linear-ortho at C=16"),
    ("main", "mlp",        "linear-ppe", {"C": 16}, "mlp vs linear-ppe at C=16"),

    # ETTh1
    ("etth1", "cat", "linear",     {}, "cat vs linear on ETTh1"),
    ("etth1", "cat", "concat",     {}, "cat vs concat on ETTh1"),
    ("etth1", "cat", "linear-ortho",  {}, "cat vs linear-ortho on ETTh1"),
    ("etth1", "cat", "mlp",        {}, "cat vs mlp on ETTh1"),
    ("etth1", "cat", "linear-ppe", {}, "cat vs linear-ppe on ETTh1"),

    # d_model sweep at C=4
    ("dmodel", "linear-ppe", "linear", {"C": 4, "d_model": 64},  "linear-ppe vs linear at d=64"),
    ("dmodel", "linear-ppe", "linear", {"C": 4, "d_model": 128}, "linear-ppe vs linear at d=128"),
    ("dmodel", "linear-ppe", "linear", {"C": 4, "d_model": 256}, "linear-ppe vs linear at d=256"),
    ("dmodel", "linear",     "mlp",    {"C": 4, "d_model": 64},  "linear vs mlp at d=64 (mlp degrades)"),
    ("dmodel", "linear",     "mlp",    {"C": 4, "d_model": 128}, "linear vs mlp at d=128"),
    ("dmodel", "linear",     "mlp",    {"C": 4, "d_model": 256}, "linear vs mlp at d=256"),

    # main_largen at C=16, N=5120
    ("main_largen", "mlp", "linear",     {"C": 16}, "mlp vs linear at C=16, N=5120"),
    ("main_largen", "mlp", "concat",     {"C": 16}, "mlp vs concat at C=16, N=5120"),
    ("main_largen", "mlp", "linear-ortho",  {"C": 16}, "mlp vs linear-ortho at C=16, N=5120"),
    ("main_largen", "mlp", "linear-ppe", {"C": 16}, "mlp vs linear-ppe at C=16, N=5120"),

    # Bias ablation (canonical 5 seeds only — not extended)
    ("bias", "linear", "linear-nobias",  {}, "linear vs linear-nobias at C=4"),
]


# ---------------------------------------------------------------------------
# Geometry-stage CIs (not paired-NLL, just bootstrap CIs on scalar
# quantities reported in tab:gram, tab:space, tab:pospro-geom).
# ---------------------------------------------------------------------------

def bootstrap_scalar(values: np.ndarray, n_boot: int = 10000,
                     ci_level: float = 0.95, rng=None) -> dict:
    """Seed-level bootstrap CI on the mean of a scalar quantity."""
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return dict(n=len(values),
                    mean=float(values.mean()) if len(values) else None,
                    ci=(None, None))
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = values[idx].mean()
    lo = float(np.percentile(means, 100 * (1 - ci_level) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci_level) / 2))
    return dict(n=n, mean=float(values.mean()), ci=(lo, hi))


def geometry_summary(data: dict) -> dict:
    """Bootstrap CIs on the gram / encoding-space numbers reported in
    Tables 5-6 of the paper."""
    out = {}
    rng = np.random.default_rng(42)
    # linear at C ∈ {4, 8, 16}
    for C in (4, 8, 16):
        runs = [r for r in data["geometry"].get("linear", []) if r["C"] == C]
        if not runs:
            continue
        # off-diagonal cosines
        mc = np.array([r["mean_off_abs_cos"] for r in runs])
        xc = np.array([r["max_off_abs_cos"] for r in runs])
        # norms split into driver (0..3) and distractor (4..C-1)
        norms = np.array([r["norms"] for r in runs])  # (n_seeds, C)
        drv_mean = norms[:, :4].mean(axis=1)
        dst_mean = norms[:, 4:].mean(axis=1) if C > 4 else None
        ratio = drv_mean / dst_mean if dst_mean is not None else None
        # variance fractions
        var = np.array([r["var_fraction"] for r in runs])
        drv_var = var[:, :4].sum(axis=1)
        dst_var = var[:, 4:].sum(axis=1) if C > 4 else None
        block = dict(
            n_seeds=len(runs),
            mean_off_abs_cos=bootstrap_scalar(mc, rng=rng),
            max_off_abs_cos=bootstrap_scalar(xc, rng=rng),
            driver_norm_mean=bootstrap_scalar(drv_mean, rng=rng),
        )
        if dst_mean is not None:
            block["distractor_norm_mean"] = bootstrap_scalar(dst_mean, rng=rng)
            block["driver_over_distractor_ratio"] = bootstrap_scalar(ratio, rng=rng)
            block["driver_variance_fraction"] = bootstrap_scalar(drv_var, rng=rng)
            block["distractor_variance_fraction"] = bootstrap_scalar(dst_var, rng=rng)
        out[f"linear_C{C}"] = block

    # linear-ortho at C=4 (legacy ``sum_ortho`` key also accepted)
    runs = (data["geometry"].get("linear_ortho", [])
            or data["geometry"].get("sum_ortho", []))
    if runs:
        mc = np.array([r["mean_off_abs_cos"] for r in runs])
        xc = np.array([r["max_off_abs_cos"] for r in runs])
        out["linear_ortho_C4"] = dict(
            n_seeds=len(runs),
            mean_off_abs_cos=bootstrap_scalar(mc, rng=rng),
            max_off_abs_cos=bootstrap_scalar(xc, rng=rng),
        )

    # pospro_geometry: linear vs linear-ppe at C=4
    pg = data.get("pospro_geometry", {})
    for key, label in (("linear", "linear_C4"),
                       ("linear_ppe", "linear_ppe_C4")):
        runs = pg.get(key, [])
        if not runs:
            continue
        eff = np.array([r["P_entropy_rank"] for r in runs])
        r99 = np.array([r["P_rank_99"] for r in runs])
        frac = np.array([r["frac_P_energy_in_span_W"] for r in runs])
        out[f"pospro_{label}"] = dict(
            n_seeds=len(runs),
            P_entropy_rank=bootstrap_scalar(eff, rng=rng),
            P_rank_99=bootstrap_scalar(r99, rng=rng),
            frac_P_energy_in_span_W=bootstrap_scalar(frac, rng=rng),
        )

    # Pair the linear vs linear-ppe pospro into paired-diff stats too.
    lin_runs = {r["seed"]: r for r in pg.get("linear", [])}
    ppe_runs = {r["seed"]: r for r in pg.get("linear_ppe", [])}
    common = sorted(set(lin_runs) & set(ppe_runs))
    if common:
        frac_diff = np.array(
            [ppe_runs[s]["frac_P_energy_in_span_W"]
             - lin_runs[s]["frac_P_energy_in_span_W"] for s in common])
        eff_diff = np.array(
            [ppe_runs[s]["P_entropy_rank"]
             - lin_runs[s]["P_entropy_rank"] for s in common])
        out["pospro_paired_diff"] = dict(
            n_seeds=len(common),
            frac_in_W_diff=dict(  # negative means ppe rotates P out of span(W)
                mean=float(frac_diff.mean()),
                paired_t=paired_t(-frac_diff),  # report ppe-is-better positive
                bootstrap=bootstrap_seed(-frac_diff, rng=rng),
            ),
            eff_rank_diff=dict(  # positive means ppe spreads P over more dirs
                mean=float(eff_diff.mean()),
                paired_t=paired_t(eff_diff),
                bootstrap=bootstrap_seed(eff_diff, rng=rng),
            ),
        )

    return out


# ---------------------------------------------------------------------------
# Run all comparisons
# ---------------------------------------------------------------------------

def run_all(data: dict, n_boot: int = 10000) -> dict:
    """Run all paired comparisons + bootstrap, return structured results."""
    rng_seed = 0
    out = dict(comparisons=[], geometry=None)
    for stage, A, B, filters, label in COMPARISONS:
        d = paired_diffs(data, stage, A, B, **filters)
        if len(d["diff"]) < 2:
            out["comparisons"].append(dict(
                label=label, stage=stage, A=A, B=B, filters=filters,
                n=len(d["diff"]),
                note=f"insufficient seeds ({len(d['diff'])})",
            ))
            continue
        rng = np.random.default_rng(rng_seed)
        rng_seed += 1
        diffs = d["diff"]
        t_res = paired_t(diffs)
        b_seed = bootstrap_seed(diffs, n_boot=n_boot, rng=rng)
        # Cluster bootstrap using per-example data if available
        rng_c = np.random.default_rng(rng_seed)
        rng_seed += 1
        ex_diffs = per_example_diffs(d)
        b_cluster = bootstrap_cluster(ex_diffs, n_boot=n_boot, rng=rng_c) \
            if ex_diffs else None
        out["comparisons"].append(dict(
            label=label, stage=stage, A=A, B=B, filters=filters,
            n=int(len(diffs)),
            mean_diff=float(diffs.mean()),  # positive => A is better
            std_diff=float(diffs.std(ddof=1)),
            n_pos=int((diffs > 0).sum()),
            paired_t=t_res,
            bootstrap_seed=b_seed,
            bootstrap_cluster=b_cluster,
            nll_A_mean=float(d["nll_A"].mean()),
            nll_A_std=float(d["nll_A"].std(ddof=1)),
            nll_B_mean=float(d["nll_B"].mean()),
            nll_B_std=float(d["nll_B"].std(ddof=1)),
        ))
    out["geometry"] = geometry_summary(data)
    return out


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _fmt_ci(ci):
    if ci is None or ci[0] is None:
        return "[n/a]"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def render_text(results: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("PAIRED COMPARISONS")
    lines.append("=" * 78)
    lines.append("Positive Δ = encoder A wins. Bootstrap p is one-sided "
                 "(prob. of bootstrap mean having opposite sign).")
    lines.append("")
    for c in results["comparisons"]:
        lines.append(f"--- {c['label']} (stage={c['stage']}) ---")
        if "note" in c:
            lines.append(f"    {c['note']}")
            lines.append("")
            continue
        lines.append(
            f"    n={c['n']}, {c['n_pos']}/{c['n']} paired diffs favour A, "
            f"mean Δ = {c['mean_diff']:+.4f} (std {c['std_diff']:.4f})")
        lines.append(
            f"    NLL  A: {c['nll_A_mean']:.4f}±{c['nll_A_std']:.4f}, "
            f"B: {c['nll_B_mean']:.4f}±{c['nll_B_std']:.4f}")
        t = c["paired_t"]
        lines.append(
            f"    paired t-test: t={t['t']:.3f}, p={t['p']:.4g}, "
            f"95% CI {_fmt_ci(t['ci'])}")
        b = c["bootstrap_seed"]
        lines.append(
            f"    bootstrap (seed-level, n={b['n']}, B=10000): "
            f"mean Δ={b['mean']:+.4f}, 95% CI {_fmt_ci(b['ci'])}, "
            f"p_one-sided={b['p_one_sided']:.4g}")
        bc = c["bootstrap_cluster"]
        if bc is not None:
            lines.append(
                f"    bootstrap (cluster: per-example within seed, "
                f"{bc['n_seeds']} seeds × ~{bc['n_examples_total']//bc['n_seeds']} ex): "
                f"mean Δ={bc['mean']:+.4f}, 95% CI {_fmt_ci(bc['ci'])}, "
                f"p_one-sided={bc['p_one_sided']:.4g}")
        else:
            lines.append("    bootstrap (cluster): not available "
                         "(no per-example data)")
        lines.append("")

    lines.append("=" * 78)
    lines.append("GEOMETRY / POSPRO BOOTSTRAP CIs")
    lines.append("=" * 78)
    geo = results.get("geometry") or {}
    for key in sorted(geo):
        block = geo[key]
        lines.append(f"--- {key} (n={block.get('n_seeds', '?')}) ---")
        for metric, val in block.items():
            if metric == "n_seeds":
                continue
            if isinstance(val, dict) and "ci" in val:
                lines.append(f"    {metric}: mean={val['mean']:.4f}, "
                             f"95% CI {_fmt_ci(val['ci'])}")
            elif isinstance(val, dict):
                # nested (e.g. pospro_paired_diff)
                for sub_name, sub_val in val.items():
                    if isinstance(sub_val, dict) and "ci" in sub_val:
                        lines.append(
                            f"    {metric}.{sub_name}: "
                            f"mean={sub_val['mean']:.4f}, "
                            f"95% CI {_fmt_ci(sub_val['ci'])}")
                    elif sub_name == "paired_t":
                        lines.append(
                            f"    {metric}.{sub_name}: t={sub_val['t']:.3f}, "
                            f"p={sub_val['p']:.4g}")
                    elif sub_name == "bootstrap":
                        lines.append(
                            f"    {metric}.bootstrap: "
                            f"mean={sub_val['mean']:.4f}, "
                            f"95% CI {_fmt_ci(sub_val['ci'])}, "
                            f"p_one-sided={sub_val['p_one_sided']:.4g}")
                    elif sub_name == "mean":
                        lines.append(f"    {metric}.mean: {sub_val:.4f}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results_paper")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    data = load_all(args.out)
    # Report seed counts so the user can sanity-check
    n_main = len({_run_field(r, "seed") for r in data["main"]
                  if _run_field(r, "encoder") == "linear"
                  and _run_field(r, "C") == 4})
    n_etth1 = len({_run_field(r, "seed") for r in data["etth1"]
                   if _run_field(r, "encoder") == "linear"})
    n_geom = len([r for r in data["geometry"].get("linear", [])
                  if r["C"] == 4])
    n_pospro = len(data["pospro_geometry"].get("linear", []))
    print(f"Seed counts (linear-anchored sanity check):")
    print(f"  main C=4: {n_main}")
    print(f"  etth1:    {n_etth1}")
    print(f"  geometry C=4: {n_geom}")
    print(f"  pospro_geometry C=4: {n_pospro}")
    print()

    results = run_all(data, n_boot=args.n_boot)

    n_seeds_main = n_main
    json_path = os.path.join(args.out, f"analysis_{n_seeds_main}seeds.json")
    txt_path = os.path.join(args.out, f"analysis_{n_seeds_main}seeds.txt")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    text = render_text(results)
    with open(txt_path, "w") as f:
        f.write(text + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    print()
    print(text)


if __name__ == "__main__":
    main()
