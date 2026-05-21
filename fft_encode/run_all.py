"""Run every experiment in the paper, end-to-end, with 100-epoch training
and per-epoch convergence tracing. One command, one output directory.

Usage:
    uv run python -m fft_encode.run_all --out results_full --epochs 100

Produces a subdirectory with:
    main.json       C in {4,8,16} x encoders x seeds (+ test-time channel mask)
    dmodel.json     d_model in {64,128,256} x encoders x seeds
    carrier.json    11 (omega_min,omega_max) bands x seeds at d_model=128
    probe.json      linear-probe channel recovery per encoder at C=4
    summary.txt     human-readable aggregate + convergence flags

Metrics are reported at each encoder/config's *best* val NLL achieved over
training (effective early stopping), with the epoch noted. Trajectories are
saved so learning curves can be re-plotted without retraining.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .data import MultiSignalDataset
from .encodings import FDMChannelEncoding
from .experiments import RunCfg, evaluate
from .model import SignalTransformer, build_model, nll_loss
from .runner import train_and_trace


def header(s: str) -> None:
    print(f"\n{'='*70}\n{s}\n{'='*70}")


def aggregate_runs(runs: list, group_keys: list) -> list:
    """Aggregate per-seed runs by group_keys (tuple of cfg keys). Returns list
    of dicts with mean/std for best and final NLL+acc, plus flags per seed."""
    buckets = defaultdict(list)
    for r in runs:
        key = tuple(r["cfg"].get(k, None) if k != "omega_range"
                    else (r.get("omega_min"), r.get("omega_max"))
                    for k in group_keys)
        buckets[key].append(r)

    out = []
    for key, rs in buckets.items():
        entry = dict(zip(group_keys, key))
        for metric in ["best_val_nll", "best_val_acc",
                       "final_val_nll", "final_val_acc"]:
            vals = [r[metric] for r in rs]
            entry[f"{metric}_mean"] = float(np.mean(vals))
            entry[f"{metric}_std"] = float(np.std(vals))
        entry["best_epoch_mean"] = float(np.mean([r["best_epoch"] for r in rs]))
        entry["flags"] = [r["convergence_flag"] for r in rs]
        entry["params"] = rs[0]["params"]
        out.append(entry)
    return out


def do_main_sweep(args, device) -> tuple[list, list]:
    """C in {4,8,16} x encoders x seeds. Then test-time channel masking
    on the C=4 seed-0 models."""
    header("Main sweep: C scaling + encoder comparison")
    Cs = args.Cs
    encoders = args.encoders
    runs = []
    for C in Cs:
        for enc in encoders:
            if enc == "cat" and C > args.cat_max_C:
                print(f"  (skip cat at C={C}; exceeds --cat-max-C={args.cat_max_C})")
                continue
            for s in range(args.seeds):
                cfg = RunCfg(
                    encoder=enc, C=C, seed=s,
                    epochs=args.epochs, n_series=args.n_series,
                    lr_schedule=args.lr_schedule,
                )
                t0 = time.time()
                r = train_and_trace(cfg, device, log_every=args.log_every)
                runs.append(r)
                print(
                    f"  [{enc:>9} C={C:>2} s={s}] "
                    f"best_nll={r['best_val_nll']:.4f}@ep{r['best_epoch']:>3} "
                    f"final_nll={r['final_val_nll']:.4f} "
                    f"acc@best={r['best_val_acc']:.3f} "
                    f"flag={r['convergence_flag']:<16} "
                    f"({time.time()-t0:.1f}s)"
                )

    # Test-time channel masking at C=4 (retrain once per encoder, seed 0)
    header("Channel-mask ablation (C=4, seed=0)")
    mask_runs = []
    for enc in encoders:
        cfg = RunCfg(encoder=enc, C=4, seed=0,
                     epochs=args.epochs, n_series=args.n_series,
                     lr_schedule=args.lr_schedule)
        t0 = time.time()
        torch.manual_seed(cfg.seed)
        ds = MultiSignalDataset(n_series=cfg.n_series, T=cfg.T, C=cfg.C,
                                K=cfg.bins, seed=cfg.seed)
        n_val = max(32, len(ds) // 10)
        tr, va = random_split(
            ds, [len(ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        val_loader = DataLoader(va, batch_size=cfg.batch_size)
        # reuse the matching run we just did if available, else retrain
        matching = [r for r in runs if r["cfg"]["encoder"] == enc
                    and r["cfg"]["C"] == 4 and r["cfg"]["seed"] == 0]
        # rebuild and retrain because we didn't keep the model handle
        r, model, _ = train_and_trace(cfg, device, log_every=args.log_every,
                                      return_model=True)
        base_nll, base_acc = r["final_val_nll"], r["final_val_acc"]
        per_channel = []
        for k in range(cfg.C):
            nll_k, acc_k = evaluate(model, val_loader, device, mask_channel=k)
            per_channel.append(dict(channel=k, val_nll=nll_k, val_acc=acc_k))
        mask_runs.append(dict(
            encoder=enc, base_nll=base_nll, base_acc=base_acc,
            per_channel=per_channel,
        ))
        print(f"  [{enc:>9}] base_acc={base_acc:.3f} mask_accs="
              f"{[round(p['val_acc'], 3) for p in per_channel]} "
              f"({time.time()-t0:.1f}s)")
    return runs, mask_runs


def do_dmodel_sweep(args, device) -> list:
    header("d_model scaling sweep")
    d_models = args.d_models
    encoders = args.encoders_dmodel
    runs = []
    for dm in d_models:
        d_ff = 4 * dm
        for enc in encoders:
            for s in range(args.seeds):
                cfg = RunCfg(
                    encoder=enc, C=4, seed=s,
                    epochs=args.epochs, n_series=args.n_series,
                    d_model=dm, d_ff=d_ff,
                    lr_schedule=args.lr_schedule,
                )
                t0 = time.time()
                r = train_and_trace(cfg, device, log_every=args.log_every)
                runs.append(r)
                print(
                    f"  [d={dm:>3} {enc:>9} s={s}] "
                    f"best_nll={r['best_val_nll']:.4f}@ep{r['best_epoch']:>3} "
                    f"final_nll={r['final_val_nll']:.4f} "
                    f"flag={r['convergence_flag']:<16} "
                    f"({time.time()-t0:.1f}s)"
                )
    return runs


def do_carrier_grid(args, device) -> list:
    header("Carrier-band grid at d_model=128")
    grid = [
        (0.03, 0.5), (0.03, 2.0), (0.1, 1.0), (0.1, 4.0), (0.1, 16.0),
        (0.3, 3.0), (0.5, 8.0), (0.5, 50.0), (1.0, 16.0), (3.0, 30.0),
        (10.0, 100.0),
    ]
    runs = []
    for (om_lo, om_hi) in grid:
        for s in range(args.seeds_grid):
            cfg = RunCfg(encoder="fdm", C=4, seed=s,
                         epochs=args.epochs, n_series=args.n_series,
                         d_model=128, d_ff=512,
                         lr_schedule=args.lr_schedule)

            def build_fn(cfg, om_lo=om_lo, om_hi=om_hi):
                enc = FDMChannelEncoding(n_channels=cfg.C, d_model=cfg.d_model)
                new_o = torch.logspace(math.log10(om_lo), math.log10(om_hi),
                                       steps=cfg.C, base=10.0)
                with torch.no_grad():
                    enc.omegas.data.copy_(new_o)
                return SignalTransformer(
                    enc, d_model=cfg.d_model, n_heads=cfg.heads,
                    n_layers=cfg.layers, d_ff=cfg.d_ff, n_bins=cfg.bins,
                    dropout=cfg.dropout,
                )

            t0 = time.time()
            r = train_and_trace(cfg, device, log_every=args.log_every,
                                build_model_fn=build_fn)
            r["omega_min"] = om_lo
            r["omega_max"] = om_hi
            runs.append(r)
            print(f"  [ω∈[{om_lo},{om_hi}] s={s}] "
                  f"best_nll={r['best_val_nll']:.4f}@ep{r['best_epoch']:>3} "
                  f"flag={r['convergence_flag']} ({time.time()-t0:.1f}s)")
    return runs


def do_probe(args, device) -> list:
    header("Linear-probe channel recovery (C=4)")
    # ci / cat have a fundamentally different hidden-state layout (per-
    # channel stream or per-(t,k) token) so a single-stream probe does not
    # apply. Restrict probing to single-stream encoder variants.
    probable = {"sum", "concat", "fdm", "fdm-learn", "sum-ortho"}
    encoders = [e for e in args.encoders if e in probable]
    results = []
    for enc in encoders:
        cfg = RunCfg(encoder=enc, C=4, seed=0,
                     epochs=args.epochs, n_series=args.n_series,
                     lr_schedule=args.lr_schedule)
        r, model, _ = train_and_trace(cfg, device, log_every=args.log_every,
                                      return_model=True)
        # Build probe dataset (independent of training data)
        from .data import MultiSignalDataset
        ds = MultiSignalDataset(n_series=256, T=cfg.T, C=cfg.C,
                                K=cfg.bins, seed=99)
        n_va = len(ds) // 5
        tr, va = random_split(ds, [len(ds) - n_va, n_va],
                              generator=torch.Generator().manual_seed(0))
        tr_loader = DataLoader(tr, batch_size=32)
        va_loader = DataLoader(va, batch_size=32)
        from .probe import collect_hidden, fit_probe
        per_layer = []
        for layer_idx in [0, cfg.layers]:
            H_tr, X_tr = collect_hidden(model, tr_loader, device, layer_idx)
            H_va, X_va = collect_hidden(model, va_loader, device, layer_idx)
            r2 = fit_probe(H_tr, X_tr, H_va, X_va)
            per_layer.append(dict(layer=layer_idx,
                                  r2_per_channel=r2.tolist(),
                                  r2_mean=float(r2.mean())))
            print(f"  [{enc:>9} layer={layer_idx}] r2={r2.tolist()}")
        results.append(dict(
            encoder=enc,
            downstream_val_nll=r["final_val_nll"],
            downstream_val_acc=r["final_val_acc"],
            downstream_best_nll=r["best_val_nll"],
            downstream_best_epoch=r["best_epoch"],
            probes=per_layer,
        ))
    return results


def write_summary(out_dir: str, main_runs, dmodel_runs, carrier_runs,
                  probe_results, mask_results) -> None:
    lines = []

    def aggregate_flags(runs):
        tally = defaultdict(int)
        for r in runs:
            tally[r["convergence_flag"]] += 1
        return dict(tally)

    lines.append("=" * 70)
    lines.append("Convergence flags (across all runs)")
    lines.append("=" * 70)
    lines.append(f"main:    {aggregate_flags(main_runs)}")
    lines.append(f"dmodel:  {aggregate_flags(dmodel_runs)}")
    lines.append(f"carrier: {aggregate_flags(carrier_runs)}")

    def fmt_table(agg, keys, metric_name="best_val_nll"):
        headers = list(keys) + [metric_name, f"{metric_name.replace('nll','acc')}",
                                "best_ep", "flags"]
        lines.append("  " + " | ".join(f"{h:>14}" for h in headers))
        for row in agg:
            vals = [str(row[k]) for k in keys]
            vals.append(f"{row[f'{metric_name}_mean']:.4f}±{row[f'{metric_name}_std']:.4f}")
            vals.append(f"{row['best_val_acc_mean']:.3f}±{row['best_val_acc_std']:.3f}")
            vals.append(f"{row['best_epoch_mean']:.0f}")
            vals.append("/".join(row["flags"]))
            lines.append("  " + " | ".join(f"{v:>14}" for v in vals))

    lines.append("\n" + "=" * 70)
    lines.append("Main sweep (metric = best val NLL across training)")
    lines.append("=" * 70)
    agg = aggregate_runs(main_runs, ["C", "encoder"])
    agg.sort(key=lambda r: (r["C"], r["encoder"]))
    fmt_table(agg, ["C", "encoder"])

    lines.append("\n" + "=" * 70)
    lines.append("d_model sweep")
    lines.append("=" * 70)
    agg = aggregate_runs(dmodel_runs, ["d_model", "encoder"])
    agg.sort(key=lambda r: (r["d_model"], r["encoder"]))
    fmt_table(agg, ["d_model", "encoder"])

    lines.append("\n" + "=" * 70)
    lines.append("Carrier grid (d_model=128, sorted by best NLL)")
    lines.append("=" * 70)
    buckets = defaultdict(list)
    for r in carrier_runs:
        buckets[(r["omega_min"], r["omega_max"])].append(r)
    rows = []
    for k, rs in buckets.items():
        rows.append(dict(
            omega_min=k[0], omega_max=k[1],
            best_nll=np.mean([r["best_val_nll"] for r in rs]),
            best_nll_std=np.std([r["best_val_nll"] for r in rs]),
            best_acc=np.mean([r["best_val_acc"] for r in rs]),
            flags="/".join(r["convergence_flag"] for r in rs),
        ))
    rows.sort(key=lambda r: r["best_nll"])
    lines.append(f"  {'ω_min':>8} {'ω_max':>8}  {'best_nll':>18}  {'best_acc':>10}  flags")
    for r in rows:
        lines.append(
            f"  {r['omega_min']:>8} {r['omega_max']:>8}  "
            f"{r['best_nll']:.4f}±{r['best_nll_std']:.4f}  "
            f"{r['best_acc']:.3f}      {r['flags']}"
        )

    lines.append("\n" + "=" * 70)
    lines.append("Linear-probe channel recovery (R²)")
    lines.append("=" * 70)
    lines.append(f"  {'encoder':>9}  {'layer':>5}  per-channel R²  (mean)")
    for r in probe_results:
        for p in r["probes"]:
            vals = ", ".join(f"{x:.3f}" for x in p["r2_per_channel"])
            lines.append(f"  {r['encoder']:>9}  {p['layer']:>5}  [{vals}]  ({p['r2_mean']:.3f})")

    lines.append("\n" + "=" * 70)
    lines.append("Test-time channel masking (C=4, seed=0)")
    lines.append("=" * 70)
    lines.append(f"  {'encoder':>9}  {'base_acc':>10}  mask accs [ch0,ch1,ch2,ch3]")
    for r in mask_results:
        mk = [f"{p['val_acc']:.3f}" for p in r["per_channel"]]
        lines.append(f"  {r['encoder']:>9}  {r['base_acc']:>10.3f}  [{', '.join(mk)}]")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_full",
                   help="output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", type=int, default=5,
                   help="seed count for main and d_model sweeps")
    p.add_argument("--seeds-grid", type=int, default=3,
                   help="seed count for the carrier-band grid search")
    p.add_argument("--n-series", type=int, default=512)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--Cs", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--d-models", type=int, nargs="+", dest="d_models",
                   default=[64, 128, 256])
    p.add_argument("--encoders", nargs="+",
                   default=["sum", "concat", "fdm", "sum-ortho", "ci", "cat"])
    p.add_argument("--encoders-dmodel", nargs="+",
                   default=["sum", "concat", "fdm", "fdm-learn"])
    p.add_argument("--skip", nargs="+", default=[],
                   choices=["main", "dmodel", "carrier", "probe"],
                   help="skip one or more stages")
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["cosine", "constant"],
                   help="LR schedule for all runs (default: cosine)")
    p.add_argument("--cat-max-C", type=int, default=8,
                   help="skip cat (channel-as-token) at C > this value "
                        "because its attention scales as (C*T)^2")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"out dir={args.out}, epochs={args.epochs}, seeds={args.seeds}")

    t_total = time.time()
    main_runs, mask_runs, dmodel_runs, carrier_runs, probe_results = (
        [], [], [], [], [])

    if "main" not in args.skip:
        main_runs, mask_runs = do_main_sweep(args, device)
        with open(os.path.join(args.out, "main.json"), "w") as f:
            json.dump(dict(runs=main_runs, mask=mask_runs), f, indent=2)
    if "dmodel" not in args.skip:
        dmodel_runs = do_dmodel_sweep(args, device)
        with open(os.path.join(args.out, "dmodel.json"), "w") as f:
            json.dump(dict(runs=dmodel_runs), f, indent=2)
    if "carrier" not in args.skip:
        carrier_runs = do_carrier_grid(args, device)
        with open(os.path.join(args.out, "carrier.json"), "w") as f:
            json.dump(dict(runs=carrier_runs), f, indent=2)
    if "probe" not in args.skip:
        probe_results = do_probe(args, device)
        with open(os.path.join(args.out, "probe.json"), "w") as f:
            json.dump(probe_results, f, indent=2)

    header(f"DONE in {time.time()-t_total:.0f}s total")
    write_summary(args.out, main_runs, dmodel_runs, carrier_runs,
                  probe_results, mask_runs)


if __name__ == "__main__":
    main()
