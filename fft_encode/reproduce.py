"""Reproduce every experiment in the paper from a clean clone.

One command, one output directory. Default is 300-epoch cosine-decay training
uniformly across stages; everything matches what the paper reports.

Usage
-----
    uv run python -m fft_encode.reproduce --out results/
    uv run python -m fft_encode.reproduce --out results/ --stages main etth1
    uv run python -m fft_encode.reproduce --out results/ --epochs 30   # smoke

Stages (default: all)
---------------------
    main        synthetic sweep: C in {4,8,16} x 8 encoders x N seeds
    dmodel      d_model in {64,128,256} x [sum, concat] x N seeds, C=4
    geometry    W_k norms / gram / variance fractions
                (linear at all Cs + sum-ortho at C=4)
    probe       linear-probe channel recovery, C=4
    mask        test-time channel masking, C=4
    etth1       ETTh1 real-data validation, C=7, d_model=56
    convergence analysis-only: derives epochs-to-target from main traces
    bias        channel-bias ablation (linear vs linear-nobias, C=4)
    geom_largen large-N geometry: linear at C=8 with 10x training data,
                probes the distractor-norm noise-floor hypothesis
    main_largen top-tier encoders at C=16 with 10x training data, to
                test whether the C=16 encoder ranking persists when
                the model is no longer data-limited
    pospro_geometry  positional projection geometry: effective rank of
                the positional basis P and principal angles between
                span(W_k) and span(P), for linear vs linear-ppe.
                Discriminates the compression vs orthogonalisation
                mechanism question for linear-ppe.

Outputs
-------
    <out>/main.json
    <out>/dmodel.json
    <out>/geometry.json
    <out>/probe.json
    <out>/mask.json
    <out>/etth1.json
    <out>/convergence.json
    <out>/bias.json
    <out>/geom_largen.json
    <out>/main_largen.json
    <out>/pospro_geometry.json
    <out>/summary.txt        human-readable aggregate of everything
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .data import MultiSignalDataset
from .experiments import RunCfg, _make_scheduler, evaluate
from .model import build_model, nll_loss
from .probe import collect_hidden, fit_probe
from .real_data import ETTh1Dataset
from .runner import train_and_trace

# Canonical encoder list, paper order.
ENCODERS = ["sum", "linear", "sum-ortho", "mlp", "linear-ppe",
            "concat", "ci", "cat"]
# Encoders the linear probe applies to (single-stream hidden states).
ENCODERS_PROBE = ["sum", "linear", "sum-ortho", "mlp", "linear-ppe", "concat"]
# Encoders varied in the d_model sweep.
ENCODERS_DMODEL = ["sum", "concat"]

ALL_STAGES = ["main", "dmodel", "geometry", "probe", "mask", "etth1",
              "convergence", "bias", "geom_largen", "main_largen",
              "pospro_geometry"]


def header(s: str) -> None:
    print(f"\n{'='*70}\n{s}\n{'='*70}", flush=True)


# ---------------------------------------------------------------------------
# Stage 1: synthetic main sweep (C in {4,8,16} x encoders x seeds)
# ---------------------------------------------------------------------------

def stage_main(args, device) -> list:
    header(f"main: C in {args.Cs} x {len(args.encoders)} encoders x "
           f"{args.seeds} seeds @ {args.epochs} epochs")
    runs = []
    for C in args.Cs:
        for enc in args.encoders:
            if enc == "cat" and C > args.cat_max_C:
                print(f"  (skip cat at C={C}; exceeds --cat-max-C={args.cat_max_C})")
                continue
            for s in range(args.seeds):
                cfg = RunCfg(encoder=enc, C=C, seed=s,
                             epochs=args.epochs, n_series=args.n_series)
                t0 = time.time()
                r = train_and_trace(cfg, device, log_every=args.log_every)
                runs.append(r)
                print(f"  [{enc:>10} C={C:>2} s={s}] "
                      f"best_nll={r['best_val_nll']:.4f}@ep{r['best_epoch']:>3} "
                      f"acc={r['best_val_acc']:.3f} "
                      f"flag={r['convergence_flag']:<16} "
                      f"({time.time()-t0:.1f}s)", flush=True)
    return runs


# ---------------------------------------------------------------------------
# Stage 2: d_model sweep
# ---------------------------------------------------------------------------

def stage_dmodel(args, device) -> list:
    header(f"dmodel: d_model in {args.d_models} x "
           f"{len(args.encoders_dmodel)} encoders x {args.seeds} seeds")
    runs = []
    for dm in args.d_models:
        d_ff = 4 * dm
        for enc in args.encoders_dmodel:
            for s in range(args.seeds):
                cfg = RunCfg(encoder=enc, C=4, seed=s,
                             epochs=args.epochs, n_series=args.n_series,
                             d_model=dm, d_ff=d_ff)
                t0 = time.time()
                r = train_and_trace(cfg, device, log_every=args.log_every)
                runs.append(r)
                print(f"  [d={dm:>3} {enc:>10} s={s}] "
                      f"best_nll={r['best_val_nll']:.4f}@ep{r['best_epoch']:>3} "
                      f"flag={r['convergence_flag']:<16} "
                      f"({time.time()-t0:.1f}s)", flush=True)
    return runs


# ---------------------------------------------------------------------------
# Stage 3: encoder geometry (W_k norms, gram, variance)
# ---------------------------------------------------------------------------

def stage_geometry(args, device) -> dict:
    header("geometry: W_k norms / off-diagonal Gram / variance fractions")
    linear_runs = []   # linear at C in {4,8,16}
    ortho_runs = []    # sum-ortho at C=4 (for gram comparison)

    for C in args.Cs:
        for s in range(args.seeds):
            cfg = RunCfg(encoder="linear", C=C, seed=s,
                         epochs=args.epochs, n_series=args.n_series)
            t0 = time.time()
            r, model, val_loader = train_and_trace(
                cfg, device, log_every=args.log_every, return_model=True)
            gram = model.encoder.gram_stats()
            xs = [xb for xb, _ in val_loader]
            x_all = torch.cat(xs, 0).to(device)
            var = model.encoder.variance_stats(x_all)
            linear_runs.append(dict(
                C=C, seed=s,
                best_val_nll=r["best_val_nll"],
                norms=gram["norms"],
                max_off_abs_cos=gram["max_off_abs_cos"],
                mean_off_abs_cos=gram["mean_off_abs_cos"],
                var_fraction=var["var_fraction"],
            ))
            print(f"  [linear   C={C:>2} s={s}] "
                  f"nll={r['best_val_nll']:.4f} "
                  f"mean|cos|={gram['mean_off_abs_cos']:.4f} "
                  f"norms={[round(n,3) for n in gram['norms']]} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    for s in range(args.seeds):
        cfg = RunCfg(encoder="sum-ortho", C=4, seed=s,
                     epochs=args.epochs, n_series=args.n_series)
        t0 = time.time()
        r, model, _ = train_and_trace(
            cfg, device, log_every=args.log_every, return_model=True)
        gram = model.encoder.gram_stats()
        ortho_runs.append(dict(
            C=4, seed=s,
            best_val_nll=r["best_val_nll"],
            norms=gram["norms"],
            max_off_abs_cos=gram["max_off_abs_cos"],
            mean_off_abs_cos=gram["mean_off_abs_cos"],
        ))
        print(f"  [sum-ortho C= 4 s={s}] "
              f"nll={r['best_val_nll']:.4f} "
              f"mean|cos|={gram['mean_off_abs_cos']:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    return dict(linear=linear_runs, sum_ortho=ortho_runs)


# ---------------------------------------------------------------------------
# Stage 4: linear-probe channel recovery
# ---------------------------------------------------------------------------

def stage_probe(args, device) -> list:
    header(f"probe: linear-probe channel recovery at C=4 "
           f"({len(ENCODERS_PROBE)} encoders, seed=0)")
    results = []
    for enc in ENCODERS_PROBE:
        cfg = RunCfg(encoder=enc, C=4, seed=0,
                     epochs=args.epochs, n_series=args.n_series)
        t0 = time.time()
        r, model, _ = train_and_trace(
            cfg, device, log_every=args.log_every, return_model=True)
        # Fresh probe dataset, fixed seed.
        ds = MultiSignalDataset(n_series=256, T=cfg.T, C=cfg.C,
                                K=cfg.bins, seed=99)
        n_va = len(ds) // 5
        tr, va = random_split(ds, [len(ds) - n_va, n_va],
                              generator=torch.Generator().manual_seed(0))
        tr_loader = DataLoader(tr, batch_size=32)
        va_loader = DataLoader(va, batch_size=32)
        per_layer = []
        for li in [0, cfg.layers]:
            H_tr, X_tr = collect_hidden(model, tr_loader, device, li)
            H_va, X_va = collect_hidden(model, va_loader, device, li)
            r2 = fit_probe(H_tr, X_tr, H_va, X_va)
            per_layer.append(dict(layer=li,
                                  r2_per_channel=r2.tolist(),
                                  r2_mean=float(r2.mean())))
        results.append(dict(
            encoder=enc, downstream_best_nll=r["best_val_nll"],
            probes=per_layer,
        ))
        rmeans = [f"{p['r2_mean']:.3f}" for p in per_layer]
        print(f"  [{enc:>10}] R²_mean by layer = {rmeans} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Stage 5: test-time channel masking
# ---------------------------------------------------------------------------

def stage_mask(args, device) -> list:
    header(f"mask: test-time channel masking at C=4 (seed=0)")
    results = []
    for enc in args.encoders:
        cfg = RunCfg(encoder=enc, C=4, seed=0,
                     epochs=args.epochs, n_series=args.n_series)
        t0 = time.time()
        r, model, val_loader = train_and_trace(
            cfg, device, log_every=args.log_every, return_model=True)
        per_channel = []
        for k in range(cfg.C):
            nll_k, acc_k = evaluate(model, val_loader, device, mask_channel=k)
            per_channel.append(dict(channel=k, val_nll=nll_k, val_acc=acc_k))
        results.append(dict(
            encoder=enc,
            base_nll=r["best_val_nll"], base_acc=r["best_val_acc"],
            per_channel=per_channel,
        ))
        accs = [round(p["val_acc"], 3) for p in per_channel]
        print(f"  [{enc:>10}] base_acc={r['best_val_acc']:.3f} "
              f"mask_accs={accs} ({time.time()-t0:.1f}s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Stage 6: ETTh1 real-data validation
# ---------------------------------------------------------------------------

def _train_etth1(encoder: str, seed: int, epochs: int, device: str,
                 log_every: int = 20) -> dict:
    """Train one ETTh1 model. Mirrors fft_encode.run_real.train_one but lives
    here so reproduce.py is self-contained."""
    torch.manual_seed(seed)
    tr = ETTh1Dataset("train", T=160, K=32)
    va = ETTh1Dataset("val", T=160, K=32)
    C = tr.C
    d_model = 56
    assert d_model % C == 0
    n_heads, n_layers, d_ff = 7, 3, 4 * d_model

    train_loader = DataLoader(tr, batch_size=32, shuffle=True)
    val_loader = DataLoader(va, batch_size=32)

    model = build_model(
        kind=encoder, n_channels=C, d_model=d_model, n_bins=32,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    from types import SimpleNamespace
    sched = _make_scheduler(opt, SimpleNamespace(
        lr_schedule="cosine", lr=3e-4, lr_min_frac=0.01, epochs=epochs))

    trace = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = nll_loss(model(x), y) + model.aux_loss()
            opt.zero_grad(); loss.backward()
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
                    pred = logits[:, :-1].argmax(-1)
                    tgt = y[:, 1:]
                    tot_correct += (pred == tgt).sum().item()
                    tot_n += tgt.numel()
                    tot_loss += nll_loss(logits, y).item() * tgt.numel()
            trace.append(dict(epoch=ep + 1,
                              val_nll=tot_loss / tot_n,
                              val_acc=tot_correct / tot_n))
    best = min(trace, key=lambda t: t["val_nll"])
    return dict(encoder=encoder, seed=seed, params=n_params,
                seconds=time.time() - t0, trace=trace,
                best_val_nll=best["val_nll"],
                best_val_acc=best["val_acc"],
                best_epoch=best["epoch"],
                final_val_nll=trace[-1]["val_nll"],
                final_val_acc=trace[-1]["val_acc"])


def stage_etth1(args, device) -> list:
    header(f"etth1: {len(args.encoders)} encoders x {args.seeds} seeds "
           f"@ {args.epochs} epochs (C=7, d_model=56)")
    runs = []
    for enc in args.encoders:
        for s in range(args.seeds):
            t0 = time.time()
            r = _train_etth1(enc, s, args.epochs, device,
                             log_every=args.log_every)
            runs.append(r)
            print(f"  [{enc:>10} s={s}] "
                  f"best_nll={r['best_val_nll']:.4f}@ep{r['best_epoch']:>3} "
                  f"acc={r['best_val_acc']:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return runs


# ---------------------------------------------------------------------------
# Stage 7b: large-N geometry test (distractor norm noise-floor hypothesis)
# ---------------------------------------------------------------------------

def stage_geom_largen(args, device) -> list:
    """Probe the distractor-norm noise-floor hypothesis. The true gradient on
    a distractor channel's W_k is zero in expectation (since the channel is
    independent of the outcome), so the equilibrium norm under L2 weight
    decay is set by stochastic gradient noise that scales as O(1/sqrt(N)).
    More training data should therefore shrink distractor norms (drivers
    largely unaffected). Train `linear` at C=8 with n_series = N_large (10x
    the standard 512), single seed by default; reports the same geometry
    metrics as the `geometry` stage so the two can be compared directly."""
    header(f"geom_largen: linear at C=8 with n_series={args.geom_largen_n_series}, "
           f"{args.geom_largen_seeds} seed(s)")
    out = []
    for s in range(args.geom_largen_seeds):
        cfg = RunCfg(encoder="linear", C=8, seed=s,
                     epochs=args.epochs, n_series=args.geom_largen_n_series)
        t0 = time.time()
        r, model, val_loader = train_and_trace(
            cfg, device, log_every=args.log_every, return_model=True)
        gram = model.encoder.gram_stats()
        xs = [xb for xb, _ in val_loader]
        x_all = torch.cat(xs, 0).to(device)
        var = model.encoder.variance_stats(x_all)
        out.append(dict(
            C=8, seed=s, n_series=args.geom_largen_n_series,
            best_val_nll=r["best_val_nll"],
            norms=gram["norms"],
            max_off_abs_cos=gram["max_off_abs_cos"],
            mean_off_abs_cos=gram["mean_off_abs_cos"],
            var_fraction=var["var_fraction"],
        ))
        print(f"  [linear C=8 s={s} N={args.geom_largen_n_series}] "
              f"nll={r['best_val_nll']:.4f} "
              f"mean|cos|={gram['mean_off_abs_cos']:.4f} "
              f"norms={[round(n,3) for n in gram['norms']]} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage 7c: large-N main sweep at C=16 (top-tier encoders)
# ---------------------------------------------------------------------------

def stage_main_largen(args, device) -> list:
    """Test whether the C=16 encoder ranking observed at the main-sweep
    N=512 persists when the model is no longer data-limited. Trains the
    top-tier encoders (linear, mlp, sum-ortho, linear-ppe, concat by
    default) at the configured C with n_series=N_large, otherwise
    identical to the main stage. Lets us paired-test mlp's lead at
    C=16 in the data-rich regime."""
    header(f"main_largen: {args.main_largen_encoders} at C={args.main_largen_C}, "
           f"n_series={args.main_largen_n_series}, "
           f"{args.main_largen_seeds} seeds")
    out = []
    for enc in args.main_largen_encoders:
        for s in range(args.main_largen_seeds):
            cfg = RunCfg(encoder=enc, C=args.main_largen_C, seed=s,
                         epochs=args.epochs,
                         n_series=args.main_largen_n_series)
            t0 = time.time()
            r = train_and_trace(cfg, device, log_every=args.log_every)
            out.append(r)
            print(f"  [{enc:>10} C={args.main_largen_C} s={s} "
                  f"N={args.main_largen_n_series}] "
                  f"best_nll={r['best_val_nll']:.4f}@ep{r['best_epoch']:>3} "
                  f"acc={r['best_val_acc']:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage 7d: positional-projection geometry (linear-ppe mechanism probe)
# ---------------------------------------------------------------------------

def _principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (in radians, sorted increasing) between span(A)
    and span(B), where A is (n_A, d) and B is (n_B, d). Returns
    min(rank(A), rank(B)) angles. Implementation: QR-decompose A^T and
    B^T to orthonormal bases, then SVD of the inner-product matrix
    gives cosines of the principal angles."""
    Q_A, _ = np.linalg.qr(A.T)
    Q_B, _ = np.linalg.qr(B.T)
    sigmas = np.linalg.svd(Q_A.T @ Q_B, compute_uv=False)
    return np.arccos(np.clip(sigmas, -1.0, 1.0))


def stage_pospro_geometry(args, device) -> dict:
    """Discriminator for the linear-ppe mechanism question. Trains
    linear and linear-ppe at C=4, d=64, args.pospro_seeds seeds.
    Extracts W (per-channel projection rows) and P (the effective
    positional basis: fixed sinusoid for linear, learned W_pos @ p(t)
    + b_pos for linear-ppe). Computes effective rank of P (entropy and
    99%-energy) and principal angles between span(W) and span(P).

    Predictions:
      Compression hypothesis: P_ppe has lower effective rank than
        P_linear; principal angles are similar.
      Orthogonalisation hypothesis: effective ranks similar; min
        principal angle between span(W) and span(P) is larger for
        linear-ppe."""
    header(f"pospro_geometry: linear vs linear-ppe at C=4, "
           f"{args.pospro_seeds} seeds")
    out: dict = {"linear": [], "linear_ppe": []}
    for enc, key in [("linear", "linear"), ("linear-ppe", "linear_ppe")]:
        for s in range(args.pospro_seeds):
            cfg = RunCfg(encoder=enc, C=4, seed=s,
                         epochs=args.epochs, n_series=args.n_series)
            t0 = time.time()
            r, model, _ = train_and_trace(
                cfg, device, log_every=args.log_every, return_model=True)

            # Extract W and the effective positional basis P.
            W = model.encoder.W.detach().cpu().numpy()              # (C, d)
            T = cfg.T
            pos_fixed = model.encoder.pos[:T].cpu().numpy()         # (T, d)
            if enc == "linear-ppe":
                W_pos = model.encoder.pos_proj.weight.detach().cpu().numpy()
                b_pos = model.encoder.pos_proj.bias.detach().cpu().numpy()
                # nn.Linear: out = in @ W_pos.T + b_pos
                P = pos_fixed @ W_pos.T + b_pos
            else:
                P = pos_fixed

            # Effective rank of P (entropy of normalised squared singular values).
            sP = np.linalg.svd(P, compute_uv=False)
            sP_sq = sP ** 2
            pi = sP_sq / sP_sq.sum()
            entropy_rank = float(np.exp(-(pi * np.log(pi + 1e-30)).sum()))
            cumsum = np.cumsum(sP_sq) / sP_sq.sum()
            rank_99 = int((cumsum < 0.99).sum() + 1)

            # Principal angles span(W) vs span(P).
            angles = _principal_angles(W, P)
            angles_deg = np.degrees(angles)

            # Fraction of P's energy in span(W). High = position lives inside
            # the channel subspace (bad for separation); low = position is
            # orthogonal to channels (good).
            Q_W, _ = np.linalg.qr(W.T)  # (d, C) basis for span(W)
            P_proj_W = P @ Q_W          # (T, C)
            frac_in_span_W = float(
                (P_proj_W ** 2).sum() / (P ** 2).sum())

            entry = dict(
                encoder=enc, seed=s,
                best_val_nll=r["best_val_nll"],
                P_singular_values=sP.tolist(),
                P_entropy_rank=entropy_rank,
                P_rank_99=rank_99,
                principal_angles_deg=angles_deg.tolist(),
                min_principal_angle_deg=float(angles_deg.min()),
                mean_principal_angle_deg=float(angles_deg.mean()),
                frac_P_energy_in_span_W=frac_in_span_W,
            )
            out[key].append(entry)
            print(f"  [{enc:>10} s={s}] "
                  f"nll={r['best_val_nll']:.4f} "
                  f"eff_rank={entropy_rank:.2f} "
                  f"rank_99={rank_99} "
                  f"min_angle={angles_deg.min():.1f}° "
                  f"frac(P in span W)={frac_in_span_W:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage 7: channel-bias ablation
# ---------------------------------------------------------------------------

def stage_bias(args, device) -> list:
    """Compare `linear` (with per-channel bias) against `linear-nobias` (bias
    zeroed and not learned) at C=4, 5 seeds. Reports best NLL and Gram-matrix
    stats for both, to test whether the per-channel biases play any role in
    the orthogonalisation."""
    header(f"bias: linear vs. linear-nobias at C=4, {args.seeds} seeds")
    out = []
    for enc in ("linear", "linear-nobias"):
        for s in range(args.seeds):
            cfg = RunCfg(encoder=enc, C=4, seed=s,
                         epochs=args.epochs, n_series=args.n_series)
            t0 = time.time()
            r, model, _ = train_and_trace(
                cfg, device, log_every=args.log_every, return_model=True)
            gram = model.encoder.gram_stats()
            out.append(dict(
                encoder=enc, seed=s,
                best_val_nll=r["best_val_nll"],
                best_val_acc=r["best_val_acc"],
                norms=gram["norms"],
                max_off_abs_cos=gram["max_off_abs_cos"],
                mean_off_abs_cos=gram["mean_off_abs_cos"],
            ))
            print(f"  [{enc:>14} s={s}] "
                  f"nll={r['best_val_nll']:.4f} "
                  f"mean|cos|={gram['mean_off_abs_cos']:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage 8: convergence analysis (derived from main.json traces)
# ---------------------------------------------------------------------------

def stage_convergence(main_runs: list) -> list:
    """For each (encoder, C), compute mean epoch at which val_nll first
    reaches within 0.05 / 0.10 of that run's best. Trace resolution-bound."""
    header("convergence: epochs-to-target derived from main traces")
    groups = defaultdict(list)
    for r in main_runs:
        groups[(r["cfg"]["encoder"], r["cfg"]["C"])].append(r)

    def first_at(trace, threshold):
        for ep, nll in zip(trace["epochs"], trace["val_nll"]):
            if nll <= threshold:
                return ep
        return None

    out = []
    for (enc, C), rs in sorted(groups.items()):
        nlls = [r["best_val_nll"] for r in rs]
        secs = [r["seconds"] for r in rs]
        ep5, ep10 = [], []
        for r in rs:
            best = r["best_val_nll"]
            t5 = first_at(r["trace"], best + 0.05)
            t10 = first_at(r["trace"], best + 0.10)
            if t5 is not None: ep5.append(t5)
            if t10 is not None: ep10.append(t10)
        out.append(dict(
            encoder=enc, C=C,
            best_val_nll=float(np.mean(nlls)),
            mean_ep_to_plus_05=float(np.mean(ep5)) if ep5 else None,
            mean_ep_to_plus_10=float(np.mean(ep10)) if ep10 else None,
            mean_seconds=float(np.mean(secs)),
        ))
        ep5_str = f"{np.mean(ep5):.0f}" if ep5 else "-"
        ep10_str = f"{np.mean(ep10):.0f}" if ep10 else "-"
        print(f"  [{enc:>10} C={C:>2}] "
              f"nll={np.mean(nlls):.4f} "
              f"ep_to_+0.05={ep5_str} ep_to_+0.10={ep10_str} "
              f"sec={np.mean(secs):.1f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def aggregate_main(runs: list) -> list:
    g = defaultdict(list)
    for r in runs:
        g[(r["cfg"]["encoder"], r["cfg"]["C"])].append(r)
    out = []
    for (enc, C), rs in sorted(g.items()):
        nlls = [r["best_val_nll"] for r in rs]
        accs = [r["best_val_acc"] for r in rs]
        out.append(dict(
            encoder=enc, C=C, n_seeds=len(rs),
            nll_mean=float(np.mean(nlls)), nll_std=float(np.std(nlls)),
            acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
        ))
    return out


def write_summary(out_dir: str, results: dict) -> None:
    lines: list[str] = []

    if "main" in results:
        lines.append("=" * 70)
        lines.append("MAIN SWEEP (synthetic, 5 seeds, best val NLL)")
        lines.append("=" * 70)
        agg = aggregate_main(results["main"])
        for C in sorted({r["C"] for r in agg}):
            lines.append(f"\n  C = {C}:")
            lines.append(f"    {'encoder':>12}  {'nll':>17}  {'acc':>17}")
            for r in [a for a in agg if a["C"] == C]:
                lines.append(f"    {r['encoder']:>12}  "
                             f"{r['nll_mean']:.4f}±{r['nll_std']:.4f}  "
                             f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}")

    if "dmodel" in results:
        lines.append("\n" + "=" * 70)
        lines.append("D_MODEL SWEEP (C=4)")
        lines.append("=" * 70)
        g = defaultdict(list)
        for r in results["dmodel"]:
            g[(r["cfg"]["encoder"], r["cfg"]["d_model"])].append(r)
        lines.append(f"  {'encoder':>10}  {'d_model':>7}  {'nll':>17}")
        for (enc, dm), rs in sorted(g.items()):
            nlls = [r["best_val_nll"] for r in rs]
            lines.append(f"  {enc:>10}  {dm:>7}  "
                         f"{np.mean(nlls):.4f}±{np.std(nlls):.4f}")

    if "geometry" in results:
        lines.append("\n" + "=" * 70)
        lines.append("GEOMETRY (W_k norms, off-diagonal cosines, variance)")
        lines.append("=" * 70)
        for C in sorted({r["C"] for r in results["geometry"]["linear"]}):
            rs = [r for r in results["geometry"]["linear"] if r["C"] == C]
            norms = np.mean([r["norms"] for r in rs], axis=0)
            fracs = np.mean([r["var_fraction"] for r in rs], axis=0)
            mc = np.mean([r["mean_off_abs_cos"] for r in rs])
            xc = np.mean([r["max_off_abs_cos"] for r in rs])
            lines.append(f"\n  linear, C={C}: "
                         f"mean|cos|={mc:.4f}, max|cos|={xc:.4f}")
            lines.append("    norms = [" +
                         ", ".join(f"{n:.3f}" for n in norms) + "]")
            lines.append("    var%  = [" +
                         ", ".join(f"{v:.3f}" for v in fracs) + "]")
            if C > 4:
                lines.append(f"    driver/distractor norm ratio = "
                             f"{np.mean(norms[:4]) / np.mean(norms[4:]):.3f}")
                lines.append(f"    driver var fraction = "
                             f"{np.sum(fracs[:4]):.3f}")
        rs = results["geometry"]["sum_ortho"]
        if rs:
            mc = np.mean([r["mean_off_abs_cos"] for r in rs])
            xc = np.mean([r["max_off_abs_cos"] for r in rs])
            lines.append(f"\n  sum-ortho, C=4: "
                         f"mean|cos|={mc:.4f}, max|cos|={xc:.4f}")

    if "probe" in results:
        lines.append("\n" + "=" * 70)
        lines.append("LINEAR PROBE (R² recovering raw channels from hidden)")
        lines.append("=" * 70)
        lines.append(f"  {'encoder':>10}  {'layer':>5}  per-channel R² (mean)")
        for r in results["probe"]:
            for p in r["probes"]:
                vals = ", ".join(f"{x:.3f}" for x in p["r2_per_channel"])
                lines.append(f"  {r['encoder']:>10}  {p['layer']:>5}  "
                             f"[{vals}]  ({p['r2_mean']:.3f})")

    if "mask" in results:
        lines.append("\n" + "=" * 70)
        lines.append("CHANNEL MASK (val acc when each channel zeroed at test)")
        lines.append("=" * 70)
        lines.append(f"  {'encoder':>10}  {'base_acc':>9}  "
                     "per-channel mask acc")
        for r in results["mask"]:
            mk = ", ".join(f"{p['val_acc']:.3f}"
                           for p in r["per_channel"])
            lines.append(f"  {r['encoder']:>10}  {r['base_acc']:>9.3f}  "
                         f"[{mk}]")

    if "etth1" in results:
        lines.append("\n" + "=" * 70)
        lines.append("ETTh1 (real data, 7 variates, d_model=56)")
        lines.append("=" * 70)
        g = defaultdict(list)
        for r in results["etth1"]:
            g[r["encoder"]].append(r)
        lines.append(f"  {'encoder':>10}  {'nll':>17}  {'acc':>17}")
        for enc, rs in sorted(g.items()):
            nlls = [r["best_val_nll"] for r in rs]
            accs = [r["best_val_acc"] for r in rs]
            lines.append(f"  {enc:>10}  "
                         f"{np.mean(nlls):.4f}±{np.std(nlls):.4f}  "
                         f"{np.mean(accs):.3f}±{np.std(accs):.3f}")

    if "pospro_geometry" in results:
        lines.append("\n" + "=" * 70)
        lines.append("POSITIONAL PROJECTION GEOMETRY "
                     "(linear vs linear-ppe at C=4)")
        lines.append("=" * 70)
        for label, key in [("linear", "linear"),
                           ("linear-ppe", "linear_ppe")]:
            rs = results["pospro_geometry"][key]
            if not rs:
                continue
            er = np.array([r["P_entropy_rank"] for r in rs])
            r99 = np.array([r["P_rank_99"] for r in rs])
            mna = np.array([r["min_principal_angle_deg"] for r in rs])
            mea = np.array([r["mean_principal_angle_deg"] for r in rs])
            fps = np.array([r["frac_P_energy_in_span_W"] for r in rs])
            lines.append(
                f"  {label:>10}: "
                f"eff_rank(P) = {er.mean():.2f}±{er.std(ddof=1):.2f}, "
                f"rank_99 = {r99.mean():.1f}, "
                f"min∠(W,P) = {mna.mean():.1f}°±{mna.std(ddof=1):.1f}°, "
                f"mean∠ = {mea.mean():.1f}°, "
                f"P-energy-in-span(W) = {fps.mean():.3f}±"
                f"{fps.std(ddof=1):.3f}"
            )

    if "main_largen" in results:
        lines.append("\n" + "=" * 70)
        lines.append("MAIN SWEEP AT LARGE N (top-tier encoders, "
                     "data-rich regime)")
        lines.append("=" * 70)
        g = defaultdict(list)
        for r in results["main_largen"]:
            g[(r["cfg"]["encoder"], r["cfg"]["C"],
               r["cfg"]["n_series"])].append(r)
        for key in sorted(g.keys()):
            enc, C, N = key
            rs = g[key]
            nlls = [r["best_val_nll"] for r in rs]
            accs = [r["best_val_acc"] for r in rs]
            lines.append(f"  {enc:>12}  C={C}  N={N}  "
                         f"nll={np.mean(nlls):.4f}±{np.std(nlls):.4f}  "
                         f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
                         f"(n={len(rs)})")

    if "geom_largen" in results:
        lines.append("\n" + "=" * 70)
        lines.append("GEOMETRY AT LARGE N (linear at C=8, distractor "
                     "noise-floor probe)")
        lines.append("=" * 70)
        for r in results["geom_largen"]:
            lines.append(f"\n  seed={r['seed']}, n_series={r['n_series']}: "
                         f"nll={r['best_val_nll']:.4f}, "
                         f"mean|cos|={r['mean_off_abs_cos']:.4f}")
            norms = r["norms"]
            fracs = r["var_fraction"]
            lines.append("    norms = [" +
                         ", ".join(f"{n:.3f}" for n in norms) + "]")
            lines.append("    var%  = [" +
                         ", ".join(f"{v:.3f}" for v in fracs) + "]")
            if len(norms) > 4:
                drv = np.mean(norms[:4])
                dst = np.mean(norms[4:])
                lines.append(f"    driver/distractor norm ratio = "
                             f"{drv/dst:.3f}  (driver={drv:.3f}, "
                             f"distractor={dst:.3f})")

    if "bias" in results:
        lines.append("\n" + "=" * 70)
        lines.append("BIAS ABLATION (linear vs. linear-nobias, C=4, 5 seeds)")
        lines.append("=" * 70)
        g = defaultdict(list)
        for r in results["bias"]:
            g[r["encoder"]].append(r)
        lines.append(f"  {'encoder':>14}  {'nll':>17}  {'mean|cos|':>11}  "
                     f"{'max|cos|':>10}")
        for enc, rs in sorted(g.items()):
            nlls = [r["best_val_nll"] for r in rs]
            mc = [r["mean_off_abs_cos"] for r in rs]
            xc = [r["max_off_abs_cos"] for r in rs]
            lines.append(f"  {enc:>14}  "
                         f"{np.mean(nlls):.4f}±{np.std(nlls):.4f}  "
                         f"{np.mean(mc):.4f}  {np.mean(xc):.4f}")

    if "convergence" in results:
        lines.append("\n" + "=" * 70)
        lines.append("CONVERGENCE (epochs to within +0.05 NLL of own best)")
        lines.append("=" * 70)
        lines.append(f"  {'encoder':>10}  {'C':>3}  {'ep_to_+0.05':>11}  "
                     f"{'sec/run':>8}")
        for r in results["convergence"]:
            ep = (f"{r['mean_ep_to_plus_05']:.0f}"
                  if r["mean_ep_to_plus_05"] is not None else "-")
            lines.append(f"  {r['encoder']:>10}  {r['C']:>3}  "
                         f"{ep:>11}  {r['mean_seconds']:>8.1f}")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text, flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-series", type=int, default=512)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--Cs", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--d-models", type=int, nargs="+", dest="d_models",
                   default=[64, 128, 256])
    p.add_argument("--encoders", nargs="+", default=ENCODERS)
    p.add_argument("--encoders-dmodel", nargs="+", default=ENCODERS_DMODEL)
    p.add_argument("--cat-max-C", type=int, default=8,
                   help="skip cat at synthetic C > this (O((CT)^2) attention)")
    p.add_argument("--geom-largen-n-series", type=int, default=5120,
                   help="n_series for the geom_largen stage "
                        "(default: 5120 = 10x the main-sweep N=512)")
    p.add_argument("--geom-largen-seeds", type=int, default=1,
                   help="seed count for the geom_largen stage (default: 1)")
    p.add_argument("--main-largen-encoders", nargs="+",
                   default=["linear", "mlp", "sum-ortho",
                            "linear-ppe", "concat"],
                   help="encoders for the main_largen stage")
    p.add_argument("--main-largen-C", type=int, default=16,
                   help="channel count for the main_largen stage")
    p.add_argument("--main-largen-n-series", type=int, default=5120,
                   help="n_series for the main_largen stage "
                        "(default: 5120 = 10x the main-sweep N=512)")
    p.add_argument("--main-largen-seeds", type=int, default=5,
                   help="seed count for the main_largen stage (default: 5)")
    p.add_argument("--pospro-seeds", type=int, default=5,
                   help="seed count for the pospro_geometry stage "
                        "(default: 5)")
    p.add_argument("--stages", nargs="+", choices=ALL_STAGES,
                   default=ALL_STAGES,
                   help="which stages to run (default: all)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, out={args.out}, epochs={args.epochs}, "
          f"seeds={args.seeds}", flush=True)

    t0 = time.time()
    results: dict = {}

    if "main" in args.stages:
        runs = stage_main(args, device)
        results["main"] = runs
        with open(os.path.join(args.out, "main.json"), "w") as f:
            json.dump(runs, f, indent=2)

    if "dmodel" in args.stages:
        runs = stage_dmodel(args, device)
        results["dmodel"] = runs
        with open(os.path.join(args.out, "dmodel.json"), "w") as f:
            json.dump(runs, f, indent=2)

    if "geometry" in args.stages:
        geo = stage_geometry(args, device)
        results["geometry"] = geo
        with open(os.path.join(args.out, "geometry.json"), "w") as f:
            json.dump(geo, f, indent=2)

    if "probe" in args.stages:
        probe = stage_probe(args, device)
        results["probe"] = probe
        with open(os.path.join(args.out, "probe.json"), "w") as f:
            json.dump(probe, f, indent=2)

    if "mask" in args.stages:
        mk = stage_mask(args, device)
        results["mask"] = mk
        with open(os.path.join(args.out, "mask.json"), "w") as f:
            json.dump(mk, f, indent=2)

    if "etth1" in args.stages:
        et = stage_etth1(args, device)
        results["etth1"] = et
        with open(os.path.join(args.out, "etth1.json"), "w") as f:
            json.dump(et, f, indent=2)

    if "bias" in args.stages:
        bi = stage_bias(args, device)
        results["bias"] = bi
        with open(os.path.join(args.out, "bias.json"), "w") as f:
            json.dump(bi, f, indent=2)

    if "geom_largen" in args.stages:
        gln = stage_geom_largen(args, device)
        results["geom_largen"] = gln
        with open(os.path.join(args.out, "geom_largen.json"), "w") as f:
            json.dump(gln, f, indent=2)

    if "main_largen" in args.stages:
        mln = stage_main_largen(args, device)
        results["main_largen"] = mln
        with open(os.path.join(args.out, "main_largen.json"), "w") as f:
            json.dump(mln, f, indent=2)

    if "pospro_geometry" in args.stages:
        pg = stage_pospro_geometry(args, device)
        results["pospro_geometry"] = pg
        with open(os.path.join(args.out, "pospro_geometry.json"), "w") as f:
            json.dump(pg, f, indent=2)

    if "convergence" in args.stages:
        # Convergence requires main traces. If `main` wasn't run this session,
        # try loading from disk so the stage still works.
        if "main" not in results:
            path = os.path.join(args.out, "main.json")
            if os.path.exists(path):
                with open(path) as f:
                    results["main"] = json.load(f)
            else:
                print("  (skip convergence: no main.json available)")
        if "main" in results:
            conv = stage_convergence(results["main"])
            results["convergence"] = conv
            with open(os.path.join(args.out, "convergence.json"), "w") as f:
                json.dump(conv, f, indent=2)

    # Load any stage outputs already on disk that weren't run this session,
    # so a partial-stage invocation still produces a complete summary.txt.
    _STAGE_FILES = {
        "main": "main.json", "dmodel": "dmodel.json",
        "geometry": "geometry.json", "probe": "probe.json",
        "mask": "mask.json", "etth1": "etth1.json",
        "convergence": "convergence.json", "bias": "bias.json",
        "geom_largen": "geom_largen.json",
        "main_largen": "main_largen.json",
        "pospro_geometry": "pospro_geometry.json",
    }
    for key, fname in _STAGE_FILES.items():
        if key in results:
            continue
        path = os.path.join(args.out, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)

    header(f"DONE in {time.time()-t0:.0f}s total")
    write_summary(args.out, results)


if __name__ == "__main__":
    main()
