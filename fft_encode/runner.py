"""Shared training utility with per-epoch tracing and convergence flagging."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .data import MultiSignalDataset
from .experiments import RunCfg, _make_scheduler, evaluate, evaluate_mse
from .model import build_model, mse_loss, nll_loss


@dataclass
class Trace:
    epochs: list        # [int]
    val_nll: list       # [float]
    val_acc: list       # [float]
    train_nll: list     # [float] (end-of-epoch avg)


def train_and_trace(
    cfg: RunCfg,
    device: str,
    log_every: int = 5,
    return_model: bool = False,
    build_model_fn=None,
):
    """Train for cfg.epochs, log val metrics every log_every epochs (+first+last).

    Returns dict with cfg (asdict), trace (Trace asdict), params, seconds,
    best_val_nll, best_epoch, final_val_nll, final_val_acc, val_acc_at_best,
    convergence_flag in {"converged", "still-improving", "overfitting",
    "diverged"}.
    """
    torch.manual_seed(cfg.seed)
    ds = MultiSignalDataset(
        n_series=cfg.n_series, T=cfg.T, C=cfg.C, K=cfg.bins, seed=cfg.seed,
    )
    n_val = max(32, len(ds) // 10)
    tr, va = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=cfg.batch_size)

    if build_model_fn is None:
        model = build_model(
            kind=cfg.encoder, n_channels=cfg.C,
            d_model=cfg.d_model, n_bins=cfg.bins,
            n_heads=cfg.heads, n_layers=cfg.layers,
            d_ff=cfg.d_ff, dropout=cfg.dropout,
        ).to(device)
    else:
        model = build_model_fn(cfg).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = _make_scheduler(opt, cfg)

    import time
    t0 = time.time()
    trace = Trace(epochs=[], val_nll=[], val_acc=[], train_nll=[])
    # Per-checkpoint per-example NLLs, parallel to trace.epochs. Not saved
    # to the trace (kept local); we extract only the best-epoch entry
    # into the run dict.
    per_example_at_checkpoint: list = []
    for ep in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = nll_loss(model(x), y) + model.aux_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        if sched is not None:
            sched.step()
        epoch_train_nll = running / max(n, 1)
        if (ep + 1) % log_every == 0 or ep == 0 or ep == cfg.epochs - 1:
            vnll, vacc, per_ex = evaluate(
                model, val_loader, device, return_per_example=True)
            trace.epochs.append(ep + 1)
            trace.val_nll.append(vnll)
            trace.val_acc.append(vacc)
            trace.train_nll.append(epoch_train_nll)
            per_example_at_checkpoint.append(per_ex)
    elapsed = time.time() - t0

    vnll = np.array(trace.val_nll)
    vacc = np.array(trace.val_acc)
    best_idx = int(np.argmin(vnll))
    best_nll = float(vnll[best_idx])
    best_epoch = int(trace.epochs[best_idx])
    final_nll = float(vnll[-1])
    final_acc = float(vacc[-1])
    best_acc = float(vacc[best_idx])
    best_per_example_nll = per_example_at_checkpoint[best_idx]
    final_per_example_nll = per_example_at_checkpoint[-1]

    # Flag convergence
    last_epoch = trace.epochs[-1]
    if np.isnan(vnll).any() or np.isinf(vnll).any():
        flag = "diverged"
    elif best_epoch >= last_epoch - log_every:
        # still at or near the end -> likely still improving
        # but check: is the last window flat?
        if len(vnll) >= 3 and (vnll[-3] - vnll[-1]) < 0.005:
            flag = "converged"
        else:
            flag = "still-improving"
    elif final_nll > best_nll + 0.01:
        flag = "overfitting"
    else:
        flag = "converged"

    out = dict(
        cfg=asdict(cfg), params=params, seconds=elapsed,
        trace=asdict(trace),
        best_val_nll=best_nll, best_epoch=best_epoch,
        best_val_acc=best_acc,
        final_val_nll=final_nll, final_val_acc=final_acc,
        convergence_flag=flag,
        # Per-sequence NLLs over the val split. Each list has one float
        # per val series (their mean cross-entropy over positions). Use
        # these for example-level bootstrap CIs alongside seed-level
        # resampling.
        best_per_example_nll=best_per_example_nll,
        final_per_example_nll=final_per_example_nll,
    )
    if return_model:
        return out, model, val_loader
    return out


def train_and_trace_mse(
    cfg: RunCfg,
    device: str,
    log_every: int = 5,
    return_model: bool = False,
):
    """Sibling of train_and_trace that swaps the categorical bin head for a
    scalar regression head and trains with MSE. Same encoder, same backbone,
    same val split (paired seeds across both targets).

    Returns a dict with cfg, params, seconds, trace (with val_mse, val_r2),
    best_val_mse, best_epoch, best_val_r2, final_val_mse, final_val_r2,
    convergence_flag, best/final_per_example_mse.
    """
    torch.manual_seed(cfg.seed)
    ds = MultiSignalDataset(
        n_series=cfg.n_series, T=cfg.T, C=cfg.C, K=cfg.bins, seed=cfg.seed,
        return_continuous=True,
    )
    n_val = max(32, len(ds) // 10)
    tr, va = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=cfg.batch_size)

    # Same encoder + backbone, but a scalar (n_bins=1) head.
    model = build_model(
        kind=cfg.encoder, n_channels=cfg.C,
        d_model=cfg.d_model, n_bins=1,
        n_heads=cfg.heads, n_layers=cfg.layers,
        d_ff=cfg.d_ff, dropout=cfg.dropout,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = _make_scheduler(opt, cfg)

    import time
    t0 = time.time()
    val_mses: list[float] = []
    val_r2s: list[float] = []
    train_losses: list[float] = []
    epochs_trace: list[int] = []
    per_example_at_checkpoint: list[list] = []
    for ep in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = mse_loss(model(x), y) + model.aux_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        if sched is not None:
            sched.step()
        epoch_train_loss = running / max(n, 1)
        if (ep + 1) % log_every == 0 or ep == 0 or ep == cfg.epochs - 1:
            vmse, vr2, per_ex = evaluate_mse(
                model, val_loader, device, return_per_example=True)
            epochs_trace.append(ep + 1)
            val_mses.append(vmse)
            val_r2s.append(vr2)
            train_losses.append(epoch_train_loss)
            per_example_at_checkpoint.append(per_ex)
    elapsed = time.time() - t0

    vmse_arr = np.array(val_mses)
    vr2_arr = np.array(val_r2s)
    best_idx = int(np.argmin(vmse_arr))
    best_mse = float(vmse_arr[best_idx])
    best_epoch = int(epochs_trace[best_idx])
    final_mse = float(vmse_arr[-1])
    final_r2 = float(vr2_arr[-1])
    best_r2 = float(vr2_arr[best_idx])

    last_epoch = epochs_trace[-1]
    if np.isnan(vmse_arr).any() or np.isinf(vmse_arr).any():
        flag = "diverged"
    elif best_epoch >= last_epoch - log_every:
        if len(vmse_arr) >= 3 and (vmse_arr[-3] - vmse_arr[-1]) < 0.005:
            flag = "converged"
        else:
            flag = "still-improving"
    elif final_mse > best_mse + 0.01:
        flag = "overfitting"
    else:
        flag = "converged"

    out = dict(
        cfg=asdict(cfg), params=params, seconds=elapsed,
        trace=dict(epochs=epochs_trace, val_mse=val_mses, val_r2=val_r2s,
                   train_loss=train_losses),
        best_val_mse=best_mse, best_epoch=best_epoch,
        best_val_r2=best_r2,
        final_val_mse=final_mse, final_val_r2=final_r2,
        convergence_flag=flag,
        best_per_example_mse=per_example_at_checkpoint[best_idx],
        final_per_example_mse=per_example_at_checkpoint[-1],
    )
    if return_model:
        return out, model, val_loader
    return out
