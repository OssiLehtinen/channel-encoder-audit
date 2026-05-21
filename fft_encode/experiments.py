"""Training primitives shared by reproduce.py and runner.py.

Exposes the config dataclass (``RunCfg``), the cosine/constant LR scheduler
helper (``_make_scheduler``), the single-pass evaluator (``evaluate``), and
a one-shot training driver (``train_run``) used by ad-hoc callers. The
sweep loop itself lives in ``fft_encode.reproduce``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .data import MultiSignalDataset
from .model import build_model, nll_loss


def _make_scheduler(opt, cfg):
    """Return an LR scheduler that steps once per epoch, or None for constant LR."""
    if cfg.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.lr * cfg.lr_min_frac,
        )
    if cfg.lr_schedule == "constant":
        return None
    raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule}")


@dataclass
class RunCfg:
    encoder: str
    C: int
    seed: int
    epochs: int = 12
    n_series: int = 512
    T: int = 160
    bins: int = 32
    d_model: int = 64
    heads: int = 4
    layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 32
    lr_schedule: str = "cosine"   # {"cosine", "constant"}
    lr_min_frac: float = 0.01      # cosine floor as fraction of lr


def make_loaders(cfg: RunCfg):
    ds = MultiSignalDataset(
        n_series=cfg.n_series, T=cfg.T, C=cfg.C, K=cfg.bins, seed=cfg.seed
    )
    n_val = max(32, len(ds) // 10)
    n_train = len(ds) - n_val
    tr, va = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
    )
    return (
        DataLoader(tr, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(va, batch_size=cfg.batch_size),
        ds,
    )


def evaluate(model, loader, device, mask_channel: int | None = None):
    model.eval()
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if mask_channel is not None:
                x = x.clone()
                x[..., mask_channel] = 0.0
            logits = model(x)
            loss = nll_loss(logits, y)
            pred = logits[:, :-1].argmax(-1)
            tgt = y[:, 1:]
            tot_correct += (pred == tgt).sum().item()
            tot_n += tgt.numel()
            tot_loss += loss.item() * tgt.numel()
    return tot_loss / tot_n, tot_correct / tot_n


def train_run(cfg: RunCfg, device, return_model: bool = False):
    torch.manual_seed(cfg.seed)
    train_loader, val_loader, _ = make_loaders(cfg)
    model = build_model(
        kind=cfg.encoder,
        n_channels=cfg.C,
        d_model=cfg.d_model,
        n_bins=cfg.bins,
        n_heads=cfg.heads,
        n_layers=cfg.layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = _make_scheduler(opt, cfg)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = nll_loss(model(x), y) + model.aux_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched is not None:
            sched.step()
    val_nll, val_acc = evaluate(model, val_loader, device)
    elapsed = time.time() - t0
    out = dict(
        cfg=asdict(cfg),
        params=n_params,
        val_nll=val_nll,
        val_acc=val_acc,
        seconds=elapsed,
    )
    if return_model:
        return out, model, val_loader
    return out


