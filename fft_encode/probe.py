"""Linear-probe channel-recovery experiment.

Train a fresh model per encoder (C=4), then freeze and fit per-channel linear
probes that predict each input channel's raw value from (a) the input
embedding and (b) the final transformer hidden state. Report R^2 on a held-
out split. The intent: if the encoding preserves channel identity, a linear
probe should recover each channel cleanly.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .data import MultiSignalDataset
from .experiments import RunCfg, train_run


def collect_hidden(model, loader, device, layer_idx: int):
    """Run model.forward(return_hidden=True) over loader; return (H, X).

    H: (N*T, d_model) hidden activations at layer_idx
    X: (N*T, C) raw inputs
    """
    model.eval()
    Hs, Xs = [], []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            _logits, hidden = model(x, return_hidden=True)
            h = hidden[layer_idx]  # (B, T, d)
            Hs.append(h.reshape(-1, h.size(-1)).cpu())
            Xs.append(x.reshape(-1, x.size(-1)).cpu())
    return torch.cat(Hs, 0), torch.cat(Xs, 0)


def fit_probe(H_tr, X_tr, H_va, X_va, ridge: float = 1e-3) -> np.ndarray:
    """Closed-form ridge regression H -> X. Returns per-channel R^2 on val."""
    H_tr = H_tr.double()
    X_tr = X_tr.double()
    H_va = H_va.double()
    X_va = X_va.double()
    d = H_tr.shape[1]
    A = H_tr.T @ H_tr + ridge * torch.eye(d, dtype=torch.float64)
    B = H_tr.T @ X_tr
    W = torch.linalg.solve(A, B)  # (d, C)
    pred = H_va @ W
    ss_res = ((X_va - pred) ** 2).sum(0)
    ss_tot = ((X_va - X_va.mean(0, keepdim=True)) ** 2).sum(0)
    r2 = 1.0 - (ss_res / ss_tot.clamp_min(1e-12))
    return r2.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="probe_results.json")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    results = []
    for enc in ["fdm", "concat", "sum"]:
        cfg = RunCfg(encoder=enc, C=4, seed=args.seed, epochs=args.epochs)
        out, model, val_loader = train_run(cfg, device, return_model=True)
        # Build a separate train loader (we need probe-train data the model
        # has seen, but we just reuse the dataset with a new split.)
        from .data import MultiSignalDataset

        ds = MultiSignalDataset(n_series=256, T=cfg.T, C=cfg.C, K=cfg.bins, seed=99)
        n_va = len(ds) // 5
        n_tr = len(ds) - n_va
        tr, va = random_split(ds, [n_tr, n_va],
                              generator=torch.Generator().manual_seed(0))
        tr_loader = DataLoader(tr, batch_size=32)
        va_loader = DataLoader(va, batch_size=32)

        per_layer = []
        # layer 0 = input embedding; layer L = final hidden
        for layer_idx in [0, cfg.layers]:
            H_tr, X_tr = collect_hidden(model, tr_loader, device, layer_idx)
            H_va, X_va = collect_hidden(model, va_loader, device, layer_idx)
            r2 = fit_probe(H_tr, X_tr, H_va, X_va)
            per_layer.append(dict(layer=layer_idx, r2_per_channel=r2.tolist(),
                                  r2_mean=float(r2.mean())))
            print(f"[probe {enc} layer={layer_idx}] r2={r2.tolist()}")
        results.append(dict(
            encoder=enc, downstream_val_acc=out["val_acc"],
            downstream_val_nll=out["val_nll"], probes=per_layer,
        ))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
