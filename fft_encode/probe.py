"""Linear-probe helpers: closed-form ridge from a hidden state back to the
raw input. Invoked from fft_encode.reproduce; this module exposes the two
building blocks (``collect_hidden`` and ``fit_probe``) without a CLI."""

from __future__ import annotations

import numpy as np
import torch


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


