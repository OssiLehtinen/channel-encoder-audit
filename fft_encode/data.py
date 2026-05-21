"""Synthetic multi-signal dataset.

We generate C input channels, each a smooth stochastic process (sum of a few
sinusoids with random phase + AR(1) noise). The outcome y_t is a deterministic
non-linear function of *several* channels at *different lags*, so the model
must keep channel identity intact through the encoder to predict it well.

Outcome is binned into K classes for a categorical generative head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def _make_signal(T: int, rng: np.random.Generator, n_modes: int = 3) -> np.ndarray:
    t = np.arange(T)
    sig = np.zeros(T)
    for _ in range(n_modes):
        freq = rng.uniform(0.005, 0.08)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.5)
        sig += amp * np.sin(2 * np.pi * freq * t + phase)
    # AR(1) noise
    noise = np.zeros(T)
    for i in range(1, T):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.3)
    sig = sig + noise
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    return sig.astype(np.float32)


def _outcome(signals: np.ndarray) -> np.ndarray:
    """signals: (C, T) -> y: (T,) continuous target.

    Outcome depends only on the first 4 channels via lagged, multiplicative
    interactions; any additional channels (C>4) are *distractors*. This lets
    us study how each encoder copes with irrelevant signals as C grows.
    """
    C, T = signals.shape
    assert C >= 4, "need >=4 driver channels"
    s = signals
    y = np.zeros(T, dtype=np.float32)
    L1, L2 = 3, 7
    y[L2:] = (
        np.tanh(s[0, L2 - L1 : T - L1] * s[1, L2:])
        + 0.6 * np.sin(1.3 * s[2, : T - L2])
        + 0.4 * (s[3, L2:] > 0.0).astype(np.float32) * s[0, L2:]
    )
    return y


class MultiSignalDataset(Dataset):
    def __init__(
        self,
        n_series: int = 256,
        T: int = 256,
        C: int = 4,
        K: int = 32,
        seed: int = 0,
    ):
        self.C = C
        self.T = T
        self.K = K
        rng = np.random.default_rng(seed)

        all_signals = np.stack(
            [
                np.stack([_make_signal(T, rng) for _ in range(C)], axis=0)
                for _ in range(n_series)
            ],
            axis=0,
        )  # (N, C, T)
        all_y = np.stack([_outcome(all_signals[i]) for i in range(n_series)], axis=0)

        # Global bin edges from the pooled outcome distribution (quantile bins)
        flat = all_y.reshape(-1)
        qs = np.linspace(0, 1, K + 1)
        edges = np.quantile(flat, qs)
        edges[0] -= 1e-3
        edges[-1] += 1e-3
        bins = np.clip(np.digitize(all_y, edges) - 1, 0, K - 1).astype(np.int64)

        self.signals = torch.from_numpy(all_signals)  # (N, C, T) float32
        self.targets = torch.from_numpy(bins)  # (N, T) long
        self.bin_edges = torch.from_numpy(edges.astype(np.float32))

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int):
        # signals: (T, C), targets: (T,)
        return self.signals[idx].transpose(0, 1), self.targets[idx]
