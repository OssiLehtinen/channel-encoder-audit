"""ETTh1 (Electricity Transformer Temperature, hourly) real-dataset adapter.

Fetches the CSV from the ETDataset GitHub mirror on first use, caches it
locally, and wraps it in a Dataset compatible with the training code:
  - 7 channels standardised per column on the train split,
  - overlapping windows of length T,
  - target: next-step value of `OT` (oil temperature), binned into K
    quantile bins from the training split.
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/"
    "ETDataset/main/ETT-small/ETTh1.csv"
)
DEFAULT_CACHE = Path(os.environ.get(
    "FFT_ENCODE_CACHE", str(Path.home() / ".cache" / "fft_encode")))
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET = "OT"


def _download() -> Path:
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    out = DEFAULT_CACHE / "ETTh1.csv"
    if not out.exists():
        print(f"downloading ETTh1.csv to {out}")
        with urllib.request.urlopen(ETTH1_URL) as resp:
            out.write_bytes(resp.read())
    return out


def _load_matrix() -> np.ndarray:
    """Return (T_total, C) float32 matrix of channels in `CHANNELS` order."""
    path = _download()
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row[c]) for c in CHANNELS])
    return np.array(rows, dtype=np.float32)


class ETTh1Dataset(Dataset):
    """Windowed ETTh1 sequences for next-step categorical prediction.

    Channels are all standardised using train-split mean/std. Bin edges for
    `OT` are quantiles computed on the training split only (no leakage).
    """

    def __init__(
        self,
        split: str = "train",
        T: int = 160,
        K: int = 32,
        stride: int = 8,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ):
        assert split in ("train", "val", "test")
        mat = _load_matrix()  # (N, C)
        n_total = mat.shape[0]
        n_tr = int(n_total * train_frac)
        n_va = int(n_total * val_frac)
        tr = mat[:n_tr]
        va = mat[n_tr : n_tr + n_va]
        te = mat[n_tr + n_va :]

        mean = tr.mean(axis=0, keepdims=True)
        std = tr.std(axis=0, keepdims=True) + 1e-8
        self.mean = mean
        self.std = std

        # Bin edges from training target channel
        target_idx = CHANNELS.index(TARGET)
        tr_target_z = (tr[:, target_idx] - mean[0, target_idx]) / std[0, target_idx]
        qs = np.linspace(0, 1, K + 1)
        edges = np.quantile(tr_target_z, qs)
        edges[0] -= 1e-3
        edges[-1] += 1e-3

        if split == "train":
            data = tr
        elif split == "val":
            data = va
        else:
            data = te

        z = (data - mean) / std
        # Build windows
        idxs = list(range(0, z.shape[0] - T, stride))
        self.windows = np.stack([z[i : i + T] for i in idxs], axis=0)
        tgt_cont = self.windows[:, :, target_idx]
        self.targets = np.clip(np.digitize(tgt_cont, edges) - 1, 0, K - 1)

        self.C = len(CHANNELS)
        self.T = T
        self.K = K
        self.target_channel = target_idx
        self.bin_edges = torch.from_numpy(edges.astype(np.float32))

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx].astype(np.float32))  # (T, C)
        y = torch.from_numpy(self.targets[idx].astype(np.int64))    # (T,)
        return x, y
