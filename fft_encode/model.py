"""Small causal transformer with a categorical generative head.

Predicts a distribution over K bins for the outcome at every time step,
trained with cross-entropy (NLL of the categorical likelihood). This is a
proper generative head — you can sample y_t ~ Cat(logits_t) and reconstruct
a value from the bin centers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encodings import ConcatEncoding, FDMChannelEncoding, SumEncoding  # noqa: F401


class SignalTransformer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        n_bins: int = 32,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.encoder = encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # Build layers manually so we can capture per-layer hidden states.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_bins)
        self.max_len = max_len

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ):
        # x: (B, T, C) -> logits (B, T, K)
        emb = self.encoder(x)
        T = emb.shape[1]
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=emb.device), diagonal=1
        )
        hiddens = [emb]
        h = emb
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
            hiddens.append(h)
        h = self.norm(h)
        logits = self.head(h)
        if return_hidden:
            return logits, hiddens
        return logits


def build_model(kind: str, n_channels: int, d_model: int, n_bins: int, **kw) -> SignalTransformer:
    if kind == "fdm":
        enc = FDMChannelEncoding(n_channels=n_channels, d_model=d_model)
    elif kind == "fdm-learn":
        enc = FDMChannelEncoding(n_channels=n_channels, d_model=d_model,
                                 learnable_omega=True)
    elif kind == "sum":
        enc = SumEncoding(n_channels=n_channels, d_model=d_model)
    elif kind == "concat":
        enc = ConcatEncoding(n_channels=n_channels, d_model=d_model)
    else:
        raise ValueError(kind)
    return SignalTransformer(enc, d_model=d_model, n_bins=n_bins, **kw)


def nll_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (B, T, K), targets: (B, T) — predict y_t from x_{<=t-1}
    # standard next-step shift
    pred = logits[:, :-1].reshape(-1, logits.size(-1))
    tgt = targets[:, 1:].reshape(-1)
    return F.cross_entropy(pred, tgt)
