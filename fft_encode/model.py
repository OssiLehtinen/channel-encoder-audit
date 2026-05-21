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

from .encodings import (  # noqa: F401
    ConcatEncoding,
    MLPEncoding,
    SumEncoding,
    SumOrthoEncoding,
)


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

    def aux_loss(self) -> torch.Tensor:
        fn = getattr(self.encoder, "aux_loss", None)
        if fn is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return fn()

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


def build_model(kind: str, n_channels: int, d_model: int, n_bins: int, **kw):
    """Return a model whose forward(x: (B,T,C)) -> logits (B,T,K).

    Encoder-only variants share SignalTransformer:
      sum, linear, linear-ortho, mlp, linear-ppe, linear-lpe, concat.
    Architectural baselines return their own module:
      ci (channel-independent, PatchTST-spirit)
      cat (channel-as-token, iTransformer/Crossformer-spirit).

    The name ``sum-perch`` is accepted as an alias for ``linear`` so that
    JSON results produced before the rename still resolve.
    """
    if kind in ("sum", "concat", "mlp"):
        if kind == "sum":
            enc = SumEncoding(n_channels=n_channels, d_model=d_model)
        elif kind == "concat":
            enc = ConcatEncoding(n_channels=n_channels, d_model=d_model)
        elif kind == "mlp":
            enc = MLPEncoding(n_channels=n_channels, d_model=d_model)
        return SignalTransformer(enc, d_model=d_model, n_bins=n_bins, **kw)
    if kind in ("linear-ortho", "sum-ortho"):
        # ``sum-ortho`` is the pre-rename alias kept for backward compat
        # with results_paper/*.json files written before the rename.
        ortho_lambda = kw.pop("ortho_lambda", 1e-2)
        enc = SumOrthoEncoding(n_channels=n_channels, d_model=d_model,
                               ortho_lambda=ortho_lambda)
        return SignalTransformer(enc, d_model=d_model, n_bins=n_bins, **kw)
    if kind in ("linear", "sum-perch"):
        kw.pop("ortho_lambda", None)
        enc = SumOrthoEncoding(n_channels=n_channels, d_model=d_model,
                               ortho_lambda=0.0)
        return SignalTransformer(enc, d_model=d_model, n_bins=n_bins, **kw)
    if kind == "linear-nobias":
        kw.pop("ortho_lambda", None)
        enc = SumOrthoEncoding(n_channels=n_channels, d_model=d_model,
                               ortho_lambda=0.0, use_bias=False)
        return SignalTransformer(enc, d_model=d_model, n_bins=n_bins, **kw)
    if kind == "linear-lpe":
        kw.pop("ortho_lambda", None)
        max_len = kw.get("max_len", 512)
        enc = SumOrthoEncoding(n_channels=n_channels, d_model=d_model,
                               ortho_lambda=0.0, learned_pos=True,
                               max_len=max_len)
        return SignalTransformer(enc, d_model=d_model, n_bins=n_bins, **kw)
    if kind == "linear-ppe":
        kw.pop("ortho_lambda", None)
        enc = SumOrthoEncoding(n_channels=n_channels, d_model=d_model,
                               ortho_lambda=0.0, project_pos=True)
        return SignalTransformer(enc, d_model=d_model, n_bins=n_bins, **kw)
    if kind in ("ci", "cat"):
        from .baselines import (
            ChannelAsTokenTransformer,
            ChannelIndependentTransformer,
        )
        cls = ChannelIndependentTransformer if kind == "ci" else ChannelAsTokenTransformer
        # strip unused kwargs that SignalTransformer accepts (max_len kept)
        return cls(
            n_channels=n_channels, d_model=d_model, n_bins=n_bins,
            n_heads=kw.get("n_heads", 4), n_layers=kw.get("n_layers", 3),
            d_ff=kw.get("d_ff", 4 * d_model), dropout=kw.get("dropout", 0.1),
            max_len=kw.get("max_len", 512),
        )
    raise ValueError(kind)


def nll_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (B, T, K), targets: (B, T) — predict y_t from x_{<=t-1}
    # standard next-step shift
    pred = logits[:, :-1].reshape(-1, logits.size(-1))
    tgt = targets[:, 1:].reshape(-1)
    return F.cross_entropy(pred, tgt)


def mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-step MSE for a continuous-target head.

    preds: (B, T, 1) or (B, T) scalar predictions, targets: (B, T) continuous.
    """
    if preds.dim() == 3 and preds.size(-1) == 1:
        preds = preds.squeeze(-1)
    pred = preds[:, :-1]
    tgt = targets[:, 1:]
    return F.mse_loss(pred, tgt)
