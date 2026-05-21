"""Input encoders for multi-channel scalar signals.

All take input x of shape (B, T, C) where C is the number of channels,
and produce embeddings of shape (B, T, d_model).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SumEncoding(nn.Module):
    def __init__(self, n_channels: int, d_model: int, max_len: int = 4096):
        super().__init__()
        self.proj = nn.Linear(1, d_model, bias=False)
        # one learned per-channel embedding, summed in
        self.channel_emb = nn.Embedding(n_channels, d_model)
        self.register_buffer("pos", _sinusoidal_positions(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        # project each channel value, add channel embedding, then sum across C
        per_ch = self.proj(x.unsqueeze(-1))  # (B, T, C, d)
        ch_ids = torch.arange(C, device=x.device)
        per_ch = per_ch + self.channel_emb(ch_ids)  # broadcast over B,T
        emb = per_ch.sum(dim=2)  # (B, T, d)
        emb = emb + self.pos[:T].unsqueeze(0)
        return emb


class SumOrthoEncoding(nn.Module):
    """Summation with a soft orthogonality regulariser on per-channel
    projections.

    Each channel k has its own learned projection W_k in R^{d_model} (no
    sharing across channels). Embeddings are summed. The encoder exposes an
    auxiliary loss:

        L_ortho = lambda * sum_{i != j} (W_i . W_j)^2 / 2

    Intended as a drop-in replacement for SumEncoding that gives the
    optimiser a nudge toward channel-separated subspaces without imposing
    hard orthogonality. If separation helps, the regulariser accelerates
    what the optimiser would do anyway; if channels genuinely need to share
    directions, the task loss overwhelms the regulariser.
    """

    def __init__(self, n_channels: int, d_model: int, max_len: int = 4096,
                 ortho_lambda: float = 1e-2, learned_pos: bool = False,
                 project_pos: bool = False, use_bias: bool = True):
        super().__init__()
        self.C = n_channels
        self.d_model = d_model
        self.W = nn.Parameter(torch.empty(n_channels, d_model))
        nn.init.normal_(self.W, std=1.0 / math.sqrt(d_model))
        if use_bias:
            self.channel_bias = nn.Parameter(torch.zeros(n_channels, d_model))
        else:
            self.register_buffer("channel_bias",
                                 torch.zeros(n_channels, d_model))
        if learned_pos:
            self.pos = nn.Embedding(max_len, d_model)
            self._learned_pos = True
        else:
            self.register_buffer("pos", _sinusoidal_positions(max_len, d_model))
            self._learned_pos = False
        self._project_pos = project_pos
        if project_pos:
            self.pos_proj = nn.Linear(d_model, d_model)
        self.ortho_lambda = ortho_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, d_model)
        per_ch = x.unsqueeze(-1) * self.W + self.channel_bias
        emb = per_ch.sum(dim=2)
        T = emb.shape[1]
        if self._learned_pos:
            pos = self.pos(torch.arange(T, device=emb.device))
        else:
            pos = self.pos[:T].unsqueeze(0)
        if self._project_pos:
            pos = self.pos_proj(pos)
        emb = emb + pos
        return emb

    def aux_loss(self) -> torch.Tensor:
        if self.ortho_lambda == 0.0:
            return torch.zeros((), device=self.W.device)
        gram = self.W @ self.W.T  # (C, C)
        off = gram - torch.diag_embed(torch.diagonal(gram))
        return self.ortho_lambda * (off ** 2).sum() / 2

    def gram_stats(self) -> dict:
        """Report structure of the learned per-channel projections."""
        with torch.no_grad():
            W = self.W.detach()
            norms = W.norm(dim=-1)
            Wn = W / (norms.unsqueeze(-1) + 1e-12)
            cos = Wn @ Wn.T  # (C, C)
            off_mask = ~torch.eye(W.shape[0], dtype=torch.bool, device=W.device)
            off_abs = cos.abs()[off_mask]
            return dict(
                norms=norms.cpu().tolist(),
                max_off_abs_cos=float(off_abs.max()),
                mean_off_abs_cos=float(off_abs.mean()),
                cos_matrix=cos.cpu().tolist(),
            )

    def variance_stats(self, x: torch.Tensor) -> dict:
        """Per-channel variance contribution to the embedding.

        Returns the fraction of total embedding variance attributable to
        each channel, computed empirically over the input batch x.
        """
        with torch.no_grad():
            # x: (B, T, C) -> per-channel contribution: (B, T, C, d_model)
            per_ch = x.unsqueeze(-1) * self.W + self.channel_bias
            # variance of each channel's contribution across (B, T)
            flat = per_ch.reshape(-1, self.C, self.d_model)  # (B*T, C, d)
            var_per_ch = flat.var(dim=0).sum(dim=-1)  # (C,)
            total_var = var_per_ch.sum()
            fracs = var_per_ch / (total_var + 1e-12)
            norms = self.W.detach().norm(dim=-1)
            return dict(
                var_per_channel=var_per_ch.cpu().tolist(),
                var_fraction=fracs.cpu().tolist(),
                total_var=float(total_var),
                norms=norms.cpu().tolist(),
            )


class MLPEncoding(nn.Module):
    """Two-layer MLP applied to the full channel vector at each time step.

    h(t) = Linear(GELU(Linear(v(t)))) + p(t), where v(t) in R^C.
    Tests whether a nonlinear input projection improves over Linear.
    """

    def __init__(self, n_channels: int, d_model: int, max_len: int = 4096,
                 d_hidden: int | None = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_model
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        self.register_buffer("pos", _sinusoidal_positions(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.mlp(x)
        T = emb.shape[1]
        emb = emb + self.pos[:T].unsqueeze(0)
        return emb


class ConcatEncoding(nn.Module):
    """Per-channel learned projection, concatenated; no carrier modulation.

    Isolates the contribution of block partitioning (channel identity is
    preserved by construction) from the contribution of the sinusoidal
    carrier in FDM. Adds a standard sinusoidal positional encoding.
    """

    def __init__(self, n_channels: int, d_model: int, max_len: int = 4096):
        super().__init__()
        assert d_model % n_channels == 0, "d_model must be divisible by C"
        self.C = n_channels
        self.d_block = d_model // n_channels
        # one (1 -> d_block) linear per channel, packed as a single weight
        self.weight = nn.Parameter(torch.empty(n_channels, self.d_block))
        self.bias = nn.Parameter(torch.zeros(n_channels, self.d_block))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.d_block))
        self.register_buffer("pos", _sinusoidal_positions(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, C, d_block)
        per_ch = x.unsqueeze(-1) * self.weight + self.bias
        B, T, C, d_block = per_ch.shape
        emb = per_ch.reshape(B, T, C * d_block)
        emb = emb + self.pos[:T].unsqueeze(0)
        return emb


def _sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
