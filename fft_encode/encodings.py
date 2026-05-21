"""Two input encoders for multi-channel scalar signals.

Both take input x of shape (B, T, C) where C is the number of channels,
and produce embeddings of shape (B, T, d_model).

- SumEncoding: per-channel scalar -> learned linear projection to d_model,
  then summed across channels (the "naive" baseline).
- FDMChannelEncoding: each channel gets a non-overlapping block of
  d_model // C dims; the scalar value modulates a sinusoidal carrier with
  channel-specific log-spaced frequency, then blocks are concatenated.
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
                 ortho_lambda: float = 1e-2):
        super().__init__()
        self.C = n_channels
        self.d_model = d_model
        self.W = nn.Parameter(torch.empty(n_channels, d_model))
        nn.init.normal_(self.W, std=1.0 / math.sqrt(d_model))
        self.channel_bias = nn.Parameter(torch.zeros(n_channels, d_model))
        self.register_buffer("pos", _sinusoidal_positions(max_len, d_model))
        self.ortho_lambda = ortho_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, d_model)
        per_ch = x.unsqueeze(-1) * self.W + self.channel_bias
        emb = per_ch.sum(dim=2)
        T = emb.shape[1]
        emb = emb + self.pos[:T].unsqueeze(0)
        return emb

    def aux_loss(self) -> torch.Tensor:
        gram = self.W @ self.W.T  # (C, C)
        off = gram - torch.diag_embed(torch.diagonal(gram))
        return self.ortho_lambda * (off ** 2).sum() / 2


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


class FDMChannelEncoding(nn.Module):
    """Per-channel block with sinusoidal carrier modulated by signal value.

    For channel k with value v_k(t) at position t:
        block_k(t)[2i]   = v_k(t) * sin(omega_k * t / base^(2i/d_block))
        block_k(t)[2i+1] = v_k(t) * cos(omega_k * t / base^(2i/d_block))
    Blocks for k=0..C-1 are concatenated -> (B, T, d_model).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        max_len: int = 4096,
        base: float = 10000.0,
        learnable_omega: bool = False,
    ):
        super().__init__()
        assert d_model % n_channels == 0, "d_model must be divisible by C"
        self.C = n_channels
        self.d_model = d_model
        self.d_block = d_model // n_channels
        assert self.d_block % 2 == 0, "per-channel block size must be even"
        self.max_len = max_len

        # Log-spaced carrier frequencies, one per channel (in [omega_lo, omega_hi])
        omegas = torch.logspace(
            math.log10(0.5), math.log10(8.0), steps=n_channels, base=10.0
        )
        if learnable_omega:
            self.omegas = nn.Parameter(omegas)
        else:
            self.register_buffer("omegas", omegas)

        # RoPE-style inverse frequencies inside each block
        i = torch.arange(0, self.d_block, 2).float()
        inv_freq = 1.0 / (base ** (i / self.d_block))  # (d_block/2,)
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_len).float()
        # angles[t, i] = t * inv_freq[i]
        self.register_buffer("positions", positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        assert C == self.C, f"expected {self.C} channels, got {C}"
        device = x.device

        t = self.positions[:T].to(device)  # (T,)
        # per-channel angle: omega_k * t * inv_freq_i  -> (C, T, d_block/2)
        ang = (
            self.omegas.view(C, 1, 1)
            * t.view(1, T, 1)
            * self.inv_freq.view(1, 1, -1)
        )
        sin = torch.sin(ang)  # (C, T, d_block/2)
        cos = torch.cos(ang)
        # interleave sin/cos -> (C, T, d_block)
        carrier = torch.stack([sin, cos], dim=-1).flatten(-2)
        # amplitude modulate by signal value
        # x: (B, T, C) -> (B, C, T, 1)
        amp = x.permute(0, 2, 1).unsqueeze(-1)
        # carrier: (C, T, d_block) -> (1, C, T, d_block)
        modulated = amp * carrier.unsqueeze(0)  # (B, C, T, d_block)
        # concat blocks across channel axis into d_model
        emb = modulated.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        return emb

    def recover_channel(self, h: torch.Tensor, k: int) -> torch.Tensor:
        """Slice channel k's block out of an embedding tensor h: (B, T, d_model)."""
        s = k * self.d_block
        e = s + self.d_block
        return h[..., s:e]


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
