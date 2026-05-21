"""Alternative architectural baselines: channel-independent (PatchTST-style)
and channel-as-token (iTransformer/Crossformer-spirit).

Both consume the same input shape (B, T, C) and emit (B, T, K) logits, so
they are drop-in replacements for the main-sweep SignalTransformer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encodings import _sinusoidal_positions


def _make_encoder_layers(d_model: int, n_heads: int, n_layers: int,
                         d_ff: int, dropout: float) -> nn.ModuleList:
    return nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        for _ in range(n_layers)
    ])


def _zero_aux(device) -> torch.Tensor:
    return torch.zeros((), device=device)


class ChannelIndependentTransformer(nn.Module):
    """PatchTST-spirit: one transformer run independently per channel with
    shared weights, concatenated at the head.

    Per-channel capacity is full d_model; total compute is C× (B*C batch).
    Head takes the concatenated C hidden states at each time step and maps
    them to K logits.
    """

    def __init__(self, n_channels: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, n_bins: int,
                 dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.C = n_channels
        self.d_model = d_model
        self.proj = nn.Linear(1, d_model)
        self.register_buffer("pos", _sinusoidal_positions(max_len, d_model))
        self.layers = _make_encoder_layers(d_model, n_heads, n_layers, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(n_channels * d_model, n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B*C, T, 1) processed independently
        B, T, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * C, T, 1)
        h = self.proj(x_flat) + self.pos[:T].unsqueeze(0)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        h = self.norm(h)
        # back to (B, T, C*d_model)
        h = h.reshape(B, C, T, -1).permute(0, 2, 1, 3).reshape(B, T, C * self.d_model)
        return self.head(h)

    def aux_loss(self) -> torch.Tensor:
        return _zero_aux(next(self.parameters()).device)


class ChannelAsTokenTransformer(nn.Module):
    """Each (time, channel) pair is its own token. Sequence length is C*T.

    Tokens are ordered as (t=0,k=0), (t=0,k=1), ..., (t=0,k=C-1), (t=1,k=0),
    ... so the last token at each time step has seen that whole time step's
    channels plus all prior time steps. Causal mask enforces:
        token i (at (t_i, k_i)) sees token j iff t_j < t_i OR (t_j == t_i and j <= i)
    Prediction at output time t uses the hidden state of the last-channel
    token at time t (index t*C + C-1), which has the full x_{<=t} context.
    """

    def __init__(self, n_channels: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, n_bins: int,
                 dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.C = n_channels
        self.d_model = d_model
        self.proj = nn.Linear(1, d_model)
        self.channel_emb = nn.Embedding(n_channels, d_model)
        self.register_buffer("pos", _sinusoidal_positions(max_len, d_model))
        self.layers = _make_encoder_layers(d_model, n_heads, n_layers, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_bins)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        C = self.C
        # positions flattened as t*C + k; time_index = positions // C
        n = T * C
        pos = torch.arange(n, device=device)
        t_idx = pos // C
        # query at i, key at j: allow iff t_j < t_i OR (t_j == t_i and j <= i)
        t_i = t_idx.view(-1, 1)
        t_j = t_idx.view(1, -1)
        i = pos.view(-1, 1)
        j = pos.view(1, -1)
        allow = (t_j < t_i) | ((t_j == t_i) & (j <= i))
        # nn.TransformerEncoder expects True = mask out (not attend)
        return ~allow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        scalars = x.unsqueeze(-1)  # (B, T, C, 1)
        h = self.proj(scalars)  # (B, T, C, d)
        ch_ids = torch.arange(C, device=x.device)
        h = h + self.channel_emb(ch_ids).view(1, 1, C, -1)
        h = h + self.pos[:T].view(1, T, 1, -1)
        h = h.reshape(B, T * C, self.d_model)
        mask = self._causal_mask(T, x.device)
        for layer in self.layers:
            # is_causal=False because our mask is richer than a pure upper triangle
            h = layer(h, src_mask=mask, is_causal=False)
        h = self.norm(h)
        # last-channel token at each time: index t*C + (C-1)
        last_of_time = torch.arange(T, device=x.device) * C + (C - 1)
        h_t = h[:, last_of_time, :]
        return self.head(h_t)

    def aux_loss(self) -> torch.Tensor:
        return _zero_aux(next(self.parameters()).device)
