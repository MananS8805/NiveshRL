"""Permutation-equivariant portfolio encoder.

    per-stock temporal encoder (weights shared across stocks)
        + sector embedding + current weight        -> stock tokens (B, N, d)
    market GRU + investor-profile MLP + cash weight -> context token (B, 1, d)
    Transformer over [context, stock_1 .. stock_N]  -> contextualised tokens

Every stock goes through the same weights, so the model does not depend on
stock order and works for any N. This matters because index membership
changes over time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WindowConv(nn.Module):
    """Conv1d(kernel=3, padding=1) written as a matmul over stacked shifted windows.

    Numerically the same as Conv1d. It is several times faster on CPU than
    the mkldnn kernel for very short sequences (L ~ 12) with huge batches
    (B * N ~ 7000 per minibatch).
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.lin = nn.Linear(3 * c_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C)
        p = nn.functional.pad(x, (0, 0, 1, 1))
        return self.lin(torch.cat([p[:, :-2], p[:, 1:-1], p[:, 2:]], -1))


class TemporalEncoder(nn.Module):
    """Encode one stock's (L, F) feature window into a d-vector."""

    def __init__(self, n_feat: int, d: int, kind: str = "cnn", lookback: int = 12):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.net = nn.Sequential(WindowConv(n_feat, d), nn.GELU(), WindowConv(d, d), nn.GELU())
            self.out = nn.Linear(2 * d, d)
        elif kind == "lstm":
            self.net = nn.LSTM(n_feat, d, batch_first=True)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(1), nn.Linear(lookback * n_feat, d), nn.GELU(),
                                     nn.Linear(d, d))
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, F)
        if self.kind == "cnn":
            h = self.net(x)                                        # (B, L, d)
            return self.out(torch.cat([h.mean(1), h[:, -1]], -1))
        if self.kind == "lstm":
            _, (h, _) = self.net(x)
            return h[-1]
        return self.net(x)


class PortfolioEncoder(nn.Module):
    def __init__(self, n_feat: int, n_mkt: int, n_profile: int, sector_ids, lookback: int, d: int = 64,
                 temporal: str = "cnn", n_layers: int = 2, n_heads: int = 4, attention: bool = True):
        super().__init__()
        self.register_buffer("sector_ids", torch.as_tensor(sector_ids, dtype=torch.long))
        n_sectors = int(self.sector_ids.max()) + 1
        self.d = d
        self.temporal = TemporalEncoder(n_feat, d, temporal, lookback)
        self.sector_emb = nn.Embedding(n_sectors, d)
        self.w_proj = nn.Linear(1, d)
        self.market = nn.GRU(n_mkt, d, batch_first=True)
        self.profile = nn.Sequential(nn.Linear(n_profile, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.Linear(3 * d + 1, d)
        self.attention = attention
        if attention:
            layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def _sequence(self, obs: dict) -> torch.Tensor:
        """Build [context, stock_1 .. stock_N] before attention: (B, 1+N, d)."""
        x = obs["stock"]                                # (B, L, N, F)
        B, L, N, F = x.shape
        tok = self.temporal(x.permute(0, 2, 1, 3).reshape(B * N, L, F)).view(B, N, self.d)
        w = obs["weights"]
        tok = tok + self.sector_emb(self.sector_ids)[None] + self.w_proj(w[:, :N, None])
        _, hm = self.market(obs["market"])
        ctx = self.ctx(torch.cat([hm[-1], self.profile(obs["profile"]), tok.mean(1), w[:, N:]], -1))
        return torch.cat([ctx[:, None], tok], 1)

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (context token (B, d), stock tokens (B, N, d))."""
        seq = self._sequence(obs)
        if self.attention:
            seq = self.transformer(seq)
        seq = self.norm(seq)
        return seq[:, 0], seq[:, 1:]

    @torch.no_grad()
    def attention_to_context(self, obs: dict) -> torch.Tensor:
        """Last-layer attention from the context token to each stock, head-averaged: (B, N)."""
        if not self.attention:
            raise RuntimeError("model built without attention")
        seq = self._sequence(obs)
        layers = self.transformer.layers
        for layer in layers[:-1]:
            seq = layer(seq)
        h = layers[-1].norm1(seq)
        _, attn = layers[-1].self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        return attn[:, 0, 1:]
