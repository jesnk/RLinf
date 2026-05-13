# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RL Token encoder/decoder modules.

Faithful to PI RLT paper:
- Eq.1: bidirectional transformer over [z_{1:M}, e_rl], output = position M+1 = z_rl.
- Eq.2: autoregressive decoder (causal mask) reconstructs z_{1:M} from z_rl + sg(z_{1:i-1}).
- App. B: 4-layer transformers, hidden 2048 (Gemma matched), 8 heads, ffn 4096.

σ-QRT uses these for offline Q-aware joint training (encoder is NOT frozen after Stage 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-norm transformer block. Supports optional attn mask for causal decoding."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class RLTokenEncoder(nn.Module):
    """RLT Eq.1: bidirectional transformer over [z_{1:M}, e_rl], output = position M+1."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 4096,
    ):
        super().__init__()
        self.in_proj = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )
        self.rl_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.rl_token, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, M, d_in] -> z_rl: [B, hidden_dim]."""
        B = z.shape[0]
        h = self.in_proj(z)
        rl = self.rl_token.expand(B, -1, -1)
        h = torch.cat([h, rl], dim=1)  # [B, M+1, hidden]
        for blk in self.blocks:
            h = blk(h, attn_mask=None)
        h = self.norm(h)
        return h[:, -1]  # [B, hidden]


class RLTokenDecoder(nn.Module):
    """RLT Eq.2: causal transformer, autoregressive reconstruction of z_{1:M}.

    Input sequence layout (length M = M-1 + 1, with z_rl at position 0):
        [z_rl,  z_1,  z_2, ..., z_{M-1}]    (positions 0 .. M-1)
    Output at position i (0-indexed) reconstructs z_{i+1}, i.e. positions 0..M-1
    produce predictions for z_1, z_2, ..., z_M. With strict causal mask, position i
    can only attend to positions 0..i, so predicting z_{i+1} from sg(z_{1:i}) ∪
    {z_rl} matches Eq.2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        max_len: int = 1024,
    ):
        super().__init__()
        self.in_proj = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )
        self.rl_proj = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, hidden_dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.max_len = max_len

    def forward(self, z_rl: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """Reconstruct z_{1:M}.

        Args:
            z_rl: [B, d_in] compressed RL token from encoder.
            z_prev: [B, M-1, d_in] preceding VLA tokens. Caller applies stop_grad.

        Returns:
            z_hat: [B, M, d_in] reconstructed VLA tokens.
        """
        B, M_minus_1, _ = z_prev.shape
        M = M_minus_1 + 1
        assert M <= self.max_len, f"M={M} exceeds max_len={self.max_len}"
        rl_in = self.rl_proj(z_rl).unsqueeze(1)  # [B, 1, hidden]
        prev_in = self.in_proj(z_prev)  # [B, M-1, hidden]
        h = torch.cat([rl_in, prev_in], dim=1)  # [B, M, hidden]
        h = h + self.pos[:, :M]
        causal_mask = torch.triu(
            torch.ones(M, M, device=h.device, dtype=torch.bool), diagonal=1
        )
        for blk in self.blocks:
            h = blk(h, attn_mask=causal_mask)
        h = self.norm(h)
        return self.out(h)  # [B, M, d_in]
