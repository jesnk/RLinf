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

"""σ-QRT Task 3 tests: RLTokenEncoder + RLTokenDecoder.

Verifies:
  1. Encoder output shape (B, d) at full π0.5 scale (M=968, d=2048).
  2. Encoder smoke test at small dims for fast iteration.
  3. Encoder gradient flow to all parameters (incl. learnable e_rl token).
  4. Decoder reconstruction shape (B, M, d).
  5. Decoder causal mask (autoregressive, future tokens not attended).

Run from repo root:
    PYTHONPATH=. python -m pytest tests/sigma_qrt/test_rl_token.py -v
"""

from __future__ import annotations

import torch


def test_encoder_output_shape():
    from rlinf.models.embodiment.modules.rl_token import RLTokenEncoder

    enc = RLTokenEncoder(
        input_dim=2048, hidden_dim=2048, num_layers=4, num_heads=8, ffn_dim=4096
    )
    # Use realistic M=968 from π0.5 (3 images × 256 + 200 lang tokens)
    z = torch.randn(2, 968, 2048)
    z_rl = enc(z)
    assert z_rl.shape == (2, 2048), f"expected [B, d], got {z_rl.shape}"
    assert torch.isfinite(z_rl).all()


def test_encoder_smaller_smoke_for_speed():
    """Small dims so test is fast (M=968 × 4-layer is heavy)."""
    from rlinf.models.embodiment.modules.rl_token import RLTokenEncoder

    enc = RLTokenEncoder(
        input_dim=128, hidden_dim=128, num_layers=2, num_heads=4, ffn_dim=256
    )
    z = torch.randn(2, 16, 128)
    z_rl = enc(z)
    assert z_rl.shape == (2, 128)


def test_encoder_gradient_flow():
    from rlinf.models.embodiment.modules.rl_token import RLTokenEncoder

    enc = RLTokenEncoder(
        input_dim=128, hidden_dim=128, num_layers=2, num_heads=4, ffn_dim=256
    )
    z = torch.randn(2, 16, 128)
    z_rl = enc(z)
    loss = z_rl.sum()
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no grad on {name}"


def test_decoder_reconstruction_shape():
    from rlinf.models.embodiment.modules.rl_token import RLTokenDecoder

    dec = RLTokenDecoder(
        input_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        max_len=64,
    )
    z_rl = torch.randn(2, 128)
    z_prev = torch.randn(2, 31, 128)  # z_{1:M-1}, M=32
    z_hat = dec(z_rl, z_prev)
    assert z_hat.shape == (2, 32, 128), f"expected [B, M, d_in], got {z_hat.shape}"


def test_decoder_causal_mask():
    """Decoder must not attend to future tokens (autoregressive)."""
    from rlinf.models.embodiment.modules.rl_token import RLTokenDecoder

    dec = RLTokenDecoder(
        input_dim=64,
        hidden_dim=64,
        num_layers=1,
        num_heads=2,
        ffn_dim=128,
        max_len=16,
    )
    dec.eval()
    z_rl = torch.randn(1, 64)
    z_prev_v1 = torch.randn(1, 7, 64)
    z_prev_v2 = z_prev_v1.clone()
    z_prev_v2[0, -1] += 100.0  # 마지막 토큰만 변경
    with torch.no_grad():
        out1 = dec(z_rl, z_prev_v1)
        out2 = dec(z_rl, z_prev_v2)
    # 마지막 input 변경은 마지막 output 만 영향 — earlier output 들은 동일.
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5), (
        "causal mask violated: earlier outputs changed when last input changed"
    )
