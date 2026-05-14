# Copyright 2025 The RLinf Authors.
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

"""σ-QRT v7 Goal 1: Random-encoder baseline for the recon-loss control.

Trains the RLT decoder ONLY for the same number of warmup steps that v6
used (cfg.training.warmup_steps, typically 2000), with a freshly random-
initialized RLTokenEncoder held frozen. We then compare the final recon
loss against chain v6's σ-QRT encoder recon loss at the equivalent step.

Verification target:
    random_recon_loss  >>  sigma_qrt_recon_loss   (e.g. ≥ 2× higher)
    → encoder is learning useful features.

    random_recon_loss  ≈   sigma_qrt_recon_loss
    → encoder isn't doing meaningful representation work, decoder can
      reconstruct from anything → encoder is not the source of σ-QRT's
      σ-QRT-vs-A1 lift (Goal 1 falsified).

This script reuses the existing buffer loader + the same optimizer/batch
size/LR/clip values as v6 Stage 1, but freezes the encoder so only the
decoder learns.

Usage
-----
    python -X utf8 experiments/random_encoder_baseline.py \\
        --config examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml \\
        --buffer_path data/sigma_qrt/libero_long/transitions_B2500.pkl \\
        --max_steps 2000 \\
        --output_log runs/random_encoder_baseline.json

CPU smoke test: drop ``--device`` (defaults to cpu). Real run after chain v6
finishes: ``--device cuda:N``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Reuse the same helpers as run_qrt_offline.py for override parsing.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "examples" / "embodiment")
)
from _sigma_qrt_helpers import apply_overrides  # noqa: E402

logging.basicConfig(
    format="[rand-enc] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_modules(cfg, device: str):
    """Construct random-init encoder + decoder matching σ-QRT shape."""
    from rlinf.models.embodiment.modules.rl_token import RLTokenDecoder, RLTokenEncoder

    m = cfg.model
    enc = RLTokenEncoder(
        input_dim=m.token_dim,
        hidden_dim=m.token_dim,
        num_layers=m.encoder_layers,
        num_heads=m.encoder_heads,
        ffn_dim=m.encoder_ffn,
    ).to(device)
    dec = RLTokenDecoder(
        input_dim=m.token_dim,
        hidden_dim=m.token_dim,
        num_layers=m.decoder_layers,
        num_heads=m.decoder_heads,
        ffn_dim=m.decoder_ffn,
        max_len=int(m.get("decoder_max_len", 1024)),
    ).to(device)

    # Freeze encoder — that's the whole point of the baseline.
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc, dec


def _load_buffer(buffer_path: str, capacity: int):
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    return TrajectoryReplayBuffer.from_offline_dataset(
        buffer_path,
        capacity=capacity,
        device="cpu",
        offline_only=True,
    )


def _adapt_z_obs(batch: dict) -> torch.Tensor:
    """Extract z_obs from the buffer's worker-contract or nested-obs schema."""
    if "z_obs" in batch:
        return batch["z_obs"].float()
    return batch["curr_obs"]["z_obs"].float()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="σ-QRT v7 random-encoder recon baseline.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--buffer_path", type=str, required=True)
    p.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Match cfg.training.warmup_steps (default 2000 = v6 stage 1).",
    )
    p.add_argument("--output_log", type=str, required=True, help="JSON output path.")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu / cuda / cuda:N — default cpu so v6 is undisturbed.",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional OmegaConf overrides; matches run_qrt_offline.py.",
    )
    p.add_argument("--seed", type=int, default=0, help="Encoder + decoder init seed.")
    p.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Log + curve sample interval (steps).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.override)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    log.info(
        "random-encoder baseline: device=%s buffer=%s max_steps=%d seed=%d",
        device,
        args.buffer_path,
        args.max_steps,
        args.seed,
    )

    # Match v6 stage 1 hparams from cfg.training so the comparison is apples
    # to apples (same LR, clip, batch size).
    tr = cfg.training
    lr_encoder = float(tr.lr_encoder)
    batch_size = int(tr.batch_size)
    grad_clip = float(tr.grad_clip_norm)

    # Modules (encoder frozen, only decoder gets gradient).
    encoder, decoder = _build_modules(cfg, device)
    # Optimizer trains decoder only. Match the v6 Adam(lr_encoder) config —
    # this is the same LR that v6 stage 1 used to train both encoder + decoder,
    # so the comparison isolates "encoder is frozen random init" vs
    # "encoder is being trained jointly". No other knob changes.
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=lr_encoder)
    log.info(
        "decoder optimizer Adam(lr=%.2e) batch_size=%d grad_clip=%.2f",
        lr_encoder,
        batch_size,
        grad_clip,
    )

    # Buffer.
    buf = _load_buffer(args.buffer_path, capacity=int(cfg.data.capacity))
    log.info("buffer loaded: %d trajectories", len(buf))

    # Import recon loss late (after potential GL bootstrap in helpers).
    from rlinf.algorithms.losses import rlt_recon_loss

    curve: list[dict] = []
    recent_losses: list[float] = []
    for step in range(args.max_steps):
        batch = buf.sample(num_chunks=batch_size)
        z_obs = _adapt_z_obs(batch).to(device)
        # Encoder forward under no_grad — frozen + eval. No gradient flows
        # back into encoder params; only decoder weights move.
        with torch.no_grad():
            z_rl = encoder(z_obs)
        z_prev = z_obs[:, :-1].detach()
        z_hat = decoder(z_rl, z_prev)
        loss = rlt_recon_loss(z_hat, z_obs.detach())

        opt_dec.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=grad_clip)
        opt_dec.step()

        loss_val = float(loss.detach().item())
        recent_losses.append(loss_val)

        if step % args.log_every == 0 or step == args.max_steps - 1:
            log.info("step=%d loss_recon=%.4f", step, loss_val)
            curve.append({"step": step, "loss_recon": loss_val})

    # Final summary: mean over the last 100 steps (matches v6 logging cadence).
    tail = recent_losses[-min(100, len(recent_losses)) :]
    final_mean = float(sum(tail) / len(tail))

    out = {
        "final_mean_recon_loss_last100": final_mean,
        "max_steps": args.max_steps,
        "batch_size": batch_size,
        "lr_encoder": lr_encoder,
        "seed": args.seed,
        "device": device,
        "config": args.config,
        "buffer_path": args.buffer_path,
        "curve": curve,
    }
    out_path = Path(args.output_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("done. final_mean_last100=%.4f → %s", final_mean, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
