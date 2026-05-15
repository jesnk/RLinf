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

"""σ-QRT G2 — critic Q sensitivity probe.

Goal
----
Explain why a G2 residual-actor lane's SR peaks at 0.24 while B0=0.48: if the
critic is uninformative (Q ≈ flat over local action perturbations), then
∇ₐ Q(s,a) ≈ 0 and the Q-max term pushes Δ nowhere — the residual gets stuck
near zero and the eval action collapses to ref. If the critic IS informative,
the issue lies elsewhere (α scale, gradient flow, etc.).

Procedure
---------
For ``--num_states`` random states sampled from a transition buffer:
    Build (z_rl, s_p) via worker.encoder(z_obs) and the buffer's s_p.
    Take ref_action straight from the buffer (it is the π0.5 ref).
    For ``--num_perturb`` random perturbations:
        a_perturb = ref_action + 𝒩(0, σ²)  clipped to [-1, 1]
        q1, q2 = worker.critic(state_feat, action_feat)
        q_mean = (q1 + q2) / 2
    Collect per-state stats: std and range of q_mean across perturbations.

Output JSON
-----------
{
    "ckpt", "buffer_path", "num_states", "num_perturb", "perturb_std",
    "q_std_mean":   mean across states of (std over perturbations of q_mean),
    "q_std_max":    max  across states,
    "q_range_mean": mean across states of (max-min over perturbations),
    "q_range_max":  max across states,
    "q_mean_overall": mean of q_mean across all (state, perturbation),
    "verdict": "informative" if q_std_mean > 0.05 else "flat"
}

Usage
-----
    python experiments/probe_critic_sensitivity.py \\
        --ckpt runs/g2_residual_actor_20260514_234926/g2res_seed2/ckpt_step7000.pt \\
        --buffer_path data/sigma_qrt/libero_long/transitions_B2500.pkl \\
        --output /tmp/critic_sens.json

CPU mode is fine (~5 min). The probe never touches the env or VLA — pure
critic forward passes on cached buffer features.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Repo root → sys.path so `examples.embodiment.*` imports resolve when run
# from any cwd. Mirrors the layout used by other probe scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    format="[probe_critic_sensitivity] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="σ-QRT critic Q sensitivity probe.")
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="σ-QRT ckpt (saved by run_qrt_offline.py). Must contain "
        "'encoder', 'critic', and a 'cfg' payload.",
    )
    p.add_argument(
        "--buffer_path",
        type=str,
        required=True,
        help="Offline transition buffer (.pkl) — used only for state sampling.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config override path. Defaults to the ckpt's saved cfg.",
    )
    p.add_argument(
        "--num_states",
        type=int,
        default=100,
        help="How many random states to sample from the buffer.",
    )
    p.add_argument(
        "--num_perturb",
        type=int,
        default=20,
        help="How many random perturbations per state.",
    )
    p.add_argument(
        "--perturb_std",
        type=float,
        default=0.1,
        help="Std-dev of Gaussian perturbation added to ref_action.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu|cuda. Default: cpu (probe is CPU-cheap).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--informative_threshold",
        type=float,
        default=0.05,
        help="q_std_mean above this → verdict='informative'. Default 0.05.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path.",
    )
    return p.parse_args(argv)


def _load_worker(ckpt_path: str, config_path: str | None, device: str):
    """Build a QRTOfflinePolicyWorker from the ckpt's saved cfg and load weights.

    We need ``encoder`` and ``critic`` only — but the worker constructor will
    happily build all heads. Loading only what's necessary keeps the script
    robust to ckpts that omit decoder/v_net.
    """
    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import (
        QRTOfflinePolicyWorker,
    )

    payload = torch.load(ckpt_path, map_location=device)
    saved_cfg = payload.get("cfg")
    if config_path is not None:
        cfg = OmegaConf.load(config_path)
        log.info("using --config override: %s", config_path)
    elif isinstance(saved_cfg, dict):
        cfg = OmegaConf.create(saved_cfg)
        log.info("using cfg from ckpt payload")
    else:
        raise RuntimeError(
            "ckpt has no 'cfg' payload and --config not provided; cannot "
            "instantiate worker"
        )

    worker = QRTOfflinePolicyWorker(cfg, device=device)
    worker.setup()
    # Strict load on the two heads we use.
    worker.encoder.load_state_dict(payload["encoder"])
    worker.critic.load_state_dict(payload["critic"])
    worker.encoder.eval()
    worker.critic.eval()
    for p_ in worker.encoder.parameters():
        p_.requires_grad = False
    for p_ in worker.critic.parameters():
        p_.requires_grad = False
    log.info("worker loaded: encoder + critic from %s", ckpt_path)
    return worker, cfg


def _sample_states(buffer_path: str, num_states: int, device: str, seed: int):
    """Return (z_obs[B,M,d], s_p[B,dp], ref_action[B,C,A]) as fp32 on device.

    Reuses TrajectoryReplayBuffer.from_offline_dataset so the buffer schema
    (Trajectory vs flat-contig) is handled centrally. We only need a single
    random-sample call.
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    g = torch.Generator()
    g.manual_seed(int(seed))

    log.info("loading buffer: %s", buffer_path)
    t0 = time.time()
    # Use CPU storage for the buffer — keeps memory pressure low; we move
    # the sampled batch to the requested device after. capacity must be
    # >= number of trajectories in the pickle. The G2 alpha-sweep
    # launcher uses capacity=8000 for B2500 (≈3× headroom) — match that
    # so the flat-cache allocation stays bounded (~tens of GB).
    buf = TrajectoryReplayBuffer.from_offline_dataset(
        buffer_path,
        capacity=8000,
        sample_window_size=8000,
        device="cpu",
        offline_only=True,
    )
    log.info("buffer ready (%.1fs); sampling %d states", time.time() - t0, num_states)
    batch = buf.sample(num_chunks=int(num_states))
    z_obs = batch["z_obs"].float().to(device)
    s_p = batch["s_p"].float().to(device)
    ref_action = batch["ref_action"].float().to(device)
    log.info(
        "sampled batch: z_obs=%s s_p=%s ref_action=%s",
        tuple(z_obs.shape),
        tuple(s_p.shape),
        tuple(ref_action.shape),
    )
    return z_obs, s_p, ref_action


def _critic_q(worker, z_rl, s_p, action):
    """Return per-sample mean of (q1, q2). Shape [B]."""
    state_feat = torch.cat([z_rl, s_p], dim=-1)
    action_feat = action.flatten(start_dim=1)
    q = worker.critic(state_feat, action_feat)  # [B, 2]
    return q.mean(dim=-1)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info(
        "ckpt=%s buffer=%s num_states=%d num_perturb=%d σ=%.3f device=%s",
        args.ckpt,
        args.buffer_path,
        args.num_states,
        args.num_perturb,
        args.perturb_std,
        device,
    )

    worker, _cfg = _load_worker(args.ckpt, args.config, device=device)
    z_obs, s_p, ref_action = _sample_states(
        args.buffer_path, args.num_states, device=device, seed=args.seed
    )
    B = z_obs.shape[0]

    # Encode once per state.
    with torch.no_grad():
        z_rl = worker.encoder(z_obs)  # [B, M, d] → encoder collapses to [B, d]
        # Encoder output shape varies (pooled vs token-sequence). The critic
        # expects state_dim = token_dim + proprio_dim. If z_rl is 3D, we
        # mean-pool it to match the training-time _q() contract — but
        # the offline worker's _q already concats along dim=-1, so z_rl must
        # already be 2D. Sanity-check.
        if z_rl.dim() != 2:
            raise RuntimeError(
                f"encoder output expected [B, d], got {tuple(z_rl.shape)}"
            )

    # For each state, generate N_perturb action perturbations and stat Q.
    # Vectorize: replicate state along the batch axis N_perturb times → one
    # critic forward of shape [B * N, *].
    N = int(args.num_perturb)
    sigma = float(args.perturb_std)
    q_std_per_state = []
    q_range_per_state = []
    q_mean_overall_acc = []
    with torch.no_grad():
        # Expand state to [B*N, ...] and ref_action similarly.
        z_rl_rep = z_rl.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        s_p_rep = s_p.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        ref_rep = (
            ref_action.unsqueeze(1)
            .expand(-1, N, -1, -1)
            .reshape(B * N, *ref_action.shape[1:])
        )
        noise = torch.randn_like(ref_rep) * sigma
        a_perturb = (ref_rep + noise).clamp(-1.0, 1.0)
        q_mean = _critic_q(worker, z_rl_rep, s_p_rep, a_perturb)  # [B*N]
        q_mean = q_mean.view(B, N)
        q_std_per_state = q_mean.std(dim=1).cpu().numpy()  # [B]
        q_range_per_state = (
            (q_mean.max(dim=1).values - q_mean.min(dim=1).values).cpu().numpy()
        )  # [B]
        q_mean_overall_acc = q_mean.flatten().cpu().numpy()

    q_std_mean = float(np.mean(q_std_per_state))
    q_std_max = float(np.max(q_std_per_state))
    q_range_mean = float(np.mean(q_range_per_state))
    q_range_max = float(np.max(q_range_per_state))
    q_mean_overall = float(np.mean(q_mean_overall_acc))
    verdict = "informative" if q_std_mean > args.informative_threshold else "flat"

    result = {
        "ckpt": args.ckpt,
        "buffer_path": args.buffer_path,
        "num_states": int(B),
        "num_perturb": N,
        "perturb_std": sigma,
        "informative_threshold": float(args.informative_threshold),
        "q_std_mean": q_std_mean,
        "q_std_max": q_std_max,
        "q_range_mean": q_range_mean,
        "q_range_max": q_range_max,
        "q_mean_overall": q_mean_overall,
        "verdict": verdict,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info(
        "q_std_mean=%.4f q_std_max=%.4f q_range_mean=%.4f verdict=%s → %s",
        q_std_mean,
        q_std_max,
        q_range_mean,
        verdict,
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
