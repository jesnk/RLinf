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

"""σ-QRT Task 11 — offline RL training entry (W4 gate test driver).

Variants
--------
- ``qrt`` (default): σ-QRT joint encoder-actor-critic. Stage 1 = encoder
  + decoder warmup (RLT-style recon). Stage 2 = full ``train_step`` from
  QRTOfflinePolicyWorker (joint Q-aware update, encoder receives critic
  gradient — the σ-QRT novelty).
- ``a1_frozen_encoder``: ablation. Stage 1 + ``freeze_encoder()`` + Stage 2
  OFFLINE only (no env rollouts). Tests whether the σ-QRT novelty
  (joint encoder updates from Q gradient) actually matters versus the
  paper-faithful frozen-encoder protocol.
- ``rlt_online_sim``: paper-faithful B2 baseline. Stage 1 + freeze_encoder
  + Stage 2 with env rollouts interleaved. Online RL with frozen encoder.

Outputs
-------
- ``<output_dir>/metrics.json``: list of per-log-interval metric dicts.
- ``<output_dir>/ckpt.pt``: encoder + decoder + actor + critic state dicts
  + the resolved config (for eval-time reload).

Usage
-----
    PYTHONPATH=$(pwd) python examples/embodiment/run_qrt_offline.py \\
        --config examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml \\
        --output_dir runs/qrt_seed1 \\
        --variant qrt
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

# Repo-local helpers (CLI override parsing).
sys.path.insert(0, str(Path(__file__).parent))
from _sigma_qrt_helpers import apply_overrides  # noqa: E402

logging.basicConfig(
    format="[run_qrt] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Buffer batch → worker batch contract adapter
# --------------------------------------------------------------------------- #
_WORKER_CONTRACT_KEYS = (
    "z_obs",
    "next_z_obs",
    "s_p",
    "next_s_p",
    "action",
    "ref_action",
    "next_ref_action",
    "reward",
    "done",
)


def adapt_buffer_batch(batch: dict) -> dict:
    """Map TrajectoryReplayBuffer.sample() output → worker.train_step() contract.

    Two input formats are supported:

    1) Fast path (σ-QRT optimization, post-W4): buffer's `_flat_storage` is
       populated, so sample() already returns the worker contract dict
       (singular keys: z_obs, next_z_obs, s_p, next_s_p, action, ref_action,
       next_ref_action, reward, done). Adapter just casts to float32 in
       case the underlying storage is bfloat16 (z_obs/next_z_obs).

    2) Slow path (legacy, when flat storage is not built): plural keys +
       curr_obs/next_obs nested dicts:
           actions:    [B, chunk_len, action_dim]
           rewards:    [B, chunk_len]
           dones:      [B]
           curr_obs.z_obs:      [B, M, d]
           curr_obs.s_p:        [B, proprio_dim]
           curr_obs.ref_action: [B, chunk_len, action_dim]
           next_obs.<same>
    """
    if all(k in batch for k in _WORKER_CONTRACT_KEYS):
        # Fast path: buffer already returned the worker contract. Cast z_obs
        # tensors to float32 here (storage may be bfloat16 to save host RAM).
        return {
            "z_obs": batch["z_obs"].float(),
            "next_z_obs": batch["next_z_obs"].float(),
            "s_p": batch["s_p"].float(),
            "next_s_p": batch["next_s_p"].float(),
            "action": batch["action"].float(),
            "ref_action": batch["ref_action"].float(),
            "next_ref_action": batch["next_ref_action"].float(),
            "reward": batch["reward"].float(),
            "done": batch["done"].float(),
        }

    # Slow-path fallback: old nested schema.
    curr = batch["curr_obs"]
    next_ = batch["next_obs"]
    return {
        "z_obs": curr["z_obs"].float(),
        "next_z_obs": next_["z_obs"].float(),
        "s_p": curr["s_p"].float(),
        "next_s_p": next_["s_p"].float(),
        "action": batch["actions"].float(),
        "ref_action": curr["ref_action"].float(),
        "next_ref_action": next_["ref_action"].float(),
        "reward": batch["rewards"].float(),
        "done": batch["dones"].float(),
    }


def _move_to_device_async(batch: dict, device) -> dict:
    """Pin host memory + non_blocking H2D copy so transfer overlaps compute.

    No-op on CPU device (pin_memory only useful for cuda destination).
    Returns a new dict with tensors on `device`.
    """
    # str("cpu")/"cuda" or torch.device — normalize.
    dev_str = str(device)
    if dev_str == "cpu" or not torch.cuda.is_available():
        return batch
    out = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        # If already on cuda, leave it (defensive — should be on CPU from buf).
        if v.is_cuda:
            out[k] = v
            continue
        # pin_memory requires contiguous tensors; index_select results
        # are already contiguous so this is cheap.
        try:
            pinned = v.pin_memory()
        except Exception:
            pinned = v
        out[k] = pinned.to(device, non_blocking=True)
    return out


# --------------------------------------------------------------------------- #
# Variant builders
# --------------------------------------------------------------------------- #
def _build_worker(cfg, variant: str, device: str):
    if variant == "qrt":
        from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import (
            QRTOfflinePolicyWorker,
        )

        worker = QRTOfflinePolicyWorker(cfg, device=device)
    elif variant in ("a1_frozen_encoder", "rlt_online_sim"):
        from rlinf.workers.actor.fsdp_rlt_online_sim_worker import (
            RLTOnlineSimWorker,
        )

        worker = RLTOnlineSimWorker(cfg, device=device)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    worker.setup()
    return worker


# --------------------------------------------------------------------------- #
# Stage 1 (token warmup)
# --------------------------------------------------------------------------- #
def _qrt_stage1_step(worker, batch_adapted: dict) -> dict:
    """σ-QRT Stage 1 = encoder + decoder warmup with recon loss only.

    Mirrors RLTOnlineSimWorker.stage1_step but lives outside it so the
    σ-QRT worker (which does NOT freeze its encoder) can also do a
    warmup phase that's compatible with the (frozen-encoder) ablations.
    """
    from rlinf.algorithms.losses import rlt_recon_loss

    z_obs = batch_adapted["z_obs"].to(worker.device)
    z_rl = worker.encoder(z_obs)
    z_prev = z_obs[:, :-1].detach()
    z_hat = worker.decoder(z_rl, z_prev)
    loss = rlt_recon_loss(z_hat, z_obs.detach())
    worker.opt_enc.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(worker.encoder.parameters()) + list(worker.decoder.parameters()),
        max_norm=worker.grad_clip_norm,
    )
    worker.opt_enc.step()
    return {"loss_recon": float(loss.detach().item())}


def _run_stage1(worker, buf, cfg, variant: str, device: str) -> list[dict]:
    """Encoder warmup loop. Same for all variants except dispatching method."""
    metrics: list[dict] = []
    log_iv = int(cfg.logging.get("log_interval", 100))
    batch_size = int(cfg.training.batch_size)
    for step in range(int(cfg.training.warmup_steps)):
        batch = buf.sample(num_chunks=batch_size)
        adapted = adapt_buffer_batch(batch)
        # Tier-2 optimization: pinned-memory non_blocking H2D copy overlaps
        # with previous step's GPU compute. No-op when device=cpu.
        adapted = _move_to_device_async(adapted, worker.device)
        if variant == "qrt":
            m = _qrt_stage1_step(worker, adapted)
        else:
            m = worker.stage1_step({"z_obs": adapted["z_obs"]})
        m = dict(m)
        m["phase"] = "stage1"
        m["step"] = step
        if step % log_iv == 0 or step == int(cfg.training.warmup_steps) - 1:
            metrics.append(m)
            log.info(
                "stage1 step=%d loss_recon=%.4f",
                step,
                m.get("loss_recon", float("nan")),
            )
    return metrics


# --------------------------------------------------------------------------- #
# Stage 2 (joint or actor-critic only)
# --------------------------------------------------------------------------- #
def _stage2_offline(worker, buf, cfg, variant: str) -> list[dict]:
    """Offline Stage 2 loop for qrt + a1_frozen_encoder."""
    metrics: list[dict] = []
    log_iv = int(cfg.logging.get("log_interval", 100))
    batch_size = int(cfg.training.batch_size)
    max_steps = int(cfg.training.max_train_steps)
    for step in range(max_steps):
        batch = buf.sample(num_chunks=batch_size)
        adapted = adapt_buffer_batch(batch)
        adapted = _move_to_device_async(adapted, worker.device)
        if variant == "qrt":
            m = worker.train_step(adapted, step=step)
        else:
            m = worker.stage2_step(adapted, step=step)
        m = dict(m)
        m["phase"] = "stage2"
        m["step"] = step
        if step % log_iv == 0 or step == max_steps - 1:
            metrics.append(m)
            log.info(
                "stage2 step=%d loss_critic=%.4f loss_actor=%.4f q_mean=%.4f",
                step,
                m.get("loss_critic", float("nan")),
                m.get("loss_actor", float("nan")),
                m.get("q_mean", float("nan")),
            )
    return metrics


def _stage2_online_sim(worker, buf, cfg, _device: str) -> list[dict]:
    """Online Stage 2: env rollout interleaved with stage2_step.

    For W4 the env rollout is a single LIBERO env, so we collect 1
    transition per worker step and append it to the buffer. This is a
    minimal sync loop — production B2 needs vectorized rollouts but
    that's W5+. For the smoke test path we never reach this branch
    (test uses --variant qrt).
    """
    # Defer the heavy import so the offline-only smoke test doesn't need
    # robosuite.
    from examples.embodiment.collect_base_vla_rollouts import (  # noqa: WPS433
        _bootstrap_gl_env,
        _load_vla,
        _make_libero_env,
        _pack_trajectory,
        _rollout_one_episode,
    )

    _bootstrap_gl_env()
    log.info("rlt_online_sim Stage 2: bootstrapping LIBERO env + π0.5 VLA")
    vla = _load_vla(cfg.model, device=worker.device)
    env = _make_libero_env(
        cfg.env, int(cfg.env.get("max_episode_len", 600)), seed=int(cfg.get("seed", 0))
    )

    metrics: list[dict] = []
    log_iv = int(cfg.logging.get("log_interval", 100))
    batch_size = int(cfg.training.batch_size)
    max_steps = int(cfg.training.max_train_steps)
    # Cadence: collect 1 episode every `cfg.training.rollout_interval` steps.
    rollout_iv = int(cfg.training.get("rollout_interval", 100))
    z_obs_dtype = str(cfg.training.get("save_z_obs_dtype", "bfloat16"))

    for step in range(max_steps):
        if step > 0 and step % rollout_iv == 0:
            log.info("[rlt_online_sim] collecting 1 episode at step %d", step)
            try:
                episode = _rollout_one_episode(
                    vla,
                    env,
                    cfg.model,
                    OmegaConf.create({"action_noise_std": 0.0}),
                    max_episode_len=int(cfg.env.get("max_episode_len", 600)),
                    z_obs_dtype=z_obs_dtype,
                )
                traj = _pack_trajectory(episode)
                if traj is not None:
                    # offline_only must be False for the online sim path; the
                    # entry script forces this via buffer construction in main().
                    buf.add_trajectories([traj])
            except Exception:
                log.exception("[rlt_online_sim] rollout %d failed; continuing", step)

        batch = buf.sample(num_chunks=batch_size)
        adapted = adapt_buffer_batch(batch)
        adapted = _move_to_device_async(adapted, worker.device)
        m = worker.stage2_step(adapted, step=step)
        m = dict(m)
        m["phase"] = "stage2"
        m["step"] = step
        if step % log_iv == 0 or step == max_steps - 1:
            metrics.append(m)
            log.info(
                "[rlt_online_sim] step=%d loss_critic=%.4f loss_actor=%.4f",
                step,
                m.get("loss_critic", float("nan")),
                m.get("loss_actor", float("nan")),
            )
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="σ-QRT W4-gate training entry.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable OmegaConf override; key.path=value (bool/int/float coerced).",
    )
    p.add_argument(
        "--variant",
        choices=["qrt", "a1_frozen_encoder", "rlt_online_sim"],
        default="qrt",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override auto device selection (cpu/cuda/cuda:N).",
    )
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.override)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    log.info(
        "variant=%s device=%s buffer=%s warmup_steps=%d max_train_steps=%d",
        args.variant,
        device,
        cfg.data.offline_buffer_path,
        int(cfg.training.warmup_steps),
        int(cfg.training.max_train_steps),
    )

    worker = _build_worker(cfg, args.variant, device)

    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    # For rlt_online_sim we must keep offline_only=False so the online
    # rollout loop can call add_trajectories().
    offline_only = args.variant != "rlt_online_sim"
    buf = TrajectoryReplayBuffer.from_offline_dataset(
        cfg.data.offline_buffer_path,
        capacity=int(cfg.data.capacity),
        device=device,
        offline_only=offline_only,
    )
    log.info("buffer loaded: %d trajectories", len(buf))

    # Stage 1 (warmup).
    metrics_log: list[dict] = []
    metrics_log.extend(_run_stage1(worker, buf, cfg, args.variant, device))

    # Freeze encoder for non-qrt variants.
    if args.variant != "qrt":
        worker.freeze_encoder()
        log.info("encoder frozen (variant=%s)", args.variant)

    # Stage 2.
    if args.variant == "rlt_online_sim":
        metrics_log.extend(_stage2_online_sim(worker, buf, cfg, device))
    else:
        metrics_log.extend(_stage2_offline(worker, buf, cfg, args.variant))

    # Persist metrics + final ckpt.
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_log, indent=2))

    ckpt = {
        "encoder": worker.encoder.state_dict(),
        "decoder": worker.decoder.state_dict(),
        "actor": worker.actor.state_dict(),
        "critic": worker.critic.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "variant": args.variant,
    }
    torch.save(ckpt, out_dir / "ckpt.pt")

    log.info(
        "done. metrics=%s ckpt=%s n_log_entries=%d",
        metrics_path,
        out_dir / "ckpt.pt",
        len(metrics_log),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
