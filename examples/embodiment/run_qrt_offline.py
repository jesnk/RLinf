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
- ``a1_raw_features``: σ-QRT v7 Goal 1 bypass ablation. Stage 1 SKIPPED.
  Stage 2 runs IQL but replaces ``encoder(z_obs)`` with mean-pooled raw
  VLA features (identity dim handling when ``cfg.model.token_dim`` equals
  the VLA hidden dim — default for π0.5 Gemma 2048 = 2048). Tests whether
  the RLT encoder is necessary; if SR ≥ A1, encoder is a bottleneck.
  Requires ``--use_iql`` (or ``cfg.training.use_iql=true``).

Outputs
-------
- ``<output_dir>/metrics.json``: list of per-log-interval metric dicts.
- ``<output_dir>/ckpt.pt``: encoder + decoder + actor + critic state dicts
  + the resolved config (for eval-time reload).
- ``<output_dir>/ckpt_step{N}.pt``: intermediate Stage-2 snapshots written
  every ``--save_interval`` steps (default 2000; set ``--save_interval -1``
  or ``0`` to disable). Same payload as the final ckpt — used by
  ``periodic_eval_watcher.py`` for learning-curve eval.
- ``<output_dir>/ckpt_latest.pt``: copy of the most recent intermediate
  ckpt (for "best-so-far" inspection).
- ``<output_dir>/.training_done``: empty sentinel touched at the end of
  ``main()`` so external watchers know to drain + exit.

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
# Pre-warmed encoder cache loader (σ-QRT G2 follow-up)
# --------------------------------------------------------------------------- #
_ENCODER_CFG_KEYS = (
    "token_dim",
    "encoder_layers",
    "encoder_heads",
    "encoder_ffn",
    "decoder_layers",
    "decoder_heads",
    "decoder_ffn",
    "decoder_max_len",
)


def _load_encoder_cache(worker, cfg, encoder_ckpt_path: str) -> None:
    """Load a stripped encoder+decoder ckpt into the worker.

    Validates that the cached cfg.model dims match the live cfg.model
    dims for every key in ``_ENCODER_CFG_KEYS`` that is present in the
    cache. Any mismatch raises ``RuntimeError`` so callers don't silently
    train against a misconfigured encoder.

    Side effect: applies state_dicts to ``worker.encoder`` and
    ``worker.decoder`` via ``load_state_dict(strict=True)``. Caller is
    responsible for skipping Stage 1 after this returns.
    """
    cache_path = Path(encoder_ckpt_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"--encoder_ckpt not found: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or "encoder" not in cache or "decoder" not in cache:
        raise RuntimeError(
            f"--encoder_ckpt {cache_path} is not a stripped encoder cache; "
            f"expected keys 'encoder' + 'decoder', got "
            f"{sorted(cache.keys()) if isinstance(cache, dict) else type(cache).__name__}"
        )

    cached_model_cfg = (cache.get("cfg_partial") or {}).get("model") or {}
    live_model = cfg.get("model", {}) or {}
    mismatches = []
    for k in _ENCODER_CFG_KEYS:
        if k not in cached_model_cfg:
            continue
        cached_v = cached_model_cfg[k]
        live_v = live_model.get(k)
        if cached_v != live_v:
            mismatches.append(f"  {k}: cache={cached_v!r} live={live_v!r}")
    if mismatches:
        raise RuntimeError(
            "[run_qrt] encoder cache cfg.model mismatch — refusing to load:\n"
            + "\n".join(mismatches)
        )

    worker.encoder.load_state_dict(cache["encoder"], strict=True)
    worker.decoder.load_state_dict(cache["decoder"], strict=True)
    log.info(
        "[run_qrt] loaded encoder from %s, skipping stage 1 (cache cfg=%s)",
        cache_path,
        cached_model_cfg,
    )


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
    elif variant == "a1_raw_features":
        # σ-QRT v7 Goal 1: bypass-encoder ablation. IQL only — TD3+BC path
        # would need a separate bypass worker class. Entry script skips
        # Stage 1 for this variant (see main()).
        from rlinf.workers.actor.fsdp_raw_features_iql_worker import (
            RawFeaturesIQLWorker,
        )

        worker = RawFeaturesIQLWorker(cfg, device=device)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    worker.setup()
    return worker


# --------------------------------------------------------------------------- #
# Stage 1 (token warmup)
# --------------------------------------------------------------------------- #
def _qrt_stage1_step(worker, batch_adapted: dict, use_bf16: bool = False) -> dict:
    """σ-QRT Stage 1 = encoder + decoder warmup with recon loss only.

    Mirrors RLTOnlineSimWorker.stage1_step but lives outside it so the
    σ-QRT worker (which does NOT freeze its encoder) can also do a
    warmup phase that's compatible with the (frozen-encoder) ablations.

    When use_bf16=True, runs the encoder/decoder fwd + loss in
    torch.autocast(bf16) so flash attention dispatches and ops use
    tensor cores (10x speedup on B200 over fp32). Backward is run
    outside autocast for numerical stability — gradients accumulate
    in the original parameter dtype (fp32), and the bf16 backward
    activations are produced inside autocast. bf16 doesn't need
    GradScaler (unlike fp16); the 8-bit exponent range is sufficient.
    """
    from rlinf.algorithms.losses import rlt_recon_loss

    z_obs = batch_adapted["z_obs"].to(worker.device)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and z_obs.is_cuda
        else _NullCtx()
    )
    with autocast_ctx:
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


class _NullCtx:
    """No-op context manager (used when bf16 autocast disabled)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# --------------------------------------------------------------------------- #
# Ckpt I/O — atomic write so the periodic_eval_watcher never sees a partial
# file. Pattern: torch.save → .tmp; os.replace(.tmp, final). os.replace is
# atomic on POSIX + Windows for same-filesystem renames.
# --------------------------------------------------------------------------- #
def _build_ckpt(worker, cfg, variant: str) -> dict:
    payload = {
        "encoder": worker.encoder.state_dict(),
        "decoder": worker.decoder.state_dict(),
        "actor": worker.actor.state_dict(),
        "critic": worker.critic.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "variant": variant,
    }
    # IQL variant: persist V network so reloads (e.g. periodic eval) can
    # reconstruct the full agent. v_net is None for TD3+BC.
    v_net = getattr(worker, "v_net", None)
    if v_net is not None:
        payload["v_net"] = v_net.state_dict()
    return payload


def _atomic_torch_save(payload: dict, dest: Path) -> None:
    """torch.save to ``dest.tmp`` then os.replace → atomic for watchers."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(payload, tmp)
    import os as _os

    _os.replace(tmp, dest)


def _save_intermediate_ckpt(
    worker, cfg, variant: str, out_dir: Path, step: int
) -> None:
    """Write ``ckpt_step{step}.pt`` + refresh ``ckpt_latest.pt`` atomically.

    Step numbering uses the 1-based "completed step" count so the first
    interval boundary at ``--save_interval N`` produces ``ckpt_stepN.pt``.
    """
    payload = _build_ckpt(worker, cfg, variant)
    step_path = out_dir / f"ckpt_step{step}.pt"
    latest_path = out_dir / "ckpt_latest.pt"
    _atomic_torch_save(payload, step_path)
    # latest = same payload, separate atomic write so partial reads can't
    # mix old/new bytes.
    _atomic_torch_save(payload, latest_path)
    log.info("saved intermediate ckpt step=%d → %s", step, step_path.name)


def _run_stage1(
    worker, buf, cfg, variant: str, device: str, use_bf16: bool = False
) -> list[dict]:
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
            m = _qrt_stage1_step(worker, adapted, use_bf16=use_bf16)
        else:
            # RLTOnlineSimWorker.stage1_step doesn't accept bf16 flag; wrap
            # in autocast externally so the encoder/decoder fwd inside use
            # bf16. Backward is also inside autocast — that's fine for bf16
            # (no GradScaler needed).
            if use_bf16 and adapted["z_obs"].is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    m = worker.stage1_step({"z_obs": adapted["z_obs"]})
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
def _stage2_offline(
    worker,
    buf,
    cfg,
    variant: str,
    use_bf16: bool = False,
    save_interval: int = 0,
    out_dir: Path | None = None,
) -> list[dict]:
    """Offline Stage 2 loop for qrt + a1_frozen_encoder.

    When ``save_interval > 0`` and ``out_dir`` is provided, an intermediate
    ckpt is written every ``save_interval`` completed steps. The first
    intermediate snapshot is written after step ``save_interval - 1`` runs
    (i.e. when ``(step + 1) % save_interval == 0``) so the snapshot filename
    ``ckpt_step{N}.pt`` matches the number of training steps that produced
    it.
    """
    metrics: list[dict] = []
    log_iv = int(cfg.logging.get("log_interval", 100))
    batch_size = int(cfg.training.batch_size)
    max_steps = int(cfg.training.max_train_steps)
    do_save = save_interval and save_interval > 0 and out_dir is not None
    for step in range(max_steps):
        batch = buf.sample(num_chunks=batch_size)
        adapted = adapt_buffer_batch(batch)
        adapted = _move_to_device_async(adapted, worker.device)
        if use_bf16 and adapted["z_obs"].is_cuda:
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = _NullCtx()
        with autocast_ctx:
            if variant == "qrt":
                m = worker.train_step(adapted, step=step)
            else:
                m = worker.stage2_step(adapted, step=step)
        m = dict(m)
        m["phase"] = "stage2"
        m["step"] = step
        if step % log_iv == 0 or step == max_steps - 1:
            metrics.append(m)
            if "loss_v" in m:
                log.info(
                    "stage2[iql] step=%d loss_q=%.4f loss_v=%.4f loss_actor=%.4f "
                    "q_mean=%.4f v_mean=%.4f adv=%.4f w=%.4f",
                    step,
                    m.get("loss_critic", float("nan")),
                    m.get("loss_v", float("nan")),
                    m.get("loss_actor", float("nan")),
                    m.get("q_mean", float("nan")),
                    m.get("v_mean", float("nan")),
                    m.get("advantage_mean", float("nan")),
                    m.get("weight_mean", float("nan")),
                )
            else:
                log.info(
                    "stage2 step=%d loss_critic=%.4f loss_actor=%.4f q_mean=%.4f",
                    step,
                    m.get("loss_critic", float("nan")),
                    m.get("loss_actor", float("nan")),
                    m.get("q_mean", float("nan")),
                )
        if do_save and (step + 1) % save_interval == 0 and (step + 1) < max_steps:
            _save_intermediate_ckpt(worker, cfg, variant, out_dir, step + 1)
    return metrics


def _stage2_online_sim(
    worker,
    buf,
    cfg,
    _device: str,
    use_bf16: bool = False,
    save_interval: int = 0,
    out_dir: Path | None = None,
    variant: str = "rlt_online_sim",
) -> list[dict]:
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
    do_save = save_interval and save_interval > 0 and out_dir is not None

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
        if use_bf16 and adapted["z_obs"].is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                m = worker.stage2_step(adapted, step=step)
        else:
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
        if do_save and (step + 1) % save_interval == 0 and (step + 1) < max_steps:
            _save_intermediate_ckpt(worker, cfg, variant, out_dir, step + 1)
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
        choices=["qrt", "a1_frozen_encoder", "rlt_online_sim", "a1_raw_features"],
        default="qrt",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override auto device selection (cpu/cuda/cuda:N).",
    )
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument(
        "--use_iql",
        action="store_true",
        help=(
            "Activate the σ-QRT IQL variant: V-network + expectile regression "
            "+ advantage-weighted actor BC. Avoids Q-extrapolation collapse "
            "seen in TD3+BC β sweep Phase 1. Sets cfg.training.use_iql=true "
            "and requires the IQL hparams (iql_tau, iql_beta, iql_weight_clip) "
            "in cfg.training — see configs/libero_long_qrt_iql_*.yaml."
        ),
    )
    p.add_argument(
        "--use_cql",
        action="store_true",
        help=(
            "Activate G2 v3 CQL conservative penalty on top of IQL. Adds "
            "α_cql · (logsumexp_a Q(s,a) − Q(s, a_data)) to the Q-loss, "
            "pushing Q(s, OOD_a) DOWN relative to in-distribution actions. "
            "Sets cfg.training.use_cql=true. Requires --use_iql (or "
            "use_iql=true in yaml) — CQL is implemented as an additive term "
            "on the IQL Q-update, not a standalone algorithm."
        ),
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help=(
            "Enable torch.autocast(bf16) for encoder/decoder/actor/critic "
            "compute. ~10x speedup on B200 (flash attention dispatch + tensor "
            "cores). bf16 has the same 8-bit exponent range as fp32, so no "
            "GradScaler is needed. Recommended for paper-faithful runs on "
            "GPU; CPU smoke tests should leave this off."
        ),
    )
    p.add_argument(
        "--save_interval",
        type=int,
        default=2000,
        help=(
            "Save an intermediate ckpt every N Stage-2 steps "
            "(``<output_dir>/ckpt_step{N}.pt``) plus a refreshed "
            "``ckpt_latest.pt`` snapshot. Set to ``-1`` or ``0`` to disable "
            "(only the final ``ckpt.pt`` is written). Default 2000."
        ),
    )
    p.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help=(
            "Path to a pre-warmed encoder/decoder ckpt produced by "
            "``experiments/extract_encoder_ckpt.py``. When set, encoder + "
            "decoder state_dicts are loaded into the worker after setup() "
            "and Stage-1 warmup is SKIPPED (saves ~100 min/lane on B200). "
            "Encoder/decoder dims in the cache must match cfg.model — a "
            "mismatch errors out. Ignored (with warning) for the "
            "``a1_raw_features`` variant since that variant bypasses the "
            "encoder entirely."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.override)

    # CLI --use_iql wins over yaml; preserves yaml's use_iql=true if set.
    if args.use_iql:
        if "training" not in cfg:
            cfg.training = {}
        cfg.training.use_iql = True

    # CLI --use_cql wins over yaml; preserves yaml's use_cql=true if set.
    # Requires use_iql (CQL is implemented as additive term on IQL Q-loss).
    if args.use_cql:
        if "training" not in cfg:
            cfg.training = {}
        cfg.training.use_cql = True
        if not cfg.training.get("use_iql", False):
            raise ValueError(
                "--use_cql requires --use_iql (or training.use_iql=true in yaml). "
                "CQL is an additive penalty on the IQL Q-loss, not standalone."
            )

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
        "variant=%s device=%s bf16=%s buffer=%s warmup_steps=%d max_train_steps=%d",
        args.variant,
        device,
        args.bf16,
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

    # Pre-warmed encoder cache → load encoder + decoder, then skip Stage 1.
    # For a1_raw_features the encoder is bypassed entirely so the cache is
    # meaningless; warn + ignore. For all other variants the cache replaces
    # the Stage-1 warmup (≈100 min/lane saved on B200).
    skip_stage1_via_cache = False
    if args.encoder_ckpt is not None:
        if args.variant == "a1_raw_features":
            log.warning(
                "[run_qrt] --encoder_ckpt=%s ignored: variant=a1_raw_features "
                "bypasses the encoder",
                args.encoder_ckpt,
            )
        else:
            _load_encoder_cache(worker, cfg, args.encoder_ckpt)
            skip_stage1_via_cache = True

    # Stage 1 (warmup). Skipped for a1_raw_features (encoder bypassed; no
    # representation to train against recon loss) and skipped when a
    # pre-warmed --encoder_ckpt is loaded.
    metrics_log: list[dict] = []
    if args.variant == "a1_raw_features":
        log.info("variant=a1_raw_features → Stage 1 token warmup SKIPPED")
    elif skip_stage1_via_cache:
        log.info(
            "--encoder_ckpt=%s loaded → Stage 1 token warmup SKIPPED",
            args.encoder_ckpt,
        )
    else:
        metrics_log.extend(
            _run_stage1(worker, buf, cfg, args.variant, device, use_bf16=args.bf16)
        )

    # Freeze encoder for non-qrt variants. For a1_raw_features the encoder
    # is already bypassed (and frozen at random init) since worker.setup(),
    # so freeze_encoder() is a no-op — but call it for uniformity.
    if args.variant != "qrt":
        worker.freeze_encoder()
        log.info("encoder frozen (variant=%s)", args.variant)

    # Stage 2. Pass save_interval + out_dir so intermediate snapshots can be
    # written; the helper short-circuits when save_interval <= 0.
    save_interval = int(args.save_interval)
    log.info(
        "stage2 save_interval=%d (intermediate ckpts %s)",
        save_interval,
        "ENABLED" if save_interval > 0 else "disabled",
    )
    if args.variant == "rlt_online_sim":
        metrics_log.extend(
            _stage2_online_sim(
                worker,
                buf,
                cfg,
                device,
                use_bf16=args.bf16,
                save_interval=save_interval,
                out_dir=out_dir,
                variant=args.variant,
            )
        )
    else:
        metrics_log.extend(
            _stage2_offline(
                worker,
                buf,
                cfg,
                args.variant,
                use_bf16=args.bf16,
                save_interval=save_interval,
                out_dir=out_dir,
            )
        )

    # Persist metrics + final ckpt.
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_log, indent=2))

    ckpt = _build_ckpt(worker, cfg, args.variant)
    _atomic_torch_save(ckpt, out_dir / "ckpt.pt")
    # Also refresh ckpt_latest.pt so consumers always have a "latest" handle
    # whether they're using interval snapshots or not.
    _atomic_torch_save(ckpt, out_dir / "ckpt_latest.pt")

    # .training_done sentinel — signals periodic_eval_watcher.py to drain
    # any remaining un-evaluated ckpt_step*.pt files and exit cleanly.
    (out_dir / ".training_done").touch()

    log.info(
        "done. metrics=%s ckpt=%s n_log_entries=%d",
        metrics_path,
        out_dir / "ckpt.pt",
        len(metrics_log),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
