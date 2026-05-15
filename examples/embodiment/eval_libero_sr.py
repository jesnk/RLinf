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

"""σ-QRT Task 11 — LIBERO success-rate (SR) eval entry.

Modes
-----
- ``--ckpt`` not provided (default): B0 baseline. Roll out frozen π0.5
  with ``predict_action_batch`` and execute the reference chunk directly.
- ``--ckpt PATH``: load a σ-QRT trained ckpt (saved by
  ``run_qrt_offline.py``) and use its actor to refine the π0.5
  reference chunk via ``actor(z_rl, s_p, ref_action, training=False)``.

Outputs
-------
- ``--output`` JSON: ``{"sr", "n_success", "n_eval", "per_task_sr",
  "ckpt"}``.

Usage
-----
    PYTHONPATH=$(pwd) python examples/embodiment/eval_libero_sr.py \\
        --config examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml \\
        --num_eval 50 --output runs/b0_eval.json
    # σ-QRT trained:
    PYTHONPATH=$(pwd) python examples/embodiment/eval_libero_sr.py \\
        --config ... --ckpt runs/qrt_seed1/ckpt.pt \\
        --num_eval 50 --output runs/qrt_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Bootstrap EGL BEFORE robosuite import (matches collect_base_vla_rollouts).
sys.path.insert(0, str(Path(__file__).parent))
from _sigma_qrt_helpers import apply_overrides, bootstrap_gl_env  # noqa: E402

bootstrap_gl_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# Reuse the collect script's env/VLA construction helpers (Task 8). These
# functions are already validated by tests/sigma_qrt/test_collect_rollouts.py.
from examples.embodiment.collect_base_vla_rollouts import (  # noqa: E402
    _build_model_cfg,
    _canonical_suite_name,
    _env_obs_for_vla,
    _proprio_from_obs,
    _to_python,
)

logging.basicConfig(
    format="[eval_libero_sr] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Build helpers tailored to the σ-QRT config (which is sparser than the
# collect config).
# --------------------------------------------------------------------------- #
def _qrt_to_model_cfg(cfg_model) -> "OmegaConf":
    """Augment the σ-QRT model config with defaults openpi expects.

    The σ-QRT yaml only has the fields the worker needs (token_dim,
    chunk_len, action_dim, proprio_dim, vla_ckpt). openpi additionally
    expects `config_name`, `num_images_in_input`, `num_steps`, etc. — we
    fall back to the LIBERO-10 SFT defaults used in the collect config.
    """
    augmented = OmegaConf.merge(
        OmegaConf.create(
            {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "num_steps": 5,
            }
        ),
        cfg_model,
    )
    return _build_model_cfg(augmented)


def _qrt_to_env_cfg(cfg_env, max_episode_len: int, seed: int) -> "OmegaConf":
    """Build the env DictConfig expected by LiberoEnv from the σ-QRT yaml.

    The σ-QRT env block only has name / num_tasks / max_episode_len /
    num_eval_episodes, so we apply the LIBERO-10 eval defaults.
    """
    suite = _canonical_suite_name(str(cfg_env.get("task_suite_name", cfg_env.name)))
    return OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": suite,
            "total_num_envs": 1,
            "auto_reset": bool(cfg_env.get("auto_reset", False)),
            "ignore_terminations": bool(cfg_env.get("ignore_terminations", False)),
            "max_steps_per_rollout_epoch": int(max_episode_len),
            "max_episode_steps": int(max_episode_len),
            "use_fixed_reset_state_ids": bool(
                cfg_env.get("use_fixed_reset_state_ids", True)
            ),
            "use_ordered_reset_state_ids": True,
            "use_rel_reward": bool(cfg_env.get("use_rel_reward", True)),
            "reward_coef": float(cfg_env.get("reward_coef", 5.0)),
            "reset_gripper_open": bool(cfg_env.get("reset_gripper_open", True)),
            "is_eval": True,
            "seed": int(seed),
            "group_size": int(cfg_env.get("group_size", 1)),
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/sigma_qrt_eval_video_disabled",
            },
            "init_params": {
                "camera_heights": int(cfg_env.get("camera_heights", 256)),
                "camera_widths": int(cfg_env.get("camera_widths", 256)),
            },
        }
    )


def _load_vla(cfg_model, device: str):
    from rlinf.models.embodiment.openpi import get_model

    model_cfg = _qrt_to_model_cfg(cfg_model)
    log.info("loading π0.5 VLA from %s", model_cfg.model_path)
    model = get_model(model_cfg)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _make_env(cfg_env, max_episode_len: int, seed: int):
    from rlinf.envs.libero.libero_env import LiberoEnv

    env_cfg = _qrt_to_env_cfg(cfg_env, max_episode_len, seed)
    log.info(
        "creating LiberoEnv suite=%s max_steps=%d seed=%d",
        env_cfg.task_suite_name,
        env_cfg.max_episode_steps,
        env_cfg.seed,
    )
    return LiberoEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


# --------------------------------------------------------------------------- #
# σ-QRT actor reload (optional)
# --------------------------------------------------------------------------- #
def _maybe_load_worker(cfg, ckpt_path: str | None, device: str):
    """Return a populated worker (encoder + actor) or None for B0."""
    if not ckpt_path:
        return None
    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import (
        QRTOfflinePolicyWorker,
    )

    worker = QRTOfflinePolicyWorker(cfg, device=device)
    worker.setup()
    payload = torch.load(ckpt_path, map_location=device)
    worker.encoder.load_state_dict(payload["encoder"])
    worker.actor.load_state_dict(payload["actor"])
    if "decoder" in payload:
        worker.decoder.load_state_dict(payload["decoder"])
    if "critic" in payload:
        worker.critic.load_state_dict(payload["critic"])
    # Honor training-time residual-actor flag from the saved cfg, in case the
    # caller's yaml/override doesn't set it. Eval-side actor call branches on
    # worker.use_residual_actor so this flag MUST match training. Falls back
    # to whatever the constructor cfg said (back-compat for non-residual ckpts).
    saved_cfg = payload.get("cfg")
    if isinstance(saved_cfg, dict):
        try:
            tr = saved_cfg.get("training", {})
            if "use_residual_actor" in tr:
                worker.use_residual_actor = bool(tr["use_residual_actor"])
                log.info(
                    "ckpt cfg → use_residual_actor=%s",
                    worker.use_residual_actor,
                )
        except Exception as e:
            log.warning("failed to read residual flag from payload cfg: %s", e)
    worker.encoder.eval()
    worker.actor.eval()
    for p in worker.encoder.parameters():
        p.requires_grad = False
    for p in worker.actor.parameters():
        p.requires_grad = False
    log.info("loaded σ-QRT ckpt from %s", ckpt_path)
    return worker


# --------------------------------------------------------------------------- #
# Rollout loop (one episode)
# --------------------------------------------------------------------------- #
def _eval_one_episode(
    vla,
    worker,
    env,
    cfg_model,
    max_episode_len: int,
    device: str,
    zero_delta: bool = False,
) -> tuple[int, int, list[float]]:
    """Roll out a single episode; return (success_flag, steps_taken, delta_norms).

    ``delta_norms`` is a per-VLA-step list of ||Δ||₂ values (mean across the
    chunk×action dims, batch=1) populated ONLY when ``worker`` is in residual-
    actor mode. Empty list otherwise (B0 or non-residual ckpts). When
    ``zero_delta=True`` Δ is zeroed AFTER measurement, so the executed chunk
    equals ref but the logged norm reflects the *would-be* correction.
    """
    chunk_len = int(cfg_model.chunk_len)
    action_dim = int(cfg_model.action_dim)

    obs, _ = env.reset()
    success = 0
    steps_taken = 0
    done = False
    delta_norms: list[float] = []
    is_residual = worker is not None and getattr(worker, "use_residual_actor", False)

    while not done and steps_taken < max_episode_len:
        env_obs = _env_obs_for_vla(obs)
        with torch.no_grad():
            actions_chunk, _ = vla.predict_action_batch(env_obs, mode="eval")
            if actions_chunk.dim() == 2:
                actions_chunk = actions_chunk.view(1, chunk_len, action_dim)
            ref_chunk = actions_chunk[0].detach().float()  # [C, A]

            if worker is not None:
                # σ-QRT refinement path.
                z = vla.extract_embeddings(env_obs).float().to(device)  # [1, M, d]
                z_rl = worker.encoder(z)
                s_p = _proprio_from_obs(obs).unsqueeze(0).to(device)
                ref_in = ref_chunk.unsqueeze(0).to(device)  # [1, C, A]
                if is_residual:
                    delta = worker.actor(
                        z_rl, s_p, ref_in, training=False, residual=True
                    )
                    # Per VLA-step Δ-norm: L2 over action_dim → [B, C] then mean
                    # across batch and chunk. Matches the diag spec.
                    delta_norm_step = delta.pow(2).sum(dim=-1).sqrt().mean().item()
                    delta_norms.append(float(delta_norm_step))
                    if zero_delta:
                        delta = torch.zeros_like(delta)
                    refined = (ref_in + delta)[0].cpu()
                else:
                    refined = worker.actor(z_rl, s_p, ref_in, training=False)[0].cpu()
                action_chunk = refined
            else:
                action_chunk = ref_chunk.cpu()

        for c in range(chunk_len):
            if done:
                break
            step_action = action_chunk[c].numpy().reshape(1, action_dim)
            obs, _r, terms, truncs, info = env.step(step_action)
            steps_taken += 1
            term_val = bool(_to_python(terms, idx=0))
            trunc_val = bool(_to_python(truncs, idx=0))
            if term_val:
                success = 1
                done = True
            elif trunc_val:
                done = True
    return success, steps_taken, delta_norms


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="σ-QRT LIBERO SR eval entry.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="σ-QRT ckpt (skip for B0 zero-shot eval).",
    )
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--max_episode_len", type=int, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable OmegaConf override.",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help=(
            "Number of parallel env worker subprocesses. 1 = sequential "
            "(backwards compat path). N>1 = async pattern: N env workers "
            "feed a central VLA/worker inference server via mp.Queue."
        ),
    )
    p.add_argument(
        "--zero_delta",
        action="store_true",
        help=(
            "Diagnostic (residual-actor mode only): force Δ=0 after computing "
            "it so the refined action equals the π0.5 reference. SR should "
            "equal B0 if the residual eval path is correct AND Δ≈0 hypothesis "
            "holds. Lower SR → worker setup drift (encoder/normalization). "
            "Has no effect on B0 (no worker) or non-residual ckpts."
        ),
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Async path: per-env subprocess workers + central inference server
# --------------------------------------------------------------------------- #
def _env_worker(
    env_id: int,
    env_cfg_dict: dict,
    max_episode_len: int,
    seed: int,
    chunk_len: int,
    action_dim: int,
    num_tasks: int,
    max_episodes: int,
    total_num_workers: int,
    req_queue,
    resp_queue,
    result_queue,
    log_level: int,
) -> None:
    """Run autonomous episode loop for one env, talking to the main
    inference server via queues.

    Per-message protocol:
      req_queue.put((env_id, obs_for_vla_dict, proprio_np))
      resp_queue.get() -> numpy.ndarray of shape [chunk_len, action_dim]

    Per result:
      result_queue.put({"env_id", "ep_idx", "success", "steps", "task_id"})
    """
    # Bootstrap inside the subprocess — needed because robosuite/MuJoCo
    # doesn't fork cleanly. CUDA_VISIBLE_DEVICES + MUJOCO_EGL_DEVICE_ID
    # are inherited from main via spawn.
    sys.path.insert(0, str(Path(__file__).parent))
    from _sigma_qrt_helpers import bootstrap_gl_env

    bootstrap_gl_env()
    import logging as _logging

    _logging.basicConfig(
        format=f"[env_worker {env_id}] %(asctime)s %(levelname)s %(message)s",
        level=log_level,
        datefmt="%H:%M:%S",
    )
    _log = _logging.getLogger(f"env_worker_{env_id}")

    # Re-import torch/np inside the subprocess.
    import numpy as _np  # noqa: F811
    import torch as _torch  # noqa: F811
    from omegaconf import OmegaConf as _OmegaConf

    from examples.embodiment.collect_base_vla_rollouts import (
        _env_obs_for_vla as _env_obs_for_vla_sub,
    )
    from examples.embodiment.collect_base_vla_rollouts import (
        _proprio_from_obs as _proprio_from_obs_sub,
    )
    from examples.embodiment.collect_base_vla_rollouts import (
        _to_python as _to_python_sub,
    )
    from rlinf.envs.libero.libero_env import LiberoEnv

    # Diversify seeds across workers so episodes don't repeat.
    _torch.manual_seed(seed + env_id * 1000)
    _np.random.seed(seed + env_id * 1000)

    env_cfg = _OmegaConf.create(env_cfg_dict)
    # LiberoEnv shards `reset_state_ids_all` along the first axis by
    # `total_num_processes`, then each worker indexes row `seed_offset`.
    # So we tell each worker the global pool size + its own slice id,
    # giving disjoint init-state sequences across workers.
    env = LiberoEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=env_id,
        total_num_processes=int(total_num_workers),
        worker_info=None,
    )
    _log.info("env ready; will run %d episodes", max_episodes)

    eps_done = 0
    try:
        while eps_done < max_episodes:
            obs, _ = env.reset()
            done = False
            success = 0
            steps_taken = 0
            while not done and steps_taken < max_episode_len:
                # Package obs for VLA. Move tensors to CPU + detach so they
                # serialize cleanly across the queue (spawn pickles them).
                env_obs = _env_obs_for_vla_sub(obs)
                payload = {}
                for k, v in env_obs.items():
                    if v is None:
                        payload[k] = None
                    elif _torch.is_tensor(v):
                        payload[k] = v.detach().cpu()
                    else:
                        payload[k] = v
                proprio = _proprio_from_obs_sub(obs).detach().cpu().numpy()

                req_queue.put((env_id, payload, proprio))
                action_chunk = resp_queue.get()  # np.ndarray [C, A]

                for c in range(chunk_len):
                    if done:
                        break
                    step_action = action_chunk[c].reshape(1, action_dim)
                    obs, _r, terms, truncs, info = env.step(step_action)
                    steps_taken += 1
                    term_val = bool(_to_python_sub(terms, idx=0))
                    trunc_val = bool(_to_python_sub(truncs, idx=0))
                    if term_val:
                        success = 1
                        done = True
                    elif trunc_val:
                        done = True

            # Episode finished. task_id semantics mirror the sequential
            # path (`ep % max(1, num_tasks)`), but each worker has its
            # own ep counter so the main process re-derives per-task
            # bucketing from a global episode index it sends down.
            result_queue.put(
                {
                    "env_id": env_id,
                    "worker_ep_idx": eps_done,
                    "success": int(success),
                    "steps": int(steps_taken),
                }
            )
            eps_done += 1
    except Exception as e:  # pragma: no cover - subprocess fatal path
        _log.exception("worker crashed: %s", e)
        # Signal a failure so main doesn't hang.
        try:
            result_queue.put(
                {
                    "env_id": env_id,
                    "worker_ep_idx": eps_done,
                    "success": 0,
                    "steps": 0,
                    "error": str(e),
                }
            )
        except Exception:
            pass
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_async_eval(args, cfg, device: str, max_episode_len: int) -> dict:
    """Async eval: spawn N env workers, serve inference centrally."""
    import multiprocessing as mp
    import queue as _queue

    # mp.spawn so robosuite/MuJoCo doesn't fork an initialized GL state.
    ctx = mp.get_context("spawn")

    chunk_len = int(cfg.model.chunk_len)
    action_dim = int(cfg.model.action_dim)
    num_tasks = int(cfg.env.get("num_tasks", 1))
    num_envs = int(args.num_envs)
    num_eval = int(args.num_eval)
    assert num_envs >= 1
    assert num_eval >= 1

    # Even-ish split (first num_eval % num_envs workers get +1 episode).
    eps_per_worker = [
        (num_eval // num_envs) + (1 if i < (num_eval % num_envs) else 0)
        for i in range(num_envs)
    ]
    log.info(
        "async eval: num_envs=%d num_eval=%d split=%s",
        num_envs,
        num_eval,
        eps_per_worker,
    )

    # Serialize env_cfg to a plain dict for queue transport.
    env_cfg_omega = _qrt_to_env_cfg(cfg.env, max_episode_len, seed=args.seed)
    env_cfg_dict = OmegaConf.to_container(env_cfg_omega, resolve=True)

    # Load VLA + optional worker in the MAIN process only.
    vla = _load_vla(cfg.model, device=device)
    worker = _maybe_load_worker(cfg, args.ckpt, device=device)

    req_queue = ctx.Queue()
    result_queue = ctx.Queue()
    resp_queues = [ctx.Queue() for _ in range(num_envs)]

    procs = []
    log_level = log.getEffectiveLevel()
    for i in range(num_envs):
        if eps_per_worker[i] <= 0:
            continue
        p = ctx.Process(
            target=_env_worker,
            args=(
                i,
                env_cfg_dict,
                max_episode_len,
                int(args.seed),
                chunk_len,
                action_dim,
                num_tasks,
                eps_per_worker[i],
                num_envs,
                req_queue,
                resp_queues[i],
                result_queue,
                log_level,
            ),
            daemon=False,
        )
        p.start()
        procs.append((i, p))
    log.info("launched %d env worker subprocesses", len(procs))

    # Serve inference + drain results.
    n_completed = 0
    n_success = 0
    per_task: dict[int, list[int]] = {}
    # Each worker gets a base ep index = sum of previous workers' eps.
    base_ep_idx = [sum(eps_per_worker[:i]) for i in range(num_envs)]

    is_residual = worker is not None and getattr(worker, "use_residual_actor", False)
    zero_delta = bool(getattr(args, "zero_delta", False))
    if zero_delta and not is_residual:
        log.warning(
            "--zero_delta passed but worker is not in residual mode; flag has "
            "no effect."
        )
    # Per-env accumulator for the in-progress episode. Flushed on each result.
    pending_deltas: dict[int, list[float]] = {i: [] for i in range(num_envs)}
    ep_delta_stats: list[dict] = []

    try:
        while n_completed < num_eval:
            # Drain finished episodes.
            try:
                while True:
                    r = result_queue.get_nowait()
                    n_completed += 1
                    n_success += int(r.get("success", 0))
                    global_ep_idx = base_ep_idx[r["env_id"]] + r["worker_ep_idx"]
                    task_id = global_ep_idx % max(1, num_tasks)
                    per_task.setdefault(task_id, []).append(int(r.get("success", 0)))
                    # Flush this env's pending Δ-norms as the just-finished
                    # episode's stats (residual mode only).
                    ev = int(r["env_id"])
                    if is_residual and pending_deltas[ev]:
                        dn = pending_deltas[ev]
                        ep_delta_stats.append(
                            {
                                "mean": float(np.mean(dn)),
                                "max": float(np.max(dn)),
                                "final": float(dn[-1]),
                            }
                        )
                    pending_deltas[ev] = []
                    log.info(
                        "ep %d/%d (env=%d worker_ep=%d) success=%d steps=%d "
                        "cumulative=%d",
                        n_completed,
                        num_eval,
                        r["env_id"],
                        r["worker_ep_idx"],
                        int(r.get("success", 0)),
                        int(r.get("steps", 0)),
                        n_success,
                    )
            except _queue.Empty:
                pass
            if n_completed >= num_eval:
                break
            # Serve next inference request.
            try:
                env_id, env_obs, proprio_np = req_queue.get(timeout=1.0)
            except _queue.Empty:
                continue
            # Move tensors back to device.
            env_obs_dev = {}
            for k, v in env_obs.items():
                if v is None:
                    env_obs_dev[k] = None
                elif torch.is_tensor(v):
                    env_obs_dev[k] = v.to(device, non_blocking=True)
                else:
                    env_obs_dev[k] = v

            with torch.no_grad():
                actions_chunk, _ = vla.predict_action_batch(env_obs_dev, mode="eval")
                if actions_chunk.dim() == 2:
                    actions_chunk = actions_chunk.view(1, chunk_len, action_dim)
                ref_chunk = actions_chunk[0].detach().float()  # [C, A]
                if worker is not None:
                    z = vla.extract_embeddings(env_obs_dev).float().to(device)
                    z_rl = worker.encoder(z)
                    s_p = (
                        torch.as_tensor(proprio_np, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                    ref_in = ref_chunk.unsqueeze(0).to(device)
                    if is_residual:
                        delta = worker.actor(
                            z_rl, s_p, ref_in, training=False, residual=True
                        )
                        delta_norm_step = delta.pow(2).sum(dim=-1).sqrt().mean().item()
                        pending_deltas[int(env_id)].append(float(delta_norm_step))
                        if zero_delta:
                            delta = torch.zeros_like(delta)
                        refined = (ref_in + delta)[0].cpu()
                    else:
                        refined = worker.actor(z_rl, s_p, ref_in, training=False)[
                            0
                        ].cpu()
                    action_chunk_cpu = refined
                else:
                    action_chunk_cpu = ref_chunk.cpu()
            # Ship numpy back over the queue (no autograd tape).
            resp_queues[env_id].put(action_chunk_cpu.numpy())
    finally:
        for env_id, p in procs:
            if p.is_alive():
                p.terminate()
        for env_id, p in procs:
            p.join(timeout=30)

    sr = n_success / max(1, num_eval)
    result = {
        "sr": sr,
        "n_success": int(n_success),
        "n_eval": int(num_eval),
        "per_task_sr": {
            int(k): float(sum(v) / len(v)) for k, v in per_task.items() if v
        },
        "ckpt": args.ckpt,
    }
    if is_residual:
        if ep_delta_stats:
            result["delta_norm_mean"] = float(
                np.mean([d["mean"] for d in ep_delta_stats])
            )
            result["delta_norm_max"] = float(np.max([d["max"] for d in ep_delta_stats]))
            result["delta_norm_final"] = float(
                np.mean([d["final"] for d in ep_delta_stats])
            )
        else:
            result["delta_norm_mean"] = float("nan")
            result["delta_norm_max"] = float("nan")
            result["delta_norm_final"] = float("nan")
        if zero_delta:
            result["zero_delta"] = True
    return result


# --------------------------------------------------------------------------- #
# Sequential path (backwards-compatible; preserved bit-for-bit)
# --------------------------------------------------------------------------- #
def _run_sequential_eval(args, cfg, device: str, max_episode_len: int) -> dict:
    vla = _load_vla(cfg.model, device=device)
    worker = _maybe_load_worker(cfg, args.ckpt, device=device)
    env = _make_env(cfg.env, max_episode_len, seed=args.seed)

    n_success = 0
    per_task: dict[int, list[int]] = {}
    # Per-episode Δ-norm aggregates (residual mode only). Each entry:
    # {"mean": float, "max": float, "final": float}. Empty for non-residual.
    ep_delta_stats: list[dict] = []
    is_residual = worker is not None and getattr(worker, "use_residual_actor", False)
    if args.zero_delta and not is_residual:
        log.warning(
            "--zero_delta passed but worker is not in residual mode "
            "(worker=%s); flag has no effect.",
            "None" if worker is None else "non-residual",
        )

    for ep in range(int(args.num_eval)):
        try:
            success, steps_taken, delta_norms = _eval_one_episode(
                vla,
                worker,
                env,
                cfg.model,
                max_episode_len,
                device,
                zero_delta=bool(args.zero_delta),
            )
        except Exception:
            log.exception("episode %d failed; marking failure", ep + 1)
            success, steps_taken, delta_norms = 0, 0, []
        n_success += success
        # Best-effort task_id from env (LIBERO doesn't surface it in obs;
        # cycle through num_tasks deterministically).
        num_tasks = int(cfg.env.get("num_tasks", 1))
        task_id = ep % max(1, num_tasks)
        per_task.setdefault(task_id, []).append(success)
        if is_residual and delta_norms:
            ep_delta_stats.append(
                {
                    "mean": float(np.mean(delta_norms)),
                    "max": float(np.max(delta_norms)),
                    "final": float(delta_norms[-1]),
                }
            )
            log.info(
                "ep %d/%d success=%d steps=%d cumulative=%d Δ(mean/max/final)=%.4f/%.4f/%.4f",
                ep + 1,
                args.num_eval,
                success,
                steps_taken,
                n_success,
                ep_delta_stats[-1]["mean"],
                ep_delta_stats[-1]["max"],
                ep_delta_stats[-1]["final"],
            )
        else:
            log.info(
                "ep %d/%d success=%d steps=%d cumulative=%d",
                ep + 1,
                args.num_eval,
                success,
                steps_taken,
                n_success,
            )

    sr = n_success / max(1, int(args.num_eval))
    result = {
        "sr": sr,
        "n_success": int(n_success),
        "n_eval": int(args.num_eval),
        "per_task_sr": {
            int(k): float(sum(v) / len(v)) for k, v in per_task.items() if v
        },
        "ckpt": args.ckpt,
    }
    # Residual-only diagnostics. Skipped entirely for B0 / non-residual ckpts
    # so existing JSON consumers (coord, summary scripts) see no new keys.
    if is_residual:
        if ep_delta_stats:
            result["delta_norm_mean"] = float(
                np.mean([d["mean"] for d in ep_delta_stats])
            )
            result["delta_norm_max"] = float(np.max([d["max"] for d in ep_delta_stats]))
            result["delta_norm_final"] = float(
                np.mean([d["final"] for d in ep_delta_stats])
            )
        else:
            result["delta_norm_mean"] = float("nan")
            result["delta_norm_max"] = float("nan")
            result["delta_norm_final"] = float("nan")
        if args.zero_delta:
            result["zero_delta"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.override)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_episode_len = int(
        args.max_episode_len
        if args.max_episode_len is not None
        else cfg.env.get("max_episode_len", 600)
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info(
        "config=%s ckpt=%s num_eval=%d num_envs=%d max_ep_len=%d device=%s",
        args.config,
        args.ckpt,
        args.num_eval,
        args.num_envs,
        max_episode_len,
        device,
    )

    if int(args.num_envs) <= 1:
        result = _run_sequential_eval(args, cfg, device, max_episode_len)
    else:
        result = _run_async_eval(args, cfg, device, max_episode_len)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info(
        "SR=%.3f n_success=%d/%d → %s",
        result["sr"],
        result["n_success"],
        result["n_eval"],
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
