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
) -> tuple[int, int]:
    """Roll out a single episode; return (success_flag, steps_taken)."""
    chunk_len = int(cfg_model.chunk_len)
    action_dim = int(cfg_model.action_dim)

    obs, _ = env.reset()
    success = 0
    steps_taken = 0
    done = False

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
    return success, steps_taken


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
    return p.parse_args(argv)


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
        "config=%s ckpt=%s num_eval=%d max_ep_len=%d device=%s",
        args.config,
        args.ckpt,
        args.num_eval,
        max_episode_len,
        device,
    )

    vla = _load_vla(cfg.model, device=device)
    worker = _maybe_load_worker(cfg, args.ckpt, device=device)
    env = _make_env(cfg.env, max_episode_len, seed=args.seed)

    n_success = 0
    per_task: dict[int, list[int]] = {}
    for ep in range(int(args.num_eval)):
        try:
            success, steps_taken = _eval_one_episode(
                vla, worker, env, cfg.model, max_episode_len, device
            )
        except Exception:
            log.exception("episode %d failed; marking failure", ep + 1)
            success, steps_taken = 0, 0
        n_success += success
        # Best-effort task_id from env (LIBERO doesn't surface it in obs;
        # cycle through num_tasks deterministically).
        num_tasks = int(cfg.env.get("num_tasks", 1))
        task_id = ep % max(1, num_tasks)
        per_task.setdefault(task_id, []).append(success)
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
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("SR=%.3f n_success=%d/%d → %s", sr, n_success, args.num_eval, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
