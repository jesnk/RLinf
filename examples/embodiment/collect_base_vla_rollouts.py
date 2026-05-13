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

"""σ-QRT Task 8 — Base VLA rollout collection for the offline buffer.

Rolls out frozen π0.5 SFT on a single LIBERO env, packages each episode as
a `Trajectory` (`rlinf.data.embodied_io_struct.Trajectory`) with σ-QRT
specific fields embedded in `curr_obs` / `next_obs`, and pickles the list
so that `TrajectoryReplayBuffer.from_offline_dataset()` (Task 5) can load
it as the offline replay buffer.

Output schema (single-env collection so B=1, T = number of action chunks):

    Trajectory(
        max_episode_length=T,
        model_weights_id="pi05_libero_sft",
        actions      = [T, 1, chunk_len, action_dim]       # taken action chunk
        rewards      = [T, 1, chunk_len]                   # per-step rewards
        dones        = [T, 1]                              # terminal flag
        terminations = [T, 1]
        truncations  = [T, 1]
        prev_logprobs= [T, 1, action_dim]                  # placeholder zeros
        curr_obs = {
            "z_obs":     [T, 1, M, d]    # VLA hidden states at s_t
            "s_p":       [T, 1, proprio_dim]
            "ref_action":[T, 1, chunk_len, action_dim]     # π0.5 reference chunk
        },
        next_obs = {  # same keys, observations at s_{t+1}
            "z_obs":     [T, 1, M, d]
            "s_p":       [T, 1, proprio_dim]
            "ref_action":[T, 1, chunk_len, action_dim]
        },
    )

The Task 6 worker contract (`z_obs`, `next_z_obs`, `s_p`, `next_s_p`,
`action`, `ref_action`, `next_ref_action`, `reward`, `done`) is satisfied
by Task 11's adapter pulling these fields out of `curr_obs` / `next_obs`
after `TrajectoryReplayBuffer.sample()`.

Usage
-----
    PYTHONPATH=$(pwd) python examples/embodiment/collect_base_vla_rollouts.py \\
        --config examples/embodiment/config/libero_long_collect_rollouts_pi05.yaml \\
        --num_episodes 200 \\
        --output data/sigma_qrt/libero_long/transitions_B5k.pkl \\
        --seed 1
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

# LIBERO/robosuite headless rendering needs a GL backend (cf.
# examples/embodiment/eval_embodiment.sh + run_embodiment.sh). brain1 ships
# EGL/Mesa but NOT OSMesa, and the NVIDIA EGL vendor lives under a non-system
# path. So we auto-bootstrap EGL (with NVIDIA driver) here BEFORE robosuite
# is ever imported. If the caller already set MUJOCO_GL we respect it.
def _bootstrap_gl_env() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("LIBERO_TYPE", "standard")
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

    if os.environ["MUJOCO_GL"].lower() != "egl":
        return

    # NVIDIA EGL ICD discovery: prefer the standard glvnd path, fall back to
    # the workspace-local NVIDIA root that brain1 ships.
    nvidia_egl_root_candidates = [
        os.environ.get("NVIDIA_EGL_ROOT"),
        "/NHNHOME/WORKSPACE/0426030005_A/jisuan/dummy/.nvidia-egl-root",
    ]
    for root in nvidia_egl_root_candidates:
        if not root:
            continue
        icd_dir = f"{root}/usr/share/glvnd/egl_vendor.d"
        lib_dir = f"{root}/usr/lib/x86_64-linux-gnu"
        if (
            os.path.isfile(f"{icd_dir}/10_nvidia.json")
            and os.path.isfile(f"{lib_dir}/libEGL_nvidia.so.0")
        ):
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", icd_dir)
            existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in existing_ld.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{lib_dir}:{existing_ld}" if existing_ld else lib_dir
                )
            break


_bootstrap_gl_env()

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    format="[collect_rollouts] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LIBERO suite name canonicalization
# --------------------------------------------------------------------------- #
# The paper / Task 7 config calls it "libero_long" but LIBERO's benchmark
# registry only knows it as "libero_10".
SUITE_NAME_MAP = {
    "libero_long": "libero_10",
    "libero-long": "libero_10",
    "long": "libero_10",
}


def _canonical_suite_name(name: str) -> str:
    """Map paper-facing suite names to LIBERO's internal registry names."""
    return SUITE_NAME_MAP.get(name.lower(), name)


# --------------------------------------------------------------------------- #
# Model construction — mirrors tests/sigma_qrt/test_extract_embeddings.py
# --------------------------------------------------------------------------- #
def _build_model_cfg(cfg_model: DictConfig) -> DictConfig:
    """Build the OmegaConf cfg consumed by `rlinf.models.embodiment.openpi.get_model`.

    Mirrors `tests/sigma_qrt/test_extract_embeddings.py::_load_pi05_libero_sft`
    so we walk the canonical π0.5 load path (norm_stats, LiberoInputs/Outputs,
    safetensors weight load).
    """
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": str(cfg_model.vla_ckpt),
            "precision": None,
            "num_action_chunks": int(cfg_model.chunk_len),
            "action_dim": int(cfg_model.action_dim),
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": int(cfg_model.get("num_steps", 5)),
            "add_value_head": False,
            "openpi": {
                "config_name": str(cfg_model.config_name),
                "num_images_in_input": int(cfg_model.get("num_images_in_input", 2)),
                "noise_level": 0.5,
                "action_chunk": int(cfg_model.chunk_len),
                "num_steps": int(cfg_model.get("num_steps", 5)),
                "train_expert_only": True,
                "action_env_dim": int(cfg_model.action_dim),
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": None,
                "use_dsrl": False,
                "dsrl_state_dim": int(cfg_model.proprio_dim),
                "dsrl_action_noise_dim": 32,
                "dsrl_num_q_heads": 10,
                "dsrl_agg_q": "mean",
                "dsrl_image_latent_dim": 64,
                "dsrl_state_latent_dim": 64,
                "dsrl_hidden_dims": [128, 128, 128],
            },
        }
    )


def _load_vla(cfg_model: DictConfig, device: str):
    """Load + freeze the π0.5 SFT model."""
    from rlinf.models.embodiment.openpi import get_model

    log.info("Loading π0.5 SFT from %s", cfg_model.vla_ckpt)
    model_cfg = _build_model_cfg(cfg_model)
    model = get_model(model_cfg)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------- #
# LIBERO env construction
# --------------------------------------------------------------------------- #
def _build_env_cfg(cfg_env: DictConfig, max_episode_len: int, seed: int) -> DictConfig:
    """Build the DictConfig that `LiberoEnv.__init__` expects.

    Modelled after `examples/embodiment/config/env/libero_10.yaml`. Video
    saving is disabled to keep the script side-effect free for offline data
    generation.
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
            "is_eval": bool(cfg_env.get("is_eval", True)),
            "seed": int(seed),
            "group_size": int(cfg_env.get("group_size", 1)),
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/sigma_qrt_collect_video_disabled",
            },
            "init_params": {
                "camera_heights": int(cfg_env.get("camera_heights", 256)),
                "camera_widths": int(cfg_env.get("camera_widths", 256)),
            },
        }
    )


def _make_libero_env(cfg_env: DictConfig, max_episode_len: int, seed: int):
    """Instantiate a single-env `LiberoEnv` for sequential collection."""
    from rlinf.envs.libero.libero_env import LiberoEnv

    env_cfg = _build_env_cfg(cfg_env, max_episode_len, seed)
    log.info(
        "Creating LiberoEnv (suite=%s, max_steps=%d, seed=%d)",
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
# Obs preparation for the VLA
# --------------------------------------------------------------------------- #
def _env_obs_for_vla(obs: dict) -> dict:
    """LIBERO `_wrap_obs` dict → π0.5 `env_obs` dict consumed by
    `extract_embeddings` / `predict_action_batch`.

    LIBERO returns torch tensors for images (uint8) and states (float).
    π0.5 expects: main_images, wrist_images, extra_view_images, states,
    task_descriptions. We pass the tensors through as-is; the model's
    `obs_processor` + `input_transform` handle the rest.
    """
    return {
        "main_images": obs["main_images"],
        "wrist_images": obs.get("wrist_images"),
        "extra_view_images": obs.get("extra_view_images"),
        "states": obs["states"],
        "task_descriptions": obs["task_descriptions"],
    }


def _proprio_from_obs(obs: dict, idx: int = 0) -> torch.Tensor:
    """Extract the proprio vector for env index `idx` as 1D float tensor."""
    states = obs["states"]
    if not torch.is_tensor(states):
        states = torch.as_tensor(states)
    return states[idx].detach().float().flatten().cpu()


def _z_obs_storage_cast(z: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Cast z_obs to the configured storage dtype (default bfloat16)."""
    dtype_name = (dtype_name or "bfloat16").lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return z.to(dtype_map.get(dtype_name, torch.bfloat16)).cpu()


# --------------------------------------------------------------------------- #
# Rollout loop
# --------------------------------------------------------------------------- #
def _rollout_one_episode(
    vla,
    env,
    cfg_model: DictConfig,
    cfg_collect: DictConfig,
    max_episode_len: int,
    z_obs_dtype: str,
):
    """Roll out one episode end-to-end, returning lists of per-chunk fields.

    Returns
    -------
    dict with keys:
        z_obs_curr / z_obs_next:    list[Tensor[M, d]]    length = T
        s_p_curr   / s_p_next:      list[Tensor[proprio]]
        ref_curr   / ref_next:      list[Tensor[C, A]]
        actions:                    list[Tensor[C, A]]
        rewards:                    list[Tensor[C]]
        dones:                      list[float]
        terminations:               list[float]
        truncations:                list[float]
        prev_logprobs:              list[Tensor[A]]
        success:                    int
        steps_taken:                int
    """
    chunk_len = int(cfg_model.chunk_len)
    action_dim = int(cfg_model.action_dim)

    obs, _ = env.reset()

    # Buffers (one entry per chunk).
    z_obs_curr_l: list[torch.Tensor] = []
    s_p_curr_l: list[torch.Tensor] = []
    ref_curr_l: list[torch.Tensor] = []

    z_obs_next_l: list[torch.Tensor] = []
    s_p_next_l: list[torch.Tensor] = []
    ref_next_l: list[torch.Tensor] = []

    actions_l: list[torch.Tensor] = []
    rewards_l: list[torch.Tensor] = []
    dones_l: list[float] = []
    terms_l: list[float] = []
    truncs_l: list[float] = []
    logprob_l: list[torch.Tensor] = []

    success = 0
    steps_taken = 0
    episode_done = False

    while not episode_done and steps_taken < max_episode_len:
        # 1. Snapshot current observation (s_t).
        env_obs_t = _env_obs_for_vla(obs)
        with torch.no_grad():
            z_t = vla.extract_embeddings(env_obs_t)  # [1, M, d]
            # 2. Predict reference action chunk via π0.5's canonical inference path.
            actions_chunk, result = vla.predict_action_batch(
                env_obs_t, mode="eval"
            )
        # actions_chunk: [1, C, A] (already unnormalized via output_transform).
        if actions_chunk.dim() == 2:
            # Flat layout [1, C*A] — reshape.
            actions_chunk = actions_chunk.view(1, chunk_len, action_dim)

        ref_chunk = actions_chunk[0].detach().float().cpu()  # [C, A]

        # Take the chunk in env (no exploration noise in v1; cfg_collect.action_noise_std).
        taken_chunk = ref_chunk.clone()
        noise_std = float(cfg_collect.get("action_noise_std", 0.0))
        if noise_std > 0.0:
            taken_chunk = taken_chunk + torch.randn_like(taken_chunk) * noise_std

        # Per-step rewards within the chunk.
        chunk_rewards: list[float] = []
        chunk_terms: list[bool] = []
        chunk_truncs: list[bool] = []
        next_obs = obs
        for c in range(chunk_len):
            if episode_done:
                # Pad: replay last action, zero reward, sticky done.
                chunk_rewards.append(0.0)
                chunk_terms.append(False)
                chunk_truncs.append(False)
                continue
            step_action = taken_chunk[c].numpy().reshape(1, action_dim)
            next_obs, r, terms, truncs, info = env.step(step_action)
            # All return values are batched along the env axis (size 1).
            r_val = float(_to_python(r, idx=0))
            term_val = bool(_to_python(terms, idx=0))
            trunc_val = bool(_to_python(truncs, idx=0))
            chunk_rewards.append(r_val)
            chunk_terms.append(term_val)
            chunk_truncs.append(trunc_val)
            steps_taken += 1
            if term_val or trunc_val:
                episode_done = True
                if term_val:
                    success = 1

        # 3. Snapshot next observation (s_{t+1}) embedding + π0.5 ref chunk.
        env_obs_next = _env_obs_for_vla(next_obs)
        with torch.no_grad():
            z_next = vla.extract_embeddings(env_obs_next)  # [1, M, d]
            actions_next_chunk, _ = vla.predict_action_batch(
                env_obs_next, mode="eval"
            )
        if actions_next_chunk.dim() == 2:
            actions_next_chunk = actions_next_chunk.view(1, chunk_len, action_dim)
        ref_next_chunk = actions_next_chunk[0].detach().float().cpu()

        # 4. Append to buffers.
        z_obs_curr_l.append(_z_obs_storage_cast(z_t[0].detach(), z_obs_dtype))
        s_p_curr_l.append(_proprio_from_obs(obs))
        ref_curr_l.append(ref_chunk)

        z_obs_next_l.append(_z_obs_storage_cast(z_next[0].detach(), z_obs_dtype))
        s_p_next_l.append(_proprio_from_obs(next_obs))
        ref_next_l.append(ref_next_chunk)

        actions_l.append(taken_chunk)
        rewards_l.append(torch.tensor(chunk_rewards, dtype=torch.float32))
        dones_l.append(1.0 if episode_done else 0.0)
        terms_l.append(1.0 if any(chunk_terms) else 0.0)
        truncs_l.append(1.0 if any(chunk_truncs) else 0.0)
        # prev_logprobs placeholder: σ-QRT does not consume it offline; use zeros.
        logprob_l.append(torch.zeros(action_dim, dtype=torch.float32))

        obs = next_obs

    return {
        "z_obs_curr": z_obs_curr_l,
        "s_p_curr": s_p_curr_l,
        "ref_curr": ref_curr_l,
        "z_obs_next": z_obs_next_l,
        "s_p_next": s_p_next_l,
        "ref_next": ref_next_l,
        "actions": actions_l,
        "rewards": rewards_l,
        "dones": dones_l,
        "terminations": terms_l,
        "truncations": truncs_l,
        "prev_logprobs": logprob_l,
        "success": success,
        "steps_taken": steps_taken,
    }


def _to_python(x: Any, idx: int = 0) -> Any:
    """Coerce env return value (tensor/np/scalar) → python scalar at index 0."""
    if torch.is_tensor(x):
        return x.detach().reshape(-1)[idx].item()
    if isinstance(x, np.ndarray):
        return x.reshape(-1)[idx].item()
    if isinstance(x, (list, tuple)):
        return _to_python(x[idx])
    return x


# --------------------------------------------------------------------------- #
# Trajectory packaging
# --------------------------------------------------------------------------- #
def _pack_trajectory(episode: dict):
    """Stack per-chunk lists into a `Trajectory` with [T, B=1, ...] tensors."""
    from rlinf.data.embodied_io_struct import Trajectory

    T = len(episode["actions"])
    if T == 0:
        return None

    def _stack(lst):
        return torch.stack(lst, dim=0).unsqueeze(1)  # [T, B=1, ...]

    z_obs_curr = _stack(episode["z_obs_curr"])
    z_obs_next = _stack(episode["z_obs_next"])
    s_p_curr = _stack(episode["s_p_curr"])
    s_p_next = _stack(episode["s_p_next"])
    ref_curr = _stack(episode["ref_curr"])
    ref_next = _stack(episode["ref_next"])

    actions = _stack(episode["actions"])
    rewards = _stack(episode["rewards"])
    prev_logprobs = _stack(episode["prev_logprobs"])

    dones = torch.tensor(episode["dones"], dtype=torch.float32).unsqueeze(1)
    terminations = torch.tensor(
        episode["terminations"], dtype=torch.float32
    ).unsqueeze(1)
    truncations = torch.tensor(
        episode["truncations"], dtype=torch.float32
    ).unsqueeze(1)

    return Trajectory(
        max_episode_length=T,
        model_weights_id="pi05_libero_sft",
        actions=actions,
        rewards=rewards,
        dones=dones,
        terminations=terminations,
        truncations=truncations,
        prev_logprobs=prev_logprobs,
        curr_obs={
            "z_obs": z_obs_curr,
            "s_p": s_p_curr,
            "ref_action": ref_curr,
        },
        next_obs={
            "z_obs": z_obs_next,
            "s_p": s_p_next,
            "ref_action": ref_next,
        },
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect frozen-π0.5 rollout trajectories for σ-QRT offline buffer."
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Override cfg.collect.num_episodes.",
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--env_override",
        type=str,
        default=None,
        help="Override env name (alias for task suite).",
    )
    p.add_argument(
        "--max_episode_len",
        type=int,
        default=None,
        help="Override cfg.env.max_episode_len (smoke tests cap this).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = OmegaConf.load(args.config)

    if args.env_override:
        cfg.env.name = args.env_override

    num_episodes = int(
        args.num_episodes if args.num_episodes is not None else cfg.collect.num_episodes
    )
    max_episode_len = int(
        args.max_episode_len
        if args.max_episode_len is not None
        else cfg.env.max_episode_len
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(
        "config=%s env=%s num_episodes=%d max_ep_len=%d seed=%d device=%s",
        args.config,
        cfg.env.name,
        num_episodes,
        max_episode_len,
        args.seed,
        device,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vla = _load_vla(cfg.model, device=device)
    env = _make_libero_env(cfg.env, max_episode_len, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories = []
    z_obs_dtype = str(cfg.collect.get("save_z_obs_dtype", "bfloat16"))

    total_start = time.perf_counter()
    for ep in range(num_episodes):
        ep_start = time.perf_counter()
        try:
            episode = _rollout_one_episode(
                vla,
                env,
                cfg.model,
                cfg.collect,
                max_episode_len=max_episode_len,
                z_obs_dtype=z_obs_dtype,
            )
        except Exception:
            log.exception("episode %d failed; skipping", ep + 1)
            continue

        traj = _pack_trajectory(episode)
        if traj is None:
            log.warning("episode %d collected 0 chunks; skipping", ep + 1)
            continue
        trajectories.append(traj)

        elapsed = time.perf_counter() - ep_start
        log.info(
            "ep %d/%d chunks=%d steps=%d success=%d wall=%.1fs cumulative=%d",
            ep + 1,
            num_episodes,
            len(episode["actions"]),
            episode["steps_taken"],
            episode["success"],
            elapsed,
            len(trajectories),
        )

    total_elapsed = time.perf_counter() - total_start
    with out_path.open("wb") as f:
        pickle.dump(trajectories, f)

    total_chunks = sum(int(t.actions.shape[0]) for t in trajectories)
    log.info(
        "Saved %d trajectories (%d chunks) → %s (total wall=%.1fs)",
        len(trajectories),
        total_chunks,
        out_path,
        total_elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
