"""σ-QRT Task 8: Base VLA rollout collection script smoke test.

End-to-end test: spawns `examples/embodiment/collect_base_vla_rollouts.py` as
a subprocess, rolls out 1 LIBERO episode with frozen π0.5 SFT, and verifies
that the output pickle file contains a list of Trajectory objects with the
σ-QRT canonical schema:

    curr_obs = {"z_obs", "s_p", "ref_action"}
    next_obs = {"z_obs", "s_p", "ref_action"}
    actions, rewards, dones, terminations, truncations, prev_logprobs

Note: this is a SLOW test (1 episode ~ 5-15 min on B200). It is gated on
`@pytest.mark.slow` and the env var `RUN_SLOW_TESTS=1`, so the full
sigma_qrt regression remains fast unless explicitly opted in.

Run:
    PYTHONPATH=. RUN_SLOW_TESTS=1 \
        pytest tests/sigma_qrt/test_collect_rollouts.py -v -m slow --timeout=1500
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples/embodiment/collect_base_vla_rollouts.py"
CONFIG = (
    REPO_ROOT
    / "examples/embodiment/config/libero_long_collect_rollouts_pi05.yaml"
)
CKPT = REPO_ROOT / "ckpts" / "pi05_libero_sft"


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS", "0") != "1",
    reason="slow integration test; set RUN_SLOW_TESTS=1 to enable",
)


@pytest.mark.slow
def test_collect_rollouts_smoke(tmp_path):
    """End-to-end: 1 LIBERO episode → pickled list of Trajectory objects."""
    if not SCRIPT.exists():
        pytest.fail(f"script missing: {SCRIPT}")
    if not CONFIG.exists():
        pytest.fail(f"config missing: {CONFIG}")
    if not CKPT.exists():
        pytest.skip(f"π0.5 SFT ckpt not present at {CKPT}")

    out = tmp_path / "transitions.pkl"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(CONFIG),
        "--num_episodes",
        "1",
        "--output",
        str(out),
        "--seed",
        "0",
        # Cap episode length for smoke test so wall-clock stays bounded.
        "--max_episode_len",
        "120",
    ]

    # Forward env so CUDA / openpi / venv all resolve.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    # LIBERO/robosuite headless rendering — EGL backend (training-side default,
    # cf. examples/embodiment/run_embodiment.sh). brain1 has no OSMesa.
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    env.setdefault("LIBERO_TYPE", "standard")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1500,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, (
        "collect script failed\n"
        f"stdout (tail):\n{result.stdout[-3000:]}\n"
        f"stderr (tail):\n{result.stderr[-3000:]}"
    )
    assert out.exists(), "output pickle was not written"

    with out.open("rb") as f:
        data = pickle.load(f)

    assert isinstance(data, list), f"expected list, got {type(data)}"
    assert len(data) >= 1, f"expected at least 1 trajectory, got {len(data)}"

    # Trajectory schema check (rlinf.data.embodied_io_struct.Trajectory).
    from rlinf.data.embodied_io_struct import Trajectory

    traj = data[0]
    assert isinstance(traj, Trajectory), (
        f"expected Trajectory, got {type(traj)}"
    )

    # Required top-level fields for the replay buffer.
    assert traj.actions is not None, "actions field missing"
    assert traj.rewards is not None, "rewards field missing"
    assert traj.dones is not None, "dones field missing"

    # curr_obs / next_obs must contain σ-QRT contract keys.
    for k in ("z_obs", "s_p", "ref_action"):
        assert k in traj.curr_obs, f"curr_obs missing key {k}"
        assert k in traj.next_obs, f"next_obs missing key {k}"

    # Shape consistency: [T, B, ...] with B=1 for single-env collection.
    T = traj.actions.shape[0]
    assert traj.actions.dim() == 4, (
        f"actions must be [T, B, chunk_len, action_dim], got {tuple(traj.actions.shape)}"
    )
    assert traj.actions.shape[1] == 1, "B must be 1 for single-env collect"
    assert traj.rewards.shape[:2] == (T, 1)
    assert traj.curr_obs["z_obs"].shape[:2] == (T, 1)
    assert traj.next_obs["z_obs"].shape[:2] == (T, 1)
    assert traj.curr_obs["s_p"].shape[:2] == (T, 1)
    assert traj.curr_obs["ref_action"].shape[:2] == (T, 1)

    # Now confirm it actually loads through the offline buffer pipeline.
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buf = TrajectoryReplayBuffer.from_offline_dataset(
        str(out),
        capacity=max(8, len(data)),
        device="cpu",
    )
    assert len(buf) == len(data)
    batch = buf.sample(num_chunks=min(4, T))
    assert "actions" in batch
    assert "curr_obs" in batch
    assert "next_obs" in batch
    assert "z_obs" in batch["curr_obs"]
    assert "s_p" in batch["curr_obs"]
    assert "ref_action" in batch["curr_obs"]
