"""Tests for σ-QRT Tier-1 optimization: contiguous flat storage in
TrajectoryReplayBuffer (W4+ data pipeline optimization).

Distinct from test_replay_buffer_offline.py which exercises the slow path
(no curr_obs/next_obs schema). These tests load Trajectories with the σ-QRT
nested obs schema (z_obs, s_p, ref_action under curr_obs/next_obs) and
verify the fast path is built and returns the worker contract dict.
"""

import pickle
import tempfile
import time
from pathlib import Path

import pytest
import torch


def make_sigma_qrt_trajectories(
    n_trajectories: int = 3,
    T: int = 4,
    B: int = 1,
    M: int = 16,
    d: int = 32,
    chunk_len: int = 10,
    action_dim: int = 7,
    proprio_dim: int = 8,
):
    """List of Trajectory objects with the σ-QRT schema (curr_obs/next_obs).

    Real σ-QRT rollouts (Task 8 collect_base_vla_rollouts.py) emit trajectories
    with B=1 per env worker step and obs nested dicts containing z_obs, s_p,
    ref_action. This helper mirrors that schema with small dims so the test
    runs fast on CPU.
    """
    from rlinf.data.embodied_io_struct import Trajectory

    trajs = []
    for idx in range(n_trajectories):
        curr_obs = {
            "z_obs": torch.randn(T, B, M, d, dtype=torch.bfloat16),
            "s_p": torch.randn(T, B, proprio_dim),
            "ref_action": torch.randn(T, B, chunk_len, action_dim),
        }
        next_obs = {
            "z_obs": torch.randn(T, B, M, d, dtype=torch.bfloat16),
            "s_p": torch.randn(T, B, proprio_dim),
            "ref_action": torch.randn(T, B, chunk_len, action_dim),
        }
        traj = Trajectory(
            max_episode_length=T,
            model_weights_id=f"sigma_qrt_{idx}",
            actions=torch.randn(T, B, chunk_len, action_dim),
            rewards=torch.zeros(T, B, chunk_len),
            dones=torch.zeros(T, B),
            terminations=torch.zeros(T, B),
            truncations=torch.zeros(T, B),
            prev_logprobs=torch.zeros(T, B, action_dim),
            curr_obs=curr_obs,
            next_obs=next_obs,
        )
        trajs.append(traj)
    return trajs


@pytest.fixture
def offline_pickle(tmp_path: Path) -> Path:
    """Materialize a σ-QRT-schema pickle for the buffer to load."""
    trajs = make_sigma_qrt_trajectories(n_trajectories=4, T=8, B=1, M=16, d=32)
    path = tmp_path / "sigma_qrt_offline.pkl"
    with path.open("wb") as f:
        pickle.dump(trajs, f)
    return path


def test_flat_storage_built_for_sigma_qrt_schema(offline_pickle: Path) -> None:
    """from_offline_dataset() builds `_flat_storage` when curr_obs/next_obs
    have z_obs/s_p/ref_action."""
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buf = TrajectoryReplayBuffer.from_offline_dataset(
        str(offline_pickle), capacity=8, device="cpu"
    )
    assert buf._flat_storage is not None, "flat storage should be built"
    assert buf._n_chunks > 0
    # 4 trajectories × T=8 × B=1 = 32 chunks
    assert buf._n_chunks == 32


def test_flat_storage_skipped_for_legacy_schema() -> None:
    """Trajectories without curr_obs/next_obs (legacy test fixtures) must NOT
    populate _flat_storage — slow path is then used. Backwards compat."""
    from rlinf.data.embodied_io_struct import Trajectory
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    T, B = 4, 2
    trajs = [
        Trajectory(
            max_episode_length=T,
            model_weights_id="legacy",
            actions=torch.randn(T, B, 10, 7),
            rewards=torch.zeros(T, B, 10),
            dones=torch.zeros(T, B, dtype=torch.bool),
            terminations=torch.zeros(T, B, dtype=torch.bool),
            truncations=torch.zeros(T, B, dtype=torch.bool),
            prev_logprobs=torch.randn(T, B, 7),
        )
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.pkl"
        with path.open("wb") as f:
            pickle.dump(trajs, f)
        buf = TrajectoryReplayBuffer.from_offline_dataset(
            str(path), capacity=4, device="cpu"
        )
    # No curr_obs/next_obs → slow path retained.
    assert buf._flat_storage is None


def test_sample_returns_worker_contract_keys(offline_pickle: Path) -> None:
    """Fast-path sample() must return the 9 σ-QRT worker contract keys."""
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buf = TrajectoryReplayBuffer.from_offline_dataset(
        str(offline_pickle), capacity=8, device="cpu"
    )
    batch = buf.sample(num_chunks=8)
    expected = {
        "z_obs",
        "next_z_obs",
        "s_p",
        "next_s_p",
        "action",
        "ref_action",
        "next_ref_action",
        "reward",
        "done",
    }
    assert set(batch.keys()) == expected, (
        f"missing/extra keys: got {set(batch.keys())}, want {expected}"
    )


def test_sample_tensor_shapes(offline_pickle: Path) -> None:
    """Each sampled tensor must have the expected shape."""
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buf = TrajectoryReplayBuffer.from_offline_dataset(
        str(offline_pickle), capacity=8, device="cpu"
    )
    B_sample = 16
    batch = buf.sample(num_chunks=B_sample)
    # M=16, d=32 per fixture
    assert batch["z_obs"].shape == (B_sample, 16, 32)
    assert batch["next_z_obs"].shape == (B_sample, 16, 32)
    # proprio_dim=8
    assert batch["s_p"].shape == (B_sample, 8)
    assert batch["next_s_p"].shape == (B_sample, 8)
    # chunk_len=10, action_dim=7
    assert batch["action"].shape == (B_sample, 10, 7)
    assert batch["ref_action"].shape == (B_sample, 10, 7)
    assert batch["next_ref_action"].shape == (B_sample, 10, 7)
    # reward [B, C]
    assert batch["reward"].shape == (B_sample, 10)
    # done [B]
    assert batch["done"].shape == (B_sample,)


def test_sample_is_fast(offline_pickle: Path) -> None:
    """Performance regression test: 100 sample() calls of B=128 must complete
    under 1 second on CPU (fast path uses single index_select, not Python loop).
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buf = TrajectoryReplayBuffer.from_offline_dataset(
        str(offline_pickle), capacity=8, device="cpu"
    )
    # Warmup (one-time setup costs).
    for _ in range(3):
        _ = buf.sample(num_chunks=128)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = buf.sample(num_chunks=128)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, (
        f"sample() too slow: 100 calls took {elapsed:.3f}s "
        f"(target < 1.0s)"
    )
