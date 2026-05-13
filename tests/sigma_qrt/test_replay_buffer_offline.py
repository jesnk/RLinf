"""Tests for TrajectoryReplayBuffer offline-dataset load mode (Task 5)."""

import pickle
import tempfile
from pathlib import Path

import pytest
import torch


def make_dummy_trajectories(
    n_trajectories=4,
    T=8,
    B=2,
    chunk_len=10,
    action_dim=7,
):
    """List of Trajectory dataclass instances matching the buffer's existing schema.

    The buffer's canonical write entry-point is `add_trajectories(list[Trajectory])`
    where each Trajectory stores tensors of shape [T, B, ...]. So the offline
    pickle file must contain a list of Trajectory objects (not flat transition
    dicts) — this matches the existing internal schema (Task 5 constraint).
    """
    from rlinf.data.embodied_io_struct import Trajectory

    trajectories = []
    for idx in range(n_trajectories):
        traj = Trajectory(
            max_episode_length=T,
            model_weights_id=f"offline_{idx}",
            actions=torch.randn(T, B, chunk_len, action_dim),
            rewards=torch.zeros(T, B, chunk_len),
            dones=torch.zeros(T, B, dtype=torch.bool),
            terminations=torch.zeros(T, B, dtype=torch.bool),
            truncations=torch.zeros(T, B, dtype=torch.bool),
            prev_logprobs=torch.randn(T, B, action_dim),
        )
        trajectories.append(traj)
    return trajectories


def test_load_from_offline_file_smoke():
    """from_offline_dataset() loads a pickle of trajectories and supports sampling."""
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    trajs = make_dummy_trajectories(n_trajectories=4, T=8, B=2)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "trajectories.pkl"
        with f.open("wb") as fh:
            pickle.dump(trajs, fh)
        buf = TrajectoryReplayBuffer.from_offline_dataset(
            str(f),
            capacity=8,
            device="cpu",
        )
        # 4 trajectories loaded
        assert len(buf) == 4
        # Sampling works and returns batched chunks
        batch = buf.sample(num_chunks=8)
        assert "actions" in batch
        assert batch["actions"].shape[0] == 8
        # Action chunk shape preserved: [B, chunk_len, action_dim]
        assert tuple(batch["actions"].shape[1:]) == (10, 7)


def test_offline_mode_disables_writes():
    """offline-loaded buffer with offline_only=True must reject add_trajectories()."""
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    trajs = make_dummy_trajectories(n_trajectories=2, T=4, B=2)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "trajectories.pkl"
        with f.open("wb") as fh:
            pickle.dump(trajs, fh)
        buf = TrajectoryReplayBuffer.from_offline_dataset(
            str(f),
            capacity=8,
            device="cpu",
            offline_only=True,
        )
        # Try to add more — should raise
        with pytest.raises(RuntimeError, match="offline"):
            buf.add_trajectories(make_dummy_trajectories(n_trajectories=1, T=4, B=2))


def test_offline_mode_off_allows_writes():
    """If offline_only=False, writes still work (offline+online hybrid)."""
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    trajs = make_dummy_trajectories(n_trajectories=2, T=4, B=2)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "trajectories.pkl"
        with f.open("wb") as fh:
            pickle.dump(trajs, fh)
        buf = TrajectoryReplayBuffer.from_offline_dataset(
            str(f),
            capacity=8,
            device="cpu",
            offline_only=False,
        )
        initial = len(buf)
        buf.add_trajectories(make_dummy_trajectories(n_trajectories=1, T=4, B=2))
        assert len(buf) == initial + 1


def test_default_buffer_writes_unchanged():
    """Regression: a buffer constructed normally (not from_offline_dataset) must
    allow add_trajectories() without raising — existing callers unaffected."""
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buf = TrajectoryReplayBuffer(seed=0, enable_cache=True, sample_window_size=8)
    buf.add_trajectories(make_dummy_trajectories(n_trajectories=2, T=4, B=2))
    assert len(buf) == 2
    batch = buf.sample(num_chunks=4)
    assert "actions" in batch
