"""σ-QRT step-time PROFILER — find the actual GPU bottleneck.

Splits one train_step() into stages and times each so we know whether the
bottleneck is:
  - sample (buffer)
  - H2D copy
  - encoder forward (z_obs)
  - encoder forward (next_z_obs, no grad)
  - critic forward/backward
  - actor forward/backward
  - decoder/recon

Usage:
    python experiments/bench_qrt_profile.py [--n 10]
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples/embodiment"))
from run_qrt_offline import adapt_buffer_batch, _move_to_device_async  # noqa

logging.basicConfig(
    format="[prof] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prof")


def _build_cfg(batch_size: int):
    return OmegaConf.create({
        "model": {
            "token_dim": 2048,
            "encoder_layers": 4,
            "encoder_heads": 8,
            "encoder_ffn": 4096,
            "decoder_layers": 4,
            "decoder_heads": 8,
            "decoder_ffn": 4096,
            "decoder_max_len": 1024,
            "actor_hidden": 256,
            "actor_layers": 2,
            "critic_hidden": 256,
            "critic_layers": 2,
            "action_dim": 7,
            "proprio_dim": 8,
            "chunk_len": 10,
        },
        "training": {
            "lr_actor": 1e-4,
            "lr_critic": 1e-4,
            "lr_encoder": 1e-4,
            "batch_size": batch_size,
            "gamma": 0.95,
            "beta_bc": 0.3,
            "alpha_recon": 0.5,
            "tau_target": 0.005,
            "actor_update_freq": 2,
            "action_std": 0.05,
            "target_noise_std": 0.2,
            "target_noise_clip": 0.5,
            "ref_action_dropout": 0.5,
            "stop_grad_z_rl_next": True,
            "grad_clip_norm": 1.0,
        },
    })


class Timer:
    def __init__(self):
        self.records: dict[str, list[float]] = {}
        self.t = None
        self.last = None
        self.device_is_cuda = torch.cuda.is_available()

    def _sync(self):
        if self.device_is_cuda:
            torch.cuda.synchronize()

    def stamp(self, label: str):
        self._sync()
        now = time.perf_counter()
        if self.last is not None:
            self.records.setdefault(label, []).append((now - self.last) * 1000.0)
        self.last = now

    def restart(self):
        self._sync()
        self.last = time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buf",
                        default="data/sigma_qrt/libero_long/transitions_B2500.pkl")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda"
    cfg = _build_cfg(args.batch_size)

    from rlinf.data.replay_buffer import TrajectoryReplayBuffer
    log.info("loading buffer...")
    t0 = time.perf_counter()
    buf = TrajectoryReplayBuffer.from_offline_dataset(
        args.buf, capacity=8000, device=device, offline_only=True,
    )
    log.info("buffer loaded in %.1fs", time.perf_counter() - t0)

    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import (
        QRTOfflinePolicyWorker,
    )
    worker = QRTOfflinePolicyWorker(cfg, device=device)
    worker.setup()

    # Direct hands-on profiling: replicate train_step internals.
    timer = Timer()

    # Warmup
    for _ in range(3):
        batch = buf.sample(num_chunks=args.batch_size)
        adapted = adapt_buffer_batch(batch)
        adapted = _move_to_device_async(adapted, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            worker.train_step(adapted, step=0)
    torch.cuda.synchronize()

    log.info("profiling %d steps...", args.n)
    for step in range(args.n):
        timer.restart()

        # 1) sample
        batch = buf.sample(num_chunks=args.batch_size)
        timer.stamp("1_sample")

        # 2) adapt (CPU cast)
        adapted = adapt_buffer_batch(batch)
        timer.stamp("2_adapt_cast")

        # 3) H2D
        adapted = _move_to_device_async(adapted, device)
        timer.stamp("3_h2d")

        # 4) Full train_step (with bf16 autocast, matching production setting)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            worker.train_step(adapted, step=step)
        timer.stamp("4_train_step")

    log.info("===== PROFILER RESULTS (median ms per stage) =====")
    total = 0.0
    for label, vals in timer.records.items():
        m = statistics.median(vals)
        total += m
        log.info("  %-20s median=%6.1fms  p90=%6.1fms",
                 label, m, sorted(vals)[int(len(vals)*0.9)])
    log.info("  %-20s %6.1fms (sum of medians)", "TOTAL", total)


if __name__ == "__main__":
    sys.exit(main())
