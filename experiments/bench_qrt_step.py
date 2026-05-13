"""σ-QRT step-time benchmark (Tier-1+Tier-2 validation).

Loads the paper-faithful 4-layer encoder/decoder + actor + critic, the
offline buffer at `transitions_B2500.pkl`, and times 50 train_steps after
10 warmup steps on GPU 0. Reports median / p90 / p99 step time.

Target after Tier-1+Tier-2 optimization: median < 300 ms (vs ~11.4 s
baseline observed in the original W4 chain).

Usage:
    cd ~/data/jskang/sigma-qrt/RLinf
    source ~/data/jskang/sigma/.venv/bin/activate
    export PYTHONPATH=$(pwd)
    export CUDA_VISIBLE_DEVICES=0
    python experiments/bench_qrt_step.py [--buf PATH] [--steps N] [--warmup N]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Repo-local helpers (CLI override parsing) — match run_qrt_offline.py path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples/embodiment"))
from run_qrt_offline import (  # noqa: E402
    adapt_buffer_batch,
    _move_to_device_async,
    _NullCtx,
)

logging.basicConfig(
    format="[bench] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench")


def _build_cfg(batch_size: int):
    """Paper-faithful (libero_long_qrt_openpi_pi05.yaml) hyperparams."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buf",
        type=str,
        default="data/sigma_qrt/libero_long/transitions_B2500.pkl",
        help="Path to offline buffer pickle.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override auto device selection (cuda/cpu/cuda:N).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="logs/bench_qrt_step.json",
        help="Output JSON with timing stats.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Wrap worker.train_step in torch.autocast(bfloat16).",
    )
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s buf=%s steps=%d warmup=%d batch=%d",
             device, args.buf, args.steps, args.warmup, args.batch_size)

    cfg = _build_cfg(args.batch_size)

    # Buffer load.
    log.info("loading buffer (this may take a minute for 34 GB pickle)...")
    t0 = time.perf_counter()
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer
    buf = TrajectoryReplayBuffer.from_offline_dataset(
        args.buf, capacity=8000, device=device, offline_only=True,
    )
    log.info("buffer loaded in %.1fs: %d trajectories, %d chunks, flat_storage=%s",
             time.perf_counter() - t0, len(buf), buf._n_chunks,
             buf._flat_storage is not None)

    # Worker setup.
    from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import (
        QRTOfflinePolicyWorker,
    )
    worker = QRTOfflinePolicyWorker(cfg, device=device)
    worker.setup()
    log.info("worker setup complete (encoder=%d-layer, decoder=%d-layer)",
             cfg.model.encoder_layers, cfg.model.decoder_layers)

    # One sample to lock in shapes.
    sample0 = buf.sample(num_chunks=args.batch_size)
    log.info("sample0 keys=%s", sorted(sample0.keys()))
    for k, v in sample0.items():
        log.info("  %s: shape=%s dtype=%s", k, tuple(v.shape), v.dtype)

    autocast_ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if args.bf16 and device.startswith("cuda")
        else (lambda: _NullCtx())
    )
    log.info("bf16 autocast: %s", args.bf16)

    # Warmup.
    log.info("warmup %d steps...", args.warmup)
    for step in range(args.warmup):
        batch = buf.sample(num_chunks=args.batch_size)
        adapted = adapt_buffer_batch(batch)
        adapted = _move_to_device_async(adapted, worker.device)
        with autocast_ctx_factory():
            m = worker.train_step(adapted, step=step)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    log.info("warmup done. q_mean=%.4f loss_critic=%.4f",
             m.get("q_mean", float("nan")),
             m.get("loss_critic", float("nan")))

    # Benchmark.
    times_ms = []
    log.info("benchmarking %d steps...", args.steps)
    for step in range(args.steps):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = buf.sample(num_chunks=args.batch_size)
        adapted = adapt_buffer_batch(batch)
        adapted = _move_to_device_async(adapted, worker.device)
        with autocast_ctx_factory():
            _ = worker.train_step(adapted, step=args.warmup + step)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    median = statistics.median(times_ms)
    p90 = sorted(times_ms)[int(len(times_ms) * 0.9)]
    p99 = sorted(times_ms)[min(int(len(times_ms) * 0.99), len(times_ms) - 1)]
    mean = sum(times_ms) / len(times_ms)
    minv = min(times_ms)
    maxv = max(times_ms)

    log.info("===== RESULTS =====")
    log.info("steps=%d batch=%d device=%s", args.steps, args.batch_size, device)
    log.info("step time ms: median=%.1f p90=%.1f p99=%.1f mean=%.1f min=%.1f max=%.1f",
             median, p90, p99, mean, minv, maxv)
    target_300 = median < 300.0
    log.info("vs 300ms target: %s (median=%.1fms)",
             "PASS" if target_300 else "FAIL", median)
    log.info("vs 11400ms baseline: %.1fx speedup", 11400.0 / median)

    # Persist JSON.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "device": device,
        "batch_size": args.batch_size,
        "n_steps": args.steps,
        "median_ms": median,
        "p90_ms": p90,
        "p99_ms": p99,
        "mean_ms": mean,
        "min_ms": minv,
        "max_ms": maxv,
        "all_times_ms": times_ms,
        "below_300ms": target_300,
        "speedup_vs_baseline": 11400.0 / median,
    }, indent=2))
    log.info("saved %s", out_path)

    return 0 if target_300 else 1


if __name__ == "__main__":
    sys.exit(main())
