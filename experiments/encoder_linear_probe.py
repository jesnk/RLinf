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

"""σ-QRT v7 Goal 1: Linear probe on z_rl for action-relevant + outcome info.

Given a frozen encoder ckpt (any ckpt_stepN.pt from chain v6 — they all share
the same RLT encoder state) and the offline buffer used for training, this
script:

  1. Instantiates RLTokenEncoder from the saved yaml config.
  2. Loads the encoder weights from the ckpt, freezes them, sets .eval().
  3. Walks the buffer's flat storage to compute z_rl for every transition.
  4. Splits the transitions into 80/20 train/holdout (last 20% held out).
  5. Action probe: sklearn LinearRegression on (z_rl_train, action_flat_train),
     reports per-dim and overall R² on the holdout.
  6. Success probe: sklearn LogisticRegression on (z_rl_train, traj_success),
     reports ROC-AUC on holdout. The trajectory-level success label is derived
     from the buffer's reward signal — LIBERO is sparse, so any positive
     reward in the chunk window indicates the episode reached the goal at
     that point. We aggregate per transition by ``reward.sum(dim=-1) > 0``
     which captures "this transition saw a success signal".

Verification target (after running, post v6 finish):
    R²_action_overall  >= 0.5   → encoder carries action info
    ROC-AUC_success    >= 0.6   → encoder carries outcome info
Lower than that → encoder bottleneck signal (Goal 1 verdict).

Usage
-----
    python -X utf8 experiments/encoder_linear_probe.py \\
        --ckpt_path runs/iql_v6_.../qrt_seed1/ckpt_step2000.pt \\
        --buffer_path data/sigma_qrt/libero_long/transitions_B2500.pkl \\
        --config examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml \\
        --variant qrt \\
        --output_json runs/probe_qrt_seed1.json

CPU-only by default — no GPU touching during chain v6. Set
``--device cuda:N`` after v6 finishes to speed up encoder forward.
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

# Reuse helpers from run_qrt_offline (variant dispatch table).
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "examples" / "embodiment")
)
from _sigma_qrt_helpers import apply_overrides  # noqa: E402

logging.basicConfig(
    format="[probe] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_encoder(cfg, device: str):
    """Instantiate RLTokenEncoder matching the σ-QRT worker setup() shape."""
    from rlinf.models.embodiment.modules.rl_token import RLTokenEncoder

    m = cfg.model
    enc = RLTokenEncoder(
        input_dim=m.token_dim,
        hidden_dim=m.token_dim,
        num_layers=m.encoder_layers,
        num_heads=m.encoder_heads,
        ffn_dim=m.encoder_ffn,
    ).to(device)
    return enc


def _load_encoder_from_ckpt(encoder, ckpt_path: str, device: str) -> dict:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "encoder" not in payload:
        raise KeyError(
            f"ckpt {ckpt_path} has no 'encoder' state dict; keys={list(payload.keys())}"
        )
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return payload


def _load_buffer_flat(buffer_path: str, capacity: int):
    """Return the buffer's flat tensor storage dict.

    Uses ``TrajectoryReplayBuffer.from_offline_dataset`` which (for σ-QRT
    nested-obs trajectories) builds ``buf._flat_storage`` — the per-chunk
    contiguous tensors. We access the storage directly because we need
    deterministic order (train/holdout split is positional, not random).
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buf = TrajectoryReplayBuffer.from_offline_dataset(
        buffer_path,
        capacity=capacity,
        device="cpu",  # force CPU storage; we encode on whatever --device says
        offline_only=True,
    )
    flat = getattr(buf, "_flat_storage", None)
    if flat is None:
        raise RuntimeError(
            "buffer has no _flat_storage — schema mismatch. Probe requires σ-QRT "
            "nested-obs trajectories (curr_obs/next_obs with z_obs, s_p, ref_action)."
        )
    return flat


def _encode_in_batches(
    encoder, z_obs: torch.Tensor, device: str, batch_size: int = 64
) -> np.ndarray:
    """Forward all transitions through the frozen encoder, return z_rl as numpy.

    z_obs is on CPU (bfloat16 typical). We chunk and cast per-batch to fp32 on
    `device`, encode, and bring back to CPU. Output shape: (N, hidden=token_dim).
    """
    n = z_obs.shape[0]
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = z_obs[start:end].to(device, dtype=torch.float32, non_blocking=True)
            z_rl = encoder(chunk)
            outs.append(z_rl.detach().to("cpu", dtype=torch.float32).numpy())
            del chunk, z_rl
    return np.concatenate(outs, axis=0)


def _derive_success_label(reward: torch.Tensor) -> np.ndarray:
    """Per-transition binary success label from chunked reward.

    LIBERO rewards are sparse: 0 on every step, 1 (or task-defined) when the
    success condition becomes true. ``reward.sum(dim=-1) > 0`` therefore says
    "this transition's chunk window contains a success step". This is a clean
    binary signal for the logistic probe — matches the spec's
    "use reward.sum() > threshold ... or terminal done+last reward" guidance,
    picking the simpler positive-reward path.

    Returns 1D int array, length N.
    """
    if reward.dim() == 1:
        return (reward > 0).to(torch.int64).cpu().numpy()
    return (reward.sum(dim=-1) > 0).to(torch.int64).cpu().numpy()


def _split_train_holdout(n: int, holdout_frac: float) -> tuple[slice, slice]:
    n_holdout = int(round(n * holdout_frac))
    n_train = n - n_holdout
    return slice(0, n_train), slice(n_train, n)


def _action_probe(
    z_rl_train: np.ndarray,
    z_rl_hold: np.ndarray,
    action_train: np.ndarray,
    action_hold: np.ndarray,
) -> dict:
    """Linear regression on flattened action chunk.

    action_train/hold shape: (N, chunk_len * action_dim). The probe fits a
    single LinearRegression on the flattened target (sklearn handles
    multi-output natively). R² is reported overall AND per output dim
    (so we can see if certain action axes are easier to recover).
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(z_rl_train, action_train)
    pred_hold = model.predict(z_rl_hold)

    # Overall R² — sklearn's default uniform_average across outputs.
    ss_res = ((action_hold - pred_hold) ** 2).sum()
    ss_tot = ((action_hold - action_hold.mean(axis=0)) ** 2).sum()
    r2_overall = float(1.0 - ss_res / max(ss_tot, 1e-12))

    # Per-dim R² on the flattened (chunk_len * action_dim) target. We fold
    # back to per-action-axis by averaging across chunk steps (the spec asks
    # for 7 per-action-dim values for action_dim=7).
    per_flat = []
    for d in range(action_hold.shape[1]):
        y_true = action_hold[:, d]
        y_pred = pred_hold[:, d]
        sse = float(((y_true - y_pred) ** 2).sum())
        sst = float(((y_true - y_true.mean()) ** 2).sum())
        per_flat.append(1.0 - sse / max(sst, 1e-12))
    return {
        "r2_overall": r2_overall,
        "r2_per_flat_dim": per_flat,
    }


def _success_probe(
    z_rl_train: np.ndarray,
    z_rl_hold: np.ndarray,
    succ_train: np.ndarray,
    succ_hold: np.ndarray,
) -> dict:
    """Logistic regression + ROC-AUC.

    If either split has only one class present, return AUC=NaN and skip the
    sklearn fit (sklearn raises). This happens with very sparse buffers; the
    spec's verification target assumes both classes exist in both splits.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if len(np.unique(succ_train)) < 2 or len(np.unique(succ_hold)) < 2:
        log.warning(
            "success probe: single-class split (train=%s, hold=%s) — AUC undefined",
            np.unique(succ_train).tolist(),
            np.unique(succ_hold).tolist(),
        )
        return {
            "auc": float("nan"),
            "n_positive_train": int(succ_train.sum()),
            "n_positive_hold": int(succ_hold.sum()),
        }
    # max_iter bumped above default 100 — z_rl is high-dim (2048) and the
    # default solver can fail to converge before max_iter at scale.
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(z_rl_train, succ_train)
    score_hold = model.predict_proba(z_rl_hold)[:, 1]
    auc = float(roc_auc_score(succ_hold, score_hold))
    return {
        "auc": auc,
        "n_positive_train": int(succ_train.sum()),
        "n_positive_hold": int(succ_hold.sum()),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="σ-QRT v7 encoder linear probe.")
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="ckpt_stepN.pt (frozen encoder source)",
    )
    p.add_argument("--buffer_path", type=str, required=True, help="transitions pickle")
    p.add_argument(
        "--config", type=str, required=True, help="yaml config for encoder dims"
    )
    p.add_argument(
        "--variant",
        type=str,
        default="qrt",
        choices=["qrt", "a1_frozen_encoder"],
        help="encoder source variant (qrt=joint, a1=frozen-after-warmup)",
    )
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu / cuda / cuda:N — defaults to cpu so chain v6 is undisturbed.",
    )
    p.add_argument(
        "--encode_batch_size",
        type=int,
        default=64,
        help="Batch size for encoder forward pass over the buffer.",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional OmegaConf overrides; matches run_qrt_offline.py.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.override)

    device = args.device
    log.info(
        "probe variant=%s ckpt=%s buffer=%s device=%s holdout_frac=%.2f",
        args.variant,
        args.ckpt_path,
        args.buffer_path,
        device,
        args.holdout_frac,
    )

    # 1+2. Encoder.
    encoder = _build_encoder(cfg, device)
    _load_encoder_from_ckpt(encoder, args.ckpt_path, device)
    log.info(
        "encoder loaded + frozen (token_dim=%d, layers=%d)",
        int(cfg.model.token_dim),
        int(cfg.model.encoder_layers),
    )

    # 3. Buffer.
    flat = _load_buffer_flat(args.buffer_path, capacity=int(cfg.data.capacity))
    z_obs = flat["z_obs"]  # (N, M, d)
    action = flat["action"]  # (N, chunk_len, action_dim)
    reward = flat["reward"]  # (N, chunk_len)
    n = int(z_obs.shape[0])
    log.info(
        "buffer loaded: N=%d, z_obs=%s, action=%s, reward=%s",
        n,
        tuple(z_obs.shape),
        tuple(action.shape),
        tuple(reward.shape),
    )

    # 4. Split (positional, deterministic).
    train_slc, hold_slc = _split_train_holdout(n, args.holdout_frac)

    # 5. Encode.
    log.info("encoding %d transitions (batch_size=%d) …", n, args.encode_batch_size)
    z_rl = _encode_in_batches(encoder, z_obs, device, batch_size=args.encode_batch_size)
    log.info("z_rl shape=%s dtype=%s", z_rl.shape, z_rl.dtype)

    # Flatten action chunks. action is (N, C, A); we predict the whole chunk
    # at once as a (C*A,)-dim target.
    action_flat = action.reshape(n, -1).cpu().numpy()
    succ = _derive_success_label(reward)
    log.info(
        "success label stats: %d positive / %d total (%.2f%%)",
        int(succ.sum()),
        n,
        100.0 * succ.sum() / max(n, 1),
    )

    # 6+7. Probes.
    log.info("running action probe (linear regression) …")
    action_res = _action_probe(
        z_rl[train_slc],
        z_rl[hold_slc],
        action_flat[train_slc],
        action_flat[hold_slc],
    )
    log.info("action R² overall=%.4f", action_res["r2_overall"])

    # Per-action-dim R² is the spec's deliverable (7 floats for action_dim=7).
    # The flat target is (chunk_len * action_dim); we average per axis across
    # chunk steps.
    chunk_len = int(cfg.model.chunk_len)
    action_dim = int(cfg.model.action_dim)
    per_flat = np.asarray(action_res["r2_per_flat_dim"]).reshape(chunk_len, action_dim)
    r2_per_dim = per_flat.mean(axis=0).tolist()  # 7 floats
    log.info("action R² per dim: %s", [round(v, 4) for v in r2_per_dim])

    log.info("running success probe (logistic regression) …")
    succ_res = _success_probe(
        z_rl[train_slc],
        z_rl[hold_slc],
        succ[train_slc],
        succ[hold_slc],
    )
    log.info("success AUC=%.4f", succ_res["auc"])

    # 7. Output JSON.
    out = {
        "action_r2_overall": float(action_res["r2_overall"]),
        "action_r2_per_dim": [float(v) for v in r2_per_dim],
        "action_r2_per_flat_dim": [float(v) for v in action_res["r2_per_flat_dim"]],
        "success_auc": float(succ_res["auc"]),
        "success_n_positive_train": succ_res["n_positive_train"],
        "success_n_positive_hold": succ_res["n_positive_hold"],
        "n_train": int(train_slc.stop - train_slc.start),
        "n_holdout": int(hold_slc.stop - hold_slc.start),
        "z_rl_dim": int(z_rl.shape[1]),
        "variant": args.variant,
        "ckpt_path": args.ckpt_path,
        "buffer_path": args.buffer_path,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
