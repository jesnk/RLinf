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

"""σ-QRT G2 buffer diversification — merge per-lane rollout pkls.

Reads N input ``.pkl`` files (each a ``list[Trajectory]`` produced by
``collect_base_vla_rollouts.py``), concatenates the trajectory lists, and
writes a single merged pickle. Optionally caps the total number of chunks
to roughly match the original ``transitions_B2500.pkl`` size for direct
SR comparison.

Also reports an action-distribution std comparison vs the original buffer
so the diversification effect is visible at a glance.

Usage
-----
    python experiments/merge_diversified_buffers.py \\
        --inputs data/sigma_qrt/libero_long/diversified_*/t005.pkl \\
                 data/sigma_qrt/libero_long/diversified_*/t010.pkl \\
                 data/sigma_qrt/libero_long/diversified_*/t020.pkl \\
                 data/sigma_qrt/libero_long/diversified_*/rand.pkl \\
        --reference data/sigma_qrt/libero_long/transitions_B2500.pkl \\
        --output data/sigma_qrt/libero_long/transitions_diversified_B2500plus.pkl \\
        --max_chunks 4600
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    format="[merge_div_buffers] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_trajs(path: str | Path):
    p = Path(path)
    log.info("loading %s", p)
    t0 = time.time()
    with p.open("rb") as f:
        trajs = pickle.load(f)
    n_chunks = sum(int(t.actions.shape[0]) for t in trajs)
    log.info(
        "loaded %d trajectories (%d chunks) in %.1fs from %s",
        len(trajs),
        n_chunks,
        time.time() - t0,
        p,
    )
    return trajs


def _action_stats(trajs, max_trajs: int = 200) -> torch.Tensor:
    """Return per-dim std of executed actions across (up to) max_trajs.

    actions shape per traj: [T, 1, C, A] → flatten [T*C, A].
    """
    if not trajs:
        return torch.zeros(0)
    sample = trajs[:max_trajs]
    flat = torch.cat(
        [t.actions.reshape(-1, t.actions.shape[-1]) for t in sample], dim=0
    )
    return flat.std(dim=0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge σ-QRT diversified rollout buffers into one pickle."
    )
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Input pkl paths to merge (in order).",
    )
    p.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference buffer for action-std comparison (not concatenated).",
    )
    p.add_argument(
        "--include_reference",
        action="store_true",
        help="Also concatenate reference buffer's trajectories into the output.",
    )
    p.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Cap total chunks in output (drops trailing trajectories).",
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    merged: list = []
    per_lane_stats: dict[str, dict] = {}

    if args.include_reference and args.reference:
        ref_trajs = _load_trajs(args.reference)
        merged.extend(ref_trajs)
        std_ref_inc = _action_stats(ref_trajs)
        per_lane_stats["__reference_included"] = {
            "trajs": len(ref_trajs),
            "chunks": sum(int(t.actions.shape[0]) for t in ref_trajs),
            "action_std_per_dim": [round(float(x), 4) for x in std_ref_inc.tolist()],
        }

    for inp in args.inputs:
        trajs = _load_trajs(inp)
        merged.extend(trajs)
        std_lane = _action_stats(trajs)
        per_lane_stats[Path(inp).stem] = {
            "trajs": len(trajs),
            "chunks": sum(int(t.actions.shape[0]) for t in trajs),
            "action_std_per_dim": [round(float(x), 4) for x in std_lane.tolist()],
        }

    total_trajs = len(merged)
    total_chunks = sum(int(t.actions.shape[0]) for t in merged)
    log.info("pre-cap merged: %d trajs %d chunks", total_trajs, total_chunks)

    if args.max_chunks is not None:
        # Drop trailing trajs until total chunks <= cap. Shuffle deterministically
        # by seed so the cap doesn't bias toward early lanes.
        rng = torch.Generator().manual_seed(int(args.seed))
        idx = torch.randperm(len(merged), generator=rng).tolist()
        merged = [merged[i] for i in idx]
        kept: list = []
        running = 0
        for t in merged:
            n = int(t.actions.shape[0])
            if running + n > int(args.max_chunks):
                break
            kept.append(t)
            running += n
        merged = kept
        log.info(
            "post-cap (max_chunks=%d, shuffled): %d trajs %d chunks",
            args.max_chunks,
            len(merged),
            running,
        )

    # Final action-std summary on the merged set vs reference.
    std_merged = _action_stats(merged)
    summary = {
        "output": str(out),
        "num_trajectories": len(merged),
        "num_chunks": sum(int(t.actions.shape[0]) for t in merged),
        "merged_action_std_per_dim": [round(float(x), 4) for x in std_merged.tolist()],
        "per_lane": per_lane_stats,
    }
    if args.reference and not args.include_reference:
        ref_trajs = _load_trajs(args.reference)
        std_ref = _action_stats(ref_trajs)
        summary["reference"] = str(args.reference)
        summary["reference_num_trajectories"] = len(ref_trajs)
        summary["reference_num_chunks"] = sum(
            int(t.actions.shape[0]) for t in ref_trajs
        )
        summary["reference_action_std_per_dim"] = [
            round(float(x), 4) for x in std_ref.tolist()
        ]
        # Per-dim ratio: merged_std / ref_std (1.0 = same, >1.0 = wider).
        ratio = (std_merged / std_ref.clamp_min(1e-8)).tolist()
        summary["std_ratio_per_dim"] = [round(float(x), 3) for x in ratio]

    log.info("writing %d trajs to %s", len(merged), out)
    t0 = time.time()
    with out.open("wb") as f:
        pickle.dump(merged, f)
    log.info("wrote %s in %.1fs", out, time.time() - t0)

    summary_path = out.with_suffix(".merge_summary.json")
    import json

    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("summary → %s", summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
