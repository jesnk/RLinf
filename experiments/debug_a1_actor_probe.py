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

"""σ-QRT W4 gate A1 failure — Task X diagnostic.

Focused probe of A1 seed1 ckpt to identify whether the 0% SR failure is:
  (a) actor output in same range/scale as ref_action (BC working) — implies
      eval pipeline bug
  (b) actor output wildly different (BC failed → actor collapsed) — implies
      training broke
  (c) actor output ~OK but env rejects it (rare)

Per-dim mean comparison + env.step single-step probe with both ref and actor
actions. Mirrors layout of debug_eval_actor_v2.py but trimmed to deliverables.

Run on GPU 3 (CUDA_VISIBLE_DEVICES=3 already set by caller).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "examples" / "embodiment")
)
from _sigma_qrt_helpers import bootstrap_gl_env  # noqa: E402

bootstrap_gl_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from examples.embodiment.collect_base_vla_rollouts import (  # noqa: E402
    _env_obs_for_vla,
    _proprio_from_obs,
    _to_python,
)
from examples.embodiment.eval_libero_sr import (  # noqa: E402
    _load_vla,
    _make_env,
    _maybe_load_worker,
)

CONFIG = "examples/embodiment/config/libero_long_qrt_openpi_pi05.yaml"
CKPT = "runs/w4_gate_parallel_20260513_233708/a1_seed1/ckpt.pt"
MAX_EPISODE_STEPS = 600


def _summary(name: str, t: torch.Tensor) -> None:
    f = t.float()
    print(
        f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={f.min().item():.4f} max={f.max().item():.4f} "
        f"mean={f.mean().item():.4f} std={f.std().item():.4f}"
    )


def _per_dim_mean(name: str, chunk: torch.Tensor) -> None:
    """chunk shape [C, A] — print per-action-dim mean over the chunk."""
    per_dim = chunk.float().mean(dim=0)  # [A]
    formatted = " ".join(f"{v:+.4f}" for v in per_dim.tolist())
    print(f"  {name} per-dim mean over chunk[{chunk.shape[0]}]: [{formatted}]")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe] device={device}", flush=True)
    print(
        f"[probe] CUDA_VISIBLE_DEVICES set; torch.cuda.device_count()={torch.cuda.device_count()}",
        flush=True,
    )

    cfg = OmegaConf.load(CONFIG)
    torch.manual_seed(0)
    np.random.seed(0)

    vla = _load_vla(cfg.model, device=device)
    worker = _maybe_load_worker(cfg, CKPT, device=device)
    assert worker is not None, "Worker must load from ckpt"

    chunk_len = int(cfg.model.chunk_len)
    action_dim = int(cfg.model.action_dim)
    print(f"[probe] chunk_len={chunk_len} action_dim={action_dim}", flush=True)

    # ----------------------------------------------------------------------
    # Step 1-7: build env, reset, single-step VLA + actor inference
    # ----------------------------------------------------------------------
    env = _make_env(cfg.env, MAX_EPISODE_STEPS, seed=0)
    obs, _ = env.reset()
    env_obs = _env_obs_for_vla(obs)

    print("\n[probe] === STEP 1: VLA reference actions ===", flush=True)
    with torch.no_grad():
        actions_chunk, _ = vla.predict_action_batch(env_obs, mode="eval")
        if actions_chunk.dim() == 2:
            actions_chunk = actions_chunk.view(1, chunk_len, action_dim)
        ref_chunk = actions_chunk[0].detach().float()  # [C, A]
    _summary("ref_actions", ref_chunk)
    _per_dim_mean("ref_actions", ref_chunk)

    print("\n[probe] === STEP 2: z_obs (vla.extract_embeddings) ===", flush=True)
    with torch.no_grad():
        z_obs = vla.extract_embeddings(env_obs).float().to(device)
    _summary("z_obs", z_obs)

    print("\n[probe] === STEP 3: z_rl (worker.encoder(z_obs)) ===", flush=True)
    with torch.no_grad():
        z_rl = worker.encoder(z_obs)
    _summary("z_rl", z_rl)

    print("\n[probe] === STEP 4: actor output ===", flush=True)
    with torch.no_grad():
        s_p = _proprio_from_obs(obs).unsqueeze(0).to(device)
        _summary("s_p", s_p)
        ref_in = ref_chunk.unsqueeze(0).to(device)
        actor_out = worker.actor(z_rl, s_p, ref_in, training=False)[0].cpu()  # [C, A]
    _summary("actor_out", actor_out)
    _per_dim_mean("actor_out", actor_out)

    print("\n[probe] === STEP 5: comparison actor_out vs ref ===", flush=True)
    diff = (actor_out - ref_chunk).abs()
    print(f"  |actor - ref| mean: {diff.mean().item():.4f}")
    print(f"  |actor - ref| max:  {diff.max().item():.4f}")
    print(
        "  |actor - ref| per-dim mean: "
        + " ".join(f"{v:.4f}" for v in diff.mean(dim=0).tolist())
    )

    print("\n[probe] === STEP 6: first 3 chunk steps side by side ===", flush=True)
    for c in range(min(3, chunk_len)):
        ref_row = " ".join(f"{v:+.4f}" for v in ref_chunk[c].tolist())
        act_row = " ".join(f"{v:+.4f}" for v in actor_out[c].tolist())
        diff_row = " ".join(f"{v:+.4f}" for v in (actor_out[c] - ref_chunk[c]).tolist())
        print(f"  step {c}:")
        print(f"    ref:   [{ref_row}]")
        print(f"    actor: [{act_row}]")
        print(f"    diff:  [{diff_row}]")

    del env

    # ----------------------------------------------------------------------
    # Step 8: env.step with ref_actions[0] from a fresh reset
    # ----------------------------------------------------------------------
    print("\n[probe] === STEP 7: env.step(ref_action) from fresh reset ===", flush=True)
    env = _make_env(cfg.env, MAX_EPISODE_STEPS, seed=0)
    obs, _ = env.reset()
    action_np = ref_chunk[0].numpy().reshape(1, action_dim)
    print(f"  action sent: [{', '.join(f'{v:+.4f}' for v in action_np[0].tolist())}]")
    obs2, r, terms, truncs, info = env.step(action_np)
    print(
        f"  result: r={float(_to_python(r, idx=0)):.4f} "
        f"term={bool(_to_python(terms, idx=0))} "
        f"trunc={bool(_to_python(truncs, idx=0))}"
    )
    del env

    # ----------------------------------------------------------------------
    # Step 9: env.step with actor_out[0] from a fresh reset
    # ----------------------------------------------------------------------
    print("\n[probe] === STEP 8: env.step(actor_out) from fresh reset ===", flush=True)
    env = _make_env(cfg.env, MAX_EPISODE_STEPS, seed=0)
    obs, _ = env.reset()
    action_np = actor_out[0].numpy().reshape(1, action_dim)
    print(f"  action sent: [{', '.join(f'{v:+.4f}' for v in action_np[0].tolist())}]")
    obs3, r, terms, truncs, info = env.step(action_np)
    print(
        f"  result: r={float(_to_python(r, idx=0)):.4f} "
        f"term={bool(_to_python(terms, idx=0))} "
        f"trunc={bool(_to_python(truncs, idx=0))}"
    )
    del env

    # ----------------------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------------------
    print("\n[probe] === VERDICT (heuristic) ===", flush=True)
    in_range = (actor_out.abs() <= 1.0).all().item()
    near_ref = diff.mean().item() < 0.05
    if near_ref:
        print(
            "  VERDICT: actor_out ≈ ref_action (BC working). Eval pipeline likely OK."
        )
        print(
            "           If A1 still 0% SR, look at obs handling / chunk indexing in eval."
        )
    elif in_range:
        print("  VERDICT: actor_out in [-1,1] but differs from ref. Actor IS refining.")
        print(
            "           If 0% SR, refinement is in wrong direction (Q gradient broke)."
        )
    else:
        print("  VERDICT: actor_out OUT OF [-1,1] range. BC reg too weak.")
        print("           Actor collapsed under Q gradient. Need stronger β_bc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
