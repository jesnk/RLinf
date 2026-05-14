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

"""σ-QRT IQL chain v5 — collapse probe (step2000 vs step6000 vs step10000).

Goal: diagnose why IQL lane iql_tau0p8_beta3_seed1 went from SR=0.16 at step
6000 to SR=0.0 at step 10000 (with intermediate step 8000 at SR=0.08).

Roll the lane forward in time on a single fixed LIBERO observation and
report:
    - actor_out range / NaN / Inf / out-of-[-1,1] count
    - per-dim stats and side-by-side compare with the π0.5 reference chunk
    - relative L2 deviation from ref over the chunk
    - Q1, Q2, V predictions on the ckpt's own action and the ref action
    - 5-step env rollout reward when executing the actor chunk vs ref chunk
    - actor drift between ckpts (||a_step10000 - a_step6000||, etc.)

Modeled on experiments/debug_a1_actor_probe.py but extended to multi-ckpt
and IQL-specific (Q + V).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap EGL BEFORE robosuite import.
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
)

# Use the same config the lane was trained/eval'd with.
CONFIG = "examples/embodiment/config/libero_long_qrt_iql_openpi_pi05.yaml"
MAX_EPISODE_STEPS = 600
ROLLOUT_STEPS = 5


def _summary(name: str, t: torch.Tensor, indent: str = "  ") -> dict:
    f = t.detach().float().cpu()
    d = {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "min": float(f.min().item()),
        "max": float(f.max().item()),
        "mean": float(f.mean().item()),
        "std": float(f.std().item()),
        "nan_count": int(torch.isnan(f).sum().item()),
        "inf_count": int(torch.isinf(f).sum().item()),
    }
    print(
        f"{indent}{name}: shape={d['shape']} dtype={d['dtype']} "
        f"min={d['min']:+.4f} max={d['max']:+.4f} "
        f"mean={d['mean']:+.4f} std={d['std']:.4f} "
        f"nan={d['nan_count']} inf={d['inf_count']}"
    )
    return d


def _per_dim_mean_str(chunk: torch.Tensor) -> str:
    """chunk shape [C, A] — return per-action-dim mean over the chunk."""
    per_dim = chunk.detach().float().cpu().mean(dim=0).tolist()
    return " ".join(f"{v:+.4f}" for v in per_dim)


def _load_iql_worker(cfg, ckpt_path: str, device: str):
    """IQL-aware worker loader.

    Differs from eval_libero_sr._maybe_load_worker by also loading v_net
    when the ckpt has it (which all chain v5 IQL ckpts do).
    """
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
    if "v_net" in payload and worker.v_net is not None:
        worker.v_net.load_state_dict(payload["v_net"])
    # Also use target critic for Q (matches how IQL uses Q at eval: same body).
    worker.encoder.eval()
    worker.actor.eval()
    worker.critic.eval()
    if worker.v_net is not None:
        worker.v_net.eval()
    for p in worker.encoder.parameters():
        p.requires_grad = False
    for p in worker.actor.parameters():
        p.requires_grad = False
    for p in worker.critic.parameters():
        p.requires_grad = False
    if worker.v_net is not None:
        for p in worker.v_net.parameters():
            p.requires_grad = False
    has_v = "v_net" in payload
    print(f"  loaded ckpt={ckpt_path}  has_v_net={has_v}", flush=True)
    return worker


def _q_predict(worker, z_rl, s_p, action):
    """Return (q1, q2) using the online critic head."""
    state_feat = torch.cat([z_rl, s_p], dim=-1)
    action_feat = action.flatten(start_dim=1)
    q12 = worker.critic(state_feat, action_feat)  # [B, 2]
    q1, q2 = q12.unbind(dim=-1)
    return q1, q2


def _v_predict(worker, z_rl, s_p):
    """Return V_phi(s)."""
    if worker.v_net is None:
        return None
    return worker.v_net(z_rl, s_p)


def _rollout_chunk(env, action_chunk, action_dim, max_steps):
    """Execute action_chunk in env (already reset), return list of rewards."""
    rewards = []
    dones = []
    truncs = []
    for c in range(min(max_steps, action_chunk.shape[0])):
        step_action = action_chunk[c].detach().cpu().numpy().reshape(1, action_dim)
        # Clip to [-1, 1] to keep env from rejecting (mimics what eval does
        # via the env's own clamp; here we want unmodified probe so we don't
        # clamp explicitly).
        obs, r, terms, tr, info = env.step(step_action)
        rewards.append(float(_to_python(r, idx=0)))
        dones.append(bool(_to_python(terms, idx=0)))
        truncs.append(bool(_to_python(tr, idx=0)))
        if dones[-1] or truncs[-1]:
            break
    return rewards, dones, truncs


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--lane", type=str, required=True)
    p.add_argument(
        "--steps",
        type=str,
        default="2000,6000,10000",
        help="Comma list of training step labels. 10000 → ckpt.pt; "
        "others → ckpt_step{n}.pt.",
    )
    p.add_argument(
        "--rollout_steps",
        type=int,
        default=ROLLOUT_STEPS,
        help="Per-ckpt env-step probe horizon.",
    )
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[probe] device={device}  cuda.count={torch.cuda.device_count()}",
        flush=True,
    )
    lane_dir = Path(args.run_dir) / args.lane
    assert lane_dir.is_dir(), f"missing lane dir: {lane_dir}"

    step_labels = [int(s) for s in args.steps.split(",")]
    print(f"[probe] lane_dir={lane_dir}", flush=True)
    print(f"[probe] step labels={step_labels}", flush=True)

    cfg = OmegaConf.load(CONFIG)
    torch.manual_seed(0)
    np.random.seed(0)

    # --------------------------------------------------------------------
    # Build VLA + env, take one obs, compute reference and z_obs.
    # --------------------------------------------------------------------
    vla = _load_vla(cfg.model, device=device)
    chunk_len = int(cfg.model.chunk_len)
    action_dim = int(cfg.model.action_dim)

    env = _make_env(cfg.env, MAX_EPISODE_STEPS, seed=0)
    obs0, _ = env.reset()
    env_obs0 = _env_obs_for_vla(obs0)
    with torch.no_grad():
        ref_chunk_t, _ = vla.predict_action_batch(env_obs0, mode="eval")
        if ref_chunk_t.dim() == 2:
            ref_chunk_t = ref_chunk_t.view(1, chunk_len, action_dim)
        ref_chunk = ref_chunk_t[0].detach().float()  # [C, A] on device
        z_obs = vla.extract_embeddings(env_obs0).float().to(device)  # [1, M, d]
        s_p0 = _proprio_from_obs(obs0).unsqueeze(0).to(device)  # [1, dp]

    print("\n[probe] === reference (B0) snapshot ===", flush=True)
    _summary("ref_chunk", ref_chunk)
    print(f"  ref per-dim mean: [{_per_dim_mean_str(ref_chunk)}]")
    _summary("z_obs", z_obs)
    _summary("s_p", s_p0)
    ref_norm = float(ref_chunk.float().norm().item())
    print(f"  ||ref||_2 = {ref_norm:.4f}", flush=True)

    # Drop env; we'll re-create one per ckpt below for the rollout test.
    del env

    # --------------------------------------------------------------------
    # Per-ckpt: load worker, forward actor + critic + V, report.
    # --------------------------------------------------------------------
    results: dict[int, dict] = {}
    for step_n in step_labels:
        print(f"\n\n[probe] ============ CKPT step={step_n} ============", flush=True)
        if step_n == 10000:
            ckpt_path = str(lane_dir / "ckpt.pt")
        else:
            ckpt_path = str(lane_dir / f"ckpt_step{step_n}.pt")
        assert Path(ckpt_path).is_file(), f"missing ckpt: {ckpt_path}"

        worker = _load_iql_worker(cfg, ckpt_path, device=device)
        with torch.no_grad():
            z_rl = worker.encoder(z_obs)
            ref_in = ref_chunk.unsqueeze(0).to(device)  # [1, C, A]
            actor_out = worker.actor(z_rl, s_p0, ref_in, training=False)[0]
            # Q on actor's action and on ref action.
            q1_act, q2_act = _q_predict(worker, z_rl, s_p0, actor_out.unsqueeze(0))
            q1_ref, q2_ref = _q_predict(worker, z_rl, s_p0, ref_in)
            v_pred = _v_predict(worker, z_rl, s_p0)

        ref_cpu = ref_chunk.detach().float().cpu()
        act_cpu = actor_out.detach().float().cpu()

        print(f"\n[probe step={step_n}]  ---- actor output ----", flush=True)
        s_actor = _summary("actor_out", act_cpu)
        print(f"  actor per-dim mean: [{_per_dim_mean_str(act_cpu)}]")

        # diff to ref
        diff = (act_cpu - ref_cpu).float()
        diff_abs = diff.abs()
        diff_l2 = float(diff.norm().item())
        rel_dev = diff_l2 / ref_norm if ref_norm > 1e-9 else float("nan")
        n_oor = int((act_cpu.abs() > 1.0).sum().item())
        n_total = int(act_cpu.numel())

        print(f"\n[probe step={step_n}]  ---- actor vs ref ----", flush=True)
        print(
            f"  |actor - ref|: mean={diff_abs.mean().item():.4f} max={diff_abs.max().item():.4f}"
        )
        print(f"  per-dim |Δ| mean: [{_per_dim_mean_str(diff_abs)}]")
        print(f"  ||actor - ref||_2 = {diff_l2:.4f}  rel_dev = {rel_dev:.4f}")
        print(
            f"  |actor|>1 elements: {n_oor}/{n_total} ({100.0 * n_oor / n_total:.1f}%)"
        )

        # Q + V predictions
        q1a, q2a = float(q1_act.item()), float(q2_act.item())
        q1r, q2r = float(q1_ref.item()), float(q2_ref.item())
        v_val = float(v_pred.item()) if v_pred is not None else float("nan")
        print(f"\n[probe step={step_n}]  ---- critic/V predictions ----", flush=True)
        print(
            f"  Q(s, actor_out): Q1={q1a:+.4f}  Q2={q2a:+.4f}  min={min(q1a, q2a):+.4f}"
        )
        print(
            f"  Q(s, ref):       Q1={q1r:+.4f}  Q2={q2r:+.4f}  min={min(q1r, q2r):+.4f}"
        )
        print(f"  V(s):            V ={v_val:+.4f}")
        print(f"  adv(actor) = min(Q1,Q2) - V = {min(q1a, q2a) - v_val:+.4f}")
        print(f"  adv(ref)   = min(Q1,Q2) - V = {min(q1r, q2r) - v_val:+.4f}")

        # Side-by-side first 3 chunk rows
        print(f"\n[probe step={step_n}]  ---- chunk rows (first 3) ----", flush=True)
        for c in range(min(3, chunk_len)):
            ref_row = " ".join(f"{v:+.4f}" for v in ref_cpu[c].tolist())
            act_row = " ".join(f"{v:+.4f}" for v in act_cpu[c].tolist())
            d_row = " ".join(f"{v:+.4f}" for v in (act_cpu[c] - ref_cpu[c]).tolist())
            print(f"  row {c}:")
            print(f"    ref:   [{ref_row}]")
            print(f"    actor: [{act_row}]")
            print(f"    diff:  [{d_row}]")

        # 5-step env rollout with the actor chunk.
        print(
            f"\n[probe step={step_n}]  ---- env.step rollout (actor) ----", flush=True
        )
        env = _make_env(cfg.env, MAX_EPISODE_STEPS, seed=0)
        env.reset()
        rew_act, done_act, tr_act = _rollout_chunk(
            env, act_cpu, action_dim, args.rollout_steps
        )
        print(f"  rewards = {rew_act}")
        print(f"  dones   = {done_act}")
        print(f"  truncs  = {tr_act}")
        del env

        # 5-step env rollout with the reference chunk (sanity baseline).
        print(f"\n[probe step={step_n}]  ---- env.step rollout (ref) ----", flush=True)
        env = _make_env(cfg.env, MAX_EPISODE_STEPS, seed=0)
        env.reset()
        rew_ref, done_ref, tr_ref = _rollout_chunk(
            env, ref_cpu, action_dim, args.rollout_steps
        )
        print(f"  rewards = {rew_ref}")
        print(f"  dones   = {done_ref}")
        print(f"  truncs  = {tr_ref}")
        del env

        results[step_n] = {
            "actor_summary": s_actor,
            "diff_abs_mean": float(diff_abs.mean().item()),
            "diff_abs_max": float(diff_abs.max().item()),
            "diff_l2": diff_l2,
            "rel_dev": rel_dev,
            "n_out_of_range": n_oor,
            "n_total": n_total,
            "actor_out": act_cpu.numpy(),
            "q1_act": q1a,
            "q2_act": q2a,
            "q1_ref": q1r,
            "q2_ref": q2r,
            "v": v_val,
            "rew_actor_sum": float(sum(rew_act)),
            "rew_ref_sum": float(sum(rew_ref)),
        }

        # free worker before loading next ckpt — they share device memory.
        del worker
        if device == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------
    # Pairwise actor drift (last vs mid vs early).
    # --------------------------------------------------------------------
    print("\n\n[probe] ============ PAIRWISE ACTOR DRIFT ============", flush=True)
    if len(step_labels) >= 2:
        # sort by step
        steps_sorted = sorted(step_labels)
        for i in range(len(steps_sorted) - 1):
            a, b = steps_sorted[i], steps_sorted[i + 1]
            ao = torch.from_numpy(results[a]["actor_out"]).float()
            bo = torch.from_numpy(results[b]["actor_out"]).float()
            d = float((bo - ao).norm().item())
            # also vs ref
            print(
                f"  ||a_step{b} - a_step{a}||_2 = {d:.4f}   "
                f"(rel to ||ref|| = {d / ref_norm:.4f})"
            )
        # also pairwise step6000 vs step10000 (always informative)
        if 6000 in results and 10000 in results:
            ao = torch.from_numpy(results[6000]["actor_out"]).float()
            bo = torch.from_numpy(results[10000]["actor_out"]).float()
            d = float((bo - ao).norm().item())
            print(
                f"  ||a_step10000 - a_step6000||_2 = {d:.4f}   "
                f"(rel to ||ref|| = {d / ref_norm:.4f})"
            )

    # --------------------------------------------------------------------
    # Summary table + verdict.
    # --------------------------------------------------------------------
    print("\n\n[probe] ============ SUMMARY TABLE ============", flush=True)
    print(
        f"{'step':>7} {'mean(|act|)':>12} {'max(|act|)':>12} "
        f"{'rel_dev':>10} {'oor':>6} {'nan':>4} "
        f"{'Q_act_min':>10} {'Q_ref_min':>10} {'V':>9} "
        f"{'rew_act':>9} {'rew_ref':>9}"
    )
    for step_n in sorted(step_labels):
        r = results[step_n]
        s = r["actor_summary"]
        # Per-row: |mean| / max(|min|,|max|) for range, plus deviation / Q / V.
        print(
            f"{step_n:>7} "
            f"{abs(s['mean']):>12.4f} "
            f"{max(abs(s['min']), abs(s['max'])):>12.4f} "
            f"{r['rel_dev']:>10.4f} "
            f"{r['n_out_of_range']:>6d} "
            f"{s['nan_count']:>4d} "
            f"{min(r['q1_act'], r['q2_act']):>+10.4f} "
            f"{min(r['q1_ref'], r['q2_ref']):>+10.4f} "
            f"{r['v']:>+9.4f} "
            f"{r['rew_actor_sum']:>+9.4f} "
            f"{r['rew_ref_sum']:>+9.4f}"
        )

    print("\n[probe] ============ VERDICT ============", flush=True)
    rel_devs = {s: results[s]["rel_dev"] for s in results}
    nans = {s: results[s]["actor_summary"]["nan_count"] for s in results}
    # "Range explosion" means max(|actor|) >> 1 (e.g. >1.5). The mild
    # excursion of the gripper dim past ±1.0 (≈ 1.05 in this lane) is NOT
    # a range explosion — it's BC inheriting π0.5's own gripper output
    # which already sits slightly outside [-1,1] and gets clamped by the
    # env. So we only flag explosion if max|act| > 1.5.
    max_abs = {
        s: max(
            abs(results[s]["actor_summary"]["min"]),
            abs(results[s]["actor_summary"]["max"]),
        )
        for s in results
    }
    sorted_steps = sorted(results.keys())
    rd_list = [rel_devs[s] for s in sorted_steps]
    rd_str = " → ".join(f"{v:.4f}" for v in rd_list)
    qmin = {s: min(results[s]["q1_act"], results[s]["q2_act"]) for s in sorted_steps}
    v_vals = {s: results[s]["v"] for s in sorted_steps}
    print(f"  rel_dev (||a-ref||/||ref||) trajectory: {rd_str}")
    print(
        "  max|actor| trajectory: "
        + " → ".join(f"{max_abs[s]:.4f}" for s in sorted_steps)
    )
    print(
        "  Q_min(actor) trajectory: "
        + " → ".join(f"{qmin[s]:+.4f}" for s in sorted_steps)
    )
    print(
        "  V(s) trajectory:         "
        + " → ".join(f"{v_vals[s]:+.4f}" for s in sorted_steps)
    )
    if any(n > 0 for n in nans.values()):
        print("  VERDICT: NaN in actor output — numerical collapse.")
        return 0
    if any(m > 1.5 for m in max_abs.values()):
        worst_s = max(max_abs.items(), key=lambda kv: kv[1])[0]
        print(
            f"  VERDICT: Actor RANGE EXPLOSION at step {worst_s} "
            f"(max|act|={max_abs[worst_s]:.4f})."
        )
        return 0
    drifted_away = rd_list[-1] > rd_list[0] * 1.5
    drifted_toward = rd_list[-1] < rd_list[0] * 0.7
    if drifted_away:
        print(
            "  VERDICT: Mode-shift collapse — actor drifted FURTHER from ref. "
            "Per-dim |Δ| grew despite training metrics looking healthy. "
            "Critic was rewarding a divergent direction."
        )
    elif drifted_toward:
        print(
            "  VERDICT: Actor regressed TOWARD ref over training — final actor "
            "is essentially the BC reference (refinement vanished). "
            "Possible causes: advantage signal near zero (check adv_mean), "
            "weight_mean ≈ 1.0 → AWR degenerates to plain BC; or actor capacity "
            "exhausted recovering ref instead of refining it. "
            "If SR(step6000) > SR(B0)/SR(step10000), this is *overtraining "
            "past the useful refinement window* — actor lost a small but "
            "task-relevant deviation. Early-stopping at the peak SR ckpt is "
            "the immediate remedy."
        )
    else:
        print(
            "  VERDICT: No range or NaN collapse, rel_dev stable. "
            "Final-ckpt SR=0 likely from a *small* but task-critical shift "
            "across the rollout horizon (chunk×N steps), not a gross actor "
            "failure. Early-stopping at peak SR ckpt is the immediate fix."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
