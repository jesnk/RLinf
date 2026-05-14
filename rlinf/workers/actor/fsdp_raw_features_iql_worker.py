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

"""σ-QRT v7 Goal 1: bypass-encoder ablation worker (a1_raw_features).

Same IQL training pipeline as ``a1_frozen_encoder`` (Stage 2 only — no token
warmup, no encoder/decoder learning) but completely **bypasses** the RLT
encoder. Instead of ``z_rl = encoder(z_obs)``, this worker uses mean-pooled
raw VLA features:

    z_rl_raw = z_obs.mean(dim=1)                       # [B, M, d] → [B, d]

When ``d == m.token_dim`` (the existing σ-QRT setup: π0.5 Gemma hidden =
2048 = z_rl dim), this is an **identity** dim handling — no projection
layer is needed. If a future model has ``d != m.token_dim``, a
``nn.Linear(d, m.token_dim)`` projection is auto-instantiated and trained
jointly with the actor/critic (no extra optimizer state; uses
``self.opt_actor``).

Why
---
Tests whether the RLT encoder is necessary at all. If the bypass variant
matches σ-QRT joint-encoder / A1 frozen-encoder SR, then the encoder is
NOT contributing anything that mean-pool of raw VLA tokens cannot — i.e.
σ-QRT's encoder is a bottleneck rather than a representation refinement.

Design notes
------------
- Stage 1 skip: the entry script ``run_qrt_offline.py`` only calls Stage 1
  when ``variant != "qrt"`` AND ``stage1_step`` is meaningful. For this
  variant, we set ``warmup_steps`` effectively to 0 via the worker's
  ``stage1_step`` raising — but the cleaner path is for the entry script
  to explicitly skip Stage 1 when ``variant == "a1_raw_features"``. We
  implement BOTH guards (entry-script branch + worker raise) so a
  copy-paste launcher that forgets the skip still produces a clean error.

- Encoder + decoder retained: even though we don't USE the encoder, we
  keep the module so the ckpt schema (encoder/decoder/actor/critic) is
  identical to other variants. Decoder is unused. Encoder is unused;
  its state at save time is the random init (frozen).

- Recon loss: zeroed out. ``loss_recon`` stays in the metrics dict for
  log-scraper compatibility but is always 0 — there's nothing to train
  the decoder with that makes sense (we're bypassing the encoder).

- Optimizer: ``self.opt_enc`` is a no-op (same pattern as
  ``RLTOnlineSimWorker.freeze_encoder()``). The actor + critic + V
  optimizers are unchanged.

- If ``cfg.model.token_dim != z_obs.shape[-1]`` (the VLA hidden dim from
  ``OpenPi0ForRLActionPrediction.extract_embeddings()``), we attach a
  ``nn.Linear(d_vla, m.token_dim)`` projection. This is NOT identical to
  the encoder — it has only ~ d_vla * m.token_dim params (4.2M for
  2048 → 2048) vs the encoder's ~150M. The point is to demonstrate that
  whatever representation work is needed can be done by a tiny linear,
  if encoder is unnecessary.
"""

import torch
import torch.nn as nn

from rlinf.algorithms.losses import (
    qrt_iql_actor_loss,
    qrt_iql_q_loss,
    qrt_iql_v_loss,
)
from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import QRTOfflinePolicyWorker


class RawFeaturesIQLWorker(QRTOfflinePolicyWorker):
    """IQL worker that bypasses the RLT encoder.

    Stage 1 is a no-op. Stage 2 uses mean-pooled raw z_obs (+ optional
    projection) in place of ``encoder(z_obs)``. All other IQL semantics
    (expectile V, AWR actor, target Q soft updates) match
    :class:`QRTOfflinePolicyWorker.train_step_iql`.
    """

    def setup(self):
        super().setup()
        # Force IQL — TD3+BC bypass would need a different code path and the
        # Goal 1 ablation only needs the IQL one.
        if not self.use_iql:
            raise RuntimeError(
                "RawFeaturesIQLWorker requires cfg.training.use_iql=True (or "
                "--use_iql on the entry script). Run with the IQL config "
                "(libero_long_qrt_iql_openpi_pi05.yaml)."
            )

        # Inspect the buffer-time VLA hidden dim by reading cfg.model.token_dim
        # — for σ-QRT's current setup this is 2048 (π0.5 Gemma). If a future
        # config uses a different VLA, the projection is added on first Stage 2
        # step (when we see the real z_obs.shape[-1]). For now, init identity
        # and trust the σ-QRT config.
        self._raw_proj: nn.Linear | None = None
        self._raw_proj_target_dim = int(self.cfg.model.token_dim)

        # Encoder + decoder are unused: zero out their optimizer so any
        # accidental call site doesn't accumulate state. Mirror the pattern
        # used by RLTOnlineSimWorker.freeze_encoder().
        for p in self.encoder.parameters():
            p.requires_grad = False
            p.grad = None
        for p in self.decoder.parameters():
            p.requires_grad = False
            p.grad = None
        dummy = torch.nn.Parameter(torch.zeros(1))
        self.opt_enc = torch.optim.Adam([dummy], lr=0.0)
        self._encoder_bypassed = True

    # ------------------------------------------------------------------ #
    # Encoder bypass: mean-pool raw VLA features, optional linear projection.
    # ------------------------------------------------------------------ #
    def _maybe_init_projection(self, z_obs: torch.Tensor) -> None:
        d_vla = int(z_obs.shape[-1])
        if d_vla == self._raw_proj_target_dim:
            return  # identity path — no projection
        if self._raw_proj is None:
            self._raw_proj = nn.Linear(d_vla, self._raw_proj_target_dim).to(self.device)
            # Add to actor's optimizer so it gets trained alongside actor/V/Q.
            # Using opt_actor (rather than a new opt) keeps the per-step
            # optimizer count identical to A1 — easier to reason about
            # compute parity.
            self.opt_actor.add_param_group(
                {"params": list(self._raw_proj.parameters())}
            )

    def _z_rl_from_raw(self, z_obs: torch.Tensor) -> torch.Tensor:
        """Replace ``encoder(z_obs)`` with mean-pool over the token dim.

        Shape: ``[B, M, d_vla] → [B, m.token_dim]``. If d_vla matches
        m.token_dim (the default σ-QRT setup), this is just mean-pool. If
        not, the lazily-initialized projection brings d_vla up/down to
        m.token_dim before the actor/critic see it.
        """
        self._maybe_init_projection(z_obs)
        pooled = z_obs.mean(dim=1)  # [B, d_vla]
        if self._raw_proj is None:
            return pooled
        return self._raw_proj(pooled)

    # ------------------------------------------------------------------ #
    # Stage 1 / 2 dispatch
    # ------------------------------------------------------------------ #
    def stage1_step(self, batch: dict) -> dict:
        """No-op. Bypass-encoder variant has no token warmup.

        The run_qrt_offline.py entry script should skip Stage 1 entirely for
        this variant (see the loop in main()). This raise is defense in
        depth so an out-of-tree launcher that doesn't skip Stage 1 gets a
        clean error rather than silently training a frozen encoder against
        itself.
        """
        raise RuntimeError(
            "RawFeaturesIQLWorker.stage1_step called — Stage 1 is skipped for "
            "the a1_raw_features variant. Update the entry script's Stage 1 "
            "branch to skip when variant=='a1_raw_features'."
        )

    def freeze_encoder(self):
        """No-op: encoder is already bypassed (and frozen) since setup()."""

    def stage2_step(self, batch: dict, step: int) -> dict:
        """IQL Stage 2 with raw features replacing encoder(z_obs).

        Mirrors :meth:`QRTOfflinePolicyWorker.train_step_iql` but every
        ``self.encoder(z_obs)`` call is replaced with
        :meth:`_z_rl_from_raw`. Encoder grad is skipped (encoder frozen),
        recon loss is zero. All other IQL update semantics
        (expectile V, advantage-weighted actor, soft target-critic) are
        unchanged.
        """
        assert self.use_iql and self.v_net is not None, (
            "RawFeaturesIQLWorker requires IQL — internal error"
        )
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)

        z_obs = batch["z_obs"]
        next_z_obs = batch["next_z_obs"]
        s_p = batch["s_p"]
        next_s_p = batch["next_s_p"]
        action = batch["action"]
        ref_action = batch["ref_action"]
        reward = batch["reward"]
        done = batch["done"]

        # ----- (1) V update: expectile regression on raw features -----
        z_rl_v = self._z_rl_from_raw(z_obs)
        v_pred = self.v_net(z_rl_v, s_p)
        with torch.no_grad():
            q_for_v = self._q(z_rl_v.detach(), s_p, action, target=True)
            q_for_v_min = q_for_v.min(dim=-1).values
        loss_v = qrt_iql_v_loss(q_for_v_min, v_pred, tau=self.iql_tau)
        # No recon — encoder bypassed.
        self.opt_v.zero_grad(set_to_none=True)
        if self._raw_proj is not None:
            # Projection params live in opt_actor; zero its grads now so the
            # V backward doesn't accumulate into stale grads from the previous
            # actor step.
            self.opt_actor.zero_grad(set_to_none=True)
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(
            self.v_net.parameters(), max_norm=self.grad_clip_norm
        )
        self.opt_v.step()
        if self._raw_proj is not None:
            # Step projection from V loss too — encoder-replacement learns
            # from all 3 losses (V, Q, actor) like the σ-QRT encoder would.
            self.opt_actor.step()

        # ----- (2) Q update: TD target uses V(s') -----
        z_rl_q = self._z_rl_from_raw(z_obs)
        if self.stop_grad_z_rl_next:
            with torch.no_grad():
                z_rl_next = self._z_rl_from_raw(next_z_obs)
        else:
            z_rl_next = self._z_rl_from_raw(next_z_obs)
        with torch.no_grad():
            v_next = self.v_net(z_rl_next, next_s_p)
        q12 = self._q(z_rl_q, s_p, action, target=False)
        q1, q2 = q12.unbind(dim=-1)
        loss_q = qrt_iql_q_loss(
            q1, q2, reward, v_next, done, self.gamma, self.chunk_len
        )

        self.opt_critic.zero_grad(set_to_none=True)
        if self._raw_proj is not None:
            self.opt_actor.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=self.grad_clip_norm
        )
        self.opt_critic.step()
        if self._raw_proj is not None:
            self.opt_actor.step()

        # ----- (3) Actor update: advantage-weighted regression -----
        z_rl_a = self._z_rl_from_raw(z_obs)
        mu_actor = self.actor(z_rl_a, s_p, ref_action, training=False)
        with torch.no_grad():
            q_for_adv = (
                self._q(z_rl_a.detach(), s_p, action, target=True).min(dim=-1).values
            )
            v_for_adv = self.v_net(z_rl_a.detach(), s_p)
        loss_a = qrt_iql_actor_loss(
            actions=action,
            mu_actor=mu_actor,
            q_target=q_for_adv,
            v_pred=v_for_adv,
            beta_iql=self.iql_beta,
            weight_clip=self.iql_weight_clip,
        )

        self.opt_actor.zero_grad(set_to_none=True)
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.grad_clip_norm
        )
        self.opt_actor.step()

        # ----- (4) Soft target-critic update only (no target V in IQL) -----
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        with torch.no_grad():
            advantage = (q_for_adv - v_for_adv).detach()
            weight = torch.exp(self.iql_beta * advantage).clamp(
                max=self.iql_weight_clip
            )
        return {
            "loss_critic": float(loss_q.detach().item()),
            "loss_actor": float(loss_a.detach().item()),
            "loss_recon": 0.0,  # encoder bypassed — no recon train signal
            "loss_v": float(loss_v.detach().item()),
            "q_mean": float(q1.detach().mean().item()),
            "v_mean": float(v_pred.detach().mean().item()),
            "advantage_mean": float(advantage.mean().item()),
            "weight_mean": float(weight.mean().item()),
        }
