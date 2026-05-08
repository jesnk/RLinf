# Copyright 2026 The RLinf Authors / sigma project.
# SPDX-License-Identifier: Apache-2.0
"""ALOE policy worker -- baseline for sigma comparison.

Implements ALOE (arXiv:2602.12691, AgiBot/HKUST/Fudan, 2026-02) on top of
the existing fsdp_sac_policy_worker scaffold. Mirrors the
EmbodiedQAMFSDPPolicy pattern from sigma-phase1 commit 6efe2a5 so that
sigma vs {QAM, LWD, ALOE} share an identical critic / replay-buffer /
target-update path -- only the actor objective differs.

================ sigma vs ALOE: head-to-head ================

  - sigma:  grad_theta J_sigma = grad_theta Q( s, denoise_chain_theta(eps) )
            backprops through the FULL flow ODE  ==>  multi-step BPTT.
  - ALOE:   grad_theta L_actor = grad_theta [ w * || eps - a - f_theta ||^2 ]
            with w = sg( exp(clip(A/beta)) )
            single-step score-matching surrogate ==>  NO denoising BPTT.

The two workers SHARE the same critic structure (K-ensemble pessimistic LCB
+ chunked TD target via rlinf.algorithms.chunked_q.compute_chunked_td_target)
so the experimental delta is *exclusively* the actor loss type.

================ Key delta vs. EmbodiedSACFSDPPolicy ================

  * Critic update:  re-uses parent K-head Q machinery; TD target is
                    rebuilt via `compute_advantage_weighted_target` so we
                    obtain `(q_target, advantage)` in one call.
  * Actor update:   replaces the SAC pathwise loss
                       J_pi = E[ alpha * log_pi - Q(s, a_theta) ]
                    with the ALOE advantage-weighted CFM loss
                       L_actor = E[ w * || eps - a - f_theta(a_tilde, s, l) ||^2 ]
                    where the critic gradient enters ONLY through the
                    detached advantage weight w. NO pathwise BPTT.

This file is a *runnable skeleton* for sigma-phase1. Heavy integration
with model.forward(ForwardType.FLOW_VELOCITY), V-head value baselines, and
the chunked replay buffer is left as TODO blocks marked
`# TODO(sigma-phase2)`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig

from rlinf.algorithms.advantage_weighted_cfm import (
    compute_advantage_weight,
    compute_advantage_weighted_cfm_loss,
    compute_advantage_weighted_target,
    sample_aloe_flow_inputs,
)
from rlinf.algorithms.chunked_q import compute_chunked_td_target
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class EmbodiedALOEFSDPPolicy(EmbodiedSACFSDPPolicy):
    """ALOE baseline policy (sigma-phase1).

    Inherits everything from EmbodiedSACFSDPPolicy and overrides only:
      - forward_actor:  advantage-weighted CFM (no pathwise grad through Q)
      - forward_critic: chunked TD with K-ensemble pessimistic LCB
      - forward_alpha:  no-op (ALOE has no entropy temperature)

    so the replay buffer / target update / weight sync logic stays
    identical -- giving us a clean apples-to-apples comparison against
    sigma's pathwise SAC-Flow worker.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # ALOE hyper-parameters (paper Sec. 4 + Appendix C2).
        aloe_cfg = cfg.algorithm.get("aloe", {})
        self.advantage_beta: float = aloe_cfg.get(
            "beta",
            cfg.algorithm.get("advantage_beta", 0.5),
        )
        self.advantage_eps_clip: float = aloe_cfg.get("eps_clip", 5.0)
        self.advantage_w_max: Optional[float] = aloe_cfg.get("w_max", None)
        # K-ensemble size for pessimistic LCB. 'min' aggregator on K Q-heads.
        self.aloe_K: int = aloe_cfg.get("K_ensemble", 4)
        # Chunk horizon H (must match num_action_chunks)
        self.aloe_H: int = aloe_cfg.get(
            "chunk_H",
            cfg.actor.model.get("num_action_chunks", 5),
        )
        # n-step chunking for TD target (=1 means standard chunked TD).
        self.aloe_n_steps: int = aloe_cfg.get("n_steps", 1)
        # Number of flow time samples per actor minibatch (paper uses 1 by default).
        self.aloe_num_flow_samples: int = aloe_cfg.get("num_flow_samples", 1)

    # ------------------------------------------------------------------ #
    # Critic update -- chunked TD with K-ensemble pessimistic LCB        #
    # ------------------------------------------------------------------ #
    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        """ALOE critic update.

        Identical mathematical form to LWD's chunked Q (Eq. 15 / 19) but
        without DIVL distributional projection -- ALOE uses a scalar
        K-ensemble Q with pessimistic min aggregation.

        TD target (Q-chunking compatible):
            y_Q = sum_{h=0..H-1} gamma^h r_{t+h}
                + gamma^H * Q_pess(s_{t+H}, a_pi(s_{t+H})) * (~done)

        Q_pess is the *target* network's K-ensemble min over heads.
        """
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]   # expected shape [B, H] (per-step rewards in chunk)
        dones = batch.get("dones", None)

        # ---- next-state pessimistic LCB Q (no_grad / target net) -----
        with torch.no_grad():
            # TODO(sigma-phase2): sample chunked next-action from policy at s_{t+H};
            # for skeleton we delegate to the parent's existing helper.
            next_pi, _, _ = self.target_model(
                forward_type=ForwardType.SAC, obs=next_obs
            )
            all_next_q = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=next_obs,
                actions=next_pi,
            )  # [B, K]
            next_q_pess, _ = torch.min(all_next_q, dim=1, keepdim=True)  # LCB

        # ---- chunked TD target (Eq. Q-chunking; matches sigma N1) ----
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)  # tolerate scalar reward
        H_eff = rewards.shape[-1]
        if dones is None:
            dones = torch.zeros_like(rewards, dtype=torch.bool)
        elif dones.ndim == 1:
            dones = dones.unsqueeze(-1).expand_as(rewards)

        q_target = compute_chunked_td_target(
            rewards=rewards,
            next_q=next_q_pess,
            gamma=self.cfg.algorithm.gamma,
            dones=dones.bool() if dones.dtype != torch.bool else dones,
        )

        # ---- predicted Q for ALL K heads ----------------------------
        all_q_pred = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
        )  # [B, K]
        # MSE on each head (paper uses independent K losses summed).
        critic_loss = torch.nn.functional.mse_loss(
            all_q_pred, q_target.expand_as(all_q_pred).to(all_q_pred.dtype)
        )

        with torch.no_grad():
            q_pess_curr, _ = torch.min(all_q_pred, dim=1, keepdim=True)

        return critic_loss, {
            "q_data": all_q_pred.mean().item(),
            "q_target_mean": q_target.mean().item(),
            "q_pess_curr": q_pess_curr.mean().item(),
        }

    # ------------------------------------------------------------------ #
    # Actor update -- ALOE advantage-weighted CFM                        #
    # ------------------------------------------------------------------ #
    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        """ALOE actor loss (Eq. ALOE-actor).

        Pipeline (paper Sec. 4):
          1) Compute advantage A^pi = Q_pess(s, a) - sg(V^pi(s)) using
             the FROZEN target critic (no grad on phi or psi).
          2) Sample eta ~ U[0,1], eps ~ N(0,I); build
                a_tilde = eta * a + (1-eta) * eps
          3) Run the policy in flow-velocity mode to get
                f_theta(a_tilde, s, l)
          4) Loss: w(A) * || eps - a - f_theta ||^2
             where w = exp(clip(A/beta)).detach() -- HARD stop-gradient.
        """
        curr_obs = batch["curr_obs"]
        actions = batch["actions"]                       # [B, D] chunk-flattened
        rewards = batch.get("rewards", None)
        dones = batch.get("dones", None)
        v_baseline = batch.get("v_baseline", None)       # [B, 1] V^pi(s, l), if precomputed

        device = actions.device
        B = actions.shape[0]
        D = actions.numel() // B
        actions_flat = actions.reshape(B, D)

        # ---- 1) advantage A^pi (no_grad on critic) -------------------
        with torch.no_grad():
            all_q = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions_flat,
            )  # [B, K]
            q_pess, _ = torch.min(all_q, dim=1, keepdim=True)  # LCB

            # Value baseline V^pi(s, l): if not provided in batch we
            # approximate with E_pi[Q_pess] via the on-policy action.
            # TODO(sigma-phase2): plug in the dedicated V-head when added.
            if v_baseline is None:
                pi_a, _, _ = self.target_model(
                    forward_type=ForwardType.SAC, obs=curr_obs
                )
                all_v_q = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=curr_obs,
                    actions=pi_a,
                )
                v_baseline, _ = torch.min(all_v_q, dim=1, keepdim=True)
            else:
                if v_baseline.ndim == 1:
                    v_baseline = v_baseline.unsqueeze(-1)

            advantage = q_pess - v_baseline   # both already no_grad

        # ---- 2) sample noise + a_tilde (Eq. ALOE-noisy) --------------
        # Optionally average over `num_flow_samples` independent eta draws
        # to reduce variance (paper uses 1; we expose as knob).
        loss_terms = []
        for _ in range(max(self.aloe_num_flow_samples, 1)):
            eps, eta, a_tilde = sample_aloe_flow_inputs(actions_flat)

            # ---- 3) predicted residual velocity f_theta(a_tilde, s, l) ----
            # NB: `actor.model.openpi.detach_critic_input: True` in config
            # ensures the encoder feature does NOT leak into critic grads.
            velocity_pred = self.model(
                forward_type=(
                    ForwardType.FLOW_VELOCITY
                    if hasattr(ForwardType, "FLOW_VELOCITY")
                    else ForwardType.SAC
                ),
                obs=curr_obs,
                actions=a_tilde,
                t=eta.squeeze(-1),
            )
            if isinstance(velocity_pred, tuple):
                velocity_pred = velocity_pred[0]
            velocity_pred = velocity_pred.reshape(B, D)

            # ---- 4) advantage-weighted CFM loss (Eq. ALOE-actor) ----
            l = compute_advantage_weighted_cfm_loss(
                epsilon=eps,
                action=actions_flat,
                predicted_velocity=velocity_pred,
                advantage=advantage,           # already detached above
                beta=self.advantage_beta,
                eps_clip=self.advantage_eps_clip,
                w_max=self.advantage_w_max,
            )
            loss_terms.append(l)

        actor_loss = torch.stack(loss_terms).mean()

        # ---- diagnostics --------------------------------------------
        with torch.no_grad():
            w_diag = compute_advantage_weight(
                advantage,
                beta=self.advantage_beta,
                eps_clip=self.advantage_eps_clip,
                w_max=self.advantage_w_max,
                detach=True,
            )
            metrics = {
                "actor/aloe_cfm_loss": actor_loss.detach().item(),
                "actor/advantage_mean": advantage.mean().item(),
                "actor/advantage_std": advantage.std().item(),
                "actor/aloe_weight_mean": w_diag.mean().item(),
                "actor/aloe_weight_max": w_diag.max().item(),
                "actor/q_pess_curr": q_pess.mean().item(),
                "actor/v_baseline": v_baseline.mean().item(),
            }

        # Parent loop expects (loss, entropy, metrics); ALOE has no entropy
        # term so we return a zero placeholder for compatibility.
        zero_entropy = torch.zeros((), device=device)
        return actor_loss, zero_entropy, metrics

    # ------------------------------------------------------------------ #
    # Alpha (entropy temperature) is NOT used in ALOE                    #
    # ------------------------------------------------------------------ #
    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        # ALOE actor objective has no log_pi term (the flow policy is a
        # deterministic ODE evaluated with a clipped exp-advantage weight),
        # so entropy auto-tuning is disabled. Keep the entry point alive
        # so the parent training loop does not crash.
        zero = torch.zeros((), device=self.device, requires_grad=True)
        return zero, {"alpha/loss": 0.0, "alpha/value": 0.0}

    # ------------------------------------------------------------------ #
    # Convenience: build (q_target, advantage) in a single call          #
    # ------------------------------------------------------------------ #
    def _build_advantage_pair(
        self,
        rewards: torch.Tensor,
        next_q_pess: torch.Tensor,
        v_baseline: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Thin wrapper around `compute_advantage_weighted_target` so a
        downstream sigma-phase2 implementation can swap in its own value
        head without re-implementing the chunked TD math."""
        return compute_advantage_weighted_target(
            rewards=rewards,
            next_q_pess=next_q_pess,
            v_baseline=v_baseline,
            gamma=self.cfg.algorithm.gamma,
            H=self.aloe_H,
            dones=dones,
        )
