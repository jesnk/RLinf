# Copyright 2026 The RLinf Authors / sigma project.
# SPDX-License-Identifier: Apache-2.0
"""QAM policy worker -- baseline for sigma comparison.

Implements Q-learning with Adjoint Matching (Li & Levine, arXiv:2601.14234)
on top of the existing fsdp_sac_policy_worker scaffold.

Key delta vs. EmbodiedSACFSDPPolicy:
  * Critic update:  identical TD MSE (re-uses parent class machinery).
  * Actor update:   replaces the SAC pathwise loss
                       J_pi = E[ alpha * log_pi - Q(s, a_theta) ]
                    with the adjoint matching loss (Eq. 9, LWD = QAM Eq. 4-7):
                       L_QAM = E[ || 2 f_delta / sigma_w + sigma_w * tilde_g ||^2 ]
                    so that actor gradients NEVER flow through the denoising
                    trajectory.

This is the *direct anti-thesis* of sigma's pathwise SAC-Flow update:
  - sigma:  grad_theta J_sigma = grad_theta Q( s, denoise_chain_theta(eps) )
            backprops through the FULL flow ODE (multi-step BPTT)
  - QAM:    grad_theta L_QAM regresses f_theta against -grad_a Q
            (single-step supervised target, NO denoising backprop)

NOTE: This file is a *runnable skeleton* for sigma-phase1. Heavy
integration with model.forward(ForwardType.FLOW_VELOCITY) and the chunked
replay buffer is left as TODO blocks marked `# TODO(sigma-phase2)`.
"""
from __future__ import annotations

from typing import Optional

import torch
from omegaconf import DictConfig

from rlinf.algorithms.adjoint_matching import (
    compute_adjoint_matching_loss,
    compute_critic_action_grad,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class EmbodiedQAMFSDPPolicy(EmbodiedSACFSDPPolicy):
    """QAM baseline policy (sigma-phase1).

    Inherits everything from EmbodiedSACFSDPPolicy and overrides only the
    actor update so the critic / replay-buffer / target-update logic stays
    identical -- giving us a clean apples-to-apples comparison against
    sigma's pathwise SAC-Flow worker.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # QAM hyper-parameters (LWD Sec 4.2).
        qam_cfg = cfg.algorithm.get("qam", {})
        self.qam_lambda: float = qam_cfg.get("lambda", 1.0)
        self.qam_num_flow_steps: int = qam_cfg.get("num_flow_steps", 4)
        # Whether to weight QAM loss by chunk advantage (sigma extension).
        self.qam_use_advantage: bool = qam_cfg.get("use_advantage", False)

    # ------------------------------------------------------------------ #
    # Actor update -- the only meaningful override                       #
    # ------------------------------------------------------------------ #
    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        """QAM actor loss (Eq. 9, LWD).

        Pipeline (Algorithm 2, lines 8-11, LWD):
          1) sample a flow time w ~ U(0, 1) per batch element
          2) build a^w = (1-w) * eps + w * a   along straight-line FM path
          3) compute critic action gradient grad_a Q_phi_bar(s, a)
             with critic params frozen (target critic; no_grad)
          4) regress velocity f_theta(s, a^w, w)  ->  -sigma_w^2 / 2 * tilde_g_w
             via L_QAM
        """
        curr_obs = batch["curr_obs"]
        actions = batch["actions"]                     # [B, D]  (chunk-flattened)
        device = actions.device
        B, D = actions.shape[0], actions.numel() // actions.shape[0]
        actions_flat = actions.reshape(B, D)

        # ---- 1) flow time sampling ------------------------------------
        w = torch.rand((B, 1), device=device, dtype=actions_flat.dtype)
        # Flow-matching schedule (rectified flow / OT-CFM): sigma_w = 1 - w.
        sigma_w = (1.0 - w).clamp_min(1e-3)

        # ---- 2) noisy action a^w --------------------------------------
        eps = torch.randn_like(actions_flat)
        a_w = (1.0 - w) * eps + w * actions_flat       # straight-line FM path

        # ---- 3) target critic action gradient (no_grad on phi) --------
        with torch.no_grad():
            def _critic_fn(o, a):
                return self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=o,
                    actions=a,
                )
            # We need grad w.r.t. action only; enable grad locally.
        with torch.enable_grad():
            q_action_grad = compute_critic_action_grad(_critic_fn, curr_obs, actions_flat)

        # ---- 4) predicted residual velocity f_delta(s, a^w, w) --------
        # TODO(sigma-phase2): replace placeholder with the real flow head.
        # The worker config flag actor.model.add_q_head must be False for QAM
        # actor pass; we forward through the policy's flow expert only.
        velocity_pred = self.model(
            forward_type=ForwardType.FLOW_VELOCITY
            if hasattr(ForwardType, "FLOW_VELOCITY")
            else ForwardType.SAC,
            obs=curr_obs,
            actions=a_w,
            t=w.squeeze(-1),
        )
        if isinstance(velocity_pred, tuple):
            velocity_pred = velocity_pred[0]
        velocity_pred = velocity_pred.reshape(B, D)

        adv = batch.get("advantages", None) if self.qam_use_advantage else None
        if adv is not None:
            adv = adv.reshape(B, 1).to(velocity_pred.dtype)

        actor_loss = compute_adjoint_matching_loss(
            velocity_pred=velocity_pred,
            a_w=a_w,
            sigma_w=sigma_w,
            q_action_grad=q_action_grad,
            advantage=adv,
            lam=self.qam_lambda,
        )

        return actor_loss, {
            "actor/qam_loss": actor_loss.detach().item(),
            "actor/sigma_w_mean": sigma_w.mean().item(),
            "actor/q_action_grad_norm": q_action_grad.norm(dim=-1).mean().item(),
        }

    # ------------------------------------------------------------------ #
    # Alpha (entropy temperature) is NOT used in QAM                     #
    # ------------------------------------------------------------------ #
    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        # QAM is deterministic in the actor objective (no log_pi term);
        # we keep the entry point so the parent training loop does not
        # crash, returning zero loss with stop-grad.
        zero = torch.zeros((), device=self.device, requires_grad=True)
        return zero, {"alpha/loss": 0.0, "alpha/value": 0.0}
