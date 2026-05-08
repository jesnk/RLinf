# Copyright 2026 The RLinf Authors.
# Copyright 2026 σ project (jesnk).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""σ N1 — Chunked SAC-Flow FSDP policy worker.

EmbodiedCSFFSDPPolicy:
- 기반: fsdp_sac_policy_worker.py 의 EmbodiedSACFSDPPolicy 구조 차용
- 차이:
  * use_csf=True 일 때 actor forward 가 CSF (pathwise gradient + Flow-G gate)
  * forward_critic 이 chunked TD target 사용 (chunked_q.compute_chunked_td_target)
  * actor loss 에 BC reg + KL-to-π0-ref 항 추가
  * detach 제거 → critic gradient 가 actor 까지 backprop
  * TD3-style target network EMA + LayerNorm critic 유지 (RLinf q_head.py 기본)
  * LCB ensemble (K=4) — chunked_advantage_lcb 호출

핵심 식:
    actor_loss = (α · log_pi - q_pi).mean()
                 + β_bc · ||a - a_demo||²
                 + β_kl · ||a - a_ref||²
    critic_target = Σ γ^h r_{t+h} + γ^H · next_q · (~done)

본 worker 는 fsdp_sac_policy_worker.py 의 update_one_epoch / run_training /
save_checkpoint / load_checkpoint 패턴을 그대로 reuse 한다 (subclass 의 동기화).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from rlinf.algorithms.chunked_q import (
    chunked_advantage_lcb,
    compute_chunked_td_target,
)
from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class EmbodiedCSFFSDPPolicy(EmbodiedSACFSDPPolicy):
    """σ N1 — Chunked SAC-Flow on π0.5.

    Inherits from EmbodiedSACFSDPPolicy to reuse:
        - replay buffer, demo buffer, target network, alpha tuning
        - rollout / weight sync infrastructure
        - save / load checkpoint
        - update_one_epoch dispatch
    Overrides:
        - forward_actor / forward_critic for CSF specifics
        - reference network (π0 ref) management for KL regularization
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # σ N1 — additional state
        self.ref_model = None  # π0 reference (frozen) for KL penalty
        self.csf_kl_beta = float(self.cfg.actor.model.openpi.get("csf_kl_beta", 0.0))
        self.csf_bc_beta = float(self.cfg.actor.model.openpi.get("csf_bc_beta", 1.0))
        self.csf_pessimism = float(
            self.cfg.actor.model.openpi.get("csf_pessimism", 1.0)
        )
        self.csf_num_q_heads = int(
            self.cfg.actor.model.openpi.get("csf_num_q_heads", 4)
        )
        # action chunk H — Q-chunking unbiased H-step backup uses this
        self.num_action_chunks = int(self.cfg.actor.model.get("num_action_chunks", 5))
        self.use_csf = bool(self.cfg.actor.model.openpi.get("use_csf", False))

    # =========================================================================
    # Setup
    # =========================================================================
    def init_worker(self):
        """σ N1 — extends EmbodiedSACFSDPPolicy.init_worker with reference model."""
        super().init_worker()
        if self.csf_kl_beta > 0:
            self._init_reference_model()

    def _init_reference_model(self):
        """Build a frozen copy of π0.5 for KL-to-π0-ref regularization.

        Pattern adapted from fsdp_nft_policy_worker.py:263-297.
        """
        self.logger.info(
            f"[σ CSF] csf_kl_beta={self.csf_kl_beta:.4f} > 0 → building π0 reference model"
        )
        ref_module = self.model_provider_func()
        # Load SFT weights only (not RL-trained ones) — same path as initial.
        # NB: in production we may want a separate reference checkpoint path.
        ref_module.requires_grad_(False)
        ref_module.eval()
        ref_module.to(self.device)
        self.ref_model = ref_module

    def soft_update_target_model(self, tau: Optional[float] = None):
        """σ N1 — TD3-style EMA. Reuse parent for safety; no behavioral change."""
        return super().soft_update_target_model(tau=tau)

    # =========================================================================
    # CSF forward / loss
    # =========================================================================
    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        """Chunked TD target with K-ensemble LCB target Q.

        Replaces RLinf's `target = r + γ * next_q` (non-chunked) or
        `target = r_0 + γ^H * next_q` (DSRL chunked-but-1-step) with
        Σ γ^h r_{t+h} + γ^H * next_q * (~done) (Q-chunking unbiased H-step).

        The reward signal in batch["rewards"] is per-step inside chunk
        (shape [B, H]); RLinf's replay buffer already stores this when
        `reward_type=chunk_level`.
        """
        if not self.use_csf:
            # fallback to parent SAC critic
            return super().forward_critic(batch)

        gamma = self.cfg.algorithm.gamma
        H = self.num_action_chunks
        per_step_term = bool(
            self.cfg.algorithm.get("chunked_per_step_termination", True)
        )

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]  # [B, H] chunk-level
        terminations = batch["terminations"]  # [B, H]

        # ---------------------------------------------------------------
        # 1. target Q from target_model: pathwise sample at next state
        # ---------------------------------------------------------------
        with torch.no_grad():
            # next-state action via target actor
            target_out = self.target_model(
                forward_type=ForwardType.CSF,
                obs=next_obs,
                train=True,
            )
            next_action = target_out["action"]
            next_log_pi = target_out["log_pi"].unsqueeze(-1)  # [B, 1]
            next_suffix = target_out["suffix_features"]

            # K-ensemble Q at (next_obs, next_action) — share suffix feature
            all_qf_next_target = self.target_model(
                forward_type=ForwardType.CSF_Q,
                obs=next_obs,
                actions=next_action,
                suffix_features=next_suffix,
            )  # [B, K]

            # LCB ensemble pessimism: mean - 1σ - α·log_π
            alpha_val = self.entropy_temp.alpha
            qf_next_target = chunked_advantage_lcb(
                all_qf_next_target,
                next_log_pi,
                alpha=float(alpha_val) if self.cfg.algorithm.get("backup_entropy", True) else 0.0,
                pessimism=self.csf_pessimism,
            )  # [B, 1]

            target_q_values = compute_chunked_td_target(
                rewards.to(qf_next_target.dtype),
                qf_next_target,
                gamma=gamma,
                dones=terminations.to(qf_next_target.dtype),
                per_step_termination=per_step_term,
            )

        # ---------------------------------------------------------------
        # 2. online Q at (curr_obs, actions)
        # ---------------------------------------------------------------
        all_data_q_values = self.model(
            forward_type=ForwardType.CSF_Q,
            obs=curr_obs,
            actions=actions,
        )  # [B, K]

        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )

        metrics = {
            "q_data": all_data_q_values.mean().item(),
            "q_target": target_q_values.mean().item(),
            "q_target_std": all_qf_next_target.std(dim=1).mean().item(),
            "log_pi_next": next_log_pi.mean().item(),
        }
        return critic_loss, metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        """SAC pathwise actor + chunked Q + BC + KL regularization.

        actor_loss = (α · log_pi - q_pi_lcb).mean()
                     + β_bc · ||a - a_demo||²
                     + β_kl · ||a - a_ref||²

        Note: detach 제거. csf_forward_impl 안에서 BPTT 가 흐른다.
        """
        if not self.use_csf:
            return super().forward_actor(batch)

        curr_obs = batch["curr_obs"]
        demo_actions = batch.get("actions")  # demo or behavior actions

        # ---------------------------------------------------------------
        # 1. actor forward — pathwise BPTT (no detach)
        # ---------------------------------------------------------------
        out = self.model(
            forward_type=ForwardType.CSF,
            obs=curr_obs,
            train=True,
        )
        pi = out["action"]  # [B, H, A]
        log_pi = out["log_pi"].unsqueeze(-1)  # [B, 1]
        suffix_features = out["suffix_features"]

        # ---------------------------------------------------------------
        # 2. Q(s, π(s)) — pathwise critic gradient (detach_encoder=False!)
        # ---------------------------------------------------------------
        all_qf_pi = self.model(
            forward_type=ForwardType.CSF_Q,
            obs=curr_obs,
            actions=pi,
            suffix_features=suffix_features,
            detach_encoder=False,  # σ pathwise SAC: gradient flows
        )  # [B, K]

        # LCB pessimism for actor target as well (matches critic)
        qf_pi_pessimistic = chunked_advantage_lcb(
            all_qf_pi,
            log_pi,
            alpha=0.0,  # entropy term added separately below
            pessimism=self.csf_pessimism,
        )

        actor_main = (self.entropy_temp.alpha * log_pi - qf_pi_pessimistic).mean()

        # ---------------------------------------------------------------
        # 3. BC regularization on demo / behavior actions
        # ---------------------------------------------------------------
        bc_loss = torch.tensor(0.0, device=pi.device, dtype=pi.dtype)
        if self.csf_bc_beta > 0 and demo_actions is not None:
            # demo_actions [B, H, A] aligned with pi
            tgt = demo_actions.to(pi.dtype)
            if tgt.shape == pi.shape:
                bc_loss = F.mse_loss(pi, tgt)

        # ---------------------------------------------------------------
        # 4. KL-to-π0-ref regularization
        # ---------------------------------------------------------------
        kl_loss = torch.tensor(0.0, device=pi.device, dtype=pi.dtype)
        if self.csf_kl_beta > 0 and self.ref_model is not None:
            with torch.no_grad():
                ref_out = self.ref_model(
                    forward_type=ForwardType.CSF,
                    obs=curr_obs,
                    train=False,
                )
                ref_action = ref_out["action"].detach()
            # action-space KL surrogate via L2 — replace with full Gaussian KL when std is exposed
            kl_loss = F.mse_loss(pi, ref_action.to(pi.dtype))

        actor_loss = actor_main + self.csf_bc_beta * bc_loss + self.csf_kl_beta * kl_loss
        entropy = -log_pi.mean()

        metrics = {
            "q_pi_mean": all_qf_pi.mean().item(),
            "q_pi_std": all_qf_pi.std(dim=1).mean().item(),
            "q_pi_lcb": qf_pi_pessimistic.mean().item(),
            "actor_main": actor_main.item(),
            "bc_loss": bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss,
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            **{
                f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
                for q_id in range(min(self.csf_num_q_heads, all_qf_pi.shape[-1]))
            },
        }
        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        """Same as parent: -α (log_pi + target_entropy)."""
        if not self.use_csf:
            return super().forward_alpha(batch)
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            out = self.model(
                forward_type=ForwardType.CSF,
                obs=curr_obs,
                train=True,
            )
            log_pi = out["log_pi"].unsqueeze(-1)

        alpha = self.entropy_temp.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

    # =========================================================================
    # Diagnostics — gradient norm tracking for BPTT divergence detection
    # =========================================================================
    def log_grad_norms(self, step):
        """σ Phase 3 S5 — Day 30 BPTT 게이트 검증용.

        gradient norm > 100 (5 step 연속) → 발산 fallback (FQL one-step).
        """
        # actor + gate gradients
        actor_grad = 0.0
        gate_grad = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().data.norm(2).item()
            if "csf_velocity_gate" in name:
                gate_grad += g ** 2
            elif "q_head" not in name:
                actor_grad += g ** 2
        actor_grad = actor_grad ** 0.5
        gate_grad = gate_grad ** 0.5
        return {"diag/actor_grad_norm": actor_grad, "diag/gate_grad_norm": gate_grad}


# ---------------------------------------------------------------------------
# Module-level export hook for actor __init__.py
# ---------------------------------------------------------------------------
__all__ = ["EmbodiedCSFFSDPPolicy"]
