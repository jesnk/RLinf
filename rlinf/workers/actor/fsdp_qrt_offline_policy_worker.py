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

"""sigma-QRT offline RL worker (Q-aware joint encoder-actor-critic training).

Design doc: work/hbz/sigma/sigma_pivot_v2_design.md section 2.3.
Losses: rlinf/algorithms/losses.py (qrt_critic_loss, qrt_compute_target,
                                    qrt_actor_loss, rlt_recon_loss).

This is a minimal CPU-runnable worker. FSDP wrapping is deferred to W2+.

Worker batch contract (decoupled from buffer schema; an adapter at the training
loop in Task 11 maps buffer batch -> worker batch):

    {
        "z_obs":          [B, M, token_dim]   # VLA tokens at s
        "next_z_obs":     [B, M, token_dim]   # VLA tokens at s'
        "s_p":            [B, proprio_dim]
        "next_s_p":       [B, proprio_dim]
        "action":         [B, C, action_dim]  # taken action chunk
        "ref_action":     [B, C, action_dim]  # VLA reference chunk at s
        "next_ref_action":[B, C, action_dim]  # VLA reference chunk at s'
        "reward":         [B, C]              # per-step reward inside chunk
        "done":           [B]                 # terminal flag
    }
"""

import copy
from typing import Any

import torch
import torch.nn as nn

from rlinf.algorithms.losses import (
    qrt_actor_loss,
    qrt_compute_target,
    qrt_cql_penalty,
    qrt_critic_loss,
    qrt_iql_actor_loss,
    qrt_iql_q_loss,
    qrt_iql_v_loss,
    rlt_recon_loss,
)
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.rl_token import RLTokenDecoder, RLTokenEncoder


class _ValueMLP(nn.Module):
    """IQL state-value network V_φ(z_rl, s_p) — scalar, no action input.

    Architecture mirrors MultiQHead's MLP body (Linear → ReLU stack) so
    the V network has comparable capacity to the Q network. The lack of
    action input is the whole point: V never needs an OOD action.
    """

    def __init__(self, in_dim: int, hidden: int = 256, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_rl: torch.Tensor, s_p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_rl, s_p], dim=-1)).squeeze(-1)


class _ActorMLP(nn.Module):
    """RLT-faithful 2-layer (default) MLP actor, Gaussian with fixed std.

    Inputs: z_rl [B, d], s_p [B, dp], ref_action [B, C, da].
    Output: action chunk [B, C, da].

    During training, applies ref_action dropout (Bernoulli mask) and adds fixed-std
    Gaussian noise. During inference (training=False), uses the deterministic mean
    and no ref dropout.
    """

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_len: int,
        hidden: int = 256,
        num_layers: int = 2,
        action_std: float = 0.05,
        ref_action_dropout: float = 0.5,
    ):
        super().__init__()
        in_dim = z_dim + proprio_dim + chunk_len * action_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        layers += [nn.Linear(prev, chunk_len * action_dim)]
        self.net = nn.Sequential(*layers)
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.action_std = action_std
        self.ref_action_dropout = ref_action_dropout

    def forward(
        self,
        z_rl: torch.Tensor,
        s_p: torch.Tensor,
        ref_action: torch.Tensor,
        training: bool = True,
        residual: bool = False,
    ) -> torch.Tensor:
        """Standard mode (residual=False): output absolute action chunk μ_θ(s).

        Residual mode (residual=True): output a *correction* Δ_θ(s) only.
        Caller is responsible for forming a_pred = ref_action + Δ. Used by
        the σ-QRT G2 residual-actor variant — the actor's job shrinks to
        learning a small delta on top of π0.5's reference, with BC fidelity
        guaranteed by the ref anchor and Q signal driving the delta. No
        Gaussian noise is added in residual mode (training noise breaks the
        clean a_pred = ref + Δ semantics; AWR weighting + Q-max term carry
        the exploration pressure instead).
        """
        if training and self.ref_action_dropout > 0:
            mask = (
                torch.rand(ref_action.shape[0], 1, 1, device=ref_action.device)
                > self.ref_action_dropout
            ).float()
            ref = ref_action * mask
        else:
            ref = ref_action
        x = torch.cat([z_rl, s_p, ref.flatten(start_dim=1)], dim=-1)
        out = self.net(x).view(-1, self.chunk_len, self.action_dim)
        if residual:
            # Caller adds ref_action externally. No noise in residual mode.
            return out
        mu = out
        if training:
            return mu + torch.randn_like(mu) * self.action_std
        return mu


class QRTOfflinePolicyWorker:
    """sigma-QRT joint encoder-actor-critic offline training (CPU-runnable minimal).

    See module docstring for batch contract. Deviation from spec: MultiQHead in
    this repo takes (hidden_size, action_feature_dim, hidden_dims=list, num_q_heads)
    and forward(state_features, action_features). So we pass z_rl||s_p as state
    features and action.flatten() as action features.
    """

    def __init__(self, cfg: Any, device: str = "cuda"):
        self.cfg = cfg
        self.device = device

    def setup(self):
        m = self.cfg.model
        tr = self.cfg.training

        self.encoder = RLTokenEncoder(
            input_dim=m.token_dim,
            hidden_dim=m.token_dim,
            num_layers=m.encoder_layers,
            num_heads=m.encoder_heads,
            ffn_dim=m.encoder_ffn,
        ).to(self.device)
        self.decoder = RLTokenDecoder(
            input_dim=m.token_dim,
            hidden_dim=m.token_dim,
            num_layers=m.decoder_layers,
            num_heads=m.decoder_heads,
            ffn_dim=m.decoder_ffn,
            max_len=m.get("decoder_max_len", 1024),
        ).to(self.device)
        self.actor = _ActorMLP(
            z_dim=m.token_dim,
            proprio_dim=m.proprio_dim,
            action_dim=m.action_dim,
            chunk_len=m.chunk_len,
            hidden=m.actor_hidden,
            num_layers=m.actor_layers,
            action_std=tr.action_std,
            ref_action_dropout=tr.ref_action_dropout,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # MultiQHead in this repo: (hidden_size=state_feat_dim,
        # action_feature_dim, hidden_dims=list[int], num_q_heads).
        state_dim = m.token_dim + m.proprio_dim
        action_feat_dim = m.chunk_len * m.action_dim
        hidden_dims = [int(m.critic_hidden)] * int(m.critic_layers)
        self.critic = MultiQHead(
            hidden_size=state_dim,
            action_feature_dim=action_feat_dim,
            hidden_dims=hidden_dims,
            num_q_heads=2,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.opt_enc = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=tr.lr_encoder,
        )
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=tr.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=tr.lr_critic)

        # stash hparams
        self.gamma = float(tr.gamma)
        self.chunk_len = int(m.chunk_len)
        self.beta_bc = float(tr.beta_bc)
        self.alpha_recon = float(tr.alpha_recon)
        self.tau = float(tr.tau_target)
        self.actor_update_freq = int(tr.actor_update_freq)
        self.target_noise_std = float(tr.target_noise_std)
        self.target_noise_clip = float(tr.target_noise_clip)
        self.stop_grad_z_rl_next = bool(tr.stop_grad_z_rl_next)
        self.grad_clip_norm = float(tr.grad_clip_norm)

        # IQL variant config (only used when cfg.training.use_iql=True).
        self.use_iql = bool(tr.get("use_iql", False))
        self.iql_tau = float(tr.get("iql_tau", 0.7))
        self.iql_beta = float(tr.get("iql_beta", 3.0))
        self.iql_weight_clip = float(tr.get("iql_weight_clip", 100.0))

        # G2 v3 CQL conservative penalty (only used when use_iql=True).
        # When use_cql=true, the IQL Q-loss adds an α_cql · CQL penalty term
        # that pushes Q(s, OOD_a) DOWN relative to Q(s, a_data). This gives
        # the residual actor a real "in-distribution preference" gradient
        # signal, fixing the q_std flat ceiling observed in G2 v2.
        self.use_cql = bool(tr.get("use_cql", False))
        self.cql_alpha = float(tr.get("cql_alpha", 1.0))
        self.cql_num_random = int(tr.get("cql_num_random", 10))
        self.cql_action_low = float(tr.get("cql_action_low", -1.0))
        self.cql_action_high = float(tr.get("cql_action_high", 1.0))
        self.cql_noise_std = float(tr.get("cql_noise_std", 0.3))

        # G2 residual-actor variant (only used when use_iql=True). Actor
        # learns a small correction Δ on top of π0.5's reference action;
        # SR floor = B0 (ref alone) is structurally guaranteed by the
        # anchor, and the Q-max term lets the actor improve where the
        # critic is reliable. See work/hbz/sigma section "G2 residual".
        self.use_residual_actor = bool(tr.get("use_residual_actor", False))
        self.residual_alpha = float(tr.get("residual_alpha", 0.1))
        self.residual_reg = float(tr.get("residual_reg", 0.0))
        if self.use_iql:
            v_hidden = int(m.get("v_hidden", m.critic_hidden))
            v_layers = int(m.get("v_layers", m.critic_layers))
            v_in_dim = m.token_dim + m.proprio_dim
            self.v_net = _ValueMLP(
                in_dim=v_in_dim, hidden=v_hidden, num_layers=v_layers
            ).to(self.device)
            self.opt_v = torch.optim.Adam(self.v_net.parameters(), lr=tr.lr_critic)
        else:
            self.v_net = None
            self.opt_v = None

    def _q(self, z_rl, s_p, action, target: bool):
        """Compute Q(s, a) using MultiQHead. Returns [B, num_q_heads]."""
        net = self.critic_target if target else self.critic
        state_feat = torch.cat([z_rl, s_p], dim=-1)
        action_feat = action.flatten(start_dim=1)
        return net(state_feat, action_feat)

    def train_step(self, batch: dict, step: int) -> dict:
        # Dispatch to IQL path when configured. Keeps TD3+BC the default
        # (backward compat with existing β sweep Phase 1 configs).
        if self.use_iql:
            return self.train_step_iql(batch, step)
        return self._train_step_td3bc(batch, step)

    def _train_step_td3bc(self, batch: dict, step: int) -> dict:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)

        z_obs = batch["z_obs"]
        next_z_obs = batch["next_z_obs"]
        s_p = batch["s_p"]
        next_s_p = batch["next_s_p"]
        action = batch["action"]
        ref_action = batch["ref_action"]
        next_ref_action = batch["next_ref_action"]
        reward = batch["reward"]
        done = batch["done"]

        # --- current encoder forward (gradient on) ---
        z_rl = self.encoder(z_obs)
        # --- next encoder forward (stop_grad default) ---
        if self.stop_grad_z_rl_next:
            with torch.no_grad():
                z_rl_next = self.encoder(next_z_obs)
        else:
            z_rl_next = self.encoder(next_z_obs)

        # --- critic target ---
        with torch.no_grad():
            a_next = self.actor_target(
                z_rl_next, next_s_p, next_ref_action, training=True
            )
            noise = (torch.randn_like(a_next) * self.target_noise_std).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            a_next = a_next + noise
            q_next = self._q(z_rl_next, next_s_p, a_next, target=True)
            q_next_min = q_next.min(dim=-1).values
            q_target = qrt_compute_target(
                reward, q_next_min, done, self.gamma, self.chunk_len
            )

        q12 = self._q(z_rl, s_p, action, target=False)
        q1, q2 = q12.unbind(dim=-1)
        loss_q = qrt_critic_loss(q1, q2, q_target)

        # --- reconstruction aux (encoder + decoder) ---
        z_prev = z_obs[:, :-1].detach()
        z_hat = self.decoder(z_rl, z_prev)
        loss_recon = rlt_recon_loss(z_hat, z_obs.detach())

        loss_critic_total = loss_q + self.alpha_recon * loss_recon
        self.opt_critic.zero_grad(set_to_none=True)
        self.opt_enc.zero_grad(set_to_none=True)
        loss_critic_total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=self.grad_clip_norm
        )
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=self.grad_clip_norm,
        )
        self.opt_critic.step()
        self.opt_enc.step()

        metrics = {
            "loss_critic": float(loss_q.detach().item()),
            "loss_recon": float(loss_recon.detach().item()),
            "q_mean": float(q1.detach().mean().item()),
            "loss_actor": 0.0,
        }

        if step % self.actor_update_freq == 0:
            z_rl_for_actor = self.encoder(z_obs)
            a_theta = self.actor(z_rl_for_actor, s_p, ref_action, training=True)
            q_pi = (
                self._q(z_rl_for_actor, s_p, a_theta, target=False).min(dim=-1).values
            )
            loss_a = qrt_actor_loss(q_pi, a_theta, ref_action, beta=self.beta_bc)
            self.opt_actor.zero_grad(set_to_none=True)
            self.opt_enc.zero_grad(set_to_none=True)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.grad_clip_norm
            )
            self.opt_actor.step()
            self.opt_enc.step()
            metrics["loss_actor"] = float(loss_a.detach().item())

            with torch.no_grad():
                for p, pt in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
                for p, pt in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return metrics

    def train_step_iql(self, batch: dict, step: int) -> dict:
        """IQL training step. Replaces TD3+BC's Q-extrapolation-prone updates.

        Update order (one optimizer step per network per call):
          1. V update — expectile regression on Q_target(s, a_data) - V(s).
             Encoder + decoder + V all backprop. Recon aux stays on encoder/decoder
             via this backward pass.
          2. Q update — TD target uses V(s') (no OOD Q query). Encoder + critic.
             G2 v3: when ``use_cql=True``, an α_cql · CQL conservative penalty
             is added to the Q loss, pushing Q(s, OOD_a) DOWN relative to
             Q(s, a_data) and giving the residual actor a prefer-in-distribution
             gradient signal. CQL queries Q at random / noisy-data actions — this
             *is* an OOD Q query, but with explicit penalty pressure (the entire
             point), not silent bootstrap drift.
          3. Actor update — advantage-weighted regression on deterministic μ_θ.
             Encoder + actor.
          4. Soft target-critic update. No target V (per IQL paper §4.2).

        Encoder ϕ receives gradient from V, Q, AND actor losses (3-way) — the
        σ-QRT Q-aware joint encoder novelty is preserved.
        """
        assert self.use_iql and self.v_net is not None, (
            "train_step_iql called but use_iql=False — internal error"
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

        # ----- (1) V update: expectile regression -----
        # z_rl with gradient on encoder so V-loss backprops to ϕ (Q-aware novelty).
        z_rl_v = self.encoder(z_obs)
        v_pred = self.v_net(z_rl_v, s_p)
        # Q_target(s, a_data) = min over twin target Qs. Detached: V-loss is the
        # only thing being optimized here; target critic is frozen.
        with torch.no_grad():
            # Use the same z_rl for the frozen target critic to keep s representations
            # consistent. detach inside since z_rl_v is in the graph.
            q_for_v = self._q(z_rl_v.detach(), s_p, action, target=True)
            q_for_v_min = q_for_v.min(dim=-1).values
        loss_v = qrt_iql_v_loss(q_for_v_min, v_pred, tau=self.iql_tau)

        # Recon aux runs through encoder+decoder, lumped into the V backward
        # pass so the encoder/decoder optimizer step happens here.
        z_prev_v = z_obs[:, :-1].detach()
        z_hat = self.decoder(z_rl_v, z_prev_v)
        loss_recon = rlt_recon_loss(z_hat, z_obs.detach())

        loss_v_total = loss_v + self.alpha_recon * loss_recon
        self.opt_v.zero_grad(set_to_none=True)
        self.opt_enc.zero_grad(set_to_none=True)
        loss_v_total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.v_net.parameters(), max_norm=self.grad_clip_norm
        )
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=self.grad_clip_norm,
        )
        self.opt_v.step()
        self.opt_enc.step()

        # ----- (2) Q update: TD target uses V(s') -----
        z_rl_q = self.encoder(z_obs)
        if self.stop_grad_z_rl_next:
            with torch.no_grad():
                z_rl_next = self.encoder(next_z_obs)
        else:
            z_rl_next = self.encoder(next_z_obs)
        with torch.no_grad():
            v_next = self.v_net(z_rl_next, next_s_p)
        q12 = self._q(z_rl_q, s_p, action, target=False)
        q1, q2 = q12.unbind(dim=-1)
        loss_q_td = qrt_iql_q_loss(
            q1, q2, reward, v_next, done, self.gamma, self.chunk_len
        )

        # G2 v3: CQL conservative penalty on critic.
        # Penalty pushes Q(s, OOD_a) DOWN relative to Q(s, a_data), giving
        # the residual actor a "prefer in-distribution Δ" gradient signal.
        # q12 is already Q at the dataset action — re-use to avoid an extra
        # forward. state_feat reconstructed here to match critic API.
        if self.use_cql and self.cql_alpha > 0:
            state_feat_cql = torch.cat([z_rl_q, s_p], dim=-1)
            action_feat_cql = action.flatten(start_dim=1)
            loss_cql = qrt_cql_penalty(
                critic=self.critic,
                state_feat=state_feat_cql,
                action_data=action_feat_cql,
                q_data=q12,  # [B, num_q]
                num_random=self.cql_num_random,
                action_low=self.cql_action_low,
                action_high=self.cql_action_high,
                noise_std=self.cql_noise_std,
            )
            loss_q = loss_q_td + self.cql_alpha * loss_cql
        else:
            loss_cql = torch.zeros((), device=q1.device, dtype=q1.dtype)
            loss_q = loss_q_td

        self.opt_critic.zero_grad(set_to_none=True)
        self.opt_enc.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=self.grad_clip_norm
        )
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=self.grad_clip_norm,
        )
        self.opt_critic.step()
        self.opt_enc.step()

        # ----- (3) Actor update: advantage-weighted regression -----
        z_rl_a = self.encoder(z_obs)
        with torch.no_grad():
            # Advantage uses Q_target on the dataset action — no OOD query.
            q_for_adv = (
                self._q(z_rl_a.detach(), s_p, action, target=True).min(dim=-1).values
            )
            v_for_adv = self.v_net(z_rl_a.detach(), s_p)

        if self.use_residual_actor:
            # G2 residual actor:
            #   Δ      = actor(z_rl, s_p, ref_action)  [residual=True → noise off]
            #   a_pred = ref_action + Δ
            #   L_AWR  = exp(β·A).clamp(max=w_clip) * ‖Δ‖² (anchors Δ→0 with BC pressure)
            #   L_Qmax = α · -Q1(s, a_pred)              (pushes Δ to maximize Q)
            #   L_reg  = λ · ‖Δ‖²                         (optional, defaults to 0)
            # ref_action serves as the anchor because at eval time only the live
            # π0.5 reference is available (no buffer); using ref_action here keeps
            # train/eval semantics aligned (action == ref_action for offline π0.5
            # data anyway, verified at impl time).
            delta = self.actor(z_rl_a, s_p, ref_action, training=False, residual=True)
            a_pred = ref_action + delta
            # Δ_squared per-sample (matches qrt_iql_actor_loss sum-norm convention).
            delta_sq = (delta**2).sum(dim=(-1, -2))  # [B]
            advantage = q_for_adv - v_for_adv  # [B], detached above
            weight = torch.exp(self.iql_beta * advantage).clamp(
                max=self.iql_weight_clip
            )
            loss_awr = (weight * delta_sq).mean()

            # Q-max term: pushes Δ in the gradient direction of Q1. Use first
            # Q head (q_id=0) — twin Qs are for critic robustness; gradient
            # signal needs just one head, plus min would zero out gradients
            # through the non-min head.
            state_feat_a = torch.cat([z_rl_a, s_p], dim=-1)
            a_pred_feat = a_pred.flatten(start_dim=1)
            q1_for_max = self.critic.q_id_forward(0, state_feat_a, a_pred_feat).squeeze(
                -1
            )
            loss_qmax = -q1_for_max.mean()

            loss_reg = (delta**2).mean() if self.residual_reg > 0 else 0.0
            loss_a = (
                loss_awr
                + self.residual_alpha * loss_qmax
                + self.residual_reg * loss_reg
            )
            # mu_actor surrogate for downstream metrics naming consistency.
            mu_actor = a_pred
        else:
            # Standard IQL actor: deterministic mean output (training=False
            # → no ref dropout, no noise).
            mu_actor = self.actor(z_rl_a, s_p, ref_action, training=False)
            loss_a = qrt_iql_actor_loss(
                actions=action,
                mu_actor=mu_actor,
                q_target=q_for_adv,
                v_pred=v_for_adv,
                beta_iql=self.iql_beta,
                weight_clip=self.iql_weight_clip,
            )

        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_enc.zero_grad(set_to_none=True)
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.grad_clip_norm
        )
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=self.grad_clip_norm,
        )
        self.opt_actor.step()
        self.opt_enc.step()

        # ----- (4) Soft target-critic update only (no target V in IQL) -----
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        # Metric naming: keep top-level "loss_critic" / "loss_actor" / "loss_recon"
        # so existing log scrapers and analysis scripts (β sweep Phase 1) still work.
        # Add "loss_v" + "advantage_mean" / "weight_mean" for IQL-specific diagnostics.
        with torch.no_grad():
            advantage = (q_for_adv - v_for_adv).detach()
            weight = torch.exp(self.iql_beta * advantage).clamp(
                max=self.iql_weight_clip
            )
        out_metrics = {
            "loss_critic": float(loss_q.detach().item()),
            "loss_actor": float(loss_a.detach().item()),
            "loss_recon": float(loss_recon.detach().item()),
            "loss_v": float(loss_v.detach().item()),
            "q_mean": float(q1.detach().mean().item()),
            "v_mean": float(v_pred.detach().mean().item()),
            "advantage_mean": float(advantage.mean().item()),
            "weight_mean": float(weight.mean().item()),
        }
        if self.use_cql:
            out_metrics["loss_q_td"] = float(loss_q_td.detach().item())
            out_metrics["loss_cql"] = float(loss_cql.detach().item())
        if self.use_residual_actor:
            with torch.no_grad():
                # delta is in scope from the residual branch above.
                delta_norm = delta.detach().pow(2).sum(dim=(-1, -2)).sqrt().mean()
                out_metrics["delta_norm"] = float(delta_norm.item())
                out_metrics["loss_awr"] = float(loss_awr.detach().item())
                out_metrics["loss_qmax"] = float(loss_qmax.detach().item())
        return out_metrics
