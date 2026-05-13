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
    qrt_critic_loss,
    rlt_recon_loss,
)
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.rl_token import RLTokenDecoder, RLTokenEncoder


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
    ) -> torch.Tensor:
        if training and self.ref_action_dropout > 0:
            mask = (
                torch.rand(ref_action.shape[0], 1, 1, device=ref_action.device)
                > self.ref_action_dropout
            ).float()
            ref = ref_action * mask
        else:
            ref = ref_action
        x = torch.cat([z_rl, s_p, ref.flatten(start_dim=1)], dim=-1)
        mu = self.net(x).view(-1, self.chunk_len, self.action_dim)
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

    def _q(self, z_rl, s_p, action, target: bool):
        """Compute Q(s, a) using MultiQHead. Returns [B, num_q_heads]."""
        net = self.critic_target if target else self.critic
        state_feat = torch.cat([z_rl, s_p], dim=-1)
        action_feat = action.flatten(start_dim=1)
        return net(state_feat, action_feat)

    def train_step(self, batch: dict, step: int) -> dict:
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
            a_next = self.actor_target(z_rl_next, next_s_p, next_ref_action, training=True)
            noise = (torch.randn_like(a_next) * self.target_noise_std).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            a_next = a_next + noise
            q_next = self._q(z_rl_next, next_s_p, a_next, target=True)
            q_next_min = q_next.min(dim=-1).values
            q_target = qrt_compute_target(reward, q_next_min, done, self.gamma, self.chunk_len)

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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip_norm)
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
            q_pi = self._q(z_rl_for_actor, s_p, a_theta, target=False).min(dim=-1).values
            loss_a = qrt_actor_loss(q_pi, a_theta, ref_action, beta=self.beta_bc)
            self.opt_actor.zero_grad(set_to_none=True)
            self.opt_enc.zero_grad(set_to_none=True)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_norm)
            self.opt_actor.step()
            self.opt_enc.step()
            metrics["loss_actor"] = float(loss_a.detach().item())

            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
                for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return metrics
