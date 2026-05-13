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

"""Paper-faithful RLT-online sim port (B2 baseline in sigma_pivot_v2_design 3.6).

Subclass of QRTOfflinePolicyWorker (Task 6). Difference from sigma-QRT:
- Stage 1: encoder + decoder trained with reconstruction loss only (RLT Eq.2).
- freeze_encoder(): encoder + decoder set requires_grad=False, frozen for Stage 2.
- Stage 2: same as sigma-QRT train_step but encoder grad is zero (frozen contract).

Used in W4 gate as:
- B2 baseline (online RL with frozen encoder after Stage 1) - full plan.
- A1 ablation (offline RL with frozen encoder after Stage 1) - same worker, offline buffer.

The env-rollout loop for B2 (online) lives in a separate training entry script
(Task 11); this worker only exposes stage1_step / stage2_step / freeze_encoder.
"""

import torch

from rlinf.algorithms.losses import rlt_recon_loss
from rlinf.workers.actor.fsdp_qrt_offline_policy_worker import QRTOfflinePolicyWorker


class RLTOnlineSimWorker(QRTOfflinePolicyWorker):
    """RLT-style worker: encoder frozen after Stage 1."""

    def setup(self):
        super().setup()
        self._encoder_frozen = False

    def stage1_step(self, batch: dict) -> dict:
        """Encoder + decoder trained with reconstruction loss only (RLT Eq.2)."""
        if self._encoder_frozen:
            raise RuntimeError("stage1_step called after freeze_encoder()")
        z_obs = batch["z_obs"].to(self.device)
        z_rl = self.encoder(z_obs)
        z_prev = z_obs[:, :-1].detach()
        z_hat = self.decoder(z_rl, z_prev)
        loss = rlt_recon_loss(z_hat, z_obs.detach())
        self.opt_enc.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=self.grad_clip_norm,
        )
        self.opt_enc.step()
        return {"loss_recon": float(loss.detach().item())}

    def freeze_encoder(self):
        """Freeze encoder + decoder for Stage 2 (paper-faithful RLT).

        Sets requires_grad=False on encoder + decoder params and clears any
        stale gradient tensors left over from Stage 1. Replaces self.opt_enc
        with a no-op optimizer so super().train_step()'s opt_enc.step() does
        not touch frozen params if the parent worker is reused.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
            p.grad = None
        for p in self.decoder.parameters():
            p.requires_grad = False
            p.grad = None
        # Replace encoder optimizer with a no-op so super().train_step()'s
        # opt_enc.step() does not touch frozen params. Adam would keep state
        # for nonexistent grads; cleaner to redirect to a sentinel param.
        dummy = torch.nn.Parameter(torch.zeros(1))
        self.opt_enc = torch.optim.Adam([dummy], lr=0.0)
        self._encoder_frozen = True

    def stage2_step(self, batch: dict, step: int) -> dict:
        """Run TD3+BC actor-critic update with encoder frozen."""
        if not self._encoder_frozen:
            raise RuntimeError("stage2_step called before freeze_encoder()")
        return super().train_step(batch, step=step)
