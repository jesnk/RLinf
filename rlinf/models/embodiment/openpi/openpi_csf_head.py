# Copyright 2026 The RLinf Authors.
# Copyright 2026 σ project (jesnk).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""σ N1 — Chunked SAC-Flow head mixin for OpenPi0ForRLActionPrediction.

Implements:
    - csf_forward(obs):   π0.5 flow VLA forward + Flow-G/T velocity reparam
                          (multi-step BPTT 유지, detach 제거).
    - csf_q_forward(obs, action): PaliGemma prefix + flow expert suffix one-pass
                                  → MultiQHead → Q value [B, K].

Differences vs sac_forward (DSRL latent-noise variant):
    - DSRL operates in a latent noise space (32-d Gaussian sampled by a small
      GaussianPolicy); SAC-Flow operates in the **weight-space** of the π0.5
      flow expert. The action is the output of the full denoising loop.
    - DSRL uses a tiny critic encoder; CSF uses the flow expert hidden state
      after the last denoising step as state features (combined with state
      proj). This shares the actor/critic backbone for sample efficiency.

Implementation strategy:
    Free functions plugged in as instance methods via openpi_action_model.py
    `csf_forward = csf_forward_impl(self, ...)`. This avoids touching
    PI0Pytorch MRO and keeps the diff small. Future refactor → mixin class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from openpi.models import model as _model

from rlinf.models.embodiment.modules.layerwise_gate import (
    LayerwiseGate,
    SingleVelocityGate,
)
from rlinf.models.embodiment.modules.q_head import MultiQHead


# ---------------------------------------------------------------------------
# Module factory used at __init__ in openpi_action_model.py
# ---------------------------------------------------------------------------
def build_csf_components(self, config) -> None:
    """Attach CSF (σ N1) components to an OpenPi0ForRLActionPrediction.

    Called from __init__ when config.use_csf=True. Adds:
        - self.csf_velocity_gate     (SingleVelocityGate, σ N1)
        - self.csf_layerwise_gate    (LayerwiseGate, σ N2; only if csf_layerwise_gate)
        - self.csf_state_proj        (linear from suffix hidden to critic_state_dim)
        - self.q_head                (MultiQHead with K=csf_num_q_heads)
    """
    d_model = 1024  # π0.5 action expert width (validated in plan)
    action_dim = config.action_dim
    num_q_heads = getattr(config, "csf_num_q_heads", 4)
    state_features = getattr(config, "csf_critic_state_dim", 512)

    # bf16 for parity with backbone
    csf_dtype = torch.bfloat16

    # σ N1: single-gate Flow-G at suffix hidden → velocity multiplication
    self.csf_velocity_gate = SingleVelocityGate(
        d_model=d_model,
        action_dim=action_dim,
        hidden_dims=(256,),
        init_value=0.0,
    ).to(dtype=csf_dtype)

    # σ N2: optional layerwise gate (default off in N1)
    self.csf_use_layerwise_gate = bool(getattr(config, "csf_layerwise_gate", False))
    if self.csf_use_layerwise_gate:
        self.csf_layerwise_gate = LayerwiseGate(
            num_layers=18,  # Gemma-300M flow expert depth
            d_model=d_model,
            init_logit=3.0,
        ).to(dtype=csf_dtype)
        # TODO(σ N2): monkey-patch each gemma layer.forward to apply gate at residual.
        # This requires gemma_pytorch.py:159 compute_layer_complete hook.

    # critic feature projection: suffix hidden → critic_state_dim
    self.csf_state_proj = nn.Linear(d_model, state_features).to(dtype=csf_dtype)

    # twin / multi Q ensemble (default K=4 for LCB pessimism)
    self.q_head = MultiQHead(
        hidden_size=state_features,
        action_feature_dim=action_dim * config.action_horizon,
        hidden_dims=[256, 256],
        num_q_heads=num_q_heads,
        output_dim=1,
    ).to(dtype=csf_dtype)


# ---------------------------------------------------------------------------
# csf_forward — pathwise action sampling with Flow-G gating
# ---------------------------------------------------------------------------
def csf_forward_impl(
    self,
    obs=None,
    data=None,
    train: bool = True,
    return_chains: bool = False,
    **kwargs,
):
    """π0.5 flow VLA forward with Flow-G gated velocity reparam.

    Returns (dict):
        action:    [B, action_horizon, action_dim] — final denoised action,
                   gradient flows through full denoise BPTT (no detach).
        log_pi:    [B] — log π(action | obs) approximated via Σ_t log N(x_t; μ, σ).
        suffix_features: [B, action_horizon, d_model] — last-step suffix hidden
                         state for downstream Q evaluation.
    """
    if not self.config.use_csf:
        raise ValueError("csf_forward called but use_csf=False")

    if obs is None:
        obs = (data.get("obs", data) if data is not None else kwargs.get("obs", {}))

    observation = self.input_transform(obs, transpose=False)
    observation = _model.Observation.from_dict(observation)
    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
        observation, train=False
    )
    device = state.device
    images = [img.to(device) for img in images]
    img_masks = [img_mask.to(device) for img_mask in img_masks]

    # Prefix cache (frozen VLM)
    prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
        images, img_masks, lang_tokens, lang_masks
    )

    bsize = state.shape[0]
    actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
    x_t = self.sample_noise(actions_shape, device).to(self.action_in_proj.weight.dtype)

    chains = [x_t]
    log_probs = []
    last_suffix = None

    num_steps = self.config.num_steps
    timesteps = self._get_timesteps(num_steps, device)

    # Multi-step denoise with **gradient retained** through all steps (BPTT).
    # NB: flow_actor.py:217 applies `.detach()` on total_log_prob — σ does NOT.
    for idx in range(num_steps):
        t_input = timesteps[idx].expand(bsize)
        delta = (timesteps[idx] - timesteps[idx + 1]).expand(bsize)

        suffix_out = self.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_input)
        v_raw = self.action_out_proj(suffix_out)

        # σ N1 — Flow-G/T single gate at velocity output
        gate_dtype = self.csf_velocity_gate.gate_net[-1].weight.dtype
        v_t = self.csf_velocity_gate(
            suffix_out.to(gate_dtype),
            v_raw.to(gate_dtype),
        ).to(v_raw.dtype)

        delta_b = delta[:, None, None].expand_as(x_t)
        t_b = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_b
        x1_pred = x_t + v_t * (1 - t_b)

        sigma = float(getattr(self.config, "csf_sde_noise", 0.05))
        sigma_t = torch.full_like(t_b, sigma)
        x0_w = 1 - (t_b - delta_b)
        x1_w = (t_b - delta_b)
        x_t_mean = x0_pred * x0_w + x1_pred * x1_w
        x_t_std = torch.sqrt(delta_b) * sigma_t

        # SDE step — gradient retained (no detach!)
        eps = self.sample_noise(x_t.shape, device).to(x_t.dtype)
        x_t = x_t_mean + x_t_std * eps

        # log_prob (Gaussian, ignore constant) — used for entropy term
        log_prob_step = -((eps ** 2).sum(dim=(-1, -2)) * 0.5)
        log_probs.append(log_prob_step)
        chains.append(x_t)
        last_suffix = suffix_out

    action = x_t  # [B, H, A]
    log_pi = torch.stack(log_probs, dim=1).sum(dim=1)  # [B]

    out = {
        "action": action,
        "log_pi": log_pi,
        "suffix_features": last_suffix,
    }
    if return_chains:
        out["chains"] = torch.stack(chains, dim=1)
    return out


# ---------------------------------------------------------------------------
# csf_q_forward — Q value of (s, a) using shared backbone features
# ---------------------------------------------------------------------------
def csf_q_forward_impl(
    self,
    obs=None,
    data=None,
    actions=None,
    detach_encoder: bool = False,
    suffix_features=None,
    train: bool = True,
    **kwargs,
):
    """Compute Q(s, a) using PaliGemma prefix + flow expert suffix as state encoder.

    If `suffix_features` is provided, reuse it (saves compute). Otherwise re-run
    a single denoising step at t=0 to extract suffix features.

    Returns:
        Q values, shape [B, K] (K = num_q_heads).
    """
    if not self.config.use_csf:
        raise ValueError("csf_q_forward called but use_csf=False")

    if obs is None:
        obs = (data.get("obs", data) if data is not None else kwargs.get("obs", {}))
    if actions is None:
        actions = kwargs.get("actions")
    if actions is None:
        raise ValueError("csf_q_forward needs `actions`")

    if suffix_features is None:
        observation = self.input_transform(obs, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )
        device = state.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
        bsize = state.shape[0]
        timesteps = self._get_timesteps(self.config.num_steps, device)
        t0 = timesteps[0].expand(bsize)
        # Use action provided as x_t to compute Q on observed action
        x_t = actions.to(self.action_in_proj.weight.dtype)
        suffix_features = self.get_suffix_out(
            state, prefix_pad_masks, past_key_values, x_t, t0
        )

    pooled = suffix_features.mean(dim=1)  # [B, d_model]
    if detach_encoder:
        pooled = pooled.detach()

    state_feat = self.csf_state_proj(pooled.to(self.csf_state_proj.weight.dtype))
    flat_actions = actions.reshape(actions.shape[0], -1).to(state_feat.dtype)

    q_values = self.q_head(state_feat, flat_actions)  # [B, K]
    return q_values
