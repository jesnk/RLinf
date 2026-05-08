"""Layerwise gated velocity reparameterization for σ N1 (single-gate)
and σ N2 (per-layer gate).

Reference: SAC-Flow (arXiv 2509.25756) Flow-G/T uses single MLP gate at the
output of the velocity network. σ N1 reuses this single-gate scheme on the
π0.5 action expert (Gemma-300M, 18 layer, hidden 1024). σ N2 extends it to
per-layer gates on each Gemma block's MLP output.

Hook points:
    - N1 single-gate (SingleVelocityGate): pi0_pytorch.py:455 outputs_embeds
      output of paligemma_with_expert.forward — apply on the action-expert
      hidden state suffix slice before action_out_proj.
    - N2 layerwise (LayerwiseGate): gemma_pytorch.py:159 compute_layer_complete
      — apply per-layer at each Gemma decoder layer's residual output.

설계 노트:
    1. Gate parameters are tied to the flow expert, not the PaliGemma VLM
       (we freeze the VLM in N1). They are trained jointly with the actor.
    2. We initialize gate logits at 0 (sigmoid → 0.5) so gating is a halving
       perturbation at the start; or at +3 (sigmoid≈0.95) for near-identity.
    3. For N2 we stash the gate as an attribute on the Gemma module and
       monkey-patch `compute_layer_complete` to multiply by gate before
       returning the residual stream. This avoids fork of HF Gemma code.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# σ N1 — single-gate Flow-G/T velocity reparam
# ---------------------------------------------------------------------------
class SingleVelocityGate(nn.Module):
    """SAC-Flow Flow-G single-gate at velocity head output.

    Equivalent to SAC-Flow Flow-G:
        v_gated = sigmoid(W_g · h + b_g) ⊙ v
    where h is the suffix hidden state and v is the predicted velocity.

    Channel-wise gate (per-feature scalar in [0, 1]) — input-conditional.

    Args:
        d_model:     Hidden width of the suffix output (π0.5 action expert = 1024).
        action_dim:  Output velocity dimension (= action_dim, e.g. 32).
        hidden_dims: Optional MLP between hidden state and gate logits.
        init_value:  sigmoid(init_value) 의 초기 gate. 0.0 → 0.5 (SAC-Flow 기본).
    """

    def __init__(
        self,
        d_model: int = 1024,
        action_dim: int = 32,
        hidden_dims: tuple = (256,),
        init_value: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        layers = []
        in_dim = d_model
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.gate_net = nn.Sequential(*layers)
        # init last linear so gate logit ≈ init_value → sigmoid(init_value)
        last_lin = self.gate_net[-1]
        assert isinstance(last_lin, nn.Linear)
        nn.init.zeros_(last_lin.weight)
        nn.init.constant_(last_lin.bias, init_value)

    def forward(self, suffix_hidden: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Apply input-conditional gate.

        Args:
            suffix_hidden: [B, T, d_model] — flow expert suffix hidden state.
            velocity:      [B, T, action_dim] — pre-gate velocity prediction.

        Returns:
            gated velocity, same shape as `velocity`.
        """
        if suffix_hidden.shape[:2] != velocity.shape[:2]:
            raise ValueError(
                f"suffix_hidden {suffix_hidden.shape} and velocity {velocity.shape} "
                "must agree on (B, T)"
            )
        gate_logits = self.gate_net(suffix_hidden)
        gate = torch.sigmoid(gate_logits)
        return gate * velocity


# ---------------------------------------------------------------------------
# σ N2 — per-layer gating on each Gemma decoder block
# ---------------------------------------------------------------------------
class LayerwiseGate(nn.Module):
    """Per-layer sigmoid gate for flow expert transformer (σ N2).

    Applied at each Gemma layer's residual output. Each layer has an
    independent per-channel gate parameter.

    Initialized at logit=3.0 (sigmoid≈0.95) so initial behavior is near-identity.

    Usage (σ N2):
        gate = LayerwiseGate(num_layers, d_model)
        for i, layer in enumerate(gemma_expert.model.layers):
            layer._sigma_layer_gate = gate.bind(i)
        # then patch GemmaDecoderLayer.forward to multiply at residual.
    """

    def __init__(self, num_layers: int = 18, d_model: int = 1024, init_logit: float = 3.0):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.gate_logits = nn.Parameter(torch.full((num_layers, d_model), init_logit))

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        gate = torch.sigmoid(self.gate_logits[layer_idx])
        return x * gate

    def bind(self, layer_idx: int):
        """Return a callable bound to a specific layer index."""
        return LayerwiseGate.BoundGate(self, layer_idx)

    class BoundGate:
        def __init__(self, parent: "LayerwiseGate", layer_idx: int):
            self.parent = parent
            self.layer_idx = layer_idx

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return self.parent(x, self.layer_idx)
