"""Layerwise gated velocity reparameterization for σ N2.

Reference: SAC-Flow (arXiv 2509.25756) Flow-G가 single-block gate를 사용.
σ N2가 추가: π0.5의 Gemma flow expert 18 transformer block에 layerwise gate.
"""
import torch
import torch.nn as nn


class LayerwiseGate(nn.Module):
    """Per-layer sigmoid gate for flow expert transformer.

    Applied at each Gemma layer's MLP output (gemma_pytorch.py:232 hook point).
    """

    def __init__(self, num_layers: int = 18, d_model: int = 1024):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        # Per-layer gate parameters (sigmoid output ∈ [0, 1])
        self.gate_logits = nn.Parameter(torch.zeros(num_layers, d_model))

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply gate at given layer.

        Args:
            x: [B, ..., d_model] hidden state
            layer_idx: 0..num_layers-1
        Returns: gated x
        """
        assert 0 <= layer_idx < self.num_layers
        gate = torch.sigmoid(self.gate_logits[layer_idx])  # [d_model]
        return x * gate
