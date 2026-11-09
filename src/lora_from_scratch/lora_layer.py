"""
Generic Low-Rank Adaptation (LoRA) Linear Layer:
1. Matrix Factorization: W = W_0 + (alpha / r) * B * A
2. Step-0 Equivalence: B is initialized to exact zeros, A is initialized via Kaiming uniform.
3. In-Place Merging: Folds delta weights into W_0 under torch.no_grad() for zero-latency serving.
4. Thread-Safe Multi-Tenant Routing: Uses Python ContextVar to route dynamic adapter weights per-request.
5. Zero-Overhead Unloading: merge_and_unload() converts LoRALinear back into a standard nn.Linear.
"""

from contextvars import ContextVar
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thread-safe ContextVar holding {layer_name: (A_weight, B_weight, scaling_float)}
active_adapter_context: ContextVar[Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]]] = ContextVar(
    "active_adapter_context", default=None
)


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear implementing Low-Rank Adaptation (LoRA).
    
    Mathematical Formulation:
        h = W_0 x + (alpha / r) * B * A * x
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        layer_name: Optional[str] = None,
    ):
        super().__init__()
        assert isinstance(base_layer, nn.Linear), f"base_layer must be an nn.Linear, got {type(base_layer)}"
        assert rank > 0, f"rank must be > 0, got {rank}"

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.layer_name = layer_name

        # Freeze the base layer weights and bias
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # LoRA projection matrices
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        # Optional LoRA input dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Idempotent merge state tracking
        self.merged: bool = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialization according to Hu et al. (2021):
        - A ~ Kaiming uniform (non-zero variance enables backpropagation to A)
        - B = 0 (exact zero ensures delta W = 0 at step 0)
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass:
        - If merged: runs base_layer directly.
        - If unmerged and ContextVar active: dynamically applies requested adapter weights.
        - If unmerged standard: computes x @ A.T @ B.T in O(B * T * d * r) FLOPs.
        """
        if self.training and self.merged:
            raise RuntimeError(
                f"Cannot train LoRALinear layer '{self.layer_name}' while merged. "
                "Call unmerge() before training."
            )

        base_out = self.base_layer(x)

        if self.merged:
            return base_out

        # Check for multi-tenant dynamic adapter context (FastAPI serving mode)
        ctx = active_adapter_context.get()
        if ctx is not None:
            if self.layer_name is not None and self.layer_name in ctx:
                weight_a, weight_b, scaling = ctx[self.layer_name]
                # Dynamic adapter computation without in-place weight mutation
                adapted_out = F.linear(F.linear(self.dropout(x), weight_a), weight_b)
                return base_out + scaling * adapted_out
            else:
                # Context is active, but this layer is not targeted -> return base output
                return base_out

        # Standard forward pass with own trainable parameters
        # Factorized sequentially: x A^T first, then @ B^T
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """
        Folds adapter weights in-place into the base layer:
            W_0 <- W_0 + (alpha / r) * (B @ A)
        Idempotent: calling merge() multiple times will not duplicate adapter updates.
        """
        if self.merged:
            return

        delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.base_layer.weight.data.add_(delta_w)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """
        Reverses the merged adapter weights:
            W_0 <- W_0 - (alpha / r) * (B @ A)
        Idempotent: calling unmerge() multiple times will not duplicate subtraction.
        """
        if not self.merged:
            return

        delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.base_layer.weight.data.sub_(delta_w)
        self.merged = False

    def merge_and_unload(self) -> nn.Linear:
        """
        Returns a clean standard nn.Linear with merged weights,
        completely discarding the LoRA adapter branches.
        """
        if not self.merged:
            self.merge()

        has_bias = self.base_layer.bias is not None
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype

        new_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=has_bias,
            device=device,
            dtype=dtype,
        )
        new_linear.weight.data.copy_(self.base_layer.weight.data)
        if has_bias:
            new_linear.bias.data.copy_(self.base_layer.bias.data)

        return new_linear

