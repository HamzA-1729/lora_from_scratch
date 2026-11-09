"""
Unit Tests for Generic LoRALinear Layer:
1. Step-0 bitwise equivalence to the base layer (B=0 initialization).
2. Two-step gradient isolation (base remains frozen; B gets grad at step 0, A gets grad at step 1).
3. Scaling factor alpha / r linear proportionality.
4. Thread-safe ContextVar dynamic adapter routing.
"""

import torch
import torch.nn as nn
from lora_from_scratch.lora_layer import LoRALinear, active_adapter_context


def test_step0_exact_equivalence():
    """
    At initialization, B=0 guarantees that delta W = (alpha/r)*B*A is identically zero.
    In eval mode (dropout disabled), LoRALinear must produce bitwise identical output to base.
    """
    torch.manual_seed(42)
    in_features, out_features = 64, 128
    base_linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(base_linear, rank=8, alpha=16.0, dropout=0.0)

    lora_linear.eval()
    base_linear.eval()

    x = torch.randn(4, 16, in_features)
    with torch.no_grad():
        out_base = base_linear(x)
        out_lora = lora_linear(x)

    # Must be bitwise identical
    assert torch.equal(out_base, out_lora), "Step-0 LoRA output must be bitwise identical to base linear!"


def test_two_step_gradient_isolation():
    """
    Tests gradient flow across initial steps:
    - Base weights remain strictly frozen (grad is None).
    - Step 0: B=0 means grad reaches B, but grad to A is zero.
    - Step 1: After optimizer updates B to non-zero, both A and B receive non-zero gradients.
    """
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    base_linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(base_linear, rank=4, alpha=8.0, dropout=0.0)
    lora_linear.train()

    optimizer = torch.optim.SGD(lora_linear.parameters(), lr=0.1)

    # --- Backward 1 (Step 0) ---
    x1 = torch.randn(2, 8, in_features)
    out1 = lora_linear(x1)
    loss1 = out1.sum()
    loss1.backward()

    # Base weight must NEVER receive gradients
    assert lora_linear.base_layer.weight.grad is None, "Base layer weight must be frozen!"
    # B receives gradient
    assert lora_linear.lora_B.weight.grad is not None
    assert lora_linear.lora_B.weight.grad.abs().sum() > 0, "lora_B must receive gradient at step 0!"
    # A receives zero gradient at step 0 because B=0
    assert lora_linear.lora_A.weight.grad is not None
    assert torch.all(lora_linear.lora_A.weight.grad == 0.0), "lora_A gradient must be zero at step 0 due to B=0!"

    # Optimizer step updates B to non-zero
    optimizer.step()
    optimizer.zero_grad()
    assert lora_linear.lora_B.weight.abs().sum() > 0, "lora_B must be non-zero after optimizer step!"

    # --- Backward 2 (Step 1) ---
    x2 = torch.randn(2, 8, in_features)
    out2 = lora_linear(x2)
    loss2 = out2.sum()
    loss2.backward()

    # Base weight remains frozen
    assert lora_linear.base_layer.weight.grad is None, "Base layer weight must remain frozen!"
    # Now BOTH A and B receive non-zero gradients
    assert lora_linear.lora_A.weight.grad is not None and lora_linear.lora_A.weight.grad.abs().sum() > 0
    assert lora_linear.lora_B.weight.grad is not None and lora_linear.lora_B.weight.grad.abs().sum() > 0


def test_scaling_alpha_over_r():
    """Output delta must scale linearly with alpha / r."""
    torch.manual_seed(42)
    in_features, out_features = 16, 32
    base_linear = nn.Linear(in_features, out_features)

    # Initialize two layers with different alpha
    lora_1 = LoRALinear(base_linear, rank=4, alpha=8.0)    # scaling = 2.0
    lora_2 = LoRALinear(base_linear, rank=4, alpha=16.0)   # scaling = 4.0

    # Give both identical non-zero weights
    with torch.no_grad():
        test_a = torch.randn_like(lora_1.lora_A.weight)
        test_b = torch.randn_like(lora_1.lora_B.weight)
        lora_1.lora_A.weight.copy_(test_a)
        lora_1.lora_B.weight.copy_(test_b)
        lora_2.lora_A.weight.copy_(test_a)
        lora_2.lora_B.weight.copy_(test_b)

    lora_1.eval()
    lora_2.eval()
    base_linear.eval()

    x = torch.randn(2, 4, in_features)
    with torch.no_grad():
        base_out = base_linear(x)
        delta_1 = lora_1(x) - base_out
        delta_2 = lora_2(x) - base_out

    # delta_2 should be exactly 2x delta_1
    assert torch.allclose(delta_2, 2.0 * delta_1, atol=1e-6)


def test_contextvar_multi_adapter_routing():
    """Verifies thread-safe ContextVar dynamic adapter overriding."""
    torch.manual_seed(42)
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, rank=4, alpha=8.0, layer_name="layer1")
    lora.eval()

    x = torch.randn(2, 16)

    # Base output (no adapter in context)
    out_base = lora(x)

    # Mock an external adapter
    custom_a = torch.randn(4, 16)
    custom_b = torch.randn(32, 4)
    custom_scaling = 1.5

    token = active_adapter_context.set({"layer1": (custom_a, custom_b, custom_scaling)})
    try:
        out_custom = lora(x)
        expected_custom = base(x) + custom_scaling * (x @ custom_a.t() @ custom_b.t())
        assert torch.allclose(out_custom, expected_custom, atol=1e-5)
    finally:
        active_adapter_context.reset(token)

    # After reset, context is cleared and output returns to base
    assert torch.equal(lora(x), out_base)

