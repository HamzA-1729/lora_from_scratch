"""
Unit Tests for LoRA In-Place Weight Merging and Unloading:
1. In-place merge numerical equivalence to unmerged forward pass.
2. Unmerge numerical reversibility (atol=1e-5).
3. Idempotency of merge() and unmerge() calls.
4. merge_and_unload() standard nn.Linear conversion and zero-latency serving parity.
5. Error enforcement when training while merged.
"""

import pytest
import torch
import torch.nn as nn
from lora_from_scratch.lora_layer import LoRALinear


def test_merge_and_unmerge_reversibility():
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    base_linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(base_linear, rank=4, alpha=8.0, dropout=0.0)

    # Populate non-zero adapter weights
    with torch.no_grad():
        lora_linear.lora_A.weight.copy_(torch.randn_like(lora_linear.lora_A.weight))
        lora_linear.lora_B.weight.copy_(torch.randn_like(lora_linear.lora_B.weight))

    original_w0 = base_linear.weight.clone()

    # 1. Merge
    lora_linear.merge()
    assert lora_linear.merged is True
    assert not torch.equal(base_linear.weight, original_w0), "Base weight must change upon merging!"

    # 2. Unmerge
    lora_linear.unmerge()
    assert lora_linear.merged is False
    assert torch.allclose(base_linear.weight, original_w0, atol=1e-5), "Unmerge must restore original W_0 within atol=1e-5!"


def test_merged_logits_equivalence():
    """Output of merged layer must match output of unmerged layer."""
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    base_linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(base_linear, rank=4, alpha=8.0, dropout=0.0)

    with torch.no_grad():
        lora_linear.lora_A.weight.copy_(torch.randn_like(lora_linear.lora_A.weight))
        lora_linear.lora_B.weight.copy_(torch.randn_like(lora_linear.lora_B.weight))

    lora_linear.eval()
    x = torch.randn(4, 16, in_features)

    # Output before merge
    with torch.no_grad():
        out_unmerged = lora_linear(x)

    # Output after merge
    lora_linear.merge()
    with torch.no_grad():
        out_merged = lora_linear(x)

    assert torch.allclose(out_merged, out_unmerged, atol=1e-5), "Merged output must match unmerged output!"


def test_merge_unmerge_idempotency():
    torch.manual_seed(42)
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, rank=2, alpha=4.0)

    with torch.no_grad():
        lora.lora_A.weight.fill_(0.5)
        lora.lora_B.weight.fill_(0.5)

    original_w = base.weight.clone()

    # Double merge
    lora.merge()
    first_merge_w = base.weight.clone()
    lora.merge()
    assert torch.equal(base.weight, first_merge_w), "Second merge() call must be a no-op!"

    # Double unmerge
    lora.unmerge()
    first_unmerge_w = base.weight.clone()
    lora.unmerge()
    assert torch.equal(base.weight, first_unmerge_w), "Second unmerge() call must be a no-op!"
    assert torch.allclose(base.weight, original_w, atol=1e-5)


def test_merge_and_unload():
    torch.manual_seed(42)
    base = nn.Linear(32, 48, bias=True)
    lora = LoRALinear(base, rank=4, alpha=8.0)

    with torch.no_grad():
        lora.lora_A.weight.copy_(torch.randn_like(lora.lora_A.weight))
        lora.lora_B.weight.copy_(torch.randn_like(lora.lora_B.weight))

    lora.eval()
    x = torch.randn(2, 32)
    expected_out = lora(x)

    # Unload to pure nn.Linear
    unloaded_linear = lora.merge_and_unload()

    assert isinstance(unloaded_linear, nn.Linear)
    assert not isinstance(unloaded_linear, LoRALinear)
    assert unloaded_linear.bias is not None

    with torch.no_grad():
        unloaded_out = unloaded_linear(x)

    assert torch.allclose(unloaded_out, expected_out, atol=1e-5)


def test_cannot_train_while_merged():
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, rank=2, alpha=4.0)
    lora.merge()
    lora.train()

    x = torch.randn(2, 16)
    with pytest.raises(RuntimeError, match="Cannot train LoRALinear"):
        lora(x)

