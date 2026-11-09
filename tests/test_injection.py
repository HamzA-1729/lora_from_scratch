"""
Unit Tests for Recursive Model Injection:
1. Target module replacement & protected module preservation (lm_head, tok_embeddings).
2. Exact mathematical parameter count verification:
   - q_proj + v_proj: exactly 4,992 * r parameters across 6 layers.
   - All linears: exactly 28,416 * r parameters across 6 layers.
3. Base parameter gradient freezing (requires_grad is False for all base weights).
4. Prevention of double-wrapping / re-injection errors.
5. merge_and_unload_model: Full model-level conversion back to pure nn.Linear with output parity.
"""

import pytest
import torch
import torch.nn as nn

from modern_decoder_llm import DecoderLLM
from lora_from_scratch.lora_layer import LoRALinear
from lora_from_scratch.injection import inject_lora, merge_and_unload_model


@pytest.fixture
def base_model() -> DecoderLLM:
    """Provides a deterministic 5.57M DecoderLLM instance."""
    torch.manual_seed(42)
    model = DecoderLLM(
        vocab_size=4096,
        max_seq_len=256,
        hidden_dim=256,
        depth=6,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=768,
        rope_theta=10000.0,
        tie_weights=True,
        norm_eps=1e-6,
    )
    return model


def test_injection_targets_and_lm_head_protection(base_model: DecoderLLM):
    """
    Verifies that only target_modules (q_proj, v_proj) are wrapped,
    while lm_head and other linear layers remain standard nn.Linear.
    """
    model, stats = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)

    # 1. Tied output LM head must remain standard nn.Linear (never wrapped!)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, LoRALinear)

    # 2. q_proj and v_proj in every decoder block must be LoRALinear
    for layer in model.layers:
        assert isinstance(layer.attn.q_proj, LoRALinear)
        assert isinstance(layer.attn.v_proj, LoRALinear)
        # Non-targeted layers must remain standard nn.Linear
        assert isinstance(layer.attn.k_proj, nn.Linear) and not isinstance(layer.attn.k_proj, LoRALinear)
        assert isinstance(layer.attn.o_proj, nn.Linear) and not isinstance(layer.attn.o_proj, LoRALinear)
        assert isinstance(layer.ffn.w_gate, nn.Linear) and not isinstance(layer.ffn.w_gate, LoRALinear)
        assert isinstance(layer.ffn.w_up, nn.Linear) and not isinstance(layer.ffn.w_up, LoRALinear)
        assert isinstance(layer.ffn.w_down, nn.Linear) and not isinstance(layer.ffn.w_down, LoRALinear)

    # 3. Base weights must be frozen
    assert model.tok_embeddings.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    assert model.layers[0].attn.k_proj.weight.requires_grad is False


def test_parameter_count_qv_formula(base_model: DecoderLLM):
    """
    Verifies exact parameter formula for q_proj + v_proj:
        q_proj: (256 * r) + (r * 256) = 512 * r
        v_proj: (64 * r) + (r * 256)  = 320 * r
        Per layer: 832 * r
        Across 6 layers: 4,992 * r
    For r = 8: 4,992 * 8 = 39,936 trainable parameters.
    """
    rank = 8
    model, stats = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=rank, alpha=16.0)

    expected_trainable = 4992 * rank  # 39,936
    actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert stats["trainable_params"] == expected_trainable
    assert actual_trainable == expected_trainable
    # 39,936 on 5.57M is ~0.72%
    assert abs(stats["trainable_pct"] - 0.71) < 0.05


def test_parameter_count_all_linears_formula(base_model: DecoderLLM):
    """
    Verifies exact parameter formula for all linears (excluding lm_head):
        W_q: 512 * r
        W_k: (64 * r) + (r * 256) = 320 * r
        W_v: (64 * r) + (r * 256) = 320 * r
        W_o: (256 * r) + (r * 256) = 512 * r
        W_gate: (768 * r) + (r * 256) = 1024 * r
        W_up: (768 * r) + (r * 256) = 1024 * r
        W_down: (256 * r) + (r * 768) = 1024 * r
        Per layer: 4,736 * r
        Across 6 layers: 28,416 * r
    For r = 4: 28,416 * 4 = 113,664 trainable parameters.
    """
    rank = 4
    all_linear_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "w_gate", "w_up", "w_down"]
    model, stats = inject_lora(base_model, target_modules=all_linear_targets, rank=rank, alpha=8.0)

    expected_trainable = 28416 * rank  # 113,664
    actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert stats["trainable_params"] == expected_trainable
    assert actual_trainable == expected_trainable
    assert isinstance(model.lm_head, nn.Linear) and not isinstance(model.lm_head, LoRALinear)


def test_re_injection_prevention(base_model: DecoderLLM):
    """Calling inject_lora on an already-injected model must raise RuntimeError."""
    model, _ = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=8)
    with pytest.raises(RuntimeError, match="already injected with LoRA"):
        inject_lora(model, target_modules=["q_proj", "v_proj"], rank=8)


def test_merge_and_unload_model(base_model: DecoderLLM):
    """
    Verifies model-level merge_and_unload_model:
    - Replaces every LoRALinear with standard nn.Linear.
    - Yields output matching the pre-unloaded model within atol=1e-5.
    """
    model, _ = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=4, alpha=8.0)
    model.eval()

    # Populate non-zero adapter weights
    with torch.no_grad():
        for layer in model.layers:
            layer.attn.q_proj.lora_A.weight.fill_(0.01)
            layer.attn.q_proj.lora_B.weight.fill_(0.01)
            layer.attn.v_proj.lora_A.weight.fill_(0.01)
            layer.attn.v_proj.lora_B.weight.fill_(0.01)

    tokens = torch.randint(0, model.vocab_size, (2, 16))
    with torch.no_grad():
        expected_logits, _ = model(tokens)

    # Unload model to pure standard PyTorch layers
    unloaded_model = merge_and_unload_model(model)

    # Verify no LoRALinear remains
    for m in unloaded_model.modules():
        assert not isinstance(m, LoRALinear)

    # Verify output parity
    with torch.no_grad():
        unloaded_logits, _ = unloaded_model(tokens)

    assert torch.allclose(unloaded_logits, expected_logits, atol=1e-5)

