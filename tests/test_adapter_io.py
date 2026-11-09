"""
Unit Tests for Adapter Serialization and Architecture Validation:
1. Save and load weight roundtrip exact match.
2. Verification that only adapter parameters are serialized (no base weights).
3. Rejection of rank or dimension mismatches during load.
4. Validation of load_adapter_for_context dynamic serving dict.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from modern_decoder_llm import DecoderLLM
from lora_from_scratch.lora_layer import LoRALinear, active_adapter_context
from lora_from_scratch.injection import inject_lora
from lora_from_scratch.adapter_io import save_adapter, load_adapter, load_adapter_for_context


@pytest.fixture
def base_model() -> DecoderLLM:
    torch.manual_seed(42)
    return DecoderLLM(
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


def test_adapter_save_and_load_roundtrip(base_model: DecoderLLM, tmp_path: Path):
    """Verifies that saved adapter weights load back with bitwise parity and base weights remain identical."""
    model, _ = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)

    # Initialize non-zero pseudo-trained adapter weights
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.weight.normal_(0.0, 0.02)
                module.lora_B.weight.normal_(0.0, 0.02)

    # Save
    save_dir = tmp_path / "test_lora_checkpoint"
    save_adapter(model, save_dir, config_overrides={"task": "lesson_continuation"})

    assert (save_dir / "adapter_config.json").exists()
    assert (save_dir / "adapter_model.pt").exists()

    # Create fresh model with same base architecture
    torch.manual_seed(42)
    fresh_base = DecoderLLM(
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
    fresh_model, _ = inject_lora(fresh_base, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)

    # Load adapter
    config = load_adapter(fresh_model, save_dir, strict=True)
    assert config["rank"] == 8
    assert config["alpha"] == 16.0
    assert config["task"] == "lesson_continuation"

    # Compare adapter weights
    for (name1, m1), (name2, m2) in zip(model.named_modules(), fresh_model.named_modules()):
        if isinstance(m1, LoRALinear):
            assert isinstance(m2, LoRALinear)
            assert torch.equal(m1.lora_A.weight, m2.lora_A.weight)
            assert torch.equal(m1.lora_B.weight, m2.lora_B.weight)

    # Forward pass equivalence
    x = torch.randint(0, 4096, (2, 32))
    with torch.no_grad():
        out1, _ = model(x)
        out2, _ = fresh_model(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_save_only_adapter_parameters(base_model: DecoderLLM, tmp_path: Path):
    """Verifies that adapter_model.pt contains exclusively lora_A and lora_B keys, zero base weights."""
    model, _ = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=4, alpha=8.0)
    save_dir = tmp_path / "small_checkpoint"
    save_adapter(model, save_dir)

    weights = torch.load(save_dir / "adapter_model.pt", weights_only=True)

    # Exactly 6 layers * 2 modules (q_proj, v_proj) * 2 tensors (A, B) = 24 tensors
    assert len(weights) == 24
    for key in weights:
        assert ".lora_A.weight" in key or ".lora_B.weight" in key
        assert "base_layer" not in key
        assert "lm_head" not in key


def test_load_adapter_dimension_and_rank_mismatch_rejection(base_model: DecoderLLM, tmp_path: Path):
    """Rejects loading an adapter with rank mismatch or layer shape mismatch."""
    model_r8, _ = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)
    save_dir = tmp_path / "r8_checkpoint"
    save_adapter(model_r8, save_dir)

    # Fresh model injected with rank 4
    torch.manual_seed(42)
    other_base = DecoderLLM(
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
    model_r4, _ = inject_lora(other_base, target_modules=["q_proj", "v_proj"], rank=4, alpha=8.0)

    # Must raise ValueError due to rank mismatch (8 != 4)
    with pytest.raises(ValueError, match="Rank mismatch"):
        load_adapter(model_r4, save_dir)


def test_load_adapter_for_context(base_model: DecoderLLM, tmp_path: Path):
    """Verifies that load_adapter_for_context creates an active_adapter_context dictionary producing identical logits."""
    model, _ = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.weight.normal_(0.0, 0.02)
                module.lora_B.weight.normal_(0.0, 0.02)

    save_dir = tmp_path / "context_checkpoint"
    save_adapter(model, save_dir)

    # Load context dictionary
    ctx = load_adapter_for_context(save_dir)
    assert len(ctx) == 12  # 6 layers * 2 projections (q_proj, v_proj)

    # Forward with stateful model
    x = torch.randint(0, 4096, (2, 16))
    with torch.no_grad():
        out_stateful, _ = model(x)

    # Create fresh model with zeroed B adapters
    torch.manual_seed(42)
    fresh_base = DecoderLLM(
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
    fresh_model, _ = inject_lora(fresh_base, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)

    # Without context, fresh_model produces base output (since B=0)
    # With context set, fresh_model must match out_stateful!
    token = active_adapter_context.set(ctx)
    try:
        with torch.no_grad():
            out_contextual, _ = fresh_model(x)
        assert torch.allclose(out_stateful, out_contextual, atol=1e-5)
    finally:
        active_adapter_context.reset(token)

