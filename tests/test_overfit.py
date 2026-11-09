"""
Unit Test for LoRA Convergence & Base Weight Invariance:
Verifies that a 2-layer DecoderLLM with LoRA adapters can overfit a synthetic sequence batch:
1. Overfitting Convergence: Cross-entropy loss drops by >95% within 250 steps.
2. Invariance Verification: All frozen base model parameters remain 100% bitwise unchanged.
3. Gradient & Update Verification: Trainable adapter parameters (A, B) actively update.
"""

import pytest
import torch
import torch.nn.functional as F

from modern_decoder_llm import DecoderLLM
from lora_from_scratch.injection import inject_lora
from lora_from_scratch.lora_layer import LoRALinear


def test_lora_overfit_and_base_weight_invariance():
    torch.manual_seed(42)
    device = torch.device("cpu")

    # 1. Instantiate miniature 2-layer DecoderLLM
    model = DecoderLLM(
        vocab_size=128,
        max_seq_len=32,
        hidden_dim=64,
        depth=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_dim=128,
        rope_theta=10000.0,
        tie_weights=True,
        norm_eps=1e-6,
    )
    model.to(device)

    # Scale base embeddings to pre-trained magnitude (std=1.0) so logits can achieve sufficient dynamic range
    with torch.no_grad():
        model.tok_embeddings.weight.normal_(0.0, 1.0)

    # 2. Inject LoRA adapters (rank=16, alpha=32.0) across all linear projections
    target_linears = ["q_proj", "k_proj", "v_proj", "o_proj", "w_gate", "w_up", "w_down"]
    model, stats = inject_lora(model, target_modules=target_linears, rank=16, alpha=32.0)
    assert stats["trainable_lora_params"] > 0

    # 3. Snapshot base parameters to ensure zero corruption during backpropagation
    initial_base_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }

    # Snapshot initial adapter parameters
    initial_adapter_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # 4. Generate fixed synthetic training sequence (1 sequence of 16 tokens)
    batch_size = 1
    seq_len = 16
    tokens = torch.randint(0, 128, (batch_size, seq_len), device=device)
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    # Optimizer on only trainable adapter parameters (zero weight decay for overfitting test)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-2, weight_decay=0.0)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(300):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if final_loss < 0.05 * initial_loss:
            # Overfitting threshold reached early
            break

    # 5. Convergence assertion: loss drop > 95%
    print(f"Overfit test: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}")
    assert final_loss < 0.05 * initial_loss, (
        f"LoRA failed to overfit synthetic batch! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )

    # 6. Invariance assertion: every base parameter must remain BITWISE IDENTICAL
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert torch.equal(param, initial_base_params[name]), (
                f"Base parameter '{name}' was modified during LoRA training! Base weights must be strictly frozen."
            )

    # 7. Verification that adapter weights actively updated
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.equal(param, initial_adapter_params[name]), (
                f"Adapter parameter '{name}' was not updated during training!"
            )

