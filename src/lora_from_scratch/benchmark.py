"""
Empirical Inference Latency and Throughput Benchmark:
Evaluates token generation throughput and per-forward-pass latency across 4 serving configurations:
1. Base Model (unadapted reference)
2. LoRA Unmerged (factorized delta computation: h = W_0 x + scaling * B(A(x)))
3. LoRA Merged In-Place (h = (W_0 + delta W) x)
4. LoRA Merged & Unloaded (reconverted to standard nn.Linear)
"""

import time
from typing import Dict, List
import torch
import torch.nn as nn

from modern_decoder_llm import DecoderLLM
from lora_from_scratch.injection import inject_lora, merge_and_unload_model
from lora_from_scratch.lora_layer import LoRALinear


def benchmark_forward_latency(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 64,
    vocab_size: int = 4096,
    num_runs: int = 50,
    warmup_runs: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Measures average forward pass latency in milliseconds."""
    model.eval()
    model.to(device)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / num_runs) * 1000.0
    return avg_latency_ms


def run_latency_comparison() -> Dict[str, float]:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning Inference Latency Benchmark on {device}...")

    # 1. Base Model
    base_model = DecoderLLM(
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
    base_latency = benchmark_forward_latency(base_model, device=device)

    # 2. LoRA Unmerged
    lora_model, _ = inject_lora(base_model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)
    unmerged_latency = benchmark_forward_latency(lora_model, device=device)

    # 3. LoRA Merged In-Place
    for m in lora_model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
    merged_latency = benchmark_forward_latency(lora_model, device=device)

    # 4. LoRA Merged and Unloaded (Back to pure nn.Linear)
    unloaded_model = merge_and_unload_model(lora_model)
    unloaded_latency = benchmark_forward_latency(unloaded_model, device=device)

    results = {
        "Base Model": round(base_latency, 2),
        "LoRA Unmerged (Dynamic)": round(unmerged_latency, 2),
        "LoRA Merged In-Place": round(merged_latency, 2),
        "LoRA Merged & Unloaded": round(unloaded_latency, 2),
    }

    print("\n" + "=" * 65)
    print("INFERENCE LATENCY COMPARISON (Batch=1, SeqLen=64)")
    print("=" * 65)
    print(f"| Serving Mode | Mean Latency (ms) | Overhead vs Base |")
    print(f"| :--- | :---: | :---: |")
    for mode, lat in results.items():
        overhead = f"{((lat - base_latency) / base_latency) * 100:+.1f}%"
        print(f"| {mode:<26} | {lat:>6.2f} ms | {overhead:>8} |")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_latency_comparison()

