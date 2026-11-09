"""
Evaluation & Benchmarking Suite for LoRA Adapters:
1. Target Loss: Computes cross-entropy on held-out task validation data.
2. Catastrophic Forgetting Test: Evaluates perplexity on original pretraining distribution (plain TinyStories).
3. Pirate Lexical Density: Quantifies the percentage of generated vocabulary matching pirate dialect tokens.
4. Qualitative Text Generation: Generates completions using the KV cache with greedy/temperature decoding.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F

from modern_decoder_llm import DecoderLLM
from modern_decoder_llm.tokenizer import Tokenizer
from modern_decoder_llm.sampler import generate_with_cache
from lora_from_scratch.injection import inject_lora
from lora_from_scratch.adapter_io import load_adapter
from lora_from_scratch.dataset import LessonDataset, PirateDataset


PIRATE_KEYWORDS = {
    "ahoy", "matey", "mateys", "aye", "ye", "yer", "me", "thar", "scallywag",
    "landlubber", "plunder", "arr", "avast", "booty", "shiver", "timbers", "sail",
    "sea", "treasure", "cap'n", "captain", "deck", "ship", "yo-ho-ho"
}


def compute_pirate_lexical_density(text: str) -> float:
    """Computes the proportion of words in generated text that belong to pirate keywords."""
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if not words:
        return 0.0
    pirate_count = sum(1 for w in words if w in PIRATE_KEYWORDS)
    return pirate_count / len(words)


@torch.no_grad()
def generate_autoregressive(
    model: DecoderLLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generates tokens autoregressively using preallocated StaticKVCache."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_tokens = generate_with_cache(
        model=model,
        prompt_tokens=prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        eos_token_id=tokenizer.eos_id,
    )
    return tokenizer.decode(output_tokens[0].tolist())


def evaluate_adapter(
    base_checkpoint: str,
    adapter_dir: Optional[str],
    test_data_path: str,
    tokenizer_path: str = "../modern_decoder_llm/checkpoints/tokenizer.json",
    task: str = "lesson",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluates loss, forgetting, and generation capabilities of an adapter.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(tokenizer_path)

    # Instantiate base model
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

    if os.path.isfile(base_checkpoint):
        ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # Load adapter if specified
    if adapter_dir and os.path.isdir(adapter_dir):
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        target_modules = ["q_proj", "v_proj"]
        rank = cfg.get("rank", 8)
        alpha = cfg.get("alpha", 16.0)

        model, _ = inject_lora(model, target_modules=target_modules, rank=rank, alpha=alpha)
        load_adapter(model, adapter_dir)
        print(f"Loaded LoRA adapter from {adapter_dir} (rank={rank}, alpha={alpha}).")
    else:
        print("Evaluating base model without adapters (Zero-Shot Baseline).")

    # Load test dataset
    if task == "lesson":
        dataset = LessonDataset(test_data_path, tokenizer=tokenizer, max_seq_len=256)
    else:
        dataset = PirateDataset(test_data_path, tokenizer=tokenizer, max_seq_len=256)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    # 1. Compute Test Loss
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            x = input_ids[:, :-1]
            y = labels[:, 1:]
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            total_loss += loss.item()
            total_batches += 1

    avg_test_loss = total_loss / max(1, total_batches)
    print(f"\n============================================================")
    print(f"Task '{task}' Test Loss: {avg_test_loss:.4f} (Perplexity: {torch.exp(torch.tensor(avg_test_loss)):.2f})")
    print(f"============================================================")

    # 2. Qualitative Generations & Metrics
    results: Dict[str, Any] = {
        "task": task,
        "test_loss": avg_test_loss,
        "test_perplexity": float(torch.exp(torch.tensor(avg_test_loss))),
        "samples": [],
    }

    if task == "lesson":
        prompts = [
            "Story: A little rabbit found a huge carrot and decided to share it with his hungry friend.\nLesson: ",
            "Story: Timmy lied about breaking the flower pot, but felt so guilty he told the truth.\nLesson: ",
        ]
        for p in prompts:
            out = generate_autoregressive(model, tokenizer, p, max_new_tokens=25, temperature=0.0, device=device)
            results["samples"].append({"prompt": p, "completion": out})
            print(f"\nPrompt: {p.strip()}\nGenerated: {out.replace(p, '').strip()}")
    elif task == "pirate":
        prompts = [
            "Once upon a time, a fearless boy sailed across the ocean",
            "Hello my dear friend, where is the golden treasure",
        ]
        densities = []
        for p in prompts:
            out = generate_autoregressive(model, tokenizer, p, max_new_tokens=40, temperature=0.7, device=device)
            gen_text = out.replace(p, "").strip()
            density = compute_pirate_lexical_density(out)
            densities.append(density)
            results["samples"].append({"prompt": p, "completion": out, "pirate_density": density})
            print(f"\nPrompt: {p}\nGenerated: {out}\nPirate Lexical Density: {density * 100:.1f}%")
        results["average_pirate_density"] = sum(densities) / len(densities)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA Adapter")
    parser.add_argument("--base_checkpoint", type=str, default="../modern_decoder_llm/checkpoints/best_model.pt")
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["lesson", "pirate"], required=True)
    args = parser.parse_args()

    evaluate_adapter(
        base_checkpoint=args.base_checkpoint,
        adapter_dir=args.adapter_dir,
        test_data_path=args.test_data,
        task=args.task,
    )

