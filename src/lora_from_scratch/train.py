"""
Training Pipeline for Low-Rank Adaptation (LoRA) and Full Fine-Tuning Baselines:
1. Base Checkpoint Loading: Initializes DecoderLLM and loads pre-trained base model weights.
2. Selective Parameter Freezing: Injects LoRALinear layers, keeping base weights frozen.
3. Memory & Efficiency Logging: Displays exact trainable vs frozen parameter counts.
4. Loss Computation: Ignores prompt and pad positions using label -100 cross-entropy.
5. Adapter Checkpointing: Automatically saves best low-rank adapter weights and config metadata.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from modern_decoder_llm import DecoderLLM
from modern_decoder_llm.tokenizer import Tokenizer
from lora_from_scratch.injection import inject_lora
from lora_from_scratch.adapter_io import save_adapter
from lora_from_scratch.dataset import LessonDataset, PirateDataset


def load_base_model(config: Dict[str, Any], device: torch.device) -> DecoderLLM:
    """Instantiates the DecoderLLM architecture and optionally loads base weights."""
    model_cfg = config["base_model"]
    model = DecoderLLM(
        vocab_size=model_cfg.get("vocab_size", 4096),
        max_seq_len=model_cfg.get("max_seq_len", 256),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        depth=model_cfg.get("depth", 6),
        n_heads=model_cfg.get("n_heads", 8),
        n_kv_heads=model_cfg.get("n_kv_heads", 2),
        head_dim=model_cfg.get("head_dim", 32),
        ffn_dim=model_cfg.get("ffn_dim", 768),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        tie_weights=True,
        norm_eps=model_cfg.get("norm_eps", 1e-6),
    )

    ckpt_path = model_cfg.get("checkpoint_path")
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"Loading pre-trained base weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Note: No pre-trained base checkpoint found. Operating with freshly initialized weights.")

    model.to(device)
    return model


def get_dataset(data_path: str, tokenizer: Tokenizer, max_seq_len: int = 256):
    """Instantiates the appropriate dataset based on file content/name."""
    if "lesson" in data_path:
        return LessonDataset(data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
    elif "pirate" in data_path:
        return PirateDataset(data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
    else:
        # Fallback to pirate/causal LM dataset for general text files
        return PirateDataset(data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_eval_steps: Optional[int] = None,
) -> float:
    """Evaluates cross-entropy loss over validation batches, ignoring label -100."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i, (input_ids, labels) in enumerate(val_loader):
        if max_eval_steps is not None and i >= max_eval_steps:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Causal shift
        x = input_ids[:, :-1]
        y = labels[:, 1:]

        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)


def train_adapter(
    config_path: str,
    output_dir: Optional[str] = None,
    mode: str = "lora",
    rank_override: Optional[int] = None,
    alpha_override: Optional[float] = None,
    seed_override: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Executes the training workflow for LoRA adaptation or full fine-tuning.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = seed_override if seed_override is not None else config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n============================================================")
    print(f"Starting LoRA Training Pipeline [Mode: {mode.upper()}] on Device: {device}")
    print(f"Seed: {seed}")
    print(f"============================================================")

    # 1. Base Model & Tokenizer
    model = load_base_model(config, device)
    tok_path = config["base_model"].get("tokenizer_path", "../modern_decoder_llm/checkpoints/tokenizer.json")
    if not os.path.isfile(tok_path):
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    tokenizer = Tokenizer.load(tok_path)

    # 2. Adaptation Mode Setup
    lora_cfg = config.get("lora", {})
    rank = rank_override if rank_override is not None else lora_cfg.get("rank", 8)
    alpha = alpha_override if alpha_override is not None else lora_cfg.get("alpha", 16.0)
    dropout = lora_cfg.get("dropout", 0.0)
    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])

    if mode == "lora":
        model, stats = inject_lora(
            model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        print(f"Injected LoRA (rank={rank}, alpha={alpha}) into {len(stats['injected_layers'])} layers.")
        print(f"Trainable LoRA Parameters: {stats['trainable_lora_params']:,} ({stats['trainable_pct']:.3f}%)")
        print(f"Frozen Base Parameters:    {stats['base_params']:,}")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        # Full Fine-Tuning Baseline
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = list(model.parameters())
        total_p = sum(p.numel() for p in trainable_params)
        print(f"Full Fine-Tuning: All {total_p:,} parameters trainable.")

    # 3. Datasets & DataLoaders
    train_cfg = config["training"]
    max_seq_len = config["base_model"].get("max_seq_len", 256)
    train_dataset = get_dataset(train_cfg["train_data_path"], tokenizer, max_seq_len=max_seq_len)
    val_dataset = get_dataset(train_cfg["val_data_path"], tokenizer, max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=False,
    )

    # 4. Optimizer & Cosine Warmup Schedule
    lr = float(train_cfg.get("lr", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    max_steps = train_cfg.get("max_steps", 500)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 2)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    eval_interval = train_cfg.get("eval_interval", 50)
    eval_steps = train_cfg.get("eval_steps", 25)

    warmup_steps = max(10, int(0.1 * max_steps))

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr * 0.1 + 0.5 * lr * 0.9 * (1.0 + math.cos(math.pi * progress))

    # 5. Training Loop
    if output_dir is None:
        task_name = "lesson" if "lesson" in train_cfg["train_data_path"] else "pirate"
        output_dir = f"outputs/{task_name}_r{rank}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "steps": []}
    step = 0
    best_val_loss = float("inf")
    pbar = tqdm(total=max_steps, desc="Adapting Model")

    data_iter = iter(train_loader)
    optimizer.zero_grad()

    while step < max_steps:
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            x = input_ids[:, :-1]
            y = labels[:, 1:]

            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            accum_loss += scaled_loss.item()

        # Optimizer step with gradient clipping
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{current_lr:.2e}"})

        # Periodic Evaluation
        if step % eval_interval == 0 or step == max_steps:
            val_loss = evaluate_loss(model, val_loader, device, max_eval_steps=eval_steps)
            history["steps"].append(step)
            history["train_loss"].append(accum_loss)
            history["val_loss"].append(val_loss)
            print(f"\n[Step {step}/{max_steps}] Train Loss: {accum_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if mode == "lora":
                    best_save_dir = out_path / "best_adapter"
                    save_adapter(
                        model,
                        best_save_dir,
                        config_overrides={
                            "task": train_cfg["train_data_path"],
                            "rank": rank,
                            "alpha": alpha,
                            "val_loss": val_loss,
                            "step": step,
                        },
                    )
                else:
                    torch.save(model.state_dict(), out_path / "best_full_model.pt")

    pbar.close()

    # Save training history
    with open(out_path / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best Val Loss: {best_val_loss:.4f}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA Adapter on DecoderLLM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--mode", type=str, choices=["lora", "full"], default="lora", help="Training mode")
    parser.add_argument("--rank", type=int, default=None, help="Override rank")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha")
    args = parser.parse_args()

    train_adapter(
        config_path=args.config,
        output_dir=args.output_dir,
        mode=args.mode,
        rank_override=args.rank,
        alpha_override=args.alpha,
    )

