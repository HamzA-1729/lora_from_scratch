"""
Adapter Serialization and Deserialization Engine:
1. Isolated Persistence: Saves and loads ONLY low-rank adapter matrices (A, B) and hyperparameter metadata.
2. Base Parameter Shielding: Guarantees base model weights are never duplicated into adapter checkpoints.
3. Strict Architecture Validation: Validates layer names, input/output dimensions, and intrinsic ranks on load.
4. Multi-Tenant Serving Loader: Directly constructs the {layer_name: (A, B, scaling)} context dictionary
   required for zero-mutation multi-adapter request multiplexing.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from lora_from_scratch.lora_layer import LoRALinear


def save_adapter(
    model: nn.Module,
    save_directory: Union[str, Path],
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Extracts and persists only trainable LoRA weights and metadata.
    
    Args:
        model: Adapted model containing LoRALinear layers.
        save_directory: Target folder to store adapter_model.pt and adapter_config.json.
        config_overrides: Optional dictionary of additional metadata (e.g., base model hash, training loss).
        
    Returns:
        Path to the saved directory.
    """
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    adapter_weights: Dict[str, torch.Tensor] = {}
    layer_shapes: Dict[str, Dict[str, Any]] = {}
    ranks = set()
    alphas = set()

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            layer_name = module.layer_name or name
            adapter_weights[f"{layer_name}.lora_A.weight"] = module.lora_A.weight.detach().cpu().clone()
            adapter_weights[f"{layer_name}.lora_B.weight"] = module.lora_B.weight.detach().cpu().clone()
            layer_shapes[layer_name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "rank": module.rank,
                "alpha": module.alpha,
                "scaling": module.scaling,
            }
            ranks.add(module.rank)
            alphas.add(module.alpha)

    if not adapter_weights:
        raise ValueError("No LoRALinear layers were found in the provided model to save!")

    # Check rank consistency across layers (standard across common adapters)
    default_rank = next(iter(ranks)) if len(ranks) == 1 else list(ranks)
    default_alpha = next(iter(alphas)) if len(alphas) == 1 else list(alphas)

    adapter_config: Dict[str, Any] = {
        "format_version": "1.0",
        "rank": default_rank,
        "alpha": default_alpha,
        "total_adapter_tensors": len(adapter_weights),
        "layer_names": list(layer_shapes.keys()),
        "layer_shapes": layer_shapes,
    }

    if config_overrides:
        adapter_config.update(config_overrides)

    # 1. Save hyperparameter metadata JSON
    config_file = save_path / "adapter_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    # 2. Save pure adapter weights
    weights_file = save_path / "adapter_model.pt"
    torch.save(adapter_weights, weights_file)

    return save_path


def load_adapter(
    model: nn.Module,
    load_directory: Union[str, Path],
    strict: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Loads saved LoRA weights into an existing model with strict dimensional validation.
    
    Args:
        model: Model containing LoRALinear layers to update.
        load_directory: Directory containing adapter_config.json and adapter_model.pt.
        strict: If True, asserts all model LoRALinear layers are matched in the checkpoint.
        device: Device to map weights onto (defaults to existing module weight device).
        
    Returns:
        The loaded adapter configuration dictionary.
    """
    load_path = Path(load_directory)
    config_file = load_path / "adapter_config.json"
    weights_file = load_path / "adapter_model.pt"

    if not config_file.is_file():
        raise FileNotFoundError(f"Missing adapter config file: {config_file}")
    if not weights_file.is_file():
        raise FileNotFoundError(f"Missing adapter weights file: {weights_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    weights: Dict[str, torch.Tensor] = torch.load(weights_file, map_location="cpu", weights_only=True)
    layer_shapes = config.get("layer_shapes", {})

    # Map name -> LoRALinear module
    model_lora_modules: Dict[str, LoRALinear] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key = module.layer_name or name
            model_lora_modules[key] = module

    # Validate and copy weights
    matched_keys = set()
    for layer_name, expected_meta in layer_shapes.items():
        if layer_name not in model_lora_modules:
            raise ValueError(
                f"Architecture mismatch: Checkpoint requires LoRALinear layer '{layer_name}', "
                f"which does not exist in target model."
            )

        module = model_lora_modules[layer_name]

        # Check dimensions and rank
        if module.rank != expected_meta["rank"]:
            raise ValueError(
                f"Rank mismatch for '{layer_name}': model has rank {module.rank}, "
                f"checkpoint has rank {expected_meta['rank']}."
            )
        if module.in_features != expected_meta["in_features"] or module.out_features != expected_meta["out_features"]:
            raise ValueError(
                f"Feature dimension mismatch for '{layer_name}': model has ({module.in_features}, {module.out_features}), "
                f"checkpoint has ({expected_meta['in_features']}, {expected_meta['out_features']})."
            )

        key_a = f"{layer_name}.lora_A.weight"
        key_b = f"{layer_name}.lora_B.weight"

        if key_a not in weights or key_b not in weights:
            raise KeyError(f"Weight tensors for '{layer_name}' ({key_a}, {key_b}) missing from checkpoint.")

        target_device = device if device is not None else module.lora_A.weight.device
        with torch.no_grad():
            module.lora_A.weight.copy_(weights[key_a].to(target_device))
            module.lora_B.weight.copy_(weights[key_b].to(target_device))
            if "alpha" in expected_meta:
                module.alpha = float(expected_meta["alpha"])
                module.scaling = module.alpha / float(module.rank)

        matched_keys.add(layer_name)

    if strict and len(matched_keys) != len(model_lora_modules):
        unmatched = set(model_lora_modules.keys()) - matched_keys
        raise ValueError(f"Strict load failed: model has unpopulated LoRA layers: {unmatched}")

    return config


def load_adapter_for_context(
    load_directory: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Loads adapter weights directly into memory format suitable for active_adapter_context.
    
    Returns:
        Dict mapping layer_name -> (A_weight, B_weight, scaling_factor).
        This dictionary can be set directly via active_adapter_context.set(...) during inference.
    """
    load_path = Path(load_directory)
    config_file = load_path / "adapter_config.json"
    weights_file = load_path / "adapter_model.pt"

    if not config_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError(f"Valid adapter checkpoint not found in {load_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    weights: Dict[str, torch.Tensor] = torch.load(weights_file, map_location="cpu", weights_only=True)
    layer_shapes = config.get("layer_shapes", {})

    target_device = device if device is not None else torch.device("cpu")
    context_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {}

    for layer_name, meta in layer_shapes.items():
        key_a = f"{layer_name}.lora_A.weight"
        key_b = f"{layer_name}.lora_B.weight"
        if key_a in weights and key_b in weights:
            weight_a = weights[key_a].to(target_device)
            weight_b = weights[key_b].to(target_device)
            scaling = float(meta.get("scaling", meta.get("alpha", 16.0) / meta.get("rank", 8)))
            context_dict[layer_name] = (weight_a, weight_b, scaling)

    return context_dict

