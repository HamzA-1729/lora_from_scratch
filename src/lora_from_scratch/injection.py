"""
Recursive Model Injection Engine for LoRA:
1. Module Traversal: Recursively traverses any PyTorch nn.Module hierarchy.
2. Parameter Freezing: Freezes all base model parameters prior to injection.
3. Leaf Replacement: Replaces matching leaf nn.Linear projections with LoRALinear.
4. Tied-Weight Protection: Explicitly protects and excludes lm_head and tok_embeddings.
5. Double-Wrapping Prevention: Rejects subsequent injections on already-wrapped models.
6. merge_and_unload_model: Recursively converts all LoRALinear modules back into pure nn.Linear.
"""

from typing import Any, Dict, List, Set, Tuple
import torch.nn as nn

from lora_from_scratch.lora_layer import LoRALinear

# Reserved modules that must never be wrapped with LoRA (tied weights, embeddings)
PROTECTED_MODULE_NAMES: Set[str] = {"lm_head", "tok_embeddings"}


def _is_already_injected(model: nn.Module) -> bool:
    """Checks if the model already contains any LoRALinear layers."""
    return any(isinstance(m, LoRALinear) for m in model.modules())


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Recursively injects LoRALinear layers into model matching target_modules.
    
    Args:
        model: PyTorch model (e.g., DecoderLLM). Base checkpoint should be loaded first.
        target_modules: List of leaf Linear names to wrap (e.g., ["q_proj", "v_proj"]).
        rank: Intrinsic rank r for the adapter matrices.
        alpha: LoRA scaling numerator.
        dropout: LoRA input dropout probability.
        
    Returns:
        model: The adapted model with frozen base weights and trainable LoRA adapters.
        stats: Dictionary containing parameter counts and injected layer names.
    """
    if _is_already_injected(model):
        raise RuntimeError("Model is already injected with LoRA layers! Refusing to double-wrap.")

    # 1. Freeze all existing base model parameters
    for param in model.parameters():
        param.requires_grad = False

    target_set = set(target_modules)
    injected_names: List[str] = []

    def _recursive_replace(module: nn.Module, parent_prefix: str = "") -> None:
        for child_name, child_module in list(module.named_children()):
            full_name = f"{parent_prefix}.{child_name}" if parent_prefix else child_name

            # Skip explicitly protected modules (e.g., tied lm_head, embeddings)
            if child_name in PROTECTED_MODULE_NAMES:
                continue

            # If child is a target leaf nn.Linear, replace it with LoRALinear
            if isinstance(child_module, nn.Linear) and child_name in target_set:
                lora_layer = LoRALinear(
                    base_layer=child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    layer_name=full_name,
                )
                setattr(module, child_name, lora_layer)
                injected_names.append(full_name)
            else:
                # Recurse down the hierarchy
                _recursive_replace(child_module, parent_prefix=full_name)

    _recursive_replace(model)

    if not injected_names:
        raise ValueError(f"No target linear layers matching {target_modules} were found in model!")

    # Compute parameter statistics
    total_base_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (total_trainable_params / (total_base_params + total_trainable_params)) * 100.0

    stats = {
        "base_params": total_base_params,
        "trainable_params": total_trainable_params,
        "trainable_lora_params": total_trainable_params,
        "trainable_pct": trainable_pct,
        "injected_layers": injected_names,
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
    }

    return model, stats


def merge_and_unload_model(model: nn.Module) -> nn.Module:
    """
    Recursively finds all LoRALinear layers in model and replaces them
    with clean, standard nn.Linear layers containing the merged weights.
    Provides true zero-overhead inference deployment.
    """
    def _recursive_unload(module: nn.Module) -> None:
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, LoRALinear):
                # Convert LoRALinear back to standard nn.Linear with folded weights
                plain_linear = child_module.merge_and_unload()
                setattr(module, child_name, plain_linear)
            else:
                _recursive_unload(child_module)

    _recursive_unload(model)
    return model

