"""
Utility helpers for parameter calculation, seeding, and device management.
"""

import os
import random
from typing import Dict
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Computes total, trainable, and frozen parameter counts for any PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }

