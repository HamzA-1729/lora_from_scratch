"""
Re-exporting injection engine for backwards-compatibility.
"""

from lora_from_scratch.injection import inject_lora, merge_and_unload_model

__all__ = ["inject_lora", "merge_and_unload_model"]

