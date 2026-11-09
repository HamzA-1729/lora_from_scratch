"""
Re-exporting Merge & Unload tests for test_merge.py.
"""

from tests.test_merge_unmerge import (
    test_merge_and_unmerge_reversibility,
    test_merged_logits_equivalence,
    test_merge_unmerge_idempotency,
    test_merge_and_unload,
    test_cannot_train_while_merged,
)

__all__ = [
    "test_merge_and_unmerge_reversibility",
    "test_merged_logits_equivalence",
    "test_merge_unmerge_idempotency",
    "test_merge_and_unload",
    "test_cannot_train_while_merged",
]

