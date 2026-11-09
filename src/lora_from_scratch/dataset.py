"""
PyTorch Datasets for LoRA Task Adaptation:
1. LessonDataset: Supervised instruction dataset for lesson-sentence generation.
   - Masks the prompt with label -100 so cross-entropy loss is computed exclusively on the target lesson.
2. PirateDataset: Causal language modeling dataset for synthetic pirate dialect adaptation.
   - Standard next-token prediction across the entire story, ignoring only pad tokens with label -100.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import torch
from torch.utils.data import Dataset

from modern_decoder_llm.tokenizer import Tokenizer


class LessonDataset(Dataset):
    """
    Dataset for supervised lesson-continuation generation.
    Masks the prompt tokens with -100 so gradient updates are derived only from the lesson target.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, str]] = []

        file_path = Path(data_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Lesson dataset not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        target = sample["target"]

        prompt_ids = self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target) + [self.tokenizer.eos_id]

        total_len = len(prompt_ids) + len(target_ids)
        if total_len > self.max_seq_len:
            # Truncate prompt from the left so the prompt suffix (e.g. '\nLesson: ') and target are preserved
            overflow = total_len - self.max_seq_len
            prompt_ids = prompt_ids[overflow:]

        input_ids = prompt_ids + target_ids
        # Prompt positions are masked with -100; target positions are supervised
        labels = [-100] * len(prompt_ids) + target_ids

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class PirateDataset(Dataset):
    """
    Dataset for causal language modeling on synthetic pirate dialect stories.
    Computes loss across all story tokens; only padding is masked with -100.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[str] = []

        file_path = Path(data_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Pirate dataset not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for item in raw_data:
                if isinstance(item, dict):
                    self.samples.append(item.get("text", item.get("story", "")))
                elif isinstance(item, str):
                    self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_id]

        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]

        input_ids = list(tokens)
        labels = list(tokens)

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

