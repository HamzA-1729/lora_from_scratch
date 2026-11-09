"""
Unit Tests for Data Preparation & Text Transformations:
1. Whole-sentence lesson extraction and complete excision from prompt.
2. Pirate dialect word boundary (\\b) and case preservation.
3. LessonDataset prompt label masking with -100 and causal shift parity.
4. PirateDataset next-token causal modeling labels.
"""

import json
from pathlib import Path
import pytest
import torch

from lora_from_scratch.prepare_data import extract_lesson_pair, rewrite_to_pirate
from lora_from_scratch.dataset import LessonDataset, PirateDataset
from modern_decoder_llm.tokenizer import Tokenizer


def test_extract_lesson_pair_whole_sentence_excision():
    story = (
        "Timmy went into the dark forest looking for his lost ball. "
        "He searched behind every tree until he finally found it. "
        "From that day on, he learned that patience brings great rewards."
    )
    pair = extract_lesson_pair(story)
    assert pair is not None
    prompt, target = pair

    # Expected prompt has the body without the last sentence
    assert "From that day on" not in prompt
    assert prompt.startswith("Story: Timmy went into the dark forest")
    assert prompt.endswith("\nLesson: ")

    # Target is the entire last sentence
    assert target == "From that day on, he learned that patience brings great rewards."


def test_extract_lesson_pair_non_matching():
    story = "Once upon a time there was a playful kitten. It chased a red yarn ball all day long."
    pair = extract_lesson_pair(story)
    assert pair is None


def test_rewrite_to_pirate_word_boundaries_and_case():
    text = "Hello! You are my best friend. Is there any food? This is wonderful."
    rewritten = rewrite_to_pirate(text)

    # Word boundary checks: "is" inside "This" must NOT be modified
    assert "This" in rewritten
    assert "Thbe" not in rewritten

    # Whole word transformations
    assert "Ahoy!" in rewritten
    assert "Ye be me best matey." in rewritten
    assert "Be thar any food?" in rewritten
    assert rewritten.endswith("Arr!")


@pytest.fixture
def dummy_tokenizer() -> Tokenizer:
    tok = Tokenizer(vocab_size=512)
    tok.train(["Hello world! Story: Once upon a time. Lesson: Sharing is caring.", "Ahoy matey!"])
    return tok


def test_lesson_dataset_prompt_masking_and_causal_shift(dummy_tokenizer: Tokenizer, tmp_path: Path):
    data_file = tmp_path / "test_lesson.json"
    sample_data = [
        {
            "prompt": "Story: A girl had two apples.\nLesson: ",
            "target": "Sharing makes people happy.",
        }
    ]
    with open(data_file, "w") as f:
        json.dump(sample_data, f)

    dataset = LessonDataset(data_file, tokenizer=dummy_tokenizer, max_seq_len=64)
    assert len(dataset) == 1

    input_ids, labels = dataset[0]

    # Verify input_ids and labels shapes
    assert input_ids.shape == (64,)
    assert labels.shape == (64,)

    prompt_tokens = dummy_tokenizer.encode(sample_data[0]["prompt"])
    target_tokens = dummy_tokenizer.encode(sample_data[0]["target"]) + [dummy_tokenizer.eos_id]
    prompt_len = len(prompt_tokens)
    target_len = len(target_tokens)

    # 1. Prompt tokens must have label -100
    for i in range(prompt_len):
        assert labels[i].item() == -100

    # 2. Target tokens must match target_tokens
    for j in range(target_len):
        assert labels[prompt_len + j].item() == target_tokens[j]

    # 3. Padding positions must have label -100
    for k in range(prompt_len + target_len, 64):
        assert labels[k].item() == -100
        assert input_ids[k].item() == dummy_tokenizer.pad_id

    # 4. In causal LM shift: x = input_ids[:-1], y = labels[1:]
    x = input_ids[:-1]
    y = labels[1:]
    # At position prompt_len - 1 in x (the end of the prompt, e.g. "Lesson: "):
    # y[prompt_len - 1] is labels[prompt_len], which is target_tokens[0]!
    assert y[prompt_len - 1].item() == target_tokens[0]
    # At earlier positions, y[i] is -100
    for i in range(prompt_len - 1):
        assert y[i].item() == -100


def test_pirate_dataset_causal_labels(dummy_tokenizer: Tokenizer, tmp_path: Path):
    data_file = tmp_path / "test_pirate.json"
    sample_data = [{"text": "Ahoy matey! Ye be welcome."}]
    with open(data_file, "w") as f:
        json.dump(sample_data, f)

    dataset = PirateDataset(data_file, tokenizer=dummy_tokenizer, max_seq_len=32)
    assert len(dataset) == 1

    input_ids, labels = dataset[0]
    assert input_ids.shape == (32,)
    assert labels.shape == (32,)

    tokens = dummy_tokenizer.encode("Ahoy matey! Ye be welcome.") + [dummy_tokenizer.eos_id]
    num_tokens = len(tokens)

    # Non-pad tokens are supervised
    for i in range(num_tokens):
        assert input_ids[i].item() == tokens[i]
        assert labels[i].item() == tokens[i]

    # Pad tokens have label -100
    for j in range(num_tokens, 32):
        assert labels[j].item() == -100
        assert input_ids[j].item() == dummy_tokenizer.pad_id
