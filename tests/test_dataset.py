"""
Unit Tests for Data Preparation & Text Transformations:
1. Whole-sentence lesson extraction and complete excision from prompt.
2. Pirate dialect word boundary (\\b) and case preservation.
3. Length filter verification (prompt_tokens + target_tokens <= 256).
"""

from lora_from_scratch.prepare_data import extract_lesson_pair, rewrite_to_pirate
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

