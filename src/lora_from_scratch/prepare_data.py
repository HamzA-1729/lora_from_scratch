"""
Data Preparation & Step 0 Feasibility Audit:
1. Lesson-Sentence Extraction:
   - Evaluates the whole last sentence of each story.
   - Anchors regex triggers to filter lesson-containing stories:
     r"^(?:The moral|Moral:|From that day|He learned|She learned|They learned|It is important|Always remember|Never forget|He realized|She realized|They realized|He understood|She understood|The lesson|That is why)\b"
   - Completely excises the last sentence from the prompt:
     prompt = "Story: " + story_without_last_sentence + "\\nLesson: "
     target = last_sentence + "<|endoftext|>"
   - Filters out any sample where len(prompt_tokens) + len(target_tokens) > 256.
   - Streams until target quotas are satisfied: 5,000 train, 500 val, 500 test.
2. Pirate Dialect Transformation:
   - Word-boundary regex with case preservation: you->ye, my->me, friend->matey, etc.
   - Appends ", Arr!" to concluding sentences.
   - Streams until 5,000 train and 1,000 val causal LM samples are generated (tokens <= 256).
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from modern_decoder_llm.tokenizer import Tokenizer

# Regex to split text into sentences
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")

# Regex anchored at sentence start identifying reflective lesson endings
LESSON_TRIGGER_REGEX = re.compile(
    r"^(?:The moral|Moral:|From that day|He learned|She learned|They learned|It is important|Always remember|Never forget|He realized|She realized|They realized|He understood|She understood|The lesson|That is why)\b",
    re.IGNORECASE,
)

# Word-boundary replacement rules for Pirate dialect transformation
PIRATE_RULES = [
    (re.compile(r"\bYou\b"), "Ye"),
    (re.compile(r"\byou\b"), "ye"),
    (re.compile(r"\bYour\b"), "Yer"),
    (re.compile(r"\byour\b"), "yer"),
    (re.compile(r"\bMy\b"), "Me"),
    (re.compile(r"\bmy\b"), "me"),
    (re.compile(r"\bFriend\b"), "Matey"),
    (re.compile(r"\bfriend\b"), "matey"),
    (re.compile(r"\bFriends\b"), "Hearties"),
    (re.compile(r"\bfriends\b"), "hearties"),
    (re.compile(r"\bHello\b"), "Ahoy"),
    (re.compile(r"\bhello\b"), "ahoy"),
    (re.compile(r"\bIs\b"), "Be"),
    (re.compile(r"\bis\b"), "be"),
    (re.compile(r"\bAre\b"), "Be"),
    (re.compile(r"\bare\b"), "be"),
    (re.compile(r"\bThere\b"), "Thar"),
    (re.compile(r"\bthere\b"), "thar"),
]


def rewrite_to_pirate(text: str) -> str:
    """Applies whole-word, case-preserving transformations to convert text to pirate dialect."""
    rewritten = text
    for pattern, replacement in PIRATE_RULES:
        rewritten = pattern.sub(replacement, rewritten)
    rewritten = rewritten.strip()
    if rewritten and not rewritten.endswith("Arr!"):
        if rewritten[-1] in ".!?":
            rewritten = rewritten[:-1] + ", Arr!"
        else:
            rewritten = rewritten + ", Arr!"
    return rewritten


def extract_lesson_pair(text: str) -> Optional[Tuple[str, str]]:
    """
    Extracts (prompt, target) if the last sentence contains a lesson trigger.
    Excises the last sentence from the story body completely.
    """
    sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text.strip()) if s.strip()]
    if len(sentences) < 2:
        return None

    last_sentence = sentences[-1]
    if LESSON_TRIGGER_REGEX.search(last_sentence):
        body = " ".join(sentences[:-1]).strip()
        prompt = f"Story: {body}\nLesson: "
        target = f"{last_sentence}"
        return prompt, target
    return None


def run_data_preparation(
    data_dir: str = "data",
    tokenizer_path: str = "../modern_decoder_llm/checkpoints/tokenizer.json",
    target_train_lessons: int = 5000,
    target_val_lessons: int = 1000,
    target_train_pirates: int = 5000,
    target_val_pirates: int = 1000,
    max_seq_len: int = 256,
) -> Dict[str, int]:
    """Prepares datasets and reports Step 0 feasibility counts."""
    os.makedirs(data_dir, exist_ok=True)

    # 1. Load tokenizer
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = Tokenizer.load(tokenizer_path)
    else:
        print(f"Warning: Tokenizer not found at {tokenizer_path}. Using base byte-level tokenizer.")
        tokenizer = Tokenizer(vocab_size=4096)

    # 2. Fetch TinyStories from datasets
    from datasets import load_dataset
    print("Fetching TinyStories training set...")
    ds_train = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    print("Fetching TinyStories validation set (to eliminate pretraining leakage)...")
    ds_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    # --- Part A: Lesson Extraction ---
    print("\n--- Auditing & Extracting Lesson-Continuation Pairs ---")
    train_lesson_pairs = []
    pbar_train = tqdm(desc="Extracting Train Lessons", total=target_train_lessons)
    for item in ds_train:
        pair = extract_lesson_pair(item["text"])
        if pair is not None:
            prompt, target = pair
            p_ids = tokenizer.encode(prompt)
            t_ids = tokenizer.encode(target + "<|endoftext|>")
            if len(p_ids) + len(t_ids) <= max_seq_len:
                train_lesson_pairs.append({
                    "prompt": prompt,
                    "target": target,
                    "prompt_tokens": len(p_ids),
                    "target_tokens": len(t_ids),
                })
                pbar_train.update(1)
                if len(train_lesson_pairs) >= target_train_lessons:
                    break
    pbar_train.close()

    val_lesson_pairs = []
    pbar_val = tqdm(desc="Extracting Val Lessons", total=target_val_lessons)
    for item in ds_val:
        pair = extract_lesson_pair(item["text"])
        if pair is not None:
            prompt, target = pair
            p_ids = tokenizer.encode(prompt)
            t_ids = tokenizer.encode(target + "<|endoftext|>")
            if len(p_ids) + len(t_ids) <= max_seq_len:
                val_lesson_pairs.append({
                    "prompt": prompt,
                    "target": target,
                    "prompt_tokens": len(p_ids),
                    "target_tokens": len(t_ids),
                })
                pbar_val.update(1)
                if len(val_lesson_pairs) >= target_val_lessons:
                    break
    pbar_val.close()

    # Split val pairs into validation (500) and test (500)
    val_subset = val_lesson_pairs[:500]
    test_subset = val_lesson_pairs[500:1000]

    # --- Part B: Pirate Dialect Generation ---
    print("\n--- Generating Pirate Dialect Causal LM Dataset ---")
    ds_train_pirate = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    train_pirate_samples = []
    pbar_pirate_train = tqdm(desc="Rewriting Train Pirate", total=target_train_pirates)
    for item in ds_train_pirate:
        pirate_text = rewrite_to_pirate(item["text"])
        tokens = tokenizer.encode(pirate_text + "<|endoftext|>")
        if len(tokens) <= max_seq_len:
            train_pirate_samples.append({
                "text": pirate_text,
                "tokens": len(tokens),
            })
            pbar_pirate_train.update(1)
            if len(train_pirate_samples) >= target_train_pirates:
                break
    pbar_pirate_train.close()

    ds_val_pirate = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_pirate_samples = []
    pbar_pirate_val = tqdm(desc="Rewriting Val Pirate", total=target_val_pirates)
    for item in ds_val_pirate:
        pirate_text = rewrite_to_pirate(item["text"])
        tokens = tokenizer.encode(pirate_text + "<|endoftext|>")
        if len(tokens) <= max_seq_len:
            val_pirate_samples.append({
                "text": pirate_text,
                "tokens": len(tokens),
            })
            pbar_pirate_val.update(1)
            if len(val_pirate_samples) >= target_val_pirates:
                break
    pbar_pirate_val.close()

    # Save to JSON
    train_lesson_file = os.path.join(data_dir, "train_lesson.json")
    val_lesson_file = os.path.join(data_dir, "val_lesson.json")
    test_lesson_file = os.path.join(data_dir, "test_lesson.json")
    train_pirate_file = os.path.join(data_dir, "train_pirate.json")
    val_pirate_file = os.path.join(data_dir, "val_pirate.json")

    with open(train_lesson_file, "w", encoding="utf-8") as f:
        json.dump(train_lesson_pairs, f, indent=2)
    with open(val_lesson_file, "w", encoding="utf-8") as f:
        json.dump(val_subset, f, indent=2)
    with open(test_lesson_file, "w", encoding="utf-8") as f:
        json.dump(test_subset, f, indent=2)
    with open(train_pirate_file, "w", encoding="utf-8") as f:
        json.dump(train_pirate_samples, f, indent=2)
    with open(val_pirate_file, "w", encoding="utf-8") as f:
        json.dump(val_pirate_samples, f, indent=2)

    stats = {
        "train_lesson_pairs": len(train_lesson_pairs),
        "val_lesson_pairs": len(val_subset),
        "test_lesson_pairs": len(test_subset),
        "train_pirate_samples": len(train_pirate_samples),
        "val_pirate_samples": len(val_pirate_samples),
    }

    print("\n" + "=" * 60)
    print("STEP 0 DATA FEASIBILITY AUDIT SUMMARY:")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<25}: {v:,}")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    run_data_preparation()
