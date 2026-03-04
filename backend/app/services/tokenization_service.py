"""Tokenization service — tokenizer management and dataset statistics."""

import json
from pathlib import Path
from collections import Counter

from app.config import settings


def _tokenizer_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "tokenizer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_tokenizer(model_name: str):
    """Load a HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ValueError("transformers not installed")

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer for {model_name}: {e}")


def analyze_dataset_tokens(
    dataset_path: str,
    model_name: str,
    max_seq_length: int = 2048,
    text_field: str = "text",
    question_field: str = "question",
    answer_field: str = "answer",
) -> dict:
    """Analyze token statistics for a JSONL dataset."""
    tokenizer = load_tokenizer(model_name)
    lengths = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            # Extract text from various formats
            if text_field in entry:
                text = entry[text_field]
            elif question_field in entry and answer_field in entry:
                text = f"Question: {entry[question_field]}\nAnswer: {entry[answer_field]}"
            else:
                text = json.dumps(entry)

            tokens = tokenizer.encode(text)
            lengths.append(len(tokens))

    if not lengths:
        return {"error": "No entries found"}

    total = len(lengths)
    avg_len = sum(lengths) / total
    max_len = max(lengths)
    min_len = min(lengths)
    truncated = sum(1 for l in lengths if l > max_seq_length)

    # Length distribution buckets
    buckets = {"0-256": 0, "256-512": 0, "512-1024": 0, "1024-2048": 0, "2048+": 0}
    for l in lengths:
        if l <= 256:
            buckets["0-256"] += 1
        elif l <= 512:
            buckets["256-512"] += 1
        elif l <= 1024:
            buckets["512-1024"] += 1
        elif l <= 2048:
            buckets["1024-2048"] += 1
        else:
            buckets["2048+"] += 1

    return {
        "total_entries": total,
        "avg_length": round(avg_len, 1),
        "max_length": max_len,
        "min_length": min_len,
        "median_length": sorted(lengths)[total // 2],
        "truncation_count": truncated,
        "truncation_percent": round(truncated / total * 100, 1),
        "max_seq_length": max_seq_length,
        "vocab_size": tokenizer.vocab_size,
        "length_distribution": buckets,
    }


def get_vocab_sample(model_name: str, sample_size: int = 100) -> dict:
    """Get a sample of the tokenizer vocabulary."""
    tokenizer = load_tokenizer(model_name)
    vocab = tokenizer.get_vocab()
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])

    return {
        "vocab_size": len(vocab),
        "sample": [{"token": t, "id": i} for t, i in sorted_tokens[:sample_size]],
        "special_tokens": {
            "bos": tokenizer.bos_token,
            "eos": tokenizer.eos_token,
            "pad": tokenizer.pad_token,
            "unk": tokenizer.unk_token,
        },
    }
