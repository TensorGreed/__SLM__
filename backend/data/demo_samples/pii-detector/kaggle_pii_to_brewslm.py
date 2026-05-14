#!/usr/bin/env python3
"""Convert the Kaggle "PII Detection Removal from Educational Data" (2024)
training file into BrewSLM's span-extraction JSONL.

Kaggle ships each essay as:

    {
      "document": 7,
      "full_text": "Hello, my name is John ...",
      "tokens": ["Hello", ",", "my", "name", "is", "John", ...],
      "trailing_whitespace": [false, true, true, ...],
      "labels": ["O", "O", "O", "O", "O", "B-NAME_STUDENT", ...]
    }

BrewSLM's StructuredExtractionHandler (span_set mode) expects:

    {
      "key": "kaggle-7",
      "text": "Hello, my name is John ...",
      "entities_json": '[{"type":"person_name","start":18,"end":22,"text":"John"}, ...]'
    }

This script walks each essay's BIO tags, merges consecutive B-X / I-X
runs into single spans, reconstructs character offsets (preferring
``full_text`` index when present; falling back to token+whitespace
join when not), and writes one JSONL line per essay.

Usage:

    python scripts/kaggle_pii_to_brewslm.py \\
        --input train.json \\
        --out backend/data/imports/kaggle-pii.jsonl

Then import into a project:

    brewslm dataset import \\
        --jsonl backend/data/imports/kaggle-pii.jsonl \\
        --project-slug pii-detector \\
        --adapter-preset structured-extraction

Stdlib-only. No deps beyond Python 3.10+.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Kaggle's tag vocabulary (taken from the competition's data dictionary)
# mapped onto the PII demo's entity types. Unknown Kaggle tags fall
# through as-is (lowercased) — useful if Kaggle adds new categories.
KAGGLE_TO_BREWSLM_TYPE: dict[str, str] = {
    "NAME_STUDENT": "person_name",
    "EMAIL": "email",
    "USERNAME": "person_name",   # closest BrewSLM concept
    "ID_NUM": "ssn",             # Kaggle's catch-all numeric ID; closest mapping
    "PHONE_NUM": "phone",
    "URL_PERSONAL": "url",
    "STREET_ADDRESS": "street_address",
}


def _reconstruct_offsets_from_tokens(
    tokens: list[str], trailing_whitespace: list[bool]
) -> tuple[str, list[tuple[int, int]]]:
    """Build (text, per_token_(start, end)) from Kaggle's tokens +
    whitespace flags. Used when ``full_text`` is absent or unreliable."""

    buf: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for tok, trail in zip(tokens, trailing_whitespace):
        start = cursor
        buf.append(tok)
        cursor += len(tok)
        end = cursor
        spans.append((start, end))
        if trail:
            buf.append(" ")
            cursor += 1
    return "".join(buf), spans


def _align_tokens_to_full_text(
    full_text: str, tokens: list[str]
) -> list[tuple[int, int]] | None:
    """Walk ``full_text`` left-to-right, matching each token. Returns
    per-token (start, end) char ranges, or None when alignment drifts
    (rare but happens on non-ASCII / normalized tokens).
    """

    spans: list[tuple[int, int]] = []
    cursor = 0
    n = len(full_text)
    for tok in tokens:
        if not tok:
            spans.append((cursor, cursor))
            continue
        # Skip any whitespace ahead of the token.
        while cursor < n and full_text[cursor].isspace():
            cursor += 1
        if cursor + len(tok) > n or full_text[cursor : cursor + len(tok)] != tok:
            return None
        start = cursor
        cursor += len(tok)
        spans.append((start, cursor))
    return spans


def _bio_runs(labels: list[str]) -> list[tuple[int, int, str]]:
    """Walk a BIO tag list and emit (token_start_idx, token_end_idx,
    entity_type) tuples — half-open ranges in token index space.

    ``B-X`` opens a new run; ``I-X`` extends the current run when the
    type matches; ``O`` (or a type switch) closes it.
    """

    runs: list[tuple[int, int, str]] = []
    current_type: str | None = None
    current_start: int | None = None
    for idx, raw in enumerate(labels):
        tag = (raw or "O").strip()
        if tag == "O":
            if current_type is not None and current_start is not None:
                runs.append((current_start, idx, current_type))
            current_type = None
            current_start = None
            continue
        prefix, _, ent_type = tag.partition("-")
        if not ent_type:
            ent_type = prefix
            prefix = "B"
        if prefix == "B" or current_type != ent_type:
            # Close any open run before opening a new one.
            if current_type is not None and current_start is not None:
                runs.append((current_start, idx, current_type))
            current_type = ent_type
            current_start = idx
    if current_type is not None and current_start is not None:
        runs.append((current_start, len(labels), current_type))
    return runs


def _map_entity_type(kaggle_type: str) -> str:
    return KAGGLE_TO_BREWSLM_TYPE.get(kaggle_type, kaggle_type.lower())


def convert_essay(essay: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one Kaggle essay record into a BrewSLM JSONL row.
    Returns None for malformed inputs so the caller can skip them.
    """

    doc_id = essay.get("document")
    tokens = essay.get("tokens")
    labels = essay.get("labels")
    trailing = essay.get("trailing_whitespace")
    full_text = essay.get("full_text")

    if not isinstance(tokens, list) or not isinstance(labels, list):
        return None
    if len(tokens) != len(labels):
        return None
    if not isinstance(trailing, list) or len(trailing) != len(tokens):
        trailing = [True] * len(tokens)

    text: str
    token_spans: list[tuple[int, int]] | None = None
    if isinstance(full_text, str) and full_text:
        token_spans = _align_tokens_to_full_text(full_text, tokens)
        text = full_text
    if token_spans is None:
        # Either no full_text or alignment drifted — reconstruct.
        text, token_spans = _reconstruct_offsets_from_tokens(tokens, trailing)

    entities: list[dict[str, Any]] = []
    for tok_start_idx, tok_end_idx, kaggle_type in _bio_runs(labels):
        if tok_end_idx <= tok_start_idx:
            continue
        char_start = token_spans[tok_start_idx][0]
        char_end = token_spans[tok_end_idx - 1][1]
        span_text = text[char_start:char_end]
        if not span_text.strip():
            continue
        entities.append(
            {
                "type": _map_entity_type(kaggle_type),
                "start": char_start,
                "end": char_end,
                "text": span_text,
            }
        )

    return {
        "key": f"kaggle-{doc_id}" if doc_id is not None else None,
        "text": text,
        # Stringified so the row drops into the demo's CSV / JSONL
        # importer without further escaping; the bundle's
        # `pii_records.csv` uses the same column name.
        "entities_json": json.dumps(
            {"entities": entities}, ensure_ascii=False
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle PII competition train.json into "
        "BrewSLM span-extraction JSONL.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to Kaggle's train.json (or any same-shape JSON array).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to write the BrewSLM-shaped JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N essays (0 = no limit). Useful for testing the "
        "converter against a small sample before running the full set.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    with args.input.open("r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"failed to parse {args.input}: {exc}", file=sys.stderr)
            return 2

    if not isinstance(payload, list):
        print(
            f"expected a JSON array of essay objects in {args.input}, got "
            f"{type(payload).__name__}",
            file=sys.stderr,
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.out.open("w", encoding="utf-8") as out_fh:
        for essay in payload:
            if args.limit and written >= args.limit:
                break
            if not isinstance(essay, dict):
                skipped += 1
                continue
            row = convert_essay(essay)
            if row is None:
                skipped += 1
                continue
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"wrote {written} rows to {args.out} "
        f"({skipped} essays skipped due to shape mismatches)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
