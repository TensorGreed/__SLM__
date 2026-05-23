"""POSITIVES_PARAPHRASE playbook for the `span-extraction` recipe.

Paraphrasing span-extraction data is tricky: the new text can't
preserve exact character offsets, so the model has to emit fresh
offsets for the new text. We validate that each declared span
actually appears at the offsets it claims.
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    PlaybookContext,
    SynthMode,
    SynthRow,
    parse_jsonl_lines,
    register_playbook,
    sample_gold_rows,
)


_PROMPT_TEMPLATE = """\
You are generating training data for a span-extraction fine-tuning task.

You are given a small set of (text, spans) examples. For each one, write {paraphrases_per_row} alternative phrasings of the TEXT that contain the SAME set of entity types in slightly different wording.

Rules:
- Each new text MUST contain every entity type from the source row's spans, with the same value (or a clearly equivalent paraphrase of that value).
- Compute fresh `start` and `end` character offsets for the new text. Offsets are 0-indexed, end-exclusive. `text[start:end]` MUST exactly equal the span's `text` field.
- Preserve the entity types (`type` field) exactly.
- Vary surrounding context, sentence structure, formality.
- Do NOT add entity types not present in the source.

Source examples:
{examples_block}

Output exactly {total_count} lines. Each line MUST be a single JSON object:
{{"text": "<paraphrased text>", "spans": [{{"type": "...", "start": N, "end": N, "text": "..."}}]}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines.
"""


class SpanExtractionParaphrasePlaybook:
    recipe_id = "span-extraction"
    mode = SynthMode.POSITIVES_PARAPHRASE

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        n_source = max(1, min(len(gold), max(2, target // 4)))
        examples = sample_gold_rows(gold, count=n_source, seed=0)
        paraphrases_per_row = max(1, target // max(1, len(examples)))

        lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            text = self._extract_text(row)
            spans = self._extract_spans(row)
            lines.append(
                f"{i}. text: {text!r}\n   spans: {json.dumps(spans, ensure_ascii=False)}"
            )

        return _PROMPT_TEMPLATE.format(
            paraphrases_per_row=paraphrases_per_row,
            examples_block="\n".join(lines),
            total_count=paraphrases_per_row * len(examples),
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        gold = ctx.get("gold_rows") or []
        known_types = self._collect_entity_types(gold)

        accepted: list[SynthRow] = []
        for row in parsed_rows:
            text = row.get("text")
            spans = row.get("spans")
            if not isinstance(text, str) or not isinstance(spans, list):
                continue
            text = text.strip()
            if not text:
                continue
            confidence = 1.0
            valid_spans: list[dict[str, Any]] = []
            for span in spans:
                if not isinstance(span, dict):
                    confidence *= 0.7
                    continue
                start = span.get("start")
                end = span.get("end")
                span_text = span.get("text")
                span_type = span.get("type")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not isinstance(span_text, str)
                    or not isinstance(span_type, str)
                    or start < 0
                    or end <= start
                    or end > len(text)
                ):
                    confidence *= 0.5
                    continue
                # Critical: the declared text MUST be what's at the offsets.
                # If the model lied about offsets, drop the row's confidence.
                if text[start:end] != span_text:
                    confidence *= 0.4
                    continue
                if known_types and span_type not in known_types:
                    confidence *= 0.75
                valid_spans.append({
                    "type": span_type,
                    "start": start,
                    "end": end,
                    "text": span_text,
                })
            # Allow empty-spans rows only when source examples also had
            # zero-span rows (negatives). Otherwise an empty spans list
            # is probably an LLM artifact, drop confidence.
            if not valid_spans and any(self._extract_spans(g) for g in gold):
                confidence *= 0.3
            accepted.append({
                "payload": {"text": text, "spans": valid_spans},
                "synth_confidence": confidence,
                "synth_source": f"playbook:span-extraction:{self.mode.value}",
            })
        return accepted

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(row: dict[str, Any]) -> str:
        for key in ("text", "question", "input"):
            value = row.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                inner = value.get("text") or value.get("ticket") or value.get("question")
                if isinstance(inner, str):
                    return inner
        return ""

    @staticmethod
    def _extract_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
        # Try a few shapes: row['spans'], row['expected']['spans'/'entities'],
        # row['answer'] as a JSON string.
        if isinstance(row.get("spans"), list):
            return row["spans"]
        if isinstance(row.get("entities"), list):
            return row["entities"]
        expected = row.get("expected")
        if isinstance(expected, dict):
            for key in ("spans", "entities"):
                if isinstance(expected.get(key), list):
                    return expected[key]
        answer = row.get("answer")
        if isinstance(answer, str):
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                for key in ("spans", "entities"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]
        return []

    @classmethod
    def _collect_entity_types(cls, gold_rows: list[dict[str, Any]]) -> set[str]:
        types: set[str] = set()
        for row in gold_rows:
            for span in cls._extract_spans(row):
                if isinstance(span, dict) and isinstance(span.get("type"), str):
                    types.add(span["type"])
        return types


register_playbook(SpanExtractionParaphrasePlaybook())
