"""HARD_NEGATIVES playbook for the `span-extraction` recipe.

Generates text that *looks like* it should contain spans (similar
vocabulary, similar context) but in fact has zero spans of any
known entity type. Train-time, this teaches the model when NOT to
emit spans — a major source of false-positive errors in extractors
trained only on positive examples.
"""

from __future__ import annotations

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
You are generating training data for a span-extraction (NER-style) task.

The model is trained to find spans of these entity types: {entity_types}

Existing positive examples (each contains real entities):
{positive_block}

Your task: generate {target_count} HARD NEGATIVE examples.

A hard negative is a text that:
  - Sounds similar to the positive examples (same domain, similar vocabulary, similar structure)
  - But contains ZERO instances of any of the entity types listed above
  - Is realistic — would plausibly appear in the same corpus

For each example, write a single JSON line:
{{"text": "...", "spans": []}}

The spans list MUST be empty. Vary topic, length, and structure.

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class SpanExtractionHardNegativesPlaybook:
    recipe_id = "span-extraction"
    mode = SynthMode.HARD_NEGATIVES

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        entity_types = sorted(self._collect_entity_types(gold))

        positives = [r for r in gold if self._extract_spans(r)]
        positives = sample_gold_rows(positives, count=min(5, len(positives)), seed=0)
        positive_block = "\n".join(
            f"  - {self._extract_text(r)!r}" for r in positives
        ) or "  (none)"

        return _PROMPT_TEMPLATE.format(
            entity_types=", ".join(entity_types) or "(unknown)",
            positive_block=positive_block,
            target_count=target,
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        accepted: list[SynthRow] = []
        for row in parsed_rows:
            text = row.get("text")
            spans = row.get("spans")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            if len(text) < 10 or len(text) > 4000:
                continue
            # Hard negative ⇒ spans MUST be empty (or missing/null).
            # If the model generated spans anyway, drop the row entirely
            # — this is the wrong mode for that output.
            if spans is None:
                spans_list: list[Any] = []
            elif isinstance(spans, list):
                spans_list = spans
            else:
                continue
            if spans_list:
                # Model generated spans for what was supposed to be a negative
                # — generation failure, drop.
                continue
            accepted.append({
                "payload": {"text": text, "spans": []},
                "synth_confidence": 1.0,
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
                for sub_key in ("text", "ticket", "question"):
                    inner = value.get(sub_key)
                    if isinstance(inner, str):
                        return inner
        return ""

    @staticmethod
    def _extract_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
        import json

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


register_playbook(SpanExtractionHardNegativesPlaybook())
