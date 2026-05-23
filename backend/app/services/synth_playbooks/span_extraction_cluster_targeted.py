"""CLUSTER_TARGETED playbook for the `span-extraction` recipe."""

from __future__ import annotations

import json
from typing import Any

from ._cluster_targeted_common import (
    cluster_provenance_suffix,
    render_cluster_block,
)
from .base import (
    PlaybookContext,
    SynthMode,
    SynthRow,
    parse_jsonl_lines,
    register_playbook,
    sample_gold_rows,
)


_PROMPT_TEMPLATE = """\
You are generating training data targeted at a specific failure pattern in a span-extraction model.

Entity types the model is trained to find: {entity_types}

The model is failing on a cluster of test rows with this signature:

{cluster_block}

Reference (correct) examples:
{anchor_block}

Your task: generate {target_count} NEW examples that:
  - Resemble the failure-cluster inputs in style and structure
  - Have correct, exactly-offset spans the model SHOULD produce
  - Cover the same entity types as the cluster's failures

For each example, write a single JSON line:
{{"text": "...", "spans": [{{"type": "...", "start": N, "end": N, "text": "..."}}]}}

Rules:
  - `start` and `end` MUST be valid 0-indexed character offsets into `text` such that `text[start:end]` equals the span's `text` field.
  - Use only entity types from the list above.
  - Empty spans list is allowed for negatives — useful for teaching the model when NOT to emit spans.

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class SpanExtractionClusterTargetedPlaybook:
    recipe_id = "span-extraction"
    mode = SynthMode.CLUSTER_TARGETED

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 15
        gold = ctx.get("gold_rows") or []
        cluster = ctx.get("failure_cluster")
        types = sorted(self._collect_entity_types(gold))

        anchors = sample_gold_rows(gold, count=min(3, len(gold)), seed=0)
        anchor_lines: list[str] = []
        for r in anchors:
            text = self._extract_text(r)
            spans = self._extract_spans(r)
            anchor_lines.append(
                f"  - text: {text!r}\n    spans: {json.dumps(spans, ensure_ascii=False)}"
            )
        anchor_block = "\n".join(anchor_lines) or "  (none)"

        return _PROMPT_TEMPLATE.format(
            entity_types=", ".join(types) or "(unknown)",
            cluster_block=render_cluster_block(cluster),
            anchor_block=anchor_block,
            target_count=target,
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        gold = ctx.get("gold_rows") or []
        cluster = ctx.get("failure_cluster")
        known_types = self._collect_entity_types(gold)
        suffix = cluster_provenance_suffix(cluster)

        accepted: list[SynthRow] = []
        for row in parsed_rows:
            text = row.get("text")
            spans = row.get("spans")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            confidence = 1.0
            valid_spans: list[dict[str, Any]] = []
            if isinstance(spans, list):
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
            accepted.append({
                "payload": {"text": text, "spans": valid_spans},
                "synth_confidence": confidence,
                "synth_source": f"playbook:span-extraction:{self.mode.value}:cluster={suffix}",
            })
        return accepted

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


register_playbook(SpanExtractionClusterTargetedPlaybook())
