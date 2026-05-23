"""CLUSTER_TARGETED playbook for the `classification` recipe."""

from __future__ import annotations

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
You are generating training data targeted at a specific failure pattern in a text classifier.

Classes in this dataset: {class_list}

The classifier is failing on a cluster of test rows with this signature:

{cluster_block}

Reference (correct) examples:
{anchor_block}

Your task: generate {target_count} NEW examples that:
  - Resemble the failure-cluster inputs (same style, same kind of text)
  - Have the correct label the classifier SHOULD produce
  - Are diverse within the cluster's pattern

For each example, write a single JSON line:
{{"text": "...", "label": "<one of the classes>"}}

Rules:
  - Only use labels from the list above.
  - Don't repeat the cluster exemplars verbatim; generate new specifics.

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class ClassificationClusterTargetedPlaybook:
    recipe_id = "classification"
    mode = SynthMode.CLUSTER_TARGETED

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        cluster = ctx.get("failure_cluster")

        labels = sorted(self._collect_labels(gold))
        anchors = sample_gold_rows(gold, count=min(5, len(gold)), seed=0)
        anchor_block = "\n".join(
            f"  - [{self._extract_label(r)}] {self._extract_text(r)!r}" for r in anchors
        ) or "  (none)"

        return _PROMPT_TEMPLATE.format(
            class_list=", ".join(labels) or "(unknown)",
            cluster_block=render_cluster_block(cluster),
            anchor_block=anchor_block,
            target_count=target,
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        gold = ctx.get("gold_rows") or []
        known = self._collect_labels(gold)
        cluster = ctx.get("failure_cluster")
        suffix = cluster_provenance_suffix(cluster)

        accepted: list[SynthRow] = []
        for row in parsed_rows:
            text = row.get("text")
            label = row.get("label")
            if not isinstance(text, str) or not isinstance(label, str):
                continue
            text, label = text.strip(), label.strip()
            if not text or not label:
                continue
            if len(text) < 5 or len(text) > 4000:
                continue
            confidence = 1.0
            if known and label not in known:
                confidence *= 0.30
            accepted.append({
                "payload": {"text": text, "label": label},
                "synth_confidence": confidence,
                "synth_source": f"playbook:classification:{self.mode.value}:cluster={suffix}",
            })
        return accepted

    @staticmethod
    def _extract_text(row: dict[str, Any]) -> str:
        for key in ("text", "input", "question"):
            value = row.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for sub in value.values():
                    if isinstance(sub, str):
                        return sub
        return ""

    @staticmethod
    def _extract_label(row: dict[str, Any]) -> str:
        if isinstance(row.get("label"), str):
            return row["label"]
        expected = row.get("expected")
        if isinstance(expected, dict) and isinstance(expected.get("label"), str):
            return expected["label"]
        if isinstance(expected, str):
            return expected
        answer = row.get("answer")
        if isinstance(answer, str) and 0 < len(answer) <= 64 and "\n" not in answer:
            return answer
        return ""

    @classmethod
    def _collect_labels(cls, gold_rows: list[dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for row in gold_rows:
            label = cls._extract_label(row)
            if label:
                out.add(label)
        return out


register_playbook(ClassificationClusterTargetedPlaybook())
