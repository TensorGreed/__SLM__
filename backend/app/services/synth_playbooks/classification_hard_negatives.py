"""HARD_NEGATIVES playbook for the `classification` recipe.

Generates examples that *look like* one class but should be labeled
as another. Targets the confusion-matrix off-diagonals — the
highest-leverage failure mode for under-resourced classifiers.

Validation explicitly drops rows where ``label == target_class``
(those would be paraphrased positives, not negatives) and rows whose
label isn't in the project's known class set (the model invented a
new class).
"""

from __future__ import annotations

import random
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
You are generating training data for a text classifier.

Classes in this dataset: {class_list}

Existing examples of class {target_class!r}:
{positive_block}

Existing examples of other classes (for context):
{other_block}

Your task: generate {target_count} HARD NEGATIVE examples for class {target_class!r}.

A hard negative is a text that:
  - LOOKS like it could be class {target_class!r} (uses similar vocabulary, similar surface features)
  - But should actually be labeled as ONE OF THE OTHER CLASSES
  - Is realistic — would plausibly appear in real data of that other class

For each example, write a single JSON line:
{{"text": "...", "label": "<one of the OTHER classes>"}}

Rules:
  - Do NOT label any output as {target_class!r}.
  - Use only classes from the list above.
  - Vary the wording — don't just copy a positive example and change the label.

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class ClassificationHardNegativesPlaybook:
    recipe_id = "classification"
    mode = SynthMode.HARD_NEGATIVES

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        target_class = ctx.get("target_class")
        gold = ctx.get("gold_rows") or []

        labels = self._collect_labels(gold)
        if not labels:
            return ""
        # Pick a class to be the "look-alike" target. Prefer the
        # caller's target_class; if absent, sample one with at least
        # 2 examples in the gold set so the prompt has anchor data.
        if target_class is None or target_class not in labels:
            target_class = self._pick_default_target(gold, labels)

        # Build the positive-example block (target class) — small set.
        positives = [
            r for r in gold if self._extract_label(r) == target_class
        ]
        positives = sample_gold_rows(positives, count=min(4, len(positives)), seed=0)
        positive_block = "\n".join(
            f"  - {self._extract_text(r)!r}" for r in positives
        ) or "  (none)"

        # Other-class examples — show 2-3 to anchor the labels.
        others = [
            r for r in gold if self._extract_label(r) != target_class and self._extract_label(r)
        ]
        others = sample_gold_rows(others, count=min(6, len(others)), seed=1)
        other_block = "\n".join(
            f"  - [{self._extract_label(r)}] {self._extract_text(r)!r}" for r in others
        ) or "  (none)"

        # Stash the resolved target_class back into ctx for validate().
        # PlaybookContext is total=False so this is safe.
        ctx["target_class"] = target_class  # type: ignore[typeddict-item]

        return _PROMPT_TEMPLATE.format(
            class_list=", ".join(sorted(labels)),
            target_class=target_class,
            positive_block=positive_block,
            other_block=other_block,
            target_count=target,
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        target_class = ctx.get("target_class")
        gold = ctx.get("gold_rows") or []
        known_labels = self._collect_labels(gold)

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
            # CRITICAL: the playbook is generating *negatives* for
            # target_class. A row labeled target_class is a generation
            # failure, drop it.
            if target_class is not None and label == target_class:
                continue
            # Label must be in the known class set.
            if known_labels and label not in known_labels:
                confidence *= 0.30
            accepted.append({
                "payload": {"text": text, "label": label},
                "synth_confidence": confidence,
                "synth_source": f"playbook:classification:{self.mode.value}:vs={target_class or 'auto'}",
            })
        return accepted

    # ── helpers ─────────────────────────────────────────────────────

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

    @classmethod
    def _pick_default_target(cls, gold_rows: list[dict[str, Any]], labels: set[str]) -> str:
        # Pick the class with the most examples — gives the LLM the most
        # anchor data for "what target_class looks like."
        counts: dict[str, int] = {}
        for row in gold_rows:
            label = cls._extract_label(row)
            if label:
                counts[label] = counts.get(label, 0) + 1
        if not counts:
            return next(iter(labels))
        # Deterministic tiebreak by label string.
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


register_playbook(ClassificationHardNegativesPlaybook())
