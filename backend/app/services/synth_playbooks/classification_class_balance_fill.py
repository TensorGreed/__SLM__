"""CLASS_BALANCE_FILL playbook for the `classification` recipe.

Identifies the under-represented class(es) in the project's gold
set and generates additional examples for them. The caller may pin
a specific class via ``ctx['target_class']``; otherwise the playbook
picks the class with the lowest count automatically.

Distinct from POSITIVES_PARAPHRASE: that mode paraphrases what's
already there in proportion to existing counts. This mode is
specifically for fixing class imbalance — it generates *more* of
the under-represented class, not a uniform spread.
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
You are generating training data for a text classifier.

Existing classes (and current example counts):
{distribution_block}

This dataset is imbalanced — class {target_class!r} is under-represented.

Existing examples of class {target_class!r}:
{anchor_block}

Your task: generate {target_count} NEW examples of class {target_class!r}.

Rules:
  - Every example MUST be labeled {target_class!r}.
  - Vary wording, length, tone — but stay clearly within the class's meaning.
  - Don't copy the anchor examples verbatim; paraphrase + create new angles.
  - Each example should sound like real data of class {target_class!r}.

For each example, write a single JSON line:
{{"text": "...", "label": {target_class!r}}}

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class ClassificationClassBalanceFillPlaybook:
    recipe_id = "classification"
    mode = SynthMode.CLASS_BALANCE_FILL

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 30
        target_class = ctx.get("target_class")
        gold = ctx.get("gold_rows") or []

        counts = self._count_labels(gold)
        if not counts:
            return ""

        # Auto-pick the lowest-count class if caller didn't specify.
        if target_class is None or target_class not in counts:
            target_class = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            ctx["target_class"] = target_class  # type: ignore[typeddict-item]

        anchors = [
            r for r in gold if self._extract_label(r) == target_class
        ]
        anchors = sample_gold_rows(anchors, count=min(5, len(anchors)), seed=0)
        anchor_block = "\n".join(
            f"  - {self._extract_text(r)!r}" for r in anchors
        ) or "  (none)"

        distribution_block = "\n".join(
            f"  - {label}: {count} examples"
            for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )

        return _PROMPT_TEMPLATE.format(
            distribution_block=distribution_block,
            target_class=target_class,
            anchor_block=anchor_block,
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
            # CRITICAL: this mode generates the target_class only.
            # A row labeled anything else is a generation failure.
            if target_class is not None and label != target_class:
                continue
            if known_labels and label not in known_labels:
                confidence *= 0.30
            accepted.append({
                "payload": {"text": text, "label": label},
                "synth_confidence": confidence,
                "synth_source": f"playbook:classification:{self.mode.value}:class={target_class or 'auto'}",
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
    def _count_labels(cls, gold_rows: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in gold_rows:
            label = cls._extract_label(row)
            if label:
                out[label] = out.get(label, 0) + 1
        return out

    @classmethod
    def _collect_labels(cls, gold_rows: list[dict[str, Any]]) -> set[str]:
        return set(cls._count_labels(gold_rows).keys())


register_playbook(ClassificationClassBalanceFillPlaybook())
