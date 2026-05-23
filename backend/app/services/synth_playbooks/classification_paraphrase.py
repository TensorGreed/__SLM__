"""POSITIVES_PARAPHRASE playbook for the `classification` recipe.

Generates paraphrased *text* that keeps the same label. Useful for
expanding training examples per class without changing the label
distribution.
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
You are generating training data for a text-classification fine-tuning task.

You are given a small set of (text, label) examples. Generate {paraphrases_per_row} alternative phrasings for each text that should map to the SAME label.

Rules:
- Do NOT change the label.
- Preserve the underlying meaning and intent — only vary the wording, length, tone, grammar.
- Do NOT introduce content that would shift the example into a different class.
- Make each paraphrase sound like real data of the same class.

Source examples:
{examples_block}

Output exactly {total_count} lines. Each line MUST be a single JSON object:
{{"text": "<paraphrased text>", "label": "<the exact original label>"}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines.
"""


class ClassificationParaphrasePlaybook:
    recipe_id = "classification"
    mode = SynthMode.POSITIVES_PARAPHRASE

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 30
        gold = ctx.get("gold_rows") or []
        n_source = max(1, min(len(gold), max(3, target // 5)))
        examples = sample_gold_rows(gold, count=n_source, seed=0)
        paraphrases_per_row = max(1, target // max(1, len(examples)))

        lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            text = self._extract_text(row)
            label = self._extract_label(row)
            lines.append(f"{i}. text: {text!r}\n   label: {label!r}")

        return _PROMPT_TEMPLATE.format(
            paraphrases_per_row=paraphrases_per_row,
            examples_block="\n".join(lines),
            total_count=paraphrases_per_row * len(examples),
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        gold = ctx.get("gold_rows") or []
        known_labels = {self._extract_label(r) for r in gold if self._extract_label(r)}

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
            # The label MUST be one of the known classes — otherwise
            # the model invented a new class.
            if known_labels and label not in known_labels:
                confidence *= 0.25
            accepted.append({
                "payload": {"text": text, "label": label},
                "synth_confidence": confidence,
                "synth_source": f"playbook:classification:{self.mode.value}",
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
        # Walk the same shape variants the trainability forecast does.
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


register_playbook(ClassificationParaphrasePlaybook())
