"""POSITIVES_PARAPHRASE playbook for the `code-review` recipe.

Generates code snippets with the same review-style feedback. The
code is paraphrased structurally (variable renames, equivalent
formulations) while preserving the issue the feedback flags.
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
You are generating training data for a code-review fine-tuning task.

You are given a small set of (code, feedback) examples. For each, write {paraphrases_per_row} alternative code snippets that exhibit the SAME issue the feedback flags. Vary variable names, the surrounding context, formatting, and language idioms when appropriate.

Rules:
- The new code MUST still have the same kind of problem the feedback describes.
- Preserve the language (Python stays Python, etc.).
- The feedback text can be paraphrased to match the new code, but the underlying complaint should be the same.
- Avoid trivially identical snippets — change variable names, surrounding lines, expression order.

Source examples:
{examples_block}

Output exactly {total_count} lines. Each line MUST be a single JSON object:
{{"code": "<the new code snippet>", "feedback": "<the matching review feedback>"}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines. Keep newlines inside `code` as \\n.
"""


class CodeReviewParaphrasePlaybook:
    recipe_id = "code-review"
    mode = SynthMode.POSITIVES_PARAPHRASE

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        n_source = max(1, min(len(gold), max(2, target // 4)))
        examples = sample_gold_rows(gold, count=n_source, seed=0)
        paraphrases_per_row = max(1, target // max(1, len(examples)))

        lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            code = self._extract_code(row)
            feedback = self._extract_feedback(row)
            lines.append(
                f"{i}. code: {code!r}\n   feedback: {feedback!r}"
            )

        return _PROMPT_TEMPLATE.format(
            paraphrases_per_row=paraphrases_per_row,
            examples_block="\n".join(lines),
            total_count=paraphrases_per_row * len(examples),
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        accepted: list[SynthRow] = []
        for row in parsed_rows:
            code = row.get("code")
            feedback = row.get("feedback")
            if not isinstance(code, str) or not isinstance(feedback, str):
                continue
            code, feedback = code.strip(), feedback.strip()
            if not code or not feedback:
                continue
            if len(code) < 10 or len(code) > 8000 or len(feedback) < 10 or len(feedback) > 4000:
                continue
            accepted.append({
                "payload": {"code": code, "feedback": feedback},
                "synth_confidence": 1.0,
                "synth_source": f"playbook:code-review:{self.mode.value}",
            })
        return accepted

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_code(row: dict[str, Any]) -> str:
        for key in ("code", "snippet", "input"):
            value = row.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict) and isinstance(value.get("code"), str):
                return value["code"]
        question = row.get("question")
        if isinstance(question, str):
            return question
        return ""

    @staticmethod
    def _extract_feedback(row: dict[str, Any]) -> str:
        for key in ("feedback", "review", "answer", "output"):
            value = row.get(key)
            if isinstance(value, str):
                return value
        expected = row.get("expected")
        if isinstance(expected, dict) and isinstance(expected.get("feedback"), str):
            return expected["feedback"]
        if isinstance(expected, str):
            return expected
        return ""


register_playbook(CodeReviewParaphrasePlaybook())
