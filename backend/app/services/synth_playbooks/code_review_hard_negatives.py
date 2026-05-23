"""HARD_NEGATIVES playbook for the `code-review` recipe.

Generates code snippets that *look clean* — well-formatted,
idiomatic, no obvious smells — but contain a subtle bug that the
review feedback should flag. Trains the reviewer-model to look
past surface polish.
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

Existing examples (each has a code snippet + the feedback it deserves):
{example_block}

Your task: generate {target_count} HARD NEGATIVE examples.

A hard negative is code that:
  - LOOKS clean at first glance — well-named variables, reasonable structure, no obvious smells
  - Contains a subtle bug the reviewer should flag (off-by-one, wrong default, race condition, dead branch, missed error case, etc.)
  - Has feedback that names the specific subtle issue, not a generic "looks fine"

For each example, write a single JSON line:
{{"code": "<snippet>", "feedback": "<specific issue the reviewer caught>"}}

Rules:
  - The feedback MUST identify the actual bug, not a stylistic comment.
  - Vary the bug type across examples; don't repeat off-by-one in every one.
  - Code can be Python, JavaScript, or whatever matches the source examples.
  - Keep newlines in `code` as literal \\n.

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class CodeReviewHardNegativesPlaybook:
    recipe_id = "code-review"
    mode = SynthMode.HARD_NEGATIVES

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 15
        gold = ctx.get("gold_rows") or []
        examples = sample_gold_rows(gold, count=min(4, len(gold)), seed=0)
        lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            code = self._extract_code(row)
            feedback = self._extract_feedback(row)
            lines.append(f"{i}. code: {code!r}\n   feedback: {feedback!r}")

        return _PROMPT_TEMPLATE.format(
            example_block="\n".join(lines) or "(none)",
            target_count=target,
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
            if len(code) < 10 or len(code) > 8000 or len(feedback) < 15 or len(feedback) > 4000:
                continue
            # Sanity guard: feedback should not be a generic "looks fine"
            # — this is hard-negatives mode, the bug must be named.
            lower = feedback.lower()
            if any(phrase in lower for phrase in ("looks fine", "no issues", "lgtm", "all good")):
                # Generation drifted into positive-mode output; drop confidence.
                confidence = 0.30
            else:
                confidence = 1.0
            accepted.append({
                "payload": {"code": code, "feedback": feedback},
                "synth_confidence": confidence,
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


register_playbook(CodeReviewHardNegativesPlaybook())
