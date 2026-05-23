"""CLUSTER_TARGETED playbook for the `code-review` recipe."""

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
You are generating training data targeted at a specific failure pattern in a code-review model.

The model is failing on a cluster of test rows with this signature:

{cluster_block}

Reference (correct) code/feedback pairs:
{anchor_block}

Your task: generate {target_count} NEW (code, feedback) pairs that:
  - Contain code with the same KIND of issue the failure cluster captures
  - Have feedback that correctly identifies the issue
  - Vary specifics across examples — different variable names, surrounding lines, language idioms

For each example, write a single JSON line:
{{"code": "<snippet>", "feedback": "<specific review comment>"}}

Rules:
  - The feedback MUST name the issue concretely.
  - Keep newlines in `code` as literal \\n.

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class CodeReviewClusterTargetedPlaybook:
    recipe_id = "code-review"
    mode = SynthMode.CLUSTER_TARGETED

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 12
        gold = ctx.get("gold_rows") or []
        cluster = ctx.get("failure_cluster")

        anchors = sample_gold_rows(gold, count=min(3, len(gold)), seed=0)
        anchor_lines: list[str] = []
        for r in anchors:
            anchor_lines.append(
                f"  - code: {self._extract_code(r)!r}\n    feedback: {self._extract_feedback(r)!r}"
            )

        return _PROMPT_TEMPLATE.format(
            cluster_block=render_cluster_block(cluster),
            anchor_block="\n".join(anchor_lines) or "  (none)",
            target_count=target,
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        cluster = ctx.get("failure_cluster")
        suffix = cluster_provenance_suffix(cluster)

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
                "synth_source": f"playbook:code-review:{self.mode.value}:cluster={suffix}",
            })
        return accepted

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


register_playbook(CodeReviewClusterTargetedPlaybook())
