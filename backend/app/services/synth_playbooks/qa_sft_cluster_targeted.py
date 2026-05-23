"""CLUSTER_TARGETED playbook for the `qa-sft` recipe.

Generates Q&A pairs that target a specific failure cluster from
the eval. The model just failed in a particular pattern on a
particular kind of question; this playbook produces new training
data that exercises the same pattern with correct answers.
"""

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
You are generating training data targeted at a specific failure pattern in a Q&A model.

The model is failing on a cluster of test rows with this signature:

{cluster_block}

Reference (correct) Q&A pairs from the training set:
{anchor_block}

Your task: generate {target_count} NEW Q&A pairs that:
  - Test the same failure pattern (same kind of question, similar context)
  - Have the correct answer the model SHOULD produce
  - Are diverse — vary the specifics so the model can't pattern-match a single template

For each example, write a single JSON line:
{{"question": "<a new question targeting the failure pattern>", "answer": "<the correct answer>"}}

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class QaSftClusterTargetedPlaybook:
    recipe_id = "qa-sft"
    mode = SynthMode.CLUSTER_TARGETED

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        cluster = ctx.get("failure_cluster")

        anchors = sample_gold_rows(gold, count=min(4, len(gold)), seed=0)
        anchor_block = "\n".join(
            f"  - Q: {self._extract_question(r)!r}\n    A: {self._extract_answer(r)!r}"
            for r in anchors
        ) or "  (none)"

        return _PROMPT_TEMPLATE.format(
            cluster_block=render_cluster_block(cluster),
            anchor_block=anchor_block,
            target_count=target,
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        cluster = ctx.get("failure_cluster")
        suffix = cluster_provenance_suffix(cluster)

        accepted: list[SynthRow] = []
        for row in parsed_rows:
            q = row.get("question")
            a = row.get("answer")
            if not isinstance(q, str) or not isinstance(a, str):
                continue
            q, a = q.strip(), a.strip()
            if not q or not a:
                continue
            if len(q) < 5 or len(q) > 2000 or len(a) > 8000:
                continue
            accepted.append({
                "payload": {"question": q, "answer": a},
                "synth_confidence": 1.0,
                "synth_source": f"playbook:qa-sft:{self.mode.value}:cluster={suffix}",
            })
        return accepted

    @staticmethod
    def _extract_question(row: dict[str, Any]) -> str:
        if isinstance(row.get("question"), str):
            return row["question"]
        if isinstance(row.get("input"), dict) and isinstance(row["input"].get("question"), str):
            return row["input"]["question"]
        if isinstance(row.get("input"), str):
            return row["input"]
        return ""

    @staticmethod
    def _extract_answer(row: dict[str, Any]) -> str:
        if isinstance(row.get("answer"), str):
            return row["answer"]
        if isinstance(row.get("expected"), dict) and isinstance(row["expected"].get("answer"), str):
            return row["expected"]["answer"]
        if isinstance(row.get("expected"), str):
            return row["expected"]
        return ""


register_playbook(QaSftClusterTargetedPlaybook())
