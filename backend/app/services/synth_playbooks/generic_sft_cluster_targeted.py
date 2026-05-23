"""CLUSTER_TARGETED playbook for the `generic-sft` recipe."""

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
You are generating training data targeted at a specific failure pattern in an instruction-tuned model.

The model is failing on a cluster of test rows with this signature:

{cluster_block}

Reference (correct) prompt/completion pairs:
{anchor_block}

Your task: generate {target_count} NEW (prompt, completion) pairs that:
  - Test the same failure pattern the cluster captured
  - Have the correct completion the model SHOULD produce
  - Are diverse — vary specifics, length, surface style

For each example, write a single JSON line:
{{"prompt": "...", "completion": "..."}}

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class GenericSftClusterTargetedPlaybook:
    recipe_id = "generic-sft"
    mode = SynthMode.CLUSTER_TARGETED

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        cluster = ctx.get("failure_cluster")

        anchors = sample_gold_rows(gold, count=min(4, len(gold)), seed=0)
        anchor_lines: list[str] = []
        for r in anchors:
            anchor_lines.append(
                f"  - prompt: {self._extract_prompt(r)!r}\n    completion: {self._extract_completion(r)!r}"
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
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                continue
            prompt, completion = prompt.strip(), completion.strip()
            if not prompt or not completion:
                continue
            if len(prompt) < 2 or len(prompt) > 4000 or len(completion) > 10000:
                continue
            accepted.append({
                "payload": {"prompt": prompt, "completion": completion},
                "synth_confidence": 1.0,
                "synth_source": f"playbook:generic-sft:{self.mode.value}:cluster={suffix}",
            })
        return accepted

    @staticmethod
    def _extract_prompt(row: dict[str, Any]) -> str:
        for key in ("prompt", "input", "question", "instruction"):
            value = row.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for sub_key in ("prompt", "text", "instruction", "draft", "question"):
                    inner = value.get(sub_key)
                    if isinstance(inner, str):
                        return inner
        return ""

    @staticmethod
    def _extract_completion(row: dict[str, Any]) -> str:
        for key in ("completion", "output", "answer", "response"):
            value = row.get(key)
            if isinstance(value, str):
                return value
        expected = row.get("expected")
        if isinstance(expected, dict):
            for sub_key in ("completion", "output", "answer", "rewrite", "response"):
                inner = expected.get(sub_key)
                if isinstance(inner, str):
                    return inner
        if isinstance(expected, str):
            return expected
        return ""


register_playbook(GenericSftClusterTargetedPlaybook())
