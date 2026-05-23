"""CLUSTER_TARGETED playbook for the `summarization` recipe."""

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
You are generating training data targeted at a specific failure pattern in a summarization model.

The model is failing on a cluster of test rows with this signature:

{cluster_block}

Reference (correct) source/summary pairs:
{anchor_block}

Your task: generate {target_count} NEW (source, summary) pairs that:
  - Resemble the failure-cluster sources in domain + length + structure
  - Have the correct summary the model SHOULD produce
  - Cover the same kind of input the cluster captured

For each example, write a single JSON line:
{{"source": "<source text>", "summary": "<correct summary>"}}

Rules:
  - The summary MUST be meaningfully shorter than the source.
  - Preserve every fact in the source; don't introduce information not present.
  - Vary specifics; don't repeat the cluster exemplars verbatim.

Output exactly {target_count} JSON lines, no preamble, no markdown code fences.
"""


class SummarizationClusterTargetedPlaybook:
    recipe_id = "summarization"
    mode = SynthMode.CLUSTER_TARGETED

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 12
        gold = ctx.get("gold_rows") or []
        cluster = ctx.get("failure_cluster")

        anchors = sample_gold_rows(gold, count=min(3, len(gold)), seed=0)
        anchor_lines: list[str] = []
        for r in anchors:
            anchor_lines.append(
                f"  - source: {self._extract_source(r)!r}\n    summary: {self._extract_summary(r)!r}"
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
            source = row.get("source")
            summary = row.get("summary")
            if not isinstance(source, str) or not isinstance(summary, str):
                continue
            source, summary = source.strip(), summary.strip()
            if not source or not summary:
                continue
            if len(source) < 20 or len(source) > 10000 or len(summary) < 5 or len(summary) > 2000:
                continue
            confidence = 1.0 if len(summary) < len(source) else 0.4
            accepted.append({
                "payload": {"source": source, "summary": summary},
                "synth_confidence": confidence,
                "synth_source": f"playbook:summarization:{self.mode.value}:cluster={suffix}",
            })
        return accepted

    @staticmethod
    def _extract_source(row: dict[str, Any]) -> str:
        for key in ("source", "input", "question", "text"):
            value = row.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for sub_key in ("source", "text", "advisory", "input"):
                    inner = value.get(sub_key)
                    if isinstance(inner, str):
                        return inner
        return ""

    @staticmethod
    def _extract_summary(row: dict[str, Any]) -> str:
        for key in ("summary", "answer", "output"):
            value = row.get(key)
            if isinstance(value, str):
                return value
        expected = row.get("expected")
        if isinstance(expected, dict):
            for sub_key in ("summary", "answer"):
                inner = expected.get(sub_key)
                if isinstance(inner, str):
                    return inner
        if isinstance(expected, str):
            return expected
        return ""


register_playbook(SummarizationClusterTargetedPlaybook())
