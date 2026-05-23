"""POSITIVES_PARAPHRASE playbook for the `summarization` recipe.

Paraphrases the source text while preserving the same summary (the
target output). This expands the diversity of inputs the model
learns to compress, without changing what counts as a correct
summary.
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
You are generating training data for a summarization fine-tuning task.

You are given a small set of (source, summary) examples. For each, write {paraphrases_per_row} alternative phrasings of the SOURCE text that should still summarize to the same gist (or an equivalent paraphrase of the summary).

Rules:
- Preserve every fact in the original source — same entities, same numbers, same events.
- Vary tone, sentence structure, length, formality.
- Do NOT lose information; do NOT add facts.
- The summary may be paraphrased lightly (different wording, same meaning) but should not change facts.

Source examples:
{examples_block}

Output exactly {total_count} lines. Each line MUST be a single JSON object:
{{"source": "<paraphrased source>", "summary": "<the summary>"}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines.
"""


class SummarizationParaphrasePlaybook:
    recipe_id = "summarization"
    mode = SynthMode.POSITIVES_PARAPHRASE

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        n_source = max(1, min(len(gold), max(2, target // 5)))
        examples = sample_gold_rows(gold, count=n_source, seed=0)
        paraphrases_per_row = max(1, target // max(1, len(examples)))

        lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            source = self._extract_source(row)
            summary = self._extract_summary(row)
            lines.append(
                f"{i}. source: {source!r}\n   summary: {summary!r}"
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
            source = row.get("source")
            summary = row.get("summary")
            if not isinstance(source, str) or not isinstance(summary, str):
                continue
            source, summary = source.strip(), summary.strip()
            if not source or not summary:
                continue
            # Sanity: a summary should be meaningfully shorter than the source.
            if len(summary) >= len(source):
                confidence = 0.4
            else:
                confidence = 1.0
            if len(source) < 20 or len(source) > 10000 or len(summary) < 5 or len(summary) > 2000:
                continue
            accepted.append({
                "payload": {"source": source, "summary": summary},
                "synth_confidence": confidence,
                "synth_source": f"playbook:summarization:{self.mode.value}",
            })
        return accepted

    # ── helpers ─────────────────────────────────────────────────────

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


register_playbook(SummarizationParaphrasePlaybook())
