"""POSITIVES_PARAPHRASE playbook for the `generic-sft` recipe.

Catch-all for free-text prompt → completion data. Paraphrases the
prompt and keeps the completion stable. The validator is intentionally
loose because generic-sft datasets vary widely in shape.
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
You are generating training data for an instruction-tuning task.

You are given a small set of (prompt, completion) pairs. For each, write {paraphrases_per_row} alternative phrasings of the PROMPT that should produce the SAME completion.

Rules:
- Do NOT change the completion.
- Preserve the prompt's intent and any specific facts/values it mentions.
- Vary wording, formality, length, sentence structure.
- Each paraphrase should sound like a realistic user request of the same kind.

Source examples:
{examples_block}

Output exactly {total_count} lines. Each line MUST be a single JSON object:
{{"prompt": "<paraphrased prompt>", "completion": "<the exact original completion>"}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines.
"""


class GenericSftParaphrasePlaybook:
    recipe_id = "generic-sft"
    mode = SynthMode.POSITIVES_PARAPHRASE

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 30
        gold = ctx.get("gold_rows") or []
        n_source = max(1, min(len(gold), max(3, target // 5)))
        examples = sample_gold_rows(gold, count=n_source, seed=0)
        paraphrases_per_row = max(1, target // max(1, len(examples)))

        lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            prompt = self._extract_prompt(row)
            completion = self._extract_completion(row)
            lines.append(
                f"{i}. prompt: {prompt!r}\n   completion: {completion!r}"
            )

        return _PROMPT_TEMPLATE.format(
            paraphrases_per_row=paraphrases_per_row,
            examples_block="\n".join(lines),
            total_count=paraphrases_per_row * len(examples),
        )

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]:
        gold = ctx.get("gold_rows") or []
        known_completions = {self._extract_completion(r) for r in gold if self._extract_completion(r)}

        accepted: list[SynthRow] = []
        for row in parsed_rows:
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                continue
            prompt, completion = prompt.strip(), completion.strip()
            if not prompt or not completion:
                continue
            if len(prompt) < 2 or len(prompt) > 4000 or len(completion) < 1 or len(completion) > 10000:
                continue
            confidence = 1.0
            if known_completions and completion not in known_completions:
                # Loose recipe — we tolerate paraphrased completions
                # but lower confidence so the reviewer queue surfaces them.
                confidence *= 0.65
            accepted.append({
                "payload": {"prompt": prompt, "completion": completion},
                "synth_confidence": confidence,
                "synth_source": f"playbook:generic-sft:{self.mode.value}",
            })
        return accepted

    # ── helpers ─────────────────────────────────────────────────────

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


register_playbook(GenericSftParaphrasePlaybook())
