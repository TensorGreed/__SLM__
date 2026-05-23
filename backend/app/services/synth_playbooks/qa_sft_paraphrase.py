"""POSITIVES_PARAPHRASE playbook for the `qa-sft` recipe.

Generates paraphrased *questions* that map to the same answer. The
answer text is preserved verbatim — paraphrasing the answer would
shift the eval target.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .base import (
    Playbook,
    PlaybookContext,
    SynthMode,
    SynthRow,
    parse_jsonl_lines,
    register_playbook,
    sample_gold_rows,
)


_PROMPT_TEMPLATE = """\
You are generating training data for a question-answering fine-tuning task.

You are given a small set of (question, answer) pairs from a real dataset.

Your task: for each pair, write {paraphrases_per_row} alternative phrasings of the QUESTION that should still produce the SAME answer. Preserve the meaning exactly; vary the wording, formality, length, and grammatical structure.

Rules:
- Do NOT change the answer.
- Do NOT add information not present in the original question.
- Do NOT generate trivial restatements (e.g., just adding "please?").
- Each paraphrase should sound like a real user might ask it.

Source pairs:
{examples_block}

Output exactly {total_count} lines. Each line MUST be a single JSON object:
{{"question": "<paraphrased question>", "answer": "<the exact original answer>"}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines.
"""


class QaSftParaphrasePlaybook:
    recipe_id = "qa-sft"
    mode = SynthMode.POSITIVES_PARAPHRASE

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 30
        gold = ctx.get("gold_rows") or []
        # Pick a manageable number of source rows; ~5 per requested
        # output keeps the prompt focused without hitting context limits.
        n_source = max(1, min(len(gold), max(3, target // 5)))
        examples = sample_gold_rows(gold, count=n_source, seed=0)
        paraphrases_per_row = max(1, target // max(1, len(examples)))

        example_lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            question = self._extract_question(row)
            answer = self._extract_answer(row)
            example_lines.append(
                f"{i}. Q: {question!r}\n   A: {answer!r}"
            )

        return _PROMPT_TEMPLATE.format(
            paraphrases_per_row=paraphrases_per_row,
            examples_block="\n".join(example_lines),
            total_count=paraphrases_per_row * len(examples),
        )

    def parse_output(
        self,
        raw_llm_output: str,
        ctx: PlaybookContext,
    ) -> list[dict[str, Any]]:
        return parse_jsonl_lines(raw_llm_output)

    def validate(
        self,
        parsed_rows: list[dict[str, Any]],
        ctx: PlaybookContext,
    ) -> list[SynthRow]:
        gold = ctx.get("gold_rows") or []
        # Build a lookup of known answers so we can score whether the
        # generated row paraphrases an existing example.
        known_answers = {self._extract_answer(r) for r in gold if self._extract_answer(r)}

        accepted: list[SynthRow] = []
        for row in parsed_rows:
            q = row.get("question")
            a = row.get("answer")
            if not isinstance(q, str) or not isinstance(a, str):
                continue
            q, a = q.strip(), a.strip()
            if not q or not a:
                continue
            # Length sanity: nothing absurdly short or absurdly long.
            if len(q) < 5 or len(q) > 2000 or len(a) < 1 or len(a) > 8000:
                continue
            # Confidence starts at 1.0, drops if the answer doesn't
            # match any known gold answer (means the model invented
            # an answer instead of paraphrasing the question).
            confidence = 1.0
            if known_answers and a not in known_answers:
                confidence *= 0.55
            accepted.append({
                "payload": {"question": q, "answer": a},
                "synth_confidence": confidence,
                "synth_source": f"playbook:qa-sft:{self.mode.value}",
            })
        return accepted

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_question(row: dict[str, Any]) -> str:
        """Walk the common gold-row shapes for the question text."""
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


register_playbook(QaSftParaphrasePlaybook())
