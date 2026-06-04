"""Arc R-1 — POSITIVES_PARAPHRASE playbook for the ``rag-protocol`` recipe.

Generates citation-drill examples from gold (context, question, answer)
triples. Same idea as ``qa_sft_paraphrase`` (paraphrase the question,
preserve the answer) but with two extras:

  - the **context** is carried through verbatim so the model learns
    that the same context grounds many question phrasings;
  - the **answer** must keep its ``[#N]`` citation marker(s) — the
    citation token IS the training signal. Rows whose generated
    answer drops the citation get scored down (0.45 confidence) so
    the review queue surfaces them.

The protocol is what the model trains on; the customer's actual facts
get bolted on at inference time via auto-RAG.
"""

from __future__ import annotations

import re
from typing import Any

from .base import (
    PlaybookContext,
    SynthMode,
    SynthRow,
    parse_jsonl_lines,
    register_playbook,
    sample_gold_rows,
)


_CITATION_RE = re.compile(r"\[#\d+\]")


_PROMPT_TEMPLATE = """\
You are generating training data for a RAG-protocol fine-tune. The model is being taught to USE a retrieval index correctly — answering grounded questions, citing the chunk it pulled from.

You are given a small set of (context, question, answer-with-citation) triples. Each ``answer`` contains a citation marker like ``[#1]`` that points at the relevant chunk in the context.

Your task: for each triple, write {paraphrases_per_row} alternative phrasings of the QUESTION that should map to the SAME context and the SAME answer (citation marker preserved verbatim). The model must learn that the same fact answers many phrasings.

Rules:
- Do NOT change the context.
- Do NOT change the answer text (citation markers stay verbatim).
- Vary the question's wording, formality, and length.
- Every output answer MUST contain at least one ``[#N]`` citation marker — that's the training signal.

Source triples:
{examples_block}

Output exactly {total_count} lines. Each line MUST be a single JSON object with this exact shape:
{{"context": "<copy the context verbatim>", "question": "<paraphrased question>", "answer": "<the exact original answer with [#N] preserved>"}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines.
"""


class RagProtocolParaphrasePlaybook:
    recipe_id = "rag-protocol"
    mode = SynthMode.POSITIVES_PARAPHRASE

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 30
        gold = ctx.get("gold_rows") or []
        n_source = max(1, min(len(gold), max(3, target // 5)))
        examples = sample_gold_rows(gold, count=n_source, seed=0)
        paraphrases_per_row = max(1, target // max(1, len(examples)))

        example_lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            context = _extract_context(row)
            question = _extract_question(row)
            answer = _extract_answer(row)
            example_lines.append(
                f"{i}. CONTEXT: {context!r}\n   Q: {question!r}\n   A: {answer!r}"
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
        known_answers = {_extract_answer(r) for r in gold if _extract_answer(r)}

        accepted: list[SynthRow] = []
        for row in parsed_rows:
            context = row.get("context")
            question = row.get("question")
            answer = row.get("answer")
            if not isinstance(context, str) or not isinstance(question, str) or not isinstance(answer, str):
                continue
            context, question, answer = context.strip(), question.strip(), answer.strip()
            if not context or not question or not answer:
                continue
            if len(question) < 5 or len(question) > 2000:
                continue
            if len(answer) < 1 or len(answer) > 8000:
                continue
            confidence = 1.0
            # Citation marker is the training signal — drop confidence
            # hard when the model omits it. We don't reject outright so
            # the review queue can surface near-misses for the user to
            # accept manually.
            if not _CITATION_RE.search(answer):
                confidence *= 0.45
            # Slight discount when the answer text drifts from any
            # known gold answer — same heuristic as qa-sft paraphrase.
            if known_answers and answer not in known_answers:
                confidence *= 0.75
            accepted.append({
                "payload": {
                    "context": context,
                    "question": question,
                    "answer": answer,
                },
                "synth_confidence": confidence,
                "synth_source": f"playbook:rag-protocol:{self.mode.value}",
            })
        return accepted


# ─────────────────────────────────────────────────────────────────────
# Shared field extractors. Same helpers used by the refusal + format
# playbooks below — kept module-level so each playbook can import
# them; the rag-protocol triple shape is consistent across modes.
# ─────────────────────────────────────────────────────────────────────


def _extract_context(row: dict[str, Any]) -> str:
    for key in ("context", "passage", "chunk", "source"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(row.get("input"), dict):
        inner = row["input"].get("context") or row["input"].get("passage")
        if isinstance(inner, str):
            return inner
    return ""


def _extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "q"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(row.get("input"), dict) and isinstance(row["input"].get("question"), str):
        return row["input"]["question"]
    if isinstance(row.get("input"), str):
        return row["input"]
    return ""


def _extract_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "expected", "output", "a"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(row.get("expected"), dict) and isinstance(row["expected"].get("answer"), str):
        return row["expected"]["answer"]
    return ""


register_playbook(RagProtocolParaphrasePlaybook())
