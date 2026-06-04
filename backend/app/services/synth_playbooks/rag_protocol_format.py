"""Arc R-1 — FORMAT_ROBUSTNESS playbook for the ``rag-protocol`` recipe.

Generates question variants that demand the SAME answer format. The
model needs to learn that *how* the user phrases their question
shouldn't shift the response format — answers stay short, cite the
chunk, and follow one templated shape.

Different from POSITIVES_PARAPHRASE: paraphrase varies the question's
wording; format-robustness varies the question's *register* (terse vs.
verbose, polite vs. direct, abbreviated vs. spelled-out) while
holding semantic content constant. The training signal is "this
matrix of input registers all produce the same formatted output."

The validator pins the OUTPUT format. Rows whose answer drifts in
length / structure / citation discipline relative to a stable
template are down-scored so review surfaces them.
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
from .rag_protocol_paraphrase import (
    _CITATION_RE,
    _extract_answer,
    _extract_context,
    _extract_question,
)


_PROMPT_TEMPLATE = """\
You are generating format-robustness training data for a RAG-protocol fine-tune. The model is being taught that the OUTPUT format is invariant — the answer always cites the chunk, stays concise, and follows the same templated structure — even as the INPUT question varies in register (terse vs. verbose, polite vs. direct, telegraphic vs. formal).

You are given a small set of (context, question, answer) triples. For each triple, write {variants_per_row} question variants with deliberately different REGISTERS:
  - one terse (≤6 words, possibly missing articles)
  - one verbose (>20 words, polite phrasing)
  - one direct (imperative, no pleasantries)
  - more if {variants_per_row} > 3

The CONTEXT stays verbatim. The ANSWER stays verbatim (including the ``[#N]`` citation marker). Only the question varies.

Source triples:
{examples_block}

Output exactly {total_count} JSON lines. Each line MUST be a single JSON object:
{{"context": "<verbatim source context>", "question": "<re-registered question>", "answer": "<verbatim source answer with [#N]>"}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines.
"""


class RagProtocolFormatPlaybook:
    recipe_id = "rag-protocol"
    mode = SynthMode.FORMAT_ROBUSTNESS

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 30
        gold = ctx.get("gold_rows") or []
        n_source = max(1, min(len(gold), max(2, target // 4)))
        examples = sample_gold_rows(gold, count=n_source, seed=0)
        variants_per_row = max(3, target // max(1, len(examples)))

        example_lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            context = _extract_context(row)
            question = _extract_question(row)
            answer = _extract_answer(row)
            example_lines.append(
                f"{i}. CONTEXT: {context!r}\n   Q: {question!r}\n   A: {answer!r}"
            )

        return _PROMPT_TEMPLATE.format(
            variants_per_row=variants_per_row,
            examples_block="\n".join(example_lines),
            total_count=variants_per_row * len(examples),
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
        # Build a set of canonical answer strings from the gold so we
        # can detect "answer drifted" cases (the whole point of this
        # mode is invariant output).
        gold = ctx.get("gold_rows") or []
        canonical_answers = {_extract_answer(r) for r in gold if _extract_answer(r)}

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
            if len(question) < 3 or len(question) > 2000:
                # Allow shorter questions than paraphrase mode — the
                # terse variant is explicitly under 6 words.
                continue
            if len(answer) < 1 or len(answer) > 8000:
                continue
            confidence = 1.0
            # The whole point: the answer should match one of the
            # canonical answers verbatim (format invariance). Drift =
            # the model hallucinated a different response for the
            # same input/output pair.
            if canonical_answers and answer not in canonical_answers:
                confidence *= 0.55
            # Citation marker is still the protocol signal — preserve
            # the same down-score as the paraphrase mode.
            if not _CITATION_RE.search(answer):
                confidence *= 0.45
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


# Re-export the citation regex so tests can pin its semantics without
# reaching into the paraphrase module's private namespace.
__all__ = ["RagProtocolFormatPlaybook"]
_ = re  # silence linter if regex never used at runtime in this file

register_playbook(RagProtocolFormatPlaybook())
