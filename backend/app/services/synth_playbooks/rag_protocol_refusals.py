"""Arc R-1 — REFUSALS playbook for the ``rag-protocol`` recipe.

Generates rows where the question CANNOT be answered from the
provided context, and the answer is the templated refusal phrase. The
model needs to learn that "no relevant context" → "refuse cleanly"
rather than "hallucinate something plausible."

Two refusal flavours are generated:
  1. **Off-topic context** — context is real but unrelated to the
     question (e.g. context covers refunds, question asks about
     shipping address changes).
  2. **No-context / empty** — the context field is empty or a stub
     placeholder; the model must refuse on the structural absence.

Refusal answers must contain the canonical refusal phrase (case-
insensitive substring match in validate()) so the model learns the
exact response shape. Rows that paper over the missing context with
a confident-sounding hallucination are heavily down-scored so review
catches them.
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
from .rag_protocol_paraphrase import _extract_context, _extract_question


# Canonical refusal phrase. The validator does a case-insensitive
# substring match on this OR a small set of close variants. Keep this
# list tight — the protocol's value is that the model emits ONE
# recognisable refusal shape downstream consumers can detect.
REFUSAL_CANONICAL_PHRASE = "I don't have enough context to answer that."
REFUSAL_PHRASE_VARIANTS = (
    "don't have enough context",
    "not enough context",
    "context doesn't cover",
    "cannot answer that based on the provided context",
    "i can't answer that with the available context",
)


_PROMPT_TEMPLATE = """\
You are generating refusal training data for a RAG-protocol fine-tune. The model is being taught to REFUSE cleanly when the retrieved context can't answer the user's question — instead of hallucinating a plausible-sounding answer.

You are given a small set of (context, question, answer) triples that DO answer correctly. Use them as a domain reference.

Your task: generate {total_count} REFUSAL training examples. For each one:

1. Pick or imagine a question about the SAME general domain but whose answer is NOT in any source context above. (E.g. if the context covers refunds, ask about international shipping rules.)
2. Choose between two refusal flavours:
   - **Off-topic context**: paste in one of the real contexts from the source set (verbatim). The model must learn that "context exists but is unrelated" still warrants refusal.
   - **No context**: set context to the empty string "". The model must learn that missing context never produces a confident answer.
3. The answer MUST be the canonical refusal phrase, optionally with one short clarifying sentence that does NOT invent facts:
   {refusal_phrase!r}

Source triples (for domain reference only — do NOT reuse their answers):
{examples_block}

Output exactly {total_count} JSON lines. Each line MUST be a single JSON object with this exact shape:
{{"context": "<verbatim source context OR empty string>", "question": "<question outside this context>", "answer": "I don't have enough context to answer that.<optional: one short clarifying sentence>"}}

No preamble, no JSON array wrapper, no markdown code fences. Just {total_count} JSON lines.
"""


class RagProtocolRefusalsPlaybook:
    recipe_id = "rag-protocol"
    mode = SynthMode.REFUSALS

    def build_prompt(self, ctx: PlaybookContext) -> str:
        target = ctx.get("target_count") or 20
        gold = ctx.get("gold_rows") or []
        examples = sample_gold_rows(gold, count=min(5, max(1, len(gold))), seed=0)

        example_lines: list[str] = []
        for i, row in enumerate(examples, start=1):
            context = _extract_context(row) or "(none)"
            question = _extract_question(row) or "(unknown)"
            example_lines.append(
                f"{i}. CONTEXT: {context!r}\n   Q: {question!r}"
            )

        return _PROMPT_TEMPLATE.format(
            total_count=target,
            refusal_phrase=REFUSAL_CANONICAL_PHRASE,
            examples_block="\n".join(example_lines) or "(no source triples — generate refusals from scratch)",
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
        accepted: list[SynthRow] = []
        for row in parsed_rows:
            context = row.get("context")
            question = row.get("question")
            answer = row.get("answer")
            # The context field MAY be empty (no-context flavour) but
            # it must still be a string. Question + answer must be
            # populated.
            if not isinstance(context, str) or not isinstance(question, str) or not isinstance(answer, str):
                continue
            context, question, answer = context.strip(), question.strip(), answer.strip()
            if not question or not answer:
                continue
            if len(question) < 5 or len(question) > 2000:
                continue
            if len(answer) < 5 or len(answer) > 1200:
                # Refusal answers should be short; >1200 chars almost
                # certainly means the model hallucinated a full
                # response in spite of the refusal instruction.
                continue
            answer_lc = answer.lower()
            has_refusal = any(variant in answer_lc for variant in REFUSAL_PHRASE_VARIANTS)
            confidence = 1.0
            if not has_refusal:
                # The whole point of this mode is the canonical
                # refusal shape — without it, the row is almost
                # certainly a hallucinated answer wearing a refusal
                # costume. Down-score hard so review surfaces these.
                confidence *= 0.30
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


register_playbook(RagProtocolRefusalsPlaybook())
