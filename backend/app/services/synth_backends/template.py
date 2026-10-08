"""Template backend — a deterministic stand-in for a generation / judge
model, for smoke tests and CI (the way ``TRAINING_BACKEND=simulate`` stands
in for a trainer). It writes no language of its own: the documents → Q&A
flow gets question → answer pairs lifted from the passage's sentences, and
the answer judge gets correct / partial / wrong by token overlap with the
reference. Quality is meaningless; the point is that every path that calls
a model runs end to end on a machine without one.

Never auto-picked unless ``BREWSLM_TEMPLATE_SYNTH=1``; always reachable by
name (``backend="template"``, ``EVAL_JUDGE_BACKEND=template``).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .base import SynthBackendError

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PASSAGE_RE = re.compile(r"<passage>\s*(.*?)\s*</passage>", re.DOTALL)
_PAIRS_RE = re.compile(r"Write (\d+) question-and-answer pairs")
_QUESTION_RE = re.compile(r"QUESTION:\s*(.*?)\n\s*\n", re.DOTALL)
_REFERENCE_RE = re.compile(r"REFERENCE ANSWER[^\n]*:\s*(.*?)\n\s*\nMODEL ANSWER", re.DOTALL)
_MODEL_RE = re.compile(r"MODEL ANSWER:\s*(.*?)\n\s*\nVerdict rules", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def overlap(reference: str, prediction: str) -> float:
    """Share of the reference's distinct words present in the prediction."""
    ref, pred = _tokens(reference), _tokens(prediction)
    return len(ref & pred) / len(ref) if ref else 0.0


def judge_verdict(reference: str, prediction: str) -> dict[str, str]:
    score = overlap(reference, prediction)
    verdict = "correct" if score >= 0.6 else "partial" if score >= 0.3 else "wrong"
    return {"verdict": verdict, "reason": f"template judge: {score:.0%} of the reference's words present"}


def passage_pairs(passage: str, pairs_per_passage: int) -> dict[str, Any]:
    """``pairs_per_passage`` training pairs + one eval pair, each a sentence
    of the passage asked about by its opening words."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(passage.strip()) if len(s.strip().split()) >= 4]
    if not sentences:
        sentences = [passage.strip()[:200] or "The passage is empty."]
    items = []
    for index in range(pairs_per_passage + 1):
        sentence = sentences[index % len(sentences)]
        words = sentence.split()
        lead = " ".join(words[:9]).rstrip(",.;:")
        items.append({
            "question": f"What does the passage say about \"{lead}\" (part {index + 1})?",
            "answer": sentence,
        })
    return {"pairs": items[:pairs_per_passage], "eval": items[pairs_per_passage]}


class TemplateBackend:
    name: str = "template"
    schema_aware: bool = True

    @classmethod
    def is_available(cls) -> bool:
        return os.getenv("BREWSLM_TEMPLATE_SYNTH", "").strip() == "1"

    def describe(self) -> str:
        return "template (deterministic smoke backend)"

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict | None = None,
    ) -> str:
        if "REFERENCE ANSWER" in prompt and "MODEL ANSWER" in prompt:
            reference = _REFERENCE_RE.search(prompt)
            model = _MODEL_RE.search(prompt)
            if not reference or not model:
                raise SynthBackendError("template backend: unrecognised judge prompt")
            return json.dumps(judge_verdict(reference.group(1), model.group(1)))
        passage = _PASSAGE_RE.search(prompt)
        if passage:
            count = _PAIRS_RE.search(prompt)
            return json.dumps(passage_pairs(passage.group(1), int(count.group(1)) if count else 3))
        raise SynthBackendError(
            "template backend only answers the documents → Q&A flow and the answer judge; "
            "use a real model for anything else"
        )
