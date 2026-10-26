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


_CF_MARKER_RE = re.compile(r"\[brewslm:code-findings:(\w+)\]")
_CF_CHUNK_RE = re.compile(r"<chunk index=\"(\d+)\"[^>]*?start_line=\"(\d+)\"[^>]*>\n(.*?)\n</chunk>", re.DOTALL)
_CF_COUNT_RE = re.compile(r"Output exactly (\d+) JSON lines")
_CF_LABEL_RE = re.compile(r"Target label id: (\S+)")
_CF_ALLOWED_RE = re.compile(r"Allowed label ids: ([^\s,]+)")
_CF_NUMBERED_RE = re.compile(r"^\s*\d+ \| ?")
_CF_KINDS = ("fixed", "safe_lookalike", "test_fixture")


def code_findings_rows(prompt: str, mode: str) -> str:
    """Deterministic teacher output for the `code-findings` playbook prompts (marked
    ``[brewslm:code-findings:<mode>]``). Seeded / cluster: append a line that builds a query by
    string concatenation and report that line. Hard negatives: append a parameterised variant,
    ``findings: []``."""
    chunks = []
    for match in _CF_CHUNK_RE.finditer(prompt):
        code = "\n".join(_CF_NUMBERED_RE.sub("", line, count=1) for line in match.group(3).split("\n"))
        chunks.append((int(match.group(1)), int(match.group(2)), code))
    if not chunks:
        raise SynthBackendError("template backend: no <chunk> blocks in the code-findings prompt")
    count_match = _CF_COUNT_RE.search(prompt)
    count = int(count_match.group(1)) if count_match else len(chunks)
    label_match = _CF_LABEL_RE.search(prompt) or _CF_ALLOWED_RE.search(prompt)
    label = label_match.group(1) if label_match else "unknown"
    out = []
    for i in range(count):
        index, start, code = chunks[i % len(chunks)]
        n_lines = code.count("\n") + 1
        if mode == "hard_negatives":
            line = f'query_{i} = db.prepare("SELECT 1 WHERE id = ?", [user_input])'
            out.append({
                "chunk": index, "negative_kind": _CF_KINDS[i % len(_CF_KINDS)],
                "code": f"{code}\n{line}", "findings": [],
            })
            continue
        line = f'query_{i} = "SELECT 1 WHERE id = " + user_input'
        number = start + n_lines
        out.append({
            "chunk": index, "code": f"{code}\n{line}",
            "finding": {"label": label, "start_line": number, "end_line": number, "evidence": line, "attrs": {}},
        })
    return "\n".join(json.dumps(item) for item in out)


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
            # Names what it asks about, never "the passage" — the flow drops
            # questions that point at the passage.
            "question": f"What is said about \"{lead}\" (part {index + 1})?",
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
        cf = _CF_MARKER_RE.search(prompt)
        if cf:
            return code_findings_rows(prompt, cf.group(1))
        passage = _PASSAGE_RE.search(prompt)
        if passage:
            count = _PAIRS_RE.search(prompt)
            return json.dumps(passage_pairs(passage.group(1), int(count.group(1)) if count else 3))
        raise SynthBackendError(
            "template backend only answers the documents → Q&A flow, the answer judge and the code-findings playbooks; "
            "use a real model for anything else"
        )
