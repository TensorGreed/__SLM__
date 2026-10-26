"""Documents → generated Q&A → answer key → split → train → lift check, as
one background Job.

The legal-assistant case: a user has a pile of documents (case law, policy
manuals) and wants a model that answers questions about them better than the
base model. Every step existed — cleaned passages, synthetic generation,
LLM-assisted answer keys, task-shape confirmation, the split, training, the
automatic lift check — but nothing chained them, so a documents-only project
sat on the training tab with nothing to train on, or trained
question→answer on raw chunks (the MamlaLegal project did exactly that:
7 runs, 4 failed, no measurable result).

What the flow does, in order:

1. Loads the cleaned document passages (``auto_rag_service.load_document_passages``).
2. For each passage (up to ``max_passages``), asks the synthetic-data backend
   (Ollama / configured teacher / cloud, via ``synth_backends.pick_backend``)
   for ``pairs_per_passage`` question→answer pairs grounded in the passage,
   plus ONE more, different, evaluation question on the same passage.
   Training pairs land in the project's synthetic dataset
   (``synth_source="documents_qa_flow"``, accepted — reviewable and
   bulk-droppable in the review queue); the evaluation questions land in the
   answer key (practice set, ``source="documents_qa_flow"``).
3. Confirms the task type as ``qa-sft`` (the project was documents-only).
4. Splits the default training sources (cleaned labelled rows + synthetic —
   document passages are unlabelled and drop out) into train / val / test.
5. Creates a training run with the project defaults and starts it; the
   training watcher Job fires the automatic lift check when it finishes, so
   the Eval tab answers "better than the base model, and by how much?".

The eval questions are *different questions on the same passages* as the
training pairs, not questions on held-out passages: the use case is a model
that knows the corpus, and the train↔gold leakage check (near-duplicate
matching) still guards against copies. Honest limits: the generated pairs
are only as good as the backend model; a 135M student will learn the style
long before the facts; and the lift check's row-level and seed-level noise
notes apply as usual.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import DatasetType
from app.models.project import Project
from app.services import provenance_service
from app.services.llm_budget_guard import LlmBudgetExceeded
from app.services.synth_backends.base import SynthBackend, SynthBackendError

_LOG = logging.getLogger("documents_qa_flow")

FLOW_SOURCE = "documents_qa_flow"
RECIPE_ID = "qa-sft"
MIN_PASSAGES = 5
DEFAULT_MAX_PASSAGES = 60
DEFAULT_PAIRS_PER_PASSAGE = 3
MAX_PASSAGE_CHARS = 2400
# Below this many usable training pairs the split + lift check can't say
# anything; the flow stops before training and says so.
MIN_TRAINING_PAIRS = 40

# ── Sizing the test examples to the decision ──────────────────────────────
#
# The gate and the sweep compare retrievals (and later the fine-tune) by a
# paired per-row judge score on the TEST split. On 19 rows nothing short of
# a huge gap clears noise. The flow therefore sizes generation to the
# decision it has to make: enough test rows that a DECISION_GAP judge-score
# difference clears the 95% interval. With per-row score differences of
# standard deviation DIFF_SD, that needs about (2 * DIFF_SD / DECISION_GAP)^2
# paired rows; judge scores are 0 / 0.5 / 1, so differences spread with
# SD ~0.5 (measured 0.48 on the legal case). 0.15 → 45 rows.
DECISION_GAP = 0.15
DIFF_SD = 0.5
# Share of the generated pairs that becomes test examples. Raised above the
# split's default (0.1) only when the documents can't supply enough passages.
TEST_RATIO = 0.1
MAX_TEST_RATIO = 0.3
MAX_PASSAGES_CAP = 2000
MIN_TEST_ROWS = 5


def target_test_rows(gap: float = DECISION_GAP, diff_sd: float = DIFF_SD) -> int:
    """Paired rows for a ``gap`` in mean judge score to clear a 95% interval."""
    import math

    return max(MIN_TEST_ROWS, math.ceil((2.0 * diff_sd / gap) ** 2))


def spread_passages(passages: list[Any], count: int) -> list[Any]:
    """``count`` passages spread evenly over the whole corpus, in document
    order. The flow used to take the first ``count`` — on a long document
    (a 140-page guide cut into ~1,100 passages, 150 used) every question came
    from its opening chapters, and the answer key measured those alone."""
    total = len(passages)
    if count >= total:
        return list(passages)
    if count <= 0:
        return []
    return [passages[(i * total) // count] for i in range(count)]


def size_for_decision(
    available_passages: int,
    pairs_per_passage: int = DEFAULT_PAIRS_PER_PASSAGE,
    *,
    gap: float = DECISION_GAP,
    diff_sd: float = DIFF_SD,
) -> dict[str, Any]:
    """How many passages to generate from, and what test share to split
    with, so the test split reaches ``target_test_rows``. Prefers more
    passages (coverage) and only raises the test share — up to
    ``MAX_TEST_RATIO`` — when the documents run out of passages."""
    import math

    target = target_test_rows(gap, diff_sd)
    pairs_needed = math.ceil(target / TEST_RATIO)
    passages_needed = math.ceil(pairs_needed / max(1, pairs_per_passage))
    passages = max(1, min(available_passages, passages_needed, MAX_PASSAGES_CAP))
    expected_pairs = passages * pairs_per_passage
    test_ratio = TEST_RATIO
    capped = passages < passages_needed
    if capped and expected_pairs > 0:
        test_ratio = min(MAX_TEST_RATIO, max(TEST_RATIO, target / expected_pairs))
    expected_test_rows = int(expected_pairs * test_ratio)
    return {
        "target_test_rows": target,
        "judge_gap": gap,
        "diff_sd": diff_sd,
        "passages": passages,
        "passages_needed": passages_needed,
        "test_ratio": round(test_ratio, 3),
        "expected_test_rows": expected_test_rows,
        "capped_by_available": capped,
        "reaches_target": expected_test_rows >= target,
    }

ProgressFn = Callable[[float, str], Awaitable[None]]

_SYSTEM_PROMPT = (
    "You write study material from documents. Every answer must be fully "
    "supported by the passage you are given — never add outside facts. "
    "Answer in the passage's own terms, in one to three sentences. Reply "
    "with JSON only."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                "required": ["question", "answer"],
            },
        },
        "eval": {
            "type": "object",
            "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
            "required": ["question", "answer"],
        },
    },
    "required": ["pairs", "eval"],
}


def build_passage_prompt(passage_text: str, pairs_per_passage: int, *, domain_hint: str = "") -> str:
    """The per-passage prompt. ``pairs_per_passage`` training pairs plus one
    evaluation question that asks something the training pairs don't.

    The evaluation question is held to a stricter standard than the training
    pairs, because it becomes an answer-key row a *reader with the passage in
    front of it* is scored against: it must stand on its own (no "this
    subsection" / "the passage"), and its answer must be something that
    reader can quote — the passage's own complete words, nothing selected
    from or added to them. A passage is a chunk and may start or end
    mid-sentence; a cut-off fragment is no basis for a question."""
    hint = f" The documents are about: {domain_hint.strip()}." if domain_hint.strip() else ""
    return (
        f"Read this passage.{hint}\n\n"
        f"<passage>\n{passage_text.strip()[:MAX_PASSAGE_CHARS]}\n</passage>\n\n"
        + _passage_instructions(pairs_per_passage)
    )


def _passage_instructions(pairs_per_passage: int) -> str:
    return (
        f"Write {pairs_per_passage} question-and-answer pairs a reader might ask about this passage, "
        "each answerable from the passage alone. Vary what they ask about (who, what, when, why, how much, "
        "which rule applies). Then write ONE more evaluation question about the same passage that none of "
        "the pairs already asks, with its answer.\n\n"
        "Every question — the pairs and the evaluation question — will be asked by someone who has never seen "
        "this passage, so each must stand on its own: name the thing you are asking about (the section, the "
        "rule, the form, the person) and never refer to the passage itself — no \"according to the passage\", "
        "\"this passage\", \"the text\", \"this subsection\", \"mentioned above\" or \"listed here\". "
        "Write \"What is the deadline for filing an appeal?\", not \"What is the deadline according to the "
        "passage?\".\n\n"
        "The evaluation question will be put to a model that has this passage in front of it, so:\n"
        "- its answer must be quotable: the passage's own complete statement of the fact, in the passage's "
        "words — not a selection of some items from a list, not a summary, nothing the passage does not say;\n"
        "- the passage may begin or end in the middle of a sentence; never build the evaluation question on a "
        "cut-off fragment.\n\n"
        'Return JSON of the form {"pairs": [{"question": "...", "answer": "..."}, ...], '
        '"eval": {"question": "...", "answer": "..."}} and nothing else.'
    )


def passage_prompt_parts(passage_text: str, pairs_per_passage: int, *, domain_hint: str = "") -> tuple[str, str]:
    """``(stable, varying)`` for a backend with prompt caching: the
    instructions (identical for every passage of a run) go first as the
    cached block, the passage follows. Same instructions as
    ``build_passage_prompt``."""
    hint = f" The documents are about: {domain_hint.strip()}." if domain_hint.strip() else ""
    stable = f"You will be sent one passage at a time.{hint}\n\n" + _passage_instructions(pairs_per_passage)
    varying = f"Read this passage.\n\n<passage>\n{passage_text.strip()[:MAX_PASSAGE_CHARS]}\n</passage>"
    return stable, varying


# ── Is the evaluation pair quotable? ─────────────────────────────────────
#
# A deterministic gate behind the prompt: an answer-key row a reader can't
# satisfy from the passage measures the key, not the model. Two checks:
# the question must not point at the passage or the prompt's framing, and
# the answer's content words must mostly occur in the passage.
_DEICTIC_RE = re.compile(
    r"\b(this|the|that|the above|the given|the provided)\s+(passage|text|excerpt|extract|paragraph|subsection|"
    r"snippet|document|quote|selection)\b|\b(mentioned|listed|described|provided|given|stated)\s+(above|here|in the "
    r"(passage|text|excerpt|document))\b|\baccording to the (passage|text|excerpt)\b",
    flags=re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'’.-]*", flags=re.IGNORECASE)
_SUPPORT_STOPWORDS = frozenset(
    "the a an and or of to in on for by with as at from that this these those is are was were be been being it its "
    "their there which who whom whose what when where why how must shall may can will would should not no any all "
    "such than then into under over between within without only also each every other more most if unless where "
    "including include includes".split()
)
# Share of the answer's content words that must appear in the passage.
EVAL_ANSWER_SUPPORT_MIN = 0.7


def _content_words(text: str) -> list[str]:
    words = [w.lower().strip(".'’") for w in _WORD_RE.findall(text or "")]
    return [w for w in words if len(w) >= 3 and w not in _SUPPORT_STOPWORDS]


def answer_support(answer: str, passage_text: str) -> float | None:
    """Share of the answer's content words present in the passage (None
    when the answer has no content words)."""
    words = _content_words(answer)
    if not words:
        return None
    present = set(_content_words(passage_text))
    return round(sum(1 for w in words if w in present) / len(words), 4)


def eval_pair_rejection(eval_pair: dict[str, str] | None, passage_text: str) -> str | None:
    """Why an evaluation pair should NOT become an answer-key row: ``"deictic"``
    (the question points at the passage / the prompt), ``"unsupported"``
    (fewer than ``EVAL_ANSWER_SUPPORT_MIN`` of the answer's content words are
    in the passage). None when it is quotable."""
    if not eval_pair:
        return None
    if _DEICTIC_RE.search(eval_pair.get("question") or ""):
        return "deictic"
    support = answer_support(eval_pair.get("answer") or "", passage_text)
    if support is not None and support < EVAL_ANSWER_SUPPORT_MIN:
        return "unsupported"
    return None


def is_deictic(question: str) -> bool:
    """The question points at the passage / the prompt instead of naming
    what it asks about."""
    return bool(_DEICTIC_RE.search(question or ""))


# A pure attribution — "…, according to the passage?" / "According to this
# text, …" — adds nothing a reader needs: the question means the same
# without it, so it is cut by code. Any other reference to the passage
# ("What does the passage define as …", "Which states are listed in the
# text?") carries meaning and goes to the model to rewrite.
_PASSAGE_NOUN = r"(?:the|this|that|the given|the provided)\s+(?:passage|text|excerpt|extract|document|paragraph|snippet)"
_ATTRIBUTION_TAIL_RE = re.compile(
    r",?\s+(?:according to|as (?:stated|described|explained|given) in)\s+" + _PASSAGE_NOUN + r"\s*(?=[?.!]?\s*$)",
    flags=re.IGNORECASE,
)
_ATTRIBUTION_HEAD_RE = re.compile(
    r"^\s*(?:according to|based on|as (?:stated|described) in)\s+" + _PASSAGE_NOUN + r"\s*,\s*",
    flags=re.IGNORECASE,
)


def strip_passage_reference(question: str) -> str:
    """``question`` without a pure attribution to the passage (unchanged
    when it has none)."""
    original = (question or "").strip()
    stripped = _ATTRIBUTION_TAIL_RE.sub("", _ATTRIBUTION_HEAD_RE.sub("", original)).strip()
    if stripped == original or len(stripped) < 8:
        return original
    stripped = stripped[0].upper() + stripped[1:]
    if not stripped.endswith(("?", ".", "!")):
        stripped += "?"
    return stripped


_REWRITE_SYSTEM_PROMPT = "You rewrite questions so they stand on their own. Reply with JSON only."
_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}


def build_rewrite_prompt(passage_text: str, questions: list[str]) -> str:
    """Ask the generation model to rewrite questions that point at the
    passage so they name what they ask about — same meaning, same answer."""
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    return (
        "Each question below was written about this passage but refers to the passage instead of naming what "
        "it asks about. Rewrite each one so it stands on its own for someone who has never seen the passage: "
        "name the thing it asks about (the rule, the form, the person, the event), keep its meaning, so the same "
        "answer still answers it. Never mention the passage, the text, the excerpt or the document.\n\n"
        f"<passage>\n{passage_text.strip()[:MAX_PASSAGE_CHARS]}\n</passage>\n\n"
        f"Questions:\n{numbered}\n\n"
        'Return JSON of the form {"questions": ["...", ...]} with one rewritten question per question above, '
        "in the same order, and nothing else."
    )


def parse_rewrite_output(raw: str | None, count: int) -> list[str | None]:
    """One rewritten question per input (None where the rewrite is unusable:
    missing, too short, or still pointing at the passage). A reply with the
    wrong number of questions is unusable as a whole — the order can't be
    trusted."""
    payload = _extract_json(raw or "")
    items = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) != count:
        return [None] * count
    out: list[str | None] = []
    for item in items:
        text = str(item or "").strip()
        out.append(text if len(text) >= 8 and not is_deictic(text) else None)
    return out


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _clean_pair(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    question = str(item.get("question") or item.get("q") or "").strip()
    answer = str(item.get("answer") or item.get("a") or "").strip()
    if len(question) < 8 or len(answer) < 2:
        return None
    return {"question": question, "answer": answer}


def parse_passage_output(raw: str, pairs_per_passage: int) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    """``(training_pairs, eval_pair)`` from the backend's reply. Drops pairs
    whose question repeats one already kept (case-insensitive) and an eval
    question that matches a training question — the eval question must be
    new. Never raises; an unusable reply yields ``([], None)``."""
    payload = _extract_json(raw)
    if not isinstance(payload, dict):
        return [], None
    seen: set[str] = set()
    pairs: list[dict[str, str]] = []
    for item in payload.get("pairs") if isinstance(payload.get("pairs"), list) else []:
        pair = _clean_pair(item)
        if pair is None:
            continue
        key = pair["question"].lower()
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pair)
        if len(pairs) >= pairs_per_passage:
            break
    eval_pair = _clean_pair(payload.get("eval"))
    if eval_pair is not None and eval_pair["question"].lower() in seen:
        eval_pair = None
    return pairs, eval_pair


@dataclass
class FlowSummary:
    project_id: int
    passages_total: int
    passages_used: int
    passages_failed: int
    training_pairs: int
    answer_key_rows: int
    backend: str
    recipe_id: str = RECIPE_ID
    split: dict[str, Any] | None = None
    experiment_id: int | None = None
    training: dict[str, Any] | None = None
    stopped_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    # Evaluation pairs the quotable gate dropped, by reason (deictic / unsupported).
    eval_rejected: dict[str, int] = field(default_factory=dict)
    # Questions that pointed at the passage and were repaired: a pure
    # attribution cut by code ("stripped"), the rest rewritten by the
    # generation model ("rewritten").
    questions_repaired: dict[str, int] = field(default_factory=dict)
    # Training pairs dropped because their question still pointed at the
    # passage after the repair, by reason.
    pairs_rejected: dict[str, int] = field(default_factory=dict)
    # The pre-training gate: what the base model + passages gets right.
    passages_gate: dict[str, Any] | None = None
    # The run was trained to answer from the retrieved passages.
    train_with_passages: bool = False
    # The teacher step (answers_from_teacher): counts, filter, usage.
    teacher: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "project_id": self.project_id,
            "passages_total": self.passages_total,
            "passages_used": self.passages_used,
            "passages_failed": self.passages_failed,
            "training_pairs": self.training_pairs,
            "answer_key_rows": self.answer_key_rows,
            "eval_rejected": dict(self.eval_rejected),
            "questions_repaired": dict(self.questions_repaired),
            "pairs_rejected": dict(self.pairs_rejected),
            "backend": self.backend,
            "recipe_id": self.recipe_id,
            "split": self.split,
            "experiment_id": self.experiment_id,
            "training": self.training,
            "stopped_reason": self.stopped_reason,
            "warnings": list(self.warnings),
            "passages_gate": self.passages_gate,
            "train_with_passages": self.train_with_passages,
            "teacher": self.teacher,
        }
        # Cloud spend of the generation + teacher calls: batch calls are not
        # seen by the Job's meter, so the flow reports its own usage list.
        usage = self._usage()
        if usage:
            out["usage"] = usage
            out["cost_usd"] = round(sum(u.get("usd") or 0 for u in usage), 4)
            out["tokens"] = sum(int(u.get("total_tokens") or 0) for u in usage)
        return out

    _llms: list[Any] = field(default_factory=list, repr=False)

    def _usage(self) -> list[dict[str, Any]] | None:
        from app.services.span_relabel_service import usage_block

        seen: list[Any] = []
        for llm in self._llms:
            if llm is not None and all(llm is not s for s in seen):
                seen.append(llm)
        block = usage_block(*seen) if seen else []
        return block or None


async def preview_documents_qa_flow(
    db: AsyncSession, project_id: int, *, backend: str | None = None, decision_gap: float = DECISION_GAP,
) -> dict[str, Any]:
    """Can this project run the flow, and what would it cost? Read-only.

    ``backend``: the generation model the user picked. A cloud pin
    (``cloud:<provider>:<model>``) is usable when the project has a key saved
    for the provider — no local model needed; otherwise the first reachable
    registered backend is the one the flow would use."""
    from app.services.auto_rag_service import load_document_passages
    from app.services.cloud_backend_pins import (
        CloudPinError,
        missing_key_message,
        parse_cloud_pin,
        saved_cloud_key,
    )
    from app.services.synth_backends import BACKEND_REGISTRY

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    # eval-only / retrieval-only documents never seed training pairs (they stay in the index).
    passages = provenance_service.drop_excluded_passages(
        load_document_passages(project_id), await provenance_service.training_excluded_document_ids(db, project_id)
    )
    backend_name: str | None = None
    backend_blocker: str | None = None
    try:
        cloud = parse_cloud_pin(backend)
    except CloudPinError as exc:
        cloud = None
        backend_blocker = exc.message
    if cloud is not None:
        if await saved_cloud_key(db, project_id, cloud[0]):
            backend_name = str(backend)
        else:
            backend_blocker = missing_key_message(cloud[0])
    elif backend_blocker is None:
        for cls in BACKEND_REGISTRY:
            if getattr(cls, "auto_pick", True) is False:
                continue  # explicit-pin-only backends (the command CLI) are never defaulted
            try:
                if cls.is_available():
                    backend_name = cls.name
                    break
            except Exception:  # noqa: BLE001
                continue
    existing_qa = await _count_flow_rows(project_id)
    recipe_id = str((project.selected_recipe or {}).get("recipe_id") or "")
    blockers: list[str] = []
    if len(passages) < MIN_PASSAGES:
        blockers.append(
            f"Needs at least {MIN_PASSAGES} cleaned document passages (found {len(passages)}). "
            "Import documents and run cleaning first."
        )
    if backend_blocker is not None:
        blockers.append(backend_blocker)
    elif backend_name is None:
        blockers.append(
            "No generation model is reachable. Start Ollama with a chat model, set "
            "TEACHER_MODEL_API_URL to an OpenAI-compatible endpoint, or pick a cloud "
            "model whose API key is saved on the project."
        )
    sizing = size_for_decision(len(passages), DEFAULT_PAIRS_PER_PASSAGE, gap=decision_gap)
    max_passages = sizing["passages"]
    return {
        "project_id": project_id,
        "eligible": not blockers,
        "blockers": blockers,
        "passages": len(passages),
        "backend": backend_name,
        "current_recipe_id": recipe_id or None,
        "already_generated": existing_qa,
        "plan": {
            "max_passages": max_passages,
            "pairs_per_passage": DEFAULT_PAIRS_PER_PASSAGE,
            "estimated_training_pairs": max_passages * DEFAULT_PAIRS_PER_PASSAGE,
            "estimated_answer_key_rows": max_passages,
            "llm_calls": max_passages,
            # Why that many: the test split must let a DECISION_GAP judge-score
            # difference clear noise (sizing above).
            "sized_for": sizing,
        },
    }


def _active_split_ratios(project_id: int) -> float | None:
    """The test ratio of the project's active prepared split, if any."""
    path = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ratios = manifest.get("ratios") if isinstance(manifest, dict) else None
    test = (ratios or {}).get("test") if isinstance(ratios, dict) else None
    return float(test) if isinstance(test, (int, float)) and 0 < float(test) < 1 else None


def _enable_batch(llm: Any) -> None:
    """Ask a backend for batched calls (Anthropic Message Batches, or the command
    backend's packed calls); ``ValueError`` when it can't."""
    if not getattr(llm, "supports_batch", False) or not hasattr(llm, "complete_batch"):
        label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "backend")
        raise ValueError(
            f"use_batch needs an Anthropic cloud backend (cloud:anthropic:<model>) or the command backend; {label} can't batch."
        )
    llm.use_batch = True


def _generation_call(passage_text: str, pairs_per_passage: int, domain_hint: str, llm: Any) -> dict[str, Any]:
    if getattr(llm, "supports_prompt_cache", False):
        # A cloud backend: the per-run instructions are the cached block,
        # only the passage varies between calls.
        stable, varying = passage_prompt_parts(passage_text, pairs_per_passage, domain_hint=domain_hint)
        extra: dict[str, Any] = {"cache_prefix": stable}
    else:
        varying = build_passage_prompt(passage_text, pairs_per_passage, domain_hint=domain_hint)
        extra = {}
    return {"prompt": varying, "system_prompt": _SYSTEM_PROMPT, "max_tokens": 1600, "temperature": 0.4, **extra}


REPLY_CHECKPOINT_FILE = "generation_checkpoint.jsonl"


def reply_checkpoint_path(project_id: int) -> Path:
    """Where a running flow appends each passage's raw reply (removed once
    the pairs are saved)."""
    return settings.DATA_DIR / "projects" / str(project_id) / "flows" / REPLY_CHECKPOINT_FILE


def _reply_key(backend_label: str, call: dict[str, Any]) -> str:
    """A reply is reusable only for the same backend and the same request."""
    raw = json.dumps({"backend": backend_label, "call": call}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _load_reply_checkpoint(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by the crash
            if isinstance(row, dict) and row.get("key") and isinstance(row.get("raw"), str):
                out[str(row["key"])] = row["raw"]
    return out


def _append_reply(path: Path, key: str, raw: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def _generate_replies(
    llm: Any,
    chosen: list[dict[str, Any]],
    pairs_per_passage: int,
    domain_hint: str,
    report: Callable[[float, str], Awaitable[None]],
    backend_label: str,
    *,
    concurrency: int = 1,
    checkpoint: Path | None = None,
) -> list[str | None]:
    """One reply per passage (None where the call failed). With the backend's
    ``use_batch`` on, every call goes as ONE Message Batch first; requests it
    didn't answer are retried directly.

    With ``checkpoint``, every reply is appended to that file as it arrives
    and replies already there (same backend + same prompt) are reused, so a
    run killed mid-way (a crash, a power cut, a backend restart) resumes
    instead of paying for every passage again."""
    calls = [_generation_call(p["text"], pairs_per_passage, domain_hint, llm) for p in chosen]
    return await _complete_calls(
        llm, calls, report, backend_label,
        ids=[p.get("id") for p in chosen], schema=_RESPONSE_SCHEMA, what="Writing questions",
        start=0.05, width=0.55, concurrency=concurrency, checkpoint=checkpoint,
    )


async def _complete_calls(
    llm: Any,
    calls: list[dict[str, Any]],
    report: Callable[[float, str], Awaitable[None]],
    backend_label: str,
    *,
    ids: list[Any],
    schema: dict[str, Any],
    what: str,
    start: float,
    width: float,
    concurrency: int = 1,
    checkpoint: Path | None = None,
) -> list[str | None]:
    """One reply per call, one call per passage (None where it failed):
    batch first when the backend batches, then direct calls ``concurrency``
    at a time; every reply checkpointed (see ``_generate_replies``)."""
    from app.services.span_relabel_service import uses_message_batch

    keys = [_reply_key(backend_label, call) for call in calls]
    saved = _load_reply_checkpoint(checkpoint) if checkpoint is not None else {}
    raws: list[str | None] = [saved.get(key) for key in keys]
    resumed = sum(1 for raw in raws if raw is not None)
    if resumed:
        await report(start, f"Resuming: {resumed}/{len(calls)} passages already written ({backend_label})")
    if calls and resumed < len(calls) and uses_message_batch(llm):
        async def _progress(message: str, done: int, total: int) -> None:
            await report(start + width * 0.9 * done / max(1, total), f"{what} for {total} passages ({backend_label}): {message}")

        todo = [i for i, raw in enumerate(raws) if raw is None]
        try:
            batch_raws = await llm.complete_batch([calls[i] for i in todo], progress=_progress)
        except SynthBackendError as exc:
            _LOG.warning("documents_qa_flow: message batch failed, generating directly: %s", exc)
            batch_raws = [None] * len(todo)
        for i, raw in zip(todo, batch_raws):
            raws[i] = raw
            if raw is not None and checkpoint is not None:
                _append_reply(checkpoint, keys[i], raw)
    import asyncio

    # Direct calls, ``concurrency`` in flight at once (a local server with
    # parallel slots, or a cloud API).
    pending = [i for i, raw in enumerate(raws) if raw is None]
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    done = {"n": 0}
    at_once = f", {concurrency} at a time" if concurrency > 1 else ""

    async def _one(index: int) -> None:
        async with sem:
            args = dict(calls[index])
            prompt = args.pop("prompt")
            try:
                raws[index] = await llm.complete(prompt, **args, response_schema=schema)
            except SynthBackendError as exc:
                _LOG.warning("documents_qa_flow passage %s failed (%s): %s", ids[index], what, exc)
                raws[index] = None
            if raws[index] is not None and checkpoint is not None:
                _append_reply(checkpoint, keys[index], raws[index])
            done["n"] += 1
            resumed_note = f", {resumed} resumed" if resumed else ""
            await report(
                start + width * done["n"] / max(1, len(pending)),
                f"{what}: {done['n']}/{len(pending)} passages done ({backend_label}{at_once}{resumed_note})",
            )

    await asyncio.gather(*(_one(i) for i in pending))
    return raws


def _rewrite_call(passage_text: str, questions: list[str]) -> dict[str, Any]:
    return {
        "prompt": build_rewrite_prompt(passage_text, questions),
        "system_prompt": _REWRITE_SYSTEM_PROMPT,
        "max_tokens": 120 * len(questions) + 200,
        "temperature": 0.2,
    }


async def repair_deictic_questions(
    llm: Any,
    parsed: list[tuple[dict[str, Any], list[dict[str, str]], dict[str, str] | None]],
    report: Callable[[float, str], Awaitable[None]],
    backend_label: str,
    *,
    concurrency: int = 1,
    checkpoint: Path | None = None,
) -> dict[str, int]:
    """Repair, in place, every question in ``parsed`` ((passage, pairs,
    eval_pair) per passage) that points at the passage: a pure attribution
    is cut by code; the rest go back to ``llm`` in one rewrite call per
    passage. A question the rewrite can't fix is left as it was (the caller
    drops it). Returns ``{"stripped": n, "rewritten": n}``."""
    counts = {"stripped": 0, "rewritten": 0}
    todo: list[tuple[int, list[dict[str, str]]]] = []
    for index, (_passage, pairs, eval_pair) in enumerate(parsed):
        items = list(pairs) + ([eval_pair] if eval_pair is not None else [])
        still: list[dict[str, str]] = []
        for item in items:
            if not is_deictic(item["question"]):
                continue
            fixed = strip_passage_reference(item["question"])
            if fixed != item["question"] and not is_deictic(fixed):
                item["question"] = fixed
                counts["stripped"] += 1
            else:
                still.append(item)
        if still:
            todo.append((index, still))
    if not todo or llm is None:
        return counts
    calls = [_rewrite_call(parsed[i][0]["text"], [item["question"] for item in items]) for i, items in todo]
    raws = await _complete_calls(
        llm, calls, report, backend_label,
        ids=[parsed[i][0].get("id") for i, _ in todo], schema=_REWRITE_SCHEMA,
        what="Rewriting questions that point at the passage", start=0.60, width=0.02,
        concurrency=concurrency, checkpoint=checkpoint,
    )
    for (_index, items), raw in zip(todo, raws):
        for item, rewritten in zip(items, parse_rewrite_output(raw, len(items))):
            if rewritten is not None:
                item["question"] = rewritten
                counts["rewritten"] += 1
    return counts


def grow_test_set(
    kept_test: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure: the answer-key rows to add to ``kept_test`` so it reaches
    ``target`` rows. Candidates whose question is already a test question
    are skipped; passages that already back a test row are used last; the
    rest are spread evenly over the documents (``spread_passages``, in
    document order). Returns ``(additions, report)``."""
    def _norm(text: Any) -> str:
        return " ".join(str(text or "").split()).lower()

    have = {_norm(r.get("question")) for r in kept_test}
    used_passages = {(r.get("source_document_id"), r.get("chunk_id")) for r in kept_test}
    need = max(0, int(target) - len(kept_test))
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        q = _norm(row.get("question"))
        if not q or q in have or q in seen:
            continue
        seen.add(q)
        pool.append(row)
    order = lambda r: (str(r.get("source_document_id") or ""), int(r.get("chunk_id") or 0) if str(r.get("chunk_id") or "0").lstrip("-").isdigit() else 0)  # noqa: E731
    fresh = sorted((r for r in pool if (r.get("source_document_id"), r.get("chunk_id")) not in used_passages), key=order)
    reused = sorted((r for r in pool if (r.get("source_document_id"), r.get("chunk_id")) in used_passages), key=order)
    additions = spread_passages(fresh, need) if need else []
    if len(additions) < need:
        additions += spread_passages(reused, need - len(additions))
    return additions, {
        "target": int(target),
        "kept": len(kept_test),
        "added": len(additions),
        "candidates": len(pool),
        "from_new_passages": sum(1 for r in additions if (r.get("source_document_id"), r.get("chunk_id")) not in used_passages),
        "reaches_target": len(kept_test) + len(additions) >= int(target),
    }


def _used_local_ollama(summary: "FlowSummary") -> bool:
    """Did this flow run a model in the local Ollama (generation, teacher,
    critic, or the teacher filter's judge)?"""
    from app.services.synth_backends.ollama import OllamaBackend

    if any(isinstance(llm, OllamaBackend) for llm in summary._llms):
        return True
    judge = ((summary.teacher or {}).get("filter") or {}).get("judge")
    return str(judge or "").startswith("ollama")


def _answer_key_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row.get(k) for k in ("question", "answer", "source_doc", "source_document_id", "chunk_id", "generated_by")}
    out["answer_key_source"] = FLOW_SOURCE
    return out


def _read_jsonl_rows(path: Any) -> list[dict[str, Any]]:
    from pathlib import Path as _Path

    path = _Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


async def _count_gold_flow_rows(db: AsyncSession, project_id: int) -> int:
    """Answer-key rows the flow wrote (GOLD_DEV rows with the flow's source)."""
    from app.services.gold_service import get_gold_entries

    try:
        rows = await get_gold_entries(db, project_id, DatasetType.GOLD_DEV)
    except Exception:  # noqa: BLE001
        return 0
    return sum(1 for row in rows if str(row.get("source") or "") == FLOW_SOURCE)


async def _count_flow_rows(project_id: int) -> int:
    path = settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if f'"synth_source": "{FLOW_SOURCE}"' in line:
                count += 1
    return count


async def run_documents_qa_flow(
    db: AsyncSession,
    project_id: int,
    *,
    max_passages: int | None = None,
    pairs_per_passage: int = DEFAULT_PAIRS_PER_PASSAGE,
    backend: str | None = None,
    backend_override: SynthBackend | None = None,
    train: bool = True,
    reuse_existing: bool = False,
    passages_check: bool = True,
    train_if_retrieval_ready: bool = False,
    train_with_passages: bool = False,
    passages_withheld_fraction: float | None = None,
    decision_gap: float = DECISION_GAP,
    keep_test: bool = False,
    sweep_split: str = "val",
    answers_from_teacher: bool = False,
    teacher_backend: str | None = None,
    teacher_backend_override: SynthBackend | None = None,
    teacher_filter: bool = True,
    teacher_filter_keep: str = "not_wrong",
    teacher_filter_judge: Any = None,
    teacher_revise_rounds: int = 0,
    teacher_critic_backend: str | None = None,
    teacher_critic_backend_override: SynthBackend | None = None,
    teacher_retrieve_fn: Callable[..., list[dict[str, Any]]] | None = None,
    use_batch: bool = False,
    concurrency: int = 1,
    progress: ProgressFn | None = None,
    start_training: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    passages_check_fn: Callable[..., dict[str, Any]] | None = None,
) -> FlowSummary:
    """Run the flow. ``train=False`` stops after the split (tests, and users
    who want to review the generated rows first). ``reuse_existing`` skips
    generation when the project already holds flow-generated pairs — the
    same pairs, answer key and split, trained again (e.g. on another base
    model), so the two lift checks are comparable. ``start_training`` is the
    launcher used for the training run (defaults to the API's start, which
    also spawns the watcher Job that runs the lift check).

    ``passages_check`` (default on): after the split and BEFORE training, score
    the base model with and without the document passages on the test
    examples, judged (``passage_rag_verdict_service.passages_gate``). When
    retrieval alone already answers well the flow stops there — training is
    the long way round — unless ``train_if_retrieval_ready``. The gate lands
    in ``summary.passages_gate`` either way. ``passages_check_fn`` injects
    the comparison (tests); the default runs the auto-RAG harness.

    ``train_with_passages``: train the model to answer FROM the retrieved
    passages (``passage_prompts_service``: each pair's prompt becomes the
    served retrieval prompt; ``passages_withheld_fraction`` of the rows get
    their answer passage removed and "the passages don't say" as target).
    Its lift check then scores it with the passages against the base model
    + the same passages. With ``reuse_existing`` it trains the same pairs on
    the same split.

    ``decision_gap``: the judge-score difference the test examples must be
    able to resolve (``size_for_decision``; 0.15 → 45 rows, 0.08 → 157).

    ``keep_test``: keep the active test examples verbatim (pre-registered
    rows stay comparable) and grow them to ``target_test_rows(decision_gap)``
    with this flow's answer-key questions, spread over the documents
    (``grow_test_set``); validation is sized to the same target; train / val
    questions matching any test question (exact or near-identical) are
    dropped and counted (``split.keep_test.question_leakage``).

    ``sweep_split``: the rows the pre-training gate's retrieval sweep picks
    on — ``"val"`` (default): the sweep chooses on validation, then the
    chosen retrieval is scored ONCE on the test examples (the gate reads
    that); ``"test"``: the older behaviour (sweep and gate on test).

    ``answers_from_teacher`` (implies ``train_with_passages``, no withheld
    rows): after the gate, a teacher backend (``teacher_backend`` pin /
    ``teacher_backend_override``, else the generation backend) answers every
    train / val question from exactly the passages the served retrieval
    returns (``teacher_passage_answers_service``); those answers are the
    targets. ``teacher_filter``: drop training answers the local answer judge
    grades wrong against the pair's reference; ``teacher_filter_keep``
    ``"not_wrong"`` (default) keeps partial answers, ``"correct"`` keeps only
    the ones graded correct. ``teacher_revise_rounds`` (0–2): answers the
    filter drops are critiqued and revised by the teacher, then re-judged
    (``teacher_passage_answers_service.critique_and_revise``); the critique
    comes from ``teacher_critic_backend`` (pin / name, or
    ``teacher_critic_backend_override``), else from the filter verdict's own
    reason; needs ``teacher_filter``. ``use_batch``: send the
    generation and the teacher calls as Anthropic Message Batches.
    ``concurrency`` (1–8): direct generation / teacher calls in flight at once
    (Ollama with ``OLLAMA_NUM_PARALLEL``, or a cloud API)."""
    from app.services.auto_rag_service import load_document_passages
    from app.services.synth_backends import pick_backend

    async def report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    sizing: dict[str, Any] | None = None
    if max_passages is None:
        # Size generation to the decision (see size_for_decision).
        sizing = None  # resolved once the passages are counted
    elif max_passages < 1 or max_passages > MAX_PASSAGES_CAP:
        raise ValueError(f"max_passages must be between 1 and {MAX_PASSAGES_CAP}")
    if pairs_per_passage < 1 or pairs_per_passage > 10:
        raise ValueError("pairs_per_passage must be between 1 and 10")
    if not (0.01 <= float(decision_gap) <= 1.0):
        raise ValueError("decision_gap must be between 0.01 and 1")
    if sweep_split not in {"val", "test"}:
        raise ValueError("sweep_split must be 'val' or 'test'")
    if answers_from_teacher:
        train_with_passages = True
    teacher_revise_rounds = int(teacher_revise_rounds or 0)
    if teacher_revise_rounds and not teacher_filter:
        raise ValueError(
            "teacher_revise_rounds needs teacher_filter: the filter judge's verdicts say which answers to revise."
        )

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    passages = provenance_service.drop_excluded_passages(
        load_document_passages(project_id), await provenance_service.training_excluded_document_ids(db, project_id)
    )
    if len(passages) < MIN_PASSAGES:
        raise ValueError(
            f"Needs at least {MIN_PASSAGES} cleaned document passages (found {len(passages)}). "
            "Import documents and run cleaning first."
        )
    if max_passages is None:
        sizing = size_for_decision(len(passages), pairs_per_passage, gap=decision_gap)
        max_passages = sizing["passages"]
    test_ratio = sizing["test_ratio"] if sizing else TEST_RATIO
    existing_pairs = await _count_flow_rows(project_id) if reuse_existing else 0
    if existing_pairs:
        # Same pairs, same split: a re-train keeps the ratios of the active
        # prepared split so the two lift checks score the same rows.
        previous = _active_split_ratios(project_id)
        if previous is not None:
            test_ratio = previous
            sizing = None
    if reuse_existing and not existing_pairs:
        raise ValueError("No flow-generated pairs to reuse; run the flow without reuse_existing first.")
    if existing_pairs:
        llm = None
        backend_label = f"reused {existing_pairs} generated pairs"
    else:
        llm = backend_override or pick_backend(backend)
        backend_label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "backend")
        if use_batch:
            _enable_batch(llm)
    domain_hint = str(project.description or project.name or "").strip()[:200]

    # ── 1 + 2: generate per passage ───────────────────────────────────────
    chosen = spread_passages(passages, max_passages) if llm is not None else []
    training_pairs: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    failed = 0
    empty_replies = 0
    # Evaluation pairs the quotable gate dropped, by reason.
    eval_rejected: dict[str, int] = {}
    if not 1 <= int(concurrency) <= 8:
        raise ValueError("concurrency must be between 1 and 8")
    checkpoint = reply_checkpoint_path(project_id)
    raws = await _generate_replies(
        llm, chosen, pairs_per_passage, domain_hint, report, backend_label,
        concurrency=int(concurrency), checkpoint=checkpoint,
    )
    parsed: list[tuple[dict[str, Any], list[dict[str, str]], dict[str, str] | None]] = []
    for index, passage in enumerate(chosen):
        raw = raws[index]
        if raw is None:
            failed += 1
            continue
        pairs, eval_pair = parse_passage_output(raw, pairs_per_passage)
        if not pairs and eval_pair is None:
            failed += 1
            if not str(raw or "").strip():
                empty_replies += 1
            else:
                _LOG.warning("documents_qa_flow passage %s: unparseable reply: %r", passage["id"], str(raw)[:200])
            continue
        parsed.append((passage, pairs, eval_pair))
    # A question that points at the passage ("… according to the passage?")
    # is unanswerable as asked by a user who never saw it — repaired before
    # any row is kept (training pairs too, not only the answer key).
    questions_repaired = await repair_deictic_questions(
        llm, parsed, report, backend_label, concurrency=int(concurrency), checkpoint=checkpoint,
    )
    pairs_rejected: dict[str, int] = {}
    for passage, pairs, eval_pair in parsed:
        provenance = {
            "source_doc": passage.get("source_doc"),
            "source_document_id": passage.get("source_document_id"),
            "chunk_id": passage.get("chunk_id"),
            # Which model wrote the question / answer (model-card provenance).
            "generated_by": backend_label,
        }
        for pair in pairs:
            if is_deictic(pair["question"]):
                pairs_rejected["deictic"] = pairs_rejected.get("deictic", 0) + 1
                continue
            training_pairs.append({**pair, **provenance})
        if eval_pair is not None:
            reason = eval_pair_rejection(eval_pair, passage["text"])
            if reason is None:
                answer_key.append({**eval_pair, **provenance})
            else:
                eval_rejected[reason] = eval_rejected.get(reason, 0) + 1
                _LOG.info("documents_qa_flow passage %s: eval question dropped (%s): %r", passage["id"], reason, eval_pair["question"][:120])

    summary = FlowSummary(
        project_id=project_id,
        passages_total=len(passages),
        passages_used=len(chosen) - failed,
        passages_failed=failed,
        training_pairs=len(training_pairs),
        answer_key_rows=len(answer_key),
        backend=backend_label,
        eval_rejected=dict(eval_rejected),
        questions_repaired={k: v for k, v in questions_repaired.items() if v},
        pairs_rejected=dict(pairs_rejected),
    )
    summary._llms.append(llm)
    if eval_rejected:
        parts = []
        if eval_rejected.get("deictic"):
            parts.append(f"{eval_rejected['deictic']} pointed at the passage instead of naming what they asked about")
        if eval_rejected.get("unsupported"):
            parts.append(f"{eval_rejected['unsupported']} had answers the passage doesn't state")
        summary.warnings.append(
            f"{sum(eval_rejected.values())} evaluation questions were dropped from the answer key: "
            + "; ".join(parts) + ". A reader with the passage in front of it couldn't have been scored fairly on them."
        )
    if pairs_rejected.get("deictic"):
        summary.warnings.append(
            f"{pairs_rejected['deictic']} training questions were dropped: they still pointed at the passage "
            "after the rewrite, and a user who never saw it couldn't ask them."
        )
    if failed:
        note = f"{failed} of {len(chosen)} passages produced no usable questions."
        if empty_replies:
            note += (
                f" {empty_replies} came back empty — a thinking model that spent its token budget on "
                "reasoning; pick a non-thinking model or a backend that can turn thinking off."
            )
        summary.warnings.append(note)

    if llm is None:
        summary.training_pairs = existing_pairs
        summary.answer_key_rows = await _count_gold_flow_rows(db, project_id)
        await report(0.62, f"Reusing {existing_pairs} generated pairs and {summary.answer_key_rows} answer-key rows")
    else:
        await report(0.62, f"Saving {len(training_pairs)} training pairs and {len(answer_key)} answer-key rows")
    if training_pairs:
        await _save_training_pairs(db, project_id, training_pairs)
    if answer_key:
        from app.services.gold_service import import_qa_pairs

        await import_qa_pairs(
            db,
            project_id,
            [{**row, "source": FLOW_SOURCE} for row in answer_key],
            dataset_type=DatasetType.GOLD_DEV,
        )
    await db.commit()
    # The pairs are saved; a later run reuses them through reuse_existing.
    checkpoint.unlink(missing_ok=True)

    if summary.training_pairs < MIN_TRAINING_PAIRS:
        summary.stopped_reason = (
            f"Only {summary.training_pairs} training pairs were generated (need {MIN_TRAINING_PAIRS} for a "
            "split the lift check can measure). They are saved; add documents or raise max_passages and re-run."
        )
        return summary

    # ── 3: the project is now a Q&A project ───────────────────────────────
    await report(0.68, "Setting the task type to Q&A")
    from app.services.recipe_apply_service import apply_recipe_to_project

    recipe_id = str((project.selected_recipe or {}).get("recipe_id") or "")
    base_model_before = str(project.base_model_name or "").strip()
    if recipe_id != RECIPE_ID:
        await apply_recipe_to_project(db, project_id, RECIPE_ID)
        project = await db.get(Project, project_id)
        # Picking a task type adopts its suggested model; a model the user
        # chose for the documents stays.
        if base_model_before and project is not None and project.base_model_name != base_model_before:
            project.base_model_name = base_model_before
        await db.commit()

    # ── 4: split ──────────────────────────────────────────────────────────
    await report(0.75, "Splitting into train / validation / test examples")
    from app.services.dataset_service import DEFAULT_TRAINING_SOURCE_TYPES, split_dataset

    split_kwargs: dict[str, Any] = {}
    growth: dict[str, Any] | None = None
    if keep_test:
        kept_test = _read_jsonl_rows(settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "test.jsonl")
        if not kept_test:
            raise ValueError("keep_test needs an active prepared test split to keep; run the flow once without it.")
        from app.services.gold_service import get_gold_entries

        candidates = [
            row for row in await get_gold_entries(db, project_id, DatasetType.GOLD_DEV)
            if str(row.get("source") or "") == FLOW_SOURCE
        ]
        target = target_test_rows(decision_gap)
        additions, growth = grow_test_set(kept_test, candidates, target)
        # Validation = eval-only answer-key questions too (sized to the same
        # target): kept verbatim once it is, else drawn from the answer key's
        # remaining questions, preferring passages no test row uses.
        kept_val = _read_jsonl_rows(settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "val.jsonl")
        keep_val = bool(kept_val) and len(kept_val) >= target and all(
            r.get("_source_dataset") == "answer_key" for r in kept_val
        )
        val_from: list[dict[str, Any]] | None = None
        if not keep_val:
            taken = kept_test + additions
            val_pick, _ = grow_test_set(taken, candidates, len(taken) + target)
            val_from = [_answer_key_row(row) for row in val_pick]
        growth["val"] = {"kept_verbatim": keep_val, "rows": len(kept_val) if keep_val else len(val_from or [])}
        additions = [_answer_key_row(row) for row in additions]
        split_kwargs = {"keep_test": True, "test_additions": additions, "keep_val": keep_val, "val_from": val_from}
        await report(0.75, f"Keeping the {len(kept_test)} test examples and adding {len(additions)} answer-key questions (target {target})")
    manifest = await split_dataset(
        db,
        project_id,
        include_types=[t.value for t in DEFAULT_TRAINING_SOURCE_TYPES],
        adapter_id="qa-pair",
        task_profile="qa",
        train_ratio=round(1.0 - test_ratio - 0.1, 3),
        val_ratio=0.1,
        test_ratio=test_ratio,
        **split_kwargs,
    )
    await db.commit()
    splits = manifest.get("splits") if isinstance(manifest, dict) else None
    summary.split = {
        "train": (splits or {}).get("train"),
        "val": (splits or {}).get("val"),
        "test": (splits or {}).get("test"),
        "test_ratio": test_ratio,
        "sized_for": sizing,
        "decision_gap": decision_gap,
        "dedup_dropped": (manifest.get("dedup_report") or {}).get("dropped_count") if isinstance(manifest, dict) else None,
    }
    if keep_test:
        summary.split["test_growth"] = growth
        summary.split["keep_test"] = manifest.get("keep_test") if isinstance(manifest, dict) else None
        summary.split["test_sha256"] = ((manifest.get("file_hashes") or {}).get("test")) if isinstance(manifest, dict) else None
        summary.split["val_sha256"] = ((manifest.get("file_hashes") or {}).get("val")) if isinstance(manifest, dict) else None
    if not train:
        summary.stopped_reason = "training skipped (train=False)"
        return summary

    # ── 4b: the pre-training gate — what retrieval alone gets right ───────
    if passages_check:
        await report(0.78, "Before training: scoring the base model with and without your passages (judged)")
        summary.passages_gate = await _run_passages_gate(project_id, report, passages_check_fn, sweep_split=sweep_split)
        gate = summary.passages_gate or {}
        if gate.get("status") == "retrieval_ready" and not train_if_retrieval_ready:
            summary.stopped_reason = (
                f"Retrieval already answers: {gate.get('reason')}. Training on generated pairs is "
                "unlikely to beat that — reroute to RAG, or train anyway to compare."
            )
            await report(1.0, "Retrieval already answers — stopped before training")
            return summary
        if gate.get("status") == "retrieval_promising":
            summary.warnings.append(
                f"Before training, retrieval alone was promising: {gate.get('reason')}."
            )

    # ── 4c: teacher answers from the same passages ────────────────────────
    if answers_from_teacher:
        teacher_llm = teacher_backend_override or (llm if teacher_backend is None and llm is not None else None) or pick_backend(teacher_backend)
        summary._llms.append(teacher_llm)
        critic_llm = None
        if teacher_revise_rounds:
            # The critic resolves like the teacher (cloud pin via the saved key,
            # else a synth backend name); None = the filter judge's own reason.
            critic_llm = teacher_critic_backend_override or (pick_backend(teacher_critic_backend) if teacher_critic_backend else None)
            if critic_llm is not None:
                summary._llms.append(critic_llm)
        summary.teacher = await _run_teacher_step(
            db, project_id, report,
            llm=teacher_llm,
            use_batch=use_batch,
            teacher_filter=teacher_filter,
            teacher_filter_keep=teacher_filter_keep,
            filter_judge=teacher_filter_judge,
            retrieve_fn=teacher_retrieve_fn,
            concurrency=int(concurrency),
            revise_rounds=teacher_revise_rounds,
            critic=critic_llm,
        )

    # ── 5: train (the watcher Job runs the lift check afterwards) ─────────
    await report(0.85, "Starting the training run")
    from app.services.training_service import create_experiment

    project = await db.get(Project, project_id)
    base_model = str(getattr(project, "base_model_name", "") or "").strip()
    if not base_model:
        summary.stopped_reason = "Project has no base model to train; pick one on the Train page."
        return summary
    run_config: dict[str, Any] = {"base_model": base_model, "task_type": "causal_lm", "training_mode": "sft"}
    run_name = f"Q&A assistant from documents · {summary.training_pairs} pairs"
    if train_with_passages:
        from app.services.passage_prompts_service import (
            CONFIG_FLAG,
            CONFIG_WITHHELD,
            DEFAULT_WITHHELD_FRACTION,
        )

        run_config[CONFIG_FLAG] = True
        run_config[CONFIG_WITHHELD] = (
            DEFAULT_WITHHELD_FRACTION if passages_withheld_fraction is None else float(passages_withheld_fraction)
        )
        run_name += " · reads your passages"
        if answers_from_teacher:
            from app.services.passage_prompts_service import CONFIG_TARGETS, TARGETS_TEACHER

            run_config[CONFIG_TARGETS] = TARGETS_TEACHER
            run_config[CONFIG_WITHHELD] = 0.0
            run_name += " · teacher answers"
    experiment = await create_experiment(
        db,
        project_id,
        run_name[:255],
        base_model,
        run_config,
        f"Started by the documents → Q&A flow ({len(chosen)} passages, {backend_label})"
        + (", trained to answer from the retrieved passages" if train_with_passages else "")
        + (f" with {summary.teacher.get('teacher')}'s answers to them." if answers_from_teacher and summary.teacher else "."),
    )
    await db.commit()
    summary.experiment_id = experiment.id
    summary.train_with_passages = bool(train_with_passages)
    launcher = start_training
    if launcher is None:
        from app.api.training import start as api_start

        launcher = api_start
    if _used_local_ollama(summary):
        # A local generation / teacher / judge model would sit on the memory
        # the training run needs (a 120B model holds ~65 GB): unload first.
        from app.services.synth_backends.ollama import unload_loaded_models

        unloaded = await unload_loaded_models()
        if unloaded:
            summary.warnings.append(f"Unloaded {', '.join(unloaded)} from Ollama before training.")
    summary.training = await launcher(project_id, experiment.id, db)
    await report(0.98, f"Training run #{experiment.id} started — the lift check follows automatically")
    return summary


async def _run_passages_gate(
    project_id: int,
    report: Callable[[float, str], Awaitable[None]],
    check_fn: Callable[..., dict[str, Any]] | None,
    *,
    sweep_split: str = "val",
) -> dict[str, Any]:
    """Run the judged base + document-passages comparison as a RETRIEVAL
    SWEEP (the auto-RAG harness on a worker thread), keep the retrieval the
    judge scores highest as the project's setting, and read the gate.

    ``sweep_split="val"``: the sweep picks on the validation rows, then the
    chosen retrieval is scored ONCE on the test examples (no sweep) — the
    gate reads that test score, so the retrieval is never selected on the
    rows it is reported on. ``gate["sweep"]`` keeps the validation side
    (split, rows, chosen arm's score). ``"test"``: sweep and gate both on
    test (older behaviour). ``check_fn(progress_callback=, split=,
    sweep_retrieval=)`` injects the comparison (tests). Best-effort: a
    failure yields ``status="error"`` and the flow trains."""
    import asyncio

    from app.services.passage_rag_verdict_service import read_passages_gate

    def _default_check(**kwargs):
        import sys as _sys
        from pathlib import Path as _Path

        backend_root = str(_Path(__file__).resolve().parents[2])
        if backend_root not in _sys.path:
            _sys.path.insert(0, backend_root)
        from scripts.auto_rag_ab import run_project_comparison

        return run_project_comparison(project_id, base_only=True, corpus="documents", **kwargs)

    loop = asyncio.get_running_loop()
    state: dict[str, Any] = {"scored": 0, "total": 0, "label": "", "phase": ""}

    def _cb(scored: int, total: int, label: str) -> None:
        state.update(scored=scored, total=total, label=label)

    async def _drain(stop: asyncio.Event) -> None:
        while not stop.is_set():
            phase = state["phase"]
            if state["total"]:
                frac = 0.78 + 0.12 * min(1.0, state["scored"] / max(1, 2 * state["total"]))
                await report(frac, f"Pre-training check{phase}: scoring row {state['scored']}/{state['total']} ({state['label']})")
            elif state["label"]:
                await report(0.88, f"Pre-training check{phase}: {state['label']}")
            try:
                await asyncio.wait_for(stop.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass

    fn = check_fn or _default_check
    stop = asyncio.Event()
    drainer = loop.create_task(_drain(stop))
    sweep_block: dict[str, Any] | None = None
    summary: Any = None
    applied: Any = None
    try:
        state["phase"] = f" (retrieval sweep on {sweep_split})"
        payload = await asyncio.to_thread(fn, progress_callback=_cb, split=sweep_split, sweep_retrieval=True)
        summary = (payload or {}).get("summary") if isinstance(payload, dict) else None
        retrieval = (summary or {}).get("retrieval") if isinstance(summary, dict) else None
        applied = None
        if isinstance(retrieval, dict):
            from app.services.auto_rag_comparison_job_service import apply_retrieval_choice

            applied = await apply_retrieval_choice(project_id, retrieval) or retrieval
        if sweep_split != "test":
            judge = (summary or {}).get("judge") if isinstance(summary, dict) else None
            on = (judge or {}).get("with_rag") if isinstance(judge, dict) else None
            off = (judge or {}).get("without_rag") if isinstance(judge, dict) else None
            sweep_block = {
                "split": sweep_split,
                "rows": (summary or {}).get("n_val_rows") if isinstance(summary, dict) else None,
                "chosen_score": (on or {}).get("score") if isinstance(on, dict) else None,
                "chosen_counts": (on or {}).get("counts") if isinstance(on, dict) else None,
                "without_retrieval_score": (off or {}).get("score") if isinstance(off, dict) else None,
            }
            # The chosen retrieval (now the project's setting), scored once
            # on the test examples.
            state.update(scored=0, total=0, label="", phase=" (chosen retrieval on test)")
            await asyncio.to_thread(fn, progress_callback=_cb, split="test", sweep_retrieval=False)
    except LlmBudgetExceeded:
        raise  # the hard API budget ends the flow
    except Exception as exc:  # noqa: BLE001 — the gate never blocks the flow
        _LOG.warning("documents_qa_flow passages check failed: %s", exc)
        return {"status": "error", "reason": f"{exc.__class__.__name__}: {exc}"[:300]}
    finally:
        stop.set()
        try:
            await drainer
        except Exception:  # noqa: BLE001
            pass
    gate = read_passages_gate(project_id)
    if applied is not None:
        gate["retrieval"] = applied
    if isinstance(summary, dict):
        sweep = summary.get("retrieval_sweep")
        if isinstance(sweep, list):
            gate["retrieval_sweep"] = [
                {k: arm.get(k) for k in ("label", "judge_score", "chosen", "current", "vs_current")}
                for arm in sweep if isinstance(arm, dict)
            ]
        if isinstance(summary.get("retrieval_decision"), dict):
            gate["retrieval_decision"] = summary["retrieval_decision"]
    gate["sweep_split"] = sweep_split
    if sweep_block is not None:
        gate["sweep"] = sweep_block
    return gate


async def _run_teacher_step(
    db: AsyncSession,
    project_id: int,
    report: Callable[[float, str], Awaitable[None]],
    *,
    llm: Any,
    use_batch: bool,
    teacher_filter: bool,
    filter_judge: Any,
    retrieve_fn: Callable[..., list[dict[str, Any]]] | None,
    concurrency: int = 1,
    teacher_filter_keep: str = "not_wrong",
    revise_rounds: int = 0,
    critic: Any = None,
) -> dict[str, Any]:
    """Teacher answers for the prepared train + val rows, from the project's
    served retrieval (set by the gate). Never reads the test rows."""
    from app.services.auto_rag_service import retrieval_settings
    from app.services.teacher_passage_answers_service import write_teacher_answers

    if use_batch and not getattr(llm, "use_batch", False):
        _enable_batch(llm)
    project = await db.get(Project, project_id)
    await db.refresh(project)
    retrieval = retrieval_settings(project)
    prepared = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    rows_by_split = {
        "train": _read_jsonl_rows(prepared / "train.jsonl"),
        "val": _read_jsonl_rows(prepared / "val.jsonl"),
    }
    judge = filter_judge
    if teacher_filter and judge is None:
        from app.services.answer_judge_service import resolve_answer_judge

        judge = await resolve_answer_judge(db, project_id, project)
    return await write_teacher_answers(
        project_id,
        rows_by_split,
        retrieval=retrieval,
        llm=llm,
        retrieve_fn=retrieve_fn,
        report=report,
        filter_judge=judge if teacher_filter else None,
        filter_keep=teacher_filter_keep,
        revise_rounds=revise_rounds,
        critic=critic,
        start=0.8,
        span=0.05,
        concurrency=max(4, concurrency) if getattr(llm, "name", "") != "ollama" else concurrency,
    )


async def _save_training_pairs(db: AsyncSession, project_id: int, pairs: list[dict[str, Any]]) -> None:
    """Append to the synthetic dataset as ACCEPTED rows tagged with the flow's
    ``synth_source`` — the user asked for a model, not a review queue, and
    the review queue can still show, reject or purge them by source."""
    from app.services.synth_playbook_service import _peek_next_id
    from app.services.synthetic_service import get_or_create_synthetic_dataset

    dataset = await get_or_create_synthetic_dataset(db, project_id)
    synthetic_dir = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    file_path = synthetic_dir / "synthetic.jsonl"
    next_id = _peek_next_id(file_path)
    rows = pairs
    with file_path.open("a", encoding="utf-8") as handle:
        for offset, pair in enumerate(pairs):
            record = {
                "id": next_id + offset,
                **pair,
                "synth_confidence": 0.8,
                "synth_source": FLOW_SOURCE,
                "review_status": "accepted",
                "status": "accepted",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    dataset.record_count = (dataset.record_count or 0) + len(rows)
    if not dataset.file_path:
        dataset.file_path = str(file_path)
    await db.flush()
