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

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import DatasetType
from app.models.project import Project
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
MAX_PASSAGES_CAP = 500
MIN_TEST_ROWS = 5


def target_test_rows(gap: float = DECISION_GAP, diff_sd: float = DIFF_SD) -> int:
    """Paired rows for a ``gap`` in mean judge score to clear a 95% interval."""
    import math

    return max(MIN_TEST_ROWS, math.ceil((2.0 * diff_sd / gap) ** 2))


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
        "The evaluation question will be put to a model that has this passage in front of it, so:\n"
        "- it must stand on its own — never say \"this passage\", \"this subsection\", \"mentioned above\" "
        "or \"listed\"; name the thing you are asking about (the section, the rule, the person);\n"
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
    # The pre-training gate: what the base model + passages gets right.
    passages_gate: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "passages_total": self.passages_total,
            "passages_used": self.passages_used,
            "passages_failed": self.passages_failed,
            "training_pairs": self.training_pairs,
            "answer_key_rows": self.answer_key_rows,
            "eval_rejected": dict(self.eval_rejected),
            "backend": self.backend,
            "recipe_id": self.recipe_id,
            "split": self.split,
            "experiment_id": self.experiment_id,
            "training": self.training,
            "stopped_reason": self.stopped_reason,
            "warnings": list(self.warnings),
            "passages_gate": self.passages_gate,
        }


async def preview_documents_qa_flow(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Can this project run the flow, and what would it cost? Read-only."""
    from app.services.auto_rag_service import load_document_passages
    from app.services.synth_backends import BACKEND_REGISTRY

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    passages = load_document_passages(project_id)
    backend_name: str | None = None
    for cls in BACKEND_REGISTRY:
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
    if backend_name is None:
        blockers.append(
            "No generation model is reachable. Start Ollama with a chat model, or set "
            "TEACHER_MODEL_API_URL to an OpenAI-compatible endpoint."
        )
    sizing = size_for_decision(len(passages), DEFAULT_PAIRS_PER_PASSAGE)
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
    the comparison (tests); the default runs the auto-RAG harness."""
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

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    passages = load_document_passages(project_id)
    if len(passages) < MIN_PASSAGES:
        raise ValueError(
            f"Needs at least {MIN_PASSAGES} cleaned document passages (found {len(passages)}). "
            "Import documents and run cleaning first."
        )
    if max_passages is None:
        sizing = size_for_decision(len(passages), pairs_per_passage)
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
    domain_hint = str(project.description or project.name or "").strip()[:200]

    # ── 1 + 2: generate per passage ───────────────────────────────────────
    chosen = passages[:max_passages] if llm is not None else []
    training_pairs: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    failed = 0
    empty_replies = 0
    # Evaluation pairs the quotable gate dropped, by reason.
    eval_rejected: dict[str, int] = {}
    for index, passage in enumerate(chosen):
        await report(
            0.05 + 0.55 * index / max(1, len(chosen)),
            f"Writing questions for passage {index + 1}/{len(chosen)} ({backend_label})",
        )
        try:
            if getattr(llm, "supports_prompt_cache", False):
                # A cloud backend: the per-run instructions are the cached
                # block, only the passage varies between calls.
                stable, varying = passage_prompt_parts(passage["text"], pairs_per_passage, domain_hint=domain_hint)
                prompt_kwargs: dict[str, Any] = {"cache_prefix": stable}
            else:
                varying = build_passage_prompt(passage["text"], pairs_per_passage, domain_hint=domain_hint)
                prompt_kwargs = {}
            raw = await llm.complete(
                varying,
                system_prompt=_SYSTEM_PROMPT,
                max_tokens=1600,
                temperature=0.4,
                response_schema=_RESPONSE_SCHEMA,
                **prompt_kwargs,
            )
        except SynthBackendError as exc:
            failed += 1
            _LOG.warning("documents_qa_flow passage %s failed: %s", passage["id"], exc)
            continue
        pairs, eval_pair = parse_passage_output(raw, pairs_per_passage)
        if not pairs and eval_pair is None:
            failed += 1
            if not str(raw or "").strip():
                empty_replies += 1
            else:
                _LOG.warning("documents_qa_flow passage %s: unparseable reply: %r", passage["id"], str(raw)[:200])
            continue
        provenance = {
            "source_doc": passage.get("source_doc"),
            "source_document_id": passage.get("source_document_id"),
            "chunk_id": passage.get("chunk_id"),
        }
        for pair in pairs:
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
    )
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

    manifest = await split_dataset(
        db,
        project_id,
        include_types=[t.value for t in DEFAULT_TRAINING_SOURCE_TYPES],
        adapter_id="qa-pair",
        task_profile="qa",
        train_ratio=round(1.0 - test_ratio - 0.1, 3),
        val_ratio=0.1,
        test_ratio=test_ratio,
    )
    await db.commit()
    splits = manifest.get("splits") if isinstance(manifest, dict) else None
    summary.split = {
        "train": (splits or {}).get("train"),
        "val": (splits or {}).get("val"),
        "test": (splits or {}).get("test"),
        "test_ratio": test_ratio,
        "sized_for": sizing,
        "dedup_dropped": (manifest.get("dedup_report") or {}).get("dropped_count") if isinstance(manifest, dict) else None,
    }
    if not train:
        summary.stopped_reason = "training skipped (train=False)"
        return summary

    # ── 4b: the pre-training gate — what retrieval alone gets right ───────
    if passages_check:
        await report(0.78, "Before training: scoring the base model with and without your passages (judged)")
        summary.passages_gate = await _run_passages_gate(project_id, report, passages_check_fn)
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

    # ── 5: train (the watcher Job runs the lift check afterwards) ─────────
    await report(0.85, "Starting the training run")
    from app.services.training_service import create_experiment

    project = await db.get(Project, project_id)
    base_model = str(getattr(project, "base_model_name", "") or "").strip()
    if not base_model:
        summary.stopped_reason = "Project has no base model to train; pick one on the Training tab."
        return summary
    experiment = await create_experiment(
        db,
        project_id,
        f"Q&A assistant from documents · {summary.training_pairs} pairs"[:255],
        base_model,
        {"base_model": base_model, "task_type": "causal_lm", "training_mode": "sft"},
        f"Started by the documents → Q&A flow ({len(chosen)} passages, {backend_label}).",
    )
    await db.commit()
    summary.experiment_id = experiment.id
    launcher = start_training
    if launcher is None:
        from app.api.training import start as api_start

        launcher = api_start
    summary.training = await launcher(project_id, experiment.id, db)
    await report(0.98, f"Training run #{experiment.id} started — the lift check follows automatically")
    return summary


async def _run_passages_gate(
    project_id: int,
    report: Callable[[float, str], Awaitable[None]],
    check_fn: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    """Run the judged base + document-passages comparison on the test split
    as a RETRIEVAL SWEEP (top-3 / top-5, each ± the cross-encoder reranker;
    the auto-RAG harness on a worker thread), keep the retrieval the judge
    scores highest as the project's setting, and read the gate from it — the
    gate judges the best retrieval the project can serve, not plain top-3.
    Best-effort: a failure yields ``status="error"`` and the flow trains."""
    import asyncio

    from app.services.passage_rag_verdict_service import read_passages_gate

    def _default_check(**kwargs):
        import sys as _sys
        from pathlib import Path as _Path

        backend_root = str(_Path(__file__).resolve().parents[2])
        if backend_root not in _sys.path:
            _sys.path.insert(0, backend_root)
        from scripts.auto_rag_ab import run_project_comparison

        return run_project_comparison(
            project_id, base_only=True, corpus="documents", split="test", sweep_retrieval=True, **kwargs
        )

    loop = asyncio.get_running_loop()
    state: dict[str, Any] = {"scored": 0, "total": 0, "label": ""}

    def _cb(scored: int, total: int, label: str) -> None:
        state.update(scored=scored, total=total, label=label)

    async def _drain(stop: asyncio.Event) -> None:
        while not stop.is_set():
            if state["total"]:
                frac = 0.78 + 0.12 * min(1.0, state["scored"] / max(1, 2 * state["total"]))
                await report(frac, f"Pre-training check: scoring row {state['scored']}/{state['total']} ({state['label']})")
            elif state["label"]:
                await report(0.88, f"Pre-training check: {state['label']}")
            try:
                await asyncio.wait_for(stop.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass

    stop = asyncio.Event()
    drainer = loop.create_task(_drain(stop))
    try:
        payload = await asyncio.to_thread(check_fn or _default_check, progress_callback=_cb)
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
    # The sweep's winner becomes what the project (and a RAG sibling cloned
    # from it) serves; the gate says which retrieval it judged.
    summary = (payload or {}).get("summary") if isinstance(payload, dict) else None
    retrieval = (summary or {}).get("retrieval") if isinstance(summary, dict) else None
    if isinstance(retrieval, dict):
        from app.services.auto_rag_comparison_job_service import apply_retrieval_choice

        gate["retrieval"] = await apply_retrieval_choice(project_id, retrieval) or retrieval
        sweep = (summary or {}).get("retrieval_sweep")
        if isinstance(sweep, list):
            gate["retrieval_sweep"] = [
                {k: arm.get(k) for k in ("label", "judge_score", "chosen", "current", "vs_current")}
                for arm in sweep if isinstance(arm, dict)
            ]
        if isinstance((summary or {}).get("retrieval_decision"), dict):
            gate["retrieval_decision"] = summary["retrieval_decision"]
    return gate


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
