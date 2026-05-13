"""Task-aware evaluation dispatcher (Phase 5.3.0).

Foundation for routing evaluation through per-task handlers. This phase
ships only the dispatcher + ``GenericHandler``, which preserves today's
behavior byte-for-byte. Future phases (5.3.1 classification, 5.3.3
seq2seq, …) register new handlers without touching this file.

The contract: ``task_profile`` is read **only** from
``prepared/manifest.json``. There is no row-shape sniffing — a seq2seq
dataset with few unique references can never be auto-mistaken for
classification. Missing tag → ``GenericHandler`` → identical behavior
to the pre-dispatcher pipeline.

Plan and per-phase user stories live in ``TASK_AWARE_EVAL_PLAN.md`` at
the repo root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from app.config import settings


# ── Data shapes ───────────────────────────────────────────────────────


@dataclass
class EvalContext:
    """Read-only context passed to every handler call.

    Carries everything a handler needs to build prompts or compute
    metrics without re-reading state. Handlers must not mutate it.
    """

    project_id: int
    experiment_id: int
    eval_type: str
    task_profile: str | None
    handler_id: str
    prepared_dir: Path
    dataset_name: str
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltPrompt:
    """One row's prompt + reference + auxiliary fields.

    ``extras`` carries handler-specific fields (image_path, context for
    RAG, candidate label set echoed for diagnostics, etc.) that the
    inference path or scorer may need. ``GenericHandler`` populates
    ``image_path`` and ``audio_path`` when present.
    """

    prompt: str
    reference: str
    extras: dict[str, Any] = field(default_factory=dict)

    def as_pair(self) -> dict[str, Any]:
        """Render as the legacy ``{prompt, reference, **extras}`` dict the
        existing inference path consumes."""

        pair: dict[str, Any] = {"prompt": self.prompt, "reference": self.reference}
        pair.update(self.extras)
        return pair


# ── Handler protocol ──────────────────────────────────────────────────


@runtime_checkable
class TaskHandler(Protocol):
    """Two-method interface every task handler implements."""

    profile_id: str

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        """Map dataset rows to prompt/reference pairs for inference."""

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        """Compute metric dict from predictions. Returned keys flow
        straight into ``EvalResult.metrics``."""


# ── GenericHandler — today's behavior, preserved verbatim ─────────────


class GenericHandler:
    """Fallback handler. Mirrors pre-5.3.0 behavior exactly.

    Delegates prompt extraction and scoring to the helpers that live in
    ``evaluation_service`` so the pre-dispatcher entry points (the
    ``/api/evaluation/run`` direct path, existing tests) score
    identically.
    """

    profile_id: str = "generic"

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        from app.services.evaluation_service import (
            _extract_prompt_and_reference,
        )

        built: list[BuiltPrompt] = []
        for row in rows:
            prompt, reference = _extract_prompt_and_reference(row)
            extras: dict[str, Any] = {}
            image_path = str(row.get("image_path") or row.get("image") or "").strip()
            audio_path = str(row.get("audio_path") or row.get("audio") or "").strip()
            if image_path:
                extras["image_path"] = image_path
            if audio_path:
                extras["audio_path"] = audio_path
            built.append(BuiltPrompt(prompt=prompt, reference=reference, extras=extras))
        return built

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        from app.services.evaluation_service import (
            evaluate_safety_response,
            exact_match,
            f1_score,
        )

        eval_type = ctx.eval_type
        if eval_type == "exact_match":
            scores = [
                exact_match(p.get("prediction", ""), p.get("reference", ""))
                for p in predictions
            ]
            return {
                "exact_match": round(sum(scores) / len(scores), 4) if scores else 0,
                "total": len(scores),
                "correct": int(sum(scores)),
            }
        if eval_type == "f1":
            scores = [
                f1_score(p.get("prediction", ""), p.get("reference", ""))
                for p in predictions
            ]
            return {
                "f1": round(sum(scores) / len(scores), 4) if scores else 0,
                "total": len(scores),
            }
        if eval_type == "safety":
            results = [
                evaluate_safety_response(
                    p.get("response", ""), p.get("test_type", "unknown")
                )
                for p in predictions
            ]
            passed = sum(1 for r in results if r["passed"])
            return {
                "pass_rate": round(passed / len(results), 4) if results else 0,
                "total_tests": len(results),
                "passed": passed,
                "failed": len(results) - passed,
            }
        # llm_judge is handled by a separate code path; the dispatcher
        # doesn't get called for it. Return empty so callers can detect
        # an unknown eval_type and fall through.
        return {}


# ── ClassificationHandler (Phase 5.3.1) ───────────────────────────────


class ClassificationHandler:
    """Task handler for classification tasks (sentiment, intent, topic …).

    Wraps each row's input with a label-list instruction, generates a short
    completion, then extracts the predicted label by substring-matching
    against the candidate set. Produces classification-native metrics
    (accuracy, macro_f1, per_class P/R/F1, confusion matrix,
    unparseable_rate) plus legacy ``exact_match`` / ``f1`` aliases so
    eval-pack gates keyed on those metric IDs keep working.
    """

    profile_id: str = "classification"

    # How many labels we'll list inline in the prompt before omitting the
    # list and just asking for a label. 30 is the threshold from the
    # plan — beyond that the prompt becomes a list of clutter.
    LABEL_LIST_PROMPT_CAP: int = 30
    # Outer cap on candidate-set size. Beyond this we still parse but
    # bail out of per-class metrics + confusion matrix.
    MAX_CANDIDATE_SET: int = 200
    # Confusion matrix only when the candidate set is small enough that
    # the resulting NxN dict is human-readable.
    CONFUSION_MATRIX_CAP: int = 20
    # Generation override: classification answers are short. Even with
    # multi-word labels ("very_positive") 16 new tokens is plenty.
    MAX_NEW_TOKENS_CAP: int = 16

    def __init__(self) -> None:
        self._cached_candidates: list[str] | None = None

    # ── Candidate-set resolution ──

    def _candidate_set_from_manifest(self, ctx: EvalContext) -> list[str]:
        raw = ctx.manifest.get("labels")
        if not isinstance(raw, list):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for value in raw:
            label = str(value).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            out.append(label)
        return out[: self.MAX_CANDIDATE_SET]

    def _candidate_set_from_records(
        self, records: list[dict[str, Any]]
    ) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for record in records:
            label = self._extract_reference_label(record)
            if not label or label in seen:
                continue
            seen.add(label)
            out.append(label)
        out.sort()
        return out[: self.MAX_CANDIDATE_SET]

    def _resolve_candidates(
        self,
        records: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[str]:
        if self._cached_candidates is not None:
            return self._cached_candidates
        from_manifest = self._candidate_set_from_manifest(ctx)
        candidates = from_manifest or self._candidate_set_from_records(records)
        self._cached_candidates = candidates
        return candidates

    # ── Row-field extraction ──

    def _extract_input_text(self, row: dict[str, Any]) -> str:
        for key in (
            "text",
            "source_text",
            "input",
            "prompt",
            "question",
            "instruction",
            "body",
            "content",
        ):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_reference_label(self, row: dict[str, Any]) -> str:
        """Pull the gold label from a raw row or a prediction dict.

        For raw rows the label may be under ``label`` / ``target_text`` /
        ``answer`` / ``class`` / ``category``. For prediction dicts
        produced by ``_load_heldout_pairs`` the label lives under
        ``reference`` because ``build_prompts`` mapped it there.
        """

        for key in (
            "label",
            "target_text",
            "reference",
            "answer",
            "output",
            "class",
            "category",
        ):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    # ── Prompt assembly ──

    def _build_prompt_text(self, input_text: str, candidates: list[str]) -> str:
        if 0 < len(candidates) <= self.LABEL_LIST_PROMPT_CAP:
            label_list = ", ".join(candidates)
            return (
                f"Classify the following text. Reply with exactly one of: "
                f"{label_list}.\n"
                f"Text: {input_text}\n"
                f"Label:"
            )
        # > cap or unknown: still ask for a single-label reply.
        return (
            "Classify the following text. Reply with just the class label, "
            "nothing else.\n"
            f"Text: {input_text}\n"
            f"Label:"
        )

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        candidates = self._resolve_candidates(rows, ctx)
        built: list[BuiltPrompt] = []
        for row in rows:
            input_text = self._extract_input_text(row)
            gold_label = self._extract_reference_label(row)
            wrapped = self._build_prompt_text(input_text, candidates)
            extras: dict[str, Any] = {
                "classification_input": input_text,
                "classification_candidates": list(candidates),
            }
            built.append(
                BuiltPrompt(prompt=wrapped, reference=gold_label, extras=extras)
            )
        return built

    # ── Label extraction from model output ──

    def parse_predicted_label(
        self,
        output: str,
        candidates: list[str],
    ) -> str | None:
        """Extract the predicted label from a generation.

        Strategy: scan the (lowercased) output for the first occurrence
        of any candidate label. Ties at the same position are resolved
        by longest label so ``very_positive`` wins over ``positive`` when
        the model said ``very_positive sentiment``. Returns ``None`` if
        no candidate appears in the output (counted as ``unparseable``).
        """

        if not candidates:
            return None
        text = (str(output) or "").strip().lower()
        if not text:
            return None
        # (position, -length, label) so default sort gives earliest-then-longest.
        hits: list[tuple[int, int, str]] = []
        for label in candidates:
            needle = label.lower()
            pos = text.find(needle)
            if pos >= 0:
                hits.append((pos, -len(needle), label))
        if not hits:
            return None
        hits.sort()
        return hits[0][2]

    # ── Scoring ──

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        candidates = self._resolve_candidates(predictions, ctx)
        total = len(predictions)
        if total == 0:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "exact_match": 0.0,
                "f1": 0.0,
                "total": 0,
                "correct": 0,
                "unparseable": 0,
                "unparseable_rate": 0.0,
                "per_class": {},
                "confusion_matrix": {},
                "candidate_set": candidates,
            }

        parsed_pairs: list[tuple[str | None, str]] = []
        unparseable = 0
        for prediction in predictions:
            gold = self._extract_reference_label(prediction)
            predicted = self.parse_predicted_label(
                prediction.get("prediction", ""), candidates
            )
            if predicted is None:
                unparseable += 1
            parsed_pairs.append((predicted, gold))

        correct = sum(
            1 for pred, gold in parsed_pairs if pred is not None and pred == gold
        )
        accuracy = round(correct / total, 4)

        per_class: dict[str, dict[str, Any]] = {}
        for label in candidates:
            tp = sum(1 for pred, gold in parsed_pairs if pred == label and gold == label)
            fp = sum(1 for pred, gold in parsed_pairs if pred == label and gold != label)
            fn = sum(1 for pred, gold in parsed_pairs if pred != label and gold == label)
            support = sum(1 for _, gold in parsed_pairs if gold == label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        macro_f1 = (
            round(sum(entry["f1"] for entry in per_class.values()) / len(per_class), 4)
            if per_class
            else 0.0
        )

        confusion: dict[str, dict[str, int]] = {}
        if 0 < len(candidates) <= self.CONFUSION_MATRIX_CAP:
            for gold_label in candidates:
                row_counts: dict[str, int] = {pred_label: 0 for pred_label in candidates}
                row_counts["__unparseable__"] = 0
                for pred, gold in parsed_pairs:
                    if gold != gold_label:
                        continue
                    if pred is None:
                        row_counts["__unparseable__"] += 1
                    elif pred in row_counts:
                        row_counts[pred] += 1
                confusion[gold_label] = row_counts

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            # Legacy aliases so eval-pack gates keyed on `exact_match` and
            # `f1` keep resolving without a pack migration.
            "exact_match": accuracy,
            "f1": macro_f1,
            "total": total,
            "correct": correct,
            "unparseable": unparseable,
            "unparseable_rate": round(unparseable / total, 4),
            "per_class": per_class,
            "confusion_matrix": confusion,
            "candidate_set": candidates,
        }

    # ── Inference hint ──

    def max_new_tokens_override(self, default: int) -> int:
        """Cap generation length. A class label is at most a few tokens —
        letting the model emit 128+ new tokens just gives it room to
        ramble and burn latency."""

        return min(max(1, int(default or 1)), self.MAX_NEW_TOKENS_CAP)


# ── Registry + dispatcher ─────────────────────────────────────────────


def _normalize_profile(value: Any) -> str:
    """Normalize a task_profile string for registry lookup. Empty / None
    becomes empty string, which dispatches to ``GenericHandler``."""

    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


# Maps normalized task_profile → handler factory. New handlers register
# themselves by appending to this dict (e.g. classification handler in
# Phase 5.3.1). Missing key falls through to ``GenericHandler``.
_HANDLER_FACTORIES: dict[str, Callable[[], TaskHandler]] = {}


def register_handler(profile: str, factory: Callable[[], TaskHandler]) -> None:
    """Register a handler factory for a given task profile.

    Intentionally tolerant: registering the same profile twice replaces
    the prior factory (useful for tests). The empty string is reserved
    for the GenericHandler fallback.
    """

    key = _normalize_profile(profile)
    if not key:
        raise ValueError("register_handler requires a non-empty profile id")
    _HANDLER_FACTORIES[key] = factory


def resolve_task_handler(task_profile: str | None) -> TaskHandler:
    """Return the handler matching ``task_profile``, or GenericHandler.

    The lookup is intentionally forgiving: unknown profiles, empty
    strings, malformed types all fall through to ``GenericHandler``. We
    log the fall-through at the call site, not here, so callers can
    decide what severity to attach.
    """

    key = _normalize_profile(task_profile)
    factory = _HANDLER_FACTORIES.get(key)
    if factory is None:
        return GenericHandler()
    try:
        return factory()
    except Exception:
        # A buggy handler factory must never break eval — fall back.
        return GenericHandler()


def list_registered_profiles() -> list[str]:
    """Diagnostic: which task profiles currently have a registered
    handler (excludes the implicit generic fallback)."""

    return sorted(_HANDLER_FACTORIES.keys())


# ── Manifest reading ──────────────────────────────────────────────────


def _project_prepared_dir(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "prepared"


def read_prepared_manifest(project_id: int) -> dict[str, Any]:
    """Load ``prepared/manifest.json`` for a project; empty dict on
    miss / parse failure. Never raises."""

    manifest_path = _project_prepared_dir(project_id) / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_task_profile_from_manifest(project_id: int) -> str | None:
    """Convenience: pull just the ``task_profile`` from the prepared
    manifest. Returns ``None`` if missing or unreadable."""

    manifest = read_prepared_manifest(project_id)
    value = manifest.get("task_profile")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_eval_context(
    *,
    project_id: int,
    experiment_id: int,
    eval_type: str,
    dataset_name: str,
) -> tuple[EvalContext, TaskHandler]:
    """One-shot helper: read manifest, resolve handler, return both.

    Callers in ``evaluation_service`` use this to keep the dispatch
    site short.
    """

    manifest = read_prepared_manifest(project_id)
    task_profile = read_task_profile_from_manifest(project_id)
    handler = resolve_task_handler(task_profile)
    ctx = EvalContext(
        project_id=project_id,
        experiment_id=experiment_id,
        eval_type=eval_type,
        task_profile=task_profile,
        handler_id=handler.profile_id,
        prepared_dir=_project_prepared_dir(project_id),
        dataset_name=dataset_name,
        manifest=manifest,
    )
    return ctx, handler


# ── Built-in handler registrations ────────────────────────────────────
# New handlers register themselves here as they land (Phase 5.3.2+).

register_handler("classification", ClassificationHandler)


__all__ = [
    "BuiltPrompt",
    "ClassificationHandler",
    "EvalContext",
    "GenericHandler",
    "TaskHandler",
    "build_eval_context",
    "list_registered_profiles",
    "read_prepared_manifest",
    "read_task_profile_from_manifest",
    "register_handler",
    "resolve_task_handler",
]
