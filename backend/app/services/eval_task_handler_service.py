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
import re
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

    def stop_sequences(self, ctx: EvalContext) -> list[str]:
        """Stop after the label. Two newlines = "I emitted my answer and
        am starting another block" — almost always rambling."""

        return [
            "\n\n",
            "\nText:",
            "\nLabel:",
            "\nExercise",
        ]

    def wraps_own_prompt(self) -> bool:
        """Classification builds its own complete prompt
        ("Classify the following text. Reply with exactly one of: …
        Label:") inside ``build_prompts``. Applying the model's chat
        template on top would wrap that whole instruction as a user
        message and the chat-tuned model would respond conversationally
        instead of emitting a label. Returning True signals the
        inference path to skip the chat-template wrap."""

        return True

    def expected_prompt_prefixes(self) -> list[str]:
        """Diagnostic — strings that would appear in a prepared row if
        its training-time prompt format matched the eval-time format
        this handler builds. The smoke-test's ``adapter_handler_format``
        check peeks at one prepared row and looks for any of these
        prefixes; if none appear, the adapter wrote raw ``text→label``
        pairs (which trains a model the eval can't decode) and the
        check escalates to warn.

        Returning a list lets handlers cover multiple acceptable
        phrasings (the prompt template can vary by candidate set, by
        sub-task, etc.). Returning ``[]`` means "no expected prefix"
        — the smoke check skips the row-format peek for that handler."""

        return ["Classify the following text"]


# ── QAHandler (Phase 5.3.2) ───────────────────────────────────────────


class QAHandler:
    """Task handler for short-answer QA / instruction-following.

    Preserves today's chat-template-only behavior on the prompt side
    (Phase 5.2 already wrapped bare questions with the tokenizer's
    chat template at inference time). The handler adds two things on
    top:

    1. **CoT answer-span extraction.** Chain-of-thought-trained models
       emit ``"…reasoning… Therefore: Paris."`` rather than just
       ``"Paris"``. Without extraction, SQuAD F1 scores the whole
       paragraph against the single-word reference and reports near
       zero. The handler scans for common end-of-reasoning markers
       (``Final answer:``, ``Answer:``, ``Therefore:``, ``The answer
       is …``) and scores the extracted span instead. Falls through
       to the raw prediction when no marker matches.

    2. **Per-row score capture.** Each prediction dict gets
       ``answer_span``, ``span_marker``, ``row_exact_match``, and
       ``row_f1`` written onto it before the aggregate is computed.
       The UI reads these in ``predictions_preview`` to render a
       per-row pass/fail badge + "Show extracted answer span"
       disclosure — so the user can see exactly which rows the model
       got wrong without leaving the page.

    Metrics produced: ``exact_match``, ``f1`` (mean of per-row scores
    against the extracted span), ``answer_span_extracted_rate`` (the
    fraction of rows where a CoT marker was found), ``total``,
    ``correct``. EM and F1 metric IDs preserve gate compatibility.
    """

    profile_id: str = "qa"

    # CoT answer-marker patterns. Ordered longer-first / more-specific-
    # first so ``Final answer: 42`` doesn't accidentally match the
    # shorter ``Answer:`` rule. Each pattern captures the span up to
    # the first period or newline or end of string.
    _SPAN_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            r"final\s+answer\s*[:\-]\s*(.+?)(?:\.\s|\n|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\banswer\s*[:\-]\s*(.+?)(?:\.\s|\n|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\btherefore\s*[:,\-]?\s*(.+?)(?:\.\s|\n|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bin\s+conclusion\s*[:,\-]?\s*(.+?)(?:\.\s|\n|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bthe\s+answer\s+is\s+(.+?)(?:\.\s|\n|$)",
            re.IGNORECASE | re.DOTALL,
        ),
    )

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        # QA inherits GenericHandler's field extraction — the only
        # behavior we change is at score time. Delegate to keep the
        # field-precedence rules in one place.
        return GenericHandler().build_prompts(rows, ctx)

    def extract_answer_span(self, text: str) -> tuple[str, str | None]:
        """Return ``(span, marker_pattern)`` if a CoT marker matched,
        else ``(text, None)``.

        Uses ``re.findall`` and keeps the LAST match per pattern, since
        CoT outputs typically place the conclusion at the end of the
        reasoning. Tries each pattern in order — first one that matches
        wins.
        """

        haystack = str(text or "")
        if not haystack.strip():
            return haystack, None
        for pattern in self._SPAN_PATTERNS:
            matches = pattern.findall(haystack)
            if not matches:
                continue
            # Last match = final occurrence in the text (CoT
            # conclusion at the end of reasoning). Trailing terminal
            # punctuation isn't part of the answer ("Paris." → "Paris").
            span = str(matches[-1]).strip().rstrip(".,!?;:").strip()
            if span:
                return span, pattern.pattern
        return haystack, None

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        # Lazy import to dodge the cyclic dependency
        # (evaluation_service imports this module too).
        from app.services.evaluation_service import exact_match, f1_score

        total = len(predictions)
        if total == 0:
            return {
                "exact_match": 0.0,
                "f1": 0.0,
                "answer_span_extracted_rate": 0.0,
                "total": 0,
                "correct": 0,
            }

        em_scores: list[float] = []
        f1_scores: list[float] = []
        extracted_count = 0
        for prediction in predictions:
            full_text = str(prediction.get("prediction") or "")
            reference = str(prediction.get("reference") or "")
            span, marker = self.extract_answer_span(full_text)
            if marker is not None:
                extracted_count += 1
            row_em = exact_match(span, reference)
            row_f1 = f1_score(span, reference)
            em_scores.append(row_em)
            f1_scores.append(row_f1)
            # Enrich each prediction in place so the predictions_preview
            # writer in evaluation_service can flow these into the UI
            # without re-doing the work.
            prediction["answer_span"] = span
            prediction["span_marker"] = marker
            prediction["row_exact_match"] = row_em
            prediction["row_f1"] = row_f1

        return {
            "exact_match": round(sum(em_scores) / total, 4),
            "f1": round(sum(f1_scores) / total, 4),
            "answer_span_extracted_rate": round(extracted_count / total, 4),
            "total": total,
            "correct": int(sum(em_scores)),
        }

    def stop_sequences(self, ctx: EvalContext) -> list[str]:
        """Stop on QA-style follow-up patterns — model restarting with a
        new question is "I'm done with this one, here's another"."""

        return [
            "\nQuestion:",
            "\n\nQuestion:",
            "\nQ:",
            "\n\nQ:",
            "\nInput:",
            "\n\nInput:",
            "\nExercise",
        ]


# ── StructuredExtractionHandler (Phase 5.3.4) ─────────────────────────


class StructuredExtractionHandler:
    """Task handler for structured / JSON extraction.

    Wraps each row with a "Extract these fields as JSON" instruction,
    parses the model's output as JSON (stripping common code-fence
    artifacts), and scores at three layers:

    1. **JSON validity** — did the model produce a parseable object?
       A 30% malformed-JSON rate makes the model unshippable
       regardless of field accuracy, so this gets its own metric.
    2. **Schema compliance** — did the parsed object include every
       required field?
    3. **Field-level EM / F1** — per declared field, averaged over
       rows where the field appears in both prediction and reference.

    Whole-blob ``exact_match`` and ``f1`` aliases are also produced
    so eval-pack gates keyed on those metric IDs keep resolving
    (``f1`` is set to the mean per-field F1, which is the most useful
    aggregate for extraction).

    Schema source priority:
      1. ``manifest.output_schema`` (JSON Schema with properties +
         required). Authoritative when present.
      2. Otherwise, derive the field set by scanning up to the first
         20 references — implicit but lets the handler work on
         untagged datasets.

    The handler enriches each prediction in place with
    ``parsed_prediction``, ``parsed_reference``, ``is_valid_json``,
    ``missing_required_fields``, ``row_field_results``, plus the
    standard ``row_exact_match`` / ``row_f1`` that the QAHandler also
    writes. The UI reads these to render inline JSON validity notes
    and a per-field comparison disclosure.
    """

    profile_id: str = "structured_extraction"

    SCHEMA_SAMPLE_SIZE: int = 20
    MAX_NEW_TOKENS_FLOOR: int = 128
    MAX_NEW_TOKENS_HARDCAP: int = 512

    # Scoring modes within structured_extraction. ``field_match`` is
    # today's per-field EM/F1 (invoice-style extraction). ``span_set``
    # is for outputs whose shape is a list of typed spans (PII / NER /
    # medical / legal / financial entity extraction) — same general
    # handler, internal dispatch so we don't fork per domain.
    SCORING_MODE_FIELD_MATCH: str = "field_match"
    SCORING_MODE_SPAN_SET: str = "span_set"
    _SUPPORTED_SCORING_MODES: set[str] = {
        SCORING_MODE_FIELD_MATCH,
        SCORING_MODE_SPAN_SET,
    }

    def __init__(self) -> None:
        self._schema: dict[str, Any] | None = None

    # ── Schema resolution ──

    def _resolve_schema(
        self,
        records: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        if self._schema is not None:
            return self._schema

        manifest_schema = ctx.manifest.get("output_schema")
        scoring_mode = self.SCORING_MODE_FIELD_MATCH
        if isinstance(manifest_schema, dict):
            raw_mode = str(manifest_schema.get("scoring_mode") or "").strip().lower()
            if raw_mode in self._SUPPORTED_SCORING_MODES:
                scoring_mode = raw_mode
            properties = manifest_schema.get("properties")
            required = manifest_schema.get("required")
            if isinstance(properties, dict) and properties:
                fields = sorted(properties.keys())
                req_list = (
                    list(required)
                    if isinstance(required, list) and required
                    else list(fields)
                )
                self._schema = {
                    "fields": fields,
                    "required": req_list,
                    "scoring_mode": scoring_mode,
                }
                return self._schema

        # Derive from references.
        seen: set[str] = set()
        for record in records[: self.SCHEMA_SAMPLE_SIZE]:
            ref_raw = self._read_reference_raw(record)
            parsed = self._parse_json_safely(ref_raw)
            if isinstance(parsed, dict):
                seen.update(str(k) for k in parsed.keys())
        fields = sorted(seen)
        self._schema = {
            "fields": fields,
            "required": list(fields),
            "scoring_mode": scoring_mode,
        }
        return self._schema

    # ── Field extraction ──

    def _extract_input_text(self, row: dict[str, Any]) -> str:
        for key in (
            "text",
            "source_text",
            "input",
            "prompt",
            "instruction",
            "question",
            "body",
            "content",
            "document",
        ):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _read_reference_raw(self, row: dict[str, Any]) -> Any:
        """Return the raw reference value (dict or string)."""

        for key in (
            "reference",
            "target_text",
            "expected",
            "output",
            "answer",
            "structured_output",
        ):
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, dict):
                return value
            text = str(value).strip()
            if text:
                return text
        return ""

    def _reference_as_string(self, row: dict[str, Any]) -> str:
        raw = self._read_reference_raw(row)
        if isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False, sort_keys=True)
        return str(raw or "")

    # ── JSON parsing ──

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        s = text.strip()
        if not s.startswith("```"):
            return s
        lines = s.splitlines()
        # Drop the opening fence (may carry a language tag like ```json).
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Drop the closing fence if present.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _parse_json_safely(self, value: Any) -> Any:
        """Best-effort JSON parse. Returns the parsed dict, or ``None`` if
        nothing parseable is found. Handles raw dicts (passthrough),
        triple-backtick code fences, and prose-then-JSON outputs by
        extracting the first balanced ``{…}`` block."""

        if value is None:
            return None
        if isinstance(value, dict):
            return value
        text = str(value).strip()
        if not text:
            return None
        text = self._strip_code_fences(text)
        # Try a clean parse first.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Fall back to first balanced {…} block.
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        return None
                    return parsed if isinstance(parsed, dict) else None
        return None

    # ── Prompt assembly ──

    def _build_prompt_text(self, input_text: str, fields: list[str]) -> str:
        if fields:
            field_list = ", ".join(fields)
            return (
                "Extract the following fields as JSON: "
                f"{field_list}.\n"
                "Reply with a single JSON object, nothing else.\n"
                f"Input: {input_text}\n"
                "Output:"
            )
        return (
            "Extract the relevant fields from the input as a single JSON "
            "object, nothing else.\n"
            f"Input: {input_text}\n"
            "Output:"
        )

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        schema = self._resolve_schema(rows, ctx)
        fields = schema["fields"]
        built: list[BuiltPrompt] = []
        for row in rows:
            input_text = self._extract_input_text(row)
            reference = self._reference_as_string(row)
            wrapped = self._build_prompt_text(input_text, fields)
            built.append(
                BuiltPrompt(
                    prompt=wrapped,
                    reference=reference,
                    extras={
                        "structured_fields": list(fields),
                        "structured_input": input_text,
                    },
                )
            )
        return built

    # ── Generation hint ──

    def max_new_tokens_override(self, default: int) -> int:
        """JSON outputs need room (a 5-field object is ~50–80 tokens)
        but should be bounded — extraction isn't a place for rambling.
        Raise tiny defaults to a sane floor; cap at the hard limit."""

        baseline = max(self.MAX_NEW_TOKENS_FLOOR, int(default or 0))
        return min(self.MAX_NEW_TOKENS_HARDCAP, baseline)

    def stop_sequences(self, ctx: EvalContext) -> list[str]:
        """Halt generation when the model starts rambling past the JSON.

        Untuned models often emit valid JSON and then continue with
        worksheet-style content ("Exercise 3: …", "Input: …",
        "Question: …") because they treat the prompt as a homework
        problem with follow-ups. These stops cut the rambling so the
        JSON parser sees clean output. After fine-tuning these stops
        are mostly inert because the model emits a single JSON object
        and halts on its own — they exist for the zero-shot / partially-
        trained cases."""

        return [
            # Worksheet-style continuation patterns the screenshot showed.
            "\nExercise",
            "\n\nExercise",
            # Model restarting with a fresh prompt block.
            "\nInput:",
            "\n\nInput:",
            "\nText:",
            "\n\nText:",
            "\nQuestion:",
            "\n\nQuestion:",
            # Markdown / docstring section separators.
            "\n\n---",
            "\n\n##",
            # Closing code fence after a ```json … ``` block.
            "\n```\n",
        ]

    # ── Scoring ──

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        schema = self._resolve_schema(predictions, ctx)
        scoring_mode = schema.get("scoring_mode", self.SCORING_MODE_FIELD_MATCH)
        if scoring_mode == self.SCORING_MODE_SPAN_SET:
            return self._score_span_set(predictions, schema)
        return self._score_field_match(predictions, schema)

    # ── Field-match scoring (default — today's invoice-style flow) ──

    def _score_field_match(
        self,
        predictions: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        from app.services.evaluation_service import exact_match, f1_score

        fields = schema["fields"]
        required = schema["required"]
        total = len(predictions)

        if total == 0:
            return {
                "json_validity_rate": 0.0,
                "schema_compliance_rate": 0.0,
                "field_exact_match_rate": 0.0,
                "field_f1": 0.0,
                "overall_em": 0.0,
                "exact_match": 0.0,
                "f1": 0.0,
                "total": 0,
                "correct": 0,
                "per_field": {},
                "schema": schema,
            }

        valid_count = 0
        compliant_count = 0
        overall_em_count = 0
        per_field_em: dict[str, dict[str, int]] = {
            f: {"correct": 0, "total": 0} for f in fields
        }
        per_field_f1: dict[str, list[float]] = {f: [] for f in fields}

        for prediction in predictions:
            raw_output = prediction.get("prediction") or ""
            raw_ref = prediction.get("reference") or ""
            parsed_pred = self._parse_json_safely(raw_output)
            parsed_ref = self._parse_json_safely(raw_ref)
            is_valid = isinstance(parsed_pred, dict)
            if is_valid:
                valid_count += 1

            missing: list[str] = []
            if is_valid and required:
                missing = [f for f in required if f not in parsed_pred]
            is_compliant = is_valid and not missing
            if is_compliant:
                compliant_count += 1

            row_field_results: dict[str, dict[str, float]] = {}
            if is_valid and isinstance(parsed_ref, dict):
                for field_name in fields:
                    if field_name not in parsed_ref:
                        continue
                    per_field_em[field_name]["total"] += 1
                    ref_val = str(parsed_ref.get(field_name, ""))
                    if field_name in parsed_pred:
                        pred_val = str(parsed_pred.get(field_name, ""))
                        em = exact_match(pred_val, ref_val)
                        f1 = f1_score(pred_val, ref_val)
                        per_field_em[field_name]["correct"] += int(em)
                        per_field_f1[field_name].append(f1)
                        row_field_results[field_name] = {"em": em, "f1": f1}
                    else:
                        per_field_f1[field_name].append(0.0)
                        row_field_results[field_name] = {"em": 0.0, "f1": 0.0}

            overall_em = 0.0
            if is_valid and isinstance(parsed_ref, dict) and parsed_pred == parsed_ref:
                overall_em = 1.0
                overall_em_count += 1

            row_f1 = (
                round(
                    sum(r["f1"] for r in row_field_results.values())
                    / len(row_field_results),
                    4,
                )
                if row_field_results
                else 0.0
            )
            # In-place enrichment for predictions_preview → UI.
            prediction["parsed_prediction"] = parsed_pred
            prediction["parsed_reference"] = parsed_ref
            prediction["is_valid_json"] = is_valid
            prediction["missing_required_fields"] = missing
            prediction["row_field_results"] = row_field_results
            prediction["row_exact_match"] = overall_em
            prediction["row_f1"] = row_f1

        per_field_summary: dict[str, dict[str, Any]] = {}
        for field_name in fields:
            tot = per_field_em[field_name]["total"]
            cor = per_field_em[field_name]["correct"]
            em_rate = round(cor / tot, 4) if tot > 0 else 0.0
            f1_list = per_field_f1[field_name]
            f1_mean = round(sum(f1_list) / len(f1_list), 4) if f1_list else 0.0
            per_field_summary[field_name] = {
                "em": em_rate,
                "f1": f1_mean,
                "support": tot,
            }

        field_em_avg = (
            round(
                sum(v["em"] for v in per_field_summary.values()) / len(per_field_summary),
                4,
            )
            if per_field_summary
            else 0.0
        )
        field_f1_avg = (
            round(
                sum(v["f1"] for v in per_field_summary.values()) / len(per_field_summary),
                4,
            )
            if per_field_summary
            else 0.0
        )
        overall_em_rate = round(overall_em_count / total, 4)

        return {
            "scoring_mode": self.SCORING_MODE_FIELD_MATCH,
            "json_validity_rate": round(valid_count / total, 4),
            "schema_compliance_rate": round(compliant_count / total, 4),
            "field_exact_match_rate": field_em_avg,
            "field_f1": field_f1_avg,
            "overall_em": overall_em_rate,
            # Legacy aliases for gate compat — exact_match keeps its
            # "whole-blob equality" meaning so existing gates don't
            # silently swap underneath them. f1 maps to the most useful
            # aggregate for extraction: mean per-field F1.
            "exact_match": overall_em_rate,
            "f1": field_f1_avg,
            "per_field": per_field_summary,
            "schema": schema,
            "total": total,
            "correct": overall_em_count,
        }

    # ── Span-set scoring (Phase 5.3.4b — for entity-list outputs) ──
    #
    # The PII/PCI demo motivates this, but the scoring mode is general
    # across span-extraction tasks: medical entity extraction, legal
    # clause extraction, financial entity extraction, generic NER —
    # anything whose output is a list of typed spans
    # ``[{type, start, end, text}, ...]``. Triggered by
    # ``output_schema.scoring_mode == "span_set"``; otherwise the
    # default field_match path runs (invoice-style extraction, etc.).

    @staticmethod
    def _entities_from_payload(parsed: Any) -> list[tuple[str, int, int, str]]:
        """Pull a list of (type, start, end, text) tuples from a parsed
        prediction or reference dict. Tolerant of bad rows: skips entries
        that aren't dicts or are missing required fields, so a malformed
        entity doesn't blow up the whole row's scoring."""

        if not isinstance(parsed, dict):
            return []
        raw = parsed.get("entities")
        if not isinstance(raw, list):
            return []
        out: list[tuple[str, int, int, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            ent_type = str(item.get("type") or "").strip()
            try:
                start = int(item.get("start"))
                end = int(item.get("end"))
            except (TypeError, ValueError):
                continue
            text = str(item.get("text") or "")
            if not ent_type:
                continue
            out.append((ent_type, start, end, text))
        return out

    @staticmethod
    def _entity_dict(entity: tuple[str, int, int, str]) -> dict[str, Any]:
        return {
            "type": entity[0],
            "start": entity[1],
            "end": entity[2],
            "text": entity[3],
        }

    @staticmethod
    def _tally_to_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        """Standard NER P/R/F1 from TP/FP/FN counts. Edge cases follow
        the CoNLL / SemEval convention: empty-on-empty is trivially
        correct (1.0); empty prediction with non-empty gold gets 0;
        non-empty prediction with empty gold gets 0."""

        if tp == 0 and fp == 0 and fn == 0:
            return 1.0, 1.0, 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    def _score_span_set(
        self,
        predictions: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        total = len(predictions)
        if total == 0:
            return {
                "scoring_mode": self.SCORING_MODE_SPAN_SET,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "exact_match": 0.0,
                "json_validity_rate": 0.0,
                "schema_compliance_rate": 0.0,
                "per_class": {},
                "total": 0,
                "correct": 0,
                "schema": schema,
            }

        # Required field set still drives schema compliance — for span_set
        # the required field is typically the entity list (e.g. "entities").
        required = schema.get("required") or []

        global_tp = 0
        global_fp = 0
        global_fn = 0
        per_class_counts: dict[str, dict[str, int]] = {}
        valid_count = 0
        compliant_count = 0
        overall_em_count = 0

        for prediction in predictions:
            raw_output = prediction.get("prediction") or ""
            raw_ref = prediction.get("reference") or ""
            parsed_pred = self._parse_json_safely(raw_output)
            parsed_ref = self._parse_json_safely(raw_ref)
            is_valid = isinstance(parsed_pred, dict)
            if is_valid:
                valid_count += 1

            missing: list[str] = []
            if is_valid and required:
                missing = [f for f in required if f not in parsed_pred]
            is_compliant = is_valid and not missing
            if is_compliant:
                compliant_count += 1

            pred_entities = self._entities_from_payload(parsed_pred) if is_valid else []
            gold_entities = self._entities_from_payload(parsed_ref)

            # Strict matching: same (type, start, end). Use Counters so
            # duplicates count correctly — if the same email appears
            # twice in the text and both are gold, the model has to
            # find both.
            from collections import Counter

            pred_keys = [(t, s, e) for (t, s, e, _) in pred_entities]
            gold_keys = [(t, s, e) for (t, s, e, _) in gold_entities]
            pred_counter = Counter(pred_keys)
            gold_counter = Counter(gold_keys)
            common = pred_counter & gold_counter
            row_tp = sum(common.values())
            row_fp = sum(pred_counter.values()) - row_tp
            row_fn = sum(gold_counter.values()) - row_tp

            global_tp += row_tp
            global_fp += row_fp
            global_fn += row_fn

            # Per-class tallies — union over all classes seen in either
            # side, per row.
            row_classes = set(t for (t, _, _) in pred_keys) | set(
                t for (t, _, _) in gold_keys
            )
            for cls in row_classes:
                if cls not in per_class_counts:
                    per_class_counts[cls] = {"tp": 0, "fp": 0, "fn": 0}
                cls_pred = Counter(k for k in pred_keys if k[0] == cls)
                cls_gold = Counter(k for k in gold_keys if k[0] == cls)
                cls_common = cls_pred & cls_gold
                cls_tp = sum(cls_common.values())
                per_class_counts[cls]["tp"] += cls_tp
                per_class_counts[cls]["fp"] += sum(cls_pred.values()) - cls_tp
                per_class_counts[cls]["fn"] += sum(cls_gold.values()) - cls_tp

            # Row-level matched / missed / hallucinated lists for the UI.
            common_keys_remaining = Counter(common)
            row_matched: list[dict[str, Any]] = []
            row_missed: list[dict[str, Any]] = []
            row_hallucinated: list[dict[str, Any]] = []
            for ent in gold_entities:
                key = (ent[0], ent[1], ent[2])
                if common_keys_remaining.get(key, 0) > 0:
                    common_keys_remaining[key] -= 1
                    row_matched.append(self._entity_dict(ent))
                else:
                    row_missed.append(self._entity_dict(ent))
            # FP: predicted entities whose key isn't in the (already-
            # consumed) common set.
            common_for_fp = Counter(common)
            for ent in pred_entities:
                key = (ent[0], ent[1], ent[2])
                if common_for_fp.get(key, 0) > 0:
                    common_for_fp[key] -= 1
                else:
                    row_hallucinated.append(self._entity_dict(ent))

            row_p, row_r, row_f1 = self._tally_to_metrics(row_tp, row_fp, row_fn)
            row_em = 1.0 if (row_fp == 0 and row_fn == 0) else 0.0
            if row_em == 1.0:
                overall_em_count += 1

            # In-place enrichment for predictions_preview → UI.
            prediction["parsed_prediction"] = parsed_pred
            prediction["parsed_reference"] = parsed_ref
            prediction["is_valid_json"] = is_valid
            prediction["missing_required_fields"] = missing
            prediction["scoring_mode"] = self.SCORING_MODE_SPAN_SET
            prediction["row_matched_entities"] = row_matched
            prediction["row_missed_entities"] = row_missed
            prediction["row_hallucinated_entities"] = row_hallucinated
            prediction["row_precision"] = round(row_p, 4)
            prediction["row_recall"] = round(row_r, 4)
            prediction["row_f1"] = round(row_f1, 4)
            prediction["row_exact_match"] = row_em

        precision, recall, f1 = self._tally_to_metrics(global_tp, global_fp, global_fn)
        per_class_summary: dict[str, dict[str, Any]] = {}
        for cls in sorted(per_class_counts.keys()):
            counts = per_class_counts[cls]
            p, r, c_f1 = self._tally_to_metrics(
                counts["tp"], counts["fp"], counts["fn"]
            )
            per_class_summary[cls] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(c_f1, 4),
                "support": counts["tp"] + counts["fn"],
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
            }
        # Macro = unweighted mean across classes (treat every class
        # equally — important for PII where SSN and email have wildly
        # different supports but both matter).
        macro_p = (
            round(sum(v["precision"] for v in per_class_summary.values()) / len(per_class_summary), 4)
            if per_class_summary
            else 0.0
        )
        macro_r = (
            round(sum(v["recall"] for v in per_class_summary.values()) / len(per_class_summary), 4)
            if per_class_summary
            else 0.0
        )
        macro_f1 = (
            round(sum(v["f1"] for v in per_class_summary.values()) / len(per_class_summary), 4)
            if per_class_summary
            else 0.0
        )

        return {
            "scoring_mode": self.SCORING_MODE_SPAN_SET,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "precision_macro": macro_p,
            "recall_macro": macro_r,
            "f1_macro": macro_f1,
            # Legacy gate-compat aliases — exact_match means "row had
            # zero FP and zero FN", the strictest possible signal.
            "exact_match": round(overall_em_count / total, 4),
            "json_validity_rate": round(valid_count / total, 4),
            "schema_compliance_rate": round(compliant_count / total, 4),
            "per_class": per_class_summary,
            "total": total,
            "correct": overall_em_count,
            "tp_total": global_tp,
            "fp_total": global_fp,
            "fn_total": global_fn,
            "schema": schema,
        }

    def wraps_own_prompt(self) -> bool:
        """``build_prompts`` produces the full instruction
        (``"Extract the following fields as JSON: …\\nInput: …\\nOutput:"``).
        Applying the model's chat template on top would wrap that
        whole instruction inside a user-message and the chat-tuned
        model would respond conversationally instead of emitting
        JSON. Skip the wrap so the constructed prompt reaches the
        model verbatim."""

        return True

    def expected_prompt_prefixes(self) -> list[str]:
        """See ClassificationHandler.expected_prompt_prefixes — same
        contract; used by the smoke check to detect train/eval
        format mismatches."""

        return [
            "Extract the following fields as JSON",
            "Extract the relevant fields from the input",
        ]


# ── RAGHandler (Phase 5.3.5) ──────────────────────────────────────────


class RAGHandler:
    """Task handler for grounded QA / retrieval-augmented generation.

    Three signals matter for a grounded bot, and they decompose cleanly:

    1. **Answer quality** — SQuAD EM/F1 of the prediction against the
       gold answer (the model-side metric). Inherits Phase 5.2's
       SQuAD-style normalization.
    2. **Faithfulness** — fraction of prediction tokens that appear
       in the retrieved context. Low faithfulness means hallucination.
       This is the model-side metric a grounded bot is *uniquely*
       judged on; pure QA models don't have it.
    3. **Context recall** — fraction of gold-answer tokens that
       appear in the context. Low context recall is a retriever
       problem, not a model problem — useful diagnostic for
       separating the two failure modes.

    Each prediction is enriched in place with the per-row signals so
    the UI can render a "Faithful" / "Hallucinated" badge + "Show
    context" disclosure and the engineer can debug both retrieval and
    generation without leaving the page.

    Falls back to QAHandler-style scoring (just EM/F1) when no
    ``context`` field is present on the row — so a project that
    transitions from plain QA to RAG can do so by adding a context
    column without rewriting eval pack gates.
    """

    profile_id: str = "rag_qa"

    # Per-row faithfulness threshold. A row with token-grounding ≥ this
    # is marked "faithful"; below is "hallucinated". 0.7 is a defensible
    # default — high enough to flag rampant hallucination, low enough
    # that legitimate paraphrasing isn't penalized.
    FAITHFULNESS_THRESHOLD: float = 0.7

    # Generation cap. Grounded answers should be short — multi-paragraph
    # answers usually mean the model lost the question.
    MAX_NEW_TOKENS_FLOOR: int = 64
    MAX_NEW_TOKENS_HARDCAP: int = 256

    # ── Row-field extraction ──

    def _extract_context(self, row: dict[str, Any]) -> str:
        for key in ("context", "passage", "document", "evidence", "retrieved_context"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_question(self, row: dict[str, Any]) -> str:
        for key in ("question", "query", "prompt", "instruction", "input"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_answer(self, row: dict[str, Any]) -> str:
        for key in ("answer", "reference", "completion", "response", "output", "target_text"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    # ── Prompt assembly ──

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        built: list[BuiltPrompt] = []
        for row in rows:
            context = self._extract_context(row)
            question = self._extract_question(row)
            answer = self._extract_answer(row)

            if context:
                wrapped = (
                    "Answer the question using only the context. If the "
                    "context does not contain the answer, say you don't "
                    "know.\n"
                    f"Context: {context}\n"
                    f"Question: {question}\n"
                    "Answer:"
                )
                extras: dict[str, Any] = {
                    "rag_context": context,
                    "rag_question": question,
                    "rag_has_context": True,
                }
            else:
                # Fall through to plain QA-style prompting when no context
                # is available. SQuAD EM/F1 still works; faithfulness /
                # context_recall just won't be computed.
                wrapped = question
                extras = {"rag_has_context": False}

            built.append(BuiltPrompt(prompt=wrapped, reference=answer, extras=extras))
        return built

    # ── Generation hint ──

    def max_new_tokens_override(self, default: int) -> int:
        """Grounded answers should be short. Raise tiny defaults to a
        sane floor; cap at 256 so the model can't ramble into
        hallucination territory."""

        baseline = max(self.MAX_NEW_TOKENS_FLOOR, int(default or 0))
        return min(self.MAX_NEW_TOKENS_HARDCAP, baseline)

    def stop_sequences(self, ctx: EvalContext) -> list[str]:
        """Stop when the model restarts the Context/Question pattern —
        that's the model rambling into "here's another grounded QA
        example" mode after giving its answer."""

        return [
            "\nContext:",
            "\n\nContext:",
            "\nQuestion:",
            "\n\nQuestion:",
            "\nExercise",
        ]

    # ── Token helpers ──

    @staticmethod
    def _normalize_to_tokens(text: str) -> list[str]:
        """Same SQuAD normalization Phase 5.2 uses for EM/F1, returned as
        a token list so we can do Counter intersections."""

        from app.services.evaluation_service import _normalize_answer

        normalized = _normalize_answer(text)
        return normalized.split() if normalized else []

    @staticmethod
    def _grounding_ratio(
        candidate_tokens: list[str],
        haystack_tokens: list[str],
    ) -> float:
        """Fraction of ``candidate_tokens`` that have a multiset match
        in ``haystack_tokens``. Returns 1.0 when candidate is empty
        (trivially grounded — a "I don't know" answer can't
        hallucinate).
        """

        if not candidate_tokens:
            return 1.0
        if not haystack_tokens:
            return 0.0
        from collections import Counter

        cand_counter = Counter(candidate_tokens)
        hay_counter = Counter(haystack_tokens)
        common = cand_counter & hay_counter
        grounded = sum(common.values())
        return grounded / len(candidate_tokens)

    # ── Scoring ──

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        from app.services.evaluation_service import exact_match, f1_score

        total = len(predictions)
        if total == 0:
            return {
                "exact_match": 0.0,
                "f1": 0.0,
                "faithfulness_rate": 0.0,
                "faithfulness_score_mean": 0.0,
                "context_recall_mean": 0.0,
                "unsupported_token_rate_mean": 0.0,
                "grounded_rows": 0,
                "rows_with_context": 0,
                "total": 0,
                "correct": 0,
            }

        em_scores: list[float] = []
        f1_scores: list[float] = []
        faithfulness_scores: list[float] = []
        context_recall_scores: list[float] = []
        unsupported_rates: list[float] = []
        rows_with_context = 0
        faithful_rows = 0

        for prediction in predictions:
            pred_text = str(prediction.get("prediction") or "")
            ref_text = str(prediction.get("reference") or "")
            context = str(prediction.get("rag_context") or "")
            has_context = bool(prediction.get("rag_has_context"))

            # Always-on: SQuAD EM/F1 on the answer span.
            row_em = exact_match(pred_text, ref_text)
            row_f1 = f1_score(pred_text, ref_text)
            em_scores.append(row_em)
            f1_scores.append(row_f1)

            row_faithfulness: float | None = None
            row_context_recall: float | None = None
            row_unsupported: float | None = None

            if has_context and context:
                rows_with_context += 1
                pred_tokens = self._normalize_to_tokens(pred_text)
                ref_tokens = self._normalize_to_tokens(ref_text)
                ctx_tokens = self._normalize_to_tokens(context)

                row_faithfulness = self._grounding_ratio(pred_tokens, ctx_tokens)
                row_context_recall = self._grounding_ratio(ref_tokens, ctx_tokens)
                row_unsupported = (
                    1.0 - row_faithfulness if pred_tokens else 0.0
                )

                faithfulness_scores.append(row_faithfulness)
                context_recall_scores.append(row_context_recall)
                unsupported_rates.append(row_unsupported)

                if row_faithfulness >= self.FAITHFULNESS_THRESHOLD:
                    faithful_rows += 1

            # In-place enrichment for predictions_preview → UI.
            prediction["row_exact_match"] = row_em
            prediction["row_f1"] = row_f1
            prediction["rag_has_context"] = has_context
            if row_faithfulness is not None:
                prediction["rag_faithfulness"] = round(row_faithfulness, 4)
                prediction["rag_context_recall"] = round(row_context_recall, 4)
                prediction["rag_unsupported_rate"] = round(row_unsupported, 4)
                prediction["rag_is_faithful"] = (
                    row_faithfulness >= self.FAITHFULNESS_THRESHOLD
                )

        em_mean = round(sum(em_scores) / total, 4)
        f1_mean = round(sum(f1_scores) / total, 4)

        # Faithfulness / context_recall / unsupported are only meaningful
        # for rows that actually had a context. Report both the mean
        # over context-rows and the "faithful_rate" (binary at threshold).
        faithfulness_mean = (
            round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
            if faithfulness_scores
            else 0.0
        )
        context_recall_mean = (
            round(sum(context_recall_scores) / len(context_recall_scores), 4)
            if context_recall_scores
            else 0.0
        )
        unsupported_mean = (
            round(sum(unsupported_rates) / len(unsupported_rates), 4)
            if unsupported_rates
            else 0.0
        )
        faithful_rate = (
            round(faithful_rows / rows_with_context, 4)
            if rows_with_context > 0
            else 0.0
        )

        return {
            # Legacy gate aliases — preserve QA-style EM/F1 so eval-pack
            # gates keyed on those metric IDs keep working.
            "exact_match": em_mean,
            "f1": f1_mean,
            # RAG-specific signals.
            "faithfulness_rate": faithful_rate,
            "faithfulness_score_mean": faithfulness_mean,
            "context_recall_mean": context_recall_mean,
            "unsupported_token_rate_mean": unsupported_mean,
            "grounded_rows": faithful_rows,
            "rows_with_context": rows_with_context,
            "total": total,
            "correct": int(sum(em_scores)),
        }

    def wraps_own_prompt(self) -> bool:
        """``build_prompts`` produces the full grounded prompt
        (``"Context: …\\nQuestion: …\\nAnswer:"``). Wrapping that as a
        chat user-message would split the context from the question
        across turns the chat template defines, and the model would
        respond to the entire wrapped block conversationally instead
        of answering the grounded question. Skip the chat-template
        wrap."""

        return True

    def expected_prompt_prefixes(self) -> list[str]:
        """See ClassificationHandler.expected_prompt_prefixes."""

        return ["Context:", "Question:"]


# ── SafetyHandler (Phase 5.3.8) ───────────────────────────────────────


class SafetyHandler:
    """Task handler for safety / refusal evaluation.

    Brings the existing ``SAFETY_PROMPTS`` + ``evaluate_safety_response``
    flow under the dispatcher arch so it lives next to every other
    eval shape and renders in the same Sample Predictions card.

    Row shape: each row carries a ``prompt`` and an optional
    ``test_type`` (``prompt_injection`` / ``secret_extraction`` /
    ``pii_regurgitation`` / ``jailbreak`` / ``unknown_answer`` /
    ``unknown``). The model's job is to refuse — emit a refusal-style
    response. The handler scores each row as refused / complied /
    ambiguous via the existing keyword heuristics and aggregates per
    test_type so the UI can show *"prompt_injection_resistance:
    98%, jailbreak_resistance: 85%"* instead of one coarse pass rate.

    Metrics produced: a top-line ``pass_rate`` (legacy gate compat)
    plus per-test-type rates so operators can gate on the categories
    that matter for their threat model.
    """

    profile_id: str = "safety"

    MAX_NEW_TOKENS_FLOOR: int = 64
    MAX_NEW_TOKENS_HARDCAP: int = 256

    def _extract_prompt(self, row: dict[str, Any]) -> str:
        for key in ("prompt", "question", "instruction", "input", "test_prompt"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_test_type(self, row: dict[str, Any]) -> str:
        raw = str(row.get("test_type") or "unknown").strip().lower()
        return raw or "unknown"

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        built: list[BuiltPrompt] = []
        for row in rows:
            prompt_text = self._extract_prompt(row)
            test_type = self._extract_test_type(row)
            # Reference is a sentinel — safety eval doesn't have a
            # "gold answer", it checks the model's response shape.
            # Keep the sentinel as the test_type label so per-row UI
            # can see the category.
            built.append(
                BuiltPrompt(
                    prompt=prompt_text,
                    reference=test_type,
                    extras={"safety_test_type": test_type},
                )
            )
        return built

    def max_new_tokens_override(self, default: int) -> int:
        baseline = max(self.MAX_NEW_TOKENS_FLOOR, int(default or 0))
        return min(self.MAX_NEW_TOKENS_HARDCAP, baseline)

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        from app.services.evaluation_service import evaluate_safety_response

        total = len(predictions)
        if total == 0:
            return {
                "pass_rate": 0.0,
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "per_test_type": {},
                "exact_match": 0.0,
                "f1": 0.0,
                "total": 0,
                "correct": 0,
            }

        per_type_counts: dict[str, dict[str, int]] = {}
        passed_total = 0

        for prediction in predictions:
            # Safety evaluator looks at the raw response — predictions
            # from the inference path land in `prediction`; legacy
            # safety code path uses `response`. Accept either so this
            # handler works both as the eval pipeline scorer and as a
            # drop-in for the existing `eval_type=safety` flow.
            response = str(
                prediction.get("response") or prediction.get("prediction") or ""
            )
            test_type = str(
                prediction.get("safety_test_type")
                or prediction.get("test_type")
                or prediction.get("reference")
                or "unknown"
            ).lower()
            verdict = evaluate_safety_response(response, test_type)
            passed = bool(verdict.get("passed"))
            if passed:
                passed_total += 1

            per_type_counts.setdefault(
                test_type, {"passed": 0, "failed": 0, "total": 0}
            )
            per_type_counts[test_type]["total"] += 1
            if passed:
                per_type_counts[test_type]["passed"] += 1
            else:
                per_type_counts[test_type]["failed"] += 1

            # In-place enrichment: per-row pass/fail + verdict reason
            # for the UI's status badge + tooltip.
            prediction["safety_test_type"] = test_type
            prediction["safety_passed"] = passed
            prediction["safety_reason"] = str(verdict.get("reason") or "")
            prediction["row_exact_match"] = 1.0 if passed else 0.0
            prediction["row_f1"] = 1.0 if passed else 0.0

        per_test_type: dict[str, dict[str, Any]] = {}
        for tt, counts in per_type_counts.items():
            rate = (
                round(counts["passed"] / counts["total"], 4)
                if counts["total"] > 0
                else 0.0
            )
            per_test_type[tt] = {
                "pass_rate": rate,
                "passed": counts["passed"],
                "failed": counts["failed"],
                "total": counts["total"],
            }

        pass_rate = round(passed_total / total, 4)
        return {
            "pass_rate": pass_rate,
            "total_tests": total,
            "passed": passed_total,
            "failed": total - passed_total,
            "per_test_type": per_test_type,
            # Convenience headlines for the most-gated subcategories,
            # so eval-pack gates can key on them without per_test_type
            # path lookups.
            **{
                f"{tt}_pass_rate": data["pass_rate"]
                for tt, data in per_test_type.items()
            },
            # Legacy aliases so existing safety gates keyed on
            # exact_match / f1 keep resolving without a pack migration.
            "exact_match": pass_rate,
            "f1": pass_rate,
            "total": total,
            "correct": passed_total,
        }


# ── Multimodal handlers (Phase 5.3.7) ─────────────────────────────────


class VisionLanguageHandler:
    """Task handler for vision-language tasks (captioning + VQA).

    Two sub-tasks routed by ``manifest.subtask``:

    - ``captioning`` (default): model gets an image, emits a caption.
      Scored with BLEU-4 and ROUGE-L against the gold caption +
      legacy EM/F1. Mean prediction-token / reference-token ratio
      reported as ``length_ratio`` for spotting over-/under-generation.
    - ``vqa``: model gets an image plus a question, emits an answer.
      Scored with SQuAD EM/F1.

    Image path flows through ``image_path`` on each row (set by the
    existing modality-detection in ``_load_heldout_pairs``). The
    handler does NOT load images itself — that's the inference path's
    job for a real vision-language model. The handler is correct on
    any caller that uses a multimodal-aware inference runtime
    (vision2seq, PaliGemma, etc.); for plain-text inference it scores
    correctly but the input image is invisible to the model.

    CIDEr scoring (the classic captioning metric) is intentionally
    deferred — ``pycocoevalcap`` has a Java dependency on some
    sub-scorers. BLEU + ROUGE are decent caption-quality signal and
    install clean.
    """

    profile_id: str = "vision_language"

    SUBTASK_CAPTIONING: str = "captioning"
    SUBTASK_VQA: str = "vqa"
    DEFAULT_SUBTASK: str = "captioning"
    _SUPPORTED_SUBTASKS: set[str] = {SUBTASK_CAPTIONING, SUBTASK_VQA}

    MAX_NEW_TOKENS_FLOOR: int = 32
    MAX_NEW_TOKENS_HARDCAP: int = 256

    def __init__(self) -> None:
        self._cached_subtask: str | None = None

    def _resolve_subtask(self, ctx: EvalContext) -> str:
        if self._cached_subtask is not None:
            return self._cached_subtask
        raw = str(ctx.manifest.get("subtask") or "").strip().lower()
        self._cached_subtask = (
            raw if raw in self._SUPPORTED_SUBTASKS else self.DEFAULT_SUBTASK
        )
        return self._cached_subtask

    def _extract_question(self, row: dict[str, Any]) -> str:
        for key in ("question", "prompt", "instruction", "query"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_caption_or_answer(self, row: dict[str, Any]) -> str:
        for key in (
            "caption",
            "answer",
            "reference",
            "target_text",
            "completion",
            "response",
            "output",
        ):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_image_path(self, row: dict[str, Any]) -> str:
        for key in ("image_path", "image", "image_url"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        subtask = self._resolve_subtask(ctx)
        built: list[BuiltPrompt] = []
        for row in rows:
            image_path = self._extract_image_path(row)
            reference = self._extract_caption_or_answer(row)
            if subtask == self.SUBTASK_VQA:
                question = self._extract_question(row)
                # Vision-language inference paths usually expect the
                # image to be passed alongside the prompt, not embedded
                # in it. We carry the image_path through extras and emit
                # a token reference in the prompt for plain-text models
                # that still want to know an image was attached.
                prompt = (
                    f"Question: {question}\n"
                    f"Image: <image:{image_path}>\nAnswer:"
                    if image_path
                    else question
                )
            else:
                prompt = (
                    f"Describe the image: <image:{image_path}>\nCaption:"
                    if image_path
                    else "Describe the image:"
                )
            extras: dict[str, Any] = {
                "vl_subtask": subtask,
                "vl_image_path": image_path,
            }
            built.append(BuiltPrompt(prompt=prompt, reference=reference, extras=extras))
        return built

    def max_new_tokens_override(self, default: int) -> int:
        baseline = max(self.MAX_NEW_TOKENS_FLOOR, int(default or 0))
        return min(self.MAX_NEW_TOKENS_HARDCAP, baseline)

    def _score_captioning(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        if not predictions:
            return {"bleu_4": 0.0, "rouge_l": 0.0}
        from sacrebleu.metrics import BLEU
        from rouge_score import rouge_scorer

        bleu = BLEU(effective_order=True)
        bleu_result = bleu.corpus_score(list(predictions), [list(references)])
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l_total = 0.0
        for pred, ref in zip(predictions, references):
            rouge_l_total += float(
                scorer.score(ref or "", pred or "")["rougeL"].fmeasure
            )
        return {
            "bleu_4": round(float(bleu_result.score) / 100.0, 4),
            "rouge_l": round(rouge_l_total / len(predictions), 4),
        }

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        from app.services.evaluation_service import exact_match, f1_score

        subtask = self._resolve_subtask(ctx)
        total = len(predictions)
        if total == 0:
            return {
                "subtask": subtask,
                "exact_match": 0.0,
                "f1": 0.0,
                "total": 0,
            }

        pred_texts = [str(p.get("prediction") or "") for p in predictions]
        ref_texts = [str(p.get("reference") or "") for p in predictions]
        em_scores = [exact_match(p, r) for p, r in zip(pred_texts, ref_texts)]
        f1_scores = [f1_score(p, r) for p, r in zip(pred_texts, ref_texts)]

        # In-place enrichment for per-row badges.
        for idx, prediction in enumerate(predictions):
            prediction["row_exact_match"] = em_scores[idx]
            prediction["row_f1"] = f1_scores[idx]
            prediction["vl_subtask"] = subtask

        metrics: dict[str, Any] = {
            "subtask": subtask,
            "exact_match": round(sum(em_scores) / total, 4),
            "f1": round(sum(f1_scores) / total, 4),
            "total": total,
            "correct": int(sum(em_scores)),
        }
        if subtask == self.SUBTASK_CAPTIONING:
            pred_token_total = sum(len((p or "").split()) for p in pred_texts)
            ref_token_total = sum(len((r or "").split()) for r in ref_texts)
            metrics["length_ratio"] = (
                round(pred_token_total / ref_token_total, 4)
                if ref_token_total > 0
                else 0.0
            )
            metrics.update(self._score_captioning(pred_texts, ref_texts))
        return metrics

    def wraps_own_prompt(self) -> bool:
        """``build_prompts`` produces task-specific multimodal prompts
        (``"Question: …\\nImage: <image:…>\\nAnswer:"`` for VQA,
        ``"Describe the image: <image:…>\\nCaption:"`` for captioning).
        The image-token placeholders and the question-answer scaffold
        must reach the model intact — wrapping in a chat template
        would either break the placeholder substitution or add a
        chat turn the model wasn't trained to expect for this task."""

        return True

    def expected_prompt_prefixes(self) -> list[str]:
        """See ClassificationHandler.expected_prompt_prefixes."""

        return ["Describe the image:", "Question:"]


class AudioTranscriptHandler:
    """Task handler for audio transcription + audio-conditioned QA.

    Two sub-tasks:

    - ``transcription`` (default): model emits a transcript. Scored
      with WER (word error rate) and CER (character error rate) via
      ``jiwer``, plus legacy F1 / EM for gate compat.
    - ``audio_qa``: model gets an audio clip + question, emits an
      answer. Scored with SQuAD EM/F1.

    Same boundary as VisionLanguageHandler: the handler doesn't load
    audio; that's the inference runtime's job. Scoring is correct on
    any caller using an audio-aware runtime (Whisper, SeamlessM4T,
    etc.) and falls back to text-only scoring for plain runtimes.
    """

    profile_id: str = "audio_transcript"

    SUBTASK_TRANSCRIPTION: str = "transcription"
    SUBTASK_AUDIO_QA: str = "audio_qa"
    DEFAULT_SUBTASK: str = "transcription"
    _SUPPORTED_SUBTASKS: set[str] = {SUBTASK_TRANSCRIPTION, SUBTASK_AUDIO_QA}

    MAX_NEW_TOKENS_FLOOR: int = 64
    MAX_NEW_TOKENS_HARDCAP: int = 512

    def __init__(self) -> None:
        self._cached_subtask: str | None = None

    def _resolve_subtask(self, ctx: EvalContext) -> str:
        if self._cached_subtask is not None:
            return self._cached_subtask
        raw = str(ctx.manifest.get("subtask") or "").strip().lower()
        self._cached_subtask = (
            raw if raw in self._SUPPORTED_SUBTASKS else self.DEFAULT_SUBTASK
        )
        return self._cached_subtask

    def _extract_question(self, row: dict[str, Any]) -> str:
        for key in ("question", "prompt", "instruction", "query"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_transcript_or_answer(self, row: dict[str, Any]) -> str:
        for key in (
            "transcript",
            "answer",
            "reference",
            "target_text",
            "completion",
            "response",
            "output",
        ):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_audio_path(self, row: dict[str, Any]) -> str:
        for key in ("audio_path", "audio", "audio_url"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        subtask = self._resolve_subtask(ctx)
        built: list[BuiltPrompt] = []
        for row in rows:
            audio_path = self._extract_audio_path(row)
            reference = self._extract_transcript_or_answer(row)
            if subtask == self.SUBTASK_AUDIO_QA:
                question = self._extract_question(row)
                prompt = (
                    f"Question: {question}\n"
                    f"Audio: <audio:{audio_path}>\nAnswer:"
                    if audio_path
                    else question
                )
            else:
                prompt = (
                    f"Transcribe the audio: <audio:{audio_path}>\nTranscript:"
                    if audio_path
                    else "Transcribe the audio:"
                )
            extras: dict[str, Any] = {
                "audio_subtask": subtask,
                "audio_path": audio_path,
            }
            built.append(BuiltPrompt(prompt=prompt, reference=reference, extras=extras))
        return built

    def max_new_tokens_override(self, default: int) -> int:
        baseline = max(self.MAX_NEW_TOKENS_FLOOR, int(default or 0))
        return min(self.MAX_NEW_TOKENS_HARDCAP, baseline)

    def _score_wer_cer(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        if not predictions:
            return {"wer": 0.0, "cer": 0.0}
        # jiwer is lazy-imported so the rest of the eval pipeline
        # doesn't pay the load cost when nobody runs audio eval.
        import jiwer

        # jiwer expects non-empty references for WER; replace empties
        # with a single-char marker so the metric doesn't crash, and
        # rely on per-row WER values being clearly bad in that case.
        refs = [r if r else "<empty>" for r in references]
        preds = [p if p else "" for p in predictions]
        try:
            wer = float(jiwer.wer(refs, preds))
        except Exception:
            wer = 1.0
        try:
            cer = float(jiwer.cer(refs, preds))
        except Exception:
            cer = 1.0
        return {"wer": round(wer, 4), "cer": round(cer, 4)}

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        from app.services.evaluation_service import exact_match, f1_score

        subtask = self._resolve_subtask(ctx)
        total = len(predictions)
        if total == 0:
            return {
                "subtask": subtask,
                "exact_match": 0.0,
                "f1": 0.0,
                "total": 0,
            }

        pred_texts = [str(p.get("prediction") or "") for p in predictions]
        ref_texts = [str(p.get("reference") or "") for p in predictions]
        em_scores = [exact_match(p, r) for p, r in zip(pred_texts, ref_texts)]
        f1_scores = [f1_score(p, r) for p, r in zip(pred_texts, ref_texts)]

        for idx, prediction in enumerate(predictions):
            prediction["row_exact_match"] = em_scores[idx]
            prediction["row_f1"] = f1_scores[idx]
            prediction["audio_subtask"] = subtask

        metrics: dict[str, Any] = {
            "subtask": subtask,
            "exact_match": round(sum(em_scores) / total, 4),
            "f1": round(sum(f1_scores) / total, 4),
            "total": total,
            "correct": int(sum(em_scores)),
        }
        if subtask == self.SUBTASK_TRANSCRIPTION:
            metrics.update(self._score_wer_cer(pred_texts, ref_texts))
        return metrics

    def wraps_own_prompt(self) -> bool:
        """``build_prompts`` produces task-specific audio prompts
        (``"Transcribe the audio: <audio:…>\\nTranscript:"`` or
        ``"Question: …\\nAudio: <audio:…>\\nAnswer:"``). The audio
        placeholders must reach the model intact + the task-specific
        instruction scaffold defines the response format — wrapping
        as a chat turn would either break the placeholder or convert
        the model into chat-completion mode."""

        return True

    def expected_prompt_prefixes(self) -> list[str]:
        """See ClassificationHandler.expected_prompt_prefixes."""

        return ["Transcribe the audio:", "Question:"]


# ── AlignmentHandler (Phase 5.3.6) ────────────────────────────────────


class AlignmentHandler:
    """Task handler for alignment / preference-trained models (DPO, ORPO).

    Row shape requires ``prompt``, ``chosen``, ``rejected``. After DPO/
    ORPO training, the model is expected to generate outputs that
    resemble ``chosen`` more than ``rejected``. This handler measures
    that by:

    1. **Similarity-to-chosen vs similarity-to-rejected**: token-level
       F1 of the model's output against both completions. Cheap,
       deterministic, and works with the existing generation
       pipeline. A row's "preference correct" if F1(output, chosen) >
       F1(output, rejected).
    2. **Mean alignment margin**: average gap between similarity-to-
       chosen and similarity-to-rejected across all rows. Positive
       margin = model learned the preference; near-zero = model
       didn't differentiate; negative = model preferred the rejected.

    Note: this is a similarity-based proxy for "judge win-rate".
    A heavier judge-based mode (LLM compares output to {chosen,
    rejected} pairwise) would land as an opt-in mode in a follow-up.
    Log-prob-margin mode (compute logp(chosen|prompt) -
    logp(rejected|prompt) on the trained model) is the gold standard
    for DPO eval but requires inference-path changes to expose
    per-completion log-probs — also a follow-up.
    """

    profile_id: str = "alignment"

    # Generation cap. Alignment outputs can be moderately long
    # (paragraphs of preferred-style response). 256 is a defensible
    # ceiling that matches typical chat / instruction-tuned outputs.
    MAX_NEW_TOKENS_FLOOR: int = 64
    MAX_NEW_TOKENS_HARDCAP: int = 256

    # ── Row-field extraction ──

    def _extract_prompt(self, row: dict[str, Any]) -> str:
        for key in ("prompt", "question", "instruction", "input"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_chosen(self, row: dict[str, Any]) -> str:
        for key in ("chosen", "preferred", "accepted", "response_chosen", "answer"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_rejected(self, row: dict[str, Any]) -> str:
        for key in ("rejected", "dispreferred", "response_rejected", "negative"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    # ── Prompt assembly ──

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        built: list[BuiltPrompt] = []
        for row in rows:
            prompt_text = self._extract_prompt(row)
            chosen = self._extract_chosen(row)
            rejected = self._extract_rejected(row)
            # Reference is the chosen completion — preserves SQuAD-style
            # EM/F1 against the preferred completion (legacy gate compat).
            built.append(
                BuiltPrompt(
                    prompt=prompt_text,
                    reference=chosen,
                    extras={
                        "alignment_chosen": chosen,
                        "alignment_rejected": rejected,
                        "alignment_has_pair": bool(chosen and rejected),
                    },
                )
            )
        return built

    def max_new_tokens_override(self, default: int) -> int:
        baseline = max(self.MAX_NEW_TOKENS_FLOOR, int(default or 0))
        return min(self.MAX_NEW_TOKENS_HARDCAP, baseline)

    # ── Scoring ──

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        from app.services.evaluation_service import exact_match, f1_score

        total = len(predictions)
        if total == 0:
            return {
                "preference_accuracy": 0.0,
                "mean_alignment_margin": 0.0,
                "chosen_alignment_mean": 0.0,
                "rejected_alignment_mean": 0.0,
                "exact_match": 0.0,
                "f1": 0.0,
                "total": 0,
                "correct": 0,
                "rows_with_pair": 0,
            }

        em_scores: list[float] = []
        f1_scores: list[float] = []
        chosen_sims: list[float] = []
        rejected_sims: list[float] = []
        margins: list[float] = []
        correct_preference = 0
        rows_with_pair = 0

        for prediction in predictions:
            pred_text = str(prediction.get("prediction") or "")
            chosen = str(prediction.get("alignment_chosen") or prediction.get("reference") or "")
            rejected = str(prediction.get("alignment_rejected") or "")
            has_pair = bool(prediction.get("alignment_has_pair") or (chosen and rejected))

            row_em = exact_match(pred_text, chosen)
            row_f1 = f1_score(pred_text, chosen)
            em_scores.append(row_em)
            f1_scores.append(row_f1)

            row_chosen_sim: float | None = None
            row_rejected_sim: float | None = None
            row_margin: float | None = None
            row_preference_correct: bool | None = None

            if has_pair and chosen and rejected:
                rows_with_pair += 1
                row_chosen_sim = f1_score(pred_text, chosen)
                row_rejected_sim = f1_score(pred_text, rejected)
                row_margin = row_chosen_sim - row_rejected_sim
                row_preference_correct = row_chosen_sim > row_rejected_sim
                chosen_sims.append(row_chosen_sim)
                rejected_sims.append(row_rejected_sim)
                margins.append(row_margin)
                if row_preference_correct:
                    correct_preference += 1

            # In-place enrichment for predictions_preview → UI.
            prediction["row_exact_match"] = row_em
            prediction["row_f1"] = row_f1
            prediction["alignment_has_pair"] = has_pair
            if row_chosen_sim is not None:
                prediction["alignment_chosen_sim"] = round(row_chosen_sim, 4)
                prediction["alignment_rejected_sim"] = round(row_rejected_sim, 4)
                prediction["alignment_margin"] = round(row_margin, 4)
                prediction["alignment_preference_correct"] = bool(row_preference_correct)

        preference_accuracy = (
            round(correct_preference / rows_with_pair, 4) if rows_with_pair > 0 else 0.0
        )
        mean_margin = round(sum(margins) / len(margins), 4) if margins else 0.0
        chosen_mean = round(sum(chosen_sims) / len(chosen_sims), 4) if chosen_sims else 0.0
        rejected_mean = (
            round(sum(rejected_sims) / len(rejected_sims), 4) if rejected_sims else 0.0
        )

        return {
            "preference_accuracy": preference_accuracy,
            "mean_alignment_margin": mean_margin,
            "chosen_alignment_mean": chosen_mean,
            "rejected_alignment_mean": rejected_mean,
            # Legacy aliases for gate compat — EM/F1 against the
            # CHOSEN completion (the preferred output).
            "exact_match": round(sum(em_scores) / total, 4),
            "f1": round(sum(f1_scores) / total, 4),
            "total": total,
            "correct": correct_preference,
            "rows_with_pair": rows_with_pair,
        }


# ── Seq2SeqHandler (Phase 5.3.3) ──────────────────────────────────────


class Seq2SeqHandler:
    """Task handler for seq2seq tasks (translation, summarization, paraphrase).

    Wraps each row with a sub-task-specific instruction template, generates
    a free-form completion, then scores with sub-task-appropriate metrics:

    - translation  → BLEU + chrF (via ``sacrebleu``) + legacy EM/F1.
    - summarization → ROUGE-1/2/L (via ``rouge_score``) + legacy EM/F1.
    - paraphrase   → BLEU + ROUGE (paraphrase wants both lexical and
                     structural overlap signal) + legacy EM/F1.

    All sub-tasks additionally report ``length_ratio`` (mean prediction
    tokens / mean reference tokens) — useful for spotting models that
    over- or under-generate independently of content quality. Legacy
    ``exact_match`` / ``f1`` aliases are produced so eval-pack gates
    keyed on those IDs keep resolving without a pack migration.

    Sub-task is read from ``prepared/manifest.json`` (``subtask`` field).
    Missing / unknown sub-task defaults to summarization, which is the
    most common case and a safe ROUGE-based fallback.
    """

    profile_id: str = "seq2seq"

    SUBTASK_TRANSLATION: str = "translation"
    SUBTASK_SUMMARIZATION: str = "summarization"
    SUBTASK_PARAPHRASE: str = "paraphrase"
    DEFAULT_SUBTASK: str = "summarization"
    _SUPPORTED_SUBTASKS: set[str] = {
        SUBTASK_TRANSLATION,
        SUBTASK_SUMMARIZATION,
        SUBTASK_PARAPHRASE,
    }

    # Cap generation at 1.5× the longest reference, hard-bounded so a
    # pathological row can't blow the budget. The caller's max_new_tokens
    # acts as a floor — we never *reduce* below what the caller asked for.
    LENGTH_MULTIPLIER: float = 1.5
    MAX_NEW_TOKENS_HARDCAP: int = 512

    def __init__(self) -> None:
        self._max_ref_tokens: int = 0
        self._cached_subtask: str | None = None

    # ── Sub-task resolution ──

    def _resolve_subtask(self, ctx: EvalContext) -> str:
        if self._cached_subtask is not None:
            return self._cached_subtask
        raw = str(ctx.manifest.get("subtask") or "").strip().lower()
        if raw in self._SUPPORTED_SUBTASKS:
            self._cached_subtask = raw
        else:
            self._cached_subtask = self.DEFAULT_SUBTASK
        return self._cached_subtask

    def _resolve_tgt_lang(self, ctx: EvalContext) -> str:
        raw = ctx.manifest.get("tgt_lang") or ctx.manifest.get("target_language")
        text = str(raw or "").strip()
        return text or "the target language"

    # ── Row-field extraction ──

    def _extract_input_text(self, row: dict[str, Any]) -> str:
        for key in (
            "source_text",
            "text",
            "input",
            "source",
            "prompt",
            "question",
            "instruction",
            "body",
            "content",
            "article",
            "document",
        ):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_reference(self, row: dict[str, Any]) -> str:
        for key in (
            "target_text",
            "reference",
            "target",
            "completion",
            "output",
            "response",
            "answer",
            "summary",
            "translation",
            "paraphrase",
        ):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    # ── Prompt assembly ──

    def _build_prompt_text(self, input_text: str, subtask: str, tgt_lang: str) -> str:
        if subtask == self.SUBTASK_TRANSLATION:
            return f"Translate the following to {tgt_lang}.\nText: {input_text}\nTranslation:"
        if subtask == self.SUBTASK_PARAPHRASE:
            return f"Paraphrase the following text in different words.\nText: {input_text}\nParaphrase:"
        # Default: summarization.
        return f"Summarize the following text concisely.\nText: {input_text}\nSummary:"

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        subtask = self._resolve_subtask(ctx)
        tgt_lang = self._resolve_tgt_lang(ctx)
        built: list[BuiltPrompt] = []
        max_ref_tokens = 0
        for row in rows:
            input_text = self._extract_input_text(row)
            reference = self._extract_reference(row)
            ref_tokens = len(reference.split())
            if ref_tokens > max_ref_tokens:
                max_ref_tokens = ref_tokens
            wrapped = self._build_prompt_text(input_text, subtask, tgt_lang)
            extras: dict[str, Any] = {
                "seq2seq_subtask": subtask,
                "seq2seq_input": input_text,
            }
            if subtask == self.SUBTASK_TRANSLATION:
                extras["seq2seq_tgt_lang"] = tgt_lang
            built.append(
                BuiltPrompt(prompt=wrapped, reference=reference, extras=extras)
            )
        self._max_ref_tokens = max_ref_tokens
        return built

    # ── Generation hint ──

    def max_new_tokens_override(self, default: int) -> int:
        """Ensure generation has room for the longest reference, capped
        at the hard limit. Unlike ClassificationHandler this only raises
        a too-low default — never reduces it — because seq2seq outputs
        legitimately need length to be correct.
        """

        baseline = max(1, int(default or 1))
        if self._max_ref_tokens > 0:
            suggested = int(self._max_ref_tokens * self.LENGTH_MULTIPLIER)
            baseline = max(baseline, suggested)
        return min(self.MAX_NEW_TOKENS_HARDCAP, baseline)

    # ── Scoring ──

    def _score_bleu_chrf(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Corpus-level BLEU + chrF via sacrebleu. Sentinel zeros when
        the dataset is empty (consistent with other metrics here)."""

        if not predictions:
            return {"bleu": 0.0, "chrf": 0.0}
        # sacrebleu is imported lazily so the rest of the eval pipeline
        # doesn't pay the import cost when nobody runs seq2seq.
        from sacrebleu.metrics import BLEU, CHRF

        bleu = BLEU(effective_order=True)
        chrf = CHRF()
        # sacrebleu expects refs as list-of-lists ([refs_per_system]).
        ref_lists = [list(references)]
        bleu_result = bleu.corpus_score(list(predictions), ref_lists)
        chrf_result = chrf.corpus_score(list(predictions), ref_lists)
        # sacrebleu reports 0–100; normalize to 0–1 so it aligns with our
        # other metric IDs (accuracy, f1, rouge) which are all 0–1.
        return {
            "bleu": round(float(bleu_result.score) / 100.0, 4),
            "chrf": round(float(chrf_result.score) / 100.0, 4),
        }

    def _score_rouge(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Mean ROUGE-1 / ROUGE-2 / ROUGE-L across rows."""

        if not predictions:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        r1_total = 0.0
        r2_total = 0.0
        rl_total = 0.0
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref or "", pred or "")
            r1_total += float(scores["rouge1"].fmeasure)
            r2_total += float(scores["rouge2"].fmeasure)
            rl_total += float(scores["rougeL"].fmeasure)
        n = len(predictions)
        return {
            "rouge_1": round(r1_total / n, 4),
            "rouge_2": round(r2_total / n, 4),
            "rouge_l": round(rl_total / n, 4),
        }

    def _legacy_em_f1(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        # Lazy import to avoid the cyclic dependency (evaluation_service
        # imports this module too).
        from app.services.evaluation_service import exact_match, f1_score

        if not predictions:
            return {"exact_match": 0.0, "f1": 0.0}
        em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
        f1_scores = [f1_score(p, r) for p, r in zip(predictions, references)]
        n = len(predictions)
        return {
            "exact_match": round(sum(em_scores) / n, 4),
            "f1": round(sum(f1_scores) / n, 4),
        }

    def _length_ratio(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        if not predictions:
            return 0.0
        pred_tokens = sum(len((p or "").split()) for p in predictions)
        ref_tokens = sum(len((r or "").split()) for r in references)
        if ref_tokens == 0:
            return 0.0
        return round(pred_tokens / ref_tokens, 4)

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        subtask = self._resolve_subtask(ctx)
        pred_texts: list[str] = [str(p.get("prediction") or "") for p in predictions]
        ref_texts: list[str] = [str(p.get("reference") or "") for p in predictions]
        total = len(predictions)

        metrics: dict[str, Any] = {
            "subtask": subtask,
            "total": total,
            "length_ratio": self._length_ratio(pred_texts, ref_texts),
        }
        # Always alongside: legacy EM/F1 so gates keyed on those keep
        # working without a pack migration.
        metrics.update(self._legacy_em_f1(pred_texts, ref_texts))

        if subtask == self.SUBTASK_TRANSLATION:
            metrics.update(self._score_bleu_chrf(pred_texts, ref_texts))
        elif subtask == self.SUBTASK_PARAPHRASE:
            # Paraphrase wants both lexical (BLEU) and structural (ROUGE)
            # signal — produce both.
            metrics.update(self._score_bleu_chrf(pred_texts, ref_texts))
            metrics.update(self._score_rouge(pred_texts, ref_texts))
        else:  # summarization (default)
            metrics.update(self._score_rouge(pred_texts, ref_texts))

        return metrics

    def wraps_own_prompt(self) -> bool:
        """``build_prompts`` produces sub-task-specific instruction
        templates (``"Translate to French: …"`` /
        ``"Summarize: …\\nSummary:"`` / ``"Paraphrase: …\\nParaphrase:"``).
        These are flat-text instruction prompts, not chat turns —
        wrapping them in the model's chat template would force the
        Seq2Seq-tuned model into chat-completion mode and break the
        sub-task scaffold."""

        return True

    def expected_prompt_prefixes(self) -> list[str]:
        """See ClassificationHandler.expected_prompt_prefixes. Sub-task
        prefixes — at least one should appear in a correctly-prepared
        Seq2Seq training row."""

        return ["Translate", "Summarize:", "Paraphrase:"]


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
    experiment_task_type: str | None = None,
) -> tuple[EvalContext, TaskHandler]:
    """One-shot helper: read manifest, resolve handler, return both.

    Callers in ``evaluation_service`` use this to keep the dispatch
    site short.

    ``experiment_task_type`` is the experiment's ``config.task_type``
    field (e.g. ``classification``, ``seq2seq``, ``causal_lm``). When
    the prepared manifest doesn't declare a ``task_profile`` — which
    happens for projects ingested via dataset-import where the
    adapter doesn't surface it — we fall back to the experiment's
    task_type so the right handler still wins. Without this fallback,
    classification projects silently routed to QAHandler + got 0%
    on held-out eval because the chat-completion decoding path
    doesn't match the classification-head training path.
    """

    manifest = read_prepared_manifest(project_id)
    task_profile = read_task_profile_from_manifest(project_id)
    if not task_profile and experiment_task_type:
        # Map the experiment's task_type → handler key. Identity for
        # most values; ``causal_lm`` maps to the qa family the
        # registry covers under several aliases.
        mapped = _normalize_profile(experiment_task_type)
        if mapped == "causal_lm":
            mapped = "language_modeling"
        task_profile = mapped
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
register_handler("seq2seq", Seq2SeqHandler)
# QAHandler covers short-answer QA + instruction following + chat-SFT +
# generic language modeling. They all share the same "score the last
# answer-shaped span against the reference" semantics, so one handler
# serves all four manifest tags.
register_handler("qa", QAHandler)
register_handler("instruction_sft", QAHandler)
register_handler("chat_sft", QAHandler)
register_handler("language_modeling", QAHandler)
# Structured extraction (JSON outputs, field-level scoring).
register_handler("structured_extraction", StructuredExtractionHandler)
register_handler("extraction", StructuredExtractionHandler)
# Grounded QA / RAG — answer quality + faithfulness + context recall.
register_handler("rag_qa", RAGHandler)
register_handler("rag", RAGHandler)
register_handler("grounded_qa", RAGHandler)
# Alignment / preference (DPO, ORPO) — similarity-based preference scoring.
register_handler("dpo", AlignmentHandler)
register_handler("orpo", AlignmentHandler)
register_handler("alignment", AlignmentHandler)
register_handler("preference", AlignmentHandler)
# Multimodal — vision-language (captioning / VQA) and audio (transcription / QA).
register_handler("vision_language", VisionLanguageHandler)
register_handler("image_captioning", VisionLanguageHandler)
register_handler("vqa", VisionLanguageHandler)
register_handler("audio_transcript", AudioTranscriptHandler)
register_handler("audio_transcription", AudioTranscriptHandler)
register_handler("speech_to_text", AudioTranscriptHandler)
# Safety — refusal-rate eval with per-test-type breakdown.
register_handler("safety", SafetyHandler)
register_handler("refusal", SafetyHandler)


__all__ = [
    "AlignmentHandler",
    "AudioTranscriptHandler",
    "BuiltPrompt",
    "ClassificationHandler",
    "EvalContext",
    "GenericHandler",
    "QAHandler",
    "RAGHandler",
    "SafetyHandler",
    "Seq2SeqHandler",
    "StructuredExtractionHandler",
    "TaskHandler",
    "VisionLanguageHandler",
    "build_eval_context",
    "list_registered_profiles",
    "read_prepared_manifest",
    "read_task_profile_from_manifest",
    "register_handler",
    "resolve_task_handler",
]
