"""One task-shape detector + one vocabulary (Wave 2b).

Before this, five heuristics guessed "what kind of task is this data?" in
four vocabularies (task profiles, adapter ids, recipe ids, import-mapper
hypotheses), and the shape a user confirmed in the import wizard was saved
to ``selected_recipe`` while prep/training/eval read
``dataset_adapter_preset``. Now:

* ``TASK_SHAPES`` is the vocabulary — canonical task-profile names (what
  adapters, eval handlers and the trainer already speak), each mapped to
  the adapter, recipe and import mapper that implement it.
* ``detect_task_shape(rows, intent=)`` combines the column introspector
  (evidence), the data adapters (can the rows actually be mapped?) and
  the user's stated goal (a small prior) into ranked candidates with a
  plain-language rationale and an explicit ``needs_confirmation`` flag.
* ``confirm_task_shape`` persists the choice where prep/training/eval read
  it (``dataset_adapter_preset``) *and* snapshots the matching recipe.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

CONFIRM_THRESHOLD = 0.8
AMBIGUITY_MARGIN = 0.1
_INTENT_BONUS = 0.08

TASK_SHAPES: dict[str, dict[str, Any]] = {
    "qa": {
        "label": "Question answering",
        "description": "Each row is a question and the answer the model should give.",
        "adapter_id": "qa-pair",
        "recipe_id": "qa-sft",
        "mapper_id": "qa_pair_passthrough",
    },
    "rag_qa": {
        "label": "Answer from provided context",
        "description": "Question + a passage of context + the grounded answer.",
        "adapter_id": "rag-grounded",
        "recipe_id": "rag-protocol",
        "mapper_id": "rag_passthrough",
    },
    "classification": {
        "label": "Classification",
        "description": "Text in, one label out (sentiment, routing, category…).",
        "adapter_id": "classification-label",
        "recipe_id": "classification",
        "mapper_id": "label_to_classification",
    },
    "summarization": {
        "label": "Summarization",
        "description": "A long text and its shorter summary.",
        # Same adapter as the summarization recipe (text → question,
        # summary → answer); the eval side is aligned to that shape.
        "adapter_id": "qa-pair",
        "recipe_id": "summarization",
        "mapper_id": "qa_pair_passthrough",
    },
    "structured_extraction": {
        "label": "Extract structured fields",
        "description": "Text in, fields/entities out (JSON, spans).",
        "adapter_id": "structured-extraction",
        "recipe_id": "span-extraction",
        "mapper_id": "kv_to_structured",
    },
    "chat_sft": {
        "label": "Chat conversations",
        "description": "Multi-turn user/assistant messages.",
        "adapter_id": "chat-messages",
        "recipe_id": "generic-sft",
        "mapper_id": "chat_messages_passthrough",
    },
    "preference": {
        "label": "Preference pairs",
        "description": "A prompt with a preferred and a rejected answer (DPO/ORPO).",
        "adapter_id": "preference-pair",
        "recipe_id": "generic-sft",
        "mapper_id": "preference_pair",
    },
    "tool_calling": {
        "label": "Tool / function calling",
        "description": "Requests paired with the tool call the model should make.",
        "adapter_id": "tool-call-json",
        "recipe_id": "generic-sft",
        "mapper_id": None,
    },
    "instruction_sft": {
        "label": "Instruction following",
        "description": "Instructions paired with the response to produce.",
        "adapter_id": "default-canonical",
        "recipe_id": "generic-sft",
        "mapper_id": "qa_pair_passthrough",
    },
    "language_modeling": {
        "label": "Plain text (learn the domain's language)",
        "description": "Documents or passages with no answer column.",
        "adapter_id": "default-canonical",
        "recipe_id": "generic-sft",
        "mapper_id": "text_only",
        # Plain documents train as continued pretraining (packed LM), not
        # SFT — see continued_pretraining_policy.
        "training_mode": "domain_pretrain",
    },
}

# Other vocabularies' names for the same shapes (introspector hypotheses,
# recipes, training modes, user-typed hints).
_PROFILE_ALIASES = {
    "dpo": "preference",
    "orpo": "preference",
    "alignment": "preference",
    "preference_pair": "preference",
    "chat": "chat_sft",
    "rag": "rag_qa",
    "grounded_qa": "rag_qa",
    "summary": "summarization",
    "summarize": "summarization",
    "extraction": "structured_extraction",
    "ner": "structured_extraction",
    "span_extraction": "structured_extraction",
    "sft": "instruction_sft",
    "causal_lm": "instruction_sft",
    "lm": "language_modeling",
    "text_only": "language_modeling",
    "seq2seq": "summarization",
}

# Adapters whose "can I map this row?" answer is a meaningful signal on
# its own (default-canonical maps nearly anything, so it isn't).
_DISCRIMINATING_ADAPTERS = {
    profile: spec["adapter_id"]
    for profile, spec in TASK_SHAPES.items()
    # default-canonical maps nearly anything; summarization shares qa-pair,
    # so only its column rule (text + summary) can tell it apart.
    if spec["adapter_id"] != "default-canonical" and profile != "summarization"
}

# introspector field_map keys → canonical record keys the adapters read.
_FIELD_MAP_TO_CANONICAL = {
    "question_field": "question",
    "answer_field": "answer",
    "text_field": "text",
    "label_field": "label",
    "context_field": "context",
    "prompt_field": "prompt",
    "chosen_field": "chosen",
    "rejected_field": "rejected",
    "messages_field": "messages",
}


def canonical_task_profile(value: str | None) -> str | None:
    token = str(value or "").strip().lower().replace("-", "_")
    if not token:
        return None
    token = _PROFILE_ALIASES.get(token, token)
    return token if token in TASK_SHAPES else None


def _intent_profiles(intent: str | None) -> set[str]:
    text = str(intent or "").strip().lower()
    if not text:
        return set()
    from app.services.newbie_autopilot_service import _INTENT_PRESETS

    # One prior, two keyword vocabularies: the autopilot presets and the
    # project-brief task keywords (domain_blueprint_service).
    keyword_table: list[tuple[str, tuple[str, ...]]] = [
        (str(preset.get("task_profile") or ""), tuple(preset.get("keywords") or ()))
        for preset in _INTENT_PRESETS
    ]
    try:
        from app.services.domain_blueprint_service import TASK_KEYWORDS

        keyword_table.extend(TASK_KEYWORDS.items())
    except Exception:  # noqa: BLE001
        pass
    matched: set[str] = set()
    for task_profile, keywords in keyword_table:
        if any(keyword in text for keyword in keywords):
            profile = canonical_task_profile(task_profile)
            if profile:
                matched.add(profile)
    return matched


def _map_rate(rows: list[dict[str, Any]], adapter_id: str, field_map: dict[str, Any] | None) -> float:
    if not rows:
        return 0.0
    from app.services.data_adapter_service import map_record_with_adapter

    field_mapping = {
        _FIELD_MAP_TO_CANONICAL[key]: value
        for key, value in (field_map or {}).items()
        if key in _FIELD_MAP_TO_CANONICAL and isinstance(value, str) and value
    }
    mapped = 0
    for row in rows:
        try:
            if map_record_with_adapter(
                row, adapter_id=adapter_id, field_mapping=field_mapping or None
            ):
                mapped += 1
        except Exception:  # noqa: BLE001
            continue
    return mapped / len(rows)


def _candidate(profile: str, **fields: Any) -> dict[str, Any]:
    spec = TASK_SHAPES[profile]
    return {
        "task_profile": profile,
        "label": spec["label"],
        "description": spec["description"],
        "adapter_id": spec["adapter_id"],
        "recipe_id": spec["recipe_id"],
        "mapper_id": fields.pop("mapper_id", None) or spec["mapper_id"],
        "field_map": fields.pop("field_map", {}) or {},
        **fields,
    }


def detect_task_shape(
    rows: list[Any],
    *,
    intent: str | None = None,
    max_rows: int = 200,
) -> dict[str, Any]:
    """Rank task shapes for these rows. Pure (no DB)."""
    from app.services.dataset_import.introspector import detect_shape, sniff_columns

    sample = [row for row in rows if isinstance(row, dict)][:max_rows]
    intent_matches = _intent_profiles(intent)
    by_profile: dict[str, dict[str, Any]] = {}

    if sample:
        signatures = sniff_columns(sample[:50])
        for hypothesis in detect_shape(signatures, sample[:50]):
            profile = canonical_task_profile(hypothesis.target_task_profile)
            if profile is None:
                continue
            existing = by_profile.get(profile)
            if existing is not None and existing["evidence"] >= hypothesis.confidence:
                continue
            by_profile[profile] = {
                "evidence": float(hypothesis.confidence),
                "mapper_id": hypothesis.mapper_id,
                "field_map": dict(hypothesis.field_map),
                "rationale": [hypothesis.rationale],
            }

    candidates: list[dict[str, Any]] = []
    for profile, found in by_profile.items():
        adapter_id = TASK_SHAPES[profile]["adapter_id"]
        rate = _map_rate(sample, adapter_id, found["field_map"])
        confidence = found["evidence"] * (0.75 + 0.25 * rate)
        rationale = list(found["rationale"])
        if adapter_id != "default-canonical":
            rationale.append(f"{round(rate * 100)}% of rows fit the {TASK_SHAPES[profile]['label'].lower()} format")
        candidates.append(_candidate(
            profile,
            mapper_id=found["mapper_id"],
            field_map=found["field_map"],
            confidence=confidence,
            map_rate=round(rate, 3),
            rationale=rationale,
            source="columns",
        ))

    # No column rule fired: fall back to "which adapter can map these
    # rows?". Adapters are permissive (qa-pair maps most text+answer rows),
    # so this is weak evidence — capped below the confirm threshold and
    # never allowed to outrank a column rule.
    for profile, adapter_id in (_DISCRIMINATING_ADAPTERS.items() if not by_profile else ()):
        rate = _map_rate(sample, adapter_id, None)
        if rate < 0.9:
            continue
        candidates.append(_candidate(
            profile,
            confidence=0.45 + 0.2 * rate,
            map_rate=round(rate, 3),
            rationale=[f"{round(rate * 100)}% of rows fit the {TASK_SHAPES[profile]['label'].lower()} format"],
            source="row_fit",
        ))

    if not candidates:
        text_like = any(isinstance(v, str) and len(v) > 40 for row in sample[:20] for v in row.values())
        fallback = "language_modeling" if text_like else "instruction_sft"
        candidates.append(_candidate(
            fallback,
            confidence=0.3,
            map_rate=None,
            rationale=["no answer/label column recognised — treating the rows as plain text"],
            source="fallback",
        ))

    for candidate in candidates:
        if candidate["task_profile"] in intent_matches:
            candidate["confidence"] = min(0.97, candidate["confidence"] + _INTENT_BONUS)
            candidate["rationale"].append("matches the goal you described")
        candidate["confidence"] = round(candidate["confidence"], 3)

    candidates.sort(key=lambda c: -c["confidence"])
    top = candidates[0]
    runner_up = candidates[1]["confidence"] if len(candidates) > 1 else 0.0
    needs_confirmation = top["confidence"] < CONFIRM_THRESHOLD or (
        top["confidence"] - runner_up < AMBIGUITY_MARGIN
    )
    return {
        "top": top,
        "candidates": candidates,
        "needs_confirmation": needs_confirmation,
        "confirm_threshold": CONFIRM_THRESHOLD,
        "rows_examined": len(sample),
    }


async def confirm_task_shape(
    db: AsyncSession,
    project_id: int,
    task_profile: str,
    *,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Persist the confirmed shape where dataset prep, training and eval
    read it (``dataset_adapter_preset``) and snapshot the matching recipe
    onto ``selected_recipe``. Keeps an existing base model."""
    from app.models.project import Project
    from app.services import recipe_service
    from app.services.dataset_service import save_project_dataset_adapter_preference
    from app.services.recipe_apply_service import build_recipe_snapshot

    profile = canonical_task_profile(task_profile)
    if profile is None:
        allowed = ", ".join(sorted(TASK_SHAPES))
        raise ValueError(f"Unknown task shape '{task_profile}'. Pick one of: {allowed}.")
    spec = TASK_SHAPES[profile]

    preference = await save_project_dataset_adapter_preference(
        db,
        project_id,
        adapter_id=spec["adapter_id"],
        adapter_config=adapter_config,
        field_mapping=field_mapping,
        task_profile=profile,
    )
    project = await db.get(Project, project_id)
    if project is not None and isinstance(project.dataset_adapter_preset, dict):
        project.dataset_adapter_preset = {**project.dataset_adapter_preset, "origin": "task_shape"}
        preference = {**preference, "origin": "task_shape"}
    recipe = recipe_service.get_recipe(spec["recipe_id"])
    if project is not None and recipe is not None:
        snapshot = build_recipe_snapshot(recipe)
        snapshot["task_profile"] = profile
        snapshot["adapter_id"] = spec["adapter_id"]
        snapshot["confirmed_via"] = "task_shape"
        snapshot["confirmed_at"] = datetime.now(timezone.utc).isoformat()
        project.selected_recipe = snapshot
        if not (project.base_model_name or "").strip():
            project.base_model_name = recipe.suggested_base_model
    await db.flush()
    return {
        "task_profile": profile,
        "label": spec["label"],
        "adapter_id": spec["adapter_id"],
        "recipe_id": spec["recipe_id"],
        "dataset_adapter_preference": preference,
    }


def adapter_for_task_profile(task_profile: str | None) -> str | None:
    profile = canonical_task_profile(task_profile)
    return TASK_SHAPES[profile]["adapter_id"] if profile else None


async def sample_project_rows(db: AsyncSession, project_id: int, limit: int = 200) -> list[dict[str, Any]]:
    """Up to ``limit`` rows of the data this project would train on:
    imported/synthetic rows, then cleaned rows, then accepted raw row files."""
    from pathlib import Path

    from sqlalchemy import select

    from app.config import settings
    from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
    from app.services.dataset_service import _load_records_from_file

    project_dir = Path(settings.DATA_DIR) / "projects" / str(project_id)
    rows: list[dict[str, Any]] = []
    for path in (project_dir / "synthetic" / "synthetic.jsonl", project_dir / "cleaned" / "cleaned.jsonl"):
        if len(rows) >= limit:
            break
        rows.extend(_load_records_from_file(path, max_records=limit - len(rows)))
    if len(rows) < limit:
        docs = (
            await db.execute(
                select(RawDocument)
                .join(Dataset, Dataset.id == RawDocument.dataset_id)
                .where(Dataset.project_id == project_id)
                .where(Dataset.dataset_type == DatasetType.RAW)
                .where(RawDocument.status == DocumentStatus.ACCEPTED)
            )
        ).scalars().all()
        for doc in docs:
            if len(rows) >= limit:
                break
            rows.extend(_load_records_from_file(Path(doc.file_path), max_records=limit - len(rows)))
    # Drop bookkeeping so the detector sees the user's columns.
    bookkeeping = {"id", "source_document_id", "source_doc", "chunk_id", "row_index", "status",
                   "source", "import_locator", "import_mapper", "row_key", "imported_at", "review_status"}
    return [
        {k: v for k, v in row.items() if k not in bookkeeping and not str(k).startswith("_")}
        for row in rows
        if isinstance(row, dict)
    ][:limit]


async def project_data_task_profile(db: AsyncSession, project_id: int) -> tuple[str | None, str | None]:
    """The task shape the project's DATA says it is: the confirmed shape
    when there is one, else a confident detection over the current rows,
    else ``(None, None)``. Returns ``(task_profile, source)`` with source
    ``"confirmed"`` / ``"detected"``. Stated goals (autopilot intent,
    project brief) are only priors — data wins when it speaks clearly."""
    from app.models.project import Project

    project = await db.get(Project, project_id)
    if project is None:
        return None, None
    preset = project.dataset_adapter_preset if isinstance(project.dataset_adapter_preset, dict) else {}
    confirmed = canonical_task_profile(preset.get("task_profile"))
    if confirmed:
        return confirmed, "confirmed"
    rows = await sample_project_rows(db, project_id)
    if not rows:
        return None, None
    detection = detect_task_shape(rows)
    if detection["needs_confirmation"]:
        return None, None
    return detection["top"]["task_profile"], "detected"
