"""Demo project seeder (newbie UX Phase 3).

Materialises a real project from a pre-curated bundle of sample data,
a hand-labelled gold set, and project metadata. The goal: cut
time-to-first-result for a new ML engineer from "find a dataset, label
it, configure the pipeline" to one click.

Bundles live under ``backend/data/demo_samples/<slug>/`` and ship
three files:

- ``manifest.json`` — slug, display name, headline, description,
  task profile, target profile, suggested autopilot brief, input /
  output field names.
- ``<dataset_filename>`` — CSV of source rows (e.g. ``tickets.csv``).
- ``gold.jsonl`` — JSONL with ``{key, input, expected, rationale}``
  rows for the locked gold set.

The seeder is idempotent: re-running with the same slug returns the
existing project if its name is already taken. That keeps the "Try a
demo" tile click safe even on a misbehaving frontend.

Stable reason codes (raised as ``ValueError`` from the API path):

- ``demo_slug_unknown:<slug>`` (404) — no bundle dir for that slug.
- ``demo_manifest_invalid:<slug>`` (400) — manifest.json missing or
  malformed.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import (
    Dataset,
    DatasetType,
    DatasetVersion,
    DocumentStatus,
    RawDocument,
)
from app.models.gold_set_annotation import (
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)
from app.models.project import PipelineStage, Project, ProjectStatus
from app.services import gold_workbench_service


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DEMO_SAMPLES_DIR = BACKEND_DIR / "data" / "demo_samples"


def _resolve_demo_dir(slug: str) -> Path:
    """Resolve a demo slug to its sample-data directory, with safety check."""

    safe = "".join(ch for ch in slug if ch.isalnum() or ch in "-_").strip("-_")
    if not safe or safe != slug:
        raise ValueError(f"demo_slug_unknown:{slug}")
    candidate = (DEMO_SAMPLES_DIR / safe).resolve()
    if not candidate.is_dir() or DEMO_SAMPLES_DIR.resolve() not in candidate.parents:
        raise ValueError(f"demo_slug_unknown:{slug}")
    return candidate


def _load_manifest(slug: str) -> dict[str, Any]:
    demo_dir = _resolve_demo_dir(slug)
    manifest_path = demo_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"demo_manifest_invalid:{slug}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"demo_manifest_invalid:{slug}:{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"demo_manifest_invalid:{slug}:not_a_dict")
    payload["_dir"] = str(demo_dir)
    return payload


def list_demo_archetypes() -> list[dict[str, Any]]:
    """Catalog of available demos, read fresh from disk on every call."""

    if not DEMO_SAMPLES_DIR.is_dir():
        return []
    archetypes: list[dict[str, Any]] = []
    for child in sorted(DEMO_SAMPLES_DIR.iterdir()):
        if not child.is_dir():
            continue
        try:
            manifest = _load_manifest(child.name)
        except ValueError:
            continue
        archetypes.append(
            {
                "slug": manifest.get("slug") or child.name,
                "name": manifest.get("name") or child.name.title(),
                "headline": manifest.get("headline") or "",
                "description": manifest.get("description") or "",
                "task_profile": manifest.get("task_profile") or "",
                "target_profile": manifest.get("target_profile") or "",
                "suggested_brief": manifest.get("suggested_brief") or "",
            }
        )
    return archetypes


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _read_gold_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


async def _find_project_by_name(db: AsyncSession, name: str) -> Project | None:
    result = await db.execute(select(Project).where(Project.name == name))
    return result.scalar_one_or_none()


async def _next_gold_version(db: AsyncSession, gold_set_id: int) -> int:
    row = await db.execute(
        select(func.max(GoldSetVersion.version)).where(GoldSetVersion.gold_set_id == gold_set_id)
    )
    current = row.scalar_one_or_none()
    return (current or 0) + 1


def _adapter_for_task(task_profile: str) -> str:
    task = str(task_profile or "").strip().lower()
    if task == "classification":
        return "classification-label"
    if task in ("structured_extraction", "extraction"):
        return "structured-extraction"
    return "qa-pair"


def _canonical_prepared_row(
    row: dict[str, Any],
    *,
    input_field: str,
    output_field: str,
    task_profile: str,
) -> dict[str, Any]:
    """Materialise a CSV row in the canonical prepared shape.

    Always sets text/source_text/target_text so seq2seq + causal_lm contracts
    pass; adds {question,answer} for QA-style tasks and {label} for
    classification. Keeps the original fields too so the row stays
    self-describing.
    """

    input_val = str(row.get(input_field) or "").strip()
    output_val = str(row.get(output_field) or "").strip()
    canonical: dict[str, Any] = dict(row)
    canonical["text"] = input_val
    canonical["source_text"] = input_val
    canonical["target_text"] = output_val
    if str(task_profile or "").strip().lower() == "classification":
        canonical["label"] = output_val
    else:
        canonical.setdefault("question", input_val)
        canonical.setdefault("answer", output_val)
    return canonical


def _split_rows(rows: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Deterministic 70/15/15 train/val/test split.

    No shuffle — order in the bundle is curated so the first rows are the
    most representative examples (helps when a beginner inspects the
    train split). Guarantees at least one row in val + test when the
    source has ≥ 3 rows so the val/test files aren't empty.
    """

    total = len(rows)
    if total == 0:
        return [], [], []
    if total < 3:
        return list(rows), [], []
    n_test = max(1, total // 7)
    n_val = max(1, total // 7)
    n_train = total - n_val - n_test
    if n_train <= 0:
        n_train = 1
        n_val = max(1, (total - n_train) // 2)
        n_test = total - n_train - n_val
    return (
        list(rows[:n_train]),
        list(rows[n_train : n_train + n_val]),
        list(rows[n_train + n_val :]),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _materialize_demo_bundle_into_project(
    db: AsyncSession,
    project: Project,
    slug: str,
    manifest: dict[str, Any],
    demo_dir: Path,
    *,
    project_name: str,
    actor_user_id: int | None = None,
) -> dict[str, Any]:
    """Write a demo bundle's data (raw + gold + prepared splits) into
    an already-created Project. Returns the summary dict that callers
    surface to the API client. The caller is responsible for adding
    the Project to the session, flushing once project.id is needed,
    and committing the transaction.
    """

    task_profile = str(manifest.get("task_profile") or "instruction_sft")
    adapter_id = _adapter_for_task(task_profile)
    input_field = str(manifest.get("dataset_input_field") or "input")
    output_field = str(manifest.get("dataset_output_field") or "output")
    suggested_brief = str(manifest.get("suggested_brief") or "")

    # 1) Raw source dataset ----------------------------------------------------
    # NOTE: dataset_type is RAW (not CLEANED) so the Pipeline → Data tab
    # picks the rows up — that tab filters on Dataset.dataset_type == RAW
    # via app/services/ingestion_service.list_documents. The Cleaning tab
    # uses the same endpoint.
    csv_filename = str(manifest.get("dataset_filename") or "data.csv")
    csv_src = demo_dir / csv_filename
    if not csv_src.exists():
        raise ValueError(f"demo_manifest_invalid:{slug}:missing_dataset_file")

    raw_dir = settings.DATA_DIR / "projects" / str(project.id) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dst = raw_dir / csv_filename
    shutil.copy(csv_src, csv_dst)

    csv_rows = _read_csv_rows(csv_src)
    dataset = Dataset(
        project_id=project.id,
        name=f"{project_name} · source",
        dataset_type=DatasetType.RAW,
        description="Pre-loaded source rows for the demo project.",
        record_count=len(csv_rows),
        file_path=str(csv_dst),
        metadata_={
            "demo_slug": slug,
            "input_field": manifest.get("dataset_input_field"),
            "output_field": manifest.get("dataset_output_field"),
        },
        is_locked=True,
    )
    db.add(dataset)
    await db.flush()

    # One RawDocument per CSV row keeps the per-stage tabs realistic.
    for idx, row in enumerate(csv_rows):
        payload = json.dumps(row, ensure_ascii=False)
        db.add(
            RawDocument(
                dataset_id=dataset.id,
                filename=f"{csv_filename}#row-{idx + 1}",
                file_type="csv-row",
                file_path=str(csv_dst),
                file_size_bytes=len(payload.encode("utf-8")),
                source="demo",
                sensitivity="public",
                status=DocumentStatus.ACCEPTED,
                metadata_={"row_index": idx, "row": row},
            )
        )

    # 2) Locked gold set -------------------------------------------------------
    # The legacy gold UI (GoldSetPanel) reads from the dataset's
    # ``file_path`` JSONL on disk and expects each line to carry
    # ``{id, question, answer, ...}``. We materialise both the legacy
    # JSONL AND the GoldSetVersion / GoldSetRow workbench rows below —
    # the older "Gold set" tab consumes the JSONL while the newer
    # workbench (Pipeline → Gold workbench) consumes the rows.
    gold_filename = str(manifest.get("gold_filename") or "gold.jsonl")
    gold_src = demo_dir / gold_filename
    if not gold_src.exists():
        raise ValueError(f"demo_manifest_invalid:{slug}:missing_gold_file")
    gold_rows = _read_gold_rows(gold_src)

    gold_dir = settings.DATA_DIR / "projects" / str(project.id) / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_jsonl_path = gold_dir / "gold_dev.jsonl"

    now_iso = datetime.now(timezone.utc).isoformat()
    legacy_entries: list[dict[str, Any]] = []
    for idx, row in enumerate(gold_rows):
        inp = row.get("input") or {}
        exp = row.get("expected") or {}
        # Map the bundle's input / expected dicts onto the legacy
        # question / answer fields the UI hardcodes. For sentiment we
        # use input.text / expected.label; for support-faq we use
        # input.question / expected.answer; for PII we use
        # input.text / expected.entities (a list, JSON-encoded below).
        question = (
            inp.get("question")
            or inp.get("text")
            or next(iter(inp.values()), "")
        )
        # Pick the most specific scalar field first (answer/label). For
        # structured-extraction gold rows whose `expected` is a typed
        # dict like `{"entities": [...]}`, preserve the whole dict so
        # the reference structure round-trips through to scoring.
        if exp.get("answer") is not None:
            answer_raw = exp["answer"]
        elif exp.get("label") is not None:
            answer_raw = exp["label"]
        elif isinstance(exp, dict) and exp:
            answer_raw = exp
        else:
            answer_raw = ""
        # JSON-encode complex answers (lists, dicts) so the gold panel
        # shows valid JSON instead of Python repr. Strings pass through.
        if isinstance(answer_raw, str):
            answer = answer_raw
        elif answer_raw is None:
            answer = ""
        else:
            answer = json.dumps(answer_raw, ensure_ascii=False)
        legacy_entries.append({
            "id": idx + 1,
            "question": str(question),
            "answer": answer,
            "difficulty": "medium",
            "criticality": "normal",
            "is_hallucination_trap": False,
            "metadata": {
                "demo_slug": slug,
                "bundle_key": row.get("key"),
                "rationale": str(row.get("rationale") or ""),
            },
            "created_at": now_iso,
        })

    with gold_jsonl_path.open("w", encoding="utf-8") as handle:
        for entry in legacy_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    gold_dataset = Dataset(
        project_id=project.id,
        name=f"{project_name} · gold",
        dataset_type=DatasetType.GOLD_DEV,
        description="Hand-labelled gold-set rows for the demo project.",
        record_count=len(gold_rows),
        file_path=str(gold_jsonl_path),
        metadata_={"demo_slug": slug, "frozen": True},
        is_locked=True,
    )
    db.add(gold_dataset)
    await db.flush()

    version_number = await _next_gold_version(db, gold_dataset.id)
    gold_version = GoldSetVersion(
        gold_set_id=gold_dataset.id,
        version=version_number,
        status=GoldSetVersionStatus.LOCKED,
        notes=f"Seeded from demo bundle '{slug}'.",
        created_by_user_id=actor_user_id,
    )
    db.add(gold_version)
    await db.flush()

    for row in gold_rows:
        db.add(
            GoldSetRow(
                gold_set_id=gold_dataset.id,
                version_id=gold_version.id,
                source_row_key=str(row.get("key") or "")[:128] or None,
                source_dataset_id=dataset.id,
                input=row.get("input") or {},
                expected=row.get("expected") or {},
                rationale=str(row.get("rationale") or ""),
                labels=row.get("labels") or {},
                status=GoldSetRowStatus.APPROVED,
                reviewer_id=actor_user_id,
            )
        )

    # 3) Prepared training splits ----------------------------------------------
    # Autopilot's readiness check (newbie_autopilot_service.evaluate_*) blocks
    # any training launch if ``prepared/train.jsonl`` is missing, and the
    # dataset-contract check (analyze_prepared_dataset_contract) requires
    # ≥90% rows match the task type's shape. We materialise canonical rows
    # so qa_pair / classification_label contracts both pass, plus a
    # manifest.json with adapter_id + task_profile + field_mapping so
    # downstream stages can resolve the right adapter without re-running prep.
    prepared_dir = settings.DATA_DIR / "projects" / str(project.id) / "prepared"
    canonical_rows = [
        _canonical_prepared_row(
            row,
            input_field=input_field,
            output_field=output_field,
            task_profile=task_profile,
        )
        for row in csv_rows
    ]
    train_rows, val_rows, test_rows = _split_rows(canonical_rows)
    train_path = prepared_dir / "train.jsonl"
    val_path = prepared_dir / "val.jsonl"
    test_path = prepared_dir / "test.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_jsonl(test_path, test_rows)

    prepared_manifest: dict[str, Any] = {
        "project_id": project.id,
        "created_at": now_iso,
        "seed": 42,
        "total_entries": len(csv_rows),
        "splits": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "file_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "adapter_id": adapter_id,
        "adapter_config": {},
        "field_mapping": {"input": input_field, "output": output_field},
        "task_profile": task_profile,
        "demo_slug": slug,
    }
    # Phase 5.3.1: forward the candidate label set into the prepared
    # manifest so the ClassificationHandler can read it from a single
    # canonical source instead of scanning the dataset on every eval.
    bundle_labels = manifest.get("labels")
    if isinstance(bundle_labels, list) and bundle_labels:
        prepared_manifest["labels"] = [str(l).strip() for l in bundle_labels if str(l).strip()]
    # Phase 5.3.4: same idea for structured-extraction demos —
    # StructuredExtractionHandler reads `output_schema` from the
    # prepared manifest to drive per-field metrics + the prompt
    # template's field list. Forward `entity_types` too as a
    # diagnostic hint for the UI / docs.
    bundle_schema = manifest.get("output_schema")
    if isinstance(bundle_schema, dict) and bundle_schema:
        prepared_manifest["output_schema"] = bundle_schema
    bundle_entity_types = manifest.get("entity_types")
    if isinstance(bundle_entity_types, list) and bundle_entity_types:
        prepared_manifest["entity_types"] = [
            str(t).strip() for t in bundle_entity_types if str(t).strip()
        ]
    manifest_path = prepared_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(prepared_manifest, indent=2), encoding="utf-8"
    )

    prepared_specs = (
        ("train", DatasetType.TRAIN, train_rows, train_path),
        ("val", DatasetType.VALIDATION, val_rows, val_path),
        ("test", DatasetType.TEST, test_rows, test_path),
    )
    prepared_dataset_ids: dict[str, int] = {}
    for split_name, ds_type, rows, file_path in prepared_specs:
        prep_ds = Dataset(
            project_id=project.id,
            name=f"{project_name} · {split_name}",
            dataset_type=ds_type,
            description=f"Prepared {split_name} split for the demo project.",
            record_count=len(rows),
            file_path=str(file_path),
            metadata_={
                "demo_slug": slug,
                "split": split_name,
                "adapter_id": adapter_id,
                "task_profile": task_profile,
            },
            is_locked=True,
        )
        db.add(prep_ds)
        await db.flush()
        prepared_dataset_ids[split_name] = prep_ds.id
        db.add(
            DatasetVersion(
                dataset_id=prep_ds.id,
                version=1,
                file_path=str(file_path),
                record_count=len(rows),
                manifest={
                    "split": split_name,
                    "seed": 42,
                    "count": len(rows),
                    "adapter_id": adapter_id,
                    "task_profile": task_profile,
                },
            )
        )

    await db.flush()

    summary = {
        "slug": slug,
        "created": True,
        "project_id": project.id,
        "project_name": project.name,
        "source_dataset_id": dataset.id,
        "source_row_count": len(csv_rows),
        "gold_set_id": gold_dataset.id,
        "gold_version_id": gold_version.id,
        "gold_row_count": len(gold_rows),
        "prepared_train_path": str(train_path),
        "prepared_train_rows": len(train_rows),
        "prepared_val_rows": len(val_rows),
        "prepared_test_rows": len(test_rows),
        "prepared_dataset_ids": prepared_dataset_ids,
        "adapter_id": adapter_id,
        "task_profile": task_profile,
        "suggested_brief": suggested_brief,
    }
    return summary


async def seed_demo_project(
    db: AsyncSession,
    slug: str,
    *,
    actor_user_id: int | None = None,
) -> tuple[Project, dict[str, Any]]:
    """Create (or return) the demo project for ``slug``. Idempotent."""

    manifest = _load_manifest(slug)
    demo_dir = Path(manifest["_dir"])

    project_name = str(manifest.get("name") or slug)
    existing = await _find_project_by_name(db, project_name)
    if existing is not None:
        return existing, {"slug": slug, "created": False, "project_id": existing.id}

    description = str(manifest.get("description") or "")
    suggested_brief = str(manifest.get("suggested_brief") or "")
    target_profile = str(manifest.get("target_profile") or "vllm_server")
    plan_profile = str(manifest.get("training_preferred_plan_profile") or "balanced")
    eval_pack = manifest.get("evaluation_preferred_pack_id") or None

    task_profile = str(manifest.get("task_profile") or "instruction_sft")
    adapter_id = _adapter_for_task(task_profile)
    input_field = str(manifest.get("dataset_input_field") or "input")
    output_field = str(manifest.get("dataset_output_field") or "output")

    project = Project(
        name=project_name,
        description=description,
        status=ProjectStatus.ACTIVE,
        pipeline_stage=PipelineStage.TRAINING,
        beginner_mode=True,
        target_profile_id=target_profile,
        training_preferred_plan_profile=plan_profile,
        evaluation_preferred_pack_id=eval_pack,
        dataset_adapter_preset={
            "demo_slug": slug,
            "suggested_brief": suggested_brief,
            "adapter_id": adapter_id,
            "task_profile": task_profile,
            "field_mapping": {"input": input_field, "output": output_field},
        },
    )
    db.add(project)
    await db.flush()  # populate project.id

    summary = await _materialize_demo_bundle_into_project(
        db,
        project,
        slug,
        manifest,
        demo_dir,
        project_name=project_name,
        actor_user_id=actor_user_id,
    )
    # Newly-seeded demo project — always assign the recipe from the
    # slug map (force=True). Without this, the project ships with
    # selected_recipe=None and every "Pick a recipe before X" surface
    # (Coach, Synthetic playbooks, gold-set review) shows a no-op
    # state until the user manually picks one. Best-effort; broken
    # mapping records the skip but doesn't fail the seed.
    summary["recipe_assignment"] = await _assign_recipe_from_slug_if_missing(
        db, project, slug, force=True,
    )
    return project, summary


async def reset_demo_project(
    db: AsyncSession,
    slug: str,
    *,
    actor_user_id: int | None = None,
) -> tuple[Project, dict[str, Any]]:
    """Delete the existing demo project for ``slug`` (if any) and re-seed a
    fresh copy — the sample-gallery reset lifecycle (Epic G phase G2). Lets
    a first-timer who broke a sample start over clean. ``seed_demo_project``
    is idempotent (returns the existing project), so a reset must drop the
    old one first. Raises ``ValueError`` on an unknown slug (via the
    manifest load), which the API maps to 404."""
    manifest = _load_manifest(slug)
    project_name = str(manifest.get("name") or slug)
    existing = await _find_project_by_name(db, project_name)
    if existing is not None:
        # ORM cascade removes the project's datasets / experiments, but NOT the
        # gold-set workbench tables (no relationship; ambiguous multi-FK). Purge
        # them explicitly first or they leak and, via SQLite rowid reuse, pollute
        # the next project that reuses a freed dataset id.
        await gold_workbench_service.purge_gold_sets_for_project(db, existing.id)
        await db.delete(existing)
        await db.flush()
    project, summary = await seed_demo_project(
        db, slug, actor_user_id=actor_user_id
    )
    summary["reset"] = True
    return project, summary


# Recipe id → preferred demo bundle slug. Used by the project-guide
# quickstart "Import sample CSV" button to pick a bundle that matches
# the task shape the user's already locked in via the recipe picker.
# When the user hasn't picked a recipe, fall back to support-faq —
# the most domain-generic Q&A bundle.
RECIPE_TO_DEMO_SLUG: dict[str, str] = {
    "qa-sft": "support-faq",
    "classification": "sentiment-classifier",
    "span-extraction": "pii-detector",
    "summarization": "support-faq",
    "code-review": "support-faq",
    "generic-sft": "support-faq",
}


# Demo slug → canonical recipe id. Inverse of the recipe→slug map for
# the three slugs that actually have bundles on disk under
# ``data/demo_samples/``. Used by ``seed_demo_project`` +
# ``apply_demo_bundle_to_project`` to assign a recipe at materialization
# time so demo projects come out of the box ready for the Coach,
# Synthetic, Eval, etc. surfaces that gate on recipe selection.
# (Pre-this-fix, demos shipped without a recipe → multiple surfaces
# fell through to "Pick a recipe before X" no-op states.)
DEMO_SLUG_TO_RECIPE_ID: dict[str, str] = {
    "support-faq": "qa-sft",
    "sentiment-classifier": "classification",
    "pii-detector": "span-extraction",
}


def derive_recipe_id_for_slug(slug: str) -> str | None:
    """Return the canonical recipe id for a demo slug, or ``None`` if
    the slug has no known recipe mapping (extensible — slugs without
    a mapped recipe still seed; the project just doesn't get auto-
    assigned a recipe)."""
    return DEMO_SLUG_TO_RECIPE_ID.get(slug)


def derive_demo_slug_for_project(project: Project) -> str:
    """Pick the best demo bundle slug for a project. Reads
    `project.selected_recipe.recipe_id` if present; otherwise
    defaults to `support-faq`."""
    snapshot = project.selected_recipe or {}
    recipe_id = str(snapshot.get("recipe_id") or "").strip()
    return RECIPE_TO_DEMO_SLUG.get(recipe_id, "support-faq")


async def _assign_recipe_from_slug_if_missing(
    db: AsyncSession,
    project: Project,
    slug: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Snapshot the recipe matching ``slug`` onto ``project`` via
    ``recipe_apply_service.apply_recipe_to_project``. Returns an
    observability dict {assigned, reason, recipe_id} so callers can
    surface what happened.

    No-op (no DB write) when:
      - ``slug`` has no entry in DEMO_SLUG_TO_RECIPE_ID
      - ``project.selected_recipe`` is already populated AND ``force``
        is False (don't clobber a user's explicit choice on the
        existing-project ``apply_demo_bundle_to_project`` path)
    """
    recipe_id = derive_recipe_id_for_slug(slug)
    if recipe_id is None:
        return {"assigned": False, "reason": f"slug_has_no_recipe:{slug}"}
    if not force and (project.selected_recipe or {}).get("recipe_id"):
        return {
            "assigned": False,
            "reason": "project_already_has_recipe",
            "existing_recipe_id": (project.selected_recipe or {}).get("recipe_id"),
        }
    # Lazy import to avoid touching the recipe_apply_service import
    # surface unless we actually assign — keeps the demo seeder's
    # module-load light for code paths that don't materialize.
    from app.services.recipe_apply_service import (
        RecipeNotFoundError,
        apply_recipe_to_project,
    )

    try:
        await apply_recipe_to_project(db, project.id, recipe_id)
    except RecipeNotFoundError as e:
        # Catalog drift — the slug map says X but recipe X isn't
        # registered. Surface as a recorded skip rather than failing
        # the seed; the seeded project will lack a recipe and the
        # user can pick one via the recipe picker.
        return {
            "assigned": False,
            "reason": f"recipe_not_in_catalog:{recipe_id}:{e}",
        }
    return {"assigned": True, "recipe_id": recipe_id}


async def apply_demo_bundle_to_project(
    db: AsyncSession,
    project_id: int,
    slug: str | None = None,
    *,
    actor_user_id: int | None = None,
) -> dict[str, Any]:
    """Materialize a demo bundle into an existing project (no new
    Project is created). Used by the project-guide "Import sample CSV"
    quickstart action.

    If `slug` is None, the bundle is derived from the project's
    `selected_recipe` (Theme 2). The project's pipeline_stage is
    advanced to TRAINING on success so downstream steps (training,
    eval) unlock immediately.

    Raises:
        ValueError("project_not_found:{project_id}") if the project doesn't exist.
        ValueError("demo_slug_unknown:{slug}") for unknown slugs.
        ValueError("demo_manifest_invalid:{slug}") for malformed bundles.
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError(f"project_not_found:{project_id}")

    resolved_slug = (slug or "").strip() or derive_demo_slug_for_project(project)
    manifest = _load_manifest(resolved_slug)
    demo_dir = Path(manifest["_dir"])

    project_name = project.name
    summary = await _materialize_demo_bundle_into_project(
        db,
        project,
        resolved_slug,
        manifest,
        demo_dir,
        project_name=project_name,
        actor_user_id=actor_user_id,
    )
    # Existing project — only assign the recipe when the project
    # doesn't already have one set, so a user who picked a recipe
    # before clicking "Import sample CSV" keeps their choice.
    summary["recipe_assignment"] = await _assign_recipe_from_slug_if_missing(
        db, project, resolved_slug, force=False,
    )
    # Data is materialized + prepared splits exist: the project is
    # now ready to train. Advance the pipeline stage past INGESTION
    # so the guide-page checklist flips its "Ingest source data"
    # checkmark on the next refresh.
    project.pipeline_stage = PipelineStage.TRAINING
    await db.flush()
    return summary


__all__ = [
    "list_demo_archetypes",
    "seed_demo_project",
    "apply_demo_bundle_to_project",
    "derive_demo_slug_for_project",
    "derive_recipe_id_for_slug",
    "RECIPE_TO_DEMO_SLUG",
    "DEMO_SLUG_TO_RECIPE_ID",
]
