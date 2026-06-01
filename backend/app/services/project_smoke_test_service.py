"""Project smoke-test service (Diagnostics Intervention C).

Runs N independent read-only checks in parallel against a project so
a user can answer "is anything obviously broken on my project?"
without committing to a real run.

Each check captures any exception it encounters and shapes it into
the same envelope the rest of the platform uses, so the frontend's
shared ``<ErrorPanel>`` can render every failure consistently.

Design rules:

  * Read-only. No writes. No mutations. No GPU. No external network
    calls except cheap reachability probes (Ollama /api/tags at 1s
    timeout). Re-running the smoke test on a project is safe.
  * Each check has a short stable ``name`` (becomes the test-id +
    log key) + a one-sentence success message + a remediation
    string for failures.
  * Checks run in parallel via asyncio.gather. The slowest single
    check dominates wall-clock — typically <3s total.
  * Per-check failures NEVER bring down the whole smoke test —
    every check returns a SmokeCheckResult, even on exception.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment
from app.models.project import Project


SmokeStatus = Literal["ok", "warn", "fail", "skip"]


@dataclass
class SmokeCheckResult:
    """One check's outcome. Serialized for the API response.

    Fields are chosen so the frontend can render this in a checklist
    AND drop the ``envelope`` into the shared <ErrorPanel> when the
    status is ``fail``.
    """
    name: str
    status: SmokeStatus
    elapsed_ms: int
    message: str
    remediation: str | None = None
    # ErrorEnvelope-shaped dict (mirrors the platform-wide error
    # contract) — only populated for fail status. None for ok/warn/skip.
    envelope: dict[str, Any] | None = None
    # Optional extra context (counts, sample row, etc.) the panel can
    # show under a "Technical details" expander.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SmokeTestSummary:
    """Aggregate of all checks. ``overall`` rolls up to the
    worst-severity check: any ``fail`` → fail; otherwise any
    ``warn`` → warn; otherwise ok."""
    project_id: int
    overall: SmokeStatus
    elapsed_ms: int
    counts: dict[str, int]
    checks: list[SmokeCheckResult]


def _new_trace_id() -> str:
    return f"err_{secrets.token_urlsafe(9)}"


def _envelope_from_exception(
    *,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    """Build an ErrorEnvelope dict from a raised exception. Mirrors the
    shape ``_structured_error_payload`` in main.py produces so the
    frontend can render this through the same ``<ErrorPanel>`` it uses
    for HTTP error responses."""
    exc_name = type(exc).__name__
    return {
        "error_code": f"SMOKE_{stage.upper().replace('-', '_')}_{exc_name.upper()}",
        "stage": stage,
        "message": f"{exc_name}: {str(exc) or '(no message)'}",
        "actionable_fix": (
            "Open the matching surface (see stage) and run the action "
            "manually to see the full error. Copy this troubleshooting "
            "id and grep the server log for it to find the traceback."
        ),
        "docs_url": "/docs/troubleshooting",
        "troubleshooting_id": _new_trace_id(),
        "metadata": {
            "exception_type": exc_name,
            "check_stage": stage,
        },
        "detail": f"{exc_name}: {str(exc) or '(no message)'}",
    }


def _peek_prepared_row_prefix_match(
    *,
    project_id: int,
    manifest: dict[str, Any],
    experiment_task_type: str | None,
    recipe_task_profile: str | None,
) -> dict[str, Any] | None:
    """Open the first prepared row and check whether the eval handler's
    expected prompt prefix appears in any string field.

    Returns ``None`` when the check can't run (no handler, no prefixes
    declared, no readable train file). Otherwise returns
    ``{status: 'match' | 'mismatch', handler_id, expected_prefixes,
    peeked_row_path}`` so the caller can phrase the SmokeCheckResult
    appropriately.

    This is the load-bearing γ′ fix: matching adapter ids was a
    necessary but insufficient signal. The peeked row catches the
    SQLi-detector case where the adapter writes raw ``text → label``
    rows even though it's tagged as task=classification.
    """
    import json as _json
    from app.services.eval_task_handler_service import (
        _project_prepared_dir,
        resolve_task_handler,
    )

    # Resolve the handler that would run at eval time. Mirrors the
    # logic in build_eval_context — manifest first, fall back to the
    # experiment's task_type, finally the recipe's task_profile.
    task_profile = manifest.get("task_profile") if isinstance(manifest, dict) else None
    if not task_profile and experiment_task_type:
        task_profile = experiment_task_type
    if not task_profile and recipe_task_profile:
        task_profile = recipe_task_profile
    if not task_profile:
        return None
    handler = resolve_task_handler(task_profile)
    prefixes = list(
        getattr(handler, "expected_prompt_prefixes", lambda: [])()
    )
    if not prefixes:
        # Generic / QA / Safety / Alignment don't wrap with a fixed
        # prefix; the check doesn't apply.
        return None

    # Find a prepared file to peek at. Train > val > test for
    # representativeness (train is always the largest split).
    prepared_dir = _project_prepared_dir(project_id)
    candidates = [prepared_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]
    peeked_row: dict[str, Any] | None = None
    peeked_path: str = ""
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        peeked_row = _json.loads(raw)
                    except _json.JSONDecodeError:
                        continue
                    peeked_path = str(path)
                    break
        except OSError:
            continue
        if peeked_row is not None:
            break
    if peeked_row is None:
        return None

    # Concatenate every string-valued field — the prefix might live in
    # any of source_text / prompt / text / input / formatted_prompt etc.
    # depending on the adapter.
    concat = " | ".join(
        str(v) for v in peeked_row.values() if isinstance(v, str)
    )
    matched_prefix = next((p for p in prefixes if p in concat), None)
    return {
        "status": "match" if matched_prefix else "mismatch",
        "handler_id": handler.profile_id,
        "expected_prefixes": prefixes,
        "matched_prefix": matched_prefix,
        "peeked_row_path": peeked_path,
    }


# ─────────────────────────────────────────────────────────────────────
# Individual checks. Each is async, captures exceptions, returns a
# SmokeCheckResult. They're independent so asyncio.gather can run
# them in parallel.
# ─────────────────────────────────────────────────────────────────────


async def _check_project_exists(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        project = await db.get(Project, project_id)
        elapsed = int((time.monotonic() - started) * 1000)
        if project is None:
            return SmokeCheckResult(
                name="project_exists",
                status="fail",
                elapsed_ms=elapsed,
                message=f"Project {project_id} not found.",
                remediation="Verify the project URL — it may have been deleted.",
                envelope={
                    "error_code": "SMOKE_PROJECT_NOT_FOUND",
                    "stage": "project",
                    "message": f"Project {project_id} not found.",
                    "actionable_fix": "Verify the project URL — it may have been deleted.",
                    "docs_url": "/docs/troubleshooting",
                    "troubleshooting_id": _new_trace_id(),
                    "metadata": None,
                    "detail": f"Project {project_id} not found.",
                },
            )
        return SmokeCheckResult(
            name="project_exists",
            status="ok",
            elapsed_ms=elapsed,
            message=f"Project '{project.name}' is accessible.",
            metadata={"name": project.name, "status": str(project.status)},
        )
    except Exception as exc:  # noqa: BLE001 — checks must never re-raise
        return SmokeCheckResult(
            name="project_exists",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Couldn't query the projects table.",
            envelope=_envelope_from_exception(stage="project", exc=exc),
        )


async def _check_recipe_applied(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        project = await db.get(Project, project_id)
        elapsed = int((time.monotonic() - started) * 1000)
        if project is None:
            return SmokeCheckResult(
                name="recipe_applied",
                status="skip",
                elapsed_ms=elapsed,
                message="Project doesn't exist — recipe check skipped.",
            )
        selected = project.selected_recipe or {}
        recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
        if not recipe_id:
            return SmokeCheckResult(
                name="recipe_applied",
                status="fail",
                elapsed_ms=elapsed,
                message="No recipe selected on this project.",
                remediation=(
                    "Open Pipeline → Recipe picker and apply a recipe. "
                    "Many downstream flows (synth, training, eval) require "
                    "a recipe to be applied first."
                ),
                envelope={
                    "error_code": "SMOKE_RECIPE_MISSING",
                    "stage": "project",
                    "message": "No recipe selected on this project.",
                    "actionable_fix": "Open Pipeline → Recipe picker and apply a recipe.",
                    "docs_url": "/docs/troubleshooting",
                    "troubleshooting_id": _new_trace_id(),
                    "metadata": None,
                    "detail": "No recipe selected.",
                },
            )
        return SmokeCheckResult(
            name="recipe_applied",
            status="ok",
            elapsed_ms=elapsed,
            message=f"Recipe '{recipe_id}' is applied.",
            metadata={"recipe_id": recipe_id},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="recipe_applied",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Couldn't read the project's selected recipe.",
            envelope=_envelope_from_exception(stage="project", exc=exc),
        )


async def _check_gold_set(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        from app.services.trainability_forecast_service import _load_gold_rows
        rows = await _load_gold_rows(db, project_id)
        elapsed = int((time.monotonic() - started) * 1000)
        n = len(rows)
        if n == 0:
            return SmokeCheckResult(
                name="gold_set",
                status="warn",
                elapsed_ms=elapsed,
                message="Gold set is empty (0 rows).",
                remediation=(
                    "Open Pipeline → Gold set and seed at least 5-10 "
                    "labelled rows. Training + evaluation require some "
                    "gold rows to learn / measure against."
                ),
                metadata={"gold_row_count": 0},
            )
        return SmokeCheckResult(
            name="gold_set",
            status="ok",
            elapsed_ms=elapsed,
            message=f"Gold set has {n} row{'s' if n != 1 else ''}.",
            metadata={"gold_row_count": n},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="gold_set",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Couldn't load gold rows.",
            envelope=_envelope_from_exception(stage="gold", exc=exc),
        )


async def _check_data_health(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        from app.services.data_health_service import compute_data_health_report
        report = await compute_data_health_report(db, project_id)
        elapsed = int((time.monotonic() - started) * 1000)
        overall = report.get("overall") or "unknown"
        counts = report.get("severity_summary") or {}
        # Map data-health verdict → smoke status. ``block`` doesn't
        # mean smoke "fail" — the project still works, just has
        # data-quality issues to address. So this check is informational.
        smoke_status: SmokeStatus = (
            "warn" if overall in ("warn", "block") else "ok"
        )
        msg = (
            f"Data Health Report computed; overall={overall} "
            f"(ok={counts.get('ok', 0)} warn={counts.get('warn', 0)} "
            f"block={counts.get('block', 0)})"
        )
        return SmokeCheckResult(
            name="data_health",
            status=smoke_status,
            elapsed_ms=elapsed,
            message=msg,
            remediation=(
                "Open the Data Health panel for the per-signal breakdown + "
                "auto-fix affordances."
                if smoke_status == "warn" else None
            ),
            metadata={"overall": overall, "counts": dict(counts)},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="data_health",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Data Health Report failed to compute.",
            envelope=_envelope_from_exception(stage="data-health", exc=exc),
        )


async def _check_trainability_forecast(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        from app.services.trainability_forecast_service import forecast_training
        forecast = await forecast_training(db, project_id, use_cache=True)
        elapsed = int((time.monotonic() - started) * 1000)
        verdict = forecast.get("verdict") or "unknown"
        gate_prob = forecast.get("gate_pass_probability")
        msg = (
            f"Trainability forecast computed; verdict={verdict}"
            + (f" gate_pass≈{int(gate_prob * 100)}%" if isinstance(gate_prob, (int, float)) else "")
        )
        # ``likely_fail`` → warn. ``likely_pass`` / ``borderline`` → ok
        # for the smoke test (the surface is reachable + computable; the
        # project's actual readiness is the forecast's job to render).
        smoke_status: SmokeStatus = "warn" if verdict == "likely_fail" else "ok"
        return SmokeCheckResult(
            name="trainability_forecast",
            status=smoke_status,
            elapsed_ms=elapsed,
            message=msg,
            remediation=(
                "Open Training → Forecast for the per-signal breakdown."
                if smoke_status == "warn" else None
            ),
            metadata={"verdict": verdict, "gate_pass_probability": gate_prob},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="trainability_forecast",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Trainability forecast failed to compute.",
            envelope=_envelope_from_exception(stage="training", exc=exc),
        )


async def _check_synth_catalog(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        project = await db.get(Project, project_id)
        elapsed_pre = int((time.monotonic() - started) * 1000)
        if project is None:
            return SmokeCheckResult(
                name="synth_catalog",
                status="skip",
                elapsed_ms=elapsed_pre,
                message="Project doesn't exist — synth catalog skipped.",
            )
        selected = project.selected_recipe or {}
        recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
        if not recipe_id:
            return SmokeCheckResult(
                name="synth_catalog",
                status="skip",
                elapsed_ms=elapsed_pre,
                message="No recipe — synth catalog has nothing to enumerate.",
            )
        from app.services.synth_playbook_service import available_playbooks_for_recipe
        playbooks = available_playbooks_for_recipe(recipe_id)
        elapsed = int((time.monotonic() - started) * 1000)
        n = len(playbooks)
        if n == 0:
            return SmokeCheckResult(
                name="synth_catalog",
                status="warn",
                elapsed_ms=elapsed,
                message=f"No playbooks registered for recipe '{recipe_id}'.",
                remediation=(
                    "The synth panel will show an empty mode list. "
                    "Confirm the recipe ID matches one with playbooks "
                    "(classification, qa-sft, span-extraction, summarization)."
                ),
                metadata={"recipe_id": recipe_id, "playbook_count": 0},
            )
        return SmokeCheckResult(
            name="synth_catalog",
            status="ok",
            elapsed_ms=elapsed,
            message=f"{n} synth playbook{'s' if n != 1 else ''} registered for recipe '{recipe_id}'.",
            metadata={"recipe_id": recipe_id, "playbook_count": n},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="synth_catalog",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Couldn't enumerate synth playbooks.",
            envelope=_envelope_from_exception(stage="synthetic", exc=exc),
        )


async def _check_synth_backend(
    db: AsyncSession, project_id: int,  # noqa: ARG001 — backends are install-global
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        from app.services.synth_backends import BACKEND_REGISTRY
        # Cheap reachability probe per backend (each ``is_available``
        # call uses a 1s timeout). Any one reachable = ok.
        available_names: list[str] = []
        for cls in BACKEND_REGISTRY:
            try:
                if cls.is_available():
                    available_names.append(cls.name)
            except Exception:  # noqa: BLE001 — broken backend ≠ block the check
                continue
        elapsed = int((time.monotonic() - started) * 1000)
        if not available_names:
            return SmokeCheckResult(
                name="synth_backend",
                status="warn",
                elapsed_ms=elapsed,
                message="No local synth backend is reachable (Ollama + teacher).",
                remediation=(
                    "Install Ollama (https://ollama.com) and pull a model "
                    "(e.g. `ollama pull qwen2.5:14b-instruct-q4_K_M`), or "
                    "set TEACHER_MODEL_API_URL to an OpenAI-compatible "
                    "endpoint. Cloud providers (OpenAI/Anthropic/Deepseek) "
                    "work via the synth panel's Cloud picker once you "
                    "save an API key under Project Settings → Secrets."
                ),
                metadata={"available_backends": []},
            )
        return SmokeCheckResult(
            name="synth_backend",
            status="ok",
            elapsed_ms=elapsed,
            message=f"Synth backend{'s' if len(available_names) > 1 else ''} reachable: {', '.join(available_names)}.",
            metadata={"available_backends": available_names},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="synth_backend",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Couldn't enumerate synth backends.",
            envelope=_envelope_from_exception(stage="synthetic", exc=exc),
        )


async def _check_prepared_splits(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type.in_([
                    DatasetType.TRAIN, DatasetType.CLEANED, DatasetType.SYNTHETIC,
                ]),
            )
        )
        datasets = list(result.scalars())
        elapsed = int((time.monotonic() - started) * 1000)
        total_rows = sum(int(d.record_count or 0) for d in datasets)
        if total_rows == 0:
            return SmokeCheckResult(
                name="prepared_splits",
                status="warn",
                elapsed_ms=elapsed,
                message="No labelled corpus yet (0 rows in train/cleaned/synthetic).",
                remediation=(
                    "Open Pipeline → Ingest or Synth to add rows. "
                    "Training needs at least the recipe's minimum row count."
                ),
                metadata={"labelled_row_count": 0, "dataset_count": len(datasets)},
            )
        return SmokeCheckResult(
            name="prepared_splits",
            status="ok",
            elapsed_ms=elapsed,
            message=f"{total_rows} labelled rows across {len(datasets)} dataset(s).",
            metadata={"labelled_row_count": total_rows, "dataset_count": len(datasets)},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="prepared_splits",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Couldn't query the project's datasets.",
            envelope=_envelope_from_exception(stage="dataset", exc=exc),
        )


async def _check_adapter_handler_format(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    """Detect train/eval prompt-format drift before it produces a
    catastrophic 0%-ish held-out F1 (the failure mode that bit the
    SQLi-detector project after a clean training run).

    The recipe declares its canonical adapter
    (``recipe.adapter_id`` — e.g. ``classification-label`` for the
    Text-Classifier recipe). The prepared manifest records which
    adapter was actually used for dataset prep
    (``manifest.adapter_id``). When they disagree, the trainer used
    a different prompt format than the eval-time handler will use,
    and held-out F1 drops to noise even though training-time eval
    looked healthy.

    Status:
      * ``ok`` — adapters match.
      * ``warn`` — adapters disagree. Includes the expected/actual
        names + a remediation pointing at Data Prep with the
        correct adapter.
      * ``skip`` — recipe missing OR manifest doesn't yet exist OR
        either side doesn't declare an adapter. The check is
        informational and not load-bearing for projects that haven't
        prepped yet.
      * ``fail`` — the check itself errored (recipe catalog read
        broke, etc.).
    """
    started = time.monotonic()
    try:
        project = await db.get(Project, project_id)
        elapsed_pre = int((time.monotonic() - started) * 1000)
        if project is None:
            return SmokeCheckResult(
                name="adapter_handler_format",
                status="skip",
                elapsed_ms=elapsed_pre,
                message="Project doesn't exist — adapter/handler check skipped.",
            )
        selected = project.selected_recipe or {}
        recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
        if not recipe_id:
            return SmokeCheckResult(
                name="adapter_handler_format",
                status="skip",
                elapsed_ms=elapsed_pre,
                message="No recipe — can't compare adapters.",
            )
        from app.services.recipe_service import get_recipe
        recipe = get_recipe(recipe_id)
        if recipe is None:
            return SmokeCheckResult(
                name="adapter_handler_format",
                status="skip",
                elapsed_ms=elapsed_pre,
                message=f"Recipe '{recipe_id}' not in catalog.",
            )
        expected_adapter = getattr(recipe, "adapter_id", None)

        from app.services.eval_task_handler_service import read_prepared_manifest
        manifest = read_prepared_manifest(project_id)
        actual_adapter = manifest.get("adapter_id") if isinstance(manifest, dict) else None
        elapsed = int((time.monotonic() - started) * 1000)

        if not expected_adapter:
            return SmokeCheckResult(
                name="adapter_handler_format",
                status="skip",
                elapsed_ms=elapsed,
                message=f"Recipe '{recipe_id}' doesn't declare an adapter.",
            )
        if not actual_adapter:
            return SmokeCheckResult(
                name="adapter_handler_format",
                status="skip",
                elapsed_ms=elapsed,
                message=(
                    "Dataset hasn't been prepped yet (manifest has no "
                    "adapter_id). Check fires after the first Data Prep run."
                ),
                metadata={"expected_adapter": expected_adapter},
            )
        if expected_adapter == actual_adapter:
            # γ′ — adapter ids agreeing is necessary but NOT sufficient.
            # The SQLi-detector run #2 hit this exact false negative:
            # recipe + manifest both said ``classification-label`` but
            # the adapter itself wrote raw ``source_text → target_text``
            # rows without the production prompt template. Training
            # taught the model "complete this text", eval asked it to
            # "Classify the following text. Reply with one of: …", and
            # the model produced 98% unparseable predictions.
            #
            # Peek at the first prepared row and check whether the
            # handler's ``expected_prompt_prefixes`` appear in any of
            # the row's string-valued fields. If the handler declares
            # prefixes (i.e. it builds an instruction prompt at eval
            # time) and none of them are present in training rows, the
            # train/eval format gap is still open even though the
            # adapter names agree. Escalate to warn.
            peek = _peek_prepared_row_prefix_match(
                project_id=project_id,
                manifest=manifest,
                experiment_task_type=str(
                    (selected.get("task_type") or "")
                ).strip() or None,
                recipe_task_profile=getattr(recipe, "task_profile", None),
            )
            if peek is not None and peek["status"] == "mismatch":
                return SmokeCheckResult(
                    name="adapter_handler_format",
                    status="warn",
                    elapsed_ms=elapsed,
                    message=(
                        f"Adapter '{actual_adapter}' matches recipe "
                        f"'{recipe_id}' BUT the prepared rows don't "
                        f"contain the prompt format the "
                        f"{peek['handler_id']} handler builds at eval "
                        f"time. Expected at least one of: "
                        f"{peek['expected_prefixes']!r}. Trainer + eval "
                        f"will use different prompt formats and the "
                        f"held-out F1 will collapse even though the "
                        f"trainer's internal eval looks fine."
                    ),
                    remediation=(
                        "The adapter id matches but the adapter isn't "
                        "wrapping rows with the handler's prompt "
                        "template. Either (a) pick a different adapter "
                        "that does wrap rows (open Pipeline → Data Prep "
                        "→ Adapter Studio to inspect what each adapter "
                        "writes), or (b) treat this as a platform bug "
                        "on the adapter and file it — the adapter id "
                        "shouldn't claim a task profile it doesn't "
                        "produce."
                    ),
                    metadata={
                        "expected_adapter": expected_adapter,
                        "actual_adapter": actual_adapter,
                        "recipe_id": recipe_id,
                        "handler_id": peek["handler_id"],
                        "expected_prefixes": peek["expected_prefixes"],
                        "peeked_row_path": peek["peeked_row_path"],
                    },
                )
            return SmokeCheckResult(
                name="adapter_handler_format",
                status="ok",
                elapsed_ms=elapsed,
                message=(
                    f"Adapter '{actual_adapter}' matches recipe '{recipe_id}'"
                    + (
                        f" + prepared rows carry the {peek['handler_id']} "
                        f"handler's prompt format."
                        if peek is not None and peek["status"] == "match"
                        else "."
                    )
                ),
                metadata={
                    "expected_adapter": expected_adapter,
                    "actual_adapter": actual_adapter,
                    "recipe_id": recipe_id,
                    **({"peek": peek} if peek else {}),
                },
            )
        return SmokeCheckResult(
            name="adapter_handler_format",
            status="warn",
            elapsed_ms=elapsed,
            message=(
                f"Recipe '{recipe_id}' expects adapter "
                f"'{expected_adapter}' but dataset was prepped with "
                f"'{actual_adapter}'. Held-out eval will likely produce "
                f"unparseable predictions because the trainer's prompt "
                f"format won't match the eval handler's expected format."
            ),
            remediation=(
                f"Re-prep the dataset with the correct adapter: open "
                f"Pipeline → Data Prep and set adapter to "
                f"'{expected_adapter}'. Then retrain. The training-time "
                f"eval may stay high either way, but the held-out F1 "
                f"won't reflect the real model quality until the "
                f"adapter matches what the eval handler expects."
            ),
            metadata={
                "expected_adapter": expected_adapter,
                "actual_adapter": actual_adapter,
                "recipe_id": recipe_id,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="adapter_handler_format",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Couldn't check adapter/handler format.",
            envelope=_envelope_from_exception(stage="dataset", exc=exc),
        )


async def _check_experiments_accessible(
    db: AsyncSession, project_id: int,
) -> SmokeCheckResult:
    started = time.monotonic()
    try:
        result = await db.execute(
            select(Experiment).where(Experiment.project_id == project_id)
        )
        experiments = list(result.scalars())
        elapsed = int((time.monotonic() - started) * 1000)
        return SmokeCheckResult(
            name="experiments_accessible",
            status="ok",
            elapsed_ms=elapsed,
            message=f"{len(experiments)} experiment{'s' if len(experiments) != 1 else ''} on this project.",
            metadata={"experiment_count": len(experiments)},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheckResult(
            name="experiments_accessible",
            status="fail",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            message="Couldn't query the experiments table.",
            envelope=_envelope_from_exception(stage="training", exc=exc),
        )


# ─────────────────────────────────────────────────────────────────────
# Orchestrator.
# ─────────────────────────────────────────────────────────────────────


# Ordered for display — the frontend renders the checks in this
# order so users see the natural pipeline progression
# (project → recipe → data → forecast → synth → experiments).
_CHECKS: tuple = (
    _check_project_exists,
    _check_recipe_applied,
    _check_gold_set,
    _check_data_health,
    _check_trainability_forecast,
    _check_synth_catalog,
    _check_synth_backend,
    _check_prepared_splits,
    _check_adapter_handler_format,
    _check_experiments_accessible,
)


async def run_smoke_test(
    db: AsyncSession, project_id: int,
) -> SmokeTestSummary:
    """Run every check in parallel + roll up the result.

    ``overall`` reflects the worst-severity check:
      * any check ``fail`` → overall ``fail``
      * else any ``warn`` → overall ``warn``
      * else ok (skip is ignored for overall — it's neutral)
    """
    started = time.monotonic()
    checks = await asyncio.gather(*(check(db, project_id) for check in _CHECKS))
    elapsed = int((time.monotonic() - started) * 1000)
    counts: dict[str, int] = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    if counts["fail"] > 0:
        overall: SmokeStatus = "fail"
    elif counts["warn"] > 0:
        overall = "warn"
    else:
        overall = "ok"
    return SmokeTestSummary(
        project_id=project_id,
        overall=overall,
        elapsed_ms=elapsed,
        counts=counts,
        checks=list(checks),
    )


def serialize_summary(summary: SmokeTestSummary) -> dict[str, Any]:
    """Wire shape for the API response. Mirrors SmokeCheckResult /
    SmokeTestSummary but flat enough that the frontend can use the
    fields directly without aliases."""
    return {
        "project_id": summary.project_id,
        "overall": summary.overall,
        "elapsed_ms": summary.elapsed_ms,
        "counts": summary.counts,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "elapsed_ms": c.elapsed_ms,
                "message": c.message,
                "remediation": c.remediation,
                "envelope": c.envelope,
                "metadata": dict(c.metadata),
            }
            for c in summary.checks
        ],
    }
