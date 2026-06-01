"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import async_session_factory, init_db
from app.models.auth import AuditLog
from app.security import authorize_request, ensure_bootstrap_auth, extract_project_id_from_path
from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.audit import router as audit_router
from app.api.drift import router as drift_router
from app.api.remediation import router as remediation_router
from app.api.settings import router as settings_router
from app.api.projects import router as projects_router
from app.api.pipeline import router as pipeline_router
from app.api.ingestion import router as ingestion_router
from app.api.cleaning import router as cleaning_router
from app.api.data_health import router as data_health_router
from app.api.distillation import router as distillation_router
from app.api.hardware import router as hardware_router
from app.api.dataset import router as dataset_router
from app.api.data_studio import router as data_studio_router
from app.api.gold import router as gold_router
from app.api.synthetic import router as synthetic_router
from app.api.gamification import router as gamification_router
from app.api.coach import router as coach_router
from app.api.curriculum import router as curriculum_router
from app.api.jobs import router as jobs_router
from app.api.archetypes import router as archetypes_router
from app.api.auto_rag import router as auto_rag_router
from app.api.annotation import router as annotation_router
from app.api.dataset_import import (
    catalog_router as dataset_import_catalog_router,
    project_router as dataset_import_project_router,
)
from app.api.tokenization import router as tokenization_router
from app.api.training import router as training_router
from app.api.evaluation import router as evaluation_router
from app.api.compression import router as compression_router
from app.api.export import router as export_router
from app.api.comparison import router as comparison_router
from app.api.registry import router as registry_router
from app.api.secrets import router as secrets_router
from app.api.domain_packs import router as domain_packs_router
from app.api.domain_profiles import router as domain_profiles_router
from app.api.domain_blueprints import router as domain_blueprints_router
from app.api.artifacts import router as artifacts_router
from app.api.targets import router as targets_router
from app.api.starter_packs import router as starter_packs_router
from app.api.recipes import router as recipes_router
from app.api.models import router as base_models_router
from app.api.adapter_studio import router as adapter_studio_router
from app.api.autopilot import router as autopilot_router
from app.api.gold_workbench import router as gold_workbench_router
from app.api.manifest import router as manifest_router, project_router as manifest_project_router
from app.api.deployments import (
    router as deployments_router,
    project_router as deployments_project_router,
)
from app.api.run_events import (
    router as run_events_router,
    project_router as run_events_project_router,
    timeline_project_router as timeline_project_router,
)
from app.api.run_event_clusters import (
    project_router as run_event_clusters_project_router,
)
from app.api.support_bundles import (
    router as support_bundles_router,
    project_router as support_bundles_project_router,
)
from app.api.extensions import router as extensions_router
from app.api.demo_projects import router as demo_projects_router
from app.api.project_templates import router as project_templates_router
from app.api.quickstart import router as quickstart_router
from app.services.domain_pack_service import ensure_default_domain_pack
from app.services.domain_hook_service import load_hook_plugins_from_settings
from app.services.domain_profile_service import ensure_default_domain_profile
from app.services.data_adapter_service import load_data_adapter_plugins_from_settings
from app.services.dataset_import.plugin_loader import (
    load_dataset_mapper_plugins_from_settings,
)
from app.services.model_selection_service import load_model_catalog_plugins_from_settings
from app.services.runtime_settings_service import apply_persisted_runtime_overrides
from app.services.starter_pack_service import load_starter_pack_plugins_from_settings
from app.services.target_profile_service import load_target_profile_plugins_from_settings
from app.exceptions import SLMError
from fastapi.responses import JSONResponse

# Stages the structured-error envelope recognises. Order matters —
# longer / more-specific stage names must come BEFORE shorter prefixes
# they'd be confused with (e.g. ``data-health`` before any future
# ``data`` stage). Adding a new stage is safe as long as its URL
# substring is unique enough.
_TARGET_STRUCTURED_ERROR_STAGES: tuple[str, ...] = (
    "ingestion",
    "training",
    "export",
    # Widened coverage (Diagnostics Intervention A) — every other
    # surface where errors used to render as raw ``{detail: "..."}``
    # toasts. Each stage gets a stable name so frontend dispatchers
    # + log aggregators can route by ``stage`` instead of regexing
    # URLs.
    "synthetic",
    "cleaning",
    "gold",
    "evaluation",
    "data-health",
    "dataset-import",
    "dataset",
    "deployments",
    "playground",
    "annotation",
    "manifest",
    "tokenization",
    "auto-rag",
    "distillation",
    "compression",
    "drift",
    "secrets",
    "jobs",
    "recipes",
    "comparison",
    "artifacts",
    "audit",
    "remediation",
    "starter-packs",
    "templates",
    "extensions",
    "support-bundles",
    "domain-profile",
    "domain-pack",
    "pipeline",
    "settings",
    "stats",
    "runtime",
    "gate-check",
)
_STAGE_DOCS_URL: dict[str, str] = {
    "ingestion": "/docs/ingestion/troubleshooting",
    "training": "/docs/training/troubleshooting",
    "export": "/docs/export/troubleshooting",
    "general": "/docs/troubleshooting",
}


def _infer_structured_error_stage(path: str) -> str | None:
    """Stage name for an API URL, used to enrich the error envelope.

    Returns ``"general"`` for any ``/api/...`` URL whose path doesn't
    match a more specific stage — every API error gets wrapped in the
    envelope shape. Non-``/api/`` paths (rare; serves static or health)
    return ``None`` so we don't wrap them.

    Stage detection is greedy on the first segment after
    ``/api/`` or ``/api/projects/{id}/``. For ``/api/projects/17/
    synthetic/run-playbook`` the stage is ``synthetic``; for
    ``/api/health`` it's ``general``.
    """
    if not path.startswith("/api/"):
        return None
    normalized = path.lower()
    for stage in _TARGET_STRUCTURED_ERROR_STAGES:
        # Match as a path segment so ``/dataset`` doesn't swallow
        # ``/dataset-import`` (we list the longer one first above so
        # the loop hits it first).
        if f"/{stage}/" in normalized or normalized.endswith(f"/{stage}"):
            return stage
    return "general"


def _default_error_code(stage: str, status_code: int) -> str:
    # Stage names use hyphens (``data-health``, ``dataset-import``)
    # but error codes follow SCREAMING_SNAKE_CASE so frontend
    # dispatchers can use them as identifiers.
    prefix = str(stage or "general").strip().upper().replace("-", "_") or "GENERAL"
    if status_code == 404:
        return f"{prefix}_NOT_FOUND"
    if status_code == 409:
        return f"{prefix}_CONFLICT"
    if status_code == 422:
        return f"{prefix}_VALIDATION_ERROR"
    if status_code >= 500:
        return f"{prefix}_INTERNAL_ERROR"
    return f"{prefix}_REQUEST_FAILED"


def _default_actionable_fix(stage: str, status_code: int) -> str:
    if status_code == 404:
        return "Verify the project/resource identifier and try again."
    if status_code in {400, 422}:
        return "Review request inputs and retry."
    if status_code == 409:
        return "Resolve the conflicting resource state and retry."
    if status_code >= 500:
        return f"Retry the {stage} action. If it persists, inspect server logs."
    return f"Retry the {stage} operation after checking configuration."


def _new_troubleshooting_id() -> str:
    """Short opaque id for the error envelope.

    The user copy-pastes this into a bug report; the developer
    greps logs for it. 12 url-safe chars is enough for non-collision
    over the project's expected error volume (millions+) without
    being overwhelming to read aloud.
    """
    import secrets as _secrets
    return f"err_{_secrets.token_urlsafe(9)}"


def _structured_error_payload(
    *,
    stage: str,
    status_code: int,
    detail: Any,
    troubleshooting_id: str | None = None,
) -> dict[str, Any]:
    docs_url = _STAGE_DOCS_URL.get(stage, _STAGE_DOCS_URL["general"])
    trace_id = troubleshooting_id or _new_troubleshooting_id()
    if isinstance(detail, dict):
        payload = dict(detail)
        message = str(
            payload.get("message")
            or payload.get("detail")
            or payload.get("error")
            or "Request failed."
        )
        payload["error_code"] = str(
            payload.get("error_code") or _default_error_code(stage, status_code)
        )
        payload["stage"] = str(payload.get("stage") or stage)
        payload["message"] = message
        payload["actionable_fix"] = str(
            payload.get("actionable_fix")
            or _default_actionable_fix(str(payload.get("stage") or stage), status_code)
        )
        payload["docs_url"] = str(payload.get("docs_url") or docs_url)
        payload.setdefault("metadata", payload.get("metadata"))
        # Diagnostics Intervention A — every error envelope carries a
        # short opaque id the user can copy-paste into a bug report.
        # Developer greps logs for it; users get a reference token
        # instead of trying to describe their failure mode in prose.
        payload.setdefault("troubleshooting_id", trace_id)
        # Keep backward compatibility with callers/tests expecting plain `detail`.
        payload.setdefault("detail", message)
        return payload

    message = str(detail or "Request failed.")
    return {
        "error_code": _default_error_code(stage, status_code),
        "stage": stage,
        "message": message,
        "actionable_fix": _default_actionable_fix(stage, status_code),
        "docs_url": docs_url,
        "metadata": None,
        "troubleshooting_id": trace_id,
        # Keep backward compatibility with callers/tests expecting plain `detail`.
        "detail": message,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings.ensure_dirs()
    apply_persisted_runtime_overrides()
    load_hook_plugins_from_settings()
    load_data_adapter_plugins_from_settings()
    load_dataset_mapper_plugins_from_settings()
    load_target_profile_plugins_from_settings()
    load_model_catalog_plugins_from_settings()
    load_starter_pack_plugins_from_settings()
    await init_db()
    await ensure_bootstrap_auth()
    async with async_session_factory() as db:
        await ensure_default_domain_profile(db)
        await ensure_default_domain_pack(db)
        await db.commit()
    # Story 1.5 Gate 3 — sweep any experiments stuck on RUNNING whose
    # training_report.json shows they actually finished. Cheap (one
    # query + a JSON read per stale row) and best-effort: failure
    # logs and continues so a flaky filesystem can't keep the API
    # from booting.
    try:
        from app.services.training_service import (
            reconcile_stale_running_experiments,
        )

        async with async_session_factory() as db:
            fixed = await reconcile_stale_running_experiments(db)
            if fixed:
                print(
                    f"[startup] reconciled {len(fixed)} stuck-RUNNING "
                    f"experiment(s): {[r['experiment_id'] for r in fixed]}",
                    flush=True,
                )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[startup] stuck-RUNNING reconciliation skipped: {exc!r}",
            flush=True,
        )
    # Hardening Phase H1 — sweep orphaned background-jobs from a
    # previous process. The asyncio runner doesn't survive a restart,
    # so leaving Job rows in QUEUED/RUNNING means the notification
    # bell spins forever showing work that's actually dead. We mark
    # them FAILED with a "lost_during_restart" error so the user can
    # see what happened and re-trigger.
    try:
        from app.services.jobs_service import reconcile_orphaned_jobs

        async with async_session_factory() as db:
            report = await reconcile_orphaned_jobs(db)
            if report["queued_swept"] or report["running_swept"]:
                print(
                    f"[startup] swept orphaned jobs: "
                    f"queued={report['queued_swept']} running={report['running_swept']}",
                    flush=True,
                )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[startup] jobs reconciliation skipped: {exc!r}",
            flush=True,
        )
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Modular platform for building, evaluating, compressing, and exporting domain-specific Small Language Models",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_DEPENDENCIES = [Depends(authorize_request)]

# Mount API routers
app.include_router(auth_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(admin_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(remediation_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(drift_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(audit_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(settings_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(projects_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(pipeline_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(ingestion_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(hardware_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(cleaning_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(data_health_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(distillation_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(dataset_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(data_studio_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(gold_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(synthetic_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(gamification_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(coach_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(curriculum_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(jobs_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(archetypes_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(auto_rag_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(annotation_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(dataset_import_catalog_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(dataset_import_project_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(tokenization_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(training_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(evaluation_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(compression_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(export_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(comparison_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(registry_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(secrets_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(domain_packs_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(domain_profiles_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(domain_blueprints_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(artifacts_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(targets_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(starter_packs_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(recipes_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(base_models_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(adapter_studio_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(autopilot_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(gold_workbench_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(manifest_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(manifest_project_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(deployments_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(deployments_project_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(run_events_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(run_events_project_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(timeline_project_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(run_event_clusters_project_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(support_bundles_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(support_bundles_project_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(extensions_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(demo_projects_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(project_templates_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(quickstart_router, prefix="/api", dependencies=API_DEPENDENCIES)


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    """Persist auditable API request entries."""
    request_id = uuid4().hex
    request.state.request_id = request_id

    start = perf_counter()
    status_code = 500
    err: str | None = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        err = str(e)
        raise
    finally:
        path = request.url.path
        should_audit = settings.AUDIT_LOG_ENABLED and path.startswith("/api")

        # Keep read-only success responses out of audit logs for lower noise.
        if should_audit and request.method in {"GET", "HEAD", "OPTIONS"} and status_code < 400:
            should_audit = False

        if should_audit:
            duration_ms = round((perf_counter() - start) * 1000, 2)
            principal = getattr(request.state, "principal", None)
            user_id = getattr(principal, "user_id", None)
            project_id = getattr(request.state, "project_id", None)
            if project_id is None:
                project_id = extract_project_id_from_path(path)

            try:
                async with async_session_factory() as db:
                    audit = AuditLog(
                        request_id=request_id,
                        method=request.method.upper(),
                        path=path,
                        status_code=status_code,
                        user_id=user_id,
                        project_id=project_id,
                        action=f"{request.method.upper()} {path}",
                        ip_address=request.client.host if request.client else None,
                        user_agent=request.headers.get("user-agent", ""),
                        metadata_={
                            "query": request.url.query,
                            "duration_ms": duration_ms,
                            "error": err,
                        },
                    )
                    db.add(audit)
                    await db.commit()
            except Exception:
                # Audit failures should not break the request path.
                pass


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "auth_enabled": settings.AUTH_ENABLED,
    }


@app.exception_handler(SLMError)
async def slm_exception_handler(request: Request, exc: SLMError):
    # SLMError already carries a structured detail dict — re-wrap it
    # through the envelope so it also picks up troubleshooting_id and
    # any docs_url defaults the bare exception didn't set.
    stage = _infer_structured_error_stage(request.url.path) or "general"
    return JSONResponse(
        status_code=exc.status_code,
        content=_structured_error_payload(
            stage=stage,
            status_code=exc.status_code,
            detail=exc.detail,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    stage = _infer_structured_error_stage(request.url.path)
    if stage:
        return JSONResponse(
            status_code=exc.status_code,
            content=_structured_error_payload(
                stage=stage,
                status_code=exc.status_code,
                detail=exc.detail,
            ),
        )
    # Path didn't match any /api/ prefix (rare — static / health
    # endpoints). Keep the legacy shape so we don't break tests.
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    stage = _infer_structured_error_stage(request.url.path)
    if stage:
        validation_errors = jsonable_encoder(exc.errors())
        payload = _structured_error_payload(
            stage=stage,
            status_code=422,
            detail={
                "message": "Request validation failed.",
                "metadata": {"validation_errors": validation_errors},
            },
        )
        # Preserve default FastAPI validation detail for compatibility.
        payload["detail"] = validation_errors
        return JSONResponse(status_code=422, content=payload)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


@app.exception_handler(Exception)
async def last_resort_exception_handler(request: Request, exc: Exception):
    """Last-resort wrapper for unhandled exceptions.

    Catches the failure-mode that bit us with the Kaggle SDK's
    ``sys.exit(1)`` at import time: an exception (or BaseException
    subclass like SystemExit) escapes every try/except in the request
    handler and crashes the request with a 500 + bare stack trace
    that's only visible in the server log.

    Wrapping it here:
      * Always returns a 500 with the structured envelope shape.
      * Logs the full traceback server-side under the same
        troubleshooting_id the user sees, so support can correlate.
      * Surfaces the exception type as the ``error_code`` (e.g.
        ``synthetic.SystemExit``) so frontend dispatch + log
        aggregation can group by failure mode.

    FastAPI's default last-resort handler returns a bare 500 with no
    body. Registering ``Exception`` re-routes it through us. We
    deliberately do NOT catch ``BaseException`` — ``KeyboardInterrupt``
    + ``SystemExit`` from the worker process itself should still
    terminate cleanly.
    """
    stage = _infer_structured_error_stage(request.url.path) or "general"
    trace_id = _new_troubleshooting_id()
    exc_name = type(exc).__name__
    # Log the full traceback under the trace_id so a developer can
    # grep the log for it after a user reports the id.
    logging.getLogger("app.last_resort").exception(
        "unhandled exception in %s [trace_id=%s] %s",
        request.url.path, trace_id, exc_name,
    )
    return JSONResponse(
        status_code=500,
        content=_structured_error_payload(
            stage=stage,
            status_code=500,
            detail={
                "message": f"{exc_name}: {str(exc) or '(no message)'}",
                "error_code": f"{stage.upper().replace('-', '_')}_UNHANDLED_{exc_name.upper()}",
                "actionable_fix": (
                    "An unexpected server-side error occurred. Copy the "
                    "troubleshooting_id below and report it — the server "
                    "log captured the full traceback under this id."
                ),
                "metadata": {
                    "exception_type": exc_name,
                    "request_path": request.url.path,
                },
            },
            troubleshooting_id=trace_id,
        ),
    )
