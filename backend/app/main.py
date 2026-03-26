"""FastAPI application entry point."""

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
from app.api.auth import router as auth_router
from app.api.audit import router as audit_router
from app.api.settings import router as settings_router
from app.api.projects import router as projects_router
from app.api.pipeline import router as pipeline_router
from app.api.ingestion import router as ingestion_router
from app.api.cleaning import router as cleaning_router
from app.api.hardware import router as hardware_router
from app.api.dataset import router as dataset_router
from app.api.gold import router as gold_router
from app.api.synthetic import router as synthetic_router
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
from app.api.artifacts import router as artifacts_router
from app.api.targets import router as targets_router
from app.api.starter_packs import router as starter_packs_router
from app.services.domain_pack_service import ensure_default_domain_pack
from app.services.domain_hook_service import load_hook_plugins_from_settings
from app.services.domain_profile_service import ensure_default_domain_profile
from app.services.data_adapter_service import load_data_adapter_plugins_from_settings
from app.services.model_selection_service import load_model_catalog_plugins_from_settings
from app.services.runtime_settings_service import apply_persisted_runtime_overrides
from app.services.starter_pack_service import load_starter_pack_plugins_from_settings
from app.services.target_profile_service import load_target_profile_plugins_from_settings
from app.exceptions import SLMError
from fastapi.responses import JSONResponse

_TARGET_STRUCTURED_ERROR_STAGES: tuple[str, ...] = ("ingestion", "training", "export")
_STAGE_DOCS_URL: dict[str, str] = {
    "ingestion": "/docs/ingestion/troubleshooting",
    "training": "/docs/training/troubleshooting",
    "export": "/docs/export/troubleshooting",
    "general": "/docs/troubleshooting",
}


def _infer_structured_error_stage(path: str) -> str | None:
    if not path.startswith("/api/projects/"):
        return None
    normalized = path.lower()
    for stage in _TARGET_STRUCTURED_ERROR_STAGES:
        if f"/{stage}" in normalized:
            return stage
    return None


def _default_error_code(stage: str, status_code: int) -> str:
    prefix = str(stage or "general").strip().upper() or "GENERAL"
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


def _structured_error_payload(
    *,
    stage: str,
    status_code: int,
    detail: Any,
) -> dict[str, Any]:
    docs_url = _STAGE_DOCS_URL.get(stage, _STAGE_DOCS_URL["general"])
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
    load_target_profile_plugins_from_settings()
    load_model_catalog_plugins_from_settings()
    load_starter_pack_plugins_from_settings()
    await init_db()
    await ensure_bootstrap_auth()
    async with async_session_factory() as db:
        await ensure_default_domain_profile(db)
        await ensure_default_domain_pack(db)
        await db.commit()
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
app.include_router(audit_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(settings_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(projects_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(pipeline_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(ingestion_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(hardware_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(cleaning_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(dataset_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(gold_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(synthetic_router, prefix="/api", dependencies=API_DEPENDENCIES)
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
app.include_router(artifacts_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(targets_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(starter_packs_router, prefix="/api", dependencies=API_DEPENDENCIES)


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
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
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
