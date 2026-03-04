"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import async_session_factory, init_db
from app.models.auth import AuditLog
from app.security import authorize_request, ensure_bootstrap_auth, extract_project_id_from_path
from app.api.auth import router as auth_router
from app.api.audit import router as audit_router
from app.api.projects import router as projects_router
from app.api.pipeline import router as pipeline_router
from app.api.ingestion import router as ingestion_router
from app.api.cleaning import router as cleaning_router
from app.api.dataset import router as dataset_router
from app.api.gold import router as gold_router
from app.api.synthetic import router as synthetic_router
from app.api.tokenization import router as tokenization_router
from app.api.training import router as training_router
from app.api.evaluation import router as evaluation_router
from app.api.compression import router as compression_router
from app.api.export import router as export_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings.ensure_dirs()
    await init_db()
    await ensure_bootstrap_auth()
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
app.include_router(projects_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(pipeline_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(ingestion_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(cleaning_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(dataset_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(gold_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(synthetic_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(tokenization_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(training_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(evaluation_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(compression_router, prefix="/api", dependencies=API_DEPENDENCIES)
app.include_router(export_router, prefix="/api", dependencies=API_DEPENDENCIES)


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
