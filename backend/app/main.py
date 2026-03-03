"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.api.projects import router as projects_router
from app.api.pipeline import router as pipeline_router
from app.api.ingestion import router as ingestion_router
from app.api.cleaning import router as cleaning_router
from app.api.gold import router as gold_router
from app.api.synthetic import router as synthetic_router
from app.api.training import router as training_router
from app.api.evaluation import router as evaluation_router
from app.api.compression import router as compression_router
from app.api.export import router as export_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings.ensure_dirs()
    await init_db()
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

# Mount API routers
app.include_router(projects_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")
app.include_router(ingestion_router, prefix="/api")
app.include_router(cleaning_router, prefix="/api")
app.include_router(gold_router, prefix="/api")
app.include_router(synthetic_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
app.include_router(compression_router, prefix="/api")
app.include_router(export_router, prefix="/api")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}
