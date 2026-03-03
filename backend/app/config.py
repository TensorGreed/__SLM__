"""Application configuration via environment variables and .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    # ── App ─────────────────────────────────────────────────────────────
    APP_NAME: str = "SLM Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── Paths ───────────────────────────────────────────────────────────
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
    MODEL_CACHE_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data" / "models"

    # ── Database ────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/slm_platform.db"

    # ── Redis / Celery ──────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── CORS ────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Teacher Model (for synthetic generation) ────────────────────────
    TEACHER_MODEL_API_URL: str = ""
    TEACHER_MODEL_API_KEY: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def ensure_dirs(self) -> None:
        """Create required data directories."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "projects").mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)


settings = Settings()
