"""Application configuration via environment variables and .env file."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    # ── App ─────────────────────────────────────────────────────────────
    APP_NAME: str = "SLM Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── Auth / RBAC ─────────────────────────────────────────────────────
    AUTH_ENABLED: bool = True
    AUTH_BOOTSTRAP_API_KEY: str = ""
    AUTH_BOOTSTRAP_USERNAME: str = "admin"
    AUTH_BOOTSTRAP_ROLE: str = "admin"
    AUDIT_LOG_ENABLED: bool = True

    # ── Paths ───────────────────────────────────────────────────────────
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
    MODEL_CACHE_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data" / "models"

    # ── Database ────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/slm_platform.db"
    DB_AUTO_CREATE: bool = False
    ALLOW_SQLITE_AUTOCREATE: bool = True
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "SLM Studio"

    # API Keys & Auth
    API_KEY: str = "sk-mock-admin-key"
    JWT_SECRET: str = "super-secret-jwt-key-replace-in-prod"
    
    # OIDC (SSO) Configuration
    OIDC_CLIENT_ID: Optional[str] = None
    OIDC_CLIENT_SECRET: Optional[str] = None
    OIDC_DISCOVERY_URL: Optional[str] = None

    # Database Settings (Already defined above with defaults)
    DB_REQUIRE_ALEMBIC_HEAD: bool = True
    ALEMBIC_CONFIG_FILE: str = "alembic.ini"

    # ── Redis / Celery ──────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── CORS ────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Teacher Model (for synthetic generation) ────────────────────────
    TEACHER_MODEL_API_URL: str = ""
    TEACHER_MODEL_API_KEY: str = ""
    ALLOW_SYNTHETIC_DEMO_FALLBACK: bool = False

    # ── Judge Model (for evaluation) ────────────────────────────────────
    JUDGE_MODEL_API_URL: str = ""
    JUDGE_MODEL_API_KEY: str = ""

    # ── Ingestion Strictness ────────────────────────────────────────────
    ALLOW_SIMULATED_INGESTION_FALLBACK: bool = False

    # ── Training Runtime Backend ────────────────────────────────────────
    TRAINING_BACKEND: str = "external"  # simulate | external
    ALLOW_SIMULATED_TRAINING: bool = False
    TRAINING_EXTERNAL_CMD: str = (
        'python "{backend_dir}/scripts/train.py" '
        "--project {project_id} --experiment {experiment_id} "
        '--output "{output_dir}" --base-model "{base_model}" '
        '--config "{config_path}" --train-file "{train_file}" --val-file "{val_file}"'
    )

    # ── Compression Runtime Backend ─────────────────────────────────────
    COMPRESSION_BACKEND: str = "external"  # external | stub
    ALLOW_STUB_COMPRESSION: bool = False
    QUANTIZE_EXTERNAL_CMD: str = (
        'python "{backend_dir}/scripts/quantize.py" '
        '--project {project_id} --model "{model_path}" --bits {bits} '
        "--format {output_format} --out {output_model_path}"
    )
    MERGE_LORA_EXTERNAL_CMD: str = (
        'python "{backend_dir}/scripts/quantize.py" '
        '--project {project_id} --model "{base_model_path}" --bits 16 '
        '--format merged --out "{output_model_path}" --adapter "{lora_adapter_path}"'
    )
    BENCHMARK_EXTERNAL_CMD: str = (
        'python "{backend_dir}/scripts/benchmark.py" '
        '--project {project_id} --model "{model_path}" --samples {num_samples} '
        '--out "{output_dir}/benchmark_results.json"'
    )

    # ── Process Runtime Controls ────────────────────────────────────────
    EXTERNAL_COMMAND_TIMEOUT_SECONDS: int = 21600

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def ensure_dirs(self) -> None:
        """Create required data directories."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "projects").mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)


settings = Settings()
