"""Application configuration via environment variables and .env file."""

from pathlib import Path
from typing import Optional
import warnings

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "f", "no", "n", "off"}


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
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_CACHE_DIR: Path = PROJECT_ROOT / "data" / "models"

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
    CELERY_VISIBILITY_TIMEOUT_SECONDS: int = 43200

    # ── CORS ────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Teacher Model (for synthetic generation) ────────────────────────
    TEACHER_MODEL_API_URL: str = ""
    TEACHER_MODEL_API_KEY: str = ""
    ALLOW_SYNTHETIC_DEMO_FALLBACK: bool = False

    # ── Judge Model (for evaluation) ────────────────────────────────────
    JUDGE_MODEL_API_URL: str = ""
    JUDGE_MODEL_API_KEY: str = ""
    SECRETS_ENCRYPTION_KEY: str = ""

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
    # JSON array of Python module paths to auto-load training runtime plugins.
    # Example:
    # TRAINING_RUNTIME_PLUGIN_MODULES='["app.plugins.training_runtimes.example_runtime"]'
    TRAINING_RUNTIME_PLUGIN_MODULES: list[str] = []

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
    MERGE_MODELS_EXTERNAL_CMD: str = (
        'python "{backend_dir}/scripts/model_merge.py" '
        '--project {project_id} --models-file "{models_file_path}" '
        '--method "{merge_method}" --out "{output_model_path}" '
        '--weights "{weights_csv}" --ties-density {ties_density}'
    )
    BENCHMARK_EXTERNAL_CMD: str = (
        'python "{backend_dir}/scripts/benchmark.py" '
        '--project {project_id} --model "{model_path}" --samples {num_samples} '
        '--out "{benchmark_output_path}"'
    )
    LLAMA_CPP_DIR: str = ""
    LLAMA_CPP_CONVERT_SCRIPT: str = ""
    LLAMA_CPP_QUANTIZE_BIN: str = ""
    ONNX_EXPORT_TASK: str = "auto"
    PYTHON_EXECUTABLE: str = ""

    # ── Process Runtime Controls ────────────────────────────────────────
    STRICT_EXECUTION_MODE: bool = False
    EXTERNAL_COMMAND_TIMEOUT_SECONDS: int = 21600

    # ── Domain Hook Plugins ─────────────────────────────────────────────
    # Example:
    # DOMAIN_HOOK_PLUGIN_MODULES='["app.plugins.domain_hooks.example_hooks"]'
    DOMAIN_HOOK_PLUGIN_MODULES: list[str] = []

    # ── Data Adapter Plugins ────────────────────────────────────────────
    # Example:
    # DATA_ADAPTER_PLUGIN_MODULES='["app.plugins.data_adapters.example_adapters"]'
    DATA_ADAPTER_PLUGIN_MODULES: list[str] = []

    # ── Target Profile Catalog Plugins ──────────────────────────────────
    # Example:
    # TARGET_PROFILE_PLUGIN_MODULES='["app.plugins.target_profiles.acme_targets"]'
    TARGET_PROFILE_PLUGIN_MODULES: list[str] = []

    # ── Model Catalog Plugins ───────────────────────────────────────────
    # Example:
    # MODEL_CATALOG_PLUGIN_MODULES='["app.plugins.model_catalogs.acme_models"]'
    MODEL_CATALOG_PLUGIN_MODULES: list[str] = []

    # ── Starter Pack Catalog Plugins ────────────────────────────────────
    # Example:
    # STARTER_PACK_PLUGIN_MODULES='["app.plugins.starter_packs.acme_domain_starters"]'
    STARTER_PACK_PLUGIN_MODULES: list[str] = []

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("DEBUG", mode="before")
    @classmethod
    def normalize_debug_bool(cls, value: object) -> bool:
        """Harden DEBUG parsing so malformed ambient env values do not crash bootstrap."""
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in _BOOL_TRUE:
                return True
            if token in _BOOL_FALSE:
                return False
            if not token:
                return False

        warnings.warn(
            (
                f"Invalid DEBUG value {value!r}; falling back to DEBUG=False. "
                "Use one of: true/false, 1/0, yes/no, on/off."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    @model_validator(mode="after")
    def normalize_paths(self) -> "Settings":
        """Normalize filesystem-backed settings for consistent behavior across CWDs."""
        self.DATA_DIR = Path(self.DATA_DIR).expanduser().resolve()
        self.MODEL_CACHE_DIR = Path(self.MODEL_CACHE_DIR).expanduser().resolve()
        self.DATABASE_URL = self._normalize_sqlite_database_url(self.DATABASE_URL)
        return self

    def _normalize_sqlite_database_url(self, value: str) -> str:
        for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
            if not value.startswith(prefix):
                continue

            path_and_query = value[len(prefix):]
            path_part, sep, query_part = path_and_query.partition("?")

            # Already absolute: sqlite+aiosqlite:////abs/path.db
            if path_part.startswith("/"):
                return value

            absolute_path = (PROJECT_ROOT / path_part).resolve()
            normalized = f"{prefix}{absolute_path.as_posix()}"
            if sep:
                normalized = f"{normalized}?{query_part}"
            return normalized
        return value

    def ensure_dirs(self) -> None:
        """Create required data directories."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "projects").mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)


settings = Settings()
