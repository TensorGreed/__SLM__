"""SQLAlchemy async database engine and session management."""

from pathlib import Path

from sqlalchemy import event, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

_engine_kwargs: dict = {
    "echo": settings.DEBUG,
    "future": True,
}

if _is_sqlite:
    # SQLite needs NullPool with async or StaticPool for single-connection use.
    # We use connect_args to disable the same-thread check and set a busy timeout.
    from sqlalchemy.pool import StaticPool

    _engine_kwargs.update(
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=StaticPool,
    )

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL journal mode and other performance pragmas for SQLite."""
    if _is_sqlite:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


BASE_REQUIRED_TABLES = {
    "domain_packs",
    "domain_profiles",
    "domain_blueprints",
    "projects",
    "artifact_records",
    "workflow_runs",
    "workflow_run_nodes",
    "datasets",
    "dataset_versions",
    "dataset_adapter_definitions",
    "raw_documents",
    "experiments",
    "checkpoints",
    "eval_results",
    "exports",
    "model_registry_entries",
    "base_model_registry_entries",
    "project_secrets",
    "playground_sessions",
    "autopilot_decisions",
    "autopilot_snapshots",
    "autopilot_repair_previews",
}

AUTH_REQUIRED_TABLES = {
    "users",
    "api_keys",
    "project_memberships",
    "audit_logs",
}

CRITICAL_COLUMN_REQUIREMENTS: dict[str, set[str]] = {
    "projects": {
        "beginner_mode",
        "active_domain_blueprint_version",
    },
}


def _list_missing_columns(sync_conn, requirements: dict[str, set[str]]) -> list[str]:
    """Return missing required columns as dotted table.column strings."""
    inspector = inspect(sync_conn)
    table_names = set(inspector.get_table_names())
    missing: list[str] = []
    for table_name, required_columns in requirements.items():
        if table_name not in table_names:
            continue
        existing_columns = {
            str(column.get("name"))
            for column in inspector.get_columns(table_name)
            if column.get("name")
        }
        for column_name in sorted(required_columns):
            if column_name not in existing_columns:
                missing.append(f"{table_name}.{column_name}")
    return missing


def _repair_sqlite_schema_drift(sync_conn) -> list[str]:
    """Apply additive SQLite repairs for known legacy-schema drift."""
    if sync_conn.dialect.name != "sqlite":
        return []

    repair_sql: dict[str, str] = {
        "projects.beginner_mode": "ALTER TABLE projects ADD COLUMN beginner_mode BOOLEAN NOT NULL DEFAULT 0",
        "projects.active_domain_blueprint_version": "ALTER TABLE projects ADD COLUMN active_domain_blueprint_version INTEGER",
    }
    existing = set(_list_missing_columns(sync_conn, CRITICAL_COLUMN_REQUIREMENTS))
    applied: list[str] = []
    for key, statement in repair_sql.items():
        if key not in existing:
            continue
        sync_conn.execute(text(statement))
        applied.append(key)
    return applied


def _assert_alembic_head(sync_conn) -> None:
    """Ensure database schema is at Alembic head revision."""
    if not settings.DB_REQUIRE_ALEMBIC_HEAD:
        return

    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory
    except ImportError as e:
        raise RuntimeError(
            "Alembic is required for DB migration checks. Install it and run migrations."
        ) from e

    backend_dir = Path(__file__).resolve().parent.parent
    config_path = Path(settings.ALEMBIC_CONFIG_FILE)
    if not config_path.is_absolute():
        config_path = backend_dir / config_path

    if not config_path.exists():
        raise RuntimeError(
            f"Alembic config file not found: {config_path}. "
            "Set ALEMBIC_CONFIG_FILE or disable DB_REQUIRE_ALEMBIC_HEAD."
        )

    alembic_cfg = Config(str(config_path))
    script = ScriptDirectory.from_config(alembic_cfg)
    heads = set(script.get_heads())
    if not heads:
        return

    inspector = inspect(sync_conn)
    table_names = set(inspector.get_table_names())
    if "alembic_version" not in table_names:
        raise RuntimeError(
            "Database is missing alembic_version table. "
            "Run `alembic -c backend/alembic.ini upgrade head`."
        )

    rows = sync_conn.execute(text("SELECT version_num FROM alembic_version")).fetchall()
    current_versions = {row[0] for row in rows if row and row[0]}
    if current_versions != heads:
        raise RuntimeError(
            f"Database revision mismatch. Current: {sorted(current_versions)}; expected head: {sorted(heads)}. "
            "Run `alembic -c backend/alembic.ini upgrade head`."
        )


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields a database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize DB schema with explicit dev/prod behavior."""
    # Ensure all ORM models are imported so metadata includes every table.
    import app.models  # noqa: F401

    is_sqlite = settings.DATABASE_URL.startswith("sqlite")
    should_autocreate = settings.DB_AUTO_CREATE or (is_sqlite and settings.ALLOW_SQLITE_AUTOCREATE)

    if should_autocreate:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            if is_sqlite:
                await conn.run_sync(_repair_sqlite_schema_drift)
            missing_columns = await conn.run_sync(
                lambda sync_conn: _list_missing_columns(sync_conn, CRITICAL_COLUMN_REQUIREMENTS)
            )
            if missing_columns:
                if is_sqlite:
                    raise RuntimeError(
                        "SQLite schema is missing required columns after compatibility repair: "
                        f"{', '.join(sorted(missing_columns))}. "
                        "Recreate the SQLite database or run migrations manually."
                    )
                raise RuntimeError(
                    "Database schema is missing required columns after auto-create bootstrap: "
                    f"{', '.join(sorted(missing_columns))}. "
                    "Run Alembic migrations before starting the API."
                )
        return

    required_tables = set(BASE_REQUIRED_TABLES)
    if settings.AUTH_ENABLED or settings.AUDIT_LOG_ENABLED:
        required_tables.update(AUTH_REQUIRED_TABLES)

    async with engine.begin() as conn:
        existing_tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))

    missing = sorted(required_tables - existing_tables)
    if missing:
        raise RuntimeError(
            "Database schema is missing required tables: "
            f"{', '.join(missing)}. Run migrations before starting the API "
            "(or set DB_AUTO_CREATE=true for local/dev bootstrap only)."
        )

    async with engine.begin() as conn:
        missing_columns = await conn.run_sync(
            lambda sync_conn: _list_missing_columns(sync_conn, CRITICAL_COLUMN_REQUIREMENTS)
        )
    if missing_columns:
        raise RuntimeError(
            "Database schema is missing required columns: "
            f"{', '.join(sorted(missing_columns))}. "
            "Run Alembic migrations before starting the API."
        )

    async with engine.begin() as conn:
        await conn.run_sync(_assert_alembic_head)
