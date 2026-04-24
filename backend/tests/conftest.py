"""Pytest session-scoped safety pin — force every test to write to /tmp.

This exists because a prior full-suite run wiped the developer's local
``data/slm_platform.db``. Root cause: not every test module overrode
``DATABASE_URL`` at import time, so whichever module was imported first
got to bind the SQLAlchemy engine. When that module was one of the
few without an override, the engine pointed at the real production DB
(``sqlite+aiosqlite:///./data/slm_platform.db``). Later test cleanup
then happily dropped-and-recreated every table, destroying user data.

This conftest takes control before any other test module can:

1.  At module import time — which pytest guarantees happens before any
    test module body runs — we redirect ``DATABASE_URL`` and ``DATA_DIR``
    onto a per-PID scratch directory under ``/tmp``.
2.  We immediately trigger ``from app.config import settings`` so the
    settings singleton is locked to those paths. Any later test module
    that writes to ``os.environ`` is too late; the engine is already bound.
3.  ``pytest_configure`` asserts after collection that the resolved
    ``settings.DATABASE_URL`` does not contain ``data/slm_platform.db``.
    If somehow it does, the run aborts before a single test executes.

Side effect: test modules that used to silently share the production DB
now run against a clean, isolated /tmp DB. They must create their own
fixtures; the previous accidental state-sharing is gone — intentionally.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


_SESSION_ROOT = Path(tempfile.gettempdir()) / f"brewslm-pytest-{os.getpid()}"
_SESSION_DB = _SESSION_ROOT / "session.db"
_SESSION_DATA_DIR = _SESSION_ROOT / "data"

_SESSION_ROOT.mkdir(parents=True, exist_ok=True)
_SESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pin the session before any test module can override.
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_SESSION_DB.as_posix()}"
os.environ["DATA_DIR"] = _SESSION_DATA_DIR.as_posix()
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

# Import settings NOW so the singleton freezes onto /tmp. Any later
# ``os.environ`` write in a test module happens after this point and can
# no longer reach the engine binding.
from app.config import settings as _settings  # noqa: E402

_PRODUCTION_DB_MARKERS = (
    "data/slm_platform.db",
    "\\data\\slm_platform.db",
)


def _looks_like_production_db(url: str) -> bool:
    lowered = url.replace("\\", "/").lower()
    return any(marker in lowered for marker in _PRODUCTION_DB_MARKERS)


def pytest_configure(config: pytest.Config) -> None:
    """Abort the run if the safety pin failed and the engine is bound to prod."""
    url = str(_settings.DATABASE_URL)
    if _looks_like_production_db(url):
        raise pytest.UsageError(
            "Test suite aborted before any test ran: "
            f"settings.DATABASE_URL resolved to the production path ({url!r}). "
            "The conftest.py safety pin was bypassed; investigate import order."
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Best-effort cleanup of the per-PID scratch directory."""
    try:
        shutil.rmtree(_SESSION_ROOT, ignore_errors=True)
    except Exception:
        pass
