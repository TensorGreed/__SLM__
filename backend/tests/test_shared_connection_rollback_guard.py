"""The SQLite StaticPool rollback guard (``app.database``).

Every session shares ONE DBAPI connection under StaticPool, so a read-only
session ending (``Session.close()`` / the pool's reset-on-checkin, both a
ROLLBACK) used to discard another session's flushed-but-uncommitted writes.
That was the mechanism behind the intermittent async-job test failures
("Project N not found", the distillation RunEvent missing) and the dev-server
Jobs stuck QUEUED / RUNNING. These tests interleave two sessions
deterministically in ONE event loop (a single ``asyncio.run``; no TestClient,
so no cross-loop use of the engine).
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory, engine, init_db
from app.models.job import Job, JobStatus

if "data/slm_platform.db" in settings.DATABASE_URL.replace("\\", "/"):
    # Only ever run against conftest.py's per-PID /tmp DB (pytest), never the
    # dev DB a bare ``python -m unittest`` would bind.
    raise unittest.SkipTest("refusing to write to the dev database; run under pytest")


def _job(title: str) -> Job:
    return Job(kind="rollback_guard_test", title=title, status=JobStatus.QUEUED,
               queued_at=datetime.now(timezone.utc))


async def _titles() -> set[str]:
    async with async_session_factory() as db:
        rows = await db.execute(select(Job.title).where(Job.kind == "rollback_guard_test"))
        return set(rows.scalars())


class SharedConnectionRollbackGuardTests(unittest.TestCase):
    def test_engine_is_the_shared_connection_config(self):
        # The guard only matters (and is only installed) on the StaticPool engine.
        self.assertEqual(type(engine.sync_engine.pool).__name__, "StaticPool")

    def test_reader_closing_does_not_discard_a_concurrent_writers_flush(self):
        async def _go() -> set[str]:
            await init_db()
            writer = async_session_factory()
            writer.add(_job("writer-row"))
            await writer.flush()  # INSERT issued, not committed

            # Another session reads and closes in the middle of the writer's
            # transaction — the request-session / runner interleaving.
            async with async_session_factory() as reader:
                await reader.execute(select(Job.id).limit(1))

            await writer.commit()
            await writer.close()
            return await _titles()

        self.assertIn("writer-row", asyncio.run(_go()))

    def test_a_session_that_wrote_still_rolls_back_its_own_work(self):
        async def _go() -> set[str]:
            await init_db()
            async with async_session_factory() as db:
                db.add(_job("rolled-back-row"))
                await db.flush()
                await db.rollback()
            return await _titles()

        self.assertNotIn("rolled-back-row", asyncio.run(_go()))

    def test_closing_a_session_with_uncommitted_writes_discards_them(self):
        async def _go() -> set[str]:
            await init_db()
            async with async_session_factory() as db:
                db.add(_job("closed-without-commit"))
                await db.flush()
            # close() without commit → its own write is rolled back as before.
            return await _titles()

        self.assertNotIn("closed-without-commit", asyncio.run(_go()))


if __name__ == "__main__":
    unittest.main()
