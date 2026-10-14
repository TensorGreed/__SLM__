"""Shared test helper: let background Jobs finish before the next test.

Import from a test module under pytest (``tests/`` is on ``sys.path`` via
pytest's rootdir-relative import): ``from _job_wait import wait_for_jobs_idle``.

A test that starts a Job (``?async_job=true``) must not leave it running: the
runner keeps using the app's single shared DB connection after the test
returns, and a Job that outlives the file keeps the process from exiting.
Polls over HTTP (never via ``asyncio.run`` on the engine from the test thread
— that's a second event loop on the StaticPool connection).
"""

from __future__ import annotations

import time


def wait_for_jobs_idle(client, *, timeout_s: float = 15.0, poll_s: float = 0.1) -> None:
    """Block until no Job is QUEUED / RUNNING. Raises ``AssertionError`` naming
    the in-flight Jobs when ``timeout_s`` elapses (a stuck Job is a bug, not
    something to wait out silently)."""
    deadline = time.monotonic() + timeout_s
    in_flight: list[dict] = []
    while True:
        resp = client.get("/api/jobs/active", params={"include_recently_completed": "false"})
        if resp.status_code == 200:
            in_flight = list(resp.json().get("jobs") or [])
            if not in_flight:
                return
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_s)
    summary = ", ".join(f"#{j.get('id')} {j.get('kind')} {j.get('status')}" for j in in_flight)
    raise AssertionError(f"background Jobs still in flight after {timeout_s}s: {summary or '(unreadable)'}")
