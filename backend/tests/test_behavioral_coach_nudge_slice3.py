"""Quality-Lift phase 5 slice 3 — Coach nudge:
``eval:behavioral-tests-without-gates``.

Pins (slice 3: nudge fires/silences against project's active pack;
ScorecardPanel rendering covered by the vitest suite):

  * Silent when no pack resolves (uses lookup-failure tolerance).
  * Silent when pack has no behavioral_tests defined.
  * Silent when AT LEAST ONE behavioral gate already references a
    behavioral metric_id.
  * Fires when behavioral_tests are defined AND no gate references
    any of them.
  * Body lists up to 3 test_ids + "and N more" framing.
  * Severity is info (not warning) — un-gated tests aren't broken,
    just under-enforced.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")


def _pack(*, behavioral_tests: list[dict] | None = None, gates: list[dict] | None = None):
    """Build a minimal pack contract dict with a single classification
    task_spec carrying the requested behavioral_tests + gates."""
    spec: dict = {
        "task_profile": "classification",
        "required_metric_ids": ["f1"],
        "metric_schema": {},
        "gates": list(gates or []),
    }
    if behavioral_tests is not None:
        spec["behavioral_tests"] = behavioral_tests
    return {"task_specs": [spec], "active_pack_id": "test_pack"}


def _patched_resolve(pack: dict | None):
    """Return an AsyncMock that the nudge's resolve_project_evaluation_pack
    call sees. ``None`` simulates a lookup failure (silent path)."""
    if pack is None:
        async def _raise(*_args, **_kwargs):
            raise RuntimeError("pack lookup failed")
        return _raise
    async def _return(*_args, **_kwargs):
        return {"pack": pack, "active_pack_id": pack.get("active_pack_id")}
    return _return


def _run_nudge(pack: dict | None):
    """Drive ``_behavioral_tests_without_gates_nudge`` against a
    patched pack resolver."""
    import asyncio

    from app.services.coach_service import _behavioral_tests_without_gates_nudge

    async def _go():
        # ``db`` argument is unused by the nudge once the resolve is
        # patched — pass None to keep the test torchless.
        return await _behavioral_tests_without_gates_nudge(None, 1)

    with patch(
        "app.services.evaluation_pack_service.resolve_project_evaluation_pack",
        side_effect=_patched_resolve(pack),
    ):
        return asyncio.run(_go())


def _inv_test(test_id: str = "typo_invariance") -> dict:
    # Match the slice 1 cleaned shape so the nudge reads test_id from
    # the same path the validator emits.
    return {"test_id": test_id, "kind": "INV"}


class BehavioralTestsWithoutGatesNudgeTests(unittest.TestCase):

    def test_silent_when_no_pack(self):
        # resolve raises → lookup-failure path → silent. Mirrors the
        # per-class nudge's never-break-eval-coach contract.
        self.assertIsNone(_run_nudge(None))

    def test_silent_when_no_behavioral_tests_defined(self):
        nudge = _run_nudge(_pack(behavioral_tests=[], gates=[
            {"gate_id": "min_f1", "metric_id": "f1", "operator": "gte",
             "threshold": 0.8, "required": True},
        ]))
        self.assertIsNone(nudge)

    def test_silent_when_behavioral_gate_already_present(self):
        # The user has a pack with both a behavioral test AND a gate
        # referencing it. Job done — no nudge needed.
        nudge = _run_nudge(_pack(
            behavioral_tests=[_inv_test()],
            gates=[{
                "gate_id": "typo_invariance_gate",
                "metric_id": "behavioral.typo_invariance.pass_rate",
                "operator": "gte", "threshold": 0.85, "required": True,
            }],
        ))
        self.assertIsNone(nudge)

    def test_fires_when_tests_defined_but_ungated(self):
        nudge = _run_nudge(_pack(
            behavioral_tests=[_inv_test("typo_invariance"), _inv_test("negation_flips")],
            gates=[{
                "gate_id": "min_f1", "metric_id": "f1",
                "operator": "gte", "threshold": 0.8, "required": True,
            }],
        ))
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["id"], "eval:behavioral-tests-without-gates")
        self.assertEqual(nudge["severity"], "info")  # un-gated, not broken
        self.assertEqual(nudge["action"]["params"]["target"], "eval-pack-editor")
        # Body lists the test_ids so the user can correlate.
        self.assertIn("typo_invariance", nudge["body"])
        self.assertIn("negation_flips", nudge["body"])
        # Context surfaces the test ids verbatim.
        self.assertEqual(
            sorted(nudge["context"]["behavioral_test_ids"]),
            ["negation_flips", "typo_invariance"],
        )

    def test_body_truncates_when_many_tests(self):
        tests = [_inv_test(f"test_{i}") for i in range(7)]
        nudge = _run_nudge(_pack(behavioral_tests=tests, gates=[]))
        self.assertIsNotNone(nudge)
        # First three named + "and 4 more" framing.
        self.assertIn("test_0", nudge["body"])
        self.assertIn("test_1", nudge["body"])
        self.assertIn("test_2", nudge["body"])
        self.assertIn("4 more", nudge["body"])
        # The full list is on the context, not truncated.
        self.assertEqual(len(nudge["context"]["behavioral_test_ids"]), 7)


if __name__ == "__main__":
    unittest.main()
