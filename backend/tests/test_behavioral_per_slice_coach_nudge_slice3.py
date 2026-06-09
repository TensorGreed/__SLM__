"""Quality-Lift phase 6 slice 3 — Coach nudge:
``eval:behavioral-tests-without-per-slice-gates``.

Pins (slice 3: nudge fires/silences against project's slices +
behavioral tests + per-slice gates; ScorecardPanel rendering covered
by the vitest suite):

  * Silent when project has no slice_definitions configured.
  * Silent when pack has no behavioral_tests defined.
  * Silent when pack already has AT LEAST ONE per-slice behavioral
    gate.
  * Top-level behavioral gates DON'T satisfy the nudge — the user
    might have top-level gates but still need per-slice ones for
    slice-specific regressions.
  * Fires when slices defined + tests defined + no per-slice gates.
  * Body suggests a concrete ``behavioral.<test>.per_slice.<slice>.pass_rate``
    example so the user has a copy-paste starting point.
  * Severity is info (un-gated, not broken).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")


def _pack(
    *,
    behavioral_tests: list[dict] | None = None,
    gates: list[dict] | None = None,
):
    """Build a minimal classification pack with the requested
    behavioral_tests + gates."""
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
    if pack is None:
        async def _raise(*_args, **_kwargs):
            raise RuntimeError("pack lookup failed")
        return _raise
    async def _return(*_args, **_kwargs):
        return {"pack": pack, "active_pack_id": pack.get("active_pack_id")}
    return _return


class _FakeProject:
    """Stand-in for an SQLAlchemy Project row. Only the
    ``slice_definitions`` attribute matters here."""

    def __init__(self, slice_ids: list[str] | None):
        if slice_ids is None:
            self.slice_definitions = None
        else:
            self.slice_definitions = {
                "slices": [
                    {
                        "slice_id": sid,
                        "display_name": sid,
                        "where": [
                            {"field": "input_length", "op": "gte", "value": 10},
                        ],
                    }
                    for sid in slice_ids
                ],
            }


def _patched_db_execute(project: _FakeProject):
    """Mock db.execute returning a result whose scalar_one_or_none()
    yields the FakeProject. The nudge only calls execute once (the
    Project select), so we can wire this single-purpose mock."""
    class _Result:
        def scalar_one_or_none(self_inner):
            return project
    async def _execute(*_args, **_kwargs):
        return _Result()
    return _execute


def _run_nudge(*, slice_ids: list[str] | None, pack: dict | None):
    """Drive _behavioral_tests_without_per_slice_gates_nudge against a
    patched DB + pack resolver."""
    import asyncio

    from app.services.coach_service import (
        _behavioral_tests_without_per_slice_gates_nudge,
    )

    project = _FakeProject(slice_ids)
    db_mock = AsyncMock()
    db_mock.execute.side_effect = _patched_db_execute(project)

    async def _go():
        return await _behavioral_tests_without_per_slice_gates_nudge(db_mock, 1)

    with patch(
        "app.services.evaluation_pack_service.resolve_project_evaluation_pack",
        side_effect=_patched_resolve(pack),
    ):
        return asyncio.run(_go())


def _inv_test(test_id: str) -> dict:
    return {"test_id": test_id, "kind": "INV"}


class PerSliceGatesNudgeTests(unittest.TestCase):

    def test_silent_when_no_slices_configured(self):
        # Pack has tests; project has no slices → nothing to gate per-slice.
        self.assertIsNone(_run_nudge(
            slice_ids=None,
            pack=_pack(behavioral_tests=[_inv_test("typo")], gates=[]),
        ))

    def test_silent_when_no_behavioral_tests(self):
        # Project has slices; pack has no tests → no test×slice
        # cross-product to gate.
        self.assertIsNone(_run_nudge(
            slice_ids=["long_input"],
            pack=_pack(behavioral_tests=[], gates=[]),
        ))

    def test_top_level_behavioral_gate_does_not_satisfy(self):
        # Critical: a top-level behavioral gate covers the
        # un-gated-tests nudge, but per-slice regressions are still
        # invisible to the ship decision. This nudge MUST still fire.
        nudge = _run_nudge(
            slice_ids=["long_input"],
            pack=_pack(
                behavioral_tests=[_inv_test("typo")],
                gates=[{
                    "gate_id": "min_pass_overall",
                    "metric_id": "behavioral.typo.pass_rate",
                    "operator": "gte", "threshold": 0.85, "required": True,
                }],
            ),
        )
        self.assertIsNotNone(nudge)

    def test_silent_when_per_slice_gate_exists(self):
        # User has AT LEAST one per-slice behavioral gate → job done.
        self.assertIsNone(_run_nudge(
            slice_ids=["long_input"],
            pack=_pack(
                behavioral_tests=[_inv_test("typo")],
                gates=[{
                    "gate_id": "min_pass_long_input",
                    "metric_id": "behavioral.typo.per_slice.long_input.pass_rate",
                    "operator": "gte", "threshold": 0.85, "required": True,
                }],
            ),
        ))

    def test_fires_when_slices_and_tests_but_no_per_slice_gate(self):
        nudge = _run_nudge(
            slice_ids=["long_input", "short_input"],
            pack=_pack(
                behavioral_tests=[_inv_test("typo"), _inv_test("negation")],
                gates=[{
                    "gate_id": "min_f1", "metric_id": "f1",
                    "operator": "gte", "threshold": 0.8, "required": True,
                }],
            ),
        )
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["id"], "eval:behavioral-tests-without-per-slice-gates")
        self.assertEqual(nudge["severity"], "info")
        # Concrete suggested metric_id surfaces in body + context so
        # the user has a copy-paste starting point.
        self.assertIn("behavioral.typo.per_slice.long_input.pass_rate", nudge["body"])
        self.assertEqual(
            nudge["context"]["suggested_metric_id"],
            "behavioral.typo.per_slice.long_input.pass_rate",
        )
        # Full lists carried on context (not truncated like body might be).
        self.assertEqual(
            sorted(nudge["context"]["behavioral_test_ids"]),
            ["negation", "typo"],
        )
        self.assertEqual(
            sorted(nudge["context"]["slice_ids"]),
            ["long_input", "short_input"],
        )

    def test_silent_on_pack_resolution_failure(self):
        # Pack resolve raises → silent (never break eval-tab Coach).
        self.assertIsNone(_run_nudge(
            slice_ids=["long_input"],
            pack=None,
        ))


if __name__ == "__main__":
    unittest.main()
