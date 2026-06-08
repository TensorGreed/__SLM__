"""Quality-Lift phase 2, slice 1 — slice_definitions schema + endpoints.

Pins (slice 1: storage + validator + CRUD only; runtime evaluation in
slice 2, gates in slice 3):

  Validator (pure function):
    * Empty/None payload normalizes to ``{"slices": []}``.
    * Closed set of ops is enforced.
    * slice_id regex rejects dots, spaces, uppercase, leading digits.
    * Duplicate slice_ids rejected.
    * Per-project cap (20 slices) + per-slice clause cap (8) enforced.
    * Numeric ops reject string values; list ops reject scalars;
      string ops reject empty strings; regex op rejects uncompilable
      patterns.
    * Boolean op (``exists``) defaults value to True when omitted.
    * Field path validates segment grammar + max depth.

  Project model + migration:
    * Round-trip the slice_definitions JSON column.
    * Existing projects with no slice_definitions read back as None
      (additive migration unchanged the dominant path).

  CRUD endpoint:
    * GET on a fresh project returns ``{"slices": []}`` (not 404, not
      null body) — editor needs a stable shape.
    * PUT writes + GET reads back the cleaned payload.
    * PUT with an invalid clause surfaces the precise service error
      verbatim (400 with the diagnostic message).
    * DELETE clears the column + subsequent GET returns the empty
      shape (idempotent).
    * Round-trip through ProjectResponse exposes the field.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.services.slice_definitions_service import (  # noqa: E402
    MAX_CLAUSES_PER_SLICE,
    MAX_SLICES_PER_PROJECT,
    SLICE_OPERATORS,
    SliceValidationError,
    validate_slice_definitions,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-slicedefs-{uuid.uuid4().hex[:8]}"
)


def setUpModule() -> None:
    settings.AUTH_ENABLED = False
    settings.DEBUG = False
    settings.DATA_DIR = TEST_DATA_DIR.resolve()
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    global _CLIENT_CM, CLIENT
    _CLIENT_CM = TestClient(app)
    CLIENT = _CLIENT_CM.__enter__()


def tearDownModule() -> None:
    _CLIENT_CM.__exit__(None, None, None)


def _create_project() -> int:
    resp = CLIENT.post(
        "/api/projects",
        json={"name": f"slicedefs-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


# ────────────────────────────────────────────────────────────────────────
# Pure validator
# ────────────────────────────────────────────────────────────────────────


class ValidateSliceDefinitionsTests(unittest.TestCase):

    def test_none_normalizes_to_empty_slices(self):
        # Critical for the additive contract: projects with no slicing
        # configured must round-trip through the validator without
        # erroring — the GET endpoint relies on this when projects
        # haven't been touched since the migration.
        self.assertEqual(validate_slice_definitions(None), {"slices": []})

    def test_empty_slices_list_is_valid(self):
        self.assertEqual(
            validate_slice_definitions({"slices": []}),
            {"slices": []},
        )

    def test_canonical_payload_roundtrips_cleanly(self):
        payload = {
            "slices": [
                {
                    "slice_id": "long_input",
                    "display_name": "Long inputs (>100 chars)",
                    "where": [
                        {"field": "input_length", "op": "gte", "value": 100},
                    ],
                },
                {
                    "slice_id": "hindi_long",
                    "display_name": "Long Hindi inputs",
                    "where": [
                        {"field": "language", "op": "eq", "value": "hi"},
                        {"field": "input_length", "op": "gte", "value": 50},
                    ],
                },
            ],
        }
        cleaned = validate_slice_definitions(payload)
        self.assertEqual(len(cleaned["slices"]), 2)
        first = cleaned["slices"][0]
        self.assertEqual(first["slice_id"], "long_input")
        # Numeric value is normalized to float for the runtime matcher.
        self.assertEqual(first["where"][0]["value"], 100.0)
        # Idempotency — re-running on the cleaned dict is a no-op.
        self.assertEqual(validate_slice_definitions(cleaned), cleaned)

    def test_payload_must_be_dict(self):
        with self.assertRaises(SliceValidationError):
            validate_slice_definitions([{"slice_id": "x"}])

    def test_slices_must_be_list(self):
        with self.assertRaises(SliceValidationError):
            validate_slice_definitions({"slices": {}})

    def test_per_project_cap_enforced(self):
        too_many = {
            "slices": [
                {
                    "slice_id": f"s{i}",
                    "display_name": "x",
                    "where": [{"field": "f", "op": "exists"}],
                }
                for i in range(MAX_SLICES_PER_PROJECT + 1)
            ],
        }
        with self.assertRaisesRegex(SliceValidationError, "too many slices"):
            validate_slice_definitions(too_many)

    def test_slice_id_regex(self):
        # Must start with letter, lowercase ASCII, no dots/spaces.
        bad_ids = ["UpperCase", "1leading_digit", "has space", "has.dot", "-dash", ""]
        for bad in bad_ids:
            with self.subTest(slice_id=bad):
                with self.assertRaises(SliceValidationError):
                    validate_slice_definitions({
                        "slices": [{
                            "slice_id": bad,
                            "display_name": "x",
                            "where": [{"field": "f", "op": "exists"}],
                        }],
                    })

    def test_duplicate_slice_ids_rejected(self):
        with self.assertRaisesRegex(SliceValidationError, "duplicate slice_id"):
            validate_slice_definitions({
                "slices": [
                    {"slice_id": "a", "display_name": "1",
                     "where": [{"field": "f", "op": "exists"}]},
                    {"slice_id": "a", "display_name": "2",
                     "where": [{"field": "f", "op": "exists"}]},
                ],
            })

    def test_clause_cap_per_slice(self):
        # AND-only grammar — OR is achieved via multiple slices. If the
        # user piles up too many ANDs, that's almost always a missed
        # restructuring opportunity, so we cap it.
        too_many = {
            "slices": [{
                "slice_id": "wide",
                "display_name": "x",
                "where": [
                    {"field": f"f{i}", "op": "exists"}
                    for i in range(MAX_CLAUSES_PER_SLICE + 1)
                ],
            }],
        }
        with self.assertRaisesRegex(SliceValidationError, "cap is"):
            validate_slice_definitions(too_many)

    def test_unknown_op_rejected(self):
        with self.assertRaisesRegex(SliceValidationError, "unknown op"):
            validate_slice_definitions({
                "slices": [{
                    "slice_id": "x", "display_name": "x",
                    "where": [{"field": "f", "op": "matches_kinda", "value": 1}],
                }],
            })

    def test_numeric_op_rejects_string(self):
        with self.assertRaisesRegex(SliceValidationError, "requires a numeric value"):
            validate_slice_definitions({
                "slices": [{
                    "slice_id": "x", "display_name": "x",
                    "where": [{"field": "f", "op": "gte", "value": "100"}],
                }],
            })

    def test_in_op_rejects_scalar(self):
        with self.assertRaisesRegex(SliceValidationError, "non-empty list"):
            validate_slice_definitions({
                "slices": [{
                    "slice_id": "x", "display_name": "x",
                    "where": [{"field": "f", "op": "in", "value": "synth"}],
                }],
            })

    def test_in_op_rejects_empty_list(self):
        with self.assertRaisesRegex(SliceValidationError, "non-empty list"):
            validate_slice_definitions({
                "slices": [{
                    "slice_id": "x", "display_name": "x",
                    "where": [{"field": "f", "op": "in", "value": []}],
                }],
            })

    def test_regex_op_rejects_invalid_pattern(self):
        with self.assertRaisesRegex(SliceValidationError, "regex .* is invalid"):
            validate_slice_definitions({
                "slices": [{
                    "slice_id": "x", "display_name": "x",
                    "where": [{"field": "f", "op": "regex", "value": "[unclosed"}],
                }],
            })

    def test_exists_op_defaults_to_true_when_value_omitted(self):
        cleaned = validate_slice_definitions({
            "slices": [{
                "slice_id": "has_field", "display_name": "x",
                "where": [{"field": "metadata.source", "op": "exists"}],
            }],
        })
        self.assertEqual(cleaned["slices"][0]["where"][0]["value"], True)

    def test_field_path_segment_grammar(self):
        # Dots are how we split into segments; each segment must be
        # an identifier — not "foo;bar" or "f-name" with dashes.
        with self.assertRaisesRegex(SliceValidationError, "not a valid identifier"):
            validate_slice_definitions({
                "slices": [{
                    "slice_id": "x", "display_name": "x",
                    "where": [{"field": "metadata..source", "op": "exists"}],
                }],
            })

    def test_field_path_max_depth(self):
        with self.assertRaisesRegex(SliceValidationError, "exceeds depth"):
            validate_slice_definitions({
                "slices": [{
                    "slice_id": "x", "display_name": "x",
                    "where": [{"field": "a.b.c.d.e", "op": "exists"}],
                }],
            })

    def test_operator_set_is_closed_and_complete(self):
        # Hardcoded sanity check on the public op tuple — if a future
        # change adds an op, both the service AND this contract must
        # update in lockstep. The slice 2 evaluator dispatch + the
        # slice 3 editor's op picker both read SLICE_OPERATORS.
        self.assertEqual(
            set(SLICE_OPERATORS),
            {"eq", "neq", "gt", "gte", "lt", "lte",
             "in", "not_in", "contains", "regex", "exists"},
        )


# ────────────────────────────────────────────────────────────────────────
# Project model column round-trip
# ────────────────────────────────────────────────────────────────────────


class ProjectSliceColumnTests(unittest.TestCase):

    def test_slice_definitions_round_trips_through_orm(self):
        pid = _create_project()
        payload = {
            "slices": [{
                "slice_id": "synthetic",
                "display_name": "Synth-sourced rows",
                "where": [{"field": "source", "op": "eq", "value": "synth"}],
            }],
        }

        async def _go() -> dict | None:
            async with async_session_factory() as session:
                project = (await session.execute(
                    select(Project).where(Project.id == pid)
                )).scalar_one()
                project.slice_definitions = payload
                await session.commit()
            async with async_session_factory() as session:
                fresh = (await session.execute(
                    select(Project).where(Project.id == pid)
                )).scalar_one()
                return fresh.slice_definitions

        fetched = asyncio.run(_go())
        self.assertEqual(fetched, payload)

    def test_legacy_project_round_trips_with_null_slices(self):
        # Additive migration contract: existing projects (no slicing
        # configured) must read back with slice_definitions == None,
        # NOT erroring on the column access.
        pid = _create_project()

        async def _go() -> dict | None:
            async with async_session_factory() as session:
                fresh = (await session.execute(
                    select(Project).where(Project.id == pid)
                )).scalar_one()
                return fresh.slice_definitions

        self.assertIsNone(asyncio.run(_go()))


# ────────────────────────────────────────────────────────────────────────
# CRUD endpoints
# ────────────────────────────────────────────────────────────────────────


class SliceDefinitionsEndpointTests(unittest.TestCase):

    def test_get_returns_empty_shape_for_unconfigured_project(self):
        # The editor renders against this response, so a missing
        # column must NOT return null/404. Stable empty shape.
        pid = _create_project()
        resp = CLIENT.get(f"/api/projects/{pid}/slice-definitions")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], pid)
        self.assertEqual(body["slice_definitions"], {"slices": []})

    def test_put_validates_then_persists(self):
        pid = _create_project()
        payload = {
            "slices": [{
                "slice_id": "long_input",
                "display_name": "Long",
                "where": [{"field": "input_length", "op": "gte", "value": 100}],
            }],
        }
        resp = CLIENT.put(
            f"/api/projects/{pid}/slice-definitions",
            json=payload,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        # Numeric value normalized through the validator on the way in.
        self.assertEqual(
            resp.json()["slice_definitions"]["slices"][0]["where"][0]["value"],
            100.0,
        )
        # GET reads back the same persisted shape.
        get_resp = CLIENT.get(f"/api/projects/{pid}/slice-definitions")
        self.assertEqual(
            get_resp.json()["slice_definitions"]["slices"][0]["slice_id"],
            "long_input",
        )

    def test_put_surfaces_service_error_verbatim(self):
        # The editor inline error needs the precise diagnostic, not
        # a generic "Bad Request". This is the contract: every
        # SliceValidationError message lands on the response body.
        pid = _create_project()
        resp = CLIENT.put(
            f"/api/projects/{pid}/slice-definitions",
            json={
                "slices": [{
                    "slice_id": "x", "display_name": "x",
                    "where": [{"field": "f", "op": "regex", "value": "[bad"}],
                }],
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("regex", resp.json()["detail"])
        self.assertIn("invalid", resp.json()["detail"])

    def test_put_on_nonexistent_project_is_404(self):
        resp = CLIENT.put(
            "/api/projects/999999/slice-definitions",
            json={"slices": []},
        )
        self.assertEqual(resp.status_code, 404)

    def test_delete_clears_column_idempotent(self):
        pid = _create_project()
        # Seed something to clear.
        CLIENT.put(
            f"/api/projects/{pid}/slice-definitions",
            json={"slices": [{
                "slice_id": "x", "display_name": "x",
                "where": [{"field": "f", "op": "exists"}],
            }]},
        )
        # First delete clears.
        resp1 = CLIENT.delete(f"/api/projects/{pid}/slice-definitions")
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp1.json()["slice_definitions"], {"slices": []})
        # Second delete is a no-op, NOT a 404 — slice 2 + slice 3 may
        # need to call this defensively without first probing state.
        resp2 = CLIENT.delete(f"/api/projects/{pid}/slice-definitions")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.json()["slice_definitions"], {"slices": []})

    def test_project_response_exposes_slice_definitions(self):
        # The field must surface on the base ProjectResponse so the
        # frontend ProjectStore can read it without a separate fetch.
        pid = _create_project()
        CLIENT.put(
            f"/api/projects/{pid}/slice-definitions",
            json={"slices": [{
                "slice_id": "synth", "display_name": "Synth",
                "where": [{"field": "source", "op": "eq", "value": "synth"}],
            }]},
        )
        resp = CLIENT.get(f"/api/projects/{pid}")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("slice_definitions", body)
        self.assertEqual(
            body["slice_definitions"]["slices"][0]["slice_id"],
            "synth",
        )


if __name__ == "__main__":
    unittest.main()
