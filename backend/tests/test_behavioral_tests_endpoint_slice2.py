"""Quality-Lift phase 7 slice 2 — behavioral_tests focused CRUD endpoint.

Pins (slice 2: focused endpoint that fronts the BehavioralTestsSection
editor without round-tripping the whole pack scaffold draft):

  * GET on a fresh project returns ``{behavioral_tests: []}`` (no
    404, no null) so the editor renders the empty state.
  * PUT validates via the phase 5 slice 1 schema validator — malformed
    payloads surface the colon-delimited error code verbatim as a
    400 detail so the editor can show an inline diagnostic.
  * PUT persists into the scaffolded pack JSON (same path the gate
    editor uses) so a subsequent GET reads back the saved list.
  * Empty list round-trips (validate_behavioral_tests normalises
    None → []).
  * Saving auto-materialises a classification task_spec if the
    recipe-resolved pack didn't surface one (sole-user UX: editor
    works on a fresh project without manual scaffold-save first).
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-bt-endpoint-{uuid.uuid4().hex[:8]}"
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
        json={"name": f"bt-endpoint-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _inv_test(test_id: str = "typo_invariance") -> dict:
    return {
        "test_id": test_id,
        "kind": "INV",
        "description": "Typos should not change predictions.",
        "seed_examples": [{"input": "This product is great.", "given_label": "positive"}],
        "perturbations": [{"kind": "typo", "intensity": 0.05}],
    }


class GetBehavioralTestsTests(unittest.TestCase):

    def test_returns_empty_list_on_fresh_project(self):
        pid = _create_project()
        resp = CLIENT.get(f"/api/projects/{pid}/behavioral-tests")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], pid)
        self.assertEqual(body["behavioral_tests"], [])

    def test_404_on_missing_project(self):
        resp = CLIENT.get("/api/projects/999999/behavioral-tests")
        self.assertEqual(resp.status_code, 404)


class PutBehavioralTestsTests(unittest.TestCase):

    def test_round_trips_a_valid_inv_test(self):
        pid = _create_project()
        resp = CLIENT.put(
            f"/api/projects/{pid}/behavioral-tests",
            json={"behavioral_tests": [_inv_test()]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        # The validator's cleaned shape comes back — same test_id +
        # kind, plus filled-in defaults (pass_rate_floor, etc.).
        saved = resp.json()["behavioral_tests"]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["test_id"], "typo_invariance")
        self.assertEqual(saved[0]["kind"], "INV")

        # GET reads back the same list.
        get_resp = CLIENT.get(f"/api/projects/{pid}/behavioral-tests")
        self.assertEqual(get_resp.json()["behavioral_tests"][0]["test_id"], "typo_invariance")

    def test_malformed_payload_surfaces_validator_code(self):
        # The editor relies on the colon-delimited code to highlight
        # the bad row — sanity-check the wire format.
        pid = _create_project()
        bad = _inv_test()
        bad["perturbations"] = [{"kind": "phase5b_paraphrase"}]
        resp = CLIENT.put(
            f"/api/projects/{pid}/behavioral-tests",
            json={"behavioral_tests": [bad]},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("unknown_perturbation_kind", resp.json()["detail"])

    def test_empty_list_roundtrips_as_clear(self):
        # User adds, then deletes everything → save an empty list →
        # GET returns empty list.
        pid = _create_project()
        CLIENT.put(
            f"/api/projects/{pid}/behavioral-tests",
            json={"behavioral_tests": [_inv_test()]},
        )
        resp = CLIENT.put(
            f"/api/projects/{pid}/behavioral-tests",
            json={"behavioral_tests": []},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["behavioral_tests"], [])

    def test_save_materialises_classification_task_spec_if_missing(self):
        # The recipe-resolved pack might not have a classification
        # task_spec yet (e.g. fresh project with a non-classification
        # recipe). The save endpoint should still succeed — the
        # behavioral runner is classification-only so we materialise
        # the spec on the fly.
        pid = _create_project()
        resp = CLIENT.put(
            f"/api/projects/{pid}/behavioral-tests",
            json={"behavioral_tests": [_inv_test()]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["task_profile"], "classification")


if __name__ == "__main__":
    unittest.main()
