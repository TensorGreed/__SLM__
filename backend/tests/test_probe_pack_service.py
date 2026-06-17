"""Coach-stage-2 phase 8 — platform-authored held-out probe packs.

Covers:
- The pure registry: applicable packs per task_profile, kind summaries,
  honest ready_not_run status, and the no-pack fallback.
- Probe content integrity (every probe well-formed; properties in the
  closed enum; robustness probes carry a base_input to compare against).
- The API: recipe → task_profile → pack resolution, no-recipe fallback,
  and 404 on a missing project.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "probe_pack.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "probe_pack_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.services.probe_pack_service import (  # noqa: E402
    PROBE_PACK_VERSION,
    ProbeProperty,
    get_probe_pack,
)

_VALID_PROPERTIES = set(ProbeProperty.__args__)  # type: ignore[attr-defined]


class ProbePackRegistryTests(unittest.TestCase):
    def test_classification_pack_is_applicable_and_well_formed(self):
        pack = get_probe_pack("classification")
        self.assertTrue(pack["applicable"])
        self.assertEqual(pack["version"], PROBE_PACK_VERSION)
        self.assertGreater(pack["probe_count"], 0)
        self.assertEqual(pack["probe_count"], len(pack["probes"]))
        # Honest status — assembled, not yet graded against the model.
        self.assertEqual(pack["status"], "ready_not_run")
        # kind_summary counts add up to probe_count.
        self.assertEqual(sum(pack["kind_summary"].values()), pack["probe_count"])

    def test_every_probe_is_well_formed_across_all_packs(self):
        for profile in (
            "classification",
            "instruction_sft",
            "rag_qa",
            "structured_extraction",
            "summarization",
        ):
            pack = get_probe_pack(profile)
            self.assertTrue(pack["applicable"], profile)
            ids = set()
            for p in pack["probes"]:
                for key in ("id", "probe_kind", "property", "input", "rationale"):
                    self.assertIn(key, p, f"{profile}:{p.get('id')}")
                    self.assertTrue(str(p[key]) != "" or key == "input")
                self.assertIn(p["property"], _VALID_PROPERTIES, p["id"])
                # ids unique within a pack.
                self.assertNotIn(p["id"], ids)
                ids.add(p["id"])
                # Stability probes must carry a base_input to compare to.
                if p["property"] == "prediction_stable_vs_base":
                    self.assertIn("base_input", p, p["id"])
                    self.assertNotEqual(p["base_input"], p["input"])

    def test_instruction_sft_has_safety_refusal_probes(self):
        pack = get_probe_pack("instruction_sft")
        kinds = {p["probe_kind"] for p in pack["probes"]}
        self.assertIn("safety_refusal", kinds)
        # The injection probe must exist — that's the canary.
        self.assertTrue(any("injection" in p["id"] for p in pack["probes"]))

    def test_unknown_profile_returns_inapplicable_pack(self):
        pack = get_probe_pack("totally-made-up-profile")
        self.assertFalse(pack["applicable"])
        self.assertEqual(pack["probe_count"], 0)
        self.assertEqual(pack["status"], "no_pack_for_profile")

    def test_none_profile_returns_inapplicable_pack(self):
        pack = get_probe_pack(None)
        self.assertFalse(pack["applicable"])


class ProbePackApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _create_project(self, recipe_id: str | None = None) -> int:
        resp = self.client.post(
            "/api/projects", json={"name": f"pp-{uuid.uuid4().hex[:8]}"}
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        if recipe_id:
            async def _set():
                async with async_session_factory() as db:
                    from app.models.project import Project
                    proj = await db.get(Project, pid)
                    proj.selected_recipe = {"recipe_id": recipe_id}
                    await db.commit()
            asyncio.run(_set())
        return pid

    def test_get_probe_pack_resolves_recipe_to_pack(self):
        pid = self._create_project(recipe_id="classification")
        resp = self.client.get(f"/api/projects/{pid}/probe-pack")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["applicable"])
        self.assertEqual(body["task_profile"], "classification")
        self.assertEqual(body["project_id"], pid)
        self.assertGreater(body["probe_count"], 0)

    def test_get_probe_pack_no_recipe_is_inapplicable(self):
        pid = self._create_project()
        body = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        self.assertFalse(body["applicable"])

    def test_get_probe_pack_missing_project_404s(self):
        resp = self.client.get("/api/projects/99887766/probe-pack")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
