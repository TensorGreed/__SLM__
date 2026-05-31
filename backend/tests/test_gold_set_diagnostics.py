"""V4 of the ML-native visualisations arc — gold-set class balance
+ class-similarity heatmap diagnostics.

Covers:
- Empty/no-classification gold sets return the empty payload (not 404).
- Class-balance counts are correct + sorted descending + share fields
  + entropy.
- Similarity matrix uses class-balance order for rows/cols, marks
  same-bucket diagonal (skips self-pairs), and surfaces None for
  classes with too few rows.
- API endpoint validates `sample_per_class` query param.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "gold_diag.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "gold_diag_data"

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
from app.models.dataset import Dataset, DatasetType  # noqa: E402


def _seed_gold_dev(project_id: int, rows: list[dict]) -> None:
    """Write a JSONL gold-dev file + register a Dataset row pointing at it."""
    import json
    project_dir = TEST_DATA_DIR / "projects" / str(project_id) / "gold"
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "gold_dev.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    async def _add():
        async with async_session_factory() as db:
            db.add(Dataset(
                project_id=project_id,
                name="gold_dev",
                dataset_type=DatasetType.GOLD_DEV,
                file_path=str(path),
                record_count=len(rows),
            ))
            await db.commit()
    asyncio.run(_add())


class GoldSetDiagnosticsTests(unittest.TestCase):
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

    def _create_project(self) -> int:
        resp = self.client.post("/api/projects", json={"name": f"gd-{uuid.uuid4().hex[:8]}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_no_gold_rows_returns_empty_classification_payload(self):
        """A project with no gold set returns classification_eligible=False
        and empty balance/similarity. UI renders the "n/a" empty state."""
        project_id = self._create_project()
        resp = self.client.get(f"/api/projects/{project_id}/gold/diagnostics")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["total_rows"], 0)
        self.assertFalse(body["classification_eligible"])
        self.assertEqual(body["class_balance"]["labels"], [])
        self.assertEqual(body["similarity"]["labels"], [])

    def test_class_balance_sorted_descending_by_count(self):
        project_id = self._create_project()
        # Imbalanced classification gold set: 6 spam + 3 ham + 1 promo.
        rows = (
            [{"text": f"earn money now {i}", "label": "spam"} for i in range(6)]
            + [{"text": f"please review {i}", "label": "ham"} for i in range(3)]
            + [{"text": "limited offer x", "label": "promo"}]
        )
        _seed_gold_dev(project_id, rows)

        body = self.client.get(f"/api/projects/{project_id}/gold/diagnostics").json()
        self.assertTrue(body["classification_eligible"])
        self.assertEqual(body["total_rows"], 10)
        bal = body["class_balance"]
        # Sorted by count desc, tie-broken by label asc.
        self.assertEqual([entry["label"] for entry in bal["labels"]], ["spam", "ham", "promo"])
        spam = next(e for e in bal["labels"] if e["label"] == "spam")
        self.assertEqual(spam["count"], 6)
        self.assertAlmostEqual(spam["share"], 0.6, places=4)
        # Entropy > 0 for a mixed-class set.
        self.assertGreater(bal["entropy_nats"], 0.0)

    def test_similarity_matrix_uses_balance_order(self):
        """Matrix rows/cols match the class-balance order so the heatmap
        reads in descending-popularity order top-to-bottom."""
        project_id = self._create_project()
        rows = (
            [{"text": f"alpha sample {i}", "label": "alpha"} for i in range(4)]
            + [{"text": f"beta sample {i}", "label": "beta"} for i in range(2)]
        )
        _seed_gold_dev(project_id, rows)

        body = self.client.get(f"/api/projects/{project_id}/gold/diagnostics").json()
        sim = body["similarity"]
        # Alpha has more rows → first in balance order → first row/col of matrix.
        self.assertEqual(sim["labels"], ["alpha", "beta"])
        self.assertEqual(len(sim["matrix"]), 2)
        self.assertEqual(len(sim["matrix"][0]), 2)
        # Diagonal cells are intra-class similarity (real numbers, not None).
        for i in (0, 1):
            self.assertIsNotNone(sim["matrix"][i][i])
            self.assertIsInstance(sim["matrix"][i][i], float)
        # Matrix is symmetric (cross-class similarity).
        self.assertAlmostEqual(sim["matrix"][0][1], sim["matrix"][1][0], places=4)

    def test_intra_class_redundancy_is_high_when_rows_are_near_duplicates(self):
        """Near-identical rows in one class → diagonal Jaccard ≈ 1.0.
        Different rows in another class → diagonal Jaccard < 0.5.
        This is the "is my gold set diverse?" diagnostic in action."""
        project_id = self._create_project()
        rows = (
            # Class "boring" — every row is the same words.
            [{"text": "renew subscription click here today", "label": "boring"} for _ in range(4)]
            # Class "diverse" — every row uses different words.
            + [
                {"text": "alpha beta gamma", "label": "diverse"},
                {"text": "epsilon zeta eta", "label": "diverse"},
                {"text": "iota kappa lambda", "label": "diverse"},
                {"text": "rho sigma tau", "label": "diverse"},
            ]
        )
        _seed_gold_dev(project_id, rows)

        body = self.client.get(f"/api/projects/{project_id}/gold/diagnostics").json()
        sim = body["similarity"]
        # Both classes tied at 4 rows; tie broken by label asc → boring, diverse.
        self.assertEqual(sim["labels"], ["boring", "diverse"])
        boring_intra = sim["matrix"][0][0]
        diverse_intra = sim["matrix"][1][1]
        # Boring class is fully duplicated → ~1.0; diverse is fully
        # non-overlapping → 0.0. Inter-class is also low.
        self.assertGreater(boring_intra, 0.9)
        self.assertLess(diverse_intra, 0.1)

    def test_insufficient_rows_per_class_surfaces_none(self):
        """A class with only 1 row can't have its intra-class similarity
        scored (no pairs to average). The cell returns None, the label
        is flagged in insufficient_labels."""
        project_id = self._create_project()
        rows = (
            [{"text": f"plenty {i}", "label": "plenty"} for i in range(3)]
            + [{"text": "lonely row", "label": "lonely"}]  # only 1 row
        )
        _seed_gold_dev(project_id, rows)
        body = self.client.get(f"/api/projects/{project_id}/gold/diagnostics").json()
        sim = body["similarity"]
        self.assertIn("lonely", sim["insufficient_labels"])
        # Find the lonely row in the matrix — its diagonal is None.
        labels = sim["labels"]
        lonely_idx = labels.index("lonely")
        self.assertIsNone(sim["matrix"][lonely_idx][lonely_idx])
        # Cross-class cells with lonely are also None (no rows to pair).
        for j, lbl in enumerate(labels):
            if lbl == "lonely":
                continue
            self.assertIsNone(sim["matrix"][lonely_idx][j])

    def test_endpoint_rejects_invalid_sample_per_class(self):
        project_id = self._create_project()
        resp = self.client.get(
            f"/api/projects/{project_id}/gold/diagnostics?sample_per_class=1"
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        resp = self.client.get(
            f"/api/projects/{project_id}/gold/diagnostics?sample_per_class=999"
        )
        self.assertEqual(resp.status_code, 400, resp.text)


if __name__ == "__main__":
    unittest.main()
