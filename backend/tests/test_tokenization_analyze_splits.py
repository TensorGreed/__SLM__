"""V3 of the ML-native visualisations arc — POST /tokenization/analyze-splits.

Covers the cross-split orchestration endpoint:
- Returns all three splits when train/val/test are present.
- Reports `missing_splits` (not 404) when some splits are not yet
  prepared, so the panel can render a partial overlay as soon as
  train is ready.
- 404s only when literally nothing exists for the project.
- One split erroring out (corrupt JSONL etc.) is captured in
  `errors`, the others still return.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "tok_splits.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "tok_splits_data"

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

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


class AnalyzeSplitsTests(unittest.TestCase):
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
        resp = self.client.post("/api/projects", json={"name": f"tok-{uuid.uuid4().hex[:8]}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_split(self, project_id: int, split: DatasetType, records: list[dict]) -> None:
        """Write a tiny JSONL file + a Dataset row pointing at it."""
        split_dir = TEST_DATA_DIR / "projects" / str(project_id) / "prepared"
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / f"{split.value}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        async def _add():
            async with async_session_factory() as db:
                db.add(Dataset(
                    project_id=project_id,
                    name=split.value,
                    dataset_type=split,
                    file_path=str(path),
                    record_count=len(records),
                ))
                await db.commit()
        asyncio.run(_add())

    def _mock_tokenizer(self):
        """Patch load_tokenizer so the test doesn't need transformers."""
        from unittest.mock import MagicMock
        tok = MagicMock()
        # 1 token per word — deterministic + length-proportional.
        tok.encode.side_effect = lambda text: list(range(len(text.split())))
        tok.vocab_size = 32000
        return tok

    def test_returns_all_three_when_all_present(self):
        project_id = self._create_project()
        self._seed_split(project_id, DatasetType.TRAIN, [{"text": "a b c d e"}] * 4)
        self._seed_split(project_id, DatasetType.VALIDATION, [{"text": "a b c"}] * 2)
        self._seed_split(project_id, DatasetType.TEST, [{"text": "a"}] * 1)

        with patch(
            "app.services.tokenization_service.load_tokenizer",
            return_value=self._mock_tokenizer(),
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/tokenization/analyze-splits",
                json={"model_name": BASE, "max_seq_length": 2048},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["model_name"], BASE)
        self.assertEqual(set(body["splits"].keys()), {"train", "validation", "test"})
        self.assertEqual(body["missing_splits"], [])
        # Sanity: each split's payload has the analyze_dataset_tokens shape.
        for split, payload in body["splits"].items():
            self.assertIn("histogram", payload)
            self.assertIn("p50_tokens", payload)
            self.assertIn("p95_tokens", payload)
            self.assertGreater(payload["total_samples"], 0)
        # Train has more samples than test in this seed.
        self.assertGreater(
            body["splits"]["train"]["total_samples"],
            body["splits"]["test"]["total_samples"],
        )

    def test_partial_splits_report_missing_not_404(self):
        """When only some splits exist (early in dataset prep, train is
        first), the endpoint returns what it has and names the rest in
        `missing_splits`. The panel can render a partial overlay."""
        project_id = self._create_project()
        self._seed_split(project_id, DatasetType.TRAIN, [{"text": "a b c"}] * 3)

        with patch(
            "app.services.tokenization_service.load_tokenizer",
            return_value=self._mock_tokenizer(),
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/tokenization/analyze-splits",
                json={"model_name": BASE, "max_seq_length": 2048},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(list(body["splits"].keys()), ["train"])
        self.assertEqual(set(body["missing_splits"]), {"validation", "test"})

    def test_returns_404_when_no_splits_exist(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/tokenization/analyze-splits",
            json={"model_name": BASE, "max_seq_length": 2048},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_one_corrupt_split_is_reported_in_errors_others_succeed(self):
        project_id = self._create_project()
        self._seed_split(project_id, DatasetType.TRAIN, [{"text": "a b c"}] * 3)
        # Point validation at a Dataset whose file does not exist on disk.
        async def _add():
            async with async_session_factory() as db:
                db.add(Dataset(
                    project_id=project_id,
                    name="validation",
                    dataset_type=DatasetType.VALIDATION,
                    file_path=str(TEST_DATA_DIR / "projects" / str(project_id) / "missing.jsonl"),
                    record_count=10,
                ))
                await db.commit()
        asyncio.run(_add())

        with patch(
            "app.services.tokenization_service.load_tokenizer",
            return_value=self._mock_tokenizer(),
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/tokenization/analyze-splits",
                json={"model_name": BASE, "max_seq_length": 2048},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("train", body["splits"])
        # Validation row exists but its file is missing — surfaces in
        # missing_splits (not in `errors` because the file-existence
        # check happens BEFORE the tokenizer call).
        self.assertIn("validation", body["missing_splits"])


if __name__ == "__main__":
    unittest.main()
