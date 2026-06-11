"""Phase-3 additions to the training-config gap scanner.

Covers the two text-sampling signals — truncation rate vs max_seq_length
and tokenizer OOV with byte-fallback awareness — without standing up a
real tokenizer (the latter is monkey-patched to keep tests offline).
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "tcg_phase3.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "tcg_phase3_data"

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


def _signal_by_id(body: dict, signal_id: str) -> dict | None:
    for group in body["groups"]:
        for sig in group["signals"]:
            if sig["id"] == signal_id:
                return sig
    return None


class TrainingConfigGapPhase3Tests(unittest.TestCase):
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

    # ── helpers ─────────────────────────────────────────────────────

    def _seed_project(
        self,
        *,
        recipe_id: str,
        row_texts: list[str],
        base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    ) -> int:
        """Create a project, drop a JSONL of training rows, register a
        CLEANED Dataset pointing at it. row_texts[i] becomes one row
        of shape {"text": ...} so the gap scanner's row-to-text picks
        it up.
        """
        resp = self.client.post(
            "/api/projects",
            json={"name": f"tcg-p3-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])

        ds_dir = TEST_DATA_DIR / f"project-{pid}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = ds_dir / "cleaned.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for t in row_texts:
                f.write(json.dumps({"text": t}) + "\n")

        async def _set():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, pid)
                proj.selected_recipe = {"recipe_id": recipe_id}
                proj.base_model_name = base_model
                db.add(Dataset(
                    project_id=pid,
                    name="cleaned",
                    dataset_type=DatasetType.CLEANED,
                    file_path=str(jsonl_path),
                    record_count=len(row_texts),
                ))
                await db.commit()
        asyncio.run(_set())
        return pid

    # ── Truncation signal tests ────────────────────────────────────

    def test_truncation_signal_ok_when_no_rows_to_sample(self):
        # No JSONL on disk → sample returns empty → signal degrades to ok.
        resp = self.client.post(
            "/api/projects",
            json={"name": f"tcg-p3-{uuid.uuid4().hex[:8]}"},
        )
        pid = int(resp.json()["id"])

        async def _set():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, pid)
                proj.selected_recipe = {
                    "recipe_id": "recipe.classification.sentiment"
                }
                proj.base_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
                # Cleaned dataset row exists but file_path is empty →
                # sampler skips it.
                db.add(Dataset(
                    project_id=pid,
                    name="cleaned",
                    dataset_type=DatasetType.CLEANED,
                    file_path="",
                    record_count=20,
                ))
                await db.commit()
        asyncio.run(_set())

        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.max_seq_truncation_risk")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertEqual(sig["context"]["sample_size"], 0)

    def test_truncation_signal_warns_above_threshold(self):
        # max_seq_length default is 2048 → chars_per_token ≈ 4 →
        # truncation fires for rows > 8192 chars. Build a sample
        # where 50% of rows exceed that.
        short = "a" * 100   # well under
        long = "a" * 12000  # well over
        rows = [short] * 5 + [long] * 5  # 50% truncation rate
        pid = self._seed_project(
            recipe_id="recipe.classification.sentiment",
            row_texts=rows,
        )
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.max_seq_truncation_risk")
        self.assertIsNotNone(sig)
        assert sig is not None
        # 50% > 25% block threshold.
        self.assertEqual(sig["severity"], "block")
        ctx = sig["context"]
        self.assertEqual(ctx["sample_size"], 10)
        self.assertEqual(ctx["truncated_count"], 5)
        # Recommended max_seq_length is > the current 2048.
        self.assertGreater(ctx["recommended_max_seq_length"], 2048)
        # Action points at training-config.
        self.assertEqual(
            sig["suggested_action"]["target"], "training-config"
        )

    def test_truncation_signal_ok_below_threshold(self):
        # All rows short → 0% truncation → ok.
        rows = ["short row " * 20 for _ in range(10)]  # ~200 chars each
        pid = self._seed_project(
            recipe_id="recipe.classification.sentiment",
            row_texts=rows,
        )
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.max_seq_truncation_risk")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertEqual(sig["context"]["truncated_count"], 0)

    # ── Tokenizer OOV signal tests ──────────────────────────────────

    def test_tokenizer_oov_degrades_ok_when_load_fails(self):
        # Default in the test env: the SmolLM2 tokenizer is unlikely to
        # be cached on disk. The signal catches the load failure and
        # emits ok with skipped_reason.
        rows = ["hello world"] * 5
        pid = self._seed_project(
            recipe_id="recipe.classification.sentiment",
            row_texts=rows,
            base_model="not-a-real-model/intentionally-broken",
        )
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.tokenizer_oov_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertIn("not available locally", sig["headline"])
        self.assertIn("skipped_reason", sig["context"])

    def test_tokenizer_oov_byte_fallback_returns_ok(self):
        # Modern byte-BPE tokenizers (SmolLM2, Qwen2.5, etc.) have
        # unk_token=None. Monkey-patch AutoTokenizer.from_pretrained
        # to return a stub with unk_token=None and confirm the signal
        # flags byte_fallback=True without scanning text.
        from unittest.mock import patch

        class _StubByteFallbackTokenizer:
            unk_token = None
            unk_token_id = None

            def encode(self, _text, add_special_tokens=False):  # noqa: ARG002
                return [1, 2, 3]

        rows = ["hello world"] * 5
        pid = self._seed_project(
            recipe_id="recipe.classification.sentiment",
            row_texts=rows,
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_StubByteFallbackTokenizer(),
        ):
            body = self.client.get(
                f"/api/projects/{pid}/training-config-gaps"
            ).json()
        sig = _signal_by_id(body, "training_config.tokenizer_oov_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertTrue(sig["context"]["byte_fallback"])

    def test_tokenizer_oov_warns_when_unk_rate_exceeds_threshold(self):
        # WordPiece-style tokenizer with an explicit unk. Stub returns
        # 50% unk_ids → warn (block, actually — 50% >= 15%).
        from unittest.mock import patch

        class _StubUnkTokenizer:
            unk_token = "[UNK]"
            unk_token_id = 0

            def encode(self, _text, add_special_tokens=False):  # noqa: ARG002
                # 4 tokens, 2 of which are unk.
                return [0, 5, 0, 7]

        rows = ["whatever"] * 5
        pid = self._seed_project(
            recipe_id="recipe.classification.sentiment",
            row_texts=rows,
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_StubUnkTokenizer(),
        ):
            body = self.client.get(
                f"/api/projects/{pid}/training-config-gaps"
            ).json()
        sig = _signal_by_id(body, "training_config.tokenizer_oov_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        # 50% unk rate >> 15% block threshold.
        self.assertEqual(sig["severity"], "block")
        self.assertEqual(sig["context"]["unk_count"], 10)
        self.assertEqual(sig["context"]["total_tokens"], 20)
        # Action points at base-model picker (vocab mismatch).
        self.assertEqual(
            sig["suggested_action"]["target"], "training-base-model-picker"
        )

    def test_tokenizer_oov_ok_when_unk_rate_below_threshold(self):
        # 5% unk threshold; stub returns 0 unks → ok with the
        # "covers your sample" headline.
        from unittest.mock import patch

        class _StubCoveringTokenizer:
            unk_token = "[UNK]"
            unk_token_id = 0

            def encode(self, _text, add_special_tokens=False):  # noqa: ARG002
                return [1, 2, 3, 4, 5]  # zero unks

        rows = ["covered text"] * 5
        pid = self._seed_project(
            recipe_id="recipe.classification.sentiment",
            row_texts=rows,
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_StubCoveringTokenizer(),
        ):
            body = self.client.get(
                f"/api/projects/{pid}/training-config-gaps"
            ).json()
        sig = _signal_by_id(body, "training_config.tokenizer_oov_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertEqual(sig["context"]["unk_count"], 0)
