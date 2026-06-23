"""The dedup re-split ("Re-split with dedup" leakage remediation) must
reproduce the ACTIVE prepared version's split config — not silently fall
back to form/profile defaults.

The risk this pins: the split form defaults ``stratify_by`` / ``disjoint_by``
to empty. The leakage button POSTs ``dedup_rows=true`` with neither set. If
the endpoint took those at face value, a project whose active prepared
version was a *stratified* classification split would be re-split UNIFORMLY
random — silently dropping the per-class proportion guarantee while
"fixing" leakage. The endpoint instead inherits the active manifest's
config (read from ``prepared/manifest.json``, the same source the trainer
reads) for every field the caller didn't explicitly provide.

Covers:
  * A dedup re-split with no stratify_by inherits the active version's
    ``stratify_by`` (the load-bearing case) → manifest reports it stratified
    and lists it in ``dedup_inherited_config``.
  * Ratios / seed are inherited too.
  * An EXPLICIT field on the dedup request still wins over the active
    manifest (no surprise override of a deliberate choice).
  * A normal split (dedup_rows omitted) does NOT inherit — unchanged
    behaviour for the common path.
"""

from __future__ import annotations

import asyncio
import json
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
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import Dataset, DatasetType  # noqa: E402


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-dedup-inherit-{uuid.uuid4().hex[:8]}"
)

# A small classification corpus with 2 labels and an exact duplicate to
# give the dedup something to drop.
_ROWS = (
    [{"text": f"billing question number {i} about a charge", "label": "billing"} for i in range(6)]
    + [{"text": f"technical issue number {i} app crashes", "label": "technical"} for i in range(6)]
    + [{"text": "billing question number 0 about a charge", "label": "billing"}]  # exact dup
)


def _seed_cleaned_with_file(project_id: int, rows: list[dict]) -> None:
    """Seed a CLEANED dataset whose file_path holds the rows, so
    combine_datasets has something to split."""
    async def _go():
        prep = TEST_DATA_DIR / "projects" / str(project_id)
        prep.mkdir(parents=True, exist_ok=True)
        fpath = prep / "cleaned.jsonl"
        with open(fpath, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        async with async_session_factory() as session:
            ds = Dataset(
                project_id=project_id,
                name="cleaned-fixture",
                dataset_type=DatasetType.CLEANED,
                record_count=len(rows),
                file_path=str(fpath),
            )
            session.add(ds)
            await session.commit()

    asyncio.run(_go())


def _write_active_manifest(project_id: int, manifest: dict) -> None:
    """Write prepared/manifest.json — the active prepared-version config the
    endpoint inherits from on a dedup re-split."""
    prep = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    prep.mkdir(parents=True, exist_ok=True)
    (prep / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _read_active_manifest(project_id: int) -> dict:
    path = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


class DedupResplitInheritsConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_project(self, label: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{label}-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _split(self, pid: int, body: dict) -> dict:
        resp = self.client.post(f"/api/projects/{pid}/dataset/split", json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    # ── The load-bearing case (end-to-end against the REAL manifest) ──

    def test_dedup_resplit_inherits_active_stratify_by(self):
        pid = self._create_project("dedup-inherit-stratify")
        _seed_cleaned_with_file(pid, _ROWS)

        # A REAL stratified split — writes the genuine on-disk manifest the
        # dedup re-split will inherit from. (Hand-authoring a manifest with
        # resolved_split_config would mask the on-disk shape, which stores
        # top-level seed/chat_template + ratios:{train,val,test} instead.)
        first = self._split(
            pid,
            {
                "stratify_by": "label",
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "seed": 7,
            },
        )
        self.assertEqual(first["stratify_by"], "label")

        # The persisted manifest now carries the canonical resolved_split_config
        # AND keeps the legacy top-level seed + ratios for existing consumers.
        on_disk = _read_active_manifest(pid)
        self.assertEqual(on_disk["seed"], 7)
        self.assertAlmostEqual(on_disk["ratios"]["train"], 0.7)
        self.assertEqual(on_disk["resolved_split_config"]["seed"], 7)
        self.assertAlmostEqual(on_disk["resolved_split_config"]["train_ratio"], 0.7)

        # The leakage button sends ONLY dedup_rows — no stratify, no ratios.
        manifest = self._split(pid, {"dedup_rows": True})

        # Inherited the stratify guarantee → manifest is stratified on label.
        self.assertEqual(manifest["stratify_by"], "label")
        self.assertIsNotNone(manifest["stratification_report"])
        self.assertEqual(
            manifest["stratification_report"]["stratify_field"], "label"
        )
        # Inherited ratios + seed from the active version, not form defaults.
        resolved = manifest["resolved_split_config"]
        self.assertAlmostEqual(resolved["train_ratio"], 0.7)
        self.assertEqual(resolved["seed"], 7)
        # Honesty field lists what was inherited.
        inherited = manifest["dedup_inherited_config"]
        self.assertIn("stratify_by", inherited)
        self.assertIn("seed", inherited)
        self.assertIn("train_ratio", inherited)
        # Dedup actually ran (the exact dup got dropped).
        self.assertTrue(manifest["dedup_requested"])
        self.assertGreaterEqual(manifest["dedup_report"]["dropped_count"], 1)

    # ── Explicit request still wins ─────────────────────────────────

    def test_explicit_field_overrides_active_manifest(self):
        pid = self._create_project("dedup-explicit-wins")
        _seed_cleaned_with_file(pid, _ROWS)
        self._split(
            pid,
            {"stratify_by": "label", "train_ratio": 0.7, "val_ratio": 0.15,
             "test_ratio": 0.15, "seed": 7},
        )
        # Caller explicitly overrides the seed on the dedup re-split.
        manifest = self._split(pid, {"dedup_rows": True, "seed": 123})
        self.assertEqual(manifest["resolved_split_config"]["seed"], 123)
        self.assertNotIn("seed", manifest["dedup_inherited_config"])
        # stratify still inherited (not explicitly provided).
        self.assertEqual(manifest["stratify_by"], "label")

    # ── Normal split path unchanged ─────────────────────────────────

    def test_normal_split_does_not_inherit(self):
        pid = self._create_project("normal-no-inherit")
        _seed_cleaned_with_file(pid, _ROWS)
        # An active stratified version exists on disk…
        self._split(pid, {"stratify_by": "label", "seed": 7})
        # …but a fresh split with no dedup_rows must NOT inherit it.
        manifest = self._split(pid, {})
        self.assertIsNone(manifest.get("stratify_by"))
        self.assertIsNone(manifest.get("stratification_report"))
        self.assertEqual(manifest.get("dedup_inherited_config"), [])
        self.assertFalse(manifest.get("dedup_requested"))


class ActiveManifestSplitConfigTests(unittest.TestCase):
    """The normalizer must read BOTH the on-disk manifest shape and the
    API-response shape — the regression that let ratios/seed inheritance
    silently no-op against a real persisted manifest."""

    def test_reads_on_disk_shape(self):
        from app.api.dataset import _active_manifest_split_config
        out = _active_manifest_split_config(
            {"seed": 7, "chat_template": "llama3",
             "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}}
        )
        self.assertEqual(out["seed"], 7)
        self.assertAlmostEqual(out["train_ratio"], 0.7)
        self.assertAlmostEqual(out["val_ratio"], 0.15)
        self.assertEqual(out["chat_template"], "llama3")

    def test_reads_api_response_shape_as_fallback(self):
        from app.api.dataset import _active_manifest_split_config
        out = _active_manifest_split_config(
            {"resolved_split_config": {
                "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2,
                "seed": 99, "chat_template": "chatml"}}
        )
        self.assertAlmostEqual(out["train_ratio"], 0.6)
        self.assertEqual(out["seed"], 99)
        self.assertEqual(out["chat_template"], "chatml")

    def test_canonical_key_wins_when_both_present(self):
        # split_dataset now persists BOTH resolved_split_config and the legacy
        # ratios/seed. The canonical key must win.
        from app.api.dataset import _active_manifest_split_config
        out = _active_manifest_split_config(
            {
                "seed": 1, "chat_template": "llama3",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "resolved_split_config": {
                    "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15,
                    "seed": 7, "chat_template": "chatml"},
            }
        )
        self.assertAlmostEqual(out["train_ratio"], 0.7)
        self.assertEqual(out["seed"], 7)
        self.assertEqual(out["chat_template"], "chatml")

    def test_empty_manifest_yields_empty(self):
        from app.api.dataset import _active_manifest_split_config
        self.assertEqual(_active_manifest_split_config({}), {})


if __name__ == "__main__":
    unittest.main()
