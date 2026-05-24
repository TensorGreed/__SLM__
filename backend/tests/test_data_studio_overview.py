"""Data Studio overview endpoint tests."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(tempfile.gettempdir()) / f"brewslm-data-studio-{uuid.uuid4().hex[:8]}.db"
TEST_DATA_DIR = Path(tempfile.gettempdir()) / f"brewslm-data-studio-{uuid.uuid4().hex[:8]}"

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
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument  # noqa: E402


class DataStudioOverviewEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.DEBUG = False
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

    def test_empty_project_is_blocked_with_next_action(self):
        project_id = self._create_project("data-studio-empty")

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/overview")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "blocked")
        self.assertEqual(payload["row_counts"]["trainable"], 0)
        self.assertEqual(payload["primary_action"]["target_tab"], "data")
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("missing_recipe", issue_ids)
        self.assertIn("no_trainable_rows", issue_ids)

    def test_counts_pending_synthetic_and_prepared_warning(self):
        project_id = self._create_project("data-studio-counts")
        recipe_resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        self.assertEqual(recipe_resp.status_code, 200, recipe_resp.text)

        async def _seed_data():
            async with async_session_factory() as db:
                synth_dir = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
                synth_dir.mkdir(parents=True, exist_ok=True)
                synth_path = synth_dir / "synthetic.jsonl"
                rows = [
                    {
                        "id": 1,
                        "text": "accepted synthetic row",
                        "label": "billing",
                        "synth_source": "playbook:classification:class_balance_fill",
                        "synth_confidence": 0.91,
                        "review_status": "accepted",
                    },
                    {
                        "id": 2,
                        "text": "pending synthetic row",
                        "label": "billing",
                        "synth_source": "playbook:classification:class_balance_fill",
                        "synth_confidence": 0.72,
                        "review_status": "pending",
                    },
                ]
                with synth_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                db.add(
                    Dataset(
                        project_id=project_id,
                        name="Gold",
                        dataset_type=DatasetType.GOLD_DEV,
                        record_count=25,
                    )
                )
                db.add(
                    Dataset(
                        project_id=project_id,
                        name="Synthetic",
                        dataset_type=DatasetType.SYNTHETIC,
                        record_count=2,
                        file_path=str(synth_path),
                    )
                )
                await db.commit()

        asyncio.run(_seed_data())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/overview")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "needs_work")
        self.assertEqual(payload["recipe"]["id"], "classification")
        self.assertEqual(payload["row_counts"]["gold"], 25)
        self.assertEqual(payload["row_counts"]["synthetic_accepted"], 1)
        self.assertEqual(payload["row_counts"]["synthetic_pending"], 1)
        self.assertEqual(payload["row_counts"]["trainable"], 26)
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("pending_synthetic_rows", issue_ids)
        self.assertIn("dataset_not_prepared", issue_ids)

    def test_sources_summary_groups_datasets_and_recent_documents(self):
        project_id = self._create_project("data-studio-sources")

        async def _seed_sources():
            async with async_session_factory() as db:
                raw_ds = Dataset(
                    project_id=project_id,
                    name="Uploaded CSV",
                    dataset_type=DatasetType.RAW,
                    record_count=42,
                    file_path=str(settings.DATA_DIR / "raw.csv"),
                )
                cleaned_ds = Dataset(
                    project_id=project_id,
                    name="Cleaned Rows",
                    dataset_type=DatasetType.CLEANED,
                    record_count=40,
                    file_path=str(settings.DATA_DIR / "cleaned.jsonl"),
                )
                db.add_all([raw_ds, cleaned_ds])
                await db.flush()
                db.add_all([
                    RawDocument(
                        dataset_id=raw_ds.id,
                        filename="tickets.csv",
                        file_type="csv",
                        file_path=str(settings.DATA_DIR / "tickets.csv"),
                        file_size_bytes=2048,
                        source="upload",
                        status=DocumentStatus.ACCEPTED,
                        chunk_count=3,
                    ),
                    RawDocument(
                        dataset_id=raw_ds.id,
                        filename="bad.jsonl",
                        file_type="jsonl",
                        file_path=str(settings.DATA_DIR / "bad.jsonl"),
                        file_size_bytes=512,
                        source="upload",
                        status=DocumentStatus.ERROR,
                    ),
                ])
                await db.commit()

        asyncio.run(_seed_sources())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/sources")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "attention")
        self.assertEqual(payload["totals"]["dataset_count"], 2)
        self.assertEqual(payload["totals"]["document_count"], 2)
        self.assertEqual(payload["totals"]["row_count"], 82)
        self.assertEqual(payload["totals"]["accepted_documents"], 1)
        self.assertEqual(payload["totals"]["error_documents"], 1)
        groups = {group["dataset_type"]: group for group in payload["dataset_groups"]}
        self.assertEqual(groups["raw"]["row_count"], 42)
        self.assertEqual(groups["cleaned"]["row_count"], 40)
        filenames = {doc["filename"] for doc in payload["recent_documents"]}
        self.assertIn("tickets.csv", filenames)
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("source_errors", issue_ids)

    def test_mapping_preview_uses_recipe_adapter_for_raw_rows(self):
        project_id = self._create_project("data-studio-mapping")
        recipe_resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        self.assertEqual(recipe_resp.status_code, 200, recipe_resp.text)

        async def _seed_mapping_source():
            async with async_session_factory() as db:
                raw_dir = settings.DATA_DIR / "projects" / str(project_id) / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "tickets.jsonl"
                rows = [
                    {"text": "Refund requested after renewal", "label": "billing"},
                    {"text": "Login code never arrived", "label": "access"},
                ]
                with raw_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

                raw_ds = Dataset(
                    project_id=project_id,
                    name="Ticket exports",
                    dataset_type=DatasetType.RAW,
                    record_count=2,
                    file_path=str(raw_path),
                )
                db.add(raw_ds)
                await db.flush()
                db.add(
                    RawDocument(
                        dataset_id=raw_ds.id,
                        filename="tickets.jsonl",
                        file_type="jsonl",
                        file_path=str(raw_path),
                        file_size_bytes=raw_path.stat().st_size,
                        source="upload",
                        status=DocumentStatus.ACCEPTED,
                        chunk_count=2,
                    )
                )
                await db.commit()

        asyncio.run(_seed_mapping_source())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/mapping-preview")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "ready")
        self.assertEqual(payload["recipe"]["id"], "classification")
        self.assertEqual(payload["effective_mapping"]["source"], "recipe")
        self.assertEqual(payload["effective_mapping"]["adapter_id"], "classification-label")
        self.assertEqual(payload["effective_mapping"]["task_profile"], "classification")
        self.assertEqual(payload["summary"]["sampled_records"], 2)
        self.assertEqual(payload["summary"]["mapped_records"], 2)
        self.assertTrue(payload["summary"]["contract_pass"])
        fields = {
            item["field"]: item["ratio"]
            for item in payload["summary"]["required_field_coverage"]
        }
        self.assertEqual(fields["label"], 1.0)
        self.assertEqual(payload["preview_rows"][0]["mapped"]["label"], "billing")

    def test_mapping_preview_empty_without_previewable_rows(self):
        project_id = self._create_project("data-studio-mapping-empty")

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/mapping-preview")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "empty")
        self.assertIsNone(payload["source"])
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("missing_recipe", issue_ids)
        self.assertIn("no_mapping_source", issue_ids)

    def test_domain_detection_uses_source_evidence_and_runtime(self):
        project_id = self._create_project("data-studio-domain")

        async def _seed_support_source():
            async with async_session_factory() as db:
                raw_dir = settings.DATA_DIR / "projects" / str(project_id) / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "support_faq.jsonl"
                rows = [
                    {
                        "question": "How do I reset my password when the login code never arrives?",
                        "answer": "Open account security, request a new password reset email, and contact support if the code expires.",
                    },
                    {
                        "question": "Can I get a refund after my subscription renewed?",
                        "answer": "Submit a billing ticket with the renewal invoice and an agent will review the refund request.",
                    },
                ]
                with raw_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

                raw_ds = Dataset(
                    project_id=project_id,
                    name="Support FAQ",
                    dataset_type=DatasetType.RAW,
                    record_count=2,
                    file_path=str(raw_path),
                )
                db.add(raw_ds)
                await db.flush()
                db.add(
                    RawDocument(
                        dataset_id=raw_ds.id,
                        filename="support_faq.jsonl",
                        file_type="jsonl",
                        file_path=str(raw_path),
                        file_size_bytes=raw_path.stat().st_size,
                        source="upload",
                        status=DocumentStatus.ACCEPTED,
                        chunk_count=2,
                    )
                )
                await db.commit()

        asyncio.run(_seed_support_source())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/domain-detection")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["detected_domain"]["id"], "support_faq")
        self.assertGreaterEqual(payload["detected_domain"]["confidence"], 0.65)
        self.assertEqual(payload["source"]["dataset_type"], "raw")
        self.assertEqual(payload["source"]["sampled_records"], 2)
        self.assertEqual(payload["applied"]["profile_id"], "generic-domain-v1")
        self.assertIn(payload["applied"]["profile_source"], {"project", "platform_default"})
        self.assertGreaterEqual(len(payload["evidence"]), 2)
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("domain_candidate_not_applied", issue_ids)

    def test_domain_detection_empty_project_uses_generic_runtime(self):
        project_id = self._create_project("data-studio-domain-empty")

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/domain-detection")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "attention")
        self.assertEqual(payload["detected_domain"]["id"], "generic_domain")
        self.assertIsNone(payload["source"])
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("domain_needs_source_evidence", issue_ids)
        self.assertIn("low_domain_confidence", issue_ids)

    def test_llm_assist_mapping_uses_ollama_default_without_applying(self):
        project_id = self._create_project("data-studio-assist")
        assistant_payload = {
            "summary": "Category appears to be the label column.",
            "suggestions": [
                {
                    "id": "map-label-category",
                    "type": "mapping",
                    "title": "Map label to category",
                    "confidence": 0.91,
                    "rationale": "The values are short repeated classes.",
                    "evidence": ["category has label-like values"],
                    "suggested_field_mapping": {"label": "category"},
                    "target_tab": "dataprep",
                    "requires_user_confirmation": True,
                }
            ],
        }

        with patch(
            "app.services.data_studio_service.call_teacher_model",
            new_callable=AsyncMock,
        ) as teacher_mock:
            teacher_mock.return_value = {
                "content": json.dumps(assistant_payload),
                "model": "llama3",
                "tokens_used": 42,
            }
            resp = self.client.post(
                f"/api/projects/{project_id}/data-studio/assist",
                json={
                    "focus": "mapping",
                    "provider": "ollama",
                    "model_name": "llama3",
                },
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["focus"], "mapping")
        self.assertEqual(payload["provider"]["provider"], "ollama")
        self.assertEqual(payload["provider"]["api_url"], "http://localhost:11434/v1/chat/completions")
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["source_of_truth"], "deterministic_data_studio_checks")
        self.assertEqual(payload["suggestions"][0]["suggested_field_mapping"], {"label": "category"})
        self.assertTrue(payload["suggestions"][0]["requires_user_confirmation"])
        teacher_mock.assert_awaited_once()
        self.assertEqual(
            teacher_mock.await_args.kwargs["api_url"],
            "http://localhost:11434/v1/chat/completions",
        )

    def test_llm_assist_invalid_response_is_non_mutating(self):
        project_id = self._create_project("data-studio-assist-invalid")

        with patch(
            "app.services.data_studio_service.call_teacher_model",
            new_callable=AsyncMock,
        ) as teacher_mock:
            teacher_mock.return_value = {
                "content": "not json",
                "model": "local-model",
                "tokens_used": 7,
            }
            resp = self.client.post(
                f"/api/projects/{project_id}/data-studio/assist",
                json={
                    "focus": "domain",
                    "provider": "ollama",
                    "model_name": "local-model",
                },
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["status"], "invalid_response")
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["suggestions"], [])
        self.assertEqual(payload["source_of_truth"], "deterministic_data_studio_checks")
        teacher_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
