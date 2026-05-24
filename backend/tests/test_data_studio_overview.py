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
from app.models.gold_set_annotation import (  # noqa: E402
    GoldSetReviewerQueue,
    GoldSetReviewerQueueStatus,
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)


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

    def test_gold_set_workbench_summarizes_rows_versions_and_queue(self):
        project_id = self._create_project("data-studio-gold")

        async def _seed_gold_workbench():
            async with async_session_factory() as db:
                gold = Dataset(
                    project_id=project_id,
                    name="Support Gold Dev",
                    dataset_type=DatasetType.GOLD_DEV,
                    record_count=3,
                )
                db.add(gold)
                await db.flush()
                version = GoldSetVersion(
                    gold_set_id=gold.id,
                    version=1,
                    status=GoldSetVersionStatus.DRAFT,
                    notes="review batch",
                )
                db.add(version)
                await db.flush()
                approved = GoldSetRow(
                    gold_set_id=gold.id,
                    version_id=version.id,
                    source_row_key="approved-1",
                    input={"question": "How do I reset my password?"},
                    expected={"answer": "Use the password reset flow."},
                    labels={"category": "account", "difficulty": "easy"},
                    status=GoldSetRowStatus.APPROVED,
                    rationale="Known support answer.",
                )
                pending = GoldSetRow(
                    gold_set_id=gold.id,
                    version_id=version.id,
                    source_row_key="pending-1",
                    input={"question": "Can I get a refund?"},
                    expected={"answer": "Open a billing ticket."},
                    labels={"category": "billing", "difficulty": "medium"},
                    status=GoldSetRowStatus.PENDING,
                    reviewer_id=7,
                )
                changes = GoldSetRow(
                    gold_set_id=gold.id,
                    version_id=version.id,
                    source_row_key="changes-1",
                    input={"question": "Where is my invoice?"},
                    expected={"answer": "Go to billing history."},
                    labels={"category": "billing"},
                    status=GoldSetRowStatus.CHANGES_REQUESTED,
                    reviewer_id=7,
                )
                db.add_all([approved, pending, changes])
                await db.flush()
                db.add_all([
                    GoldSetReviewerQueue(
                        gold_set_id=gold.id,
                        row_id=pending.id,
                        reviewer_id=7,
                        priority=2,
                        status=GoldSetReviewerQueueStatus.PENDING,
                    ),
                    GoldSetReviewerQueue(
                        gold_set_id=gold.id,
                        row_id=changes.id,
                        reviewer_id=7,
                        priority=1,
                        status=GoldSetReviewerQueueStatus.IN_PROGRESS,
                    ),
                ])
                await db.commit()

        asyncio.run(_seed_gold_workbench())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/gold-set")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "attention")
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["entry_point"]["target_tab"], "goldset")
        self.assertEqual(payload["totals"]["gold_set_count"], 1)
        self.assertEqual(payload["totals"]["example_count"], 3)
        self.assertEqual(payload["totals"]["trusted_examples"], 1)
        self.assertEqual(payload["totals"]["review_needed"], 2)
        dataset = payload["datasets"][0]
        self.assertEqual(dataset["validation_status"], "needs_review")
        self.assertEqual(dataset["row_status_counts"]["approved"], 1)
        self.assertEqual(dataset["queue_status_counts"]["pending"], 1)
        self.assertEqual(dataset["versions"]["draft_count"], 1)
        input_fields = {item["field"]: item["ratio"] for item in dataset["coverage"]["input_fields"]}
        expected_fields = {item["field"]: item["ratio"] for item in payload["coverage"]["expected_fields"]}
        label_fields = {item["field"]: item["ratio"] for item in payload["coverage"]["label_fields"]}
        self.assertEqual(input_fields["question"], 1.0)
        self.assertEqual(expected_fields["answer"], 1.0)
        self.assertIn("category", label_fields)
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("gold_rows_need_review", issue_ids)

    def test_gold_set_workbench_empty_routes_to_gold_set_panel(self):
        project_id = self._create_project("data-studio-gold-empty")

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/gold-set")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "empty")
        self.assertEqual(payload["datasets"], [])
        self.assertEqual(payload["validation"]["status"], "empty")
        self.assertEqual(payload["entry_point"]["target_tab"], "goldset")
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("no_gold_sets", issue_ids)

    def test_synthetic_playbook_center_summarizes_local_backend_and_review_queue(self):
        project_id = self._create_project("data-studio-synth-playbooks")
        recipe_resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        self.assertEqual(recipe_resp.status_code, 200, recipe_resp.text)

        async def _seed_synth_state():
            async with async_session_factory() as db:
                gold_dir = settings.DATA_DIR / "projects" / str(project_id) / "gold"
                gold_dir.mkdir(parents=True, exist_ok=True)
                gold_path = gold_dir / "gold_dev.jsonl"
                gold_rows = [
                    {"text": "Refund request", "label": "billing"},
                    {"text": "Password reset failed", "label": "account"},
                ]
                with gold_path.open("w", encoding="utf-8") as handle:
                    for row in gold_rows:
                        handle.write(json.dumps(row) + "\n")

                synth_dir = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
                synth_dir.mkdir(parents=True, exist_ok=True)
                synth_path = synth_dir / "synthetic.jsonl"
                synth_rows = [
                    {
                        "id": 1,
                        "text": "I need a renewal refund",
                        "label": "billing",
                        "synth_source": "playbook:classification:positives_paraphrase",
                        "synth_confidence": 0.91,
                        "review_status": "pending",
                    },
                    {
                        "id": 2,
                        "text": "My login code expired",
                        "label": "account",
                        "synth_source": "playbook:classification:positives_paraphrase",
                        "synth_confidence": 0.88,
                        "review_status": "pending",
                    },
                    {
                        "id": 3,
                        "text": "Billing portal is unavailable",
                        "label": "billing",
                        "synth_source": "playbook:classification:hard_negatives",
                        "synth_confidence": 0.82,
                        "review_status": "accepted",
                    },
                ]
                with synth_path.open("w", encoding="utf-8") as handle:
                    for row in synth_rows:
                        handle.write(json.dumps(row) + "\n")

                db.add_all([
                    Dataset(
                        project_id=project_id,
                        name="Gold Dev",
                        dataset_type=DatasetType.GOLD_DEV,
                        record_count=len(gold_rows),
                        file_path=str(gold_path),
                    ),
                    Dataset(
                        project_id=project_id,
                        name="Synthetic",
                        dataset_type=DatasetType.SYNTHETIC,
                        record_count=len(synth_rows),
                        file_path=str(synth_path),
                    ),
                ])
                await db.commit()

        class FakeOllamaBackend:
            name = "ollama"

            @classmethod
            def is_available(cls):
                return True

            def describe(self):
                return "ollama:llama3"

        class FakeTeacherBackend:
            name = "teacher"

            @classmethod
            def is_available(cls):
                return False

        asyncio.run(_seed_synth_state())

        with patch(
            "app.services.data_studio_service.BACKEND_REGISTRY",
            [FakeOllamaBackend, FakeTeacherBackend],
        ):
            resp = self.client.get(f"/api/projects/{project_id}/data-studio/synthetic-playbooks")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "attention")
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["recipe"]["id"], "classification")
        self.assertTrue(payload["recommended_backend"]["available"])
        self.assertFalse(payload["recommended_backend"]["paid_required"])
        self.assertGreaterEqual(payload["catalog"]["compatible_playbooks"], 1)
        self.assertIn("classification", payload["catalog"]["supported_recipes"])
        self.assertEqual(payload["review_queue"]["total_pending"], 2)
        self.assertEqual(payload["review_queue"]["total_accepted"], 1)
        prereq_status = {item["id"]: item["status"] for item in payload["prerequisites"]}
        self.assertEqual(prereq_status["recipe"], "met")
        self.assertEqual(prereq_status["compatible_playbooks"], "met")
        self.assertEqual(prereq_status["gold_examples"], "met")
        self.assertEqual(prereq_status["local_ollama"], "met")
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("synthetic_rows_pending_review", issue_ids)
        self.assertEqual(payload["entry_point"]["target_tab"], "synthetic")

    def test_synthetic_playbook_center_no_recipe_keeps_free_local_default(self):
        project_id = self._create_project("data-studio-synth-empty")

        class OfflineOllamaBackend:
            name = "ollama"

            @classmethod
            def is_available(cls):
                return False

        with patch(
            "app.services.data_studio_service.BACKEND_REGISTRY",
            [OfflineOllamaBackend],
        ):
            resp = self.client.get(f"/api/projects/{project_id}/data-studio/synthetic-playbooks")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "attention")
        self.assertIsNone(payload["recipe"])
        self.assertFalse(payload["recommended_backend"]["available"])
        self.assertTrue(payload["recommended_backend"]["local_default"])
        self.assertFalse(payload["recommended_backend"]["paid_required"])
        self.assertGreaterEqual(payload["catalog"]["total_playbooks"], 1)
        prereq_status = {item["id"]: item["status"] for item in payload["prerequisites"]}
        self.assertEqual(prereq_status["recipe"], "missing")
        self.assertEqual(prereq_status["local_ollama"], "attention")
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("synthetic_recipe_missing", issue_ids)
        self.assertIn("synthetic_ollama_unavailable", issue_ids)

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
