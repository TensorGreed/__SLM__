"""Data Studio overview endpoint tests."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
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
from app.models.dataset import Dataset, DatasetType, DatasetVersion, DocumentStatus, RawDocument  # noqa: E402
from app.models.domain_pack import DomainPack, DomainPackStatus  # noqa: E402
from app.models.domain_profile import DomainProfile, DomainProfileStatus  # noqa: E402
from app.models.gold_set_annotation import (  # noqa: E402
    GoldSetReviewerQueue,
    GoldSetReviewerQueueStatus,
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)
from app.models.label_job import LabelJob, LabelRow  # noqa: E402
from app.models.project import Project  # noqa: E402


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
        templates = payload["mapping_templates"]
        self.assertTrue(templates["read_only"])
        self.assertGreaterEqual(templates["template_count"], 1)
        self.assertEqual(
            templates["entry_points"][0]["target_tab"],
            "dataprep",
        )
        self.assertTrue(templates["entry_points"][0]["requires_confirmation"])
        detected_fields = {item["field"] for item in templates["detected_fields"]}
        self.assertIn("text", detected_fields)
        self.assertIn("label", detected_fields)
        recommended = next(item for item in templates["templates"] if item["recommended"])
        self.assertIn(recommended["source"], {"recipe", "adapter", "auto_fix"})
        self.assertEqual(recommended["apply_action"]["target_tab"], "dataprep")
        self.assertTrue(recommended["apply_action"]["requires_confirmation"])
        recommended_fields = {item["canonical_field"]: item for item in recommended["fields"]}
        self.assertIn("text", recommended_fields)
        self.assertIn("label", recommended_fields)
        self.assertIn(recommended_fields["text"]["status"], {"available", "applied"})

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
        self.assertTrue(payload["mapping_templates"]["read_only"])
        self.assertEqual(payload["mapping_templates"]["detected_fields"], [])
        if payload["mapping_templates"]["templates"]:
            self.assertEqual(
                payload["mapping_templates"]["templates"][0]["apply_action"]["target_tab"],
                "dataprep",
            )
            self.assertTrue(
                payload["mapping_templates"]["templates"][0]["apply_action"]["requires_confirmation"]
            )

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
        setup = payload["domain_setup"]
        self.assertTrue(setup["available"])
        self.assertTrue(setup["recommended"])
        self.assertEqual(setup["detected_domain_id"], "support_faq")
        self.assertEqual(setup["profile_id"], "support-faq-profile-v1")
        self.assertEqual(setup["pack_id"], "support-faq-pack-v1")
        self.assertEqual(setup["profile_contract"]["status"], "draft")
        self.assertEqual(setup["pack_contract"]["default_profile_id"], "support-faq-profile-v1")

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
        self.assertIsNone(payload["domain_setup"])

    def test_quality_safety_scan_flags_sensitive_duplicates_leakage_and_reviews(self):
        project_id = self._create_project("data-studio-quality-safety")
        recipe_resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        self.assertEqual(recipe_resp.status_code, 200, recipe_resp.text)

        async def _seed_quality_source():
            async with async_session_factory() as db:
                raw_dir = settings.DATA_DIR / "projects" / str(project_id) / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "policy_rows.jsonl"
                duplicate_text = (
                    "Which leave policy covers caregiver emergency eligibility? "
                    "The employee handbook policy mentions compliance exceptions. "
                    "Contact jane@example.com and use card 4111 1111 1111 1111."
                )
                rows = [
                    {"text": duplicate_text, "label": "covered"},
                    {"text": duplicate_text, "label": "covered"},
                    {
                        "text": "Policy exception question with SSN 123-45-6789 and cvv: 123.",
                    },
                    {"text": "N/A", "label": "unknown"},
                ]
                with raw_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

                raw_ds = Dataset(
                    project_id=project_id,
                    name="Policy Rows",
                    dataset_type=DatasetType.RAW,
                    record_count=len(rows),
                    file_path=str(raw_path),
                )
                db.add(raw_ds)
                await db.flush()
                db.add(
                    RawDocument(
                        dataset_id=raw_ds.id,
                        filename="policy_rows.jsonl",
                        file_type="jsonl",
                        file_path=str(raw_path),
                        file_size_bytes=raw_path.stat().st_size,
                        source="upload",
                        status=DocumentStatus.ACCEPTED,
                        chunk_count=len(rows),
                        quality_score=0.4,
                    )
                )

                synth_dir = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
                synth_dir.mkdir(parents=True, exist_ok=True)
                synth_path = synth_dir / "synthetic.jsonl"
                with synth_path.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "id": 1,
                        "text": "Pending synthetic policy answer",
                        "label": "covered",
                        "synth_source": "playbook:policy:test",
                        "review_status": "pending",
                    }) + "\n")
                db.add(
                    Dataset(
                        project_id=project_id,
                        name="Synthetic",
                        dataset_type=DatasetType.SYNTHETIC,
                        record_count=1,
                        file_path=str(synth_path),
                    )
                )

                prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
                prepared_dir.mkdir(parents=True, exist_ok=True)
                train_path = prepared_dir / "train.jsonl"
                val_path = prepared_dir / "validation.jsonl"
                test_path = prepared_dir / "test.jsonl"
                leaked_row = {"text": "Policy leakage row", "label": "covered"}
                train_path.write_text(json.dumps(leaked_row) + "\n", encoding="utf-8")
                val_path.write_text(json.dumps(leaked_row) + "\n", encoding="utf-8")
                test_path.write_text(json.dumps({"text": "Unique test row", "label": "covered"}) + "\n", encoding="utf-8")
                db.add_all([
                    Dataset(
                        project_id=project_id,
                        name="Train",
                        dataset_type=DatasetType.TRAIN,
                        record_count=1,
                        file_path=str(train_path),
                    ),
                    Dataset(
                        project_id=project_id,
                        name="Validation",
                        dataset_type=DatasetType.VALIDATION,
                        record_count=1,
                        file_path=str(val_path),
                    ),
                    Dataset(
                        project_id=project_id,
                        name="Test",
                        dataset_type=DatasetType.TEST,
                        record_count=1,
                        file_path=str(test_path),
                    ),
                ])
                await db.commit()

        asyncio.run(_seed_quality_source())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/quality-safety")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "blocked")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["assist"]["default_provider"], "ollama")
        self.assertEqual(payload["assist"]["purpose"], "explanations_only")
        self.assertEqual(payload["domain"]["id"], "policy_qa")
        self.assertGreaterEqual(payload["summary"]["scanned_rows"], 4)
        self.assertGreaterEqual(payload["summary"]["pii_pci_signal_count"], 3)
        self.assertGreaterEqual(payload["summary"]["duplicate_signal_count"], 1)
        self.assertGreaterEqual(payload["summary"]["leakage_overlap_count"], 1)
        self.assertGreaterEqual(payload["summary"]["pending_review_count"], 1)

        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("pii_pci_sensitive_values", issue_ids)
        self.assertIn("duplicate_or_near_duplicate_rows", issue_ids)
        self.assertIn("train_validation_test_leakage", issue_ids)
        self.assertIn("required_fields_missing", issue_ids)
        self.assertIn("synthetic_review_contamination", issue_ids)
        self.assertIn("domain_policy_context_missing", issue_ids)

        owner_labels = {item["label"] for item in payload["findings_by_owner"]}
        self.assertIn("Source Ingestion", owner_labels)
        self.assertIn("Data Prep", owner_labels)
        status_counts = {item["status"]: item["count"] for item in payload["findings_by_status"]}
        self.assertGreaterEqual(status_counts["blocked"], 2)
        source_labels = {item["label"] for item in payload["findings_by_source"]}
        self.assertIn("policy_rows.jsonl", source_labels)

        checks_by_id = {check["id"]: check for check in payload["checks"]}
        pii_drilldown = checks_by_id["pii_pci_sensitive_values"]["drilldown"]
        self.assertTrue(pii_drilldown["read_only"])
        self.assertTrue(pii_drilldown["redacted"])
        self.assertEqual(pii_drilldown["action"]["target_tab"], "data")
        self.assertEqual(pii_drilldown["action"]["label"], "Inspect sources")
        self.assertTrue(pii_drilldown["action"]["requires_confirmation"])
        self.assertGreaterEqual(pii_drilldown["total_affected"], 3)
        self.assertTrue(pii_drilldown["source_counts"])
        self.assertTrue(pii_drilldown["rows"])
        redacted_preview = json.dumps(pii_drilldown["rows"])
        self.assertIn("[EMAIL]", redacted_preview)
        self.assertIn("[CARD]", redacted_preview)
        self.assertNotIn("jane@example.com", redacted_preview)
        self.assertNotIn("4111 1111 1111 1111", redacted_preview)

        leakage_drilldown = checks_by_id["train_validation_test_leakage"]["drilldown"]
        leakage_sources = {item["source"] for item in leakage_drilldown["source_counts"]}
        self.assertIn("train split", leakage_sources)
        self.assertIn("validation split", leakage_sources)
        self.assertEqual(leakage_drilldown["action"]["target_tab"], "dataprep")

    def test_quality_safety_applies_domain_authored_profile_and_pack_checks(self):
        project_id = self._create_project("data-studio-domain-authored-quality")

        async def _seed_domain_authored_quality():
            async with async_session_factory() as db:
                profile_id = f"policy-authored-profile-{project_id}"
                pack_id = f"policy-authored-pack-{project_id}"
                profile_contract = {
                    "$schema": "slm.domain-profile/v1",
                    "profile_id": profile_id,
                    "version": "1.0.0",
                    "display_name": "Policy Authored Quality Profile",
                    "description": "Test profile with domain-authored quality checks.",
                    "owner": "workspace",
                    "status": "active",
                    "tasks": [
                        {
                            "task_id": "policy-qa",
                            "output_mode": "text",
                            "required_fields": ["question", "answer", "context"],
                            "optional_fields": ["policy_section", "citation"],
                        }
                    ],
                    "canonical_schema": {
                        "required": ["input_text", "target_text", "context"],
                        "aliases": {
                            "input_text": ["question", "prompt"],
                            "target_text": ["answer", "response"],
                            "context": ["context", "policy_text"],
                            "citation": ["policy_section", "citation"],
                        },
                    },
                    "normalization": {
                        "trim_whitespace": True,
                        "drop_empty_records": True,
                        "dedupe": {"enabled": True},
                        "pii_redaction": {"enabled": True, "policy": "mask_training_values"},
                    },
                    "data_quality": {
                        "min_records": 10,
                        "max_null_ratio": 0.05,
                        "max_duplicate_ratio": 0.2,
                        "required_coverage": {"context": 1.0},
                        "forbidden_phrases": ["always guarantee"],
                        "citation_required": True,
                        "citation_fields": ["policy_section", "citation"],
                        "recommended_checks": [
                            {
                                "id": "policy-disallow-guarantee",
                                "type": "regex",
                                "label": "No guarantee language",
                                "pattern": "always guarantee",
                                "mode": "forbid",
                                "severity": "warning",
                                "target_tab": "data",
                                "action_label": "Inspect sources",
                            }
                        ],
                    },
                    "dataset_split": {
                        "train": 0.8,
                        "val": 0.1,
                        "test": 0.1,
                        "stratify_by": ["policy_section"],
                        "seed": 42,
                        "leakage_checks": ["exact_text_overlap", "policy_section_overlap"],
                    },
                    "audit": {"require_human_approval_for_production": True},
                }
                pack_contract = {
                    "$schema": "slm.domain-pack/v1",
                    "pack_id": pack_id,
                    "version": "1.0.0",
                    "display_name": "Policy Authored Quality Pack",
                    "description": "Test pack overlay with required field checks.",
                    "owner": "workspace",
                    "status": "active",
                    "default_profile_id": profile_id,
                    "overlay": {
                        "data_quality": {
                            "context_required": True,
                            "context_fields": ["context", "policy_text"],
                            "recommended_checks": [
                                {
                                    "id": "must-have-policy-section",
                                    "type": "required_field",
                                    "label": "Policy section required",
                                    "fields": ["policy_section"],
                                    "min_coverage": 1.0,
                                    "target_tab": "dataprep",
                                    "action_label": "Review mapping",
                                }
                            ],
                        }
                    },
                }
                profile = DomainProfile(
                    profile_id=profile_id,
                    version="1.0.0",
                    display_name="Policy Authored Quality Profile",
                    description="Test profile with domain-authored quality checks.",
                    owner="workspace",
                    status=DomainProfileStatus.ACTIVE,
                    schema_ref="slm.domain-profile/v1",
                    contract=profile_contract,
                    is_system=False,
                )
                db.add(profile)
                await db.flush()
                pack = DomainPack(
                    pack_id=pack_id,
                    version="1.0.0",
                    display_name="Policy Authored Quality Pack",
                    description="Test pack overlay with required field checks.",
                    owner="workspace",
                    status=DomainPackStatus.ACTIVE,
                    schema_ref="slm.domain-pack/v1",
                    default_profile_id=profile_id,
                    contract=pack_contract,
                    is_system=False,
                )
                db.add(pack)
                await db.flush()

                project = await db.get(Project, project_id)
                project.domain_pack_id = pack.id
                project.domain_profile_id = profile.id

                raw_dir = settings.DATA_DIR / "projects" / str(project_id) / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "domain_authored_policy.jsonl"
                rows = [
                    {
                        "question": "Which leave policy covers a caregiver emergency?",
                        "answer": "The policy answer should never always guarantee approval.",
                    },
                    {
                        "question": "Can a manager approve a benefit exception?",
                        "answer": "Exceptions require compliance review.",
                    },
                ]
                with raw_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

                raw_ds = Dataset(
                    project_id=project_id,
                    name="Domain Authored Policy",
                    dataset_type=DatasetType.RAW,
                    record_count=len(rows),
                    file_path=str(raw_path),
                )
                db.add(raw_ds)
                await db.flush()
                db.add(
                    RawDocument(
                        dataset_id=raw_ds.id,
                        filename="domain_authored_policy.jsonl",
                        file_type="jsonl",
                        file_path=str(raw_path),
                        file_size_bytes=raw_path.stat().st_size,
                        source="upload",
                        status=DocumentStatus.ACCEPTED,
                        chunk_count=len(rows),
                    )
                )
                await db.commit()

        asyncio.run(_seed_domain_authored_quality())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/quality-safety")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertTrue(payload["domain_authored"]["available"])
        self.assertTrue(payload["domain_authored"]["preview_only"])
        self.assertEqual(
            payload["domain_authored"]["applied_profile_id"],
            f"policy-authored-profile-{project_id}",
        )
        self.assertEqual(
            payload["domain_authored"]["applied_pack_id"],
            f"policy-authored-pack-{project_id}",
        )
        self.assertGreaterEqual(payload["summary"]["domain_authored_check_count"], 5)
        self.assertGreaterEqual(payload["summary"]["domain_authored_warning_count"], 1)
        self.assertGreaterEqual(payload["domain_authored"]["failing_count"], 1)

        authored_checks = [
            check for check in payload["checks"] if check.get("domain_authored")
        ]
        self.assertTrue(authored_checks)
        self.assertTrue(all(check.get("read_only_preview") for check in authored_checks))
        check_ids = {check["id"] for check in authored_checks}
        self.assertIn("domain_authored_required_coverage", check_ids)
        self.assertIn("domain_authored_forbidden_phrases_data-quality", check_ids)
        self.assertIn("domain_authored_explicit_regex_policy-disallow-guarantee", check_ids)
        self.assertIn("domain_authored_explicit_field_must-have-policy-section", check_ids)
        self.assertIn("domain_authored_context_gate", check_ids)
        self.assertIn("domain_authored_review_gate", check_ids)

        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("domain_authored_required_coverage", issue_ids)
        self.assertIn("domain_authored_explicit_regex_policy-disallow-guarantee", issue_ids)
        owner_labels = {item["label"] for item in payload["findings_by_owner"]}
        self.assertIn("Domain Managers", owner_labels)

    def test_domain_detection_policy_setup_creates_missing_drafts_after_confirmation(self):
        project_id = self._create_project("data-studio-policy-domain")

        async def _seed_policy_source():
            async with async_session_factory() as db:
                raw_dir = settings.DATA_DIR / "projects" / str(project_id) / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "policy_qa.jsonl"
                rows = [
                    {
                        "question": "Which leave policy covers a caregiver emergency?",
                        "answer": "The handbook says caregiver leave is covered when eligibility and notice requirements are met.",
                        "context": "Caregiver leave policy section 4.2 describes eligibility, exceptions, and notice rules.",
                        "policy_section": "leave-4.2",
                    },
                    {
                        "question": "Can a manager approve an exception to the benefit policy?",
                        "answer": "Exceptions require compliance review and written approval before the benefit is applied.",
                        "context": "Benefit policy section 7 lists exception handling and compliance guidance.",
                        "policy_section": "benefits-7",
                    },
                ]
                with raw_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

                raw_ds = Dataset(
                    project_id=project_id,
                    name="Policy Q&A",
                    dataset_type=DatasetType.RAW,
                    record_count=2,
                    file_path=str(raw_path),
                )
                db.add(raw_ds)
                await db.flush()
                db.add(
                    RawDocument(
                        dataset_id=raw_ds.id,
                        filename="policy_qa.jsonl",
                        file_type="jsonl",
                        file_path=str(raw_path),
                        file_size_bytes=raw_path.stat().st_size,
                        source="upload",
                        status=DocumentStatus.ACCEPTED,
                        chunk_count=2,
                    )
                )
                await db.commit()

        asyncio.run(_seed_policy_source())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/domain-detection")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["detected_domain"]["id"], "policy_qa")
        setup = payload["domain_setup"]
        self.assertEqual(setup["detected_domain_label"], "Policy Q&A")
        self.assertEqual(setup["profile_id"], "policy-qa-profile-v1")
        self.assertEqual(setup["pack_id"], "policy-qa-pack-v1")
        self.assertTrue(setup["can_create_profile"])
        self.assertTrue(setup["can_create_pack"])
        guidance_ids = {item["id"] for item in setup["guidance"]}
        self.assertIn("unknowns", guidance_ids)
        task = setup["profile_contract"]["tasks"][0]
        self.assertIn("policy_section", task["optional_fields"])
        self.assertEqual(setup["pack_contract"]["status"], "draft")

        rejected = self.client.post(
            f"/api/projects/{project_id}/data-studio/domain-detection/domain-setup",
            json={"confirm": False},
        )
        self.assertEqual(rejected.status_code, 400, rejected.text)

        create_resp = self.client.post(
            f"/api/projects/{project_id}/data-studio/domain-detection/domain-setup",
            json={"confirm": True},
        )
        self.assertEqual(create_resp.status_code, 200, create_resp.text)
        created = create_resp.json()
        self.assertEqual(created["status"], "created")
        self.assertTrue(created["created_profile"])
        self.assertTrue(created["created_pack"])
        self.assertFalse(created["assigned_to_project"])
        self.assertEqual(created["profile"]["profile_id"], "policy-qa-profile-v1")
        self.assertEqual(created["profile"]["status"], "draft")
        self.assertEqual(created["pack"]["pack_id"], "policy-qa-pack-v1")

        profile_resp = self.client.get("/api/domain-profiles/policy-qa-profile-v1")
        self.assertEqual(profile_resp.status_code, 200, profile_resp.text)
        self.assertEqual(profile_resp.json()["status"], "draft")
        pack_resp = self.client.get("/api/domain-packs/policy-qa-pack-v1")
        self.assertEqual(pack_resp.status_code, 200, pack_resp.text)
        self.assertEqual(pack_resp.json()["default_profile_id"], "policy-qa-profile-v1")

        repeat_resp = self.client.post(
            f"/api/projects/{project_id}/data-studio/domain-detection/domain-setup",
            json={"confirm": True},
        )
        self.assertEqual(repeat_resp.status_code, 200, repeat_resp.text)
        repeat = repeat_resp.json()
        self.assertEqual(repeat["status"], "already_exists")
        self.assertFalse(repeat["created_profile"])
        self.assertFalse(repeat["created_pack"])

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

    def test_synthetic_recommendations_use_domain_mapping_gold_and_queue(self):
        project_id = self._create_project("data-studio-synth-recs")
        recipe_resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        self.assertEqual(recipe_resp.status_code, 200, recipe_resp.text)

        async def _seed_recommendation_state():
            async with async_session_factory() as db:
                raw_dir = settings.DATA_DIR / "projects" / str(project_id) / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "support_faq.jsonl"
                raw_rows = [
                    {
                        "question": "How do I reset my password when the login code fails?",
                        "answer": "Open account security and request a new reset code.",
                    },
                    {
                        "question": "Can I get a refund after subscription renewal?",
                        "answer": "Open a billing ticket and include the renewal invoice.",
                    },
                ]
                with raw_path.open("w", encoding="utf-8") as handle:
                    for row in raw_rows:
                        handle.write(json.dumps(row) + "\n")

                gold_dir = settings.DATA_DIR / "projects" / str(project_id) / "gold"
                gold_dir.mkdir(parents=True, exist_ok=True)
                gold_path = gold_dir / "gold_dev.jsonl"
                with gold_path.open("w", encoding="utf-8") as handle:
                    for row in raw_rows:
                        handle.write(json.dumps(row) + "\n")

                synth_dir = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
                synth_dir.mkdir(parents=True, exist_ok=True)
                synth_path = synth_dir / "synthetic.jsonl"
                synth_rows = [
                    {
                        "id": 1,
                        "text": "Need a refund for renewal",
                        "label": "billing",
                        "synth_source": "playbook:classification:positives_paraphrase",
                        "synth_confidence": 0.91,
                        "review_status": "pending",
                    },
                    {
                        "id": 2,
                        "text": "Reset code did not arrive",
                        "label": "account",
                        "synth_source": "playbook:classification:positives_paraphrase",
                        "synth_confidence": 0.9,
                        "review_status": "pending",
                    },
                ]
                with synth_path.open("w", encoding="utf-8") as handle:
                    for row in synth_rows:
                        handle.write(json.dumps(row) + "\n")

                raw_ds = Dataset(
                    project_id=project_id,
                    name="Support FAQ",
                    dataset_type=DatasetType.RAW,
                    record_count=len(raw_rows),
                    file_path=str(raw_path),
                )
                db.add(raw_ds)
                await db.flush()
                db.add_all([
                    RawDocument(
                        dataset_id=raw_ds.id,
                        filename="support_faq.jsonl",
                        file_type="jsonl",
                        file_path=str(raw_path),
                        file_size_bytes=raw_path.stat().st_size,
                        source="upload",
                        status=DocumentStatus.ACCEPTED,
                        chunk_count=2,
                    ),
                    Dataset(
                        project_id=project_id,
                        name="Gold Dev",
                        dataset_type=DatasetType.GOLD_DEV,
                        record_count=len(raw_rows),
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

        asyncio.run(_seed_recommendation_state())

        with patch(
            "app.services.data_studio_service.BACKEND_REGISTRY",
            [FakeOllamaBackend],
        ):
            resp = self.client.get(
                f"/api/projects/{project_id}/data-studio/synthetic-recommendations"
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "attention")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["source_of_truth"], "deterministic_data_studio_checks")
        self.assertEqual(payload["domain"]["id"], "support_faq")
        self.assertEqual(payload["signals"]["synthetic_pending"], 2)
        self.assertTrue(payload["signals"]["ollama_available"])
        recommendation_ids = {item["id"] for item in payload["recommendations"]}
        self.assertIn("domain_support_faq_customer_phrasing", recommendation_ids)
        self.assertIn("review_pending_synthetic_before_more_generation", recommendation_ids)
        self.assertIn("strengthen_gold_set_before_synthetic_generation", recommendation_ids)
        domain_rec = next(
            item for item in payload["recommendations"]
            if item["id"] == "domain_support_faq_customer_phrasing"
        )
        self.assertEqual(domain_rec["target_tab"], "synthetic")
        self.assertEqual(domain_rec["playbook_mode"], "positives_paraphrase")
        self.assertTrue(domain_rec["requires_user_confirmation"])
        self.assertFalse(domain_rec["generation_path"]["paid_required"])
        self.assertIn("Support", domain_rec["domain_reason"])

    def test_synthetic_recommendations_no_recipe_are_non_mutating_setup_guidance(self):
        project_id = self._create_project("data-studio-synth-recs-empty")

        class OfflineOllamaBackend:
            name = "ollama"

            @classmethod
            def is_available(cls):
                return False

        with patch(
            "app.services.data_studio_service.BACKEND_REGISTRY",
            [OfflineOllamaBackend],
        ):
            resp = self.client.get(
                f"/api/projects/{project_id}/data-studio/synthetic-recommendations"
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "attention")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["auto_apply"])
        self.assertIsNone(payload["recipe"])
        recommendation_ids = {item["id"] for item in payload["recommendations"]}
        self.assertIn("setup_recipe_for_synthetic_recommendations", recommendation_ids)
        self.assertIn("start_local_ollama_for_synthetic_generation", recommendation_ids)
        self.assertFalse(payload["signals"]["ollama_available"])
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("synthetic_recommendation_recipe_missing", issue_ids)

    def test_review_queue_summarizes_synthetic_gold_and_annotation_work(self):
        project_id = self._create_project("data-studio-review-queue")

        async def _seed_review_state():
            async with async_session_factory() as db:
                now = datetime.now(timezone.utc)
                raw_dir = settings.DATA_DIR / "projects" / str(project_id) / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "support_faq.jsonl"
                raw_rows = [
                    {
                        "question": "How do I reset my password?",
                        "answer": "Use the account password reset flow.",
                    },
                    {
                        "question": "Can I get a refund for renewal?",
                        "answer": "Open a billing ticket with the invoice.",
                    },
                ]
                with raw_path.open("w", encoding="utf-8") as handle:
                    for row in raw_rows:
                        handle.write(json.dumps(row) + "\n")

                synth_dir = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
                synth_dir.mkdir(parents=True, exist_ok=True)
                synth_path = synth_dir / "synthetic.jsonl"
                synth_rows = [
                    {
                        "id": 1,
                        "text": "Reset code never arrived",
                        "label": "account",
                        "synth_source": "playbook:classification:positives_paraphrase",
                        "review_status": "pending",
                    },
                    {
                        "id": 2,
                        "text": "Need a subscription refund",
                        "label": "billing",
                        "synth_source": "playbook:classification:positives_paraphrase",
                        "review_status": "pending",
                    },
                    {
                        "id": 3,
                        "text": "Billing portal unavailable",
                        "label": "billing",
                        "synth_source": "playbook:classification:hard_negatives",
                        "review_status": "accepted",
                    },
                ]
                with synth_path.open("w", encoding="utf-8") as handle:
                    for row in synth_rows:
                        handle.write(json.dumps(row) + "\n")

                raw_ds = Dataset(
                    project_id=project_id,
                    name="Support FAQ",
                    dataset_type=DatasetType.RAW,
                    record_count=len(raw_rows),
                    file_path=str(raw_path),
                )
                gold = Dataset(
                    project_id=project_id,
                    name="Support Gold Dev",
                    dataset_type=DatasetType.GOLD_DEV,
                    record_count=2,
                )
                synthetic = Dataset(
                    project_id=project_id,
                    name="Synthetic",
                    dataset_type=DatasetType.SYNTHETIC,
                    record_count=len(synth_rows),
                    file_path=str(synth_path),
                )
                db.add_all([raw_ds, gold, synthetic])
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
                version = GoldSetVersion(
                    gold_set_id=gold.id,
                    version=1,
                    status=GoldSetVersionStatus.DRAFT,
                )
                db.add(version)
                await db.flush()
                approved = GoldSetRow(
                    gold_set_id=gold.id,
                    version_id=version.id,
                    input={"question": "How do I reset my password?"},
                    expected={"answer": "Use account reset."},
                    labels={"category": "account"},
                    status=GoldSetRowStatus.APPROVED,
                )
                pending = GoldSetRow(
                    gold_set_id=gold.id,
                    version_id=version.id,
                    input={"question": "Refund?"},
                    expected={"answer": "Open billing ticket."},
                    labels={"category": "billing"},
                    status=GoldSetRowStatus.PENDING,
                    reviewer_id=11,
                )
                db.add_all([approved, pending])
                await db.flush()
                db.add(
                    GoldSetReviewerQueue(
                        gold_set_id=gold.id,
                        row_id=pending.id,
                        reviewer_id=11,
                        priority=2,
                        status=GoldSetReviewerQueueStatus.PENDING,
                    )
                )

                job = LabelJob(
                    project_id=project_id,
                    name="Support annotation pass",
                    label_type="classification",
                    label_schema={"allowed_labels": ["account", "billing"]},
                    status="active",
                    target_rows=5,
                )
                db.add(job)
                await db.flush()
                db.add_all([
                    LabelRow(
                        job_id=job.id,
                        source_row_id="assigned",
                        raw_payload={"text": "Please reset login"},
                        assigned_to=7,
                        assigned_at=now,
                    ),
                    LabelRow(
                        job_id=job.id,
                        source_row_id="unlabeled",
                        raw_payload={"text": "Invoice missing"},
                    ),
                    LabelRow(
                        job_id=job.id,
                        source_row_id="labeled-unpromoted",
                        raw_payload={"text": "Refund renewal"},
                        label_payload={"label": "billing"},
                        labeled_at=now,
                    ),
                    LabelRow(
                        job_id=job.id,
                        source_row_id="promoted",
                        raw_payload={"text": "Password reset"},
                        label_payload={"label": "account"},
                        labeled_at=now,
                        promoted_at=now,
                        promoted_to_dataset_id=synthetic.id,
                    ),
                ])
                await db.commit()

        asyncio.run(_seed_review_state())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/review-queue")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "attention")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["source_of_truth"], "deterministic_data_studio_checks")
        self.assertEqual(payload["domain"]["id"], "support_faq")
        self.assertEqual(payload["totals"]["synthetic_pending"], 2)
        self.assertEqual(payload["totals"]["synthetic_accepted"], 1)
        self.assertEqual(payload["totals"]["gold_review_needed"], 1)
        self.assertEqual(payload["totals"]["gold_trusted_examples"], 1)
        self.assertEqual(payload["totals"]["annotation_jobs"], 1)
        self.assertEqual(payload["totals"]["annotation_review_needed"], 2)
        self.assertEqual(payload["totals"]["annotation_labeled_unpromoted"], 1)
        self.assertEqual(payload["totals"]["annotation_promoted"], 1)
        triage_ids = {item["id"] for item in payload["triage"]}
        self.assertIn("review_pending_synthetic_rows", triage_ids)
        self.assertIn("review_gold_set_rows", triage_ids)
        self.assertIn("promote_labeled_annotation_rows", triage_ids)
        self.assertIn("continue_annotation_review", triage_ids)
        source_kinds = {item["kind"] for item in payload["groupings"]["by_source"]}
        self.assertIn("synthetic", source_kinds)
        self.assertIn("gold_set", source_kinds)
        self.assertIn("annotation", source_kinds)
        status_counts = {
            item["status"]: item["count"]
            for item in payload["groupings"]["by_status"]
        }
        self.assertEqual(status_counts["synthetic_pending"], 2)
        self.assertEqual(status_counts["annotation_needs_promotion"], 1)
        entry_targets = {item["target_tab"] for item in payload["entry_points"]}
        self.assertIn("synthetic", entry_targets)
        self.assertIn("goldset", entry_targets)
        self.assertIn("annotate", entry_targets)
        self.assertIn("eval", entry_targets)
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("review_queue_synthetic_pending", issue_ids)
        self.assertIn("review_queue_gold_needs_review", issue_ids)
        self.assertIn("review_queue_annotation_labeled_unpromoted", issue_ids)

    def test_review_queue_empty_state_is_read_only(self):
        project_id = self._create_project("data-studio-review-empty")

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/review-queue")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "empty")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["totals"]["open_review_items"], 0)
        self.assertEqual(payload["totals"]["accepted_or_promoted"], 0)
        self.assertEqual(payload["triage"][0]["id"], "create_review_source")
        self.assertEqual(payload["triage"][0]["target_tab"], "synthetic")
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("review_queue_no_review_sources", issue_ids)

    def test_prepare_dataset_empty_project_is_blocked_and_read_only(self):
        project_id = self._create_project("data-studio-prepare-empty")

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/prepare-dataset")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "blocked")
        self.assertFalse(payload["can_prepare"])
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["source_of_truth"], "deterministic_data_studio_checks")
        self.assertEqual(payload["entry_point"]["target_tab"], "dataprep")
        self.assertTrue(payload["entry_point"]["requires_confirmation"])
        self.assertEqual(payload["recipe"]["status"], "missing")
        self.assertEqual(payload["mapping"]["status"], "missing")
        self.assertEqual(payload["splits"]["status"], "missing")
        self.assertEqual(payload["manifest"]["status"], "missing")
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("prepare_missing_recipe", issue_ids)
        self.assertIn("prepare_no_trainable_rows", issue_ids)
        self.assertIn("prepare_no_mapping_source", issue_ids)

    def test_prepare_dataset_ready_when_mapping_splits_manifest_and_versions_align(self):
        project_id = self._create_project("data-studio-prepare-ready")
        recipe_resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        self.assertEqual(recipe_resp.status_code, 200, recipe_resp.text)

        async def _seed_prepare_state():
            async with async_session_factory() as db:
                source_dir = settings.DATA_DIR / "projects" / str(project_id) / "cleaned"
                source_dir.mkdir(parents=True, exist_ok=True)
                cleaned_path = source_dir / "cleaned.jsonl"
                cleaned_rows = [
                    {"text": f"Support ticket {idx}", "label": "billing" if idx % 2 else "account"}
                    for idx in range(24)
                ]
                with cleaned_path.open("w", encoding="utf-8") as handle:
                    for row in cleaned_rows:
                        handle.write(json.dumps(row) + "\n")

                prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
                prepared_dir.mkdir(parents=True, exist_ok=True)
                split_rows = {
                    "train": cleaned_rows[:20],
                    "val": cleaned_rows[20:22],
                    "test": cleaned_rows[22:],
                }
                split_paths = {}
                for split_name, rows in split_rows.items():
                    split_path = prepared_dir / f"{split_name}.jsonl"
                    split_paths[split_name] = str(split_path)
                    with split_path.open("w", encoding="utf-8") as handle:
                        for row in rows:
                            handle.write(json.dumps(row) + "\n")

                cleaned = Dataset(
                    project_id=project_id,
                    name="Cleaned Rows",
                    dataset_type=DatasetType.CLEANED,
                    record_count=len(cleaned_rows),
                    file_path=str(cleaned_path),
                )
                train = Dataset(
                    project_id=project_id,
                    name="Train Set",
                    dataset_type=DatasetType.TRAIN,
                    record_count=len(split_rows["train"]),
                    file_path=split_paths["train"],
                )
                validation = Dataset(
                    project_id=project_id,
                    name="Validation Set",
                    dataset_type=DatasetType.VALIDATION,
                    record_count=len(split_rows["val"]),
                    file_path=split_paths["val"],
                )
                test = Dataset(
                    project_id=project_id,
                    name="Test Set",
                    dataset_type=DatasetType.TEST,
                    record_count=len(split_rows["test"]),
                    file_path=split_paths["test"],
                )
                db.add_all([cleaned, train, validation, test])
                await db.flush()
                db.add_all([
                    DatasetVersion(
                        dataset_id=train.id,
                        version=1,
                        file_path=split_paths["train"],
                        record_count=len(split_rows["train"]),
                        manifest={"split": "train", "count": len(split_rows["train"])},
                    ),
                    DatasetVersion(
                        dataset_id=validation.id,
                        version=1,
                        file_path=split_paths["val"],
                        record_count=len(split_rows["val"]),
                        manifest={"split": "val", "count": len(split_rows["val"])},
                    ),
                    DatasetVersion(
                        dataset_id=test.id,
                        version=1,
                        file_path=split_paths["test"],
                        record_count=len(split_rows["test"]),
                        manifest={"split": "test", "count": len(split_rows["test"])},
                    ),
                ])
                manifest = {
                    "project_id": project_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "seed": 42,
                    "total_entries": len(cleaned_rows),
                    "splits": {
                        "train": len(split_rows["train"]),
                        "val": len(split_rows["val"]),
                        "test": len(split_rows["test"]),
                    },
                    "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                    "file_paths": split_paths,
                    "file_hashes": {},
                    "dataset_versions": {"train": 1, "val": 1, "test": 1},
                    "chat_template": "llama3",
                    "included_types": ["cleaned"],
                    "adapter_id": "classification-label",
                    "adapter_config": {},
                    "field_mapping": {},
                    "task_profile": "classification",
                }
                (prepared_dir / "manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                await db.commit()

        asyncio.run(_seed_prepare_state())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/prepare-dataset")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "ready")
        self.assertTrue(payload["can_prepare"])
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["recipe"]["selected"]["id"], "classification")
        self.assertEqual(payload["mapping"]["status"], "met")
        self.assertTrue(payload["mapping"]["contract_pass"])
        self.assertEqual(payload["splits"]["status"], "ready")
        self.assertEqual(payload["splits"]["total_prepared_rows"], 24)
        self.assertEqual(payload["manifest"]["status"], "ready")
        self.assertEqual(payload["manifest"]["dataset_versions"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(payload["inclusion"]["cleaned_rows"], 24)
        self.assertEqual(payload["inclusion"]["included_source_types"], ["cleaned"])
        self.assertEqual(payload["review_blockers"], [])
        self.assertFalse(any(item["severity"] == "blocker" for item in payload["issues"]))

    def test_dataset_versions_empty_state_is_read_only(self):
        project_id = self._create_project("data-studio-versions-empty")

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/dataset-versions")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "empty")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["source_of_truth"], "deterministic_data_studio_checks")
        self.assertEqual(payload["summary"]["total_version_count"], 0)
        self.assertFalse(payload["summary"]["training_reuse_ready"])
        self.assertFalse(payload["summary"]["eval_reuse_ready"])
        self.assertFalse(payload["manifest"]["exists"])
        self.assertEqual(payload["entry_points"][0]["target_tab"], "dataprep")
        self.assertTrue(payload["entry_points"][0]["requires_confirmation"])
        issue_ids = {item["id"] for item in payload["issues"]}
        self.assertIn("dataset_versions_empty", issue_ids)

    def test_dataset_versions_summarizes_manifest_history_and_reuse(self):
        project_id = self._create_project("data-studio-versions-ready")
        recipe_resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        self.assertEqual(recipe_resp.status_code, 200, recipe_resp.text)

        async def _seed_version_state():
            async with async_session_factory() as db:
                prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
                prepared_dir.mkdir(parents=True, exist_ok=True)
                rows = [
                    {"text": f"Versioned support ticket {idx}", "label": "billing" if idx % 2 else "account"}
                    for idx in range(30)
                ]
                split_rows = {
                    "train": rows[:24],
                    "val": rows[24:27],
                    "test": rows[27:],
                }
                split_paths = {}
                file_hashes = {}
                for split_name, split_data in split_rows.items():
                    split_path = prepared_dir / f"{split_name}.jsonl"
                    split_paths[split_name] = str(split_path)
                    with split_path.open("w", encoding="utf-8") as handle:
                        for row in split_data:
                            handle.write(json.dumps(row) + "\n")
                    file_hashes[split_name] = f"hash-{split_name}"

                train = Dataset(
                    project_id=project_id,
                    name="Train Set",
                    dataset_type=DatasetType.TRAIN,
                    record_count=len(split_rows["train"]),
                    file_path=split_paths["train"],
                )
                validation = Dataset(
                    project_id=project_id,
                    name="Validation Set",
                    dataset_type=DatasetType.VALIDATION,
                    record_count=len(split_rows["val"]),
                    file_path=split_paths["val"],
                )
                test = Dataset(
                    project_id=project_id,
                    name="Test Set",
                    dataset_type=DatasetType.TEST,
                    record_count=len(split_rows["test"]),
                    file_path=split_paths["test"],
                )
                db.add_all([train, validation, test])
                await db.flush()
                db.add_all([
                    DatasetVersion(
                        dataset_id=train.id,
                        version=1,
                        file_path=split_paths["train"],
                        record_count=20,
                        manifest={"split": "train", "count": 20},
                    ),
                    DatasetVersion(
                        dataset_id=train.id,
                        version=2,
                        file_path=split_paths["train"],
                        record_count=len(split_rows["train"]),
                        manifest={"split": "train", "count": len(split_rows["train"])},
                    ),
                    DatasetVersion(
                        dataset_id=validation.id,
                        version=1,
                        file_path=split_paths["val"],
                        record_count=len(split_rows["val"]),
                        manifest={"split": "val", "count": len(split_rows["val"])},
                    ),
                    DatasetVersion(
                        dataset_id=test.id,
                        version=1,
                        file_path=split_paths["test"],
                        record_count=len(split_rows["test"]),
                        manifest={"split": "test", "count": len(split_rows["test"])},
                    ),
                ])
                manifest = {
                    "project_id": project_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "seed": 42,
                    "total_entries": len(rows),
                    "splits": {
                        "train": len(split_rows["train"]),
                        "val": len(split_rows["val"]),
                        "test": len(split_rows["test"]),
                    },
                    "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                    "file_paths": split_paths,
                    "file_hashes": file_hashes,
                    "dataset_versions": {"train": 2, "val": 1, "test": 1},
                    "chat_template": "llama3",
                    "included_types": ["cleaned", "gold_dev", "synthetic"],
                    "adapter_id": "classification-label",
                    "adapter_config": {},
                    "field_mapping": {},
                    "task_profile": "classification",
                }
                (prepared_dir / "manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                await db.commit()

        asyncio.run(_seed_version_state())

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/dataset-versions")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "ready")
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["summary"]["prepared_dataset_count"], 3)
        self.assertEqual(payload["summary"]["total_version_count"], 4)
        self.assertEqual(payload["summary"]["latest_total_rows"], 30)
        self.assertTrue(payload["summary"]["training_reuse_ready"])
        self.assertTrue(payload["summary"]["eval_reuse_ready"])
        self.assertEqual(payload["manifest"]["dataset_versions"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(payload["manifest"]["included_types"], ["cleaned", "gold_dev", "synthetic"])
        self.assertEqual(payload["source_context"]["recipe"]["id"], "classification")
        self.assertEqual(payload["source_context"]["adapter_id"], "classification-label")
        artifacts = {item["key"]: item for item in payload["latest_artifacts"]}
        self.assertEqual(artifacts["train"]["latest_version_number"], 2)
        self.assertTrue(artifacts["train"]["version_matches_manifest"])
        self.assertTrue(artifacts["validation"]["row_count_matches_manifest"])
        history = {item["dataset_type"]: item for item in payload["version_history"]}
        self.assertEqual(history["train"]["version_count"], 2)
        signal_status = {item["id"]: item["status"] for item in payload["reproducibility"]}
        self.assertEqual(signal_status["manifest"], "met")
        self.assertEqual(signal_status["version_refs"], "met")
        targets = {item["target_tab"] for item in payload["entry_points"]}
        self.assertIn("dataprep", targets)
        self.assertIn("training", targets)
        self.assertIn("eval", targets)
        self.assertFalse(any(item["severity"] == "warning" for item in payload["issues"]))

    def test_coach_rail_empty_project_prioritizes_cross_section_blockers(self):
        project_id = self._create_project("data-studio-coach-empty")

        resp = self.client.get(f"/api/projects/{project_id}/data-studio/coach")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["verdict"], "blocked")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["auto_apply"])
        self.assertEqual(payload["source_of_truth"], "deterministic_data_studio_checks")
        self.assertGreater(payload["summary"]["blocker_count"], 0)
        self.assertEqual(payload["next_action"]["severity"], "blocker")
        self.assertIn(payload["next_action"]["target_tab"], {"data", "goldset"})
        check_ids = {item["id"] for item in payload["checks"]}
        self.assertIn("sources", check_ids)
        self.assertIn("mapping", check_ids)
        self.assertIn("prepare_dataset", check_ids)
        issue_sections = {item["section_id"] for item in payload["issues"]}
        self.assertIn("overview", issue_sections)
        self.assertIn("mapping", issue_sections)
        self.assertEqual(payload["entry_points"][0]["target_tab"], "dataprep")

    def test_coach_rail_routes_to_training_when_reusable_versions_are_ready(self):
        project_id = self._create_project("data-studio-coach-ready")
        recipe_resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        self.assertEqual(recipe_resp.status_code, 200, recipe_resp.text)

        async def _seed_coach_ready_state():
            async with async_session_factory() as db:
                gold_dir = settings.DATA_DIR / "projects" / str(project_id) / "gold"
                gold_dir.mkdir(parents=True, exist_ok=True)
                gold_path = gold_dir / "gold_dev.jsonl"
                gold_rows = [
                    {"text": f"Trusted classification example {idx}", "label": "billing" if idx % 2 else "account"}
                    for idx in range(25)
                ]
                with gold_path.open("w", encoding="utf-8") as handle:
                    for row in gold_rows:
                        handle.write(json.dumps(row) + "\n")

                prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
                prepared_dir.mkdir(parents=True, exist_ok=True)
                rows = [
                    {"text": f"Prepared classification example {idx}", "label": "billing" if idx % 2 else "account"}
                    for idx in range(30)
                ]
                split_rows = {
                    "train": rows[:24],
                    "val": rows[24:27],
                    "test": rows[27:],
                }
                split_paths = {}
                file_hashes = {}
                for split_name, split_data in split_rows.items():
                    split_path = prepared_dir / f"{split_name}.jsonl"
                    split_paths[split_name] = str(split_path)
                    with split_path.open("w", encoding="utf-8") as handle:
                        for row in split_data:
                            handle.write(json.dumps(row) + "\n")
                    file_hashes[split_name] = f"hash-{split_name}"

                gold = Dataset(
                    project_id=project_id,
                    name="Gold Dev",
                    dataset_type=DatasetType.GOLD_DEV,
                    record_count=len(gold_rows),
                    file_path=str(gold_path),
                )
                train = Dataset(
                    project_id=project_id,
                    name="Train Set",
                    dataset_type=DatasetType.TRAIN,
                    record_count=len(split_rows["train"]),
                    file_path=split_paths["train"],
                )
                validation = Dataset(
                    project_id=project_id,
                    name="Validation Set",
                    dataset_type=DatasetType.VALIDATION,
                    record_count=len(split_rows["val"]),
                    file_path=split_paths["val"],
                )
                test = Dataset(
                    project_id=project_id,
                    name="Test Set",
                    dataset_type=DatasetType.TEST,
                    record_count=len(split_rows["test"]),
                    file_path=split_paths["test"],
                )
                db.add_all([gold, train, validation, test])
                await db.flush()
                gold_version = GoldSetVersion(
                    gold_set_id=gold.id,
                    version=1,
                    status=GoldSetVersionStatus.DRAFT,
                )
                db.add(gold_version)
                await db.flush()
                db.add_all([
                    GoldSetRow(
                        gold_set_id=gold.id,
                        version_id=gold_version.id,
                        source_row_key=f"gold-{idx}",
                        input={"text": row["text"]},
                        expected={"label": row["label"]},
                        labels={"category": row["label"]},
                        status=GoldSetRowStatus.APPROVED,
                    )
                    for idx, row in enumerate(gold_rows)
                ])
                db.add_all([
                    DatasetVersion(
                        dataset_id=train.id,
                        version=1,
                        file_path=split_paths["train"],
                        record_count=len(split_rows["train"]),
                        manifest={"split": "train", "count": len(split_rows["train"])},
                    ),
                    DatasetVersion(
                        dataset_id=validation.id,
                        version=1,
                        file_path=split_paths["val"],
                        record_count=len(split_rows["val"]),
                        manifest={"split": "val", "count": len(split_rows["val"])},
                    ),
                    DatasetVersion(
                        dataset_id=test.id,
                        version=1,
                        file_path=split_paths["test"],
                        record_count=len(split_rows["test"]),
                        manifest={"split": "test", "count": len(split_rows["test"])},
                    ),
                ])
                manifest = {
                    "project_id": project_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "seed": 42,
                    "total_entries": len(rows),
                    "splits": {
                        "train": len(split_rows["train"]),
                        "val": len(split_rows["val"]),
                        "test": len(split_rows["test"]),
                    },
                    "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                    "file_paths": split_paths,
                    "file_hashes": file_hashes,
                    "dataset_versions": {"train": 1, "val": 1, "test": 1},
                    "chat_template": "llama3",
                    "included_types": ["gold_dev"],
                    "adapter_id": "classification-label",
                    "adapter_config": {},
                    "field_mapping": {},
                    "task_profile": "classification",
                }
                (prepared_dir / "manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                await db.commit()

        class FakeOllamaBackend:
            name = "ollama"

            @classmethod
            def is_available(cls):
                return True

            def describe(self):
                return "ollama:llama3"

        asyncio.run(_seed_coach_ready_state())
        with patch(
            "app.services.data_studio_service.BACKEND_REGISTRY",
            [FakeOllamaBackend],
        ):
            resp = self.client.get(f"/api/projects/{project_id}/data-studio/coach")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertEqual(payload["summary"]["blocker_count"], 0)
        self.assertEqual(payload["summary"]["warning_count"], 0)
        self.assertEqual(payload["next_action"]["target_tab"], "training")
        self.assertEqual(payload["next_action"]["action_label"], "Open Training")
        check_status = {item["id"]: item["status"] for item in payload["checks"]}
        self.assertEqual(check_status["prepare_dataset"], "ready")
        self.assertEqual(check_status["dataset_versions"], "ready")
        self.assertEqual(payload["power_details"]["training_reuse_ready"], True)

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
