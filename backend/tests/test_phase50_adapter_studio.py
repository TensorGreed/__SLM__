"""Phase 50 tests: Dataset Structure Explorer + Adapter Studio."""

from __future__ import annotations

import csv
import json
import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase50_adapter_studio.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase50_adapter_studio_data"
TEST_FIXTURE_DIR = TEST_DATA_DIR / "fixtures"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.main import app
from app.services.adapter_studio_service import profile_rows
from app.services.data_adapter_service import preview_data_adapter


def _clear_path(path: Path) -> None:
    if not path.exists():
        return
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    if path.exists() and path.is_file():
        path.unlink()
    if path.exists() and path.is_dir():
        path.rmdir()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _make_fixtures() -> dict[str, Path]:
    csv_path = TEST_FIXTURE_DIR / "qa.csv"
    _write_csv(
        csv_path,
        [
            {
                "question": "How do I reset my password?",
                "answer": "Use the account reset flow.",
                "label": "support",
                "email": "alpha@example.com",
            },
            {
                "question": "How do I update billing info?",
                "answer": "Open billing settings and edit card.",
                "label": "billing",
                "email": "beta@example.com",
            },
        ],
    )

    tsv_path = TEST_FIXTURE_DIR / "snapshot.tsv"
    _write_tsv(
        tsv_path,
        [
            {"ticket_id": "1", "prompt": "Issue A", "response": "Resolution A"},
            {"ticket_id": "2", "prompt": "Issue B", "response": "Resolution B"},
        ],
    )

    extraction_json_path = TEST_FIXTURE_DIR / "extraction.json"
    _write_json(
        extraction_json_path,
        [
            {
                "doc": {"text": "Invoice #123 total is $98.50."},
                "entities": {"invoice_id": "123", "amount": "98.50"},
            },
            {
                "doc": {"text": "Invoice #456 total is $143.20."},
                "entities": {"invoice_id": "456", "amount": "143.20"},
            },
        ],
    )

    chat_jsonl_path = TEST_FIXTURE_DIR / "chat.jsonl"
    _write_jsonl(
        chat_jsonl_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Summarize this ticket."},
                    {"role": "assistant", "content": "This ticket requests invoice correction."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Return a short answer."},
                    {"role": "assistant", "content": "Done."},
                ]
            },
        ],
    )

    preference_jsonl_path = TEST_FIXTURE_DIR / "preference.jsonl"
    _write_jsonl(
        preference_jsonl_path,
        [
            {
                "prompt": "Answer politely",
                "chosen": "Sure, I can help with that.",
                "rejected": "No.",
            },
            {
                "prompt": "Explain refund policy",
                "chosen": "Refunds are available within 30 days.",
                "rejected": "Read docs.",
            },
        ],
    )

    docs_jsonl_path = TEST_FIXTURE_DIR / "documents.jsonl"
    _write_jsonl(
        docs_jsonl_path,
        [
            {"document_id": "a1", "text": "Long document text chunk one."},
            {"document_id": "a1", "text": "Long document text chunk two."},
        ],
    )

    parquet_fallback_path = TEST_FIXTURE_DIR / "fallback.parquet"
    _write_jsonl(
        parquet_fallback_path,
        [
            {"input": "Translate hello", "target": "hola"},
            {"input": "Translate bye", "target": "adios"},
        ],
    )

    messy_jsonl_path = TEST_FIXTURE_DIR / "messy_raw.jsonl"
    _write_jsonl(
        messy_jsonl_path,
        [
            {"prompt_text": "How to reset MFA?", "response_text": "Use account security settings."},
            {"prompt_text": "Where is invoice?", "response_text": "Open billing history tab."},
            {"prompt_text": "Need refund steps", "response_text": "Refund request form is under orders."},
            {"prompt_text": "Broken row with no response", "meta": "missing"},
        ],
    )

    return {
        "csv": csv_path,
        "tsv": tsv_path,
        "json": extraction_json_path,
        "jsonl_chat": chat_jsonl_path,
        "jsonl_preference": preference_jsonl_path,
        "document_corpus": docs_jsonl_path,
        "parquet": parquet_fallback_path,
        "messy_jsonl": messy_jsonl_path,
    }


class Phase50AdapterStudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            _clear_path(TEST_DATA_DIR)
        TEST_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        cls.fixtures = _make_fixtures()

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            _clear_path(TEST_DATA_DIR)

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": name,
                "description": "phase50",
                "target_profile_id": "vllm_server",
                "beginner_mode": True,
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_backend_unit_schema_profiling_and_adapter_inference_fixtures(self):
        extraction_rows = json.loads(self.fixtures["json"].read_text(encoding="utf-8"))
        profile = profile_rows(extraction_rows, source_meta={"source_type": "json"})
        self.assertGreaterEqual(int(profile.get("sampled_rows") or 0), 2)
        fields = [item for item in ((profile.get("schema") or {}).get("fields") or []) if isinstance(item, dict)]
        self.assertTrue(any(str(item.get("path") or "").startswith("doc.text") for item in fields))
        sensitive = list((profile.get("dataset_characteristics") or {}).get("potential_sensitive_columns") or [])
        self.assertIsInstance(sensitive, list)

        chat_rows = [json.loads(line) for line in self.fixtures["jsonl_chat"].read_text(encoding="utf-8").splitlines() if line.strip()]
        chat_preview = preview_data_adapter(chat_rows, adapter_id="auto", preview_limit=10)
        self.assertEqual(str(chat_preview.get("resolved_adapter_id")), "chat-messages")

        preference_rows = [json.loads(line) for line in self.fixtures["jsonl_preference"].read_text(encoding="utf-8").splitlines() if line.strip()]
        pref_preview = preview_data_adapter(preference_rows, adapter_id="auto", preview_limit=10)
        self.assertEqual(str(pref_preview.get("resolved_adapter_id")), "preference-pair")

        qa_rows = []
        with self.fixtures["csv"].open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            qa_rows.extend(list(reader))
        qa_preview = preview_data_adapter(qa_rows, adapter_id="auto", preview_limit=10)
        self.assertIn(str(qa_preview.get("resolved_adapter_id")), {"qa-pair", "default-canonical"})

    def test_api_profile_infer_preview_save_export_and_validate_flows(self):
        project_id = self._create_project("phase50-api")

        # Profile across representative source families.
        for source_type, fixture_key in (
            ("csv", "csv"),
            ("tsv", "tsv"),
            ("json", "json"),
            ("jsonl", "jsonl_chat"),
            ("parquet", "parquet"),
            ("sql_snapshot", "tsv"),
            ("document_corpus", "document_corpus"),
            ("chat_transcripts", "jsonl_chat"),
            ("pairwise_preference", "jsonl_preference"),
            ("chunk_corpus", "document_corpus"),
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/adapter-studio/profile",
                json={
                    "source": {
                        "source_type": source_type,
                        "source_ref": str(self.fixtures[fixture_key]),
                    },
                    "sample_size": 100,
                },
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertGreaterEqual(int(resp.json().get("sampled_rows") or 0), 1)

        infer_resp = self.client.post(
            f"/api/projects/{project_id}/adapter-studio/infer",
            json={
                "source": {
                    "source_type": "csv",
                    "source_ref": str(self.fixtures["csv"]),
                },
                "sample_size": 120,
            },
        )
        self.assertEqual(infer_resp.status_code, 200, infer_resp.text)
        infer_payload = infer_resp.json()
        inference = dict(infer_payload.get("inference") or {})
        resolved_adapter = str(inference.get("resolved_adapter_id") or "auto")
        mapping = dict(inference.get("mapping_canvas") or {})

        preview_resp = self.client.post(
            f"/api/projects/{project_id}/adapter-studio/preview",
            json={
                "source": {
                    "source_type": "csv",
                    "source_ref": str(self.fixtures["csv"]),
                },
                "adapter_id": resolved_adapter,
                "field_mapping": mapping,
                "sample_size": 120,
                "preview_limit": 10,
            },
        )
        self.assertEqual(preview_resp.status_code, 200, preview_resp.text)
        preview_payload = preview_resp.json()
        self.assertIn("drop_analysis", preview_payload)
        self.assertIn("preview", preview_payload)

        validate_resp = self.client.post(
            f"/api/projects/{project_id}/adapter-studio/validate",
            json={
                "source": {
                    "source_type": "csv",
                    "source_ref": str(self.fixtures["csv"]),
                },
                "adapter_id": resolved_adapter,
                "field_mapping": mapping,
                "sample_size": 120,
                "preview_limit": 10,
            },
        )
        self.assertEqual(validate_resp.status_code, 200, validate_resp.text)
        validate_payload = validate_resp.json()
        self.assertIn("status", validate_payload)
        self.assertIn("reason_codes", validate_payload)

        save_resp = self.client.post(
            f"/api/projects/{project_id}/adapter-studio/adapters",
            json={
                "adapter_name": "qa_studio",
                "source_type": "csv",
                "source_ref": str(self.fixtures["csv"]),
                "base_adapter_id": resolved_adapter,
                "task_profile": inference.get("resolved_task_profile"),
                "field_mapping": mapping,
                "adapter_config": {},
                "output_contract": dict((inference.get("adapter_contract") or {}).get("output_contract") or {}),
                "schema_profile": infer_payload.get("profile") or {},
                "inference_summary": inference,
                "validation_report": validate_payload,
            },
        )
        self.assertEqual(save_resp.status_code, 201, save_resp.text)
        saved = save_resp.json()
        self.assertEqual(str(saved.get("adapter_name")), "qa_studio")
        self.assertEqual(int(saved.get("version") or 0), 1)

        list_resp = self.client.get(f"/api/projects/{project_id}/adapter-studio/adapters", params={"adapter_name": "qa_studio"})
        self.assertEqual(list_resp.status_code, 200, list_resp.text)
        self.assertGreaterEqual(int(list_resp.json().get("count") or 0), 1)

        export_resp = self.client.post(
            f"/api/projects/{project_id}/adapter-studio/adapters/qa_studio/versions/1/export",
            json={},
        )
        self.assertEqual(export_resp.status_code, 200, export_resp.text)
        export_payload = export_resp.json()
        written = dict(export_payload.get("written_files") or {})
        self.assertTrue(str(written.get("template_json") or "").endswith("adapter_template.json"))
        self.assertTrue(str(written.get("plugin_python") or "").endswith("adapter_plugin.py"))

    def test_e2e_messy_dataset_inferred_adapter_to_valid_prepared_split(self):
        project_id = self._create_project("phase50-e2e")

        with self.fixtures["messy_jsonl"].open("rb") as handle:
            upload_resp = self.client.post(
                f"/api/projects/{project_id}/ingestion/upload",
                files={"file": ("messy_raw.jsonl", handle, "application/jsonl")},
                data={"source": "phase50", "sensitivity": "internal", "license_info": "test"},
            )
        self.assertEqual(upload_resp.status_code, 201, upload_resp.text)
        document_id = int(upload_resp.json().get("id") or 0)
        self.assertGreater(document_id, 0)

        process_resp = self.client.post(f"/api/projects/{project_id}/ingestion/documents/{document_id}/process")
        self.assertIn(process_resp.status_code, {200, 202}, process_resp.text)

        infer_resp = self.client.post(
            f"/api/projects/{project_id}/adapter-studio/infer",
            json={
                "source": {
                    "source_type": "project_dataset",
                    "dataset_type": "raw",
                    "document_id": document_id,
                },
                "sample_size": 200,
            },
        )
        self.assertEqual(infer_resp.status_code, 200, infer_resp.text)
        infer_payload = infer_resp.json()
        inference = dict(infer_payload.get("inference") or {})
        resolved_adapter = str(inference.get("resolved_adapter_id") or "default-canonical")
        mapping = dict(inference.get("mapping_canvas") or {})

        validate_resp = self.client.post(
            f"/api/projects/{project_id}/adapter-studio/validate",
            json={
                "source": {
                    "source_type": "project_dataset",
                    "dataset_type": "raw",
                    "document_id": document_id,
                },
                "adapter_id": resolved_adapter,
                "field_mapping": mapping,
                "sample_size": 200,
                "preview_limit": 20,
            },
        )
        self.assertEqual(validate_resp.status_code, 200, validate_resp.text)

        split_resp = self.client.post(
            f"/api/projects/{project_id}/dataset/split",
            json={
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
                "include_types": ["raw"],
                "adapter_id": resolved_adapter,
                "field_mapping": mapping,
                "task_profile": inference.get("resolved_task_profile"),
            },
        )
        self.assertEqual(split_resp.status_code, 200, split_resp.text)
        split_payload = split_resp.json()
        self.assertGreater(int(split_payload.get("total_entries") or 0), 0)
        self.assertIn("splits", split_payload)
        self.assertGreater(int((split_payload.get("splits") or {}).get("train") or 0), 0)


if __name__ == "__main__":
    unittest.main()
