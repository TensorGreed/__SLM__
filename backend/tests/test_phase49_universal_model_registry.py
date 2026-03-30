"""Phase 49 tests: universal base model registry + compatibility engine."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase49_universal_model_registry.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase49_universal_model_registry_data"
TEST_MODELS_DIR = TEST_DATA_DIR / "models"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"
os.environ["DOMAIN_BLUEPRINT_ENABLE_LLM_ENRICHMENT"] = "false"

from fastapi.testclient import TestClient

from app.main import app
from app.services.base_model_registry_service import normalize_base_model_metadata


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _make_model_dir(
    root: Path,
    name: str,
    *,
    config: dict,
    tokenizer: dict | None = None,
) -> Path:
    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(model_dir / "config.json", config)
    if tokenizer is not None:
        _write_json(model_dir / "tokenizer_config.json", tokenizer)
    return model_dir


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


class Phase49UniversalModelRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            _clear_path(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        TEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        cls.causal_dir = _make_model_dir(
            TEST_MODELS_DIR,
            "causal_model",
            config={
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 1024,
                "num_hidden_layers": 18,
                "vocab_size": 32000,
                "max_position_embeddings": 8192,
                "license": "apache-2.0",
            },
            tokenizer={
                "tokenizer_class": "LlamaTokenizer",
                "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            },
        )
        cls.seq2seq_dir = _make_model_dir(
            TEST_MODELS_DIR,
            "seq2seq_model",
            config={
                "architectures": ["T5ForConditionalGeneration"],
                "model_type": "t5",
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vocab_size": 32128,
                "is_encoder_decoder": True,
                "max_position_embeddings": 4096,
                "license": "apache-2.0",
            },
            tokenizer={
                "tokenizer_class": "T5Tokenizer",
            },
        )
        cls.classifier_dir = _make_model_dir(
            TEST_MODELS_DIR,
            "classifier_model",
            config={
                "architectures": ["BertForSequenceClassification"],
                "model_type": "bert",
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vocab_size": 30522,
                "max_position_embeddings": 512,
                "license": "mit",
            },
            tokenizer={
                "tokenizer_class": "BertTokenizer",
            },
        )

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
        create_resp = self.client.post(
            "/api/projects",
            json={
                "name": name,
                "description": "phase49",
                "target_profile_id": "vllm_server",
                "beginner_mode": True,
            },
        )
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        return int(create_resp.json()["id"])

    def _save_blueprint(self, project_id: int, task_family: str = "qa") -> None:
        save_resp = self.client.post(
            f"/api/projects/{project_id}/domain-blueprints",
            json={
                "source": "test.phase49",
                "brief_text": "Build a support assistant with safe responses.",
                "apply_immediately": True,
                "blueprint": {
                    "domain_name": "Support",
                    "problem_statement": "Answer support tickets accurately.",
                    "target_user_persona": "Support agents",
                    "task_family": task_family,
                    "input_modality": "text",
                    "expected_output_schema": {
                        "type": "object",
                        "properties": {"answer": "string"},
                        "required": ["answer"],
                    },
                    "expected_output_examples": [{"answer": "Use the reset flow."}],
                    "safety_compliance_notes": ["No PII leaks."],
                    "deployment_target_constraints": {"target_profile_id": "vllm_server"},
                    "success_metrics": [{"metric_id": "answer_correctness", "label": "Answer Correctness"}],
                    "glossary": [{"term": "task family", "plain_language": "Type of behavior", "category": "general"}],
                    "confidence_score": 0.82,
                    "unresolved_assumptions": [],
                },
            },
        )
        self.assertEqual(save_resp.status_code, 201, save_resp.text)

    def test_unit_normalization_across_causal_seq2seq_classifier(self):
        causal = normalize_base_model_metadata(
            source_type="local_path",
            source_ref=str(self.causal_dir),
            allow_network=False,
        )
        seq2seq = normalize_base_model_metadata(
            source_type="local_path",
            source_ref=str(self.seq2seq_dir),
            allow_network=False,
        )
        classifier = normalize_base_model_metadata(
            source_type="local_path",
            source_ref=str(self.classifier_dir),
            allow_network=False,
        )

        self.assertEqual(causal["architecture"], "causal_lm")
        self.assertIn("qa", set(causal.get("supported_task_families") or []))
        self.assertTrue(bool(causal.get("peft_support")))

        self.assertEqual(seq2seq["architecture"], "seq2seq")
        self.assertIn("summarization", set(seq2seq.get("supported_task_families") or []))
        self.assertIn("sft", set(seq2seq.get("training_mode_support") or []))

        self.assertEqual(classifier["architecture"], "classification")
        self.assertIn("classification", set(classifier.get("supported_task_families") or []))
        self.assertIn("sft", set(classifier.get("training_mode_support") or []))

    def test_api_import_refresh_validate_and_recommend(self):
        project_id = self._create_project("phase49-api-flows")
        self._save_blueprint(project_id, task_family="qa")

        import_causal = self.client.post(
            "/api/models/import",
            json={
                "source_type": "local_path",
                "source_ref": str(self.causal_dir),
                "allow_network": False,
            },
        )
        self.assertEqual(import_causal.status_code, 201, import_causal.text)
        causal_model = import_causal.json()["model"]

        import_seq2seq = self.client.post(
            "/api/models/import",
            json={
                "source_type": "local_path",
                "source_ref": str(self.seq2seq_dir),
                "allow_network": False,
            },
        )
        self.assertEqual(import_seq2seq.status_code, 201, import_seq2seq.text)
        seq2seq_model = import_seq2seq.json()["model"]

        import_classifier = self.client.post(
            "/api/models/import",
            json={
                "source_type": "local_path",
                "source_ref": str(self.classifier_dir),
                "allow_network": False,
            },
        )
        self.assertEqual(import_classifier.status_code, 201, import_classifier.text)
        classifier_model = import_classifier.json()["model"]

        import_catalog = self.client.post(
            "/api/models/import",
            json={
                "source_type": "catalog",
                "source_ref": "meta-llama/Llama-3.2-1B-Instruct",
                "allow_network": False,
            },
        )
        self.assertEqual(import_catalog.status_code, 201, import_catalog.text)

        import_hf = self.client.post(
            "/api/models/import",
            json={
                "source_type": "huggingface",
                "source_ref": "Qwen/Qwen2.5-1.5B-Instruct",
                "allow_network": False,
            },
        )
        self.assertEqual(import_hf.status_code, 201, import_hf.text)

        list_laptop = self.client.get("/api/models", params={"hardware_fit": "laptop"})
        self.assertEqual(list_laptop.status_code, 200, list_laptop.text)
        self.assertGreaterEqual(int(list_laptop.json().get("count") or 0), 1)

        refresh_resp = self.client.post(
            "/api/models/refresh",
            json={"model_id": int(causal_model["id"]), "allow_network": False},
        )
        self.assertEqual(refresh_resp.status_code, 200, refresh_resp.text)
        self.assertGreaterEqual(int(refresh_resp.json()["model"].get("refresh_count") or 0), 1)

        validate_pass = self.client.post(
            f"/api/projects/{project_id}/models/validate",
            json={"model_id": int(causal_model["id"]), "allow_network": False},
        )
        self.assertEqual(validate_pass.status_code, 200, validate_pass.text)
        validate_payload = validate_pass.json()
        self.assertIn("compatibility_score", validate_payload)
        self.assertIn("reason_codes", validate_payload)
        self.assertTrue(bool(validate_payload.get("compatible")))

        validate_block = self.client.post(
            f"/api/projects/{project_id}/models/validate",
            json={"model_id": int(classifier_model["id"]), "allow_network": False},
        )
        self.assertEqual(validate_block.status_code, 200, validate_block.text)
        block_payload = validate_block.json()
        self.assertFalse(bool(block_payload.get("compatible")))
        self.assertIn("TASK_FAMILY_UNSUPPORTED", set(block_payload.get("reason_codes") or []))

        recommend_resp = self.client.get(
            f"/api/projects/{project_id}/models/compatible",
            params={"limit": 10, "include_incompatible": True, "allow_network": False},
        )
        self.assertEqual(recommend_resp.status_code, 200, recommend_resp.text)
        recommend_payload = recommend_resp.json()
        self.assertGreaterEqual(int(recommend_payload.get("count") or 0), 3)
        self.assertIn("compatible_count", recommend_payload)
        rows = [item for item in recommend_payload.get("models", []) if isinstance(item, dict)]
        self.assertTrue(rows)
        self.assertTrue(any(str(item.get("model_key")) == str(seq2seq_model.get("model_key")) for item in rows))

    def test_integration_causal_seq2seq_classifier_recommendation_behavior(self):
        project_id = self._create_project("phase49-integration")
        self._save_blueprint(project_id, task_family="qa")

        for model_dir in (self.causal_dir, self.seq2seq_dir, self.classifier_dir):
            resp = self.client.post(
                "/api/models/import",
                json={
                    "source_type": "local_path",
                    "source_ref": str(model_dir),
                    "allow_network": False,
                },
            )
            self.assertEqual(resp.status_code, 201, resp.text)

        recommend_resp = self.client.get(
            f"/api/projects/{project_id}/models/compatible",
            params={"include_incompatible": True, "limit": 20, "allow_network": False},
        )
        self.assertEqual(recommend_resp.status_code, 200, recommend_resp.text)
        rows = [item for item in recommend_resp.json().get("models", []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(rows), 3)

        by_arch: dict[str, list[dict]] = {}
        for item in rows:
            arch = str((item.get("model") or {}).get("architecture") or "unknown")
            by_arch.setdefault(arch, []).append(item)

        self.assertIn("causal_lm", by_arch)
        self.assertIn("seq2seq", by_arch)
        self.assertIn("classification", by_arch)

        classifier_rows = by_arch.get("classification") or []
        self.assertTrue(classifier_rows)
        self.assertTrue(all(not bool(item.get("compatible")) for item in classifier_rows))


if __name__ == "__main__":
    unittest.main()
