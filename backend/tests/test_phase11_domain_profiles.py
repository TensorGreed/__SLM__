"""Phase 11 tests: domain profile contracts and project assignment."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase11_test.db"

os.environ["AUTH_ENABLED"] = "true"
os.environ["AUTH_BOOTSTRAP_API_KEY"] = "phase11-admin-key"
os.environ["AUTH_BOOTSTRAP_USERNAME"] = "phase11-admin"
os.environ["AUTH_BOOTSTRAP_ROLE"] = "admin"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DEBUG"] = "false"
os.environ["DOMAIN_HOOK_PLUGIN_MODULES"] = '["app.plugins.domain_hooks.example_hooks"]'

from fastapi.testclient import TestClient

from app.main import app


def _sample_contract(profile_id: str, display_name: str) -> dict:
    return {
        "$schema": "slm.domain-profile/v1",
        "profile_id": profile_id,
        "version": "1.0.0",
        "display_name": display_name,
        "description": "Domain profile for testing",
        "owner": "qa",
        "status": "active",
        "tasks": [
            {
                "task_id": "qa",
                "output_mode": "text",
                "required_fields": ["question", "answer"],
            }
        ],
        "canonical_schema": {
            "required": ["input_text", "target_text"],
            "aliases": {
                "input_text": ["question"],
                "target_text": ["answer"],
            },
        },
        "normalization": {
            "trim_whitespace": True,
            "drop_empty_records": True,
        },
        "data_quality": {
            "min_records": 20,
            "max_null_ratio": 0.1,
            "max_duplicate_ratio": 0.2,
            "required_coverage": {"input_text": 0.95, "target_text": 0.95},
        },
        "dataset_split": {"train": 0.8, "val": 0.1, "test": 0.1},
        "training_defaults": {
            "training_mode": "sft",
            "chat_template": "llama3",
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.0001,
            "use_lora": True,
        },
        "evaluation": {
            "metrics": [
                {"metric_id": "f1", "weight": 0.5, "threshold": 0.6},
                {"metric_id": "safety_pass_rate", "weight": 0.5, "threshold": 0.9},
            ],
            "required_metrics_for_promotion": ["f1", "safety_pass_rate"],
        },
        "tools": {
            "retrieval": {"enabled": False, "adapter": None},
            "function_calling": {"enabled": False, "adapter": None},
            "required_secrets": [],
        },
        "registry_gates": {
            "to_staging": {"min_metrics": {"f1": 0.6}},
            "to_production": {"min_metrics": {"f1": 0.7, "safety_pass_rate": 0.92}},
        },
        "audit": {
            "require_human_approval_for_production": True,
            "notes_required_on_force_promotion": True,
        },
    }


def _sample_pack(
    pack_id: str,
    display_name: str,
    *,
    default_profile_id: str = "generic-domain-v1",
    batch_size: int = 8,
    train_ratio: float = 0.75,
    learning_rate: float = 0.0003,
    normalizer_hook_id: str = "default-normalizer",
    normalizer_hook_config: dict | None = None,
    validator_hook_id: str = "default-validator",
    validator_hook_config: dict | None = None,
    evaluator_hook_id: str = "default-evaluator",
    evaluator_hook_config: dict | None = None,
) -> dict:
    return {
        "$schema": "slm.domain-pack/v1",
        "pack_id": pack_id,
        "version": "1.0.0",
        "display_name": display_name,
        "description": "Pack contract for testing",
        "owner": "qa",
        "status": "active",
        "default_profile_id": default_profile_id,
        "hooks": {
            "normalizer": {
                "id": normalizer_hook_id,
                "config": normalizer_hook_config or {},
            },
            "validator": {
                "id": validator_hook_id,
                "config": validator_hook_config or {},
            },
            "evaluator": {
                "id": evaluator_hook_id,
                "config": evaluator_hook_config or {},
            },
        },
        "overlay": {
            "dataset_split": {"train": train_ratio, "val": 0.15, "test": 0.1, "seed": 77},
            "training_defaults": {
                "training_mode": "sft",
                "chat_template": "chatml",
                "num_epochs": 2,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_lora": True,
            },
            "registry_gates": {
                "to_staging": {"min_metrics": {"f1": 0.66, "llm_judge_pass_rate": 0.76}},
                "to_production": {
                    "min_metrics": {"f1": 0.71, "llm_judge_pass_rate": 0.81, "safety_pass_rate": 0.93}
                },
            },
        },
    }


class Phase11DomainProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()
        cls.admin_headers = {"x-api-key": "phase11-admin-key"}

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    def test_default_profile_is_bootstrapped(self):
        unauthorized = self.client.get("/api/domain-profiles")
        self.assertEqual(unauthorized.status_code, 401)

        listed = self.client.get("/api/domain-profiles", headers=self.admin_headers)
        self.assertEqual(listed.status_code, 200, listed.text)
        payload = listed.json()
        self.assertGreaterEqual(payload.get("count", 0), 1)
        ids = {item["profile_id"] for item in payload.get("profiles", [])}
        self.assertIn("generic-domain-v1", ids)

        listed_packs = self.client.get("/api/domain-packs", headers=self.admin_headers)
        self.assertEqual(listed_packs.status_code, 200, listed_packs.text)
        packs_payload = listed_packs.json()
        self.assertGreaterEqual(packs_payload.get("count", 0), 1)
        pack_ids = {item["pack_id"] for item in packs_payload.get("packs", [])}
        self.assertIn("general-pack-v1", pack_ids)

    def test_create_and_assign_domain_profile(self):
        create_profile = self.client.post(
            "/api/domain-profiles",
            headers=self.admin_headers,
            json=_sample_contract("finance-risk-v1", "Finance Risk"),
        )
        self.assertEqual(create_profile.status_code, 201, create_profile.text)
        profile_payload = create_profile.json()
        self.assertEqual(profile_payload["profile_id"], "finance-risk-v1")

        create_project = self.client.post(
            "/api/projects",
            headers=self.admin_headers,
            json={"name": "phase11-project", "description": "phase11"},
        )
        self.assertEqual(create_project.status_code, 201, create_project.text)
        project_id = create_project.json()["id"]

        assign = self.client.put(
            f"/api/projects/{project_id}/domain-profile",
            headers=self.admin_headers,
            json={"profile_id": "finance-risk-v1"},
        )
        self.assertEqual(assign.status_code, 200, assign.text)
        assigned = assign.json()
        self.assertIsNotNone(assigned.get("domain_profile_id"))

        fetched = self.client.get(f"/api/projects/{project_id}", headers=self.admin_headers)
        self.assertEqual(fetched.status_code, 200, fetched.text)
        self.assertEqual(fetched.json()["domain_profile_id"], assigned["domain_profile_id"])

    def test_domain_pack_assignment_and_runtime_resolution(self):
        create_profile = self.client.post(
            "/api/domain-profiles",
            headers=self.admin_headers,
            json={
                **_sample_contract("support-domain-v1", "Support Domain"),
                "training_defaults": {
                    "training_mode": "sft",
                    "chat_template": "llama3",
                    "num_epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.0001,
                    "use_lora": False,
                },
                "dataset_split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
            },
        )
        self.assertEqual(create_profile.status_code, 201, create_profile.text)

        create_pack = self.client.post(
            "/api/domain-packs",
            headers=self.admin_headers,
            json=_sample_pack(
                "support-pack-v1",
                "Support Pack",
                default_profile_id="support-domain-v1",
                batch_size=10,
                train_ratio=0.74,
                learning_rate=0.00035,
            ),
        )
        self.assertEqual(create_pack.status_code, 201, create_pack.text)
        pack_payload = create_pack.json()
        pack_db_id = pack_payload["id"]

        create_project = self.client.post(
            "/api/projects",
            headers=self.admin_headers,
            json={"name": "phase11-project-pack", "description": "phase11", "domain_pack_id": pack_db_id},
        )
        self.assertEqual(create_project.status_code, 201, create_project.text)
        project_payload = create_project.json()
        project_id = project_payload["id"]
        self.assertEqual(project_payload["domain_pack_id"], pack_db_id)
        self.assertIsNotNone(project_payload["domain_profile_id"])

        runtime = self.client.get(
            f"/api/projects/{project_id}/domain-runtime",
            headers=self.admin_headers,
        )
        self.assertEqual(runtime.status_code, 200, runtime.text)
        runtime_payload = runtime.json()
        self.assertEqual(runtime_payload.get("domain_pack_applied"), "support-pack-v1")
        self.assertEqual(runtime_payload.get("domain_pack_source"), "project")
        self.assertEqual(runtime_payload.get("domain_profile_applied"), "support-domain-v1")
        self.assertEqual(runtime_payload.get("domain_profile_source"), "project")

        effective = runtime_payload.get("effective_contract", {})
        self.assertEqual(effective.get("training_defaults", {}).get("batch_size"), 10)
        self.assertAlmostEqual(
            float(effective.get("training_defaults", {}).get("learning_rate", 0)),
            0.00035,
            places=8,
        )
        self.assertAlmostEqual(
            float(effective.get("dataset_split", {}).get("train", 0)),
            0.74,
            places=6,
        )

        preview = self.client.post(
            f"/api/projects/{project_id}/training/experiments/effective-config",
            headers=self.admin_headers,
            json={"config": {"base_model": "microsoft/phi-2"}},
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        preview_payload = preview.json()
        self.assertEqual(preview_payload.get("domain_pack_applied"), "support-pack-v1")
        self.assertEqual(preview_payload.get("domain_profile_applied"), "support-domain-v1")
        self.assertEqual(preview_payload.get("resolved_training_config", {}).get("batch_size"), 10)
        self.assertAlmostEqual(
            float(preview_payload.get("resolved_training_config", {}).get("learning_rate", 0)),
            0.00035,
            places=8,
        )

    def test_domain_pack_hooks_catalog_and_eval_hook(self):
        catalog = self.client.get("/api/domain-packs/hooks/catalog", headers=self.admin_headers)
        self.assertEqual(catalog.status_code, 200, catalog.text)
        catalog_payload = catalog.json()
        self.assertIn("normalizers", catalog_payload)
        self.assertIn("validators", catalog_payload)
        self.assertIn("evaluators", catalog_payload)
        self.assertIn("pass-rate-band-evaluator", catalog_payload.get("evaluators", {}))
        self.assertIn("caps-normalizer", catalog_payload.get("normalizers", {}))
        self.assertIn("target-ratio-validator", catalog_payload.get("validators", {}))
        self.assertIn("confidence-band-evaluator", catalog_payload.get("evaluators", {}))
        self.assertIn(
            "app.plugins.domain_hooks.example_hooks",
            catalog_payload.get("plugin_modules_loaded", []),
        )

        reload_resp = self.client.post(
            "/api/domain-packs/hooks/reload",
            headers=self.admin_headers,
        )
        self.assertEqual(reload_resp.status_code, 200, reload_resp.text)
        reload_payload = reload_resp.json()
        self.assertIn("reload", reload_payload)
        self.assertIn("catalog", reload_payload)

        create_profile = self.client.post(
            "/api/domain-profiles",
            headers=self.admin_headers,
            json=_sample_contract("hooks-profile-v1", "Hooks Profile"),
        )
        self.assertEqual(create_profile.status_code, 201, create_profile.text)

        create_pack = self.client.post(
            "/api/domain-packs",
            headers=self.admin_headers,
            json=_sample_pack(
                "hooks-pack-v1",
                "Hooks Pack",
                default_profile_id="hooks-profile-v1",
                evaluator_hook_id="pass-rate-band-evaluator",
                evaluator_hook_config={"high_threshold": 0.8, "medium_threshold": 0.6},
                normalizer_hook_id="qa-required-normalizer",
                normalizer_hook_config={"require_question": True, "require_answer": True},
                validator_hook_id="min-text-length-validator",
                validator_hook_config={"min_chars": 10},
            ),
        )
        self.assertEqual(create_pack.status_code, 201, create_pack.text)
        pack_db_id = create_pack.json()["id"]

        create_project = self.client.post(
            "/api/projects",
            headers=self.admin_headers,
            json={"name": "phase11-project-hooks", "description": "phase11", "domain_pack_id": pack_db_id},
        )
        self.assertEqual(create_project.status_code, 201, create_project.text)
        project_id = create_project.json()["id"]

        runtime = self.client.get(
            f"/api/projects/{project_id}/domain-runtime",
            headers=self.admin_headers,
        )
        self.assertEqual(runtime.status_code, 200, runtime.text)
        runtime_payload = runtime.json()
        self.assertEqual(runtime_payload.get("pack_hooks", {}).get("evaluator", {}).get("id"), "pass-rate-band-evaluator")
        self.assertEqual(runtime_payload.get("pack_hooks", {}).get("normalizer", {}).get("id"), "qa-required-normalizer")
        self.assertEqual(runtime_payload.get("pack_hooks", {}).get("validator", {}).get("id"), "min-text-length-validator")

        profile_preview = self.client.post(
            f"/api/projects/{project_id}/dataset/profile",
            headers=self.admin_headers,
            json={"dataset_type": "raw", "sample_size": 50},
        )
        self.assertEqual(profile_preview.status_code, 200, profile_preview.text)
        profile_payload = profile_preview.json()
        self.assertEqual(profile_payload.get("domain_hooks", {}).get("normalizer", {}).get("id"), "qa-required-normalizer")
        self.assertEqual(profile_payload.get("domain_hooks", {}).get("validator", {}).get("id"), "min-text-length-validator")
        self.assertEqual(profile_payload.get("validator_report", {}).get("hook_id"), "min-text-length-validator")

        create_exp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            headers=self.admin_headers,
            json={
                "name": "phase11-hooks-exp",
                "config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(create_exp.status_code, 201, create_exp.text)
        experiment_id = create_exp.json()["id"]

        run_eval = self.client.post(
            f"/api/projects/{project_id}/evaluation/run",
            headers=self.admin_headers,
            json={
                "experiment_id": experiment_id,
                "dataset_name": "unit",
                "eval_type": "exact_match",
                "predictions": [
                    {"prediction": "A", "reference": "A"},
                    {"prediction": "B", "reference": "C"},
                ],
            },
        )
        self.assertEqual(run_eval.status_code, 201, run_eval.text)
        eval_payload = run_eval.json()
        self.assertEqual(eval_payload.get("eval_type"), "exact_match")
        self.assertEqual(eval_payload.get("metrics", {}).get("hook_evaluator_id"), "pass-rate-band-evaluator")
        self.assertEqual(eval_payload.get("metrics", {}).get("quality_band"), "low")

    def test_duplicate_domain_profile_auto_versions(self):
        create_profile = self.client.post(
            "/api/domain-profiles",
            headers=self.admin_headers,
            json=_sample_contract("duplicate-domain-v1", "Duplicate Domain"),
        )
        self.assertEqual(create_profile.status_code, 201, create_profile.text)

        duplicate_a = self.client.post(
            "/api/domain-profiles/duplicate-domain-v1/duplicate",
            headers=self.admin_headers,
            json={},
        )
        self.assertEqual(duplicate_a.status_code, 201, duplicate_a.text)
        payload_a = duplicate_a.json()
        self.assertEqual(payload_a["profile_id"], "duplicate-domain-v2")
        self.assertEqual(payload_a["version"], "1.0.1")
        self.assertEqual(payload_a["status"], "draft")

        duplicate_b = self.client.post(
            "/api/domain-profiles/duplicate-domain-v1/duplicate",
            headers=self.admin_headers,
            json={},
        )
        self.assertEqual(duplicate_b.status_code, 201, duplicate_b.text)
        payload_b = duplicate_b.json()
        self.assertEqual(payload_b["profile_id"], "duplicate-domain-v3")
        self.assertEqual(payload_b["version"], "1.0.1")
        self.assertEqual(payload_b["status"], "draft")

    def test_dataset_split_uses_domain_profile_defaults(self):
        create_profile = self.client.post(
            "/api/domain-profiles",
            headers=self.admin_headers,
            json={
                **_sample_contract("ops-domain-v1", "Ops Domain"),
                "dataset_split": {"train": 0.7, "val": 0.2, "test": 0.1, "seed": 99},
                "training_defaults": {"chat_template": "chatml"},
            },
        )
        self.assertEqual(create_profile.status_code, 201, create_profile.text)

        create_project = self.client.post(
            "/api/projects",
            headers=self.admin_headers,
            json={"name": "phase11-project-split", "description": "phase11"},
        )
        self.assertEqual(create_project.status_code, 201, create_project.text)
        project_id = create_project.json()["id"]

        assign = self.client.put(
            f"/api/projects/{project_id}/domain-profile",
            headers=self.admin_headers,
            json={"profile_id": "ops-domain-v1"},
        )
        self.assertEqual(assign.status_code, 200, assign.text)

        mocked_split = AsyncMock(
            return_value={"status": "ok", "splits": {"train": 7, "val": 2, "test": 1}}
        )
        with patch("app.api.dataset.split_dataset", mocked_split):
            resp = self.client.post(
                f"/api/projects/{project_id}/dataset/split",
                headers=self.admin_headers,
                json={},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload.get("domain_profile_applied"), "ops-domain-v1")
        self.assertEqual(payload.get("resolved_split_config", {}).get("train_ratio"), 0.7)
        self.assertEqual(payload.get("resolved_split_config", {}).get("val_ratio"), 0.2)
        self.assertEqual(payload.get("resolved_split_config", {}).get("test_ratio"), 0.1)
        self.assertEqual(payload.get("resolved_split_config", {}).get("seed"), 99)
        self.assertEqual(payload.get("resolved_split_config", {}).get("chat_template"), "chatml")
        self.assertEqual(payload.get("profile_split_defaults", {}).get("train_ratio"), 0.7)
        self.assertEqual(payload.get("profile_split_defaults", {}).get("chat_template"), "chatml")
        self.assertCountEqual(
            payload.get("profile_defaults_applied", []),
            ["train_ratio", "val_ratio", "test_ratio", "seed", "chat_template"],
        )
        self.assertEqual(mocked_split.await_count, 1)
        kwargs = mocked_split.await_args.kwargs
        self.assertAlmostEqual(kwargs["train_ratio"], 0.7, places=6)
        self.assertAlmostEqual(kwargs["val_ratio"], 0.2, places=6)
        self.assertAlmostEqual(kwargs["test_ratio"], 0.1, places=6)
        self.assertEqual(kwargs["seed"], 99)
        self.assertEqual(kwargs["chat_template"], "chatml")

        preview = self.client.post(
            f"/api/projects/{project_id}/dataset/split/effective-config",
            headers=self.admin_headers,
            json={},
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        preview_payload = preview.json()
        self.assertEqual(preview_payload.get("domain_profile_applied"), "ops-domain-v1")
        self.assertEqual(preview_payload.get("resolved_split_config", {}).get("train_ratio"), 0.7)
        self.assertEqual(preview_payload.get("resolved_split_config", {}).get("chat_template"), "chatml")
        self.assertCountEqual(
            preview_payload.get("profile_defaults_applied", []),
            ["train_ratio", "val_ratio", "test_ratio", "seed", "chat_template"],
        )

    def test_training_experiment_uses_domain_profile_defaults(self):
        create_profile = self.client.post(
            "/api/domain-profiles",
            headers=self.admin_headers,
            json={
                **_sample_contract("training-domain-v1", "Training Domain"),
                "training_defaults": {
                    "training_mode": "orpo",
                    "chat_template": "phi3",
                    "num_epochs": 5,
                    "batch_size": 12,
                    "learning_rate": 0.0005,
                    "use_lora": False,
                },
            },
        )
        self.assertEqual(create_profile.status_code, 201, create_profile.text)

        create_project = self.client.post(
            "/api/projects",
            headers=self.admin_headers,
            json={"name": "phase11-project-training", "description": "phase11"},
        )
        self.assertEqual(create_project.status_code, 201, create_project.text)
        project_id = create_project.json()["id"]

        assign = self.client.put(
            f"/api/projects/{project_id}/domain-profile",
            headers=self.admin_headers,
            json={"profile_id": "training-domain-v1"},
        )
        self.assertEqual(assign.status_code, 200, assign.text)

        create_exp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            headers=self.admin_headers,
            json={
                "name": "phase11-exp",
                "config": {
                    "base_model": "microsoft/phi-2"
                },
            },
        )
        self.assertEqual(create_exp.status_code, 201, create_exp.text)
        exp_payload = create_exp.json()
        self.assertEqual(exp_payload["training_mode"], "orpo")
        self.assertEqual(exp_payload["config"]["chat_template"], "phi3")
        self.assertEqual(exp_payload["config"]["num_epochs"], 5)
        self.assertEqual(exp_payload["config"]["batch_size"], 12)
        self.assertAlmostEqual(float(exp_payload["config"]["learning_rate"]), 0.0005, places=8)
        self.assertFalse(exp_payload["config"]["use_lora"])
        self.assertEqual(exp_payload.get("domain_profile_applied"), "training-domain-v1")
        self.assertEqual(exp_payload.get("profile_training_defaults", {}).get("chat_template"), "phi3")
        self.assertEqual(exp_payload.get("resolved_training_config", {}).get("chat_template"), "phi3")
        self.assertEqual(exp_payload.get("resolved_training_config", {}).get("training_mode"), "orpo")
        self.assertCountEqual(
            exp_payload.get("profile_defaults_applied", []),
            ["training_mode", "chat_template", "num_epochs", "batch_size", "learning_rate", "use_lora"],
        )

        preview = self.client.post(
            f"/api/projects/{project_id}/training/experiments/effective-config",
            headers=self.admin_headers,
            json={"config": {"base_model": "microsoft/phi-2"}},
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        preview_payload = preview.json()
        self.assertEqual(preview_payload.get("domain_profile_applied"), "training-domain-v1")
        self.assertEqual(preview_payload.get("resolved_training_mode"), "orpo")
        self.assertEqual(preview_payload.get("resolved_training_config", {}).get("chat_template"), "phi3")
        self.assertFalse(preview_payload.get("resolved_training_config", {}).get("use_lora"))
        self.assertCountEqual(
            preview_payload.get("profile_defaults_applied", []),
            ["training_mode", "chat_template", "num_epochs", "batch_size", "learning_rate", "use_lora"],
        )


if __name__ == "__main__":
    unittest.main()
