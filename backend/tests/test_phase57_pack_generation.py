"""Phase 57 tests — auto-generate starter eval pack from blueprint + dataset + adapter (P9).

Exercises `POST /projects/{project_id}/evaluation/packs/generate`:

- default resolution (no ids → latest blueprint, latest dataset, no adapter)
- task-family-driven gates (classification → macro_f1 / accuracy)
- inverted metrics (hallucination_rate → lte operator)
- safety compliance notes → auto-added safety_pass_rate gate + safety rubric
- include_judge_rubric toggle
- provenance echoes all resolved inputs
- 404 for missing project blueprint
- 404 when blueprint belongs to a different project
- sampling plan scales with dataset row count
- adapter.task_profile overrides blueprint.task_family
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase57_pack_generation.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase57_pack_generation_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"
os.environ["DOMAIN_BLUEPRINT_ENABLE_LLM_ENRICHMENT"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.dataset_adapter_definition import DatasetAdapterDefinition


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class Phase57PackGenerationTests(unittest.TestCase):
    @classmethod
    def _cleanup_artifacts(cls):
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        for suffix in ("", "-shm", "-wal"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_artifacts()

    # -- helpers ----------------------------------------------------------

    def _create_project_with_brief(self, brief: str, name: str = "phase57") -> int:
        unique = f"{name}-{uuid.uuid4().hex[:8]}"
        resp = self.client.post(
            "/api/projects",
            json={
                "name": unique,
                "description": "phase57 pack generation",
                "beginner_mode": True,
                "brief_text": brief,
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_project_without_brief(self, name: str = "phase57-bare") -> int:
        unique = f"{name}-{uuid.uuid4().hex[:8]}"
        resp = self.client.post(
            "/api/projects",
            json={"name": unique, "description": "phase57 bare"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _save_blueprint(
        self,
        project_id: int,
        blueprint: dict,
        source: str = "phase57",
    ) -> int:
        resp = self.client.post(
            f"/api/projects/{project_id}/domain-blueprints",
            json={
                "blueprint": blueprint,
                "source": source,
                "brief_text": blueprint.get("problem_statement") or "phase57",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["revision"]["id"])

    async def _insert_dataset(
        self,
        project_id: int,
        *,
        name: str,
        record_count: int,
        dataset_type: DatasetType = DatasetType.RAW,
    ) -> int:
        async with async_session_factory() as db:
            dataset = Dataset(
                project_id=project_id,
                name=name,
                dataset_type=dataset_type,
                record_count=record_count,
                file_path=f"{TEST_DATA_DIR.as_posix()}/{name}.jsonl",
                metadata_={},
            )
            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)
            return int(dataset.id)

    async def _insert_adapter(
        self,
        project_id: int | None,
        *,
        name: str,
        task_profile: str | None = None,
        field_mapping: dict | None = None,
    ) -> int:
        async with async_session_factory() as db:
            adapter = DatasetAdapterDefinition(
                project_id=project_id,
                adapter_name=name,
                version=1,
                status="active",
                source_type="raw",
                base_adapter_id="default-canonical",
                task_profile=task_profile,
                field_mapping=dict(field_mapping or {}),
                adapter_config={},
                output_contract={},
                schema_profile={},
                inference_summary={},
                validation_report={},
                export_template={},
            )
            db.add(adapter)
            await db.commit()
            await db.refresh(adapter)
            return int(adapter.id)

    def _generate(self, project_id: int, **body) -> dict:
        resp = self.client.post(
            f"/api/projects/{project_id}/evaluation/packs/generate",
            json=body,
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    # -- tests ------------------------------------------------------------

    def test_generate_basic_pack_from_brief_bootstrap(self):
        project_id = self._create_project_with_brief(
            "Build a support assistant that answers FAQ questions from customer tickets."
        )
        pack = self._generate(project_id)

        self.assertEqual(pack["owner"], "brewslm.pack_generator")
        self.assertIn("auto_generated", pack["tags"])
        self.assertTrue(pack["pack_id"].startswith("evalpack.generated."))
        self.assertGreaterEqual(len(pack["task_specs"]), 1)
        self.assertGreaterEqual(len(pack["gates"]), 1)
        self.assertIsInstance(pack["rubric_prompts"], list)
        self.assertGreaterEqual(len(pack["rubric_prompts"]), 1)
        self.assertEqual(pack["rubric_prompts"][0]["rubric_id"], "quality_helpfulness")
        self.assertIn("gold_set_sampling_plan", pack)
        self.assertIn("provenance", pack)
        self.assertEqual(pack["provenance"]["adapter_id"], None)

    def test_classification_task_produces_label_gates_and_stratified_plan(self):
        project_id = self._create_project_with_brief(
            "Classify support tickets into urgency categories.",
            name="phase57-classify",
        )
        dataset_id = asyncio.run(
            self._insert_dataset(project_id, name="phase57-classify-raw", record_count=500)
        )
        adapter_id = asyncio.run(
            self._insert_adapter(
                project_id,
                name="phase57-classify-adapter",
                task_profile="classification",
                field_mapping={"label": "category"},
            )
        )

        pack = self._generate(project_id, dataset_id=dataset_id, adapter_id=adapter_id)

        self.assertEqual(pack["default_task_profile"], "classification")
        gate_metrics = {gate["metric_id"] for gate in pack["gates"]}
        self.assertIn("macro_f1", gate_metrics)
        self.assertIn("accuracy", gate_metrics)

        plan = pack["gold_set_sampling_plan"]
        self.assertEqual(plan["strategy"], "stratified")
        self.assertEqual(plan["stratify_by"], "category")
        self.assertEqual(plan["coverage_goals"]["per_class_min"], 10)
        # 10% of 500 = 50, clamped within [20, 200].
        self.assertEqual(plan["target_size"], 50)
        self.assertEqual(plan["source"]["dataset_id"], dataset_id)

    def test_inverted_metric_gets_lte_gate(self):
        project_id = self._create_project_without_brief(name="phase57-inverted")
        self._save_blueprint(
            project_id,
            {
                "domain_name": "Support Ops",
                "problem_statement": "Answer support questions with bounded hallucinations.",
                "target_user_persona": "Support agents",
                "task_family": "qa",
                "input_modality": "text",
                "success_metrics": [
                    {
                        "metric_id": "hallucination_rate",
                        "label": "Hallucination Rate",
                        "target": "<= 0.05",
                        "why_it_matters": "Cap fabricated answers.",
                    },
                ],
                "confidence_score": 0.8,
            },
        )

        pack = self._generate(project_id)
        halluc_gate = next(
            (g for g in pack["gates"] if g["metric_id"] == "hallucination_rate"),
            None,
        )
        self.assertIsNotNone(halluc_gate, pack["gates"])
        self.assertEqual(halluc_gate["operator"], "lte")
        self.assertAlmostEqual(halluc_gate["threshold"], 0.05)
        self.assertTrue(halluc_gate["gate_id"].startswith("max_"))

    def test_safety_compliance_notes_auto_add_gate_and_rubric(self):
        project_id = self._create_project_without_brief(name="phase57-safety")
        self._save_blueprint(
            project_id,
            {
                "domain_name": "Clinical Assistant",
                "problem_statement": "Answer clinical questions without advice that bypasses clinicians.",
                "target_user_persona": "Clinical operations",
                "task_family": "qa",
                "input_modality": "text",
                "safety_compliance_notes": [
                    "HIPAA-sensitive data must not be surfaced in responses.",
                    "Do not suggest medication dosages directly.",
                ],
                "success_metrics": [
                    {
                        "metric_id": "answer_correctness",
                        "label": "Answer Correctness",
                        "target": ">= 0.82",
                        "why_it_matters": "Keep clinical answers grounded.",
                    },
                ],
                "confidence_score": 0.75,
            },
        )

        pack = self._generate(project_id)
        gate_metrics = {gate["metric_id"] for gate in pack["gates"]}
        self.assertIn("safety_pass_rate", gate_metrics)

        rubric_ids = {rubric["rubric_id"] for rubric in pack["rubric_prompts"]}
        self.assertIn("safety_compliance", rubric_ids)
        safety_rubric = next(
            r for r in pack["rubric_prompts"] if r["rubric_id"] == "safety_compliance"
        )
        self.assertIn("HIPAA-sensitive", safety_rubric["prompt_template"])

    def test_include_judge_rubric_false_omits_rubric_prompts(self):
        project_id = self._create_project_with_brief(
            "Summarize customer tickets into a short description.",
            name="phase57-norubric",
        )
        pack = self._generate(project_id, include_judge_rubric=False)
        self.assertEqual(pack["rubric_prompts"], [])

    def test_provenance_echoes_resolved_inputs(self):
        project_id = self._create_project_with_brief(
            "Extract parties and effective dates from legal contracts.",
            name="phase57-provenance",
        )
        dataset_id = asyncio.run(
            self._insert_dataset(project_id, name="phase57-prov-raw", record_count=120)
        )
        adapter_id = asyncio.run(
            self._insert_adapter(
                project_id,
                name="phase57-prov-adapter",
                task_profile="structured_extraction",
                field_mapping={},
            )
        )

        pack = self._generate(
            project_id,
            dataset_id=dataset_id,
            adapter_id=adapter_id,
        )
        prov = pack["provenance"]
        self.assertEqual(prov["dataset_id"], dataset_id)
        self.assertEqual(prov["adapter_id"], adapter_id)
        self.assertEqual(prov["blueprint_version"], 1)
        self.assertIsNotNone(prov["generated_at"])
        self.assertEqual(pack["default_task_profile"], "structured_extraction")

    def test_404_when_project_has_no_blueprint(self):
        project_id = self._create_project_without_brief(name="phase57-noblueprint")
        resp = self.client.post(
            f"/api/projects/{project_id}/evaluation/packs/generate",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertIn("blueprint", resp.json().get("detail", ""))

    def test_404_when_blueprint_id_belongs_to_other_project(self):
        project_a = self._create_project_with_brief(
            "Classify returns as refund or repair.",
            name="phase57-cross-a",
        )
        project_b = self._create_project_with_brief(
            "Classify feedback as positive, neutral, or negative.",
            name="phase57-cross-b",
        )
        # Look up project_b's blueprint id and try to generate for project_a with it.
        revisions = self.client.get(f"/api/projects/{project_b}/domain-blueprints")
        self.assertEqual(revisions.status_code, 200, revisions.text)
        foreign_id = int(revisions.json()["revisions"][0]["id"])

        resp = self.client.post(
            f"/api/projects/{project_a}/evaluation/packs/generate",
            json={"blueprint_id": foreign_id},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertIn("blueprint", resp.json().get("detail", ""))

    def test_sampling_plan_scales_with_dataset_record_count(self):
        project_id = self._create_project_with_brief(
            "Answer questions about product specifications.",
            name="phase57-size",
        )
        # Big dataset → capped at 200.
        big_dataset = asyncio.run(
            self._insert_dataset(project_id, name="phase57-big", record_count=4000)
        )
        pack_big = self._generate(project_id, dataset_id=big_dataset)
        self.assertEqual(pack_big["gold_set_sampling_plan"]["target_size"], 200)

        # Small dataset → floored at 20.
        small_dataset = asyncio.run(
            self._insert_dataset(project_id, name="phase57-small", record_count=30)
        )
        pack_small = self._generate(project_id, dataset_id=small_dataset)
        self.assertEqual(pack_small["gold_set_sampling_plan"]["target_size"], 20)


if __name__ == "__main__":
    unittest.main()
