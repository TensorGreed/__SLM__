"""Phase 10 tests: model registry lifecycle and project secrets."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"


class Phase10RegistrySecretsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmp.name)

        import app.models  # noqa: F401
        from app.database import Base
        from app.models.experiment import EvalResult, Experiment, ExperimentStatus
        from app.models.domain_profile import DomainProfile, DomainProfileStatus
        from app.models.project import Project
        from app.models.registry import RegistryStage
        from app.models.secret import ProjectSecret
        from app.services.registry_service import (
            mark_model_deployed,
            promote_model,
            register_model,
        )
        from app.services.secret_service import (
            get_project_secret_value,
            list_project_secrets,
            upsert_project_secret,
        )

        self.Base = Base
        self.EvalResult = EvalResult
        self.Experiment = Experiment
        self.ExperimentStatus = ExperimentStatus
        self.Project = Project
        self.DomainProfile = DomainProfile
        self.DomainProfileStatus = DomainProfileStatus
        self.RegistryStage = RegistryStage
        self.ProjectSecret = ProjectSecret
        self.mark_model_deployed = mark_model_deployed
        self.promote_model = promote_model
        self.register_model = register_model
        self.get_project_secret_value = get_project_secret_value
        self.list_project_secrets = list_project_secrets
        self.upsert_project_secret = upsert_project_secret

        db_path = self.tmp_root / "phase10.db"
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", future=True)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async with self.engine.begin() as conn:
            await conn.run_sync(self.Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def _seed_project_and_experiment(self, db: AsyncSession, name_suffix: str = "") -> tuple[Project, Experiment]:
        project = self.Project(
            name=f"phase10-{name_suffix or uuid4().hex[:8]}",
            description="phase10 project",
            base_model_name="microsoft/phi-2",
        )
        db.add(project)
        await db.flush()

        exp = self.Experiment(
            project_id=project.id,
            name=f"phase10-exp-{uuid4().hex[:8]}",
            status=self.ExperimentStatus.COMPLETED,
            base_model="microsoft/phi-2",
            output_dir=str(self.tmp_root / "exp_output" / str(project.id)),
        )
        db.add(exp)
        await db.flush()
        return project, exp

    async def test_project_secret_roundtrip_masks_and_decrypts(self):
        async with self.session_factory() as db:
            project, _ = await self._seed_project_and_experiment(db)
            secret = await self.upsert_project_secret(
                db=db,
                project_id=project.id,
                provider="huggingface",
                key_name="token",
                value="hf_super_secret_token_12345",
            )
            await db.commit()

            self.assertNotIn("hf_super_secret_token_12345", secret.encrypted_value)
            self.assertTrue(secret.value_hint.startswith("hf"))

            listed = await self.list_project_secrets(db, project.id)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].provider, "huggingface")

            value = await self.get_project_secret_value(db, project.id, "huggingface", "token")
            self.assertEqual(value, "hf_super_secret_token_12345")

            row = await db.get(self.ProjectSecret, secret.id)
            self.assertIsNotNone(row.last_used_at)

    async def test_registry_promotion_gates_and_deploy(self):
        async with self.session_factory() as db:
            project, exp = await self._seed_project_and_experiment(db, name_suffix="registry")
            db.add(
                self.EvalResult(
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="exact_match",
                    metrics={"exact_match": 0.82},
                    pass_rate=0.82,
                )
            )
            db.add(
                self.EvalResult(
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="f1",
                    metrics={"f1": 0.79},
                    pass_rate=0.79,
                )
            )
            db.add(
                self.EvalResult(
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="llm_judge",
                    metrics={"pass_rate": 0.85},
                    pass_rate=0.85,
                )
            )
            db.add(
                self.EvalResult(
                    experiment_id=exp.id,
                    dataset_name="safety",
                    eval_type="safety",
                    metrics={"pass_rate": 0.97},
                    pass_rate=0.97,
                )
            )
            await db.flush()

            entry = await self.register_model(db, project.id, exp.id)
            self.assertEqual(entry.stage.value, "candidate")

            promoted, report = await self.promote_model(
                db=db,
                project_id=project.id,
                model_id=entry.id,
                target_stage=self.RegistryStage.PRODUCTION,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(promoted.stage.value, "production")

            deployed = await self.mark_model_deployed(
                db=db,
                project_id=project.id,
                model_id=promoted.id,
                environment="production",
                endpoint_url="https://inference.example.com/model-a",
                notes="initial production deploy",
            )
            self.assertEqual(deployed.deployment_status.value, "deployed")
            self.assertEqual((deployed.deployment or {}).get("environment"), "production")

            await db.commit()

    async def test_registry_gate_failure_without_force(self):
        async with self.session_factory() as db:
            project, exp = await self._seed_project_and_experiment(db, name_suffix="gates")
            db.add(
                self.EvalResult(
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="exact_match",
                    metrics={"exact_match": 0.2},
                    pass_rate=0.2,
                )
            )
            await db.flush()

            entry = await self.register_model(db, project.id, exp.id)

            with self.assertRaises(ValueError):
                await self.promote_model(
                    db=db,
                    project_id=project.id,
                    model_id=entry.id,
                    target_stage=self.RegistryStage.PRODUCTION,
                    force=False,
                )

    async def test_registry_uses_domain_profile_gate_defaults(self):
        async with self.session_factory() as db:
            profile = self.DomainProfile(
                profile_id="strict-registry-v1",
                version="1.0.0",
                display_name="Strict Registry",
                description="Strict gate defaults",
                owner="qa",
                status=self.DomainProfileStatus.ACTIVE,
                schema_ref="slm.domain-profile/v1",
                contract={
                    "$schema": "slm.domain-profile/v1",
                    "profile_id": "strict-registry-v1",
                    "version": "1.0.0",
                    "display_name": "Strict Registry",
                    "registry_gates": {
                        "to_staging": {
                            "min_metrics": {
                                "f1": 0.9,
                            }
                        }
                    },
                },
                is_system=False,
            )
            db.add(profile)
            await db.flush()

            project, exp = await self._seed_project_and_experiment(db, name_suffix="profile-gates")
            project.domain_profile_id = profile.id

            db.add(
                self.EvalResult(
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="exact_match",
                    metrics={"exact_match": 0.95},
                    pass_rate=0.95,
                )
            )
            db.add(
                self.EvalResult(
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="f1",
                    metrics={"f1": 0.82},
                    pass_rate=0.82,
                )
            )
            db.add(
                self.EvalResult(
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="llm_judge",
                    metrics={"pass_rate": 0.95},
                    pass_rate=0.95,
                )
            )
            await db.flush()

            entry = await self.register_model(db, project.id, exp.id)

            with self.assertRaises(ValueError):
                await self.promote_model(
                    db=db,
                    project_id=project.id,
                    model_id=entry.id,
                    target_stage=self.RegistryStage.STAGING,
                    force=False,
                )


if __name__ == "__main__":
    unittest.main()
