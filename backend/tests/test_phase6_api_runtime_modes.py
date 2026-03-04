"""Phase 6 integration tests: strict runtime API fail-fast behavior."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database as database_module
import app.main as main_module
from app.config import settings


class StrictRuntimeApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_settings = {
            "AUTH_ENABLED": settings.AUTH_ENABLED,
            "DB_AUTO_CREATE": settings.DB_AUTO_CREATE,
            "DB_REQUIRE_ALEMBIC_HEAD": settings.DB_REQUIRE_ALEMBIC_HEAD,
            "DATABASE_URL": settings.DATABASE_URL,
            "TRAINING_BACKEND": settings.TRAINING_BACKEND,
            "ALLOW_SIMULATED_TRAINING": settings.ALLOW_SIMULATED_TRAINING,
            "COMPRESSION_BACKEND": settings.COMPRESSION_BACKEND,
            "QUANTIZE_EXTERNAL_CMD": settings.QUANTIZE_EXTERNAL_CMD,
            "ALLOW_SYNTHETIC_DEMO_FALLBACK": settings.ALLOW_SYNTHETIC_DEMO_FALLBACK,
            "TEACHER_MODEL_API_URL": settings.TEACHER_MODEL_API_URL,
        }
        cls._original_engine = database_module.engine
        cls._original_session_factory = database_module.async_session_factory
        cls._original_main_session_factory = main_module.async_session_factory

        # Keep this suite self-contained and deterministic.
        cls._tmp_db_path = Path(tempfile.gettempdir()) / f"phase6_runtime_{uuid4().hex}.db"
        settings.DATABASE_URL = f"sqlite+aiosqlite:///{cls._tmp_db_path.as_posix()}"
        settings.AUTH_ENABLED = False
        settings.DB_AUTO_CREATE = True
        settings.DB_REQUIRE_ALEMBIC_HEAD = False

        database_module.engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            future=True,
        )
        database_module.async_session_factory = async_sessionmaker(
            database_module.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        main_module.async_session_factory = database_module.async_session_factory

        cls._client_cm = TestClient(main_module.app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        asyncio.run(database_module.engine.dispose())
        database_module.engine = cls._original_engine
        database_module.async_session_factory = cls._original_session_factory
        main_module.async_session_factory = cls._original_main_session_factory
        for key, value in cls._original_settings.items():
            setattr(settings, key, value)
        if cls._tmp_db_path.exists():
            cls._tmp_db_path.unlink()

    def _create_project(self) -> int:
        payload = {
            "name": f"phase6-{uuid4().hex[:10]}",
            "description": "phase6 strict runtime test project",
            "base_model_name": "microsoft/phi-2",
        }
        resp = self.client.post("/api/projects", json=payload)
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def _create_experiment(self, project_id: int) -> int:
        payload = {
            "name": f"phase6-exp-{uuid4().hex[:8]}",
            "description": "phase6 strict runtime test experiment",
            "config": {
                "base_model": "microsoft/phi-2",
                "training_mode": "sft",
                "chat_template": "llama3",
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.0002,
                "optimizer": "paged_adamw_8bit",
                "lr_scheduler": "cosine",
                "num_epochs": 1,
                "max_seq_length": 1024,
                "warmup_ratio": 0.03,
                "weight_decay": 0.01,
                "sequence_packing": True,
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "fp16": False,
                "bf16": True,
                "gradient_checkpointing": True,
                "flash_attention": True,
                "save_steps": 50,
                "eval_steps": 50,
                "early_stopping_patience": 3,
                "seed": 42,
            },
        }
        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json=payload,
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def test_training_start_rejects_disallowed_simulation(self):
        prev_backend = settings.TRAINING_BACKEND
        prev_allow_sim = settings.ALLOW_SIMULATED_TRAINING
        try:
            settings.TRAINING_BACKEND = "simulate"
            settings.ALLOW_SIMULATED_TRAINING = False

            project_id = self._create_project()
            exp_id = self._create_experiment(project_id)
            start = self.client.post(
                f"/api/projects/{project_id}/training/experiments/{exp_id}/start"
            )
            self.assertEqual(start.status_code, 400, start.text)
            self.assertIn("Simulated training backend is disabled", start.json()["detail"])
        finally:
            settings.TRAINING_BACKEND = prev_backend
            settings.ALLOW_SIMULATED_TRAINING = prev_allow_sim

    def test_compression_quantize_rejects_missing_external_command(self):
        prev_backend = settings.COMPRESSION_BACKEND
        prev_quantize_cmd = settings.QUANTIZE_EXTERNAL_CMD
        try:
            settings.COMPRESSION_BACKEND = "external"
            settings.QUANTIZE_EXTERNAL_CMD = ""

            resp = self.client.post(
                "/api/projects/999/compression/quantize",
                json={
                    "model_path": "/tmp/nonexistent-model",
                    "bits": 4,
                    "output_format": "gguf",
                },
            )
            self.assertEqual(resp.status_code, 400, resp.text)
            self.assertIn("QUANTIZE_EXTERNAL_CMD is required", resp.json()["detail"])
        finally:
            settings.COMPRESSION_BACKEND = prev_backend
            settings.QUANTIZE_EXTERNAL_CMD = prev_quantize_cmd

    def test_synthetic_generate_rejects_missing_teacher_and_demo_fallback(self):
        prev_allow_demo = settings.ALLOW_SYNTHETIC_DEMO_FALLBACK
        prev_teacher_url = settings.TEACHER_MODEL_API_URL
        try:
            settings.ALLOW_SYNTHETIC_DEMO_FALLBACK = False
            settings.TEACHER_MODEL_API_URL = ""

            resp = self.client.post(
                "/api/projects/999/synthetic/generate",
                json={
                    "source_text": "This source text is intentionally long enough for synthetic generation testing.",
                    "num_pairs": 3,
                    "api_url": "",
                    "api_key": "",
                    "model_name": "llama3",
                },
            )
            self.assertEqual(resp.status_code, 400, resp.text)
            self.assertIn("Teacher model API URL is not configured", resp.json()["detail"])
        finally:
            settings.ALLOW_SYNTHETIC_DEMO_FALLBACK = prev_allow_demo
            settings.TEACHER_MODEL_API_URL = prev_teacher_url


if __name__ == "__main__":
    unittest.main()
