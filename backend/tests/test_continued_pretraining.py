"""Continued pretraining for documents-only projects (Wave 2c-1).

Pins:
  * packing: documents join into full blocks (EOS between docs, loss on
    every token), a tiny tail is dropped;
  * a documents-only project (confirmed task shape ``language_modeling``)
    trains as ``domain_pretrain`` unless the caller picked a mode, with CPT
    defaults (all-linear LoRA, r=64, lr 1e-4…) only for fields the caller
    didn't set;
  * held-out eval measures perplexity for plain-text splits (it used to
    raise "No valid evaluation rows"), and the lift summary treats a lower
    perplexity as the improvement;
  * real run (SmolLM2, skipped without torch / cached model): continued
    pretraining through ``run_training`` packs the corpus and lowers
    held-out perplexity on unseen documents about the same domain.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import Base, async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode
from app.models.project import Project
from app.services import evaluation_service
from app.services.continued_pretraining_policy import (
    CPT_DEFAULTS,
    apply_cpt_defaults,
    resolve_training_mode,
)
from app.services.sft_lift_summary_service import _compute_metric_lifts
from scripts.train import _pack_lm_blocks

REAL_MODEL = os.environ.get("BREWSLM_REAL_TRAIN_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


class PackingTests(unittest.TestCase):
    def test_documents_join_into_full_blocks_with_eos(self):
        blocks = _pack_lm_blocks([[1, 2, 3], [4, 5], [6, 7, 8, 9]], block_size=4, eos_id=0)
        # stream: 1 2 3 0 4 5 0 6 7 8 9 0  → 3 full blocks
        self.assertEqual([b["input_ids"] for b in blocks], [[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]])
        self.assertTrue(all(b["labels"] == b["input_ids"] for b in blocks))

    def test_tiny_tail_is_dropped_but_a_short_corpus_is_kept(self):
        blocks = _pack_lm_blocks([[1] * 8], block_size=8, eos_id=0)  # 9 tokens → 8 + a 1-token tail
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(_pack_lm_blocks([[1, 2]], block_size=8, eos_id=0)), 1)


class PolicyTests(unittest.TestCase):
    class _P:
        def __init__(self, profile):
            self.dataset_adapter_preset = {"task_profile": profile} if profile else None

    def test_documents_only_projects_continue_pretraining(self):
        self.assertEqual(
            resolve_training_mode(self._P("language_modeling"), "sft", mode_was_provided=False),
            TrainingMode.DOMAIN_PRETRAIN,
        )
        self.assertEqual(resolve_training_mode(self._P("qa"), None, mode_was_provided=False), TrainingMode.SFT)
        # An explicit choice always wins.
        self.assertEqual(
            resolve_training_mode(self._P("language_modeling"), "sft", mode_was_provided=True),
            TrainingMode.SFT,
        )

    def test_cpt_defaults_fill_only_untouched_fields(self):
        config = {"learning_rate": 3e-5}
        applied = apply_cpt_defaults(config, {"learning_rate"})
        self.assertEqual(config["learning_rate"], 3e-5)
        self.assertEqual(config["target_modules"], "all-linear")
        self.assertNotIn("learning_rate", applied)

    def test_lower_perplexity_is_an_improvement(self):
        rows = _compute_metric_lifts({"perplexity": 40.0, "exact_match": 0.1}, {"perplexity": 12.0, "exact_match": 0.3})
        by_id = {r["metric_id"]: r for r in rows}
        self.assertEqual(by_id["perplexity"]["direction"], "improved")
        self.assertEqual(by_id["exact_match"]["direction"], "improved")
        self.assertEqual(_compute_metric_lifts({"perplexity": 10.0}, {"perplexity": 11.0})[0]["direction"], "regressed")


class CreateExperimentTests(unittest.TestCase):
    def setUp(self):
        resp = client.post("/api/projects", json={"name": f"cpt-{uuid4().hex[:6]}", "description": ""})
        self.pid = int(resp.json()["id"])
        confirm = client.post(f"/api/projects/{self.pid}/task-shape/confirm", json={"task_profile": "language_modeling"})
        self.assertEqual(confirm.status_code, 200, confirm.text)

    def _create(self, config: dict) -> dict:
        resp = client.post(
            f"/api/projects/{self.pid}/training/experiments",
            json={"name": "run", "config": {"base_model": REAL_MODEL, **config}},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_documents_project_defaults_to_cpt_with_cpt_settings(self):
        body = self._create({})
        self.assertEqual(body["training_mode"], "domain_pretrain")
        cfg = body["config"]
        for key, value in CPT_DEFAULTS.items():
            self.assertEqual(cfg[key], value, key)

    def test_explicit_choices_are_kept(self):
        body = self._create({"training_mode": "sft", "learning_rate": 5e-5})
        self.assertEqual(body["training_mode"], "sft")
        body = self._create({"learning_rate": 5e-5})
        self.assertEqual(body["training_mode"], "domain_pretrain")
        self.assertEqual(body["config"]["learning_rate"], 5e-5)


class PerplexityEvalRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'ppl.db'}", future=True)
        self.sf = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        test_file = root / "test.jsonl"
        test_file.write_text("\n".join(json.dumps({"text": f"Passage {i} about warranties."}) for i in range(5)))
        (root / "run" / "model").mkdir(parents=True)
        async with self.sf() as db:
            project = Project(name=f"p-{uuid4().hex[:6]}", description="")
            db.add(project)
            await db.flush()
            db.add(Dataset(project_id=project.id, name="Test Set", dataset_type=DatasetType.TEST, file_path=str(test_file)))
            exp = Experiment(project_id=project.id, name="cpt", status=ExperimentStatus.COMPLETED,
                             training_mode=TrainingMode.DOMAIN_PRETRAIN, base_model=REAL_MODEL,
                             output_dir=str(root / "run"), config={"task_type": "causal_lm"})
            db.add(exp)
            await db.commit()
            self.pid, self.eid = project.id, exp.id

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def test_plain_text_split_is_scored_by_perplexity(self):
        fake = mock.Mock(return_value={"perplexity": 7.5, "mean_nll": 2.01, "bits_per_byte": 0.9,
                                        "eval_tokens": 40, "eval_documents": 5})
        with mock.patch.object(evaluation_service, "_compute_perplexity", fake):
            async with self.sf() as db:
                result = await evaluation_service.run_heldout_evaluation(db, self.pid, self.eid, eval_type="exact_match")
        self.assertEqual(result.eval_type, "perplexity")
        self.assertEqual(result.metrics["perplexity"], 7.5)
        self.assertEqual(result.details["inference"]["reason"], "plain_text_split")
        texts = fake.call_args.args[1]
        self.assertEqual(len(texts), 5)


def _real_skip() -> str | None:
    if os.environ.get("BREWSLM_SKIP_REAL_TRAINING"):
        return "BREWSLM_SKIP_REAL_TRAINING set"
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(REAL_MODEL, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        return f"real training unavailable: {exc}"
    return None


_PRODUCTS = ["Zorbex K-7 blender", "Quillon M2 kettle", "Varnet X9 toaster", "Pellix T4 grill", "Drumo S3 mixer"]
_FACTS = [
    "carries a {n}-month Zorbex Assurance warranty that covers the motor housing",
    "must be descaled with Zorbex CitraClean every {n} weeks to keep the seal intact",
    "ships with the Zorbex Loop charger rated for {n} watts",
    "has a Zorbex FlowLock lid that clicks {n} times when it is sealed",
]


def _manual(rng: random.Random, n: int) -> list[str]:
    docs = []
    for _ in range(n):
        product = rng.choice(_PRODUCTS)
        sentences = [f"The {product} {rng.choice(_FACTS).format(n=rng.randint(2, 30))}." for _ in range(4)]
        docs.append(" ".join(sentences))
    return docs


_SKIP = _real_skip()


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class RealContinuedPretrainingTests(unittest.TestCase):
    def test_cpt_packs_the_corpus_and_lowers_heldout_perplexity(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from scripts.train import run_training

        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_docs, heldout = _manual(rng, 160), _manual(rng, 20)
            (root / "train.jsonl").write_text("\n".join(json.dumps({"text": d}) for d in train_docs))
            config = {
                "training_mode": "domain_pretrain",
                "task_type": "causal_lm",
                **CPT_DEFAULTS,
                "learning_rate": 5e-4,
                "max_seq_length": 256,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "flash_attention": False,
                "observability_enabled": False,
                "save_steps": 10_000,
                "auto_oom_retry": False,
            }
            (root / "cfg.json").write_text(json.dumps(config))
            report = run_training(argparse.Namespace(
                project=0, experiment=0, output=str(root / "run"), base_model=REAL_MODEL,
                config=str(root / "cfg.json"), train_file=str(root / "train.jsonl"), val_file="",
                data_dir=str(root), max_train_samples=0, max_eval_samples=0, seed=7,
            ))
            env = report["runtime_environment"]
            self.assertEqual(env["loss_masking"], "packed_lm")
            self.assertLess(env["packing"]["train_blocks"], env["packing"]["train_rows"])

            base = evaluation_service._compute_perplexity(REAL_MODEL, heldout)
            tuned = evaluation_service._compute_perplexity(report["model_dir"], heldout)
            print(f"held-out perplexity: base {base['perplexity']} → CPT {tuned['perplexity']}")
            self.assertLess(tuned["perplexity"], base["perplexity"] * 0.7, (base, tuned))


if __name__ == "__main__":
    unittest.main()
