"""Playground "experiment" provider — chat with a trained run in-process
(Wave 1b).

Pins:
  * API validation: a run must be picked, belong to the project, be a real
    completed training run with weights (not a baseline / running /
    simulated row);
  * the chat + stream routes hand the run's checkpoint path to
    ``local_chat_service`` and label the session with the run;
  * real generation (SmolLM2 + a LoRA adapter): the in-process reply is the
    same greedy output as loading base+adapter by hand, and streaming
    deltas concatenate to the final reply. Skips without torch / cached
    model.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode

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


def _weights_dir(root: Path) -> str:
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": REAL_MODEL}))
    (model_dir / "adapter_model.safetensors").write_bytes(b"\0" * 8)
    return str(root)


async def _seed_experiment(project_id: int, **fields) -> int:
    async with async_session_factory() as db:
        exp = Experiment(
            project_id=project_id,
            name=fields.pop("name", f"run-{uuid4().hex[:6]}"),
            status=fields.pop("status", ExperimentStatus.COMPLETED),
            training_mode=TrainingMode.SFT,
            base_model=fields.pop("base_model", REAL_MODEL),
            **fields,
        )
        db.add(exp)
        await db.commit()
        return exp.id


class PlaygroundRunApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        resp = client.post("/api/projects", json={"name": f"pg-run-{uuid4().hex[:6]}", "description": ""})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.project_id = int(resp.json()["id"])

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, **fields) -> int:
        return asyncio.run(_seed_experiment(self.project_id, **fields))

    def _chat(self, **body):
        payload = {"provider": "experiment", "messages": [{"role": "user", "content": "hi"}], **body}
        return client.post(f"/api/projects/{self.project_id}/training/playground/chat", json=payload)

    def test_requires_a_run(self):
        resp = self._chat()
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Pick a training run", resp.text)

    def test_rejects_ineligible_runs(self):
        cases = {
            "baseline": dict(output_dir=None, config={"is_baseline": True}),
            "only completed runs": dict(status=ExperimentStatus.RUNNING, output_dir=_weights_dir(self.root / "r")),
            "no saved model weights": dict(output_dir=str(self.root / "empty")),
        }
        for needle, fields in cases.items():
            with self.subTest(needle):
                resp = self._chat(experiment_id=self._seed(**fields))
                self.assertEqual(resp.status_code, 400, resp.text)
                self.assertIn(needle, resp.text)

    def test_other_projects_run_is_404(self):
        other = client.post("/api/projects", json={"name": f"pg-other-{uuid4().hex[:6]}", "description": ""})
        other_run = asyncio.run(_seed_experiment(int(other.json()["id"]), output_dir=_weights_dir(self.root / "o")))
        self.assertEqual(self._chat(experiment_id=other_run).status_code, 404)

    def test_chat_routes_run_checkpoint_to_local_chat(self):
        run_id = self._seed(name="ticket-router", output_dir=_weights_dir(self.root / "ok"))
        calls = []

        def fake_generate(model_ref, messages, **kwargs):
            calls.append((model_ref, messages, kwargs))
            return {"reply": "Routed to Billing.", "usage": None, "finish_reason": "stop"}

        with mock.patch("app.services.local_chat_service.generate_reply", fake_generate):
            resp = self._chat(experiment_id=run_id)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["reply"], "Routed to Billing.")
        self.assertEqual(body["resolved_provider"], "experiment")
        self.assertEqual(body["requested_model_name"], f"run #{run_id} · ticket-router")
        model_ref, messages, kwargs = calls[0]
        self.assertEqual(model_ref, str(self.root / "ok" / "model"))
        self.assertEqual(messages[-1]["content"], "hi")
        self.assertEqual(kwargs["base_model_hint"], REAL_MODEL)

    def test_stream_route_streams_from_local_chat(self):
        run_id = self._seed(output_dir=_weights_dir(self.root / "s"))

        async def fake_stream(model_ref, messages, **kwargs):
            yield {"type": "delta", "content": "Routed "}
            yield {"type": "delta", "content": "to Billing."}
            yield {"type": "final", "reply": "Routed to Billing.", "usage": None,
                   "finish_reason": "stop", "latency_ms": 1.0}

        with mock.patch("app.services.local_chat_service.astream_reply", fake_stream):
            resp = client.post(
                f"/api/projects/{self.project_id}/training/playground/chat/stream",
                json={"provider": "experiment", "experiment_id": run_id,
                      "messages": [{"role": "user", "content": "hi"}]},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        events = [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")]
        self.assertEqual([e["type"] for e in events], ["delta", "delta", "final"])
        self.assertEqual(events[-1]["reply"], "Routed to Billing.")
        self.assertEqual(events[-1]["resolved_provider"], "experiment")


def _real_skip_reason() -> str | None:
    if os.environ.get("BREWSLM_SKIP_REAL_TRAINING"):
        return "BREWSLM_SKIP_REAL_TRAINING set"
    if not os.environ.get("BREWSLM_REAL_TRAINING"):
        try:
            import torch

            has_gpu = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            has_gpu = False
        if not has_gpu:
            # Real fine-tunes on CPU (+ a model download) take many minutes —
            # opt in with BREWSLM_REAL_TRAINING=1; GPU boxes run them by default.
            return "needs a GPU or BREWSLM_REAL_TRAINING=1"
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return f"ML deps unavailable: {exc}"
    try:
        AutoTokenizer.from_pretrained(REAL_MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        return f"{REAL_MODEL} not in the local HF cache"
    return None


_SKIP = _real_skip_reason()


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class LocalChatRealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls._tmp = tempfile.TemporaryDirectory()
        cls.adapter_dir = Path(cls._tmp.name) / "model"
        torch.manual_seed(0)
        model = get_peft_model(
            AutoModelForCausalLM.from_pretrained(REAL_MODEL),
            LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        )
        for name, param in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.02)
        model.save_pretrained(str(cls.adapter_dir))
        AutoTokenizer.from_pretrained(REAL_MODEL).save_pretrained(str(cls.adapter_dir))
        cls.messages = [{"role": "user", "content": "Where should a refund ticket go?"}]

    @classmethod
    def tearDownClass(cls):
        from app.services import local_chat_service

        local_chat_service.unload()
        cls._tmp.cleanup()

    def test_reply_matches_manual_base_plus_adapter_greedy(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from app.services import local_chat_service

        out = local_chat_service.generate_reply(
            str(self.adapter_dir), self.messages, max_tokens=24, temperature=0.0
        )
        tok = AutoTokenizer.from_pretrained(str(self.adapter_dir))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        manual = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(REAL_MODEL), str(self.adapter_dir)
        ).to(device).eval()
        prompt = tok.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = manual.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        expected = tok.decode(gen[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        self.assertTrue(out["reply"])
        self.assertEqual(out["reply"], expected)
        self.assertIn(out["finish_reason"], {"stop", "length"})

    def test_stream_deltas_concatenate_to_final_reply(self):
        from app.services import local_chat_service

        async def _collect():
            return [
                event
                async for event in local_chat_service.astream_reply(
                    str(self.adapter_dir), self.messages, max_tokens=24, temperature=0.0
                )
            ]

        events = asyncio.run(_collect())
        deltas = "".join(e["content"] for e in events if e["type"] == "delta")
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(deltas.strip(), events[-1]["reply"])
        self.assertTrue(events[-1]["reply"])


if __name__ == "__main__":
    unittest.main()
