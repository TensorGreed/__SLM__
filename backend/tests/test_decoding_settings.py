"""Decoding settings are explicit, never inherited from the model:

* ``generation_kwargs`` turns off every logits processor a model's
  ``generation_config.json`` can carry (repetition penalty, n-gram block,
  top-k, top-p) — checked on a real (tiny, random) model whose config sets
  ``repetition_penalty=1.1``: greedy output equals the raw argmax chain;
* the held-out eval passes those settings to ``generate()`` and records them
  in the result (``metrics.inference.decoding`` + ``details.inference.decoding``);
* the playground uses the same settings;
* an HF export rewrites ``generation_config.json`` to them (token ids kept)
  and says what it replaced in the manifest and release notes.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus
from app.services.decoding_settings import (
    NEUTRAL_PROCESSORS,
    NEUTRAL_SAMPLING,
    align_exported_generation_config,
    decoding_record,
    generation_kwargs,
    neutral_generation_config,
)

QWEN_INSTRUCT_CONFIG = {
    "bos_token_id": 151643, "do_sample": True, "eos_token_id": [151645, 151643], "pad_token_id": 151643,
    "repetition_penalty": 1.1, "temperature": 0.7, "top_k": 20, "top_p": 0.8, "transformers_version": "5.2.0",
}


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


class KwargsTests(unittest.TestCase):
    def test_greedy_turns_every_processor_off(self):
        kw = generation_kwargs(max_new_tokens=32, temperature=0.0, pad_token_id=5)
        self.assertEqual(kw["do_sample"], False)
        self.assertEqual(kw["pad_token_id"], 5)
        for key, value in NEUTRAL_PROCESSORS.items():
            self.assertEqual(kw[key], value)
        self.assertEqual((kw["repetition_penalty"], kw["num_beams"]), (1.0, 1))
        # no sampling warper runs under greedy; passing them only trips warnings
        self.assertFalse(set(NEUTRAL_SAMPLING) & set(kw))

    def test_sampling_keeps_temperature_only(self):
        kw = generation_kwargs(max_new_tokens=8, temperature=0.7)
        self.assertTrue(kw["do_sample"])
        self.assertEqual(kw["temperature"], 0.7)
        for key, value in {**NEUTRAL_PROCESSORS, **NEUTRAL_SAMPLING}.items():
            self.assertEqual(kw[key], value)
        self.assertNotIn("pad_token_id", kw)

    def test_record_is_the_decoding_subset(self):
        rec = decoding_record(generation_kwargs(max_new_tokens=8, temperature=0.0, pad_token_id=1))
        self.assertEqual(set(rec), {"do_sample", "temperature", *NEUTRAL_PROCESSORS})
        rec = decoding_record(generation_kwargs(max_new_tokens=8, temperature=0.5))
        self.assertEqual(set(rec), {"do_sample", "temperature", *NEUTRAL_PROCESSORS, *NEUTRAL_SAMPLING})


class RealGenerateTests(unittest.TestCase):
    """A random 2-layer GPT-2 on CPU with Qwen-Instruct's generation_config."""

    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, GenerationConfig

        torch.manual_seed(0)
        cfg = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2)
        cls.model = GPT2LMHeadModel(cfg).to(torch.float32).eval()
        cls.model.generation_config = GenerationConfig(
            do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=2, top_k=20, top_p=0.8, min_p=0.5,
            pad_token_id=0, eos_token_id=None,
        )
        cls.prompt = torch.tensor([[3, 7, 7, 3, 9, 3, 7]])

    def _argmax_chain(self, steps):
        import torch

        ids = self.prompt
        with torch.no_grad():
            for _ in range(steps):
                nxt = self.model(ids).logits[0, -1].argmax().view(1, 1)
                ids = torch.cat([ids, nxt], 1)
        return ids[0, self.prompt.shape[1]:].tolist()

    def test_greedy_equals_raw_argmax(self):
        import torch

        with torch.no_grad():
            out = self.model.generate(self.prompt, **generation_kwargs(max_new_tokens=12, temperature=0.0, pad_token_id=0))
        self.assertEqual(out[0, self.prompt.shape[1]:].tolist(), self._argmax_chain(12))

    def test_inherited_penalty_would_have_changed_it(self):
        """Guards the test above: the inherited config does change the output."""
        import torch

        with torch.no_grad():
            out = self.model.generate(self.prompt, max_new_tokens=12, do_sample=False, pad_token_id=0)
        self.assertNotEqual(out[0, self.prompt.shape[1]:].tolist(), self._argmax_chain(12))

    def test_sampling_adds_no_inherited_warper(self):
        """The config sets top_k 20 / top_p 0.8 / min_p 0.5; sampling with the
        explicit settings builds only the temperature warper (plus no-ops)."""
        from transformers.generation.logits_process import (
            MinPLogitsWarper, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
        )

        kw = generation_kwargs(max_new_tokens=4, temperature=0.7)
        cfg = self.model._prepare_generation_config(None, **kw)[0]
        procs = self.model._get_logits_processor(
            generation_config=cfg, input_ids_seq_length=self.prompt.shape[1], encoder_input_ids=self.prompt,
            prefix_allowed_tokens_fn=None, logits_processor=None, device="cpu",
        )
        kinds = {type(p) for p in procs}
        self.assertIn(TemperatureLogitsWarper, kinds)
        self.assertFalse(kinds & {TopKLogitsWarper, TopPLogitsWarper})
        self.assertTrue(all(p.min_p == 0.0 for p in procs if isinstance(p, MinPLogitsWarper)))
        self.assertEqual(len([p for p in procs if not isinstance(p, (TemperatureLogitsWarper, MinPLogitsWarper))]), 0)


class PlaygroundTests(unittest.TestCase):
    def test_playground_uses_the_explicit_settings(self):
        from app.services import local_chat_service

        bundle = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=9))
        self.assertEqual(
            local_chat_service._generation_kwargs(bundle, 40, 0.0),
            generation_kwargs(max_new_tokens=40, temperature=0.0, pad_token_id=9),
        )
        self.assertEqual(local_chat_service._generation_kwargs(bundle, 40, 0.5)["temperature"], 0.5)


class EvalPassesAndRecordsTests(unittest.TestCase):
    def test_transformers_eval_passes_neutral_kwargs_and_records_them(self):
        import torch

        from app.services import evaluation_service

        seen: list[dict] = []

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 1
            chat_template = None

            def __len__(self):
                return 8

            def __call__(self, text, return_tensors=None, **kw):
                return {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

            def decode(self, ids, skip_special_tokens=True):
                return "x"

        class FakeModel:
            dtype = torch.float32
            generation_config = SimpleNamespace(**QWEN_INSTRUCT_CONFIG)

            def get_input_embeddings(self):
                return SimpleNamespace(num_embeddings=8)

            def eval(self):
                return self

            def to(self, *_a, **_k):
                return self

            def generate(self, **kw):
                seen.append({k: v for k, v in kw.items() if k not in ("input_ids", "attention_mask")})
                return torch.tensor([[2, 3, 4, 5]])

        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTok()), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=FakeModel()), \
             mock.patch("torch.cuda.is_available", return_value=False):
            preds, runtime = evaluation_service._run_transformers_inference(
                "fake/model", [{"prompt": "p", "reference": "r"}], 16, 0.0, apply_chat_template=False,
            )
        self.assertEqual(len(preds), 1)
        self.assertEqual(seen[0]["repetition_penalty"], 1.0)
        self.assertEqual((seen[0]["num_beams"], seen[0]["no_repeat_ngram_size"]), (1, 0))
        self.assertFalse(seen[0]["do_sample"])
        self.assertEqual(runtime["decoding"]["repetition_penalty"], 1.0)
        self.assertFalse(runtime["decoding"]["do_sample"])


class ExportAlignsGenerationConfigTests(unittest.TestCase):
    def test_neutral_config_turns_off_processors_and_keeps_the_rest(self):
        out = neutral_generation_config(QWEN_INSTRUCT_CONFIG)
        self.assertEqual(out["eos_token_id"], [151645, 151643])
        self.assertEqual(out["pad_token_id"], 151643)
        self.assertFalse(out["do_sample"])
        self.assertEqual((out["repetition_penalty"], out["num_beams"], out["temperature"]), (1.0, 1, 1.0))
        self.assertNotIn("top_k", out)
        self.assertNotIn("top_p", out)
        # model-specific keys (mBART target language, Whisper maps, lengths) survive
        whisper_like = {**QWEN_INSTRUCT_CONFIG, "forced_bos_token_id": 7, "lang_to_id": {"<|en|>": 3},
                        "suppress_tokens": [1, 2], "max_length": 448, "num_beams": 4}
        out = neutral_generation_config(whisper_like)
        self.assertEqual((out["forced_bos_token_id"], out["lang_to_id"], out["suppress_tokens"], out["max_length"]),
                         (7, {"<|en|>": 3}, [1, 2], 448))
        self.assertEqual(out["num_beams"], 1)

    def test_written_config_loads_and_saves_strictly(self):
        """transformers validates strictly on save: e.g. top_k 0 under greedy fails."""
        from transformers import GenerationConfig

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "generation_config.json").write_text(json.dumps(QWEN_INSTRUCT_CONFIG))
            align_exported_generation_config(Path(tmp))
            cfg = GenerationConfig.from_pretrained(tmp)
            self.assertEqual(cfg.repetition_penalty, 1.0)
            cfg.save_pretrained(Path(tmp) / "again")

    def test_align_reports_what_it_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(align_exported_generation_config(Path(tmp)))
            (Path(tmp) / "generation_config.json").write_text(json.dumps(QWEN_INSTRUCT_CONFIG))
            note = align_exported_generation_config(Path(tmp))
            self.assertEqual(note["replaced"]["repetition_penalty"], 1.1)
            self.assertEqual(note["replaced"]["top_k"], 20)
            self.assertTrue(note["replaced"]["do_sample"])
            written = json.loads((Path(tmp) / "generation_config.json").read_text())
            self.assertEqual(written, note["written"])
            self.assertEqual(written["repetition_penalty"], 1.0)

    def test_hf_export_rewrites_generation_config(self):
        from app.models.export import ExportFormat, ExportStatus
        from app.services.export_service import create_export, run_export

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "exp" / "model"
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
            (model / "model.safetensors").write_bytes(b"w")
            (model / "generation_config.json").write_text(json.dumps(QWEN_INSTRUCT_CONFIG))
            pid = client.post("/api/projects", json={"name": f"decoding-{time.time_ns()}", "description": "x"}).json()["id"]

            async def _go():
                async with async_session_factory() as db:
                    exp = Experiment(project_id=pid, name="run", status=ExperimentStatus.COMPLETED,
                                     base_model="org/base-1b", output_dir=str(model.parent), config={"task_type": "causal_lm"})
                    db.add(exp)
                    await db.commit()
                    export = await create_export(db, pid, exp.id, ExportFormat.HUGGINGFACE)
                    export = await run_export(db, pid, export.id, run_smoke_tests=False)
                    await db.commit()
                    return export.status, export.manifest, export.output_path

            status, manifest, output_path = asyncio.run(_go())
            self.assertEqual(status, ExportStatus.COMPLETED)
            run_dir = Path(output_path) / manifest["run_dir"]
            shipped = json.loads((run_dir / "model" / "generation_config.json").read_text())
            self.assertEqual(shipped["repetition_penalty"], 1.0)
            self.assertFalse(shipped["do_sample"])
            self.assertEqual(manifest["decoding"]["replaced"]["repetition_penalty"], 1.1)
            notes = (run_dir / "RELEASE_NOTES.md").read_text()
            self.assertIn("- Decoding: generation_config.json set to greedy", notes)
            self.assertIn("repetition_penalty=1.1", notes)
            # the source run is untouched
            self.assertEqual(json.loads((model / "generation_config.json").read_text())["repetition_penalty"], 1.1)


if __name__ == "__main__":
    unittest.main()
