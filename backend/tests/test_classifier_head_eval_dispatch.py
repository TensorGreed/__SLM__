"""δ-fix tests — classifier-head dispatch on held-out eval.

Pins the contract for ``_resolve_classifier_head_artifacts`` +
``_run_local_inference``: when an experiment was trained with
``AutoModelForSequenceClassification`` (PEFT adapter carries
``task_type: SEQ_CLS`` and the runtime_environment lists a
``label_space_preview``), the held-out eval path must use the
classifier head's logits — not free generation from the LM head.
The β commit message documented this as a separate architectural
bug; δ closes it.

  * Detector inspects ``adapter_config.json`` + ``training_report.json``
    siblings to decide. Missing dir / missing adapter / no SEQ_CLS
    signal / no label space → ``None`` (caller stays on
    generation), so a normal SFT checkpoint isn't regressed.
  * ``_run_local_inference`` dispatches: when the detector returns
    artifacts, it routes to ``_run_classifier_head_inference``;
    otherwise it stays on ``_run_transformers_inference``. We
    verify this by patching both inference functions with stubs
    and asserting the right one fired.
  * The classifier-head path returns predictions with the label
    *string* (mapped through ``id2label``), so the downstream
    ClassificationHandler parser doesn't need to know whether
    the prediction came from generation or from head logits.

We don't exercise the real torch + transformers stack in this test
file (CI may not have a GPU). Phase β's tests already pin the
prompt-format math byte-for-byte against
``ClassificationHandler._build_prompt_text``; the integration
shows up in the held-out run we kick off after this lands.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.evaluation_service import (
    _resolve_classifier_head_artifacts,
    _resolve_multimodal_artifacts,
    _resolve_seq2seq_artifacts,
    _run_local_inference,
)


# ── Detector ─────────────────────────────────────────────────────────


class ResolveClassifierHeadArtifactsTests(unittest.TestCase):
    def _make_checkpoint(
        self,
        *,
        adapter_extra: dict | None = None,
        report: dict | None = None,
        report_in_checkpoint: bool = False,
    ) -> Path:
        """Build a fake PEFT-classifier checkpoint on disk: an
        experiment dir + a checkpoint-N subdir, each carrying the
        files the detector reads."""
        root = Path(tempfile.mkdtemp(prefix="delta-test-"))
        exp = root / "exp"
        ckpt = exp / "checkpoint-100"
        ckpt.mkdir(parents=True)
        adapter_cfg = {
            "base_model_name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "task_type": "SEQ_CLS",
            "modules_to_save": ["classifier", "score"],
            "peft_type": "LORA",
        }
        adapter_cfg.update(adapter_extra or {})
        (ckpt / "adapter_config.json").write_text(json.dumps(adapter_cfg))
        if report is not None:
            target = (ckpt if report_in_checkpoint else exp) / "training_report.json"
            target.write_text(json.dumps(report))
        return ckpt

    def test_detects_seq_cls_adapter_with_label_space(self):
        ckpt = self._make_checkpoint(
            report={
                "runtime_environment": {
                    "label_space_size": 2,
                    "label_space_preview": ["benign", "injection"],
                },
            },
        )
        artifacts = _resolve_classifier_head_artifacts(str(ckpt))
        self.assertIsNotNone(artifacts)
        self.assertEqual(artifacts["num_labels"], 2)
        self.assertEqual(
            artifacts["id2label"],
            {0: "benign", 1: "injection"},
        )
        self.assertEqual(
            artifacts["base_model"],
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        )

    def test_detects_via_modules_to_save_when_task_type_missing(self):
        # Older PEFT versions might not write ``task_type``; the
        # ``modules_to_save`` signal is enough on its own (the
        # SEQ_CLS-only modules ``classifier`` / ``score`` are not
        # something a CausalLM adapter would save).
        ckpt = self._make_checkpoint(
            adapter_extra={"task_type": None},
            report={
                "runtime_environment": {
                    "label_space_preview": ["pos", "neg"],
                },
            },
        )
        artifacts = _resolve_classifier_head_artifacts(str(ckpt))
        self.assertIsNotNone(artifacts)
        self.assertEqual(artifacts["num_labels"], 2)

    def test_returns_none_when_no_seq_cls_signal(self):
        # A regular causal-LM adapter: no task_type, no classifier
        # modules. Detector must say "no, stay on generation".
        ckpt = self._make_checkpoint(
            adapter_extra={
                "task_type": "CAUSAL_LM",
                "modules_to_save": [],
            },
            report={"runtime_environment": {"label_space_preview": ["a"]}},
        )
        self.assertIsNone(_resolve_classifier_head_artifacts(str(ckpt)))

    def test_returns_none_when_label_space_missing(self):
        # SEQ_CLS adapter but no label space anywhere we can find
        # → we can't map id → label string, so fall back to
        # generation rather than emitting integer ids the eval
        # parser would mark unparseable.
        ckpt = self._make_checkpoint(report=None)
        self.assertIsNone(_resolve_classifier_head_artifacts(str(ckpt)))

    def test_walks_up_to_experiment_dir_for_report(self):
        # The training_report.json lives in the experiment dir, not
        # the checkpoint dir. Detector should walk one level up.
        ckpt = self._make_checkpoint(
            report={"runtime_environment": {"label_space_preview": ["a", "b"]}},
            report_in_checkpoint=False,
        )
        artifacts = _resolve_classifier_head_artifacts(str(ckpt))
        self.assertIsNotNone(artifacts)

    def test_finds_report_when_colocated_with_checkpoint(self):
        # Some checkpoints copy the report into their own dir. The
        # detector should accept either location.
        ckpt = self._make_checkpoint(
            report={"runtime_environment": {"label_space_preview": ["a", "b"]}},
            report_in_checkpoint=True,
        )
        artifacts = _resolve_classifier_head_artifacts(str(ckpt))
        self.assertIsNotNone(artifacts)

    def test_returns_none_for_nonexistent_path(self):
        self.assertIsNone(
            _resolve_classifier_head_artifacts("/tmp/does-not-exist-delta")
        )

    def test_returns_none_for_malformed_adapter_config(self):
        root = Path(tempfile.mkdtemp(prefix="delta-malformed-"))
        (root / "adapter_config.json").write_text("not-json-{[}")
        self.assertIsNone(_resolve_classifier_head_artifacts(str(root)))


# ── Dispatch ─────────────────────────────────────────────────────────


class RunLocalInferenceDispatchTests(unittest.TestCase):
    def _seq_cls_checkpoint(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="delta-dispatch-"))
        exp = root / "exp"
        ckpt = exp / "checkpoint-1"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "fixture/base",
                    "task_type": "SEQ_CLS",
                    "modules_to_save": ["classifier", "score"],
                }
            )
        )
        (exp / "training_report.json").write_text(
            json.dumps(
                {
                    "runtime_environment": {
                        "label_space_preview": ["benign", "injection"],
                    }
                }
            )
        )
        return ckpt

    def test_dispatches_to_classifier_head_when_detector_returns_artifacts(self):
        ckpt = self._seq_cls_checkpoint()

        called_with: dict = {}

        def _fake_classifier(artifacts, pairs):
            called_with["artifacts"] = artifacts
            called_with["pairs"] = pairs
            return (
                [
                    {
                        "prompt": "p", "reference": "benign",
                        "prediction": "benign", "latency_ms": 1.0,
                        "generated_tokens": 1,
                    }
                ],
                {"engine": "transformers", "head": "sequence_classification"},
            )

        with (
            patch(
                "app.services.evaluation_service._run_classifier_head_inference",
                new=_fake_classifier,
            ),
            patch(
                "app.services.evaluation_service._run_transformers_inference",
                side_effect=AssertionError(
                    "generation path must not run for SEQ_CLS dispatch"
                ),
            ),
        ):
            preds, runtime = _run_local_inference(
                str(ckpt),
                [{"prompt": "p", "reference": "benign"}],
                max_new_tokens=8,
                temperature=0.0,
            )
        # The classifier path was the one that fired, with the
        # artifacts the detector resolved.
        self.assertEqual(preds[0]["prediction"], "benign")
        self.assertEqual(runtime.get("head"), "sequence_classification")
        self.assertEqual(called_with["artifacts"]["num_labels"], 2)

    def test_falls_back_to_generation_for_non_seq_cls_checkpoint(self):
        # A normal causal-LM adapter must stay on the existing
        # generation path. δ is opt-in via detector signal — no
        # silent rerouting of unrelated experiments.
        root = Path(tempfile.mkdtemp(prefix="delta-noseqcls-"))
        (root / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "fixture/base",
                    "task_type": "CAUSAL_LM",
                    "modules_to_save": [],
                }
            )
        )

        with (
            patch(
                "app.services.evaluation_service._run_classifier_head_inference",
                side_effect=AssertionError(
                    "classifier head path must not run for CAUSAL_LM"
                ),
            ),
            patch(
                "app.services.evaluation_service._run_transformers_inference",
                return_value=(
                    [{"prediction": "from-generation"}],
                    {"engine": "transformers"},
                ),
            ),
        ):
            preds, runtime = _run_local_inference(
                str(root),
                [{"prompt": "p", "reference": "x"}],
                max_new_tokens=8,
                temperature=0.0,
            )
        self.assertEqual(preds[0]["prediction"], "from-generation")
        self.assertEqual(runtime.get("engine"), "transformers")


class ResolveSeq2SeqArtifactsTests(unittest.TestCase):
    """ε-fix detector (mirrors ResolveClassifierHeadArtifactsTests
    but keyed on ``task_type: SEQ_2_SEQ_LM``)."""

    def _make_checkpoint(
        self, *, adapter_extra: dict | None = None,
    ) -> Path:
        root = Path(tempfile.mkdtemp(prefix="epsilon-test-"))
        ckpt = root / "exp" / "checkpoint-100"
        ckpt.mkdir(parents=True)
        adapter_cfg = {
            "base_model_name_or_path": "t5-small",
            "task_type": "SEQ_2_SEQ_LM",
            "peft_type": "LORA",
        }
        adapter_cfg.update(adapter_extra or {})
        (ckpt / "adapter_config.json").write_text(json.dumps(adapter_cfg))
        return ckpt

    def test_detects_seq2seq_adapter(self):
        ckpt = self._make_checkpoint()
        artifacts = _resolve_seq2seq_artifacts(str(ckpt))
        self.assertIsNotNone(artifacts)
        self.assertEqual(artifacts["base_model"], "t5-small")
        self.assertEqual(artifacts["adapter_path"], str(ckpt))

    def test_returns_none_for_causal_lm_adapter(self):
        # CausalLM adapters must NOT be misdetected as seq2seq —
        # the generation path is the right one for them.
        ckpt = self._make_checkpoint(adapter_extra={"task_type": "CAUSAL_LM"})
        self.assertIsNone(_resolve_seq2seq_artifacts(str(ckpt)))

    def test_returns_none_for_seq_cls_adapter(self):
        # The δ branch handles SEQ_CLS; the ε detector must not
        # claim a SEQ_CLS adapter (otherwise dispatch ordering
        # could double-fire).
        ckpt = self._make_checkpoint(adapter_extra={"task_type": "SEQ_CLS"})
        self.assertIsNone(_resolve_seq2seq_artifacts(str(ckpt)))

    def test_returns_none_for_nonexistent_path(self):
        self.assertIsNone(_resolve_seq2seq_artifacts("/tmp/no-epsilon"))

    def test_returns_none_for_missing_adapter_config(self):
        root = Path(tempfile.mkdtemp(prefix="epsilon-empty-"))
        self.assertIsNone(_resolve_seq2seq_artifacts(str(root)))


class RunLocalInferenceSeq2SeqDispatchTests(unittest.TestCase):
    def _seq2seq_checkpoint(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="epsilon-dispatch-"))
        ckpt = root / "exp" / "checkpoint-1"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "fixture/t5",
                    "task_type": "SEQ_2_SEQ_LM",
                }
            )
        )
        return ckpt

    def test_dispatches_to_seq2seq_when_detector_returns_artifacts(self):
        ckpt = self._seq2seq_checkpoint()

        def _fake_seq2seq(artifacts, pairs, max_new_tokens, temperature):
            return (
                [
                    {
                        "prompt": "summarise: hi",
                        "reference": "hi",
                        "prediction": "hello",
                        "latency_ms": 0.1,
                        "generated_tokens": 1,
                    }
                ],
                {"engine": "transformers", "head": "seq2seq_lm"},
            )

        with (
            patch(
                "app.services.evaluation_service._run_seq2seq_inference",
                new=_fake_seq2seq,
            ),
            patch(
                "app.services.evaluation_service._run_transformers_inference",
                side_effect=AssertionError(
                    "generation path must not run for SEQ_2_SEQ_LM dispatch"
                ),
            ),
            patch(
                "app.services.evaluation_service._run_classifier_head_inference",
                side_effect=AssertionError(
                    "δ path must not run for seq2seq checkpoint"
                ),
            ),
        ):
            preds, runtime = _run_local_inference(
                str(ckpt),
                [{"prompt": "summarise: hi", "reference": "hi"}],
                max_new_tokens=8,
                temperature=0.0,
            )
        self.assertEqual(runtime.get("head"), "seq2seq_lm")
        self.assertEqual(preds[0]["prediction"], "hello")

    def test_dispatch_order_seq_cls_wins_over_seq2seq_when_both_signals_appear(self):
        # Defensive: an adapter_config can't legitimately carry both
        # task_type values, but if a bug ever produced one, the
        # SEQ_CLS branch must run first (δ is the more-tested path
        # and head logits are richer than a generic generate call).
        # We test by having the SEQ_CLS detector return artifacts
        # while the seq2seq detector also could; only the SEQ_CLS
        # path should fire.
        root = Path(tempfile.mkdtemp(prefix="epsilon-order-"))
        exp = root / "exp"
        ckpt = exp / "checkpoint-1"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "fixture/base",
                    "task_type": "SEQ_CLS",
                    "modules_to_save": ["classifier", "score"],
                }
            )
        )
        (exp / "training_report.json").write_text(
            json.dumps(
                {"runtime_environment": {"label_space_preview": ["a", "b"]}}
            )
        )

        def _fake_classifier(artifacts, pairs):
            return ([{"prediction": "a"}], {"head": "sequence_classification"})

        with (
            patch(
                "app.services.evaluation_service._run_classifier_head_inference",
                new=_fake_classifier,
            ),
            patch(
                "app.services.evaluation_service._run_seq2seq_inference",
                side_effect=AssertionError(
                    "ε path must not run when SEQ_CLS detector fires first"
                ),
            ),
        ):
            preds, runtime = _run_local_inference(
                str(ckpt),
                [{"prompt": "x", "reference": "a"}],
                max_new_tokens=8,
                temperature=0.0,
            )
        self.assertEqual(runtime["head"], "sequence_classification")


class ResolveMultimodalArtifactsTests(unittest.TestCase):
    """Multimodal detector — extends δ/ε to vision/audio. Same
    shape as δ's tests: positive signal + every negative path."""

    def _make_checkpoint(
        self,
        *,
        loader_name: str | None = "AutoModelForVision2Seq",
        adapter_task_type: str = "SEQ_2_SEQ_LM",
        report_in_checkpoint: bool = False,
    ) -> Path:
        root = Path(tempfile.mkdtemp(prefix="multimodal-test-"))
        exp = root / "exp"
        ckpt = exp / "checkpoint-100"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "fixture/vlm-base",
                    "task_type": adapter_task_type,
                    "peft_type": "LORA",
                }
            )
        )
        if loader_name is not None:
            report = {
                "runtime_environment": {
                    "multimodal_model_loader": loader_name,
                }
            }
            target = exp if not report_in_checkpoint else ckpt
            (target / "training_report.json").write_text(json.dumps(report))
        return ckpt

    def test_detects_vision_loader(self):
        ckpt = self._make_checkpoint(loader_name="AutoModelForVision2Seq")
        out = _resolve_multimodal_artifacts(str(ckpt))
        self.assertIsNotNone(out)
        self.assertEqual(out["modality"], "vision")
        self.assertEqual(out["model_loader_class"], "AutoModelForVision2Seq")
        self.assertEqual(out["base_model"], "fixture/vlm-base")

    def test_detects_audio_loader(self):
        ckpt = self._make_checkpoint(loader_name="AutoModelForSpeechSeq2Seq")
        out = _resolve_multimodal_artifacts(str(ckpt))
        self.assertIsNotNone(out)
        self.assertEqual(out["modality"], "audio")
        self.assertEqual(out["model_loader_class"], "AutoModelForSpeechSeq2Seq")

    def test_returns_none_when_loader_field_missing(self):
        # Plain seq2seq checkpoint — no multimodal_model_loader in
        # the runtime_environment. ε should claim this, not the
        # multimodal detector.
        ckpt = self._make_checkpoint(loader_name=None)
        self.assertIsNone(_resolve_multimodal_artifacts(str(ckpt)))

    def test_returns_none_for_non_multimodal_loader_names(self):
        # Unknown loader name (e.g., trainer added a new
        # specialized class we haven't taught the detector
        # about). Defensive: don't claim it; fall through to ε.
        ckpt = self._make_checkpoint(loader_name="AutoModelForSomeNewThing")
        self.assertIsNone(_resolve_multimodal_artifacts(str(ckpt)))

    def test_returns_none_for_missing_adapter_config(self):
        root = Path(tempfile.mkdtemp(prefix="multimodal-no-adapter-"))
        # Has training_report but no adapter_config — detector
        # can't determine base model, falls through.
        report = {
            "runtime_environment": {
                "multimodal_model_loader": "AutoModelForVision2Seq",
            }
        }
        (root / "training_report.json").write_text(json.dumps(report))
        self.assertIsNone(_resolve_multimodal_artifacts(str(root)))

    def test_finds_report_when_colocated_with_checkpoint(self):
        # Some runtimes copy the report into the checkpoint dir
        # alongside adapter_config.json. Detector accepts both.
        ckpt = self._make_checkpoint(
            loader_name="AutoModelForVision2Seq",
            report_in_checkpoint=True,
        )
        self.assertIsNotNone(_resolve_multimodal_artifacts(str(ckpt)))


class RunLocalInferenceMultimodalDispatchTests(unittest.TestCase):
    def _multimodal_checkpoint(self, *, loader_name: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix="multimodal-dispatch-"))
        exp = root / "exp"
        ckpt = exp / "checkpoint-1"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "fixture/vlm",
                    "task_type": "SEQ_2_SEQ_LM",
                }
            )
        )
        (exp / "training_report.json").write_text(
            json.dumps(
                {
                    "runtime_environment": {
                        "multimodal_model_loader": loader_name,
                    }
                }
            )
        )
        return ckpt

    def test_vision_checkpoint_dispatches_to_multimodal_path(self):
        # Vision adapter must win over ε's seq2seq dispatch even
        # though adapter_config also says SEQ_2_SEQ_LM. Dispatch
        # order is the load-bearing contract.
        ckpt = self._multimodal_checkpoint(
            loader_name="AutoModelForVision2Seq"
        )

        def _fake_mm(artifacts, pairs, max_new_tokens, temperature):
            return (
                [{"prediction": "vision-routed"}],
                {"engine": "transformers", "head": "vision2seq"},
            )

        with (
            patch(
                "app.services.evaluation_service._run_multimodal_inference",
                new=_fake_mm,
            ),
            patch(
                "app.services.evaluation_service._run_seq2seq_inference",
                side_effect=AssertionError(
                    "ε must not fire when multimodal detector wins"
                ),
            ),
            patch(
                "app.services.evaluation_service._run_classifier_head_inference",
                side_effect=AssertionError(
                    "δ must not fire for multimodal checkpoint"
                ),
            ),
            patch(
                "app.services.evaluation_service._run_transformers_inference",
                side_effect=AssertionError(
                    "generation path must not fire for multimodal"
                ),
            ),
        ):
            preds, runtime = _run_local_inference(
                str(ckpt),
                [{"prompt": "p", "reference": "r"}],
                max_new_tokens=8,
                temperature=0.0,
            )
        self.assertEqual(preds[0]["prediction"], "vision-routed")
        self.assertEqual(runtime.get("head"), "vision2seq")

    def test_audio_checkpoint_dispatches_to_multimodal_path(self):
        ckpt = self._multimodal_checkpoint(
            loader_name="AutoModelForSpeechSeq2Seq"
        )

        captured: dict = {}

        def _fake_mm(artifacts, pairs, max_new_tokens, temperature):
            captured["modality"] = artifacts.get("modality")
            return (
                [{"prediction": "audio-routed"}],
                {"engine": "transformers", "head": "speech_seq2seq"},
            )

        with patch(
            "app.services.evaluation_service._run_multimodal_inference",
            new=_fake_mm,
        ):
            _run_local_inference(
                str(ckpt),
                [{"prompt": "p", "reference": "r"}],
                max_new_tokens=8,
                temperature=0.0,
            )
        # Detector correctly tagged the artifacts as audio so the
        # inference function knows which loader to import.
        self.assertEqual(captured["modality"], "audio")

    def test_plain_seq2seq_checkpoint_still_routes_to_epsilon(self):
        # Regression guard: a seq2seq adapter WITHOUT
        # multimodal_model_loader in its training_report keeps
        # going through ε. Multimodal dispatch shouldn't
        # accidentally claim non-multimodal seq2seq runs.
        root = Path(tempfile.mkdtemp(prefix="multimodal-plain-s2s-"))
        exp = root / "exp"
        ckpt = exp / "checkpoint-1"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "fixture/t5",
                    "task_type": "SEQ_2_SEQ_LM",
                }
            )
        )
        # No training_report.json → no multimodal signal.

        def _fake_s2s(artifacts, pairs, max_new_tokens, temperature):
            return (
                [{"prediction": "from-epsilon"}],
                {"head": "seq2seq_lm"},
            )

        with (
            patch(
                "app.services.evaluation_service._run_seq2seq_inference",
                new=_fake_s2s,
            ),
            patch(
                "app.services.evaluation_service._run_multimodal_inference",
                side_effect=AssertionError(
                    "multimodal must not fire on plain seq2seq"
                ),
            ),
        ):
            preds, runtime = _run_local_inference(
                str(ckpt),
                [{"prompt": "p", "reference": "r"}],
                max_new_tokens=8,
                temperature=0.0,
            )
        self.assertEqual(preds[0]["prediction"], "from-epsilon")


if __name__ == "__main__":
    unittest.main()
