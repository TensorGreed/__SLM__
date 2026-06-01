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


if __name__ == "__main__":
    unittest.main()
