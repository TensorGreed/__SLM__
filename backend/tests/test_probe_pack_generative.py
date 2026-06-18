"""Coach-stage-2 phase 10 — generative predict_fn wiring for probes.

The actual generation needs torch, so these tests mock
``_run_transformers_inference`` (the shared generation path) and verify:
- the generative predict_fn wrapper maps texts → completion strings;
- ``_safe_run_probe_pack`` dispatches generative profiles through it and
  scores the refusal / grounding probes end-to-end;
- non-classification, non-generative profiles still return ``{}``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services import evaluation_service  # noqa: E402


def _fake_inference(model_ref, pairs, max_new_tokens, temperature, **kwargs):
    """Stand-in for _run_transformers_inference: echoes a fixed refusal
    so the refusal/grounding probes deterministically pass."""
    preds = [
        {"prompt": p["prompt"], "prediction": "I cannot help with that request."}
        for p in pairs
    ]
    return preds, {"engine": "mock"}


class GenerativePredictFnTests(unittest.TestCase):
    def test_wrapper_returns_completion_strings_in_order(self):
        with patch.object(
            evaluation_service, "_run_transformers_inference",
            side_effect=lambda model_ref, pairs, *a, **k: (
                [{"prompt": p["prompt"], "prediction": f"out:{p['prompt']}"} for p in pairs],
                {},
            ),
        ):
            predict_fn = evaluation_service._build_generative_predict_fn("/fake/ckpt")
            out = predict_fn(["alpha", "beta"])
        self.assertEqual(out, ["out:alpha", "out:beta"])


class SafeRunProbePackGenerativeTests(unittest.TestCase):
    def _run(self, task_profile: str):
        with tempfile.TemporaryDirectory() as ckpt:
            exp = SimpleNamespace(id=1, output_dir=ckpt)
            with patch.object(
                evaluation_service, "_run_transformers_inference",
                side_effect=_fake_inference,
            ):
                return asyncio.run(
                    evaluation_service._safe_run_probe_pack(
                        None,  # generative path never touches db
                        project_id=1,
                        experiment=exp,
                        task_profile=task_profile,
                    )
                )

    def test_instruction_sft_runs_refusal_probes_and_scores(self):
        result = self._run("instruction_sft")
        self.assertTrue(result, "expected a non-empty probe snapshot")
        self.assertIn("probe_pass_rate", result)
        self.assertGreater(result["total"], 0)
        # A constant refusal output passes the refusal + no-fabrication
        # probes (and is stable for the robustness probe), so the pack
        # grades a perfect score here.
        self.assertEqual(result["probe_pass_rate"], 1.0)
        self.assertIn("refuses_or_declines", result["per_property"])

    def test_rag_qa_grounding_probes_execute(self):
        result = self._run("rag_qa")
        self.assertTrue(result)
        self.assertIn("no_fabrication_when_unsupported", result["per_property"])

    def test_unknown_profile_returns_empty(self):
        result = self._run("totally_unknown_profile")
        self.assertEqual(result, {})

    def test_missing_checkpoint_returns_empty(self):
        exp = SimpleNamespace(id=1, output_dir="/nonexistent/path/xyz")
        with patch.object(
            evaluation_service, "_run_transformers_inference",
            side_effect=_fake_inference,
        ):
            result = asyncio.run(
                evaluation_service._safe_run_probe_pack(
                    None, project_id=1, experiment=exp, task_profile="instruction_sft",
                )
            )
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
