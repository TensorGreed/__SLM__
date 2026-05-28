"""Unit tests for the warm-start checkpoint trainer's non-GPU logic.

Covers corpus resolution, the license-audit gate, instruction formatting, the
fairness spot-check aggregation, and the manifest flip — all pure functions, so
no GPU / network / model download is touched. The training core itself is
exercised by a separate on-hardware smoke run.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from app.services import checkpoint_registry_service as registry


def _load_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_warmstart_checkpoint.py"
    spec = importlib.util.spec_from_file_location("warmstart_trainer_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ws = _load_module()


class CorpusResolutionTests(unittest.TestCase):
    def test_resolves_known_shapes(self):
        self.assertEqual(ws.resolve_corpus_spec("qa").categories, ["open_qa", "closed_qa", "general_qa"])
        self.assertEqual(ws.resolve_corpus_spec("classification").categories, ["classification"])
        self.assertEqual(ws.resolve_corpus_spec("span-extraction").categories, ["information_extraction"])

    def test_unknown_shape_raises(self):
        with self.assertRaises(ValueError):
            ws.resolve_corpus_spec("text-to-sql")  # no corpus wired yet

    def test_configured_corpora_are_permissively_licensed(self):
        for spec in ws.CORPUS_REGISTRY.values():
            ws.assert_license_permitted(spec)  # must not raise


class LicenseGateTests(unittest.TestCase):
    def test_permissive_passes(self):
        spec = ws.CorpusSpec(
            dataset="x", license="cc-by-sa-3.0", split="train", categories=None,
            instruction_field="i", context_field="c", response_field="r",
        )
        ws.assert_license_permitted(spec)  # no raise

    def test_non_permissive_raises(self):
        spec = ws.CorpusSpec(
            dataset="x", license="cc-by-nc-4.0", split="train", categories=None,
            instruction_field="i", context_field="c", response_field="r",
        )
        with self.assertRaises(ValueError):
            ws.assert_license_permitted(spec)


class FormatTests(unittest.TestCase):
    def test_with_context(self):
        ex = ws.format_instruction_example("Classify.", "I loved it.", "positive")
        self.assertEqual(ex["completion"], "positive")
        self.assertIn("Classify.", ex["prompt"])
        self.assertIn("I loved it.", ex["prompt"])

    def test_without_context(self):
        ex = ws.format_instruction_example("Capital of France?", "", "Paris.")
        self.assertEqual(ex["prompt"], "Capital of France?")

    def test_empty_returns_none(self):
        self.assertIsNone(ws.format_instruction_example("", "ctx", "resp"))
        self.assertIsNone(ws.format_instruction_example("inst", "ctx", ""))


class FairnessSummaryTests(unittest.TestCase):
    def test_balanced_passes(self):
        samples = [
            {"group": "a", "text": "one two three four"},
            {"group": "b", "text": "one two three five"},
        ]
        out = ws.summarize_fairness(samples)
        self.assertEqual(out["verdict"], "pass")
        self.assertIn("a", out["per_group"])
        self.assertAlmostEqual(out["length_disparity_ratio"], 1.0, places=2)

    def test_length_disparity_warns(self):
        samples = [
            {"group": "a", "text": "one two three four five six seven eight nine ten"},
            {"group": "b", "text": "short"},
        ]
        out = ws.summarize_fairness(samples)
        self.assertEqual(out["verdict"], "warn")
        self.assertGreater(out["length_disparity_ratio"], 2.0)
        self.assertTrue(out["notes"])

    def test_refusal_spread_warns(self):
        samples = [
            {"group": "a", "text": "Sure, here is a bio about them today."},
            {"group": "b", "text": "I can't help with that request right now."},
        ]
        out = ws.summarize_fairness(samples)
        self.assertEqual(out["per_group"]["b"]["refusal_rate"], 1.0)
        self.assertEqual(out["verdict"], "warn")

    def test_probe_prompts_cover_groups_and_templates(self):
        prompts = ws.fairness_probe_prompts()
        groups = {p["group"] for p in prompts}
        self.assertEqual(len(groups), len(ws._FAIRNESS_GROUPS))
        self.assertEqual(len(prompts), len(ws._FAIRNESS_GROUPS) * len(ws._FAIRNESS_TEMPLATES))


class ManifestFlipTests(unittest.TestCase):
    def _seed(self, root: Path) -> None:
        d = root / "qa-base-135m"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({
            "name": "qa-base-135m", "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "task_shape": "qa", "status": "planned", "artifact_path": "weights",
        }), encoding="utf-8")

    def test_flip_sets_available_and_stamps_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root)
            prov = {"corpus": "databricks/databricks-dolly-15k", "train_rows": 1200}
            fairness = {"verdict": "pass", "length_disparity_ratio": 1.1}
            body = ws.write_available_manifest(
                "qa-base-135m", root=root, training_provenance=prov, fairness=fairness
            )
            self.assertEqual(body["status"], "available")
            self.assertEqual(body["training_provenance"]["train_rows"], 1200)
            self.assertEqual(body["fairness"]["verdict"], "pass")
            # Registry now resolves it as available (weights still absent here, but
            # status + provenance round-trip through the manifest).
            reloaded = registry.load_checkpoint("qa-base-135m", root=root)
            self.assertEqual(reloaded["status"], "available")

    def test_flip_unregistered_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ws.write_available_manifest("ghost", root=Path(tmp), training_provenance={}, fairness={})


if __name__ == "__main__":
    unittest.main()
