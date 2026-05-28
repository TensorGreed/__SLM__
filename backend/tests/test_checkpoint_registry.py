"""Unit tests for the warm-start checkpoint registry (Track 1, Epic B).

Covers manifest load/normalization and the resolve-or-fall-back logic that
training uses to pick effective starting weights. Pure unit tests — no DB,
no FastAPI lifespan — driven against temp registry roots plus the committed
``backend/data/pretrained_checkpoints`` manifests.
"""

import json
import tempfile
import unittest
from pathlib import Path

from app.services import checkpoint_registry_service as registry
from app.services.training_recipe_service import (
    list_training_recipes,
    resolve_training_recipe,
)

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _write_manifest(root: Path, name: str, body: dict, *, with_weights: bool = False) -> Path:
    ckpt_dir = root / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "manifest.json").write_text(json.dumps(body), encoding="utf-8")
    if with_weights:
        weights = ckpt_dir / str(body.get("artifact_path") or "weights")
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "config.json").write_text("{}", encoding="utf-8")
    return ckpt_dir


class CheckpointRegistryLoadTests(unittest.TestCase):
    def test_missing_root_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist"
            self.assertEqual(registry.list_checkpoints(root=missing), [])
            self.assertIsNone(registry.load_checkpoint("anything", root=missing))

    def test_list_and_load_normalizes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_manifest(
                root,
                "qa-base-135m",
                {
                    "name": "qa-base-135m",
                    "display_name": "QABase",
                    "base_model": BASE,
                    "task_shape": "qa",
                    "status": "planned",
                    "artifact_path": "weights",
                },
            )
            listed = registry.list_checkpoints(root=root)
            self.assertEqual(len(listed), 1)
            manifest = registry.load_checkpoint("QA-Base-135M", root=root)  # case-insensitive
            self.assertIsNotNone(manifest)
            self.assertEqual(manifest["name"], "qa-base-135m")
            self.assertEqual(manifest["base_model"], BASE)
            self.assertEqual(manifest["status"], "planned")
            self.assertFalse(manifest["artifact_exists"])
            self.assertFalse(manifest["available"])
            self.assertTrue(manifest["resolved_artifact_path"].endswith("weights"))

    def test_available_when_weights_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_manifest(
                root,
                "ready-base",
                {"name": "ready-base", "base_model": BASE, "status": "available", "artifact_path": "weights"},
                with_weights=True,
            )
            manifest = registry.load_checkpoint("ready-base", root=root)
            self.assertTrue(manifest["artifact_exists"])
            self.assertTrue(manifest["available"])

    def test_malformed_manifest_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "broken"
            bad.mkdir(parents=True)
            (bad / "manifest.json").write_text("{not valid json", encoding="utf-8")
            self.assertEqual(registry.list_checkpoints(root=root), [])
            self.assertIsNone(registry.load_checkpoint("broken", root=root))


class CheckpointResolutionTests(unittest.TestCase):
    def test_no_recommendation_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = registry.resolve_starting_checkpoint(
                base_model=BASE, recommended_checkpoint=None, root=Path(tmp)
            )
            self.assertEqual(res["source"], "base_model")
            self.assertEqual(res["effective_base_model"], BASE)
            self.assertEqual(res["reason"], "no_checkpoint_recommended")

    def test_unregistered_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = registry.resolve_starting_checkpoint(
                base_model=BASE, recommended_checkpoint="ghost", root=Path(tmp)
            )
            self.assertEqual(res["source"], "base_model")
            self.assertEqual(res["effective_base_model"], BASE)
            self.assertEqual(res["reason"], "checkpoint_not_registered:ghost")

    def test_planned_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_manifest(
                root, "qa", {"name": "qa", "base_model": BASE, "status": "planned", "artifact_path": "weights"}
            )
            res = registry.resolve_starting_checkpoint(
                base_model=BASE, recommended_checkpoint="qa", root=root
            )
            self.assertEqual(res["source"], "base_model")
            self.assertEqual(res["effective_base_model"], BASE)
            self.assertEqual(res["reason"], "checkpoint_planned:qa")
            self.assertEqual(res["checkpoint_name"], "qa")
            self.assertIsNotNone(res["manifest"])

    def test_missing_weights_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_manifest(
                root, "qa", {"name": "qa", "base_model": BASE, "status": "available", "artifact_path": "weights"}
            )
            res = registry.resolve_starting_checkpoint(
                base_model=BASE, recommended_checkpoint="qa", root=root
            )
            self.assertEqual(res["source"], "base_model")
            self.assertEqual(res["reason"], "checkpoint_artifact_missing:qa")

    def test_base_model_mismatch_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_manifest(
                root,
                "qa",
                {"name": "qa", "base_model": BASE, "status": "available", "artifact_path": "weights"},
                with_weights=True,
            )
            res = registry.resolve_starting_checkpoint(
                base_model="Qwen/Qwen2.5-0.5B", recommended_checkpoint="qa", root=root
            )
            self.assertEqual(res["source"], "base_model")
            self.assertEqual(res["effective_base_model"], "Qwen/Qwen2.5-0.5B")
            self.assertEqual(res["reason"], "checkpoint_base_model_mismatch:qa")

    def test_available_compatible_resolves_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_dir = _write_manifest(
                root,
                "qa",
                {"name": "qa", "base_model": BASE, "status": "available", "artifact_path": "weights"},
                with_weights=True,
            )
            res = registry.resolve_starting_checkpoint(
                base_model=BASE, recommended_checkpoint="qa", root=root
            )
            self.assertEqual(res["source"], "checkpoint")
            self.assertEqual(res["checkpoint_name"], "qa")
            self.assertEqual(res["reason"], "warm_start:qa")
            self.assertEqual(res["effective_base_model"], str(ckpt_dir / "weights"))


class RecipeWiringTests(unittest.TestCase):
    def test_summary_surfaces_field(self):
        for recipe in list_training_recipes():
            self.assertIn("recommended_starting_checkpoint", recipe)

    def test_kd_recipes_recommend_task_bases(self):
        summaries = {r["recipe_id"]: r for r in list_training_recipes()}
        expected = {
            "recipe.kd.classification": "classifier-base-135m",
            "recipe.kd.qa": "qa-base-135m",
            "recipe.kd.span_extraction": "ner-base-135m",
        }
        for rid, name in expected.items():
            self.assertEqual(summaries[rid]["recommended_starting_checkpoint"], name)

    def test_general_recipe_has_no_recommendation(self):
        summaries = {r["recipe_id"]: r for r in list_training_recipes()}
        self.assertEqual(summaries["recipe.sft.balanced"]["recommended_starting_checkpoint"], "")

    def test_resolve_carries_recommendation_into_config(self):
        resolved = resolve_training_recipe("recipe.kd.qa", base_config={"base_model": BASE})
        self.assertEqual(
            resolved["resolved_config"]["recommended_starting_checkpoint"], "qa-base-135m"
        )

    def test_explicit_override_not_clobbered(self):
        resolved = resolve_training_recipe(
            "recipe.kd.qa",
            base_config={"base_model": BASE},
            overrides={"recommended_starting_checkpoint": "custom-base"},
        )
        self.assertEqual(
            resolved["resolved_config"]["recommended_starting_checkpoint"], "custom-base"
        )


class SchemaSurvivalTests(unittest.TestCase):
    """The field must survive TrainingConfig so it reaches start_training via
    the stored experiment config (Pydantic drops unknown keys by default)."""

    def test_field_round_trips_through_training_config(self):
        from app.schemas.training import TrainingConfig

        cfg = TrainingConfig(base_model=BASE, recommended_starting_checkpoint="qa-base-135m")
        self.assertEqual(cfg.model_dump()["recommended_starting_checkpoint"], "qa-base-135m")

    def test_field_defaults_empty(self):
        from app.schemas.training import TrainingConfig

        cfg = TrainingConfig(base_model=BASE)
        self.assertEqual(cfg.model_dump()["recommended_starting_checkpoint"], "")


class StartTrainingWiringContractTests(unittest.TestCase):
    """Guard the start_training wiring against a silent rename/removal: training
    must resolve the checkpoint and pass the resolved model to the runtime."""

    def test_start_training_resolves_and_passes_effective_model(self):
        src = Path(
            "/home/anuragj/Desktop/GitHub/__SLM__/backend/app/services/training_service.py"
        ).read_text(encoding="utf-8")
        self.assertIn("resolve_starting_checkpoint(", src)
        self.assertIn("effective_base_model", src)
        self.assertIn("base_model=effective_base_model", src)


class CommittedRegistryTests(unittest.TestCase):
    def test_planned_task_bases_present(self):
        names = {c["name"] for c in registry.list_checkpoints()}
        self.assertTrue(
            {"classifier-base-135m", "ner-base-135m", "qa-base-135m", "sql-base-135m"} <= names
        )

    def test_committed_bases_resolve_consistently_with_status(self):
        # Data-driven so it survives bases being trained over time. Each base's
        # resolution must match its manifest status + on-disk weights:
        #   planned                      -> cold-start fallback (checkpoint_planned)
        #   available + weights present  -> warm start (checkpoint)
        #   available + weights missing  -> clean fallback (checkpoint_artifact_missing)
        # (weights are gitignored, so CI without them lands the last case.)
        for manifest in registry.list_checkpoints():
            name = manifest["name"]
            res = registry.resolve_starting_checkpoint(
                base_model=BASE, recommended_checkpoint=name
            )
            if manifest["status"] == "planned":
                self.assertEqual(res["source"], "base_model", name)
                self.assertEqual(res["reason"], f"checkpoint_planned:{name}", name)
            elif manifest["artifact_exists"]:
                self.assertEqual(res["source"], "checkpoint", name)
                self.assertEqual(res["reason"], f"warm_start:{name}", name)
            else:
                self.assertEqual(res["source"], "base_model", name)
                self.assertEqual(res["reason"], f"checkpoint_artifact_missing:{name}", name)

    def test_sql_base_still_planned(self):
        # No text-to-SQL corpus is wired and no recipe maps to it yet.
        self.assertEqual(registry.load_checkpoint("sql-base-135m")["status"], "planned")


if __name__ == "__main__":
    unittest.main()
