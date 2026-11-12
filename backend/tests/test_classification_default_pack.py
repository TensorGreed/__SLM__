"""The built-in ``evalpack.classification.default`` pack.

The classification recipe, the classification project templates and the
classification demo sample all name this id. Before it existed it resolved to
nothing: the readiness ledger showed no gate breakdown and the project
resolver warned "not available; falling back". The pack reuses the general
default task specs, so a project that used to fall back is gated identically.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.services.evaluation_pack_service import (
    DEFAULT_EVALUATION_PACK_ID,
    _select_task_spec,
    get_evaluation_pack,
    is_supported_evaluation_pack_id,
    list_evaluation_packs,
)
from app.services.recipe_service import list_recipes

PACK_ID = "evalpack.classification.default"
_DATA = Path(__file__).resolve().parents[1] / "data"


def _gates(spec: dict) -> list[tuple]:
    return sorted(
        (g["gate_id"], g["metric_id"], g["operator"], g["threshold"], g["required"])
        for g in spec["gates"]
    )


class ClassificationDefaultPackTests(unittest.TestCase):
    def test_pack_is_registered_and_supported(self):
        self.assertIn(PACK_ID, {p["pack_id"] for p in list_evaluation_packs()})
        self.assertTrue(is_supported_evaluation_pack_id(PACK_ID))
        pack = get_evaluation_pack(PACK_ID)
        self.assertIsNotNone(pack)
        self.assertEqual(pack["default_task_profile"], "classification")

    def test_classification_spec_gates_accuracy_and_macro_f1(self):
        pack = get_evaluation_pack(PACK_ID)
        spec, profile, _ = _select_task_spec(pack, "classification")
        self.assertEqual(profile, "classification")
        required = {g["metric_id"] for g in spec["gates"] if g["required"]}
        self.assertEqual(required, {"accuracy", "macro_f1"})

    def test_every_profile_is_gated_exactly_like_the_general_default(self):
        pack = get_evaluation_pack(PACK_ID)
        general = get_evaluation_pack(DEFAULT_EVALUATION_PACK_ID)
        profiles = {s["task_profile"] for s in general["task_specs"]}
        self.assertEqual(profiles, {s["task_profile"] for s in pack["task_specs"]})
        for profile in profiles:
            ours, _, _ = _select_task_spec(pack, profile)
            theirs, _, _ = _select_task_spec(general, profile)
            self.assertEqual(_gates(ours), _gates(theirs), profile)

    def test_every_recipe_eval_pack_resolves(self):
        for recipe in list_recipes():
            self.assertIsNotNone(get_evaluation_pack(recipe.eval_pack_id), f"{recipe.id}: {recipe.eval_pack_id}")

    def test_every_shipped_manifest_preferred_pack_resolves(self):
        manifests = list(_DATA.glob("project_templates/*/manifest.json")) + list(
            _DATA.glob("demo_samples/*/manifest.json")
        )
        self.assertTrue(manifests)
        for path in manifests:
            pack_id = json.loads(path.read_text()).get("evaluation_preferred_pack_id")
            if pack_id:
                self.assertTrue(is_supported_evaluation_pack_id(pack_id), f"{path}: {pack_id}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
