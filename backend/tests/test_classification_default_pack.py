"""The built-in ``evalpack.classification.default`` pack.

The classification recipe, the classification project templates and the
classification demo sample all name this id. Before it existed it resolved to
nothing: the readiness ledger showed no gate breakdown and the project
resolver warned "not available; falling back" and gated on the next pack in
the chain — the domain-profile pack, whose qa spec requires f1 /
llm_judge_pass_rate / safety, metrics a classification eval does not emit.
Those projects' gates change (as a fix) to accuracy + macro-F1 required,
safety optional. The resolver-level tests pin both sides of that change.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import asyncio
import uuid

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.project import Project
from app.services.evaluation_pack_service import (
    DEFAULT_EVALUATION_PACK_ID,
    resolve_project_evaluation_pack,
    _select_task_spec,
    get_evaluation_pack,
    is_supported_evaluation_pack_id,
    list_evaluation_packs,
)
from app.services.recipe_service import list_recipes

PACK_ID = "evalpack.classification.default"
_CLIENT_CM: TestClient | None = None
_CLIENT: TestClient | None = None


def setUpModule():
    global _CLIENT_CM, _CLIENT
    _CLIENT_CM = TestClient(app)
    _CLIENT = _CLIENT_CM.__enter__()


def tearDownModule():
    if _CLIENT_CM is not None:
        _CLIENT_CM.__exit__(None, None, None)
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

    def test_specs_are_the_general_default_specs(self):
        # Structural: non-classification eval types gate as under general.default.
        # (Not a claim about what projects gated on before — see the resolver tests.)
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


    def test_every_pipeline_recipe_preferred_pack_resolves(self):
        from app.services.pipeline_recipe_service import list_pipeline_recipes

        found = []

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "preferred_pack_id" and isinstance(value, str):
                        found.append(value)
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(list_pipeline_recipes(include_blueprint=True))
        self.assertTrue(found)
        for pack_id in found:
            self.assertTrue(is_supported_evaluation_pack_id(pack_id), pack_id)


class ResolverTests(unittest.TestCase):
    """A project that prefers this pack and has no domain profile of its own,
    as a classification template creates it."""

    def _project_preferring(self, pack_id: str) -> int:
        resp = _CLIENT.post("/api/projects", json={"name": f"cls-pack-{uuid.uuid4().hex[:8]}", "description": "t"})
        self.assertEqual(resp.status_code, 201, resp.text)
        project_id = int(resp.json()["id"])

        async def go():
            async with async_session_factory() as db:
                project = await db.get(Project, project_id)
                project.evaluation_preferred_pack_id = pack_id
                await db.commit()
        asyncio.run(go())
        return project_id

    def _resolve(self, project_id: int) -> dict:
        async def go():
            async with async_session_factory() as db:
                return await resolve_project_evaluation_pack(db, project_id)
        return asyncio.run(go())

    def _classification_gates(self, resolved: dict) -> dict:
        spec, _, _ = _select_task_spec(resolved["pack"], "classification")
        return {g["metric_id"]: g["required"] for g in spec["gates"]}

    def test_preference_resolves_to_the_pack_with_classification_gates(self):
        resolved = self._resolve(self._project_preferring(PACK_ID))
        self.assertEqual(resolved["source"], "project")
        self.assertEqual(resolved["active_pack_id"], PACK_ID)
        self.assertEqual(resolved["warnings"], [])
        gates = self._classification_gates(resolved)
        self.assertTrue(gates["accuracy"])
        self.assertTrue(gates["macro_f1"])
        self.assertFalse(gates["safety_pass_rate"])

    def test_an_unknown_preference_still_falls_back_to_the_profile_pack(self):
        # What every classification template project got before this pack
        # existed: a warning, then the domain-profile pack's qa gates.
        resolved = self._resolve(self._project_preferring("evalpack.does-not-exist"))
        self.assertEqual(resolved["source"], "domain_profile_default")
        self.assertTrue(any("not available" in w for w in resolved["warnings"]))
        gates = self._classification_gates(resolved)
        self.assertNotIn("accuracy", gates)
        self.assertNotIn("macro_f1", gates)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
