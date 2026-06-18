"""Coach-stage-2 phase 8 — platform-authored held-out probe packs.

Covers:
- The pure registry: applicable packs per task_profile, kind summaries,
  honest ready_not_run status, and the no-pack fallback.
- Probe content integrity (every probe well-formed; properties in the
  closed enum; robustness probes carry a base_input to compare against).
- The API: recipe → task_profile → pack resolution, no-recipe fallback,
  and 404 on a missing project.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "probe_pack.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "probe_pack_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from app.services.probe_pack_service import (  # noqa: E402
    PROBE_GATE_DEFAULT_THRESHOLD,
    PROBE_PACK_VERSION,
    ProbeProperty,
    get_probe_pack,
    read_probe_gate_config,
)

_VALID_PROPERTIES = set(ProbeProperty.__args__)  # type: ignore[attr-defined]


class ProbePackRegistryTests(unittest.TestCase):
    def test_classification_pack_is_applicable_and_well_formed(self):
        pack = get_probe_pack("classification")
        self.assertTrue(pack["applicable"])
        self.assertEqual(pack["version"], PROBE_PACK_VERSION)
        self.assertGreater(pack["probe_count"], 0)
        self.assertEqual(pack["probe_count"], len(pack["probes"]))
        # Honest status — assembled, not yet graded against the model.
        self.assertEqual(pack["status"], "ready_not_run")
        # kind_summary counts add up to probe_count.
        self.assertEqual(sum(pack["kind_summary"].values()), pack["probe_count"])

    def test_every_probe_is_well_formed_across_all_packs(self):
        for profile in (
            "classification",
            "instruction_sft",
            "rag_qa",
            "structured_extraction",
            "summarization",
        ):
            pack = get_probe_pack(profile)
            self.assertTrue(pack["applicable"], profile)
            ids = set()
            for p in pack["probes"]:
                for key in ("id", "probe_kind", "property", "input", "rationale"):
                    self.assertIn(key, p, f"{profile}:{p.get('id')}")
                    self.assertTrue(str(p[key]) != "" or key == "input")
                self.assertIn(p["property"], _VALID_PROPERTIES, p["id"])
                # ids unique within a pack.
                self.assertNotIn(p["id"], ids)
                ids.add(p["id"])
                # Stability probes must carry a base_input to compare to.
                if p["property"] == "prediction_stable_vs_base":
                    self.assertIn("base_input", p, p["id"])
                    self.assertNotEqual(p["base_input"], p["input"])

    def test_instruction_sft_has_safety_refusal_probes(self):
        pack = get_probe_pack("instruction_sft")
        kinds = {p["probe_kind"] for p in pack["probes"]}
        self.assertIn("safety_refusal", kinds)
        # The injection probe must exist — that's the canary.
        self.assertTrue(any("injection" in p["id"] for p in pack["probes"]))

    ALL_PROFILES = (
        "classification", "instruction_sft", "rag_qa",
        "structured_extraction", "summarization",
    )

    def test_every_profile_meets_minimum_depth(self):
        # Phase 19 — each profile carries a meaningful battery, not 2-3 probes.
        for profile in self.ALL_PROFILES:
            pack = get_probe_pack(profile)
            self.assertGreaterEqual(pack["probe_count"], 5, profile)

    def test_profiles_cover_their_key_property_kinds(self):
        expected = {
            "classification": {
                "prediction_stable_vs_base", "handles_degenerate_gracefully",
            },
            "instruction_sft": {
                "refuses_or_declines", "prediction_stable_vs_base",
                "handles_degenerate_gracefully", "does_not_over_refuse",
            },
            "rag_qa": {
                "no_fabrication_when_unsupported", "refuses_or_declines",
                "does_not_over_refuse",
            },
            "structured_extraction": {
                "no_fabrication_when_unsupported", "handles_degenerate_gracefully",
                "prediction_stable_vs_base", "does_not_over_refuse",
            },
            "summarization": {
                "no_fabrication_when_unsupported", "handles_degenerate_gracefully",
                "does_not_over_refuse",
            },
        }
        for profile, props in expected.items():
            present = {p["property"] for p in get_probe_pack(profile)["probes"]}
            self.assertTrue(
                props.issubset(present),
                f"{profile} missing {props - present}",
            )

    def test_unknown_profile_returns_inapplicable_pack(self):
        pack = get_probe_pack("totally-made-up-profile")
        self.assertFalse(pack["applicable"])
        self.assertEqual(pack["probe_count"], 0)
        self.assertEqual(pack["status"], "no_pack_for_profile")

    def test_none_profile_returns_inapplicable_pack(self):
        pack = get_probe_pack(None)
        self.assertFalse(pack["applicable"])


class ProbeGateConfigTests(unittest.TestCase):
    def test_default_config_is_off(self):
        cfg = read_probe_gate_config(SimpleNamespace(runtime_config=None))
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["min_pass_rate"], PROBE_GATE_DEFAULT_THRESHOLD)
        self.assertTrue(cfg["required"])

    def test_reads_enabled_config(self):
        proj = SimpleNamespace(runtime_config={
            "probe_gate": {"enabled": True, "min_pass_rate": 0.8, "required": False},
        })
        cfg = read_probe_gate_config(proj)
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["min_pass_rate"], 0.8)
        self.assertFalse(cfg["required"])

    def test_resolve_probe_gate_off_by_default(self):
        from app.services.evaluation_pack_service import _resolve_probe_gate
        self.assertIsNone(_resolve_probe_gate(SimpleNamespace(runtime_config={})))

    def test_resolve_probe_gate_when_enabled(self):
        from app.services.evaluation_pack_service import _resolve_probe_gate
        proj = SimpleNamespace(runtime_config={
            "probe_gate": {"enabled": True, "min_pass_rate": 0.75},
        })
        gate = _resolve_probe_gate(proj)
        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual(gate["metric_id"], "probe_pass_rate")
        self.assertEqual(gate["operator"], "gte")
        self.assertEqual(gate["threshold"], 0.75)
        self.assertEqual(gate["gate_id"], "min_probe_pass_rate")

    def test_read_kind_weights_merges_and_validates(self):
        from app.services.probe_pack_service import read_probe_kind_weights
        proj = SimpleNamespace(runtime_config={"probe_kind_weights": {
            "safety_refusal": 5.0,   # valid override
            "bogus_kind": 9,         # unknown → dropped
            "robustness": 99,        # out of range → default kept
        }})
        w = read_probe_kind_weights(proj)
        self.assertEqual(w["safety_refusal"], 5.0)
        self.assertNotIn("bogus_kind", w)
        self.assertEqual(w["robustness"], 1.0)
        self.assertEqual(w["format_robustness"], 2.0)  # default preserved

    def test_read_kind_weights_defaults_when_unset(self):
        from app.services.probe_pack_service import read_probe_kind_weights
        w = read_probe_kind_weights(SimpleNamespace(runtime_config=None))
        self.assertEqual(w["safety_refusal"], 3.0)


class ProbePackApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _create_project(self, recipe_id: str | None = None) -> int:
        resp = self.client.post(
            "/api/projects", json={"name": f"pp-{uuid.uuid4().hex[:8]}"}
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        if recipe_id:
            async def _set():
                async with async_session_factory() as db:
                    from app.models.project import Project
                    proj = await db.get(Project, pid)
                    proj.selected_recipe = {"recipe_id": recipe_id}
                    await db.commit()
            asyncio.run(_set())
        return pid

    def test_get_probe_pack_resolves_recipe_to_pack(self):
        pid = self._create_project(recipe_id="classification")
        resp = self.client.get(f"/api/projects/{pid}/probe-pack")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["applicable"])
        self.assertEqual(body["task_profile"], "classification")
        self.assertEqual(body["project_id"], pid)
        self.assertGreater(body["probe_count"], 0)

    def test_get_probe_pack_no_recipe_is_inapplicable(self):
        pid = self._create_project()
        body = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        self.assertFalse(body["applicable"])

    def test_get_probe_pack_missing_project_404s(self):
        resp = self.client.get("/api/projects/99887766/probe-pack")
        self.assertEqual(resp.status_code, 404)

    def test_divergence_history_is_chronological_and_in_payload(self):
        pid = self._create_project(recipe_id="classification")

        async def _seed_and_read():
            async with async_session_factory() as db:
                from app.models.experiment import (
                    EvalResult, Experiment, ExperimentStatus, TrainingMode,
                )
                from app.services.probe_pack_service import get_divergence_history
                # Two training runs; the gap narrows + the weight regime
                # changes between them (phase 23).
                for i, (gold, probe, regime) in enumerate(
                    [(0.90, 0.50, "regimeAAA"), (0.92, 0.70, "regimeBBB")]
                ):
                    exp = Experiment(
                        project_id=pid, name=f"e{i}", base_model="m",
                        status=ExperimentStatus.COMPLETED,
                        training_mode=TrainingMode.SFT,
                    )
                    db.add(exp)
                    await db.flush()
                    db.add(EvalResult(
                        experiment_id=exp.id, dataset_name="gold_test",
                        eval_type="classification", pass_rate=gold,
                        metrics={
                            "pass_rate": gold, "probe_pass_rate": probe,
                            "probe": {
                                "probe_pass_rate": probe, "weight_regime": regime,
                            },
                        },
                    ))
                await db.commit()
                return await get_divergence_history(db, pid, limit=8)

        history = asyncio.run(_seed_and_read())
        self.assertEqual(len(history), 2)
        # Chronological: oldest run (the wider 0.40 gap) first.
        self.assertAlmostEqual(history[0]["probe_pass_rate"], 0.50)
        self.assertAlmostEqual(history[0]["divergence"], 0.40, places=5)
        self.assertAlmostEqual(history[-1]["probe_pass_rate"], 0.70)
        # Phase 23 — the weight regime is surfaced + differs across the runs.
        self.assertEqual(history[0]["weight_regime"], "regimeAAA")
        self.assertEqual(history[1]["weight_regime"], "regimeBBB")
        # The pack payload carries the history for the panel sparkline.
        body = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        self.assertIn("divergence_history", body)
        self.assertEqual(len(body["divergence_history"]), 2)

    def test_gate_config_defaults_off_and_put_round_trips(self):
        pid = self._create_project(recipe_id="classification")
        body = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        self.assertIn("gate_config", body)
        self.assertFalse(body["gate_config"]["enabled"])

        put = self.client.put(
            f"/api/projects/{pid}/probe-pack/gate",
            json={"enabled": True, "min_pass_rate": 0.8, "required": True},
        )
        self.assertEqual(put.status_code, 200, put.text)
        self.assertTrue(put.json()["enabled"])
        # GET reflects the new config.
        body2 = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        self.assertTrue(body2["gate_config"]["enabled"])
        self.assertEqual(body2["gate_config"]["min_pass_rate"], 0.8)

    def test_kind_weights_put_round_trips_and_validates(self):
        pid = self._create_project(recipe_id="classification")
        # Defaults present in the pack payload.
        body0 = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        self.assertEqual(body0["kind_weights"]["safety_refusal"], 3.0)

        put = self.client.put(
            f"/api/projects/{pid}/probe-pack/kind-weights",
            json={"weights": {"safety_refusal": 5.0, "bogus": 9, "robustness": 99}},
        )
        self.assertEqual(put.status_code, 200, put.text)
        effective = put.json()
        self.assertEqual(effective["safety_refusal"], 5.0)   # valid override
        self.assertNotIn("bogus", effective)                 # unknown dropped
        self.assertEqual(effective["robustness"], 1.0)       # out of range → default
        # GET reflects the persisted override.
        body = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        self.assertEqual(body["kind_weights"]["safety_refusal"], 5.0)

    def test_put_gate_rejects_out_of_range_threshold(self):
        pid = self._create_project(recipe_id="classification")
        resp = self.client.put(
            f"/api/projects/{pid}/probe-pack/gate",
            json={"enabled": True, "min_pass_rate": 1.5},
        )
        self.assertEqual(resp.status_code, 422)

    def test_enabled_probe_gate_is_evaluated_in_auto_gates(self):
        """The whole point: an enabled probe gate enforces the independent
        ruler — it fails when probe_pass_rate is below threshold and passes
        when above."""
        pid = self._create_project(recipe_id="classification")
        self.client.put(
            f"/api/projects/{pid}/probe-pack/gate",
            json={"enabled": True, "min_pass_rate": 0.8, "required": True},
        )

        def _run(probe_rate: float) -> dict:
            async def _go():
                async with async_session_factory() as db:
                    from app.models.experiment import (
                        EvalResult, Experiment, ExperimentStatus, TrainingMode,
                    )
                    from app.services.evaluation_pack_service import (
                        evaluate_experiment_auto_gates,
                    )
                    exp = Experiment(
                        project_id=pid, name="e", base_model="m",
                        status=ExperimentStatus.COMPLETED,
                        training_mode=TrainingMode.SFT,
                    )
                    db.add(exp)
                    await db.flush()
                    db.add(EvalResult(
                        experiment_id=exp.id, dataset_name="gold_test",
                        eval_type="classification", pass_rate=0.95,
                        metrics={"pass_rate": 0.95, "probe_pass_rate": probe_rate},
                    ))
                    await db.commit()
                    return await evaluate_experiment_auto_gates(
                        db, project_id=pid, experiment_id=exp.id,
                    )
            return asyncio.run(_go())

        below = _run(0.5)
        probe_check = next(
            (c for c in below["checks"] if c.get("gate_id") == "min_probe_pass_rate"),
            None,
        )
        self.assertIsNotNone(probe_check, [c.get("gate_id") for c in below["checks"]])
        assert probe_check is not None
        self.assertFalse(probe_check["passed"])  # 0.5 < 0.8
        # Phase 15 — the gate's origin is propagated onto the check, and
        # the probe gate is distinguishable from the pack's own gates.
        self.assertEqual(probe_check.get("gate_source"), "probe_pack")
        for c in below["checks"]:
            if c.get("gate_id") != "min_probe_pass_rate":
                self.assertNotEqual(c.get("gate_source"), "probe_pack")

        above = _run(0.9)
        probe_check2 = next(
            (c for c in above["checks"] if c.get("gate_id") == "min_probe_pass_rate"),
            None,
        )
        assert probe_check2 is not None
        self.assertTrue(probe_check2["passed"])  # 0.9 >= 0.8

    def test_pack_is_ready_not_run_before_any_eval(self):
        pid = self._create_project(recipe_id="classification")
        body = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        self.assertEqual(body["status"], "ready_not_run")
        self.assertNotIn("run", body)

    def test_pack_flips_to_graded_after_a_run_lands_in_eval_metrics(self):
        pid = self._create_project(recipe_id="classification")

        async def _seed():
            async with async_session_factory() as db:
                from app.models.experiment import (
                    EvalResult, Experiment, ExperimentStatus, TrainingMode,
                )
                exp = Experiment(
                    project_id=pid,
                    name="exp",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    status=ExperimentStatus.COMPLETED,
                    training_mode=TrainingMode.SFT,
                )
                db.add(exp)
                await db.flush()
                db.add(EvalResult(
                    experiment_id=exp.id,
                    dataset_name="gold_test",
                    eval_type="classification",
                    pass_rate=0.92,
                    metrics={
                        "pass_rate": 0.92,
                        "probe_pass_rate": 0.75,
                        "probe": {
                            "probe_pass_rate": 0.75,
                            "passed": 3,
                            "total": 4,
                            "per_property": {
                                "prediction_stable_vs_base": {
                                    "passed": 2, "total": 2, "pass_rate": 1.0,
                                },
                                "handles_degenerate_gracefully": {
                                    "passed": 1, "total": 2, "pass_rate": 0.5,
                                },
                            },
                            "results": [
                                {
                                    "id": "clf.robust.casing",
                                    "probe_kind": "robustness",
                                    "property": "prediction_stable_vs_base",
                                    "passed": True,
                                    "output": "billing",
                                    "base_output": "billing",
                                    "reason": "stable",
                                },
                            ],
                        },
                    },
                ))
                await db.commit()
        asyncio.run(_seed())

        body = self.client.get(f"/api/projects/{pid}/probe-pack").json()
        # The pack now reflects the independent run.
        self.assertEqual(body["status"], "graded")
        self.assertIn("run", body)
        run = body["run"]
        self.assertEqual(run["probe_pass_rate"], 0.75)
        self.assertEqual(run["passed"], 3)
        self.assertEqual(run["total"], 4)
        self.assertIn("prediction_stable_vs_base", run["per_property"])
        self.assertEqual(run["results"][0]["id"], "clf.robust.casing")


if __name__ == "__main__":
    unittest.main()
