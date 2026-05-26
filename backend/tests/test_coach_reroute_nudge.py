"""Tests for the Phase 7d Coach Mode reroute nudge.

Covers ``_reroute_recommendation_nudge`` pure-function truth table
+ the integration into ``_eval_stage_suggestions`` (which reads the
cached RerouteAnalysis off ``EvalResult.details``).
"""

from __future__ import annotations

import asyncio
import os
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.coach_service import (  # noqa: E402
    _reroute_recommendation_nudge,
)


def _analysis(
    *,
    kind: str = "try_rag",
    pass_rate: float | None = 0.35,
    fired_signal_ids: tuple[str, ...] = (
        "brief_mentions_retrieval",
        "goldset_answer_diversity_high",
    ),
    eval_result_id: int = 101,
    confidence: float = 0.85,
) -> dict:
    all_signals = [
        {
            "id": "brief_mentions_retrieval",
            "fired": "brief_mentions_retrieval" in fired_signal_ids,
            "detail": "Your brief mentions retrieval-style language",
            "evidence": {},
        },
        {
            "id": "goldset_answer_diversity_high",
            "fired": "goldset_answer_diversity_high" in fired_signal_ids,
            "detail": "Gold-set answers are highly diverse (mean Jaccard 0.05)",
            "evidence": {},
        },
        {
            "id": "input_output_density_low",
            "fired": "input_output_density_low" in fired_signal_ids,
            "detail": "Output is a tiny slice of input (mean ratio 0.02)",
            "evidence": {},
        },
    ]
    return {
        "eval_result_id": eval_result_id,
        "project_id": 4,
        "pass_rate": pass_rate,
        "signals": all_signals,
        "recommendation": {
            "kind": kind,
            "confidence": confidence,
            "rationale": "Test rationale.",
        },
        "computed_at": "2026-05-26T12:00:00Z",
    }


class RerouteRecommendationNudgePureTests(unittest.TestCase):
    def test_fires_on_try_rag_with_fired_signals(self):
        nudge = _reroute_recommendation_nudge(
            project_id=4,
            reroute_analysis=_analysis(kind="try_rag"),
            is_rag_first_project=False,
        )
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["id"], "eval:reroute-to-rag-recommended")
        self.assertEqual(nudge["severity"], "info")
        self.assertEqual(nudge["action"]["kind"], "navigate")
        self.assertEqual(
            nudge["action"]["params"]["target"], "reroute-recommendation-panel"
        )
        # Body cites the fired signals' detail strings verbatim so the
        # recommendation is auditable from inside the Coach card.
        self.assertIn("retrieval-style language", nudge["body"])
        self.assertIn("highly diverse", nudge["body"])
        # Context surfaces the fired signal ids + the eval_result_id
        # so the frontend can do follow-up queries.
        self.assertEqual(
            set(nudge["context"]["fired_signal_ids"]),
            {"brief_mentions_retrieval", "goldset_answer_diversity_high"},
        )
        self.assertEqual(nudge["context"]["eval_result_id"], 101)
        self.assertEqual(nudge["context"]["recommendation_confidence"], 0.85)

    def test_skipped_when_recommendation_kind_is_not_try_rag(self):
        for kind in ("stay_the_course", "try_prompt_engineering", "expand_data"):
            with self.subTest(kind=kind):
                self.assertIsNone(
                    _reroute_recommendation_nudge(
                        project_id=4,
                        reroute_analysis=_analysis(kind=kind),
                        is_rag_first_project=False,
                    )
                )

    def test_skipped_when_project_is_already_rag_first(self):
        """No infinite reroute chain — don't recommend RAG on a
        project that's already a RAG-first clone."""
        self.assertIsNone(
            _reroute_recommendation_nudge(
                project_id=4,
                reroute_analysis=_analysis(kind="try_rag"),
                is_rag_first_project=True,
            )
        )

    def test_skipped_when_analysis_missing(self):
        self.assertIsNone(
            _reroute_recommendation_nudge(
                project_id=4,
                reroute_analysis=None,
                is_rag_first_project=False,
            )
        )

    def test_skipped_when_analysis_is_not_dict(self):
        # Defensive — cached value could be malformed.
        self.assertIsNone(
            _reroute_recommendation_nudge(
                project_id=4,
                reroute_analysis="not-a-dict",  # type: ignore[arg-type]
                is_rag_first_project=False,
            )
        )

    def test_handles_empty_fired_signals_gracefully(self):
        """try_rag recommendation with no fired signals (e.g. analyzer
        edge case) still fires the nudge — but body omits the evidence
        block so it doesn't render an empty bullet list."""
        nudge = _reroute_recommendation_nudge(
            project_id=4,
            reroute_analysis=_analysis(kind="try_rag", fired_signal_ids=()),
            is_rag_first_project=False,
        )
        self.assertIsNotNone(nudge)
        self.assertNotIn("Signals that fired:", nudge["body"])

    def test_handles_pass_rate_none_in_title_and_body(self):
        nudge = _reroute_recommendation_nudge(
            project_id=4,
            reroute_analysis=_analysis(kind="try_rag", pass_rate=None),
            is_rag_first_project=False,
        )
        self.assertIsNotNone(nudge)
        self.assertIn("below the healthy threshold", nudge["title"])
        self.assertIsNone(nudge["context"]["pass_rate"])


# ─────────────────────────────────────────────────────────────────────
# End-to-end via TestClient — nudge surfaces in /coach/eval response
# ─────────────────────────────────────────────────────────────────────


class CoachEvalEndpointRerouteNudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def _seed_failing_eval(self, project_id: int, description: str) -> int:
        from app.database import async_session_factory
        from app.models.experiment import (
            EvalResult,
            Experiment,
            ExperimentStatus,
            TrainingMode,
        )
        from app.models.project import Project

        async def _go() -> int:
            async with async_session_factory() as db:
                # Overwrite project description so the brief-mentions-
                # retrieval signal fires.
                proj = await db.get(Project, project_id)
                proj.description = description
                await db.flush()
                exp = Experiment(
                    project_id=project_id,
                    name="phase7d-fixture",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    # auto_rag explicitly off — analyzer's "already_rag"
                    # suppression won't fire, so we can drive the
                    # try_rag verdict from the brief + diversity signals.
                    config={"auto_rag": {"enabled": False}},
                    training_mode=TrainingMode.SFT,
                    status=ExperimentStatus.COMPLETED,
                )
                db.add(exp)
                await db.flush()
                ev = EvalResult(
                    experiment_id=exp.id,
                    dataset_name="gold_test",
                    eval_type="f1",
                    metrics={"f1": 0.30},
                    pass_rate=0.30,
                    details={},
                )
                db.add(ev)
                await db.flush()
                await db.commit()
                return ev.id

        return asyncio.run(_go())

    def test_coach_eval_surfaces_reroute_nudge_on_struggling_qa_sft_project(self):
        # policy-qa-style template → qa-sft recipe → eligible for the
        # reroute analyzer. Failing eval (0.30) + retrieval-shaped
        # brief → try_rag recommendation → nudge.
        project = self._instantiate_template(
            "policy-qa-style", "Phase7d Coach Eval Nudge"
        )
        self._seed_failing_eval(
            project["id"],
            description=(
                "An assistant that answers questions about company "
                "policies. The model looks up the relevant policy and "
                "responds accurately."
            ),
        )
        resp = self.client.get(f"/api/projects/{project['id']}/coach/eval")
        self.assertEqual(resp.status_code, 200, resp.text)
        suggestion_ids = [s["id"] for s in resp.json()["suggestions"]]
        self.assertIn("eval:reroute-to-rag-recommended", suggestion_ids)

    def test_coach_eval_omits_nudge_on_rag_first_clone(self):
        # Clone a qa-sft project, then seed an eval on the clone. The
        # clone's runtime_config.rag_first = True so the nudge should
        # skip (don't recommend what's already on).
        source = self._instantiate_template(
            "policy-qa-style", "Phase7d Coach No Nudge On Clone"
        )
        clone_resp = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(clone_resp.status_code, 201, clone_resp.text)
        new_id = clone_resp.json()["new_project_id"]
        self._seed_failing_eval(
            new_id,
            description=(
                "Look up answers from the policy. Knowledge base style."
            ),
        )
        resp = self.client.get(f"/api/projects/{new_id}/coach/eval")
        self.assertEqual(resp.status_code, 200, resp.text)
        suggestion_ids = [s["id"] for s in resp.json()["suggestions"]]
        self.assertNotIn("eval:reroute-to-rag-recommended", suggestion_ids)


if __name__ == "__main__":
    unittest.main()
