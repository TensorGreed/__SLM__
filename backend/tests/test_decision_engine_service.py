"""Tests for the Theme 7 decision engine — `infer_recommended_approach`."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "decision_engine_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "decision_engine_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemas.domain_blueprint import DomainBlueprintContract
from app.services.decision_engine_service import (
    ApproachKind,
    infer_recommended_approach,
)


def _blueprint(
    *,
    task_family: str = "instruction_sft",
    confidence_score: float = 0.7,
    expected_output_schema: dict | None = None,
) -> DomainBlueprintContract:
    """Build a minimal blueprint for unit-testing the decision
    engine in isolation from the full brief-analysis pipeline."""
    return DomainBlueprintContract(
        domain_name="test",
        problem_statement="test problem",
        target_user_persona="test persona",
        task_family=task_family,
        input_modality="text",
        expected_output_schema=expected_output_schema or {},
        expected_output_examples=[],
        safety_compliance_notes=[],
        deployment_target_constraints={},
        success_metrics=[],
        glossary=[],
        confidence_score=confidence_score,
        unresolved_assumptions=[],
    )


class DecisionEngineUnitTests(unittest.TestCase):
    """Heuristic checks — one test per branch of the decision tree."""

    def test_distillation_signal_wins_with_size_cost_language(self):
        rec = infer_recommended_approach(
            brief_text="I want a smaller, cheaper model that runs on edge devices.",
            blueprint=_blueprint(),
        )
        self.assertEqual(rec.approach, "distillation")
        self.assertGreater(rec.confidence, 0.6)
        self.assertTrue(any(s.startswith("keywords:distillation") for s in rec.signals))
        self.assertIn("smaller", rec.rationale.lower())

    def test_dpo_signal_wins_with_preference_language(self):
        rec = infer_recommended_approach(
            brief_text="Train on (chosen, rejected) preference pairs from human feedback.",
            blueprint=_blueprint(),
        )
        self.assertEqual(rec.approach, "dpo")
        self.assertTrue(any("dpo" in s.lower() for s in rec.signals))

    def test_rag_signal_wins_with_lookup_language(self):
        rec = infer_recommended_approach(
            brief_text="Answer FAQs by looking up policies from our knowledge base.",
            blueprint=_blueprint(task_family="qa"),
        )
        self.assertEqual(rec.approach, "rag_first")
        self.assertIn("RAG", rec.headline)

    def test_rag_qa_task_family_alone_triggers_rag(self):
        rec = infer_recommended_approach(
            brief_text="Some generic brief about a model.",
            blueprint=_blueprint(task_family="rag_qa"),
        )
        self.assertEqual(rec.approach, "rag_first")
        self.assertIn("task_family:rag_qa", rec.signals)

    def test_low_blueprint_confidence_triggers_rag_with_confidence_chip(self):
        rec = infer_recommended_approach(
            brief_text="I need a model that helps with my domain.",
            blueprint=_blueprint(confidence_score=0.36),
        )
        self.assertEqual(rec.approach, "rag_first")
        self.assertIn("36%", rec.headline)
        self.assertTrue(any(s.startswith("blueprint.confidence") for s in rec.signals))

    def test_prompt_only_needs_all_three_signals_present(self):
        # Few-shot keyword + small schema + short brief — all three required.
        rec = infer_recommended_approach(
            brief_text="Few-shot examples for sentiment.",
            blueprint=_blueprint(
                expected_output_schema={
                    "properties": {"sentiment": {"type": "string"}},
                },
            ),
        )
        self.assertEqual(rec.approach, "prompt_only")
        self.assertIn("few-shot", rec.headline.lower())

    def test_few_shot_keyword_alone_does_not_trigger_prompt_only(self):
        # Long brief — prompt-only guard requires <= 25 words.
        long_brief = " ".join(
            ["few-shot examples please"]
            + ["word" for _ in range(40)]
        )
        rec = infer_recommended_approach(
            brief_text=long_brief,
            blueprint=_blueprint(
                expected_output_schema={"properties": {"x": {"type": "string"}}},
            ),
        )
        self.assertNotEqual(rec.approach, "prompt_only")

    def test_style_brief_recommends_sft_with_style_signals(self):
        rec = infer_recommended_approach(
            brief_text="Make the model reply in our company's friendly tone and JSON schema.",
            blueprint=_blueprint(),
        )
        self.assertEqual(rec.approach, "sft")
        self.assertTrue(any(s.startswith("keyword:style") for s in rec.signals))
        self.assertIn("SFT", rec.headline)

    def test_default_falls_through_to_sft_when_no_signals_match(self):
        rec = infer_recommended_approach(
            brief_text="Build a model that does the thing.",
            blueprint=_blueprint(confidence_score=0.7),
        )
        self.assertEqual(rec.approach, "sft")
        self.assertIn("default:no_signals_matched", rec.signals)

    def test_empty_brief_still_returns_a_recommendation(self):
        rec = infer_recommended_approach(
            brief_text="",
            blueprint=_blueprint(confidence_score=0.8),
        )
        self.assertIsInstance(rec.approach, str)
        self.assertIn(
            rec.approach,
            {"prompt_only", "rag_first", "sft", "dpo", "distillation"},
        )

    def test_all_returned_approaches_are_in_the_union(self):
        # Belt-and-suspenders — the response Literal contract.
        for brief, expected in [
            ("Knowledge base lookup", "rag_first"),
            ("Distill into a tiny model", "distillation"),
            ("Train on preference pairs", "dpo"),
            ("Just match this JSON schema and tone", "sft"),
        ]:
            rec = infer_recommended_approach(brief, _blueprint())
            self.assertIn(rec.approach, ApproachKind.__args__)
            self.assertEqual(rec.approach, expected, f"brief={brief!r}")

    def test_rationale_quotes_a_clipped_brief_snippet_when_relevant(self):
        rec = infer_recommended_approach(
            brief_text=(
                "Help our agents look up policies and answer customer "
                "questions from the documentation we've already shipped."
            ),
            blueprint=_blueprint(),
        )
        self.assertEqual(rec.approach, "rag_first")
        # Rationale mentions either a quoted snippet from the brief
        # or a confidence/signal callout.
        self.assertGreater(len(rec.rationale), 40)


class AnalyzeBriefIntegrationTests(unittest.TestCase):
    """Confirm `recommended_approach` shows up in the live
    `/api/domain-blueprints/analyze` response shape."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def test_analyze_response_includes_recommended_approach(self):
        resp = self.client.post(
            "/api/domain-blueprints/analyze",
            json={
                "brief_text": "Train a classifier that labels support tickets as urgent / normal / spam.",
                "llm_enrich": False,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertIn("recommended_approach", payload)
        rec = payload["recommended_approach"]
        self.assertIn(
            rec["approach"],
            {"prompt_only", "rag_first", "sft", "dpo", "distillation"},
        )
        self.assertGreater(len(rec["headline"]), 0)
        self.assertGreater(len(rec["rationale"]), 0)

    def test_lookup_brief_returns_rag_first_via_api(self):
        resp = self.client.post(
            "/api/domain-blueprints/analyze",
            json={
                "brief_text": "Look up answers to support questions from our help center docs.",
                "llm_enrich": False,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        rec = resp.json()["recommended_approach"]
        self.assertEqual(rec["approach"], "rag_first")
        self.assertGreater(rec["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
