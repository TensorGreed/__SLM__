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

    def test_low_confidence_with_a_task_recipe_is_not_sent_to_rag(self):
        """Phase 7b, gap N6: a crisp tool-call brief scored low on the analyzer's confidence, yet resolved to the
        tool-calling recipe; the chip said "try RAG first" anyway."""
        rec = infer_recommended_approach(
            brief_text="Turn a user request into exactly one JSON tool call.",
            blueprint=_blueprint(task_family="tool_calling", confidence_score=0.2),
            task_evidence=["matched:tool call"],
        )
        self.assertEqual(rec.approach, "sft")
        self.assertIn("task_family:tool_calling (recipe tool-calling)", rec.signals)
        self.assertEqual(rec.headline, "Fine-tuning fits: it matched the tool calling task.")
        self.assertIn("confidence 20%", rec.rationale)

    def test_low_confidence_task_family_without_evidence_stays_rag_first(self):
        """Review M2 (phase 7b): a task family nothing in the brief picked (the analyzer's default) is not a
        clear task shape; a low score still says "try RAG first"."""
        for evidence in (None, []):
            rec = infer_recommended_approach(
                brief_text="I need a model that helps with my domain and my team every day.",
                blueprint=_blueprint(task_family="structured_extraction", confidence_score=0.15),
                task_evidence=evidence,
            )
            self.assertEqual(rec.approach, "rag_first", rec)
            self.assertIn("15%", rec.headline)

    def test_short_span_words_match_as_whole_words_only(self):
        """A short span word inside another word ("tag" in "stage", "span" in "spanish") is no span hit."""
        from app.services.domain_blueprint_service import _infer_task_family, keyword_in_text

        self.assertTrue(keyword_in_text("tag every name", "tag"))
        self.assertTrue(keyword_in_text("each span.", "span"))
        self.assertFalse(keyword_in_text("the market stage", "tag"))
        self.assertFalse(keyword_in_text("translate spanish letters", "span"))
        self.assertTrue(keyword_in_text("labels tickets", "label"))  # other keywords stay substring matches
        family, _ = _infer_task_family("Classify the market stage of each startup pitch.")
        self.assertEqual(family, "classification")
        family, _ = _infer_task_family("Summarize meeting transcripts, highlighting the decisions at each stage.")
        self.assertEqual(family, "summarization")

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
        # Phase 7b, gap N7: the recipe the create applies and its suggested base model, for the create form.
        self.assertEqual(payload["suggested_recipe_id"], "classification")
        self.assertTrue(payload["suggested_base_model"])

    def test_a_crisp_tool_call_brief_is_not_sent_to_rag_via_api(self):
        resp = self.client.post(
            "/api/domain-blueprints/analyze",
            json={
                "brief_text": (
                    "Turn a user request into exactly one JSON tool call. Each request lists the tools available "
                    "for it, then gives today's date and the request. Reply with {\"tool\": \"<name>\", "
                    "\"args\": {...}} using only a listed tool, or {\"tool\": null} when none fits."
                ),
                "llm_enrich": False,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["suggested_recipe_id"], "tool-calling")
        self.assertEqual(payload["recommended_approach"]["approach"], "sft", payload["recommended_approach"])

    def test_a_vague_brief_with_no_keyword_hit_stays_rag_first_via_api(self):
        """Review M2 (phase 7b): the review's repro brief matched nothing, yet the analyzer started below zero and
        picked the first task family (structured extraction → the span-extraction recipe), and the chip said
        "Fine-tuning fits: the brief reads as a clear task shape". Now: instruction_sft, no task recipe, RAG first."""
        resp = self.client.post(
            "/api/domain-blueprints/analyze",
            json={"brief_text": "I need a model that helps with my domain and my team every day.", "llm_enrich": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["blueprint"]["task_family"], "instruction_sft", payload["blueprint"])
        self.assertNotEqual(payload["suggested_recipe_id"], "span-extraction")
        rec = payload["recommended_approach"]
        self.assertEqual(rec["approach"], "rag_first", rec)
        self.assertNotIn("clear task shape", rec["headline"])

    def _analyze(self, brief: str) -> dict:
        resp = self.client.post("/api/domain-blueprints/analyze", json={"brief_text": brief, "llm_enrich": False})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_a_span_tagging_brief_gets_the_span_family_and_sft_via_api(self):
        """Phase 7c: TASK_KEYWORDS had no tag / span wording, so the short check's span brief fell to the
        generic default and the chip said "try RAG first". Span wording now maps to the span family."""
        for brief in (
            "Tag every person name, email address and phone number span",
            "Redact names and addresses from customer letters.",
            "Find named entities in news articles.",
            "Mark mentions of drugs and dosages in clinical notes.",
        ):
            payload = self._analyze(brief)
            self.assertEqual(payload["blueprint"]["task_family"], "structured_extraction", (brief, payload["blueprint"]))
            self.assertEqual(payload["suggested_recipe_id"], "span-extraction", brief)
            self.assertEqual(payload["recommended_approach"]["approach"], "sft", (brief, payload["recommended_approach"]))

    def test_vague_briefs_still_go_rag_first_via_api(self):
        for brief in (
            "I need a model that helps with my domain and my team every day.",
            "Build a model that does the thing.",
            "Make our team more productive at work.",
        ):
            payload = self._analyze(brief)
            self.assertEqual(payload["blueprint"]["task_family"], "instruction_sft", (brief, payload["blueprint"]))
            self.assertEqual(payload["recommended_approach"]["approach"], "rag_first", (brief, payload["recommended_approach"]))

    def test_the_clear_briefs_keep_their_task_family_via_api(self):
        """The four clear briefs from the phase 7b short check map as before the span wording was added."""
        cases = (
            ("Classify support tickets as billing, bug or other.", "classification", "classification"),
            ("Extract the invoice number, date and total from each invoice as JSON.", "structured_extraction", None),
            ("Turn a user request into exactly one JSON tool call.", "tool_calling", "tool-calling"),
            ("Summarize call transcripts into a short recap.", "summarization", None),
        )
        for brief, family, recipe in cases:
            payload = self._analyze(brief)
            self.assertEqual(payload["blueprint"]["task_family"], family, (brief, payload["blueprint"]))
            if recipe:
                self.assertEqual(payload["suggested_recipe_id"], recipe, brief)
            self.assertEqual(payload["recommended_approach"]["approach"], "sft", (brief, payload["recommended_approach"]))

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
