"""Tests for the post-eval decision engine service
(USER-SUCCESS Epic 7 Phase 7a).

Covers:
- Pure-function signal truth table (3 signals × fires/doesn't-fire).
- Row-shape walkers across the gold-row shape variants the platform
  ships (template-style ``{question, answer}``, structured
  ``{input, expected: {label: ...}}``, raw ``{expected: <str>}``).
- ``_classify_recommendation`` priority logic — including the
  "RAG wins over density" tie-break and the
  "suppress try_rag when project is already qa-sft + auto_rag on"
  guard.
- ``analyze_eval_for_reroute`` end-to-end via the FastAPI
  TestClient: instantiates a real project template, manually
  creates an Experiment + EvalResult, hits
  ``GET /api/projects/{id}/evaluation/{eval_id}/reroute-analysis``,
  asserts the cache hit on the 2nd call and the
  ``?refresh=true`` recompute path.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.post_eval_decision_engine_service import (  # noqa: E402
    _DENSITY_THRESHOLD,
    _DIVERSITY_THRESHOLD,
    _PASS_RATE_THRESHOLD,
    _classify_recommendation,
    _extract_input_text,
    _extract_output_text,
    _jaccard,
    _matching_retrieval_keywords,
    _mean_pairwise_jaccard,
    _signal_brief_mentions_retrieval,
    _signal_goldset_answer_diversity_high,
    _signal_input_output_density_low,
    _tokenize,
)


# ─────────────────────────────────────────────────────────────────────
# Row-shape walkers
# ─────────────────────────────────────────────────────────────────────


class RowShapeWalkerTests(unittest.TestCase):
    def test_extract_output_from_flat_answer(self):
        self.assertEqual(_extract_output_text({"question": "q", "answer": "a"}), "a")

    def test_extract_output_from_nested_expected_label(self):
        self.assertEqual(
            _extract_output_text({"input": "x", "expected": {"label": "ROUTE_A"}}),
            "ROUTE_A",
        )

    def test_extract_output_from_raw_expected_string(self):
        self.assertEqual(
            _extract_output_text({"input": "x", "expected": "the answer"}),
            "the answer",
        )

    def test_extract_output_returns_empty_when_unstructured(self):
        self.assertEqual(_extract_output_text({"foo": "bar"}), "")

    def test_extract_input_walks_question_then_input_then_prompt(self):
        self.assertEqual(_extract_input_text({"question": "Q"}), "Q")
        self.assertEqual(_extract_input_text({"input": "I"}), "I")
        self.assertEqual(_extract_input_text({"prompt": "P"}), "P")

    def test_tokenize_lowercases_and_drops_punctuation(self):
        self.assertEqual(
            _tokenize("Hello, World! Don't worry."),
            frozenset({"hello", "world", "don't", "worry"}),
        )

    def test_jaccard_basic_overlap(self):
        a = frozenset({"hello", "world"})
        b = frozenset({"hello", "earth"})
        # {hello} / {hello, world, earth} = 1/3
        self.assertAlmostEqual(_jaccard(a, b), 1 / 3)

    def test_jaccard_empty_returns_zero(self):
        self.assertEqual(_jaccard(frozenset(), frozenset()), 0.0)

    def test_mean_pairwise_jaccard_none_when_fewer_than_two(self):
        self.assertIsNone(_mean_pairwise_jaccard([frozenset({"a", "b"})]))
        self.assertIsNone(_mean_pairwise_jaccard([]))

    def test_mean_pairwise_jaccard_high_when_rows_overlap(self):
        sets = [frozenset({"a", "b", "c"}) for _ in range(5)]
        # All identical → mean Jaccard = 1.0
        self.assertEqual(_mean_pairwise_jaccard(sets), 1.0)

    def test_mean_pairwise_jaccard_low_when_rows_distinct(self):
        sets = [frozenset({"a"}), frozenset({"b"}), frozenset({"c"})]
        # Pairwise Jaccard = 0 for all three pairs → mean = 0
        self.assertEqual(_mean_pairwise_jaccard(sets), 0.0)


# ─────────────────────────────────────────────────────────────────────
# Signal: brief_mentions_retrieval
# ─────────────────────────────────────────────────────────────────────


class BriefMentionsRetrievalSignalTests(unittest.TestCase):
    def test_matches_explicit_retrieval_phrasing(self):
        sig = _signal_brief_mentions_retrieval(
            "Build a chatbot that answers questions about our HR policies."
        )
        self.assertTrue(sig["fired"])
        # Either the singular or plural "answer(s) questions about"
        # variant should match.
        matched = sig["evidence"]["matched_keywords"]
        self.assertTrue(
            any("answer" in kw and "questions about" in kw for kw in matched),
            f"expected an 'answer(s) questions about' match, got {matched}",
        )

    def test_matches_knowledge_base_phrase(self):
        sig = _signal_brief_mentions_retrieval(
            "Support chatbot powered by our internal knowledge base."
        )
        self.assertTrue(sig["fired"])
        matched = sig["evidence"]["matched_keywords"]
        # Both "knowledge base" and "support chatbot" should match.
        self.assertIn("knowledge base", matched)
        self.assertIn("support chatbot", matched)

    def test_case_insensitive(self):
        sig = _signal_brief_mentions_retrieval(
            "ANSWER QUESTIONS ABOUT compliance topics."
        )
        self.assertTrue(sig["fired"])

    def test_does_not_match_generic_classification_brief(self):
        sig = _signal_brief_mentions_retrieval(
            "Classify incoming support tickets by department."
        )
        self.assertFalse(sig["fired"])
        self.assertEqual(sig["evidence"]["matched_keywords"], [])

    def test_does_not_match_empty_description(self):
        sig = _signal_brief_mentions_retrieval("")
        self.assertFalse(sig["fired"])

    def test_keyword_list_avoids_bare_documentation(self):
        """``documentation`` alone is too common to fire — there must
        be retrieval-specific context. This protects against the
        false positive of a non-RAG project whose brief just happens
        to mention writing or reading docs."""
        self.assertEqual(
            _matching_retrieval_keywords(
                "Generate user-facing documentation for our API endpoints."
            ),
            [],
        )


# ─────────────────────────────────────────────────────────────────────
# Signal: goldset_answer_diversity_high
# ─────────────────────────────────────────────────────────────────────


class AnswerDiversitySignalTests(unittest.TestCase):
    def test_fires_when_answers_are_distinct(self):
        # 5 rows, each answer uses entirely different vocabulary →
        # mean pairwise Jaccard near 0 → far below the 0.20 threshold.
        rows = [
            {"question": "q1", "answer": "alpha beta gamma"},
            {"question": "q2", "answer": "delta epsilon zeta"},
            {"question": "q3", "answer": "eta theta iota"},
            {"question": "q4", "answer": "kappa lambda mu"},
            {"question": "q5", "answer": "nu xi omicron"},
        ]
        sig = _signal_goldset_answer_diversity_high(rows)
        self.assertTrue(sig["fired"])
        self.assertLess(sig["evidence"]["mean_pairwise_jaccard"], _DIVERSITY_THRESHOLD)
        self.assertEqual(sig["evidence"]["n_rows"], 5)

    def test_does_not_fire_when_answers_share_vocabulary(self):
        # Classification-style rows where every answer is one of two
        # labels → mean Jaccard high.
        rows = [{"input": f"row {i}", "expected": {"label": "ROUTE_A"}} for i in range(5)]
        sig = _signal_goldset_answer_diversity_high(rows)
        self.assertFalse(sig["fired"])

    def test_does_not_fire_when_too_few_rows(self):
        sig = _signal_goldset_answer_diversity_high(
            [{"question": "q", "answer": "a"}]
        )
        self.assertFalse(sig["fired"])
        self.assertIsNone(sig["evidence"]["mean_pairwise_jaccard"])

    def test_does_not_fire_on_empty_gold_set(self):
        sig = _signal_goldset_answer_diversity_high([])
        self.assertFalse(sig["fired"])
        self.assertIsNone(sig["evidence"]["mean_pairwise_jaccard"])


# ─────────────────────────────────────────────────────────────────────
# Signal: input_output_density_low
# ─────────────────────────────────────────────────────────────────────


class InputOutputDensitySignalTests(unittest.TestCase):
    def test_fires_when_output_is_tiny_slice_of_input(self):
        # 1000-char inputs, 20-char outputs → ratio 0.02 < 0.05.
        long_doc = "a" * 1000
        rows = [{"input": long_doc, "expected": "yes"} for _ in range(3)]
        sig = _signal_input_output_density_low(rows)
        self.assertTrue(sig["fired"])
        self.assertLess(sig["evidence"]["mean_density"], _DENSITY_THRESHOLD)

    def test_does_not_fire_when_output_is_substantial(self):
        # 100-char input, 80-char output → ratio 0.8.
        rows = [
            {"question": "x" * 100, "answer": "y" * 80} for _ in range(3)
        ]
        sig = _signal_input_output_density_low(rows)
        self.assertFalse(sig["fired"])
        self.assertGreater(sig["evidence"]["mean_density"], _DENSITY_THRESHOLD)

    def test_does_not_fire_on_empty_gold_set(self):
        sig = _signal_input_output_density_low([])
        self.assertFalse(sig["fired"])
        self.assertIsNone(sig["evidence"]["mean_density"])

    def test_skips_rows_with_empty_input(self):
        rows = [
            {"input": "", "expected": "yes"},  # skipped
            {"input": "a" * 1000, "expected": "yes"},  # counted
        ]
        sig = _signal_input_output_density_low(rows)
        self.assertEqual(sig["evidence"]["n_rows"], 1)


# ─────────────────────────────────────────────────────────────────────
# _classify_recommendation
# ─────────────────────────────────────────────────────────────────────


def _signal(sid: str, fired: bool, **evidence) -> dict:
    return {"id": sid, "fired": fired, "detail": "", "evidence": evidence}


class ClassifyRecommendationTests(unittest.TestCase):
    def _build_signals(self, brief: bool, diversity: bool, density: bool) -> list[dict]:
        return [
            _signal("brief_mentions_retrieval", brief),
            _signal("goldset_answer_diversity_high", diversity),
            _signal("input_output_density_low", density),
        ]

    def test_passing_eval_returns_stay_the_course(self):
        # Even with every signal firing, a passing eval doesn't
        # recommend rerouting — panel self-hides.
        rec = _classify_recommendation(
            signals=self._build_signals(True, True, True),
            pass_rate=0.85,
            recipe_id="qa-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec["kind"], "stay_the_course")

    def test_at_threshold_returns_stay_the_course(self):
        rec = _classify_recommendation(
            signals=self._build_signals(False, False, False),
            pass_rate=_PASS_RATE_THRESHOLD,
            recipe_id="generic-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec["kind"], "stay_the_course")

    def test_brief_signal_alone_triggers_try_rag(self):
        rec = _classify_recommendation(
            signals=self._build_signals(True, False, False),
            pass_rate=0.40,
            recipe_id="generic-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec["kind"], "try_rag")
        # Single-signal confidence floors below the both-fired bonus.
        self.assertLess(rec["confidence"], 0.85)

    def test_both_rag_signals_lift_confidence(self):
        rec_both = _classify_recommendation(
            signals=self._build_signals(True, True, False),
            pass_rate=0.40,
            recipe_id="generic-sft",
            auto_rag_enabled=False,
        )
        rec_one = _classify_recommendation(
            signals=self._build_signals(True, False, False),
            pass_rate=0.40,
            recipe_id="generic-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec_both["kind"], "try_rag")
        self.assertEqual(rec_one["kind"], "try_rag")
        self.assertGreater(rec_both["confidence"], rec_one["confidence"])

    def test_density_alone_triggers_try_prompt_engineering(self):
        rec = _classify_recommendation(
            signals=self._build_signals(False, False, True),
            pass_rate=0.30,
            recipe_id="generic-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec["kind"], "try_prompt_engineering")

    def test_rag_wins_over_density_when_both_fire(self):
        """When a RAG signal AND density signal both fire, RAG wins
        — the recommendation priority is RAG > density > catch-all."""
        rec = _classify_recommendation(
            signals=self._build_signals(True, False, True),
            pass_rate=0.30,
            recipe_id="generic-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec["kind"], "try_rag")

    def test_no_signal_fires_yields_expand_data(self):
        rec = _classify_recommendation(
            signals=self._build_signals(False, False, False),
            pass_rate=0.30,
            recipe_id="generic-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec["kind"], "expand_data")

    def test_try_rag_suppressed_when_already_qa_sft_with_auto_rag_on(self):
        """No point recommending RAG to a project that already runs
        auto-RAG — that route is already taken."""
        rec = _classify_recommendation(
            signals=self._build_signals(True, True, False),
            pass_rate=0.30,
            recipe_id="qa-sft",
            auto_rag_enabled=True,
        )
        self.assertNotEqual(rec["kind"], "try_rag")
        # With density off, the catch-all is expand_data.
        self.assertEqual(rec["kind"], "expand_data")

    def test_try_rag_not_suppressed_when_recipe_is_qa_sft_but_auto_rag_off(self):
        """A qa-sft project where someone explicitly turned auto_rag
        off is still a candidate for the reroute recommendation —
        the user might benefit from turning it back on (or
        cloning into a true RAG-first project in Phase 7b)."""
        rec = _classify_recommendation(
            signals=self._build_signals(True, True, False),
            pass_rate=0.30,
            recipe_id="qa-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec["kind"], "try_rag")

    def test_handles_pass_rate_none_as_struggling(self):
        """An eval with no pass_rate (e.g. safety eval) shouldn't
        block reroute — treat as struggling so the user gets a
        recommendation if any signal fires."""
        rec = _classify_recommendation(
            signals=self._build_signals(True, True, False),
            pass_rate=None,
            recipe_id="generic-sft",
            auto_rag_enabled=False,
        )
        self.assertEqual(rec["kind"], "try_rag")


# ─────────────────────────────────────────────────────────────────────
# analyze_eval_for_reroute — end-to-end via FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────


class RerouteAnalysisApiTests(unittest.TestCase):
    """End-to-end: instantiate a real template, manually create an
    Experiment + EvalResult, hit the endpoint twice (cache miss +
    cache hit), then once more with ``?refresh=true``."""

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

    def _create_eval_result(
        self,
        project_id: int,
        *,
        pass_rate: float,
        description_override: str | None = None,
    ) -> int:
        """Manually insert an Experiment + EvalResult (bypasses the
        full training loop). Returns the new EvalResult.id."""
        from app.database import async_session_factory
        from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode
        from app.models.project import Project

        async def _go() -> int:
            async with async_session_factory() as db:
                if description_override is not None:
                    project = await db.get(Project, project_id)
                    project.description = description_override
                    await db.flush()
                exp = Experiment(
                    project_id=project_id,
                    name="phase7a-fixture",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
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
                    metrics={"f1": pass_rate},
                    pass_rate=pass_rate,
                    details={},
                )
                db.add(ev)
                await db.flush()
                eval_id = ev.id
                await db.commit()
                return eval_id

        return asyncio.run(_go())

    def test_endpoint_404s_on_unknown_eval_result(self):
        resp = self.client.get(
            "/api/projects/1/evaluation/999999999/reroute-analysis"
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_endpoint_400s_on_eval_not_in_project(self):
        # Two projects; eval lives under project A but we query
        # project B's URL.
        proj_a = self._instantiate_template(
            "policy-qa-style", "Phase7a Eval-Project Mismatch A"
        )
        proj_b = self._instantiate_template(
            "ticket-router", "Phase7a Eval-Project Mismatch B"
        )
        eval_id = self._create_eval_result(proj_a["id"], pass_rate=0.30)
        resp = self.client.get(
            f"/api/projects/{proj_b['id']}/evaluation/{eval_id}/reroute-analysis"
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("eval_result_not_in_project", resp.text)

    def test_endpoint_returns_try_rag_for_low_f1_qa_with_retrieval_brief(self):
        project = self._instantiate_template(
            "policy-qa-style", "Phase7a Try-RAG Recommendation"
        )
        # Patch the project description to include retrieval-shaped
        # language. Policy-QA's default brief might or might not
        # match — be explicit so the test is deterministic.
        eval_id = self._create_eval_result(
            project["id"],
            pass_rate=0.35,
            description_override=(
                "An assistant that should answer questions about company "
                "policies. The model needs to look up the relevant policy "
                "and respond accurately."
            ),
        )
        resp = self.client.get(
            f"/api/projects/{project['id']}/evaluation/{eval_id}/reroute-analysis"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["eval_result_id"], eval_id)
        self.assertEqual(body["project_id"], project["id"])
        self.assertEqual(body["pass_rate"], 0.35)
        # The brief signal must fire (we forced retrieval phrasing).
        signals_by_id = {s["id"]: s for s in body["signals"]}
        self.assertTrue(signals_by_id["brief_mentions_retrieval"]["fired"])
        # Recommendation should be try_rag (low F1 + retrieval brief).
        self.assertEqual(body["recommendation"]["kind"], "try_rag")

    def test_endpoint_returns_stay_the_course_on_passing_eval(self):
        project = self._instantiate_template(
            "policy-qa-style", "Phase7a Stay-The-Course Passing"
        )
        eval_id = self._create_eval_result(project["id"], pass_rate=0.90)
        resp = self.client.get(
            f"/api/projects/{project['id']}/evaluation/{eval_id}/reroute-analysis"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["recommendation"]["kind"], "stay_the_course")

    def test_endpoint_caches_on_eval_details_and_refresh_recomputes(self):
        project = self._instantiate_template(
            "policy-qa-style", "Phase7a Cache + Refresh"
        )
        eval_id = self._create_eval_result(
            project["id"],
            pass_rate=0.30,
            description_override="Look up answers from the policy.",
        )

        # First call computes and writes the cache.
        first = self.client.get(
            f"/api/projects/{project['id']}/evaluation/{eval_id}/reroute-analysis"
        )
        self.assertEqual(first.status_code, 200, first.text)
        computed_at_first = first.json()["computed_at"]

        # Verify the analysis landed on EvalResult.details.
        from app.database import async_session_factory
        from app.models.experiment import EvalResult

        async def _read_details() -> dict:
            async with async_session_factory() as db:
                ev = await db.get(EvalResult, eval_id)
                return dict(ev.details or {})

        details = asyncio.run(_read_details())
        self.assertIn("reroute_analysis", details)
        self.assertEqual(details["reroute_analysis"]["eval_result_id"], eval_id)

        # Second call without ?refresh — must return the cached
        # payload byte-identical (same computed_at).
        second = self.client.get(
            f"/api/projects/{project['id']}/evaluation/{eval_id}/reroute-analysis"
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["computed_at"], computed_at_first)

        # Third call with ?refresh=true — must recompute (new
        # computed_at timestamp, even if the analysis content is
        # identical).
        third = self.client.get(
            f"/api/projects/{project['id']}/evaluation/{eval_id}/reroute-analysis"
            "?refresh=true"
        )
        self.assertEqual(third.status_code, 200, third.text)
        self.assertNotEqual(third.json()["computed_at"], computed_at_first)


if __name__ == "__main__":
    unittest.main()
