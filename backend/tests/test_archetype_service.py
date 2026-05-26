"""Tests for the archetype-extraction service + endpoint
(USER-SUCCESS Epic 8 Phase 8a).

Coverage:

- Pure feature-check truth table (Shannon entropy / class-balance
  ratio / hard-negative ratio / diversity / lengths).
- Per-recipe applicability — features fire only on the recipes
  the map gates them on.
- Template seed loader: each recipe gets the expected number of
  seeded contributions, and ``code-review`` (no shipped template)
  returns an empty list.
- Cohort threshold: a project with ``pass_rate < 0.6`` is
  excluded; ``pass_rate >= 0.6`` included.
- Module-level cache: same recipe within TTL returns identical
  ``computed_at``; ``clear_archetype_cache`` re-triggers compute.
- Endpoint: 200 happy path with template-only cohort, 400 unknown
  recipe, 404 empty cohort (no template seed).
- ``?refresh=true`` bypasses the cache.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from datetime import datetime, timezone

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.archetype_service import (  # noqa: E402
    _diversity_score,
    _hard_negative_ratio,
    _min_over_max_class_ratio,
    _per_row_input_lengths,
    _per_row_output_lengths,
    _shannon_entropy,
    clear_archetype_cache,
    extract_features_from_rows,
)
from app.services.archetype_seeds import (  # noqa: E402
    clear_seed_cache,
    load_seed_contributions,
)


_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__exit__(None, None, None)


# ─────────────────────────────────────────────────────────────────────
# Pure feature checks
# ─────────────────────────────────────────────────────────────────────


class ShannonEntropyTests(unittest.TestCase):
    def test_balanced_two_class_yields_one_bit(self):
        self.assertAlmostEqual(_shannon_entropy(["a", "b", "a", "b"]), 1.0)

    def test_single_class_yields_zero(self):
        self.assertEqual(_shannon_entropy(["a", "a", "a"]), 0.0)

    def test_four_class_uniform_yields_two_bits(self):
        self.assertAlmostEqual(
            _shannon_entropy(["a", "b", "c", "d"]), 2.0,
        )

    def test_empty_yields_zero(self):
        self.assertEqual(_shannon_entropy([]), 0.0)


class ClassBalanceRatioTests(unittest.TestCase):
    def test_perfectly_balanced_yields_one(self):
        self.assertEqual(_min_over_max_class_ratio(["a", "b", "a", "b"]), 1.0)

    def test_skewed_yields_small_ratio(self):
        # 1 vs 9 → 1/9 ≈ 0.111
        labels = ["a"] + ["b"] * 9
        self.assertAlmostEqual(_min_over_max_class_ratio(labels), 1 / 9)

    def test_single_class_yields_none(self):
        # Only one class → ratio undefined.
        self.assertIsNone(_min_over_max_class_ratio(["a", "a", "a"]))


class HardNegativeRatioTests(unittest.TestCase):
    def test_counts_only_hard_negative_synth_sources(self):
        rows = [
            {"synth_source": "playbook:classification:hard_negatives:vs=billing"},
            {"synth_source": "playbook:classification:positives_paraphrase"},
            {"synth_source": "playbook:classification:hard_negatives:vs=technical"},
        ]
        self.assertAlmostEqual(_hard_negative_ratio(rows), 2 / 3)

    def test_returns_none_when_no_synth_rows(self):
        # Project with only gold rows → ratio undefined.
        rows = [{"input": "x", "expected": {"label": "a"}}]
        self.assertIsNone(_hard_negative_ratio(rows))

    def test_returns_zero_when_synth_present_but_no_hard_negatives(self):
        rows = [
            {"synth_source": "playbook:qa-sft:positives_paraphrase"},
            {"synth_source": "playbook:qa-sft:edge_cases"},
        ]
        self.assertEqual(_hard_negative_ratio(rows), 0.0)


class DiversityScoreTests(unittest.TestCase):
    def test_identical_rows_yield_low_diversity(self):
        rows = [{"input": "the cat sat on the mat"}] * 5
        score = _diversity_score(rows)
        self.assertIsNotNone(score)
        self.assertLess(score, 0.05)  # near-zero diversity

    def test_distinct_rows_yield_high_diversity(self):
        rows = [
            {"input": "alpha beta gamma"},
            {"input": "delta epsilon zeta"},
            {"input": "eta theta iota"},
            {"input": "kappa lambda mu"},
        ]
        score = _diversity_score(rows)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.9)

    def test_too_few_rows_yields_none(self):
        self.assertIsNone(_diversity_score([{"input": "lone"}]))


class LengthExtractionTests(unittest.TestCase):
    def test_per_row_input_lengths_walks_nested_dicts(self):
        rows = [
            {"input": {"ticket": "abc"}},
            {"input": {"text": "hello world"}},
            {"question": "Q?"},
        ]
        self.assertEqual(_per_row_input_lengths(rows), [3, 11, 2])

    def test_per_row_output_lengths_prefers_nested_expected(self):
        rows = [
            {"expected": {"answer": "yes"}},
            {"expected": {"summary": "long summary text"}},
            {"expected": "raw string"},
            {"answer": "fallback path"},
        ]
        self.assertEqual(_per_row_output_lengths(rows), [3, 17, 10, 13])

    def test_skips_rows_with_no_extractable_text(self):
        self.assertEqual(_per_row_input_lengths([{"meta": {"foo": "bar"}}]), [])


# ─────────────────────────────────────────────────────────────────────
# Per-recipe applicability
# ─────────────────────────────────────────────────────────────────────


class FeatureApplicabilityTests(unittest.TestCase):
    def test_classification_recipe_emits_class_features(self):
        rows = [{"expected": {"label": l}} for l in ("a", "b", "a", "b")]
        features = extract_features_from_rows(rows, recipe_id="classification")
        self.assertIn("class_entropy", features)
        self.assertIn("class_balance_ratio", features)
        self.assertIn("row_count", features)
        # class_entropy should be ~1.0 (balanced binary).
        self.assertAlmostEqual(features["class_entropy"], 1.0)

    def test_qa_sft_recipe_omits_class_features(self):
        rows = [
            {"input": {"question": "Q"}, "expected": {"answer": "A"}}
            for _ in range(5)
        ]
        features = extract_features_from_rows(rows, recipe_id="qa-sft")
        self.assertNotIn("class_entropy", features)
        self.assertNotIn("class_balance_ratio", features)
        self.assertIn("output_length_chars", features)

    def test_summarization_recipe_skips_hard_negative(self):
        # hard_negative isn't in the applicability set for
        # summarization — that recipe doesn't have a hard-negative
        # playbook.
        rows = [{"input": "src", "expected": {"summary": "summary"}}]
        features = extract_features_from_rows(rows, recipe_id="summarization")
        self.assertNotIn("hard_negative_ratio", features)


# ─────────────────────────────────────────────────────────────────────
# Template seed loader
# ─────────────────────────────────────────────────────────────────────


class SeedLoaderTests(unittest.TestCase):
    def setUp(self):
        clear_seed_cache()

    def test_classification_gets_two_seeds(self):
        seeds = load_seed_contributions("classification")
        self.assertEqual(len(seeds), 2)
        slugs = {s["template_slug"] for s in seeds}
        self.assertEqual(slugs, {"ticket-router", "log-triage"})
        for seed in seeds:
            self.assertIn("class_entropy", seed["features"])
            self.assertIsNotNone(seed["features"]["row_count"])

    def test_generic_sft_gets_three_seeds(self):
        seeds = load_seed_contributions("generic-sft")
        self.assertEqual(len(seeds), 3)

    def test_qa_sft_gets_one_seed(self):
        seeds = load_seed_contributions("qa-sft")
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["template_slug"], "policy-qa-style")

    def test_code_review_gets_zero_seeds(self):
        # No shipped template for code-review.
        self.assertEqual(load_seed_contributions("code-review"), [])

    def test_pseudo_ids_are_negative_to_avoid_collision(self):
        for recipe in ("classification", "qa-sft", "generic-sft"):
            for seed in load_seed_contributions(recipe):
                self.assertLess(
                    seed["pseudo_id"], 0,
                    f"seed for {seed['template_slug']} has positive pseudo_id",
                )

    def test_memoised_across_calls(self):
        first = load_seed_contributions("classification")
        second = load_seed_contributions("classification")
        # Same list object — memoisation working.
        self.assertIs(first, second)


# ─────────────────────────────────────────────────────────────────────
# End-to-end via TestClient + DB
# ─────────────────────────────────────────────────────────────────────


def _seed_passing_project(
    name: str,
    recipe_id: str,
    *,
    pass_rate: float,
) -> int:
    """Insert a Project + completed Experiment + passing EvalResult.
    Returns the project id."""
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
            project = Project(
                name=name,
                description="archetype test project",
                base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                selected_recipe={"recipe_id": recipe_id},
            )
            db.add(project)
            await db.flush()
            pid = project.id
            exp = Experiment(
                project_id=pid,
                name="archetype-fixture",
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                config={},
                training_mode=TrainingMode.SFT,
                status=ExperimentStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc),
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
            await db.commit()
            return pid

    return asyncio.run(_go())


class CohortThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls.client = _MODULE_CLIENT_CM

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def setUp(self):
        clear_archetype_cache()

    def test_below_threshold_excluded(self):
        from app.database import async_session_factory
        from app.services.archetype_service import _find_passing_projects

        # Seed a borderline-fail project — its pass_rate is just
        # below 0.6.
        _seed_passing_project(
            "Archetype-cohort-fail", "classification", pass_rate=0.55,
        )

        async def _go() -> list:
            async with async_session_factory() as db:
                return await _find_passing_projects(db, "classification")

        passing = asyncio.run(_go())
        names = [p.name for p, _ in passing]
        self.assertNotIn("Archetype-cohort-fail", names)

    def test_at_or_above_threshold_included(self):
        from app.database import async_session_factory
        from app.services.archetype_service import _find_passing_projects

        _seed_passing_project(
            "Archetype-cohort-pass", "classification", pass_rate=0.65,
        )

        async def _go() -> list:
            async with async_session_factory() as db:
                return await _find_passing_projects(db, "classification")

        passing = asyncio.run(_go())
        names = [p.name for p, _ in passing]
        self.assertIn("Archetype-cohort-pass", names)


class CacheBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls.client = _MODULE_CLIENT_CM

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def setUp(self):
        clear_archetype_cache()

    def test_second_call_within_ttl_returns_same_computed_at(self):
        first = self.client.get("/api/archetypes/classification")
        self.assertEqual(first.status_code, 200, first.text)
        second = self.client.get("/api/archetypes/classification")
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(
            first.json()["computed_at"],
            second.json()["computed_at"],
            "Cache miss: timestamps differ within TTL window",
        )

    def test_refresh_true_recomputes(self):
        first = self.client.get("/api/archetypes/classification")
        self.assertEqual(first.status_code, 200, first.text)
        # Tiny sleep to ensure ISO-second ticks (timestamps include
        # microseconds, so even a sub-millisecond diff suffices).
        import time as _time
        _time.sleep(0.01)
        third = self.client.get("/api/archetypes/classification?refresh=true")
        self.assertEqual(third.status_code, 200, third.text)
        self.assertNotEqual(
            first.json()["computed_at"],
            third.json()["computed_at"],
            "refresh=true should have re-stamped computed_at",
        )


class EndpointStatusCodesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls.client = _MODULE_CLIENT_CM

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def setUp(self):
        clear_archetype_cache()

    def test_unknown_recipe_returns_400(self):
        resp = self.client.get("/api/archetypes/not-a-real-recipe")
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("unknown_recipe_id", resp.text)

    def test_classification_returns_archetype_from_templates(self):
        # No user projects passing — should fall back to the 2
        # classification template seeds.
        resp = self.client.get("/api/archetypes/classification")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["recipe_id"], "classification")
        # 2 templates seed classification.
        self.assertGreaterEqual(body["n_template_seeds"], 2)
        self.assertGreaterEqual(body["n_passing_projects"], 2)
        # All applicable features present.
        feature_ids = {f["feature_id"] for f in body["features"]}
        self.assertIn("class_entropy", feature_ids)
        self.assertIn("row_count", feature_ids)
        # Cohort provenance carries each template.
        sources = {c["source"] for c in body["cohort_provenance"]}
        self.assertIn("template", sources)

    def test_code_review_with_no_user_projects_returns_404(self):
        # code-review has no template seed AND no user project
        # passing in this test session — empty cohort → 404.
        resp = self.client.get("/api/archetypes/code-review")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertIn("empty_cohort", resp.text)


# ─────────────────────────────────────────────────────────────────────
# Phase 8b — per-project comparison
# ─────────────────────────────────────────────────────────────────────


from app.services.archetype_service import (  # noqa: E402
    _classify_feature_status,
    _minority_label_for,
    _suggestion_for,
    _summarise,
)


class ClassifyFeatureStatusTests(unittest.TestCase):
    def test_below_p25(self):
        self.assertEqual(
            _classify_feature_status(your_value=10, p25=20, p75=80),
            "below",
        )

    def test_above_p75(self):
        self.assertEqual(
            _classify_feature_status(your_value=100, p25=20, p75=80),
            "above",
        )

    def test_in_band(self):
        self.assertEqual(
            _classify_feature_status(your_value=50, p25=20, p75=80),
            "ok",
        )

    def test_missing_when_value_none(self):
        self.assertEqual(
            _classify_feature_status(your_value=None, p25=20, p75=80),
            "missing",
        )

    def test_missing_when_archetype_band_none(self):
        # Single-contribution cohort — p25/p75 might be None.
        self.assertEqual(
            _classify_feature_status(your_value=42, p25=None, p75=None),
            "missing",
        )


class SuggestionForTests(unittest.TestCase):
    def test_row_count_below_emits_paraphrase_action(self):
        suggestion, action = _suggestion_for(
            feature_id="row_count",
            status="below",
            your_value=50,
            p50=200,
            minority_label=None,
        )
        self.assertIsNotNone(suggestion)
        self.assertIsNotNone(action)
        self.assertEqual(action["kind"], "run_playbook")
        self.assertEqual(action["params"]["mode"], "positives_paraphrase")
        self.assertEqual(action["params"]["target_count"], 150)

    def test_row_count_paraphrase_capped_at_200(self):
        # Cohort median way above current → cap delta at 200.
        _suggestion, action = _suggestion_for(
            feature_id="row_count",
            status="below",
            your_value=10,
            p50=2000,
            minority_label=None,
        )
        self.assertEqual(action["params"]["target_count"], 200)

    def test_class_entropy_below_with_minority_emits_balance_fill(self):
        suggestion, action = _suggestion_for(
            feature_id="class_entropy",
            status="below",
            your_value=0.3,
            p50=1.5,
            minority_label="billing",
        )
        self.assertIn("billing", suggestion)
        self.assertEqual(action["kind"], "run_playbook")
        self.assertEqual(action["params"]["mode"], "class_balance_fill")
        self.assertEqual(action["params"]["target_class"], "billing")

    def test_class_entropy_below_without_minority_emits_text_only(self):
        # No minority label known → suggestion exists, action is None.
        suggestion, action = _suggestion_for(
            feature_id="class_entropy",
            status="below",
            your_value=0.3,
            p50=1.5,
            minority_label=None,
        )
        self.assertIsNotNone(suggestion)
        self.assertIsNone(action)

    def test_hard_negative_below_emits_hard_negatives_playbook(self):
        _s, action = _suggestion_for(
            feature_id="hard_negative_ratio",
            status="below",
            your_value=0.02,
            p50=0.1,
            minority_label=None,
        )
        self.assertEqual(action["kind"], "run_playbook")
        self.assertEqual(action["params"]["mode"], "hard_negatives")

    def test_diversity_below_emits_navigate(self):
        _s, action = _suggestion_for(
            feature_id="goldset_diversity",
            status="below",
            your_value=0.1,
            p50=0.8,
            minority_label=None,
        )
        self.assertEqual(action["kind"], "navigate")
        self.assertEqual(action["params"]["target"], "data-studio-diversity")

    def test_length_features_emit_diagnostic_no_action(self):
        for fid in ("input_length_chars", "output_length_chars"):
            with self.subTest(fid=fid):
                suggestion, action = _suggestion_for(
                    feature_id=fid,
                    status="below",
                    your_value=10,
                    p50=200,
                    minority_label=None,
                )
                self.assertIsNotNone(suggestion)
                self.assertIsNone(action)

    def test_ok_or_missing_emits_nothing(self):
        for status in ("ok", "missing"):
            with self.subTest(status=status):
                self.assertEqual(
                    _suggestion_for(
                        feature_id="row_count",
                        status=status,
                        your_value=200,
                        p50=200,
                        minority_label=None,
                    ),
                    (None, None),
                )


class MinorityLabelTests(unittest.TestCase):
    def test_returns_smallest_class(self):
        rows = (
            [{"expected": {"label": "billing"}}] * 1
            + [{"expected": {"label": "tech"}}] * 10
            + [{"expected": {"label": "shipping"}}] * 5
        )
        self.assertEqual(_minority_label_for(rows), "billing")

    def test_returns_none_for_single_class(self):
        rows = [{"expected": {"label": "tech"}}] * 5
        self.assertIsNone(_minority_label_for(rows))

    def test_returns_none_for_no_labels(self):
        rows = [{"input": "x", "expected": {"answer": "y"}}] * 5
        self.assertIsNone(_minority_label_for(rows))


class SummariseTests(unittest.TestCase):
    def _feature(self, status: str) -> dict:
        return {
            "feature_id": "x",
            "label": "x",
            "unit": "rows",
            "your_value": None,
            "archetype_p25": None,
            "archetype_p50": None,
            "archetype_p75": None,
            "status": status,
            "suggestion": None,
            "suggested_action": None,
        }

    def test_all_ok_is_healthy(self):
        self.assertEqual(
            _summarise([self._feature("ok"), self._feature("ok")]),
            "healthy",
        )

    def test_below_majority_is_below_cohort(self):
        self.assertEqual(
            _summarise([self._feature("below"), self._feature("below"), self._feature("ok")]),
            "below_cohort",
        )

    def test_above_majority_is_above_cohort(self):
        self.assertEqual(
            _summarise([self._feature("above"), self._feature("above"), self._feature("ok")]),
            "above_cohort",
        )

    def test_both_directions_is_mixed(self):
        self.assertEqual(
            _summarise([self._feature("below"), self._feature("above")]),
            "mixed",
        )

    def test_only_missing_is_healthy(self):
        # Nothing measurable → don't lecture the user.
        self.assertEqual(
            _summarise([self._feature("missing"), self._feature("missing")]),
            "healthy",
        )


class ComparisonEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls.client = _MODULE_CLIENT_CM

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _instantiate_template(self, slug: str, name: str) -> int:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def setUp(self):
        clear_archetype_cache()

    def test_404_when_project_not_found(self):
        resp = self.client.get(
            "/api/projects/9999999/archetype-comparison",
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertIn("project_not_found", resp.text)

    def test_200_with_template_archetype_for_classification_project(self):
        pid = self._instantiate_template(
            "ticket-router", "Archetype Cmp endpoint happy",
        )
        resp = self.client.get(
            f"/api/projects/{pid}/archetype-comparison",
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], pid)
        self.assertEqual(body["recipe_id"], "classification")
        # Archetype was loaded — template seeds present.
        self.assertGreaterEqual(body["archetype"]["n_passing_projects"], 1)
        # Per-feature statuses are present for the applicable
        # classification features.
        feature_ids = {f["feature_id"] for f in body["features"]}
        self.assertIn("class_entropy", feature_ids)
        self.assertIn("row_count", feature_ids)
        # Summary is one of the four known values.
        self.assertIn(
            body["summary"],
            {"healthy", "below_cohort", "above_cohort", "mixed"},
        )


if __name__ == "__main__":
    unittest.main()
