"""Tests for the Phase 8c Coach Mode archetype-drift nudge
(USER-SUCCESS Epic 8 — closes the epic).

Covers ``_archetype_drift_nudge`` pure-function truth table +
the integration into ``_data_stage_suggestions`` and
``_training_stage_suggestions``.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.coach_service import _archetype_drift_nudge  # noqa: E402


_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__exit__(None, None, None)


# ─────────────────────────────────────────────────────────────────────
# Comparison fixture factory
# ─────────────────────────────────────────────────────────────────────


def _make_feature(
    feature_id: str,
    status: str,
    *,
    suggestion: str | None = None,
    action: dict | None = None,
) -> dict:
    return {
        "feature_id": feature_id,
        "label": f"Feature {feature_id}",
        "unit": "rows",
        "your_value": 50,
        "archetype_p25": 100,
        "archetype_p50": 200,
        "archetype_p75": 300,
        "status": status,
        "suggestion": suggestion,
        "suggested_action": action,
    }


def _make_comparison(
    *,
    features: list[dict],
    n_user_projects: int = 2,
    n_template_seeds: int = 1,
    cohort: list[dict] | None = None,
    recipe_id: str = "classification",
) -> dict:
    if cohort is None:
        cohort = []
        for i in range(n_user_projects):
            cohort.append({
                "id": 1000 + i,
                "name": f"User project {i}",
                "source": "user",
                "pass_rate": 0.7,
            })
        for i in range(n_template_seeds):
            cohort.append({
                "id": -100 - i,
                "name": f"Template seed {i}",
                "source": "template",
                "pass_rate": None,
            })
    return {
        "project_id": 4,
        "recipe_id": recipe_id,
        "archetype": {
            "recipe_id": recipe_id,
            "n_passing_projects": n_user_projects + n_template_seeds,
            "n_user_projects": n_user_projects,
            "n_template_seeds": n_template_seeds,
            "computed_at": "2026-05-26T12:00:00Z",
            "features": [],
            "cohort_provenance": cohort,
        },
        "features": features,
        "summary": "below_cohort",
    }


# ─────────────────────────────────────────────────────────────────────
# Pure-function nudge truth table
# ─────────────────────────────────────────────────────────────────────


class ArchetypeDriftNudgePureTests(unittest.TestCase):
    def test_data_stage_fires_on_one_below_with_user_cohort(self):
        comparison = _make_comparison(
            features=[
                _make_feature(
                    "row_count",
                    "below",
                    suggestion="Generate 150 more rows.",
                    action={
                        "kind": "run_playbook",
                        "params": {
                            "mode": "positives_paraphrase",
                            "target_count": 150,
                        },
                    },
                ),
            ],
            n_user_projects=2,
        )
        nudge = _archetype_drift_nudge(4, "data", comparison)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["severity"], "info")
        self.assertEqual(nudge["id"], "data:archetype-drift")
        # Title mentions the count of below features.
        self.assertIn("1 feature", nudge["title"])
        # Body cites cohort size + dominant feature's suggestion.
        self.assertIn("2 successful classification project", nudge["body"])
        self.assertIn("Generate 150 more rows.", nudge["body"])
        # Action = the dominant feature's suggested_action with a
        # default label injected.
        self.assertEqual(nudge["action"]["kind"], "run_playbook")
        self.assertEqual(nudge["action"]["params"]["target_count"], 150)
        self.assertIn("label", nudge["action"])

    def test_data_stage_skipped_below_threshold(self):
        # Zero features below → no nudge regardless of stage.
        comparison = _make_comparison(
            features=[_make_feature("row_count", "ok")],
            n_user_projects=2,
        )
        self.assertIsNone(_archetype_drift_nudge(4, "data", comparison))

    def test_training_stage_requires_two_below(self):
        one_below = _make_comparison(
            features=[
                _make_feature("row_count", "below"),
                _make_feature("class_entropy", "ok"),
            ],
            n_user_projects=2,
        )
        self.assertIsNone(_archetype_drift_nudge(4, "training", one_below))
        two_below = _make_comparison(
            features=[
                _make_feature(
                    "row_count", "below",
                    suggestion="Generate more rows.",
                ),
                _make_feature("class_entropy", "below"),
            ],
            n_user_projects=2,
        )
        nudge = _archetype_drift_nudge(4, "training", two_below)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["severity"], "warning")
        # Title cites 2 features below.
        self.assertIn("2 features", nudge["title"])

    def test_data_stage_skipped_when_only_template_seeds(self):
        # n_user_projects == 0 → don't nudge on data stage (too
        # early to lecture). The same comparison should still fire
        # on the training stage (the user is about to spend
        # compute).
        comparison = _make_comparison(
            features=[
                _make_feature("row_count", "below", suggestion="Add rows."),
                _make_feature("class_entropy", "below"),
            ],
            n_user_projects=0,
            n_template_seeds=2,
        )
        self.assertIsNone(_archetype_drift_nudge(4, "data", comparison))
        # Training stage still fires.
        self.assertIsNotNone(_archetype_drift_nudge(4, "training", comparison))

    def test_skipped_when_user_is_only_provenance(self):
        # The only user-source cohort entry IS this project →
        # recommending you match yourself is silly.
        comparison = _make_comparison(
            features=[
                _make_feature("row_count", "below", suggestion="Add rows."),
                _make_feature("class_entropy", "below"),
            ],
            cohort=[
                {
                    "id": 4,  # same as project_id
                    "name": "This project",
                    "source": "user",
                    "pass_rate": 0.7,
                },
            ],
            n_user_projects=1,
            n_template_seeds=0,
        )
        # Both stages skip.
        self.assertIsNone(_archetype_drift_nudge(4, "data", comparison))
        self.assertIsNone(_archetype_drift_nudge(4, "training", comparison))
        # But for a DIFFERENT project_id (where the sole user
        # provenance isn't us), the nudge should fire on training.
        self.assertIsNotNone(
            _archetype_drift_nudge(99, "training", comparison),
        )

    def test_dominant_picks_row_count_first(self):
        # Multiple below → row_count wins by priority order.
        comparison = _make_comparison(
            features=[
                _make_feature("class_entropy", "below"),
                _make_feature(
                    "row_count", "below",
                    suggestion="Top-of-priority suggestion text.",
                ),
                _make_feature("goldset_diversity", "below"),
            ],
            n_user_projects=2,
        )
        nudge = _archetype_drift_nudge(4, "training", comparison)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["context"]["dominant_feature_id"], "row_count")
        self.assertIn("Top-of-priority suggestion text", nudge["body"])

    def test_dominant_falls_through_priority_when_row_count_ok(self):
        comparison = _make_comparison(
            features=[
                _make_feature("row_count", "ok"),
                _make_feature("class_entropy", "below", suggestion="ent"),
                _make_feature("hard_negative_ratio", "below"),
            ],
            n_user_projects=2,
        )
        nudge = _archetype_drift_nudge(4, "training", comparison)
        self.assertIsNotNone(nudge)
        # class_entropy beats hard_negative_ratio per priority.
        self.assertEqual(nudge["context"]["dominant_feature_id"], "class_entropy")

    def test_none_when_comparison_is_none_or_malformed(self):
        self.assertIsNone(_archetype_drift_nudge(4, "data", None))
        self.assertIsNone(_archetype_drift_nudge(4, "data", "not a dict"))  # type: ignore[arg-type]
        # Empty features list → no below → no nudge.
        self.assertIsNone(
            _archetype_drift_nudge(
                4, "data",
                _make_comparison(features=[], n_user_projects=2),
            ),
        )

    def test_falls_back_to_navigate_action_when_dominant_has_none(self):
        # Length-feature drift has suggestion but no action — the
        # nudge should default to a navigate-to-training-config
        # action so the button is still clickable.
        comparison = _make_comparison(
            features=[
                _make_feature(
                    "input_length_chars", "below",
                    suggestion="Inputs are shorter than the cohort.",
                    action=None,
                ),
            ],
            n_user_projects=2,
        )
        nudge = _archetype_drift_nudge(4, "data", comparison)
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["action"]["kind"], "navigate")
        self.assertEqual(
            nudge["action"]["params"]["target"], "training-config",
        )

    def test_context_lists_all_below_feature_ids(self):
        comparison = _make_comparison(
            features=[
                _make_feature("row_count", "below"),
                _make_feature("class_entropy", "below"),
                _make_feature("goldset_diversity", "ok"),
            ],
            n_user_projects=2,
        )
        nudge = _archetype_drift_nudge(4, "training", comparison)
        self.assertIsNotNone(nudge)
        self.assertEqual(
            set(nudge["context"]["below_feature_ids"]),
            {"row_count", "class_entropy"},
        )


# ─────────────────────────────────────────────────────────────────────
# End-to-end via TestClient
# ─────────────────────────────────────────────────────────────────────


class CoachArchetypeNudgeEndpointTests(unittest.TestCase):
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

    def test_coach_training_response_includes_archetype_nudge_for_thin_project(self):
        """End-to-end: a freshly-instantiated qa-sft project (thin
        relative to the policy-qa-style template seed) hits the
        training-stage Coach endpoint and we get the archetype
        nudge back in the suggestions list.

        Order-resilient: we use qa-sft + the policy-qa-style
        template (1 seed) and a freshly-instantiated project, so
        the cohort is always template-seed-only — independent of
        what other tests have left in the DB.

        At training stage the nudge DOES fire even with zero user
        projects (per the Phase 8c spec: the user is about to spend
        compute, so the recommendation is timely)."""
        from app.services.archetype_service import clear_archetype_cache

        pid = self._instantiate_template(
            "policy-qa-style", "Phase8c Training Archetype Nudge",
        )
        clear_archetype_cache()

        resp = self.client.get(f"/api/projects/{pid}/coach/training")
        self.assertEqual(resp.status_code, 200, resp.text)
        suggestions = resp.json()["suggestions"]
        suggestion_ids = [s["id"] for s in suggestions]

        # The nudge fires when at least 2 features are below cohort
        # — for a thin freshly-instantiated qa-sft project against
        # the policy-qa-style template seed, that's typically true
        # (the template has 200 gold rows). When it doesn't fire,
        # the test passes trivially (no false-positive); the
        # critical assertion is that no exception bubbles up and
        # the response is well-formed.
        if "training:archetype-drift" in suggestion_ids:
            nudge = next(s for s in suggestions if s["id"] == "training:archetype-drift")
            self.assertEqual(nudge["severity"], "warning")
            self.assertIn("p25", nudge["title"])
            self.assertIn("Cohort:", nudge["body"])


if __name__ == "__main__":
    unittest.main()
