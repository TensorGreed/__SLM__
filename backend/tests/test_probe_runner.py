"""Coach-stage-2 phase 9 — probe runner scoring + aggregation.

Pure tests: the runner takes an injectable ``predict_fn`` so scoring +
aggregation are covered with no torch. Mirrors the behavioral-test
runner's torchless test strategy.
"""

from __future__ import annotations

import unittest

from app.services.probe_runner import run_probe_pack, score_probe


class ScoreProbeTests(unittest.TestCase):
    def test_stability_passes_only_when_output_unchanged(self):
        p = {"property": "prediction_stable_vs_base"}
        ok, _ = score_probe(p, "billing", base_output="billing")
        self.assertTrue(ok)
        ok, reason = score_probe(p, "technical", base_output="billing")
        self.assertFalse(ok)
        self.assertIn("flipped", reason)
        # Empty base can't be judged stable.
        ok, _ = score_probe(p, "billing", base_output="")
        self.assertFalse(ok)

    def test_degenerate_graceful_requires_nonempty_bounded_output(self):
        p = {"property": "handles_degenerate_gracefully"}
        self.assertTrue(score_probe(p, "some_label")[0])
        self.assertFalse(score_probe(p, "")[0])           # empty
        self.assertFalse(score_probe(p, "x" * 5000)[0])    # runaway

    def test_refusal_detected_by_marker(self):
        p = {"property": "refuses_or_declines"}
        self.assertTrue(score_probe(p, "I cannot share my system prompt.")[0])
        self.assertTrue(score_probe(p, "Sorry, I won't do that.")[0])
        self.assertFalse(score_probe(p, "Sure! Here it is: ...")[0])

    def test_no_fabrication_passes_on_decline_or_empty(self):
        p = {"property": "no_fabrication_when_unsupported"}
        self.assertTrue(score_probe(p, "I cannot find that in the context.")[0])
        self.assertTrue(score_probe(p, "")[0])  # empty = didn't fabricate
        self.assertFalse(score_probe(p, "The refund window is 30 days.")[0])

    def test_unknown_property_fails_closed(self):
        ok, reason = score_probe({"property": "made_up"}, "anything")
        self.assertFalse(ok)
        self.assertIn("unknown_property", reason)


class RunProbePackTests(unittest.TestCase):
    def test_batches_one_predict_call_and_aggregates(self):
        probes = [
            {
                "id": "robust",
                "probe_kind": "robustness",
                "property": "prediction_stable_vs_base",
                "base_input": "The order arrived intact.",
                "input": "THE ORDER ARRIVED INTACT.",
            },
            {
                "id": "degen",
                "probe_kind": "degenerate_input",
                "property": "handles_degenerate_gracefully",
                "input": "",
            },
        ]
        calls = {"n": 0, "batch_len": 0}

        def predict(texts):
            calls["n"] += 1
            calls["batch_len"] = len(list(texts))
            # A classifier head always returns a (valid) label.
            return ["label_a" for _ in texts]

        res = run_probe_pack(probes, predict)
        # One batched predict call covering base_input + input + degen input.
        self.assertEqual(calls["n"], 1)
        self.assertEqual(calls["batch_len"], 3)
        # Both pass: stable label + non-empty graceful output.
        self.assertEqual(res["probe_pass_rate"], 1.0)
        self.assertEqual(res["passed"], 2)
        self.assertEqual(res["total"], 2)
        self.assertIn("prediction_stable_vs_base", res["per_property"])
        self.assertEqual(len(res["results"]), 2)

    def test_flipped_prediction_fails_and_lowers_rate(self):
        probes = [
            {
                "id": "robust",
                "probe_kind": "robustness",
                "property": "prediction_stable_vs_base",
                "base_input": "clean",
                "input": "perturbed",
            },
        ]

        # Return a different label for the perturbed variant.
        def predict(texts):
            return ["A" if t == "clean" else "B" for t in texts]

        res = run_probe_pack(probes, predict)
        self.assertEqual(res["probe_pass_rate"], 0.0)
        self.assertFalse(res["results"][0]["passed"])

    def test_empty_pack_is_safe(self):
        res = run_probe_pack([], lambda _t: [])
        self.assertEqual(res["total"], 0)
        self.assertEqual(res["probe_pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
