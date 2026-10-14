"""Compliance bundles: built-in bundles are data (types + definitions +
regime tag), applying one merges definitions without overwriting the
user's, the type→regimes map is editable, and span evaluations report
micro P / R / F1 per regime."""

from __future__ import annotations

import asyncio
import time
import unittest

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.services import compliance_bundles_service as cb
from app.services.post_training_eval_service import span_block
from app.services.span_type_definitions_service import DEFINITIONS_KEY, _SUGGESTIONS

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


class BundleDataTests(unittest.TestCase):
    def test_every_bundle_is_well_formed(self):
        ids = [b["id"] for b in cb.BUILTIN_BUNDLES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({"pci_dss", "gdpr", "hipaa", "ccpa"} <= set(ids))
        for bundle in cb.list_bundles():
            self.assertTrue(bundle["name"] and bundle["description"] and bundle["regime"])
            names = [t["type"] for t in bundle["types"]]
            self.assertEqual(len(names), len(set(names)), bundle["id"])
            for t in bundle["types"]:
                self.assertRegex(t["type"], r"^[a-z0-9_]+$")
                self.assertTrue(t["definition"], f"{bundle['id']}:{t['type']} has no definition")

    def test_definitions_reuse_the_suggestions(self):
        pci = next(b for b in cb.list_bundles() if b["id"] == "pci_dss")
        self.assertEqual([t["type"] for t in pci["types"]], ["credit_card", "card_expiry", "cvv", "bank_account", "iban"])
        for t in pci["types"]:
            self.assertEqual(t["definition"], _SUGGESTIONS[t["type"]])
        hipaa = next(b for b in cb.BUILTIN_BUNDLES if b["id"] == "hipaa")
        self.assertTrue({"medical_record_number", "health_plan_id", "provider_name", "person_name"} <= set(hipaa["types"]))


class ApplyTests(unittest.TestCase):
    def test_apply_merges_without_overwriting_user_definitions(self):
        cfg = {DEFINITIONS_KEY: {"iban": "my own IBAN rule"}, "other": 1}
        new, result = cb.apply_bundles(cfg, ["pci_dss"])
        self.assertEqual(new["other"], 1)
        self.assertEqual(new[DEFINITIONS_KEY]["iban"], "my own IBAN rule")
        self.assertEqual(new[DEFINITIONS_KEY]["cvv"], _SUGGESTIONS["cvv"])
        self.assertEqual(result["definitions_kept"], ["iban"])
        self.assertNotIn("iban", result["definitions_added"])
        self.assertEqual(result["entity_types"], ["credit_card", "card_expiry", "cvv", "bank_account", "iban"])
        self.assertEqual(new["compliance"]["bundles"], ["pci_dss"])
        self.assertEqual(cfg[DEFINITIONS_KEY], {"iban": "my own IBAN rule"}, "input not mutated")

    def test_overlapping_bundles_tag_shared_types_with_both_regimes(self):
        new, _ = cb.apply_bundles({}, ["gdpr", "hipaa"])
        regimes = new["compliance"]["type_regimes"]
        self.assertEqual(regimes["person_name"], ["gdpr", "hipaa"])
        self.assertEqual(regimes["diagnosis"], ["hipaa"])
        again, result = cb.apply_bundles(new, ["gdpr"])
        self.assertEqual(again["compliance"]["type_regimes"]["person_name"], ["gdpr", "hipaa"], "idempotent")
        self.assertEqual(result["bundles"], ["gdpr", "hipaa"])

    def test_remove_takes_off_the_tag_but_keeps_definitions(self):
        new, _ = cb.apply_bundles({}, ["gdpr", "hipaa"])
        out, result = cb.apply_bundles(new, [], ["hipaa"])
        regimes = out["compliance"]["type_regimes"]
        self.assertEqual(regimes["person_name"], ["gdpr"])
        self.assertNotIn("diagnosis", regimes)
        self.assertIn("diagnosis", out[DEFINITIONS_KEY])
        self.assertEqual(result["bundles"], ["gdpr"])

    def test_unknown_bundle_raises(self):
        with self.assertRaises(ValueError):
            cb.apply_bundles({}, ["sox"])

    def test_custom_tags(self):
        new, _ = cb.apply_bundles({}, ["pci_dss"])
        out, result = cb.set_type_regimes(new, {"credit_card": ["pci_dss", "Internal Tier 1"], "Employee ID": ["internal tier 1"], "x": []})
        self.assertEqual(result["type_regimes"], {"credit_card": ["pci_dss", "internal_tier_1"], "employee_id": ["internal_tier_1"]})
        self.assertEqual(result["custom"], ["internal_tier_1"])
        self.assertEqual(out["compliance"]["bundles"], ["pci_dss"])


class RegimeScoreTests(unittest.TestCase):
    PER_CLASS = {
        "credit_card": {"tp": 8, "fp": 1, "fn": 2, "f1": 0.84},
        "iban": {"tp": 2, "fp": 0, "fn": 0, "f1": 1.0},
        "email": {"tp": 5, "fp": 0, "fn": 0, "f1": 1.0},
        "person_name": {"tp": 3, "fp": 2, "fn": 5, "f1": 0.46},
    }

    def test_micro_prf_per_regime_weakest_recall_first(self):
        mapping = {"credit_card": ["pci_dss"], "iban": ["pci_dss", "gdpr"], "email": ["gdpr"], "person_name": ["gdpr"], "cvv": ["pci_dss"], "ssn": ["custom_tag"]}
        rows = cb.regime_scores(self.PER_CLASS, mapping)
        by = {r["regime"]: r for r in rows}
        pci = by["pci_dss"]
        self.assertEqual((pci["tp"], pci["fp"], pci["fn"]), (10, 1, 2))
        self.assertEqual(pci["recall"], round(10 / 12, 4))
        self.assertEqual(pci["precision"], round(10 / 11, 4))
        self.assertEqual(pci["unscored_types"], ["cvv"])
        self.assertEqual(pci["label"], "PCI-DSS")
        gdpr = by["gdpr"]
        self.assertEqual((gdpr["tp"], gdpr["fn"]), (10, 5))
        self.assertEqual([r["regime"] for r in rows], ["gdpr", "pci_dss", "custom_tag"])
        self.assertIsNone(by["custom_tag"]["recall"])
        self.assertTrue(by["custom_tag"]["custom"])

    def test_span_block_carries_by_regime_only_with_a_map(self):
        class R:
            id = 7
            dataset_name = "gold_test"
            metrics = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "total": 10, "per_class": RegimeScoreTests.PER_CLASS}

        self.assertEqual(span_block(R())["by_regime"], [])
        block = span_block(R(), {"email": ["gdpr"]})
        self.assertEqual(block["by_regime"][0]["recall"], 1.0)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"compliance-{time.time_ns()}", "description": "x"}).json()["id"]

    def test_get_apply_edit(self):
        base = f"/api/projects/{self.pid}/compliance"
        got = client.get(f"{base}/bundles").json()
        self.assertEqual({b["id"] for b in got["builtin"]} >= {"pci_dss", "gdpr", "hipaa", "ccpa"}, True)
        self.assertFalse(any(b["applied"] for b in got["builtin"]))
        self.assertEqual(got["entity_types"], [])

        client.put(f"/api/projects/{self.pid}/synthetic/type-definitions", json={"definitions": {"cvv": "user rule"}})
        applied = client.post(f"{base}/apply", json={"bundle_ids": ["pci_dss"]})
        self.assertEqual(applied.status_code, 200, applied.text)
        body = applied.json()
        self.assertEqual(body["definitions"]["cvv"], "user rule")
        self.assertIn("cvv", body["definitions_kept"])
        self.assertIn("credit_card", body["entity_types"])

        self.assertEqual(client.post(f"{base}/apply", json={"bundle_ids": ["nope"]}).status_code, 400)

        put = client.put(f"{base}/type-regimes", json={"type_regimes": {**body["type_regimes"], "cvv": ["pci_dss", "vault"]}}).json()
        self.assertEqual(put["custom"], ["vault"])
        self.assertIn({"regime": "vault", "label": "vault", "custom": True}, put["regimes"])

        again = client.get(f"{base}/bundles").json()
        self.assertTrue(next(b for b in again["builtin"] if b["id"] == "pci_dss")["applied"])
        self.assertEqual(again["type_regimes"]["cvv"], ["pci_dss", "vault"])

    def test_span_eval_summary_reports_by_regime(self):
        client.post(f"/api/projects/{self.pid}/compliance/apply", json={"bundle_ids": ["pci_dss"]})

        async def _go():
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=self.pid, name="tc", status=ExperimentStatus.COMPLETED,
                    base_model="microsoft/deberta-v3-base", output_dir="/tmp/none",
                    config={"task_type": "token_classification"},
                )
                db.add(exp)
                await db.flush()
                db.add(EvalResult(
                    experiment_id=exp.id, dataset_name="test", eval_type="f1",
                    metrics={"precision": 0.9, "recall": 0.8, "f1": 0.85, "total": 10, "scoring_mode": "span_set", "per_class": RegimeScoreTests.PER_CLASS},
                    details={"failures_preview": [], "failed_count": 0},
                ))
                await db.commit()
                return exp.id

        exp_id = asyncio.run(_go())
        body = client.get(f"/api/projects/{self.pid}/evaluation/summary", params={"experiment_id": exp_id}).json()
        self.assertEqual(body["kind"], "span_eval")
        regimes = body["test"]["by_regime"]
        self.assertEqual([r["regime"] for r in regimes], ["pci_dss"])
        self.assertEqual(regimes[0]["recall"], round(10 / 12, 4))


if __name__ == "__main__":
    unittest.main()
