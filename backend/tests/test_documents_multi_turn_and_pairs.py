"""`documents_multi_turn` playbook (generic-sft) + preference pairs from an eval result.

Neutral toy data. The teacher backend is a local stub registered in BOTH backend registries; the
eval result is seeded directly. pytest-only (Jobs are awaited in tearDown).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services import preference_pairs_from_eval_service as pp
from app.services.auto_rag_service import document_preamble
from app.services.synth_playbooks import SynthMode, get_playbook
from app.services.synth_playbooks.generic_sft_documents_multi_turn import check_params, pick_groups, resolve_params

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


TEXT_A = "The calibration routine needs a torque wrench and takes twenty minutes. Operators record the reading in the shared logbook."
TEXT_B = "Spare filters are stored in the east cabinet and replaced every quarter by the maintenance crew."
CHUNKS = [
    {"text": TEXT_A, "source_document_id": 1, "chunk_id": 0, "source_doc": "manual.txt"},
    {"text": TEXT_B, "source_document_id": 1, "chunk_id": 1, "source_doc": "manual.txt"},
]


def dialogue(group=1, *, a1=TEXT_A + " [1]", a2=TEXT_B + " [2]", u2="What about the spare filters?", u1="How is the routine done?"):
    return {"group": group, "messages": [
        {"role": "user", "content": u1}, {"role": "assistant", "content": a1},
        {"role": "user", "content": u2}, {"role": "assistant", "content": a2},
    ]}


def make_ctx(**params):
    return {"recipe_id": "generic-sft", "project_id": 1, "gold_rows": [], "raw_rows": None, "failure_cluster": None,
            "target_class": None, "target_count": 1, "params": {"passages_per_dialogue": 2, **params}, "rejects": [],
            "document_chunks": list(CHUNKS), "seed_nonce": 1}


def reasons(ctx):
    return [r["reason"] for r in ctx["rejects"]]


class PlaybookValidatorTests(unittest.TestCase):
    def setUp(self):
        self.pb = get_playbook("generic-sft", SynthMode.DOCUMENTS_MULTI_TURN)
        self.ctx = make_ctx()

    def validate(self, *rows, ctx=None):
        return self.pb.validate(list(rows), ctx or self.ctx)

    def test_registered_for_the_chat_recipe(self):
        self.assertIsNotNone(self.pb)
        self.assertEqual(SynthMode("documents_multi_turn"), SynthMode.DOCUMENTS_MULTI_TURN)

    def test_good_dialogue_keeps_messages_and_passage_ids_only(self):
        out = self.validate(dialogue())
        self.assertEqual(reasons(self.ctx), [])
        payload = out[0]["payload"]
        self.assertEqual(set(payload), {"messages", "source_passages"})
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual([p["chunk_id"] for p in payload["source_passages"]], [0, 1])
        self.assertEqual(out[0]["synth_source"], "playbook:generic-sft:documents_multi_turn")

    def test_passages_ride_as_the_document_preamble_system_message_in_citation_order(self):
        out = self.validate(dialogue())
        system = out[0]["payload"]["messages"][0]["content"]
        expected = document_preamble([{"row_id": n, "score": 0.0, "payload": pg} for n, pg in enumerate(CHUNKS)])
        self.assertEqual(system, expected)
        # [1] / [2] in the preamble are the passages the assistant's [n] citations point at
        self.assertLess(system.index("[1] (manual.txt"), system.index("[2] (manual.txt"))
        self.assertLess(system.index(TEXT_A[:30]), system.index(TEXT_B[:30]))

    def test_a_dialogue_asking_a_test_question_is_rejected_exact_or_near_duplicate(self):
        cases = {
            "exact": [{"question": "How is the routine done?"}],
            # same words, reordered: token-set Jaccard 1.0 (>= the leakage matcher's 0.9)
            "near": [{"question": "Is the routine how done"}],
            "chat row": [{"messages": [{"role": "user", "content": "How is the routine done?"},
                                       {"role": "assistant", "content": "x"}]}],
        }
        for name, test_rows in cases.items():
            with self.subTest(name):
                ctx = make_ctx()
                ctx["test_rows"] = test_rows
                self.assertEqual(self.validate(dialogue(), ctx=ctx), [])
                self.assertEqual(reasons(ctx), ["test_question_leak"])
        ctx = make_ctx()
        ctx["test_rows"] = [{"question": "Which cabinet holds the gloves?"}]
        self.assertEqual(len(self.validate(dialogue(), ctx=ctx)), 1)

    def test_an_answer_without_content_words_counts_as_supported(self):
        ctx = make_ctx()
        out = self.validate(dialogue(a2="[2]"), ctx=ctx)
        self.assertEqual(reasons(ctx), [])
        self.assertEqual(len(out), 1)

    def test_pick_groups_scales_linearly_and_stays_deterministic(self):
        many = [{"text": f"passage number {i}", "source_document_id": i // 50, "chunk_id": i} for i in range(4000)]
        ctx = make_ctx()
        ctx["document_chunks"] = many
        start = time.monotonic()
        first = pick_groups(ctx, 8, 2)
        self.assertLess(time.monotonic() - start, 2.0)
        self.assertEqual(first, pick_groups(ctx, 8, 2))
        self.assertTrue(all(len(g) == 2 for g in first))

    def test_every_reject_reason(self):
        cases = {
            "unknown_group": [dialogue(group=9)],
            "bad_shape": [{"group": 1, "messages": "x"}],
            "bad_message": [{"group": 1, "messages": [{"role": "system", "content": "x"}]}],
            "bad_roles": [{"group": 1, "messages": [{"role": "assistant", "content": "a [1]"}, {"role": "user", "content": "q"}]}],
            "turn_count": [{"group": 1, "messages": dialogue()["messages"][:2]}],
            "no_citation": [dialogue(a2=TEXT_B)],
            "bad_citation": [dialogue(a2=TEXT_B + " [7]")],
            "unsupported": [dialogue(a2="Quarterly zeppelin inspections guarantee fluorescent harmony [2]")],
            "repeated_user_turn": [dialogue(u2="how is the routine done")],
        }
        for reason, rows in cases.items():
            with self.subTest(reason):
                ctx = make_ctx()
                self.assertEqual(self.validate(*rows, ctx=ctx), [])
                self.assertEqual(reasons(ctx), [reason])

    def test_turn_range_param(self):
        ctx = make_ctx(turns_min=3, turns_max=4)
        self.assertEqual(self.validate(dialogue(), ctx=ctx), [])
        self.assertEqual(reasons(ctx), ["turn_count"])

    def test_params_are_range_checked_and_resolved(self):
        self.assertIsNone(check_params({}))
        self.assertEqual(check_params({"turns_min": 1})[0], "turns_min")
        self.assertEqual(check_params({"turns_min": 5, "turns_max": 3})[0], "turns_min")
        self.assertEqual(check_params({"passages_per_dialogue": 4})[0], "passages_per_dialogue")
        self.assertEqual(resolve_params({}), {"turns_min": 2, "turns_max": 5, "dialogues_per_passage_group": 1, "passages_per_dialogue": 2})

    def test_prompt_numbers_passages_and_schema_constrains_roles(self):
        prompt = self.pb.build_prompt(self.ctx)
        self.assertIn("GROUP 1:", prompt)
        self.assertIn("[1] The calibration", prompt)
        schema = self.pb.response_schema(self.ctx)
        self.assertEqual(schema["properties"]["messages"]["items"]["properties"]["role"]["enum"], ["user", "assistant"])


class _StubTeacher:
    """Offline teacher: one two-exchange dialogue per GROUP in the prompt."""

    name = "stub-teacher"
    schema_aware = True
    reply: str | None = None

    @classmethod
    def is_available(cls):
        return True

    def describe(self):
        return "stub teacher"

    async def complete(self, prompt, *, system_prompt=None, max_tokens=1024, temperature=0.7, response_schema=None):
        if self.reply is not None:
            return self.reply
        lines = []
        for g, block in enumerate(re.findall(r"GROUP \d+:\n((?:\[\d+\][^\n]*\n?)+)", prompt), start=1):
            passages = re.findall(r"\[(\d+)\] ([^\n]*)", block)
            first = passages[0][1]
            lines.append(json.dumps({"group": g, "messages": [
                {"role": "user", "content": f"Tell me about item {g}?"},
                {"role": "assistant", "content": f"{first} [1]"},
                {"role": "user", "content": "And what else does it say about that?"},
                {"role": "assistant", "content": f"{first} [1]"},
            ]}))
        return "\n".join(lines)


class PlaybookApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._patches = [
            patch("app.services.synth_backends.BACKEND_REGISTRY", [_StubTeacher]),
            patch("app.services.synth_playbook_service.BACKEND_REGISTRY", [_StubTeacher]),
        ]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        settings.AUTH_ENABLED = cls._prev_auth

    def tearDown(self):
        _StubTeacher.reply = None
        wait_for_jobs_idle(client)

    def setUp(self):
        r = client.post("/api/projects", json={"name": f"mt-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        resp = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "generic-sft"})
        if resp.status_code != 200:
            from app.database import async_session_factory
            from app.models.project import Project

            async def go():
                async with async_session_factory() as db:
                    (await db.get(Project, self.pid)).selected_recipe = {"recipe_id": "generic-sft"}
                    await db.commit()

            asyncio.run(go())
        self.pdir = Path(settings.DATA_DIR) / "projects" / str(self.pid)
        self.url = f"/api/projects/{self.pid}/synthetic/run-playbook"

    def _passages(self, chunks=CHUNKS):
        path = self.pdir / "cleaned" / "cleaned.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")

    def _run(self, **kw):
        body = {"mode": "documents_multi_turn", "target_count": 2, "params": {"passages_per_dialogue": 2}}
        body.update(kw)
        return client.post(self.url, json=body)

    def test_run_writes_messages_rows_with_provenance(self):
        self._passages()
        resp = self._run()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(body["rows"]), 2)
        saved = [json.loads(l) for l in (self.pdir / "synthetic" / "synthetic.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(len(saved), 2)
        for row in saved:
            self.assertEqual(len(row["messages"]), 5)
            self.assertEqual(row["messages"][0]["role"], "system")
            self.assertIn("[1] (manual.txt", row["messages"][0]["content"])
            self.assertNotIn("system_prompt", row)
            self.assertEqual(row["source_passages"][0]["source_doc"], "manual.txt")
            self.assertEqual(row["review_status"], "pending")

    def test_zero_accepted_raises_with_the_reject_breakdown(self):
        self._passages()
        _StubTeacher.reply = json.dumps(dialogue(a2=TEXT_B))  # no citation
        resp = self._run(target_count=1)
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("no_citation", resp.text)

    def test_answer_key_test_questions_are_rejected_via_the_api(self):
        from app.database import async_session_factory
        from app.models.dataset import Dataset, DatasetType

        self._passages()
        gold = self.pdir / "gold_test.jsonl"
        gold.write_text(json.dumps({"question": "Tell me about item 1?", "answer": "x"}) + "\n", encoding="utf-8")

        async def go():
            async with async_session_factory() as db:
                db.add(Dataset(project_id=self.pid, name="gold", dataset_type=DatasetType.GOLD_TEST,
                               file_path=str(gold), record_count=1))
                await db.commit()

        asyncio.run(go())
        resp = self._run(target_count=1)
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("test_question_leak", resp.text)

    def test_no_passages_is_an_error_not_an_empty_success(self):
        resp = self._run()
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("document passages", resp.text)

    def test_param_out_of_range_is_400(self):
        self._passages()
        resp = self._run(params={"turns_min": 1})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["error_code"], "invalid_playbook_param")


class BuildPairsTests(unittest.TestCase):
    def build(self, records, include=("wrong", "partial"), **kw):
        return pp.build_pairs(records, include=include, eval_result_id=5, experiment_id=7, source_split="validation", **kw)

    def test_wrong_and_partial_become_pairs_and_every_skip_is_counted(self):
        records = [
            {"row_index": 0, "prompt": "q0", "reference": "ref0", "prediction": "bad0", "row_judge_verdict": "wrong"},
            {"row_index": 1, "prompt": "q1", "reference": "ref1", "prediction": "meh1", "row_judge_verdict": "partial"},
            {"row_index": 2, "prompt": "q2", "reference": "ref2", "prediction": "ref2", "row_judge_verdict": "correct"},
            {"row_index": 3, "prompt": "q3", "reference": "ref3", "prediction": "x"},
            {"row_index": 4, "prompt": "q4", "reference": "ref4", "prediction": " ", "row_judge_verdict": "wrong"},
            {"row_index": 5, "prompt": "q5", "reference": "", "prediction": "x", "row_judge_verdict": "wrong"},
            {"row_index": 6, "prompt": "q6", "reference": "Same  text", "prediction": "Same text", "row_judge_verdict": "wrong"},
            {"row_index": 7, "prompt": "q7", "reference": "r", "row_judge_verdict": "wrong"},
        ]
        pairs, skipped = self.build(records)
        self.assertEqual([p["row_index"] for p in pairs], [0, 1])
        self.assertEqual(pairs[0], {"prompt": "q0", "chosen": "ref0", "rejected": "bad0", "source": "eval",
                                    "source_split": "validation", "eval_result_id": 5, "experiment_id": 7,
                                    "row_index": 0, "verdict": "wrong"})
        self.assertEqual(skipped, {"correct": 1, "not_judged": 1, "empty_prediction": 1, "empty_reference": 1,
                                   "prediction_equals_reference": 1, "no_prediction_stored": 1})
        _pairs, skipped = self.build(records[:2], include=("wrong",))
        self.assertEqual(skipped, {"not_included": 1})

    def test_chat_rows_use_the_eval_render_as_prompt_and_keep_messages(self):
        history = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        record = {"row_index": 0, "reference": "r", "prediction": "p", "row_judge_verdict": "wrong", "history": history}
        pairs, _ = self.build([record])
        self.assertEqual(pairs[0]["prompt"], "system: S\nuser: U\nassistant: ")
        self.assertEqual(pairs[0]["messages"], history)

    def test_a_chat_row_without_an_answer_verdict_uses_the_dialogue_judges_final_turn(self):
        history = [{"role": "user", "content": "U"}]
        block = {"turns": [{"turn": 1, "score": 0.0, "verdict": "wrong"}, {"turn": 3, "score": 1.0, "verdict": "correct"},
                           {"turn": 2, "score": 0.0, "verdict": "wrong"}]}
        records = [
            {"row_index": 0, "reference": "r", "prediction": "p", "history": history, "dialogue_judge": block},
            {"row_index": 1, "reference": "r", "prediction": "p", "history": history,
             "dialogue_judge": {"turns": [{"turn": 1, "verdict": "correct"}, {"turn": 2, "verdict": "partial"}]}},
            # the answer judge's verdict wins when both exist
            {"row_index": 2, "reference": "r", "prediction": "p", "history": history, "row_judge_verdict": "wrong",
             "dialogue_judge": block},
        ]
        pairs, skipped = self.build(records)
        self.assertEqual([(p["row_index"], p["verdict"]) for p in pairs], [(1, "partial"), (2, "wrong")])
        self.assertEqual(skipped, {"correct": 1})

    def test_rows_past_the_caps_are_reported_as_not_stored(self):
        history = [{"role": "user", "content": "U"}]
        records = [{"row_index": 0, "reference": "r", "prediction": "p", "history": history,
                    "row_judge_verdict": "wrong", "history_truncated": True}]
        pairs, skipped = self.build(records, not_stored=4)
        self.assertEqual(pairs, [])
        self.assertEqual(skipped, {"not_stored": 5})


class PairsEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

    @classmethod
    def tearDownClass(cls):
        settings.AUTH_ENABLED = cls._prev_auth

    def setUp(self):
        r = client.post("/api/projects", json={"name": f"pp-{time.time_ns()}", "description": "x"})
        self.pid = r.json()["id"]
        self.url = f"/api/projects/{self.pid}/alignment/preference-pairs/from-eval"

    def _eval(self, records=None, *, split="validation", pid=None, truncated=0, details=None):
        """Seed an eval result; ``records`` go to the side file exactly where the eval writes them."""
        from app.database import async_session_factory
        from app.models.experiment import EvalResult, Experiment, ExperimentStatus

        project_id = pid or self.pid

        async def go():
            async with async_session_factory() as db:
                exp = Experiment(project_id=project_id, name="e", description="", status=ExperimentStatus.COMPLETED,
                                 base_model="toy/model")
                db.add(exp)
                await db.flush()
                res = EvalResult(experiment_id=exp.id, dataset_name=split, eval_type="f1", metrics={},
                                 details=dict(details or {}))
                db.add(res)
                await db.flush()
                if records is not None:
                    rel = f"experiments/{exp.id}/evals/result_{res.id}_rows.jsonl"
                    path = Path(settings.DATA_DIR) / "projects" / str(project_id) / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
                    res.details = {**(res.details or {}),
                                   "row_predictions_file": {"path": rel, "count": len(records), "truncated": truncated}}
                await db.commit()
                return res.id, exp.id

        return asyncio.run(go())

    def test_builds_pairs_and_writes_the_dataset_file(self):
        rid, eid = self._eval([
            {"row_index": 0, "prompt": "q0", "reference": "ref0", "prediction": "bad0", "row_judge_verdict": "wrong"},
            {"row_index": 1, "prompt": "q1", "reference": "ref1", "prediction": "ref1", "row_judge_verdict": "correct"},
        ])
        resp = client.post(self.url, json={"eval_result_id": rid})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["created"], body["skipped"]), (1, {"correct": 1}))
        self.assertIsNone(body["dataset_id"])
        self.assertIsNone(body["warning"])
        self.assertEqual(body["source_split"], "validation")
        self.assertIn(f"eval #{rid}", body["dataset_name"])
        self.assertIn(f"run #{eid}", body["dataset_name"])
        self.assertEqual(body["examples"][0]["chosen"], "ref0")
        lines = Path(body["dataset_path"]).read_text().splitlines()
        self.assertEqual(json.loads(lines[0])["rejected"], "bad0")
        self.assertEqual(json.loads(lines[0])["source_split"], "validation")

    def test_the_test_split_is_refused_unless_overridden(self):
        records = [{"row_index": 0, "prompt": "q0", "reference": "ref0", "prediction": "bad0", "row_judge_verdict": "wrong"}]
        for split in ("test", "gold_test", ""):
            with self.subTest(split=split):
                rid, _ = self._eval(records, split=split)
                resp = client.post(self.url, json={"eval_result_id": rid})
                self.assertEqual(resp.status_code, 400, resp.text)
                body = resp.json()
                self.assertEqual(body["error_code"], "test_split_refused")
                self.assertIn("memorisation", body["message"])
                allowed = client.post(self.url, json={"eval_result_id": rid, "allow_test_split": True})
                self.assertEqual(allowed.status_code, 200, allowed.text)
                out = allowed.json()
                self.assertIn("memorisation", out["warning"])
                self.assertEqual(out["examples"][0]["source_split"], split or None)
                for line in Path(out["dataset_path"]).read_text().splitlines():
                    self.assertIn("source_split", json.loads(line))

    def test_zero_pairs_is_a_400_with_the_breakdown(self):
        rid, _ = self._eval([
            {"row_index": 0, "prompt": "q", "reference": "r", "prediction": "r", "row_judge_verdict": "correct"},
            {"row_index": 1, "prompt": "q", "reference": "r", "prediction": "x"},
        ], truncated=2)
        resp = client.post(self.url, json={"eval_result_id": rid})
        self.assertEqual(resp.status_code, 400, resp.text)
        body = resp.json()
        self.assertEqual(body["error_code"], "no_pairs_built")
        self.assertEqual(body["metadata"]["skipped"], {"correct": 1, "not_judged": 1, "not_stored": 2})

    def test_eval_without_a_side_file_and_foreign_eval(self):
        rid, _ = self._eval(details={"predictions_preview": [{"prompt": "q"}], "row_scores": {"keys": ["a", "b", "c"]}})
        resp = client.post(self.url, json={"eval_result_id": rid})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["metadata"]["skipped"], {"no_prediction_stored": 3})
        other = client.post("/api/projects", json={"name": f"pp2-{time.time_ns()}", "description": "x"}).json()["id"]
        rid2, _ = self._eval([], pid=other)
        self.assertEqual(client.post(self.url, json={"eval_result_id": rid2}).status_code, 404)
        self.assertEqual(client.post(self.url, json={"eval_result_id": rid, "include": ["bogus"]}).status_code, 422)

    def test_results_list_never_carries_row_arrays(self):
        rid, eid = self._eval([{"row_index": 0, "prompt": "q", "reference": "r", "prediction": "x",
                                "row_judge_verdict": "wrong"}],
                              details={"row_predictions": [{"prompt": "legacy inline rows"}], "failed_count": 1})
        resp = client.get(f"/api/projects/{self.pid}/evaluation/results/{eid}")
        self.assertEqual(resp.status_code, 200, resp.text)
        details = resp.json()[0]["details"]
        self.assertNotIn("row_predictions", details)
        self.assertEqual(details["row_predictions_file"]["count"], 1)
        self.assertEqual(details["failed_count"], 1)


if __name__ == "__main__":
    unittest.main()
