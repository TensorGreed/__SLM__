"""Multi-stage pipeline evaluation (G10): definitions, held-out upload, the
``pipeline_eval`` runner (fan-out, item-key alignment, both views,
attribution), auth and the internal gateway helper. Toy data only: stage 1
extracts ``items`` from a text, stage 2 rewrites each item. Generation is
stubbed (``local_chat_service.generate_reply``)."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
import _gateway_helpers as h  # noqa: E402
from _job_wait import wait_for_jobs_idle  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.models.auth import GlobalRole, ProjectRole  # noqa: E402
from app.security import create_user_with_key, upsert_project_membership  # noqa: E402
from app.services import gateway_service  # noqa: E402
from app.services import pipeline_eval_service as pes  # noqa: E402

ITEMS_AB = [{"label": "a", "name": "apple"}, {"label": "b", "name": "bean"}]
ITEMS_ABC = ITEMS_AB + [{"label": "c", "name": "carrot"}]
X = {"label": "x", "name": "xeno"}

EXTRACT = {
    "r1": {"items": ITEMS_AB + [X]},  # one spurious (x), one missed (c)
    "r2": {"items": ITEMS_AB},
    "r3": {"items": [ITEMS_AB[0], X]},
}
REWRITE = {"apple": "APPLE", "bean": "wrong", "xeno": "XENO", "carrot": "CARROT"}


class Stub:
    def __init__(self):
        self.calls = []

    def __call__(self, model_ref, messages, **kw):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if prompt.startswith("Extract items from: "):
            key = prompt.split(": ", 1)[1]
            reply = "```json\n" + json.dumps(EXTRACT.get(key, {"items": []})) + "\n```"
        elif prompt.startswith("Rewrite: "):
            reply = REWRITE.get(prompt.split(": ", 1)[1], "?")
        else:
            reply = "?"
        return {"reply": reply, "usage": {"prompt_tokens": 3, "completion_tokens": 2}, "finish_reason": "stop"}


@pytest.fixture(scope="module")
def client():
    cm, c, prev = h.open_client()
    yield c
    wait_for_jobs_idle(c)
    h.close_client(cm, prev)


@pytest.fixture(autouse=True)
def _reset(client):
    h.reset()
    yield
    wait_for_jobs_idle(client)
    h.reset()


@pytest.fixture(scope="module")
def root():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def definition(p1, p2, **over):
    body = {
        "name": f"toy-{uuid4().hex[:6]}",
        "description": "extract then rewrite",
        "stages": [
            {"name": "extract", "project_id": p1, "model": "",
             "prompt": {"mode": "template", "template": "Extract items from: {text}", "system": "Answer in JSON."},
             "inputs": {"text": {"from": "row", "pointer": "/text"}}, "output": "json"},
            {"name": "rewrite", "project_id": p2, "model": "",
             "prompt": {"mode": "template", "template": "Rewrite: {name}"},
             "inputs": {"name": {"from": "item", "pointer": "/name"}},
             "fan_out": {"pointer": "/items/*", "item_key": ["/label"]}, "output": "text"},
        ],
    }
    body.update(over)
    return body


def rw(items):
    return [{"label": it["label"], "output": out} for it, out in items]


ROWS = [
    {"id": "r1", "input": {"text": "r1"},
     "gold": {"extract": {"items": ITEMS_ABC},
              "rewrite": rw([(ITEMS_ABC[0], "APPLE"), (ITEMS_ABC[1], "BEAN"), (ITEMS_ABC[2], "CARROT")])}},
    {"id": "r2", "input": {"text": "r2"},
     "gold": {"extract": {"items": ITEMS_AB}, "rewrite": rw([(ITEMS_AB[0], "APPLE"), (ITEMS_AB[1], "BEAN")])}},
    # no stage-1 gold: stage 1 unscored, isolated view skipped; c is missed upstream
    {"id": "r3", "input": {"text": "r3"}, "gold": {"rewrite": rw([(ITEMS_AB[0], "APPLE"), (ITEMS_ABC[2], "CARROT")])}},
    {"id": "r4", "input": {"text": "r4"}, "gold": {}},  # no gold at all
    {"id": "r5", "input": {"other": "no text field"}, "gold": {}},  # missing_input
]


def make_pipeline(client, root, rows=ROWS, **over):
    p1, p2 = h.make_project(client), h.make_project(client)
    r1, r2 = h.seed_run(p1, root), h.seed_run(p2, root)
    resp = client.post("/api/pipelines", json=definition(p1, p2, **over))
    assert resp.status_code == 201, resp.text
    pl = resp.json()
    up = client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": rows})
    assert up.status_code == 200, up.text
    return pl, (p1, p2), (r1, r2)


def test_two_stage_fan_out_both_views_and_attribution(client, root):
    pl, (p1, p2), (run1, run2) = make_pipeline(client, root)
    stub = Stub()
    with mock.patch("app.services.local_chat_service.generate_reply", stub):
        resp = client.post(f"/api/pipelines/{pl['id']}/runs", json={"max_rows": 5})
    assert resp.status_code == 200, resp.text
    run = resp.json()
    assert run["status"] == "done" and run["rows"] == 5 and run["truncated"] is False
    # r1: 1 + 3 rewrites + 1 isolated (carrot); r2 / r3: extract only (rewrites cached); r4: extract
    assert run["calls"] == 8 == len(stub.calls)
    assert run["errors_by_reason"] == {"missing_input": 1}

    extract, rewrite = run["stages"]
    assert extract["project_id"] == p1 and extract["run_id"] == run1 and extract["model_id"].endswith(f":run-{run1}")
    assert rewrite["project_id"] == p2 and rewrite["run_id"] == run2 and rewrite["project_name"]
    assert extract["handler"] == "generic"
    assert extract["isolated"] is None
    assert extract["judge"] is None and extract["judge_not_run"] is False and extract["scoring_error"] is None
    assert extract["end_to_end"] == {"scored": 2, "correct": 1, "accuracy": 0.5, "unscored": 3,
                                     "unscored_by_reason": {"no_gold": 3}, "spurious": 0, "missed_upstream": 0,
                                     "lost_to_error": 0, "missing_item_key": 0, "duplicate_item_key": 0, "errors": 1}
    assert rewrite["end_to_end"] == {"scored": 5, "correct": 3, "accuracy": 0.6, "unscored": 0, "unscored_by_reason": {},
                                     "spurious": 2, "missed_upstream": 2, "lost_to_error": 0, "missing_item_key": 0,
                                     "duplicate_item_key": 0, "errors": 0}
    iso = rewrite["isolated"]
    assert (iso["scored"], iso["correct"], iso["spurious"], iso["missed_upstream"], iso["skipped_rows"]) == (5, 3, 0, 0, 3)
    # The producing stage is judged per item: r1 / r2 items equal to the gold items, r3 (no extract gold) by key.
    assert extract["item_compare"] == "mixed" and extract["item_compare_counts"] == {"deep_equal": 4, "key_only": 1}

    assert run["end_to_end"] == {"gold_final_items": 7, "unscored_items": 0, "unscored_by_reason": {},
                                 "fully_correct": 3, "share_fully_correct": round(3 / 7, 4)}
    assert run["attribution"] == {"first_wrong_stage": {"extract": 2, "rewrite": 2}, "correct": 3,
                                  "missed_upstream_at": {"extract": 2}}

    detail = {r["id"]: r for r in run["rows_detail"]}
    r1 = detail["r1"]
    assert r1["stages"][1]["inputs_count"] == 3
    outs = {tuple(o["item_key_values"]): o for o in r1["stages"][1]["outputs"]}
    assert outs[("a",)]["correct"] is True and outs[("b",)]["correct"] is False and outs[("x",)]["correct"] is None
    # item a was extracted exactly as gold and rewritten right: fully correct despite the spurious x / missed c
    assert [(f["fully_correct"], f["first_wrong_stage"]) for f in r1["final"]] == [
        (True, None), (False, "rewrite"), (False, "extract")]
    assert {tuple(f["gold_item_key"]): f["first_wrong_stage"] for f in detail["r3"]["final"]} == {
        ("a",): None, ("c",): "extract"}
    assert detail["r5"]["stages"][0]["outputs"][0]["error"] == "missing_input"
    assert detail["r4"]["final"] == []

    listed = client.get(f"/api/pipelines/{pl['id']}/runs").json()["runs"]
    assert listed[0]["run_id"] == run["run_id"] and listed[0]["share_fully_correct"] == round(3 / 7, 4)
    assert client.get(f"/api/pipelines/{pl['id']}/runs/{run['run_id']}").json()["calls"] == 8
    summary = next(p for p in client.get("/api/pipelines").json()["pipelines"] if p["id"] == pl["id"])
    assert summary["heldout_rows"] == 5 and summary["last_run"]["run_id"] == run["run_id"]
    assert [s["name"] for s in summary["stages"]] == ["extract", "rewrite"]
    # project -> pipelines lookup
    for pid in (p1, p2):
        got = client.get(f"/api/projects/{pid}/pipelines").json()["pipelines"]
        assert [p["id"] for p in got] == [pl["id"]]


def test_job_end_to_end(client, root):
    pl, _, _ = make_pipeline(client, root)
    stub = Stub()
    with mock.patch("app.services.local_chat_service.generate_reply", stub):
        resp = client.post(f"/api/pipelines/{pl['id']}/runs?async_job=true", json={"max_rows": 2})
        assert resp.status_code == 202, resp.text
        job = resp.json()
        assert job["kind"] == "pipeline_eval" and job["project_id"] is None
        run_id = job["params"]["run_id"]
        assert job["params"]["pipeline_id"] == pl["id"] and job["params"]["max_rows"] == 2
        wait_for_jobs_idle(client)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "succeeded", done
    result = done["result"]
    assert result["pipeline_id"] == pl["id"] and result["run_id"] == run_id and result["rows"] == 2
    assert result["attribution"]["first_wrong_stage"] == {"extract": 1, "rewrite": 2}
    assert result["share_fully_correct"] == 0.4
    run = client.get(f"/api/pipelines/{pl['id']}/runs/{run_id}").json()
    assert run["status"] == "done" and run["rows"] == 2


def test_nothing_scored_raises_and_is_recorded(client, root):
    rows = [{"id": "q1", "input": {"other": 1}, "gold": {"extract": {"items": []}}}]
    pl, _, _ = make_pipeline(client, root, rows=rows)
    with mock.patch("app.services.local_chat_service.generate_reply", Stub()):
        resp = client.post(f"/api/pipelines/{pl['id']}/runs", json={"max_rows": 5})
        assert resp.status_code == 400
        assert "no stage produced a model output" in resp.json()["detail"] and "missing_input: 1" in resp.json()["detail"]
        # outputs but no gold anywhere -> also refused
        client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": [{"id": "q2", "input": {"text": "r2"}}]})
        resp = client.post(f"/api/pipelines/{pl['id']}/runs", json={"max_rows": 5})
        assert resp.status_code == 400 and "nothing was scored" in resp.json()["detail"]
        # async: the Job fails with the diagnostic
        job = client.post(f"/api/pipelines/{pl['id']}/runs?async_job=true", json={}).json()
        wait_for_jobs_idle(client)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "failed" and "nothing was scored" in done["error"]
    run = client.get(f"/api/pipelines/{pl['id']}/runs/{job['params']['run_id']}").json()
    assert run["status"] == "failed" and "nothing was scored" in run["error"]
    # no held-out rows -> 400 before any Job
    client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": []})
    assert client.post(f"/api/pipelines/{pl['id']}/runs?async_job=true", json={}).status_code == 400
    assert client.post(f"/api/pipelines/{pl['id']}/runs", json={"max_rows": 501}).status_code == 400
    # the in-request path is for tiny runs only
    client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": ROWS})
    big = client.post(f"/api/pipelines/{pl['id']}/runs", json={"max_rows": 6})
    assert big.status_code == 400 and "async_job=true" in big.json()["detail"]
    assert client.post(f"/api/pipelines/{pl['id']}/runs", json={}).status_code == 400  # default 50 rows


def test_call_cap_truncates(client, root):
    pl, _, _ = make_pipeline(client, root)
    with mock.patch("app.services.local_chat_service.generate_reply", Stub()), mock.patch.object(pes, "MAX_MODEL_CALLS", 6):
        run = client.post(f"/api/pipelines/{pl['id']}/runs", json={"max_rows": 5}).json()
    assert run["truncated"] is True and run["rows"] == 2 and run["calls"] == 6
    assert run["truncated_at_row"] == "r3" and "Stopped at row 'r3'" in run["truncated_message"]


def test_validation_errors(client, root):
    p1, p2 = h.make_project(client), h.make_project(client)
    foreign = h.seed_run(p2, root)
    body = definition(p1, p2)
    body["name"] = ""
    body["stages"][0]["inputs"]["prev"] = {"from": "previous", "pointer": ""}
    body["stages"][0]["model"] = f":run-{foreign}"
    body["stages"][1]["name"] = "extract"
    body["stages"][1]["prompt"] = {"mode": "recipe"}
    body["stages"][1]["inputs"]["bad"] = {"from": "row", "pointer": "no-slash"}
    body["stages"].append({"name": "third", "project_id": 99999999, "prompt": {"mode": "template", "template": "{x}"},
                           "fan_out": {"pointer": "/items/*", "item_key": []}, "max_tokens": 0,
                           "inputs": {"x": {"from": "item", "pointer": ""}}})
    resp = client.post("/api/pipelines", json=body)
    assert resp.status_code == 400
    out = resp.json()
    fields = {e["field"] for e in out["errors"]}
    assert {
        "name", "stages[0].inputs.prev.from", "stages[0].model", "stages[1].name", "stages[1].prompt",
        "stages[1].inputs.bad.pointer", "stages[2].project_id", "stages[2].fan_out.item_key",
        "stages[2].max_tokens", "stages[2].fan_out",
    } <= fields, fields
    assert out["metadata"]["errors"] == out["errors"] and isinstance(out["detail"], str)
    assert out["error_code"] == "PIPELINE_INVALID"
    fan_err = [e["message"] for e in out["errors"] if e["field"] == "stages[2].fan_out"]
    assert any("only one stage" in m for m in fan_err)
    many = definition(p1, p2)
    many["stages"] = many["stages"][:1] * 9
    assert client.post("/api/pipelines", json=many).status_code == 400
    # 'item' without fan_out
    bad = definition(p1, p2)
    del bad["stages"][1]["fan_out"]
    errs = client.post("/api/pipelines", json=bad).json()["errors"]
    assert {"field": "stages[1].inputs.name.from", "message": "'item' needs fan_out on this stage"} in errs
    # update keeps the id and validates too
    ok = client.post("/api/pipelines", json=definition(p1, p2)).json()
    upd = definition(p1, p2, name="renamed")
    put = client.put(f"/api/pipelines/{ok['id']}", json=upd)
    assert put.status_code == 200 and put.json()["id"] == ok["id"] and put.json()["name"] == "renamed"
    assert put.json()["created_at"] == ok["created_at"]
    assert client.put(f"/api/pipelines/{ok['id']}", json={**upd, "stages": []}).status_code == 400
    assert client.get("/api/pipelines/pl-00000000").status_code == 404
    assert client.get("/api/pipelines/..%2Fetc").status_code == 404
    assert client.delete(f"/api/pipelines/{ok['id']}").status_code == 200
    assert client.get(f"/api/pipelines/{ok['id']}").status_code == 404


def test_heldout_upload_partial_accept(client, root):
    p1, p2 = h.make_project(client), h.make_project(client)
    pl = client.post("/api/pipelines", json=definition(p1, p2)).json()
    lines = [
        json.dumps({"id": "a", "input": {"text": "x"}, "gold": {"extract": {"items": []}}}),
        "{not json",
        json.dumps({"id": "a", "input": {"text": "y"}}),
        "",
        json.dumps({"id": "b", "input": {"text": "y"}, "gold": {"nope": 1}}),
        json.dumps({"id": "c", "input": {"text": "y"}, "gold": {"rewrite": {"label": "a"}}}),
        json.dumps({"id": "d", "input": "text"}),
        json.dumps({"input": {"text": "y"}}),
        json.dumps({"id": "e", "input": {"text": "z"}, "gold": {"rewrite": [{"label": "a", "output": "A"}]}}),
        json.dumps({"id": "f", "input": {"text": "z"}, "gold": {"rewrite": [{"output": "A"}]}}),
        json.dumps({"id": "g", "input": {"text": "z"}, "gold": {"rewrite": [{"label": None, "output": "A"}]}}),
    ]
    resp = client.put(
        f"/api/pipelines/{pl['id']}/heldout",
        files={"file": ("heldout.jsonl", "\n".join(lines).encode(), "application/jsonl")},
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["accepted"] == 2
    assert out["rejected_by_reason"] == {
        "invalid_json": [{"line": 2, "id": None}],
        "duplicate_id": [{"line": 3, "id": "a"}],
        "unknown_stage_in_gold": [{"line": 5, "id": "b"}],
        "fan_out_gold_not_list": [{"line": 6, "id": "c"}],
        "input_not_object": [{"line": 7, "id": "d"}],
        "missing_id": [{"line": 8, "id": None}],
        "fan_out_gold_item_missing_key": [{"line": 10, "id": "f"}, {"line": 11, "id": "g"}],
    }
    got = client.get(f"/api/pipelines/{pl['id']}/heldout?limit=1").json()
    assert got["total"] == 2 and [r["id"] for r in got["rows"]] == ["a"]
    assert client.put(f"/api/pipelines/{pl['id']}/heldout", json={"nope": 1}).status_code == 400


def test_auth_visibility_and_roles(client, root):
    pl, (p1, p2), _ = make_pipeline(client, root)
    p3 = h.make_project(client)

    async def _user(role_by_project):
        async with async_session_factory() as db:
            u = await create_user_with_key(db, f"u-{uuid4().hex[:6]}", GlobalRole.ENGINEER)
            for pid, role in role_by_project.items():
                await upsert_project_membership(db, pid, u["user_id"], role)
            await db.commit()
            return {"x-api-key": u["api_key"]}

    viewer = asyncio.run(_user({p1: ProjectRole.VIEWER, p2: ProjectRole.VIEWER}))
    half = asyncio.run(_user({p1: ProjectRole.EDITOR}))
    editor = asyncio.run(_user({p1: ProjectRole.EDITOR, p2: ProjectRole.EDITOR, p3: ProjectRole.EDITOR}))
    settings.AUTH_ENABLED = True
    try:
        assert client.get("/api/pipelines").status_code == 401

        def ids(hdr):
            return {p["id"] for p in client.get("/api/pipelines", headers=hdr).json()["pipelines"]}

        assert pl["id"] in ids(viewer) and pl["id"] in ids(editor) and pl["id"] not in ids(half)
        assert client.get(f"/api/pipelines/{pl['id']}", headers=viewer).status_code == 200
        assert client.get(f"/api/pipelines/{pl['id']}", headers=half).status_code == 403
        assert client.get(f"/api/pipelines/{pl['id']}/runs", headers=half).status_code == 403
        assert client.get(f"/api/pipelines/{pl['id']}/heldout", headers=half).status_code == 403
        # project lookup: hidden when the caller can't see every member; 403 without access to the project
        assert client.get(f"/api/projects/{p1}/pipelines", headers=half).json()["pipelines"] == []
        assert client.get(f"/api/projects/{p2}/pipelines", headers=half).status_code == 403
        assert [p["id"] for p in client.get(f"/api/projects/{p1}/pipelines", headers=viewer).json()["pipelines"]] == [pl["id"]]
        # writes need EDITOR on every member
        assert client.post(f"/api/pipelines/{pl['id']}/runs", json={}, headers=viewer).status_code == 403
        assert client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": []}, headers=viewer).status_code == 403
        assert client.delete(f"/api/pipelines/{pl['id']}", headers=half).status_code == 403
        assert client.post("/api/pipelines", json=definition(p1, p2), headers=half).status_code == 403
        created = client.post("/api/pipelines", json=definition(p1, p3), headers=editor)
        assert created.status_code == 201, created.text
        # editing to add a project the user can't edit is refused
        assert client.put(f"/api/pipelines/{created.json()['id']}", json=definition(p1, p2),
                          headers=half).status_code == 403
    finally:
        settings.AUTH_ENABLED = False


def test_internal_gateway_helper_no_telemetry_no_rate_limit(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    stub = Stub()

    async def _count():
        async with async_session_factory() as db:
            return (await db.execute(text("select count(*) from gateway_request_logs where project_id = :p"),
                                     {"p": pid})).scalar()

    async def _calls(n):
        out = []
        async with async_session_factory() as db:
            for _ in range(n):
                out.append(await gateway_service.run_internal(
                    db, f"p{pid}", [{"role": "user", "content": "Rewrite: apple"}], project_ids=[pid], max_tokens=16))
        return out

    before = asyncio.run(_count())
    with mock.patch("app.services.local_chat_service.generate_reply", stub), \
            mock.patch("app.services.served_model_telemetry_service.record_gateway_request") as rec:
        results = asyncio.run(_calls(5))  # the internal key's limit is 0/min: any limiter call would 429
    assert [r["choices"][0]["message"]["content"] for r in results] == ["APPLE"] * 5
    assert rec.call_count == 0
    assert asyncio.run(_count()) == before
    assert gateway_service.INTERNAL_KEY_ID not in gateway_service._RATE_WINDOWS
    assert gateway_service.internal_key([pid]).rate_limit_per_minute == 0
    # a pipeline run writes no gateway request rows either
    pl, (p1, p2), _ = make_pipeline(client, root)

    async def _count_all():
        async with async_session_factory() as db:
            return (await db.execute(text("select count(*) from gateway_request_logs where project_id in (:a, :b)"),
                                     {"a": p1, "b": p2})).scalar()

    with mock.patch("app.services.local_chat_service.generate_reply", Stub()):
        assert client.post(f"/api/pipelines/{pl['id']}/runs", json={"max_rows": 5}).status_code == 200
    assert asyncio.run(_count_all()) == 0


def test_parse_output_and_fence_helper():
    from app.services.eval_task_handler_service import strip_code_fences

    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('```json {"a": 1}```') == '{"a": 1}'
    assert strip_code_fences("plain") == "plain"
    assert pes.parse_output('```\n[1, 2]\n```', "json") == (True, [1, 2])
    assert pes.parse_output('Sure: {"a": 1} done', "json") == (True, {"a": 1})
    assert pes.parse_output("nope", "json") == (False, None)
    assert pes.parse_output("  hi ", "text") == (True, "hi")
    assert pes.key_of({"label": " A ", "line": 3}, ["/label", "/line"]) == (("a", "3"), [" A ", 3])


# ---------------------------------------------------------------- G10 hardening: scoring = held-out scoring

import copy  # noqa: E402

from app.services import evaluation_service as es  # noqa: E402
from app.services import pipeline_service as ps  # noqa: E402
from app.services.answer_judge_service import ResolvedJudge  # noqa: E402
from app.services.data_adapter_service import map_record_with_adapter  # noqa: E402
from app.services.unified_diff import make_unified_diff  # noqa: E402


@pytest.fixture(autouse=True)
def _no_judge_backend():
    # No judge is ever resolved over the network: tests that want one stub resolve_answer_judge.
    with mock.patch.dict("os.environ", {"EVAL_JUDGE_BACKEND": "none"}):
        yield


class MapStub:
    """generate_reply stub: the reply for a prompt is ``replies[prompt]`` ("?" otherwise)."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def __call__(self, model_ref, messages, **kw):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        reply = self.replies.get(prompt)
        if reply is None:  # prefix match (multi-line prompts)
            reply = next((v for k, v in self.replies.items() if k.endswith("*") and prompt.startswith(k[:-1])), "?")
        return {"reply": reply if isinstance(reply, str) else json.dumps(reply),
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}, "finish_reason": "stop"}


def prepared(pid: int, manifest: dict, rows: list[dict]) -> None:
    d = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def one_stage(client, pid, template, inputs, rows, output="json"):
    body = {"name": f"one-{uuid4().hex[:6]}", "stages": [
        {"name": "solo", "project_id": pid, "prompt": {"mode": "template", "template": template},
         "inputs": {k: {"from": "row", "pointer": f"/{k}"} for k in inputs}, "output": output}]}
    resp = client.post("/api/pipelines", json=body)
    assert resp.status_code == 201, resp.text
    pl = resp.json()
    up = client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": rows})
    assert up.status_code == 200 and up.json()["accepted"] == len(rows), up.text
    return pl


def run_sync(client, pl, stub, n=5):
    with mock.patch("app.services.local_chat_service.generate_reply", stub):
        return client.post(f"/api/pipelines/{pl['id']}/runs", json={"max_rows": n})


def outputs_of(run):
    return {r["id"]: r["stages"][0]["outputs"][0] for r in run["rows_detail"]}


CODE = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = " + uid\n    cur.execute(q)\n    return cur.fetchone()\n'
FIXED = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = ?"\n    cur.execute(q, (uid,))\n    return cur.fetchone()\n'
BROKEN = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = ?"\n    cur.execute(q, (uid,\n    return cur.fetchone()\n'
FINDING = {"label": "query.unsafe", "start_line": 11, "end_line": 12, "evidence": "q = ... + uid", "attrs": {}}


def fix_payload(new_code):
    return {"patch": make_unified_diff(CODE, new_code, path="app/db.py", start_line=10),
            "rationale": "Bind the id as a parameter.", "citations": []}


def test_remediation_stage_scores_each_patch_on_its_own_row(client, root):
    """Without the row behind the patch the remediation scorer skips the row (counted correct); the
    pseudo-row carries code + finding, so a good patch scores 1, a broken one 0, a row without a
    finding is unscored."""
    pid = h.make_project(client, recipe="code-remediation")
    h.seed_run(pid, root)
    schema = {"scoring_mode": "remediation", "required": ["patch", "rationale", "citations"], "properties": {},
              "remediation": {"checks": None, "retrieve_passages": False, "top_k": 3, "retrieval": None},
              "label_parents": {}, "label_titles": {}}
    sample = {"row_kind": "remediation", "code": CODE, "language": "python", "path": "app/db.py", "start_line": 10,
              "finding": dict(FINDING), "remediation_json": fix_payload(FIXED)}
    sample = {**sample, **map_record_with_adapter(sample, adapter_id="structured-extraction", adapter_config={
        "remediation": {"extra_fields": {}, "project_id": 0, "retrieve_passages": False, "top_k": 3, "retrieval": None,
                        "label_titles": {}}})}
    prepared(pid, {"task_profile": "structured_extraction", "adapter_id": "structured-extraction",
                   "output_schema": schema}, [sample])
    base = {"code": CODE, "language": "python", "path": "app/db.py", "start_line": 10, "finding": FINDING}
    rows = [
        {"id": "good", "input": {**base, "case": "good"}, "gold": {"solo": fix_payload(FIXED)}},
        {"id": "broken", "input": {**base, "case": "broken"}, "gold": {"solo": fix_payload(FIXED)}},
        {"id": "nofinding", "input": {"code": CODE, "language": "python", "case": "nofinding"},
         "gold": {"solo": fix_payload(FIXED)}},
    ]
    pl = one_stage(client, pid, "Fix ({case}): {code}", ["code", "language", "path", "start_line", "finding", "case"],
                   rows[:2])
    pl_nf = one_stage(client, pid, "Fix ({case}): {code}", ["code", "language", "case"], rows[2:])
    stub = MapStub({"Fix (good)*": fix_payload(FIXED), "Fix (broken)*": fix_payload(BROKEN),
                    "Fix (nofinding)*": fix_payload(FIXED)})
    resp = run_sync(client, pl, stub)
    assert resp.status_code == 200, resp.text
    run = resp.json()
    stage = run["stages"][0]
    assert stage["scoring_mode"] == "remediation"
    assert (stage["end_to_end"]["scored"], stage["end_to_end"]["correct"]) == (2, 1)
    assert {k: v["correct"] for k, v in outputs_of(run).items()} == {"good": True, "broken": False}
    # The remediation judge would apply, but none is reachable: reported -- and the patch's own checks
    # still decide pass / fail exactly as in held-out, so nothing is unscored.
    assert stage["judge_not_run"] is True and stage["judge"]["reason"] == "no_judge_available"
    # No finding to fix: the scorer has nothing to apply the patch to -> unscored, never correct.
    resp = run_sync(client, pl_nf, stub)
    assert resp.status_code == 400 and "row_context_unavailable: 1" in resp.json()["detail"], resp.text


STEP = {"type": "object", "required": ["action", "target"],
        "properties": {"action": {"enum": ["create", "update", "delete"]}, "target": {"type": "string"}},
        "additionalProperties": False}
PLAN = {"type": "object", "required": ["name", "steps"],
        "properties": {"name": {"type": "string"}, "steps": {"type": "array", "minItems": 1,
                                                             "items": {"$ref": "step.schema.json"}}}}
FILES = {"plan.schema.json": PLAN, "step.schema.json": STEP}
RUBRIC = """
id: plan-quality
title: Plan quality
criteria:
  - id: naming
    description: The name says what is rolled out.
    anchors: {1: none, 2: vague, 3: ok, 4: clear, 5: precise}
"""
GOOD_PLAN = {"name": "rollout", "steps": [{"action": "update", "target": "svc"}]}


def plan_project(client, root, *, rubric: bool):
    from app.services.dataset_service import _recipe_output_schema

    pid = h.make_project(client, recipe="structured-plan")
    h.seed_run(pid, root)
    cfg = {"schema_files": FILES, "root_schema": "plan.schema.json", "retrieve_passages": False,
           "judge_enabled": rubric}
    if rubric:
        cfg["rubric_yaml"] = RUBRIC
    assert client.put(f"/api/projects/{pid}/structured-plan-config", json=cfg).status_code == 200

    async def _schema():
        async with async_session_factory() as db:
            return await _recipe_output_schema(db, pid)

    schema = asyncio.run(_schema())
    row = {"row_kind": "structured_plan", "scenario": "Roll out svc.", "plan_json": GOOD_PLAN}
    mapped = map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config={"structured_plan": {
        "schema_files": FILES, "root_schema": "plan.schema.json", "project_id": 0, "retrieve_passages": False}})
    prepared(pid, {"task_profile": "structured_extraction", "adapter_id": "structured-extraction",
                   "output_schema": schema}, [{**row, **mapped}])
    rows = [{"id": "ok", "input": {"scenario": "Roll out svc."}, "gold": {"solo": GOOD_PLAN}},
            {"id": "bad", "input": {"scenario": "Roll back svc."}, "gold": {"solo": GOOD_PLAN}}]
    pl = one_stage(client, pid, "Plan: {scenario}", ["scenario"], rows)
    stub = MapStub({"Plan: Roll out svc.": GOOD_PLAN, "Plan: Roll back svc.": {"name": "rollout"}})  # no steps
    return pid, pl, stub


def test_structured_plan_stage_and_rubric_judge(client, root):
    _, pl, stub = plan_project(client, root, rubric=False)
    run = run_sync(client, pl, stub).json()
    stage = run["stages"][0]
    assert stage["scoring_mode"] == "structured_plan" and stage["judge"] is None
    assert {k: v["correct"] for k, v in outputs_of(run).items()} == {"ok": True, "bad": False}

    # A rubric applies but no judge is reachable: the plan that passed its checks cannot be called right
    # -> unscored (needs judge); the plan that failed its schema is wrong whatever the rubric says.
    pid, pl, stub = plan_project(client, root, rubric=True)
    run = run_sync(client, pl, stub).json()
    stage = run["stages"][0]
    assert stage["judge_not_run"] is True and stage["judge"] == {
        "kind": "rubric", "label": None, "ran": False, "units": 0, "reason": "no_judge_available"}
    assert stage["end_to_end"]["unscored_by_reason"] == {"needs_judge": 1}
    assert {k: (v["correct"], v["unscored_reason"]) for k, v in outputs_of(run).items()} == {
        "ok": (None, "needs_judge"), "bad": (False, None)}
    assert run["end_to_end"]["unscored_items"] == 1 and run["end_to_end"]["share_fully_correct"] == 0.0

    # With a (stubbed) judge it runs per stage view and decides the check-passing plan.
    judge = ResolvedJudge(label="stub:judge", judge=None, complete=lambda *a: None)

    def fake_grade(predictions, rubric, resolved, cache=None, view=None):
        for p in predictions:
            p["row_rubric_passed"] = True
        return {"rubric": {"status": "judged", "pass_rate": 1.0, "id": rubric.id}}

    async def fake_resolve(*a, **k):
        return judge

    with mock.patch("app.services.answer_judge_service.resolve_answer_judge", fake_resolve), \
            mock.patch("app.services.rubric_service.judge_rows_with_rubric", fake_grade):
        run = run_sync(client, pl, stub).json()
    stage = run["stages"][0]
    assert stage["judge_not_run"] is False
    assert stage["judge"] == {"kind": "rubric", "label": "stub:judge", "ran": True, "units": 2}
    assert {k: v["correct"] for k, v in outputs_of(run).items()} == {"ok": True, "bad": False}


def test_answer_judge_runs_per_stage_or_leaves_units_unscored(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    long_ref = "The capital city of France is Paris, on the river Seine."
    prepared(pid, {"task_profile": "qa"}, [{"question": "q", "answer": long_ref}])
    rows = [{"id": "q1", "input": {"q": "capital?"}, "gold": {"solo": long_ref}},
            {"id": "q2", "input": {"q": "river?"}, "gold": {"solo": long_ref}}]
    pl = one_stage(client, pid, "Q: {q}", ["q"], rows, output="text")
    stub = MapStub({"Q: capital?": "It is Paris, which sits on the Seine river in France.",
                    "Q: river?": "Lyon is the capital of France."})
    resp = run_sync(client, pl, stub)
    # A long-answer judge applies and none is reachable: every unit is unscored, so nothing is scored.
    assert resp.status_code == 400 and "needs_judge: 2" in resp.json()["detail"], resp.text

    async def verdict(question, reference, prediction):
        return (1.0, "correct", "ok", 0) if "Paris" in prediction else (0.0, "wrong", "no", 0)

    async def fake_resolve(*a, **k):
        return ResolvedJudge(label="stub:answers", judge=verdict)

    with mock.patch("app.services.answer_judge_service.resolve_answer_judge", fake_resolve):
        run = run_sync(client, pl, stub).json()
    stage = run["stages"][0]
    assert stage["judge"] == {"kind": "answer", "label": "stub:answers", "ran": True, "units": 2}
    assert {k: v["correct"] for k, v in outputs_of(run).items()} == {"q1": True, "q2": False}


def test_tool_call_stage_perfect_call_scores_one(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    sample = map_record_with_adapter({"prompt": "Look up svc-0", "tool_name": "get_service",
                                      "tool_args": {"name": "svc-0"}, "result": "ok"},
                                     adapter_id="tool-call-json", adapter_config={})
    prepared(pid, {"task_profile": "tool_calling", "adapter_id": "tool-call-json"}, [sample])
    call = {"tool": "get_service", "args": {"name": "svc-1"}}
    rows = [{"id": "t1", "input": {"ask": "Look up svc-1"}, "gold": {"solo": call}},
            {"id": "t2", "input": {"ask": "Look up svc-2"}, "gold": {"solo": {**call, "args": {"name": "svc-2"}}}}]
    pl = one_stage(client, pid, "{ask}", ["ask"], rows)
    stub = MapStub({"Look up svc-1": call, "Look up svc-2": {**call, "args": {"name": "svc-9"}}})
    run = run_sync(client, pl, stub).json()
    assert run["stages"][0]["handler"] == "tool_calling"
    assert {k: v["correct"] for k, v in outputs_of(run).items()} == {"t1": True, "t2": False}


def test_field_match_stage(client, root):
    pid = h.make_project(client)
    h.seed_run(pid, root)
    schema = {"scoring_mode": "field_match", "required": ["name", "env"],
              "properties": {"name": {"type": "string"}, "env": {"type": "string"}}}
    prepared(pid, {"task_profile": "structured_extraction", "adapter_id": "structured-extraction",
                   "output_schema": schema}, [{"text": "x", "structured_output": {"name": "a", "env": "b"}}])
    rows = [{"id": "f1", "input": {"text": "one"}, "gold": {"solo": {"name": "api", "env": "prod"}}},
            {"id": "f2", "input": {"text": "two"}, "gold": {"solo": {"name": "api", "env": "prod"}}}]
    pl = one_stage(client, pid, "Extract: {text}", ["text"], rows)
    stub = MapStub({"Extract: one": '```json\n{"env": "prod", "name": "api"}```',
                    "Extract: two": {"name": "api", "env": "dev"}})
    run = run_sync(client, pl, stub).json()
    assert run["stages"][0]["handler"] == "structured_extraction"
    assert {k: v["correct"] for k, v in outputs_of(run).items()} == {"f1": True, "f2": False}


def test_scorer_failure_leaves_units_unscored(client, root):
    pl, (p1, p2), _ = make_pipeline(client, root)
    real = es.score_predictions_like_heldout

    async def flaky(db, project_id, predictions, **kw):
        if project_id == p2:
            raise RuntimeError("scorer exploded")
        return await real(db, project_id, predictions, **kw)

    with mock.patch.object(es, "score_predictions_like_heldout", flaky):
        run = run_sync(client, pl, Stub()).json()
    extract, rewrite = run["stages"]
    assert rewrite["scoring_error"] == "scorer failed: scorer exploded"
    assert rewrite["end_to_end"]["scored"] == 0 and rewrite["end_to_end"]["unscored_by_reason"] == {"scoring_error": 5}
    assert extract["end_to_end"]["scored"] == 2 and extract["scoring_error"] is None
    # Final items that reach the unscored stage are neither correct nor blamed on it.
    e2e = run["end_to_end"]
    assert e2e["fully_correct"] == 0 and e2e["unscored_by_reason"] == {"scoring_error": 5}
    assert e2e["share_fully_correct"] == 0.0 and run["attribution"]["first_wrong_stage"] == {"extract": 2}


def test_three_stages_gold_free_middle_and_lost_to_error(client, root):
    p1, p2, p3 = (h.make_project(client) for _ in range(3))
    for p in (p1, p2, p3):
        h.seed_run(p, root)
    body = {"name": f"three-{uuid4().hex[:6]}", "stages": [
        {"name": "extract", "project_id": p1, "prompt": {"mode": "template", "template": "Extract items from: {text}"},
         "inputs": {"text": {"from": "row", "pointer": "/text"}}},
        {"name": "classify", "project_id": p2, "prompt": {"mode": "template", "template": "Classify: {name}"},
         "inputs": {"name": {"from": "item", "pointer": "/name"}},
         "fan_out": {"pointer": "/items/*", "item_key": ["/label"]}},
        {"name": "fix", "project_id": p3, "prompt": {"mode": "template", "template": "Fix: {kind}"},
         "inputs": {"kind": {"from": "previous", "pointer": "/kind"}}, "output": "text"},
    ]}
    pl = client.post("/api/pipelines", json=body).json()
    rows = [
        {"id": "m1", "input": {"text": "r2"}, "gold": {
            "extract": {"items": ITEMS_AB},
            "fix": [{"label": "a", "output": "FRUIT"}, {"label": "b", "output": "VEG"}]}},
        # extract gold only: no fix gold, so the isolated view of fix is skipped (no wasted calls)
        {"id": "m2", "input": {"text": "r2"}, "gold": {"extract": {"items": ITEMS_AB}}},
    ]
    assert client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": rows}).json()["accepted"] == 2
    stub = MapStub({"Extract items from: r2": {"items": ITEMS_AB}, "Classify: apple": {"kind": "fruit"},
                    "Classify: bean": "not json", "Fix: fruit": "FRUIT"})
    run = run_sync(client, pl, stub).json()
    extract, classify, fix = run["stages"]
    assert classify["end_to_end"]["unscored_by_reason"] == {"no_gold": 4} and classify["end_to_end"]["errors"] == 2
    # m1's item b never reached fix because classify errored on it: lost to the error, not "missed"
    assert fix["end_to_end"]["lost_to_error"] == 1 and fix["end_to_end"]["missed_upstream"] == 0
    # isolated views need gold for the stage AND the one before: none here, so nothing extra ran
    assert classify["isolated"]["skipped_rows"] == 2 and fix["isolated"]["skipped_rows"] == 2
    assert len(stub.calls) == 4
    assert run["errors_by_reason"] == {"invalid_json": 2}
    final = {tuple(f["gold_item_key"]): f for f in run["rows_detail"][0]["final"]}
    assert final[("a",)]["fully_correct"] is True
    assert final[("b",)]["first_wrong_stage"] == "classify"
    assert run["attribution"] == {"first_wrong_stage": {"classify": 1}, "correct": 1, "missed_upstream_at": {}}


def test_duplicate_and_missing_item_keys(client, root):
    pl, _, _ = make_pipeline(client, root, rows=[
        {"id": "d1", "input": {"text": "dup"}, "gold": {
            "extract": {"items": [ITEMS_AB[0]]}, "rewrite": [{"label": "a", "output": "APPLE"}]}}])
    stub = MapStub({"Extract items from: dup": {"items": [ITEMS_AB[0], ITEMS_AB[0], {"name": "nokey"}]},
                    "Rewrite: apple": "APPLE", "Rewrite: nokey": "?"})
    run = run_sync(client, pl, stub).json()
    rw_view = run["stages"][1]["end_to_end"]
    assert (rw_view["duplicate_item_key"], rw_view["spurious"], rw_view["missing_item_key"]) == (1, 1, 1)
    assert (rw_view["scored"], rw_view["correct"]) == (1, 1)


def test_cancellation_is_checked_per_unit(client, root):
    pl, _, _ = make_pipeline(client, root)

    class Handle:
        job_id = None

        def __init__(self):
            self.checks = 0

        async def check_cancelled(self):
            self.checks += 1
            return self.checks > 2  # row check + the first unit, then cancelled before the next unit

        async def set_progress(self, **kw):
            pass

    stub = Stub()

    async def _go():
        async with async_session_factory() as db:
            await pes.run_pipeline_eval(db, pl["id"], "run-00000000aa", max_rows=5, handle=Handle())

    with mock.patch("app.services.local_chat_service.generate_reply", stub), pytest.raises(asyncio.CancelledError):
        asyncio.run(_go())
    assert len(stub.calls) == 1
    assert ps.load_run(pl["id"], "run-00000000aa")["status"] == "cancelled"


def test_heldout_upload_size_limit(client, root):
    p1, p2 = h.make_project(client), h.make_project(client)
    pl = client.post("/api/pipelines", json=definition(p1, p2)).json()
    big = [{"id": f"r{i}", "input": {"text": "x" * 50}} for i in range(5)]
    with mock.patch.object(ps, "MAX_HELDOUT_BYTES", 200):
        resp = client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": big})
        assert resp.status_code == 413
        payload = "\n".join(json.dumps(r) for r in big).encode()
        resp = client.put(f"/api/pipelines/{pl['id']}/heldout",
                          files={"file": ("h.jsonl", payload, "application/jsonl")})
        assert resp.status_code == 413
        assert client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": big[:1]}).status_code == 200


def test_fences_closed_on_the_last_line():
    from app.services.eval_task_handler_service import StructuredExtractionHandler, strip_code_fences

    assert strip_code_fences('```json\n{"a":1}```') == '{"a":1}'
    assert strip_code_fences('```json\n[1,2]```') == "[1,2]"
    assert gateway_service._validate_reply({"type": "object"}, '```json\n{"a": 1}```')["valid"] is True
    assert gateway_service._validate_reply({"type": "array"}, '```json\n[1, 2]```')["valid"] is True
    assert StructuredExtractionHandler()._parse_json_safely('```json\n{"a": 1}```') == {"a": 1}
    assert pes.parse_output('```json\n[1,2]```', "json") == (True, [1, 2])
    assert pes.parse_output('Here: [{"a": 1}] -- done', "json") == (True, [{"a": 1}])


def test_validation_placeholders_slugs_mismatch_and_stale_runs(client, root):
    p1, p2 = h.make_project(client), h.make_project(client)
    bad = definition(p1, p2)
    bad["stages"][1]["prompt"]["template"] = "Rewrite: {name} as {style}"
    errs = client.post("/api/pipelines", json=bad).json()["errors"]
    assert any(e["field"] == "stages[1].prompt.template" and "{style}" in e["message"] for e in errs)
    assert client.get("/api/pipelines/pl-0000000a%0A").status_code == 404  # fullmatch: no trailing newline

    # A project named like another's numeric alias ("p<id>") does not hijack it: stages resolve by slug.
    pl, (p1, p2), (run1, _) = make_pipeline(client, root)
    h.make_project(client, name=f"p{p1}")
    run = run_sync(client, pl, Stub()).json()
    assert run["stages"][0]["project_id"] == p1 and run["stages"][0]["run_id"] == run1

    # Rename a stage after the upload: rows whose gold names the old stage are left out and reported.
    client.put(f"/api/pipelines/{pl['id']}/heldout", json={"rows": [
        {"id": "k1", "input": {"text": "r2"}, "gold": {"extract": {"items": ITEMS_AB}}}, ROWS[0], ROWS[1]]})
    renamed = definition(p1, p2)
    renamed["stages"][1]["name"] = "polish"
    assert client.put(f"/api/pipelines/{pl['id']}", json=renamed).status_code == 200
    run = run_sync(client, pl, Stub()).json()
    assert run["gold_mismatch"] == {"unknown_stage_in_gold": ["r1", "r2"]}
    assert run["rows"] == 1

    # A "running" run whose Job is gone is listed as stale; summaries come from the sidecars.
    ps.save_run(pl["id"], {"run_id": "run-00000000ff", "pipeline_id": pl["id"], "status": "running",
                           "started_at": "2999-01-01T00:00:00+00:00", "job_id": 99999999})
    listed = client.get(f"/api/pipelines/{pl['id']}/runs").json()["runs"]
    assert listed[0]["run_id"] == "run-00000000ff" and listed[0]["status"] == "stale"
    assert client.get(f"/api/pipelines/{pl['id']}/runs/run-00000000ff").json()["status"] == "stale"
    assert ps.summary_path(pl["id"], run["run_id"]).is_file()
    summary = next(p for p in client.get("/api/pipelines").json()["pipelines"] if p["id"] == pl["id"])
    assert summary["last_run"]["status"] == "stale" and summary["heldout_rows"] == 3

    # A member project deleted later is flagged in the list.
    assert client.delete(f"/api/projects/{p2}").status_code == 204
    summary = next(p for p in client.get("/api/pipelines").json()["pipelines"] if p["id"] == pl["id"])
    assert [s["project_missing"] for s in summary["stages"]] == [False, True]


def test_recipe_builder_crash_is_a_prompt_error(client, root):
    pl, _, _ = make_pipeline(client, root)
    pipeline = ps.load_pipeline(pl["id"])
    pipeline["stages"][1]["prompt"] = {"mode": "recipe"}
    ps.atomic_write_json(ps.pipeline_dir(pl["id"]) / "pipeline.json", pipeline)
    with mock.patch("app.services.gateway_tools_service.build_recipe_prompt", side_effect=KeyError("boom")), \
            mock.patch("app.services.gateway_service._adapter_config", mock.AsyncMock(return_value={})):
        run = run_sync(client, pl, Stub()).json()
    assert run["errors_by_reason"]["prompt_error"] >= 1
