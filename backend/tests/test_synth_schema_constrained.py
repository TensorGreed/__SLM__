"""Tests for USER-SUCCESS Epic 5 Phase 5b — schema-constrained synth generation.

Coverage:
  * NemoBackend.complete honors the ``response_schema`` kwarg —
    forwards it as ``response_format={"type":"json_schema",...}``
    on the chat-completion payload.
  * NemoBackend.complete omits ``response_format`` when no schema
    is passed (no regression on Phase 5a behavior).
  * OllamaBackend.complete silently accepts + ignores the schema
    kwarg (no exception, payload doesn't grow a ``response_format``).
  * TeacherModelBackend.complete silently accepts the schema kwarg.
  * class_balance_fill playbook builds a JSON Schema from the gold
    labels via ``response_schema(ctx)`` and narrows the ``label``
    enum to the resolved target_class.
  * orchestrator's ``get_response_schema`` helper picks up the
    playbook-defined schema and returns it.
  * End-to-end: class_balance_fill + NeMo-shaped fake backend
    that returns fenced JSON still produces valid-shape rows after
    the playbook's parse/validate pass — the fence-strip in
    parse_jsonl_lines is the defense-in-depth for cases where the
    NIM constraint slips (or the schema is silently ignored by a
    fallback backend).
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

import httpx  # noqa: E402

from app.services.synth_backends import (  # noqa: E402
    NemoBackend,
    OllamaBackend,
    TeacherModelBackend,
)
from app.services.synth_playbooks import (  # noqa: E402
    SynthMode,
    get_playbook,
    get_response_schema,
)


# ─────────────────────────────────────────────────────────────────────
# Test doubles — minimal httpx.AsyncClient stand-ins.
# ─────────────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_data: dict | None = None):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = json.dumps(self._json)

    def json(self):
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code < 600:  # pragma: no cover
            req = httpx.Request("POST", "http://test/v1/chat/completions")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=req,
                response=httpx.Response(self.status_code, request=req),
            )


class _FakeAsyncClient:
    def __init__(self, handler):
        self._handler = handler

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        return self._handler(url=url, **kwargs)


def _patch_async_client(handler):
    return patch.object(httpx, "AsyncClient", _FakeAsyncClient(handler))


# ─────────────────────────────────────────────────────────────────────
# NeMo: schema forwarded as response_format=json_schema
# ─────────────────────────────────────────────────────────────────────


class NemoBackendSchemaForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_forwards_response_schema_as_response_format(self):
        """When the playbook passes a JSON Schema, the NIM payload
        gains ``response_format`` in the OpenAI Structured-Outputs
        shape (``type=json_schema``, ``json_schema.schema=<schema>``,
        ``strict=True``)."""
        captured: dict = {}

        def handler(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _FakeResponse(
                status_code=200,
                json_data={"choices": [{"message": {"content": "{}"}}]},
            )

        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "label": {"type": "string", "enum": ["billing"]},
            },
            "required": ["text", "label"],
            "additionalProperties": False,
        }
        backend = NemoBackend(host="http://nim", model="meta/llama-3.1-70b-instruct")
        with _patch_async_client(handler):
            await backend.complete("Generate one row.", response_schema=schema)

        rf = captured["json"].get("response_format")
        self.assertIsNotNone(rf, "response_format missing from NIM payload")
        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["schema"], schema)
        self.assertTrue(rf["json_schema"]["strict"])

    async def test_complete_omits_response_format_when_no_schema(self):
        """No schema → no ``response_format`` key (Phase 5a behavior
        preserved). NIM should fall through to free-form decoding."""
        captured: dict = {}

        def handler(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _FakeResponse(
                status_code=200,
                json_data={"choices": [{"message": {"content": "x"}}]},
            )

        backend = NemoBackend(host="http://nim", model="m")
        with _patch_async_client(handler):
            await backend.complete("Just generate.")

        self.assertNotIn("response_format", captured["json"])


# ─────────────────────────────────────────────────────────────────────
# Ollama + Teacher: schema kwarg accepted + silently ignored
# ─────────────────────────────────────────────────────────────────────


class OllamaBackendIgnoresSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_accepts_schema_kwarg_without_error(self):
        """Ollama's OpenAI-compatible endpoint ignores
        ``response_format=json_schema`` — the backend mirrors that
        by accepting the kwarg and not threading it into the payload.
        Existing callers (and the auto-pick path) see zero change."""
        captured: dict = {}

        def handler(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _FakeResponse(
                status_code=200,
                json_data={"choices": [{"message": {"content": "ok"}}]},
            )

        backend = OllamaBackend(host="http://ollama", model="llama3.1:8b")
        with _patch_async_client(handler):
            out = await backend.complete(
                "Generate.",
                response_schema={"type": "object", "properties": {}},
            )

        self.assertEqual(out, "ok")
        # Ollama backend must NOT have leaked the schema into the
        # payload (it would just be ignored, but we want the request
        # shape to stay identical to Phase 5a).
        self.assertNotIn("response_format", captured["json"])

    async def test_teacher_backend_accepts_schema_kwarg_without_error(self):
        """The legacy teacher dispatcher (``call_teacher_model``) has
        no structured-output hook, so the backend silently ignores
        the schema arg. Just verify no TypeError on the keyword."""
        async def _fake_call_teacher_model(**kwargs):
            return {"choices": [{"message": {"content": "teacher-said"}}]}

        backend = TeacherModelBackend(model="llama3", api_url="http://teacher/v1")
        with patch(
            "app.services.synthetic_service.call_teacher_model",
            side_effect=_fake_call_teacher_model,
        ):
            out = await backend.complete(
                "Generate.",
                response_schema={"type": "object"},
            )

        self.assertEqual(out, "teacher-said")


# ─────────────────────────────────────────────────────────────────────
# Playbook → schema construction from gold labels
# ─────────────────────────────────────────────────────────────────────


class ClassBalanceFillSchemaTests(unittest.TestCase):
    def test_response_schema_uses_known_labels_enum(self):
        pb = get_playbook("classification", SynthMode.CLASS_BALANCE_FILL)
        gold = [
            {"text": "a", "label": "billing"},
            {"text": "b", "label": "billing"},
            {"text": "c", "label": "technical"},
            {"text": "d", "label": "shipping"},
        ]
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": gold,
            "target_count": 3,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        # No target pinned yet — schema enumerates ALL known labels.
        schema = pb.response_schema(ctx)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["text", "label"])
        self.assertFalse(schema["additionalProperties"])
        label_enum = schema["properties"]["label"]["enum"]
        self.assertEqual(sorted(label_enum), ["billing", "shipping", "technical"])

    def test_response_schema_narrows_label_to_pinned_target(self):
        """Once build_prompt picks the minority class, the schema
        narrows ``label`` to that single value so the decoder can't
        drift to a different class."""
        pb = get_playbook("classification", SynthMode.CLASS_BALANCE_FILL)
        gold = [{"text": f"row{i}", "label": "common"} for i in range(5)] + [
            {"text": "the rare", "label": "rare"},
        ]
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": gold,
            "target_count": 3,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        pb.build_prompt(ctx)  # stashes target_class="rare" on ctx
        self.assertEqual(ctx["target_class"], "rare")
        schema = pb.response_schema(ctx)
        self.assertEqual(schema["properties"]["label"]["enum"], ["rare"])

    def test_response_schema_falls_back_to_freeform_when_no_labels(self):
        pb = get_playbook("classification", SynthMode.CLASS_BALANCE_FILL)
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": [],
            "target_count": 3,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        schema = pb.response_schema(ctx)
        # Without known labels we can't build an enum — fall back to
        # a plain string so the decoder isn't blocked.
        self.assertNotIn("enum", schema["properties"]["label"])
        self.assertEqual(schema["properties"]["label"]["type"], "string")

    def test_get_response_schema_helper_returns_none_for_playbooks_without_schema(self):
        # POSITIVES_PARAPHRASE playbook doesn't implement response_schema —
        # the helper must return None, not error.
        pb = get_playbook("classification", SynthMode.POSITIVES_PARAPHRASE)
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": [{"text": "x", "label": "billing"}],
            "target_count": 1,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        self.assertIsNone(get_response_schema(pb, ctx))


# ─────────────────────────────────────────────────────────────────────
# End-to-end: class_balance_fill + NeMo-shaped backend + fenced JSON
# ─────────────────────────────────────────────────────────────────────


class _FakeNeMoFencedBackend:
    """Stand-in for NemoBackend that records the schema it received
    and returns markdown-fenced JSON (the failure mode we want the
    playbook parser to absorb even when schema constraints slip)."""

    name = "nemo"

    def __init__(self, response: str):
        self._response = response
        self.last_schema: dict | None = None

    @classmethod
    def is_available(cls):  # pragma: no cover
        return True

    def describe(self):
        return "nemo:test"

    async def complete(
        self,
        prompt,
        *,
        system_prompt=None,
        max_tokens=1024,
        temperature=0.7,
        response_schema=None,
    ):
        self.last_schema = response_schema
        return self._response


class ClassBalanceFillEndToEndTests(unittest.TestCase):
    def test_fenced_json_from_nemo_still_validates(self):
        """The end-to-end playbook flow with a NeMo-shaped backend:
        the playbook builds a schema, the backend records it (the
        Phase 5b contract), and even though the canned response is
        wrapped in ```json fences, parse_output + validate still
        accept the rows.

        This guards against the case where (a) the NIM ignores
        ``strict`` and emits fenced text, or (b) we ever route the
        schema-bearing flow through a non-schema-aware fallback —
        the parser is the safety net."""
        pb = get_playbook("classification", SynthMode.CLASS_BALANCE_FILL)
        gold = [{"text": f"row{i}", "label": "common"} for i in range(8)] + [
            {"text": "minority1", "label": "rare"},
        ]
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": gold,
            "target_count": 3,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }

        # Simulate a NIM that wrapped its (already-on-schema) output
        # in a ```json fence. Two valid rows + one row drifted to the
        # majority class (the validator must drop that one).
        fenced_response = (
            "```json\n"
            '{"text": "another rare example", "label": "rare"}\n'
            '{"text": "yet another rare one", "label": "rare"}\n'
            '{"text": "drifted to common", "label": "common"}\n'
            "```\n"
        )
        backend = _FakeNeMoFencedBackend(fenced_response)

        prompt = pb.build_prompt(ctx)
        schema = get_response_schema(pb, ctx)
        # Sanity-check the orchestrator-equivalent path:
        # - prompt is non-empty
        # - schema is the narrowed enum for target_class
        self.assertGreater(len(prompt), 20)
        self.assertEqual(schema["properties"]["label"]["enum"], ["rare"])

        raw = asyncio.run(
            backend.complete(prompt, response_schema=schema)
        )
        # Backend recorded the schema we passed (Phase 5b contract).
        self.assertEqual(backend.last_schema, schema)

        parsed = pb.parse_output(raw, ctx)
        # Fence-strip in parse_jsonl_lines must yield all 3 rows.
        self.assertEqual(len(parsed), 3)

        validated = pb.validate(parsed, ctx)
        # 2 rows accepted ("rare" matches target). The "common" row
        # is dropped — wrong class for class_balance_fill.
        self.assertEqual(len(validated), 2)
        for row in validated:
            self.assertEqual(row["payload"]["label"], "rare")
            self.assertGreaterEqual(row["synth_confidence"], 0.5)
            self.assertIn("class_balance_fill", row["synth_source"])
            self.assertIn("class=rare", row["synth_source"])


if __name__ == "__main__":
    unittest.main()
