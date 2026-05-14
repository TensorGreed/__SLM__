"""Phase H — plugin mappers + optional LLM-assist.

Pins:
- ``load_dataset_mapper_plugins`` registers extra mappers from
  module-level ``DATASET_MAPPERS`` dicts *and* from a
  ``register_dataset_mappers(register)`` hook; failures in a single
  module don't crash the rest of the load.
- A plugin-registered mapper is resolvable via the regular registry
  and the wizard / CLI ``--auto`` path treats it identically to a
  built-in mapper.
- LLM-assist returns ``None`` when disabled by setting, when the
  teacher URL is unset, or when the teacher errors out.
- When enabled + the teacher returns a JSON proposal naming a
  registered mapper, the proposal joins the ranked hypothesis list
  and gets flagged with the ``proposal-source: llm-assist`` warning.
- LLM proposal naming a non-existent mapper is rejected (no
  hallucinated mapper escapes the registry check).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.dataset_import import (  # noqa: E402
    list_registered_mappers,
    resolve_mapper,
)
from app.services.dataset_import.llm_assist import (  # noqa: E402
    _coerce_response,
    _normalize_proposal,
    llm_assisted_proposal,
)
from app.services.dataset_import.plugin_loader import (  # noqa: E402
    load_dataset_mapper_plugins,
    load_dataset_mapper_plugins_from_settings,
)
from app.services.dataset_import.protocols import (  # noqa: E402
    ImportContext,
    TransformedRow,
)
from app.services.dataset_import.service import introspect_locator  # noqa: E402


# ── Plugin loader ────────────────────────────────────────────────────


def _install_fake_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _make_register_hook_module(name: str) -> types.ModuleType:
    """Module exporting ``register_dataset_mappers(register)`` that
    adds a single ``echo_uppercase`` mapper."""

    module = types.ModuleType(name)

    class _EchoMapper:
        mapper_id = "echo_uppercase"

        def declared_target(self) -> str:
            return "language_modeling"

        def transform(self, rows, field_map, *, ctx: ImportContext):
            for row in rows:
                text = str(row.get("text") or "").strip().upper()
                yield TransformedRow(payload={"text": text})

    def register_dataset_mappers(register):
        register("echo_uppercase", _EchoMapper)

    module.register_dataset_mappers = register_dataset_mappers
    return module


def _make_data_module(name: str) -> types.ModuleType:
    """Module exporting a ``DATASET_MAPPERS`` dict — the declarative form."""

    module = types.ModuleType(name)

    class _PrefixMapper:
        mapper_id = "prefix_meow"

        def declared_target(self) -> str:
            return "language_modeling"

        def transform(self, rows, field_map, *, ctx: ImportContext):
            for row in rows:
                yield TransformedRow(
                    payload={"text": "meow: " + str(row.get("text") or "")}
                )

    module.DATASET_MAPPERS = {"prefix_meow": _PrefixMapper}
    return module


def _make_broken_module(name: str) -> types.ModuleType:
    """Module that raises during the registration hook."""

    module = types.ModuleType(name)

    def register_dataset_mappers(register):
        raise RuntimeError("boom from inside the plugin")

    module.register_dataset_mappers = register_dataset_mappers
    return module


class PluginLoaderTests(unittest.TestCase):
    def setUp(self):
        # Snapshot the global registry so each test sees an isolated
        # plugin-set added on top of the built-ins.
        from app.services.dataset_import.registry import _MAPPERS

        self._registry_snapshot = dict(_MAPPERS)
        # Same for the plugin-loader's loaded-modules tracker.
        from app.services.dataset_import import plugin_loader as pl

        self._loaded_snapshot = set(pl._LOADED_PLUGIN_MODULES)
        self._error_snapshot = dict(pl._PLUGIN_ERRORS)

    def tearDown(self):
        from app.services.dataset_import.registry import _MAPPERS
        from app.services.dataset_import import plugin_loader as pl

        _MAPPERS.clear()
        _MAPPERS.update(self._registry_snapshot)
        pl._LOADED_PLUGIN_MODULES.clear()
        pl._LOADED_PLUGIN_MODULES.update(self._loaded_snapshot)
        pl._PLUGIN_ERRORS.clear()
        pl._PLUGIN_ERRORS.update(self._error_snapshot)

    def test_register_dataset_mappers_hook_registers_mapper(self):
        module_name = "_ph_test_plugin_hook"
        _install_fake_module(module_name, _make_register_hook_module(module_name))
        try:
            result = load_dataset_mapper_plugins([module_name])
            self.assertEqual(result["loaded_modules"], [module_name])
            self.assertEqual(result["failed_modules"], {})
            self.assertGreaterEqual(result["registered_mappers"], 1)
            self.assertIn("echo_uppercase", list_registered_mappers())
            mapper = resolve_mapper("echo_uppercase")
            self.assertEqual(mapper.declared_target(), "language_modeling")
        finally:
            sys.modules.pop(module_name, None)

    def test_dataset_mappers_dict_form_registers_mapper(self):
        module_name = "_ph_test_plugin_data"
        _install_fake_module(module_name, _make_data_module(module_name))
        try:
            result = load_dataset_mapper_plugins([module_name])
            self.assertEqual(result["loaded_modules"], [module_name])
            self.assertIn("prefix_meow", list_registered_mappers())
        finally:
            sys.modules.pop(module_name, None)

    def test_broken_module_doesnt_block_the_rest(self):
        broken_name = "_ph_test_plugin_broken"
        ok_name = "_ph_test_plugin_ok"
        _install_fake_module(broken_name, _make_broken_module(broken_name))
        _install_fake_module(ok_name, _make_register_hook_module(ok_name))
        try:
            result = load_dataset_mapper_plugins([broken_name, ok_name])
            self.assertIn(ok_name, result["loaded_modules"])
            self.assertIn(broken_name, result["failed_modules"])
            self.assertIn("boom from inside the plugin", result["failed_modules"][broken_name])
            # The OK mapper still landed.
            self.assertIn("echo_uppercase", list_registered_mappers())
        finally:
            sys.modules.pop(broken_name, None)
            sys.modules.pop(ok_name, None)

    def test_no_export_module_rejected_with_clear_message(self):
        empty_name = "_ph_test_plugin_empty"
        _install_fake_module(empty_name, types.ModuleType(empty_name))
        try:
            result = load_dataset_mapper_plugins([empty_name])
            self.assertIn(empty_name, result["failed_modules"])
            self.assertIn(
                "register_dataset_mappers",
                result["failed_modules"][empty_name],
            )
        finally:
            sys.modules.pop(empty_name, None)

    def test_settings_loader_returns_empty_diagnostic_when_unset(self):
        with mock.patch("app.config.settings.DATASET_MAPPER_PLUGIN_MODULES", []):
            result = load_dataset_mapper_plugins_from_settings()
        self.assertEqual(result["status"], "no_plugin_modules_configured")
        self.assertEqual(result["requested_modules"], [])

    def test_settings_loader_routes_to_real_loader(self):
        module_name = "_ph_test_plugin_settings"
        _install_fake_module(module_name, _make_register_hook_module(module_name))
        try:
            with mock.patch(
                "app.config.settings.DATASET_MAPPER_PLUGIN_MODULES",
                [module_name],
            ):
                result = load_dataset_mapper_plugins_from_settings()
            self.assertIn(module_name, result["loaded_modules"])
            self.assertIn("echo_uppercase", list_registered_mappers())
        finally:
            sys.modules.pop(module_name, None)


# ── LLM-assist ───────────────────────────────────────────────────────


def _write_jsonl(rows: list[dict]) -> str:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    try:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    finally:
        fh.close()
    return fh.name


class LLMAssistResponseCoercionTests(unittest.TestCase):
    def test_raw_json_object_parses(self):
        result = _coerce_response('{"mapper_id": "x", "confidence": 0.5}')
        self.assertEqual(result, {"mapper_id": "x", "confidence": 0.5})

    def test_strips_markdown_code_fences(self):
        raw = '```json\n{"mapper_id": "x", "confidence": 0.5}\n```'
        result = _coerce_response(raw)
        self.assertEqual(result, {"mapper_id": "x", "confidence": 0.5})

    def test_extracts_largest_object_from_prose(self):
        raw = (
            "Sure, here is my suggestion:\n"
            '{"mapper_id": "x", "confidence": 0.5}\n'
            "Hope that helps."
        )
        result = _coerce_response(raw)
        self.assertEqual(result, {"mapper_id": "x", "confidence": 0.5})

    def test_empty_returns_none(self):
        self.assertIsNone(_coerce_response(""))

    def test_array_payload_rejected(self):
        # Top-level array is not a valid mapping proposal.
        self.assertIsNone(_coerce_response("[]"))


class LLMAssistNormalizeProposalTests(unittest.TestCase):
    def test_unknown_mapper_rejected(self):
        self.assertIsNone(
            _normalize_proposal(
                {"mapper_id": "no_such_mapper", "confidence": 0.9}
            )
        )

    def test_known_mapper_accepted_and_target_profile_resolved(self):
        proposal = _normalize_proposal(
            {
                "mapper_id": "label_to_classification",
                "field_map": {"text_field": "review", "label_field": "tag"},
                "confidence": 0.7,
                "rationale": "small label set + free-text body",
            }
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.mapper_id, "label_to_classification")
        # target_task_profile is resolved from the mapper, not trusted
        # from the LLM payload.
        self.assertEqual(proposal.target_task_profile, "classification")
        self.assertEqual(proposal.confidence, 0.7)
        # Provenance is recorded so callers can highlight LLM proposals.
        self.assertIn("proposal-source: llm-assist", proposal.warnings)
        self.assertIn("[llm-assist]", proposal.rationale)

    def test_confidence_clamped_to_zero_one(self):
        proposal = _normalize_proposal(
            {
                "mapper_id": "text_only",
                "field_map": {"text_field": "body"},
                "confidence": 5.0,
            }
        )
        self.assertEqual(proposal.confidence, 1.0)


class LLMAssistGatesTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_setting_returns_none(self):
        with mock.patch(
            "app.config.settings.DATASET_IMPORT_LLM_ASSIST_ENABLED", False
        ):
            with mock.patch(
                "app.config.settings.TEACHER_MODEL_API_URL", "http://t"
            ):
                result = await llm_assisted_proposal(
                    columns=["text"], sample_rows=[{"text": "hi"}]
                )
        self.assertIsNone(result)

    async def test_missing_teacher_url_returns_none(self):
        with mock.patch(
            "app.config.settings.DATASET_IMPORT_LLM_ASSIST_ENABLED", True
        ):
            with mock.patch("app.config.settings.TEACHER_MODEL_API_URL", ""):
                result = await llm_assisted_proposal(
                    columns=["text"], sample_rows=[{"text": "hi"}]
                )
        self.assertIsNone(result)

    async def test_empty_sample_rows_returns_none(self):
        with mock.patch(
            "app.config.settings.DATASET_IMPORT_LLM_ASSIST_ENABLED", True
        ):
            with mock.patch(
                "app.config.settings.TEACHER_MODEL_API_URL", "http://t"
            ):
                result = await llm_assisted_proposal(
                    columns=["text"], sample_rows=[]
                )
        self.assertIsNone(result)

    async def test_teacher_error_returns_none(self):
        async def _raises(*args, **kwargs):
            raise RuntimeError("teacher offline")

        with mock.patch(
            "app.config.settings.DATASET_IMPORT_LLM_ASSIST_ENABLED", True
        ):
            with mock.patch(
                "app.config.settings.TEACHER_MODEL_API_URL", "http://t"
            ):
                with mock.patch(
                    "app.services.synthetic_service.call_teacher_model",
                    side_effect=_raises,
                ):
                    result = await llm_assisted_proposal(
                        columns=["text"], sample_rows=[{"text": "hi"}]
                    )
        self.assertIsNone(result)


class LLMAssistIntrospectIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_proposal_joins_hypothesis_list_and_is_marked(self):
        # Three short rows — deterministic sniffer can match
        # classification weakly. The LLM proposes the same mapper
        # with high confidence; both surface in the ranked list.
        path = _write_jsonl(
            [
                {"text": "great", "label": "positive"},
                {"text": "bad", "label": "negative"},
                {"text": "neutral take", "label": "neutral"},
            ]
        )

        async def _fake_teacher(*args, **kwargs):
            return {
                "content": json.dumps(
                    {
                        "mapper_id": "label_to_classification",
                        "field_map": {
                            "text_field": "text",
                            "label_field": "label",
                        },
                        "confidence": 0.97,
                        "rationale": "free text + small label set",
                    }
                )
            }

        try:
            with mock.patch(
                "app.config.settings.DATASET_IMPORT_LLM_ASSIST_ENABLED", True
            ):
                with mock.patch(
                    "app.config.settings.TEACHER_MODEL_API_URL", "http://t"
                ):
                    with mock.patch(
                        "app.services.synthetic_service.call_teacher_model",
                        side_effect=_fake_teacher,
                    ):
                        payload = await introspect_locator(
                            f"jsonl:{path}", llm_assist=True
                        )
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertTrue(payload["llm_assist_used"])
        # The LLM proposal should be the top hypothesis (0.97 > anything
        # the deterministic sniffer produces for tiny / short inputs).
        top = payload["hypotheses"][0]
        self.assertEqual(top["mapper_id"], "label_to_classification")
        self.assertEqual(top["confidence"], 0.97)
        self.assertIn("proposal-source: llm-assist", top["warnings"])
        # The selected proposal mirrors the top hypothesis.
        self.assertEqual(payload["proposal"]["mapper_id"], "label_to_classification")
        self.assertEqual(payload["proposal"]["confidence"], 0.97)

    async def test_llm_assist_disabled_doesnt_mark_llm_assist_used(self):
        path = _write_jsonl(
            [
                {"text": "outstanding service indeed", "label": "positive"},
                {"text": "absolute disaster of a product", "label": "negative"},
                {"text": "average experience, nothing more", "label": "neutral"},
            ]
        )
        try:
            payload = await introspect_locator(
                f"jsonl:{path}", llm_assist=False
            )
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertFalse(payload["llm_assist_used"])
        # No hypothesis carries the llm-assist marker.
        markers = [
            "proposal-source: llm-assist" in (h.get("warnings") or [])
            for h in payload["hypotheses"]
        ]
        self.assertFalse(any(markers))


if __name__ == "__main__":
    unittest.main()
