"""Phase 84 — plugin contract validators (priority.md P37, Wave H).

Verifies the contract suite in
:mod:`app.services.plugin_contracts` plus the orchestration layer in
:mod:`app.services.plugin_contract_service` and the
``/api/extensions`` HTTP surface.

What's covered:

1. **Per-kind validators** accept a well-formed in-memory module and
   reject modules that are missing the hook, have a wrong signature,
   declare an incompatible contract version, or carry a malformed
   payload.

2. **list_extensions** returns one entry per kind with the runtime
   metadata (recognised exports, contract version, settings key,
   whether reload is supported).

3. **validate_extension** surfaces import failures as a structured
   report (rather than letting the exception propagate) and runs the
   schema-compliance checks when import succeeds.

4. **reload_extensions** returns ``status="not_supported"`` for kinds
   without a live loader and dispatches to the data-adapter /
   training-runtime reloaders otherwise.

5. **HTTP API** — ``GET /api/extensions``, ``POST /api/extensions/validate``,
   ``POST /api/extensions/reload`` agree with the service layer.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase84_plugin_contracts.db"
TEST_DATA_DIR = (
    Path(__file__).resolve().parent / "phase84_plugin_contracts_data"
)

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.plugin_contract_service import (
    list_extensions,
    reload_extensions,
    validate_extension,
)
from app.services.plugin_contracts import (
    KNOWN_PLUGIN_KINDS,
    PLUGIN_CONTRACT_VERSIONS,
    validate_plugin_module,
)


_MODULE_PREFIX = "phase84_plugin_modules"


def _make_module(name: str, attrs: dict[str, object]) -> types.ModuleType:
    fq_name = f"{_MODULE_PREFIX}.{name}_{uuid.uuid4().hex[:8]}"
    module = types.ModuleType(fq_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[fq_name] = module
    return module


def _cleanup_artifacts() -> None:
    if TEST_DATA_DIR.exists():
        for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
            if path.is_file():
                try:
                    path.unlink()
                except PermissionError:
                    pass
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    for suffix in ("", "-shm", "-wal"):
        path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass


class Phase84PluginContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        _cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        for name in [n for n in sys.modules if n.startswith(f"{_MODULE_PREFIX}.")]:
            sys.modules.pop(name, None)
        _cleanup_artifacts()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checks_by_name(report: dict) -> dict[str, dict]:
        return {check["name"]: check for check in report["checks"]}

    @staticmethod
    def _ok_named(report: dict, name: str) -> bool:
        return Phase84PluginContractTests._checks_by_name(report).get(name, {}).get(
            "ok", False
        )

    # ------------------------------------------------------------------
    # 1. Data adapter contract
    # ------------------------------------------------------------------

    def test_data_adapter_register_signature_ok(self):
        def register_data_adapters(register):
            register("phase84-adapter-ok", lambda record, config: record)

        module = _make_module("adapter_ok", {
            "register_data_adapters": register_data_adapters,
            "CONTRACT_VERSION": PLUGIN_CONTRACT_VERSIONS["data_adapter"],
        })
        report = validate_plugin_module(module, "data_adapter").to_dict()
        self.assertTrue(report["ok"], report)
        self.assertTrue(self._ok_named(report, "module_interface"))
        self.assertTrue(self._ok_named(report, "schema_compliance"))
        self.assertTrue(self._ok_named(report, "version_metadata"))

    def test_data_adapter_dict_form_collects_ids(self):
        module = _make_module("adapter_dict", {
            "DATA_ADAPTERS": {
                "phase84-dict-a": {"map_row": lambda r, c: r, "description": "a"},
                "phase84-dict-b": lambda r, c: r,
            },
        })
        report = validate_plugin_module(module, "data_adapter").to_dict()
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            sorted(report["declared_ids"]),
            ["phase84-dict-a", "phase84-dict-b"],
        )

    def test_data_adapter_missing_hook_fails(self):
        module = _make_module("adapter_no_hook", {"UNRELATED": 1})
        report = validate_plugin_module(module, "data_adapter").to_dict()
        self.assertFalse(report["ok"])
        checks = self._checks_by_name(report)
        self.assertFalse(checks["module_interface"]["ok"])
        self.assertIn("register_data_adapters", checks["module_interface"]["message"])

    def test_data_adapter_wrong_signature_fails(self):
        def register_data_adapters(register, extra):
            register("bad", lambda r, c: r)

        module = _make_module("adapter_wrong_sig", {
            "register_data_adapters": register_data_adapters,
        })
        report = validate_plugin_module(module, "data_adapter").to_dict()
        checks = self._checks_by_name(report)
        self.assertFalse(report["ok"])
        self.assertFalse(checks["schema_compliance"]["ok"])
        self.assertIn("one positional parameter", checks["schema_compliance"]["message"])

    def test_data_adapter_bad_payload_fails(self):
        module = _make_module("adapter_bad_payload", {
            "DATA_ADAPTERS": {"id1": "not-a-dict-or-callable"},
        })
        report = validate_plugin_module(module, "data_adapter").to_dict()
        self.assertFalse(report["ok"])
        checks = self._checks_by_name(report)
        self.assertFalse(checks["schema_compliance"]["ok"])

    def test_data_adapter_version_mismatch_fails(self):
        def register_data_adapters(register):
            register("ok", lambda r, c: r)

        module = _make_module("adapter_wrong_ver", {
            "register_data_adapters": register_data_adapters,
            "CONTRACT_VERSION": "slm.data_adapter/v999",
        })
        report = validate_plugin_module(module, "data_adapter").to_dict()
        self.assertFalse(report["ok"])
        checks = self._checks_by_name(report)
        self.assertFalse(checks["version_metadata"]["ok"])
        self.assertIn("slm.data_adapter/v3", checks["version_metadata"]["message"])

    # ------------------------------------------------------------------
    # 2. Training runtime contract
    # ------------------------------------------------------------------

    def test_training_runtime_zero_arg_register_ok(self):
        def register_training_runtime_plugins():
            return None

        module = _make_module("rt_zero", {
            "register_training_runtime_plugins": register_training_runtime_plugins,
        })
        report = validate_plugin_module(module, "training_runtime").to_dict()
        self.assertTrue(report["ok"], report)

    def test_training_runtime_one_arg_register_ok(self):
        def register_training_runtime_plugins(register):
            return None

        module = _make_module("rt_one", {
            "register_training_runtime_plugins": register_training_runtime_plugins,
        })
        report = validate_plugin_module(module, "training_runtime").to_dict()
        self.assertTrue(report["ok"], report)

    def test_training_runtime_two_arg_register_fails(self):
        def register_training_runtime_plugins(a, b):
            return None

        module = _make_module("rt_two", {
            "register_training_runtime_plugins": register_training_runtime_plugins,
        })
        report = validate_plugin_module(module, "training_runtime").to_dict()
        self.assertFalse(report["ok"])
        checks = self._checks_by_name(report)
        self.assertFalse(checks["schema_compliance"]["ok"])

    def test_training_runtime_missing_hook_fails(self):
        module = _make_module("rt_missing", {})
        report = validate_plugin_module(module, "training_runtime").to_dict()
        self.assertFalse(report["ok"])
        checks = self._checks_by_name(report)
        self.assertFalse(checks["module_interface"]["ok"])

    # ------------------------------------------------------------------
    # 3. Domain pack + eval pack contracts (declarative)
    # ------------------------------------------------------------------

    def test_domain_pack_constant_ok(self):
        module = _make_module("dp_const", {
            "DOMAIN_PACKS": [
                {"pack_id": "phase84-pack", "display_name": "Phase 84 Pack"},
            ],
        })
        report = validate_plugin_module(module, "domain_pack").to_dict()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["declared_ids"], ["phase84-pack"])
        # Domain pack reload check is informational; safe_reload should be
        # marked ok=True with a "not implemented yet" message.
        checks = self._checks_by_name(report)
        self.assertTrue(checks["safe_reload"]["ok"])
        self.assertIn("P38", checks["safe_reload"]["message"])

    def test_domain_pack_missing_required_field_fails(self):
        module = _make_module("dp_bad", {
            "DOMAIN_PACKS": [{"pack_id": "phase84-bad"}],  # no display_name
        })
        report = validate_plugin_module(module, "domain_pack").to_dict()
        self.assertFalse(report["ok"])
        checks = self._checks_by_name(report)
        self.assertFalse(checks["schema_compliance"]["ok"])
        self.assertIn("display_name", checks["schema_compliance"]["message"])

    def test_eval_pack_constant_ok(self):
        module = _make_module("ep_const", {
            "EVALUATION_PACKS": [
                {
                    "pack_id": "phase84-eval",
                    "display_name": "Phase 84 Eval",
                    "task_specs": [{"task_profile": "instruction_sft"}],
                },
            ],
        })
        report = validate_plugin_module(module, "eval_pack").to_dict()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["declared_ids"], ["phase84-eval"])

    def test_eval_pack_bad_task_specs_fails(self):
        module = _make_module("ep_bad", {
            "EVALUATION_PACKS": [
                {
                    "pack_id": "phase84-eval-bad",
                    "display_name": "Bad eval",
                    "task_specs": "not-a-list",
                },
            ],
        })
        report = validate_plugin_module(module, "eval_pack").to_dict()
        self.assertFalse(report["ok"])

    # ------------------------------------------------------------------
    # 4. Service surface — list / validate / reload
    # ------------------------------------------------------------------

    def test_list_extensions_covers_every_kind(self):
        payload = list_extensions()
        kinds = [entry["kind"] for entry in payload["kinds"]]
        self.assertEqual(set(kinds), set(KNOWN_PLUGIN_KINDS))
        by_kind = {entry["kind"]: entry for entry in payload["kinds"]}
        self.assertTrue(by_kind["data_adapter"]["supports_safe_reload"])
        self.assertFalse(by_kind["domain_pack"]["supports_safe_reload"])
        self.assertEqual(
            by_kind["data_adapter"]["contract_version"],
            PLUGIN_CONTRACT_VERSIONS["data_adapter"],
        )

    def test_validate_extension_against_existing_module(self):
        # Reuse the in-process module API itself as a valid runtime
        # module: it exports register_training_runtime_plugin (the SDK
        # entrypoint), not register_training_runtime_plugins — so the
        # validator should fail with a clear module_interface message.
        report = validate_extension(
            kind="training_runtime",
            module_path="app.services.training_runtime_service",
        )
        self.assertFalse(report["ok"])
        checks = {check["name"]: check for check in report["checks"]}
        self.assertFalse(checks["module_interface"]["ok"])

    def test_validate_extension_import_error_is_structured(self):
        report = validate_extension(
            kind="data_adapter",
            module_path="phase84_definitely_no_such_module",
        )
        self.assertFalse(report["ok"])
        self.assertIsNotNone(report["import_error"])
        self.assertEqual(report["checks"][0]["name"], "module_importable")
        self.assertFalse(report["checks"][0]["ok"])

    def test_validate_extension_unknown_kind_raises(self):
        with self.assertRaises(ValueError) as cm:
            validate_extension(
                kind="not_a_kind",
                module_path="phase84_missing",
            )
        self.assertIn("unknown_plugin_kind", str(cm.exception))

    def test_reload_extensions_dispatches_per_kind(self):
        payload = reload_extensions()
        statuses = {result["kind"]: result["status"] for result in payload["results"]}
        self.assertEqual(statuses["data_adapter"], "ok")
        self.assertEqual(statuses["training_runtime"], "ok")
        self.assertEqual(statuses["domain_pack"], "not_supported")
        self.assertEqual(statuses["eval_pack"], "not_supported")

    # ------------------------------------------------------------------
    # 5. HTTP API
    # ------------------------------------------------------------------

    def test_get_extensions_route(self):
        resp = self.client.get("/api/extensions")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("kinds", body)
        self.assertEqual(
            sorted(entry["kind"] for entry in body["kinds"]),
            sorted(KNOWN_PLUGIN_KINDS),
        )
        self.assertEqual(set(body["known_kinds"]), set(KNOWN_PLUGIN_KINDS))

    def test_post_validate_route_with_real_module(self):
        # Materialize a temporary module under a regular import path by
        # writing to a tmp directory and pointing sys.path at it. Tests
        # the loader end-to-end (importlib.import_module path).
        tmp_root = TEST_DATA_DIR / "extmod"
        tmp_root.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(tmp_root))
        module_name = f"phase84_inline_ok_{uuid.uuid4().hex[:8]}"
        module_path = tmp_root / f"{module_name}.py"
        module_path.write_text(
            (
                "CONTRACT_VERSION = 'slm.data_adapter/v3'\n"
                "def register_data_adapters(register):\n"
                "    register('phase84-http-adapter', lambda r, c: r)\n"
            ),
            encoding="utf-8",
        )
        try:
            resp = self.client.post(
                "/api/extensions/validate",
                json={"kind": "data_adapter", "module": module_name},
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertTrue(body["ok"], body)
            self.assertIsNone(body["import_error"])
        finally:
            sys.path.remove(str(tmp_root))
            sys.modules.pop(module_name, None)
            importlib.invalidate_caches()

    def test_post_validate_route_unknown_kind(self):
        resp = self.client.post(
            "/api/extensions/validate",
            json={"kind": "bogus", "module": "irrelevant"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("unknown_plugin_kind", resp.json()["detail"])

    def test_post_reload_route_returns_per_kind_results(self):
        resp = self.client.post("/api/extensions/reload", json={})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        kinds = {entry["kind"] for entry in body["results"]}
        self.assertEqual(kinds, set(KNOWN_PLUGIN_KINDS))

    def test_post_reload_single_kind(self):
        resp = self.client.post(
            "/api/extensions/reload",
            json={"kind": "data_adapter"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(body["results"]), 1)
        self.assertEqual(body["results"][0]["kind"], "data_adapter")
        self.assertEqual(body["results"][0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
