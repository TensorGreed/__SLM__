"""Phase 85 — extension scaffold generator (priority.md P38, Wave H).

Verifies that :func:`app.services.scaffold_service.generate_extension_scaffold`
emits contract-valid starter modules for every plugin kind, plus the
HTTP surface at ``POST /api/extensions/scaffold``.

What's covered:

1. **Round trip**: for each plugin kind the generated module imports
   cleanly and passes
   :func:`app.services.plugin_contracts.validate_plugin_module` with
   ``ok=True`` (no FAIL checks).

2. **File layout**: each scaffold produces exactly ``<module>.py``,
   ``test_<module>.py``, and ``README.md``. The test stub references
   the module file by name and the README documents the right
   settings key.

3. **Plugin id slug**: dashes and uppercase characters are normalised
   into a Python-safe module basename; leading digits get an
   underscore prefix.

4. **Validation**: unknown kind → ``ValueError("unknown_plugin_kind:...")``;
   empty plugin id → ``ValueError("scaffold_plugin_id_required:...")``.

5. **HTTP**: ``POST /api/extensions/scaffold`` returns the same payload
   as the service. Bad input returns 400 with the stable reason code
   in ``detail``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase85_scaffolds.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase85_scaffolds_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.plugin_contracts import (
    KNOWN_PLUGIN_KINDS,
    PLUGIN_CONTRACT_VERSIONS,
    validate_plugin_module,
)
from app.services.scaffold_service import generate_extension_scaffold


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


def _exec_module(module_basename: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_basename, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_basename] = module
    spec.loader.exec_module(module)
    return module


class Phase85ScaffoldTests(unittest.TestCase):
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
        cls._loaded_module_names: list[str] = []

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        for name in cls._loaded_module_names:
            sys.modules.pop(name, None)
        _cleanup_artifacts()

    def _tmp_export_dir(self, label: str) -> Path:
        out = TEST_DATA_DIR / f"{label}_{uuid.uuid4().hex[:8]}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ------------------------------------------------------------------
    # 1. Round trip: scaffold → import → validate, every kind
    # ------------------------------------------------------------------

    def test_every_kind_round_trips(self):
        for kind in KNOWN_PLUGIN_KINDS:
            with self.subTest(kind=kind):
                out_dir = self._tmp_export_dir(f"rt_{kind}")
                payload = generate_extension_scaffold(
                    kind=kind,
                    plugin_id=f"phase85-{kind.replace('_', '-')}-rt",
                    description=f"Round-trip scaffold for {kind}",
                    export_dir=out_dir,
                )
                self.assertEqual(payload["kind"], kind)
                self.assertEqual(
                    payload["contract_version"], PLUGIN_CONTRACT_VERSIONS[kind]
                )
                self.assertEqual(payload["output_dir"], str(out_dir.resolve()))

                module_basename = payload["module_basename"]
                module_path = Path(payload["output_dir"]) / f"{module_basename}.py"
                self.assertTrue(module_path.is_file(), module_path)
                unique_name = f"phase85_{kind}_{uuid.uuid4().hex[:8]}_{module_basename}"
                module = _exec_module(unique_name, module_path)
                self._loaded_module_names.append(unique_name)

                report = validate_plugin_module(module, kind)
                self.assertTrue(report.ok, [check.to_dict() for check in report.checks])

    # ------------------------------------------------------------------
    # 2. File layout
    # ------------------------------------------------------------------

    def test_scaffold_writes_three_files(self):
        out_dir = self._tmp_export_dir("layout")
        payload = generate_extension_scaffold(
            kind="data_adapter",
            plugin_id="phase85-layout-test",
            export_dir=out_dir,
        )
        module_basename = payload["module_basename"]
        expected_paths = {
            f"{module_basename}.py",
            f"test_{module_basename}.py",
            "README.md",
        }
        self.assertEqual(set(payload["files"]), expected_paths)
        for name in expected_paths:
            self.assertTrue(
                (Path(payload["output_dir"]) / name).is_file(),
                f"expected scaffold file: {name}",
            )

    def test_readme_mentions_settings_key_for_loader_backed_kinds(self):
        out_dir = self._tmp_export_dir("readme_loader")
        payload = generate_extension_scaffold(
            kind="training_runtime",
            plugin_id="phase85-readme-loader",
            export_dir=out_dir,
        )
        readme = payload["files"]["README.md"]
        self.assertIn("TRAINING_RUNTIME_PLUGIN_MODULES", readme)

    def test_readme_signals_pending_loader_for_declarative_kinds(self):
        out_dir = self._tmp_export_dir("readme_decl")
        payload = generate_extension_scaffold(
            kind="domain_pack",
            plugin_id="phase85-readme-decl",
            export_dir=out_dir,
        )
        readme = payload["files"]["README.md"]
        self.assertIn("Module loader for this plugin kind", readme)

    def test_test_stub_references_module_basename(self):
        out_dir = self._tmp_export_dir("test_stub")
        payload = generate_extension_scaffold(
            kind="eval_pack",
            plugin_id="phase85-test-stub",
            export_dir=out_dir,
        )
        stub = payload["files"][f"test_{payload['module_basename']}.py"]
        self.assertIn(payload["module_basename"], stub)
        self.assertIn("register_evaluation_packs", stub)
        self.assertIn("CONTRACT_VERSION", stub)

    # ------------------------------------------------------------------
    # 3. Plugin id normalisation
    # ------------------------------------------------------------------

    def test_dashes_and_caps_normalise_to_module_basename(self):
        out_dir = self._tmp_export_dir("slug")
        payload = generate_extension_scaffold(
            kind="data_adapter",
            plugin_id="Phase85--Weird_Mix",
            export_dir=out_dir,
        )
        # Dashes/uppercase collapse to a python-safe ident.
        self.assertEqual(payload["module_basename"], "phase85_weird_mix")
        self.assertEqual(payload["plugin_id"], "phase85-weird_mix")

    def test_leading_digit_gets_underscore_prefix(self):
        out_dir = self._tmp_export_dir("leading_digit")
        payload = generate_extension_scaffold(
            kind="data_adapter",
            plugin_id="2024-feature-adapter",
            export_dir=out_dir,
        )
        self.assertTrue(payload["module_basename"].startswith("_"))
        self.assertIn("2024_feature_adapter", payload["module_basename"])

    # ------------------------------------------------------------------
    # 4. Validation
    # ------------------------------------------------------------------

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError) as cm:
            generate_extension_scaffold(
                kind="not_a_kind",
                plugin_id="phase85-x",
                write=False,
            )
        self.assertIn("unknown_plugin_kind", str(cm.exception))

    def test_empty_plugin_id_raises(self):
        with self.assertRaises(ValueError) as cm:
            generate_extension_scaffold(
                kind="data_adapter",
                plugin_id="   ",
                write=False,
            )
        self.assertIn("scaffold_plugin_id_required", str(cm.exception))

    def test_plugin_id_that_normalises_to_empty_raises(self):
        with self.assertRaises(ValueError) as cm:
            generate_extension_scaffold(
                kind="data_adapter",
                plugin_id="@@@",
                write=False,
            )
        self.assertIn("scaffold_plugin_id_invalid", str(cm.exception))

    def test_write_false_returns_files_without_disk_writes(self):
        payload = generate_extension_scaffold(
            kind="domain_pack",
            plugin_id="phase85-no-write",
            write=False,
        )
        self.assertEqual(payload["written_files"], [])
        self.assertFalse(Path(payload["output_dir"]).exists())

    # ------------------------------------------------------------------
    # 5. HTTP API
    # ------------------------------------------------------------------

    def test_post_scaffold_route_returns_files(self):
        out_dir = self._tmp_export_dir("http_scaffold")
        resp = self.client.post(
            "/api/extensions/scaffold",
            json={
                "kind": "data_adapter",
                "plugin_id": "phase85-http",
                "description": "HTTP scaffold",
                "export_dir": str(out_dir),
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["kind"], "data_adapter")
        self.assertEqual(body["plugin_id"], "phase85-http")
        self.assertIn(f"{body['module_basename']}.py", body["files"])
        self.assertGreater(len(body["written_files"]), 0)
        # And the file actually lives on disk.
        self.assertTrue(
            (out_dir / f"{body['module_basename']}.py").is_file()
        )

    def test_post_scaffold_route_unknown_kind_returns_400(self):
        resp = self.client.post(
            "/api/extensions/scaffold",
            json={"kind": "bogus", "plugin_id": "phase85-bad"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("unknown_plugin_kind", resp.json()["detail"])

    def test_post_scaffold_route_empty_plugin_id_returns_400(self):
        # Pydantic Field(min_length=1) rejects empty strings before the
        # service sees them, surfacing as a 422 validation error; the
        # service-level error path is exercised by passing a slug-only
        # string of separators that normalises empty.
        resp = self.client.post(
            "/api/extensions/scaffold",
            json={"kind": "data_adapter", "plugin_id": "@@@"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("scaffold_plugin_id_invalid", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
