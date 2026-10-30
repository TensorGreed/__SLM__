"""Opt-in "format before check" for held-out code-example scoring
(``runtime_config["remediation"]["format_before_check"]``, default off).

The predicted code is run through the language's canonical formatter (``code_check_service.format_code``)
before the code checks, so a format-only failure (``terraform fmt -check`` on ``=`` alignment) no longer
fails otherwise valid code. Covers: the stubbed formatter path (always runs), the real terraform path
(skipped without the binary), off by default, a formatter that rejects the code, a missing binary, the
settings API, the live-config overlay and the scoring fingerprint W1's Compare reads.

Pytest-only; the sandbox is stubbed except in the real-terraform test.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.project import Project
from app.services import code_check_service as svc
from app.services import code_example_scoring as ces
from app.services import sandbox_exec
from app.services.code_remediation_prompt import effective_remediation_config
from app.services.compare_row_evidence_service import _scoring, compare_provenance
from app.services.remediation_detector import remediation_scoring_fingerprint

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


ALIGNED = 'resource "example_bucket" "logs" {\n  name     = "logs"\n  location = "eu"\n}\n'
MISALIGNED = 'resource "example_bucket" "logs" {\n  name = "logs"\n  location = "eu"\n}\n'
BROKEN = 'resource "example_bucket" "logs" {\n  name = \n'


def sbx(rc: int = 0, out: str = "", err: str = "", timed_out: bool = False):
    return SimpleNamespace(returncode=rc, stdout=out, stderr=err, timed_out=timed_out, truncated=False)


def fake_terraform(argv, files=None, stdin=None, **kw):
    """``terraform fmt -check`` passes only canonical code; ``terraform fmt -`` aligns, or fails on a syntax error."""
    if "-check" in argv:
        return sbx(0) if stdin == ALIGNED else sbx(3, out="<stdin>\n")
    if "=\n" in (stdin or "").replace(" ", ""):
        return sbx(2, err="Error: Invalid expression\n")
    return sbx(0, out=ALIGNED)


def parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def prediction(code: str) -> dict:
    return {
        "prompt": "p",
        "reference": json.dumps({"code": ALIGNED, "rationale": "r", "citations": []}),
        "prediction": json.dumps({"code": code, "rationale": "r", "citations": []}),
        "code_example_row": {"task": "Add a log bucket.", "language": "terraform", "passages": []},
    }


def schema(fmt: bool | None) -> dict:
    rem = {"checks": ["terraform_fmt"], "retrieve_passages": False, "top_k": 3, "retrieval": None,
           "score_code_examples": True}
    if fmt is not None:
        rem["format_before_check"] = fmt
    return {"scoring_mode": "remediation", "remediation": rem}


class StubbedTerraform:
    """terraform_fmt allowed, the binary 'present', the sandbox faked."""

    def __enter__(self):
        self.patches = [
            mock.patch.object(settings, "CODE_CHECK_ALLOWED", "terraform_fmt"),
            mock.patch.object(shutil, "which", return_value="/usr/bin/terraform"),
            mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=fake_terraform),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in reversed(self.patches):
            p.stop()


def score(code: str, fmt: bool | None) -> tuple[dict, dict]:
    pred = prediction(code)
    metrics = ces.score_code_examples([pred], schema(fmt), parse_json)
    return pred, metrics


class FormatCodeTests(unittest.TestCase):
    def test_formatter_registry(self):
        self.assertEqual(svc.formatter_for("hcl").id, "terraform_fmt")
        self.assertEqual(svc.formatter_for("golang").id, "go_syntax")
        self.assertEqual(svc.formatter_for("rust").id, "rust_syntax")
        self.assertIsNone(svc.formatter_for("python"))
        self.assertEqual(svc.format_code("x = 1", language="python")["status"], "no_formatter")

    def test_formatted_failed_and_gated(self):
        with StubbedTerraform() as stub:
            done = svc.format_code(MISALIGNED, language="terraform")
            same = svc.format_code(ALIGNED, language="terraform")
            bad = svc.format_code(BROKEN, language="terraform")
            argv = sandbox_exec.run_sandboxed.call_args_list[0].args[0]
        self.assertIsNotNone(stub)
        self.assertEqual((done["status"], done["formatter"], done["code"]), ("formatted", "terraform_fmt", ALIGNED))
        self.assertEqual(same["status"], "unchanged")
        self.assertEqual(argv, ["terraform", "fmt", "-no-color", "-"])
        self.assertEqual((bad["status"], bad["code"]), ("failed", BROKEN))  # the original code is kept
        self.assertIn("Invalid expression", bad["messages"][0])
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", ""):
            self.assertEqual(svc.format_code(MISALIGNED, language="terraform")["status"], "not_allowed")

    def test_missing_binary_changes_nothing(self):
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "terraform_fmt"), \
                mock.patch.object(shutil, "which", return_value=None), \
                mock.patch.object(sandbox_exec, "run_sandboxed") as run:
            out = svc.format_code(MISALIGNED, language="terraform")
        run.assert_not_called()
        self.assertEqual((out["status"], out["code"]), ("unavailable", MISALIGNED))

    def test_truncated_output_is_never_used(self):
        cut = SimpleNamespace(returncode=0, stdout=ALIGNED[:20], stderr="", timed_out=False, truncated=True)
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "terraform_fmt"), \
                mock.patch.object(shutil, "which", return_value="/x"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", return_value=cut):
            out = svc.format_code(MISALIGNED, language="terraform")
        self.assertEqual((out["status"], out["code"]), ("truncated", MISALIGNED))

    def test_timeout_keeps_the_code(self):
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "terraform_fmt"), \
                mock.patch.object(shutil, "which", return_value="/x"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", return_value=sbx(timed_out=True)):
            out = svc.format_code(MISALIGNED, language="terraform")
        self.assertEqual((out["status"], out["code"]), ("timeout", MISALIGNED))


    def test_rust_fragment_is_not_formattable_not_failed(self):
        def fake_rustfmt(argv, files=None, stdin=None, **kw):
            if (stdin or "").startswith("fn __snippet() {"):
                return sbx(0, out="fn __snippet() {\n    let x = 1;\n}\n")
            return sbx(1, err="error: expected item, found keyword `let`\n")

        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "rust_syntax"), \
                mock.patch("app.services.code_checks.external.RustSyntax.available", return_value=True), \
                mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=fake_rustfmt):
            frag = svc.format_code("let x=1;", language="rust")
        self.assertEqual((frag["status"], frag["code"]), ("not_formattable", "let x=1;"))

        def broken_rustfmt(argv, files=None, stdin=None, **kw):
            return sbx(1, err="error: expected one of `;`\n")

        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "rust_syntax"), \
                mock.patch("app.services.code_checks.external.RustSyntax.available", return_value=True), \
                mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=broken_rustfmt):
            self.assertEqual(svc.format_code("fn f( {", language="rust")["status"], "failed")


class ScoringTests(unittest.TestCase):
    def test_off_by_default_a_format_only_difference_fails(self):
        for fmt in (None, False):
            with StubbedTerraform():
                pred, metrics = score(MISALIGNED, fmt)
                argvs = [c.args[0] for c in sandbox_exec.run_sandboxed.call_args_list]
            self.assertIs(pred["row_code_example_checks_pass"], False, fmt)
            self.assertNotIn("row_code_example_format", pred)
            self.assertNotIn("code_example_format_before_check", metrics)
            self.assertTrue(argvs and all("-check" in a for a in argvs), argvs)  # the formatter never ran

    def test_on_formats_before_the_check(self):
        with StubbedTerraform():
            pred, metrics = score(MISALIGNED, True)
        self.assertIs(pred["row_code_example_checkable"], True)
        self.assertIs(pred["row_code_example_checks_pass"], True)
        self.assertEqual(pred["row_code_example_format"]["status"], "formatted")
        self.assertEqual(metrics["code_example_format_before_check"], {"enabled": True, "by_status": {"formatted": 1}})

    def test_on_a_formatter_failure_keeps_the_row_failing(self):
        with StubbedTerraform():
            pred, metrics = score(BROKEN, True)
        self.assertIs(pred["row_code_example_checks_pass"], False)
        self.assertEqual(pred["row_code_example_format"]["status"], "failed")
        self.assertEqual(metrics["code_example_format_before_check"]["by_status"], {"failed": 1})

    def test_on_without_the_binary_is_unchanged_and_recorded(self):
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "terraform_fmt"), \
                mock.patch.object(shutil, "which", return_value=None):
            pred, metrics = score(MISALIGNED, True)
        self.assertIs(pred["row_code_example_checkable"], False)  # no check ran on the reference either
        self.assertIsNone(pred["row_code_example_checks_pass"])
        self.assertEqual(pred["row_code_example_format"]["status"], "not_checkable")
        self.assertEqual(metrics["code_example_format_before_check"]["by_status"], {"not_checkable": 1})

    @unittest.skipUnless(shutil.which("terraform"), "terraform not installed")
    def test_real_terraform_alignment(self):
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "terraform_fmt"):
            off, _ = score(MISALIGNED, None)
            on, _ = score(MISALIGNED, True)
            broken, _ = score(BROKEN, True)
        self.assertIs(off["row_code_example_checks_pass"], False)
        self.assertIs(on["row_code_example_checks_pass"], True, on["row_code_example_format"])
        self.assertIs(broken["row_code_example_checks_pass"], False)
        self.assertEqual(broken["row_code_example_format"]["status"], "failed")


class FingerprintTests(unittest.TestCase):
    def fp(self, runtime_config: dict) -> str:
        return asyncio.run(remediation_scoring_fingerprint(None, 0, runtime_config, {"properties": {}}))

    def test_setting_changes_the_fingerprint_only_when_on_with_code_example_scoring(self):
        base = self.fp({})
        self.assertEqual(self.fp({"remediation": {"format_before_check": False}}), base)
        # Without code-example scoring nothing is formatted: same scorer, same fingerprint.
        self.assertEqual(self.fp({"remediation": {"format_before_check": True}}), base)
        scored = self.fp({"remediation": {"score_code_examples": True}})
        self.assertNotEqual(self.fp({"remediation": {"score_code_examples": True, "format_before_check": True}}), scored)
        self.assertIs(effective_remediation_config({}, 0)["format_before_check"], False)

    def test_compare_flags_evals_with_and_without_it(self):
        fp_off = self.fp({"remediation": {"score_code_examples": True}})
        fp_on = self.fp({"remediation": {"score_code_examples": True, "format_before_check": True}})

        def er(fp: str, fmt: bool):
            metrics = {"scoring_mode": "remediation", "remediation_fingerprint": fp}
            if fmt:
                metrics["code_example_format_before_check"] = {"enabled": True, "by_status": {"formatted": 3}}
            return SimpleNamespace(eval_type="exact_match", metrics=metrics, details={})

        # No code-example row scored: the stamped twin of the fingerprint part still names the setting.
        bare_on = SimpleNamespace(eval_type="exact_match", details={}, metrics={
            "scoring_mode": "remediation", "remediation_fingerprint": fp_on, "remediation_format_before_check": True})
        named = compare_provenance({"scoring": _scoring(bare_on)}, {"scoring": _scoring(er(fp_off, False))})
        self.assertIn("format before check on vs off", named["differences"][0]["message"])

        same = compare_provenance({"scoring": _scoring(er(fp_on, True))}, {"scoring": _scoring(er(fp_on, True))})
        self.assertTrue(same["comparable"])
        diff = compare_provenance({"scoring": _scoring(er(fp_off, False))}, {"scoring": _scoring(er(fp_on, True))})
        self.assertFalse(diff["comparable"])
        self.assertEqual(diff["differences"][0]["kind"], "scoring")
        self.assertIn("format before check off vs on", diff["differences"][0]["message"])


class ApiAndOverlayTests(unittest.TestCase):
    def setUp(self):
        async def make() -> int:
            async with async_session_factory() as db:
                project = Project(name="fmt-before-check", selected_recipe={"recipe_id": "code-remediation"})
                db.add(project)
                await db.commit()
                return project.id

        self.pid = asyncio.run(make())

    def test_put_get_and_live_overlay(self):
        url = f"/api/projects/{self.pid}/remediation-config"
        res = client.get(url)
        self.assertEqual(res.status_code, 200, res.text)
        self.assertIs(res.json()["is_code_remediation"], True)
        self.assertIs(res.json()["effective"]["format_before_check"], False)
        self.assertIsNone(res.json()["config"]["format_before_check"])
        put = client.put(url, json={"score_code_examples": True, "format_before_check": True})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertIs(put.json()["config"]["format_before_check"], True)
        self.assertIs(put.json()["effective"]["format_before_check"], True)

        from app.services.evaluation_service import _apply_remediation_live_config

        async def overlay(manifest_schema: dict) -> dict:
            ctx = SimpleNamespace(manifest={"output_schema": manifest_schema})
            async with async_session_factory() as db:
                await _apply_remediation_live_config(db, self.pid, ctx)
            return ctx.manifest["output_schema"]

        stamped = asyncio.run(overlay({"scoring_mode": "remediation", "remediation": {"checks": None}}))
        self.assertIs(stamped["remediation"]["format_before_check"], True)
        client.put(url, json={})
        cleared = asyncio.run(overlay({"scoring_mode": "remediation",
                                       "remediation": {"checks": None, "format_before_check": True}}))
        self.assertNotIn("format_before_check", cleared["remediation"])


if __name__ == "__main__":
    unittest.main()
