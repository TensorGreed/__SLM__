"""Operator-registered external commands (``external_checks``): the operator file, the two-key gate
(file + ``CODE_CHECK_ALLOWED``), the sandbox call, the option-verifier protocol and its error
handling, the verdict cache, and the detector contract through ``remediation_detector``.
pytest-only; no DB."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import settings
from app.services import code_check_service as ccs
from app.services import external_checks as ec
from app.services import remediation_detector as rd

import _multi_option_toy as toy


def _result(stdout="", returncode=0, timed_out=False, truncated=False, stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr, timed_out=timed_out,
                           truncated=truncated, network_isolated=True, duration_s=0.01)


class _OperatorFileCase(unittest.TestCase):
    allowed = "toy_option_verifier,toy_detector"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="brewslm-extchecks-"))
        self.file = toy.write_operator_file(self.tmp)
        self._patches = [
            patch.object(settings, "EXTERNAL_CHECKS_FILE", str(self.file)),
            patch.object(settings, "CODE_CHECK_ALLOWED", self.allowed),
        ]
        for p in self._patches:
            p.start()
        ec.ensure_loaded(force=True)

    def tearDown(self):
        for p in self._patches:
            p.stop()
        ec.ensure_loaded(force=True)  # unset -> unregister everything the file added
        shutil.rmtree(self.tmp, ignore_errors=True)


class OperatorFileTests(_OperatorFileCase):
    def test_entries_register_as_opt_in_checks(self):
        check = ccs.get_code_check("toy_option_verifier")
        self.assertIsInstance(check, ec.ExternalCommandCheck)
        self.assertEqual(check.kind, "option_verifier")
        self.assertFalse(check.default_enabled)
        self.assertEqual(check.identity(), "toy_option_verifier:p1:r1")
        listed = {c["id"]: c for c in ccs.list_code_checks()}
        self.assertTrue(listed["toy_option_verifier"]["allowed"])
        self.assertFalse(listed["toy_option_verifier"]["default"])
        self.assertFalse(listed["toy_detector"]["default"])
        self.assertNotIn("toy_detector", ccs.default_check_ids())
        self.assertEqual([v["id"] for v in ec.list_option_verifiers()], ["toy_option_verifier"])

    def test_removed_entries_are_unregistered_and_bad_entries_reported(self):
        self.file.write_text(json.dumps([
            {"id": "toy_detector", "kind": "detector", "argv": ["/bin/true"], "languages": ["ini"]},
            {"id": "no argv", "kind": "option_verifier"},
            {"id": "x", "kind": "shell"},
            {"id": "y", "kind": "option_verifier", "argv": ["/bin/true"], "protocol": 2},
            {"id": "python_ast", "kind": "detector", "argv": ["/bin/true"], "languages": ["python"]},
        ]))
        os.utime(self.file, (time.time() + 5, time.time() + 5))
        diag = ec.ensure_loaded()
        self.assertEqual(diag["registered"], ["toy_detector"])
        self.assertEqual(len(diag["errors"]), 4)
        self.assertTrue(any("already registered" in e for e in diag["errors"]))  # never shadows a built-in
        self.assertIsNone(ccs.get_code_check("toy_option_verifier"))
        self.assertNotIsInstance(ccs.get_code_check("python_ast"), ec.ExternalCommandCheck)

    def test_unset_setting_registers_nothing(self):
        with patch.object(settings, "EXTERNAL_CHECKS_FILE", ""):
            diag = ec.ensure_loaded()
            self.assertFalse(diag["configured"])
            self.assertIsNone(ccs.get_code_check("toy_option_verifier"))

    def test_a_missing_or_garbage_file_never_raises(self):
        self.file.write_text("{not json")
        os.utime(self.file, (time.time() + 5, time.time() + 5))
        self.assertEqual(ec.ensure_loaded()["registered"], [])
        with patch.object(settings, "EXTERNAL_CHECKS_FILE", str(self.tmp / "nope.json")):
            self.assertEqual(ec.ensure_loaded()["registered"], [])


class GateTests(_OperatorFileCase):
    allowed = "toy_detector"  # the verifier is declared but NOT allow-listed

    def test_a_declared_but_not_allow_listed_verifier_is_refused(self):
        with self.assertRaises(ec.VerifierUnavailable) as ctx:
            ec.resolve_option_verifier("toy_option_verifier")
        self.assertIn("CODE_CHECK_ALLOWED", str(ctx.exception))
        self.assertTrue(ec.verifier_identity("toy_option_verifier").startswith("unavailable:"))

    def test_unknown_ids_and_non_verifiers_are_refused(self):
        for cid in ("nope", "toy_detector", "python_ast", ""):
            with self.assertRaises(ec.VerifierUnavailable):
                ec.resolve_option_verifier(cid)

    def test_a_named_option_verifier_never_runs_as_a_code_check(self):
        with patch.object(settings, "CODE_CHECK_ALLOWED", "toy_option_verifier"), \
                patch("app.services.sandbox_exec.run_sandboxed", side_effect=AssertionError("ran")):
            report = ccs.run_code_checks("x=1", language="python", check_ids=["toy_option_verifier"])
        self.assertIsNone(report["ok"])


class ProtocolTests(_OperatorFileCase):
    def check(self) -> ec.ExternalCommandCheck:
        return ec.resolve_option_verifier("toy_option_verifier")

    def test_sandbox_call_carries_the_payload_on_stdin_with_limits(self):
        seen = {}

        def fake(argv, **kw):
            seen.update(kw, argv=argv)
            return _result(json.dumps({"formats": {"config_file": {"checks": {"parses": True}, "passes": True},
                                                   "commands": {"checks": {"parses": True, "extra": "n/a"}}},
                                       "agreement": {"agree": True}, "option_passes": True}))

        with patch("app.services.sandbox_exec.run_sandboxed", side_effect=fake):
            verdict = self.check().verify_option(toy.PAYLOAD, toy.good_option(), toy.FORMATS)
        self.assertEqual(seen["argv"], [str(self.tmp / "toy_verify_option")])
        self.assertEqual(seen["files"], {})
        self.assertEqual((seen["timeout_s"], seen["cpu_s"], seen["mem_mb"]), (20.0, 20, 1024))
        sent = json.loads(seen["stdin"])
        self.assertEqual(sent, {"protocol": 1, "payload": toy.PAYLOAD, "option": toy.good_option(), "formats": toy.FORMATS})
        self.assertTrue(verdict["formats"]["config_file"]["passes"])
        # passes derived from the decided checks; non-boolean check values are "not applicable"
        self.assertTrue(verdict["formats"]["commands"]["passes"])
        self.assertIsNone(verdict["formats"]["commands"]["checks"]["extra"])
        self.assertEqual(verdict["agreement"], {"agree": True, "differences": []})

    def test_failures_become_verifier_errors(self):
        # Option-level: the verifier ran on THIS option and broke -> the scorer fails the option.
        cases = [
            (_result(returncode=2, stderr="bad input\n"), "exit 2: bad input", "verifier_crash"),
            (_result(timed_out=True, returncode=-9), "timed out", "verifier_timeout"),
            (_result(stdout="garbage"), "not a JSON object", "verifier_bad_output"),
            (_result(stdout="{}", truncated=True), "truncated", "verifier_output_too_large"),
        ]
        for res, expected, reason in cases:
            with patch("app.services.sandbox_exec.run_sandboxed", return_value=res):
                verdict = self.check().verify_option({}, {}, toy.FORMATS)
            self.assertIn(expected, verdict.get("error", ""), res)
            self.assertEqual(verdict["error_kind"], "option")
            self.assertEqual(verdict["reason"], reason)
        from app.services.sandbox_exec import SandboxError

        # Infrastructure: the sandbox could not launch it -> the option stays unverified.
        with patch("app.services.sandbox_exec.run_sandboxed", side_effect=SandboxError("network isolation is required")):
            verdict = self.check().verify_option({}, {}, toy.FORMATS)
        self.assertIn("network isolation", verdict["error"])
        self.assertEqual(verdict["error_kind"], "infrastructure")

    def test_cache_hit_by_sha256_and_revision_bump_invalidates(self):
        cache = ec.VerdictCache(self.tmp / "cache.json")
        good = _result(json.dumps({"formats": {f: {"checks": {"ok": True}} for f in toy.FORMATS}, "agreement": {"agree": True}}))
        with patch("app.services.sandbox_exec.run_sandboxed", return_value=good) as run:
            verify = ec.make_verify_fn(self.check(), cache)
            verify(toy.PAYLOAD, toy.good_option(), toy.FORMATS)
            verify(toy.PAYLOAD, toy.good_option(), toy.FORMATS)
            self.assertEqual(run.call_count, 1)
            cache.flush()
            # a different option is a different key
            verify(toy.PAYLOAD, toy.restrict_option(), toy.FORMATS)
            self.assertEqual(run.call_count, 2)
        reloaded = ec.VerdictCache(self.tmp / "cache.json")
        with patch("app.services.sandbox_exec.run_sandboxed", side_effect=AssertionError("not cached")):
            ec.make_verify_fn(self.check(), reloaded)(toy.PAYLOAD, toy.good_option(), toy.FORMATS)
        # bump the revision in the operator file: same cache file, new identity -> re-run
        toy.write_operator_file(self.tmp, revision=2)
        os.utime(self.file, (time.time() + 5, time.time() + 5))
        ec.ensure_loaded()
        self.assertEqual(self.check().identity(), "toy_option_verifier:p1:r2")
        with patch("app.services.sandbox_exec.run_sandboxed", return_value=good) as run:
            ec.make_verify_fn(self.check(), ec.VerdictCache(self.tmp / "cache.json"))(toy.PAYLOAD, toy.good_option(), toy.FORMATS)
            self.assertEqual(run.call_count, 1)

    def test_a_tampered_cache_entry_is_ignored(self):
        path = self.tmp / "cache.json"
        cache = ec.VerdictCache(path)
        good = _result(json.dumps({"formats": {f: {"checks": {"ok": False}} for f in toy.FORMATS}, "agreement": {"agree": True}}))
        with patch("app.services.sandbox_exec.run_sandboxed", return_value=good):
            ec.make_verify_fn(self.check(), cache)(toy.PAYLOAD, toy.good_option(), toy.FORMATS)
        cache.flush()
        stored = json.loads(path.read_text())
        (key, entry), = stored.items()
        self.assertIn("mac", entry)
        self.assertEqual(ec.VerdictCache(path).rejected, 0)
        # an untrusted writer flips the verdict to a pass but cannot re-sign it
        for f in toy.FORMATS:
            entry["v"]["formats"][f]["passes"] = True
        path.write_text(json.dumps(stored))
        reloaded = ec.VerdictCache(path)
        self.assertEqual(reloaded.rejected, 1)
        self.assertIsNone(reloaded.get(key))
        # an unsigned (older-format) entry is dropped too
        path.write_text(json.dumps({key: entry["v"]}))
        self.assertIsNone(ec.VerdictCache(path).get(key))
        # so the verifier runs again instead of trusting the forged pass
        with patch("app.services.sandbox_exec.run_sandboxed", return_value=good) as run:
            verdict = ec.make_verify_fn(self.check(), ec.VerdictCache(path))(toy.PAYLOAD, toy.good_option(), toy.FORMATS)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(verdict["formats"][toy.FORMATS[0]]["passes"])

    def test_errors_are_never_cached(self):
        cache = ec.VerdictCache(None)
        with patch("app.services.sandbox_exec.run_sandboxed", return_value=_result(returncode=1)) as run:
            verify = ec.make_verify_fn(self.check(), cache)
            verify({}, {}, toy.FORMATS)
            verify({}, {}, toy.FORMATS)
        self.assertEqual(run.call_count, 2)


class RealSandboxTests(_OperatorFileCase):
    """The toy verifier run for real through ``sandbox_exec`` (a Python script; no network needed)."""

    def test_toy_verifier_end_to_end(self):
        check = ec.resolve_option_verifier("toy_option_verifier")
        good = check.verify_option(toy.PAYLOAD, toy.good_option(), toy.FORMATS)
        self.assertNotIn("error", good, good)
        self.assertTrue(good["formats"]["config_file"]["passes"])
        self.assertTrue(good["formats"]["commands"]["passes"])
        self.assertTrue(good["agreement"]["agree"])
        broken = check.verify_option(toy.PAYLOAD, toy.breaking_option(), toy.FORMATS)
        self.assertFalse(broken["formats"]["commands"]["checks"]["behaviour_preserved"])
        disagree = check.verify_option(toy.PAYLOAD, toy.disagreeing_option(), toy.FORMATS)
        self.assertFalse(disagree["agreement"]["agree"])
        crash = check.verify_option(toy.PAYLOAD, {"strategy": "exit-nonzero"}, toy.FORMATS)
        self.assertIn("exit 3", crash["error"])
        self.assertEqual(crash["reason"], "verifier_crash")
        garbage = check.verify_option(toy.PAYLOAD, {"strategy": "bad-json"}, toy.FORMATS)
        self.assertIn("not a JSON object", garbage["error"])

    def test_timeout(self):
        toy.write_operator_file(self.tmp, timeout_s=1)
        os.utime(self.file, (time.time() + 5, time.time() + 5))
        ec.ensure_loaded()
        verdict = ec.resolve_option_verifier("toy_option_verifier").verify_option({}, {"strategy": "sleep"}, toy.FORMATS)
        self.assertIn("timed out", verdict["error"])
        self.assertEqual((verdict["error_kind"], verdict["reason"]), ("option", "verifier_timeout"))

    def test_operator_detector_through_the_remediation_contract(self):
        resolved = rd._resolve_command_detector({"kind": "command", "check_id": "toy_detector"})
        self.assertIn("toy_detector:p1:r1", resolved.identity)
        labels = resolved.detect([{"code": "debug=on\nport=80\n", "language": "ini"}, {"code": "debug=off\n", "language": "ini"}])
        self.assertEqual(labels, [["debug.enabled"], []])


class DetectorGateTests(_OperatorFileCase):
    allowed = "toy_option_verifier"

    def test_detector_not_allow_listed(self):
        with self.assertRaises(rd.DetectorUnavailable):
            rd._resolve_command_detector({"kind": "command", "check_id": "toy_detector"})


if __name__ == "__main__":
    unittest.main()
