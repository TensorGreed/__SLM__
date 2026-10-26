"""Sandbox runner: real tiny commands only (/bin/echo, /bin/sleep, env, sh-free)."""

import os
import shutil
import sys
import unittest
from unittest import mock

from app.config import settings
from app.services import sandbox_exec
from app.services.sandbox_exec import SandboxError, run_sandboxed

ENV_BIN = shutil.which("env")
ECHO = shutil.which("echo")
SLEEP = shutil.which("sleep")
CAT = shutil.which("cat")


def _run(argv, **kw):
    kw.setdefault("files", {})
    kw.setdefault("timeout_s", 10)
    kw.setdefault("cpu_s", 5)
    kw.setdefault("mem_mb", 512)
    return run_sandboxed(argv, **kw)


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self._p = mock.patch.object(settings, "CODE_CHECK_NETWORK_ISOLATION", "off")
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_echo_and_cwd_files(self):
        r = _run([ECHO, "hi"])
        self.assertEqual((r.returncode, r.stdout.strip(), r.timed_out, r.truncated, r.network_isolated), (0, "hi", False, False, False))
        r = _run([CAT, "a/b.txt"], files={"a/b.txt": "payload"})
        self.assertEqual(r.stdout, "payload")

    def test_stdin(self):
        self.assertEqual(_run([CAT], stdin="from stdin").stdout, "from stdin")

    def test_timeout_kills(self):
        r = _run([SLEEP, "30"], timeout_s=0.5)
        self.assertTrue(r.timed_out)
        self.assertLess(r.duration_s, 10)

    def test_output_truncated(self):
        r = _run([CAT], stdin="x" * 5000, max_output_bytes=100)
        self.assertTrue(r.truncated)
        self.assertEqual(len(r.stdout), 100)

    def test_env_is_minimal(self):
        with mock.patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k", "HTTP_PROXY": "http://p", "https_proxy": "http://p",
            "MY_SECRET_TOKEN": "t", "SYNTH_COMMAND_ENV_PASSTHROUGH": "x",
        }):
            out = _run([ENV_BIN]).stdout
        names = {ln.split("=", 1)[0] for ln in out.splitlines()}
        for bad in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HTTP_PROXY", "https_proxy", "MY_SECRET_TOKEN"):
            self.assertNotIn(bad, names)
        self.assertIn("PATH", names)

    def test_home_is_the_temp_dir(self):
        out = dict(ln.split("=", 1) for ln in _run([ENV_BIN]).stdout.splitlines() if "=" in ln)
        self.assertTrue(out["HOME"].startswith("/") and "brewslm-sbx-" in out["HOME"])
        self.assertFalse(os.path.exists(out["HOME"]))  # cleaned up

    def test_file_name_traversal_refused(self):
        for bad in ("../x", "/etc/x", "a/../../x", ""):
            with self.assertRaises(SandboxError):
                _run([ECHO], files={bad: "x"})

    def test_missing_binary(self):
        with self.assertRaises(SandboxError):
            _run(["definitely-not-a-binary-xyz"])

    def test_require_mode_refuses_when_unavailable(self):
        with mock.patch.object(settings, "CODE_CHECK_NETWORK_ISOLATION", "require"), \
                mock.patch.object(sandbox_exec, "network_isolation_available", return_value=False):
            with self.assertRaises(SandboxError):
                _run([ECHO, "x"])

    def test_auto_mode_prefixes_unshare_and_reports(self):
        calls = []
        real_popen = sandbox_exec.subprocess.Popen

        def spy(argv, *a, **k):
            calls.append(list(argv))
            return real_popen([ECHO, "ok"], *a, **k)

        with mock.patch.object(settings, "CODE_CHECK_NETWORK_ISOLATION", "auto"), \
                mock.patch.object(sandbox_exec, "network_isolation_available", return_value=True), \
                mock.patch.object(sandbox_exec.subprocess, "Popen", side_effect=spy):
            r = _run([ECHO, "x"])
        self.assertTrue(r.network_isolated)
        self.assertEqual(calls[0][1], "-rn")
        self.assertTrue(calls[0][0].endswith("unshare"))

    def test_auto_mode_unavailable_runs_unisolated(self):
        with mock.patch.object(settings, "CODE_CHECK_NETWORK_ISOLATION", "auto"), \
                mock.patch.object(sandbox_exec, "network_isolation_available", return_value=False):
            r = _run([ECHO, "x"])
        self.assertFalse(r.network_isolated)
        self.assertEqual(r.returncode, 0)

    def test_shared_helpers(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k", "FOO_PASS": "1"}):
            env = sandbox_exec.minimal_env("FOO_PASS,ANTHROPIC_API_KEY")
        self.assertEqual(env.get("FOO_PASS"), "1")
        self.assertNotIn("ANTHROPIC_API_KEY", env)


if __name__ == "__main__":
    unittest.main()
