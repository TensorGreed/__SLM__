"""Command (local CLI) synth backend + the hard API budget (phase G9).

A fake CLI (a Python script with a shebang, written to a tmp dir and
allow-listed by absolute path) stands in for the real tool; nothing here
touches the network or a real model. Every cloud provider call goes through
``httpx.MockTransport``.

* safety: allow-list, no shell, minimal env, temp cwd, argv/prompt handling
* output parsing, ``is_error``, packed batches + malformed-item retries
* usage-limit pause -> resume (Job message, no re-calls of finished chunks),
  chunked playbook checkpoint reuse
* ``auto_pick`` False, judge ``local`` False, spend accounting ($0 subscription)
* hard budget: global / project caps, 402 on a Job start, mid-run Job stop
* eval-judge GET / PUT, backends billing / detail
* acceptance: run-playbook ``?async_job=true`` with ``backend="command"``
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services import (
    answer_judge_service,
    cloud_llm_service,
    llm_budget_guard,
    llm_pricing,
    llm_spend_service,
    llm_usage,
    synth_backends,
    synth_playbook_service,
)
from app.services.synth_backends import BACKEND_REGISTRY, CommandBackend, SynthBackendError, pick_backend
from app.services.synth_backends import command as command_module

client: TestClient | None = None
_REAL_ASYNC_CLIENT = httpx.AsyncClient

FAKE_CLI = '''#!{python}
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "config.json")))
argv = sys.argv[1:]
if cfg.get("mode") == "arg":
    prompt = argv[-1]
else:
    prompt = sys.stdin.read()
counter_path = os.path.join(here, "calls.count")
n = int(open(counter_path).read()) + 1 if os.path.exists(counter_path) else 1
open(counter_path, "w").write(str(n))
with open(os.path.join(here, "calls.log"), "a") as fh:
    fh.write(json.dumps({{"argv": argv, "prompt": prompt, "env": sorted(os.environ), "cwd": os.getcwd(), "n": n}}) + "\\n")
if cfg.get("sleep"):
    time.sleep(cfg["sleep"])
if n <= cfg.get("limit_first_k", 0):
    sys.stderr.write("Error: usage limit reached for this plan\\n")
    sys.exit(1)
packed = "EXACTLY one JSON line per request" in prompt
if "[[FAIL]]" in prompt or ("[[BAD]]" in prompt and not packed):
    sys.stderr.write("boom\\n")
    sys.exit(2)

def emit(text):
    fmt = cfg.get("format", "json")
    usage = {{"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2}}
    if fmt == "raw":
        sys.stdout.write(text)
    elif fmt == "nested":
        sys.stdout.write(json.dumps({{"data": {{"text": text}}, "usage": usage}}))
    elif fmt == "choices":
        sys.stdout.write(json.dumps({{"choices": [{{"text": text}}]}}))
    else:
        sys.stdout.write(json.dumps({{"result": text, "usage": usage}}))

if "[[ISERR]]" in prompt:
    sys.stdout.write(json.dumps({{"is_error": True, "result": "model refused"}}))
    sys.exit(0)
if "[[ENV]]" in prompt:
    emit(json.dumps(sorted(os.environ)))
elif "[[CWD]]" in prompt:
    emit(os.getcwd())
elif "Requests:" in prompt and "EXACTLY one JSON line per request" in prompt:
    lines = []
    for raw in prompt.split("Requests:\\n", 1)[1].splitlines():
        item = json.loads(raw)
        if "[[BAD]]" in item["prompt"]:
            continue
        if "[[MALFORMED]]" in item["prompt"]:
            lines.append('{{"id": "%s", "output": ' % item["id"])
            continue
        lines.append(json.dumps({{"id": item["id"], "output": "ANS:" + item["prompt"][:40]}}))
    body = "\\n".join(lines)
    emit(("```json\\n" + body + "\\n```") if cfg.get("fence") else body)
elif "JSON lines" in prompt:
    rows = [json.dumps({{"text": "Could I get a refund on order %d-%d please" % (n, i), "label": "billing"}}) for i in range(4)]
    emit("\\n".join(rows))
else:
    emit("ECHO:" + prompt[:200])
'''


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    wait_for_jobs_idle(client)
    client.__exit__(None, None, None)


class _FakeCli:
    """A fake CLI dir: ``cli`` (executable), ``config.json``, ``calls.*``."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="brewslm-fakecli-")).resolve()
        self.cli = self.dir / "fakecli"
        self.cli.write_text(FAKE_CLI.format(python=sys.executable))
        self.cli.chmod(self.cli.stat().st_mode | stat.S_IEXEC)
        self.configure()

    def configure(self, **cfg) -> None:
        (self.dir / "config.json").write_text(json.dumps(cfg))

    def calls(self) -> list[dict]:
        path = self.dir / "calls.log"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def reset_calls(self) -> None:
        for name in ("calls.log", "calls.count"):
            (self.dir / name).unlink(missing_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class CommandTestCase(unittest.TestCase):
    """Settings patched to point at the fake CLI; usage-limit pause reset."""

    argv_extra: list[str] = []
    prompt_mode = "stdin"

    def setUp(self) -> None:
        self.fake = _FakeCli()
        self.addCleanup(self.fake.cleanup)
        argv = [str(self.fake.cli), "--model", "{model}"]
        if self.prompt_mode == "arg":
            argv.append("{prompt}")
        patches = {
            "SYNTH_COMMAND_ALLOWED_BINARIES": str(self.fake.cli),
            "SYNTH_COMMAND_ARGV": json.dumps(argv),
            "SYNTH_COMMAND_PROMPT_MODE": self.prompt_mode,
            "SYNTH_COMMAND_OUTPUT": ".result",
            "SYNTH_COMMAND_DEFAULT_MODEL": "sonnet",
            "SYNTH_COMMAND_BACKOFF_SECONDS": 0.01,
            "SYNTH_COMMAND_COOLDOWN_SECONDS": 0.3,
            "SYNTH_COMMAND_TIMEOUT_SECONDS": 20.0,
            "SYNTH_COMMAND_BATCH_SIZE": 8,
            "SYNTH_COMMAND_MAX_RETRIES": 1,
            "SYNTH_COMMAND_CONCURRENCY": 1,
            "SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL": 25,
        }
        for key, value in patches.items():
            p = mock.patch.object(settings, key, value)
            p.start()
            self.addCleanup(p.stop)
        # the production floor (60 s) would make every pause test minutes long
        floor = mock.patch.object(command_module, "_MIN_COOLDOWN_SECONDS", 0.0)
        floor.start()
        self.addCleanup(floor.stop)
        command_module.reset_pause()
        self.addCleanup(command_module.reset_pause)
        llm_budget_guard.reset()
        self.addCleanup(llm_budget_guard.reset)
        self.fake.configure(mode=self.prompt_mode)
        self.fake.reset_calls()

    def cfg(self, **cfg) -> None:
        self.fake.configure(mode=self.prompt_mode, **cfg)

    def run_async(self, coro):
        return asyncio.run(coro)


# ── A: safety, parsing, batching ─────────────────────────────────────


class TestAllowList(CommandTestCase):
    def test_empty_allow_list_disables_the_backend(self):
        with mock.patch.object(settings, "SYNTH_COMMAND_ALLOWED_BINARIES", ""):
            self.assertFalse(CommandBackend.is_available())
            with self.assertRaises(SynthBackendError):
                CommandBackend()
            with self.assertRaises(SynthBackendError):
                pick_backend("command")

    def test_binary_not_on_the_list_is_refused_even_when_on_path(self):
        other = shutil.which("echo") or "/bin/echo"
        with mock.patch.object(settings, "SYNTH_COMMAND_ARGV", json.dumps([other, "hi"])):
            self.assertFalse(CommandBackend.is_available())
            with self.assertRaisesRegex(SynthBackendError, "binary not allow-listed: echo"):
                CommandBackend()

    def test_empty_argument_elements_are_allowed(self):
        # the documented recipe turns a CLI's tools off with `--tools ""`
        argv = [str(self.fake.cli), "--model", "{model}", "--tools", ""]
        with mock.patch.object(settings, "SYNTH_COMMAND_ARGV", json.dumps(argv)):
            self.assertTrue(CommandBackend.is_available())
            built, _ = CommandBackend()._build_argv("hello")
            self.assertEqual(built[-2:], ["--tools", ""])
        with mock.patch.object(settings, "SYNTH_COMMAND_ARGV", json.dumps(["", "x"])):
            self.assertFalse(CommandBackend.is_available())

    def test_bare_name_only_matches_a_bare_argv0(self):
        resolved = str(self.fake.cli)
        with mock.patch.object(settings, "SYNTH_COMMAND_ALLOWED_BINARIES", "fakecli"):
            # an absolute argv[0] whose basename equals the bare entry is NOT allowed
            self.assertFalse(command_module._is_allowed(resolved, resolved))
            self.assertFalse(CommandBackend.is_available())
            self.assertTrue(command_module._is_allowed("fakecli", resolved))

    def test_allow_list_rechecked_per_call(self):
        backend = CommandBackend()
        with mock.patch.object(settings, "SYNTH_COMMAND_ALLOWED_BINARIES", "/somewhere/else"):
            with self.assertRaisesRegex(SynthBackendError, "not allow-listed"):
                self.run_async(backend.complete("hello"))
        self.assertEqual(self.fake.calls(), [])

    def test_model_must_match_pattern(self):
        with self.assertRaises(ValueError):
            CommandBackend(model="bad model; rm -rf /")
        self.assertEqual(pick_backend("command:opus").describe(), "command:opus")

    def test_describe_round_trips_and_detail(self):
        self.assertEqual(CommandBackend().describe(), "command")
        pinned = CommandBackend(model="opus")
        self.assertEqual(pinned.describe(), "command:opus")
        self.assertEqual(pick_backend(pinned.describe()).describe(), "command:opus")
        self.assertIn("fakecli CLI (subscription)", CommandBackend().detail())
        self.assertIn("batch 8", CommandBackend().detail())

    def test_registry_has_command_before_template(self):
        self.assertIn(CommandBackend, BACKEND_REGISTRY)
        self.assertEqual(BACKEND_REGISTRY[-1].name, "template")
        self.assertLess(BACKEND_REGISTRY.index(CommandBackend), len(BACKEND_REGISTRY) - 1)

    def test_arg_mode_requires_a_prompt_placeholder(self):
        with mock.patch.object(settings, "SYNTH_COMMAND_PROMPT_MODE", "arg"):
            with self.assertRaisesRegex(SynthBackendError, "{prompt}"):
                CommandBackend()


class _InjectionMixin:
    def test_shell_metacharacters_arrive_literally(self):
        backend = CommandBackend()
        # absolute targets: a relative one would land in the throw-away cwd and prove nothing
        t = str(self.fake.dir)
        evil = f"; touch {t}/PWNED; $(touch {t}/PWNED2) `touch {t}/PWNED3`"
        reply = self.run_async(backend.complete(evil))
        self.assertEqual(reply, "ECHO:" + evil[:200])
        for name in ("PWNED", "PWNED2", "PWNED3"):
            self.assertFalse(Path(name).exists(), name)
            self.assertFalse((self.fake.dir / name).exists(), name)
        self.assertEqual(self.fake.calls()[-1]["prompt"], evil)

    def test_prompt_starting_with_a_flag_is_not_a_flag(self):
        reply = self.run_async(CommandBackend().complete("--help"))
        argv = self.fake.calls()[-1]["argv"]
        if self.prompt_mode == "arg":
            self.assertEqual(reply, "ECHO: --help")
            self.assertEqual(argv[-1], " --help")  # prefixed with a single space
        else:
            self.assertEqual(reply, "ECHO:--help")
            self.assertNotIn("--help", argv)


class TestInjectionStdin(_InjectionMixin, CommandTestCase):
    prompt_mode = "stdin"


class TestInjectionArg(_InjectionMixin, CommandTestCase):
    prompt_mode = "arg"


class TestEnvironmentAndCwd(CommandTestCase):
    def test_env_is_minimal_and_denies_api_keys(self):
        with mock.patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-secret", "ANTHROPIC_AUTH_TOKEN": "t", "OPENAI_API_KEY": "o",
            "SOME_OTHER_SECRET": "x", "EXTRA_OK": "1",
        }):
            with mock.patch.object(settings, "SYNTH_COMMAND_ENV_PASSTHROUGH", "EXTRA_OK,ANTHROPIC_API_KEY"):
                reply = self.run_async(CommandBackend().complete("[[ENV]]"))
        env = set(json.loads(reply))
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("SOME_OTHER_SECRET", env)
        self.assertIn("EXTRA_OK", env)
        self.assertIn("PATH", env)

    def test_cwd_is_a_fresh_temp_dir(self):
        first = self.run_async(CommandBackend().complete("[[CWD]]"))
        second = self.run_async(CommandBackend().complete("[[CWD]]"))
        repo = str(Path(__file__).resolve().parents[2])
        self.assertFalse(first.startswith(repo))
        self.assertIn("brewslm-cmd-", first)
        self.assertNotEqual(first, second)
        self.assertFalse(Path(first).exists())  # removed after the call


class TestOutputParsing(CommandTestCase):
    def test_default_result_path_and_usage_recorded_as_subscription(self):
        with llm_usage.metered() as meter:
            reply = self.run_async(CommandBackend().complete("hello", system_prompt="be brief"))
        self.assertTrue(reply.startswith("ECHO:be brief"))
        entries = meter.entries()
        self.assertEqual(len(entries), 1)
        row = entries[0]
        self.assertEqual(row["model"], "subscription:fakecli")
        self.assertEqual(row["calls"], 1)
        self.assertEqual((row["input_tokens"], row["output_tokens"]), (11, 7))
        self.assertEqual((row["cache_read_tokens"], row["cache_write_tokens"]), (3, 2))
        self.assertEqual(row["usd"], 0.0)

    def test_raw_output(self):
        self.cfg(format="raw")
        with mock.patch.object(settings, "SYNTH_COMMAND_OUTPUT", "raw"):
            with llm_usage.metered() as meter:
                reply = self.run_async(CommandBackend().complete("hello"))
        self.assertEqual(reply, "ECHO:hello")
        self.assertEqual(meter.entries()[0]["calls"], 1)  # the call counts even without token counts
        self.assertEqual(meter.entries()[0]["total_tokens"], 0)

    def test_nested_and_indexed_paths(self):
        self.cfg(format="nested")
        with mock.patch.object(settings, "SYNTH_COMMAND_OUTPUT", ".data.text"):
            self.assertEqual(self.run_async(CommandBackend().complete("x")), "ECHO:x")
        self.cfg(format="choices")
        with mock.patch.object(settings, "SYNTH_COMMAND_OUTPUT", ".choices.0.text"):
            self.assertEqual(self.run_async(CommandBackend().complete("y")), "ECHO:y")

    def test_missing_path_is_an_error(self):
        self.cfg(format="nested")
        with mock.patch.object(settings, "SYNTH_COMMAND_MAX_RETRIES", 0):
            with self.assertRaisesRegex(SynthBackendError, "not found"):
                self.run_async(CommandBackend().complete("x"))

    def test_is_error_and_exit_code_failures_retry_then_raise(self):
        with mock.patch.object(settings, "SYNTH_COMMAND_MAX_RETRIES", 1):
            with self.assertRaisesRegex(SynthBackendError, "model refused"):
                self.run_async(CommandBackend().complete("[[ISERR]]"))
            self.assertEqual(len(self.fake.calls()), 2)  # first try + 1 retry
            self.fake.reset_calls()
            with self.assertRaisesRegex(SynthBackendError, "boom"):
                self.run_async(CommandBackend().complete("[[FAIL]]"))
            self.assertEqual(len(self.fake.calls()), 2)

    def test_response_schema_becomes_an_instruction(self):
        self.run_async(CommandBackend().complete("hi", response_schema={"type": "object"}))
        self.assertIn("matching this JSON Schema", self.fake.calls()[-1]["prompt"])

    def test_timeout_kills_and_fails(self):
        self.cfg(sleep=5)
        with mock.patch.object(settings, "SYNTH_COMMAND_TIMEOUT_SECONDS", 0.5), mock.patch.object(
            settings, "SYNTH_COMMAND_MAX_RETRIES", 0
        ):
            started = time.monotonic()
            with self.assertRaisesRegex(SynthBackendError, "timed out"):
                self.run_async(CommandBackend().complete("slow"))
        self.assertLess(time.monotonic() - started, 4.0)

    def test_project_overrides_are_clamped_and_argv_is_never_overridable(self):
        ctx = {"concurrency": 99, "batch_size": 0, "model": "opus", "argv": ["rm"], "timeout_seconds": 5}
        with llm_budget_guard.project_context(1, ctx):
            backend = CommandBackend()
        self.assertEqual(backend.concurrency, 8)
        self.assertEqual(backend.batch_size, 1)
        self.assertEqual(backend.model, "opus")
        self.assertEqual(backend.timeout_seconds, 5.0)
        self.assertEqual(backend.binary, "fakecli")  # settings' argv, not the project's


class TestBatching(CommandTestCase):
    def _requests(self, n, marks=None):
        mark = marks or {}
        return [{"prompt": f"question {i} {mark.get(i, '')}", "system_prompt": "sys", "max_tokens": 50, "temperature": 0.0}
                for i in range(n)]

    def test_requests_are_packed_into_few_cli_calls(self):
        with mock.patch.object(settings, "SYNTH_COMMAND_BATCH_SIZE", 4):
            backend = CommandBackend()
            seen: list[tuple[int, int]] = []

            async def progress(message, done, total):
                seen.append((done, total))

            with llm_usage.metered() as meter:
                out = self.run_async(backend.complete_batch(self._requests(10), progress=progress))
        self.assertEqual(len(self.fake.calls()), 3)  # 4 + 4 + 2
        self.assertEqual([o[:4] for o in out], ["ANS:"] * 10)
        self.assertTrue(out[7].startswith("ANS:question 7"))
        self.assertEqual(seen[-1], (10, 10))
        # exactly one usage record per CLI call (no double counting)
        self.assertEqual(meter.entries()[0]["calls"], 3)
        self.assertTrue(backend.use_batch)

    def test_fenced_replies_parse(self):
        self.cfg(fence=True)
        out = self.run_async(CommandBackend().complete_batch(self._requests(3)))
        self.assertTrue(all(o and o.startswith("ANS:") for o in out))

    def test_missing_item_is_retried_individually_once_then_rejected_with_reason(self):
        backend = CommandBackend()
        out = self.run_async(backend.complete_batch(self._requests(3, {1: "[[BAD]]"})))
        self.assertEqual(out[0][:4], "ANS:")
        self.assertIsNone(out[1])  # the packed reply omits it and its single retry exits non-zero
        self.assertEqual(out[2][:4], "ANS:")
        self.assertEqual(backend.rejected_items, [{"id": "r1", "reason": "call_failed"}])
        self.assertEqual(backend.batch_report[-1]["rejected"], backend.rejected_items)
        self.assertEqual(backend.batch_report[-1]["failed"], 1)

    def test_malformed_item_is_retried_individually_and_can_succeed(self):
        backend = CommandBackend()
        out = self.run_async(backend.complete_batch(self._requests(3, {2: "[[MALFORMED]]"})))
        # packed reply malformed for r2 -> one individual call, which echoes
        self.assertTrue(out[2].startswith("ECHO:"))
        self.assertEqual(backend.rejected_items, [])
        self.assertEqual(len(self.fake.calls()), 2)

    def test_parse_reasons(self):
        answers, bad = command_module._parse_packed(
            '{"id": "r0", "output": "a"}\n{"id": "r1", "output": ""}\n{"id": "r2", "output": 5}\n'
            '{"id": "r3", "output": "x"}\n{"id": "r3", "output": "y"}\n{"id": "r4", "outp',
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(answers, {0: "a"})
        self.assertEqual(bad[1], "empty_output")
        self.assertEqual(bad[2], "malformed_json")
        self.assertEqual(bad[3], "malformed_json")  # duplicate
        self.assertEqual(bad[4], "malformed_json")  # truncated line naming it
        self.assertEqual(bad[5], "missing_from_batch")

    def test_batch_size_one_does_not_pack(self):
        with mock.patch.object(settings, "SYNTH_COMMAND_BATCH_SIZE", 1):
            backend = CommandBackend()
            self.assertFalse(backend.use_batch)
            out = self.run_async(backend.complete_batch(self._requests(2)))
        self.assertEqual(len(self.fake.calls()), 2)
        self.assertTrue(all(o.startswith("ECHO:") for o in out))

    def test_cache_prefix_is_not_lost(self):
        reqs = [{"prompt": "p", "system_prompt": "s", "cache_prefix": "STABLE-INSTRUCTIONS"}]
        self.run_async(CommandBackend().complete_batch(reqs))
        self.assertIn("STABLE-INSTRUCTIONS", self.fake.calls()[-1]["prompt"])


class TestBatchWiring(CommandTestCase):
    def test_uses_message_batch_and_enable_batch_accept_the_command_backend(self):
        from app.services.documents_qa_flow_service import _enable_batch
        from app.services.span_relabel_service import uses_message_batch

        backend = CommandBackend()
        self.assertTrue(uses_message_batch(backend))
        backend.use_batch = False
        _enable_batch(backend)
        self.assertTrue(backend.use_batch)
        with self.assertRaises(ValueError):
            _enable_batch(SimpleNamespace(name="ollama"))


# ── usage-limit pause ────────────────────────────────────────────────


class TestUsageLimitPause(CommandTestCase):
    def test_limit_pauses_then_retries_without_consuming_a_retry(self):
        self.cfg(limit_first_k=2)
        with mock.patch.object(settings, "SYNTH_COMMAND_MAX_RETRIES", 0):
            started = time.monotonic()
            reply = self.run_async(CommandBackend().complete("hi"))
        self.assertEqual(reply, "ECHO:hi")
        self.assertEqual(len(self.fake.calls()), 3)  # 2 limited + the success
        self.assertGreaterEqual(time.monotonic() - started, 0.5)  # two 0.3s cool-downs

    def test_reset_epoch_overrides_the_cooldown(self):
        backend = CommandBackend()
        with mock.patch.object(settings, "SYNTH_COMMAND_USAGE_LIMIT_REGEX", r"limit.*reset (?P<reset>\d+)"):
            backend._start_pause(f"usage limit reset {int(time.time()) + 2}")
        self.assertTrue(command_module.is_paused())
        remaining = command_module._PAUSE["until_mono"] - time.monotonic()
        self.assertTrue(0.5 < remaining <= 3.1, remaining)

    def test_pause_beyond_the_per_job_cap_fails(self):
        self.cfg(limit_first_k=50)
        with mock.patch.object(settings, "SYNTH_COMMAND_MAX_PAUSE_SECONDS", 0.5):
            with self.assertRaisesRegex(SynthBackendError, "usage window did not reopen"):
                self.run_async(CommandBackend().complete("hi"))


# ── auto_pick / judge ────────────────────────────────────────────────


class TestAutoPickAndJudge(CommandTestCase):
    def test_pick_backend_none_never_returns_command(self):
        self.assertFalse(CommandBackend.auto_pick)
        with self.assertRaises(SynthBackendError):
            pick_backend(None, registry=[CommandBackend])
        self.assertIsInstance(pick_backend("command", registry=[CommandBackend]), CommandBackend)

    def test_judge_auto_walk_skips_it_and_pinned_command_is_not_local(self):
        no_cloud = mock.AsyncMock(return_value=None)
        with mock.patch.object(synth_backends, "BACKEND_REGISTRY", [CommandBackend]), mock.patch.object(
            answer_judge_service, "_cloud_config", no_cloud
        ), mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": ""}):
            self.assertIsNone(self.run_async(answer_judge_service.resolve_answer_judge(None, 1, None)))
            project = SimpleNamespace(runtime_config={"eval_judge": {"enabled": True, "backend": "command"}})
            judge = self.run_async(answer_judge_service.resolve_answer_judge(None, 1, project))
        self.assertIsNotNone(judge)
        self.assertEqual(judge.label, "command")
        self.assertFalse(judge.local)

    def test_sends_data_off_machine_flags(self):
        from app.services.synth_backends import CloudLlmBackend, OllamaBackend

        self.assertTrue(CommandBackend.sends_data_off_machine)
        self.assertTrue(CloudLlmBackend.sends_data_off_machine)
        self.assertFalse(getattr(OllamaBackend, "sends_data_off_machine", False))


# ── D: pricing / tally ───────────────────────────────────────────────


class TestSpendAccounting(unittest.TestCase):
    def test_subscription_rows_are_priced_zero_before_price_matching(self):
        est = llm_pricing.estimate_cost("subscription:sonnet", {"calls": 2, "input_tokens": 1_000_000, "output_tokens": 500_000})
        self.assertEqual(est["usd"], 0.0)
        self.assertEqual(est["usd_uncached"], 0.0)
        self.assertEqual(est["basis"], "subscription (not billed per call)")
        self.assertEqual(est["total_tokens"], 1_500_000)

    def test_tally_separates_subscription_from_metered(self):
        sub = llm_pricing.estimate_cost("subscription:claude", {"calls": 3, "input_tokens": 100, "output_tokens": 50})
        met = llm_pricing.estimate_cost("claude-sonnet-4-6", {"calls": 1, "input_tokens": 1000, "output_tokens": 1000})
        out = llm_spend_service.tally([
            {"kind": "synth_playbook", "usage": [{"backend": "subscription:claude", **sub}]},
            {"kind": "relabel", "usage": [{"backend": "anthropic:claude-sonnet-4-6", **met}]},
        ])
        rows = {r["model"]: r for r in out["by_model"]}
        self.assertEqual(rows["subscription:claude"]["billing"], "subscription")
        self.assertEqual(rows["claude-sonnet-4-6"]["billing"], "metered")
        self.assertEqual(out["totals"]["subscription_calls"], 3)
        self.assertEqual(out["totals"]["subscription_tokens"], 150)
        self.assertEqual(out["totals"]["cost_usd"], round(met["usd"], 4))

    def test_subscription_usage_never_touches_the_budget_ledger(self):
        llm_budget_guard.reset()
        self.addCleanup(llm_budget_guard.reset)
        llm_usage.record("", "subscription:claude", input_tokens=10_000_000, output_tokens=10_000_000)
        self.assertEqual(llm_budget_guard.snapshot()["global"], 0.0)
        llm_usage.record("anthropic", "claude-sonnet-4-6", input_tokens=1_000_000)
        self.assertAlmostEqual(llm_budget_guard.snapshot()["global"], 3.0, places=3)


# ── C: hard budget ───────────────────────────────────────────────────


class _Anthropic:
    def __init__(self) -> None:
        self.requests = 0

    def __enter__(self):
        def handler(request: httpx.Request):
            self.requests += 1
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "hi"}], "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 10, "output_tokens": 10},
            })

        def make(*a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            return _REAL_ASYNC_CLIENT(*a, **kw)

        self._patch = mock.patch.object(httpx, "AsyncClient", side_effect=make)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


class TestBudgetGuard(CommandTestCase):
    def _call(self):
        return self.run_async(cloud_llm_service.call_anthropic_chat(
            api_key="k", model="claude-sonnet-4-6", system_prompt="s", user_prompt="u",
        ))

    def test_global_cap_blocks_before_any_request(self):
        llm_budget_guard.seed_for_tests(global_spent=100.0)
        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 100.0), _Anthropic() as api:
            with self.assertRaises(llm_budget_guard.LlmBudgetExceeded) as ctx:
                self._call()
        self.assertEqual(api.requests, 0)
        self.assertEqual(ctx.exception.scope, "global")
        self.assertIn("API budget reached (global: $100.00 of $100.00 this month)", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, SynthBackendError)

    def test_under_the_cap_proceeds_and_counts(self):
        llm_budget_guard.seed_for_tests(global_spent=1.0)
        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 100.0), _Anthropic() as api:
            resp = self._call()
        self.assertEqual(resp.content, "hi")
        self.assertEqual(api.requests, 1)

    def test_project_cap_blocks_inside_that_project_only(self):
        llm_budget_guard.seed_for_tests(global_spent=6.0, projects={7: 6.0}, caps={7: 5.0})
        with _Anthropic() as api:
            with llm_budget_guard.project_context(7):
                with self.assertRaises(llm_budget_guard.LlmBudgetExceeded) as ctx:
                    self._call()
            self.assertEqual(ctx.exception.scope, "project")
            self.assertEqual(api.requests, 0)
            with llm_budget_guard.project_context(8):  # another project, no cap, no global cap
                self.assertEqual(self._call().content, "hi")

    def test_cloud_backend_checks_before_complete_and_batch(self):
        from app.services.synth_backends import CloudLlmBackend

        llm_budget_guard.seed_for_tests(global_spent=5.0)
        backend = CloudLlmBackend(provider="anthropic", model="claude-sonnet-4-6", api_key="k")
        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 5.0), _Anthropic() as api:
            with self.assertRaises(llm_budget_guard.LlmBudgetExceeded):
                self.run_async(backend.complete("x"))
            with self.assertRaises(llm_budget_guard.LlmBudgetExceeded):
                self.run_async(backend.complete_batch([{"prompt": "x"}]))
        self.assertEqual(api.requests, 0)

    def test_local_and_subscription_calls_are_never_blocked(self):
        llm_budget_guard.seed_for_tests(global_spent=500.0, projects={1: 500.0}, caps={1: 1.0})
        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 1.0), llm_budget_guard.project_context(1):
            self.assertEqual(self.run_async(CommandBackend().complete("hi")), "ECHO:hi")
            llm_budget_guard.check_budget("", "subscription:claude")
            # a local (LAN / localhost) OpenAI-compatible endpoint skips the guard
            def handler(request: httpx.Request):
                return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

            def make(*a, **kw):
                kw["transport"] = httpx.MockTransport(handler)
                return _REAL_ASYNC_CLIENT(*a, **kw)

            with mock.patch.object(httpx, "AsyncClient", side_effect=make):
                resp = self.run_async(cloud_llm_service.call_openai_chat(
                    api_key="k", model="llama3", system_prompt="s", user_prompt="u",
                    api_url="http://localhost:11434/v1/chat/completions",
                ))
            self.assertEqual(resp.content, "ok")

    def test_month_rollover_resets_the_ledger(self):
        llm_budget_guard.seed_for_tests(global_spent=500.0)
        with llm_budget_guard._LOCK:
            llm_budget_guard._STATE["month"] = "1999-01"
        self.assertEqual(llm_budget_guard.snapshot()["global"], 0.0)

    def test_runtime_setting_for_the_global_cap(self):
        from app.services import runtime_settings_service as rss

        spec = rss._SPEC_BY_KEY["LLM_MONTHLY_BUDGET_USD"]
        self.assertEqual(rss._coerce_value(spec, "100"), 100.0)
        self.assertIsNone(rss._coerce_value(spec, ""))
        with self.assertRaises(ValueError):
            rss._coerce_value(spec, "-5")


# ── API: 402, mid-run stop, backends, eval judge, acceptance ─────────


def _project_from_template(name: str = "ticket-router") -> int:
    resp = client.post(f"/api/project-templates/{name}/instantiate", json={"project_name": f"G9-{time.time_ns()}"})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _wait_job(job_id: int, *, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} still running")


class TestBudgetApi(CommandTestCase):
    def tearDown(self):
        wait_for_jobs_idle(client)

    def test_cloud_pinned_job_start_is_402_when_capped(self):
        pid = _project_from_template()
        llm_budget_guard.seed_for_tests(global_spent=50.0)
        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 50.0), mock.patch(
            "app.api.synthetic.resolve_cloud_backend", mock.AsyncMock(return_value=object())
        ):
            resp = client.post(
                f"/api/projects/{pid}/synthetic/run-playbook?async_job=true",
                json={"mode": "positives_paraphrase", "target_count": 3, "backend": "cloud:anthropic:claude-sonnet-4-6"},
            )
        self.assertEqual(resp.status_code, 402, resp.text)
        body = resp.json()
        self.assertEqual(body["error_code"], "llm_budget_exceeded")
        self.assertEqual(body["metadata"]["scope"], "global")
        self.assertEqual(body["metadata"]["monthly"], 50.0)
        self.assertIsInstance(body["detail"], str)
        self.assertIn("API budget reached", body["detail"])

    def test_command_job_start_is_never_blocked(self):
        pid = _project_from_template()
        llm_budget_guard.seed_for_tests(global_spent=500.0)
        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 1.0):
            resp = client.post(
                f"/api/projects/{pid}/synthetic/run-playbook?async_job=true",
                json={"mode": "positives_paraphrase", "target_count": 3, "backend": "command"},
            )
            self.assertEqual(resp.status_code, 202, resp.text)
            _wait_job(resp.json()["id"])

    def test_job_crossing_the_cap_mid_run_fails_with_the_message(self):
        pid = _project_from_template()
        llm_budget_guard.seed_for_tests(global_spent=0.5)

        async def fake_run_playbook(db, project_id, mode, **kwargs):
            # one expensive call lands ($1.50) ...
            llm_usage.record("anthropic", "claude-sonnet-4-6", output_tokens=100_000)
            # ... and the next metered call is refused before any request
            await cloud_llm_service.call_anthropic_chat(
                api_key="k", model="claude-sonnet-4-6", system_prompt="s", user_prompt="u",
            )
            raise AssertionError("not reached")

        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 1.0), mock.patch.object(
            synth_playbook_service, "run_playbook", fake_run_playbook
        ), _Anthropic() as api:
            resp = client.post(
                f"/api/projects/{pid}/synthetic/run-playbook?async_job=true",
                json={"mode": "positives_paraphrase", "target_count": 3},
            )
            self.assertEqual(resp.status_code, 202, resp.text)
            job = _wait_job(resp.json()["id"])
        self.assertEqual(job["status"], "failed")
        self.assertTrue(job["error"].startswith("API budget reached (global: $2.00 of $1.00 this month)"), job["error"])
        self.assertIn("switch to a local or subscription backend", job["error"])
        self.assertEqual(api.requests, 0)

    def test_budget_endpoint_has_global_block_and_hard_stop(self):
        pid = _project_from_template()
        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 100.0):
            body = client.get(f"/api/projects/{pid}/llm-budget").json()
        self.assertTrue(body["hard_stop"])
        self.assertEqual(body["global"]["monthly"], 100.0)
        self.assertIn(body["global"]["status"], {"ok", "warning", "over"})
        self.assertIn("spent_usd", body["global"])
        body = client.get(f"/api/projects/{pid}/llm-budget").json()
        self.assertIsNone(body["global"]["monthly"])
        self.assertEqual(body["global"]["status"], "ok")


class TestBackendsAndJudgeApi(CommandTestCase):
    def test_backends_list_has_billing_and_detail(self):
        pid = _project_from_template()
        entries = {e["name"]: e for e in client.get(f"/api/projects/{pid}/synthetic/backends").json()["backends"]}
        self.assertEqual(entries["command"]["billing"], "subscription")
        self.assertTrue(entries["command"]["available"])
        self.assertEqual(entries["command"]["describe"], "command")
        self.assertIn("(subscription)", entries["command"]["detail"])
        self.assertEqual(entries["ollama"]["billing"], "local")
        self.assertIsNone(entries["ollama"]["detail"])
        with mock.patch.object(settings, "SYNTH_COMMAND_ALLOWED_BINARIES", ""):
            entries = {e["name"]: e for e in client.get(f"/api/projects/{pid}/synthetic/backends").json()["backends"]}
        self.assertFalse(entries["command"]["available"])
        self.assertIsNone(entries["command"]["detail"])

    def test_eval_judge_round_trip(self):
        pid = _project_from_template()
        with mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": ""}):
            got = client.get(f"/api/projects/{pid}/eval-judge")
            self.assertEqual(got.status_code, 200, got.text)
            body = got.json()
            self.assertTrue(body["enabled"])
            self.assertIsNone(body["backend"])
            self.assertIsNone(body["env_override"])
            values = {o["value"]: o for o in body["options"]}
            self.assertIn("", values)
            self.assertEqual(values["command"]["billing"], "subscription")
            self.assertEqual(values["anthropic"]["billing"], "metered")
            self.assertIn("available", values["anthropic"])

            put = client.put(f"/api/projects/{pid}/eval-judge", json={"enabled": True, "backend": "command"})
            self.assertEqual(put.status_code, 200, put.text)
            self.assertEqual(put.json()["backend"], "command")
            resolved = put.json()["resolved"]
            self.assertEqual(resolved["describe"], "command")
            self.assertFalse(resolved["local"])
            self.assertEqual(resolved["billing"], "subscription")
            self.assertEqual(client.get(f"/api/projects/{pid}/eval-judge").json()["backend"], "command")

            off = client.put(f"/api/projects/{pid}/eval-judge", json={"enabled": False, "backend": None})
            self.assertFalse(off.json()["enabled"])
            self.assertIsNone(off.json()["resolved"])

        with mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": "ollama"}):
            self.assertEqual(client.get(f"/api/projects/{pid}/eval-judge").json()["env_override"], "ollama")

    def test_eval_judge_put_rejects_unknown_backend(self):
        pid = _project_from_template()
        resp = client.put(f"/api/projects/{pid}/eval-judge", json={"enabled": True, "backend": "mystery:x"})
        self.assertEqual(resp.status_code, 422, resp.text)


class TestPlaybookAcceptance(CommandTestCase):
    def tearDown(self):
        wait_for_jobs_idle(client)

    def _start(self, pid: int, target: int) -> dict:
        resp = client.post(
            f"/api/projects/{pid}/synthetic/run-playbook?async_job=true",
            json={"mode": "positives_paraphrase", "target_count": target, "backend": "command"},
        )
        self.assertEqual(resp.status_code, 202, resp.text)
        return resp.json()

    def test_acceptance_command_playbook_job_and_subscription_spend(self):
        pid = _project_from_template()
        job = _wait_job(self._start(pid, 3)["id"])
        self.assertEqual(job["status"], "succeeded", job)
        self.assertGreaterEqual(job["result"]["rows_generated"], 1)
        self.assertEqual(job["result"]["backend_used"], "command")
        spend = client.get(f"/api/projects/{pid}/llm-spend").json()
        rows = {r["model"]: r for r in spend["by_model"]}
        self.assertIn("subscription:fakecli", rows)
        self.assertEqual(rows["subscription:fakecli"]["billing"], "subscription")
        self.assertGreater(rows["subscription:fakecli"]["calls"], 0)
        self.assertEqual(spend["month"]["cost_usd"], 0.0)
        self.assertEqual(spend["totals"]["cost_usd"], 0.0)
        self.assertGreater(spend["totals"]["subscription_calls"], 0)
        # a subscription run never moves the API budget
        self.assertEqual(llm_budget_guard.snapshot()["global"], 0.0)

    def test_chunked_run_pauses_on_usage_limit_and_resumes_without_recalling_chunks(self):
        pid = _project_from_template()
        # 10 rows in chunks of 4 -> 3 chunks; the first CLI call hits the usage limit once.
        self.cfg(limit_first_k=1)
        with mock.patch.object(settings, "SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL", 4), mock.patch.object(
            settings, "SYNTH_COMMAND_COOLDOWN_SECONDS", 1.5
        ):
            started = self._start(pid, 10)
            seen_waiting = False
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                job = client.get(f"/api/jobs/{started['id']}").json()
                if "waiting for usage window" in (job.get("progress_message") or ""):
                    seen_waiting = True
                if job["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.05)
        self.assertTrue(seen_waiting, "the Job message never showed the usage-window pause")
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["result"]["chunks"], 3)
        self.assertEqual(job["result"]["failed_chunks"], [])
        self.assertGreater(job["result"].get("paused_seconds", 0), 0.5)
        # 1 limited call + 3 chunk calls: finished chunks were never called twice
        self.assertEqual(len(self.fake.calls()), 4)
        prompts = [c["prompt"] for c in self.fake.calls()]
        self.assertTrue(all("of 3: vary topics" in p for p in prompts))
        self.assertGreaterEqual(job["result"]["rows_generated"], 3)
        self.assertGreater(job["result"]["paused_seconds"], 0)

    def test_checkpoint_is_reused_after_a_crash_and_removed_on_success(self):
        from app.services.synth_playbooks import SynthMode

        pid = _project_from_template()
        project_dir = Path(settings.DATA_DIR) / "projects" / str(pid) / "synthetic" / "playbook_checkpoints"
        backend = CommandBackend()
        path = synth_playbook_service.checkpoint_path_for(
            pid, "classification", SynthMode.POSITIVES_PARAPHRASE, backend.describe(), 10, None, None,
        )
        # What a crashed run left behind: chunk 0 finished and fsynced.
        row = {"payload": {"text": "Could I get a refund on order 99 please", "label": "billing"},
               "synth_confidence": 0.9, "synth_source": "playbook:classification:positives_paraphrase"}
        synth_playbook_service._append_checkpoint(path, {"chunk": 0, "rows": [row], "raw_snippet": "x"})
        self.assertTrue(path.exists())
        self.assertEqual(path.parent, project_dir)
        real = synth_playbook_service.checkpoint_path_for
        # the prompt hash is part of the real key; this test seeds the file by hand without it
        with mock.patch.object(settings, "SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL", 5), mock.patch.object(
            synth_playbook_service, "checkpoint_path_for", lambda *a, **kw: real(*a, **{**kw, "prompt_hash": ""}),
        ):
            job = _wait_job(self._start(pid, 10)["id"])
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["result"]["chunks"], 2)
        self.assertEqual(len(self.fake.calls()), 1)  # only chunk 2 was called
        self.assertIn("Batch 2 of 2", self.fake.calls()[0]["prompt"])
        self.assertFalse(path.exists())  # deleted on success

    def test_chunk_helpers(self):
        self.assertEqual(synth_playbook_service.chunk_sizes(10, 4), [4, 3, 3])
        self.assertEqual(synth_playbook_service.chunk_sizes(25, 25), [25])
        self.assertEqual(sum(synth_playbook_service.chunk_sizes(101, 25)), 101)

    def test_stale_checkpoints_are_pruned(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        old, fresh = d / "old.jsonl", d / "fresh.jsonl"
        old.write_text("{}\n")
        fresh.write_text("{}\n")
        week_ago = time.time() - 8 * 24 * 3600
        os.utime(old, (week_ago, week_ago))
        synth_playbook_service._prune_stale_checkpoints(d)
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())

    def test_non_chunking_backend_keeps_the_single_call(self):
        class OneShot:
            name = "oneshot"
            schema_aware = False
            calls = 0

            async def complete(self, prompt, **kw):
                OneShot.calls += 1
                return '{"text": "Please refund me", "label": "billing"}\n'

            def describe(self):
                return "oneshot"

        pid = _project_from_template()
        with mock.patch.object(synth_playbook_service, "pick_backend", return_value=OneShot()):
            resp = client.post(
                f"/api/projects/{pid}/synthetic/run-playbook",
                json={"mode": "positives_paraphrase", "target_count": 200},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(OneShot.calls, 1)
        self.assertNotIn("chunks", resp.json())


if __name__ == "__main__":
    unittest.main()


# ── fix lane: review findings ────────────────────────────────────────


class _FakeHandle:
    """A stand-in for ``JobProgressHandle`` (records messages; scripted cancel)."""

    def __init__(self, cancel_after_checks: int | None = None) -> None:
        self.last_message: str | None = None
        self.pre_pause_message: str | None = None
        self.messages: list[str] = []
        self.checks = 0
        self.cancel_after = cancel_after_checks

    async def set_message(self, message):
        self.messages.append(message)
        self.last_message = message

    async def check_cancelled(self) -> bool:
        self.checks += 1
        return self.cancel_after is not None and self.checks > self.cancel_after


def _with_handle(handle, coro_fn):
    from app.services import jobs_service

    async def runner():
        token = jobs_service.current_job_handle.set(handle)
        try:
            return await coro_fn()
        finally:
            jobs_service.current_job_handle.reset(token)

    return asyncio.run(runner())


class TestModelFlagInjection(CommandTestCase):
    def test_a_model_may_not_start_with_a_dash(self):
        for bad in ("--dangerously-skip", "-x", "-"):
            with self.assertRaises(ValueError):
                CommandBackend(model=bad)
            with self.assertRaises(ValueError):
                pick_backend(f"command:{bad}")
        CommandBackend(model="opus-4.1")  # a dash inside is fine

    def test_project_override_model_is_validated_too(self):
        with llm_budget_guard.project_context(1, {"model": "--flag"}):
            with self.assertRaises(ValueError):
                CommandBackend()

    def test_eval_judge_put_rejects_a_flag_like_model(self):
        pid = _project_from_template()
        resp = client.put(f"/api/projects/{pid}/eval-judge", json={"enabled": True, "backend": "command:--x"})
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_run_playbook_and_flows_reject_a_flag_like_pin(self):
        pid = _project_from_template()
        resp = client.post(
            f"/api/projects/{pid}/synthetic/run-playbook?async_job=true",
            json={"mode": "positives_paraphrase", "target_count": 3, "backend": "command:--x"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(self.fake.calls(), [])
        from app.services.cloud_backend_pins import CloudPinError, resolve_cloud_backend

        with self.assertRaises(CloudPinError) as ctx:
            self.run_async(resolve_cloud_backend(None, pid, "command:-x"))
        self.assertEqual(ctx.exception.status_code, 400)


class TestCooldownFloorAndHitCap(CommandTestCase):
    def test_reset_epoch_in_the_past_falls_back_to_the_cooldown_floor(self):
        backend = CommandBackend()
        with mock.patch.object(command_module, "_MIN_COOLDOWN_SECONDS", 60.0), mock.patch.object(
            settings, "SYNTH_COMMAND_USAGE_LIMIT_REGEX", r"limit.*reset (?P<reset>\d+)"
        ):
            backend._start_pause(f"usage limit reset {int(time.time()) - 500}")
        remaining = command_module._PAUSE["until_mono"] - time.monotonic()
        self.assertTrue(55 < remaining <= 61, remaining)

    def test_tiny_cooldown_is_raised_to_the_floor(self):
        backend = CommandBackend()
        with mock.patch.object(command_module, "_MIN_COOLDOWN_SECONDS", 60.0), mock.patch.object(
            settings, "SYNTH_COMMAND_COOLDOWN_SECONDS", 0.0
        ):
            backend._start_pause("usage limit reached")
        self.assertGreater(command_module._PAUSE["until_mono"] - time.monotonic(), 55)

    def test_consecutive_usage_limit_hits_are_capped(self):
        self.cfg(limit_first_k=1000)
        with mock.patch.object(command_module, "_MAX_LIMIT_HITS", 3), mock.patch.object(
            settings, "SYNTH_COMMAND_COOLDOWN_SECONDS", 0.0
        ):
            with self.assertRaisesRegex(SynthBackendError, "usage limit 3 times"):
                self.run_async(CommandBackend().complete("hi"))
        self.assertEqual(len(self.fake.calls()), 4)

    def test_usage_limit_text_in_a_successful_reply_does_not_pause(self):
        reply = self.run_async(CommandBackend().complete("tell me about the usage limit and rate limit 429"))
        self.assertTrue(reply.startswith("ECHO:"))
        self.assertFalse(command_module.is_paused())
        self.assertEqual(settings.SYNTH_COMMAND_USAGE_LIMIT_REGEX, r"usage limit|rate limit|\b429\b")


class TestPackedParsingAndSchema(CommandTestCase):
    def test_unhashable_or_odd_ids_are_skipped_and_retried(self):
        reply = (
            '{"id": ["r0"], "output": "x"}\n{"id": {"a": 1}, "output": "y"}\n{"id": 5, "output": "z"}\n'
            '{"id": null, "output": "n"}\n{"id": "r1", "output": "ok"}'
        )
        answers, bad = command_module._parse_packed(reply, [0, 1])
        self.assertEqual(answers, {1: "ok"})
        self.assertEqual(bad[0], "missing_from_batch")

    def test_each_packed_item_carries_its_response_schema(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        reqs = [{"prompt": "p0", "response_schema": schema}, {"prompt": "p1"}]
        packed = command_module._pack_prompt(reqs, [0, 1])
        items = [json.loads(line) for line in packed.split("Requests:\n", 1)[1].splitlines()]
        self.assertEqual(items[0]["response_schema"], schema)
        self.assertNotIn("response_schema", items[1])

    def test_individual_retry_keeps_the_schema_instruction(self):
        schema = {"type": "object"}
        reqs = [{"prompt": "a", "response_schema": schema}, {"prompt": "b [[MALFORMED]]", "response_schema": schema}]
        self.run_async(CommandBackend().complete_batch(reqs))
        single = self.fake.calls()[-1]["prompt"]
        self.assertIn("Reply with ONLY JSON matching this JSON Schema", single)


class TestPauseHandling(CommandTestCase):
    def test_a_call_queued_behind_the_semaphore_rechecks_the_pause(self):
        async def go():
            command_module._PAUSE["until_mono"] = time.monotonic() + 5
            return await CommandBackend()._invoke("hi")

        outcome = self.run_async(go())
        self.assertTrue(outcome.paused)
        self.assertEqual(self.fake.calls(), [])  # nothing was launched during the pause

    def test_pre_pause_message_is_remembered_once_and_restored(self):
        handle = _FakeHandle()
        handle.last_message = "chunk 2/3 · 10 rows so far"
        backend = CommandBackend()

        async def go():
            command_module._PAUSE["until_mono"] = time.monotonic() + 0.2
            command_module._PAUSE["until_wall"] = time.time() + 0.2
            await backend._wait_if_paused()

        _with_handle(handle, go)
        self.assertEqual(handle.pre_pause_message, "chunk 2/3 · 10 rows so far")
        self.assertTrue(handle.messages[0].startswith("waiting for usage window"))
        self.assertEqual(handle.last_message, "chunk 2/3 · 10 rows so far")

    def test_a_waiting_message_is_never_stored_or_restored(self):
        handle = _FakeHandle()
        handle.last_message = "waiting for usage window — resumes ~01:00 UTC"
        handle.pre_pause_message = "real message"
        backend = CommandBackend()

        async def go():
            command_module._PAUSE["until_mono"] = time.monotonic() + 0.2
            command_module._PAUSE["until_wall"] = time.time() + 0.2
            await backend._wait_if_paused()

        _with_handle(handle, go)
        self.assertEqual(handle.pre_pause_message, "real message")
        self.assertEqual(handle.last_message, "real message")
        self.assertFalse(any(m == "waiting for usage window — resumes ~01:00 UTC" for m in handle.messages[1:]))

    def test_never_restores_a_waiting_text_when_nothing_preceded_it(self):
        handle = _FakeHandle()  # no message before the pause
        backend = CommandBackend()

        async def go():
            command_module._PAUSE["until_mono"] = time.monotonic() + 0.2
            command_module._PAUSE["until_wall"] = time.time() + 0.2
            await backend._wait_if_paused()

        _with_handle(handle, go)
        self.assertTrue(all(m.startswith("waiting for usage window") for m in handle.messages))


class TestCancellation(CommandTestCase):
    def test_cancelled_job_never_launches_the_cli(self):
        handle = _FakeHandle(cancel_after_checks=0)
        with self.assertRaises(asyncio.CancelledError):
            _with_handle(handle, lambda: CommandBackend().complete("hi"))
        self.assertEqual(self.fake.calls(), [])

    def test_cancel_kills_the_in_flight_cli(self):
        self.cfg(sleep=15)
        handle = _FakeHandle(cancel_after_checks=2)  # the pre-invoke check + the first polls
        started = time.monotonic()
        with mock.patch.object(command_module, "_CANCEL_POLL_SECONDS", 0.1):
            with self.assertRaises(asyncio.CancelledError):
                _with_handle(handle, lambda: CommandBackend().complete("slow"))
        self.assertLess(time.monotonic() - started, 6)  # not the CLI's 15 s
        self.assertEqual(len(self.fake.calls()), 1)

    def test_cancel_stops_sibling_chunks_of_a_batch(self):
        self.cfg(sleep=15)
        handle = _FakeHandle(cancel_after_checks=4)
        reqs = [{"prompt": f"q{i}"} for i in range(4)]
        started = time.monotonic()
        with mock.patch.object(settings, "SYNTH_COMMAND_BATCH_SIZE", 1), mock.patch.object(
            settings, "SYNTH_COMMAND_CONCURRENCY", 4
        ), mock.patch.object(command_module, "_CANCEL_POLL_SECONDS", 0.1):
            with self.assertRaises(asyncio.CancelledError):
                _with_handle(handle, lambda: CommandBackend().complete_batch(reqs))
        self.assertLess(time.monotonic() - started, 8)


class TestConcurrencyIsProcessWide(CommandTestCase):
    def _elapsed(self, n: int, *, project_override: dict | None = None) -> float:
        self.cfg(sleep=0.4)

        async def go():
            with llm_budget_guard.project_context(3, project_override):
                backend = CommandBackend()
                await asyncio.gather(*(backend.complete(f"q{i}") for i in range(n)))

        started = time.monotonic()
        self.run_async(go())
        return time.monotonic() - started

    def test_a_project_override_can_only_lower_the_limit(self):
        with mock.patch.object(settings, "SYNTH_COMMAND_CONCURRENCY", 4):
            lowered = self._elapsed(4, project_override={"concurrency": 1})
        self.assertGreaterEqual(lowered, 1.5)  # serialised: 4 x 0.4 s
        self.fake.reset_calls()
        with mock.patch.object(settings, "SYNTH_COMMAND_CONCURRENCY", 1):
            raised = self._elapsed(4, project_override={"concurrency": 8})
        self.assertGreaterEqual(raised, 1.5)  # still the global limit of 1


class TestJudgeSelection(CommandTestCase):
    def test_teacher_sends_data_off_machine_only_for_a_non_local_url(self):
        from app.services.synth_backends.teacher import TeacherModelBackend

        self.assertTrue(TeacherModelBackend(api_url="https://api.example.com/v1/chat/completions").sends_data_off_machine)
        self.assertFalse(TeacherModelBackend(api_url="http://localhost:11434/v1/chat/completions").sends_data_off_machine)
        self.assertFalse(TeacherModelBackend(api_url="http://192.168.1.5:8000/v1/chat/completions").sends_data_off_machine)

    def test_resolve_judge_outside_a_job_applies_project_command_overrides(self):
        project = SimpleNamespace(runtime_config={"eval_judge": {"enabled": True, "backend": "command"},
                                                  "synth_command": {"model": "opus"}})
        with mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": ""}):
            judge = self.run_async(answer_judge_service.resolve_answer_judge(None, 1, project))
        self.assertEqual(judge.label, "command:opus")
        self.assertFalse(judge.local)

    def test_auto_option_billing_reflects_what_it_resolves_to(self):
        class FakeMetered:
            name = "fakemetered"
            billing = "metered"
            auto_pick = True

            @classmethod
            def is_available(cls):
                return True

            def describe(self):
                return "fakemetered"

        pid = _project_from_template()
        with mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": ""}), mock.patch(
            "app.services.synth_backends.BACKEND_REGISTRY", [FakeMetered, CommandBackend]
        ):
            body = client.get(f"/api/projects/{pid}/eval-judge").json()
        auto = {o["value"]: o for o in body["options"]}[""]
        self.assertEqual(auto["billing"], "metered")
        # the command backend alone is never what Automatic picks
        with mock.patch.dict(os.environ, {"EVAL_JUDGE_BACKEND": ""}), mock.patch(
            "app.services.synth_backends.BACKEND_REGISTRY", [CommandBackend]
        ), mock.patch("app.services.answer_judge_service._cloud_config", mock.AsyncMock(return_value=None)):
            body = client.get(f"/api/projects/{pid}/eval-judge").json()
        self.assertEqual({o["value"]: o for o in body["options"]}[""]["billing"], "local")


class TestCheckpointKey(CommandTestCase):
    def test_prompt_hash_changes_the_checkpoint_file(self):
        from app.services.synth_playbooks import SynthMode

        def path(h):
            return synth_playbook_service.checkpoint_path_for(
                1, "classification", SynthMode.POSITIVES_PARAPHRASE, "command", 10, None, None, prompt_hash=h,
            )

        self.assertNotEqual(path("a"), path("b"))
        self.assertEqual(path("a"), path("a"))


class TestBudgetLedgerFixes(CommandTestCase):
    def test_unpriced_metered_model_counts_at_the_highest_listed_price(self):
        llm_budget_guard.seed_for_tests()
        self.assertIsNone(llm_pricing.estimate_cost("mystery-model-9", {"output_tokens": 1_000_000})["usd"])
        llm_usage.record("someprovider", "mystery-model-9", output_tokens=1_000_000)
        top_out = max(p[1] for p in llm_pricing._prices().values())
        self.assertAlmostEqual(llm_budget_guard.snapshot()["global"], top_out, places=3)
        # and a project cap sees it too
        llm_budget_guard.seed_for_tests(caps={5: 1.0})
        with llm_budget_guard.project_context(5):
            llm_usage.record("someprovider", "mystery-model-9", output_tokens=1_000_000)
            with self.assertRaises(llm_budget_guard.LlmBudgetExceeded):
                llm_budget_guard.check_budget("someprovider", "mystery-model-9")
        # the spend view still says price unknown
        self.assertEqual(llm_pricing.estimate_cost("mystery-model-9", {"output_tokens": 5})["basis"], "price unknown")

    def test_subscription_calls_never_count_even_when_unpriced(self):
        llm_budget_guard.seed_for_tests()
        llm_usage.record("", "subscription:mystery", output_tokens=5_000_000)
        self.assertEqual(llm_budget_guard.snapshot()["global"], 0.0)

    def _run_batch(self, backend, results_by_attempt, checks):
        from app.services.synth_backends import CloudLlmBackend  # noqa: F401

        attempts = iter(results_by_attempt)

        async def create(*, api_key, requests):
            return {"id": "b1"}

        async def wait(**kw):
            return {"results_url": "u"}

        async def fetch(**kw):
            return next(attempts)

        async def acheck(provider=None, model=None):
            checks.append(model)

        with mock.patch.object(cloud_llm_service, "create_anthropic_batch", create), mock.patch.object(
            cloud_llm_service, "wait_for_anthropic_batch", wait
        ), mock.patch.object(cloud_llm_service, "fetch_anthropic_batch_results", fetch), mock.patch.object(
            llm_budget_guard, "acheck_budget", acheck
        ):
            return self.run_async(backend.complete_batch([{"prompt": "x"}, {"prompt": "y"}]))

    def test_message_batch_spend_reaches_the_ledger_at_half_price_without_double_metering(self):
        from app.services.synth_backends import CloudLlmBackend

        llm_budget_guard.seed_for_tests()
        backend = CloudLlmBackend(provider="anthropic", model="claude-sonnet-4-6", api_key="k")
        msg = {"type": "succeeded", "message": {"content": [{"type": "text", "text": "ok"}], "model": "claude-sonnet-4-6",
                                               "usage": {"input_tokens": 500_000, "output_tokens": 0}}}
        checks: list = []
        with llm_usage.metered() as meter:
            out = self._run_batch(backend, [{"r0": msg, "r1": msg}], checks)
        self.assertEqual(out, ["ok", "ok"])
        # 1M input tokens of sonnet at 3 USD/M = 3 USD list -> 1.5 USD batch
        self.assertAlmostEqual(llm_budget_guard.snapshot()["global"], 1.5, places=3)
        self.assertEqual(meter.entries(), [])  # batch usage is NOT recorded through the meter (no double count)
        self.assertEqual(backend.batch_usage["calls"], 2)

    def test_temperature_resubmit_rechecks_the_budget(self):
        from app.services.synth_backends import CloudLlmBackend

        backend = CloudLlmBackend(provider="anthropic", model="claude-fixlane-temp-1", api_key="k")
        self.addCleanup(cloud_llm_service._NO_TEMPERATURE_MODELS.discard, "claude-fixlane-temp-1")
        bad = {"type": "errored", "error": {"error": {"message": "temperature is deprecated for this model"}}}
        good = {"type": "succeeded", "message": {"content": [{"type": "text", "text": "ok"}], "usage": {}}}
        checks: list = []
        out = self._run_batch(backend, [{"r0": bad, "r1": bad}, {"r0": good, "r1": good}], checks)
        self.assertEqual(out, ["ok", "ok"])
        self.assertEqual(len(checks), 2)  # the initial check + one before the resubmit


class TestBudgetReachesOtherCallers(CommandTestCase):
    def test_playground_metered_chat_is_402_when_capped(self):
        pid = _project_from_template()
        llm_budget_guard.seed_for_tests(global_spent=10.0)
        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 10.0), _Anthropic() as api:
            for path in ("chat", "chat/stream"):
                resp = client.post(
                    f"/api/projects/{pid}/training/playground/{path}",
                    json={"provider": "openai_compatible", "model_name": "gpt-4o",
                          "api_url": "https://api.openai.com/v1/chat/completions", "api_key": "k",
                          "messages": [{"role": "user", "content": "hi"}], "save_history": False},
                )
                self.assertEqual(resp.status_code, 402, resp.text)
                self.assertEqual(resp.json()["error_code"], "llm_budget_exceeded")
        self.assertEqual(api.requests, 0)

    def test_playground_local_chat_is_not_checked(self):
        llm_budget_guard.seed_for_tests(global_spent=10.0)
        from app.services import playground_service

        with mock.patch.object(settings, "LLM_MONTHLY_BUDGET_USD", 10.0):
            self.run_async(playground_service.check_chat_budget(1, "openai_compatible", None, "llama3"))
            self.run_async(playground_service.check_chat_budget(1, "mock", "https://api.openai.com/x", "m"))
            with self.assertRaises(llm_budget_guard.LlmBudgetExceeded):
                self.run_async(playground_service.check_chat_budget(1, "openai_compatible", "https://api.openai.com/x", "m"))

    def test_probe_judge_and_plan_refinement_reraise_the_budget_error(self):
        from app.services import evaluation_service, pipeline_refinement_service as prs

        boom = mock.AsyncMock(side_effect=llm_budget_guard.LlmBudgetExceeded("global", 1.0, 2.0))
        with mock.patch.object(cloud_llm_service, "call_openai_chat", boom), mock.patch.object(
            cloud_llm_service, "call_anthropic_chat", boom
        ):
            with self.assertRaises(llm_budget_guard.LlmBudgetExceeded):
                self.run_async(evaluation_service._judge_probe_via_cloud(
                    {"property": "refuses_or_declines", "input": "i", "_model_output": "o"},
                    provider="openai", model="gpt-4o", api_key="k",
                ))
            fn = prs._build_cloud_strategy_fn({"provider": "openai", "api_key": "k", "model": "gpt-4o"})
            with self.assertRaises(llm_budget_guard.LlmBudgetExceeded):
                self.run_async(fn({"rows": 5}))
