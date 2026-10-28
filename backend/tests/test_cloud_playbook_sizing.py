"""Cloud playbook synthesis: chunked runs and reply budgets sized to the rows asked for.

A cloud playbook run used to be ONE call for the whole target with max_tokens=2048
(20 code examples came back as 4, the reply truncated). The cloud backend now
states ``playbook_rows_per_call`` (chunked + checkpointed like the command
backend) and ``max_output_tokens`` (the ceiling a call's budget is sized up to).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services.synth_backends.cloud_llm import CloudLlmBackend  # noqa: E402
from app.services.synth_playbook_service import PLAYBOOK_TOKENS_PER_ROW, _sized_max_tokens  # noqa: E402


class _Ceiling:
    def __init__(self, ceiling):
        self.max_output_tokens = ceiling


class SizedMaxTokensTests(unittest.TestCase):
    def test_backends_without_a_ceiling_keep_the_fixed_budget(self):
        self.assertEqual(_sized_max_tokens(object(), 50, PLAYBOOK_TOKENS_PER_ROW, 2048), 2048)
        self.assertEqual(_sized_max_tokens(_Ceiling(None), 50, PLAYBOOK_TOKENS_PER_ROW, 2048), 2048)
        self.assertEqual(_sized_max_tokens(_Ceiling(True), 50, PLAYBOOK_TOKENS_PER_ROW, 2048), 2048)

    def test_budget_grows_with_the_rows_up_to_the_ceiling_never_below_the_floor(self):
        self.assertEqual(_sized_max_tokens(_Ceiling(16000), 1, 1024, 2048), 2048)
        self.assertEqual(_sized_max_tokens(_Ceiling(16000), 10, 1024, 2048), 10240)
        self.assertEqual(_sized_max_tokens(_Ceiling(16000), 50, 1024, 2048), 16000)


class CloudBackendTests(unittest.TestCase):
    def backend(self, provider):
        return CloudLlmBackend(provider=provider, model="m", api_key="k")

    def test_output_ceiling_per_provider(self):
        self.assertEqual(self.backend("anthropic").max_output_tokens, 16000)
        self.assertEqual(self.backend("openai").max_output_tokens, 16384)
        self.assertEqual(self.backend("deepseek").max_output_tokens, 8192)

    def test_rows_per_call_follows_the_setting_clamped(self):
        b = self.backend("anthropic")
        self.assertEqual(b.playbook_rows_per_call, settings.SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL)
        with patch.object(settings, "SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL", 0):
            self.assertEqual(b.playbook_rows_per_call, 1)
        with patch.object(settings, "SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL", 5000):
            self.assertEqual(b.playbook_rows_per_call, 100)


class ThinkingBudgetTests(unittest.TestCase):
    """A model that thinks by default spends thinking tokens from max_tokens: a big
    prompt at the default effort could use the whole budget and return no text.
    Cloud calls send SYNTH_CLOUD_EFFORT, get thinking headroom, and say so plainly
    when the budget still ran out."""

    def test_effort_is_sent_only_to_models_that_take_it(self):
        from app.services.cloud_llm_service import anthropic_message_params

        def params(model, effort="medium"):
            return anthropic_message_params(model=model, system_prompt="s", user_prompt="u", effort=effort)

        self.assertEqual(params("claude-sonnet-5-5")["output_config"], {"effort": "medium"})
        self.assertEqual(params("claude-opus-4-8", "low")["output_config"], {"effort": "low"})
        self.assertNotIn("output_config", params("claude-haiku-4-5-20251001"))
        self.assertNotIn("output_config", params("claude-sonnet-5-5", "turbo"))
        self.assertNotIn("output_config", params("claude-sonnet-5-5", None))

    def test_effort_setting(self):
        b = CloudLlmBackend(provider="anthropic", model="claude-sonnet-5-5", api_key="k")
        self.assertEqual(b.effort, "medium")
        with patch.object(settings, "SYNTH_CLOUD_EFFORT", " HIGH "):
            self.assertEqual(b.effort, "high")
        with patch.object(settings, "SYNTH_CLOUD_EFFORT", ""):
            self.assertIsNone(b.effort)
        # a model that only thinks when asked keeps its own default effort
        self.assertIsNone(CloudLlmBackend(provider="anthropic", model="claude-opus-4-8", api_key="k").effort)

    def test_thinking_headroom_for_models_that_think_by_default(self):
        from app.services.synth_backends.cloud_llm import THINKING_HEADROOM_TOKENS

        thinker = CloudLlmBackend(provider="anthropic", model="claude-sonnet-5-5", api_key="k")
        # callers size answers against the ceiling less the headroom, so a full-size call still gets it
        self.assertEqual(thinker.max_output_tokens, 16000 - THINKING_HEADROOM_TOKENS)
        self.assertEqual(thinker._reply_budget(thinker.max_output_tokens), 16000)
        self.assertEqual(thinker._reply_budget(2048), 2048 + THINKING_HEADROOM_TOKENS)
        self.assertEqual(thinker._reply_budget(12000), 16000, "direct calls stay under the HTTP-timeout ceiling")
        self.assertEqual(thinker._reply_budget(12000, batch=True), 12000 + THINKING_HEADROOM_TOKENS)
        self.assertEqual(thinker._reply_budget(30000, batch=True), 32000)
        for other in (CloudLlmBackend(provider="anthropic", model="claude-opus-4-8", api_key="k"),
                      CloudLlmBackend(provider="openai", model="gpt-4o", api_key="k")):
            self.assertEqual(other._reply_budget(2048), 2048)

    def _complete(self, response):
        import asyncio

        from app.services import llm_budget_guard
        from app.services.synth_backends import cloud_llm

        seen = {}

        async def fake_call(**kw):
            seen.update(kw)
            return response

        async def ok(*a, **kw):
            return None

        b = CloudLlmBackend(provider="anthropic", model="claude-sonnet-5-5", api_key="k")
        self.backend = b
        with patch.object(cloud_llm, "call_anthropic_chat", fake_call), patch.object(llm_budget_guard, "acheck_budget", ok):
            return seen, asyncio.run(b.complete("p", max_tokens=6000))

    def test_direct_call_sends_effort_and_the_headroom_budget(self):
        from app.services.cloud_llm_service import CloudLlmResponse

        seen, text = self._complete(CloudLlmResponse("answer", "claude-sonnet-5-5", 10, 5, stop_reason="end_turn"))
        self.assertEqual(text, "answer")
        self.assertEqual((seen["effort"], seen["max_tokens"]), ("medium", 14000))

    def test_a_reply_with_no_text_at_max_tokens_is_a_clear_error(self):
        from app.services.cloud_llm_service import CloudLlmResponse
        from app.services.synth_backends.base import SynthBackendError

        with self.assertRaisesRegex(SynthBackendError, "spent its whole reply budget .* thinking"):
            self._complete(CloudLlmResponse("", "claude-sonnet-5-5", 10, 14000, stop_reason="max_tokens"))
        self.assertEqual((self.backend.usage["calls"], self.backend.usage["output_tokens"]), (1, 14000),
                         "the call was paid for: the backend's own usage still counts it")

    def test_batch_requests_get_effort_headroom_and_report_budget_spent_thinking(self):
        import asyncio
        import json

        from app.services import llm_budget_guard
        from test_relabel_batch import FakeAnthropic, _patched

        fake = FakeAnthropic()
        fake.results_for = lambda requests: "\n".join(json.dumps({"custom_id": r["custom_id"], "result": {
            "type": "succeeded", "message": {
                "content": [{"type": "thinking", "thinking": ""}] + (
                    [] if r["custom_id"] == "r1" else [{"type": "text", "text": "ok"}]),
                "model": "claude-sonnet-5-5", "stop_reason": "max_tokens" if r["custom_id"] == "r1" else "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5}}}}) for r in requests)

        async def ok(*a, **kw):
            return None

        b = CloudLlmBackend(provider="anthropic", model="claude-sonnet-5-5", api_key="k")
        reqs = [{"prompt": "a", "max_tokens": 12000}, {"prompt": "b", "max_tokens": 12000}]
        with _patched(fake), patch.dict(os.environ, {"ANTHROPIC_BATCH_POLL_SECONDS": "0"}), \
                patch.object(llm_budget_guard, "acheck_budget", ok), patch.object(llm_budget_guard, "note_usage", lambda *a, **kw: None):
            out = asyncio.run(b.complete_batch(reqs))
        self.assertEqual(out, ["ok", None], "no text = unanswered, so the caller retries it directly")
        sent = [r["params"] for r in fake.batches[0]]
        self.assertEqual({p["max_tokens"] for p in sent}, {20000})
        self.assertEqual({json.dumps(p["output_config"]) for p in sent}, {json.dumps({"effort": "medium"})})
        self.assertEqual(b.batch_report[-1]["budget_spent_thinking"], 1)


class PriceTests(unittest.TestCase):
    def test_model_generations_are_priced_on_their_own_rate(self):
        from app.services.llm_pricing import price_for

        self.assertEqual(price_for("claude-sonnet-5-5"), (2.0, 10.0))
        self.assertEqual(price_for("claude-sonnet-5"), (2.0, 10.0))
        self.assertEqual(price_for("claude-sonnet-4-6"), (3.0, 15.0))
        self.assertEqual(price_for("claude-opus-5-5"), (4.0, 20.0))
        self.assertEqual(price_for("claude-opus-4-8"), (5.0, 25.0))
        self.assertEqual(price_for("claude-haiku-5-5"), (0.1, 0.5))
        self.assertEqual(price_for("claude-haiku-4-5-20251001"), (1.0, 5.0))


class ChunkedCloudRunTests(unittest.TestCase):
    """A cloud-shaped backend goes through the chunked path with sized budgets."""

    PROJECT_ID = 987311

    def tearDown(self):
        import shutil

        shutil.rmtree(Path(settings.DATA_DIR) / "projects" / str(self.PROJECT_ID), ignore_errors=True)

    def test_each_chunk_asks_for_its_rows_with_a_sized_budget(self):
        import asyncio

        from app.services.synth_playbook_service import _run_chunked
        from app.services.synth_playbooks import SynthMode, get_playbook
        from test_code_findings_g4_fixes import clean_row, make_ctx

        class CloudShaped:
            max_output_tokens = 16000
            playbook_rows_per_call = 3

            def __init__(self):
                self.budgets = []

            def describe(self):
                return "cloud-shaped"

            async def complete(self, prompt, **kw):
                self.budgets.append(kw["max_tokens"])
                from app.services.synth_backends.template import code_findings_rows

                return code_findings_rows(prompt, "seeded_defect")

        gold = [clean_row(code=f"def f{i}(x):\n    return x + {i}", path=f"mod{i}.py") for i in range(8)]
        ctx = make_ctx(SynthMode.SEEDED_DEFECT, gold=gold, params={"label_id": "injection.query"}, target_count=7,
                       project_id=self.PROJECT_ID)
        backend = CloudShaped()
        with patch.dict(os.environ, {"BREWSLM_TEMPLATE_SYNTH": "1"}):
            _p, _r, rows, _e, meta, _path = asyncio.run(_run_chunked(
                get_playbook("code-findings", SynthMode.SEEDED_DEFECT), ctx, SynthMode.SEEDED_DEFECT, backend, 3,
                progress=None, recipe_id="code-findings", target_class=None, failure_cluster=None))
        self.assertEqual(len(backend.budgets), 3, "7 rows at 3 per call")
        self.assertEqual(backend.budgets, [_sized_max_tokens(backend, n, PLAYBOOK_TOKENS_PER_ROW, 2048) for n in (3, 2, 2)])
        self.assertEqual(len(rows), 7)


if __name__ == "__main__":
    unittest.main()
