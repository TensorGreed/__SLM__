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
