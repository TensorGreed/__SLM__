"""The template backend — a deterministic stand-in for a generation / judge
model (CI + smoke). Pins: never auto-picked without the env flag; answers
the documents → Q&A flow's prompt with pairs lifted from the passage; answers
the judge prompt by token overlap; refuses anything else."""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest import mock

os.environ.setdefault("DEBUG", "false")

from app.services.answer_judge_service import build_judge_prompt, parse_verdict
from app.services.documents_qa_flow_service import build_passage_prompt, parse_passage_output
from app.services.synth_backends import BACKEND_REGISTRY, SynthBackendError, pick_backend
from app.services.synth_backends.template import TemplateBackend, judge_verdict, passage_pairs

PASSAGE = (
    "Section 4. The registrar shall keep a disclosure register under section 4 and shall review it within "
    "thirty days. A record kept under this section forms part of the registrar's file. The registrar shall "
    "notify the applicant in writing when the review is complete."
)


class TemplateBackendTests(unittest.TestCase):
    def test_only_available_with_the_flag(self):
        with mock.patch.dict(os.environ, {"BREWSLM_TEMPLATE_SYNTH": ""}):
            self.assertFalse(TemplateBackend.is_available())
            with self.assertRaises(SynthBackendError):
                pick_backend("template")
        with mock.patch.dict(os.environ, {"BREWSLM_TEMPLATE_SYNTH": "1"}):
            self.assertTrue(TemplateBackend.is_available())
            self.assertEqual(pick_backend("template").name, "template")
        self.assertIs(BACKEND_REGISTRY[-1], TemplateBackend, "auto-pick order: last")

    def test_flow_prompt_gets_pairs_from_the_passage(self):
        raw = asyncio.run(TemplateBackend().complete(build_passage_prompt(PASSAGE, 3, domain_hint="law")))
        pairs, eval_pair = parse_passage_output(raw, 3)
        self.assertEqual(len(pairs), 3)
        self.assertIsNotNone(eval_pair)
        self.assertTrue(all(p["answer"] in PASSAGE for p in pairs))
        self.assertNotIn(eval_pair["question"], {p["question"] for p in pairs})
        self.assertEqual(passage_pairs(PASSAGE, 2), passage_pairs(PASSAGE, 2), "deterministic")

    def test_judge_prompt_gets_an_overlap_verdict(self):
        reference = "The registrar shall keep a disclosure register and review it within thirty days."
        for prediction, expected in (
            ("The registrar must keep a disclosure register and review it within thirty days.", "correct"),
            ("The registrar keeps a register.", "partial"),
            ("Fishing licences are issued by the minister.", "wrong"),
        ):
            raw = asyncio.run(TemplateBackend().complete(build_judge_prompt("Q?", reference, prediction), system_prompt="judge"))
            score, verdict, reason = parse_verdict(raw)
            self.assertEqual(verdict, expected, (prediction, reason))
        self.assertEqual(judge_verdict("a b c d", "")["verdict"], "wrong")

    def test_anything_else_is_refused(self):
        with self.assertRaises(SynthBackendError):
            asyncio.run(TemplateBackend().complete("Write me a poem about fine-tuning."))


if __name__ == "__main__":
    unittest.main()
