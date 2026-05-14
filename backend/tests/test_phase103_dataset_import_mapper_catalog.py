"""Phase C — expanded mapper catalog + introspector detection rules.

Pins behaviour for the six mappers shipped in Phase C:

- ``text_only`` — single-column LM passthrough
- ``qa_pair_passthrough`` — ``{question, answer}`` → QA
- ``chat_messages_passthrough`` — multi-turn chat → chat SFT
- ``preference_pair`` — DPO/ORPO triples
- ``rag_passthrough`` — grounded QA triples
- ``kv_to_structured`` — flat fields → entities for field_match scoring

Plus the introspector's new detection rules: each shape's
conventional column-name fingerprint should yield the right
hypothesis at high confidence under ``--auto``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.cli.dataset_import import main as cli_main  # noqa: E402
from app.services.dataset_import import (  # noqa: E402
    list_registered_mappers,
    resolve_mapper,
)
from app.services.dataset_import.introspector import propose_mapping  # noqa: E402
from app.services.dataset_import.protocols import (  # noqa: E402
    ImportContext,
    RejectedRow,
    TransformedRow,
)
from app.services.dataset_import.service import introspect_locator  # noqa: E402


def _write_jsonl(rows: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        return fh.name


def _ctx(mapper_id: str, source_id: str = "jsonl") -> ImportContext:
    return ImportContext(
        project_id=0,
        project_task_profile=None,
        source_id=source_id,
        mapper_id=mapper_id,
        locator=f"{source_id}:/tmp/x.jsonl",
        field_map={},
    )


def _split(items):
    accepted: list[TransformedRow] = []
    rejected: list[RejectedRow] = []
    for it in items:
        if isinstance(it, TransformedRow):
            accepted.append(it)
        else:
            rejected.append(it)
    return accepted, rejected


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            code = cli_main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, buf.getvalue()


# ── Registry ─────────────────────────────────────────────────────────


class RegistryTests(unittest.TestCase):
    def test_all_phase_c_mappers_registered(self):
        registered = set(list_registered_mappers())
        for mapper_id in (
            "text_only",
            "qa_pair_passthrough",
            "chat_messages_passthrough",
            "preference_pair",
            "rag_passthrough",
            "kv_to_structured",
        ):
            self.assertIn(mapper_id, registered)

    def test_declared_targets_match_task_handler_profiles(self):
        # If a mapper declares a profile that no handler is registered
        # for, eval would fall through to GenericHandler — that's the
        # exact bug the orchestrator's compatibility check exists to
        # catch.
        expected = {
            "text_only": "language_modeling",
            "qa_pair_passthrough": "qa",
            "chat_messages_passthrough": "chat_sft",
            "preference_pair": "dpo",
            "rag_passthrough": "rag_qa",
            "kv_to_structured": "structured_extraction",
        }
        for mapper_id, profile in expected.items():
            self.assertEqual(
                resolve_mapper(mapper_id).declared_target(), profile
            )


# ── text_only ────────────────────────────────────────────────────────


class TextOnlyMapperTests(unittest.TestCase):
    def test_emits_text_only_payload(self):
        mapper = resolve_mapper("text_only")
        rows = [
            {"text": "The first chunk of text in the corpus."},
            {"text": "  whitespace gets   collapsed  "},
        ]
        accepted, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("text_only"))
        )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected, [])
        self.assertEqual(
            accepted[0].payload, {"text": "The first chunk of text in the corpus."}
        )
        self.assertEqual(
            accepted[1].payload, {"text": "whitespace gets collapsed"}
        )

    def test_min_chars_rejects_too_short(self):
        mapper = resolve_mapper("text_only")
        rows = [
            {"text": "ok longer text"},
            {"text": "hi"},
        ]
        accepted, rejected = _split(
            mapper.transform(
                rows, {"min_chars": 5}, ctx=_ctx("text_only")
            )
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].reason, "text_too_short")

    def test_missing_text_rejected(self):
        mapper = resolve_mapper("text_only")
        rows = [{"text": ""}, {"text": "   "}]
        accepted, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("text_only"))
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(r.reason == "missing_text" for r in rejected))


# ── qa_pair_passthrough ──────────────────────────────────────────────


class QaPairPassthroughMapperTests(unittest.TestCase):
    def test_canonical_qa_columns(self):
        mapper = resolve_mapper("qa_pair_passthrough")
        rows = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]
        accepted, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("qa_pair_passthrough"))
        )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0].payload["prompt"], "Q1?")
        self.assertEqual(accepted[0].payload["reference"], "A1")
        self.assertEqual(accepted[0].payload["question"], "Q1?")
        self.assertEqual(accepted[0].payload["answer"], "A1")

    def test_fallback_column_names_kick_in(self):
        # No "question"/"answer" columns — falls back to
        # prompt/response per the same precedence eval uses.
        mapper = resolve_mapper("qa_pair_passthrough")
        rows = [{"prompt": "P1", "response": "R1"}]
        accepted, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("qa_pair_passthrough"))
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].payload["prompt"], "P1")
        self.assertEqual(accepted[0].payload["reference"], "R1")

    def test_missing_question_rejected(self):
        mapper = resolve_mapper("qa_pair_passthrough")
        accepted, rejected = _split(
            mapper.transform(
                [{"answer": "Paris"}], {}, ctx=_ctx("qa_pair_passthrough")
            )
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0].reason, "missing_question")

    def test_field_map_overrides(self):
        mapper = resolve_mapper("qa_pair_passthrough")
        rows = [{"q": "Q1?", "a": "A1"}]
        accepted, _ = _split(
            mapper.transform(
                rows,
                {"question_field": "q", "answer_field": "a"},
                ctx=_ctx("qa_pair_passthrough"),
            )
        )
        self.assertEqual(accepted[0].payload["question"], "Q1?")


# ── chat_messages_passthrough ────────────────────────────────────────


class ChatMessagesMapperTests(unittest.TestCase):
    def test_two_turn_conversation_emits_canonical_payload(self):
        mapper = resolve_mapper("chat_messages_passthrough")
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello there!"},
                ]
            }
        ]
        accepted, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("chat_messages_passthrough"))
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].payload["reference"], "Hello there!")
        self.assertEqual(accepted[0].payload["prompt"], "user: Hi")
        self.assertEqual(
            [m["role"] for m in accepted[0].payload["messages"]],
            ["user", "assistant"],
        )

    def test_no_assistant_reply_rejected_by_default(self):
        mapper = resolve_mapper("chat_messages_passthrough")
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "user", "content": "anyone?"},
                ]
            }
        ]
        accepted, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("chat_messages_passthrough"))
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0].reason, "missing_assistant_reply")

    def test_invalid_shape_rejected_with_clear_reason(self):
        mapper = resolve_mapper("chat_messages_passthrough")
        rows = [{"messages": "not a list"}, {"messages": [{"role": "user"}]}]
        accepted, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("chat_messages_passthrough"))
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0].reason, "invalid_messages_shape")
        self.assertEqual(rejected[1].reason, "no_valid_turns")

    def test_alt_content_key_value_falls_back(self):
        # Some chat datasets use "value" instead of "content".
        mapper = resolve_mapper("chat_messages_passthrough")
        rows = [
            {
                "messages": [
                    {"role": "user", "value": "Hi"},
                    {"role": "assistant", "value": "Hello"},
                ]
            }
        ]
        accepted, _ = _split(
            mapper.transform(rows, {}, ctx=_ctx("chat_messages_passthrough"))
        )
        self.assertEqual(accepted[0].payload["reference"], "Hello")


# ── preference_pair ──────────────────────────────────────────────────


class PreferencePairMapperTests(unittest.TestCase):
    def test_canonical_triple_passthrough(self):
        mapper = resolve_mapper("preference_pair")
        rows = [
            {"prompt": "Q", "chosen": "good answer", "rejected": "bad answer"}
        ]
        accepted, _ = _split(
            mapper.transform(rows, {}, ctx=_ctx("preference_pair"))
        )
        self.assertEqual(accepted[0].payload["chosen"], "good answer")
        self.assertEqual(accepted[0].payload["rejected"], "bad answer")
        # Legacy reference alias = chosen.
        self.assertEqual(accepted[0].payload["reference"], "good answer")

    def test_identical_pair_rejected(self):
        mapper = resolve_mapper("preference_pair")
        rows = [{"prompt": "Q", "chosen": "same", "rejected": "same"}]
        _, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("preference_pair"))
        )
        self.assertEqual(rejected[0].reason, "identical_pair")

    def test_alt_field_names_via_fallback(self):
        mapper = resolve_mapper("preference_pair")
        rows = [{"prompt": "Q", "preferred": "good", "negative": "bad"}]
        accepted, _ = _split(
            mapper.transform(rows, {}, ctx=_ctx("preference_pair"))
        )
        self.assertEqual(accepted[0].payload["chosen"], "good")
        self.assertEqual(accepted[0].payload["rejected"], "bad")


# ── rag_passthrough ──────────────────────────────────────────────────


class RagPassthroughMapperTests(unittest.TestCase):
    def test_canonical_rag_triple(self):
        mapper = resolve_mapper("rag_passthrough")
        rows = [
            {
                "question": "Where is Paris?",
                "context": "Paris is the capital of France.",
                "answer": "France",
            }
        ]
        accepted, _ = _split(
            mapper.transform(rows, {}, ctx=_ctx("rag_passthrough"))
        )
        self.assertEqual(accepted[0].payload["question"], "Where is Paris?")
        self.assertEqual(
            accepted[0].payload["context"], "Paris is the capital of France."
        )
        self.assertEqual(accepted[0].payload["answer"], "France")
        # Legacy aliases for non-RAG handlers reading the same row.
        self.assertEqual(accepted[0].payload["prompt"], "Where is Paris?")
        self.assertEqual(accepted[0].payload["reference"], "France")

    def test_missing_context_rejected_distinctly(self):
        mapper = resolve_mapper("rag_passthrough")
        rows = [{"question": "Q", "answer": "A"}]
        _, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("rag_passthrough"))
        )
        self.assertEqual(rejected[0].reason, "missing_context")


# ── kv_to_structured ─────────────────────────────────────────────────


class KvToStructuredMapperTests(unittest.TestCase):
    def test_list_fields_emits_entity_array(self):
        mapper = resolve_mapper("kv_to_structured")
        rows = [
            {
                "text": "Invoice INV-001 — Total: $42",
                "invoice_number": "INV-001",
                "total": "42",
                "vendor": "Acme",
            }
        ]
        accepted, _ = _split(
            mapper.transform(
                rows,
                {"fields": ["invoice_number", "total", "vendor"]},
                ctx=_ctx("kv_to_structured"),
            )
        )
        self.assertEqual(len(accepted), 1)
        entities = json.loads(accepted[0].payload["entities_json"])["entities"]
        self.assertEqual(
            [e["field"] for e in entities],
            ["invoice_number", "total", "vendor"],
        )
        self.assertEqual(
            [e["value"] for e in entities], ["INV-001", "42", "Acme"]
        )

    def test_dict_fields_remaps_source_columns(self):
        mapper = resolve_mapper("kv_to_structured")
        rows = [{"text": "Body", "InvoiceNumber": "INV-9", "Total": "1.50"}]
        accepted, _ = _split(
            mapper.transform(
                rows,
                {
                    "fields": {
                        "invoice_number": "InvoiceNumber",
                        "total": "Total",
                    }
                },
                ctx=_ctx("kv_to_structured"),
            )
        )
        entities = json.loads(accepted[0].payload["entities_json"])["entities"]
        self.assertEqual(entities[0]["field"], "invoice_number")
        self.assertEqual(entities[0]["value"], "INV-9")

    def test_empty_fields_dropped_by_default(self):
        mapper = resolve_mapper("kv_to_structured")
        rows = [{"text": "Body", "a": "x", "b": ""}]
        accepted, _ = _split(
            mapper.transform(
                rows,
                {"fields": ["a", "b"]},
                ctx=_ctx("kv_to_structured"),
            )
        )
        entities = json.loads(accepted[0].payload["entities_json"])["entities"]
        self.assertEqual([e["field"] for e in entities], ["a"])

    def test_skip_empty_fields_false_keeps_empty(self):
        mapper = resolve_mapper("kv_to_structured")
        rows = [{"text": "Body", "a": "x", "b": ""}]
        accepted, _ = _split(
            mapper.transform(
                rows,
                {"fields": ["a", "b"], "skip_empty_fields": False},
                ctx=_ctx("kv_to_structured"),
            )
        )
        entities = json.loads(accepted[0].payload["entities_json"])["entities"]
        self.assertEqual([e["field"] for e in entities], ["a", "b"])

    def test_missing_fields_config_rejects_every_row(self):
        mapper = resolve_mapper("kv_to_structured")
        rows = [{"text": "Body"}, {"text": "Body 2"}]
        _, rejected = _split(
            mapper.transform(rows, {}, ctx=_ctx("kv_to_structured"))
        )
        self.assertEqual(len(rejected), 2)
        self.assertTrue(
            all(r.reason == "missing_fields_config" for r in rejected)
        )


# ── Introspector — Phase C detection rules ──────────────────────────


class IntrospectorPhaseCTests(unittest.TestCase):
    def test_qa_pair_detected_from_question_answer_columns(self):
        proposal = propose_mapping(
            [
                {
                    "question": "What is the capital of France?",
                    "answer": "Paris is the capital of France.",
                },
                {
                    "question": "Who wrote Hamlet?",
                    "answer": "William Shakespeare wrote Hamlet.",
                },
            ]
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.mapper_id, "qa_pair_passthrough")
        self.assertGreaterEqual(proposal.confidence, 0.8)

    def test_rag_detected_with_context_column(self):
        proposal = propose_mapping(
            [
                {
                    "question": "Where is Paris?",
                    "context": "Paris is the capital of France in Europe.",
                    "answer": "France",
                },
                {
                    "question": "When did WWII end?",
                    "context": "WWII ended in 1945 after Japan surrendered.",
                    "answer": "1945",
                },
            ]
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.mapper_id, "rag_passthrough")

    def test_preference_pair_detected_from_canonical_triple(self):
        proposal = propose_mapping(
            [
                {
                    "prompt": "Write a haiku about rain.",
                    "chosen": "Soft rain drums slowly on the roof tiles in the spring.",
                    "rejected": "It rains. Wet. Cold.",
                },
                {
                    "prompt": "Explain photosynthesis.",
                    "chosen": "Plants convert sunlight, water, and CO2 to glucose and oxygen.",
                    "rejected": "Plants eat sun.",
                },
            ]
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.mapper_id, "preference_pair")
        self.assertGreaterEqual(proposal.confidence, 0.8)

    def test_chat_messages_detected_from_messages_column(self):
        proposal = propose_mapping(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                    ]
                }
            ]
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.mapper_id, "chat_messages_passthrough")

    def test_text_only_when_no_label_or_structured_signal(self):
        proposal = propose_mapping(
            [
                {"text": "Long-form text for language model pretraining one."},
                {"text": "Another paragraph of text for the corpus, no labels."},
                {"text": "A third chunk of corpus text — purely unsupervised."},
            ]
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.mapper_id, "text_only")

    def test_rag_outranks_classification_when_context_present(self):
        # Even with short answers that look categorical on a small
        # sample, RAG should win because of the conventional "context"
        # column name.
        payload = asyncio.run(introspect_locator(
            "jsonl:" + _write_jsonl(
                [
                    {
                        "question": "Where is Paris?",
                        "context": "Paris is the capital of France.",
                        "answer": "France",
                    },
                    {
                        "question": "When did WWII end?",
                        "context": "World War II ended in 1945.",
                        "answer": "1945",
                    },
                ]
            )
        ))
        self.assertEqual(payload["proposal"]["mapper_id"], "rag_passthrough")


# ── CLI ──────────────────────────────────────────────────────────────


class CliEndToEndTests(unittest.TestCase):
    def test_mappers_subcommand_lists_phase_c_catalog(self):
        code, out = _run_cli(["mappers"])
        self.assertEqual(code, 0)
        for mapper_id in (
            "text_only",
            "qa_pair_passthrough",
            "chat_messages_passthrough",
            "preference_pair",
            "rag_passthrough",
            "kv_to_structured",
        ):
            self.assertIn(mapper_id, out)

    def test_auto_preview_on_preference_pair(self):
        path = _write_jsonl(
            [
                {
                    "prompt": "Write a haiku about rain.",
                    "chosen": "Soft rain drums slowly on the roof tiles.",
                    "rejected": "Rain bad. Wet.",
                },
                {
                    "prompt": "Explain photosynthesis briefly.",
                    "chosen": "Plants convert sunlight to glucose and oxygen.",
                    "rejected": "Plants eat sun.",
                },
            ]
        )
        try:
            code, out = _run_cli(
                ["preview", "--locator", f"jsonl:{path}", "--auto", "--json"]
            )
            self.assertEqual(code, 0)
            lines = out.splitlines()
            json_start = next(i for i, line in enumerate(lines) if line.startswith("{"))
            payload = json.loads("\n".join(lines[json_start:]))
            self.assertEqual(payload["mapper_id"], "preference_pair")
            self.assertEqual(payload["accepted_count"], 2)
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
