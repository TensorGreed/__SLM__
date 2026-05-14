"""Phase B — schema introspector + --auto / --force CLI gate.

Pins the introspector's column-sniffer + shape-detector behaviour and
the CLI's no-silent-auto-mapping contract: a proposal below the
confidence threshold must require ``--force`` before a ``preview`` /
``run`` will accept it.
"""

from __future__ import annotations

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
from app.services.dataset_import.introspector import (  # noqa: E402
    BIO_TAG_LIST,
    CATEGORICAL,
    CHAT_MESSAGES,
    CONFIDENCE_HIGH,
    TEXT_LIKE,
    TOKENS_LIST,
    detect_shape,
    propose_mapping,
    sniff_columns,
)
from app.services.dataset_import.service import introspect_locator  # noqa: E402


def _write_jsonl(rows: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        return fh.name


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            code = cli_main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, buf.getvalue()


# ── Column sniffer ───────────────────────────────────────────────────


class ColumnSnifferTests(unittest.TestCase):
    def test_bio_tag_list_detected_over_plain_string_list(self):
        rows = [
            {
                "tokens": ["Alice", "lives", "here"],
                "labels": ["B-NAME", "O", "O"],
            },
            {
                "tokens": ["Bob", "too"],
                "labels": ["B-NAME", "O"],
            },
        ]
        sigs = sniff_columns(rows)
        self.assertEqual(sigs["tokens"].column_type, TOKENS_LIST)
        self.assertEqual(sigs["labels"].column_type, BIO_TAG_LIST)

    def test_chat_messages_detected_from_role_content_shape(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ]
        sigs = sniff_columns(rows)
        self.assertEqual(sigs["messages"].column_type, CHAT_MESSAGES)

    def test_categorical_vs_text_like_split_by_cardinality(self):
        # Long sentences in `text`; small label set in `class`.
        rows = [
            {
                "text": "the absolute best customer service I have ever experienced",
                "class": "pos",
            },
            {
                "text": "broken product arrived damaged and refund denied entirely",
                "class": "neg",
            },
            {
                "text": "exactly as described nothing more nothing less to say",
                "class": "neu",
            },
            {
                "text": "would buy again twice over honestly an amazing find",
                "class": "pos",
            },
        ]
        sigs = sniff_columns(rows)
        self.assertEqual(sigs["text"].column_type, TEXT_LIKE)
        self.assertEqual(sigs["class"].column_type, CATEGORICAL)
        self.assertEqual(
            sorted(sigs["class"].unique_values), ["neg", "neu", "pos"]
        )

    def test_all_unique_short_strings_are_not_categorical(self):
        # Every value distinct → text_like, not categorical.
        rows = [{"text": f"row {i}"} for i in range(5)]
        sigs = sniff_columns(rows)
        self.assertEqual(sigs["text"].column_type, TEXT_LIKE)


# ── Shape detector + proposal builder ────────────────────────────────


class ShapeDetectorTests(unittest.TestCase):
    def test_bio_to_spans_when_tokens_and_bio_labels_match(self):
        rows = [
            {
                "tokens": ["Alice", "loves", "Paris"],
                "labels": ["B-PERSON", "O", "B-LOC"],
            },
            {
                "tokens": ["Bob", "works", "in", "Berlin"],
                "labels": ["B-PERSON", "O", "O", "B-LOC"],
            },
        ]
        sigs = sniff_columns(rows)
        hyps = detect_shape(sigs, rows)
        self.assertTrue(hyps)
        top = hyps[0]
        self.assertEqual(top.mapper_id, "bio_to_spans")
        self.assertEqual(top.target_task_profile, "structured_extraction")
        self.assertEqual(top.field_map["tokens_field"], "tokens")
        self.assertEqual(top.field_map["labels_field"], "labels")
        self.assertGreaterEqual(top.confidence, CONFIDENCE_HIGH)

    def test_bio_to_spans_picks_full_text_alignment_column(self):
        rows = [
            {
                "tokens": ["Alice", "lives", "here"],
                "labels": ["B-PERSON", "O", "O"],
                "full_text": "Alice lives here",
            }
        ]
        sigs = sniff_columns(rows)
        hyps = detect_shape(sigs, rows)
        self.assertEqual(hyps[0].field_map.get("full_text_field"), "full_text")

    def test_classification_hypothesis_keys_off_text_plus_categorical(self):
        rows = [
            {
                "text": "the absolute best customer service ever experienced",
                "label": "positive",
            },
            {
                "text": "broken on arrival and refund denied entirely",
                "label": "negative",
            },
            {
                "text": "exactly as described nothing more nothing less",
                "label": "neutral",
            },
        ]
        sigs = sniff_columns(rows)
        hyps = detect_shape(sigs, rows)
        top = next(h for h in hyps if h.mapper_id == "label_to_classification")
        self.assertEqual(top.field_map["text_field"], "text")
        self.assertEqual(top.field_map["label_field"], "label")
        self.assertGreaterEqual(top.confidence, CONFIDENCE_HIGH)
        self.assertIn(
            "positive", top.field_map["allowed_labels"]
        )

    def test_low_confidence_proposal_when_text_too_short(self):
        # Multi-word but very short snippets land as text_like with
        # weak confidence (0.5). Paired with a categorical that
        # doesn't match a conventional label name, the classification
        # hypothesis stays below the auto-run threshold.
        rows = [
            {"snippet": "ok now", "tag": "a"},
            {"snippet": "no way", "tag": "b"},
            {"snippet": "ok now", "tag": "a"},
            {"snippet": "see ya", "tag": "c"},
        ]
        proposal = propose_mapping(rows)
        self.assertIsNotNone(proposal)
        self.assertLess(proposal.confidence, CONFIDENCE_HIGH)


# ── Service-level introspect ─────────────────────────────────────────


class IntrospectLocatorTests(unittest.TestCase):
    def test_introspect_locator_returns_proposal_for_classification_dataset(self):
        path = _write_jsonl(
            [
                {
                    "text": "the absolute best customer service ever experienced",
                    "label": "positive",
                },
                {
                    "text": "broken on arrival and refund denied entirely",
                    "label": "negative",
                },
                {
                    "text": "exactly as described nothing more nothing less",
                    "label": "neutral",
                },
                {
                    "text": "would buy again twice over a genuinely lovely product",
                    "label": "positive",
                },
            ]
        )
        try:
            payload = introspect_locator(f"jsonl:{path}")
            self.assertEqual(payload["source_id"], "jsonl")
            self.assertGreaterEqual(len(payload["hypotheses"]), 1)
            self.assertEqual(
                payload["proposal"]["mapper_id"], "label_to_classification"
            )
            # Result of sniffer is exposed so the CLI / UI can show it.
            sig_names = {s["name"] for s in payload["column_signatures"]}
            self.assertEqual(sig_names, {"text", "label"})
        finally:
            Path(path).unlink()


# ── CLI: --auto / --force gate ───────────────────────────────────────


class CliAutoFlowTests(unittest.TestCase):
    def test_introspect_subcommand_prints_proposal(self):
        path = _write_jsonl(
            [
                {
                    "tokens": ["Hi", ",", "I'm", "Alice"],
                    "labels": ["O", "O", "O", "B-NAME"],
                }
            ]
        )
        try:
            code, out = _run(
                ["introspect", "--locator", f"jsonl:{path}"]
            )
            self.assertEqual(code, 0)
            self.assertIn("bio_to_spans", out)
            self.assertIn("rationale", out)
        finally:
            Path(path).unlink()

    def test_auto_high_confidence_runs_without_force(self):
        path = _write_jsonl(
            [
                {
                    "tokens": ["Hi", ",", "I'm", "Alice"],
                    "labels": ["O", "O", "O", "B-NAME"],
                    "trailing_whitespace": [False, True, True, False],
                },
                {
                    "tokens": ["Bob", "works", "in", "Berlin"],
                    "labels": ["B-NAME", "O", "O", "B-LOC"],
                    "trailing_whitespace": [True, True, True, False],
                },
            ]
        )
        try:
            code, out = _run(
                ["preview", "--locator", f"jsonl:{path}", "--auto", "--json"]
            )
            self.assertEqual(code, 0)
            # Strip the "--auto picked …" header line, parse JSON tail.
            lines = out.splitlines()
            json_start = next(i for i, line in enumerate(lines) if line.startswith("{"))
            payload = json.loads("\n".join(lines[json_start:]))
            self.assertEqual(payload["mapper_id"], "bio_to_spans")
            self.assertEqual(payload["accepted_count"], 2)
        finally:
            Path(path).unlink()

    def test_auto_low_confidence_blocks_without_force(self):
        # Multi-word but very short snippets + non-conventional column
        # names keep the proposal below the 0.8 threshold.
        path = _write_jsonl(
            [
                {"snippet": "ok now", "tag": "a"},
                {"snippet": "no way", "tag": "b"},
                {"snippet": "ok now", "tag": "a"},
                {"snippet": "see ya", "tag": "c"},
            ]
        )
        try:
            with self.assertRaises(SystemExit) as cm:
                cli_main(
                    ["preview", "--locator", f"jsonl:{path}", "--auto"]
                )
            self.assertIn("below the 0.80 threshold", str(cm.exception))
        finally:
            Path(path).unlink()

    def test_auto_force_overrides_low_confidence(self):
        path = _write_jsonl(
            [
                {"snippet": "ok now", "tag": "a"},
                {"snippet": "no way", "tag": "b"},
                {"snippet": "ok now", "tag": "a"},
                {"snippet": "see ya", "tag": "c"},
            ]
        )
        try:
            code, out = _run(
                [
                    "preview",
                    "--locator",
                    f"jsonl:{path}",
                    "--auto",
                    "--force",
                    "--json",
                ]
            )
            self.assertEqual(code, 0)
            lines = out.splitlines()
            json_start = next(i for i, line in enumerate(lines) if line.startswith("{"))
            payload = json.loads("\n".join(lines[json_start:]))
            self.assertEqual(payload["mapper_id"], "label_to_classification")
        finally:
            Path(path).unlink()

    def test_missing_mapper_without_auto_is_rejected(self):
        path = _write_jsonl([{"text": "hi", "label": "p"}])
        try:
            with self.assertRaises(SystemExit) as cm:
                cli_main(["preview", "--locator", f"jsonl:{path}"])
            self.assertIn("--mapper is required", str(cm.exception))
        finally:
            Path(path).unlink()

    def test_explicit_map_overrides_auto_suggestion(self):
        # Source uses non-standard column names; introspector won't
        # auto-detect, but --auto + explicit --map fills the gap.
        # Confidence will be low (no convention match), so --force.
        rows = [
            {
                "review_text": "this product is wonderful, very pleased with it",
                "sentiment": "positive",
            },
            {
                "review_text": "terrible, broke after first use of the device",
                "sentiment": "negative",
            },
            {
                "review_text": "fine but nothing to write home about really",
                "sentiment": "neutral",
            },
        ]
        path = _write_jsonl(rows)
        try:
            code, out = _run(
                [
                    "preview",
                    "--locator",
                    f"jsonl:{path}",
                    "--auto",
                    "--force",
                    "--map",
                    "text_field=review_text",
                    "--map",
                    "label_field=sentiment",
                    "--json",
                ]
            )
            self.assertEqual(code, 0)
            lines = out.splitlines()
            json_start = next(i for i, line in enumerate(lines) if line.startswith("{"))
            payload = json.loads("\n".join(lines[json_start:]))
            # Explicit --map must override the auto-suggested keys.
            self.assertEqual(payload["accepted_count"], 3)
            self.assertEqual(
                payload["accepted_sample"][0]["payload"]["label"], "positive"
            )
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
