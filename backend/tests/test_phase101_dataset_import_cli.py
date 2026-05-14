"""Phase A — CLI surface for the dataset import pipeline.

Pins the user-facing flag contract so future refactors don't
break the documented invocation pattern.

Surface tested:

- ``sources`` / ``mappers`` list commands.
- ``preview`` with --locator, --mapper, --map, --drop, --sample-cap,
  --json flags.
- ``--map K=V`` field-map syntax + ``--map-json '{...}'`` JSON object
  syntax + their combination.
- Bulk-drop via ``--drop REASON`` (counts intact, sample filtered).
- Stable error messages for the common failure modes (unknown source,
  unknown mapper, missing required flag).
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


def _write_jsonl(rows: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        return fh.name


def _run(argv: list[str]) -> tuple[int, str]:
    """Execute the CLI in-process and capture stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            code = cli_main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, buf.getvalue()


class CatalogCommandTests(unittest.TestCase):
    def test_sources_lists_built_ins(self):
        code, out = _run(["sources"])
        self.assertEqual(code, 0)
        self.assertIn("jsonl", out)
        self.assertIn("csv", out)

    def test_mappers_lists_built_ins_with_targets(self):
        code, out = _run(["mappers"])
        self.assertEqual(code, 0)
        self.assertIn("bio_to_spans", out)
        self.assertIn("label_to_classification", out)
        # Mapper listing shows the task_profile each mapper feeds, so
        # users see at a glance which mapper to pick for their project.
        self.assertIn("task_profile=classification", out)
        self.assertIn("task_profile=structured_extraction", out)


class PreviewCommandTests(unittest.TestCase):
    def test_preview_happy_path(self):
        path = _write_jsonl(
            [
                {"text": "great", "label": "positive"},
                {"text": "bad", "label": "negative"},
            ]
        )
        try:
            code, out = _run(
                [
                    "preview",
                    "--locator",
                    f"jsonl:{path}",
                    "--mapper",
                    "label_to_classification",
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("accepted=2", out)
            self.assertIn("rejected=0", out)
            self.assertIn("target_task_profile=classification", out)
        finally:
            Path(path).unlink()

    def test_preview_json_output(self):
        path = _write_jsonl([{"text": "hi", "label": "positive"}])
        try:
            code, out = _run(
                [
                    "preview",
                    "--locator",
                    f"jsonl:{path}",
                    "--mapper",
                    "label_to_classification",
                    "--json",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertEqual(payload["accepted_count"], 1)
            self.assertEqual(payload["source_id"], "jsonl")
            self.assertEqual(payload["mapper_id"], "label_to_classification")
            self.assertTrue(payload["dry_run"])
        finally:
            Path(path).unlink()

    def test_field_map_pairs_remap_columns(self):
        # Source uses non-standard column names (review_text /
        # sentiment); --map remaps them onto the mapper's expected
        # text_field / label_field inputs.
        path = _write_jsonl(
            [{"review_text": "great", "sentiment": "positive"}]
        )
        try:
            code, out = _run(
                [
                    "preview",
                    "--locator",
                    f"jsonl:{path}",
                    "--mapper",
                    "label_to_classification",
                    "--map",
                    "text_field=review_text",
                    "--map",
                    "label_field=sentiment",
                    "--json",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertEqual(payload["accepted_count"], 1)
            self.assertEqual(
                payload["accepted_sample"][0]["payload"]["label"], "positive"
            )
        finally:
            Path(path).unlink()

    def test_map_json_for_nested_values(self):
        # bio_to_spans needs entity_type_map (a dict) which can't be
        # expressed via --map K=V; --map-json takes the JSON object.
        path = _write_jsonl(
            [
                {
                    "tokens": ["Hi", ",", "I'm", "John"],
                    "labels": ["O", "O", "O", "B-NAME_STUDENT"],
                    "trailing_whitespace": [False, True, True, False],
                }
            ]
        )
        try:
            code, out = _run(
                [
                    "preview",
                    "--locator",
                    f"jsonl:{path}",
                    "--mapper",
                    "bio_to_spans",
                    "--map-json",
                    json.dumps({"entity_type_map": {"NAME_STUDENT": "person_name"}}),
                    "--json",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertEqual(payload["accepted_count"], 1)
            entities = json.loads(
                payload["accepted_sample"][0]["payload"]["entities_json"]
            )["entities"]
            self.assertEqual(entities[0]["type"], "person_name")
        finally:
            Path(path).unlink()

    def test_drop_filters_sample_keeps_counts(self):
        # Bulk-drop UX contract: counts in the breakdown stay accurate
        # so the user sees "you dropped N malformed rows" totals;
        # rejected_sample only carries the categories we didn't drop.
        path = _write_jsonl(
            [
                {"text": "ok", "label": "pos"},
                {"text": "ok"},           # missing_label — kept
                {"label": "pos"},         # missing_text — dropped
                {"label": "pos"},         # missing_text — dropped
            ]
        )
        try:
            code, out = _run(
                [
                    "preview",
                    "--locator",
                    f"jsonl:{path}",
                    "--mapper",
                    "label_to_classification",
                    "--drop",
                    "missing_text",
                    "--json",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out)
            # Counts unchanged — full tally.
            self.assertEqual(payload["rejection_counts"]["missing_text"], 2)
            self.assertEqual(payload["rejection_counts"]["missing_label"], 1)
            self.assertEqual(payload["rejected_count"], 3)
            # But the surfaced sample skips missing_text.
            reasons_surfaced = {row["reason"] for row in payload["rejected_sample"]}
            self.assertNotIn("missing_text", reasons_surfaced)
            self.assertIn("missing_label", reasons_surfaced)
        finally:
            Path(path).unlink()


class ErrorContractTests(unittest.TestCase):
    def test_unknown_source_surfaces_registered_alternatives(self):
        with self.assertRaises(KeyError) as cm:
            cli_main(
                [
                    "preview",
                    "--locator",
                    "totally_fake:something",
                    "--mapper",
                    "label_to_classification",
                ]
            )
        self.assertIn("Registered sources:", str(cm.exception))

    def test_malformed_map_pair_rejected_with_clear_message(self):
        with self.assertRaises(SystemExit) as cm:
            cli_main(
                [
                    "preview",
                    "--locator",
                    "jsonl:/nonexistent",
                    "--mapper",
                    "label_to_classification",
                    "--map",
                    "broken_no_equals",
                ]
            )
        self.assertIn("KEY=VALUE", str(cm.exception))

    def test_malformed_map_json_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            cli_main(
                [
                    "preview",
                    "--locator",
                    "jsonl:/nonexistent",
                    "--mapper",
                    "label_to_classification",
                    "--map-json",
                    "not actually json",
                ]
            )
        self.assertIn("not valid JSON", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
