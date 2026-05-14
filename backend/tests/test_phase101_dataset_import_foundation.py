"""Phase A of DATASET_IMPORT_PLAN.md — foundation.

Pins the contracts every later phase relies on:

- Source + mapper registries dispatch on id, never silently pick.
- JSONL + CSV source loaders stream rows lazily, surface parse errors
  as rejection-worthy sentinel rows rather than crashing.
- bio_to_spans mapper generalizes the kaggle PII converter: handles
  arbitrary entity types via config, reconstructs offsets from either
  full_text alignment or tokens+trailing-whitespace fallback, rejects
  malformed rows with stable reason codes.
- label_to_classification mapper coerces non-string labels (bool/int)
  to canonical strings, filters by allowed_labels when supplied,
  rejects with reason codes for missing fields / out-of-set labels.
- Orchestrator's preview_import returns capped accepted sample + full
  rejected list + rejection counts grouped by reason — the load-
  bearing shape for the bulk-drop UX contract.
- Drop-reasons filter removes whole rejection categories from the
  surfaced rejected_sample but keeps counts intact for transparency.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.dataset_import import (  # noqa: E402
    ImportContext,
    RejectedRow,
    TransformedRow,
    list_registered_mappers,
    list_registered_sources,
    resolve_mapper,
    resolve_source,
    split_locator,
)
from app.services.dataset_import.service import preview_import  # noqa: E402


def _ctx(
    *,
    source_id: str = "jsonl",
    mapper_id: str = "bio_to_spans",
    locator: str = "jsonl:/tmp/test.jsonl",
    task_profile: str | None = None,
    field_map: dict | None = None,
) -> ImportContext:
    return ImportContext(
        project_id=0,
        project_task_profile=task_profile,
        source_id=source_id,
        mapper_id=mapper_id,
        locator=locator,
        field_map=field_map or {},
    )


class RegistryTests(unittest.TestCase):
    def test_built_in_sources_registered(self):
        sources = list_registered_sources()
        self.assertIn("jsonl", sources)
        self.assertIn("csv", sources)

    def test_built_in_mappers_registered(self):
        mappers = list_registered_mappers()
        self.assertIn("bio_to_spans", mappers)
        self.assertIn("label_to_classification", mappers)

    def test_resolve_unknown_source_raises_helpful_keyerror(self):
        with self.assertRaises(KeyError) as cm:
            resolve_source("nonexistent_source")
        # Error message names the registered alternatives so callers
        # can surface a useful error to the user.
        self.assertIn("Registered sources:", str(cm.exception))

    def test_resolve_unknown_mapper_raises_helpful_keyerror(self):
        with self.assertRaises(KeyError) as cm:
            resolve_mapper("nonexistent_mapper")
        self.assertIn("Registered mappers:", str(cm.exception))

    def test_split_locator_parses_prefix(self):
        self.assertEqual(split_locator("jsonl:/path/file"), ("jsonl", "/path/file"))
        # Multi-colon locators keep everything after the first colon as rest.
        self.assertEqual(
            split_locator("hf:ai4privacy/pii-masking-200k:train"),
            ("hf", "ai4privacy/pii-masking-200k:train"),
        )

    def test_split_locator_rejects_missing_prefix(self):
        with self.assertRaises(ValueError):
            split_locator("/path/without/prefix")


class JsonlSourceTests(unittest.TestCase):
    def test_streams_object_rows(self):
        source = resolve_source("jsonl")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"a": 1}) + "\n")
            fh.write(json.dumps({"b": 2}) + "\n")
            path = fh.name
        try:
            rows = list(source.load(path))
            self.assertEqual(rows, [{"a": 1}, {"b": 2}])
        finally:
            Path(path).unlink()

    def test_unparseable_lines_become_sentinel_rows(self):
        # Per the source contract: never silently drop. Bad lines turn
        # into sentinel rows with __parse_error__ so the mapper can
        # reject them with a stable reason code.
        source = resolve_source("jsonl")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"ok": True}) + "\n")
            fh.write("definitely not json\n")
            fh.write(json.dumps({"ok": False}) + "\n")
            path = fh.name
        try:
            rows = list(source.load(path))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], {"ok": True})
            self.assertIn("__parse_error__", rows[1])
            self.assertEqual(rows[2], {"ok": False})
        finally:
            Path(path).unlink()

    def test_limit_truncates_streaming(self):
        source = resolve_source("jsonl")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            for i in range(100):
                fh.write(json.dumps({"i": i}) + "\n")
            path = fh.name
        try:
            rows = list(source.load(path, limit=5))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[-1]["i"], 4)
        finally:
            Path(path).unlink()

    def test_describe_returns_columns_and_sample(self):
        source = resolve_source("jsonl")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"text": "hi", "label": "pos"}) + "\n")
            fh.write(json.dumps({"text": "bye", "label": "neg"}) + "\n")
            path = fh.name
        try:
            meta = source.describe(path)
            self.assertEqual(set(meta["columns"]), {"text", "label"})
            self.assertEqual(meta["approximate_total_rows"], 2)
            self.assertEqual(len(meta["sample_rows"]), 2)
        finally:
            Path(path).unlink()

    def test_missing_file_raises_filenotfound(self):
        source = resolve_source("jsonl")
        with self.assertRaises(FileNotFoundError):
            list(source.load("/nonexistent/path/that-does-not-exist.jsonl"))


class CsvSourceTests(unittest.TestCase):
    def test_uses_first_row_as_header(self):
        source = resolve_source("csv")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fh:
            fh.write("text,label\n")
            fh.write("hello,positive\n")
            fh.write("bye,negative\n")
            path = fh.name
        try:
            rows = list(source.load(path))
            self.assertEqual(rows[0], {"text": "hello", "label": "positive"})
            self.assertEqual(rows[1], {"text": "bye", "label": "negative"})
        finally:
            Path(path).unlink()

    def test_describe_returns_columns_and_sample(self):
        source = resolve_source("csv")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fh:
            fh.write("a,b,c\n")
            fh.write("1,2,3\n")
            path = fh.name
        try:
            meta = source.describe(path)
            self.assertEqual(meta["columns"], ["a", "b", "c"])
            self.assertEqual(meta["approximate_total_rows"], 1)
        finally:
            Path(path).unlink()


class BioToSpansMapperTests(unittest.TestCase):
    def test_full_text_alignment_path(self):
        mapper = resolve_mapper("bio_to_spans")
        rows = [
            {
                "document": 7,
                "tokens": ["Hi", ",", "my", "name", "is", "John"],
                "labels": ["O", "O", "O", "O", "O", "B-NAME_STUDENT"],
                "trailing_whitespace": [False, True, True, True, True, True],
                "full_text": "Hi, my name is John ",
            }
        ]
        results = list(
            mapper.transform(
                rows,
                {"entity_type_map": {"NAME_STUDENT": "person_name"}},
                ctx=_ctx(),
            )
        )
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], TransformedRow)
        payload = results[0].payload
        entities = json.loads(payload["entities_json"])["entities"]
        self.assertEqual(len(entities), 1)
        ent = entities[0]
        # Offset sanity: text[start:end] must equal claimed text.
        self.assertEqual(payload["text"][ent["start"] : ent["end"]], ent["text"])
        self.assertEqual(ent["type"], "person_name")
        self.assertEqual(ent["text"], "John")

    def test_token_fallback_when_full_text_absent(self):
        mapper = resolve_mapper("bio_to_spans")
        rows = [
            {
                "tokens": ["Email", "is", "foo@example.com"],
                "labels": ["O", "O", "B-EMAIL"],
                "trailing_whitespace": [True, True, False],
            }
        ]
        results = list(mapper.transform(rows, {"entity_type_map": {"EMAIL": "email"}}, ctx=_ctx()))
        self.assertEqual(len(results), 1)
        payload = results[0].payload
        entities = json.loads(payload["entities_json"])["entities"]
        self.assertEqual(entities[0]["type"], "email")
        # Offsets reconstructed correctly.
        self.assertEqual(
            payload["text"][entities[0]["start"] : entities[0]["end"]],
            entities[0]["text"],
        )

    def test_merges_b_and_i_into_single_span(self):
        mapper = resolve_mapper("bio_to_spans")
        rows = [
            {
                "tokens": ["I", "am", "John", "Smith"],
                "labels": ["O", "O", "B-NAME", "I-NAME"],
                "trailing_whitespace": [True, True, True, False],
            }
        ]
        results = list(mapper.transform(rows, {}, ctx=_ctx()))
        entities = json.loads(results[0].payload["entities_json"])["entities"]
        # Single span covering "John Smith", not two separate tokens.
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["text"], "John Smith")

    def test_missing_tokens_rejected_with_stable_code(self):
        mapper = resolve_mapper("bio_to_spans")
        results = list(
            mapper.transform([{"labels": ["O"]}], {}, ctx=_ctx())
        )
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RejectedRow)
        self.assertEqual(results[0].reason, "missing_tokens")

    def test_length_mismatch_rejected_with_stable_code(self):
        mapper = resolve_mapper("bio_to_spans")
        rows = [{"tokens": ["a", "b"], "labels": ["O", "O", "O"]}]
        results = list(mapper.transform(rows, {}, ctx=_ctx()))
        self.assertEqual(results[0].reason, "length_mismatch")

    def test_parse_error_sentinel_rejected_with_stable_code(self):
        mapper = resolve_mapper("bio_to_spans")
        # The JSONL source's bad-line sentinel — mapper must surface it
        # as parse_error rather than crashing on the missing tokens.
        rows = [{"__parse_error__": "invalid_json", "__raw_line__": "..."}]
        results = list(mapper.transform(rows, {}, ctx=_ctx()))
        self.assertEqual(results[0].reason, "parse_error")

    def test_declared_target_is_structured_extraction(self):
        self.assertEqual(resolve_mapper("bio_to_spans").declared_target(), "structured_extraction")


class LabelToClassificationMapperTests(unittest.TestCase):
    def test_passthrough_for_clean_rows(self):
        mapper = resolve_mapper("label_to_classification")
        results = list(
            mapper.transform(
                [{"text": "Great product!", "label": "positive"}],
                {},
                ctx=_ctx(),
            )
        )
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], TransformedRow)
        self.assertEqual(
            results[0].payload, {"text": "Great product!", "label": "positive"}
        )

    def test_coerces_int_labels_to_strings(self):
        # Datasets with 0/1 binary labels are common; we coerce to
        # canonical strings so the classifier handler sees consistent
        # types.
        mapper = resolve_mapper("label_to_classification")
        results = list(
            mapper.transform(
                [{"text": "hi", "label": 1}, {"text": "bye", "label": 0}],
                {},
                ctx=_ctx(),
            )
        )
        self.assertEqual(results[0].payload["label"], "1")
        self.assertEqual(results[1].payload["label"], "0")

    def test_collapses_whitespace_in_text_and_label(self):
        mapper = resolve_mapper("label_to_classification")
        results = list(
            mapper.transform(
                [{"text": "  hello\n\n  world  ", "label": "  positive\n"}],
                {},
                ctx=_ctx(),
            )
        )
        self.assertEqual(results[0].payload["text"], "hello world")
        self.assertEqual(results[0].payload["label"], "positive")

    def test_filters_to_allowed_labels(self):
        mapper = resolve_mapper("label_to_classification")
        rows = [
            {"text": "x", "label": "positive"},
            {"text": "y", "label": "weird"},
            {"text": "z", "label": "neutral"},
        ]
        results = list(
            mapper.transform(
                rows,
                {"allowed_labels": ["positive", "neutral"]},
                ctx=_ctx(),
            )
        )
        # Two accepted, one rejected.
        accepted = [r for r in results if isinstance(r, TransformedRow)]
        rejected = [r for r in results if isinstance(r, RejectedRow)]
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].reason, "label_not_allowed")

    def test_missing_fields_rejected_with_stable_codes(self):
        mapper = resolve_mapper("label_to_classification")
        results = list(
            mapper.transform(
                [
                    {"label": "positive"},  # missing text
                    {"text": "hi"},  # missing label
                ],
                {},
                ctx=_ctx(),
            )
        )
        self.assertEqual(results[0].reason, "missing_text")
        self.assertEqual(results[1].reason, "missing_label")

    def test_remaps_field_names(self):
        # Source has weird column names → field_map remaps them.
        mapper = resolve_mapper("label_to_classification")
        results = list(
            mapper.transform(
                [{"review_text": "great!", "sentiment": "positive"}],
                {"text_field": "review_text", "label_field": "sentiment"},
                ctx=_ctx(),
            )
        )
        self.assertEqual(results[0].payload["text"], "great!")
        self.assertEqual(results[0].payload["label"], "positive")

    def test_declared_target_is_classification(self):
        self.assertEqual(
            resolve_mapper("label_to_classification").declared_target(),
            "classification",
        )


class PreviewImportTests(unittest.TestCase):
    """End-to-end pipeline through preview_import.

    Pins the bulk-drop UX contract: rejection_counts surfaces all
    reasons grouped by code; drop_reasons filters whole categories out
    of the surfaced rejected sample while still counting them.
    """

    def _write_jsonl(self, rows: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            return fh.name

    def test_label_to_classification_end_to_end(self):
        path = self._write_jsonl(
            [
                {"text": "Great!", "label": "positive"},
                {"text": "Awful.", "label": "negative"},
            ]
        )
        try:
            result = preview_import(
                project_id=0,
                project_task_profile=None,
                locator=f"jsonl:{path}",
                mapper_id="label_to_classification",
                field_map={},
            )
            self.assertTrue(result.dry_run)
            self.assertEqual(result.accepted_count, 2)
            self.assertEqual(result.rejected_count, 0)
            self.assertEqual(result.target_task_profile, "classification")
            self.assertEqual(result.source_id, "jsonl")
        finally:
            Path(path).unlink()

    def test_rejections_grouped_by_reason(self):
        path = self._write_jsonl(
            [
                {"text": "ok", "label": "pos"},
                {"text": "ok"},  # missing_label
                {"label": "pos"},  # missing_text
                {"label": "pos"},  # missing_text again
            ]
        )
        try:
            result = preview_import(
                project_id=0,
                project_task_profile=None,
                locator=f"jsonl:{path}",
                mapper_id="label_to_classification",
                field_map={},
            )
            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.rejected_count, 3)
            # Reasons grouped, counts accurate.
            self.assertEqual(result.rejection_counts["missing_text"], 2)
            self.assertEqual(result.rejection_counts["missing_label"], 1)
        finally:
            Path(path).unlink()

    def test_drop_reasons_filters_rejected_sample_but_keeps_counts(self):
        # Bulk-drop UX contract: counts still reflect the full set so
        # users see "you dropped N malformed rows", but the surfaced
        # rejected_sample only contains the categories the user wants
        # to inspect.
        path = self._write_jsonl(
            [
                {"text": "ok", "label": "pos"},
                {"text": "ok"},  # missing_label — to be inspected
                {"label": "pos"},  # missing_text — to be dropped
                {"label": "pos"},  # missing_text — to be dropped
            ]
        )
        try:
            result = preview_import(
                project_id=0,
                project_task_profile=None,
                locator=f"jsonl:{path}",
                mapper_id="label_to_classification",
                field_map={},
                drop_reasons={"missing_text"},
            )
            # rejection_counts is unchanged — full tally.
            self.assertEqual(result.rejection_counts["missing_text"], 2)
            self.assertEqual(result.rejection_counts["missing_label"], 1)
            # rejected_count = sum(rejection_counts.values()) — also full.
            self.assertEqual(result.rejected_count, 3)
            # But the surfaced sample only contains the non-dropped
            # reasons so the user can focus.
            reasons_in_sample = {r.reason for r in result.rejected_rows}
            self.assertNotIn("missing_text", reasons_in_sample)
            self.assertIn("missing_label", reasons_in_sample)
        finally:
            Path(path).unlink()

    def test_unknown_source_raises_keyerror(self):
        with self.assertRaises(KeyError):
            preview_import(
                project_id=0,
                project_task_profile=None,
                locator="bogus:something",
                mapper_id="label_to_classification",
                field_map={},
            )

    def test_unknown_mapper_raises_keyerror(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"a": 1}) + "\n")
            path = fh.name
        try:
            with self.assertRaises(KeyError):
                preview_import(
                    project_id=0,
                    project_task_profile=None,
                    locator=f"jsonl:{path}",
                    mapper_id="bogus_mapper",
                    field_map={},
                )
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
