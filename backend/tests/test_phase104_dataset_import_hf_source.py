"""Phase D — HuggingFace source connector.

Pins locator parsing, streaming-load semantics, multi-split fallback,
auth-token resolution, and the missing-package error contract.

All tests stub ``datasets.load_dataset`` so the suite runs offline.
The real library is exercised via the manual smoke test documented
in DATASET_IMPORT_PLAN.md Phase D.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest import mock

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.dataset_import import (  # noqa: E402
    list_registered_sources,
    resolve_source,
    split_locator,
)
from app.services.dataset_import.sources.hf import (  # noqa: E402
    HuggingFaceSource,
    _materialize_first_split,
    _parse_locator,
    _resolve_token,
)


# ── Fake HF Dataset / DatasetDict ────────────────────────────────────


class _FakeIterableDataset:
    """Mimics ``datasets.IterableDataset`` enough for our connector
    code paths (iteration; the ``features`` attribute lets
    ``_materialize_first_split`` distinguish it from DatasetDict)."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.features = {"_marker": None}  # any non-None hits the type guard

    def __iter__(self):
        for row in self._rows:
            yield dict(row)


class _FakeDatasetDict(dict):
    """Mimics ``datasets.DatasetDict`` — dict-like, no ``features``."""

    # No ``features`` attribute — that's how _materialize_first_split
    # detects multi-split returns from load_dataset.


# ── Locator parsing ──────────────────────────────────────────────────


class LocatorParsingTests(unittest.TestCase):
    def test_dataset_id_only(self):
        self.assertEqual(
            _parse_locator("Anthropic/hh-rlhf"),
            ("Anthropic/hh-rlhf", None, None),
        )

    def test_dataset_id_with_split(self):
        self.assertEqual(
            _parse_locator("imdb:train"),
            ("imdb", "train", None),
        )

    def test_dataset_id_with_split_and_revision(self):
        self.assertEqual(
            _parse_locator("ai4privacy/pii-masking-200k:train:abc1234"),
            ("ai4privacy/pii-masking-200k", "train", "abc1234"),
        )

    def test_dataset_id_with_slash_preserved(self):
        # The ``/`` in the org/dataset id must survive parsing — the
        # registry's split_locator already partitioned on the first ':'.
        dataset_id, split, _ = _parse_locator("OpenAssistant/oasst1:train")
        self.assertEqual(dataset_id, "OpenAssistant/oasst1")
        self.assertEqual(split, "train")

    def test_empty_locator_rejected(self):
        with self.assertRaises(ValueError) as cm:
            _parse_locator("")
        self.assertIn("dataset id", str(cm.exception))

    def test_locator_with_only_colon_rejected(self):
        with self.assertRaises(ValueError):
            _parse_locator(":train")

    def test_split_locator_routes_to_hf(self):
        # registry-level: the prefix split + the HF locator parser
        # cooperate so `hf:org/dataset:split` lands cleanly.
        source_id, rest = split_locator("hf:Anthropic/hh-rlhf:train")
        self.assertEqual(source_id, "hf")
        dataset_id, split, _ = _parse_locator(rest)
        self.assertEqual(dataset_id, "Anthropic/hh-rlhf")
        self.assertEqual(split, "train")


# ── Auth ─────────────────────────────────────────────────────────────


class TokenResolutionTests(unittest.TestCase):
    def test_no_env_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            self.assertIsNone(_resolve_token())

    def test_hf_token_preferred(self):
        with mock.patch.dict(
            os.environ, {"HF_TOKEN": "tok_a", "HUGGING_FACE_HUB_TOKEN": "tok_b"}
        ):
            self.assertEqual(_resolve_token(), "tok_a")

    def test_legacy_env_var_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ["HUGGING_FACE_HUB_TOKEN"] = "tok_legacy"
            self.assertEqual(_resolve_token(), "tok_legacy")


# ── Split-dict materialization ───────────────────────────────────────


class MaterializeFirstSplitTests(unittest.TestCase):
    def test_single_dataset_returned_verbatim(self):
        ds = _FakeIterableDataset([{"a": 1}])
        out, picked = _materialize_first_split(ds)
        self.assertIs(out, ds)
        self.assertIsNone(picked)

    def test_dataset_dict_picks_first_key(self):
        ds_train = _FakeIterableDataset([{"a": 1}])
        ds_test = _FakeIterableDataset([{"a": 2}])
        dd = _FakeDatasetDict({"train": ds_train, "test": ds_test})
        out, picked = _materialize_first_split(dd)
        self.assertIs(out, ds_train)
        self.assertEqual(picked, "train")

    def test_empty_dataset_dict_rejected(self):
        with self.assertRaises(ValueError):
            _materialize_first_split(_FakeDatasetDict({}))


# ── load() / describe() with stubbed load_dataset ────────────────────


def _stub_loader(
    rows: list[dict],
    *,
    expect_split: str | None = None,
    expect_revision: str | None = None,
    expect_token: str | None = None,
    expect_streaming: bool = True,
):
    """Returns a fake ``load_dataset`` callable that asserts the
    expected kwargs the connector passed in."""

    def _fn(**kwargs):
        # Sanity-check the kwargs the connector forwards to load_dataset.
        if expect_split is None:
            assert "split" not in kwargs, kwargs
        else:
            assert kwargs.get("split") == expect_split, kwargs
        if expect_revision is None:
            assert "revision" not in kwargs, kwargs
        else:
            assert kwargs.get("revision") == expect_revision, kwargs
        if expect_token is None:
            assert "token" not in kwargs, kwargs
        else:
            assert kwargs.get("token") == expect_token, kwargs
        assert kwargs.get("streaming") is expect_streaming, kwargs
        return _FakeIterableDataset(rows)

    return _fn


def _no_token_env():
    """Clear any real HF auth env vars so the no-token assertions in
    the stub loader hold even on a developer machine that has
    ``HF_TOKEN`` set."""

    cleared = dict(os.environ)
    cleared.pop("HF_TOKEN", None)
    cleared.pop("HUGGING_FACE_HUB_TOKEN", None)
    return mock.patch.dict(os.environ, cleared, clear=True)


class LoadAndDescribeTests(unittest.TestCase):
    def test_load_yields_streaming_rows_with_limit(self):
        rows = [{"text": f"r{i}", "label": "pos"} for i in range(10)]
        source = HuggingFaceSource()
        with _no_token_env(), mock.patch(
            "app.services.dataset_import.sources.hf._import_datasets",
            return_value=(_stub_loader(rows, expect_split="train"), Exception),
        ):
            out = list(source.load("dataset_id:train", limit=3))
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], {"text": "r0", "label": "pos"})

    def test_load_forwards_revision_and_token(self):
        rows = [{"a": 1}]
        with mock.patch.dict(os.environ, {"HF_TOKEN": "tok_x"}):
            source = HuggingFaceSource()
            with mock.patch(
                "app.services.dataset_import.sources.hf._import_datasets",
                return_value=(
                    _stub_loader(
                        rows,
                        expect_split="train",
                        expect_revision="abc1234",
                        expect_token="tok_x",
                    ),
                    Exception,
                ),
            ):
                _ = list(source.load("org/ds:train:abc1234", limit=1))

    def test_describe_caps_sample_and_collects_columns(self):
        rows = [{"text": f"r{i}", "label": "pos"} for i in range(30)]
        source = HuggingFaceSource()
        with _no_token_env(), mock.patch(
            "app.services.dataset_import.sources.hf._import_datasets",
            return_value=(_stub_loader(rows), Exception),
        ):
            description = source.describe("dataset_id")
        # 20-row sample cap regardless of dataset size.
        self.assertEqual(len(description["sample_rows"]), 20)
        self.assertEqual(description["columns"], ["text", "label"])
        self.assertEqual(description["source_id"], "hf")
        self.assertEqual(description["dataset_id"], "dataset_id")
        self.assertIsNone(description["approximate_total_rows"])

    def test_describe_multi_split_picks_first(self):
        # No split in locator → DatasetDict comes back → pick first.
        ds_train = _FakeIterableDataset([{"a": 1}, {"a": 2}])
        ds_test = _FakeIterableDataset([{"a": 99}])

        def _fake_loader(**kwargs):
            assert "split" not in kwargs
            return _FakeDatasetDict({"train": ds_train, "test": ds_test})

        source = HuggingFaceSource()
        with _no_token_env(), mock.patch(
            "app.services.dataset_import.sources.hf._import_datasets",
            return_value=(_fake_loader, Exception),
        ):
            description = source.describe("dataset_id")
        self.assertEqual(description["split"], "train")
        self.assertEqual(len(description["sample_rows"]), 2)


# ── Missing-dep error contract ───────────────────────────────────────


class MissingDatasetsPackageTests(unittest.TestCase):
    def test_load_surfaces_clear_error_when_datasets_missing(self):
        # Simulate the package being uninstalled by making
        # _import_datasets raise. The connector must re-raise with a
        # human-readable hint pointing at the install command.
        source = HuggingFaceSource()
        with mock.patch(
            "app.services.dataset_import.sources.hf._import_datasets",
            side_effect=ImportError(
                "HF source requires the `datasets` package. "
                "Install it with `pip install datasets`, …"
            ),
        ):
            with self.assertRaises(ImportError) as cm:
                list(source.load("any:train"))
        self.assertIn("pip install datasets", str(cm.exception))


# ── Registry wiring ──────────────────────────────────────────────────


class RegistryWiringTests(unittest.TestCase):
    def test_hf_source_registered(self):
        self.assertIn("hf", list_registered_sources())

    def test_resolve_returns_huggingface_source(self):
        source = resolve_source("hf")
        self.assertEqual(source.source_id, "hf")
        self.assertIsInstance(source, HuggingFaceSource)


# ── End-to-end: introspect + preview through the orchestrator ────────


class IntrospectAndPreviewIntegrationTests(unittest.TestCase):
    """The HF source plugged into service.introspect_locator() +
    preview_import() should behave identically to jsonl / csv. We
    stub load_dataset to keep the test offline."""

    def test_introspect_on_hf_locator_proposes_classification(self):
        rows = [
            {"text": f"sample review {i} with enough length to be text-like", "label": lab}
            for i, lab in enumerate(["pos", "neg", "neu", "pos", "neg", "neu"])
        ]
        from app.services.dataset_import.service import introspect_locator

        with _no_token_env(), mock.patch(
            "app.services.dataset_import.sources.hf._import_datasets",
            return_value=(_stub_loader(rows), Exception),
        ):
            payload = asyncio.run(introspect_locator("hf:my-org/my-dataset"))

        self.assertEqual(payload["source_id"], "hf")
        self.assertEqual(payload["proposal"]["mapper_id"], "label_to_classification")

    def test_preview_runs_end_to_end_through_hf_source(self):
        rows = [
            {"text": f"sample review {i} with enough length", "label": lab}
            for i, lab in enumerate(["pos", "neg", "neu"])
        ]
        from app.services.dataset_import.service import preview_import

        with _no_token_env(), mock.patch(
            "app.services.dataset_import.sources.hf._import_datasets",
            return_value=(_stub_loader(rows), Exception),
        ):
            result = preview_import(
                project_id=0,
                project_task_profile=None,
                locator="hf:my-org/my-dataset",
                mapper_id="label_to_classification",
                field_map={},
                sample_cap=3,
            )
        self.assertEqual(result.accepted_count, 3)
        self.assertEqual(result.source_id, "hf")
        self.assertEqual(result.target_task_profile, "classification")


if __name__ == "__main__":
    unittest.main()
