"""Phase E — Kaggle source connector.

Pins locator parsing (competition / dataset / ?file=), the auth
pre-flight check, the file-picking heuristic on a pre-extracted
cache dir, format iteration (.jsonl / .json / .csv / .tsv), and
the missing-``kaggle``-package error contract.

The Kaggle API itself is stubbed everywhere — no network, no
``~/.kaggle/kaggle.json`` required. The download path is tested by
short-circuiting ``_download_and_extract`` via a pre-extracted cache
dir (matches the "second-run reuse" code path; the actual API call
is exercised manually per DATASET_IMPORT_PLAN.md Phase E).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
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
from app.services.dataset_import.sources.kaggle import (  # noqa: E402
    KaggleSource,
    _check_auth,
    _find_data_files,
    _iter_rows,
    _parse_locator,
    _pick_data_file,
    _slug_to_dirname,
)


# ── Locator parsing ──────────────────────────────────────────────────


class LocatorParsingTests(unittest.TestCase):
    def test_competition_only(self):
        self.assertEqual(
            _parse_locator("competition:pii-detection"),
            ("competition", "pii-detection", None),
        )

    def test_dataset_owner_slug(self):
        self.assertEqual(
            _parse_locator("dataset:ekohrt/pii-data-detection-dataset"),
            ("dataset", "ekohrt/pii-data-detection-dataset", None),
        )

    def test_picked_file_in_query(self):
        kind, slug, picked = _parse_locator(
            "competition:pii-detection?file=data/train.json"
        )
        self.assertEqual(kind, "competition")
        self.assertEqual(slug, "pii-detection")
        self.assertEqual(picked, "data/train.json")

    def test_dataset_requires_owner_slash_slug(self):
        with self.assertRaises(ValueError):
            _parse_locator("dataset:nameonly")

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            _parse_locator("notebook:slug")

    def test_empty_locator_rejected(self):
        with self.assertRaises(ValueError):
            _parse_locator("")

    def test_split_locator_routes_to_kaggle(self):
        source_id, rest = split_locator(
            "kaggle:competition:pii-detection?file=train.json"
        )
        self.assertEqual(source_id, "kaggle")
        kind, slug, picked = _parse_locator(rest)
        self.assertEqual((kind, slug, picked), ("competition", "pii-detection", "train.json"))


class SlugSanitizationTests(unittest.TestCase):
    def test_owner_slash_becomes_double_underscore(self):
        # Critical for filesystem safety: ekohrt/pii-data turns into a
        # single flat dir name without colliding into a subdir.
        self.assertEqual(
            _slug_to_dirname("dataset", "ekohrt/pii-data-detection-dataset"),
            "dataset__ekohrt_pii-data-detection-dataset",
        )

    def test_competition_slug_passthrough(self):
        self.assertEqual(
            _slug_to_dirname("competition", "pii-detection"),
            "competition__pii-detection",
        )


# ── Auth pre-flight ──────────────────────────────────────────────────


class AuthPreflightTests(unittest.TestCase):
    def test_missing_creds_raises_permission_error(self):
        # Both env and ~/.kaggle/kaggle.json must be absent.
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch(
                "app.services.dataset_import.sources.kaggle.Path.exists",
                return_value=False,
            ):
                with self.assertRaises(PermissionError) as cm:
                    _check_auth()
        self.assertIn("KAGGLE_USERNAME", str(cm.exception))
        self.assertIn("kaggle.json", str(cm.exception))

    def test_env_creds_satisfy_preflight(self):
        with mock.patch.dict(
            os.environ,
            {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"},
            clear=True,
        ):
            with mock.patch(
                "app.services.dataset_import.sources.kaggle.Path.exists",
                return_value=False,
            ):
                # No raise == pass.
                _check_auth()

    def test_kaggle_json_file_satisfies_preflight(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch(
                "app.services.dataset_import.sources.kaggle.Path.exists",
                return_value=True,
            ):
                _check_auth()


# ── File picking ─────────────────────────────────────────────────────


class FilePickingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel: str, content: str = "") -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_explicit_picked_file_returned(self):
        target = self._write("data/train.json", "[]")
        self._write("data/test.json", "[]")
        out = _pick_data_file(self.root, "data/train.json")
        self.assertEqual(out, target)

    def test_picked_file_missing_raises(self):
        self._write("train.json", "[]")
        with self.assertRaises(FileNotFoundError):
            _pick_data_file(self.root, "does/not/exist.json")

    def test_picked_file_with_bad_extension_rejected(self):
        self._write("README.md", "hello")
        with self.assertRaises(ValueError):
            _pick_data_file(self.root, "README.md")

    def test_train_dot_star_preferred(self):
        self._write("train.json", "[]")
        self._write("test.json", "[]")
        out = _pick_data_file(self.root, None)
        self.assertEqual(out.name, "train.json")

    def test_single_data_file_picked_when_no_train(self):
        self._write("alpha.csv", "a,b\n1,2\n")
        out = _pick_data_file(self.root, None)
        self.assertEqual(out.name, "alpha.csv")

    def test_multiple_data_files_no_train_surfaces_candidates(self):
        self._write("alpha.csv", "")
        self._write("beta.jsonl", "")
        with self.assertRaises(ValueError) as cm:
            _pick_data_file(self.root, None)
        msg = str(cm.exception)
        self.assertIn("alpha.csv", msg)
        self.assertIn("beta.jsonl", msg)
        self.assertIn("?file=", msg)

    def test_multiple_train_files_surfaces_candidates(self):
        self._write("data1/train.json", "")
        self._write("data2/train.csv", "")
        with self.assertRaises(ValueError) as cm:
            _pick_data_file(self.root, None)
        self.assertIn("train.", str(cm.exception))

    def test_no_data_files_clear_error(self):
        self._write("README.md", "hi")
        with self.assertRaises(FileNotFoundError):
            _pick_data_file(self.root, None)

    def test_find_data_files_is_recursive_and_sorted(self):
        self._write("z.csv", "")
        self._write("a/b/c.jsonl", "")
        found = _find_data_files(self.root)
        rel = [str(p.relative_to(self.root)) for p in found]
        self.assertEqual(rel, ["a/b/c.jsonl", "z.csv"])


# ── Row iteration ────────────────────────────────────────────────────


class RowIterationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content)
        return path

    def test_jsonl_streams_rows(self):
        path = self._write(
            "data.jsonl",
            '{"text": "a"}\n{"text": "b"}\n',
        )
        rows = list(_iter_rows(path))
        self.assertEqual(rows, [{"text": "a"}, {"text": "b"}])

    def test_jsonl_unparseable_lines_surface_as_sentinel(self):
        path = self._write(
            "data.jsonl",
            '{"text": "ok"}\nnot json\n',
        )
        rows = list(_iter_rows(path))
        self.assertEqual(rows[0], {"text": "ok"})
        self.assertEqual(rows[1]["__parse_error__"], "invalid_json")

    def test_json_array_streams_rows(self):
        # The Kaggle PII competition's train.json is a top-level
        # array of objects, not JSONL. Common shape.
        path = self._write(
            "data.json",
            json.dumps(
                [
                    {"document": 1, "tokens": ["Hi"], "labels": ["O"]},
                    {"document": 2, "tokens": ["By"], "labels": ["O"]},
                ]
            ),
        )
        rows = list(_iter_rows(path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["document"], 1)

    def test_json_non_array_rejected(self):
        path = self._write("data.json", '{"not_an_array": true}')
        with self.assertRaises(ValueError):
            list(_iter_rows(path))

    def test_csv_yields_dict_rows(self):
        path = self._write(
            "data.csv",
            "text,label\nhello,pos\nworld,neg\n",
        )
        rows = list(_iter_rows(path))
        self.assertEqual(
            rows,
            [{"text": "hello", "label": "pos"}, {"text": "world", "label": "neg"}],
        )

    def test_tsv_yields_dict_rows(self):
        path = self._write(
            "data.tsv",
            "text\tlabel\nhello\tpos\n",
        )
        rows = list(_iter_rows(path))
        self.assertEqual(rows, [{"text": "hello", "label": "pos"}])

    def test_unsupported_extension_rejected(self):
        path = self._write("data.txt", "hi")
        with self.assertRaises(ValueError):
            list(_iter_rows(path))


# ── End-to-end: connector with pre-extracted cache ───────────────────


class ConnectorEndToEndTests(unittest.TestCase):
    """Exercises load()/describe() by short-circuiting the download:
    a pre-extracted directory in the cache root simulates the
    "second-run reuse" path."""

    def setUp(self):
        self._cache_root = tempfile.TemporaryDirectory()
        # Pre-populate the slug-scoped dir so _download_and_extract
        # finds data already + skips the API call entirely.
        cache_root = Path(self._cache_root.name)
        slug_dir = cache_root / "competition__pii-detection"
        slug_dir.mkdir(parents=True)
        (slug_dir / "train.json").write_text(
            json.dumps(
                [
                    {"document": 1, "tokens": ["Alice"], "labels": ["B-NAME"]},
                    {"document": 2, "tokens": ["Bob"], "labels": ["B-NAME"]},
                ]
            )
        )
        self._env = mock.patch.dict(
            os.environ,
            {
                "BREWSLM_KAGGLE_CACHE": self._cache_root.name,
                "KAGGLE_USERNAME": "u",
                "KAGGLE_KEY": "k",
            },
        )
        self._env.start()
        # Stub the lazy import — auth pre-flight passes via env vars
        # but we never want to touch the real kaggle library here.
        self._patcher = mock.patch(
            "app.services.dataset_import.sources.kaggle._import_kaggle_api",
            return_value=mock.Mock(
                competition_download_files=mock.Mock(),
                dataset_download_files=mock.Mock(),
            ),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._env.stop()
        self._cache_root.cleanup()

    def test_describe_returns_sample_and_columns(self):
        source = KaggleSource()
        description = source.describe("competition:pii-detection")
        self.assertEqual(description["source_id"], "kaggle")
        self.assertEqual(description["kind"], "competition")
        self.assertEqual(description["slug"], "pii-detection")
        self.assertEqual(description["picked_file"], "train.json")
        self.assertEqual(description["approximate_total_rows"], 2)
        self.assertEqual(
            sorted(description["columns"]), ["document", "labels", "tokens"]
        )

    def test_load_streams_rows_with_limit(self):
        source = KaggleSource()
        rows = list(source.load("competition:pii-detection", limit=1))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["document"], 1)


# ── Registry wiring ──────────────────────────────────────────────────


class RegistryWiringTests(unittest.TestCase):
    def test_kaggle_source_registered(self):
        self.assertIn("kaggle", list_registered_sources())

    def test_resolve_returns_kaggle_source(self):
        source = resolve_source("kaggle")
        self.assertEqual(source.source_id, "kaggle")
        self.assertIsInstance(source, KaggleSource)


# ── Missing-package error contract ───────────────────────────────────


class MissingKagglePackageTests(unittest.TestCase):
    def test_import_error_surfaces_install_hint(self):
        # Simulate the package being uninstalled. The kaggle library's
        # SystemExit-on-import is intercepted by _check_auth → the
        # ImportError below would only fire if creds existed but the
        # library was missing.
        with mock.patch.dict(
            os.environ, {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}
        ):
            with mock.patch(
                "app.services.dataset_import.sources.kaggle._import_kaggle_api",
                side_effect=ImportError(
                    "Kaggle source requires the `kaggle` package. "
                    "Install it with `pip install kaggle`, …"
                ),
            ):
                source = KaggleSource()
                with self.assertRaises(ImportError) as cm:
                    list(source.load("competition:foo"))
        self.assertIn("pip install kaggle", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
