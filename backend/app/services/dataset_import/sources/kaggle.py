"""Kaggle source connector — competitions + datasets.

Locator formats::

    kaggle:competition:<slug>          # competition data
    kaggle:dataset:<owner/slug>        # community dataset
    kaggle:competition:<slug>?file=train.json
    kaggle:dataset:<owner/slug>?file=data/train.csv

The connector downloads the archive once into
``$BREWSLM_KAGGLE_CACHE`` (default ``~/.cache/brewslm/kaggle``),
auto-extracts the zip, then picks the data file inside. File picking
order:

1. Explicit ``?file=<path>`` in the locator.
2. ``train.*`` if present and there's exactly one match.
3. Single ``.jsonl`` / ``.json`` / ``.csv`` / ``.tsv`` file in the
   archive when only one exists.

Anything else surfaces a clear error listing the candidate files so
the user can re-run with ``?file=…``. Subsequent calls reuse the
extracted dir — no re-download.

Auth: reads ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` env vars (the
standard Kaggle env names), or falls back to ``~/.kaggle/kaggle.json``
which the Kaggle library reads on its own. The ``kaggle`` PyPI
package has the unusual behavior of calling ``sys.exit(1)`` when it
can't find creds at *import time*, so this module imports it lazily
inside ``describe`` / ``load`` and surfaces a clean error first.

Why we don't just shell out to the ``kaggle`` CLI: the same package
ships both the binary and the Python API. The Python API is cheaper
to call (no subprocess, no output parsing) and gives us direct error
codes — and it's already a dep via ``requirements-base.txt``.
"""

from __future__ import annotations

import csv as _csv
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

from app.services.dataset_import.protocols import DatasetSource, RawRow
from app.services.dataset_import.registry import register_source


DESCRIBE_SAMPLE_CAP: int = 20
"""Match the jsonl / csv / hf connectors so the introspector sees a
consistent sample regardless of source."""


_DATA_EXTENSIONS: tuple[str, ...] = (".jsonl", ".json", ".csv", ".tsv")


def _default_cache_dir() -> Path:
    """Resolve the cache root: env override → XDG default."""

    override = os.environ.get("BREWSLM_KAGGLE_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "brewslm" / "kaggle"


def _slug_to_dirname(kind: str, slug: str) -> str:
    """Sanitize the Kaggle slug for use as a directory name. Replaces
    ``/`` with ``__`` so an ``owner/dataset`` slug lands in a single
    flat dir without collisions."""

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", slug)
    return f"{kind}__{safe}"


def _parse_locator(
    rest: str,
) -> tuple[str, str, str | None]:
    """Split ``<kind>:<slug>[?file=<path>]`` into
    ``(kind, slug, picked_file | None)``.

    ``kind`` is ``competition`` or ``dataset``. ``slug`` is everything
    between the kind colon and the optional ``?file=`` suffix.
    """

    text = (rest or "").strip()
    if not text:
        raise ValueError(
            "kaggle source requires '<kind>:<slug>' — kind is "
            "'competition' or 'dataset'"
        )

    # Optional ?file= suffix.
    picked_file: str | None = None
    if "?" in text:
        text, _, query = text.partition("?")
        for part in query.split("&"):
            if "=" not in part:
                continue
            key, _, value = part.partition("=")
            if key.strip().lower() == "file":
                value = value.strip()
                if value:
                    picked_file = value

    if ":" not in text:
        raise ValueError(
            f"kaggle locator '{rest}' is missing a kind — use "
            "'kaggle:competition:<slug>' or 'kaggle:dataset:<owner/slug>'"
        )
    kind, _, slug = text.partition(":")
    kind = kind.strip().lower()
    slug = slug.strip()
    if kind not in {"competition", "dataset"}:
        raise ValueError(
            f"kaggle locator kind must be 'competition' or 'dataset' "
            f"(got '{kind}')"
        )
    if not slug:
        raise ValueError(
            f"kaggle locator '{rest}' is missing a slug"
        )
    if kind == "dataset" and "/" not in slug:
        raise ValueError(
            f"kaggle dataset slug must be '<owner>/<dataset>' (got '{slug}')"
        )
    return kind, slug, picked_file


def _check_auth() -> None:
    """Refuse to proceed if no Kaggle creds are findable.

    Surfaces a clear message *before* importing the ``kaggle`` package
    (which calls ``sys.exit(1)`` on its own when auth is missing —
    catastrophic for callers that just want a graceful error)."""

    has_env = bool(os.environ.get("KAGGLE_USERNAME")) and bool(
        os.environ.get("KAGGLE_KEY")
    )
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not has_env and not has_file:
        raise PermissionError(
            "Kaggle source needs auth. Set KAGGLE_USERNAME + KAGGLE_KEY "
            "env vars, or drop a kaggle.json into ~/.kaggle/. Generate "
            "one at https://www.kaggle.com/settings/account → "
            "'Create New API Token'."
        )


def _import_kaggle_api():
    """Lazy import so module-load doesn't trip the ``kaggle``
    package's import-time auth exit. Returns an authenticated
    ``KaggleApi`` instance.
    """

    _check_auth()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Kaggle source requires the `kaggle` package. "
            "Install it with `pip install kaggle`, or skip the kaggle "
            "source and use the hf / jsonl / csv sources."
        ) from exc
    api = KaggleApi()
    api.authenticate()
    return api


def _download_and_extract(
    api: Any, kind: str, slug: str, cache_root: Path
) -> Path:
    """Download the archive into the slug-scoped cache dir and
    extract it. No-op when the extracted directory already exists +
    has at least one data file inside — re-runs skip the network hop.

    Returns the extracted directory path.
    """

    target = cache_root / _slug_to_dirname(kind, slug)
    target.mkdir(parents=True, exist_ok=True)

    # Skip the download when we already have data files extracted —
    # a partially-extracted dir (zip present, no data files) re-runs.
    existing_data = list(_find_data_files(target))
    if existing_data:
        return target

    target.mkdir(parents=True, exist_ok=True)
    if kind == "competition":
        # Pulls a zip named ``<slug>.zip`` into the target dir.
        api.competition_download_files(slug, path=str(target), quiet=False)
    else:
        # Dataset: ``api.dataset_download_files(...)`` writes a zip
        # at ``<target>/<dataset_basename>.zip``.
        api.dataset_download_files(
            slug, path=str(target), quiet=False, unzip=False
        )

    # Extract every zip we just downloaded.
    for zip_path in target.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
        except zipfile.BadZipFile as exc:
            raise IOError(
                f"Kaggle archive at {zip_path} is not a valid zip: {exc}"
            ) from exc
    return target


def _find_data_files(root: Path) -> list[Path]:
    """Recursively enumerate data-shaped files under ``root``. Sorted
    so the picking heuristic is deterministic."""

    if not root.exists():
        return []
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in _DATA_EXTENSIONS:
            matches.append(path)
    return sorted(matches)


def _pick_data_file(
    extracted_dir: Path, picked_file: str | None
) -> Path:
    """Pick the data file inside the extracted archive. See module
    docstring for the picking order. Surfaces a clear error listing
    the candidates when picking is ambiguous."""

    if picked_file:
        candidate = extracted_dir / picked_file
        if not candidate.exists():
            raise FileNotFoundError(
                f"file '{picked_file}' not found inside the Kaggle "
                f"archive at {extracted_dir}"
            )
        if candidate.suffix.lower() not in _DATA_EXTENSIONS:
            raise ValueError(
                f"file '{picked_file}' has unsupported extension "
                f"'{candidate.suffix}'. Supported: "
                f"{', '.join(_DATA_EXTENSIONS)}"
            )
        return candidate

    candidates = _find_data_files(extracted_dir)
    if not candidates:
        raise FileNotFoundError(
            f"no data files ({', '.join(_DATA_EXTENSIONS)}) found in "
            f"the Kaggle archive at {extracted_dir}"
        )

    # Prefer train.*
    train = [p for p in candidates if p.stem.lower() == "train"]
    if len(train) == 1:
        return train[0]
    if len(train) > 1:
        rel = [str(p.relative_to(extracted_dir)) for p in train]
        raise ValueError(
            f"multiple 'train.*' files in the Kaggle archive "
            f"({', '.join(rel)}); pass ?file=<path> to disambiguate"
        )

    if len(candidates) == 1:
        return candidates[0]

    rel = [str(p.relative_to(extracted_dir)) for p in candidates]
    raise ValueError(
        f"multiple data files in the Kaggle archive "
        f"({', '.join(rel)}); pass ?file=<path> to disambiguate"
    )


def _iter_rows(path: Path) -> Iterable[RawRow]:
    """Stream rows from the picked file. Supports .jsonl (line per
    object), .json (top-level array of objects — common in Kaggle
    competitions like the PII detection one), .csv, and .tsv."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    yield {
                        "__parse_error__": "invalid_json",
                        "__raw_line__": stripped[:200],
                    }
                    continue
                if not isinstance(row, dict):
                    yield {
                        "__parse_error__": "not_an_object",
                        "__raw_value__": str(row)[:200],
                    }
                else:
                    yield row
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON file at {path} is not valid JSON: {exc}"
                ) from exc
        if not isinstance(payload, list):
            raise ValueError(
                f"JSON file at {path} must be a top-level array of "
                f"objects (got {type(payload).__name__})"
            )
        for value in payload:
            if isinstance(value, dict):
                yield value
            else:
                yield {
                    "__parse_error__": "not_an_object",
                    "__raw_value__": str(value)[:200],
                }
        return
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh, delimiter=delimiter)
            for row in reader:
                cleaned = {
                    str(k): (v if v is not None else "")
                    for k, v in row.items()
                    if k is not None
                }
                yield cleaned
        return
    raise ValueError(f"unsupported file extension '{suffix}' for {path}")


class KaggleSource:
    source_id: str = "kaggle"

    def _resolve_file(self, rest: str) -> tuple[Path, str, str, str | None]:
        """Run the locator → file pipeline. Returns
        ``(data_path, kind, slug, picked_file_arg)``."""

        kind, slug, picked = _parse_locator(rest)
        api = _import_kaggle_api()
        extracted = _download_and_extract(api, kind, slug, _default_cache_dir())
        data_path = _pick_data_file(extracted, picked)
        return data_path, kind, slug, picked

    def load(
        self,
        locator: str,
        *,
        limit: int | None = None,
    ) -> Iterable[RawRow]:
        data_path, _kind, _slug, _picked = self._resolve_file(locator)
        yielded = 0
        for row in _iter_rows(data_path):
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    def describe(self, locator: str) -> dict[str, Any]:
        data_path, kind, slug, picked = self._resolve_file(locator)
        sample: list[dict[str, Any]] = []
        columns: list[str] = []
        seen: set[str] = set()
        total_rows = 0
        for row in _iter_rows(data_path):
            total_rows += 1
            if len(sample) < DESCRIBE_SAMPLE_CAP and isinstance(row, dict):
                sample.append(row)
                for key in row.keys():
                    if key not in seen:
                        seen.add(key)
                        columns.append(key)
        return {
            "source_id": self.source_id,
            "locator": locator,
            "kind": kind,
            "slug": slug,
            "picked_file": str(data_path.name),
            "resolved_path": str(data_path),
            "approximate_total_rows": total_rows,
            "sample_rows": sample,
            "columns": columns,
        }


register_source("kaggle", KaggleSource)
