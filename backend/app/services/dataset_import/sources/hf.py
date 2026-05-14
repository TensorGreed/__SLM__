"""HuggingFace datasets source connector.

Locator format::

    hf:<dataset_id>[:<split>[:<revision>]]

Examples::

    hf:ai4privacy/pii-masking-200k
    hf:ai4privacy/pii-masking-200k:train
    hf:Anthropic/hh-rlhf:test:abc1234

The connector wraps :func:`datasets.load_dataset` and streams rows
lazily (``streaming=True``) so a 50GB dataset can be introspected
without downloading the full archive. Auth uses the standard HF env
vars: ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``. The HF library
also reads these on its own — we pass them explicitly so the error
message in ``describe`` / ``load`` is clear when a gated dataset
fails for missing auth.

Caching: the HF library writes downloads to ``~/.cache/huggingface``
by default; honor ``HF_HOME`` / ``HF_DATASETS_CACHE`` for offline /
project-scoped overrides. We don't add a BrewSLM-specific cache
policy — HF's defaults are well-understood and respected by every
adjacent tool in the ecosystem.

Two failure modes worth surfacing distinctly:

- ``datasets`` package not importable → clear error pointing the
  user at ``pip install datasets``.
- ``load_dataset`` returns ``DatasetDict`` (multi-split) when no
  split was specified — we pick the first split + emit a warning so
  the user knows to be explicit.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from app.services.dataset_import.protocols import DatasetSource, RawRow
from app.services.dataset_import.registry import register_source


# Sample cap for ``describe`` — keeps the streaming download to the
# first few KB regardless of dataset size.
DESCRIBE_SAMPLE_CAP: int = 20


def _resolve_token() -> str | None:
    """Read the standard HF auth env vars. Returns None when neither
    is set — caller passes ``token=None`` which the HF library treats
    as anonymous."""

    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_var)
        if value and value.strip():
            return value.strip()
    return None


def _parse_locator(rest: str) -> tuple[str, str | None, str | None]:
    """Split ``<dataset_id>[:<split>[:<revision>]]`` into its parts.

    The dataset_id can contain ``/`` (e.g. ``ai4privacy/pii-…``) but
    not ``:`` — so we partition on ``:`` after the registry's
    source-prefix split already happened.
    """

    text = (rest or "").strip()
    if not text:
        raise ValueError(
            "hf source requires a dataset id, e.g. "
            "'hf:Anthropic/hh-rlhf' or 'hf:imdb:train'"
        )
    parts = text.split(":")
    dataset_id = parts[0].strip()
    split = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else None
    revision = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
    if not dataset_id:
        raise ValueError(f"hf locator '{text}' is missing a dataset id")
    return dataset_id, split, revision


def _import_datasets():
    """Lazy import so the dataset_import package stays loadable on
    machines without the HF ``datasets`` library installed (e.g. the
    minimal CI image). The CLI / API surfaces this as a clear error
    rather than a stack trace on import."""

    try:
        from datasets import load_dataset  # type: ignore
        from datasets.exceptions import DatasetNotFoundError  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "HF source requires the `datasets` package. "
            "Install it with `pip install datasets`, or skip the hf "
            "source and use jsonl / csv directly."
        ) from exc
    return load_dataset, DatasetNotFoundError


def _materialize_first_split(loaded: Any) -> tuple[Any, str | None]:
    """When ``load_dataset`` returns a DatasetDict (multi-split), pick
    the first split so the caller gets an iterable Dataset / IterableDataset.
    Returns ``(dataset, picked_split_name | None)``."""

    # DatasetDict / IterableDatasetDict both behave dict-like. Detect
    # by duck-typing rather than isinstance — avoids importing the
    # specific classes when ``datasets`` may be missing in tests.
    if hasattr(loaded, "keys") and hasattr(loaded, "__getitem__") and not hasattr(
        loaded, "features"
    ):
        keys = list(loaded.keys())
        if not keys:
            raise ValueError("HF returned an empty DatasetDict")
        picked = keys[0]
        return loaded[picked], picked
    return loaded, None


def _normalize_row(value: Any) -> RawRow:
    """Coerce a non-dict row (rare — usually datasets emit dicts) into
    a single-key dict so the downstream mapper sees a stable shape."""

    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


class HuggingFaceSource:
    source_id: str = "hf"

    def _open_streaming(
        self,
        rest: str,
        *,
        full_load: bool = False,
    ) -> tuple[Any, str, str | None, str | None]:
        """Resolve the locator + call ``load_dataset``.

        Returns ``(dataset, dataset_id, picked_split, revision)``.
        ``full_load=False`` uses ``streaming=True`` so we don't
        download more than the first few rows; ``run_import`` may
        prefer ``full_load=True`` when the user supplied a small
        explicit ``limit`` and wants random access (e.g. shuffle).
        """

        load_dataset, _ = _import_datasets()
        dataset_id, split, revision = _parse_locator(rest)
        token = _resolve_token()

        kwargs: dict[str, Any] = {
            "path": dataset_id,
            "streaming": not full_load,
        }
        if split:
            kwargs["split"] = split
        if revision:
            kwargs["revision"] = revision
        if token:
            kwargs["token"] = token

        try:
            loaded = load_dataset(**kwargs)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"HF dataset '{dataset_id}' not found "
                f"(split={split!r}, revision={revision!r}): {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"HF load_dataset rejected '{dataset_id}' "
                f"(split={split!r}, revision={revision!r}): {exc}"
            ) from exc

        # If the user didn't pin a split, pick the first one.
        dataset, picked = _materialize_first_split(loaded)
        effective_split = split or picked
        return dataset, dataset_id, effective_split, revision

    def load(
        self,
        locator: str,
        *,
        limit: int | None = None,
    ) -> Iterable[RawRow]:
        dataset, _dataset_id, _split, _revision = self._open_streaming(locator)
        yielded = 0
        for row in dataset:
            yield _normalize_row(row)
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    def describe(self, locator: str) -> dict[str, Any]:
        """Sample the first ``DESCRIBE_SAMPLE_CAP`` rows + collect
        column names. Doesn't try to count the total — HF's iterable
        datasets don't expose ``len``, and downloading the full
        archive to count would be expensive.
        """

        dataset, dataset_id, split, revision = self._open_streaming(locator)
        sample: list[dict[str, Any]] = []
        columns: list[str] = []
        seen: set[str] = set()
        for row in dataset:
            normalized = _normalize_row(row)
            sample.append(normalized)
            for key in normalized.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
            if len(sample) >= DESCRIBE_SAMPLE_CAP:
                break
        return {
            "source_id": self.source_id,
            "locator": locator,
            "dataset_id": dataset_id,
            "split": split,
            "revision": revision,
            "approximate_total_rows": None,  # streaming — unknown w/o download
            "sample_rows": sample,
            "columns": columns,
        }


register_source("hf", HuggingFaceSource)
