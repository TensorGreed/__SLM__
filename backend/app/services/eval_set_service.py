"""Named, versioned eval sets + prompt-format variants (hardening Phase 3).

An eval set is a project's own named set of test rows, beside the four built-in sets (prepared test /
validation, the two answer keys). Task-agnostic: rows are stored as given (any shape the eval path
reads: question/answer, text/label, spans, messages, tool calls, SQL ...), plus an optional
``row_context``.

Storage: one ``Dataset`` row (``DatasetType.EVAL_SET``; ``name`` = display name; ``file_path`` = the
current version's file) whose ``metadata`` holds ``{eval_set: true, slug, current_version, versions:
[...], variants: {...}}``; one ``DatasetVersion`` per version; files at
``DATA_DIR/projects/<id>/eval_sets/<slug>/v<N>.jsonl``. A version is an immutable snapshot: a new
upload, a row delete or a copy makes a new version, and the old files stay for the results that used
them.

Prompt-format variants are declarative render options over the SAME rows (no file is rewritten):
``{system: "strip"}`` drops every system message (``row_context.system``, the legacy ``system_prompt``
field, ``messages`` system turns); ``{system: "replace", text}`` gives every row ``text`` as its system
message; ``{system: "add", text}`` gives it only to rows without one. Applied at eval time to the rows
before the task handler builds prompts, so it works for every task profile that renders a system
message (every profile, see ``row_context_service.CONSUMERS``). A variant is immutable once created
(delete + recreate under a new name), and its digest is stamped on each result.

Results: an eval-set eval is a normal ``EvalResult`` whose ``dataset_name`` is
``eval_set:<slug>@v<N>`` (``+<variant>`` when one is picked) and whose ``details.dataset.eval_set``
carries the provenance. ``is_default_eval_dataset`` / ``default_eval_results_clause`` keep the "latest
eval" readers (summary, gates, lift, Compare's default pick, Coach ...) on the built-in sets (decision
D3).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DatasetVersion

PREFIX = "eval_set:"
VARIANT_MODES = ("strip", "replace", "add")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
MAX_NAME_CHARS = 120
ROWS_PAGE_MAX = 200
# Reject reason (gold_import_service.REASON_TEXT) for an imported row that leaks training data.
REASON_LEAKS_TRAINING = "leaks_training_data"
# A version's record of rows the user kept although the leak scan flagged them (eval sets only: a
# training-direction leak is never keepable). ``{count, row_indexes, row_ids[, lines]}``.
KEPT_LEAK_KEY = "kept_despite_leak"
# Built-in sets an eval set can be copied from (``from_split``).
SPLIT_SOURCES = ("test", "validation", "gold_test", "gold_dev")

_LOCKS: dict[tuple[int, int, str], asyncio.Lock] = {}
# Per project: the highest version each slug ever had (survives a delete, so a recreated slug never
# reuses a version number and old ``@vN`` results are never paired with new rows).
TOMBSTONES_FILE = "_tombstones.json"


class EvalSetError(ValueError):
    """A request the eval-set service refuses. ``code`` is the API error code, ``status`` the HTTP status."""

    def __init__(self, message: str, *, code: str = "EVAL_SET_INVALID", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def write_lock(project_id: int, slug: str) -> asyncio.Lock:
    """Serializes every write to one set (create, versions, row deletes, variants, lock, delete). A
    writer re-reads the set INSIDE the lock (``get_eval_set(..., fresh=True)``) and mutates only that
    copy: a copy loaded before the lock is stale. Keyed per event loop (an ``asyncio.Lock`` binds to
    one loop)."""
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    key = (loop_id, int(project_id), normalize_slug(slug))
    lock = _LOCKS.get(key)
    if lock is None:
        lock = _LOCKS[key] = asyncio.Lock()
    return lock


# --------------------------------------------------------------------------
# Names and labels.
# --------------------------------------------------------------------------


def normalize_slug(slug: Any) -> str:
    return str(slug or "").strip().lower()


def is_eval_set_name(name: Any) -> bool:
    return isinstance(name, str) and name.strip().lower().startswith(PREFIX)


def is_default_eval_dataset(name: Any) -> bool:
    """True for a result on a built-in set (or any non-eval-set name). "Latest eval" readers use this
    (or ``default_eval_results_clause``) so an eval-set run never becomes "the" eval of a run."""
    return not is_eval_set_name(name)


def default_eval_results_clause():
    """SQL filter twin of ``is_default_eval_dataset`` for ``select(EvalResult)`` readers."""
    from app.models.experiment import EvalResult

    return or_(EvalResult.dataset_name.is_(None), ~EvalResult.dataset_name.startswith(PREFIX, autoescape=True))


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", str(name or "").strip().lower()).strip("-_")
    return slug[:63] or "eval-set"


def dataset_label(slug: str, version: int, variant: str | None = None) -> str:
    """``eval_set:<slug>@v<N>`` (+ ``+<variant>``): the ``EvalResult.dataset_name`` of an eval-set run.
    Unique per (set, version, variant), so readers that pick "the latest eval on X" pair like for like."""
    label = f"{PREFIX}{slug}@v{int(version)}"
    return f"{label}+{variant}" if variant else label


def parse_label(name: str) -> tuple[str, int | None, str | None]:
    """``eval_set:<slug>[@v<N>][+<variant>]`` -> ``(slug, version | None, variant | None)``. Raises
    ``EvalSetError`` on a malformed name."""
    if not is_eval_set_name(name):
        raise EvalSetError(f"'{name}' is not an eval-set name (eval_set:<slug>[@vN][+variant]).")
    rest = name.strip()[len(PREFIX):]
    variant: str | None = None
    if "+" in rest:
        rest, variant = rest.split("+", 1)
        variant = variant.strip() or None
    version: int | None = None
    if "@" in rest:
        rest, ver = rest.split("@", 1)
        m = re.fullmatch(r"v?(\d+)", ver.strip().lower())
        if not m:
            raise EvalSetError(f"Bad eval-set version in '{name}' (use @v1, @v2 ...).")
        version = int(m.group(1))
    slug = rest.strip().lower()
    if not SLUG_RE.match(slug):
        raise EvalSetError(f"Bad eval-set slug in '{name}'.")
    return slug, version, variant


def human_label(name: Any) -> str:
    """"eval set fresh v1 (native)" for an eval-set dataset name; the name itself otherwise."""
    if not is_eval_set_name(name):
        return str(name or "")
    try:
        slug, version, variant = parse_label(str(name))
    except EvalSetError:
        return str(name)
    text = f"eval set {slug}" + (f" v{version}" if version is not None else "")
    return text + (f" ({variant})" if variant else "")


# --------------------------------------------------------------------------
# Prompt-format variants (pure functions; every task profile).
# --------------------------------------------------------------------------


def normalize_variant_name(name: Any) -> str:
    value = str(name or "").strip().lower()
    if not SLUG_RE.match(value):
        raise EvalSetError("A variant name is 1-63 lowercase letters, digits, '-' or '_'.")
    return value


def normalize_variant(spec: Any) -> dict[str, Any]:
    """``{"system": "strip"}`` / ``{"system": "replace" | "add", "text": str}``. Raises ``EvalSetError``."""
    from app.services import row_context_service as rcs

    if not isinstance(spec, dict):
        raise EvalSetError("A variant is an object like {\"system\": \"strip\"}.")
    mode = str(spec.get("system") or "").strip().lower()
    if mode not in VARIANT_MODES:
        raise EvalSetError(f"variant.system must be one of {list(VARIANT_MODES)}.")
    unknown = sorted(set(spec) - {"system", "text"})
    if unknown:
        raise EvalSetError(f"Unknown variant field(s): {unknown}.")
    if mode == "strip":
        return {"system": "strip"}
    text = spec.get("text")
    if not isinstance(text, str) or not text.strip():
        raise EvalSetError(f"variant.text is required for system '{mode}'.")
    if len(text) > rcs.MAX_SYSTEM_CHARS:
        raise EvalSetError(f"variant.text is longer than {rcs.MAX_SYSTEM_CHARS} characters.")
    return {"system": mode, "text": text.strip()}


def variant_digest(spec: dict[str, Any] | None) -> str | None:
    if not spec:
        return None
    from app.services import row_context_service as rcs

    return rcs.digest(spec)


def variant_label(spec: dict[str, Any] | None) -> str:
    if not spec:
        return "system message as in the rows"
    mode = spec.get("system")
    if mode == "strip":
        return "system message stripped"
    if mode == "replace":
        return "system message replaced"
    return "system message added where missing"


def _strip_system_turns(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    out = []
    for turn in value:
        if isinstance(turn, dict) and str(turn.get("role") or turn.get("from") or "").strip().lower() == "system":
            continue
        out.append(turn)
    return out


def _has_system(row: dict[str, Any]) -> bool:
    from app.services import row_context_service as rcs

    if rcs.row_system(row)[0]:
        return True
    for key in ("messages", "conversations"):
        turns = row.get(key)
        if isinstance(turns, list) and any(
            isinstance(t, dict) and str(t.get("role") or t.get("from") or "").strip().lower() == "system"
            for t in turns
        ):
            return True
    return False


def apply_variant(row: dict[str, Any], spec: dict[str, Any] | None) -> dict[str, Any]:
    """The row rendered under ``spec`` (a new dict; the input is not changed). ``None`` = unchanged."""
    if not spec or not isinstance(row, dict):
        return row
    from app.services import row_context_service as rcs

    mode = spec.get("system")
    if mode == "add" and _has_system(row):
        return row
    out = dict(row)
    ctx = dict(rcs.decode(out.get(rcs.ROW_CONTEXT_KEY)) or {})
    if mode in ("strip", "replace"):
        ctx.pop("system", None)
        out.pop(rcs.SYSTEM_ALIAS, None)
        for key in ("messages", "conversations"):
            if key in out:
                out[key] = _strip_system_turns(out[key])
    if mode in ("replace", "add"):
        ctx["system"] = spec["text"]
    if ctx:
        out[rcs.ROW_CONTEXT_KEY] = ctx
    else:
        out.pop(rcs.ROW_CONTEXT_KEY, None)
    return out


# Training task types whose model never sees a system message (the eval engine reads only the row's
# text / input, not a rendered prompt): a prompt-format variant would change nothing there.
_NO_SYSTEM_TASK_TYPES = {
    "classification": "a classifier-head run (it scores the row text with its label head)",
    "token_classification": "a token-classification run (it tags the raw text)",
    "seq2seq": "a seq2seq run (the encoder reads the prompt as-is, with no system message)",
}


def variant_unsupported_reason(experiment: Any, model_path: str | None = None) -> str | None:
    """Why a prompt-format variant cannot apply to evals of ``experiment`` (None = it can). Variants
    change only the system message, so they are refused for the engines that never render one:
    classifier-head, token-classification, seq2seq and multimodal (vision / audio) runs. Read from the
    run's training ``task_type`` and, when its artifacts exist, from their shape (as the eval engine
    picks it). Never raises."""
    cfg = getattr(experiment, "config", None)
    task_type = str((cfg or {}).get("task_type") or "").strip().lower() if isinstance(cfg, dict) else ""
    why: str | None = _NO_SYSTEM_TASK_TYPES.get(task_type)
    if why is None:
        try:
            from app.services import evaluation_service as es

            model_ref = es._resolve_model_reference(experiment, model_path)
            if es._resolve_token_classifier_artifacts(model_ref) is not None:
                why = _NO_SYSTEM_TASK_TYPES["token_classification"]
            elif es._resolve_classifier_head_artifacts(model_ref) is not None:
                why = _NO_SYSTEM_TASK_TYPES["classification"]
            elif es._resolve_multimodal_artifacts(model_ref) is not None:
                why = "a multimodal (vision / audio) run (its inputs are not chat prompts)"
            elif es._resolve_seq2seq_artifacts(model_ref) is not None:
                why = _NO_SYSTEM_TASK_TYPES["seq2seq"]
        except Exception:  # noqa: BLE001 - no artifacts yet: nothing says the variant cannot apply
            why = None
    if why is None:
        return None
    return (f"Prompt-format variants change only the system message, and this is {why}: the variant "
            "would change nothing. Evaluate on the set without a variant.")


def apply_variant_rows(rows: list[dict[str, Any]], spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not spec:
        return rows
    return [apply_variant(r, spec) for r in rows]


# --------------------------------------------------------------------------
# Files.
# --------------------------------------------------------------------------


def eval_sets_dir(project_id: int) -> Path:
    return Path(settings.DATA_DIR) / "projects" / str(project_id) / "eval_sets"


def version_path(project_id: int, slug: str, version: int) -> Path:
    return eval_sets_dir(project_id) / slug / f"v{int(version)}.jsonl"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.services.dataset_service import _sha256_file, split_content_hash

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{datetime.now(timezone.utc).timestamp():.6f}")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    return {"sha256": _sha256_file(path), "content_sha256": split_content_hash(path)}


def _tombstones_path(project_id: int) -> Path:
    return eval_sets_dir(project_id) / TOMBSTONES_FILE


def read_tombstones(project_id: int) -> dict[str, int]:
    """``{slug: highest version it ever had}`` for the project's deleted sets. A missing file is ``{}``;
    an unreadable or corrupt one raises ``EvalSetError`` (fail closed: guessing ``{}`` would let a
    deleted slug reuse version numbers) and is left untouched."""
    path = _tombstones_path(project_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalSetError(
            f"eval_sets/{TOMBSTONES_FILE} is unreadable or corrupt ({exc}); it was left untouched. "
            "Repair or remove it by hand, then retry.",
            code="EVAL_SET_TOMBSTONES_CORRUPT", status=409,
        ) from exc
    if not isinstance(data, dict):
        raise EvalSetError(
            f"eval_sets/{TOMBSTONES_FILE} is not a JSON object; it was left untouched. Repair or remove it, then retry.",
            code="EVAL_SET_TOMBSTONES_CORRUPT", status=409,
        )
    out: dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


_TOMBSTONE_THREAD_LOCK = threading.Lock()


def _write_tombstone(project_id: int, slug: str, max_version: int) -> None:
    """Read-modify-write under a project-wide lock (deletes of different slugs share one file), then an
    atomic temp-file + rename. A corrupt file raises before anything is written."""
    with _TOMBSTONE_THREAD_LOCK:
        data = read_tombstones(project_id)
        data[slug] = max(int(max_version), int(data.get(slug) or 0))
        path = _tombstones_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)


def read_version_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


# --------------------------------------------------------------------------
# Metadata helpers.
# --------------------------------------------------------------------------


def meta(ds: Dataset) -> dict[str, Any]:
    return dict(ds.metadata_ or {}) if isinstance(ds.metadata_, dict) else {}


def slug_of(ds: Dataset) -> str:
    return str(meta(ds).get("slug") or "")


def versions_of(ds: Dataset) -> list[dict[str, Any]]:
    versions = meta(ds).get("versions")
    return [v for v in versions if isinstance(v, dict)] if isinstance(versions, list) else []


def variants_of(ds: Dataset) -> dict[str, dict[str, Any]]:
    variants = meta(ds).get("variants")
    return {str(k): v for k, v in variants.items() if isinstance(v, dict)} if isinstance(variants, dict) else {}


def current_version(ds: Dataset) -> int:
    return int(meta(ds).get("current_version") or 0)


def _version_entry(ds: Dataset, version: int) -> dict[str, Any] | None:
    return next((v for v in versions_of(ds) if int(v.get("version") or 0) == int(version)), None)


def kept_leak_record(
    rows: list[dict[str, Any]], positions: list[int], lines: list[int] | None = None,
) -> dict[str, Any] | None:
    """The provenance of rows kept despite a leak flag, by their 0-based position in the version: the
    caller's row ``id`` when it has one, else ``#<1-based position>``. None when nothing was kept."""
    positions = sorted({int(p) for p in positions if 0 <= int(p) < len(rows)})
    if not positions:
        return None

    def rid(i: int) -> str:
        value = rows[i].get("id") if isinstance(rows[i], dict) else None
        return str(value) if value not in (None, "") else f"#{i + 1}"

    return {
        "count": len(positions),
        "row_indexes": positions,
        "row_ids": [rid(i) for i in positions],
        **({"lines": sorted(lines)} if lines else {}),
    }


def kept_leak_of(ds: Dataset, version: int) -> dict[str, Any] | None:
    value = (_version_entry(ds, version) or {}).get(KEPT_LEAK_KEY)
    return value if isinstance(value, dict) and value.get("count") else None


def _remap_kept_leak(record: dict[str, Any] | None, rows: list[dict[str, Any]], kept_old_positions: list[int]) -> dict[str, Any] | None:
    """``record`` (positions in an older version) for a new version made of the old rows at
    ``kept_old_positions`` (in order)."""
    if not record:
        return None
    flagged = {int(i) for i in record.get("row_indexes") or []}
    return kept_leak_record(rows, [new for new, old in enumerate(kept_old_positions) if old in flagged])


def serialize(ds: Dataset, *, last_evals: dict[str, Any] | None = None) -> dict[str, Any]:
    m = meta(ds)
    versions = versions_of(ds)
    cur = current_version(ds)
    cur_entry = _version_entry(ds, cur) or {}
    variants = [
        {"name": name, "spec": {k: v for k, v in spec.items() if k in ("system", "text")},
         "label": variant_label(spec), "digest": spec.get("digest"), "created_at": spec.get("created_at")}
        for name, spec in sorted(variants_of(ds).items())
    ]
    return {
        "id": ds.id,
        "slug": m.get("slug"),
        "name": ds.name,
        "description": ds.description or "",
        "current_version": cur,
        "rows": int(cur_entry.get("rows") or ds.record_count or 0),
        "is_locked": bool(ds.is_locked),
        "versions": [
            {k: v.get(k) for k in ("version", "rows", "sha256", "content_sha256", "created_at", "source", "dropped_lines",
                                   KEPT_LEAK_KEY)}
            for v in sorted(versions, key=lambda x: int(x.get("version") or 0))
        ],
        "variants": variants,
        "dataset_name": dataset_label(str(m.get("slug") or ""), cur) if cur else None,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "updated_at": ds.updated_at.isoformat() if ds.updated_at else None,
        **({"last_evals": last_evals} if last_evals is not None else {}),
    }


# --------------------------------------------------------------------------
# Queries + writes.
# --------------------------------------------------------------------------


async def list_eval_sets(db: AsyncSession, project_id: int) -> list[Dataset]:
    rows = await db.execute(
        select(Dataset)
        .where(Dataset.project_id == project_id, Dataset.dataset_type == DatasetType.EVAL_SET)
        .order_by(Dataset.id)
    )
    return list(rows.scalars().all())


async def get_eval_set(db: AsyncSession, project_id: int, slug: str, *, fresh: bool = False) -> Dataset:
    """The set with this slug. ``fresh=True`` (writers, under ``write_lock``): re-load it from the
    database, overwriting any copy this session already holds, so a write never starts from stale
    metadata."""
    slug = normalize_slug(slug)
    stmt = (
        select(Dataset)
        .where(Dataset.project_id == project_id, Dataset.dataset_type == DatasetType.EVAL_SET)
        .order_by(Dataset.id)
    )
    if fresh:
        stmt = stmt.execution_options(populate_existing=True)
    for ds in (await db.execute(stmt)).scalars().all():
        if slug_of(ds) == slug:
            return ds
    raise EvalSetError(f"Eval set '{slug}' not found in project {project_id}.", code="EVAL_SET_NOT_FOUND", status=404)


async def _record_version(
    db: AsyncSession, ds: Dataset, project_id: int, slug: str, rows: list[dict[str, Any]], source: dict[str, Any],
    dropped_lines: list[int] | None = None, kept_leak: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Never reuse a number: past versions in the metadata, any DatasetVersion row of this set, and the
    # slug's tombstone (a deleted set of the same name) all count.
    existing = await db.execute(select(DatasetVersion.version).where(DatasetVersion.dataset_id == ds.id))
    version = max(
        [int(v.get("version") or 0) for v in versions_of(ds)]
        + [int(v or 0) for v in existing.scalars().all()]
        + [int(read_tombstones(project_id).get(slug) or 0), 0]
    ) + 1
    path = version_path(project_id, slug, version)
    hashes = await asyncio.to_thread(_write_rows, path, rows)
    entry = {
        "version": version,
        "rows": len(rows),
        "sha256": hashes["sha256"],
        "content_sha256": hashes["content_sha256"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        **({"dropped_lines": sorted(dropped_lines)} if dropped_lines else {}),
        **({KEPT_LEAK_KEY: kept_leak} if kept_leak else {}),
    }
    m = meta(ds)
    m["versions"] = [*versions_of(ds), entry]
    m["current_version"] = version
    ds.metadata_ = m
    ds.file_path = str(path)
    ds.record_count = len(rows)
    ds.updated_at = datetime.now(timezone.utc)
    db.add(DatasetVersion(
        dataset_id=ds.id, version=version, file_path=str(path), record_count=len(rows),
        manifest={"sha256": hashes["sha256"], "content_sha256": hashes["content_sha256"], "rows": len(rows),
                  "source": source, "eval_set": slug, **({KEPT_LEAK_KEY: kept_leak} if kept_leak else {})},
    ))
    return entry


async def create_eval_set(
    db: AsyncSession, project_id: int, *, name: str, rows: list[dict[str, Any]], source: dict[str, Any],
    description: str = "", dropped_lines: list[int] | None = None, kept_leak: dict[str, Any] | None = None,
) -> Dataset:
    name = str(name or "").strip()
    if not name:
        raise EvalSetError("An eval set needs a name.")
    if len(name) > MAX_NAME_CHARS:
        raise EvalSetError(f"The name is longer than {MAX_NAME_CHARS} characters.")
    if not rows:
        raise EvalSetError("An eval set needs at least one row.", code="EVAL_SET_NO_ROWS", status=422)
    slug = slugify(name)
    existing = {slug_of(d) for d in await list_eval_sets(db, project_id)}
    if slug in existing:
        raise EvalSetError(
            f"An eval set '{slug}' already exists. Upload a new version of it, or pick another name.",
            code="EVAL_SET_EXISTS", status=409,
        )
    ds = Dataset(
        project_id=project_id, name=name, dataset_type=DatasetType.EVAL_SET, description=description or "",
        record_count=0, file_path="",
        metadata_={"eval_set": True, "slug": slug, "current_version": 0, "versions": [], "variants": {}},
    )
    db.add(ds)
    await db.flush()
    await _record_version(db, ds, project_id, slug, rows, source, dropped_lines, kept_leak)
    await db.commit()
    await db.refresh(ds)
    return ds


def _refuse_locked(ds: Dataset) -> None:
    if ds.is_locked:
        raise EvalSetError(
            f"Eval set '{slug_of(ds)}' is locked; it cannot change. Create a new set (from this one) instead.",
            code="EVAL_SET_LOCKED", status=409,
        )


async def add_version(
    db: AsyncSession, ds: Dataset, rows: list[dict[str, Any]], source: dict[str, Any],
    dropped_lines: list[int] | None = None, kept_leak: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _refuse_locked(ds)
    if not rows:
        raise EvalSetError("A version needs at least one row.", code="EVAL_SET_NO_ROWS", status=422)
    entry = await _record_version(db, ds, int(ds.project_id), slug_of(ds), rows, source, dropped_lines, kept_leak)
    await db.commit()
    await db.refresh(ds)
    return entry


def version_file(ds: Dataset, version: int | None = None) -> tuple[int, Path]:
    ver = int(version) if version is not None else current_version(ds)
    if _version_entry(ds, ver) is None:
        raise EvalSetError(
            f"Eval set '{slug_of(ds)}' has no version {ver} (versions: "
            f"{[v.get('version') for v in versions_of(ds)]}).", code="EVAL_SET_VERSION_NOT_FOUND", status=404,
        )
    return ver, version_path(int(ds.project_id), slug_of(ds), ver)


async def delete_rows(db: AsyncSession, ds: Dataset, row_indexes: list[int], *, version: int | None = None) -> dict[str, Any]:
    """A new version without the rows at these 0-based positions of ``version`` (default: current)."""
    _refuse_locked(ds)
    ver, path = version_file(ds, version)
    rows = await asyncio.to_thread(read_version_rows, path)
    drop = {int(i) for i in row_indexes}
    missing = sorted(i for i in drop if i < 0 or i >= len(rows))
    if missing:
        raise EvalSetError(f"Row position(s) {missing[:10]} are not in version {ver} ({len(rows)} rows).")
    kept_positions = [i for i in range(len(rows)) if i not in drop]
    kept = [rows[i] for i in kept_positions]
    if not kept:
        raise EvalSetError("That would leave no rows; delete the eval set instead.", code="EVAL_SET_NO_ROWS", status=422)
    entry = await add_version(
        db, ds, kept, {"kind": "rows_deleted", "from_version": ver, "deleted": len(drop)},
        kept_leak=_remap_kept_leak(kept_leak_of(ds, ver), kept, kept_positions),
    )
    return {"deleted": len(drop), "version": entry}


async def add_variant(db: AsyncSession, ds: Dataset, name: Any, spec: Any) -> dict[str, Any]:
    vname = normalize_variant_name(name)
    norm = normalize_variant(spec)
    variants = variants_of(ds)
    if vname in variants:
        raise EvalSetError(
            f"Variant '{vname}' already exists on '{slug_of(ds)}' (variants are immutable: delete it or pick another name).",
            code="EVAL_SET_VARIANT_EXISTS", status=409,
        )
    record = {**norm, "digest": variant_digest(norm), "created_at": datetime.now(timezone.utc).isoformat()}
    m = meta(ds)
    m["variants"] = {**variants, vname: record}
    ds.metadata_ = m
    await db.commit()
    await db.refresh(ds)
    return {"name": vname, **record}


async def delete_variant(db: AsyncSession, ds: Dataset, name: str) -> None:
    variants = variants_of(ds)
    if name not in variants:
        raise EvalSetError(f"Variant '{name}' not found on '{slug_of(ds)}'.", code="EVAL_SET_VARIANT_NOT_FOUND", status=404)
    variants.pop(name)
    m = meta(ds)
    m["variants"] = variants
    ds.metadata_ = m
    await db.commit()
    await db.refresh(ds)


async def lock(db: AsyncSession, ds: Dataset) -> None:
    ds.is_locked = True
    await db.commit()
    await db.refresh(ds)


async def active_eval_job_ids(db: AsyncSession, project_id: int, slug: str) -> list[int]:
    from app.models.job import Job, JobStatus

    rows = await db.execute(
        select(Job).where(Job.project_id == project_id, Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
    )
    out: list[int] = []
    for job in rows.scalars().all():
        name = (job.params or {}).get("dataset_name") if isinstance(job.params, dict) else None
        if is_eval_set_name(name):
            try:
                if parse_label(str(name))[0] == slug:
                    out.append(int(job.id))
            except EvalSetError:
                continue
    return out


async def delete_eval_set(db: AsyncSession, ds: Dataset) -> None:
    _refuse_locked(ds)
    slug = slug_of(ds)
    jobs = await active_eval_job_ids(db, int(ds.project_id), slug)
    if jobs:
        raise EvalSetError(
            f"Eval set '{slug}' is in use by eval Job(s) {jobs}; wait for them to finish.",
            code="EVAL_SET_IN_USE", status=409,
        )
    project_id = int(ds.project_id)
    max_version = max([int(v.get("version") or 0) for v in versions_of(ds)] + [0])
    # The tombstone first: a recreated slug continues after the highest version this one had.
    await asyncio.to_thread(_write_tombstone, project_id, slug, max_version)
    await db.delete(ds)
    await db.commit()
    target = eval_sets_dir(project_id) / slug
    if target.is_dir():
        await asyncio.to_thread(shutil.rmtree, target, True)


async def rows_page(ds: Dataset, *, version: int | None, offset: int, limit: int) -> dict[str, Any]:
    ver, path = version_file(ds, version)
    rows = await asyncio.to_thread(read_version_rows, path)
    page = rows[offset: offset + limit]
    return {
        "slug": slug_of(ds), "version": ver, "total": len(rows), "offset": offset, "limit": limit,
        "rows": [{"position": offset + i, "row": r} for i, r in enumerate(page)],
    }


# --------------------------------------------------------------------------
# Sources: another eval set, or a built-in set.
# --------------------------------------------------------------------------


async def rows_from_source(db: AsyncSession, project_id: int, source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(rows, provenance)`` for ``{kind: "eval_set", slug, version?}`` or ``{kind: "split", split}``
    (test / validation / gold_test / gold_dev, read as the eval reads them: rejected answer-key rows left
    out)."""
    kind = str((source or {}).get("kind") or "").strip()
    if kind == "eval_set":
        src = await get_eval_set(db, project_id, str(source.get("slug") or ""))
        ver, path = version_file(src, source.get("version"))
        rows = await asyncio.to_thread(read_version_rows, path)
        prov: dict[str, Any] = {"kind": "eval_set", "slug": slug_of(src), "version": ver}
        kept = kept_leak_of(src, ver)
        if kept:  # a copy keeps the record of rows kept despite a leak flag (same rows, same order)
            prov[KEPT_LEAK_KEY] = _remap_kept_leak(kept, rows, list(range(len(rows))))
        return rows, prov
    if kind == "split":
        split = str(source.get("split") or "").strip().lower()
        if split not in SPLIT_SOURCES:
            raise EvalSetError(f"split must be one of {list(SPLIT_SOURCES)}.")
        from app.services.evaluation_service import _load_eval_rows, _resolve_heldout_dataset
        from app.services.compare_row_evidence_service import answer_key_stamp

        try:
            ds = await _resolve_heldout_dataset(db, project_id, split)
        except ValueError as exc:
            raise EvalSetError(str(exc), code="EVAL_SET_SOURCE_NOT_FOUND", status=404) from exc
        path = Path(str(ds.file_path)).expanduser().resolve()

        async def read() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            try:
                rows = await asyncio.to_thread(_load_eval_rows, path)
            except FileNotFoundError as exc:
                raise EvalSetError(str(exc), code="EVAL_SET_SOURCE_NOT_FOUND", status=404) from exc
            return rows, await answer_key_stamp(db, ds, path)

        if ds.dataset_type in (DatasetType.GOLD_DEV, DatasetType.GOLD_TEST):
            # An answer key is rewritten in place by its writers (review, import, row deletes): read the
            # rows and stamp the sha under their lock, so the copy and its provenance agree.
            from app.services.gold_service import gold_write_lock

            async with gold_write_lock(project_id, ds.dataset_type):
                rows, stamp = await read()
        else:
            rows, stamp = await read()
        return rows, {"kind": "split", "split": split, "dataset_type": ds.dataset_type.value,
                      "dataset_id": ds.id, "sha256": stamp.get("sha256"), "split_version": stamp.get("version")}
    raise EvalSetError("source.kind must be 'eval_set' or 'split'.")


# --------------------------------------------------------------------------
# Leakage scan on import.
# --------------------------------------------------------------------------


async def _prepared_like(db: AsyncSession, project_id: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``rows`` mapped the way the project's training rows were (its adapter + task profile, so a tool
    row gets its rendered context block in ``question`` exactly as training did), one output per input
    row: a row the adapter cannot map stays as given. Best-effort, never raises."""
    try:
        from app.services.dataset_service import (
            _normalize_rows_for_training,
            resolve_project_dataset_adapter_preference,
        )
        from app.services.eval_task_handler_service import read_prepared_manifest

        preset = await resolve_project_dataset_adapter_preference(db, project_id)
        manifest = await asyncio.to_thread(read_prepared_manifest, int(project_id))
        manifest = manifest if isinstance(manifest, dict) else {}
        rejected: list[tuple[int, str]] = []
        mapped = _normalize_rows_for_training(
            [dict(r) for r in rows], DatasetType.TEST, str(manifest.get("chat_template") or "llama3"),
            adapter_id=str(manifest.get("adapter_id") or preset.get("adapter_id") or "default-canonical"),
            adapter_config=dict(preset.get("adapter_config") or {}),
            field_mapping=dict(preset.get("field_mapping") or {}),
            task_profile=(manifest.get("task_profile") or preset.get("task_profile") or None),
            rejected=rejected,
        )
    except Exception:  # noqa: BLE001 - the scan falls back to the rows as given
        return rows
    dropped = {i for i, _reason in rejected}
    it = iter(mapped)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i in dropped:
            out.append(row)
            continue
        out.append(dict(next(it, row)))
    return out


def _match_view(row: dict[str, Any]) -> dict[str, Any]:
    """A prepared row without its chat-rendered ``text`` (a copy of question + answer wrapped in
    template tokens), so a prepared training row and an imported row compare on their content."""
    if isinstance(row.get("question"), str) and "answer" in row and "text" in row:
        return {k: v for k, v in row.items() if k != "text"}
    return row


async def leak_scan(db: AsyncSession, project_id: int, rows: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    """Rows of an eval-set import that leak the project's training data: ``[(row index, match kind,
    matched training excerpt)]``. The data-health train/test matcher, reused as is (exact normalised
    text, then token-set near-duplicates), against what the model trains on (the training corpus +
    the prepared validation split), with the same ignored lines: the rendered context blocks (a row's
    own tool list or the project tool block, Phase 2) and, when the project opted in, shared lines.
    Empty when the project has no training rows yet."""
    if not rows:
        return []
    from app.models.project import Project
    from app.services import data_health_service as dhs
    from app.services.dataset_service import _load_records_from_file
    from app.services.trainability_forecast_service import _row_to_text

    train_rows, _truncated = await dhs._load_training_corpus_rows(db, project_id, cap=dhs.LEAKAGE_TRAIN_SCAN_CAP)
    val = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id, Dataset.dataset_type == DatasetType.VALIDATION)
    )
    for ds in val.scalars().all():
        if ds.file_path and Path(ds.file_path).exists():
            train_rows.extend(r for r in _load_records_from_file(Path(ds.file_path)) if isinstance(r, dict))
    if not train_rows:
        return []
    train_rows = [_match_view(r) for r in train_rows]
    needles = [_match_view(r) for r in await _prepared_like(db, project_id, rows)]
    project = await db.get(Project, project_id)
    boilerplate = dhs.leakage_boilerplate(
        train_rows + needles, getattr(project, "runtime_config", None),
        stored=dhs.load_stored_shared_lines(project_id),
    )

    def scan() -> list[tuple[int, str, str]]:
        index = dhs._build_leakage_index(train_rows, boilerplate)
        hits: list[tuple[int, str, str]] = []
        for i, row in enumerate(needles):
            text = _row_to_text(row)
            if not text.strip():
                continue
            kind, _jaccard, midx = dhs._match_row_against_index(text, index)
            if kind:
                hits.append((i, kind, dhs._excerpt(index["texts"][midx])[:80] if midx >= 0 else ""))
        return hits

    return await asyncio.to_thread(scan)


# --------------------------------------------------------------------------
# Eval-time resolution + provenance.
# --------------------------------------------------------------------------


async def resolve_for_eval(
    db: AsyncSession, project_id: int, dataset_name: str, prompt_variant: str | None = None,
) -> tuple[Dataset, Path, dict[str, Any]]:
    """``(dataset, version file, info)`` for an eval on ``eval_set:<slug>[@vN][+variant]``. ``prompt_variant``
    (the request field) sets the variant when the name has none; both given and different is an error."""
    slug, version, variant = parse_label(dataset_name)
    if prompt_variant:
        pv = normalize_variant_name(prompt_variant)
        if variant and variant != pv:
            raise EvalSetError(f"dataset_name names variant '{variant}' but prompt_variant is '{pv}'.")
        variant = pv
    ds = await get_eval_set(db, project_id, slug)
    ver, path = version_file(ds, version)
    spec: dict[str, Any] | None = None
    if variant:
        record = variants_of(ds).get(variant)
        if record is None:
            raise EvalSetError(
                f"Eval set '{slug}' has no variant '{variant}' (variants: {sorted(variants_of(ds))}).",
                code="EVAL_SET_VARIANT_NOT_FOUND", status=404,
            )
        spec = {k: v for k, v in record.items() if k in ("system", "text")}
    entry = _version_entry(ds, ver) or {}
    await verify_version_file(ds, ver, path, entry)
    info = {
        "slug": slug,
        "name": ds.name,
        "version": ver,
        "sha256": entry.get("sha256"),
        "content_sha256": entry.get("content_sha256"),
        "variant": variant,
        "variant_spec": spec,
        "variant_digest": variant_digest(spec),
        "rows": int(entry.get("rows") or 0),
        "label": dataset_label(slug, ver, variant),
    }
    kept = kept_leak_of(ds, ver)
    if kept:  # shown on the result: rows scored although they leak training data
        info[KEPT_LEAK_KEY] = {k: kept.get(k) for k in ("count", "row_indexes", "row_ids")}
    return ds, path, info


async def verify_version_file(ds: Dataset, version: int, path: Path, entry: dict[str, Any] | None = None) -> None:
    """Refuse an eval on a version file that is missing or whose sha256 is not the one recorded when
    the version was made (an edited or swapped file would score other rows under the same label)."""
    from app.services.dataset_service import _sha256_file

    entry = entry if entry is not None else (_version_entry(ds, version) or {})
    expected = entry.get("sha256")
    if not path.exists():
        raise EvalSetError(
            f"Eval set '{slug_of(ds)}' v{version}: its file is missing ({path.name}). Upload a new version.",
            code="EVAL_SET_FILE_MISSING", status=409,
        )
    if not expected:
        return
    actual = await asyncio.to_thread(_sha256_file, path)
    if actual != expected:
        raise EvalSetError(
            f"Eval set '{slug_of(ds)}' v{version}: the file changed after the version was made (sha256 "
            f"{str(actual)[:12]} vs recorded {str(expected)[:12]}). Versions are immutable; upload the rows "
            "as a new version instead.",
            code="EVAL_SET_FILE_CHANGED", status=409,
        )


def eval_set_block(details: Any) -> dict[str, Any] | None:
    """``details.dataset.eval_set`` of a result (None for a built-in-set result)."""
    if not isinstance(details, dict):
        return None
    dataset = details.get("dataset")
    block = dataset.get("eval_set") if isinstance(dataset, dict) else None
    if not isinstance(block, dict):
        return None
    if not block.get("content_sha256") and isinstance(dataset.get("content_sha256"), str):
        # A result from before the block carried the hash: the file's content hash the eval stamped.
        block = {**block, "content_sha256": dataset["content_sha256"]}
    return block


def _side_label(block: dict[str, Any] | None) -> str:
    if not block:
        return "a built-in set"
    text = f"eval set {block.get('slug')} v{block.get('version')}"
    return text + (f" ({block.get('variant')})" if block.get("variant") else "")


def _variant_text(variant: Any) -> str:
    return f"'{variant}' variant" if variant else "rows as written (no variant)"


def cross_variant_pair(
    a: dict[str, Any] | None, b: dict[str, Any] | None, *, sides: tuple[str, str] = ("A", "B"),
) -> dict[str, Any] | None:
    """Two results on DIFFERENT prompt-format variants of the SAME eval-set version: same slug, same
    version and the same recorded content hash (both must be recorded), different variant names. Such a
    pair may be compared on the rows both evaluated, labelled: both the model and the prompt differ.
    None for any other pair (same variant, a redefined variant under one name, another set / version, a
    missing or different content hash)."""
    if not a or not b:
        return None
    if a.get("slug") != b.get("slug") or a.get("version") != b.get("version"):
        return None
    ca, cb = a.get("content_sha256") or None, b.get("content_sha256") or None
    if not ca or not cb or ca != cb:
        return None
    va, vb = a.get("variant") or None, b.get("variant") or None
    if va == vb:
        return None
    return {
        "slug": a.get("slug"),
        "version": a.get("version"),
        "variant_a": va,
        "variant_b": vb,
        "label": (
            f"Cross-variant pair on eval set {a.get('slug')} v{a.get('version')}: {sides[0]} answered the "
            f"{_variant_text(va)}, {sides[1]} the {_variant_text(vb)}. Both the model and the prompt differ, so the "
            "change is not the model's alone; only rows both evaluated are paired."
        ),
    }


def rekey_by_set_row(row_scores: Any) -> Any:
    """``details.row_scores`` keyed by the row's position in the eval-set version (``set_rows``) instead of
    its prompt + reference, so two variants of one version (other system message, same rows) pair row for
    row. Unchanged when ``set_rows`` is missing or misaligned (older results pair on identical prompts only)."""
    if not isinstance(row_scores, dict):
        return row_scores
    rows, keys = row_scores.get("set_rows"), row_scores.get("keys")
    if not isinstance(rows, list) or not isinstance(keys, list) or len(rows) != len(keys):
        return row_scores
    if any(r is None for r in rows):
        return row_scores
    return {**row_scores, "keys": [f"set-row-{int(r)}" for r in rows]}


def eval_set_difference(
    a: dict[str, Any] | None, b: dict[str, Any] | None, *, allow_cross_variant: bool = False,
) -> str | None:
    """Why two results' eval sets cannot be paired (None = same set + version + variant, or neither is an
    eval-set result). Compare and the lift view refuse such a pair with this reason. ``allow_cross_variant``:
    a ``cross_variant_pair`` (same version + content hash, other variant) is pairable too (labelled)."""
    if allow_cross_variant and cross_variant_pair(a, b) is not None:
        return None
    if not a and not b:
        return None
    if not a or not b:
        return f"different eval set: {_side_label(a)} vs {_side_label(b)}"
    if a.get("slug") != b.get("slug"):
        return f"different eval set: {_side_label(a)} vs {_side_label(b)}"
    if a.get("version") != b.get("version"):
        return f"different eval-set version: {_side_label(a)} vs {_side_label(b)}"
    ca, cb = a.get("content_sha256") or None, b.get("content_sha256") or None
    if ca and cb and ca != cb:
        # Same label, other rows: a deleted set recreated under the same name before version numbers
        # were kept across deletes, or a file edited outside BrewSLM.
        return (f"different rows under the same label: {_side_label(a)} (content {str(ca)[:8]}) vs "
                f"{_side_label(b)} (content {str(cb)[:8]})")
    if (a.get("variant") or None) != (b.get("variant") or None):
        return f"different prompt-format variant: {_side_label(a)} vs {_side_label(b)}"
    if (a.get("variant_digest") or None) != (b.get("variant_digest") or None):
        return (f"prompt-format variant '{a.get('variant')}' was redefined between the two evals "
                f"({a.get('variant_digest')} vs {b.get('variant_digest')})")
    return None


async def eval_set_results_for_run(db: AsyncSession, experiment_id: int, limit: int = 5) -> list[dict[str, Any]]:
    """A run's eval-set results, newest per label (decision D3 keeps them out of "the latest eval"):
    ``[{eval_result_id, dataset_name, label, pass_rate}]``, newest first, at most ``limit``. Readers that find
    no built-in result use it to say "evaluated on eval sets only" instead of "not evaluated".

    Each entry also says ``has_base``: a base-model result (a baseline run of the run's own base model) exists on
    that label, so base vs fine-tuned is measured there (UI simplification phase 7b: the Ship Coach no longer warns
    "not evaluated" over a complete eval-set comparison)."""
    from app.models.experiment import EvalResult, Experiment

    rows = await db.execute(
        select(EvalResult.id, EvalResult.dataset_name, EvalResult.pass_rate)
        .where(EvalResult.experiment_id == experiment_id, EvalResult.dataset_name.startswith(PREFIX, autoescape=True))
        .order_by(desc(EvalResult.id))
        .limit(200)
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rid, name, pass_rate in rows.all():
        if name in seen:
            continue
        seen.add(name)
        out.append({"eval_result_id": rid, "dataset_name": name, "label": human_label(name), "pass_rate": pass_rate})
        if len(out) >= limit:
            break
    if out:
        run = (await db.execute(
            select(Experiment.project_id, Experiment.base_model).where(Experiment.id == experiment_id)
        )).first()
        if run is not None:
            project_id, run_base = run
            last = await last_evals_by_label(db, int(project_id))
            for entry in out:
                base = (last.get(entry["dataset_name"]) or {}).get("base") or {}
                entry["has_base"] = bool(base) and (not run_base or base.get("base_model") in (None, run_base))
    return out


async def last_evals_by_label(
    db: AsyncSession, project_id: int, sets: list[Dataset] | None = None,
) -> dict[str, dict[str, Any]]:
    """Newest result per eval-set dataset name in the project: ``{label: {eval_result_id, experiment_id,
    pass_rate, created_at, content_sha256}}`` (for the card's "last eval" column) -- the newest result of a
    trained run when there is one, else of a baseline run (``baseline: true``) -- plus ``base``: the newest
    baseline-run (base model) result on that label, ``{eval_result_id, experiment_id, pass_rate, base_model}``,
    when there is one, so the card shows both sides of base vs fine-tuned. With ``sets``, a result
    counts only when its rows are the rows of that set's version today (``content_sha256`` of the result's
    ``eval_set`` block = the version's): a result from a deleted set that had the same name, or from an
    edited file, is never shown as this set's last eval."""
    from app.models.experiment import EvalResult, Experiment

    hashes: dict[tuple[str, int], str | None] | None = None
    if sets is not None:
        hashes = {
            (slug_of(ds), int(v.get("version") or 0)): v.get("content_sha256")
            for ds in sets for v in versions_of(ds)
        }
    rows = await db.execute(
        select(EvalResult.id, EvalResult.experiment_id, EvalResult.dataset_name, EvalResult.pass_rate,
               EvalResult.created_at, EvalResult.details, Experiment.config, Experiment.base_model)
        .join(Experiment, Experiment.id == EvalResult.experiment_id)
        .where(Experiment.project_id == project_id,
               EvalResult.dataset_name.startswith(PREFIX, autoescape=True))
        .order_by(desc(EvalResult.id))
        .limit(1000)
    )
    out: dict[str, dict[str, Any]] = {}
    for rid, exp_id, name, pass_rate, created, details, exp_config, base_model in rows.all():
        is_base = isinstance(exp_config, dict) and exp_config.get("is_baseline") is True
        current = out.get(name)
        if current is not None and (current.get("base") is not None or not is_base) and (is_base or not current.get("baseline")):
            continue
        block = eval_set_block(details) or {}
        content = block.get("content_sha256")
        if hashes is not None:
            try:
                slug, version, _variant = parse_label(str(name))
            except EvalSetError:
                continue
            key = (slug, int(version or 0))
            if key not in hashes:
                continue
            expected = hashes[key]
            if expected and content != expected:
                continue
        entry = {"eval_result_id": rid, "experiment_id": exp_id, "pass_rate": pass_rate,
                 "created_at": created.isoformat() if created else None, "content_sha256": content}
        if is_base:
            entry["baseline"] = True
        if current is None:
            out[name] = {**entry, "base": None}
        elif current.get("baseline") and not is_base:
            out[name] = {**entry, "base": current.get("base")}  # a trained run's result leads the row
        if is_base and out[name].get("base") is None:
            out[name]["base"] = {"eval_result_id": rid, "experiment_id": exp_id, "pass_rate": pass_rate,
                                 "base_model": base_model}
    return out
