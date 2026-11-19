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
import re
import shutil
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
# Built-in sets an eval set can be copied from (``from_split``).
SPLIT_SOURCES = ("test", "validation", "gold_test", "gold_dev")

_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


class EvalSetError(ValueError):
    """A request the eval-set service refuses. ``code`` is the API error code, ``status`` the HTTP status."""

    def __init__(self, message: str, *, code: str = "EVAL_SET_INVALID", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def write_lock(project_id: int, slug: str) -> asyncio.Lock:
    key = (int(project_id), slug)
    lock = _LOCKS.get(key)
    if lock is None:
        lock = _LOCKS[key] = asyncio.Lock()
    return lock


# --------------------------------------------------------------------------
# Names and labels.
# --------------------------------------------------------------------------


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
            {k: v.get(k) for k in ("version", "rows", "sha256", "content_sha256", "created_at", "source", "dropped_lines")}
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


async def get_eval_set(db: AsyncSession, project_id: int, slug: str) -> Dataset:
    slug = str(slug or "").strip().lower()
    for ds in await list_eval_sets(db, project_id):
        if slug_of(ds) == slug:
            return ds
    raise EvalSetError(f"Eval set '{slug}' not found in project {project_id}.", code="EVAL_SET_NOT_FOUND", status=404)


async def _record_version(
    db: AsyncSession, ds: Dataset, project_id: int, slug: str, rows: list[dict[str, Any]], source: dict[str, Any],
    dropped_lines: list[int] | None = None,
) -> dict[str, Any]:
    version = max([int(v.get("version") or 0) for v in versions_of(ds)] + [0]) + 1
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
                  "source": source, "eval_set": slug},
    ))
    return entry


async def create_eval_set(
    db: AsyncSession, project_id: int, *, name: str, rows: list[dict[str, Any]], source: dict[str, Any],
    description: str = "", dropped_lines: list[int] | None = None,
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
    await _record_version(db, ds, project_id, slug, rows, source, dropped_lines)
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
    dropped_lines: list[int] | None = None,
) -> dict[str, Any]:
    _refuse_locked(ds)
    if not rows:
        raise EvalSetError("A version needs at least one row.", code="EVAL_SET_NO_ROWS", status=422)
    entry = await _record_version(db, ds, int(ds.project_id), slug_of(ds), rows, source, dropped_lines)
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
    kept = [r for i, r in enumerate(rows) if i not in drop]
    if not kept:
        raise EvalSetError("That would leave no rows; delete the eval set instead.", code="EVAL_SET_NO_ROWS", status=422)
    entry = await add_version(db, ds, kept, {"kind": "rows_deleted", "from_version": ver, "deleted": len(drop)})
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


async def lock(db: AsyncSession, ds: Dataset) -> None:
    ds.is_locked = True
    await db.commit()


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
        return rows, {"kind": "eval_set", "slug": slug_of(src), "version": ver}
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
        try:
            rows = await asyncio.to_thread(_load_eval_rows, path)
        except FileNotFoundError as exc:
            raise EvalSetError(str(exc), code="EVAL_SET_SOURCE_NOT_FOUND", status=404) from exc
        stamp = await answer_key_stamp(db, ds, path)
        return rows, {"kind": "split", "split": split, "dataset_type": ds.dataset_type.value,
                      "dataset_id": ds.id, "sha256": stamp.get("sha256"), "split_version": stamp.get("version")}
    raise EvalSetError("source.kind must be 'eval_set' or 'split'.")


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
    info = {
        "slug": slug,
        "name": ds.name,
        "version": ver,
        "variant": variant,
        "variant_spec": spec,
        "variant_digest": variant_digest(spec),
        "rows": int(entry.get("rows") or 0),
        "label": dataset_label(slug, ver, variant),
    }
    return ds, path, info


def eval_set_block(details: Any) -> dict[str, Any] | None:
    """``details.dataset.eval_set`` of a result (None for a built-in-set result)."""
    if not isinstance(details, dict):
        return None
    dataset = details.get("dataset")
    block = dataset.get("eval_set") if isinstance(dataset, dict) else None
    return block if isinstance(block, dict) else None


def _side_label(block: dict[str, Any] | None) -> str:
    if not block:
        return "a built-in set"
    text = f"eval set {block.get('slug')} v{block.get('version')}"
    return text + (f" ({block.get('variant')})" if block.get("variant") else "")


def eval_set_difference(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str | None:
    """Why two results' eval sets cannot be paired (None = same set + version + variant, or neither is an
    eval-set result). Compare and the lift view refuse such a pair with this reason."""
    if not a and not b:
        return None
    if not a or not b:
        return f"different eval set: {_side_label(a)} vs {_side_label(b)}"
    if a.get("slug") != b.get("slug"):
        return f"different eval set: {_side_label(a)} vs {_side_label(b)}"
    if a.get("version") != b.get("version"):
        return f"different eval-set version: {_side_label(a)} vs {_side_label(b)}"
    if (a.get("variant") or None) != (b.get("variant") or None):
        return f"different prompt-format variant: {_side_label(a)} vs {_side_label(b)}"
    if (a.get("variant_digest") or None) != (b.get("variant_digest") or None):
        return (f"prompt-format variant '{a.get('variant')}' was redefined between the two evals "
                f"({a.get('variant_digest')} vs {b.get('variant_digest')})")
    return None


async def last_evals_by_label(db: AsyncSession, project_id: int) -> dict[str, dict[str, Any]]:
    """Newest result per eval-set dataset name in the project: ``{label: {eval_result_id, experiment_id,
    pass_rate, created_at}}`` (for the card's "last eval" column)."""
    from app.models.experiment import EvalResult, Experiment

    rows = await db.execute(
        select(EvalResult.id, EvalResult.experiment_id, EvalResult.dataset_name, EvalResult.pass_rate,
               EvalResult.created_at)
        .join(Experiment, Experiment.id == EvalResult.experiment_id)
        .where(Experiment.project_id == project_id,
               EvalResult.dataset_name.startswith(PREFIX, autoescape=True))
        .order_by(desc(EvalResult.id))
        .limit(2000)
    )
    out: dict[str, dict[str, Any]] = {}
    for rid, exp_id, name, pass_rate, created in rows.all():
        if name in out:
            continue
        out[name] = {"eval_result_id": rid, "experiment_id": exp_id, "pass_rate": pass_rate,
                     "created_at": created.isoformat() if created else None}
    return out
