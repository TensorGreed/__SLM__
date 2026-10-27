"""Domain packs as data: a pack zip carries the contract (overlay / hooks) *and*
content (taxonomy, user probes, rubrics, schemas, tool catalog, documents, gold
rows, a project template). This module validates a pack, stores each imported
version on disk, diffs versions, instantiates a project from a pack, and exports
a project back to the same layout.

Layout (zip root, or a single top-level folder, holds ``pack.yaml``)::

    pack.yaml  taxonomy.yaml  probes/*.yaml  rubrics/*  schemas/*.json
    tool_catalog.yaml  documents/*  gold/dev.jsonl  gold/test.jsonl

No table and no migration: files of each version live under
``DATA_DIR/domain_packs/<pack_id>/<version>/`` and the existing ``DomainPack``
row indexes them in ``contract["content"]`` / ``contract["content_versions"]``.

Storage choices worth knowing:

* ``project_template.training_defaults`` lands in
  ``runtime_config["training_config_overrides"]`` -- the block the Train panel
  prefills its form from (``GET .../training-config-gaps/overrides``) and the
  gap scanner reads. Keys are validated against ``TrainingConfig``.
* Schemas / rubrics / tool catalog are carried as validated, versioned
  artifacts in ``runtime_config["pack_artifacts"]`` (no consumer yet); a schema
  flagged ``apply_as_output_schema`` also goes through the recipe output-schema
  setter when the recipe supports one.
* The content hash ignores the ``version`` field, so re-importing identical
  content under a new version number is "unchanged".
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
import stat
import tempfile
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import yaml
from pydantic import ValidationError

from app.services.label_taxonomy_service import _NoAliasLoader  # the one alias-refusing loader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

SCHEMA_V2 = "brewslm.domain-pack/v2"
KNOWN_SCHEMAS = {SCHEMA_V2, "slm.domain-pack/v1"}
PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")  # >= 3 chars, as DomainPackContract requires
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES = 2000
MAX_ARTIFACT_BYTES = 1024 * 1024

DEFAULT_TARGET_PROFILE = "vllm_server"
DEFAULT_PLAN_PROFILE = "balanced"
GOLD_DEFAULTS = {"difficulty": "medium", "criticality": "normal", "is_hallucination_trap": False, "metadata": {}}
GOLD_SYSTEM_FIELDS = ("id", "created_at")

KNOWN_KEYS = {
    "schema", "$schema", "pack_id", "version", "display_name", "description", "owner", "tags", "status",
    "overlay", "hooks", "default_profile_id", "taxonomy", "probe_packs", "rubrics", "schemas",
    "tool_catalog", "documents", "gold", "project_template",
}
TEMPLATE_KEYS = ("recipe_id", "base_model", "target_profile", "plan_profile", "training_defaults", "retrieval")


class PackContentError(Exception):
    """A pack operation failed. ``status`` maps to HTTP; ``errors`` is a list of issues."""

    def __init__(self, status: int, code: str, message: str, errors: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.errors = errors or []


class PackValidation(dict):
    """The validation result: a dict (JSON-ready) whose keys also read as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class Parsed:
    """Everything validation parsed, kept for import / diff / instantiate."""

    taxonomy: dict[str, Any] | None = None
    probe_packs: list[dict[str, Any]] = field(default_factory=list)
    rubrics: list[dict[str, Any]] = field(default_factory=list)  # {id, content}
    schemas: list[dict[str, Any]] = field(default_factory=list)  # {id, schema, apply_as_output_schema}
    tool_catalog: Any = None
    has_tool_catalog: bool = False
    documents: list[dict[str, Any]] = field(default_factory=list)  # {path, sha256, title, licence, usage, locator}
    gold: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    template: dict[str, Any] = field(default_factory=dict)


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _load_text(data: bytes, path: str) -> Any:
    text = data.decode("utf-8-sig", "replace")
    if path.lower().endswith(".json"):
        return json.loads(text)
    return yaml.load(text, Loader=_NoAliasLoader)  # noqa: S506 - SafeLoader subclass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jdump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _version_tuple(version: str) -> tuple[int, int, int]:
    m = VERSION_RE.match(str(version or "").strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)


def _bump_patch(version: str) -> str:
    a, b, c = _version_tuple(version)
    return f"{a}.{b}.{c + 1}"


def pack_storage_dir(pack_id: str, version: str) -> Path:
    return settings.DATA_DIR / "domain_packs" / pack_id / version


# ── Zip handling ───────────────────────────────────────────────────────────


def _unsafe_name(name: str) -> bool:
    if not name or name.startswith(("/", "\\")) or "\\" in name or "\x00" in name:
        return True
    if re.match(r"^[A-Za-z]:", name):
        return True
    return ".." in PurePosixPath(name).parts


@contextlib.contextmanager
def extracted_archive(data: bytes) -> Iterator[tuple[Path, list[dict[str, str]]]]:
    """Extract a pack zip to a temp dir; yields ``(root, early_errors)``. Unsafe
    members (absolute, ``..``, symlinks) and size / count overruns are reported
    as errors and never written."""
    errors: list[dict[str, str]] = []
    tmp = Path(tempfile.mkdtemp(prefix="brewslm-pack-"))
    try:
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            errors.append(_issue("bad_zip", "", "The upload is not a valid zip file."))
            yield tmp, errors
            return
        with zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            infos = [i for i in infos if not i.filename.startswith("__MACOSX/") and PurePosixPath(i.filename).name != ".DS_Store"]
            if len(infos) > MAX_FILES:
                errors.append(_issue("too_large", "", f"The archive holds {len(infos)} files; the limit is {MAX_FILES}."))
            total = sum(i.file_size for i in infos)
            if total > MAX_TOTAL_BYTES:
                errors.append(_issue("too_large", "", f"The archive expands to {total} bytes; the limit is {MAX_TOTAL_BYTES}."))
            for info in infos:
                if info.file_size > MAX_FILE_BYTES:
                    errors.append(_issue("too_large", info.filename, f"{info.filename} is {info.file_size} bytes; the per-file limit is {MAX_FILE_BYTES}."))
                if _unsafe_name(info.filename):
                    errors.append(_issue("unsafe_path", info.filename, f"Unsafe path in the archive: {info.filename!r}."))
                elif stat.S_ISLNK(info.external_attr >> 16):
                    errors.append(_issue("unsafe_path", info.filename, f"{info.filename!r} is a symlink."))
            if errors:
                yield tmp, errors
                return
            names = [i.filename for i in infos]
            prefix = ""
            if "pack.yaml" not in names:
                tops = {PurePosixPath(n).parts[0] for n in names if len(PurePosixPath(n).parts) > 1}
                if len(tops) == 1 and all(len(PurePosixPath(n).parts) > 1 for n in names):
                    top = next(iter(tops))
                    if f"{top}/pack.yaml" in names:
                        prefix = top + "/"
            try:
                for info in infos:
                    rel = info.filename[len(prefix):] if prefix and info.filename.startswith(prefix) else info.filename
                    if prefix and not info.filename.startswith(prefix):
                        continue
                    dest = tmp / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    written = 0
                    with zf.open(info) as src, open(dest, "wb") as out:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            written += len(chunk)
                            if written > MAX_FILE_BYTES:
                                errors.append(_issue("too_large", info.filename, f"{info.filename} expands beyond {MAX_FILE_BYTES} bytes."))
                                break
                            out.write(chunk)
            except (zipfile.BadZipFile, zlib.error, EOFError, OSError, NotImplementedError, RuntimeError) as exc:
                errors.append(_issue("bad_zip", "", f"The archive is corrupt or unreadable: {str(exc) or type(exc).__name__}."))
            yield tmp, errors
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Validation ─────────────────────────────────────────────────────────────


class _Ctx:
    def __init__(self, root: Path):
        self.root = root
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []
        self.files: dict[str, tuple[str, int]] = {}
        self.total = 0

    def err(self, code: str, path: str, message: str) -> None:
        self.errors.append(_issue(code, path, message))

    def warn(self, code: str, path: str, message: str) -> None:
        self.warnings.append(_issue(code, path, message))

    def read(self, raw: Any, where: str, *, limit: int = MAX_FILE_BYTES) -> tuple[str, bytes] | None:
        """Resolve + read a referenced file. Returns ``(normalized_rel, bytes)`` or None (error recorded)."""
        if not isinstance(raw, str) or not raw.strip():
            self.err("missing_file", where, "A file path is required.")
            return None
        text = raw.strip()
        if _unsafe_name(text):
            self.err("unsafe_path", text, f"Unsafe path {text!r} (absolute, '..' or backslash).")
            return None
        rel = PurePosixPath(text).as_posix()
        rel = str(PurePosixPath(*PurePosixPath(rel).parts)) if PurePosixPath(rel).parts else ""
        if not rel or rel == ".":
            self.err("missing_file", where, "A file path is required.")
            return None
        cur = self.root
        for part in PurePosixPath(rel).parts:
            cur = cur / part
            if cur.is_symlink():
                self.err("unsafe_path", rel, f"{rel!r} passes through a symlink.")
                return None
        try:
            inside = cur.resolve().is_relative_to(self.root.resolve())
        except OSError:
            inside = False
        if not inside:
            self.err("unsafe_path", rel, f"{rel!r} resolves outside the pack.")
            return None
        if not cur.is_file():
            self.err("missing_file", rel, f"{where}: file {rel!r} does not exist in the pack.")
            return None
        size = cur.stat().st_size
        if size > limit or size > MAX_FILE_BYTES:
            self.err("too_large", rel, f"{rel!r} is {size} bytes; the limit is {min(limit, MAX_FILE_BYTES)}.")
            return None
        data = cur.read_bytes()
        if rel not in self.files:
            self.total += size
            if self.total > MAX_TOTAL_BYTES:
                self.err("too_large", rel, f"The pack's referenced files exceed {MAX_TOTAL_BYTES} bytes.")
        self.files[rel] = (_sha(data), size)
        return rel, data


def _str_opt(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _issue_message(err: Any) -> tuple[str, str]:
    if isinstance(err, dict):
        code = str(err.get("code") or "invalid")
        msg = str(err.get("message") or err.get("detail") or "")
        where = err.get("probe_id") or err.get("id") or err.get("path")
        text = f"{code}: {msg}" if msg else code
        if where:
            text += f" ({where})"
        return code, text
    return "invalid", str(err)


def _build_contract(manifest: dict[str, Any], raw: dict[str, Any], pack_id: str):
    """The ``DomainPackContract`` for a manifest (raises ``ValidationError``)."""
    from app.schemas.domain_pack import DomainPackContract

    return DomainPackContract.model_validate({
        "pack_id": pack_id,
        "version": manifest.get("version") or "",
        "display_name": manifest["display_name"],
        "description": manifest.get("description", ""),
        "owner": manifest.get("owner") or "imported",
        "tags": manifest.get("tags") or [],
        "overlay": raw.get("overlay") or {},
        "hooks": raw.get("hooks") or {},
        **({"default_profile_id": manifest["default_profile_id"]} if "default_profile_id" in manifest else {}),
    })


def _validate(root: Path) -> tuple[PackValidation, Parsed]:
    ctx = _Ctx(root)
    parsed = Parsed()
    manifest: dict[str, Any] = {}
    summary = {
        "taxonomy_entries": 0, "probe_packs": 0, "probes": 0, "critical_probes": 0, "rubrics": 0,
        "schemas": 0, "tool_catalog": False, "documents": 0, "gold_dev_rows": 0, "gold_test_rows": 0,
        "project_template": False,
    }

    def finish() -> tuple[PackValidation, Parsed]:
        content_hash = None
        if not ctx.errors:
            basis = {k: v for k, v in manifest.items() if k != "version"}
            content_hash = _sha(
                _jdump({"manifest": basis, "files": sorted((p, h) for p, (h, _s) in ctx.files.items())}).encode()
            )
        result = PackValidation(
            valid=not ctx.errors,
            errors=ctx.errors,
            warnings=ctx.warnings,
            manifest=manifest,
            summary=summary,
            content_hash=content_hash,
            files=[{"path": p, "sha256": h, "size": s} for p, (h, s) in sorted(ctx.files.items())],
        )
        return result, parsed

    pack_yaml = root / "pack.yaml"
    if pack_yaml.is_symlink() or not pack_yaml.is_file():
        ctx.err("missing_pack_yaml", "pack.yaml", "pack.yaml was not found at the pack root.")
        return finish()
    try:
        raw = _load_text(pack_yaml.read_bytes(), "pack.yaml")
    except Exception as exc:  # noqa: BLE001
        ctx.err("bad_yaml", "pack.yaml", f"pack.yaml is not valid YAML: {str(exc).splitlines()[0] if str(exc) else exc}")
        return finish()
    if not isinstance(raw, dict):
        ctx.err("bad_yaml", "pack.yaml", "pack.yaml must be a mapping.")
        return finish()

    for key in raw:
        if key not in KNOWN_KEYS:
            ctx.warn("unknown_key", "pack.yaml", f"Unknown top-level key {key!r} is ignored.")
    schema_val = _str_opt(raw.get("schema") or raw.get("$schema"))
    if schema_val and schema_val not in KNOWN_SCHEMAS:
        ctx.warn("unknown_schema", "pack.yaml", f"Unrecognised schema {schema_val!r}; treating it as {SCHEMA_V2}.")

    pack_id = _str_opt(raw.get("pack_id")) or ""
    if not PACK_ID_RE.match(pack_id):
        ctx.err("bad_pack_id", "pack.yaml", f"pack_id {pack_id!r} must match {PACK_ID_RE.pattern}.")
    elif pack_id == "general-pack-v1":
        ctx.warn("system_pack_id", "pack.yaml", f"{pack_id!r} is a platform pack id; importing over it is refused.")
    version = _str_opt(raw.get("version")) or ""
    if not VERSION_RE.match(version):
        ctx.err("bad_version", "pack.yaml", f"version {version!r} must look like MAJOR.MINOR.PATCH.")

    tags = raw.get("tags")
    tags = [str(t).strip() for t in tags if str(t).strip()] if isinstance(tags, list) else []
    manifest.update({
        "schema": SCHEMA_V2,
        "pack_id": pack_id,
        "version": version,
        "display_name": _str_opt(raw.get("display_name")) or pack_id,
        "description": str(raw.get("description") or ""),
        "owner": _str_opt(raw.get("owner")) or "imported",
        "tags": tags,
    })
    if _str_opt(raw.get("default_profile_id")):
        manifest["default_profile_id"] = _str_opt(raw.get("default_profile_id"))

    # The real contract payload (what import will store): pack_id, owner,
    # display_name, description, overlay + hooks ... must all validate here, so
    # a pack that previews as valid cannot fail at import.
    try:
        contract = _build_contract(manifest, raw, pack_id)
        manifest["overlay"] = contract.overlay.model_dump(mode="json")
        manifest["hooks"] = contract.hooks.model_dump(mode="json", by_alias=True)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            ctx.err("invalid_contract", "pack.yaml", f"{loc or 'pack.yaml'}: {err.get('msg', 'invalid')}")
    except Exception as exc:  # noqa: BLE001
        ctx.err("invalid_contract", "pack.yaml", f"overlay / hooks are invalid: {str(exc).splitlines()[0]}")

    _validate_taxonomy(ctx, raw, manifest, parsed, summary)
    _validate_probes(ctx, raw, manifest, parsed, summary, pack_id)
    _validate_artifacts(ctx, raw, manifest, parsed, summary)
    _validate_documents(ctx, raw, manifest, parsed, summary)
    _validate_gold(ctx, raw, manifest, parsed, summary)
    _validate_template(ctx, raw, manifest, parsed, summary)

    # unreferenced files
    on_disk: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            ctx.err("unsafe_path", p.relative_to(root).as_posix(), f"{p.relative_to(root).as_posix()!r} is a symlink.")
        elif p.is_file():
            on_disk.append(p.relative_to(root).as_posix())
    if len(on_disk) > MAX_FILES:
        ctx.err("too_large", "", f"The pack holds {len(on_disk)} files; the limit is {MAX_FILES}.")
    for rel in on_disk:
        if rel != "pack.yaml" and rel not in ctx.files and not rel.startswith("__MACOSX/") and not rel.endswith(".DS_Store"):
            ctx.warn("unreferenced_file", rel, f"{rel} is not referenced from pack.yaml and was ignored.")
    return finish()


def _validate_taxonomy(ctx: _Ctx, raw: dict, manifest: dict, parsed: Parsed, summary: dict) -> None:
    section = raw.get("taxonomy")
    if section is None:
        return
    if not isinstance(section, dict):
        ctx.err("invalid_taxonomy", "pack.yaml", "taxonomy must be a mapping with a 'path'.")
        return
    got = ctx.read(section.get("path"), "taxonomy.path")
    cw = [str(c).strip() for c in (section.get("crosswalk_from") or []) if str(c).strip()] if isinstance(section.get("crosswalk_from"), list) else []
    entry: dict[str, Any] = {"path": got[0] if got else str(section.get("path") or "")}
    if cw:
        entry["crosswalk_from"] = cw
    if _str_opt(section.get("name")):
        entry["name"] = _str_opt(section.get("name"))
    manifest["taxonomy"] = entry
    if not got:
        return
    from app.services import label_taxonomy_service as lts

    tax, errors = lts.parse_taxonomy_lenient(got[1], crosswalk_from=cw, name=entry.get("name"))
    if tax is not None:
        errors = errors + lts.validate_taxonomy(tax)
    for e in errors:
        ctx.err("invalid_taxonomy", got[0], f"{e.get('code')}: {e.get('message')}")
    if tax is not None and not errors:
        parsed.taxonomy = tax
        summary["taxonomy_entries"] = len(tax["entries"])


def _validate_probes(ctx: _Ctx, raw: dict, manifest: dict, parsed: Parsed, summary: dict, pack_id: str) -> None:
    section = raw.get("probe_packs")
    if section is None:
        return
    if not isinstance(section, list):
        ctx.err("invalid_probe", "pack.yaml", "probe_packs must be a list of file paths.")
        return
    from app.services import probe_pack_service as pps

    paths: list[str] = []
    seen_packs: dict[str, str] = {}
    for item in section:
        got = ctx.read(item, "probe_packs[]")
        if not got:
            continue
        rel, data = got
        paths.append(rel)
        try:
            doc = pps.parse_user_probe_pack_text(data.decode("utf-8-sig", "replace"))
        except Exception as exc:  # noqa: BLE001
            ctx.err("bad_yaml", rel, f"Probe pack is not valid YAML/JSON: {str(exc).splitlines()[0] if str(exc) else exc}")
            continue
        # duplicate probe ids within the pack file (checked here so the code is ours)
        probes_raw = doc.get("probes") if isinstance(doc, dict) else None
        dupes: set[str] = set()
        ids_seen: set[str] = set()
        for p in probes_raw if isinstance(probes_raw, list) else []:
            pid = str(p.get("id")) if isinstance(p, dict) and p.get("id") is not None else None
            if pid is None:
                continue
            if pid in ids_seen and pid not in dupes:
                dupes.add(pid)
                ctx.err("duplicate_id", rel, f"Probe id {pid!r} appears more than once in {rel}.")
            ids_seen.add(pid)
        norm, verrors = pps.validate_user_probe_pack(doc)
        for e in verrors or []:
            code, text = _issue_message(e)
            if code == "duplicate_id":
                if not dupes:
                    ctx.err("duplicate_id", rel, text)
                continue
            ctx.err("invalid_probe", rel, text)
        if verrors or not isinstance(norm, dict):
            continue
        pid = str(norm.get("id"))
        if pid in seen_packs:
            ctx.err("duplicate_id", rel, f"Probe pack id {pid!r} is declared by both {seen_packs[pid]} and {rel}.")
            continue
        seen_packs[pid] = rel
        parsed.probe_packs.append(norm)
        probes = [p for p in (norm.get("probes") or []) if isinstance(p, dict)]
        summary["probe_packs"] += 1
        summary["probes"] += len(probes)
        summary["critical_probes"] += sum(1 for p in probes if p.get("critical"))
    manifest["probe_packs"] = paths


def _validate_artifacts(ctx: _Ctx, raw: dict, manifest: dict, parsed: Parsed, summary: dict) -> None:
    for key, code, bucket in (("rubrics", "invalid_rubric", parsed.rubrics), ("schemas", "invalid_schema", parsed.schemas)):
        section = raw.get(key)
        if section is None:
            continue
        if not isinstance(section, list):
            ctx.err(code, "pack.yaml", f"{key} must be a list of {{id, path}}.")
            continue
        entries: list[dict[str, Any]] = []
        ids: set[str] = set()
        for item in section:
            if not isinstance(item, dict):
                ctx.err(code, "pack.yaml", f"Each {key} entry must be a mapping with 'id' and 'path'.")
                continue
            ident = _str_opt(item.get("id"))
            if not ident:
                ctx.err(code, "pack.yaml", f"A {key} entry has no id.")
                continue
            if ident in ids:
                ctx.err("duplicate_id", "pack.yaml", f"Duplicate {key[:-1]} id {ident!r}.")
                continue
            ids.add(ident)
            got = ctx.read(item.get("path"), f"{key}[{ident}].path", limit=MAX_ARTIFACT_BYTES)
            apply_flag = bool(item.get("apply_as_output_schema")) if key == "schemas" else None
            entry = {"id": ident, "path": got[0] if got else str(item.get("path") or "")}
            if key == "schemas":
                entry["apply_as_output_schema"] = apply_flag
            entries.append(entry)
            if not got:
                continue
            try:
                content = _load_text(got[1], got[0])
            except Exception as exc:  # noqa: BLE001
                ctx.err(code, got[0], f"{got[0]} does not parse: {str(exc).splitlines()[0] if str(exc) else exc}")
                continue
            if key == "rubrics":
                if not isinstance(content, (dict, list)):
                    ctx.err(code, got[0], "A rubric must be a mapping or a list.")
                    continue
                if isinstance(content, list):
                    inner = [str(i["id"]) for i in content if isinstance(i, dict) and "id" in i]
                    for dup in sorted({i for i in inner if inner.count(i) > 1}):
                        ctx.err("duplicate_id", got[0], f"Duplicate item id {dup!r} in {got[0]}.")
                bucket.append({"id": ident, "content": content})
            else:
                if not isinstance(content, dict):
                    ctx.err(code, got[0], "A schema must be a JSON object.")
                    continue
                if apply_flag:
                    from app.services.output_schema_service import validate_output_schema

                    try:
                        validate_output_schema(content)
                    except ValueError as exc:
                        ctx.err(code, got[0], f"Not usable as an output schema: {exc}")
                        continue
                bucket.append({"id": ident, "schema": content, "apply_as_output_schema": apply_flag})
        manifest[key] = entries
        summary[key] = len(bucket)
    section = raw.get("tool_catalog")
    if section is not None:
        path = section.get("path") if isinstance(section, dict) else section
        got = ctx.read(path, "tool_catalog.path", limit=MAX_ARTIFACT_BYTES)
        manifest["tool_catalog"] = {"path": got[0] if got else str(path or "")}
        if got:
            try:
                content = _load_text(got[1], got[0])
            except Exception as exc:  # noqa: BLE001
                ctx.err("invalid_tool_catalog", got[0], f"{got[0]} does not parse: {str(exc).splitlines()[0] if str(exc) else exc}")
                content = None
            else:
                if not isinstance(content, (dict, list)):
                    ctx.err("invalid_tool_catalog", got[0], "A tool catalog must be a mapping or a list.")
                else:
                    from app.services.tool_catalog_service import check_pack_tools

                    tool_errors, tool_warnings = check_pack_tools(content)
                    for e in tool_errors:
                        ctx.err(e["code"], got[0], e["message"])
                    for w in tool_warnings:
                        ctx.warn(w["code"], got[0], w["message"])
                    parsed.tool_catalog = content
                    parsed.has_tool_catalog = True
                    summary["tool_catalog"] = True
    if sum(1 for s in parsed.schemas if s.get("apply_as_output_schema")) > 1:
        ctx.warn("multiple_output_schemas", "pack.yaml", "More than one schema is flagged apply_as_output_schema; only the first is applied.")


def _validate_documents(ctx: _Ctx, raw: dict, manifest: dict, parsed: Parsed, summary: dict) -> None:
    section = raw.get("documents")
    if section is None:
        return
    if not isinstance(section, list):
        ctx.err("missing_file", "pack.yaml", "documents must be a list.")
        return
    from app.services.provenance_service import USAGES
    from app.utils.file_parsers import SUPPORTED_EXTENSIONS

    out: list[dict[str, Any]] = []
    for item in section:
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            ctx.err("missing_file", "pack.yaml", "Each document must be a path or a mapping with 'path'.")
            continue
        got = ctx.read(item.get("path"), "documents[].path")
        path = got[0] if got else str(item.get("path") or "")
        if any(d["path"] == path for d in out):
            ctx.err("duplicate_id", path, f"Document {path!r} is listed more than once.")
            continue
        usage = str(item.get("usage") or "any").strip().lower()
        entry: dict[str, Any] = {"path": path}
        for k in ("title", "licence", "locator"):
            if _str_opt(item.get(k)):
                entry[k] = _str_opt(item.get(k))
        entry["usage"] = usage
        out.append(entry)
        if usage not in USAGES:
            ctx.err("bad_licence_usage", path, f"usage {usage!r} must be one of {', '.join(USAGES)}.")
        if item.get("licence") is not None and not isinstance(item.get("licence"), (str, int, float)):
            ctx.err("bad_licence_usage", path, "licence must be text (an SPDX id or free text).")
        if not entry.get("licence"):
            ctx.warn("missing_licence", path, f"{path} declares no licence; it will be recorded as unknown.")
        ext = PurePosixPath(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            ctx.err("unsupported_document", path, f"{path}: file type {ext or '(none)'} cannot be ingested.")
        if got:
            parsed.documents.append({**entry, "sha256": ctx.files[got[0]][0]})
    manifest["documents"] = out
    summary["documents"] = len(parsed.documents)


def _validate_gold(ctx: _Ctx, raw: dict, manifest: dict, parsed: Parsed, summary: dict) -> None:
    section = raw.get("gold")
    if section is None:
        return
    if not isinstance(section, dict):
        ctx.err("bad_gold_row", "pack.yaml", "gold must be a mapping with 'dev' and/or 'test' paths.")
        return
    out: dict[str, str] = {}
    for split in ("dev", "test"):
        if section.get(split) is None:
            continue
        got = ctx.read(section.get(split), f"gold.{split}")
        out[split] = got[0] if got else str(section.get(split))
        if not got:
            continue
        rows: list[dict[str, Any]] = []
        for n, line in enumerate(got[1].decode("utf-8-sig", "replace").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                ctx.err("bad_gold_row", got[0], f"line {n} is not valid JSON.")
                continue
            if not isinstance(row, dict):
                ctx.err("bad_gold_row", got[0], f"line {n} is not a JSON object.")
                continue
            rows.append(row)
        parsed.gold[split] = rows
        summary[f"gold_{split}_rows"] = len(rows)
    manifest["gold"] = out


def training_default_fields() -> set[str]:
    from app.schemas.training import TrainingConfig

    return set(TrainingConfig.model_fields) - {"base_model"}


def _validate_template(ctx: _Ctx, raw: dict, manifest: dict, parsed: Parsed, summary: dict) -> None:
    section = raw.get("project_template")
    if section is None:
        return
    if not isinstance(section, dict):
        ctx.err("unknown_recipe", "pack.yaml", "project_template must be a mapping.")
        return
    out: dict[str, Any] = {}
    for key in section:
        if key not in TEMPLATE_KEYS:
            ctx.warn("unknown_key", "pack.yaml", f"Unknown project_template key {key!r} is ignored.")
    recipe_id = _str_opt(section.get("recipe_id"))
    if recipe_id:
        from app.services import recipe_service

        if recipe_service.get_recipe(recipe_id) is None:
            ctx.err("unknown_recipe", "pack.yaml", f"project_template.recipe_id {recipe_id!r} is not in the recipe catalog.")
        out["recipe_id"] = recipe_id
    for key in ("base_model", "target_profile", "plan_profile"):
        if _str_opt(section.get(key)):
            out[key] = _str_opt(section.get(key))
    td = section.get("training_defaults")
    if td is not None:
        if not isinstance(td, dict):
            ctx.err("invalid_template", "pack.yaml", "project_template.training_defaults must be a mapping.")
        else:
            fields = training_default_fields()
            bad = sorted(k for k in td if k not in fields)
            if bad:
                ctx.err("invalid_template", "pack.yaml", f"training_defaults has unknown keys: {', '.join(map(str, bad))}.")
            else:
                from app.schemas.training import TrainingConfig

                try:
                    TrainingConfig(base_model="x", **td)
                except Exception as exc:  # noqa: BLE001
                    ctx.err("invalid_template", "pack.yaml", f"training_defaults is invalid: {str(exc).splitlines()[0]}")
                else:
                    out["training_defaults"] = dict(td)
    if section.get("retrieval") is not None:
        if not isinstance(section.get("retrieval"), dict):
            ctx.err("invalid_template", "pack.yaml", "project_template.retrieval must be a mapping.")
        else:
            from app.services.auto_rag_service import normalize_retrieval

            out["retrieval"] = normalize_retrieval(section["retrieval"])
    manifest["project_template"] = out
    parsed.template = out
    summary["project_template"] = bool(out)


def validate_pack(root: Path) -> PackValidation:
    """Validate an extracted pack directory."""
    return _validate(Path(root))[0]


def validate_pack_archive(data: bytes) -> PackValidation:
    """Validate a pack zip (extracts to a temp dir)."""
    if len(data) > MAX_TOTAL_BYTES + 1024 * 1024:
        return _failed([_issue("too_large", "", "The upload is larger than the pack size limit.")])
    with extracted_archive(data) as (root, early):
        if early:
            return _failed(early)
        return validate_pack(root)


def _failed(errors: list[dict[str, str]]) -> PackValidation:
    return PackValidation(
        valid=False, errors=errors, warnings=[], manifest={}, content_hash=None, files=[],
        summary={"taxonomy_entries": 0, "probe_packs": 0, "probes": 0, "critical_probes": 0, "rubrics": 0,
                 "schemas": 0, "tool_catalog": False, "documents": 0, "gold_dev_rows": 0, "gold_test_rows": 0,
                 "project_template": False},
    )


# ── Diff ───────────────────────────────────────────────────────────────────


def _gold_norm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = {k: v for k, v in row.items() if k not in GOLD_SYSTEM_FIELDS}
        for k, default in GOLD_DEFAULTS.items():
            if k in r and r[k] == default:
                r.pop(k)
        out.append(r)
    return out


def _norm_template(template: dict[str, Any]) -> dict[str, Any]:
    """Defaults that a project carries anyway compare equal to 'not declared'."""
    t = {k: v for k, v in (template or {}).items() if v not in (None, "", {}, [])}
    if t.get("target_profile") == DEFAULT_TARGET_PROFILE:
        t.pop("target_profile")
    if t.get("plan_profile") == DEFAULT_PLAN_PROFILE:
        t.pop("plan_profile")
    if t.get("recipe_id") and t.get("base_model"):
        from app.services import recipe_service

        recipe = recipe_service.get_recipe(t["recipe_id"])
        if recipe is not None and t["base_model"] == recipe.suggested_base_model:
            t.pop("base_model")
    return t


def _snapshot(manifest: dict[str, Any], parsed: Parsed) -> dict[str, Any]:
    tax = {str(e["id"]): _jdump(e) for e in (parsed.taxonomy or {}).get("entries", [])}
    probes: dict[str, str] = {}
    for pack in parsed.probe_packs:
        for probe in pack.get("probes") or []:
            probes[f"{pack.get('id')}/{probe.get('id')}"] = _jdump({"applies_to": pack.get("applies_to"), "probe": probe})
    return {
        "taxonomy": tax,
        "probes": probes,
        "rubrics": {r["id"]: _jdump(r["content"]) for r in parsed.rubrics},
        "schemas": {s["id"]: _jdump([s["schema"], bool(s["apply_as_output_schema"])]) for s in parsed.schemas},
        "documents": {d["path"]: d["sha256"] for d in parsed.documents},
        "gold": {
            split: (len(rows), _sha(_jdump(_gold_norm(rows)).encode()))
            for split, rows in ((s, parsed.gold.get(s, [])) for s in ("dev", "test"))
        },
        "template": _norm_template(parsed.template),
        "overlay": _jdump({k: manifest.get(k) for k in ("overlay", "hooks", "default_profile_id")}),
        "tool_catalog": _sha(_jdump(parsed.tool_catalog).encode()) if parsed.has_tool_catalog else None,
    }


def _delta(old: dict[str, str], new: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
    }


def compute_diff(old: tuple[dict, Parsed] | None, new: tuple[dict, Parsed], *, version_from: str | None, version_to: str) -> dict[str, Any]:
    """Diff two ``(manifest, parsed)`` pairs; ``old=None`` compares against an empty pack."""
    a = _snapshot(*old) if old else _snapshot({}, Parsed())
    b = _snapshot(*new)
    gold = {s: {"from": a["gold"][s][0], "to": b["gold"][s][0], "changed": a["gold"][s] != b["gold"][s]} for s in ("dev", "test")}
    ta, tb = a["template"], b["template"]
    return {
        "version": {"from": version_from, "to": version_to},
        "sections": {
            "taxonomy": _delta(a["taxonomy"], b["taxonomy"]),
            "probes": _delta(a["probes"], b["probes"]),
            "rubrics": _delta(a["rubrics"], b["rubrics"]),
            "schemas": _delta(a["schemas"], b["schemas"]),
            "documents": _delta(a["documents"], b["documents"]),
            "gold": gold,
            "project_template": {"changed_fields": sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))},
            "overlay": {"changed": a["overlay"] != b["overlay"]},
            "tool_catalog": {"changed": a["tool_catalog"] != b["tool_catalog"]},
        },
    }


def diff_has_changes(diff: dict[str, Any]) -> bool:
    for name, sec in diff["sections"].items():
        if name == "gold":
            if any(v["changed"] for v in sec.values()):
                return True
        elif name == "project_template":
            if sec["changed_fields"]:
                return True
        elif name in ("overlay", "tool_catalog"):
            if sec["changed"]:
                return True
        elif any(sec.values()):
            return True
    return False


# ── Import ─────────────────────────────────────────────────────────────────


def _existing_content(pack: Any) -> dict[str, Any] | None:
    contract = pack.contract if isinstance(getattr(pack, "contract", None), dict) else {}
    content = contract.get("content")
    return content if isinstance(content, dict) else None


def _stored_pair(pack_id: str, version: str) -> tuple[dict, Parsed] | None:
    root = pack_storage_dir(pack_id, version)
    if not (root / "pack.yaml").is_file():
        return None
    result, parsed = _validate(root)
    if not result["valid"]:
        return None
    return result["manifest"], parsed


async def _plan_import(db: AsyncSession, data: bytes, root: Path, early: list[dict[str, str]]):
    """Validate + compare with what's stored. Returns ``(preview, validation, parsed, existing_pack)``."""
    from app.services.domain_pack_service import get_domain_pack

    if early:
        validation, parsed = _failed(early), Parsed()
    else:
        validation, parsed = _validate(root)
    errors = list(validation["errors"])
    existing = None
    existing_info = None
    proposed = None
    diff = None
    unchanged = False
    manifest = validation["manifest"]
    pack_id = manifest.get("pack_id")
    if pack_id and PACK_ID_RE.match(pack_id):
        existing = await get_domain_pack(db, pack_id)
    if existing is not None:
        content = _existing_content(existing)
        existing_info = {"version": existing.version, "content_hash": (content or {}).get("content_hash")}
        if existing.is_system:
            errors.append(_issue("system_pack", "pack.yaml", f"{pack_id!r} is a platform pack and cannot be imported over."))
    if validation["valid"] and not errors:
        version = manifest["version"]
        if existing is not None:
            if existing_info["content_hash"] and existing_info["content_hash"] == validation["content_hash"]:
                unchanged = True
                proposed = existing.version
            elif _version_tuple(version) > _version_tuple(existing.version):
                proposed = version
            else:
                proposed = _bump_patch(existing.version)
            old = _stored_pair(existing.pack_id, str((content or {}).get("version") or existing.version)) if content else None
            diff = compute_diff(old, (manifest, parsed), version_from=existing.version, version_to=proposed)
        else:
            proposed = version
    preview = {
        "valid": validation["valid"] and not errors,
        "errors": errors,
        "warnings": validation["warnings"],
        "summary": validation["summary"],
        "manifest": manifest,
        "content_hash": validation["content_hash"],
        "existing": existing_info,
        "proposed_version": proposed,
        "unchanged": unchanged,
        "diff": diff,
    }
    return preview, validation, parsed, existing


async def preview_import(db: AsyncSession, data: bytes) -> dict[str, Any]:
    if len(data) > MAX_TOTAL_BYTES + 1024 * 1024:
        preview, *_ = await _plan_import(db, data, Path("."), [_issue("too_large", "", "The upload is larger than the pack size limit.")])
        return preview
    with extracted_archive(data) as (root, early):
        preview, *_ = await _plan_import(db, data, root, early)
    return preview


def _stage_version_dir(root: Path, manifest: dict[str, Any], validation: PackValidation) -> Path:
    """Write the version's files to a hidden staging dir beside the final one.
    The caller renames it into place only after the database commit; an
    existing version dir is never touched."""
    dest = pack_storage_dir(manifest["pack_id"], manifest["version"])
    staging = dest.parent / f".{dest.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        (staging / "pack.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
        for f in validation["files"]:
            target = staging / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / f["path"], target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


async def import_pack(db: AsyncSession, data: bytes) -> dict[str, Any]:
    """Validate and import a pack zip. Raises ``PackContentError`` (422 invalid,
    409 system pack). Same content hash as the current version: ``unchanged``,
    nothing written."""
    from app.schemas.domain_pack import DomainPackContract
    from app.services.domain_pack_service import _hydrate_pack_from_contract, create_domain_pack

    if len(data) > MAX_TOTAL_BYTES + 1024 * 1024:
        raise PackContentError(422, "INVALID_DOMAIN_PACK", "The upload is larger than the pack size limit.",
                               [_issue("too_large", "", "The upload is larger than the pack size limit.")])
    with extracted_archive(data) as (root, early):
        preview, validation, parsed, existing = await _plan_import(db, data, root, early)
        for e in preview["errors"]:
            if e["code"] == "system_pack":
                raise PackContentError(409, "system_pack", e["message"], [e])
        if not preview["valid"]:
            first = preview["errors"][0]["message"] if preview["errors"] else "Invalid pack."
            more = f" (and {len(preview['errors']) - 1} more)" if len(preview["errors"]) > 1 else ""
            raise PackContentError(422, "INVALID_DOMAIN_PACK", f"{first}{more}", preview["errors"])
        if preview["unchanged"]:
            return {"pack": existing, "version": existing.version, "unchanged": True, "diff": preview["diff"],
                    "warnings": preview["warnings"]}

        manifest = dict(validation["manifest"])
        version = preview["proposed_version"]
        # Never overwrite a version dir that already exists on disk.
        while pack_storage_dir(manifest["pack_id"], version).exists():
            version = _bump_patch(version)
        manifest["version"] = version
        now = datetime.now(timezone.utc).isoformat()

        prior_versions: list[dict[str, Any]] = []
        if existing is not None:
            prior = existing.contract.get("content_versions") if isinstance(existing.contract, dict) else None
            prior_versions = [v for v in (prior or []) if isinstance(v, dict) and v.get("version") != version]
        versions = prior_versions + [{"version": version, "content_hash": validation["content_hash"], "imported_at": now}]
        payload: dict[str, Any] = {
            "$schema": SCHEMA_V2,
            "pack_id": manifest["pack_id"],
            "version": version,
            "display_name": manifest["display_name"],
            "description": manifest["description"],
            "owner": manifest["owner"],
            "tags": manifest["tags"],
            "overlay": manifest["overlay"],
            "hooks": manifest["hooks"],
            "content": {
                "version": version,
                "content_hash": validation["content_hash"],
                "imported_at": now,
                "files": validation["files"],
                "summary": validation["summary"],
                "manifest": manifest,
            },
            "content_versions": versions,
        }
        if manifest.get("default_profile_id"):
            payload["default_profile_id"] = manifest["default_profile_id"]
        # Build + validate the contract BEFORE anything is written.
        try:
            contract = DomainPackContract.model_validate(payload)
        except ValidationError as exc:
            issues = [
                _issue("invalid_contract", "pack.yaml", f"{'.'.join(str(x) for x in e.get('loc', ()))}: {e.get('msg', 'invalid')}")
                for e in exc.errors()
            ]
            raise PackContentError(422, "INVALID_DOMAIN_PACK", issues[0]["message"], issues)
        staging = _stage_version_dir(root, manifest, validation)

    dest = pack_storage_dir(manifest["pack_id"], version)
    try:
        if existing is None:
            pack = await create_domain_pack(db, contract)
        else:
            pack = _hydrate_pack_from_contract(existing, contract)
            await db.flush()
        await db.commit()  # row first; the files move into place only once it is durable
        await db.refresh(pack)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        staging.rename(dest)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"pack": pack, "version": version, "unchanged": False, "diff": preview["diff"], "warnings": preview["warnings"]}


# ── Read ───────────────────────────────────────────────────────────────────


async def get_pack_content(db: AsyncSession, pack_id: str, version: str | None = None) -> dict[str, Any]:
    from app.services.domain_pack_service import get_domain_pack

    pack = await get_domain_pack(db, pack_id)
    if pack is None:
        raise PackContentError(404, "pack_not_found", f"Domain pack '{pack_id}' not found")
    content = _existing_content(pack)
    if not content:
        raise PackContentError(404, "pack_has_no_content", f"Domain pack '{pack_id}' has no imported content")
    versions = [v for v in (pack.contract.get("content_versions") or []) if isinstance(v, dict)]
    if version is None or version == content.get("version"):
        return {"pack_id": pack.pack_id, "version": content.get("version"), "versions": versions,
                "manifest": content.get("manifest") or {}, "summary": content.get("summary") or {},
                "files": content.get("files") or []}
    if version not in {v.get("version") for v in versions}:
        raise PackContentError(404, "version_not_found", f"Pack '{pack_id}' has no version {version!r}")
    root = pack_storage_dir(pack.pack_id, version)
    if not (root / "pack.yaml").is_file():
        raise PackContentError(404, "version_files_missing", f"Files for {pack_id}@{version} are missing on disk")
    result = validate_pack(root)
    return {"pack_id": pack.pack_id, "version": version, "versions": versions, "manifest": result["manifest"],
            "summary": result["summary"], "files": result["files"]}


# ── Instantiate ────────────────────────────────────────────────────────────


def _runtime_set(project: Any, key: str, value: Any) -> None:
    cfg = dict(project.runtime_config or {})
    cfg[key] = value
    project.runtime_config = cfg


async def instantiate_pack(
    db: AsyncSession,
    pack_id: str,
    *,
    version: str | None = None,
    project_name: str | None = None,
    actor_user_id: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a project from a pack version, all-or-nothing: on any failure the
    project's data directory is removed and the error re-raised (the caller's
    session rolls the rows back)."""
    from app.models.dataset import DatasetType
    from app.models.project import PipelineStage, Project, ProjectStatus
    from app.services.domain_pack_service import assign_project_domain_pack, get_domain_pack

    pack = await get_domain_pack(db, pack_id)
    if pack is None:
        raise PackContentError(404, "pack_not_found", f"Domain pack '{pack_id}' not found")
    content = _existing_content(pack)
    if not content:
        raise PackContentError(404, "pack_has_no_content", f"Domain pack '{pack_id}' has no imported content")
    version = version or str(content.get("version"))
    if version not in {v.get("version") for v in (pack.contract.get("content_versions") or []) if isinstance(v, dict)}:
        raise PackContentError(404, "version_not_found", f"Pack '{pack_id}' has no version {version!r}")
    root = pack_storage_dir(pack.pack_id, version)
    validation, parsed = _validate(root) if (root / "pack.yaml").is_file() else (None, None)
    if validation is None or not validation["valid"]:
        raise PackContentError(409, "pack_files_invalid", f"Stored files for {pack_id}@{version} are missing or invalid",
                               validation["errors"] if validation else [])
    manifest = validation["manifest"]
    ref = f"{pack.pack_id}@{version}"

    from app.services.project_template_service import _unique_project_name

    name = await _unique_project_name(db, (project_name or "").strip() or manifest.get("display_name") or pack.pack_id)
    project = Project(
        name=name,
        description=str(manifest.get("description") or ""),
        status=ProjectStatus.ACTIVE,
        pipeline_stage=PipelineStage.INGESTION,
    )
    db.add(project)
    await db.flush()
    project_dir = settings.DATA_DIR / "projects" / str(project.id)
    warnings: list[dict[str, str]] = list(validation["warnings"])
    try:
        await assign_project_domain_pack(db, project.id, pack.pack_id)
        project = await db.get(Project, project.id)

        tpl = parsed.template
        if tpl.get("recipe_id"):
            from app.services.recipe_apply_service import apply_recipe_to_project

            await apply_recipe_to_project(db, project.id, tpl["recipe_id"])
        if tpl.get("base_model"):
            project.base_model_name = tpl["base_model"]
        if tpl.get("target_profile"):
            project.target_profile_id = tpl["target_profile"]
        if tpl.get("plan_profile"):
            project.training_preferred_plan_profile = tpl["plan_profile"]
        if tpl.get("training_defaults"):
            _runtime_set(project, "training_config_overrides", dict(tpl["training_defaults"]))
        if tpl.get("retrieval"):
            from app.services.auto_rag_service import RETRIEVAL_SETTINGS_KEY, normalize_retrieval

            _runtime_set(project, RETRIEVAL_SETTINGS_KEY, normalize_retrieval(tpl["retrieval"]))

        if parsed.taxonomy:
            from app.services import label_taxonomy_service as lts
            from app.services.code_findings_prompt import refresh_guided_json

            lts.set_project_taxonomy(project, parsed.taxonomy)
            refresh_guided_json(project)

        if parsed.probe_packs:
            from app.services import probe_pack_service as pps

            pps.set_project_user_probe_packs(
                project, parsed.probe_packs,
                source={"kind": "domain_pack", "pack_id": pack.pack_id, "version": version},
            )

        warnings.extend(await _copy_documents(db, project, root, parsed, pack.pack_id, version, ref))

        from app.services.gold_service import import_qa_pairs

        for split, dtype in (("dev", DatasetType.GOLD_DEV), ("test", DatasetType.GOLD_TEST)):
            if parsed.gold.get(split):
                await import_qa_pairs(db, project.id, [dict(r) for r in parsed.gold[split]], dtype)

        if parsed.schemas or parsed.rubrics or parsed.has_tool_catalog:
            _runtime_set(project, "pack_artifacts", {
                "source": {"pack_id": pack.pack_id, "version": version},
                "schemas": [dict(s) for s in parsed.schemas],
                "rubrics": [dict(r) for r in parsed.rubrics],
                "tool_catalog": parsed.tool_catalog if parsed.has_tool_catalog else None,
            })
            if parsed.has_tool_catalog:
                from app.services import tool_catalog_service as tcs

                pack_tools = tcs.tools_from_pack(parsed.tool_catalog)
                if pack_tools:
                    try:
                        tcs.set_catalog_tools(
                            project, pack_tools,
                            {"kind": "domain_pack", "pack_id": pack.pack_id, "version": version},
                        )
                    except tcs.ToolCatalogError as exc:
                        warnings.append(_issue("tool_catalog_not_applied", "tool_catalog", f"The pack's tool catalog was kept as an artifact but not applied: {exc}"))
            flagged = next((s for s in parsed.schemas if s.get("apply_as_output_schema")), None)
            if flagged is not None:
                from app.services.recipe_apply_service import OutputSchemaNotSupportedError, set_recipe_output_schema

                try:
                    await set_recipe_output_schema(db, project.id, flagged["schema"])
                except (OutputSchemaNotSupportedError, ValueError) as exc:
                    warnings.append(_issue("output_schema_not_applied", flagged["id"], f"Schema {flagged['id']!r} was kept as an artifact but not applied as the output schema: {exc}"))

        _runtime_set(project, "domain_pack_source", {
            "pack_id": pack.pack_id, "version": version, "content_hash": validation["content_hash"],
            "instantiated_at": datetime.now(timezone.utc).isoformat(),
        })
        await db.flush()
        await db.refresh(project)
    except BaseException:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise

    summary = {
        "pack_id": pack.pack_id,
        "version": version,
        "content_hash": validation["content_hash"],
        "recipe_id": tpl.get("recipe_id"),
        "base_model": project.base_model_name or None,
        "taxonomy_entries": validation["summary"]["taxonomy_entries"],
        "probe_packs": validation["summary"]["probe_packs"],
        "probes": validation["summary"]["probes"],
        "critical_probes": validation["summary"]["critical_probes"],
        "documents": validation["summary"]["documents"],
        "gold_dev_rows": validation["summary"]["gold_dev_rows"],
        "gold_test_rows": validation["summary"]["gold_test_rows"],
        "rubrics": validation["summary"]["rubrics"],
        "schemas": validation["summary"]["schemas"],
        "tool_catalog": validation["summary"]["tool_catalog"],
        "warnings": warnings,
    }
    return project, summary


async def _copy_documents(db: AsyncSession, project: Any, root: Path, parsed: Parsed, pack_id: str, version: str, ref: str) -> list[dict[str, str]]:
    from app.services.ingestion_service import ingest_file
    from app.services.provenance_service import KIND_DOMAIN_PACK, attach_document, create_source

    from app.services.ingestion_service import process_document

    warnings: list[dict[str, str]] = []
    needs_index = False
    for doc in parsed.documents:
        data = (root / doc["path"]).read_bytes()
        row = await ingest_file(
            db, project.id, PurePosixPath(doc["path"]).name, data,
            source="domain_pack", license_info=doc.get("licence") or "",
        )
        source = await create_source(
            db, project.id,
            kind=KIND_DOMAIN_PACK,
            title=doc.get("title") or PurePosixPath(doc["path"]).name,
            locator=f"{ref}:{doc['path']}",
            ref=ref,
            licence=doc.get("licence"),
            usage=doc.get("usage") or "any",
            metadata={"pack_id": pack_id, "version": version, "path": doc["path"], "original_locator": doc.get("locator")},
        )
        await attach_document(row, source)
        # The upload flow processes (parses) a document as a separate step; do it
        # here so `rag` documents are searchable. A failure is a warning, not an abort.
        try:
            await process_document(db, project.id, row.id)
        except Exception as exc:  # noqa: BLE001
            warnings.append(_issue("document_not_processed", doc["path"], f"{doc['path']} was imported but not processed: {exc}. Process it from the Data page."))
        if (doc.get("usage") or "any") in ("rag", "any"):
            needs_index = True
    if needs_index:
        warnings.append(_issue(
            "passages_index_not_built", "documents",
            "Documents were imported and processed; the passages index is built separately -- rebuild it before using retrieval.",
        ))
    await db.flush()
    return warnings


# ── Export ─────────────────────────────────────────────────────────────────


def slugify_pack_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", str(text or "").lower()).strip("-._")
    slug = slug[:64].strip("-._")
    if len(slug) < 3:
        slug = (slug + "-pack").strip("-._") if slug else "project-pack"
    return slug


def _file_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-.") or "item"


def _dump_yaml(value: Any) -> bytes:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")


async def export_project_pack(
    db: AsyncSession,
    project_id: int,
    *,
    pack_id: str | None = None,
    version: str | None = None,
    include_documents: str = "pack",
) -> bytes:
    data, _report = await export_project_pack_report(
        db, project_id, pack_id=pack_id, version=version, include_documents=include_documents
    )
    return data


async def export_project_pack_report(
    db: AsyncSession,
    project_id: int,
    *,
    pack_id: str | None = None,
    version: str | None = None,
    include_documents: str = "pack",
) -> tuple[bytes, dict[str, Any]]:
    """Write the project's current artifacts as a pack zip. The zip is checked
    with ``validate_pack_archive`` before it is returned."""
    from app.models.dataset import DatasetType
    from app.models.domain_pack import DomainPack
    from app.models.project import Project
    from app.services import label_taxonomy_service as lts

    if include_documents not in ("pack", "all", "none"):
        raise PackContentError(400, "bad_include_documents", "include_documents must be pack, all or none")
    project = await db.get(Project, project_id)
    if project is None:
        raise PackContentError(404, "project_not_found", f"Project {project_id} not found")
    runtime = dict(project.runtime_config or {})
    source = runtime.get("domain_pack_source") if isinstance(runtime.get("domain_pack_source"), dict) else {}
    if not pack_id:
        pack_id = source.get("pack_id") or slugify_pack_id(project.name)
    if not PACK_ID_RE.match(pack_id):
        raise PackContentError(400, "bad_pack_id", f"pack_id {pack_id!r} must match {PACK_ID_RE.pattern}")
    if not version:
        version = _bump_patch(source["version"]) if source.get("pack_id") == pack_id and source.get("version") else "1.0.0"
    if not VERSION_RE.match(version):
        raise PackContentError(400, "bad_version", f"version {version!r} must look like MAJOR.MINOR.PATCH")

    bound: DomainPack | None = None
    if project.domain_pack_id:
        bound = await db.get(DomainPack, project.domain_pack_id)
    files: dict[str, bytes] = {}
    skipped: list[str] = []
    manifest: dict[str, Any] = {
        "schema": SCHEMA_V2,
        "pack_id": pack_id,
        "version": version,
        "display_name": (bound.display_name if bound and not bound.is_system else project.name) or pack_id,
        "description": (bound.description if bound and not bound.is_system else project.description) or "",
        "owner": (bound.owner if bound and not bound.is_system else "exported") or "exported",
        "tags": list((bound.contract or {}).get("tags") or []) if bound and not bound.is_system else [],
    }
    if bound and not bound.is_system:
        contract = bound.contract or {}
        manifest["overlay"] = contract.get("overlay") or {}
        manifest["hooks"] = contract.get("hooks") or {}
        if bound.default_profile_id:
            manifest["default_profile_id"] = bound.default_profile_id

    tax = lts.project_taxonomy(project)
    if tax:
        entries = []
        for e in tax["entries"]:
            item: dict[str, Any] = {"id": e["id"], "title": e.get("title") or e["id"]}
            for k in ("parent", "aliases", "attrs", "crosswalk"):
                if e.get(k):
                    item[k] = e[k]
            entries.append(item)
        doc: dict[str, Any] = {}
        if tax.get("name"):
            doc["name"] = tax["name"]
        if tax.get("source_version"):
            doc["version"] = tax["source_version"]
        doc["entries"] = entries
        files["taxonomy.yaml"] = _dump_yaml(doc)
        manifest["taxonomy"] = {"path": "taxonomy.yaml", **({"name": tax["name"]} if tax.get("name") else {})}

    from app.services import probe_pack_service as pps

    packs = pps.get_project_user_probe_packs(project) or []
    paths = []
    for pack in packs:
        rel = f"probes/{_file_slug(pack.get('id'))}.yaml"
        files[rel] = _dump_yaml(pack)
        paths.append(rel)
    if paths:
        manifest["probe_packs"] = paths

    artifacts = runtime.get("pack_artifacts") if isinstance(runtime.get("pack_artifacts"), dict) else {}
    if artifacts.get("rubrics"):
        manifest["rubrics"] = []
        for r in artifacts["rubrics"]:
            rel = f"rubrics/{_file_slug(r['id'])}.yaml"
            files[rel] = _dump_yaml(r["content"])
            manifest["rubrics"].append({"id": r["id"], "path": rel})
    if artifacts.get("schemas"):
        manifest["schemas"] = []
        for s in artifacts["schemas"]:
            rel = f"schemas/{_file_slug(s['id'])}.json"
            files[rel] = json.dumps(s["schema"], indent=2, ensure_ascii=False).encode("utf-8")
            manifest["schemas"].append({"id": s["id"], "path": rel, "apply_as_output_schema": bool(s.get("apply_as_output_schema"))})
    from app.services import tool_catalog_service as tcs

    catalog_state = tcs.raw_state(runtime)
    if catalog_state is not None and artifacts.get("tool_catalog") is not None:
        # The catalog still equals what the pack shipped: export the pack's own text form unchanged.
        try:
            shipped, _ = tcs.normalize_tools(tcs.tools_from_pack(artifacts["tool_catalog"]), strict=False)
            if tcs._sha12(shipped) == catalog_state.get("version"):
                catalog_state = None
        except tcs.ToolCatalogError:
            pass
    if catalog_state is not None:
        exported = []
        for t in catalog_state["tools"]:
            item: dict[str, Any] = {"name": t["name"], "description": t.get("description") or "", "inputSchema": t["input_schema"]}
            if t.get("annotations"):
                item["annotations"] = t["annotations"]
            else:  # a hand-entered tool: carry its top-level flags as MCP hints
                derived = {k: t[f] for f, k in (("read_only", "readOnlyHint"), ("destructive", "destructiveHint")) if isinstance(t.get(f), bool)}
                if derived:
                    item["annotations"] = derived
            exported.append(item)
        files["tool_catalog.yaml"] = _dump_yaml({"tools": exported})
        manifest["tool_catalog"] = {"path": "tool_catalog.yaml"}
    elif artifacts.get("tool_catalog") is not None:
        files["tool_catalog.yaml"] = _dump_yaml(artifacts["tool_catalog"])
        manifest["tool_catalog"] = {"path": "tool_catalog.yaml"}

    if include_documents != "none":
        manifest_docs = await _export_documents(db, project_id, include_documents, files, skipped)
        if manifest_docs:
            manifest["documents"] = manifest_docs

    from app.services.gold_service import get_gold_entries

    gold: dict[str, str] = {}
    for split, dtype in (("dev", DatasetType.GOLD_DEV), ("test", DatasetType.GOLD_TEST)):
        rows = _gold_norm(await get_gold_entries(db, project_id, dtype))
        if rows:
            rel = f"gold/{split}.jsonl"
            files[rel] = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
            gold[split] = rel
    if gold:
        manifest["gold"] = gold

    template: dict[str, Any] = {}
    snap = project.selected_recipe if isinstance(project.selected_recipe, dict) else {}
    if snap.get("recipe_id"):
        template["recipe_id"] = snap["recipe_id"]
    template["base_model"] = (project.base_model_name or "").strip()
    template["target_profile"] = project.target_profile_id
    template["plan_profile"] = project.training_preferred_plan_profile
    if isinstance(runtime.get("training_config_overrides"), dict):
        template["training_defaults"] = runtime["training_config_overrides"]
    from app.services.auto_rag_service import RETRIEVAL_SETTINGS_KEY

    if isinstance(runtime.get(RETRIEVAL_SETTINGS_KEY), dict):
        template["retrieval"] = runtime[RETRIEVAL_SETTINGS_KEY]
    template = _norm_template(template)
    if template:
        manifest["project_template"] = template

    files["pack.yaml"] = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(files, key=lambda p: (p != "pack.yaml", p)):
            zf.writestr(rel, files[rel])
    out = buf.getvalue()
    check = validate_pack_archive(out)
    if not check["valid"]:
        raise PackContentError(500, "export_invalid", "The exported pack did not pass validation.", check["errors"])
    return out, {"pack_id": pack_id, "version": version, "skipped_documents": skipped, "warnings": check["warnings"]}


async def _export_documents(db: AsyncSession, project_id: int, mode: str, files: dict[str, bytes], skipped: list[str]) -> list[dict[str, Any]]:
    from app.models.data_source import DataSource
    from app.services.provenance_service import KIND_DOMAIN_PACK, _project_documents, effective_usage
    from app.utils.file_parsers import SUPPORTED_EXTENSIONS

    docs = await _project_documents(db, project_id)
    sources = {s.id: s for s in (await db.execute(select(DataSource).where(DataSource.project_id == project_id))).scalars().all()}
    out: list[dict[str, Any]] = []
    total = 0
    for doc in docs:
        src = sources.get(doc.data_source_id) if doc.data_source_id else None
        if mode == "pack" and not (src is not None and src.source_kind == KIND_DOMAIN_PACK):
            continue
        name = doc.filename
        path = Path(doc.file_path or "")
        if not path.is_file() or PurePosixPath(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(name)
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES or len(files) + 2 > MAX_FILES:
            skipped.append(name)
            continue
        rel = f"documents/{name}"
        if src is not None and src.source_kind == KIND_DOMAIN_PACK:
            # Round-trip: a pack document keeps the path it was imported from.
            stored = (src.metadata_ if isinstance(src.metadata_, dict) else {}).get("path")
            if isinstance(stored, str) and stored and stored != "pack.yaml" and not _unsafe_name(stored):
                rel = stored
        n = 1
        base = PurePosixPath(rel)
        while rel in files:
            rel = (base.parent / f"{base.stem}_{n}{base.suffix}").as_posix()
            n += 1
        files[rel] = path.read_bytes()
        total += size
        entry: dict[str, Any] = {"path": rel}
        if src is not None:
            entry["title"] = src.title or name
            if src.licence:
                entry["licence"] = src.licence
            entry["usage"] = effective_usage(src)
            meta = src.metadata_ if isinstance(src.metadata_, dict) else {}
            original = meta.get("original_locator") if src.source_kind == KIND_DOMAIN_PACK else src.locator
            if original:
                entry["locator"] = original
        else:
            entry["title"] = name
            entry["usage"] = "any"
        out.append(entry)
    return out
