"""Operator-registered external commands: option verifiers and detectors.

Projects never supply a command line. An operator lists commands in the JSON file named by
``settings.EXTERNAL_CHECKS_FILE`` (unset = none)::

    [{"id": "ext_option_verifier_a", "kind": "option_verifier", "protocol": 1,
      "argv": ["/opt/verifiers/a/verify"], "formats": ["format_a", "format_b"],
      "timeout_s": 120, "cpu_s": 120, "mem_mb": 2048, "revision": 3},
     {"id": "ext_detector_a", "kind": "detector", "protocol": 1,
      "argv": ["/opt/verifiers/a/detect", "-"], "languages": ["python"],
      "timeout_s": 30, "cpu_s": 30, "mem_mb": 1024, "revision": 2}]

Each entry becomes an ``ExternalCommandCheck`` in the code-check registry (``default_enabled =
False``), and still runs only when its id is ALSO in ``CODE_CHECK_ALLOWED`` -- two keys: the
operator file declares it, the allow-list enables it. Execution goes through
``sandbox_exec.run_sandboxed`` (temp dir, scrubbed env, rlimits, network isolation per
``CODE_CHECK_NETWORK_ISOLATION``); the payload travels on stdin, no files are shared.

**Detector protocol** (the ``remediation_detector`` contract): stdin = code; stdout
``{"findings": [{"label", "start_line", "end_line"}]}``.

**Option-verifier protocol v1**: stdin
``{"protocol": 1, "payload": <row verifier_payload>, "option": <one option>, "formats": [...]}``;
stdout ``{"formats": {fmt: {"checks": {name: bool|null}, "passes": bool, "messages": [str]}},
"agreement": {"agree": bool, "differences": [str]}, "option_passes": bool}``. Exit 0 = a verdict
(pass or fail). Errors come back as ``{"error": str, "error_kind": ..., "reason": ...}``:

* ``error_kind = "infrastructure"`` -- the sandbox could not launch the command (or it is not an
  option verifier). The scorer leaves the option unverified (a row that needs it is excluded).
* ``error_kind = "option"`` -- a timeout (``reason = verifier_timeout``), a non-zero exit
  (``verifier_crash``), stdout over the size cap (``verifier_output_too_large``) or unparseable stdout
  (``verifier_bad_output``) on THIS option. The option under test caused it, so the scorer fails the
  option (with the reason) rather than excluding the row.

The verifier runs the model's option; an operator verifier MUST isolate whatever it executes (its own
container / VM / throwaway copy) -- ``sandbox_exec`` limits time, memory and network, not the
filesystem the verifier itself can reach.

The file is re-read when its path, mtime or size changes (``ensure_loaded``); removed entries are
unregistered. A malformed entry is skipped and reported (``load_diagnostics``), never fatal.

``revision`` joins the verdict cache key and the ``multi_option`` scoring fingerprint, so bumping it
invalidates cached verdicts and the lift check's cached base-model result.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from app.config import settings
from app.services import sandbox_exec
from app.services.code_checks.base import BaseCheck, make_result

_LOG = logging.getLogger(__name__)

KINDS = ("option_verifier", "detector")
PROTOCOL = 1
# Pseudo-language of an option verifier: it is never selected for real code.
OPTION_VERIFIER_LANGUAGE = "multi_option"
MAX_TIMEOUT_S = 600.0
MAX_MEM_MB = 16384
MAX_OUTPUT_BYTES = 262144
MAX_MESSAGES = 10
MAX_MESSAGE_CHARS = 300
MAX_CHECKS_PER_FORMAT = 40
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"key": None, "ids": set(), "errors": [], "path": None}


class ExternalCheckError(ValueError):
    """A malformed operator-file entry."""


# ── the check ────────────────────────────────────────────────────────────


class ExternalCommandCheck(BaseCheck):
    """One operator-registered command. ``kind`` is ``option_verifier`` or ``detector``."""

    default_enabled = False
    source = "operator_file"

    def __init__(self, entry: dict[str, Any]) -> None:
        self.id = entry["id"]
        self.kind = entry["kind"]
        self.protocol = entry["protocol"]
        self.argv = list(entry["argv"])
        self.binary = self.argv[0]
        self.formats = list(entry.get("formats") or [])
        self.languages = frozenset(entry.get("languages") or [OPTION_VERIFIER_LANGUAGE])
        self.timeout_s = float(entry["timeout_s"])
        self.cpu_s = int(entry["cpu_s"])
        self.mem_mb = int(entry["mem_mb"])
        self.revision = int(entry["revision"])
        self.revision_note = str(entry.get("revision_note") or "")[:200]

    def available(self) -> bool:
        exe = self.binary
        if os.path.isabs(exe):
            return os.path.isfile(exe) and os.access(exe, os.X_OK)
        return super().available()

    def identity(self) -> str:
        """Cache / fingerprint identity: id, protocol and revision (never the argv: an operator
        moving the install directory does not change verdicts; bumping ``revision`` does)."""
        return f"{self.id}:p{self.protocol}:r{self.revision}"

    def _sandbox(self, stdin: str) -> Any:
        return sandbox_exec.run_sandboxed(
            self.argv, files={}, timeout_s=self.timeout_s, cpu_s=self.cpu_s, mem_mb=self.mem_mb,
            stdin=stdin, max_output_bytes=MAX_OUTPUT_BYTES,
        )

    # The detector contract (``remediation_detector._resolve_command_detector`` calls ``run``).
    def run(self, code: str, *, language: str, context: dict[str, Any]) -> dict[str, Any]:
        if self.kind != "detector":
            return make_result(self.id, None, ["an option verifier does not check code"], skipped="language_mismatch")
        try:
            res = self._sandbox(code)
        except sandbox_exec.SandboxError as exc:
            return make_result(self.id, None, [str(exc)], skipped="unavailable")
        if res.timed_out:
            return make_result(self.id, None, [f"timed out after {self.timeout_s:g}s"], skipped="timeout")
        if res.returncode != 0 or getattr(res, "truncated", False):
            return make_result(self.id, None, [_tail(res.stderr) or f"exit {res.returncode}"], skipped="unavailable")
        parsed = _parse_stdout(res.stdout)
        findings = parsed.get("findings") if isinstance(parsed, dict) else None
        if not isinstance(findings, list):
            return make_result(self.id, None, ["the detector printed no findings list"], skipped="unavailable")
        out = make_result(self.id, True)
        out["findings"] = [f for f in findings if isinstance(f, dict) and f.get("label")]
        return out

    def verify_option(self, payload: Any, option: Any, formats: list[str]) -> dict[str, Any]:
        """Run the option verifier on one option. Returns the normalized verdict
        (``normalize_verdict``) or ``{"error": str}``; never raises."""
        if self.kind != "option_verifier":
            return _infra_error(f"'{self.id}' is not an option verifier")
        stdin = json.dumps(
            {"protocol": self.protocol, "payload": payload, "option": option, "formats": list(formats)},
            ensure_ascii=False, default=str,
        )
        try:
            res = self._sandbox(stdin)
        except sandbox_exec.SandboxError as exc:
            return _infra_error(f"could not run: {exc}")
        except Exception as exc:  # noqa: BLE001
            return _infra_error(f"could not run: {type(exc).__name__}")
        if res.timed_out:
            return _option_error("verifier_timeout", f"timed out after {self.timeout_s:g}s")
        if getattr(res, "truncated", False):
            return _option_error("verifier_output_too_large", "output was truncated (over the size cap)")
        if res.returncode != 0:
            tail = _tail(res.stderr)
            return _option_error("verifier_crash", f"exit {res.returncode}" + (f": {tail}" if tail else ""))
        parsed = _parse_stdout(res.stdout)
        if not isinstance(parsed, dict):
            return _option_error("verifier_bad_output", "stdout was not a JSON object")
        return normalize_verdict(parsed, formats)


def _infra_error(message: str) -> dict[str, Any]:
    """The verifier could not be run at all: the option stays unverified (never failed)."""
    return {"error": message[:MAX_MESSAGE_CHARS], "error_kind": "infrastructure"}


def _option_error(reason: str, message: str) -> dict[str, Any]:
    """The verifier ran on this option and broke (timeout / crash / bad or oversized output): the
    option fails with ``reason``."""
    return {"error": message[:MAX_MESSAGE_CHARS], "error_kind": "option", "reason": reason}


def _tail(text: str | None) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return " | ".join(lines[-2:])[:MAX_MESSAGE_CHARS]


def _parse_stdout(text: str | None) -> Any:
    try:
        return json.loads(str(text or "").strip() or "null")
    except (ValueError, TypeError):
        return None


def _clip(messages: Any) -> list[str]:
    out: list[str] = []
    for m in messages if isinstance(messages, list) else []:
        text = " ".join(str(m).split())
        if text:
            out.append(text[:MAX_MESSAGE_CHARS])
        if len(out) >= MAX_MESSAGES:
            break
    return out


def normalize_verdict(raw: dict[str, Any], formats: list[str]) -> dict[str, Any]:
    """The verifier's stdout -> ``{"formats": {fmt: {checks, passes, messages}}, "agreement":
    {agree, differences} | None, "option_passes": bool | None}``. Check values other than true /
    false become None (not applicable). A requested format the verifier did not report is
    ``passes: None`` with a message; ``passes`` defaults to "every applicable check passed"."""
    reported = raw.get("formats") if isinstance(raw.get("formats"), dict) else {}
    out_formats: dict[str, Any] = {}
    for fmt in formats:
        entry = reported.get(fmt)
        if not isinstance(entry, dict):
            out_formats[fmt] = {"checks": {}, "passes": None, "messages": ["the verifier did not report this format"]}
            continue
        checks: dict[str, bool | None] = {}
        raw_checks = entry.get("checks") if isinstance(entry.get("checks"), dict) else {}
        for name, value in list(raw_checks.items())[:MAX_CHECKS_PER_FORMAT]:
            checks[str(name)[:80]] = value if isinstance(value, bool) else None
        passes = entry.get("passes")
        if not isinstance(passes, bool):
            decided = [v for v in checks.values() if v is not None]
            passes = all(decided) if decided else None
        out_formats[fmt] = {"checks": checks, "passes": passes, "messages": _clip(entry.get("messages"))}
    agreement = raw.get("agreement")
    agree_block: dict[str, Any] | None = None
    if isinstance(agreement, dict) and isinstance(agreement.get("agree"), bool):
        agree_block = {"agree": agreement["agree"], "differences": _clip(agreement.get("differences"))}
    option_passes = raw.get("option_passes")
    return {
        "formats": out_formats,
        "agreement": agree_block,
        "option_passes": option_passes if isinstance(option_passes, bool) else None,
    }


# ── the operator file ────────────────────────────────────────────────────


def _num(entry: dict[str, Any], key: str, default: float, cap: float) -> float:
    value = entry.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ExternalCheckError(f"{key} must be a positive number")
    return float(min(value, cap))


def parse_entry(raw: Any) -> dict[str, Any]:
    """Validate one operator-file entry; ``ExternalCheckError`` when malformed."""
    if not isinstance(raw, dict):
        raise ExternalCheckError("an entry must be a JSON object")
    cid = raw.get("id")
    if not isinstance(cid, str) or not _ID_RE.match(cid):
        raise ExternalCheckError("id must be 1-80 letters, digits, '_', '.' or '-'")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ExternalCheckError(f"{cid}: kind must be one of {', '.join(KINDS)}")
    if raw.get("protocol", PROTOCOL) != PROTOCOL:
        raise ExternalCheckError(f"{cid}: unsupported protocol {raw.get('protocol')!r} (this server speaks {PROTOCOL})")
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a and "\x00" not in a for a in argv):
        raise ExternalCheckError(f"{cid}: argv must be a non-empty list of strings")
    revision = raw.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ExternalCheckError(f"{cid}: revision must be a positive integer")
    entry: dict[str, Any] = {
        "id": cid, "kind": kind, "protocol": PROTOCOL, "argv": list(argv), "revision": revision,
        "timeout_s": _num(raw, "timeout_s", 60, MAX_TIMEOUT_S),
        "cpu_s": int(_num(raw, "cpu_s", 60, MAX_TIMEOUT_S)),
        "mem_mb": int(_num(raw, "mem_mb", 1024, MAX_MEM_MB)),
        "revision_note": raw.get("revision_note") if isinstance(raw.get("revision_note"), str) else "",
    }
    if kind == "option_verifier":
        formats = raw.get("formats")
        if formats is not None and (not isinstance(formats, list) or not all(isinstance(f, str) and f for f in formats)):
            raise ExternalCheckError(f"{cid}: formats must be a list of strings")
        entry["formats"] = list(dict.fromkeys(formats or []))
    else:
        langs = raw.get("languages")
        if not isinstance(langs, list) or not langs or not all(isinstance(x, str) and x for x in langs):
            raise ExternalCheckError(f"{cid}: a detector needs a non-empty languages list")
        entry["languages"] = list(langs)
    return entry


def _file_key(path: Path | None) -> tuple[Any, ...] | None:
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return (str(path), None, None)
    return (str(path), st.st_mtime_ns, st.st_size)


def configured_path() -> Path | None:
    raw = str(getattr(settings, "EXTERNAL_CHECKS_FILE", "") or "").strip()
    return Path(raw).expanduser() if raw else None


def ensure_loaded(*, force: bool = False) -> dict[str, Any]:
    """(Re)register the operator file's entries when it changed; returns ``load_diagnostics()``."""
    from app.services import code_check_service as ccs

    path = configured_path()
    key = _file_key(path)
    with _LOCK:
        if not force and key == _STATE["key"]:
            return load_diagnostics()
        entries: list[dict[str, Any]] = []
        errors: list[str] = []
        if path is not None:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                errors.append("the external checks file does not exist")
                data = []
            except (OSError, ValueError) as exc:
                errors.append(f"the external checks file could not be read: {type(exc).__name__}")
                data = []
            if isinstance(data, dict) and isinstance(data.get("checks"), list):
                data = data["checks"]
            if not isinstance(data, list):
                errors.append("the external checks file must hold a JSON list of entries")
                data = []
            seen: set[str] = set()
            for i, raw in enumerate(data):
                try:
                    entry = parse_entry(raw)
                except ExternalCheckError as exc:
                    errors.append(f"entry {i}: {exc}")
                    continue
                existing = ccs._REGISTRY.get(entry["id"])
                if entry["id"] in seen or (existing is not None and getattr(existing, "source", None) != "operator_file"):
                    errors.append(f"entry {i}: id '{entry['id']}' is already registered")
                    continue
                seen.add(entry["id"])
                entries.append(entry)
        for old in list(_STATE["ids"]):
            ccs.unregister_code_check(old)
        for entry in entries:
            ccs.register_code_check(ExternalCommandCheck(entry))
        _STATE.update({"key": key, "ids": {e["id"] for e in entries}, "errors": errors,
                       "path": str(path) if path else None})
        if errors:
            _LOG.warning("external checks file: %s", "; ".join(errors))
        return load_diagnostics()


def load_diagnostics() -> dict[str, Any]:
    return {"configured": _STATE["path"] is not None, "registered": sorted(_STATE["ids"]), "errors": list(_STATE["errors"])}


def list_option_verifiers() -> list[dict[str, Any]]:
    """Every registered option verifier (operator file or plugin) with its gate state. Read-only:
    what a project may pick from."""
    from app.services import code_check_service as ccs

    ensure_loaded()
    out: list[dict[str, Any]] = []
    for cid in sorted(ccs._REGISTRY):
        check = ccs._REGISTRY[cid]
        if getattr(check, "kind", None) != "option_verifier":
            continue
        out.append({
            "id": cid,
            "revision": int(getattr(check, "revision", 1) or 1),
            "formats": list(getattr(check, "formats", []) or []),
            "allowed": ccs.is_allowed(check),
            "available": bool(check.available()),
        })
    return out


class VerifierUnavailable(Exception):
    """The configured option verifier cannot run (unknown, not allow-listed, not installed)."""


def resolve_option_verifier(check_id: Any) -> ExternalCommandCheck:
    """The registered, allow-listed, installed option verifier with this id, else
    ``VerifierUnavailable`` (with a message safe to show)."""
    from app.services import code_check_service as ccs

    cid = str(check_id or "").strip()
    if not cid:
        raise VerifierUnavailable("no option verifier is configured")
    check = ccs.get_code_check(cid)
    if check is None or getattr(check, "kind", None) != "option_verifier" or not callable(getattr(check, "verify_option", None)):
        raise VerifierUnavailable(f"'{cid}' is not a registered option verifier")
    if not ccs.is_allowed(check):
        raise VerifierUnavailable(f"option verifier '{cid}' is not allow-listed (CODE_CHECK_ALLOWED)")
    if not check.available():
        raise VerifierUnavailable(f"option verifier '{cid}' is not installed")
    return check


def verifier_identity(check_id: Any) -> str:
    """Fingerprint part for a configured verifier (never raises): ``id:p<protocol>:r<revision>`` of
    a usable one, else ``unavailable:<id>`` / ``none``."""
    cid = str(check_id or "").strip()
    if not cid:
        return "none"
    try:
        check = resolve_option_verifier(cid)
    except VerifierUnavailable:
        return f"unavailable:{cid}"
    ident = getattr(check, "identity", None)
    return ident() if callable(ident) else f"{cid}:r{int(getattr(check, 'revision', 1) or 1)}"


# ── verdict cache ────────────────────────────────────────────────────────

CACHE_MAX_ENTRIES = 20000
_CACHE_KEY_FILE = ".multi_option_verdict_cache.key"
_PROCESS_KEY = os.urandom(32)


def verdict_cache_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "multi_option_verifier_cache.json"


def verdict_key(identity: str, payload: Any, option: Any, formats: list[str]) -> str:
    raw = json.dumps([identity, payload, option, list(formats)], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _install_key() -> bytes:
    """A per-install random key in ``DATA_DIR`` (outside ``DATA_DIR/projects``, mode 0600), created on
    first use; a per-process random key when it cannot be read or written (the cache then lives for
    this process only)."""
    path = Path(settings.DATA_DIR) / _CACHE_KEY_FILE
    try:
        if path.exists():
            data = path.read_bytes()
            if len(data) >= 32:
                return data[:64]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = os.urandom(32)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return data
    except FileExistsError:
        try:
            return path.read_bytes()[:64] or _PROCESS_KEY
        except OSError:
            return _PROCESS_KEY
    except OSError:
        return _PROCESS_KEY


def _cache_mac_key() -> bytes:
    """HMAC key for verdict-cache entries: the server's secret setting (``SECRETS_ENCRYPTION_KEY``, else
    ``JWT_SECRET``) mixed with the per-install key, so neither alone forges an entry."""
    seed = str(getattr(settings, "SECRETS_ENCRYPTION_KEY", "") or getattr(settings, "JWT_SECRET", "") or "")
    return hmac.new(_install_key(), b"multi_option_verdict_cache\x00" + seed.encode("utf-8"), hashlib.sha256).digest()


def _entry_mac(mac_key: bytes, key: str, verdict: Any) -> str:
    body = json.dumps([key, verdict], ensure_ascii=False, sort_keys=True, default=str)
    return hmac.new(mac_key, body.encode("utf-8", "replace"), hashlib.sha256).hexdigest()


class VerdictCache:
    """File-backed ``{sha256: {"v": verdict, "mac": hmac}}`` (errors are never cached). Each entry is
    HMAC-signed with a server-side key (``_cache_mac_key``); an entry whose MAC does not verify (an
    edited file, an older unsigned format) is dropped on load. Best-effort: read / write failures
    degrade to an in-memory cache."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self.hits = 0
        self.rejected = 0
        self._mac_key = _cache_mac_key()
        try:
            if path is not None and path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    for k, entry in loaded.items():
                        verdict = entry.get("v") if isinstance(entry, dict) else None
                        mac = entry.get("mac") if isinstance(entry, dict) else None
                        if (
                            isinstance(verdict, dict) and isinstance(mac, str)
                            and hmac.compare_digest(mac, _entry_mac(self._mac_key, str(k), verdict))
                        ):
                            self._data[str(k)] = verdict
                        else:
                            self.rejected += 1
        except Exception:  # noqa: BLE001
            self._data = {}
        if self.rejected:
            _LOG.warning("multi-option verdict cache: dropped %d entr(ies) that failed verification", self.rejected)

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._data.get(key)
        if isinstance(value, dict):
            self.hits += 1
            return json.loads(json.dumps(value))
        return None

    def set(self, key: str, verdict: dict[str, Any]) -> None:
        if verdict.get("error"):
            return
        self._data[key] = verdict
        self._dirty = True
        while len(self._data) > CACHE_MAX_ENTRIES:
            self._data.pop(next(iter(self._data)), None)

    def flush(self) -> None:
        if not self._dirty or self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            signed = {k: {"v": v, "mac": _entry_mac(self._mac_key, k, v)} for k, v in self._data.items()}
            tmp.write_text(json.dumps(signed), encoding="utf-8")
            tmp.replace(self._path)
            self._dirty = False
        except Exception:  # noqa: BLE001
            pass


def make_verify_fn(check: ExternalCommandCheck, cache: VerdictCache | None):
    """``verify(payload, option, formats) -> verdict | {"error"}`` with the cache in front."""
    identity = check.identity()

    def verify(payload: Any, option: Any, formats: list[str]) -> dict[str, Any]:
        key = verdict_key(identity, payload, option, formats)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        verdict = check.verify_option(payload, option, formats)
        if cache is not None:
            cache.set(key, verdict)
        return verdict

    return verify
