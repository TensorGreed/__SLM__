"""Read a git repository as a set of documents (``git`` CLI, no library).

Shallow-fetches one ref into a temp directory, pins the exact commit, finds
the repo's licence, and classifies every file: prose (``doc``), source / config
(``code``) or rejected with a stable reason code (``excluded_by_glob``,
``too_large``, ``binary``, ``unsupported_type``, ``empty_file`` (a supported
file with no text, e.g. an empty ``__init__.py``); files past ``max_files`` are
only counted as ``limit_reached``).

Safety: a URL starting with ``-`` is refused (it would be read as a ``git``
option); only ``https://``, ``http://``, ``ssh://`` and ``git@host:path`` are
accepted — ``file://``, local paths and ssh / scp targets on this machine
(``localhost``, ``127.0.0.0/8``, ``::1``) only when
``settings.DOCUMENT_SOURCES_ALLOW_LOCAL_GIT`` is on. A URL with credentials in
it (``https://user:token@host/…``) is refused — they would be stored and shown
with the Job — and so is a host starting with ``-`` (read as an ssh option). ``GIT_TERMINAL_PROMPT=0``
(a private repo fails instead of hanging on a password prompt),
``GIT_ALLOW_PROTOCOL`` pins the transports, submodules are never fetched and
symlinks are never followed.

The ``git`` runner is injectable for tests.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from app.services.code_chunker import detect_language

DEFAULT_EXCLUDE_GLOBS = [
    "**/.git/**", "**/node_modules/**", "**/vendor/**", "**/dist/**", "**/build/**", "*.min.js", "*.lock",
    "package-lock.json",
]
DOC_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".rst", ".html", ".htm"})
LICENSE_FILENAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENCE.md", "LICENCE.txt", "COPYING", "COPYING.txt")
GIT_TIMEOUT_SECONDS = 300
_BINARY_PROBE_BYTES = 8192
_REF_RE = re.compile(r"^[A-Za-z0-9_./@^~{}+-]+$")
_SCP_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:[^\s]+$")

GitRunner = Callable[[list[str], Path | None, dict[str, str], int], "subprocess.CompletedProcess[str]"]


class GitSourceError(ValueError):
    """The repository can't be read (message is user-facing)."""


def _is_loopback_host(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in {"localhost", "::1", "0:0:0:0:0:0:0:1"} or host.endswith(".localhost"):
        return True
    return bool(re.fullmatch(r"127(?:\.\d{1,3}){3}", host))


def _remote_host(token: str) -> str | None:
    """Host of an ssh / scp-style target, else ``None``."""
    if token.lower().startswith("ssh://"):
        try:
            return urlsplit(token).hostname or ""
        except ValueError:
            return ""
    match = _SCP_URL_RE.match(token)
    if match:
        return token.split("@", 1)[1].split(":", 1)[0]
    return None


def validate_repo_url(url: str, *, allow_local: bool | None = None) -> str:
    """The URL to hand to ``git``, or :class:`GitSourceError`."""
    if allow_local is None:
        from app.config import settings

        allow_local = bool(settings.DOCUMENT_SOURCES_ALLOW_LOCAL_GIT)
    token = (url or "").strip()
    if not token:
        raise GitSourceError("A repository URL is required.")
    if token.startswith("-"):
        raise GitSourceError("A repository URL can't start with '-'.")
    if re.search(r"\s", token) or "\x00" in token:
        raise GitSourceError("A repository URL can't contain whitespace.")
    lowered = token.lower()
    if lowered.startswith(("https://", "http://", "ssh://")) or _SCP_URL_RE.match(token):
        if lowered.startswith(("https://", "http://", "ssh://")):
            try:
                parts = urlsplit(token)
                parts.port  # noqa: B018 — raises ValueError on a bad port
            except ValueError as exc:
                raise GitSourceError(f"Invalid repository URL: {exc}") from exc
            userinfo = parts.netloc.rpartition("@")[0] if "@" in parts.netloc else ""
            if userinfo and (not lowered.startswith("ssh://") or ":" in userinfo):
                raise GitSourceError(
                    "Repository URLs can't contain credentials (user:token@host); use a public URL or an ssh key."
                )
        host = _remote_host(token)
        if host is not None:
            if not host or host.startswith("-"):
                raise GitSourceError("A repository host can't be empty or start with '-'.")
            if _is_loopback_host(host) and not allow_local:
                raise GitSourceError(
                    "Repositories on this machine are disabled on this server "
                    "(set DOCUMENT_SOURCES_ALLOW_LOCAL_GIT=true to allow them); use a remote repository."
                )
        return token
    if lowered.startswith("file://") or token.startswith(("/", "./", "../", "~")):
        if not allow_local:
            raise GitSourceError(
                "Local repository paths are disabled on this server "
                "(set DOCUMENT_SOURCES_ALLOW_LOCAL_GIT=true to allow them); use an https:// or ssh URL."
            )
        return token
    raise GitSourceError("Repository URL must start with https://, http://, ssh:// or look like git@host:path.")


@dataclass
class GitRepoConfig:
    url: str
    ref: str = "HEAD"
    include_globs: list[str] = field(default_factory=list)
    exclude_globs: list[str] = field(default_factory=list)
    default_excludes: bool = True
    max_file_bytes: int = 200_000
    max_files: int = 2000

    @property
    def effective_excludes(self) -> list[str]:
        """``exclude_globs`` added to the built-in defaults (unless ``default_excludes`` is off)."""
        return [*(DEFAULT_EXCLUDE_GLOBS if self.default_excludes else []), *self.exclude_globs]

    def __post_init__(self) -> None:
        self.ref = (self.ref or "HEAD").strip() or "HEAD"
        if self.ref.startswith("-") or not _REF_RE.match(self.ref):
            raise GitSourceError(f"Invalid ref {self.ref!r}: use a branch, tag or commit id.")
        self.max_file_bytes = max(1, int(self.max_file_bytes))
        self.max_files = max(1, min(20000, int(self.max_files)))


def _glob_regex(pattern: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif ch == "*":
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "[" and "]" in pattern[i + 1 :]:
            close = pattern.index("]", i + 1)
            body = pattern[i + 1 : close]
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append(f"[{body}]")
            i = close + 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(path: str, pattern: str) -> bool:
    """``pattern`` against a posix relative ``path``. ``*`` stays inside one
    segment, ``**`` crosses segments, ``**/`` also matches zero directories.
    A pattern with no ``/`` matches the file name at any depth
    (``*.min.js``); one with a ``/`` is anchored at the repo root."""
    pattern = pattern.strip().lstrip("/")
    if not pattern:
        return False
    target = path if "/" in pattern else path.rsplit("/", 1)[-1]
    return bool(_glob_regex(pattern).match(target))


def detect_licence(text: str) -> str:
    """SPDX id for a licence text by its characteristic wording, else
    ``NOASSERTION``."""
    t = " ".join(text.split()).lower()
    if "gnu affero general public license" in t:
        return "AGPL-3.0"
    if "gnu lesser general public license" in t:
        return "LGPL-2.1" if "version 2.1" in t else "LGPL-3.0"
    if "gnu general public license" in t:
        return "GPL-3.0" if "version 3" in t else "GPL-2.0" if "version 2" in t else "NOASSERTION"
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "mozilla public license" in t and "2.0" in t:
        return "MPL-2.0"
    if "eclipse public license" in t and "2.0" in t:
        return "EPL-2.0"
    if "creative commons" in t and ("cc0" in t or "public domain dedication" in t):
        return "CC0-1.0"
    if "creative commons" in t and "attribution 4.0" in t:
        return "CC-BY-4.0"
    if "free and unencumbered software released into the public domain" in t:
        return "Unlicense"
    if "permission is hereby granted, free of charge" in t:
        return "MIT"
    if "permission to use, copy, modify, and/or distribute this software for any purpose with or without fee" in t:
        return "ISC"
    if "redistribution and use in source and binary forms" in t:
        return "BSD-3-Clause" if ("neither the name" in t or "endorse or promote" in t) else "BSD-2-Clause"
    return "NOASSERTION"


@dataclass
class RepoFile:
    path: str
    abs_path: Path
    size: int
    content_kind: str  # "doc" | "code"
    language: str | None

    def read_bytes(self) -> bytes:
        return self.abs_path.read_bytes()


@dataclass
class RepoSnapshot:
    url: str
    ref: str
    commit: str
    licence: str
    licence_file: str | None
    files: list[RepoFile] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    limit_reached: int = 0
    _tmp: tempfile.TemporaryDirectory | None = None

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rejected:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
        return counts

    def cleanup(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __enter__(self) -> "RepoSnapshot":
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()


class GitCancelled(Exception):
    """The import was cancelled while git was running."""


def _make_runner(cancel: threading.Event | None) -> GitRunner:
    """The real ``git`` runner. With ``cancel`` it polls the event and kills the
    process promptly when it is set (a clone can run for minutes)."""

    def runner(args: list[str], cwd: Path | None, env: dict[str, str], timeout: int) -> "subprocess.CompletedProcess[str]":
        proc = subprocess.Popen(  # noqa: S603 — argv list, no shell; url/ref validated above
            args, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        waited = 0.0
        while True:
            try:
                out, err = proc.communicate(timeout=0.2)
                return subprocess.CompletedProcess(args, proc.returncode, out, err)
            except subprocess.TimeoutExpired:
                waited += 0.2
                if cancel is not None and cancel.is_set():
                    proc.kill()
                    proc.communicate()
                    raise GitCancelled("cancelled") from None
                if waited >= timeout:
                    proc.kill()
                    proc.communicate()
                    raise subprocess.TimeoutExpired(args, timeout) from None

    return runner


def _default_runner(args: list[str], cwd: Path | None, env: dict[str, str], timeout: int) -> "subprocess.CompletedProcess[str]":
    return _make_runner(None)(args, cwd, env, timeout)


def _git_env(allow_local: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ALLOW_PROTOCOL"] = "https:http:ssh" + (":file" if allow_local else "")
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    return env


def _run(runner: GitRunner, args: list[str], cwd: Path, env: dict[str, str], *, what: str) -> str:
    try:
        proc = runner(["git", *args], cwd, env, GIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise GitSourceError(f"git {what} timed out after {GIT_TIMEOUT_SECONDS}s.") from exc
    except FileNotFoundError as exc:
        raise GitSourceError("The git command isn't installed on this server.") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise GitSourceError(f"git {what} failed: {detail[-1][:300] if detail else 'exit ' + str(proc.returncode)}")
    return proc.stdout or ""


def clone_repo(
    config: GitRepoConfig,
    *,
    runner: GitRunner | None = None,
    allow_local: bool | None = None,
    cancel: threading.Event | None = None,
) -> RepoSnapshot:
    """Shallow-clone ``config.ref`` into a temp dir and classify its files.
    The caller owns the snapshot: use it as a context manager (or call
    ``cleanup()``) when done reading the files. ``cancel`` (a ``threading.Event``)
    stops the clone promptly with :class:`GitCancelled` when set."""
    from app.config import settings

    if allow_local is None:
        allow_local = bool(settings.DOCUMENT_SOURCES_ALLOW_LOCAL_GIT)
    url = validate_repo_url(config.url, allow_local=allow_local)
    base_runner = runner or _make_runner(cancel)

    def runner(args: list[str], cwd: Path | None, env: dict[str, str], timeout: int) -> "subprocess.CompletedProcess[str]":
        _check_cancel(cancel)
        return base_runner(args, cwd, env, timeout)

    env = _git_env(allow_local)
    tmp = tempfile.TemporaryDirectory(prefix="brewslm-git-")
    root = Path(tmp.name)
    try:
        _run(runner, ["init", "-q"], root, env, what="init")
        _run(runner, ["remote", "add", "origin", url], root, env, what="remote add")
        try:
            _run(runner, ["fetch", "-q", "--depth", "1", "--no-tags", "origin", config.ref], root, env, what="fetch")
            _run(runner, ["checkout", "-q", "--detach", "FETCH_HEAD"], root, env, what="checkout")
        except GitSourceError as exc:
            # Only a ref the server can't serve shallowly (an arbitrary commit, a short
            # name) is worth the wider fetch; auth / network / timeout errors are final.
            if config.ref == "HEAD" or not _ref_not_found(str(exc)):
                raise
            is_commit = bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", config.ref))
            _check_cancel(cancel)
            _run(
                runner,
                ["fetch", "-q", "--tags", *(["--filter=blob:none"] if is_commit else ["--depth", "1"]),
                 "origin", "+refs/heads/*:refs/remotes/origin/*"],
                root, env, what="fetch",
            )
            target = config.ref if is_commit else None
            if target is None:
                for candidate in (f"refs/tags/{config.ref}", f"refs/remotes/origin/{config.ref}", config.ref):
                    probe = runner(["git", "rev-parse", "--verify", "-q", f"{candidate}^{{commit}}"], root, env, GIT_TIMEOUT_SECONDS)
                    if probe.returncode == 0:
                        target = candidate
                        break
            if target is None:
                raise GitSourceError(f"Couldn't find ref {config.ref!r} in the repository.")
            _run(runner, ["checkout", "-q", "--detach", target], root, env, what="checkout")
        commit = _run(runner, ["rev-parse", "HEAD"], root, env, what="rev-parse").strip()
        snapshot = RepoSnapshot(url=config.url, ref=config.ref, commit=commit, licence="NOASSERTION", licence_file=None, _tmp=tmp)
        _classify_files(snapshot, root, config, cancel)
        return snapshot
    except BaseException:
        tmp.cleanup()
        raise


_NOT_FOUND_MARKERS = (
    "couldn't find remote ref", "could not find remote ref", "not our ref", "unadvertised object",
    "no such remote ref", "invalid refspec",
)


def _ref_not_found(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _NOT_FOUND_MARKERS)


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise GitCancelled("cancelled")


def _find_licence(root: Path) -> tuple[str, str | None]:
    names = {p.name.lower(): p for p in root.iterdir() if p.is_file() and not p.is_symlink()}
    for wanted in LICENSE_FILENAMES:
        path = names.get(wanted.lower())
        if path is not None:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:100_000]
            except OSError:
                continue
            return detect_licence(text), path.name
    return "NOASSERTION", None


def _classify_files(
    snapshot: RepoSnapshot, root: Path, config: GitRepoConfig, cancel: threading.Event | None = None
) -> None:
    snapshot.licence, snapshot.licence_file = _find_licence(root)
    candidates: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        for name in sorted(filenames):
            full = Path(dirpath) / name
            candidates.append((full.relative_to(root).as_posix(), full))
    candidates.sort(key=lambda item: item[0])

    def reject(path: str, reason: str, detail: str = "") -> None:
        snapshot.rejected.append({"path": path, "reason": reason, "detail": detail})

    for index, (rel, full) in enumerate(candidates):
        if index % 200 == 0:
            _check_cancel(cancel)
        if any(glob_match(rel, g) for g in config.effective_excludes) or (
            config.include_globs and not any(glob_match(rel, g) for g in config.include_globs)
        ):
            reject(rel, "excluded_by_glob")
            continue
        if full.is_symlink():
            reject(rel, "unsupported_type", "symlink")
            continue
        try:
            size = full.stat().st_size
            with full.open("rb") as handle:
                head = handle.read(_BINARY_PROBE_BYTES)
        except OSError as exc:
            reject(rel, "unsupported_type", f"unreadable: {exc}"[:200])
            continue
        if size > config.max_file_bytes:
            reject(rel, "too_large", f"{size} bytes")
            continue
        if b"\x00" in head:
            reject(rel, "binary")
            continue
        ext = full.suffix.lower()
        language = None
        kind = None
        if ext in DOC_EXTENSIONS:
            kind = "doc"
        else:
            language = detect_language(rel, head.decode("utf-8", errors="replace"))
            if language:
                kind = "code"
        if kind is None:
            reject(rel, "unsupported_type", ext or "no extension")
            continue
        if len(head) == size and not head.strip():
            reject(rel, "empty_file", "no content" if size == 0 else "whitespace only")
            continue
        if len(snapshot.files) >= config.max_files:
            snapshot.limit_reached += 1
            continue
        snapshot.files.append(RepoFile(path=rel, abs_path=full, size=size, content_kind=kind, language=language))
