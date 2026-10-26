"""Git repository source: one row per source chunk, one per prose file.

Locator format: ``git:<url>[@<ref>]``, e.g.
``git:https://github.com/org/repo@v1.2``. Source / config files are cut at
declaration boundaries (:mod:`app.services.code_chunker`) and yield
``{path, language, start_line, end_line, symbol, text}``; Markdown / text /
HTML files yield one ``{path, text}`` row, so the ``text_only`` mapper can use
either. ``describe`` adds the repo's licence and the pinned commit. For the
full document workflow (provenance, rejected-file review) use Data →
Documents from website / repo instead.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from app.services.code_chunker import chunk_code
from app.services.dataset_import.protocols import RawRow
from app.services.dataset_import.registry import register_source
from app.services.document_sources.git_repo import GitRepoConfig, clone_repo
from app.utils.file_parsers import DocumentParseError, html_to_text

CHUNK_MAX_CHARS = 1500
_SAMPLE_ROWS = 20
_REMOTE_WITH_PATH = re.compile(r"^(?:(?:https?|ssh)://[^/@]+(?:@[^/]+)?/.+|[\w.-]+@[\w.-]+:.+|(?:file://|/|\./|~).+)$")

# Tests swap this (a fake git runner).
_runner: Callable[..., Any] | None = None


def split_repo_locator(rest: str) -> tuple[str, str]:
    """``<url>[@<ref>]`` → ``(url, ref)``; ref defaults to ``HEAD``. The last
    ``@`` starts a ref only when what precedes it already names a repository
    path (so ``git@host:org/repo`` and ``https://user@host/x`` are not cut)."""
    rest = rest.strip()
    if "@" in rest:
        url, _, ref = rest.rpartition("@")
        if ref and ":" not in ref and "@" not in ref and _REMOTE_WITH_PATH.match(url):
            return url, ref
    return rest, "HEAD"


class GitSource:
    source_id: str = "git"

    def _snapshot(self, rest: str, limit: int | None):
        url, ref = split_repo_locator(rest)
        config = GitRepoConfig(url=url, ref=ref, **({"max_files": limit} if limit else {}))
        return clone_repo(config, runner=_runner)

    def _rows(self, snapshot, limit: int | None) -> Iterable[RawRow]:
        emitted = 0
        for repo_file in snapshot.files:
            text = repo_file.read_bytes().decode("utf-8", errors="replace")
            if repo_file.content_kind == "code":
                rows = [
                    {
                        "path": repo_file.path, "language": repo_file.language, "start_line": chunk.start_line,
                        "end_line": chunk.end_line, "symbol": chunk.symbol, "text": chunk.text,
                    }
                    for chunk in chunk_code(text, repo_file.language, max_chars=CHUNK_MAX_CHARS)
                ]
            else:
                if repo_file.path.lower().endswith((".html", ".htm")):
                    try:
                        text = html_to_text(text)
                    except DocumentParseError:
                        continue
                rows = [{"path": repo_file.path, "text": text}] if text.strip() else []
            for row in rows:
                yield row
                emitted += 1
                if limit and emitted >= limit:
                    return

    def load(self, locator: str, *, limit: int | None = None) -> Iterable[RawRow]:
        with self._snapshot(locator, None) as snapshot:
            yield from self._rows(snapshot, limit)

    def describe(self, locator: str) -> dict[str, Any]:
        with self._snapshot(locator, None) as snapshot:
            rows = list(self._rows(snapshot, _SAMPLE_ROWS))
            columns: list[str] = []
            for row in rows:
                for key in row:
                    if key not in columns:
                        columns.append(key)
            return {
                "source_id": self.source_id,
                "locator": locator,
                "approximate_total_rows": None,
                "sample_rows": rows,
                "columns": columns,
                "licence": snapshot.licence,
                "commit": snapshot.commit,
                "ref": snapshot.ref,
                "files": len(snapshot.files),
            }


register_source("git", GitSource)
