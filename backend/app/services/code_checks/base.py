"""Shared types for code checks (no registry dependency, so built-ins can import it)."""

from __future__ import annotations

import shutil
from typing import Any, Protocol, runtime_checkable

MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 300

_ALIASES = {
    "py": "python", "python3": "python", "js": "javascript", "node": "javascript", "mjs": "javascript",
    "sh": "shell", "bash": "shell", "zsh": "shell", "tf": "terraform", "hcl": "terraform",
    "cs": "csharp", "c#": "csharp", "golang": "go", "yml": "yaml",
}


def normalize_language(language: str | None) -> str:
    key = str(language or "").strip().lower()
    return _ALIASES.get(key, key)


def clip_messages(messages: list[str] | None) -> list[str]:
    out: list[str] = []
    for m in messages or []:
        text = " ".join(str(m).split())
        if text:
            out.append(text[:MAX_MESSAGE_CHARS])
        if len(out) >= MAX_MESSAGES:
            break
    return out


def make_result(check_id: str, ok: bool | None, messages: list[str] | None = None, skipped: str | None = None) -> dict[str, Any]:
    return {"check_id": check_id, "ok": ok, "messages": clip_messages(messages), "skipped": skipped}


@runtime_checkable
class CodeCheck(Protocol):
    id: str
    languages: frozenset[str]
    kind: str  # "builtin" | "external" | "detector"
    binary: str | None

    def available(self) -> bool: ...

    def run(self, code: str, *, language: str, context: dict[str, Any]) -> dict[str, Any]: ...


class BaseCheck:
    id = ""
    languages: frozenset[str] = frozenset()
    kind = "builtin"
    binary: str | None = None
    requires_context: str | None = None  # a context key that must be present for default selection

    def available(self) -> bool:
        return True if not self.binary else shutil.which(self.binary) is not None

    def run(self, code: str, *, language: str, context: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError
