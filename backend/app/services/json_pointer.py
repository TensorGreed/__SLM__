"""RFC 6901 JSON-pointer helpers shared by the probe runner, the schema bundle and structured-plan scoring."""

from __future__ import annotations

import re
from typing import Any, Iterable


def split_json_pointer(pointer: str) -> list[str] | None:
    """RFC 6901 tokens of ``pointer`` (``*`` is kept literally as a wildcard
    token); ``None`` when it is not a valid pointer."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw):
            return None
        tokens.append(raw.replace("~1", "/").replace("~0", "~"))
    return tokens


def resolve_json_pointer(doc: Any, tokens: list[str]) -> list[Any]:
    """Every value ``tokens`` resolves to; a ``*`` token fans out over all
    array elements / object values. Missing paths contribute nothing."""
    current = [doc]
    for tok in tokens:
        nxt: list[Any] = []
        for node in current:
            if tok == "*":
                if isinstance(node, list):
                    nxt.extend(node)
                elif isinstance(node, dict):
                    nxt.extend(node.values())
            elif isinstance(node, dict):
                if tok in node:
                    nxt.append(node[tok])
            elif isinstance(node, list) and tok.isascii() and tok.isdigit() and int(tok) < len(node):
                nxt.append(node[int(tok)])
        current = nxt
        if not current:
            break
    return current


def format_json_pointer(path_parts: Iterable[str | int]) -> str:
    """RFC 6901 pointer for ``path_parts`` (``~`` -> ``~0``, ``/`` -> ``~1``); empty path -> ``""``."""
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in path_parts)
