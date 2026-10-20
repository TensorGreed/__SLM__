"""Strip machine-specific absolute paths from what an export records.

An export's manifest and release notes travel with the model (the Hub
publish uploads them under ``brewslm/``), so they must not carry the
builder's home directory. Paths inside the export run directory become
relative to it (the run dir itself is ``.``); anything else under the home
directory becomes ``~/…``. Pure — no filesystem access."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _prefixes(path: Path | str | None) -> list[str]:
    """The spellings of ``path`` to look for: as given and resolved."""
    if path is None:
        return []
    out: list[str] = []
    for candidate in (Path(path), Path(path).expanduser()):
        for spelled in (str(candidate), _resolved(candidate)):
            token = spelled.rstrip("/")
            if token and token != "/" and token not in out:
                out.append(token)
    # Longest first, so a resolved symlink target isn't half-replaced.
    return sorted(out, key=len, reverse=True)


def _resolved(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def portable_text(text: str, run_dir: Path | str | None, home: Path | str | None = None) -> str:
    """``text`` with ``run_dir`` paths made relative and the home dir as ``~``."""
    if home is None:
        home = Path.home()
    for prefix in _prefixes(run_dir):
        text = text.replace(prefix + "/", "").replace(prefix, ".")
    for prefix in _prefixes(home):
        text = text.replace(prefix + "/", "~/").replace(prefix, "~")
    return text


def portable_paths(value: Any, run_dir: Path | str | None, home: Path | str | None = None) -> Any:
    """A copy of a JSON-like ``value`` with every string run through
    :func:`portable_text` (dict keys too)."""
    if isinstance(value, str):
        return portable_text(value, run_dir, home)
    if isinstance(value, dict):
        return {
            (portable_text(k, run_dir, home) if isinstance(k, str) else k): portable_paths(v, run_dir, home)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [portable_paths(v, run_dir, home) for v in value]
    if isinstance(value, tuple):
        return [portable_paths(v, run_dir, home) for v in value]
    return value
