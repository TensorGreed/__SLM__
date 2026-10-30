"""Code-check registry: cheap, deterministic "does this code still parse" checks.

A check has an id, a set of languages, a kind (``builtin`` in-process, ``external``
sandboxed binary, ``detector`` an external finder) and returns a result dict::

    {"check_id", "ok": bool|None, "messages": [str] (<=20, each <=300 chars), "skipped": str|None}

``ok`` None means it did not run; ``skipped`` says why: ``unavailable``,
``not_allowed``, ``language_mismatch``, ``timeout``, ``no_schema``,
``no_module_dir``, ``unknown_check``.

Built-ins are always available. Anything else runs only when its id is in
``settings.CODE_CHECK_ALLOWED`` (comma list; default empty) AND its binary
resolves on PATH. Plugins (``settings.CODE_CHECK_PLUGIN_MODULES``, comma list)
export ``register_code_checks(register)`` or a ``CODE_CHECKS`` list.

Formatters: a check may also define ``format(code, *, language, context)`` (the language's canonical
formatter: gofmt, terraform fmt, rustfmt). ``format_code`` runs it under the check's gate (allowed +
available), sandbox and timeouts; used only by the opt-in "format before check" eval setting.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

from app.config import settings
from app.services import sandbox_exec
from app.services.code_checks.base import CodeCheck, make_result, normalize_language

__all__ = [
    "CodeCheck", "normalize_language", "register_code_check", "list_code_checks", "checks_for_language",
    "run_code_checks", "load_code_check_plugins", "load_code_check_plugins_from_settings",
    "get_plugin_diagnostics", "allowed_check_ids", "is_allowed", "get_code_check", "format_code", "formatter_for", "check_revisions",
]

_REGISTRY: dict[str, Any] = {}
_CORE_BUILTINS: list[Any] = []  # the in-process built-in instances registered by the core module
_LOCK = threading.RLock()
_LOADED_PLUGIN_MODULES: set[str] = set()
_PLUGIN_ERRORS: dict[str, str] = {}


def _split(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value or "").split(",") if p.strip()]


def allowed_check_ids() -> set[str]:
    return set(_split(settings.CODE_CHECK_ALLOWED))


def is_allowed(check: Any) -> bool:
    """Real built-ins (the instances the core registers, by identity) always run; a plugin that
    merely declares ``kind == "builtin"`` still needs its id in ``CODE_CHECK_ALLOWED``."""
    if any(check is b for b in _CORE_BUILTINS):
        return True
    return check.id in allowed_check_ids()


def register_code_check(check: Any) -> None:
    cid = getattr(check, "id", None)
    if not isinstance(cid, str) or not cid.strip():
        raise ValueError("a code check needs a non-empty string id")
    if not callable(getattr(check, "run", None)) or not callable(getattr(check, "available", None)):
        raise ValueError(f"code check '{cid}' must define available() and run()")
    langs = getattr(check, "languages", None)
    if not langs:
        raise ValueError(f"code check '{cid}' must declare languages")
    check.languages = frozenset(normalize_language(x) for x in langs)
    with _LOCK:
        _REGISTRY[cid] = check


def get_code_check(check_id: str) -> Any | None:
    return _REGISTRY.get(check_id)


def check_revisions(check_ids: list[str]) -> dict[str, int]:
    """``{id: revision}`` for the given checks whose semantics changed since they shipped (revision > 1);
    checks still at revision 1 are left out, so fingerprints that never included them stay as they were."""
    out: dict[str, int] = {}
    for cid in check_ids:
        rev = getattr(_REGISTRY.get(cid), "revision", 1)
        if isinstance(rev, int) and rev > 1:
            out[cid] = rev
    return out


def list_code_checks() -> list[dict[str, Any]]:
    out = []
    for cid in sorted(_REGISTRY):
        c = _REGISTRY[cid]
        out.append({
            "id": cid,
            "languages": sorted(c.languages),
            "kind": c.kind,
            "binary": c.binary,
            "available": bool(c.available()),
            "allowed": is_allowed(c),
        })
    return out


def checks_for_language(language: str | None, check_ids: list[str] | None = None) -> list[Any]:
    """Registered checks that support ``language``. Without ``check_ids`` this is the
    default selection: runnable (available + allowed) non-detector checks. With ids,
    every named check in the order given (unknown ids are dropped here)."""
    lang = normalize_language(language)
    if check_ids is not None:
        return [_REGISTRY[i] for i in check_ids if i in _REGISTRY and lang in _REGISTRY[i].languages]
    return [
        c for _id, c in sorted(_REGISTRY.items())
        if lang in c.languages and c.kind != "detector" and is_allowed(c) and c.available()
    ]


def _execute(check: Any, code: str, lang: str, context: dict[str, Any]) -> dict[str, Any]:
    if lang not in check.languages:
        return make_result(check.id, None, skipped="language_mismatch")
    if not is_allowed(check):
        return make_result(check.id, None, skipped="not_allowed")
    if not check.available():
        return make_result(check.id, None, skipped="unavailable")
    try:
        res = check.run(code, language=lang, context=context)
    except sandbox_exec.SandboxError as exc:
        return make_result(check.id, None, [str(exc)], skipped="unavailable")
    except Exception as exc:  # noqa: BLE001 - a broken plugin must not fail the caller
        return make_result(check.id, None, [f"{type(exc).__name__}: {exc}"], skipped="unavailable")
    return make_result(check.id, res.get("ok"), res.get("messages"), res.get("skipped"))


def run_code_checks(
    code: str, *, language: str | None, check_ids: list[str] | None = None, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run checks over ``code``. ``ok`` = every check that ran passed; None when none ran."""
    lang = normalize_language(language)
    ctx = dict(context or {})
    results: list[dict[str, Any]] = []
    if check_ids is None:
        selected = [c for c in checks_for_language(lang)
                    if getattr(c, "requires_context", None) is None or ctx.get(c.requires_context) is not None]
        results = [_execute(c, code, lang, ctx) for c in selected]
    else:
        for cid in check_ids:
            c = _REGISTRY.get(cid)
            results.append(make_result(cid, None, skipped="unknown_check") if c is None else _execute(c, code, lang, ctx))
    ran = [r for r in results if r["ok"] is not None]
    return {"ok": (all(r["ok"] for r in ran) if ran else None), "results": results}


FORMAT_STATUSES = (
    "formatted", "unchanged", "failed", "not_formattable", "truncated", "unavailable", "not_allowed", "timeout",
    "no_formatter",
)


def formatter_for(language: str | None) -> Any | None:
    """The registered check that formats ``language`` (first by id), or None."""
    lang = normalize_language(language)
    for _id, c in sorted(_REGISTRY.items()):
        if lang in c.languages and callable(getattr(c, "format", None)):
            return c
    return None


def format_code(code: str, *, language: str | None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the language's canonical formatter over ``code``.

    Returns ``{"status", "formatter", "code", "messages"}``; ``code`` is what to check next: the
    formatted text on ``formatted`` / ``unchanged``, the ORIGINAL code otherwise. ``failed`` = the
    formatter rejected the code (a syntax error: the checks then fail it as before); ``unavailable`` /
    ``not_allowed`` / ``timeout`` = it did not run (binary missing, not in ``CODE_CHECK_ALLOWED``, too
    slow); ``not_formattable`` = a fragment the formatter cannot take on its own but that parses (rustfmt
    on statements); ``truncated`` = the output hit the sandbox cap (never used); ``no_formatter`` = the
    language has none. Never raises."""
    lang = normalize_language(language)
    check = formatter_for(lang)

    def out(status: str, text: str, messages: list[str] | None = None) -> dict[str, Any]:
        return {"status": status, "formatter": getattr(check, "id", None), "code": text,
                "messages": list(messages or [])[:5]}

    if check is None:
        return out("no_formatter", code)
    if not is_allowed(check):
        return out("not_allowed", code)
    if not check.available():
        return out("unavailable", code)
    try:
        res = check.format(code, language=lang, context=dict(context or {}))
    except Exception as exc:  # noqa: BLE001 - a broken formatter must not fail the caller
        return out("unavailable", code, [f"{type(exc).__name__}: {exc}"])
    ok = res.get("ok") if isinstance(res, dict) else None
    if ok is True and isinstance(res.get("code"), str):
        formatted = res["code"]
        return out("unchanged" if formatted == code else "formatted", formatted)
    if ok is False:
        return out("failed", code, res.get("messages"))
    skipped = (res or {}).get("skipped")
    status = skipped if skipped in ("timeout", "not_formattable", "truncated") else "unavailable"
    return out(status, code, (res or {}).get("messages"))


# ── plugins ──


def _register_builtins() -> None:
    from app.services.code_checks.builtin import BUILTINS
    from app.services.code_checks.external import EXTERNALS

    for c in BUILTINS + EXTERNALS:
        register_code_check(c)
    _CORE_BUILTINS[:] = list(BUILTINS)


def load_code_check_plugins(module_paths: list[str], *, force_reload: bool = False) -> dict[str, Any]:
    loaded: list[str] = []
    failed: dict[str, str] = {}
    registered = 0
    with _LOCK:
        for raw in module_paths:
            name = str(raw or "").strip()
            if not name:
                continue
            if name in _LOADED_PLUGIN_MODULES and not force_reload:
                loaded.append(name)
                continue
            try:
                module = importlib.import_module(name)
                if force_reload:
                    module = importlib.reload(module)
                before = len(_REGISTRY)
                hook = getattr(module, "register_code_checks", None)
                items = getattr(module, "CODE_CHECKS", None)
                if not callable(hook) and not isinstance(items, (list, tuple)):
                    raise ValueError("module must export register_code_checks(register) or a CODE_CHECKS list")
                if callable(hook):
                    hook(register_code_check)
                for item in items or []:
                    register_code_check(item() if isinstance(item, type) else item)
                _LOADED_PLUGIN_MODULES.add(name)
                _PLUGIN_ERRORS.pop(name, None)
                loaded.append(name)
                registered += max(0, len(_REGISTRY) - before)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                _PLUGIN_ERRORS[name] = msg
                failed[name] = msg
    return {
        "requested_modules": [str(m).strip() for m in module_paths if str(m).strip()],
        "loaded_modules": sorted(set(loaded)),
        "failed_modules": failed,
        "registered_checks": registered,
    }


def load_code_check_plugins_from_settings(*, force_reload: bool = False) -> dict[str, Any]:
    modules = _split(settings.CODE_CHECK_PLUGIN_MODULES)
    if not modules:
        return {"requested_modules": [], "loaded_modules": sorted(_LOADED_PLUGIN_MODULES),
                "failed_modules": dict(_PLUGIN_ERRORS), "registered_checks": 0, "status": "no_plugin_modules_configured"}
    return load_code_check_plugins(modules, force_reload=force_reload)


def get_plugin_diagnostics() -> dict[str, Any]:
    return {"loaded_modules": sorted(_LOADED_PLUGIN_MODULES), "failed_modules": dict(_PLUGIN_ERRORS)}


_register_builtins()
