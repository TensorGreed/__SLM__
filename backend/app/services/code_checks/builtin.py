"""In-process code checks: no subprocess, always available."""

from __future__ import annotations

import ast
import re
import json
import textwrap
from typing import Any

from .base import BaseCheck, make_result


class PythonAst(BaseCheck):
    id = "python_ast"
    languages = frozenset({"python"})

    def run(self, code, *, language, context):
        try:
            ast.parse(code)
            return make_result(self.id, True)
        except SyntaxError as first:
            dedented = textwrap.dedent(code)
            if dedented != code:  # a chunk cut from inside a block is indented as a whole
                try:
                    ast.parse(dedented)
                    return make_result(self.id, True)
                except SyntaxError as second:
                    first = second
            line = f"line {first.lineno}: " if first.lineno else ""
            return make_result(self.id, False, [f"{line}{first.msg}"])
        except (ValueError, RecursionError, MemoryError) as exc:
            return make_result(self.id, False, [str(exc) or type(exc).__name__])


class JsonParse(BaseCheck):
    id = "json_parse"
    languages = frozenset({"json"})

    def run(self, code, *, language, context):
        try:
            json.loads(code)
            return make_result(self.id, True)
        except ValueError as exc:
            return make_result(self.id, False, [str(exc)])


# Template expressions in templated YAML (Helm / Go templates, Jinja): not YAML
# until rendered. A line holding only template tags (``{{- if ... }}``,
# ``{% for %}``) is blanked; an inline expression becomes a placeholder scalar.
# One linear scan (an unclosed tag ends it), so a huge line can't make it slow;
# an expression may span lines (its newlines are kept, so line numbers hold).
_TEMPLATE_TAGS = (("{{", "}}"), ("{%", "%}"), ("{#", "#}"))
TEMPLATE_PLACEHOLDER = "TEMPLATE_VALUE"
_MARK = "\x00"


def strip_yaml_templates(code: str) -> str:
    """``code`` with template expressions neutralised (line numbers kept)."""
    if not any(open_ in code for open_, _ in _TEMPLATE_TAGS):
        return code
    out: list[str] = []
    i = 0
    while True:
        starts = [(code.find(open_, i), close) for open_, close in _TEMPLATE_TAGS]
        starts = [(at, close) for at, close in starts if at >= 0]
        if not starts:
            break
        at, close = min(starts)
        end = code.find(close, at + 2)
        if end < 0:
            break
        out.append(code[i:at])
        out.append(_MARK + "\n" * code.count("\n", at, end))
        i = end + len(close)
    out.append(code[i:])
    lines = []
    for line in "".join(out).split("\n"):
        if _MARK in line and not line.replace(_MARK, "").strip():
            lines.append("")
        else:
            lines.append(line.replace(_MARK, TEMPLATE_PLACEHOLDER))
    return "\n".join(lines)


_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"


def _strict_yaml_loader() -> type:
    """A SafeLoader subclass (built on first use) that refuses aliases, like the taxonomy loader, AND
    duplicate mapping keys at any depth: PyYAML silently keeps the last value, so a doubled key
    (``envFrom`` twice in one container) parsed as valid. Keys are compared by their SOURCE text and
    resolved tag (``1.0`` and ``1``, or ``on`` and ``true``, are different keys even though YAML 1.1
    resolves them to equal values; ``a`` and ``"a"`` are the same key; ``1`` and ``"1"`` differ). Only
    scalar keys are compared; ``<<`` merge keys are not counted, a key that overrides a merged one is not
    a duplicate (merging happens after this check), and a key made from a template expression
    (``TEMPLATE_PLACEHOLDER``) is never compared."""
    global _STRICT_LOADER
    if _STRICT_LOADER is not None:
        return _STRICT_LOADER
    import yaml

    from app.services.label_taxonomy_service import _NoAliasLoader

    class _StrictYamlLoader(_NoAliasLoader):
        def construct_mapping(self, node, deep=False):  # type: ignore[override]
            if isinstance(node, yaml.MappingNode):
                seen: dict[Any, Any] = {}
                for key_node, _value_node in node.value:
                    if key_node.tag == _YAML_MERGE_TAG or not isinstance(key_node, yaml.ScalarNode):
                        continue
                    if TEMPLATE_PLACEHOLDER in key_node.value:
                        continue  # two template-expression keys are not the same key once rendered
                    key = (key_node.tag, key_node.value)
                    first = seen.get(key)
                    if first is not None:
                        raise yaml.constructor.ConstructorError(
                            "while constructing a mapping", node.start_mark,
                            f"found duplicate key '{key_node.value}' (first at line {first.line + 1})",
                            key_node.start_mark,
                        )
                    seen[key] = key_node.start_mark
            return super().construct_mapping(node, deep=deep)

    _STRICT_LOADER = _StrictYamlLoader
    return _STRICT_LOADER


_STRICT_LOADER: type | None = None


def _had_control_lines(code: str, stripped: str) -> bool:
    """True when ``strip_yaml_templates`` blanked a line that held only template tags (``{{- if }}``,
    ``{% else %}``, ``{{- end }}``, ``{{- range }}``): the text is then several alternatives / repetitions
    flattened together, so the same key may legitimately appear in two branches."""
    if stripped == code:
        return False
    return any(
        o.strip() and not s.strip() and not _TEMPLATE_COMMENT_LINE.match(o)
        for o, s in zip(code.split("\n"), stripped.split("\n"))
    )


# A line holding only template comments (``{{/* … */}}``, ``{{- /* … */ -}}``, Jinja ``{# … #}``): no branch,
# so it does not switch the duplicate check off.
_TEMPLATE_COMMENT_LINE = re.compile(r"^\s*(?:(?:\{\{-?\s*/\*.*?\*/\s*-?\}\}|\{#.*?#\})\s*)+$")


class YamlParse(BaseCheck):
    """Parses every document (multi-document streams included) after ``strip_yaml_templates``. Fails on
    a syntax error, an alias (``*x``) and a duplicate mapping key at any nesting level. The duplicate check
    is skipped for templated YAML with control lines (if / else / range lines removed by the template
    strip): the branches flatten into sibling keys, so duplicates there cannot be told from alternatives."""

    id = "yaml_parse"
    languages = frozenset({"yaml"})
    revision = 2  # 2: duplicate mapping keys fail (2026-10-30)
    revision_note = "duplicate keys now fail"

    def run(self, code, *, language, context):
        import yaml

        text = strip_yaml_templates(code)
        try:
            from app.services.label_taxonomy_service import _NoAliasLoader

            loader = _NoAliasLoader if _had_control_lines(code, text) else _strict_yaml_loader()
            list(yaml.load_all(text, Loader=loader))  # noqa: S506 - SafeLoader subclasses
            return make_result(self.id, True)
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            where = f"line {mark.line + 1}: " if mark is not None else ""
            note = " (after replacing template expressions)" if text != code else ""
            return make_result(self.id, False, [where + (getattr(exc, "problem", None) or str(exc)) + note])


class JsonSchemaCheck(BaseCheck):
    id = "json_schema"
    languages = frozenset({"json"})
    requires_context = "json_schema"

    def run(self, code, *, language, context):
        schema = (context or {}).get("json_schema")
        if not isinstance(schema, dict):
            return make_result(self.id, None, skipped="no_schema")
        try:
            instance = json.loads(code)
        except ValueError as exc:
            return make_result(self.id, False, [f"not valid JSON: {exc}"])
        import jsonschema

        try:
            validator = jsonschema.validators.validator_for(schema)
            validator.check_schema(schema)
            errors = sorted(validator(schema).iter_errors(instance), key=lambda e: list(e.absolute_path))
        except jsonschema.SchemaError as exc:
            return make_result(self.id, None, [f"invalid schema: {exc.message}"], skipped="unavailable")
        if not errors:
            return make_result(self.id, True)
        msgs = [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors]
        return make_result(self.id, False, msgs)


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


class PythonUndefinedNames(BaseCheck):
    """pyflakes, in-process (it only parses: nothing is imported or run). Fails on undefined names (incl.
    ``__all__`` entries and locals used before assignment) and on syntax errors; unused imports, redefinitions
    and other style warnings are ignored. An indented fragment (a method body) is dedented first and may use
    ``self`` / ``cls``. A name that appears in the text the model was shown (``context["grounding_text"]``: the
    question and its passages) counts as defined: a snippet continues code the question shows (a notebook's
    ``spark`` / ``df``, a constant from an earlier cell). Gated like an external check: registered outside the core
    built-ins, so it runs only when its id is in ``CODE_CHECK_ALLOWED`` and ``pyflakes`` is importable."""

    id = "python_undefined_names"
    languages = frozenset({"python"})
    revision = 2  # 2: names shown in the question / passages count as defined (2026-11-09)
    revision_note = "names shown in the question count as defined"

    def available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("pyflakes") is not None

    def run(self, code, *, language, context):
        from pyflakes import checker
        from pyflakes import messages as pm

        extra: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError as first:
            dedented = textwrap.dedent(code)
            try:
                if dedented == code:
                    raise first
                tree = ast.parse(dedented)
                extra = {"self", "cls"}  # a method body cut out of its class
            except SyntaxError as exc:
                line = f"line {exc.lineno}: " if exc.lineno else ""
                return make_result(self.id, False, [f"{line}{exc.msg}"])
        except (ValueError, RecursionError, MemoryError) as exc:
            return make_result(self.id, False, [str(exc) or type(exc).__name__])
        grounding = (context or {}).get("grounding_text") if isinstance(context, dict) else None
        if isinstance(grounding, str) and grounding:
            extra |= set(_IDENTIFIER_RE.findall(grounding))
        found = checker.Checker(tree, filename="<snippet>", builtins=extra).messages
        bad = [m for m in found if isinstance(m, (pm.UndefinedName, pm.UndefinedExport, pm.UndefinedLocal))]
        if not bad:
            return make_result(self.id, True)
        bad.sort(key=lambda m: (getattr(m, "lineno", 0), getattr(m, "col", 0)))
        return make_result(self.id, False, [f"line {m.lineno}: {m.message % m.message_args}" for m in bad])


def properties_logical_lines(code: str) -> list[tuple[int, str]]:
    """``(first physical line number, text)`` of each logical line of a ``.properties`` file, per the Java
    ``Properties.load`` rules: blank lines and comment lines (first non-blank character ``#`` or ``!``) are
    skipped; a line ending in an odd number of backslashes continues on the next line, whose leading
    whitespace is dropped. A comment line never continues."""
    out: list[tuple[int, str]] = []
    lines = re.split(r"\r\n|\r|\n", code)  # Java breaks lines only on these
    i = 0
    while i < len(lines):
        start = i
        text = lines[i].lstrip(" \t\f")
        i += 1
        if not text or text[0] in "#!":
            continue
        parts: list[str] = []
        while True:
            trailing = len(text) - len(text.rstrip("\\"))
            if trailing % 2 == 1 and i < len(lines):
                parts.append(text[:-1])
                text = lines[i].lstrip(" \t\f")
                i += 1
                continue
            parts.append(text[:-1] if trailing % 2 == 1 else text)
            break
        out.append((start + 1, "".join(parts)))
    return out


# A logical line starting with one of these is JSON / YAML structure, not ``key=value``.
_PROPERTIES_BAD_STARTS = ("{", "}", "[", "]", "- ")
_PROPERTIES_MAX_LINES_REPORTED = 5


class PropertiesParse(BaseCheck):
    """A ``.properties`` file (Java ``Properties.load`` syntax). Almost any text loads as properties, so the
    check fails only on the clear mistakes: the whole text parses as a JSON object or array, or a logical line
    (after ``\\`` continuations; ``#`` / ``!`` comments and blank lines skipped) starts with ``{``, ``}``,
    ``[``, ``]`` or ``- `` (a YAML list item). ``key=value``, ``key: value``, ``key value`` and a bare ``key``
    (empty value) are valid. Gated like ``python_undefined_names``: runs only when its id is in
    ``CODE_CHECK_ALLOWED``."""

    id = "properties_parse"
    languages = frozenset({"properties"})

    def run(self, code, *, language, context):
        stripped = code.strip()
        if stripped[:1] in ("{", "["):
            try:
                parsed = json.loads(stripped)
            except ValueError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                kind = "object" if isinstance(parsed, dict) else "array"
                return make_result(self.id, False, [
                    f"the text is a JSON {kind}, not a .properties file; write one key=value pair per line"
                ])
        bad = [
            (lineno, text) for lineno, text in properties_logical_lines(code)
            if text.startswith(_PROPERTIES_BAD_STARTS)
        ]
        if not bad:
            return make_result(self.id, True)
        msgs = []
        for lineno, text in bad[:_PROPERTIES_MAX_LINES_REPORTED]:
            start = "-" if text.startswith("- ") else text[0]
            msgs.append(
                f"line {lineno}: starts with '{start}' -- a .properties line is key=value "
                "(JSON or YAML structure is not valid here)"
            )
        if len(bad) > _PROPERTIES_MAX_LINES_REPORTED:
            msgs.append(f"and {len(bad) - _PROPERTIES_MAX_LINES_REPORTED} more line(s) like these")
        return make_result(self.id, False, msgs)


from .grounding import ContextGrounding  # noqa: E402 - grounding imports this module's YAML helpers lazily

BUILTINS: list[Any] = [PythonAst(), JsonParse(), YamlParse(), JsonSchemaCheck(), ContextGrounding()]
# In-process, but gated like external checks (CODE_CHECK_ALLOWED + importable dependency).
GATED_IN_PROCESS: list[Any] = [PythonUndefinedNames(), PropertiesParse()]
