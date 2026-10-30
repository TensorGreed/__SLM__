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


BUILTINS: list[Any] = [PythonAst(), JsonParse(), YamlParse(), JsonSchemaCheck()]
