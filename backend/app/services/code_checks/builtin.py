"""In-process code checks: no subprocess, always available."""

from __future__ import annotations

import ast
import json
import re
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
_TEMPLATE_EXPR = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}")
TEMPLATE_PLACEHOLDER = "TEMPLATE_VALUE"


def strip_yaml_templates(code: str) -> str:
    """``code`` with template expressions neutralised (line numbers kept)."""
    if "{{" not in code and "{%" not in code and "{#" not in code:
        return code
    lines = []
    for line in code.split("\n"):
        if line.strip() and not _TEMPLATE_EXPR.sub("", line).strip():
            lines.append("")
        else:
            lines.append(_TEMPLATE_EXPR.sub(TEMPLATE_PLACEHOLDER, line))
    return "\n".join(lines)


class YamlParse(BaseCheck):
    id = "yaml_parse"
    languages = frozenset({"yaml"})

    def run(self, code, *, language, context):
        import yaml

        from app.services.label_taxonomy_service import _NoAliasLoader

        text = strip_yaml_templates(code)
        try:
            list(yaml.load_all(text, Loader=_NoAliasLoader))  # noqa: S506 - SafeLoader subclass
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
