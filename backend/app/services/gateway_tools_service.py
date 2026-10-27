"""Named tools of the serving gateway (``runtime_config["gateway"]["tools"]``).

A named tool is a saved prompt over one model of a project: a JSON-Schema
input, a prompt template and optional retrieval. MCP clients see each as a
tool; calling it renders the prompt from the arguments and runs the normal
chat-completion path.

Templates are rendered with a SAFE substitution: only ``{identifier}`` tokens
are replaced from the arguments, everything else stays literal (no
``str.format`` attribute / index access).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from jsonschema import Draft202012Validator

TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# Built-in MCP tool names a project tool may not shadow.
RESERVED_TOOL_NAMES = frozenset({"list_models", "run_model", "retrieve"})
_MODEL_SUFFIX_RE = re.compile(r"^(|:base|:run-\d+)$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
RECIPE_TEMPLATE = "@recipe"
MAX_TEMPLATE_CHARS = 20_000
MAX_TOOLS_PER_PROJECT = 50


def template_placeholders(template: str) -> set[str]:
    """The ``{name}`` placeholders ``render_template`` would substitute."""
    return set(_PLACEHOLDER_RE.findall(str(template or "")))


def render_template(template: str, arguments: dict[str, Any]) -> str:
    """Replace ``{name}`` with ``arguments[name]`` (strings as-is, anything else
    as JSON); unknown names and every other brace stay literal."""

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in arguments:
            return match.group(0)
        value = arguments[key]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return _PLACEHOLDER_RE.sub(_sub, template)


def argument_errors(schema: dict[str, Any], arguments: Any, *, limit: int = 5) -> list[str]:
    if not isinstance(arguments, dict):
        return ["arguments must be an object"]
    validator = Draft202012Validator(schema)
    out: list[str] = []
    for err in sorted(validator.iter_errors(arguments), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path)
        out.append(f"{where}: {err.message}" if where else err.message)
        if len(out) >= limit:
            break
    return out


# ``prompt_template: "@recipe"`` = "use this project's training prompt format". A recipe
# that has a deterministic training-prompt builder registers it here:
# ``builder(project, arguments, adapter_config) -> prompt`` (``adapter_config`` is the
# project's dataset-adapter config; the builder may read the disk, so the gateway calls it
# in a thread). Recipes without a builder get a clear 400 and need a ``prompt_template``.
RecipeBuilder = Callable[[Any, dict[str, Any], dict[str, Any]], str]
RECIPE_PROMPT_BUILDERS: dict[str, RecipeBuilder] = {}


def register_recipe_prompt_builder(recipe_id: str, builder: RecipeBuilder) -> None:
    RECIPE_PROMPT_BUILDERS[str(recipe_id)] = builder


def project_recipe_id(project: Any) -> str:
    snap = getattr(project, "selected_recipe", None)
    return str(snap.get("recipe_id") or "") if isinstance(snap, dict) else ""


def recipe_supported(project: Any) -> bool:
    return project_recipe_id(project) in RECIPE_PROMPT_BUILDERS


def _code_findings_builder(project: Any, arguments: dict[str, Any], adapter_config: dict[str, Any]) -> str:
    """Code-findings: label precedence mirrors prep -- the adapter config's explicit
    ``finding_labels``, else the manifest's ``prompt_labels``, else the project taxonomy."""
    from app.services import label_taxonomy_service as lts
    from app.services.code_findings_prompt import build_code_findings_prompt, taxonomy_label_ids
    from app.services.eval_task_handler_service import read_prepared_manifest

    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("argument 'code' (a non-empty string) is required for @recipe")
    explicit = adapter_config.get("finding_labels")
    if isinstance(explicit, list):
        labels = [str(x) for x in explicit]
    else:
        schema = read_prepared_manifest(project.id).get("output_schema") or {}
        labels = list(schema.get("prompt_labels") or []) if isinstance(schema, dict) else []
        if not labels:
            labels = taxonomy_label_ids(lts.project_taxonomy(project))
    start = arguments.get("start_line")
    start = start if isinstance(start, int) and not isinstance(start, bool) else None
    path = arguments.get("path") if isinstance(arguments.get("path"), str) else None
    language = arguments.get("language") if isinstance(arguments.get("language"), str) else None
    return build_code_findings_prompt(code, language, path, start, labels or None)


register_recipe_prompt_builder("code-findings", _code_findings_builder)


def build_recipe_prompt(project: Any, arguments: dict[str, Any], adapter_config: dict[str, Any] | None = None) -> str:
    """The project's training-format prompt for ``arguments``; reads the disk."""
    builder = RECIPE_PROMPT_BUILDERS.get(project_recipe_id(project))
    if builder is None:
        raise ValueError("no recipe prompt builder; write a prompt_template")
    return builder(project, arguments, adapter_config or {})


def build_tool_prompt(
    project: Any, tool: dict[str, Any], arguments: dict[str, Any], adapter_config: dict[str, Any] | None = None
) -> str:
    template = str(tool.get("prompt_template") or "")
    if template.strip() == RECIPE_TEMPLATE:
        return build_recipe_prompt(project, arguments, adapter_config)
    return render_template(template, arguments)


def normalize_tools(raw: Any, project: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """``(tools, errors)``: cleaned tools and per-field messages (``tools[0].name``)."""
    errors: list[dict[str, str]] = []
    if raw is None:
        return [], errors
    if not isinstance(raw, list):
        return [], [{"field": "tools", "message": "tools must be a list"}]
    if len(raw) > MAX_TOOLS_PER_PROJECT:
        errors.append({"field": "tools", "message": f"at most {MAX_TOOLS_PER_PROJECT} tools per project"})
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(raw[:MAX_TOOLS_PER_PROJECT]):
        pre = f"tools[{i}]"
        if not isinstance(item, dict):
            errors.append({"field": pre, "message": "each tool must be an object"})
            continue
        name = str(item.get("name") or "").strip()
        if not TOOL_NAME_RE.match(name):
            errors.append({"field": f"{pre}.name", "message": "name must match ^[a-z][a-z0-9_]{0,63}$"})
        elif name in RESERVED_TOOL_NAMES:
            errors.append({"field": f"{pre}.name", "message": f"'{name}' is a reserved tool name"})
        elif name in seen:
            errors.append({"field": f"{pre}.name", "message": f"duplicate tool name '{name}'"})
        seen.add(name)
        schema = item.get("input_schema")
        if not isinstance(schema, dict):
            errors.append({"field": f"{pre}.input_schema", "message": "input_schema must be a JSON Schema object"})
            schema = {"type": "object"}
        else:
            if schema.get("type") not in (None, "object"):
                errors.append({"field": f"{pre}.input_schema", "message": "input_schema.type must be 'object'"})
            try:
                Draft202012Validator.check_schema(schema)
            except Exception as exc:  # noqa: BLE001
                errors.append({"field": f"{pre}.input_schema", "message": f"invalid JSON Schema: {str(exc).splitlines()[0][:200]}"})
        model = item.get("model") or ""
        if not isinstance(model, str) or not _MODEL_SUFFIX_RE.match(model):
            errors.append({"field": f"{pre}.model", "message": "model must be '', ':base' or ':run-<id>'"})
            model = ""
        template = item.get("prompt_template")
        if not isinstance(template, str) or not template.strip():
            errors.append({"field": f"{pre}.prompt_template", "message": "prompt_template is required"})
            template = ""
        elif len(template) > MAX_TEMPLATE_CHARS:
            errors.append({"field": f"{pre}.prompt_template", "message": f"at most {MAX_TEMPLATE_CHARS} characters"})
        elif template.strip() == RECIPE_TEMPLATE and not recipe_supported(project):
            errors.append({"field": f"{pre}.prompt_template", "message": "no recipe prompt builder; write a prompt_template"})
        max_tokens = item.get("max_tokens")
        if max_tokens is not None:
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
                errors.append({"field": f"{pre}.max_tokens", "message": "max_tokens must be an integer 1..4096"})
                max_tokens = None
        system = item.get("system_prompt")
        if system is not None and not isinstance(system, str):
            errors.append({"field": f"{pre}.system_prompt", "message": "system_prompt must be a string"})
            system = None
        retrieval = item.get("retrieval")
        if retrieval is not None and not isinstance(retrieval, bool):
            errors.append({"field": f"{pre}.retrieval", "message": "retrieval must be true or false"})
            retrieval = None
        tools.append(
            {
                "name": name,
                "description": str(item.get("description") or "")[:1000],
                "input_schema": schema,
                "model": model,
                "prompt_template": template,
                "system_prompt": system,
                "retrieval": retrieval,
                "max_tokens": max_tokens,
            }
        )
    return tools, errors
