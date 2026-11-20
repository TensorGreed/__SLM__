"""The chat template a GGUF package's Ollama ``Modelfile`` carries (``TEMPLATE`` / ``SYSTEM`` / ``stop``).

The run was trained on its tokenizer's chat template (``tokenizer_config.json`` ``chat_template`` or
``chat_template.jinja``, saved with the run) unless ``use_tokenizer_chat_template`` was off. A Jinja
template cannot be translated to Ollama's Go template in general, so only well-known turn formats are
converted, recognised by their turn markers and checked against a small render of the Jinja template. The
Ollama templates render the system message once, from ``.System`` (a request's own, else ``SYSTEM``), and
the user / assistant turns from ``.Messages`` (system turns skipped there):

  * ChatML (``<|im_start|>`` / ``<|im_end|>``), with a literal default system message when the template
    injects one (written as ``SYSTEM``, which Ollama applies only when a request has none, as the Jinja does);
  * Llama 3 headers (``<|start_header_id|>`` / ``<|eot_id|>``) without injected date / tool preambles;
  * Gemma turns (``<start_of_turn>`` / ``<end_of_turn>``, the ``model`` role, system folded into the
    first user turn is NOT reproduced, so only templates without system support are converted).

Anything else returns a reason instead, and the package notes say Ollama will use the template from the
GGUF metadata or its default for the base family. Pure (file reads only).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CHATML = (
    '{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
    '{{- range .Messages }}{{- if ne .Role "system" }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}{{ end }}'
    '<|im_start|>assistant\n'
)
LLAMA3 = (
    '{{- if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}'
    '{{- range .Messages }}{{- if ne .Role "system" }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n{{ .Content }}<|eot_id|>{{ end }}{{ end }}'
    '<|start_header_id|>assistant<|end_header_id|>\n\n'
)
GEMMA = (
    '{{- range .Messages }}{{- if ne .Role "system" }}<start_of_turn>{{ if eq .Role "assistant" }}model{{ else }}{{ .Role }}{{ end }}\n'
    '{{ .Content }}<end_of_turn>\n{{ end }}{{ end }}<start_of_turn>model\n'
)

_CHATML_DEFAULT_SYSTEM = re.compile(r"<\|im_start\|>system\\n(.+?)<\|im_end\|>", re.S)


def find_chat_template(dirs: list[Path]) -> tuple[str, str] | None:
    """``(jinja template, source file)`` from the first dir holding one (``chat_template.jinja``, else
    ``tokenizer_config.json``'s ``chat_template``: a string, or the ``default`` entry of a named list)."""
    for d in dirs:
        try:
            jinja = d / "chat_template.jinja"
            if jinja.is_file():
                text = jinja.read_text(encoding="utf-8")
                if text.strip():
                    return text, str(jinja)
            cfg = d / "tokenizer_config.json"
            if cfg.is_file():
                value = json.loads(cfg.read_text(encoding="utf-8")).get("chat_template")
                if isinstance(value, list):
                    named = {str(x.get("name")): x.get("template") for x in value if isinstance(x, dict)}
                    value = named.get("default")
                if isinstance(value, str) and value.strip():
                    return value, str(cfg)
        except (OSError, ValueError):
            continue
    return None


_NO_JINJA = object()


def _render(jinja: str, messages: list[dict[str, str]]) -> Any:
    """The Jinja template rendered for ``messages`` (+ generation prompt); ``_NO_JINJA`` when jinja2 is not
    installed, None when the template errors."""
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except Exception:  # noqa: BLE001 - no jinja2: callers rely on the marker match
        return _NO_JINJA

    def _raise(msg: str) -> None:
        raise ValueError(msg)

    try:
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = _raise
        return env.from_string(jinja).render(
            messages=messages, add_generation_prompt=True, bos_token="", eos_token="", tools=None,
        )
    except Exception:  # noqa: BLE001 - an unrenderable template is not converted
        return None


def _render_check(jinja: str, expected_markers: list[str]) -> bool:
    """Render a one-turn chat with the Jinja template and check the turn markers appear in order; a template
    that errors or omits them is not converted (no jinja2: the marker match alone decides)."""
    out = _render(jinja, [{"role": "user", "content": "U1"}])
    if out is _NO_JINJA:
        return True
    if out is None:
        return False
    pos = 0
    for marker in expected_markers:
        found = out.find(marker, pos)
        if found < 0:
            return False
        pos = found + len(marker)
    return True


def _chatml_default_system(jinja: str) -> str | None:
    """The system message the template adds when a chat has none (rendered, so only literal text counts)."""
    out = _render(jinja, [{"role": "user", "content": "U1"}])
    if out is _NO_JINJA:
        for m in _CHATML_DEFAULT_SYSTEM.finditer(jinja):
            text = m.group(1)
            if not any(ch in text for ch in ("'", '"', "+", "{", "}")):
                return text.replace("\\n", "\n")
        return None
    if not isinstance(out, str):
        return None
    m = re.match(r"\s*<\|im_start\|>system\n(.*?)<\|im_end\|>", out, re.S)
    return m.group(1) if m and m.group(1).strip() else None


def ollama_template(jinja: str) -> dict[str, Any]:
    """``{family, template, system?, stop}`` for a recognised turn format, else ``{reason}``."""
    if "<|im_start|>" in jinja and "<|im_end|>" in jinja:
        if not _render_check(jinja, ["<|im_start|>user\nU1<|im_end|>", "<|im_start|>assistant\n"]):
            return {"reason": "a ChatML-style template that did not render a plain two-turn chat as expected"}
        out: dict[str, Any] = {"family": "chatml", "template": CHATML, "stop": ["<|im_end|>"]}
        own = _render(jinja, [{"role": "system", "content": "S0"}, {"role": "user", "content": "U1"}])
        if isinstance(own, str) and "<|im_start|>system\nS0<|im_end|>" not in own:
            return {"reason": "a ChatML-style template that does not render a request's own system message as a system turn"}
        system = _chatml_default_system(jinja)
        if system:
            out["system"] = system
        return out
    if "<|start_header_id|>" in jinja and "<|eot_id|>" in jinja:
        if re.search(r"Cutting Knowledge|Today Date|Environment: ipython", jinja):
            return {"reason": "a Llama 3 template that injects a date / tool preamble Ollama's template cannot reproduce"}
        if not _render_check(jinja, ["<|start_header_id|>user<|end_header_id|>\n\nU1<|eot_id|>",
                                     "<|start_header_id|>assistant<|end_header_id|>\n\n"]):
            return {"reason": "a Llama 3-style template that did not render a plain two-turn chat as expected"}
        return {"family": "llama3", "template": LLAMA3, "stop": ["<|eot_id|>"]}
    if "<start_of_turn>" in jinja and "<end_of_turn>" in jinja:
        if "system" in jinja and "System role not supported" not in jinja:
            return {"reason": "a Gemma-style template that folds the system message into the first turn"}
        if not _render_check(jinja, ["<start_of_turn>user\nU1<end_of_turn>", "<start_of_turn>model\n"]):
            return {"reason": "a Gemma-style template that did not render a plain two-turn chat as expected"}
        return {"family": "gemma", "template": GEMMA, "stop": ["<end_of_turn>"]}
    return {"reason": "a chat template in a format BrewSLM does not convert to Ollama's template syntax"}


def modelfile_template(
    dirs: list[Path], *, use_tokenizer_chat_template: bool = True, chat_wrap: str | None = "tokenizer_chat_template",
) -> dict[str, Any]:
    """The Modelfile's template block: ``{written: True, family, source, lines}`` or ``{written: False,
    reason, note}`` (the note goes into the Modelfile comments and the release notes). ``chat_wrap``: what
    training recorded (``training_report.json`` ``runtime_environment.chat_template_rewrap``); a TEMPLATE is
    written only when it says the rows were wrapped with the tokenizer's chat template."""
    note_tail = (
        "Ollama will use the chat template stored in the GGUF metadata, or its default template for the base "
        "model family; check `ollama show --modelfile <name>` and add a TEMPLATE if prompts come out untemplated."
    )
    if not use_tokenizer_chat_template:
        reason = "the run was trained with its own prompt format (use_tokenizer_chat_template off), not the tokenizer's chat template"
        return {"written": False, "reason": reason, "note": f"No TEMPLATE written: {reason}. {note_tail}"}
    if chat_wrap != "tokenizer_chat_template":
        reason = (
            "the training report does not record how the rows were rendered"
            if not chat_wrap else
            f"training did not wrap the rows with the tokenizer's chat template (recorded: {chat_wrap})"
        )
        return {"written": False, "reason": reason, "note": f"No TEMPLATE written: {reason}. {note_tail}"}
    found = find_chat_template(dirs)
    if found is None:
        reason = "no chat template was saved with the run"
        return {"written": False, "reason": reason, "note": f"No TEMPLATE written: {reason}. {note_tail}"}
    jinja, source = found
    conv = ollama_template(jinja)
    if "template" not in conv:
        reason = f"the tokenizer has {conv['reason']}"
        return {"written": False, "reason": reason, "source": source, "note": f"No TEMPLATE written: {reason}. {note_tail}"}
    if any('"""' in str(conv.get(k) or "") for k in ("template", "system")):
        # A Modelfile triple-quoted value has no escape for three quotes: never write a broken file.
        reason = "the chat template's text holds a triple quote, which a Modelfile value cannot carry"
        return {"written": False, "reason": reason, "source": source, "note": f"No TEMPLATE written: {reason}. {note_tail}"}
    lines = [f'TEMPLATE """{conv["template"]}"""']
    if conv.get("system"):
        lines.append(f'SYSTEM """{conv["system"]}"""')
    lines += [f"PARAMETER stop {json.dumps(s)}" for s in conv["stop"]]
    return {
        "written": True, "family": conv["family"], "source": source,
        "default_system": bool(conv.get("system")), "lines": lines,
    }
