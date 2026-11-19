"""Golden fingerprints of training and eval prompt rendering for rows WITHOUT ``row_context``.

``tests/test_row_context_golden.py`` asserts these hashes equal the ones recorded from the code
BEFORE per-row context existed (base commit df7ffea), so rows without ``row_context`` render
byte-identically after it. Pure functions only (no DB, no model): the training side is the data prep
(``_normalize_rows_for_training``) plus ``scripts/train.py``'s re-wrap; the eval side is
``_load_heldout_pairs`` plus the text one pair is generated from.

Regenerate against an older tree: ``python tests/_row_context_golden.py`` with that tree's backend on
``sys.path`` prints the hashes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TOOL_BLOCK = (
    "Available tools (call only these):\n- get_item(id: string) — Fetch one item. [read-only]\n"
    "- list_items() — List items. [read-only]"
)

CASES: dict[str, dict[str, Any]] = {
    "qa": {
        "adapter": "default-canonical", "profile": "qa", "config": {},
        "rows": [{"question": "What is the capital of France?", "answer": "Paris"},
                 {"question": "Name a prime number.", "answer": "7"}],
    },
    "classification": {
        "adapter": "classification-label", "profile": "classification", "config": {"candidates": ["neg", "pos"]},
        "rows": [{"text": "I loved it", "label": "pos"}, {"text": "Terrible service", "label": "neg"}],
    },
    "tool_calling": {
        "adapter": "tool-call-json", "profile": "tool_calling", "config": {"tool_block": TOOL_BLOCK},
        "rows": [{"prompt": "Get item 7", "tool_name": "get_item", "tool_args": {"id": "7"}, "result": "ok"},
                 {"prompt": "List everything", "tool_name": "list_items", "tool_args": {}, "result": "ok"}],
    },
    "tool_calling_no_catalog": {
        "adapter": "tool-call-json", "profile": "tool_calling", "config": {},
        "rows": [{"prompt": "Get item 7", "tool_name": "get_item", "tool_args": {"id": "7"}, "result": "ok"}],
    },
    "extraction": {
        "adapter": "structured-extraction", "profile": "structured_extraction", "config": {},
        "rows": [{"text": "Invoice 12 from ACME, total 40", "json": {"vendor": "ACME", "total": 40}}],
    },
    "chat": {
        "adapter": "chat-messages", "profile": "qa", "config": {},
        "rows": [{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                               {"role": "assistant", "content": "Hello."}]}],
    },
}


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ANN001, ARG002
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


class NoTemplateTokenizer:
    chat_template = None


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def _train_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    if "brewslm_train_golden" in sys.modules:
        return sys.modules["brewslm_train_golden"]
    spec = importlib.util.spec_from_file_location("brewslm_train_golden", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["brewslm_train_golden"] = mod
    spec.loader.exec_module(mod)
    return mod


def training_render(case: dict[str, Any]) -> list[dict[str, Any]]:
    from app.models.dataset import DatasetType
    from app.services.dataset_service import _normalize_rows_for_training

    return _normalize_rows_for_training(
        [dict(r) for r in case["rows"]], DatasetType.RAW, "llama3", adapter_id=case["adapter"],
        adapter_config=dict(case["config"]), task_profile=case["profile"],
    )


def train_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """What ``scripts/train.py`` trains on for prepared ``rows`` (causal LM): its record adapter (the
    ``system_prompt`` column rides along when the split has one) then the prompt->completion re-wrap."""
    train = _train_module()
    contract = train._build_data_adapter_contract("causal_lm", "llama3")
    carries_system = any("system_prompt" in r for r in rows)
    out = []
    for row in rows:
        mapped = train._adapt_record_to_text(dict(row), contract, "llama3")
        if carries_system:
            mapped["system_prompt"] = str(row.get("system_prompt") or "")
        out.append(train._rewrap_prompt_completion_row(mapped, tokenizer))
    return out


def train_rewrap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``train_rows`` with and without a chat template."""
    return train_rows(rows, FakeTokenizer()) + train_rows(rows, NoTemplateTokenizer())


def eval_render(case: dict[str, Any], rows: list[dict[str, Any]], tmp: Path) -> list[Any]:
    from app.services import evaluation_service as es
    from app.services.eval_task_handler_service import EvalContext, handler_wraps_own_prompt, resolve_task_handler

    path = tmp / f"{case['profile']}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    handler = resolve_task_handler(case["profile"])
    manifest = {"task_profile": case["profile"]}
    if case["profile"] == "tool_calling":
        manifest["tool_block"] = case["config"].get("tool_block", "")
    ctx = EvalContext(project_id=0, experiment_id=0, eval_type="exact_match", task_profile=case["profile"],
                      handler_id=handler.profile_id, prepared_dir=tmp, dataset_name="test", manifest=manifest)
    pairs = es._load_heldout_pairs(path, 100, handler=handler, ctx=ctx)
    apply = not handler_wraps_own_prompt(handler, manifest)
    fmt = getattr(es, "_format_eval_prompt", None)
    texts = []
    for pair in pairs:
        for tok in (FakeTokenizer(), NoTemplateTokenizer()):
            if fmt is not None:
                texts.append(fmt(tok, pair, apply)[0])
            elif apply:
                texts.append(es._apply_chat_template_if_present(tok, pair["prompt"], pair.get("history"))[0])
            else:
                texts.append(pair["prompt"])
    return [pairs, texts]


def fingerprints(tmp: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name, case in CASES.items():
        prepared = training_render(case)
        out[name] = {
            "prepared": _sha(prepared),
            "train_text": _sha(train_rewrap(prepared)),
            "eval": _sha(eval_render(case, prepared, tmp)),
        }
    return out


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(fingerprints(Path(d)), indent=2, sort_keys=True))
