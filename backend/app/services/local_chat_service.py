"""In-process chat with a trained run's checkpoint (Playground "experiment"
provider).

Before this, chatting with a model you just trained meant exporting it,
starting an external server and typing its URL into the Playground. This
loads the run's weights directly — a LoRA adapter on its base model, or a
full fine-tune — keeps one model cached (GPU memory), and renders the
conversation with the tokenizer's chat template: the same prompt shape the
trainer and held-out eval use.

Blocking model work runs in a worker thread; a lock serialises generation
so concurrent requests can't interleave on one model.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

from app.services.adapter_merge_service import adapter_base_model, read_adapter_config

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"key": None, "bundle": None}


class _Bundle:
    def __init__(self, model: Any, tokenizer: Any, kind: str, device: str, labels: dict | None):
        self.model = model
        self.tokenizer = tokenizer
        self.kind = kind  # "causal_lm" | "classifier"
        self.device = device
        self.labels = labels or {}


def _cache_key(model_ref: str) -> tuple[str, float]:
    path = Path(model_ref)
    try:
        mtime = max(p.stat().st_mtime for p in path.iterdir()) if path.is_dir() else 0.0
    except ValueError:
        mtime = 0.0
    return str(path.resolve()), mtime


def _is_classifier(model_ref: str, adapter_cfg: dict | None) -> bool:
    if adapter_cfg is not None:
        return str(adapter_cfg.get("task_type") or "").upper() == "SEQ_CLS"
    config_path = Path(model_ref) / "config.json"
    if not config_path.is_file():
        return False
    import json

    try:
        archs = json.loads(config_path.read_text(encoding="utf-8")).get("architectures") or []
    except Exception:  # noqa: BLE001
        return False
    return any(str(a).endswith("ForSequenceClassification") for a in archs)


def _load_bundle(model_ref: str, base_model_hint: str | None) -> _Bundle:
    key = _cache_key(model_ref)
    if _CACHE["key"] == key and _CACHE["bundle"] is not None:
        return _CACHE["bundle"]

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    # One resident model at a time.
    _CACHE["key"], _CACHE["bundle"] = None, None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_cfg = read_adapter_config(model_ref)
    classifier = _is_classifier(model_ref, adapter_cfg)
    model_cls = AutoModelForSequenceClassification if classifier else AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # On CPU load in fp32: transformers >= 5 would keep the checkpoint's saved
    # dtype (bf16 for most small instruct models), which is emulated — and
    # very slow — on CPUs without native bf16.
    load_kwargs: dict[str, Any] = {} if device == "cuda" else {"dtype": torch.float32}

    if adapter_cfg is not None:
        from peft import PeftModel

        base = adapter_base_model(model_ref, fallback=base_model_hint)
        if not base:
            raise ValueError(f"LoRA adapter at {model_ref} doesn't record its base model.")
        model = PeftModel.from_pretrained(model_cls.from_pretrained(base, **load_kwargs), model_ref)
        tokenizer_source = model_ref if (Path(model_ref) / "tokenizer_config.json").exists() else base
    else:
        model = model_cls.from_pretrained(model_ref, **load_kwargs)
        tokenizer_source = model_ref
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device).eval()

    labels = None
    if classifier:
        config = getattr(model, "config", None)
        labels = dict(getattr(config, "id2label", {}) or {})
    bundle = _Bundle(model, tokenizer, "classifier" if classifier else "causal_lm", device, labels)
    _CACHE["key"], _CACHE["bundle"] = key, bundle
    return bundle


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """The prompt the run was trained on: the tokenizer's chat template over
    the whole conversation; without a template, the last user turn + newline
    (the trainer's raw-prompt fallback)."""
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    return f"{last_user}\n"


def _generation_kwargs(bundle: _Bundle, max_tokens: int, temperature: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_tokens),
        "pad_token_id": bundle.tokenizer.pad_token_id,
    }
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature))
    else:
        kwargs["do_sample"] = False
    return kwargs


def _classify(bundle: _Bundle, messages: list[dict[str, str]]) -> dict[str, Any]:
    import torch

    text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    enc = bundle.tokenizer(text, return_tensors="pt", truncation=True).to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(**enc).logits[0]
    probs = torch.softmax(logits.float(), dim=-1)
    idx = int(probs.argmax())
    label = bundle.labels.get(idx, bundle.labels.get(str(idx), str(idx)))
    return {
        "reply": f"{label}",
        "usage": {"prompt_tokens": int(enc["input_ids"].shape[-1]), "completion_tokens": 0},
        "finish_reason": "classification",
        "confidence": round(float(probs[idx]), 4),
    }


def generate_reply(
    model_ref: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.2,
    base_model_hint: str | None = None,
) -> dict[str, Any]:
    """Blocking single reply. Call via ``asyncio.to_thread``."""
    import torch

    with _LOCK:
        bundle = _load_bundle(model_ref, base_model_hint)
        if bundle.kind == "classifier":
            return _classify(bundle, messages)
        prompt = render_chat_prompt(bundle.tokenizer, messages)
        enc = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
        with torch.no_grad():
            out = bundle.model.generate(**enc, **_generation_kwargs(bundle, max_tokens, temperature))
        new_tokens = out[0][enc["input_ids"].shape[-1]:]
        reply = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        stopped = bundle.tokenizer.eos_token_id in new_tokens.tolist()
        return {
            "reply": reply,
            "usage": {
                "prompt_tokens": int(enc["input_ids"].shape[-1]),
                "completion_tokens": int(new_tokens.shape[-1]),
            },
            "finish_reason": "stop" if stopped else "length",
        }


async def astream_reply(
    model_ref: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.2,
    base_model_hint: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield ``{"type": "delta", "content"}`` events, then one
    ``{"type": "final", ...}``, generating in a worker thread."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    started = perf_counter()

    def _put(item: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def _work() -> None:
        try:
            import torch
            from transformers import TextIteratorStreamer

            with _LOCK:
                bundle = _load_bundle(model_ref, base_model_hint)
                if bundle.kind == "classifier":
                    result = _classify(bundle, messages)
                    _put(("delta", result["reply"]))
                    _put(("final", result))
                    return
                prompt = render_chat_prompt(bundle.tokenizer, messages)
                enc = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
                streamer = TextIteratorStreamer(
                    bundle.tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                kwargs = _generation_kwargs(bundle, max_tokens, temperature)
                holder: dict[str, Any] = {}

                def _gen() -> None:
                    with torch.no_grad():
                        holder["out"] = bundle.model.generate(**enc, streamer=streamer, **kwargs)

                gen_thread = threading.Thread(target=_gen, daemon=True)
                gen_thread.start()
                pieces: list[str] = []
                for text in streamer:
                    if text:
                        pieces.append(text)
                        _put(("delta", text))
                gen_thread.join()
                new_tokens = holder["out"][0][enc["input_ids"].shape[-1]:]
                stopped = bundle.tokenizer.eos_token_id in new_tokens.tolist()
                _put((
                    "final",
                    {
                        "reply": "".join(pieces).strip(),
                        "usage": {
                            "prompt_tokens": int(enc["input_ids"].shape[-1]),
                            "completion_tokens": int(new_tokens.shape[-1]),
                        },
                        "finish_reason": "stop" if stopped else "length",
                    },
                ))
        except Exception as exc:  # noqa: BLE001
            _put(("error", exc))

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    while True:
        kind, payload = await queue.get()
        if kind == "delta":
            yield {"type": "delta", "content": payload}
        elif kind == "error":
            raise ValueError(f"Local model chat failed: {payload}") from payload
        else:
            yield {
                "type": "final",
                **payload,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
            }
            return


def unload() -> None:
    """Drop the cached model (tests / memory pressure)."""
    with _LOCK:
        _CACHE["key"], _CACHE["bundle"] = None, None
