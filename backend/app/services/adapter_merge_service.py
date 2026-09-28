"""Detect PEFT/LoRA adapter directories and merge them into full models.

LoRA training saves only the adapter (``adapter_config.json`` + adapter
weights) under ``<run>/model``. Anything downstream that expects a
standalone model — HF/Docker export, vLLM/TGI serving, GGUF/ONNX
conversion — needs the adapter merged into its base first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_adapter_config(model_dir: str | Path | None) -> dict[str, Any] | None:
    """The adapter's ``adapter_config.json`` when ``model_dir`` is a PEFT
    adapter directory, else ``None``."""
    if not model_dir:
        return None
    path = Path(model_dir) / "adapter_config.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def adapter_base_model(model_dir: str | Path | None, fallback: str | None = None) -> str | None:
    cfg = read_adapter_config(model_dir)
    if cfg is None:
        return None
    base = str(cfg.get("base_model_name_or_path") or "").strip()
    return base or (fallback or None)


def _auto_model_class(task_type: str):
    import transformers

    task = (task_type or "").upper()
    if task == "SEQ_CLS":
        return transformers.AutoModelForSequenceClassification
    if task == "SEQ_2_SEQ_LM":
        return transformers.AutoModelForSeq2SeqLM
    return transformers.AutoModelForCausalLM


def merge_adapter(
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    base_model: str | None = None,
) -> dict[str, Any]:
    """Merge ``adapter_dir`` into its base model and save a standalone
    model (safetensors + tokenizer) to ``out_dir``. Blocking — call via
    ``asyncio.to_thread`` from async code. Raises with an actionable
    message on failure."""
    adapter_path = Path(adapter_dir)
    cfg = read_adapter_config(adapter_path)
    if cfg is None:
        raise ValueError(f"{adapter_path} is not a LoRA adapter directory (no adapter_config.json).")
    base = adapter_base_model(adapter_path, fallback=base_model)
    if not base:
        raise ValueError(
            f"Adapter at {adapter_path} doesn't record its base model; "
            "re-run the export with the experiment's base model available."
        )
    try:
        from peft import PeftModel
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Merging a LoRA adapter needs torch + transformers + peft in the backend environment."
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    model_cls = _auto_model_class(str(cfg.get("task_type") or ""))
    # Merge in fp32 (W + BA in bf16 loses the small LoRA delta to
    # rounding), then save in the checkpoint's own dtype so the export
    # isn't twice the size.
    from transformers import AutoConfig

    # Read the storage dtype before the fp32 load overwrites config.dtype.
    base_config = AutoConfig.from_pretrained(base, trust_remote_code=True)
    saved_dtype = getattr(base_config, "dtype", None) or getattr(base_config, "torch_dtype", None)
    base_model_obj = model_cls.from_pretrained(base, trust_remote_code=True, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base_model_obj, str(adapter_path)).merge_and_unload()
    if isinstance(saved_dtype, str):
        saved_dtype = getattr(torch, saved_dtype, None)
    if isinstance(saved_dtype, torch.dtype) and saved_dtype != torch.float32:
        merged = merged.to(saved_dtype)
    merged.save_pretrained(str(out), safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:  # noqa: BLE001
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.save_pretrained(str(out))

    files = [p for p in out.rglob("*") if p.is_file()]
    return {
        "base_model": base,
        "adapter_dir": str(adapter_path),
        "task_type": cfg.get("task_type"),
        "merged_dir": str(out),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }
