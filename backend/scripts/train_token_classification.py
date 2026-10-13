"""Token classification for span extraction: exact character offsets.

A span-extraction project (``{text, entities: [{type, start, end, text}]}``
rows — PII / PCI / PHI detection, NER, any "find and label pieces of
text") can be trained two ways. The generative way (the default
``span-extraction`` recipe) has a causal LM *write* the entities as JSON,
offsets included — a small model gets the offsets wrong, because it has
to count characters. This mode trains an encoder with a token-
classification head (``AutoModelForTokenClassification``, e.g.
DeBERTa-v3) on BIO labels derived from the character spans, so at
inference the offsets come from the tokenizer's offset mapping — exact
by construction — and the result is a plain Hugging Face model:
``pipeline("token-classification", model=<dir>, aggregation_strategy="simple")``
returns ``{entity_group, start, end, word, score}``.

Invoked by ``train.py`` when ``task_type == "token_classification"``.
Writes the same artifacts the platform expects from a run: ``model/``
(full model + tokenizer + ``label_space.json``), ``metrics.jsonl``
(per-epoch span P/R/F1, the same strict (type, start, end) match the
``span_set`` evaluation scorer uses) and a ``training_report.json``-shaped
dict returned to the caller.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Keys a span row may carry (synthetic rows, prepared rows, imports).
TEXT_KEYS = ("text", "input", "source_text", "content", "document", "passage")
ENTITY_KEYS = ("entities", "entities_json", "expected", "structured_output", "target", "target_text", "labels", "output")
OUTSIDE = "O"


# ── Rows → (text, spans) ─────────────────────────────────────────────────


def _parse_entities(value: Any) -> list[dict[str, Any]]:
    """``[{type,start,end,text}]`` from a list, a dict with ``entities``,
    or a JSON string of either."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = value.get("entities", [])
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        ent_type = str(item.get("type") or item.get("label") or "").strip()
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not ent_type or end <= start:
            continue
        out.append({"type": ent_type, "start": start, "end": end, "text": str(item.get("text") or "")})
    return out


def row_to_example(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    """``(text, spans)`` with spans validated against the text — an entity
    whose ``text`` doesn't match ``text[start:end]`` (or runs off the end)
    is dropped, never trained on. None when the row has no text."""
    text = None
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("text") or value.get("input")
        if isinstance(value, str) and value.strip():
            text = value
            break
    if text is None:
        return None
    spans: list[dict[str, Any]] = []
    for key in ENTITY_KEYS:
        if key in row:
            spans = _parse_entities(row.get(key))
            if spans or key in ("entities", "entities_json"):
                break
    valid: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        if span["end"] > len(text):
            continue
        if span["text"] and text[span["start"]:span["end"]] != span["text"]:
            continue
        # Overlaps are not representable in BIO: keep the first.
        if valid and span["start"] < valid[-1]["end"]:
            continue
        valid.append(span)
    return text, valid


def load_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_label_space(examples: list[tuple[str, list[dict[str, Any]]]]) -> list[str]:
    """``["O", "B-type", "I-type", …]`` over every type seen, sorted."""
    types = sorted({span["type"] for _, spans in examples for span in spans})
    labels = [OUTSIDE]
    for t in types:
        labels.extend([f"B-{t}", f"I-{t}"])
    return labels


# ── Char spans ↔ token labels ────────────────────────────────────────────


def spans_to_token_labels(
    offsets: list[tuple[int, int]],
    spans: list[dict[str, Any]],
    label_to_id: dict[str, int],
) -> list[int]:
    """BIO label id per token from its character offsets. Special tokens
    (offset (0, 0)) get -100 (ignored by the loss). A token is inside a
    span when its characters overlap it; the first such token is B-."""
    labels = [-100] * len(offsets)
    for span in spans:
        first = True
        for index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if start < span["end"] and end > span["start"]:
                prefix = "B-" if first else "I-"
                labels[index] = label_to_id.get(f"{prefix}{span['type']}", 0)
                first = False
    for index, (start, end) in enumerate(offsets):
        if end > start and labels[index] == -100:
            labels[index] = label_to_id[OUTSIDE]
    return labels


def token_labels_to_spans(
    offsets: list[tuple[int, int]],
    label_ids: list[int],
    id_to_label: dict[int, str],
    text: str,
) -> list[dict[str, Any]]:
    """Decode BIO predictions into character spans. A stray I- without a
    B- starts a span (lenient, like the HF pipeline's "simple" strategy);
    a type change closes the span."""
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for (start, end), label_id in zip(offsets, label_ids):
        if end <= start:
            continue
        label = id_to_label.get(int(label_id), OUTSIDE)
        if label == OUTSIDE:
            if current:
                spans.append(current)
                current = None
            continue
        prefix, ent_type = label[:2], label[2:]
        if current and prefix == "I-" and current["type"] == ent_type:
            current["end"] = end
            continue
        if current:
            spans.append(current)
        current = {"type": ent_type, "start": start, "end": end}
    if current:
        spans.append(current)
    for span in spans:
        # Trim whitespace the tokenizer attached to the span's edges.
        while span["start"] < span["end"] and text[span["start"]].isspace():
            span["start"] += 1
        while span["end"] > span["start"] and text[span["end"] - 1].isspace():
            span["end"] -= 1
        span["text"] = text[span["start"]:span["end"]]
    return [s for s in spans if s["end"] > s["start"]]


def span_prf(gold: list[list[dict[str, Any]]], pred: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Strict (type, start, end) precision / recall / F1, micro over rows,
    plus per-type counts — the same match the span_set scorer uses."""
    tp = fp = fn = 0
    per_type: dict[str, dict[str, int]] = {}
    for g_spans, p_spans in zip(gold, pred):
        g = Counter((s["type"], s["start"], s["end"]) for s in g_spans)
        p = Counter((s["type"], s["start"], s["end"]) for s in p_spans)
        common = g & p
        tp += sum(common.values())
        fp += sum(p.values()) - sum(common.values())
        fn += sum(g.values()) - sum(common.values())
        for t in {k[0] for k in g} | {k[0] for k in p}:
            tg = Counter(k for k in g.elements() if k[0] == t)
            tp_ = Counter(k for k in p.elements() if k[0] == t)
            c = tg & tp_
            slot = per_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
            slot["tp"] += sum(c.values())
            slot["fp"] += sum(tp_.values()) - sum(c.values())
            slot["fn"] += sum(tg.values()) - sum(c.values())

    def _prf(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
        precision = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        recall = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_type": {
            t: dict(zip(("precision", "recall", "f1"), _prf(c["tp"], c["fp"], c["fn"])), **c) for t, c in sorted(per_type.items())
        },
    }


# ── Inference helper (shared with the evaluation service) ───────────────


def predict_spans(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    id_to_label: dict[int, str],
    *,
    max_length: int = 512,
    batch_size: int = 16,
    device: str | None = None,
) -> list[list[dict[str, Any]]]:
    """Character spans for each text with a loaded token-classification
    model: tokenize with offsets (long texts in overlapping windows are a
    later refinement — this truncates at ``max_length`` tokens), argmax,
    decode BIO → spans."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    results: list[list[dict[str, Any]]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, truncation=True, max_length=max_length, padding=True,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        for text, offs, ids in zip(batch, offsets, pred_ids):
            results.append(token_labels_to_spans([tuple(o) for o in offs], ids, id_to_label, text))
    return results


# ── Training ─────────────────────────────────────────────────────────────


def run_token_classification(
    *,
    base_model: str,
    config: dict[str, Any],
    output_dir: Path,
    model_dir: Path,
    train_file: Path,
    val_file: Path | None,
    started_at: str,
    config_path: Path | None = None,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    seed: int = 42,
    emit_metric: Any = None,
) -> dict[str, Any]:
    """Fine-tune ``base_model`` with a token-classification head on the
    span rows in ``train_file`` and return the report dict."""
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(int(seed))
    warnings: list[str] = []
    train_rows = load_rows(train_file)
    eval_rows = load_rows(val_file)
    if max_train_samples:
        train_rows = train_rows[:max_train_samples]
    if max_eval_samples:
        eval_rows = eval_rows[:max_eval_samples]
    train_examples = [ex for ex in (row_to_example(r) for r in train_rows) if ex is not None]
    eval_examples = [ex for ex in (row_to_example(r) for r in eval_rows) if ex is not None]
    if not train_examples:
        raise RuntimeError(
            "token_classification needs rows shaped {text, entities: [{type, start, end, text}]} — "
            f"none of the {len(train_rows)} training rows had text."
        )
    labelled = sum(1 for _, spans in train_examples if spans)
    if labelled == 0:
        raise RuntimeError("token_classification: no training row carries an entity span — nothing to learn.")
    label_space = build_label_space(train_examples + eval_examples)
    label_to_id = {label: i for i, label in enumerate(label_space)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"token_classification needs a fast tokenizer for character offsets; {base_model} has none."
        )
    max_seq_length = int(config.get("max_seq_length") or 512)
    max_seq_length = max(64, min(max_seq_length, int(getattr(tokenizer, "model_max_length", 512) or 512), 4096))

    def _encode(examples: list[tuple[str, list[dict[str, Any]]]]) -> Dataset:
        texts = [t for t, _ in examples]
        encoded = tokenizer(texts, truncation=True, max_length=max_seq_length, return_offsets_mapping=True)
        labels = [
            spans_to_token_labels([tuple(o) for o in offs], spans, label_to_id)
            for offs, (_, spans) in zip(encoded["offset_mapping"], examples)
        ]
        encoded.pop("offset_mapping")
        data = {k: encoded[k] for k in encoded.keys()}
        data["labels"] = labels
        return Dataset.from_dict(data)

    train_ds = _encode(train_examples)
    eval_ds = _encode(eval_examples) if eval_examples else None

    use_cuda = torch.cuda.is_available()
    dtype_note = "bf16" if use_cuda and torch.cuda.is_bf16_supported() else ("fp16" if use_cuda else "fp32")
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(label_space), id2label=id_to_label, label2id=label_to_id,
        dtype=torch.float32,
    )

    batch_size = int(config.get("batch_size") or 16)
    # ``auto_epochs`` (the platform default) sizes epochs to the row count so
    # a freshly initialised head on dozens of BIO labels gets enough optimizer
    # steps; an explicit num_epochs only counts when auto_epochs is off.
    auto_epochs = _coerce_bool(config.get("auto_epochs"), True)
    num_epochs = (
        _auto_epochs(len(train_examples), batch_size)
        if auto_epochs or not config.get("num_epochs")
        else float(config.get("num_epochs"))
    )
    learning_rate = float(config.get("learning_rate") or 5e-5)
    warmup_ratio = float(config.get("warmup_ratio") or 0.06)
    eval_strategy = "epoch" if eval_ds is not None else "no"
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size, 16),
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=float(config.get("weight_decay") or 0.01),
        eval_strategy=eval_strategy,
        save_strategy="no",
        logging_steps=max(1, int(config.get("logging_steps") or 10)),
        bf16=dtype_note == "bf16",
        fp16=dtype_note == "fp16",
        report_to=[],
        seed=int(seed),
        dataloader_pin_memory=use_cuda,
    )

    eval_texts = [t for t, _ in eval_examples]
    eval_gold = [spans for _, spans in eval_examples]
    eval_offsets = (
        [[tuple(o) for o in offs] for offs in tokenizer(eval_texts, truncation=True, max_length=max_seq_length, return_offsets_mapping=True)["offset_mapping"]]
        if eval_examples else []
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, _labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)
        pred_spans = [
            token_labels_to_spans(offs, ids[: len(offs)].tolist(), id_to_label, text)
            for offs, ids, text in zip(eval_offsets, pred_ids, eval_texts)
        ]
        prf = span_prf(eval_gold, pred_spans)
        return {"span_precision": prf["precision"], "span_recall": prf["recall"], "span_f1": prf["f1"]}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )
    train_result = trainer.train()

    # Per-epoch metrics the platform's charts read (metrics.jsonl).
    metrics_path = output_dir / "metrics.jsonl"
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    with metrics_path.open("w", encoding="utf-8") as handle:
        for entry in log_history:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if emit_metric is not None:
                emit_metric(entry)

    # Final held-out span scores (strict match) + per-type breakdown.
    final_eval: dict[str, Any] = {}
    if eval_ds is not None:
        device = "cuda" if use_cuda else "cpu"
        model.to(device)
        pred_spans = predict_spans(eval_texts, model, tokenizer, id_to_label, max_length=max_seq_length, device=device)
        final_eval = span_prf(eval_gold, pred_spans)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "label_space.json").write_text(
        json.dumps({"labels": label_space, "entity_types": sorted({l[2:] for l in label_space if l != OUTSIDE})}, indent=2),
        encoding="utf-8",
    )

    final_train_loss = next((e.get("loss") for e in reversed(log_history) if "loss" in e), None)
    final_eval_loss = next((e.get("eval_loss") for e in reversed(log_history) if "eval_loss" in e), None)
    from datetime import datetime, timezone

    return {
        "status": "completed",
        "task_type": "token_classification",
        "training_mode": "token_classification",
        "base_model": base_model,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path else None,
        "effective_config": {
            **dict(config),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
        },
        "train_file": str(train_file),
        "val_file": str(val_file) if val_file is not None else None,
        "model_dir": str(model_dir),
        "metrics_path": str(metrics_path),
        "epochs": num_epochs,
        "total_steps": int(getattr(trainer.state, "global_step", 0) or 0),
        "train_records": len(train_examples),
        "eval_records": len(eval_examples),
        "train_records_labelled": labelled,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "final_eval_accuracy": None,
        "best_eval_loss": final_eval_loss,
        "task_metrics": {"span_set": final_eval} if final_eval else {},
        "checkpoints": [],
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "label_space": label_space,
        "entity_types": sorted({l[2:] for l in label_space if l != OUTSIDE}),
        "runtime_environment": {
            "model_loader": "AutoModelForTokenClassification",
            "device": "cuda" if use_cuda else "cpu",
            "dtype": dtype_note,
            "fast_tokenizer": True,
            "label_space_size": len(label_space),
        },
        "warnings": warnings,
        "attempt_index": 1,
    }


# A token-classification head starts from random weights over 2·types + 1
# labels; it needs on the order of a few hundred optimizer steps whatever
# the row count. Target ~400 steps, within 5–40 epochs.
TARGET_OPTIMIZER_STEPS = 400
MIN_EPOCHS = 5
MAX_EPOCHS = 40


def _auto_epochs(rows: int, batch_size: int = 16) -> float:
    steps_per_epoch = max(1, math.ceil(max(1, rows) / max(1, batch_size)))
    return float(max(MIN_EPOCHS, min(MAX_EPOCHS, math.ceil(TARGET_OPTIMIZER_STEPS / steps_per_epoch))))


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
