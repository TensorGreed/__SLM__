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

Long texts: an encoder sees ``max_length`` tokens (512 for DeBERTa-v3)
at a time. Training, validation, prediction and evaluation all cut a
longer text into overlapping windows (``encode_windows``: ``stride``
tokens shared between neighbours, default a quarter of the window); each
token's label comes from the window where it sits furthest from an edge
(``merge_window_scores``), so a document of any length is tagged end to
end. The saved tokenizer's ``model_max_length`` is the window size, so
``pipeline(..., stride=N)`` windows the exported model the same way.
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
# Overlap between neighbouring windows, as a fraction of the window.
DEFAULT_STRIDE_FRACTION = 0.25


# ── OCR-style noise (augmentation) ───────────────────────────────────────
#
# Scanned / OCR'd text swaps characters for look-alikes. Each swap keeps
# the text's length, so a noisy copy of a row keeps every span's offsets
# (and a noisy entity is still the same entity). TRAIN_CONFUSIONS is a
# deliberate subset of the look-alikes the measurement uses, so part of
# the evaluated noise is never seen in training.

TRAIN_CONFUSIONS: dict[str, str] = {
    "O": "0", "0": "O", "o": "0", "l": "1", "1": "l", "I": "l", "i": "l",
    "S": "5", "5": "S", "B": "8", "8": "B", "e": "c", "c": "e", "a": "o",
    "n": "m", "m": "n", ".": ",", ",": ".",
}


def ocr_noise(text: str, rate: float, rng: Any, confusions: dict[str, str] | None = None) -> str:
    """``text`` with each swappable character replaced by its look-alike
    with probability ``rate``. Same length, so offsets are unchanged."""
    table = confusions or TRAIN_CONFUSIONS
    return "".join(table[ch] if ch in table and rng.random() < rate else ch for ch in text)


def augment_with_ocr_noise(
    examples: list[tuple[str, list[dict[str, Any]]]], rate: float, copies: int = 1, seed: int = 42,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """``copies`` noisy copies of every example, each at a rate drawn from
    [rate/4, rate] (a spread, not one fixed level); spans unchanged except
    their ``text``, which follows the noised characters."""
    import random as _random

    rng = _random.Random(seed)
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for _ in range(max(0, int(copies))):
        for text, spans in examples:
            noisy = ocr_noise(text, rng.uniform(rate / 4, rate), rng)
            if noisy == text:
                continue
            out.append((noisy, [{**sp, "text": noisy[sp["start"]:sp["end"]]} for sp in spans]))
    return out


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
    span when its characters overlap it; the token holding the span's
    first character is B-, the rest I- — so in a window that starts in the
    middle of a span, the span continues (I-) rather than restarting."""
    labels = [-100] * len(offsets)
    for span in spans:
        for index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if start < span["end"] and end > span["start"]:
                prefix = "B-" if start <= span["start"] else "I-"
                labels[index] = label_to_id.get(f"{prefix}{span['type']}", 0)
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


def window_stride(max_length: int, stride: int | None = None) -> int:
    """Tokens shared by neighbouring windows: ``stride`` when given, else a
    quarter of the window; always below half the window."""
    value = int(stride) if stride is not None else int(max_length * DEFAULT_STRIDE_FRACTION)
    return max(0, min(value, max_length // 2 - 2))


def encode_windows(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    stride: int | None = None,
    padding: bool = False,
    return_tensors: str | None = None,
) -> Any:
    """Tokenize ``texts`` into windows of at most ``max_length`` tokens that
    overlap by ``window_stride`` tokens. A text that fits is one window.
    The result carries ``offset_mapping`` (character offsets into the
    ORIGINAL text, so labels and predictions stay in text coordinates) and
    ``overflow_to_sample_mapping`` (window → text index)."""
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=window_stride(max_length, stride),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
        return_tensors=return_tensors,
    )


def merge_window_scores(
    num_texts: int,
    window_offsets: list[list[tuple[int, int]]],
    sample_map: list[int],
    window_scores: list[Any],
) -> list[tuple[list[tuple[int, int]], list[Any]]]:
    """Per text: its tokens (character offsets, in order) and one score
    vector per token. A token seen by several windows takes the scores
    from the window where it sits furthest from either edge — the one
    that saw the most context on both sides. Special and padding tokens
    (empty offsets) are dropped."""
    best: list[dict[tuple[int, int], tuple[int, Any]]] = [{} for _ in range(num_texts)]
    for offsets, sample, scores in zip(window_offsets, sample_map, window_scores):
        real = [i for i, (a, b) in enumerate(offsets) if b > a]
        if not real:
            continue
        first, last = real[0], real[-1]
        for i in real:
            key = (int(offsets[i][0]), int(offsets[i][1]))
            margin = min(i - first, last - i)
            held = best[sample].get(key)
            if held is None or margin > held[0]:
                best[sample][key] = (margin, scores[i])
    merged: list[tuple[list[tuple[int, int]], list[Any]]] = []
    for per_text in best:
        keys = sorted(per_text)
        merged.append((keys, [per_text[k][1] for k in keys]))
    return merged


def decode_window_predictions(
    texts: list[str],
    window_offsets: list[list[tuple[int, int]]],
    sample_map: list[int],
    window_logits: list[Any],
    id_to_label: dict[int, str],
) -> list[list[dict[str, Any]]]:
    """Merge per-window logits into whole-text spans (argmax per token
    after the merge, then BIO decode)."""
    import numpy as np

    merged = merge_window_scores(len(texts), window_offsets, sample_map, window_logits)
    out: list[list[dict[str, Any]]] = []
    for text, (offsets, scores) in zip(texts, merged):
        ids = [int(np.argmax(np.asarray(v))) for v in scores]
        out.append(token_labels_to_spans(offsets, ids, id_to_label, text))
    return out


def predict_spans(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    id_to_label: dict[int, str],
    *,
    max_length: int = 512,
    stride: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
    with_scores: bool = False,
    thresholds: dict[str, float] | None = None,
    min_score: float | None = None,
    recall_bias: float = 0.0,
    ocr_tolerant: bool = False,
) -> list[list[dict[str, Any]]]:
    """``ocr_tolerant``: tidy the spans for OCR'd text (``ocr_tidy_spans``).

    Character spans for each text with a loaded token-classification
    model. A text longer than ``max_length`` tokens is read in overlapping
    windows (``encode_windows``) and merged (``merge_window_scores``), so
    nothing past the first window is dropped. ``batch_size`` counts
    windows.

    By default every token takes its argmax label (the eval path). With
    ``with_scores``, ``thresholds`` / ``min_score`` or a ``recall_bias``
    the scored decoder runs instead (``decode_scored_spans``): each span
    carries ``score`` / ``min_score`` (its tokens' probability of being
    that type), spans under their type's threshold are dropped, and a
    recall bias lowers the "O" probability so borderline tokens become
    entities ("high-recall mode")."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if not texts:
        return []
    encoded = encode_windows(tokenizer, texts, max_length=max_length, stride=stride)
    window_offsets = [[tuple(o) for o in offs] for offs in encoded["offset_mapping"]]
    sample_map = list(encoded["overflow_to_sample_mapping"])
    window_logits: list[Any] = []
    for start in range(0, len(window_offsets), batch_size):
        features = [
            {"input_ids": encoded["input_ids"][i], "attention_mask": encoded["attention_mask"][i]}
            for i in range(start, min(start + batch_size, len(window_offsets)))
        ]
        batch = tokenizer.pad(features, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.float().cpu().numpy()
        for row, feature in zip(logits, features):
            window_logits.append(row[: len(feature["input_ids"])])
    if with_scores or thresholds or min_score or recall_bias:
        merged = merge_window_scores(len(texts), window_offsets, sample_map, window_logits)
        out = [
            decode_scored_spans(
                text, offsets, scores, id_to_label,
                recall_bias=recall_bias, thresholds=thresholds, min_score=min_score,
            )
            for text, (offsets, scores) in zip(texts, merged)
        ]
    else:
        out = decode_window_predictions(texts, window_offsets, sample_map, window_logits, id_to_label)
    if ocr_tolerant:
        out = [ocr_tidy_spans(text, spans) for text, spans in zip(texts, out)]
    return out


# ── OCR-tolerant decoding ────────────────────────────────────────────────
#
# On OCR'd text a line break is flattened into a space and leaves stray
# marks behind (‘ ~ _ « :), and a value can be broken by them
# ("Sarah - Jenkins", "XJ9LKM2P-QW 3R- T5Y6"). The model then predicts the
# value in pieces, or pulls a stray mark into its edge ("_jsmith@…").
# ``ocr_tidy_spans`` trims stray marks off span edges and joins neighbouring
# spans of the same type separated only by whitespace and stray marks — so a
# predicted span can cross where an OCR line break was. A comma, a letter or
# a digit in the gap keeps two spans apart ("John, Mary" stays two names).

OCR_EDGE_MARKS = "‘’'`~_|«»:;,—–-"
OCR_LEADING_MARKS = OCR_EDGE_MARKS + "."
OCR_GAP_RE = re.compile(r"^[\s‘’'`~_|«»:;—–\-.]*$")
OCR_MAX_GAP = 3


def ocr_tidy_spans(text: str, spans: list[dict[str, Any]], *, max_gap: int = OCR_MAX_GAP) -> list[dict[str, Any]]:
    """Trim stray OCR marks off span edges, then merge same-type neighbours
    whose gap is at most ``max_gap`` characters of whitespace / stray marks
    (never a comma, letter or digit). Pure; returns new span dicts."""
    tidy: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda e: (e["start"], e["end"])):
        start, end = int(span["start"]), int(span["end"])
        while start < end and (text[start].isspace() or text[start] in OCR_LEADING_MARKS):
            start += 1
        while end > start and (text[end - 1].isspace() or text[end - 1] in OCR_EDGE_MARKS):
            end -= 1
        if end <= start:
            continue
        item = {**span, "start": start, "end": end, "text": text[start:end]}
        if tidy:
            prev = tidy[-1]
            gap = text[prev["end"]:start]
            if (
                prev["type"] == item["type"]
                and len(gap) <= max_gap
                and OCR_GAP_RE.match(gap)
                and "," not in gap
            ):
                prev["end"] = end
                prev["text"] = text[prev["start"]:end]
                if "score" in prev and "score" in item:
                    prev["score"] = min(prev["score"], item["score"])
                continue
        tidy.append(item)
    return tidy


def _softmax(values: Any) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()


def decode_scored_spans(
    text: str,
    offsets: list[tuple[int, int]],
    token_logits: list[Any],
    id_to_label: dict[int, str],
    *,
    recall_bias: float = 0.0,
    thresholds: dict[str, float] | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """BIO-decode one text's merged token logits with confidences.

    Per token: softmax probabilities; ``recall_bias`` (0–1) is subtracted
    from the "O" probability before the argmax, so a token whose entity
    probability is within the bias of "O" is tagged. Spans decode exactly
    like ``token_labels_to_spans``; each gets ``score`` (mean over its
    tokens of P(B-type) + P(I-type), unbiased) and ``min_score`` (the
    weakest token). A span whose ``score`` is below ``thresholds[type]``
    (else ``min_score``) is dropped. With bias 0 and no thresholds the
    spans equal the argmax decoder's."""
    import numpy as np

    label_to_id = {label: int(i) for i, label in id_to_label.items()}
    outside_id = label_to_id.get(OUTSIDE)
    probs = [_softmax(v) for v in token_logits]
    ids: list[int] = []
    for p in probs:
        adjusted = np.array(p, copy=True)
        if recall_bias and outside_id is not None and outside_id < len(adjusted):
            adjusted[outside_id] -= float(recall_bias)
        ids.append(int(np.argmax(adjusted)))
    spans = token_labels_to_spans(offsets, ids, id_to_label, text)

    def _type_prob(p: Any, ent_type: str) -> float:
        total = 0.0
        for prefix in ("B-", "I-"):
            idx = label_to_id.get(f"{prefix}{ent_type}")
            if idx is not None and idx < len(p):
                total += float(p[idx])
        return total

    kept: list[dict[str, Any]] = []
    for span in spans:
        token_scores = [
            _type_prob(p, span["type"])
            for (start, end), p, label_id in zip(offsets, probs, ids)
            if end > start and start < span["end"] and end > span["start"]
            and id_to_label.get(label_id, OUTSIDE)[2:] == span["type"]
        ]
        if token_scores:
            span["score"] = round(sum(token_scores) / len(token_scores), 4)
            span["min_score"] = round(min(token_scores), 4)
        else:  # pragma: no cover - a decoded span always has a token
            span["score"] = span["min_score"] = 0.0
        floor = (thresholds or {}).get(span["type"], min_score)
        if floor is not None and span["score"] < float(floor):
            continue
        kept.append(span)
    return kept


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
    noise_rate = float(config.get("ocr_noise_augment_rate") or 0.0)
    noise_copies = int(config.get("ocr_noise_augment_copies") or 1)
    noisy_examples = (
        augment_with_ocr_noise(train_examples, min(noise_rate, 0.3), noise_copies, seed=int(seed))
        if noise_rate > 0 else []
    )
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

    stride = window_stride(max_seq_length, config.get("window_stride"))

    def _encode(examples: list[tuple[str, list[dict[str, Any]]]]) -> tuple[Dataset, list[list[tuple[int, int]]], list[int]]:
        """One training example per window: a text longer than the window
        contributes every window, each labelled from its own offsets."""
        texts = [t for t, _ in examples]
        encoded = encode_windows(tokenizer, texts, max_length=max_seq_length, stride=stride)
        offsets = [[tuple(o) for o in offs] for offs in encoded["offset_mapping"]]
        sample_map = list(encoded["overflow_to_sample_mapping"])
        labels = [
            spans_to_token_labels(offs, examples[sample][1], label_to_id)
            for offs, sample in zip(offsets, sample_map)
        ]
        data = {k: encoded[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in encoded}
        data["labels"] = labels
        return Dataset.from_dict(data), offsets, sample_map

    train_ds, _train_offsets, train_map = _encode(train_examples + noisy_examples)
    # Rows that needed more than one window (not the number of extra windows).
    train_long_rows = sum(1 for n in Counter(train_map).values() if n > 1)
    if eval_examples:
        eval_ds, eval_offsets, eval_map = _encode(eval_examples)
    else:
        eval_ds, eval_offsets, eval_map = None, [], []

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
        _auto_epochs(len(train_examples) + len(noisy_examples), batch_size)
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
        # A float in [0, 1) is a ratio of total steps (transformers 5.x);
        # warmup_ratio itself is gone in newer releases (5.19 rejects it).
        warmup_steps=min(max(warmup_ratio, 0.0), 0.99),
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

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        # Rows of ``logits`` are the eval windows, in order; merge them back
        # into whole texts before scoring, like predict_spans.
        logits, _labels = eval_pred
        window_logits = [np.asarray(row)[: len(offs)] for row, offs in zip(logits, eval_offsets)]
        pred_spans = decode_window_predictions(eval_texts, eval_offsets, eval_map, window_logits, id_to_label)
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
        pred_spans = predict_spans(eval_texts, model, tokenizer, id_to_label, max_length=max_seq_length, stride=stride, device=device)
        final_eval = span_prf(eval_gold, pred_spans)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    # The window size, not the tokenizer's "unlimited" default (DeBERTa-v3
    # ships ~1e30): with it, pipeline(..., stride=N) windows a long text
    # instead of feeding the whole document to an encoder trained on 512.
    tokenizer.model_max_length = max_seq_length
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
            "window_stride": stride,
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
        # Long texts are trained / scored in overlapping windows.
        "augmentation": (
            {"ocr_noise_rate": noise_rate, "copies": noise_copies, "rows_added": len(noisy_examples)}
            if noise_rate > 0 else None
        ),
        "windowing": {
            "max_length": max_seq_length,
            "stride": stride,
            "train_windows": len(train_map),
            "train_long_rows": train_long_rows,
            "eval_windows": len(eval_map),
        },
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
