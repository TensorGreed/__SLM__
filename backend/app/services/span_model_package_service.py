"""Packaging for token-classification (span) models — Epic P5.

An exported span tagger should be usable without BrewSLM: this writes, into
the exported ``model/`` directory,

* ``README.md`` — a Hugging Face model card built ONLY from evidence the
  platform holds (label space, the run's training report, the latest span
  EvalResults; answer key first, test examples second). A section without
  data is left out rather than filled with a guess.
* ``thresholds.json`` — per-type minimum span scores + the recall bias the
  scored decoder (``train_token_classification.decode_scored_spans``) and
  the redact endpoint read. Defaults keep the argmax behaviour.

``build_span_model_card`` is pure; ``package_token_classifier`` gathers the
inputs and writes both files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment

THRESHOLDS_FILE = "thresholds.json"
MODEL_CARD_FILE = "README.md"
# Types with fewer gold spans than this get a "noisy score" note.
LOW_SUPPORT_SPANS = 10
MAX_SOURCES_LISTED = 8
# Below this many texts an evaluation is called small (noisy) in the card.
SMALL_EVAL_TEXTS = 200
_SPAN_DATASETS = (("gold_test", "answer key"), ("test", "test examples"))


def pipeline_stride(window: int | None) -> int:
    """The ``stride`` to pass the HF pipeline for this window: a quarter of
    it (128 for a 512-token window), always below half the window — the
    same overlap training and eval use (``window_stride``)."""
    if not window:
        return 128
    return max(0, min(int(window) // 4, int(window) // 2 - 2))


def default_thresholds(entity_types: list[str]) -> dict[str, Any]:
    """thresholds.json defaults: every type at 0.0 and no recall bias — the
    same spans as the plain argmax decoder. Raise a type's threshold to
    trade its recall for precision; raise ``recall_bias`` (0–1) to tag
    borderline tokens ("high-recall mode")."""
    return {
        "version": 1,
        "score": "mean over a span's tokens of P(B-type) + P(I-type)",
        "default_min_score": 0.0,
        "recall_bias": 0.0,
        "per_type": {t: 0.0 for t in sorted(entity_types)},
    }


def read_thresholds(model_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(model_dir) / THRESHOLDS_FILE).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _fmt(value: Any) -> str:
    return f"{float(value):.2f}" if isinstance(value, (int, float)) else "–"


def _is_hub_id(name: str | None) -> bool:
    if not name:
        return False
    return not (name.startswith(("/", ".", "~")) or "\\" in name) and name.count("/") <= 1


def build_span_model_card(
    *,
    entity_types: list[str],
    base_model: str | None = None,
    model_name: str | None = None,
    experiment_id: int | None = None,
    project_name: str | None = None,
    window: int | None = None,
    training: dict[str, Any] | None = None,
    evaluations: list[dict[str, Any]] | None = None,
    thresholds_file: bool = True,
) -> str:
    """Markdown model card (with HF front matter) for a span tagger.

    ``training``: ``{train_records, train_records_labelled, eval_records,
    epochs, sources: {name: rows}, included_types: [...], augmentation}``
    — any key may be missing. ``evaluations``: one block per scored split,
    in the order to show (answer key first): ``{label, dataset_name,
    precision, recall, f1, rows, per_type: {type: {precision, recall, f1,
    tp, fp, fn}}, written_by}``. Numbers are copied, never derived beyond
    a gold-span count (tp + fn); a section with no data is omitted."""
    training = training or {}
    evaluations = [e for e in (evaluations or []) if isinstance(e, dict) and e.get("f1") is not None]
    types = sorted({str(t) for t in entity_types if t})
    stride = pipeline_stride(window)
    lines: list[str] = []

    front = ["---", "library_name: transformers", "pipeline_tag: token-classification"]
    if _is_hub_id(base_model):
        front.append(f"base_model: {base_model}")
    front += ["tags:", "- token-classification", "- span-extraction", "- brewslm", "---", ""]
    lines += front

    title = model_name or (f"{project_name} span tagger" if project_name else "Span tagger")
    lines.append(f"# {title}")
    lines.append("")
    intro = "A token-classification model that finds spans of text and labels each with a type"
    lines.append(intro + (f", fine-tuned from `{base_model}`." if base_model else "."))
    provenance = []
    if project_name:
        provenance.append(f"project “{project_name}”")
    if experiment_id is not None:
        provenance.append(f"run #{experiment_id}")
    if provenance:
        lines.append(f"Trained with BrewSLM ({', '.join(provenance)}).")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    if base_model:
        lines.append(f"- **Base model:** `{base_model}`")
    lines.append("- **Task:** token classification (BIO tags → character spans)")
    if window:
        lines.append(f"- **Window:** {window} tokens (the tokenizer's `model_max_length`); longer texts are read in overlapping windows")
    lines.append("")

    if types:
        lines.append("## Entity types")
        lines.append("")
        lines.append(f"{len(types)} types: " + ", ".join(f"`{t}`" for t in types) + ".")
        lines.append("Labels are BIO: `O`, then `B-<type>` / `I-<type>` per type.")
        lines.append("")

    train_lines: list[str] = []
    if isinstance(training.get("train_records"), int):
        detail = f"- **Training rows:** {training['train_records']}"
        if isinstance(training.get("train_records_labelled"), int):
            detail += f" ({training['train_records_labelled']} with at least one span)"
        train_lines.append(detail)
    if isinstance(training.get("eval_records"), int) and training["eval_records"]:
        train_lines.append(f"- **Validation rows:** {training['eval_records']}")
    if training.get("included_types"):
        train_lines.append("- **Data kinds:** " + ", ".join(str(t) for t in training["included_types"]))
    sources = training.get("sources") or {}
    if sources:
        listed = sorted(sources.items(), key=lambda kv: -kv[1])[:MAX_SOURCES_LISTED]
        train_lines.append("- **Row sources:** " + "; ".join(f"`{name}` {n}" for name, n in listed))
    aug = training.get("augmentation")
    if isinstance(aug, dict) and aug.get("rows_added"):
        train_lines.append(
            f"- **Augmentation:** {aug['rows_added']} OCR-noise copies (character noise rate {aug.get('ocr_noise_rate')})"
        )
    if isinstance(training.get("epochs"), (int, float)):
        train_lines.append(f"- **Epochs:** {training['epochs']:g}")
    if train_lines:
        lines.append("## Training data")
        lines.append("")
        lines += train_lines
        lines.append("")

    if evaluations:
        lines.append("## Evaluation")
        lines.append("")
        lines.append("Strict span match: a prediction counts only when its type, start and end all equal a reference span.")
        lines.append("")
        for block in evaluations:
            label = block.get("label") or block.get("dataset_name") or "held-out"
            heading = f"### {label[:1].upper()}{label[1:]}"
            if block.get("rows"):
                heading += f" ({block['rows']} texts)"
            lines.append(heading)
            lines.append("")
            if block.get("written_by"):
                lines.append(f"Reference spans written by `{block['written_by']}`.")
                lines.append("")
            lines.append(
                f"Overall: precision {_fmt(block.get('precision'))} · recall {_fmt(block.get('recall'))} · F1 {_fmt(block.get('f1'))}"
            )
            lines.append("")
            per_type = block.get("per_type") or {}
            if per_type:
                lines.append("| Type | Precision | Recall | F1 | Gold spans |")
                lines.append("|---|---|---|---|---|")
                for t in sorted(per_type):
                    v = per_type[t] if isinstance(per_type[t], dict) else {}
                    support = (v.get("tp") or 0) + (v.get("fn") or 0) if ("tp" in v or "fn" in v) else None
                    lines.append(
                        f"| `{t}` | {_fmt(v.get('precision'))} | {_fmt(v.get('recall'))} | {_fmt(v.get('f1'))} | "
                        f"{support if support is not None else '–'} |"
                    )
                lines.append("")

    model_ref = "path/to/model"
    lines.append("## Usage")
    lines.append("")
    lines.append("```python")
    lines.append("from transformers import pipeline")
    lines.append("")
    lines.append(
        f'tagger = pipeline("token-classification", model="{model_ref}", aggregation_strategy="simple", stride={stride})'
    )
    lines.append("")
    lines.append("def find_spans(text):")
    lines.append("    spans = []")
    lines.append("    for ent in tagger(text):")
    lines.append('        start, end = ent["start"], ent["end"]')
    lines.append("        # The tokenizer can attach the preceding space to a span; trim it.")
    lines.append("        while start < end and text[start].isspace():")
    lines.append("            start += 1")
    lines.append('        spans.append({"type": ent["entity_group"], "start": start, "end": end,')
    lines.append('                      "text": text[start:end], "score": float(ent["score"])})')
    lines.append("    return spans")
    lines.append("```")
    lines.append("")
    lines.append(
        f"`stride={stride}` matters: without it the pipeline cuts a text at the {window or 'model'}-token window "
        "and never tags anything after it. With it, long texts are read in overlapping windows."
    )
    lines.append("")
    if thresholds_file:
        lines.append(
            f"`{THRESHOLDS_FILE}` holds per-type minimum scores (all 0.0 = keep every predicted span) and a "
            "`recall_bias`. To favour precision for a type, drop its spans whose score is under that type's value. "
            "BrewSLM's redact endpoint and scored decoder read this file; there the score is the span's mean "
            "per-token probability of its type, and `recall_bias` (0–1) is subtracted from each token's `O` "
            "probability so borderline tokens are tagged (more recall, less precision)."
        )
        lines.append("")

    limits: list[str] = []
    if evaluations:
        sizes = [f"{b['rows']} texts ({b.get('label') or b.get('dataset_name')})" for b in evaluations if b.get("rows")]
        largest = max((int(b["rows"]) for b in evaluations if isinstance(b.get("rows"), int)), default=0)
        if sizes and largest < SMALL_EVAL_TEXTS:
            limits.append("Scores come from small held-out sets — " + ", ".join(sizes) + " — so expect a few points of noise either way.")
        elif sizes:
            limits.append("Evaluated on " + ", ".join(sizes) + "; your own texts may differ from these.")
        low = sorted({
            t for b in evaluations for t, v in (b.get("per_type") or {}).items()
            if isinstance(v, dict) and ("tp" in v or "fn" in v) and (v.get("tp") or 0) + (v.get("fn") or 0) < LOW_SUPPORT_SPANS
        })
        if low:
            limits.append(
                f"Types with fewer than {LOW_SUPPORT_SPANS} reference spans in an evaluation ("
                + ", ".join(f"`{t}`" for t in low) + ") have unreliable per-type scores."
            )
        if not any(b.get("dataset_name") == "gold_test" for b in evaluations):
            limits.append(
                "The test examples were labelled the same way as the training data, so this score cannot see "
                "labelling mistakes the two share; there is no independent answer key."
            )
    else:
        limits.append("This model has no held-out span evaluation recorded; measure it on your own texts before relying on it.")
    if "synthetic" in [str(t).lower() for t in (training.get("included_types") or [])]:
        limits.append(
            "The training data is synthetic or model-labelled, not human-annotated; real documents with other "
            "layouts, formats or languages may score lower."
        )
    limits.append("It finds what it was trained to find: entity types outside the list above are never tagged.")
    lines.append("## Limitations")
    lines.append("")
    lines += [f"- {item}" for item in limits]
    lines.append("")
    return "\n".join(lines)


# ── Evidence gathering ──────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _row_sources(train_file: Path, expected_rows: int | None) -> dict[str, int]:
    """``{source: rows}`` from the training file the run read — only when
    its parseable row count still matches the run's report (a re-split
    since would make the tally describe different data)."""
    counts: Counter[str] = Counter()
    total = 0
    try:
        with train_file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    continue
                total += 1
                source = row.get("source") or row.get("synth_source") or row.get("_source_type")
                if isinstance(source, str) and source:
                    counts[source] += 1
    except Exception:  # noqa: BLE001
        return {}
    if expected_rows is not None and total != expected_rows:
        return {}
    return dict(counts)


def training_evidence(output_dir: Path | None) -> dict[str, Any]:
    """The run's own report (``training_report.json``) → the model card's
    training block. Empty when there is no report."""
    if output_dir is None:
        return {}
    report = _read_json(output_dir / "training_report.json") or {}
    if not report:
        return {}
    out: dict[str, Any] = {}
    for key in ("train_records", "train_records_labelled", "eval_records", "augmentation"):
        if report.get(key) is not None:
            out[key] = report[key]
    if isinstance(report.get("epochs"), (int, float)):
        out["epochs"] = report["epochs"]
    train_file = Path(str(report.get("train_file"))) if report.get("train_file") else None
    if train_file is not None and train_file.is_file():
        rows = report.get("train_records") if isinstance(report.get("train_records"), int) else None
        sources = _row_sources(train_file, rows)
        if sources:
            out["sources"] = sources
        manifest = _read_json(train_file.parent / "manifest.json") or {}
        splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
        if manifest.get("included_types") and (rows is None or splits.get("train") == rows):
            out["included_types"] = list(manifest["included_types"])
    return out


async def span_evaluations(db: AsyncSession, project_id: int, experiment_id: int) -> list[dict[str, Any]]:
    """The run's latest span EvalResult per dataset — answer key
    (``gold_test``) first, test examples second — as model-card blocks."""
    from app.services.eval_summary_service import _answer_key_source

    rows = (
        await db.execute(
            select(EvalResult)
            .where(EvalResult.experiment_id == experiment_id)
            .where(EvalResult.dataset_name.in_([name for name, _ in _SPAN_DATASETS]))
            .order_by(desc(EvalResult.id))
        )
    ).scalars().all()
    latest: dict[str, EvalResult] = {}
    for result in rows:
        metrics = result.metrics if isinstance(result.metrics, dict) else {}
        if isinstance(metrics.get("per_class"), dict) and result.dataset_name not in latest:
            latest[result.dataset_name] = result
    blocks: list[dict[str, Any]] = []
    for name, label in _SPAN_DATASETS:
        result = latest.get(name)
        if result is None:
            continue
        metrics = result.metrics
        block = {
            "label": label,
            "dataset_name": name,
            "eval_result_id": int(result.id),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "rows": metrics.get("total") or metrics.get("evaluated_samples"),
            "per_type": metrics.get("per_class") or {},
        }
        if name == "gold_test":
            block["written_by"] = _answer_key_source(project_id)
        blocks.append(block)
    return blocks


def _label_space(model_dir: Path) -> list[str]:
    data = _read_json(model_dir / "label_space.json") or {}
    types = data.get("entity_types")
    if isinstance(types, list) and types:
        return [str(t) for t in types]
    config = _read_json(model_dir / "config.json") or {}
    labels = (config.get("id2label") or {}).values() if isinstance(config.get("id2label"), dict) else []
    return sorted({str(l)[2:] for l in labels if str(l) != "O" and len(str(l)) > 2})


def _window(model_dir: Path) -> int | None:
    tok = _read_json(model_dir / "tokenizer_config.json") or {}
    try:
        value = float(tok.get("model_max_length"))
    except (TypeError, ValueError):
        return None
    return int(value) if 0 < value <= 1_000_000 else None


def is_token_classifier_dir(model_dir: Path) -> bool:
    config = _read_json(Path(model_dir) / "config.json") or {}
    return any(str(a).endswith("ForTokenClassification") for a in (config.get("architectures") or []))


async def package_token_classifier(
    db: AsyncSession,
    *,
    project_id: int,
    experiment: Experiment,
    model_dir: Path,
    project_name: str | None = None,
) -> dict[str, Any] | None:
    """Write ``README.md`` + ``thresholds.json`` into an exported span
    model directory. None (nothing written) for any other model. An
    existing thresholds.json (copied from the run) is kept."""
    model_dir = Path(model_dir)
    if not is_token_classifier_dir(model_dir):
        return None
    entity_types = _label_space(model_dir)
    output_dir = Path(experiment.output_dir).expanduser() if experiment.output_dir else None
    training = training_evidence(output_dir)
    evaluations = await span_evaluations(db, project_id, experiment.id)
    window = _window(model_dir)
    card = build_span_model_card(
        entity_types=entity_types,
        base_model=experiment.base_model,
        experiment_id=experiment.id,
        project_name=project_name,
        window=window,
        training=training,
        evaluations=evaluations,
    )
    (model_dir / MODEL_CARD_FILE).write_text(card, encoding="utf-8")
    thresholds_path = model_dir / THRESHOLDS_FILE
    if not thresholds_path.exists():
        thresholds_path.write_text(json.dumps(default_thresholds(entity_types), indent=2), encoding="utf-8")
    return {
        "model_card": MODEL_CARD_FILE,
        "thresholds": THRESHOLDS_FILE,
        "entity_types": entity_types,
        "evaluations": [b["dataset_name"] for b in evaluations],
        "eval_result_ids": [b["eval_result_id"] for b in evaluations],
        "window": window,
        "pipeline_stride": pipeline_stride(window),
    }
