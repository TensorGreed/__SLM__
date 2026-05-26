"""A/B harness for auto-RAG (USER-SUCCESS Epic 9 Phase 9c).

Decides whether to promote Phase 9d (default-on + UI + target profile)
by running real fine-tunes of the QA-SFT template (``policy-qa-style``)
and measuring eval F1 with and without auto-RAG retrieval at
inference time. Per seed: ONE training run (auto-RAG is an inference-
only swap given the same trained model) feeds TWO eval passes
(``with_rag=False`` and ``with_rag=True``). The lift between the two
is the signal Phase 9d is gated on.

Gate criterion (strict): Phase 9d ships **only if** mean F1 lift ≥ 5%
on the one available QA-SFT template, with non-overlapping ``mean ± 1σ``
bands across 5 seeds. The 1-template coverage is weaker than Epic 6's
2-template gate; the harness logs that limitation in the roadmap
block so the result is auditable.

Eval metric: ``evaluation_service.f1_score`` (token-level SQuAD-style
F1 over normalized multisets) — the same primitive the auto-gate
uses, so a 9c PASS maps directly to the real F1 the user will see
on their eval runs.

Usage:

  python -m backend.scripts.auto_rag_ab \
      [--seeds 5] [--num-epochs 3] \
      [--output auto_rag_ab_results.json]

Per-run cost on GB10 with SmolLM2-135M, 140 train rows, 3 epochs:
~30-60s training + ~3 min eval inference (2 conditions × ~28 val
rows × ~3s each). 5 seeds ≈ 20-25 minutes total wall time.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Phase 9c covers the one QA-SFT template that ships. When a second
# QA-SFT template lands, add it here — the strict gate requires lift
# on every listed template, so adding one tightens the bar.
QA_SFT_TEMPLATES: tuple[str, ...] = ("policy-qa-style",)

# Gate threshold mirrors Epic 6c: ≥5% lift, non-overlapping 1σ bands.
GATE_MIN_LIFT_PCT: float = 5.0

# Generation budget per val row. SmolLM2-135M with max_new_tokens=200
# costs ~2-4s on GB10. Larger budget yields longer answers but the
# QA template answers are typically 1-3 sentences — 200 is plenty.
GENERATION_MAX_NEW_TOKENS: int = 200

# Top-K retrieval count for the with-RAG condition. Matches the
# Phase 9b default (PlaygroundChatRequest.auto_rag_k default is 3).
RAG_K: int = 3


# ─────────────────────────────────────────────────────────────────────
# Result types (mirror curriculum_ab.py's shape; minor field renames)
# ─────────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    template: str
    seed: int
    without_rag_f1: float | None
    with_rag_f1: float | None
    train_runtime_seconds: float | None
    eval_runtime_seconds: float | None
    output_dir: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "seed": self.seed,
            "without_rag_f1": self.without_rag_f1,
            "with_rag_f1": self.with_rag_f1,
            "train_runtime_seconds": self.train_runtime_seconds,
            "eval_runtime_seconds": self.eval_runtime_seconds,
            "output_dir": self.output_dir,
            "error": self.error,
        }


@dataclass
class TemplateSummary:
    template: str
    on_f1s: list[float] = field(default_factory=list)   # with RAG
    off_f1s: list[float] = field(default_factory=list)  # without RAG

    @property
    def on_mean(self) -> float | None:
        return statistics.mean(self.on_f1s) if self.on_f1s else None

    @property
    def off_mean(self) -> float | None:
        return statistics.mean(self.off_f1s) if self.off_f1s else None

    @property
    def on_std(self) -> float:
        return statistics.stdev(self.on_f1s) if len(self.on_f1s) > 1 else 0.0

    @property
    def off_std(self) -> float:
        return statistics.stdev(self.off_f1s) if len(self.off_f1s) > 1 else 0.0

    @property
    def absolute_lift(self) -> float | None:
        if self.on_mean is None or self.off_mean is None:
            return None
        return self.on_mean - self.off_mean

    @property
    def relative_lift_pct(self) -> float | None:
        if self.on_mean is None or self.off_mean is None or self.off_mean == 0:
            return None
        return (self.on_mean - self.off_mean) / self.off_mean * 100.0

    @property
    def bands_non_overlapping(self) -> bool:
        if self.on_mean is None or self.off_mean is None:
            return False
        return (self.on_mean - self.on_std) > (self.off_mean + self.off_std)


# ─────────────────────────────────────────────────────────────────────
# Template data prep — mirrors curriculum_ab pattern, qa-sft variant
# ─────────────────────────────────────────────────────────────────────


def _read_template_gold(template_slug: str) -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[2]
    path = (
        repo_root
        / "backend"
        / "data"
        / "project_templates"
        / template_slug
        / "gold.jsonl"
    )
    if not path.exists():
        raise FileNotFoundError(f"Template gold set not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _flatten_qa_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten ``{input:{question}, expected:{answer}}`` into the flat
    ``{question, answer}`` shape train.py's causal_lm adapter reads
    (input_fields=question, target_fields=answer)."""
    input_block = row.get("input") or {}
    expected = row.get("expected") or {}
    if not isinstance(input_block, dict) or not isinstance(expected, dict):
        return None
    question = ""
    for key in ("question", "input", "prompt", "instruction"):
        v = input_block.get(key)
        if isinstance(v, str) and v.strip():
            question = v.strip()
            break
    answer = expected.get("answer")
    if not (question and isinstance(answer, str) and answer.strip()):
        return None
    return {"question": question, "answer": answer.strip()}


def _split_70_15_15(rows: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Matches ``demo_project_service._split_rows`` so harness eval
    runs on the same val split a real demo project would."""
    total = len(rows)
    if total < 3:
        return list(rows), [], []
    n_test = max(1, total // 7)
    n_val = max(1, total // 7)
    n_train = total - n_val - n_test
    return (
        rows[:n_train],
        rows[n_train : n_train + n_val],
        rows[n_train + n_val :],
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_template_splits(template_slug: str, prepared_dir: Path) -> dict[str, int]:
    """Read template gold → flatten → 70/15/15 split → write JSONL.
    Returns row counts so the harness can sanity-check the prep."""
    gold = _read_template_gold(template_slug)
    flat = [r for r in (_flatten_qa_row(r) for r in gold) if r is not None]
    train, val, test = _split_70_15_15(flat)
    _write_jsonl(prepared_dir / "train.jsonl", train)
    _write_jsonl(prepared_dir / "val.jsonl", val)
    _write_jsonl(prepared_dir / "test.jsonl", test)
    return {"train": len(train), "val": len(val), "test": len(test)}


# ─────────────────────────────────────────────────────────────────────
# Training subprocess (mirrors curriculum_ab.run_one_finetune shape)
# ─────────────────────────────────────────────────────────────────────


def _build_run_config(*, num_epochs: int, seed: int) -> dict[str, Any]:
    """Minimal QA-SFT training config: causal_lm task, llama3 chat
    template, LoRA. Matches what a real policy-qa-style project would
    train with at thin-data defaults."""
    return {
        "task_type": "causal_lm",
        "training_mode": "sft",
        "chat_template": "llama3",
        "num_epochs": num_epochs,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "max_seq_length": 512,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "save_steps": 1000,
        "eval_steps": 50,
        "warmup_ratio": 0.03,
        "seed": seed,
    }


def run_one_training(
    *,
    template_slug: str,
    seed: int,
    num_epochs: int,
    base_model: str,
    workdir: Path,
) -> tuple[Path | None, str | None, float]:
    """Subprocess train.py for one (template, seed). Returns
    (model_dir, error, runtime_seconds). On error, model_dir is None
    and error carries a tail of stderr/stdout."""
    run_id = f"{template_slug}_seed{seed}"
    template_dir = workdir / template_slug
    output_dir = workdir / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = template_dir / "train.jsonl"
    val_file = template_dir / "val.jsonl"
    config_path = output_dir / "training_config.json"
    config_path.write_text(
        json.dumps(_build_run_config(num_epochs=num_epochs, seed=seed), indent=2),
        encoding="utf-8",
    )

    train_script = Path(__file__).resolve().parent / "train.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--project", "0",
        "--experiment", str(abs(hash(run_id)) % (10**8)),
        "--output", str(output_dir),
        "--base-model", base_model,
        "--config", str(config_path),
        "--train-file", str(train_file),
        "--val-file", str(val_file),
        "--seed", str(seed),
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - started

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "(no output)")[-2000:]
        return None, f"train.py exited rc={proc.returncode}: {err}", elapsed
    model_dir = output_dir / "model"
    if not model_dir.exists():
        return None, f"model dir not written at {model_dir}", elapsed
    return model_dir, None, elapsed


# ─────────────────────────────────────────────────────────────────────
# Eval inference — load LoRA, generate, score against val.jsonl
# ─────────────────────────────────────────────────────────────────────


# Mirrors train.py's _qa_to_chat_text(llama3) but emits only the
# prompt portion (user turn + assistant header). The model generates
# the answer + <|eot_id|>.
def _format_llama3_inference_prompt(question: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _build_rag_preamble(retrieved_pairs: list[dict[str, Any]]) -> str:
    """Mirrors auto_rag_service._AUTO_RAG_PREAMBLE_TEMPLATE +
    _format_pair so the A/B condition matches what the real
    playground produces in Phase 9b."""
    parts: list[str] = [
        "Reference Q&A pairs from the knowledge base "
        "(use them to ground your answer; cite the matching pair "
        "number if you use one):",
    ]
    for idx, pair in enumerate(retrieved_pairs, start=1):
        q = str(pair.get("question") or "").strip()
        a = str(pair.get("answer") or "").strip()
        parts.append(f"[{idx}] Q: {q}\n    A: {a}")
    parts.append("Now answer the user's next question.")
    return "\n\n".join(parts)


def _format_llama3_rag_prompt(question: str, retrieved_pairs: list[dict[str, Any]]) -> str:
    """With-RAG prompt: system message preamble (the retrieved pairs)
    + user question + assistant header. Mirrors the playground path's
    insert-after-existing-system-messages shape."""
    preamble = _build_rag_preamble(retrieved_pairs)
    return (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{preamble}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


_ASSISTANT_TAIL_RE = re.compile(r"<\|eot_id\|>.*", flags=re.DOTALL)


def _clean_generated_answer(decoded: str) -> str:
    """The model's output decodes to the entire conversation including
    the prompt + the new tokens. ``model.generate`` returns the prompt
    too, so we strip up to the LAST assistant header before splitting
    on the eot marker."""
    last_header = decoded.rfind("<|start_header_id|>assistant<|end_header_id|>")
    if last_header >= 0:
        after = decoded[last_header:].split(">\n\n", 1)
        decoded = after[1] if len(after) == 2 else decoded
    return _ASSISTANT_TAIL_RE.sub("", decoded).strip()


def evaluate_with_inference(
    *,
    base_model: str,
    model_dir: Path,
    val_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    with_rag: bool,
    rag_k: int = RAG_K,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Load the trained model (base + LoRA), generate an answer for
    each val row, score via ``evaluation_service.f1_score``. Returns
    (per-row F1s, per-row record dicts for debugging).

    When ``with_rag`` is True, the harness builds a BM25 index over
    ``train_rows`` (using ``auto_rag_service.build_bm25_index``) and
    prepends the top-K retrievals to each prompt — exactly the shape
    Phase 9b's playground path uses, so the A/B is faithful to what
    the user will see if Phase 9d ships."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.services.auto_rag_service import (
        AutoRagUnavailable,
        build_bm25_index,
        retrieve,
    )
    from app.services.evaluation_service import f1_score

    # Build a transient BM25 index for the with-RAG condition. Lives
    # in a sibling dir of the model output so each seed's index is
    # isolated (mirrors what training-completion would do at the
    # project level).
    index_dir = model_dir.parent / "auto_rag"
    if with_rag:
        try:
            build_bm25_index(
                train_rows,
                recipe_id="qa-sft",
                output_dir=index_dir,
            )
        except AutoRagUnavailable as e:
            raise RuntimeError(f"failed to build BM25 index for with-RAG eval: {e}") from e

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()

    f1s: list[float] = []
    records: list[dict[str, Any]] = []
    for row in val_rows:
        question = str(row.get("question") or "").strip()
        reference = str(row.get("answer") or "").strip()
        if not question or not reference:
            continue
        retrieved_pairs: list[dict[str, Any]] = []
        if with_rag:
            try:
                hits = retrieve(question, index_dir=index_dir, k=rag_k)
            except AutoRagUnavailable:
                hits = []
            for hit in hits:
                payload = hit.get("payload") or {}
                retrieved_pairs.append({
                    "question": payload.get("question", ""),
                    "answer": payload.get("answer", ""),
                })
            prompt = _format_llama3_rag_prompt(question, retrieved_pairs)
        else:
            prompt = _format_llama3_inference_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        generated = _clean_generated_answer(decoded)
        score = f1_score(generated, reference)
        f1s.append(score)
        records.append({
            "question": question,
            "reference": reference,
            "generated": generated[:400],
            "f1": score,
            "retrieved_row_count": len(retrieved_pairs),
        })

    # Free GPU memory so the next seed can fresh-load.
    del model, base
    torch.cuda.empty_cache()
    return f1s, records


# ─────────────────────────────────────────────────────────────────────
# Per-seed orchestration
# ─────────────────────────────────────────────────────────────────────


def run_one_seed(
    *,
    template_slug: str,
    seed: int,
    num_epochs: int,
    base_model: str,
    workdir: Path,
) -> RunResult:
    """One seed end-to-end: train ONCE, eval TWICE (with + without
    RAG). Auto-RAG is inference-only, so reusing the trained model
    for both conditions controls for training-side noise — the only
    variable between conditions is the inference-time prompt."""
    template_dir = workdir / template_slug
    train_path = template_dir / "train.jsonl"
    val_path = template_dir / "val.jsonl"
    with train_path.open(encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f if line.strip()]
    with val_path.open(encoding="utf-8") as f:
        val_rows = [json.loads(line) for line in f if line.strip()]

    model_dir, train_err, train_runtime = run_one_training(
        template_slug=template_slug,
        seed=seed,
        num_epochs=num_epochs,
        base_model=base_model,
        workdir=workdir,
    )
    if train_err:
        return RunResult(
            template=template_slug,
            seed=seed,
            without_rag_f1=None,
            with_rag_f1=None,
            train_runtime_seconds=round(train_runtime, 1),
            eval_runtime_seconds=None,
            output_dir=str(workdir / "runs" / f"{template_slug}_seed{seed}"),
            error=train_err,
        )

    eval_started = time.time()
    try:
        off_f1s, off_records = evaluate_with_inference(
            base_model=base_model, model_dir=model_dir,
            val_rows=val_rows, train_rows=train_rows, with_rag=False,
        )
        on_f1s, on_records = evaluate_with_inference(
            base_model=base_model, model_dir=model_dir,
            val_rows=val_rows, train_rows=train_rows, with_rag=True,
        )
    except Exception as e:  # noqa: BLE001 — eval failure is recoverable, log + continue
        return RunResult(
            template=template_slug,
            seed=seed,
            without_rag_f1=None,
            with_rag_f1=None,
            train_runtime_seconds=round(train_runtime, 1),
            eval_runtime_seconds=round(time.time() - eval_started, 1),
            output_dir=str(model_dir.parent),
            error=f"eval failed: {type(e).__name__}: {e}",
        )
    eval_runtime = time.time() - eval_started

    # Persist per-row eval records for offline inspection.
    debug_path = model_dir.parent / "eval_records.json"
    debug_path.write_text(
        json.dumps({"without_rag": off_records, "with_rag": on_records}, indent=2),
        encoding="utf-8",
    )

    return RunResult(
        template=template_slug,
        seed=seed,
        without_rag_f1=statistics.mean(off_f1s) if off_f1s else None,
        with_rag_f1=statistics.mean(on_f1s) if on_f1s else None,
        train_runtime_seconds=round(train_runtime, 1),
        eval_runtime_seconds=round(eval_runtime, 1),
        output_dir=str(model_dir.parent),
    )


# ─────────────────────────────────────────────────────────────────────
# Aggregation + gate
# ─────────────────────────────────────────────────────────────────────


def aggregate_results(results: list[RunResult]) -> dict[str, TemplateSummary]:
    summaries: dict[str, TemplateSummary] = {}
    for r in results:
        if r.without_rag_f1 is None or r.with_rag_f1 is None:
            continue
        s = summaries.setdefault(r.template, TemplateSummary(template=r.template))
        s.off_f1s.append(r.without_rag_f1)
        s.on_f1s.append(r.with_rag_f1)
    return summaries


@dataclass
class GateDecision:
    passed: bool
    reason: str
    per_template: dict[str, dict[str, Any]]


def apply_gate(summaries: dict[str, TemplateSummary]) -> GateDecision:
    """Phase 9d ships iff every template clears the ≥5% lift AND
    has non-overlapping ``mean ± 1σ`` bands. Anything weaker stops
    Epic 9 at 9b as the power-user feature."""
    per_template: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for slug, s in summaries.items():
        lift = s.relative_lift_pct
        non_overlap = s.bands_non_overlapping
        passed = lift is not None and lift >= GATE_MIN_LIFT_PCT and non_overlap
        per_template[slug] = {
            "on_mean": s.on_mean,
            "off_mean": s.off_mean,
            "on_std": s.on_std,
            "off_std": s.off_std,
            "absolute_lift": s.absolute_lift,
            "relative_lift_pct": lift,
            "bands_non_overlapping": non_overlap,
            "passed": passed,
            "n_on": len(s.on_f1s),
            "n_off": len(s.off_f1s),
        }
        if not passed:
            if lift is None:
                failures.append(f"{slug}: lift undefined (missing runs)")
            elif lift < GATE_MIN_LIFT_PCT:
                failures.append(
                    f"{slug}: lift={lift:.2f}% < {GATE_MIN_LIFT_PCT}% threshold"
                )
            elif not non_overlap:
                failures.append(
                    f"{slug}: bands overlap "
                    f"(on={s.on_mean:.3f}±{s.on_std:.3f}, "
                    f"off={s.off_mean:.3f}±{s.off_std:.3f})"
                )
    if not summaries:
        return GateDecision(
            passed=False, reason="no successful runs", per_template={},
        )
    if failures:
        return GateDecision(passed=False, reason="; ".join(failures), per_template=per_template)
    return GateDecision(
        passed=True,
        reason=(
            f"all templates lifted ≥ {GATE_MIN_LIFT_PCT}% with non-overlapping "
            f"1σ bands — ship Phase 9d"
        ),
        per_template=per_template,
    )


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────


def format_markdown_block(
    results: list[RunResult],
    summaries: dict[str, TemplateSummary],
    gate: GateDecision,
    *,
    base_model: str,
    num_epochs: int,
    seeds: list[int],
) -> str:
    lines: list[str] = []
    lines.append("**Phase 9c A/B results.**")
    lines.append("")
    lines.append(
        f"Setup: base model `{base_model}` · {num_epochs} epochs · "
        f"seeds {seeds} · LoRA r=16 · GB10 GPU · token-level F1 over "
        f"the val split."
    )
    lines.append("")
    lines.append("| Template | n (with/without) | Mean F1 (with RAG) | Mean F1 (without RAG) | Lift | Non-overlap 1σ? |")
    lines.append("|---|---|---|---|---|---|")
    for slug in QA_SFT_TEMPLATES:
        summary = summaries.get(slug)
        if summary is None or not summary.on_f1s or not summary.off_f1s:
            lines.append(f"| {slug} | -/- | — | — | — | — |")
            continue
        lift_str = (
            f"{summary.relative_lift_pct:+.2f}%"
            if summary.relative_lift_pct is not None else "—"
        )
        lines.append(
            f"| {slug} | {len(summary.on_f1s)}/{len(summary.off_f1s)} | "
            f"{summary.on_mean:.4f} ± {summary.on_std:.4f} | "
            f"{summary.off_mean:.4f} ± {summary.off_std:.4f} | "
            f"{lift_str} | {'✓' if summary.bands_non_overlapping else '✗'} |"
        )
    lines.append("")
    if gate.passed:
        lines.append(f"**Gate: PASS** — {gate.reason}. Phase 9d cleared to ship.")
    else:
        lines.append(f"**Gate: FAIL** — {gate.reason}.")
        lines.append(
            "Per the Phase 9c criterion, Epic 9 stops at Phase 9b as a "
            "power-user feature: auto-RAG works (opt-in via the playground "
            "flag) but the lift doesn't justify shipping a default-on "
            "heuristic. Phase 9a.1 (embedding hybrid) or a different "
            "template might revisit."
        )
    lines.append("")
    lines.append(
        "_Coverage caveat: Phase 9c gates on the **one** QA-SFT template "
        "that ships (`policy-qa-style`). Epic 6c had 2-template coverage; "
        "adding a second QA-SFT template would tighten the statistical "
        "case. The strict gate compensates by requiring non-overlapping "
        "bands at 5 seeds, but a second template is the right next step "
        "before scaling auto-RAG to other recipes._"
    )
    failed = [r for r in results if r.error is not None]
    if failed:
        lines.append("")
        lines.append(f"_{len(failed)} run(s) failed:_")
        for r in failed:
            excerpt = (r.error or "").splitlines()[0][:200]
            lines.append(f"  - `{r.template}` seed={r.seed}: {excerpt}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# CLI driver
# ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B harness for auto-RAG (Phase 9c)."
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument(
        "--base-model", type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument("--workdir", type=str, default="")
    parser.add_argument(
        "--output", type=str, default="auto_rag_ab_results.json",
    )
    parser.add_argument(
        "--markdown-output", type=str, default="auto_rag_ab_results.md",
    )
    parser.add_argument("--templates", type=str, nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    templates = tuple(args.templates) if args.templates else QA_SFT_TEMPLATES
    seeds = list(range(args.seeds))
    workdir = (
        Path(args.workdir).expanduser().resolve()
        if args.workdir
        else Path("/tmp") / f"auto_rag_ab_{int(time.time())}"
    )
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[harness] workdir={workdir}")
    for slug in templates:
        template_dir = workdir / slug
        if (template_dir / "train.jsonl").exists():
            continue
        counts = prepare_template_splits(slug, template_dir)
        print(f"[harness] {slug}: train={counts['train']} val={counts['val']} test={counts['test']}")

    results: list[RunResult] = []
    results_path = Path(args.output).expanduser().resolve()
    if results_path.exists():
        try:
            prior = json.loads(results_path.read_text(encoding="utf-8"))
            for r in prior.get("runs", []):
                results.append(RunResult(**r))
            print(f"[harness] resumed: {len(results)} runs already on disk")
        except (json.JSONDecodeError, TypeError):
            pass
    already_done = {(r.template, r.seed) for r in results}

    total_combos = len(templates) * len(seeds)
    done = len(already_done)
    for slug in templates:
        for seed in seeds:
            if (slug, seed) in already_done:
                continue
            done += 1
            label = f"{slug} seed={seed}"
            print(f"[harness] ({done}/{total_combos}) {label} …")
            t0 = time.time()
            result = run_one_seed(
                template_slug=slug, seed=seed,
                num_epochs=args.num_epochs, base_model=args.base_model,
                workdir=workdir,
            )
            results.append(result)
            _persist_results(
                results=results,
                results_path=results_path,
                markdown_path=Path(args.markdown_output).expanduser().resolve(),
                base_model=args.base_model,
                num_epochs=args.num_epochs,
                seeds=seeds,
            )
            if result.error:
                print(f"  ✗ failed in {time.time() - t0:.1f}s: {result.error.splitlines()[0][:160]}")
            else:
                print(
                    f"  ✓ off={result.without_rag_f1:.4f} on={result.with_rag_f1:.4f} "
                    f"(train {result.train_runtime_seconds}s + eval {result.eval_runtime_seconds}s)"
                )

    summaries = aggregate_results(results)
    gate = apply_gate(summaries)
    _persist_results(
        results=results,
        results_path=results_path,
        markdown_path=Path(args.markdown_output).expanduser().resolve(),
        base_model=args.base_model,
        num_epochs=args.num_epochs,
        seeds=seeds,
    )

    print()
    print(format_markdown_block(
        results, summaries, gate,
        base_model=args.base_model,
        num_epochs=args.num_epochs,
        seeds=seeds,
    ))
    print()
    print(f"[harness] raw results → {results_path}")
    print(f"[harness] roadmap block → {args.markdown_output}")
    return 0 if gate.passed else 1


def _persist_results(
    *,
    results: list[RunResult],
    results_path: Path,
    markdown_path: Path,
    base_model: str,
    num_epochs: int,
    seeds: list[int],
) -> None:
    summaries = aggregate_results(results)
    gate = apply_gate(summaries)
    payload = {
        "base_model": base_model,
        "num_epochs": num_epochs,
        "seeds": seeds,
        "gate": {
            "passed": gate.passed,
            "reason": gate.reason,
            "per_template": gate.per_template,
        },
        "summaries": {
            slug: {
                "on_mean": s.on_mean,
                "off_mean": s.off_mean,
                "on_std": s.on_std,
                "off_std": s.off_std,
                "absolute_lift": s.absolute_lift,
                "relative_lift_pct": s.relative_lift_pct,
                "bands_non_overlapping": s.bands_non_overlapping,
            }
            for slug, s in summaries.items()
        },
        "runs": [r.as_dict() for r in results],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(
        format_markdown_block(
            results, summaries, gate,
            base_model=base_model, num_epochs=num_epochs, seeds=seeds,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
