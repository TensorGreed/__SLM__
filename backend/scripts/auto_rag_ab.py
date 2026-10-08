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
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# Phase 9c covers the one QA-SFT template that ships. When a second
# QA-SFT template lands, add it here — the strict gate requires lift
# on every listed template, so adding one tightens the bar.
QA_SFT_TEMPLATES: tuple[str, ...] = ("policy-qa-style",)

# Gate threshold mirrors Epic 6c: ≥5% lift, non-overlapping 1σ bands.
GATE_MIN_LIFT_PCT: float = 5.0

# Upper bound on the generation budget per val row (the held-out eval's
# default). The budget actually used is sized to the project's answers —
# see ``_generation_cap``.
GENERATION_MAX_NEW_TOKENS: int = 128
GENERATION_MIN_NEW_TOKENS: int = 32
# Headroom over the longest training answer.
GENERATION_CAP_HEADROOM: float = 1.5

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


# Fallback for tokenizers WITHOUT a chat template (see
# ``_build_inference_prompt``). Mirrors train.py's _qa_to_chat_text(llama3)
# but emits only the prompt portion (user turn + assistant header).
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


def _build_inference_prompt(
    tokenizer: Any,
    question: str,
    retrieved_pairs: list[dict[str, Any]] | None = None,
    preamble: str | None = None,
) -> tuple[str, bool]:
    """Prompt for one eval row, in the format the model was trained on.

    Returns ``(prompt, used_chat_template)``. When the tokenizer ships a chat
    template (virtually every modern checkpoint), the prompt is rendered with
    it — the same contract as ``train.py`` (``use_tokenizer_chat_template``)
    and ``evaluation_service._apply_chat_template_if_present``. The harness
    used to hard-code Llama-3 headers for every model, so a ChatML model
    (SmolLM2, Qwen) saw tokens it was never trained on and both A/B arms
    scored near zero. The Llama-3 builders remain the fallback for tokenizers
    without a template.

    ``retrieved_pairs`` (with-RAG arm) become a leading system message, as in
    the playground.
    """
    messages: list[dict[str, str]] = []
    if preamble is not None:
        # Document passages (``--corpus documents``): the playground's
        # cite-or-say-you-don't-know preamble.
        messages.append({"role": "system", "content": preamble})
    elif retrieved_pairs is not None:
        messages.append({"role": "system", "content": _build_rag_preamble(retrieved_pairs)})
    messages.append({"role": "user", "content": question.strip()})
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001 — e.g. a template with no system role
            rendered = None
        if isinstance(rendered, str) and rendered.strip():
            return rendered, True
    if preamble is not None:
        return _format_llama3_rag_prompt(question, [{"question": "", "answer": preamble}]), False
    if retrieved_pairs is not None:
        return _format_llama3_rag_prompt(question, retrieved_pairs), False
    return _format_llama3_inference_prompt(question), False


def _project_id_from_index_dir(index_dir: Path | None) -> int:
    """``data/projects/<id>/auto_rag`` → ``<id>`` (documents mode needs the
    project's passage index, which lives next to the Q&A one)."""
    if index_dir is None:
        raise RuntimeError("documents corpus needs the project's auto_rag index dir")
    try:
        return int(index_dir.parent.name)
    except ValueError as exc:
        raise RuntimeError(f"cannot derive a project id from {index_dir}") from exc


def _judge_comparison_rows(
    project_id: int, off_records: list[dict[str, Any]], on_records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Score both arms with the answer judge (``answer_judge_service``) when
    the task is long-answer and a judge is reachable. Annotates each record
    with ``judge`` = {score, verdict, reason}; returns the judge summary
    (label + per-arm means) or None when it doesn't apply."""
    import asyncio

    from app.services.answer_judge_service import (
        judge_predictions,
        project_cache,
        resolve_answer_judge,
        should_judge,
    )

    references = [r["reference"] for r in off_records]
    if not should_judge("qa", references):
        return None

    async def _run() -> dict[str, Any] | None:
        resolved = await resolve_answer_judge(None, project_id, None)
        if resolved is None:
            return None
        cache = project_cache(project_id)
        out: dict[str, Any] = {"judge": resolved.label}
        for arm, records in (("without_rag", off_records), ("with_rag", on_records)):
            preds = [
                {"prompt": r["question"], "reference": r["reference"], "prediction": r["generated"]}
                for r in records
            ]
            snap = await judge_predictions(preds, resolved.judge, label=resolved.label, cache=cache)
            for rec, pred in zip(records, preds):
                if "row_judge_score" in pred:
                    rec["judge"] = {
                        "score": pred["row_judge_score"],
                        "verdict": pred["row_judge_verdict"],
                        "reason": pred["row_judge_reason"],
                    }
            out[arm] = {k: snap.get(k) for k in ("score", "counts", "judged", "unjudged", "judge_calls", "judge_cached")}
        return out

    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 — the judge never breaks the comparison
        print(f"[harness] judge skipped: {exc}")
        return None


def _generation_cap(tokenizer: Any, train_rows: list[dict[str, Any]]) -> int:
    """Generation budget for one answer: 1.5x the longest TRAINING answer,
    within [GENERATION_MIN_NEW_TOKENS, GENERATION_MAX_NEW_TOKENS].

    A small model that never emits its end-of-turn token rambles until the
    budget runs out; with a flat 200-token budget a correct first sentence
    was buried under ~150 tokens of filler and token-F1 scored it near zero
    (the with-RAG arm, which copies the preamble's Q/A pattern, worst of
    all). The cap comes from the training answers, never the val references.
    """
    longest = 0
    for row in train_rows:
        answer = str(row.get("answer") or "").strip()
        if not answer:
            continue
        try:
            longest = max(longest, len(tokenizer(answer, add_special_tokens=False)["input_ids"]))
        except Exception:  # noqa: BLE001
            continue
    if longest <= 0:
        return GENERATION_MAX_NEW_TOKENS
    return max(
        GENERATION_MIN_NEW_TOKENS,
        min(GENERATION_MAX_NEW_TOKENS, math.ceil(longest * GENERATION_CAP_HEADROOM)),
    )


def _stop_token_ids(tokenizer: Any) -> list[int]:
    """EOS plus whichever end-of-turn markers this tokenizer actually has."""
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.append(eos)
    unk = getattr(tokenizer, "unk_token_id", None)
    for marker in ("<|eot_id|>", "<|im_end|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(marker)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk and token_id not in ids:
            ids.append(token_id)
    return ids


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
    model_dir: Path | None,
    val_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    with_rag: bool,
    rag_k: int = RAG_K,
    index_dir_override: Path | None = None,
    progress_callback: "Callable[[int, int, str], None] | None" = None,
    corpus: str = "qa",
    reranker: str | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Load the trained model (base + LoRA adapter, or the full fine-tuned
    model when the run saved one), generate an answer for
    each val row, score via ``evaluation_service.f1_score``. Returns
    (per-row F1s, per-row record dicts for debugging).

    When ``with_rag`` is True, the harness needs a BM25 index to
    retrieve from. Two modes:

    * ``index_dir_override=None`` (Phase 9c gate path) — build a
      **transient** BM25 over ``train_rows`` next to ``model_dir``.
      Each seed gets its own index because each seed's training
      corpus may differ.
    * ``index_dir_override=<path>`` (Phase 9d per-project path) —
      use an **existing** BM25 index at the given path. The
      ``train_rows`` only sizes the generation cap in this mode. Use this when
      you want the comparison to predict what the project's actual
      playground will do (the playground reads
      ``data/projects/{id}/auto_rag/bm25_index.json``, built over the
      project's training rows — never the answer key or val/test).

    ``corpus="documents"`` retrieves the project's cleaned document
    *passages* (``auto_rag_service.ensure_document_index``) instead of Q&A
    pairs and prepends the playground's cite-or-say-you-don't-know preamble
    — the retrieval a documents-only project actually serves.
    ``index_dir_override`` is ignored in that mode.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.services.auto_rag_service import (
        AutoRagUnavailable,
        build_bm25_index,
        retrieve_ranked,
    )
    from app.services.evaluation_service import f1_score

    use_documents = corpus == "documents"
    if use_documents:
        from app.services.auto_rag_service import document_index_dir

        index_dir = document_index_dir(_project_id_from_index_dir(index_dir_override))
        if with_rag and not (index_dir / "bm25_index.json").exists():
            raise RuntimeError(
                f"{index_dir} has no document index — the project has no cleaned "
                "document passages (ensure_document_index builds it)."
            )
    elif index_dir_override is not None:
        # Phase 9d path — use the existing project-deployed index.
        # Refuse to silently fall back to building a transient one;
        # if the override points at a missing index we want a loud
        # error so the caller can either build it first or drop the
        # override.
        index_dir = index_dir_override
        if with_rag and not (index_dir / "bm25_index.json").exists():
            raise RuntimeError(
                f"index_dir_override={index_dir} has no bm25_index.json — "
                f"build the project's BM25 index first via "
                f"``auto_rag_service.build_index_for_project`` (normally "
                f"fired automatically at training completion)."
            )
    else:
        # Phase 9c gate path — build a transient BM25 next to the
        # model dir so each seed's index is isolated.
        if model_dir is None:
            raise RuntimeError("base-model evaluation needs index_dir_override")
        index_dir = model_dir.parent / "auto_rag"
        if with_rag:
            try:
                build_bm25_index(
                    train_rows,
                    recipe_id="qa-sft",
                    output_dir=index_dir,
                )
            except AutoRagUnavailable as e:
                raise RuntimeError(
                    f"failed to build BM25 index for with-RAG eval: {e}"
                ) from e

    # The run's own tokenizer (saved next to the adapter) carries the chat
    # template it was trained with; fall back to the base model's.
    tokenizer_source = (
        str(model_dir)
        if model_dir is not None and (model_dir / "tokenizer_config.json").exists()
        else base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    stop_ids = _stop_token_ids(tokenizer)
    max_new_tokens = _generation_cap(tokenizer, train_rows)
    # A LoRA run saves only the adapter (load base + adapter); a full
    # fine-tune saves the whole model (load it directly). The harness used to
    # assume LoRA and crashed on full fine-tunes — the quickstart default.
    # ``model_dir=None`` scores the untouched base model (what a RAG-first
    # project serves: base + retrieval, no fine-tune).
    is_adapter_run = model_dir is not None and (model_dir / "adapter_config.json").exists()
    # CUDA in fp16; CPU in fp32 (bf16/fp16 are emulated — very slow — on
    # CPUs without native support, and the CI gate runs on CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kwargs: dict[str, Any] = (
        {"dtype": torch.float16, "device_map": "cuda"} if device == "cuda" else {"dtype": torch.float32}
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model if (is_adapter_run or model_dir is None) else str(model_dir),
        **load_kwargs,
    )
    model = PeftModel.from_pretrained(base, str(model_dir)) if is_adapter_run else base
    model = model.to(device).eval()

    f1s: list[float] = []
    records: list[dict[str, Any]] = []
    condition_label = "with-RAG" if with_rag else "without-RAG"
    # Total = the number of val rows we'll actually score (skipping
    # any malformed ones). Cheaper to pre-count than to publish a
    # moving total.
    scoreable_total = sum(
        1
        for r in val_rows
        if str(r.get("question") or "").strip() and str(r.get("answer") or "").strip()
    )
    scored = 0
    for row in val_rows:
        question = str(row.get("question") or "").strip()
        reference = str(row.get("answer") or "").strip()
        if not question or not reference:
            continue
        retrieved_pairs: list[dict[str, Any]] = []
        preamble: str | None = None
        retrieved_sources: list[str] = []
        if with_rag:
            try:
                hits = retrieve_ranked(question, index_dir=index_dir, k=rag_k, reranker=reranker)
            except AutoRagUnavailable:
                hits = []
            if use_documents:
                from app.services.auto_rag_service import (
                    _DOCUMENT_PREAMBLE_TEMPLATE,
                    _format_passage,
                )

                if hits:
                    passages_text = "\n\n".join(
                        _format_passage(idx, hit) for idx, hit in enumerate(hits, start=1)
                    )
                    preamble = _DOCUMENT_PREAMBLE_TEMPLATE.format(passages=passages_text)
                    for hit in hits:
                        payload = hit.get("payload") or {}
                        chunk = payload.get("chunk_id")
                        where = f"{payload.get('source_doc') or 'document'}"
                        if isinstance(chunk, int):
                            where += f" · passage {chunk + 1}"
                        retrieved_sources.append(where)
                        retrieved_pairs.append({"question": "", "answer": str(payload.get("text") or "")})
            else:
                for hit in hits:
                    payload = hit.get("payload") or {}
                    retrieved_pairs.append({
                        "question": payload.get("question", ""),
                        "answer": payload.get("answer", ""),
                    })
        prompt, used_template = _build_inference_prompt(
            tokenizer,
            question,
            (retrieved_pairs if with_rag and not use_documents else None),
            preamble=preamble,
        )
        # A rendered chat template already carries its special tokens.
        inputs = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=not used_template
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_ids or None,
            )
        if used_template:
            # Only the newly generated tokens are the answer.
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
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
            "retrieved_sources": retrieved_sources,
        })
        scored += 1
        # Per-row progress hook — used by the API Job runner to
        # publish "scoring row 12/28 (with-RAG)" into the bell.
        # Best-effort; a buggy callback never blocks the scoring loop.
        if progress_callback is not None:
            try:
                progress_callback(scored, scoreable_total, condition_label)
            except Exception:  # noqa: BLE001 — observability is non-load-bearing
                pass

    # Free GPU memory so the next seed can fresh-load.
    del model, base
    if device == "cuda":
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
        description="A/B harness for auto-RAG (Phase 9c) + per-project comparison cache (Phase 9d)."
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
    parser.add_argument(
        "--project", type=int, default=None,
        help=(
            "Phase 9d per-project mode. Runs ONE A/B (1 seed, training "
            "already done — points at the project's latest experiment's "
            "model_dir) and writes the comparison to data/projects/"
            "{project_id}/auto_rag/comparison.json so the Eval-tab "
            "AutoRagComparisonPanel can render it. Skips the multi-"
            "template gate flow used for Phase 9c."
        ),
    )
    parser.add_argument(
        "--corpus", choices=("qa", "documents"), default="qa",
        help=(
            "With --project: what the with-retrieval arm retrieves. 'qa' = the "
            "project's training Q&A pairs (default, the playground's Q&A index); "
            "'documents' = the project's cleaned document passages with the "
            "cite-or-say-you-don't-know preamble (what a documents-only project serves)."
        ),
    )
    parser.add_argument("--k", type=int, default=RAG_K, help="With --project: BM25 top-k for the with-retrieval arm.")
    parser.add_argument(
        "--reranker", type=str, default=None,
        help="With --project: cross-encoder model id to rerank BM25 candidates ('default' = cross-encoder/ms-marco-MiniLM-L-6-v2).",
    )
    parser.add_argument(
        "--sweep-retrieval", action="store_true",
        help="With --project: try top-3 / top-5, each with and without the reranker; keep the judge's best as the comparison and write auto_rag/retrieval_sweep.json.",
    )
    parser.add_argument(
        "--split", choices=("val", "test"), default=None,
        help=(
            "With --project: rows to score. Default: val for Q&A-pair retrieval, "
            "test for --corpus documents (same rows as the lift check)."
        ),
    )
    parser.add_argument(
        "--base-only", action="store_true",
        help=(
            "With --project: score the project's BASE model with and without "
            "retrieval (no fine-tuned run needed) and write "
            "auto_rag/comparison_base.json. Answers 'does retrieval alone "
            "help?' — the RAG-first question."
        ),
    )
    return parser.parse_args(argv)


def run_project_comparison(
    project_id: int,
    *,
    seed: int = 0,
    progress_callback: "Callable[[int, int, str], None] | None" = None,
    base_only: bool = False,
    corpus: str = "qa",
    split: str | None = None,
    rag_k: int = RAG_K,
    reranker: str | None = None,
    sweep_retrieval: bool = False,
) -> dict[str, Any]:
    """Phase 9d — generate the per-project comparison the Eval-tab
    panel reads. Reuses the per-row eval inference loop with the
    project's latest COMPLETED experiment's model_dir. Writes
    ``data/projects/{project_id}/auto_rag/comparison.json``.

    ``base_only=True`` scores the project's **base model** with and without
    retrieval instead of the latest fine-tuned run — the question a
    RAG-first project asks ("does retrieval alone get me there?"). A model
    fine-tuned on bare question → answer pairs never saw a retrieval
    preamble, so its with-RAG arm understates what retrieval can do. The
    result goes to ``comparison_base.json`` (``"model": "base"``) and never
    overwrites the fine-tuned comparison the panel reads.

    ``corpus="documents"`` retrieves document passages instead of Q&A pairs
    (``evaluate_with_inference``); the result is stamped ``corpus`` and goes
    to ``comparison_base_documents.json`` / ``comparison_documents.json``.
    Both arms are also scored by the answer judge when one applies
    (``_judge_comparison_rows``) — on long answers the judge's
    correct / partial / wrong is the number to read, not F1.

    ``split``: the rows scored — ``"val"`` (default for Q&A-pair retrieval)
    or ``"test"`` (default for document passages, so the passages verdict
    pairs the same rows as the lift check, which scores the test split).

    ``rag_k`` / ``reranker``: the retrieval the with-RAG arm uses (BM25 top-k,
    optionally cross-encoder reranked — ``retrieval_reranker``).
    ``sweep_retrieval=True`` scores the with-RAG arm under every
    ``RETRIEVAL_SWEEP`` config (the without-RAG arm is shared), judges each,
    keeps the config the judge scores highest (ties → the cheaper one) as
    the comparison, and records all of them in ``summary["retrieval_sweep"]``
    + ``auto_rag/retrieval_sweep.json``. The chosen config is what the
    project should serve (``runtime_config.auto_rag_retrieval``).

    This function is invoked from the CLI's ``--project`` mode (and
    can be called programmatically by a future API trigger if we
    decide to make the comparison runnable from the UI). For now,
    it's an opt-in manual step the user runs via the CLI.
    """
    import sqlite3
    from datetime import datetime, timezone
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.config import settings
    from app.services.evaluation_service import f1_score

    # Reach into the SQLite directly — avoids the async-DB session
    # complexity for a CLI-only path. The schema is stable.
    db_path = str(settings.DATABASE_URL).replace("sqlite+aiosqlite:///", "")
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT id, output_dir, base_model FROM experiments "
            "WHERE project_id = ? AND status = 'COMPLETED' "
            "ORDER BY completed_at DESC LIMIT 1",
            (project_id,),
        )
        exp_row = cur.fetchone()
        project_row = con.execute(
            "SELECT base_model_name FROM projects WHERE id = ?", (project_id,)
        ).fetchone()

    model_dir: Path | None
    if base_only:
        base_model = str((project_row["base_model_name"] if project_row else "") or "").strip()
        if not base_model and exp_row is not None:
            base_model = str(exp_row["base_model"])
        if not base_model:
            raise RuntimeError(
                f"Project {project_id} has no base model. Choose a task type "
                f"(it suggests one) or train a run first."
            )
        model_dir = None
    else:
        if exp_row is None:
            raise RuntimeError(
                f"No COMPLETED experiment found for project {project_id}. "
                f"Train a QA-SFT experiment first."
            )
        model_dir = Path(str(exp_row["output_dir"])) / "model"
        base_model = str(exp_row["base_model"])
        if not model_dir.exists():
            raise RuntimeError(f"Trained model dir missing at {model_dir}.")

    # The project's prepared train split sizes generation + (Q&A mode)
    # feeds the index; the scored rows come from val or test.
    split = split or ("test" if corpus == "documents" else "val")
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    train_file = prepared_dir / "train.jsonl"
    val_file = prepared_dir / f"{split}.jsonl"
    if not train_file.exists() or not val_file.exists():
        raise RuntimeError(
            f"Prepared train/{split} missing at {prepared_dir}. "
            f"Run dataset prep first."
        )
    with train_file.open(encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f if line.strip()]
    with val_file.open(encoding="utf-8") as f:
        val_rows = [json.loads(line) for line in f if line.strip()]

    print(f"[harness] project={project_id} model={model_dir or base_model + ' (base model)'}")
    print(f"[harness] train_rows={len(train_rows)} {split}_rows={len(val_rows)}")

    # Use the project's DEPLOYED BM25 index (built at training-
    # completion by Phase 9b's hook over the training rows). This is what the playground
    # actually reads at inference time — so the comparison's lift
    # numbers predict real playground behavior. Falls through to the
    # Phase 9c transient-build path only when this project hasn't
    # had the index built yet (rare; loud error tells the user to
    # train + let the hook fire).
    project_index_dir = (
        settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
    )
    # An index built before the corpus excluded the answer key (or never
    # built) is rebuilt over the prepared train split — the same rows the
    # training-completion hook indexes. Retrieval must not be able to return
    # a val row (or a gold row) together with its reference answer.
    from app.services.auto_rag_service import (
        QA_CORPUS_SOURCE,
        build_bm25_index,
        qa_index_is_current,
    )
    project_index_path = project_index_dir / "bm25_index.json"
    if corpus == "documents":
        from app.services.auto_rag_service import ensure_document_index

        doc_index = ensure_document_index(project_id)
        if not doc_index.get("available"):
            raise RuntimeError(
                f"Project {project_id} has no cleaned document passages to retrieve from."
            )
        print(f"[harness] document index: {doc_index.get('passages')} passages")
    elif not qa_index_is_current(project_index_path):
        recipe_id = "qa-sft"
        if project_index_path.exists():
            try:
                recipe_id = str(
                    json.loads(project_index_path.read_text(encoding="utf-8")).get("recipe_id")
                    or recipe_id
                )
            except (json.JSONDecodeError, OSError):
                pass
        build_bm25_index(
            train_rows,
            recipe_id=recipe_id,
            output_dir=project_index_dir,
            corpus_source=QA_CORPUS_SOURCE,
        )
        print(f"[harness] rebuilt auto-RAG index over {len(train_rows)} train rows")
    off_f1s, off_records = evaluate_with_inference(
        base_model=base_model, model_dir=model_dir,
        val_rows=val_rows, train_rows=train_rows, with_rag=False,
        index_dir_override=project_index_dir,
        progress_callback=progress_callback,
        corpus=corpus,
    )
    configs = list(RETRIEVAL_SWEEP) if sweep_retrieval else [{"k": int(rag_k), "reranker": reranker}]
    arms: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        label = retrieval_label(config)
        if progress_callback is not None and sweep_retrieval:
            try:
                progress_callback(0, 0, f"retrieval {index + 1}/{len(configs)}: {label}")
            except Exception:  # noqa: BLE001
                pass
        arm_f1s, arm_records = evaluate_with_inference(
            base_model=base_model, model_dir=model_dir,
            val_rows=val_rows, train_rows=train_rows, with_rag=True,
            index_dir_override=project_index_dir,
            progress_callback=progress_callback,
            corpus=corpus,
            rag_k=int(config["k"]),
            reranker=config.get("reranker"),
        )
        if progress_callback is not None:
            try:
                progress_callback(0, 0, f"judging ({label})")
            except Exception:  # noqa: BLE001
                pass
        # The judge sees the shared without-RAG answers each time; its cache
        # makes the repeats free.
        arm_judge = _judge_comparison_rows(project_id, off_records, arm_records)
        from app.services.retrieval_reranker import normalize_reranker

        arms.append({
            "retrieval": {"k": int(config["k"]), "reranker": normalize_reranker(config.get("reranker"))},
            "label": label,
            "f1s": arm_f1s,
            "records": arm_records,
            "judge": arm_judge,
            "on_mean_f1": statistics.mean(arm_f1s) if arm_f1s else 0.0,
            "judge_score": ((arm_judge or {}).get("with_rag") or {}).get("score"),
        })
    best = pick_best_retrieval(arms)
    on_f1s, on_records, judge_summary = best["f1s"], best["records"], best["judge"]
    retrieval_used = best["retrieval"]
    retrieval_sweep = (
        [
            {
                "retrieval": arm["retrieval"],
                "label": arm["label"],
                "on_mean_f1": round(arm["on_mean_f1"], 4),
                "judge_score": arm["judge_score"],
                "judge_counts": ((arm["judge"] or {}).get("with_rag") or {}).get("counts"),
                "chosen": arm is best,
            }
            for arm in arms
        ]
        if sweep_retrieval
        else None
    )
    if sweep_retrieval:
        for arm in retrieval_sweep or []:
            print(f"[harness] sweep {arm['label']}: judge={arm['judge_score']} f1={arm['on_mean_f1']}{'  <- chosen' if arm['chosen'] else ''}")

    off_mean = statistics.mean(off_f1s) if off_f1s else 0.0
    on_mean = statistics.mean(on_f1s) if on_f1s else 0.0
    lift = (on_mean - off_mean) / off_mean * 100.0 if off_mean else None

    # Combine per-row records into one list (off + on side-by-side
    # by row index) — the UI panel renders an expandable card per
    # row showing both generations + the retrieved chunks.
    combined_rows: list[dict[str, Any]] = []
    for off_r, on_r in zip(off_records, on_records):
        combined_rows.append({
            "question": off_r["question"],
            "reference": off_r["reference"],
            "without_rag": {
                "generated": off_r["generated"],
                "f1": off_r["f1"],
                "judge": off_r.get("judge"),
            },
            "with_rag": {
                "generated": on_r["generated"],
                "f1": on_r["f1"],
                "judge": on_r.get("judge"),
                "retrieved_row_count": on_r["retrieved_row_count"],
                "retrieved_sources": on_r.get("retrieved_sources") or [],
            },
        })

    payload = {
        "project_id": project_id,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": None if base_only else int(exp_row["id"]),
        "model": "base" if base_only else "fine_tuned",
        "corpus": corpus,
        "split": split,
        "base_model": base_model,
        "model_dir": None if base_only else str(model_dir),
        "summary": {
            "off_mean_f1": off_mean,
            "on_mean_f1": on_mean,
            "absolute_lift": on_mean - off_mean,
            "relative_lift_pct": lift,
            "n_val_rows": len(off_records),
            "rag_k": retrieval_used["k"],
            "retrieval": retrieval_used,
            "retrieval_sweep": retrieval_sweep,
            "phase_9c_reference_lift_pct": 146.49,
            "judge": judge_summary,
        },
        "rows": combined_rows,
    }
    if retrieval_sweep:
        sweep_path = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / "retrieval_sweep.json"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(json.dumps({
            "project_id": project_id, "corpus": corpus, "split": split, "base_model": base_model,
            "model": "base" if base_only else "fine_tuned",
            "cached_at": datetime.now(timezone.utc).isoformat(), "arms": retrieval_sweep,
        }, indent=2), encoding="utf-8")
    cache_path = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / comparison_file_name(
        base_only=base_only, corpus=corpus
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[harness] wrote comparison to {cache_path}")
    print(f"[harness] off_mean={off_mean:.4f}  on_mean={on_mean:.4f}  lift={lift:.2f}%")
    if judge_summary:
        print(
            f"[harness] judge={judge_summary['judge']}  "
            f"off={judge_summary['without_rag'].get('score')}  on={judge_summary['with_rag'].get('score')}"
        )
    return payload


# The retrieval configs a sweep tries, cheapest first (ties go to the
# earlier one): BM25 top-3 / top-5, each with and without the cross-encoder.
RETRIEVAL_SWEEP: tuple[dict[str, Any], ...] = (
    {"k": 3, "reranker": None},
    {"k": 5, "reranker": None},
    {"k": 3, "reranker": "default"},
    {"k": 5, "reranker": "default"},
)


def retrieval_label(config: dict[str, Any]) -> str:
    from app.services.retrieval_reranker import normalize_reranker

    name = normalize_reranker(config.get("reranker"))
    return f"top-{int(config['k'])}" + (f" + reranker {name.split('/')[-1]}" if name else "")


def pick_best_retrieval(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """The arm the judge scores highest; without a judge, the highest mean
    F1. Ties keep the earlier (cheaper) arm — strict ``>`` only."""
    best = arms[0]
    for arm in arms[1:]:
        if best.get("judge_score") is not None and arm.get("judge_score") is not None:
            if arm["judge_score"] > best["judge_score"]:
                best = arm
        elif arm.get("judge_score") is not None and best.get("judge_score") is None:
            best = arm
        elif best.get("judge_score") is None and arm["on_mean_f1"] > best["on_mean_f1"]:
            best = arm
    return best


def comparison_file_name(*, base_only: bool, corpus: str = "qa") -> str:
    """``comparison[_base][_documents].json`` — the four cached comparisons
    never overwrite each other."""
    name = "comparison_base" if base_only else "comparison"
    if corpus == "documents":
        name += "_documents"
    return name + ".json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Phase 9d per-project mode — short-circuits the template gate
    # flow and writes the cached comparison for the Eval-tab panel.
    if args.project is not None:
        payload = run_project_comparison(
            args.project, base_only=bool(args.base_only), corpus=str(args.corpus), split=args.split,
            rag_k=int(args.k), reranker=args.reranker, sweep_retrieval=bool(args.sweep_retrieval),
        )
        if args.sweep_retrieval:
            chosen = (payload.get("summary") or {}).get("retrieval")
            print(f"[harness] chosen retrieval: {chosen} — apply with runtime_config.auto_rag_retrieval")
        return 0

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
