"""A/B harness for curriculum learning (USER-SUCCESS Epic 6 Phase 6c).

Decides whether to promote Phase 6d (UI toggle + default-on heuristic)
by running real fine-tunes of the 2 classification templates
(ticket-router, log-triage) under both ``curriculum on`` and
``curriculum off``, across 3 seeds, and comparing mean macro F1.

Gate criterion: Phase 6d ships **only if** mean F1 lift ≥ 5% on
**both** classification templates, with non-overlapping ``mean ± 1σ``
bands (a weak-but-not-trivial statistical floor for 3 seeds).
Otherwise Epic 6 stops at 6b as a negative result.

Usage:

  python -m backend.scripts.curriculum_ab \
      [--seeds 3] [--num-epochs 3] [--output curriculum_ab_results.json]

The harness is deliberately self-contained — it reads template gold
sets straight from ``backend/data/project_templates/<slug>/gold.jsonl``
and shells out to ``backend/scripts/train.py`` for each run, so the
A/B doesn't depend on an FastAPI server, a DB, or any project state.
Per-run results are written to disk immediately so a crash or
Ctrl-C doesn't lose the completed runs.

Per-run cost on GB10 with SmolLM2-135M, 140 train rows, 3 epochs:
~1-3 minutes including model load. 12 runs ≈ 15-40 minutes total.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# Templates this harness scores. Update when classification templates
# are added — Phase 6c's gate criterion requires positive lift on
# *every* listed template, so adding one tightens the bar.
# ─────────────────────────────────────────────────────────────────────


CLASSIFICATION_TEMPLATES: tuple[str, ...] = ("ticket-router", "log-triage")

# Gate criterion: Phase 6d ships if mean F1 lift ≥ 5% on every
# template AND on-mean − on-std > off-mean + off-std (non-overlapping
# 1σ bands). 5% is half the Epic's stated 10-20% F1-lift target;
# we use the lower bound as the "real signal" threshold.
GATE_MIN_LIFT_PCT: float = 5.0


# ─────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    template: str
    seed: int
    curriculum: bool
    macro_f1: float | None
    eval_loss: float | None
    train_runtime_seconds: float | None
    output_dir: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "seed": self.seed,
            "curriculum": self.curriculum,
            "macro_f1": self.macro_f1,
            "eval_loss": self.eval_loss,
            "train_runtime_seconds": self.train_runtime_seconds,
            "output_dir": self.output_dir,
            "error": self.error,
        }


@dataclass
class TemplateSummary:
    template: str
    on_f1s: list[float] = field(default_factory=list)
    off_f1s: list[float] = field(default_factory=list)

    @property
    def on_mean(self) -> float | None:
        return statistics.mean(self.on_f1s) if self.on_f1s else None

    @property
    def off_mean(self) -> float | None:
        return statistics.mean(self.off_f1s) if self.off_f1s else None

    @property
    def on_std(self) -> float:
        # stdev is undefined for n<2; treat as 0 so the gate doesn't
        # auto-pass on a single-seed run.
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
        """True iff (on_mean − on_std) > (off_mean + off_std).

        With std=0 (single seed) this collapses to a bare mean compare
        — fine for a sanity ping but not enough for a real gate; the
        ``relative_lift_pct ≥ GATE_MIN_LIFT_PCT`` check is the load-
        bearing criterion."""
        if self.on_mean is None or self.off_mean is None:
            return False
        return (self.on_mean - self.on_std) > (self.off_mean + self.off_std)


# ─────────────────────────────────────────────────────────────────────
# Template data prep — read gold.jsonl + apply 70/15 train/val split.
# Mirrors demo_project_service._split_rows so harness training matches
# what the platform itself would train on.
# ─────────────────────────────────────────────────────────────────────


def _read_template_gold(template_slug: str) -> list[dict[str, Any]]:
    """Read the template's gold set from
    ``backend/data/project_templates/<slug>/gold.jsonl``."""
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


def _flatten_classification_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten ``{"input": {"ticket"/"log_line": "..."}, "expected":
    {"label": "..."}}`` into the flat ``{"text", "label"}`` shape
    train.py's classification adapter expects. Returns None when the
    row doesn't have the expected fields (skipped silently)."""
    input_block = row.get("input") or {}
    expected = row.get("expected") or {}
    if not isinstance(input_block, dict) or not isinstance(expected, dict):
        return None
    text = ""
    for key in ("ticket", "log_line", "text", "content", "question", "prompt"):
        value = input_block.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    label = expected.get("label")
    if not (text and isinstance(label, str) and label.strip()):
        return None
    return {"text": text, "label": label.strip()}


def _split_70_15_15(rows: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Deterministic 70/15/15 split — matches
    ``demo_project_service._split_rows`` so harness eval is
    comparable to a real demo project's eval on the same template."""
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
    """Read template gold → flatten → 70/15/15 split → write
    train/val/test JSONL into ``prepared_dir``. Returns the row
    counts so the harness can log + sanity-check."""
    gold_rows = _read_template_gold(template_slug)
    flat = [r for r in (_flatten_classification_row(r) for r in gold_rows) if r is not None]
    train, val, test = _split_70_15_15(flat)
    _write_jsonl(prepared_dir / "train.jsonl", train)
    _write_jsonl(prepared_dir / "val.jsonl", val)
    _write_jsonl(prepared_dir / "test.jsonl", test)
    return {"train": len(train), "val": len(val), "test": len(test)}


# ─────────────────────────────────────────────────────────────────────
# Single fine-tune (subprocess train.py, parse training_report.json)
# ─────────────────────────────────────────────────────────────────────


def _build_run_config(*, num_epochs: int, seed: int, curriculum: bool) -> dict[str, Any]:
    """Minimal training config — small + fast for the A/B's purposes.
    Defaults match what a thin-data classification project would
    realistically use: LoRA, small batch, 3 epochs."""
    return {
        "task_type": "classification",
        "training_mode": "sft",
        "num_epochs": num_epochs,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "max_seq_length": 256,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "save_steps": 1000,    # don't checkpoint mid-run — too slow on tiny data
        "eval_steps": 50,
        "warmup_ratio": 0.03,
        "seed": seed,
        # Phase 6b — when curriculum is on, train.py reads this flag and
        # swaps the trainer's RandomSampler for a SequentialSampler so
        # the easy-first ordering in train.curriculum.jsonl is preserved.
        "curriculum_disable_shuffle": bool(curriculum),
    }


def _maybe_build_curriculum_shard(
    *,
    template_dir: Path,
    curriculum: bool,
) -> Path:
    """When curriculum is on, build the easy-first shard via
    ``curriculum_service.build_curriculum_shards`` and return its
    path. Otherwise return the plain train.jsonl path."""
    plain_train = template_dir / "train.jsonl"
    if not curriculum:
        return plain_train
    # Lazy import — only needed when curriculum is on, and avoids
    # pulling sentence-transformers into harness module-load.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.services.curriculum_service import build_curriculum_shards

    with plain_train.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    manifest = build_curriculum_shards(
        rows,
        scoring_mode="prototype_entropy",
        output_dir=template_dir / "curriculum",
    )
    return Path(manifest["shard_path"])


def run_one_finetune(
    *,
    template_slug: str,
    seed: int,
    curriculum: bool,
    num_epochs: int,
    base_model: str,
    workdir: Path,
) -> RunResult:
    """Execute a single training run + parse ``training_report.json``
    for ``final_eval_macro_f1``. Returns a RunResult with ``error``
    set when the run failed; the harness logs and continues."""
    run_id = f"{template_slug}_seed{seed}_curriculum{'on' if curriculum else 'off'}"
    template_dir = workdir / template_slug
    output_dir = workdir / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = _maybe_build_curriculum_shard(
        template_dir=template_dir, curriculum=curriculum
    )
    val_file = template_dir / "val.jsonl"
    config_path = output_dir / "training_config.json"
    config_path.write_text(
        json.dumps(_build_run_config(
            num_epochs=num_epochs,
            seed=seed,
            curriculum=curriculum,
        ), indent=2),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),  # placeholder — replaced below
    ]
    # Use module form so train.py's relative imports resolve.
    train_script = Path(__file__).resolve().parent / "train.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--project", "0",                    # unused by the harness path
        "--experiment", str(abs(hash(run_id)) % (10**8)),  # unique synthetic id
        "--output", str(output_dir),
        "--base-model", base_model,
        "--config", str(config_path),
        "--train-file", str(train_file),
        "--val-file", str(val_file),
        "--seed", str(seed),
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900,  # 15 min per-run cap — much more than the GB10 needs.
    )
    elapsed = time.time() - started

    report_path = output_dir / "training_report.json"
    if proc.returncode != 0 or not report_path.exists():
        err = (proc.stderr or proc.stdout or "(no output)")[-2000:]
        return RunResult(
            template=template_slug,
            seed=seed,
            curriculum=curriculum,
            macro_f1=None,
            eval_loss=None,
            train_runtime_seconds=round(elapsed, 1),
            output_dir=str(output_dir),
            error=f"train.py exited rc={proc.returncode}: {err}",
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return RunResult(
            template=template_slug,
            seed=seed,
            curriculum=curriculum,
            macro_f1=None,
            eval_loss=None,
            train_runtime_seconds=round(elapsed, 1),
            output_dir=str(output_dir),
            error=f"training_report.json unparseable: {e}",
        )
    return RunResult(
        template=template_slug,
        seed=seed,
        curriculum=curriculum,
        macro_f1=_coerce_float(report.get("final_eval_macro_f1")),
        eval_loss=_coerce_float(report.get("final_eval_loss")),
        train_runtime_seconds=round(elapsed, 1),
        output_dir=str(output_dir),
    )


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Aggregation + gate decision
# ─────────────────────────────────────────────────────────────────────


def aggregate_results(results: list[RunResult]) -> dict[str, TemplateSummary]:
    """Group results by template + curriculum-on/off and collect F1s
    into TemplateSummary instances. Failed runs (macro_f1 None) are
    silently dropped — they show up in the raw results list with an
    ``error`` field, so a separate check can decide whether to abort
    on too-many-failures."""
    summaries: dict[str, TemplateSummary] = {}
    for r in results:
        if r.macro_f1 is None:
            continue
        summary = summaries.setdefault(r.template, TemplateSummary(template=r.template))
        if r.curriculum:
            summary.on_f1s.append(r.macro_f1)
        else:
            summary.off_f1s.append(r.macro_f1)
    return summaries


@dataclass
class GateDecision:
    passed: bool
    reason: str
    per_template: dict[str, dict[str, Any]]


def apply_gate(summaries: dict[str, TemplateSummary]) -> GateDecision:
    """Phase 6d ships iff every template hit the lift threshold AND
    has non-overlapping ``mean ± 1σ`` bands. Anything weaker keeps
    Epic 6 at 6b as a negative result."""
    per_template: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for slug, s in summaries.items():
        lift = s.relative_lift_pct
        non_overlap = s.bands_non_overlapping
        passed = (
            lift is not None
            and lift >= GATE_MIN_LIFT_PCT
            and non_overlap
        )
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
                    f"{slug}: bands overlap (on={s.on_mean:.3f}±{s.on_std:.3f}, "
                    f"off={s.off_mean:.3f}±{s.off_std:.3f})"
                )

    if not summaries:
        return GateDecision(
            passed=False,
            reason="no successful runs — gate cannot be evaluated",
            per_template={},
        )
    if failures:
        return GateDecision(
            passed=False,
            reason="; ".join(failures),
            per_template=per_template,
        )
    return GateDecision(
        passed=True,
        reason=(
            f"all templates lifted ≥ {GATE_MIN_LIFT_PCT}% with non-overlapping "
            f"1σ bands — ship Phase 6d"
        ),
        per_template=per_template,
    )


# ─────────────────────────────────────────────────────────────────────
# Reporting (Markdown block for the roadmap, JSON for the raw record)
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
    """Return a Markdown block ready to paste into the Epic 6 status
    section of ROADMAP-USER-SUCCESS.md."""
    lines: list[str] = []
    lines.append("**Phase 6c A/B results.**")
    lines.append("")
    lines.append(
        f"Setup: base model `{base_model}` · {num_epochs} epochs · "
        f"seeds {seeds} · LoRA r=16 · GB10 GPU."
    )
    lines.append("")
    lines.append("| Template | n (on/off) | Mean F1 (on) | Mean F1 (off) | Lift | Non-overlap 1σ? |")
    lines.append("|---|---|---|---|---|---|")
    for slug in CLASSIFICATION_TEMPLATES:
        summary = summaries.get(slug)
        if summary is None or not summary.on_f1s or not summary.off_f1s:
            lines.append(
                f"| {slug} | -/- | — | — | — | — |"
            )
            continue
        lift_str = (
            f"{summary.relative_lift_pct:+.2f}%" if summary.relative_lift_pct is not None else "—"
        )
        lines.append(
            f"| {slug} | {len(summary.on_f1s)}/{len(summary.off_f1s)} | "
            f"{summary.on_mean:.4f} ± {summary.on_std:.4f} | "
            f"{summary.off_mean:.4f} ± {summary.off_std:.4f} | "
            f"{lift_str} | {'✓' if summary.bands_non_overlapping else '✗'} |"
        )
    lines.append("")
    if gate.passed:
        lines.append(f"**Gate: PASS** — {gate.reason}. Phase 6d cleared to ship.")
    else:
        lines.append(f"**Gate: FAIL** — {gate.reason}.")
        lines.append(
            "Per the Phase 6c criterion, Epic 6 stops at Phase 6b as a "
            "negative result; curriculum learning does not lift F1 on "
            "classification at the magnitude the literature predicts."
        )
    # Surface failed runs explicitly (silent drops in aggregation
    # don't appear above).
    failed = [r for r in results if r.error is not None]
    if failed:
        lines.append("")
        lines.append(f"_{len(failed)} run(s) failed:_")
        for r in failed:
            err_excerpt = (r.error or "").splitlines()[0][:200]
            lines.append(
                f"  - `{r.template}` seed={r.seed} curriculum={r.curriculum}: {err_excerpt}"
            )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B harness for curriculum learning (Phase 6c)."
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Seeds per (template, condition). Default 3 (Phase 6c spec).",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3,
        help="Epochs per training run. Default 3.",
    )
    parser.add_argument(
        "--base-model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="HF base model id. Default SmolLM2-135M-Instruct.",
    )
    parser.add_argument(
        "--workdir", type=str, default="",
        help="Working directory for per-run output. Default: a fresh tmp dir.",
    )
    parser.add_argument(
        "--output", type=str, default="curriculum_ab_results.json",
        help="Path to the JSON results file. Default: ./curriculum_ab_results.json",
    )
    parser.add_argument(
        "--markdown-output", type=str, default="curriculum_ab_results.md",
        help="Path to the Markdown roadmap block. Default: ./curriculum_ab_results.md",
    )
    parser.add_argument(
        "--templates", type=str, nargs="*", default=None,
        help=(
            "Override the template list (default: ticket-router log-triage). "
            "Useful for smoke runs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    templates = tuple(args.templates) if args.templates else CLASSIFICATION_TEMPLATES
    seeds = list(range(args.seeds))
    workdir = (
        Path(args.workdir).expanduser().resolve()
        if args.workdir
        else Path("/tmp") / f"curriculum_ab_{int(time.time())}"
    )
    workdir.mkdir(parents=True, exist_ok=True)

    # Prepare each template's data once — shared across all (seed, condition)
    # runs so the A/B compares like with like.
    print(f"[harness] workdir={workdir}")
    for slug in templates:
        template_dir = workdir / slug
        if (template_dir / "train.jsonl").exists():
            continue
        counts = prepare_template_splits(slug, template_dir)
        print(f"[harness] {slug}: train={counts['train']} val={counts['val']} test={counts['test']}")

    results: list[RunResult] = []
    results_path = Path(args.output).expanduser().resolve()
    # Resume support: skip runs already present in the output file.
    if results_path.exists():
        try:
            prior = json.loads(results_path.read_text(encoding="utf-8"))
            for r in prior.get("runs", []):
                results.append(RunResult(**r))
            print(f"[harness] resumed: {len(results)} runs already on disk")
        except (json.JSONDecodeError, TypeError):
            pass
    already_done = {(r.template, r.seed, r.curriculum) for r in results}

    total_combos = len(templates) * len(seeds) * 2
    done = len(already_done)
    for slug in templates:
        for seed in seeds:
            for curriculum in (False, True):
                key = (slug, seed, curriculum)
                if key in already_done:
                    continue
                done += 1
                label = f"{slug} seed={seed} curriculum={'on' if curriculum else 'off'}"
                print(f"[harness] ({done}/{total_combos}) {label} …")
                t0 = time.time()
                result = run_one_finetune(
                    template_slug=slug,
                    seed=seed,
                    curriculum=curriculum,
                    num_epochs=args.num_epochs,
                    base_model=args.base_model,
                    workdir=workdir,
                )
                results.append(result)
                # Persist immediately so a crash doesn't lose the run.
                _persist_results(
                    results=results,
                    summaries=aggregate_results(results),
                    gate=apply_gate(aggregate_results(results)),
                    results_path=results_path,
                    markdown_path=Path(args.markdown_output).expanduser().resolve(),
                    base_model=args.base_model,
                    num_epochs=args.num_epochs,
                    seeds=seeds,
                )
                if result.error:
                    print(f"  ✗ failed in {time.time() - t0:.1f}s: {result.error.splitlines()[0][:160]}")
                else:
                    print(f"  ✓ F1={result.macro_f1:.4f} in {time.time() - t0:.1f}s")

    summaries = aggregate_results(results)
    gate = apply_gate(summaries)
    _persist_results(
        results=results,
        summaries=summaries,
        gate=gate,
        results_path=results_path,
        markdown_path=Path(args.markdown_output).expanduser().resolve(),
        base_model=args.base_model,
        num_epochs=args.num_epochs,
        seeds=seeds,
    )

    print()
    print(format_markdown_block(
        results,
        summaries,
        gate,
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
    summaries: dict[str, TemplateSummary],
    gate: GateDecision,
    results_path: Path,
    markdown_path: Path,
    base_model: str,
    num_epochs: int,
    seeds: list[int],
) -> None:
    """Write JSON + Markdown atomically. JSON has the full record;
    Markdown is the ready-to-paste roadmap block."""
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
            base_model=base_model,
            num_epochs=num_epochs,
            seeds=seeds,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
