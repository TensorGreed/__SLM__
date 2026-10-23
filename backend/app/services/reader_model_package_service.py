"""Model card for a passage reader — a causal LM trained with retrieved
passages in its prompt (``passage_prompts_service``).

A span tagger gets its card from ``span_model_package_service``; every other
export used to ship without one, so a published reader had a blank Hub page
until a card was hand-written. This writes ``README.md`` into the exported
``model/`` directory, built ONLY from evidence the run itself owns:

* the run's lift check — ``auto_rag/passage_reader/<run>.json``
  (``passage_reader_eval_service``): base + passages (the bar) vs this run
  + passages, both without passages too, the paired CI and the retrieval
  judge's found / missed split. Never the project-wide
  ``comparison*_documents.json`` files: any later comparison overwrites
  them, on any split;
* the run's passage-prompt report (``passage_prompts/report.json``) — the
  retrieval it trained with, the answer targets, the row counts;
* the run's training file, tallied only while its sha256 still matches the
  training report (question generators, source documents);
* ``training_report.json`` + the adapter's ``adapter_config.json``.

The headline claim reads the paired evidence's ``verdict`` (CI-based), never
the raw score delta. A run with no lift check says so instead of dropping
the section. ``build_reader_model_card`` is pure; ``package_passage_reader``
gathers the inputs and writes the file.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.models.experiment import Experiment

MODEL_CARD_FILE = "README.md"
MAX_SOURCES_LISTED = 8

_ARMS = (
    ("base_with_passages", "Base model + passages (the bar)"),
    ("finetune_with_passages", "**This model + passages**"),
    ("finetune_without_passages", "This model, no passages"),
    ("base_without_passages", "Base model, no passages"),
)

_VERDICT_TEXT = {
    "better": "better than the bar: the 95% CI of the paired difference is above 0",
    "worse": "worse than the bar: the 95% CI of the paired difference is below 0",
    "within_noise": "not distinguishable from the bar: the 95% CI of the paired difference includes 0",
    "too_few_rows": "not decidable: too few paired rows for a confidence interval",
}


def _is_hub_id(name: str | None) -> bool:
    if not name:
        return False
    return not (name.startswith(("/", ".", "~")) or "\\" in name) and name.count("/") <= 1


def _score(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "–"


def _signed(value: Any) -> str:
    return f"{float(value):+.3f}" if isinstance(value, (int, float)) else "–"


def _counts_text(counts: Any) -> str:
    if not isinstance(counts, dict) or not all(isinstance(counts.get(k), int) for k in ("correct", "partial", "wrong")):
        return ""
    return f" ({counts['correct']} / {counts['partial']} / {counts['wrong']})"


def _retrieval_label(retrieval: Any) -> str | None:
    if not isinstance(retrieval, dict):
        return None
    from app.services.auto_rag_service import retrieval_label

    return retrieval_label(retrieval)


def _example_preamble() -> str:
    """The system message layout the reader was trained on, with
    placeholders — rendered from the same template serving, the harness and
    passage-prompt training use, so the card can't drift from it."""
    from app.services.auto_rag_service import _DOCUMENT_PREAMBLE_TEMPLATE

    passages = "\n\n".join(f"[{i}] (<source document> · passage <n>) <passage text>" for i in (1, 2, 3))
    return _DOCUMENT_PREAMBLE_TEMPLATE.format(passages=passages)


def _instruction() -> str:
    """The instruction sentence after the passages (the template's tail)."""
    from app.services.auto_rag_service import _DOCUMENT_PREAMBLE_TEMPLATE

    return _DOCUMENT_PREAMBLE_TEMPLATE.split("{passages}\n", 1)[-1]


def _result_lines(record: dict[str, Any] | None, trained_retrieval: Any) -> list[str]:
    lines = ["## Result", ""]
    arms = record.get("arms") if isinstance(record, dict) and isinstance(record.get("arms"), dict) else {}
    ours = arms.get("finetune_with_passages") if isinstance(arms.get("finetune_with_passages"), dict) else None
    if not ours or ours.get("score") is None:
        lines.append(
            "No held-out evaluation is recorded for this run, so this card makes no claim about how well it "
            "answers. Run the lift check (“Does it read your passages better?”) in BrewSLM and export again."
        )
        lines.append("")
        return lines

    split = str(record.get("split") or "test")
    n_rows = record.get("n_rows")
    judge = record.get("judge")
    eval_label = record.get("retrieval_label") or _retrieval_label(record.get("retrieval"))
    head = f"{n_rows} held-out {split} questions" if isinstance(n_rows, int) else f"The held-out {split} questions"
    head += (
        f", scored by an answer-judge model (`{judge}`) as correct / partial / wrong against an answer key "
        "(score = correct + ½ partial, averaged)."
        if judge else ", scored against an answer key."
    )
    if eval_label:
        head += f" Every arm gets the same retrieval: {eval_label}."
    lines.append(head)
    lines.append("")
    lines.append(f"| | {split.capitalize()} score (correct / partial / wrong) |")
    lines.append("|---|---|")
    for key, label in _ARMS:
        arm = arms.get(key)
        if isinstance(arm, dict) and arm.get("score") is not None:
            value = _score(arm["score"]) + _counts_text(arm.get("counts"))
            if key == "finetune_with_passages":
                value = f"**{value}**"
            lines.append(f"| {label} | {value} |")
    lines.append("")

    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    verdict = str(evidence.get("verdict") or "")
    if evidence.get("n") and verdict:
        sentence = (
            f"Compared row by row with the bar on the same {evidence['n']} questions, this model is better on "
            f"{evidence.get('better', '–')}, worse on {evidence.get('worse', '–')} and the same on "
            f"{evidence.get('same', '–')}"
        )
        if evidence.get("mean_diff") is not None:
            sentence += f": {_signed(evidence['mean_diff'])}"
            if evidence.get("ci_low") is not None and evidence.get("ci_high") is not None:
                sentence += f", 95% CI {_signed(evidence['ci_low'])} to {_signed(evidence['ci_high'])}"
        sentence += f". That is {_VERDICT_TEXT.get(verdict, verdict)}."
        lines.append(sentence)
        lines.append("")
    if record.get("comparable") is False:
        lines.append(
            "**The two arms were not scored under identical conditions** (different retrieval or rows), so the "
            "comparison above is indicative only."
        )
        lines.append("")
    if trained_retrieval is not None and isinstance(record.get("retrieval"), dict):
        from app.services.auto_rag_service import same_retrieval

        if not same_retrieval(trained_retrieval, record["retrieval"]):
            lines.append(
                f"**Retrieval differs from training:** this model was trained with "
                f"{_retrieval_label(trained_retrieval)} but evaluated with {eval_label}."
            )
            lines.append("")

    caveats: list[str] = []
    retrieval = ours.get("retrieval") if isinstance(ours.get("retrieval"), dict) else {}
    if isinstance(retrieval.get("found"), int) and isinstance(retrieval.get("judged"), int):
        text = (
            f"**Retrieval is the ceiling.** By the judge's reading, the retrieved passages held the answer for "
            f"{retrieval['found']} of {retrieval['judged']} questions, the same for every arm."
        )
        wrong = (ours.get("counts") or {}).get("wrong") if isinstance(ours.get("counts"), dict) else None
        if isinstance(retrieval.get("wrong_missed"), int) and isinstance(wrong, int):
            text += f" {retrieval['wrong_missed']} of this model's {wrong} wrong answers are questions where it wasn't."
            if isinstance(retrieval.get("wrong_misread"), int):
                text += f" The other {retrieval['wrong_misread']} misread a passage that held the answer."
        caveats.append(text)
    if judge:
        caveats.append(
            f"**The judge is one model.** Every score above is `{judge}`'s reading, not a human's."
        )
    without = arms.get("finetune_without_passages")
    if isinstance(without, dict) and without.get("score") is not None:
        caveats.append(
            f"**It is a reader, not a knowledge base:** without passages it scores {_score(without['score'])}. "
            "Serve it behind a retriever over the same documents."
        )
    for caveat in caveats:
        lines.append(f"- {caveat}")
    if caveats:
        lines.append("")
    return lines


_SEED_VERDICT_TEXT = {
    "better": "the gain holds across seeds (the 95% CI of the mean seed-to-bar difference is above 0)",
    "worse": "the loss holds across seeds (the 95% CI of the mean seed-to-bar difference is below 0)",
    "within_noise": "a gain across seeds is not established (the 95% CI of the mean seed-to-bar difference includes 0)",
    "too_few_seeds": "there are too few seeds for an interval across seeds",
}


def _same_scoring(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two lift-check records scored the same way: split, judge, rows,
    retrieval and the bar's own score."""
    from app.services.auto_rag_service import same_retrieval

    bar_a = ((a.get("arms") or {}).get("base_with_passages") or {}).get("score")
    bar_b = ((b.get("arms") or {}).get("base_with_passages") or {}).get("score")
    return (
        a.get("split") == b.get("split") and a.get("judge") == b.get("judge") and a.get("n_rows") == b.get("n_rows")
        and same_retrieval(a.get("retrieval"), b.get("retrieval")) and bar_a is not None and bar_a == bar_b
    )


def _seed_lines(experiment_id: int | None, record: dict[str, Any] | None, seed_runs: list[dict[str, Any]]) -> list[str]:
    """Card section for the run's seed siblings (``seed_runs`` includes the
    run itself): one row per seed scored like this run, the across-seed
    interval when every listed seed shares the bar, and which siblings are
    left out and why. Fewer than two seeds → no section."""
    from app.services.paired_comparison_stats import seed_spread_evidence

    if not isinstance(record, dict) or len(seed_runs) < 2:
        return []
    listed: list[dict[str, Any]] = []
    left_out: list[str] = []
    for run in sorted(seed_runs, key=lambda r: (r.get("seed") is None, r.get("seed"), r.get("experiment_id"))):
        rec = record if run.get("experiment_id") == experiment_id else run.get("record")
        score = (((rec or {}).get("arms") or {}).get("finetune_with_passages") or {}).get("score")
        if not isinstance(rec, dict) or score is None:
            left_out.append(f"run #{run.get('experiment_id')} (no lift check)")
        elif not _same_scoring(record, rec):
            left_out.append(f"run #{run.get('experiment_id')} (scored on a different split, judge, retrieval or bar)")
        else:
            listed.append({**run, "record": rec, "score": score})
    if len(listed) < 2:
        return []
    lines = [
        "### Across seeds",
        "",
        f"This run is one of {len(seed_runs)} runs trained on the identical training and validation files with "
        "identical settings, differing only in the random seed. Each was scored the same way:",
        "",
        f"| Seed | Run | {str(record.get('split') or 'test').capitalize()} score | Paired difference vs bar (95% CI) | Validation loss |",
        "|---|---|---|---|---|",
    ]
    for run in listed:
        ev = run["record"].get("evidence") if isinstance(run["record"].get("evidence"), dict) else {}
        diff = _signed(ev.get("mean_diff"))
        if ev.get("ci_low") is not None and ev.get("ci_high") is not None:
            diff += f" ({_signed(ev['ci_low'])} to {_signed(ev['ci_high'])})"
        loss = f"{run['val_loss']:.4f}" if isinstance(run.get("val_loss"), (int, float)) else "–"
        cells = [str(run.get("seed") if run.get("seed") is not None else "–"), f"#{run['experiment_id']}",
                 _score(run["score"]), diff, loss]
        if run.get("experiment_id") == experiment_id:
            cells = [f"**{c}**" for c in cells]
            cells[1] = f"**#{run['experiment_id']} (this model)**"
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    bar = record["arms"]["base_with_passages"]["score"]
    spread = seed_spread_evidence(bar, [run["score"] for run in listed])
    text = (
        f"Across the {len(listed)} seeds the mean score is {_score(spread['mean'])} "
        f"(range {_score(spread['min'])} to {_score(spread['max'])}), {_signed(spread['mean'] - bar)} against the bar"
    )
    if spread.get("ci_low") is not None and spread.get("ci_high") is not None:
        text += f" (95% CI across seeds {_signed(spread['ci_low'])} to {_signed(spread['ci_high'])})"
    text += f", so {_SEED_VERDICT_TEXT.get(spread['verdict'], spread['verdict'])}."
    if spread.get("all_better"):
        text += f" All {len(listed)} seeds beat the bar."
    elif spread.get("all_worse"):
        text += f" All {len(listed)} seeds are below the bar."
    else:
        text += f" {sum(1 for run in listed if run['score'] > bar)} of {len(listed)} seeds beat the bar."
    lines += [text, ""]
    if left_out:
        lines += ["Not in the table: " + ", ".join(left_out) + ".", ""]
    return lines


def _data_lines(passage_report: dict[str, Any] | None, tallies: dict[str, dict[str, int]]) -> list[str]:
    report = passage_report if isinstance(passage_report, dict) else {}
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    train = counts.get("train") if isinstance(counts.get("train"), dict) else {}
    val = counts.get("val") if isinstance(counts.get("val"), dict) else {}
    items: list[str] = []
    generators = tallies.get("generators") or {}
    if generators:
        items.append(
            "**Questions:** written by "
            + ", ".join(f"`{name}` ({rows:,} rows)" for name, rows in sorted(generators.items(), key=lambda kv: -kv[1]))
            + "."
        )
    sources = tallies.get("sources") or {}
    if sources:
        listed = sorted(sources.items(), key=lambda kv: -kv[1])[:MAX_SOURCES_LISTED]
        more = len(sources) - len(listed)
        items.append(
            "**Source documents:** "
            + ", ".join(f"`{name}` ({rows:,} rows)" for name, rows in listed)
            + (f", and {more} more" if more > 0 else "")
            + "."
        )
    label = report.get("retrieval_label") or _retrieval_label(report.get("retrieval"))
    if report.get("targets") == "teacher":
        teacher = f"`{report['teacher']}`" if report.get("teacher") else "a teacher model"
        text = (
            f"**Training answers:** {teacher} answered each question from the same "
            + (f"retrieved passages ({label})" if label else "retrieved passages")
            + " the model sees at inference time, so the targets cite passages"
        )
        if isinstance(train.get("abstained"), int):
            text += f" and {train['abstained']:,} of them say the passages don't hold the answer"
        text += "."
        items.append(text)
        filt = report.get("filter") if isinstance(report.get("filter"), dict) else None
        if filt or (isinstance(train.get("filtered"), int) and train["filtered"]):
            judge = _labels(filt.get("judge")) if filt else None
            text = f"**Filter:** {judge or 'an answer judge'} graded the teacher's answers against the reference answers"
            keep = filt.get("keep") if filt else None
            if keep == "correct":
                text += "; only answers graded correct were kept (partial answers were dropped as well as wrong ones)"
            elif keep == "not_wrong":
                text += "; answers graded wrong were dropped"
            if isinstance(train.get("filtered"), int):
                text += f". {train['filtered']:,} answers were left out of training"
            items.append(text + ".")
        revise = report.get("revise") if isinstance(report.get("revise"), dict) else None
        kept = train.get("revised_kept") if isinstance(train.get("revised_kept"), int) else (revise or {}).get("kept")
        if revise:
            rounds = revise.get("rounds")
            text = "**Critique and revise:** each answer the filter dropped got "
            text += f"up to {rounds} revision round{'s' if rounds != 1 else ''}" if isinstance(rounds, int) else "a revision round"
            critic = _labels(revise.get("critic"))
            text += (f": {critic} critiqued it" if critic else ": it was critiqued") + ", the teacher rewrote it, and "
            text += f"{_labels(revise.get('judge'))} re-graded the rewrite" if revise.get("judge") else "the filter re-graded the rewrite"
            text += "."
            if isinstance(kept, int):
                text += f" {kept:,} revisions passed and are training targets"
                if isinstance(train.get("filtered_revised"), int):
                    text += f"; {train['filtered_revised']:,} still failed and stayed out"
                    if train.get("filtered_revised_abstained"):
                        text += f" ({train['filtered_revised_abstained']:,} of them because the revision said it didn't know)"
                text += "."
            if kept:
                text += " **Some training answers are therefore revisions written under critique, not first drafts.**"
            items.append(text)
    elif report.get("targets") == "reference":
        text = (
            "**Training answers:** the reference answer with the passage that holds it cited"
            + (f", over retrieved passages ({label})" if label else "")
            + "."
        )
        if isinstance(train.get("withheld"), int) and train["withheld"]:
            text += (
                f" For {train['withheld']:,} rows the passage holding the answer was withheld and the target says "
                "the passages don't say, which teaches it to abstain."
            )
        items.append(text)
    if isinstance(train.get("rows_out"), int):
        text = f"**Training set:** {train['rows_out']:,} passage-prompt rows (question + retrieved passages → answer)"
        if isinstance(val.get("rows_out"), int):
            text += f"; {val['rows_out']:,} validation rows"
        items.append(text + ".")
    if not items:
        return []
    return ["## Training data", ""] + [f"- {item}" for item in items] + [""]


def _labels(value: Any) -> str | None:
    if isinstance(value, list):
        return ", ".join(f"`{v}`" for v in value) if value else None
    return f"`{value}`" if value else None


def _training_lines(training: dict[str, Any], adapter: dict[str, Any] | None, merged: bool) -> list[str]:
    parts: list[str] = []
    if isinstance(adapter, dict) and adapter.get("r") is not None:
        lora = f"LoRA r {adapter['r']}"
        if adapter.get("lora_alpha") is not None:
            lora += f", alpha {adapter['lora_alpha']}"
        if adapter.get("lora_dropout") is not None:
            lora += f", dropout {adapter['lora_dropout']:g}"
        targets = adapter.get("target_modules")
        if isinstance(targets, list) and targets:
            lora += " on " + ", ".join(f"`{t}`" for t in sorted(targets))
        parts.append(lora + (" (merged into the base weights)" if merged else "") + ".")
    env = training.get("runtime_environment") if isinstance(training.get("runtime_environment"), dict) else {}
    loop: list[str] = []
    if isinstance(training.get("epochs"), (int, float)):
        loop.append(f"{training['epochs']:g} epochs")
    if isinstance(training.get("total_steps"), int):
        loop.append(f"{training['total_steps']:,} steps")
    micro = env.get("micro_batch") if isinstance(env.get("micro_batch"), dict) else {}
    if micro.get("effective_batch"):
        loop.append(f"effective batch {micro['effective_batch']}")
    fit = env.get("max_seq_length_fit") if isinstance(env.get("max_seq_length_fit"), dict) else {}
    if fit.get("to"):
        loop.append(f"max sequence length {fit['to']}")
    if env.get("loss_masking") == "completion_only":
        loop.append("loss on the answer tokens only")
    if loop:
        parts.append(", ".join(loop).capitalize() + ".")
    losses: list[str] = []
    if isinstance(training.get("final_train_loss"), (int, float)):
        losses.append(f"final train loss {training['final_train_loss']:.3f}")
    if isinstance(training.get("final_eval_loss"), (int, float)):
        losses.append(f"validation loss {training['final_eval_loss']:.3f}")
    if losses:
        parts.append(", ".join(losses).capitalize() + ".")
    repro = training.get("reproducibility") if isinstance(training.get("reproducibility"), dict) else {}
    if repro.get("seed") is not None:
        parts.append(f"Seed {repro['seed']}.")
    if env.get("device_name"):
        parts.append(f"Trained on one {env['device_name']}.")
    if not parts:
        return []
    return ["## Training", "", " ".join(parts), ""]


def _prompt_lines(model_ref: str) -> list[str]:
    instruction = _instruction()
    return [
        "## Prompt format",
        "",
        "The model was trained on its chat template. The **system message** holds the numbered passages and the "
        "instruction, and the **user message** is the question. Keep this layout; others were never trained or "
        "tested.",
        "",
        "```",
        _example_preamble(),
        "```",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'repo = "{model_ref}"',
        "tok = AutoTokenizer.from_pretrained(repo)",
        'model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")',
        "",
        "passages = [(\"<source document> · passage <n>\", \"<passage text>\")]  # from your retriever",
        'system = "Passages from this project\'s documents:\\n" + "\\n\\n".join(',
        '    f"[{i}] ({where}) {text}" for i, (where, text) in enumerate(passages, 1)',
        f') + "\\n" + {json.dumps(instruction, ensure_ascii=False)}',
        'messages = [{"role": "system", "content": system}, {"role": "user", "content": "<question>"}]',
        "inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,",
        '                                 return_tensors="pt").to(model.device)',
        "out = model.generate(**inputs, max_new_tokens=256, do_sample=False)",
        'print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))',
        "```",
        "",
    ]


def build_reader_model_card(
    *,
    base_model: str | None = None,
    model_name: str | None = None,
    experiment_id: int | None = None,
    project_name: str | None = None,
    reader_record: dict[str, Any] | None = None,
    passage_report: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    adapter: dict[str, Any] | None = None,
    tallies: dict[str, dict[str, int]] | None = None,
    merged: bool = False,
    seed_selection: str | None = None,
    seed_runs: list[dict[str, Any]] | None = None,
) -> str:
    """Markdown model card (with HF front matter) for a passage reader.

    ``reader_record``: ``auto_rag/passage_reader/<run>.json``.
    ``passage_report``: the run's ``passage_prompts/report.json``.
    ``training``: ``training_report.json``. ``adapter``: the run's
    ``adapter_config.json``. ``tallies``: ``{generators, sources}`` row
    counts from the run's own training file. ``seed_runs``: the run and its
    seed siblings (``seed_siblings``) — ``{experiment_id, seed, val_loss,
    record}``. Numbers are copied, never derived (bar the across-seed
    interval); a section with no data is omitted — except the result, which
    says when there is none."""
    training = training or {}
    passage_report = passage_report if isinstance(passage_report, dict) else {}
    lines: list[str] = ["---", "library_name: transformers", "pipeline_tag: text-generation"]
    if _is_hub_id(base_model):
        lines.append(f"base_model: {base_model}")
    lines += ["tags:", "- retrieval-augmented-generation", "- question-answering"]
    if adapter is not None:
        lines.append("- lora")
    if merged:
        lines.append("- merged")
    lines += ["- brewslm", "---", ""]

    lines.append(f"# {model_name or (f'{project_name} passage reader' if project_name else 'Passage reader')}")
    lines.append("")
    base = f"`{base_model}`" if base_model else "A language model"
    lines.append(
        f"{base}, fine-tuned to answer questions **from passages you retrieve and put in the prompt**. "
        "It cites the passages it used and says it doesn't know when they don't hold the answer."
    )
    provenance = []
    if project_name:
        provenance.append(f"project “{project_name}”")
    if experiment_id is not None:
        provenance.append(f"run #{experiment_id}")
    lines.append(
        "Trained with [BrewSLM](https://brewslm.com)" + (f" ({', '.join(provenance)})." if provenance else ".")
    )
    lines.append("")
    lines.append(
        "> It is not a subject-matter expert. It can be wrong, even when it cites a passage. Check answers "
        "against the source documents before relying on them."
    )
    lines.append("")

    lines += _result_lines(reader_record, passage_report.get("retrieval"))
    lines += _seed_lines(experiment_id, reader_record, seed_runs or [])
    if seed_selection:
        lines += [f"**Seed selection:** {seed_selection}", ""]
    lines += _data_lines(passage_report, tallies or {})
    lines += _training_lines(training, adapter, merged)
    lines += _prompt_lines(model_name if _is_hub_id(model_name) else "<this model's repo id or path>")

    limits = ["Trained on this project's documents only; it knows nothing about other documents."]
    if isinstance(reader_record, dict):
        limits.append(
            "The test questions come from the same documents as the training questions, so the score measures "
            "in-distribution reading, not robustness to how people phrase questions."
        )
    counts = passage_report.get("counts") if isinstance(passage_report.get("counts"), dict) else {}
    train = counts.get("train") if isinstance(counts.get("train"), dict) else {}
    if isinstance(train.get("abstained"), int) and isinstance(train.get("rows_out"), int) and train["rows_out"]:
        limits.append(
            f"{train['abstained']:,} of {train['rows_out']:,} training answers "
            f"({train['abstained'] / train['rows_out']:.0%}) are “I don't know”, so it abstains when unsure; "
            "an abstention counts as wrong against the answer key."
        )
    filt = passage_report.get("filter") if isinstance(passage_report.get("filter"), dict) else {}
    eval_judge = reader_record.get("judge") if isinstance(reader_record, dict) else None
    filter_judges = filt.get("judge") if isinstance(filt.get("judge"), list) else [filt.get("judge")]
    if eval_judge and eval_judge in filter_judges:
        limits.append(
            f"The same model (`{eval_judge}`) filtered the training answers and judges the test, so its "
            "preferences are built into both the data and the score."
        )
    lines += ["## Limitations", ""] + [f"- {item}" for item in limits] + [""]
    return "\n".join(lines).rstrip() + "\n"


# ── Evidence gathering ──────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def training_file_tallies(train_file: Path | None, expected_sha256: str | None) -> dict[str, dict[str, int]]:
    """``{generators: {model: rows}, sources: {document: rows}}`` from the
    run's passage-prompt training file — only while its sha256 still
    matches the training report (a file rewritten since would describe
    different data). Empty otherwise."""
    if train_file is None or not expected_sha256 or not train_file.is_file():
        return {}
    try:
        raw = train_file.read_bytes()
    except OSError:
        return {}
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        return {}
    generators: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("generated_by"), str) and row["generated_by"]:
            generators[row["generated_by"]] += 1
        if isinstance(row.get("source_doc"), str) and row["source_doc"]:
            sources[row["source_doc"]] += 1
    return {"generators": dict(generators), "sources": dict(sources)}


_SIBLING_IGNORED_KEYS = {"seed", "training_runtime_id", "seed_selection"}


def _sibling_config(config: Any) -> dict[str, Any]:
    """The settings two seed siblings must share: the run config minus the
    seed, run-local ids and ``_``-prefixed runtime notes."""
    if not isinstance(config, dict):
        return {}
    return {k: v for k, v in config.items() if k not in _SIBLING_IGNORED_KEYS and not str(k).startswith("_")}


def _run_fingerprint(experiment: Any) -> tuple[str, str, Any, float | None] | None:
    output_dir = Path(experiment.output_dir).expanduser() if getattr(experiment, "output_dir", None) else None
    report = (_read_json(output_dir / "training_report.json") if output_dir else None) or {}
    repro = report.get("reproducibility") if isinstance(report.get("reproducibility"), dict) else {}
    if not repro.get("train_file_sha256") or not repro.get("val_file_sha256"):
        return None
    loss = report.get("final_eval_loss")
    return repro["train_file_sha256"], repro["val_file_sha256"], repro.get("seed"), loss if isinstance(loss, (int, float)) else None


async def seed_siblings(db: Any, project_id: int, experiment: Experiment) -> list[dict[str, Any]]:
    """The run + every COMPLETED passage-trained run in the project that is
    the same training with another seed: same base model, byte-identical
    training and validation files (the reports' sha256), the same config
    apart from the seed. A formal seed group qualifies; so do runs cloned by
    hand. Each: ``{experiment_id, seed, val_loss, record}``. Fewer than two
    distinct seeds → []."""
    from sqlalchemy import select

    from app.models.experiment import ExperimentStatus
    from app.services.passage_reader_eval_service import is_passage_trained, read_record

    mine = _run_fingerprint(experiment)
    if mine is None or not is_passage_trained(experiment):
        return []
    candidates = (
        await db.execute(
            select(Experiment)
            .where(Experiment.project_id == project_id)
            .where(Experiment.status == ExperimentStatus.COMPLETED)
            .where(Experiment.base_model == experiment.base_model)
            .where(Experiment.id != experiment.id)
        )
    ).scalars().all()
    runs = [{"experiment_id": experiment.id, "seed": mine[2], "val_loss": mine[3], "record": None}]
    seeds = {mine[2]}
    for other in candidates:
        if not is_passage_trained(other) or _sibling_config(other.config) != _sibling_config(experiment.config):
            continue
        theirs = _run_fingerprint(other)
        if theirs is None or theirs[:2] != mine[:2] or theirs[2] in seeds:
            continue
        seeds.add(theirs[2])
        runs.append({"experiment_id": other.id, "seed": theirs[2], "val_loss": theirs[3],
                     "record": read_record(project_id, other.id)})
    return runs if len(runs) > 1 else []


def _cached_teacher_provenance(project_id: int, train_file: Path | None, report: dict[str, Any]) -> dict[str, Any]:
    from app.services.teacher_passage_answers_service import load_answers, teacher_provenance

    if train_file is None or not isinstance(report.get("retrieval"), dict):
        return {}
    rows: list[dict[str, Any]] = []
    try:
        for line in train_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return {}
    provenance = teacher_provenance(rows, load_answers(project_id), report["retrieval"])
    return {"filter": provenance.get("filter"), "revise": provenance.get("revise")} if provenance else {}


def package_passage_reader(
    *,
    project_id: int,
    experiment: Experiment,
    model_dir: Path,
    project_name: str | None = None,
    merged: bool = False,
    seed_selection: dict[str, Any] | None = None,
    seed_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Write ``README.md`` into an exported passage-reader model directory.
    None (nothing written) for a run not trained with passages."""
    from app.services.passage_reader_eval_service import is_passage_trained, read_record

    if not is_passage_trained(experiment):
        return None
    model_dir = Path(model_dir)
    output_dir = Path(experiment.output_dir).expanduser() if experiment.output_dir else None
    training = (_read_json(output_dir / "training_report.json") if output_dir else None) or {}
    passage_report = (_read_json(output_dir / "passage_prompts" / "report.json") if output_dir else None) or {}
    adapter = _read_json(output_dir / "model" / "adapter_config.json") if output_dir else None
    repro = training.get("reproducibility") if isinstance(training.get("reproducibility"), dict) else {}
    train_file = Path(str(training["train_file"])) if training.get("train_file") else None
    tallies = training_file_tallies(train_file, repro.get("train_file_sha256"))
    if passage_report.get("targets") == "teacher" and "filter" not in passage_report and tallies:
        # A run trained before the report recorded who filtered / revised
        # its targets: read it from the cache, but only when every trained
        # answer is still the cached one (tallies ⇒ the file is unchanged).
        passage_report = {**passage_report, **_cached_teacher_provenance(project_id, train_file, passage_report)}
    record = read_record(project_id, experiment.id)
    card = build_reader_model_card(
        base_model=experiment.base_model,
        experiment_id=experiment.id,
        project_name=project_name,
        reader_record=record,
        passage_report=passage_report,
        training=training,
        adapter=adapter,
        tallies=tallies,
        merged=merged,
        seed_selection=(seed_selection or {}).get("text"),
        seed_runs=seed_runs,
    )
    (model_dir / MODEL_CARD_FILE).write_text(card, encoding="utf-8")
    evidence = (record or {}).get("evidence") if isinstance((record or {}).get("evidence"), dict) else {}
    return {
        "model_card": MODEL_CARD_FILE,
        "evaluated": bool(record),
        "split": (record or {}).get("split"),
        "judge": (record or {}).get("judge"),
        "verdict": evidence.get("verdict"),
        "seed_runs": [run["experiment_id"] for run in seed_runs or []],
    }
