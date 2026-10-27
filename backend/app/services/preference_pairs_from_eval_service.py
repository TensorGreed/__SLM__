"""Preference pairs from a held-out eval: the judge's wrong / partial answers.

For an eval result whose rows a judge graded (the answer judge's ``row_judge_verdict``, or for a
chat row the per-turn dialogue judge's verdict on its FINAL turn when the answer judge didn't
grade it), every row marked wrong (or partial) becomes a pair: ``chosen`` = the answer-key
reference, ``rejected`` = the model's own prediction. No teacher revision and no review-queue
edits — the evidence is the eval alone.

The rows come from the eval's side file (``details["row_predictions_file"]``,
``evaluation_service.write_row_predictions_file``) — judged rows only, so an unjudged row never
reaches a pair. ``prompt`` is flat text, the shape the preference loader expects: the row's prompt
for a single-turn row, and for a chat row the same no-template render the eval uses (``role:
content`` lines plus ``assistant: ``); the chat row's ``messages`` ride along on the pair.

**Test split is refused by default.** Pairs built from the held-out test split would teach the model
the very rows it is scored on (memorisation), so an eval whose split is ``test`` / ``gold_test`` -
or unknown - raises ``test_split_refused`` unless ``allow_test_split`` is set; every pair carries
``source_split`` either way and the response carries a ``warning`` when the override was used.

Written to ``prepared/alignment/eval_<eval_result_id>_pairs.jsonl`` (re-running replaces it).
Skipped rows are counted by reason: ``not_judged``, ``correct``, ``not_included``,
``prediction_equals_reference``, ``empty_prediction``, ``empty_reference``, ``no_prediction_stored``,
``not_stored`` (judged rows past the side file's caps, or whose chat history was cut).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment
from app.services import alignment_dataset_service as ads
from app.services.dialogue_eval_service import render_history_fallback
from app.services.evaluation_service import read_row_predictions_file

DEFAULT_INCLUDE = ("wrong", "partial")
TEST_SPLITS = frozenset({"test", "gold_test"})
SKIP_REASONS = (
    "not_judged", "correct", "not_included", "prediction_equals_reference",
    "empty_prediction", "empty_reference", "no_prediction_stored", "not_stored",
)
TEST_SPLIT_WARNING = (
    "These pairs come from the held-out test split. Training on them teaches the model the rows it is "
    "scored on, so a later eval on that split measures memorisation, not generalisation."
)


class PreferencePairsError(ValueError):
    """Carries the skip breakdown so the API can show why nothing was built."""

    def __init__(self, message: str, *, code: str, skipped: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.skipped = skipped or {}


def record_verdict(record: dict[str, Any]) -> str | None:
    """The row's verdict: the answer judge's, else the dialogue judge's on the row's FINAL turn
    (highest turn number) — a chat dataset's judge may be the per-turn one."""
    verdict = record.get("row_judge_verdict")
    if verdict in ("correct", "partial", "wrong"):
        return verdict
    block = record.get("dialogue_judge")
    turns = block.get("turns") if isinstance(block, dict) else None
    if isinstance(turns, list) and turns:
        final = max((t for t in turns if isinstance(t, dict)), key=lambda t: t.get("turn") or 0, default=None)
        if final and final.get("verdict") in ("correct", "partial", "wrong"):
            return final["verdict"]
    return None


def build_pairs(
    records: list[dict[str, Any]],
    *,
    include: tuple[str, ...],
    eval_result_id: int,
    experiment_id: int,
    source_split: str | None = None,
    not_stored: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pure: ``(pairs, skipped-by-reason)`` from the side file's records. ``not_stored`` is the
    number of judged rows the side file could not hold (reported, never silently dropped)."""
    skipped: dict[str, int] = {}

    def skip(reason: str, n: int = 1) -> None:
        skipped[reason] = skipped.get(reason, 0) + n

    if not_stored > 0:
        skip("not_stored", not_stored)
    pairs: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or "prediction" not in record:
            skip("no_prediction_stored")
            continue
        verdict = record_verdict(record)
        if verdict is None:
            skip("not_judged")
            continue
        if verdict == "correct":
            skip("correct")
            continue
        if verdict not in include:
            skip("not_included")
            continue
        reference = str(record.get("reference") or "").strip()
        prediction = str(record.get("prediction") or "").strip()
        if not reference:
            skip("empty_reference")
            continue
        if not prediction:
            skip("empty_prediction")
            continue
        if " ".join(prediction.split()) == " ".join(reference.split()):
            skip("prediction_equals_reference")
            continue
        history = record.get("history")
        if isinstance(history, list) and history:
            if record.get("history_truncated"):
                skip("not_stored")  # the start of the dialogue (system prompt...) was cut
                continue
            prompt = render_history_fallback(history)
        else:
            history = None
            prompt = str(record.get("prompt") or "").strip()
        if not prompt:
            skip("no_prediction_stored")
            continue
        pair: dict[str, Any] = {
            "prompt": prompt,
            "chosen": reference,
            "rejected": prediction,
            "source": "eval",
            "source_split": source_split,
            "eval_result_id": eval_result_id,
            "experiment_id": experiment_id,
            "row_index": record.get("row_index"),
            "verdict": verdict,
        }
        if history:
            pair["messages"] = history
        pairs.append(pair)
    return pairs, skipped


async def preference_pairs_from_eval(
    db: AsyncSession,
    project_id: int,
    eval_result_id: int,
    include: tuple[str, ...] | list[str] = DEFAULT_INCLUDE,
    dataset_name: str | None = None,
    allow_test_split: bool = False,
) -> dict[str, Any]:
    """Build and write the pairs; raises ``PreferencePairsError`` (not found / test split refused /
    nothing built)."""
    chosen_include = tuple(v for v in include if v in ("wrong", "partial"))
    if not chosen_include:
        raise PreferencePairsError("include must list 'wrong' and/or 'partial'.", code="invalid_include")
    result = await db.get(EvalResult, eval_result_id)
    experiment = await db.get(Experiment, result.experiment_id) if result is not None else None
    if result is None or experiment is None or experiment.project_id != project_id:
        raise PreferencePairsError(f"Eval result {eval_result_id} not found in this project.", code="eval_result_not_found")
    # The split an eval ran on is its ``dataset_name`` ("test", "validation", ...); one that
    # recorded none is treated as the test split.
    split = str(result.dataset_name or "").strip().lower()
    test_like = split in TEST_SPLITS or not split
    if test_like and not allow_test_split:
        raise PreferencePairsError(
            f"Eval #{eval_result_id} ran on the {split or 'unknown'} split. " + TEST_SPLIT_WARNING
            + " Evaluate on a validation split instead, or set allow_test_split to build them anyway.",
            code="test_split_refused",
        )
    details = result.details if isinstance(result.details, dict) else {}
    pointer = details.get("row_predictions_file")
    records = read_row_predictions_file(project_id, pointer)
    if records is None:
        # An eval from before judged rows were stored: only a few truncated previews survive.
        count = len(((details.get("row_scores") or {}).get("keys")) or []) or len(details.get("predictions_preview") or [])
        raise PreferencePairsError(
            "This eval did not store its judged rows; re-run the eval to build pairs from it.",
            code="no_pairs_built", skipped={"no_prediction_stored": count},
        )
    pairs, skipped = build_pairs(
        records, include=chosen_include, eval_result_id=eval_result_id, experiment_id=experiment.id,
        source_split=split or None, not_stored=int((pointer or {}).get("truncated") or 0),
    )
    if not pairs:
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items())) or "no rows"
        raise PreferencePairsError(
            f"No preference pairs could be built from eval #{eval_result_id} ({summary}).",
            code="no_pairs_built", skipped=skipped,
        )
    name = (dataset_name or "").strip() or f"Preference pairs from eval #{eval_result_id} (run #{experiment.id})"
    path = ads._alignment_dir(project_id) / f"eval_{eval_result_id}_pairs.jsonl"
    ads._write_jsonl_rows(path, pairs)
    return {
        "dataset_id": None,
        "dataset_name": name,
        "dataset_path": str(path),
        "eval_result_id": eval_result_id,
        "experiment_id": experiment.id,
        "source_split": split or None,
        "warning": TEST_SPLIT_WARNING if test_like else None,
        "created": len(pairs),
        "skipped": skipped,
        "examples": pairs[:3],
    }
