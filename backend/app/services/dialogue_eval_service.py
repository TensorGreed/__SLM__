"""Multi-turn (chat) rows at eval time: last-turn history + per-turn replay judge.

Two things live here, both generic (any chat dataset, any recipe):

* **Last-turn history** — a held-out row with ``messages`` is evaluated on its
  FINAL assistant turn, given everything before it (system prompt and earlier
  exchanges) rendered with the tokenizer's chat template exactly like training.
  ``row_history`` / ``history_for_pair`` / ``history_messages`` produce the message
  list; ``evaluation_service`` hands it to the chat-template wrapper.
* **Per-turn replay judge** (``dialogue_judge: true``) — for a row with 2+
  assistant turns, every assistant turn is generated from the GOLD history up to it
  and judged by the existing answer judge against that turn's reference. Reported
  per row (``turns``, ``mean``, ``worst_turn``) and as a summary
  (``rows``, ``mean``, ``worst_turn_mean``, ``by_turn_index``). Turn numbers are
  1-based assistant-turn ordinals.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services import llm_budget_guard, llm_usage

_LOG = logging.getLogger(__name__)

# Upper bound on extra generations one eval may spend on replayed turns.
MAX_REPLAY_GENERATIONS = 500


def normalized_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """The row's chat messages, normalised the way the chat adapter does
    (``messages`` or ``conversations``; human/gpt/model aliases; empty turns dropped)."""
    from app.services.data_adapter_service import _messages_from_record

    return _messages_from_record(row) if isinstance(row, dict) else []


def final_assistant_index(messages: list[dict[str, str]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return idx
    return -1


def row_history(row: dict[str, Any]) -> tuple[list[dict[str, str]], str] | None:
    """``(history, reference)`` for a chat row: every message before the final assistant
    turn (system included) and that turn's content. None when the row has no messages or
    no user turn before its final assistant turn."""
    messages = normalized_messages(row)
    idx = final_assistant_index(messages)
    if idx < 1:
        return None
    history = messages[:idx]
    if not any(m.get("role") == "user" for m in history):
        return None
    return history, messages[idx]["content"]


def history_for_pair(history: list[dict[str, str]], prompt: str) -> list[dict[str, str]] | None:
    """The message list to render for one eval pair, or None when a bare user prompt is
    enough (a single user message, no system) — keeping single-turn rows byte-identical.
    When the handler rewrote the last user turn (``prompt`` differs, e.g. retrieved
    context added) the rewritten prompt replaces that turn's content."""
    msgs = [dict(m) for m in history]
    if len(msgs) == 1 and msgs[0].get("role") == "user":
        return None
    last_user = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"), -1)
    if prompt and last_user >= 0 and str(msgs[last_user].get("content") or "").strip() != prompt.strip():
        msgs[last_user]["content"] = prompt
    return msgs


def history_text(messages: list[dict[str, str]]) -> str:
    """Flat ``role: content`` lines — the judge's QUESTION and the DPO prompt string."""
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def render_history_fallback(messages: list[dict[str, str]]) -> str:
    """Prompt for a tokenizer without a chat template: ``role: content\\n`` lines plus the
    ``assistant: `` generation prompt — exactly how the trainer renders such rows."""
    return "".join(f"{m.get('role', 'user')}: {m.get('content', '')}\n" for m in messages) + "assistant: "


def last_user_content(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def assistant_turns(messages: list[dict[str, str]]) -> list[int]:
    """Indices of assistant turns that have a user turn somewhere before them."""
    out: list[int] = []
    seen_user = False
    for idx, m in enumerate(messages):
        if m.get("role") == "user":
            seen_user = True
        elif m.get("role") == "assistant" and seen_user:
            out.append(idx)
    return out


def build_replay_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extra eval pairs, one per NON-final assistant turn of every multi-turn pair
    (``pair["dialogue_messages"]`` with 2+ assistant turns). The final turn is the pair's
    own prediction. Each carries ``_replay = {pair_index, turn, turns}``. Capped at
    ``MAX_REPLAY_GENERATIONS`` (whole rows, in order): rows past the cap are not replayed, and
    ``judge_dialogues`` leaves them out and reports them (``rows_not_replayed``)."""
    replay: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        messages = pair.get("dialogue_messages")
        if not isinstance(messages, list):
            continue
        turns = assistant_turns(messages)
        if len(turns) < 2:
            continue
        if len(replay) + len(turns) - 1 > MAX_REPLAY_GENERATIONS:
            break
        for ordinal, idx in enumerate(turns[:-1], start=1):
            history = messages[:idx]
            prompt = last_user_content(history)
            replay.append(
                {
                    "prompt": prompt,
                    "reference": messages[idx]["content"],
                    "history": history_for_pair(history, prompt),
                    "_replay": {"pair_index": pair_index, "turn": ordinal, "turns": len(turns)},
                }
            )
    return replay


def _round(value: float) -> float:
    return round(float(value), 4)


async def judge_dialogues(
    pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    replay_predictions: list[dict[str, Any]],
    replay_pairs: list[dict[str, Any]],
    judge: Any,
    *,
    cache: Any = None,
) -> dict[str, Any]:
    """Judge every assistant turn of every multi-turn row (see module doc).

    ``predictions[i]`` belongs to ``pairs[i]`` (its final turn); ``replay_predictions[j]``
    to ``replay_pairs[j]``. Annotates ``predictions[i]["dialogue_judge"]`` and returns the
    summary (``rows``, ``mean``, ``worst_turn_mean``, ``by_turn_index``, ``rows_not_replayed`` /
    ``replay_cap_hit``, plus judge label, call counts and cloud usage). A turn the judge couldn't score is left out of that
    row's numbers and counted in ``unjudged``."""
    from app.services.answer_judge_service import cache_key

    by_row: dict[int, dict[int, dict[str, Any]]] = {}
    for rp, pred in zip(replay_pairs, replay_predictions):
        meta = rp["_replay"]
        by_row.setdefault(meta["pair_index"], {})[meta["turn"]] = {
            "history": rp.get("history") or [{"role": "user", "content": rp["prompt"]}],
            "reference": rp["reference"],
            "prediction": str((pred or {}).get("prediction") or ""),
        }
    replayed = set(by_row)
    rows_not_replayed = 0
    for pair_index, pair in enumerate(pairs):
        messages = pair.get("dialogue_messages")
        if not isinstance(messages, list) or pair_index >= len(predictions):
            continue
        turns = assistant_turns(messages)
        if len(turns) < 2:
            continue
        if pair_index not in replayed:
            # Past the replay cap: its earlier turns were never generated, so judging only
            # the final turn would pass off a one-turn score as a dialogue score.
            rows_not_replayed += 1
            continue
        final_idx = turns[-1]
        by_row.setdefault(pair_index, {})[len(turns)] = {
            "history": messages[:final_idx],
            "reference": messages[final_idx]["content"],
            "prediction": str(predictions[pair_index].get("prediction") or ""),
        }

    calls = cached = tokens = unjudged = 0
    row_means: list[float] = []
    row_worsts: list[float] = []
    per_turn: dict[int, list[float]] = {}
    with llm_usage.metered() as meter:
        for pair_index in sorted(by_row):
            turn_rows: list[dict[str, Any]] = []
            for turn in sorted(by_row[pair_index]):
                item = by_row[pair_index][turn]
                question = history_text(item["history"])
                key = cache_key(judge.label, question, item["reference"], item["prediction"])
                hit = cache.get(key) if cache is not None else None
                if hit is not None:
                    score, verdict, _reason = hit
                    cached += 1
                else:
                    result = None
                    try:
                        result = await judge.judge(question, item["reference"], item["prediction"])
                    except llm_budget_guard.LlmBudgetExceeded:
                        raise
                    except Exception as exc:  # noqa: BLE001 - the judge never breaks the eval
                        _LOG.warning("dialogue judge failed on row %s turn %s: %s", pair_index, turn, exc)
                    calls += 1
                    if result is None:
                        unjudged += 1
                        continue
                    score, verdict, reason, row_tokens = result
                    tokens += int(row_tokens or 0)
                    if cache is not None:
                        cache.set(key, verdict, reason)
                turn_rows.append({"turn": turn, "score": float(score), "verdict": verdict})
                per_turn.setdefault(turn, []).append(float(score))
            if not turn_rows:
                continue
            scores = [t["score"] for t in turn_rows]
            worst = min(turn_rows, key=lambda t: t["score"])
            block = {
                "turns": turn_rows,
                "mean": _round(sum(scores) / len(scores)),
                "worst_turn": worst["turn"],
            }
            predictions[pair_index]["dialogue_judge"] = block
            row_means.append(block["mean"])
            row_worsts.append(worst["score"])
    if cache is not None:
        cache.flush()
    summary: dict[str, Any] = {
        "judge": judge.label,
        "rows": len(row_means),
        "mean": _round(sum(row_means) / len(row_means)) if row_means else None,
        "worst_turn_mean": _round(sum(row_worsts) / len(row_worsts)) if row_worsts else None,
        "by_turn_index": [
            {"turn": turn, "mean": _round(sum(vals) / len(vals)), "n": len(vals)}
            for turn, vals in sorted(per_turn.items())
        ],
        "rows_not_replayed": rows_not_replayed,
        "replay_cap_hit": rows_not_replayed > 0,
        "turns_judged": sum(len(v) for v in per_turn.values()),
        "unjudged": unjudged,
        "judge_calls": calls,
        "judge_cached": cached,
        "judge_tokens": tokens,
    }
    summary.update(meter.block())
    return summary
