"""LLM-judge correctness for long-answer held-out evals.

Token-F1 / exact match are fine rulers for labels and short spans, but on
paragraph-length answers (statutes, policies, manuals) they measure word
overlap, not whether the answer is right: a correct paraphrase scores low,
a fluent wrong answer that reuses the question's words scores high. This
module adds ``judge_correct`` next to F1: a judge model reads the question,
the reference answer and the model's answer and returns correct / partial /
wrong (1 / 0.5 / 0). The mean is the metric; the per-row scores go into
``details["row_scores"]`` so the lift check pairs base vs fine-tuned rows
and reports evidence exactly as it does for F1.

Best-effort by design: no judge reachable → the metric is absent and F1
stays the headline; a judge error on a row → that row is "unjudged" and
excluded (counted in the snapshot). Tests inject ``judge_fn``.

Judge resolution (``resolve_answer_judge``): ``EVAL_JUDGE_BACKEND`` env
(``none`` disables; ``ollama[:model]`` / ``teacher`` pick a local synth
backend; ``anthropic`` / ``openai`` / ``deepseek`` use the probe-judge
cloud config) → ``project.runtime_config["eval_judge"]`` (``enabled``,
``backend``) → the probe judge's cloud config when one is configured →
the first available local synth backend (Ollama on a dev box). Cloud is
never auto-picked over a reachable local model: judging costs one call
per row per eval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Awaitable, Callable

from app.config import settings

_LOG = logging.getLogger(__name__)

JUDGE_METRIC = "judge_correct"
# Generative profiles where overlap metrics are weak rulers. Labels and
# structured extraction have exact / field-level scoring already.
JUDGE_PROFILES: frozenset[str] = frozenset({"qa", "rag_qa", "instruction_sft", "summarization"})
# Short-answer Q&A (names, numbers, spans) is what exact match + F1 were
# built for; the judge is for answers long enough that paraphrase matters.
MIN_REFERENCE_WORDS = 8
VERDICT_SCORES: dict[str, float] = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
MAX_JUDGE_TOKENS = 200
CACHE_MAX_ENTRIES = 5000

# judge(question, reference, prediction) -> (score, verdict, reason, tokens) | None
JudgeFn = Callable[[str, str, str], Awaitable["tuple[float, str, str, int] | None"]]

_SYSTEM_PROMPT = (
    "You grade a small language model's answer against a reference answer. "
    "Judge factual content only — not length, style or wording. Respond ONLY "
    'with JSON: {"verdict": "correct" | "partial" | "wrong", "reason": "<one short sentence>"}.'
)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def should_judge(task_profile: str | None, references: list[str]) -> bool:
    """Long-answer generative task? (profile in ``JUDGE_PROFILES`` and the
    median reference is at least ``MIN_REFERENCE_WORDS`` words.)"""
    profile = str(task_profile or "").strip().lower()
    if profile not in JUDGE_PROFILES:
        return False
    lengths = [len(str(r or "").split()) for r in references if str(r or "").strip()]
    if not lengths:
        return False
    return median(lengths) >= MIN_REFERENCE_WORDS


def build_judge_prompt(question: str, reference: str, prediction: str) -> str:
    return (
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "REFERENCE ANSWER (treat as ground truth):\n"
        f"{reference.strip()}\n\n"
        "MODEL ANSWER:\n"
        f"{prediction.strip() or '(empty)'}\n\n"
        "Verdict rules:\n"
        "- correct: the model answer states the reference's key facts (paraphrase is fine) "
        "and nothing that contradicts them.\n"
        "- partial: it gets some key facts right but misses important ones, or adds a "
        "claim that is not in the reference.\n"
        "- wrong: it contradicts the reference, answers a different question, is empty, "
        "or is mostly unsupported.\n"
        "An answer that reuses the question's words without giving the facts is wrong."
    )


def parse_verdict(raw: str | None) -> tuple[float, str, str] | None:
    """``(score, verdict, reason)`` from the judge's reply, or None."""
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    payload: Any = None
    try:
        payload = json.loads(text)
    except Exception:  # noqa: BLE001
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:  # noqa: BLE001
                payload = None
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in VERDICT_SCORES:
        return None
    reason = str(payload.get("reason") or "").strip()[:300]
    return VERDICT_SCORES[verdict], verdict, reason


def cache_key(judge_label: str, question: str, reference: str, prediction: str) -> str:
    raw = "\x00".join((judge_label, question.strip(), reference.strip(), prediction.strip()))
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()


class AnswerJudgeCache:
    """File-backed ``{key: {score, verdict, reason}}``; greedy decoding makes
    the model's answers deterministic, so re-evaluating the same checkpoint
    (and every run's reuse of the base-model eval) costs no judge calls.
    Best-effort: load / write errors degrade to an in-memory cache."""

    def __init__(self, path: Any) -> None:
        self._path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        try:
            if path is not None and path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = {k: v for k, v in loaded.items() if isinstance(v, dict)}
        except Exception:  # noqa: BLE001
            self._data = {}

    def get(self, key: str) -> tuple[float, str, str] | None:
        entry = self._data.get(key)
        if not isinstance(entry, dict) or entry.get("verdict") not in VERDICT_SCORES:
            return None
        return float(VERDICT_SCORES[entry["verdict"]]), str(entry["verdict"]), str(entry.get("reason") or "")

    def set(self, key: str, verdict: str, reason: str) -> None:
        self._data[key] = {"score": VERDICT_SCORES.get(verdict, 0.0), "verdict": verdict, "reason": reason}
        self._dirty = True
        while len(self._data) > CACHE_MAX_ENTRIES:
            self._data.pop(next(iter(self._data)), None)

    def flush(self) -> None:
        if not self._dirty or self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data), encoding="utf-8")
            self._dirty = False
        except Exception:  # noqa: BLE001
            pass


def project_cache(project_id: int) -> AnswerJudgeCache:
    return AnswerJudgeCache(settings.DATA_DIR / "projects" / str(project_id) / "eval_judge_cache.json")


@dataclass
class ResolvedJudge:
    label: str  # e.g. "ollama:gemma4:12b" / "anthropic:claude-…" — stamped on results
    judge: JudgeFn


def _judge_from_synth_backend(backend: Any, label: str) -> JudgeFn:
    async def _judge(question: str, reference: str, prediction: str):
        raw = await backend.complete(
            build_judge_prompt(question, reference, prediction),
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=MAX_JUDGE_TOKENS,
            temperature=0.0,
            response_schema=_RESPONSE_SCHEMA,
        )
        parsed = parse_verdict(raw)
        if parsed is None:
            return None
        score, verdict, reason = parsed
        return score, verdict, reason, 0

    return _judge


def _judge_from_cloud(provider: str, model: str, api_key: str) -> JudgeFn:
    async def _judge(question: str, reference: str, prediction: str):
        from app.services.cloud_llm_service import call_anthropic_chat, call_openai_chat

        user = build_judge_prompt(question, reference, prediction)
        if provider == "anthropic":
            resp = await call_anthropic_chat(
                api_key=api_key, model=model, system_prompt=_SYSTEM_PROMPT,
                user_prompt=user, max_tokens=MAX_JUDGE_TOKENS, temperature=0.0,
            )
        else:
            api_url = "https://api.deepseek.com/v1/chat/completions" if provider == "deepseek" else None
            resp = await call_openai_chat(
                api_key=api_key, model=model, system_prompt=_SYSTEM_PROMPT,
                user_prompt=user, max_tokens=MAX_JUDGE_TOKENS, temperature=0.0,
                api_url=api_url, force_json=True,
            )
        parsed = parse_verdict(getattr(resp, "content", None))
        if parsed is None:
            return None
        score, verdict, reason = parsed
        tokens = int(getattr(resp, "prompt_tokens", 0) or 0) + int(getattr(resp, "completion_tokens", 0) or 0)
        return score, verdict, reason, tokens

    return _judge


async def resolve_answer_judge(db: Any, project_id: int, project: Any = None) -> ResolvedJudge | None:
    """Pick the judge for this project, or None (→ no ``judge_correct``)."""
    from app.services.synth_backends import BACKEND_REGISTRY, CloudLlmBackend, pick_backend

    requested = (os.getenv("EVAL_JUDGE_BACKEND") or "").strip()
    config = {}
    if project is not None:
        rc = getattr(project, "runtime_config", None)
        if isinstance(rc, dict) and isinstance(rc.get("eval_judge"), dict):
            config = rc["eval_judge"]
    if not requested:
        if config.get("enabled") is False:
            return None
        requested = str(config.get("backend") or "").strip()
    if requested.lower() == "none":
        return None

    provider_names = {"anthropic", "openai", "deepseek"}
    if requested and requested.split(":", 1)[0].lower() in provider_names:
        cloud = await _cloud_config(db, project_id, requested.split(":", 1)[0].lower())
        if cloud is None:
            _LOG.warning("eval judge %r requested but no API key is configured", requested)
            return None
        if ":" in requested:
            cloud["model"] = requested.split(":", 1)[1]
        return ResolvedJudge(
            label=f"{cloud['provider']}:{cloud['model']}",
            judge=_judge_from_cloud(cloud["provider"], cloud["model"], cloud["api_key"]),
        )
    if requested:
        try:
            backend = pick_backend(requested)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("eval judge %r unavailable: %s", requested, exc)
            return None
        return ResolvedJudge(label=await _backend_label(backend), judge=_judge_from_synth_backend(backend, requested))

    # Nothing requested: a local model first, then a configured cloud judge.
    for cls in BACKEND_REGISTRY:
        if cls is CloudLlmBackend:
            continue
        try:
            if cls.is_available():
                backend = cls()
                return ResolvedJudge(label=await _backend_label(backend), judge=_judge_from_synth_backend(backend, cls.name))
        except Exception:  # noqa: BLE001
            continue
    cloud = await _cloud_config(db, project_id, None)
    if cloud is not None:
        return ResolvedJudge(
            label=f"{cloud['provider']}:{cloud['model']}",
            judge=_judge_from_cloud(cloud["provider"], cloud["model"], cloud["api_key"]),
        )
    return None


async def _backend_label(backend: Any) -> str:
    """``describe()`` with the model resolved — Ollama picks its model
    lazily, and the label must name it (results judged by different
    models are never compared silently)."""
    resolve = getattr(backend, "_resolve_model", None)
    if callable(resolve):
        try:
            await resolve()
        except Exception:  # noqa: BLE001
            pass
    try:
        return str(backend.describe())
    except Exception:  # noqa: BLE001
        return str(getattr(backend, "name", "backend"))


async def _cloud_config(db: Any, project_id: int, provider: str | None) -> dict[str, str] | None:
    """The probe judge's provider resolution, optionally pinned to one provider."""
    from app.services.evaluation_service import _resolve_probe_judge_config

    try:
        cfg = await _resolve_probe_judge_config(db, project_id)
    except Exception:  # noqa: BLE001
        return None
    if not cfg or not cfg.get("api_key") or not cfg.get("model"):
        return None
    if provider and cfg.get("provider") != provider:
        return None
    return {"provider": str(cfg["provider"]), "model": str(cfg["model"]), "api_key": str(cfg["api_key"])}


async def judge_predictions(
    predictions: list[dict[str, Any]],
    judge: JudgeFn,
    *,
    label: str,
    cache: AnswerJudgeCache | None = None,
    progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Annotate each prediction in place with ``row_judge_score`` /
    ``row_judge_verdict`` / ``row_judge_reason`` and return the snapshot
    for ``metrics["judge"]``. Rows the judge couldn't score stay
    unannotated and are counted as ``unjudged``."""
    counts = {"correct": 0, "partial": 0, "wrong": 0}
    scores: list[float] = []
    unjudged = 0
    calls = 0
    cached = 0
    tokens = 0
    for index, p in enumerate(predictions):
        if not isinstance(p, dict):
            unjudged += 1
            continue
        question = str(p.get("prompt") or "")
        reference = str(p.get("reference") or "")
        prediction = str(p.get("prediction") or "")
        key = cache_key(label, question, reference, prediction)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            score, verdict, reason = hit
            cached += 1
        else:
            result = None
            try:
                result = await judge(question, reference, prediction)
            except Exception as exc:  # noqa: BLE001 — the judge never breaks the eval
                _LOG.warning("answer judge failed on row %s: %s", index, exc)
            calls += 1
            if result is None:
                unjudged += 1
                continue
            score, verdict, reason, row_tokens = result
            tokens += int(row_tokens or 0)
            if cache is not None:
                cache.set(key, verdict, reason)
        p["row_judge_score"] = float(score)
        p["row_judge_verdict"] = verdict
        p["row_judge_reason"] = reason
        counts[verdict] = counts.get(verdict, 0) + 1
        scores.append(float(score))
        if progress is not None:
            await progress(index + 1, len(predictions))
    if cache is not None:
        cache.flush()
    judged = len(scores)
    return {
        "judge": label,
        "metric": JUDGE_METRIC,
        "judged": judged,
        "unjudged": unjudged,
        "counts": counts,
        "score": round(sum(scores) / judged, 4) if judged else None,
        "strict_correct_rate": round(counts["correct"] / judged, 4) if judged else None,
        "judge_calls": calls,
        "judge_cached": cached,
        "judge_tokens": tokens,
    }
