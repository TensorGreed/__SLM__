"""Pipeline plan refinement — Phase 1 (deterministic, no cloud LLM).

The post-data pass: once a project has data, assess how well its current
pipeline *plan* (recipe / task shape / base model / target) fits the *measured*
data, and assemble a privacy-safe aggregate profile that a future cloud-LLM
strategy pass (Phase 2) can reason over.

PRIVACY BOUNDARY — load-bearing
================================
``cloud_safe_profile`` is the **only** structure a Phase-2 cloud-LLM call may
ever send off-box. It contains exclusively aggregate signals — counts, rates,
distribution *shape*, severities, and config strings the user themselves chose
(recipe id, base model, target profile). It must NEVER contain ingested rows,
document text, gold answers, or even label names.

It's built by *whitelisting scalar fields*, never by copying a signal's nested
``context`` (which can carry raw row excerpts — e.g. the leakage signals stash
matched rows under ``context.examples``). ``test_pipeline_refinement`` asserts
no row/text/label-name leaks into ``cloud_safe_profile``.

Provider note (Phase 2): the cloud call will ride the existing
``cloud_llm_service`` OpenAI-compatible path — which already covers DeepSeek —
plus Qwen via a DashScope-compatible base URL or a local Ollama model. Phase 1
makes no call; ``cloud_refinement.available`` is always False here.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project

# A strategy fn takes the cloud-safe profile and returns the raw LLM dict (or
# None on any failure). Injectable so tests never touch the network — the same
# pattern as probe_runner's judge_fn.
StrategyFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


# Severity → plan-health bucket. Gap/forecast signals use ok / warn / block.
def _bucket(severity: str | None) -> str:
    s = str(severity or "").lower()
    if s == "block":
        return "mismatch"
    if s in ("warn", "attention"):
        return "attention"
    return "ready"


def _label_distribution_shape(counts: dict[str, int]) -> dict[str, Any] | None:
    """Distribution *shape* metrics from a {label: count} map — counts only,
    NO label names (the names are user-chosen vocabulary we keep off-box)."""
    from app.services.trainability_forecast_service import PER_CLASS_MINIMUM

    if not counts:
        return None
    values = list(counts.values())
    min_count = min(values)
    max_count = max(values)
    return {
        "num_classes": len(counts),
        "min_class_count": int(min_count),
        "max_class_count": int(max_count),
        # 1.0 = perfectly balanced, → 0 as the rarest class shrinks vs the biggest.
        "imbalance_ratio": round(min_count / max_count, 4) if max_count else 0.0,
        "classes_below_floor": sum(1 for v in values if v < PER_CLASS_MINIMUM),
    }


def _gap_severity(gaps_report: dict[str, Any], signal_id: str) -> str | None:
    """Pull a single training-config-gap signal's severity by id (aggregate)."""
    for group in gaps_report.get("groups") or []:
        for signal in group.get("signals") or []:
            if signal.get("id") == signal_id:
                return str(signal.get("severity") or "") or None
    return None


def _archetype_below_band_features(comparison: dict[str, Any]) -> list[str]:
    """Feature *names* that sit below the archetype's expected band (e.g.
    ``answer_length``). Feature names describe the task shape, not user data."""
    out: list[str] = []
    features = comparison.get("features") if isinstance(comparison, dict) else None
    for feat in features or []:
        if not isinstance(feat, dict):
            continue
        status = str(feat.get("status") or feat.get("band") or "").lower()
        below = bool(feat.get("below_band")) or status in ("below", "low", "under")
        if below:
            name = feat.get("name") or feat.get("feature") or feat.get("id")
            if name:
                out.append(str(name))
    return out


async def build_cloud_safe_profile(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Aggregate, off-box-safe data + plan profile. See the module PRIVACY
    BOUNDARY note — only whitelisted scalars/shapes, never raw content."""
    from app.services.trainability_forecast_service import (
        _estimate_train_row_count,
        _label_counts,
        _load_gold_rows,
        forecast_training,
    )
    from app.services.training_config_gap_service import scan_training_config_gaps

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    task_profile = selected_recipe.get("task_profile") or selected_recipe.get("task_family")

    gold_rows = await _load_gold_rows(db, project_id)
    label_shape = _label_distribution_shape(_label_counts(gold_rows))
    labelled_row_count = await _estimate_train_row_count(db, project_id)

    # Aggregate severities — best-effort; a producer failure must not sink the
    # whole profile (it just narrows it).
    truncation_risk: str | None = None
    tokenizer_oov: str | None = None
    try:
        gaps = await scan_training_config_gaps(db, project_id)
        truncation_risk = _gap_severity(gaps, "training_config.max_seq_truncation_risk")
        tokenizer_oov = _gap_severity(gaps, "training_config.tokenizer_oov_high")
    except Exception:  # noqa: BLE001 — narrow the profile, don't fail it
        pass

    forecast_verdict: str | None = None
    try:
        forecast = await forecast_training(db, project_id)
        forecast_verdict = str(forecast.get("overall") or "") or None
    except Exception:  # noqa: BLE001
        pass

    archetype_features: list[str] = []
    try:
        from app.services.archetype_service import compare_project_to_archetype

        comparison = await compare_project_to_archetype(db, project_id)
        archetype_features = _archetype_below_band_features(comparison)
    except Exception:  # noqa: BLE001
        pass

    # WHITELIST — every field here is an aggregate/scalar/enum or user-chosen
    # config. No raw rows, document text, gold answers, or label names.
    return {
        "recipe_id": recipe_id,
        "task_profile": task_profile,
        "base_model_name": project.base_model_name or None,
        "target_profile_id": project.target_profile_id or None,
        "labelled_row_count": int(labelled_row_count),
        "label_distribution_shape": label_shape,
        "truncation_risk": truncation_risk,
        "tokenizer_oov": tokenizer_oov,
        "archetype_below_band_features": archetype_features,
        "forecast_verdict": forecast_verdict,
    }


def _signal(sid: str, severity: str, headline: str, *, target_tab: str | None = None) -> dict[str, Any]:
    return {"id": sid, "severity": severity, "headline": headline, "target_tab": target_tab}


def assess_plan_health(profile: dict[str, Any], *, recipe_min_rows: int | None) -> dict[str, Any]:
    """Deterministic plan-fit lens: does the *current plan* suit the *measured
    data*? Distinct from the data-health / config-gap panels — this cross-checks
    plan ↔ data. Verdict = worst signal (mismatch > attention > ready)."""
    signals: list[dict[str, Any]] = []

    if not profile.get("recipe_id"):
        signals.append(_signal(
            "plan.no_recipe", "block",
            "No recipe selected — pick a task shape before refining the plan.",
            target_tab="data",
        ))

    verdict = str(profile.get("forecast_verdict") or "")
    if verdict == "likely_fail":
        signals.append(_signal(
            "plan.forecast_likely_fail", "block",
            "Trainability forecast says this run would likely fail on the current data.",
            target_tab="training-config",
        ))
    elif verdict == "borderline":
        signals.append(_signal(
            "plan.forecast_borderline", "warn",
            "Trainability forecast is borderline — the plan may underdeliver without more/cleaner data.",
            target_tab="training-config",
        ))

    rows = int(profile.get("labelled_row_count") or 0)
    if recipe_min_rows and rows < int(recipe_min_rows):
        signals.append(_signal(
            "plan.rows_below_recommended", "warn",
            f"{rows} labelled rows is below the ~{int(recipe_min_rows)} recommended for this task shape.",
            target_tab="goldset",
        ))

    shape = profile.get("label_distribution_shape") or {}
    below_floor = int(shape.get("classes_below_floor") or 0)
    if below_floor > 0:
        signals.append(_signal(
            "plan.classes_below_floor", "warn",
            f"{below_floor} class(es) sit below the per-class floor — the model will underfit them.",
            target_tab="synthetic",
        ))

    for sid, label in (("truncation_risk", "max_seq_length truncation"),
                       ("tokenizer_oov", "tokenizer OOV")):
        bucket = _bucket(profile.get(sid))
        if bucket == "mismatch":
            signals.append(_signal(
                f"plan.{sid}", "block",
                f"High {label} risk for this base model — the plan will train on corrupted inputs.",
                target_tab="training-config",
            ))
        elif bucket == "attention":
            signals.append(_signal(
                f"plan.{sid}", "warn",
                f"Elevated {label} risk — worth resolving before training.",
                target_tab="training-config",
            ))

    if any(s["severity"] == "block" for s in signals):
        overall = "mismatch"
    elif any(s["severity"] == "warn" for s in signals):
        overall = "attention"
    else:
        overall = "ready"
    return {"verdict": overall, "signals": signals}


# ─────────────────────────────────────────────────────────────────────
# Phase 2 — cloud LLM strategy pass (best-effort, validated, privacy-safe).
# ─────────────────────────────────────────────────────────────────────

# What the LLM is allowed to touch. Anything outside these menus is dropped —
# the LLM proposes a *strategy*; the deterministic engine owns the numbers and
# vetoes anything off-menu, so a hallucinated config can't reach the user.
_KNOWN_TASK_PROFILES = {
    "instruction_sft", "chat_sft", "qa", "rag_qa", "tool_calling",
    "structured_extraction", "summarization", "seq2seq", "classification", "preference",
}
_KNOWN_SIZE_CLASSES = {"small", "mid", "large"}
_KNOWN_TRAINING_MODES = {"sft", "distillation"}
# Directional config kinds map 1:1 to the deterministic patch engine's
# apply_patch_kinds (+ two safe extras). The LLM gives a *direction*; the real
# number comes from training_config_gap_service, never the LLM.
_KNOWN_DIRECTIONAL_KINDS = {
    "eval_steps_recommend", "num_epochs_recommend", "warmup_ratio_recommend",
    "max_seq_length_raise", "stratify_split",
}
# Each data-gap kind only survives if the deterministic profile independently
# supports it — the LLM can *explain* a gap, never *invent* one.
_DATA_GAP_KINDS = {"class_balance", "more_rows", "seq_length", "leakage", "label_consistency"}

_REFINE_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "qwen": "qwen-plus",
}
# DeepSeek + Qwen ride the OpenAI-compatible path with their own base URLs.
_OPENAI_COMPAT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
}

_PLAN_REFINEMENT_KEY = "plan_refinement"  # runtime_config cache slot


def _data_gap_supported(kind: str, profile: dict[str, Any]) -> bool:
    """Deterministic precondition for a data-gap kind — the anti-hallucination
    gate. The profile must independently evidence the gap."""
    shape = profile.get("label_distribution_shape") or {}
    verdict = str(profile.get("forecast_verdict") or "")
    if kind == "class_balance":
        return int(shape.get("classes_below_floor") or 0) > 0 or (
            float(shape.get("imbalance_ratio") or 1.0) < 0.34
        )
    if kind == "more_rows":
        return verdict in ("borderline", "likely_fail")
    if kind == "seq_length":
        return _bucket(profile.get("truncation_risk")) != "ready"
    if kind == "label_consistency":
        return _bucket(profile.get("tokenizer_oov")) != "ready"
    if kind == "leakage":
        # Leakage isn't in the cloud profile (it carries raw matches), so the
        # LLM can't see it — never accept a leakage gap from the model.
        return False
    return False


def validate_strategy(raw: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Clamp the raw LLM strategy to the known menus + cross-check data gaps
    against the deterministic profile. Pure; the trust-preserving core.

    Returns a sanitized refinement with ``dropped`` counts so the UI can be
    honest about what the model said vs. what survived validation."""
    from app.services.recipe_service import get_recipe

    raw = raw if isinstance(raw, dict) else {}
    dropped: dict[str, int] = {"plan_delta": 0, "directional": 0, "data_gaps": 0}

    # plan_delta — keep only valid fields that represent an actual change.
    plan_delta: dict[str, Any] = {}
    rd = raw.get("plan_delta") if isinstance(raw.get("plan_delta"), dict) else {}
    recipe_id = rd.get("recipe_id")
    if isinstance(recipe_id, str) and recipe_id.strip() and recipe_id != profile.get("recipe_id"):
        try:
            if get_recipe(recipe_id) is not None:
                plan_delta["recipe_id"] = recipe_id
            else:
                dropped["plan_delta"] += 1
        except Exception:  # noqa: BLE001
            dropped["plan_delta"] += 1
    tp = rd.get("task_profile")
    if isinstance(tp, str) and tp in _KNOWN_TASK_PROFILES and tp != profile.get("task_profile"):
        plan_delta["task_profile"] = tp
    elif isinstance(tp, str) and tp and tp not in _KNOWN_TASK_PROFILES:
        dropped["plan_delta"] += 1
    sc = rd.get("base_model_size_class")
    if isinstance(sc, str) and sc in _KNOWN_SIZE_CLASSES:
        plan_delta["base_model_size_class"] = sc
    if isinstance(rd.get("rag_first"), bool):
        plan_delta["rag_first"] = rd["rag_first"]
    tm = rd.get("training_mode")
    if isinstance(tm, str) and tm in _KNOWN_TRAINING_MODES:
        plan_delta["training_mode"] = tm

    # directional_config — keep only known kinds.
    directional: list[dict[str, Any]] = []
    for item in raw.get("directional_config") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind in _KNOWN_DIRECTIONAL_KINDS:
            directional.append({
                "kind": kind,
                "direction": str(item.get("direction") or "")[:24] or None,
                "reason": str(item.get("reason") or "")[:280],
            })
        else:
            dropped["directional"] += 1

    # data_gaps — only those the deterministic profile evidences.
    data_gaps: list[dict[str, Any]] = []
    for item in raw.get("data_gaps") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind in _DATA_GAP_KINDS and _data_gap_supported(kind, profile):
            entry: dict[str, Any] = {
                "kind": kind,
                "detail": str(item.get("detail") or "")[:280],
            }
            count = item.get("suggested_count")
            if isinstance(count, (int, float)) and 0 < count <= 1000:
                entry["suggested_count"] = int(count)
            data_gaps.append(entry)
        else:
            dropped["data_gaps"] += 1

    confidence = raw.get("confidence")
    confidence = (
        round(float(confidence), 3)
        if isinstance(confidence, (int, float)) and 0 <= confidence <= 1
        else None
    )

    return {
        "plan_delta": plan_delta,
        "directional_config": directional,
        "data_gaps": data_gaps,
        "rationale": str(raw.get("rationale") or "")[:1200],
        "confidence": confidence,
        "dropped": dropped,
    }


def _resolve_strategy_config(db: AsyncSession, project_id: int):
    """Resolve a cloud-LLM config (provider/model/api_key/api_url) for the
    strategy pass, or ``None`` (→ deterministic fallback). Order mirrors the
    probe judge: PLAN_REFINE_* env → project secret → provider env keys.
    Returns a coroutine."""
    async def _resolve() -> dict[str, Any] | None:
        model_override = (os.getenv("PLAN_REFINE_MODEL") or "").strip()

        def _cfg(provider: str, key: str) -> dict[str, Any]:
            return {
                "provider": provider,
                "api_key": key,
                "model": model_override or _REFINE_DEFAULT_MODELS.get(provider, ""),
                "api_url": _OPENAI_COMPAT_BASE_URLS.get(provider),
            }

        prov = (os.getenv("PLAN_REFINE_PROVIDER") or "").strip().lower()
        key = (os.getenv("PLAN_REFINE_API_KEY") or "").strip()
        if prov and key:
            return _cfg(prov, key)

        if db is not None:
            try:
                from app.services.secret_service import get_project_secret_value
                for provider in ("anthropic", "openai", "deepseek", "qwen"):
                    val = await get_project_secret_value(db, project_id, provider, "api_key", touch=False)
                    if val:
                        return _cfg(provider, val)
            except Exception:  # noqa: BLE001 — best-effort
                pass

        for env_name, provider in (("ANTHROPIC_API_KEY", "anthropic"), ("OPENAI_API_KEY", "openai")):
            val = (os.getenv(env_name) or "").strip()
            if val:
                return _cfg(provider, val)
        return None

    return _resolve()


def _build_strategy_prompt(profile: dict[str, Any]) -> tuple[str, str]:
    """System + user prompt. PRIVACY: the user prompt contains ONLY the
    cloud-safe profile (aggregates) — never raw rows. The schema constrains the
    model to a strategy menu; numbers are deliberately excluded."""
    system = (
        "You are an expert ML architect advising on a small-model fine-tuning pipeline. "
        "You are given an AGGREGATE profile of a project's data + current plan — never the raw data. "
        "Recommend STRATEGY only; do NOT emit hyperparameter numbers (learning rate, epochs, etc.) — "
        "the platform computes those from the data.\n\n"
        "Return ONLY JSON: {\n"
        '  "plan_delta": { "recipe_id"?: str, "task_profile"?: one of '
        "[instruction_sft,chat_sft,qa,rag_qa,tool_calling,structured_extraction,summarization,seq2seq,classification,preference], "
        '"base_model_size_class"?: one of [small,mid,large], "rag_first"?: bool, "training_mode"?: one of [sft,distillation] },\n'
        '  "directional_config": [ { "kind": one of '
        "[eval_steps_recommend,num_epochs_recommend,warmup_ratio_recommend,max_seq_length_raise,stratify_split], "
        '"direction": str, "reason": str } ],\n'
        '  "data_gaps": [ { "kind": one of [class_balance,more_rows,seq_length,label_consistency], "detail": str, "suggested_count"?: int } ],\n'
        '  "rationale": str, "confidence": number 0..1\n'
        "}\nOmit a plan_delta field if no change is warranted."
    )
    user = "AGGREGATE PROFILE (no raw data):\n" + json.dumps(profile, sort_keys=True)
    return system, user


def _build_cloud_strategy_fn(config: dict[str, Any]) -> StrategyFn:
    """Wrap a resolved provider config into a StrategyFn that calls the cloud
    and parses JSON. DeepSeek/Qwen ride the OpenAI-compatible path."""
    async def _fn(profile: dict[str, Any]) -> dict[str, Any] | None:
        from app.services.cloud_llm_service import (
            call_anthropic_chat,
            call_openai_chat,
            extract_json_payload,
        )
        system, user = _build_strategy_prompt(profile)
        try:
            if config["provider"] == "anthropic":
                resp = await call_anthropic_chat(
                    api_key=config["api_key"], model=config["model"],
                    system_prompt=system, user_prompt=user, max_tokens=800, temperature=0.2,
                )
            else:
                resp = await call_openai_chat(
                    api_key=config["api_key"], model=config["model"],
                    system_prompt=system, user_prompt=user, max_tokens=800, temperature=0.2,
                    api_url=config.get("api_url"), force_json=True,
                )
            payload = extract_json_payload(resp.content)
            return payload if isinstance(payload, dict) else None
        except Exception:  # noqa: BLE001 — any failure → deterministic fallback
            return None

    return _fn


def _profile_hash(profile: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode("utf-8")).hexdigest()[:32]


async def run_cloud_strategy_pass(
    db: AsyncSession,
    project_id: int,
    *,
    strategy_fn: StrategyFn | None = None,
    model_label: str | None = None,
) -> dict[str, Any] | None:
    """Phase 2 — run the cloud strategy pass and return a validated refinement,
    or ``None`` (→ deterministic fallback) when no provider is configured or the
    call fails. Caches by profile hash in ``runtime_config`` so repeat reads on
    unchanged data don't re-bill. Best-effort throughout.

    Only ``cloud_safe_profile`` is ever passed to ``strategy_fn`` — the privacy
    boundary from Phase 1."""
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    profile = await build_cloud_safe_profile(db, project_id)
    phash = _profile_hash(profile)

    runtime_config = dict(project.runtime_config or {})
    cached = runtime_config.get(_PLAN_REFINEMENT_KEY)
    if isinstance(cached, dict) and cached.get("profile_hash") == phash:
        return {**cached["refinement"], "from_cache": True}

    fn = strategy_fn
    provider_label = model_label
    if fn is None:
        config = await _resolve_strategy_config(db, project_id)
        if config is None:
            return None  # no provider → fallback to deterministic only
        fn = _build_cloud_strategy_fn(config)
        provider_label = f"{config['provider']}:{config['model']}"

    raw = await fn(profile)
    if raw is None:
        return None

    refinement = validate_strategy(raw, profile)
    refinement["provenance"] = {
        "model": provider_label or "injected",
        "shared": "cloud_safe_profile",  # what left the box — aggregates only
    }

    runtime_config[_PLAN_REFINEMENT_KEY] = {"profile_hash": phash, "refinement": refinement}
    project.runtime_config = runtime_config
    await db.flush()
    await db.commit()
    return {**refinement, "from_cache": False}


# ─────────────────────────────────────────────────────────────────────
# Phase 3 — accept / apply a validated refinement through existing machinery.
# ─────────────────────────────────────────────────────────────────────

# task_profile → data adapter (mirrors nl2pipeline's mapping).
_TASK_PROFILE_TO_ADAPTER = {
    "structured_extraction": "structured-extraction",
    "rag_qa": "rag-grounded",
    "tool_calling": "tool-call-json",
}
# Coarse size class → a concrete base model (mirrors nl2pipeline's fallback).
_SIZE_CLASS_TO_BASE = {
    "small": "Qwen/Qwen1.5-1.8B-Chat",
    "mid": "microsoft/phi-2",
    "large": "meta-llama/Meta-Llama-3-8B-Instruct",
}


async def _apply_plan_delta_field(
    db: AsyncSession, project: Project, project_id: int, field: str, value: Any
) -> dict[str, Any]:
    """Apply one validated plan_delta field through the canonical machinery.
    Returns an outcome record. Never trusts raw values — ``value`` came from the
    already-validated cached refinement."""
    if field == "recipe_id":
        from app.services.pipeline_recipe_service import apply_pipeline_recipe_blueprint
        await apply_pipeline_recipe_blueprint(
            db, project_id=project_id, recipe_id=str(value),
            include_preflight=False, mark_active=True,
        )
        return {"field": field, "status": "applied", "to": str(value)}

    if field == "task_profile":
        from app.services.dataset_service import save_project_dataset_adapter_preference
        adapter = _TASK_PROFILE_TO_ADAPTER.get(str(value), "default-canonical")
        await save_project_dataset_adapter_preference(
            db, project_id, adapter_id=adapter, task_profile=str(value),
        )
        return {"field": field, "status": "applied", "to": str(value)}

    if field == "base_model_size_class":
        base = _SIZE_CLASS_TO_BASE.get(str(value))
        if not base:
            return {"field": field, "status": "skipped", "reason": "unknown_size_class"}
        project.base_model_name = base
        return {"field": field, "status": "applied", "to": base}

    if field == "rag_first" and value:
        runtime = dict(project.runtime_config or {})
        runtime["rag_first"] = True
        auto_rag = dict(runtime.get("auto_rag") or {})
        auto_rag["enabled"] = True
        runtime["auto_rag"] = auto_rag
        project.runtime_config = runtime
        return {"field": field, "status": "applied", "to": True}

    if field == "training_mode":
        # Distillation needs a teacher capture first — surface as manual, never
        # silently flip the training mode.
        return {"field": field, "status": "manual",
                "reason": "training_mode change needs the Distillation setup (teacher capture)."}

    return {"field": field, "status": "skipped", "reason": "unsupported_field"}


async def apply_strategy_refinement(
    db: AsyncSession,
    project_id: int,
    *,
    plan_delta_fields: list[str] | None = None,
    directional_kinds: list[str] | None = None,
) -> dict[str, Any]:
    """Apply selected items of the cached, *validated* refinement (Phase 3).

    The cached refinement is the source of truth — the client can only choose
    WHICH validated items to accept, never inject new ones. Directional patches
    go through ``training_config_gap_service.apply_patch``, which only lands when
    the scanner *currently* flags that gap (a second deterministic guardrail).
    Plan-delta fields go through the canonical recipe/adapter/base apply paths.
    All reversible. Raises ``ValueError`` (→ 404) on a missing project / no
    cached refinement."""
    from app.services.training_config_gap_service import (
        apply_patch,
        signal_id_for_patch_kind,
    )

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    cache = (project.runtime_config or {}).get(_PLAN_REFINEMENT_KEY)
    refinement = cache.get("refinement") if isinstance(cache, dict) else None
    if not isinstance(refinement, dict):
        raise ValueError("No refinement to apply — run the cloud strategy pass first.")

    plan_delta = refinement.get("plan_delta") or {}
    available_directional = {d.get("kind") for d in (refinement.get("directional_config") or [])}

    # Default = accept all available items.
    fields = plan_delta_fields if plan_delta_fields is not None else list(plan_delta.keys())
    kinds = directional_kinds if directional_kinds is not None else list(available_directional)

    plan_outcomes: list[dict[str, Any]] = []
    for field in fields:
        if field not in plan_delta:
            plan_outcomes.append({"field": field, "status": "skipped", "reason": "not_in_refinement"})
            continue
        try:
            plan_outcomes.append(
                await _apply_plan_delta_field(db, project, project_id, field, plan_delta[field])
            )
        except Exception as exc:  # noqa: BLE001 — one bad apply shouldn't sink the rest
            plan_outcomes.append({"field": field, "status": "error", "reason": str(exc)[:200]})

    directional_outcomes: list[dict[str, Any]] = []
    for kind in kinds:
        if kind not in available_directional:
            directional_outcomes.append({"kind": kind, "status": "skipped", "reason": "not_in_refinement"})
            continue
        signal_id = signal_id_for_patch_kind(kind)
        if not signal_id:
            # max_seq_length_raise / stratify_split have no one-click patch.
            directional_outcomes.append({"kind": kind, "status": "manual", "reason": "no_auto_patch"})
            continue
        try:
            await apply_patch(db, project_id, signal_id)
            directional_outcomes.append({"kind": kind, "status": "applied"})
        except ValueError:
            # apply_patch raises when the gap isn't currently in the report —
            # i.e. the deterministic scanner no longer agrees. Don't apply.
            directional_outcomes.append({"kind": kind, "status": "skipped", "reason": "gap_not_currently_present"})

    # Stamp what was applied onto the cache so the UI can show "applied".
    runtime = dict(project.runtime_config or {})
    cache_block = dict(runtime.get(_PLAN_REFINEMENT_KEY) or {})
    cache_block["applied"] = {
        "plan_delta": [o["field"] for o in plan_outcomes if o["status"] == "applied"],
        "directional": [o["kind"] for o in directional_outcomes if o["status"] == "applied"],
    }
    runtime[_PLAN_REFINEMENT_KEY] = cache_block
    project.runtime_config = runtime

    await db.flush()
    await db.commit()
    return {
        "project_id": int(project_id),
        "plan_delta": plan_outcomes,
        "directional_config": directional_outcomes,
        "applied": cache_block["applied"],
    }


async def refine_pipeline_plan(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Phase 1 — deterministic plan-refinement report: current plan + the
    privacy-safe aggregate profile + a plan-fit roll-up. No cloud call (but it
    surfaces a previously-cached Phase-2 refinement when one matches the current
    data, and whether a cloud provider is configured)."""
    from app.services.recipe_service import get_recipe

    profile = await build_cloud_safe_profile(db, project_id)

    recipe_min_rows: int | None = None
    recipe_id = profile.get("recipe_id")
    if recipe_id:
        try:
            recipe = get_recipe(str(recipe_id))
            if recipe is not None:
                recipe_min_rows = int(recipe.gold_template.min_rows_recommended)
        except Exception:  # noqa: BLE001
            recipe_min_rows = None

    plan_health = assess_plan_health(profile, recipe_min_rows=recipe_min_rows)

    # Cheap, free-of-charge cloud readiness: is a provider configured, and is
    # there a cached Phase-2 refinement still valid for the current data? No
    # cloud call happens on this GET — the billable call is POST /refine-plan/cloud.
    project = await db.get(Project, project_id)
    provider_config = await _resolve_strategy_config(db, project_id)
    cached_refinement = None
    cached = (project.runtime_config or {}).get(_PLAN_REFINEMENT_KEY) if project else None
    if isinstance(cached, dict) and cached.get("profile_hash") == _profile_hash(profile):
        cached_refinement = {
            **cached["refinement"],
            "from_cache": True,
            "applied": cached.get("applied"),
        }

    return {
        "project_id": int(project_id),
        "plan": {
            "recipe_id": profile.get("recipe_id"),
            "task_profile": profile.get("task_profile"),
            "base_model_name": profile.get("base_model_name"),
            "target_profile_id": profile.get("target_profile_id"),
        },
        "cloud_safe_profile": profile,
        "plan_health": plan_health,
        "refinement": cached_refinement,
        "privacy": {
            "cloud_sharing": "aggregate_only",
            "note": (
                "Only the aggregate signals in cloud_safe_profile are ever eligible "
                "to be sent to a cloud model. Your ingested rows, document text, "
                "gold answers, and label names never leave BrewSLM."
            ),
        },
        "cloud_refinement": {
            # Configured = a provider resolves; the strategy pass is a separate
            # billable POST. Off → the report stays fully deterministic.
            "available": provider_config is not None,
            "supported_providers": ["anthropic", "openai", "deepseek", "qwen", "ollama"],
            "reason": "configured" if provider_config is not None else "no_provider_configured",
        },
    }
