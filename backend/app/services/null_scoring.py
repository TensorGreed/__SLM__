"""Opt-in "null counts as absent" scoring for JSON answers (any task whose scorer compares JSON to gold).

Many JSON APIs treat ``"key": null`` as "not given". With this setting on, a ``null`` value counts the same
as the key being left out when a prediction is compared to its gold, on both sides and in nested objects
(``output_schema_service.drop_nulls``). Readers: structured-extraction field match (per-field values,
nested objects, whole-object match) and tool-call argument match (default and strict). Required-field and
type / schema checks still read the raw prediction.

Default OFF. Set per project (``runtime_config["json_scoring"]["null_as_absent"] = true``,
``GET/PUT /api/projects/{id}/settings/null-as-absent``) or per held-out eval run
(``HeldoutEvalRunRequest.null_as_absent``: true / false overrides the project, null follows it).

Provenance: an eval scored with it on carries ``metrics.json_scoring = {null_as_absent: true, source:
"project" | "run"}`` (absent when off, so other results are unchanged). Compare refuses to pair two results
whose settings differ, the lift summary returns ``not_comparable`` for such a pair, and the lift check never
reuses a cached base result scored under the other setting.
"""

from __future__ import annotations

from typing import Any

SETTING_KEY = "json_scoring"
METRICS_KEY = "json_scoring"


def project_setting(runtime_config: Any) -> bool:
    """The project's saved setting (False when unset)."""
    block = runtime_config.get(SETTING_KEY) if isinstance(runtime_config, dict) else None
    return isinstance(block, dict) and block.get("null_as_absent") is True


def with_project_setting(runtime_config: Any, enabled: bool | None) -> dict[str, Any]:
    """``runtime_config`` with the setting saved (True) or removed (False / None: the default, off)."""
    cfg = dict(runtime_config or {}) if isinstance(runtime_config, dict) else {}
    block = dict(cfg.get(SETTING_KEY) or {}) if isinstance(cfg.get(SETTING_KEY), dict) else {}
    if enabled:
        block["null_as_absent"] = True
    else:
        block.pop("null_as_absent", None)
    if block:
        cfg[SETTING_KEY] = block
    else:
        cfg.pop(SETTING_KEY, None)
    return cfg


def effective(runtime_config: Any, override: bool | None = None) -> tuple[bool, str]:
    """``(on, source)``: a run's explicit override wins (``"run"``), else the project setting."""
    if override is not None:
        return bool(override), "run"
    return project_setting(runtime_config), "project"


def record(enabled: bool, source: str) -> dict[str, Any] | None:
    """``metrics.json_scoring`` for an eval scored with the setting on; None when off."""
    return {"null_as_absent": True, "source": source} if enabled else None


def result_setting(metrics: Any) -> bool:
    """Whether a stored eval result was scored with null-as-absent on."""
    block = metrics.get(METRICS_KEY) if isinstance(metrics, dict) else None
    return isinstance(block, dict) and block.get("null_as_absent") is True


def label(enabled: bool) -> str:
    return "null counts as absent" if enabled else "null is a value"


def difference(metrics_a: Any, metrics_b: Any) -> str | None:
    """Why two results cannot be paired on this setting, or None when they agree."""
    a, b = result_setting(metrics_a), result_setting(metrics_b)
    if a == b:
        return None
    return f"scored with different null handling ({label(a)} vs {label(b)})"
