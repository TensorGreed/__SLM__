"""Compliance bundles: named groups of entity types, each type tagged with
the regimes (PCI-DSS, GDPR, HIPAA, CCPA, or the user's own tags) that care
about it.

The mechanism is generic — a bundle is DATA: a regime tag plus a list of
entity types with one-line labelling definitions. Applying a bundle to a
span project:

* adds its types' definitions to the project's labelling conventions
  (``runtime_config["span_type_definitions"]``) — only for types that have
  no definition yet, so a definition the user wrote is never overwritten;
* records the regime tag on each type in
  ``runtime_config["compliance"] = {bundles, custom, type_regimes}``;
* returns the resulting entity-type list for the span tools.

``regime_scores`` turns a span evaluation's ``metrics.per_class`` (tp / fp
/ fn per type) into micro P / R / F1 per regime — recall is the headline:
for redaction, a missed span is the leak. Each built-in bundle also carries
a ``recall_floor``; ``suggested_recall_floors`` offers it for the regime's
types, but the floors the threshold tuner uses are only ever set by the user
(``span_threshold_service``, ``runtime_config["span_recall_floors"]``).
"""

from __future__ import annotations

import re
from typing import Any

from app.services.span_type_definitions_service import (
    DEFINITIONS_KEY,
    _SUGGESTIONS,
    _normalise_type,
    normalise_definitions,
)

COMPLIANCE_KEY = "compliance"
MAX_TAGS_PER_TYPE = 20
MAX_TAG_CHARS = 40

# Definitions for bundle types that span_type_definitions_service has no
# suggestion for. Same style: what the type is AND what it is not.
_EXTRA_DEFINITIONS: dict[str, str] = {
    "device_id": "a device or advertising identifier (IMEI, MAC address, IDFA/AAID, serial number tied to a person's device); not an IP address",
    "geolocation": "precise coordinates or a location fix tied to a person (lat/long, GPS fix); a city or country alone is not geolocation",
    "username": "an account username, login or handle; not an email address",
}


def type_definition(name: str) -> str | None:
    return _SUGGESTIONS.get(name) or _EXTRA_DEFINITIONS.get(name)


# Built-in starter bundles — data, not handlers. Each type list is a
# starting point the user edits (types and regime tags both).
BUILTIN_BUNDLES: list[dict[str, Any]] = [
    {
        "id": "pci_dss",
        "regime": "pci_dss",
        "recall_floor": 0.95,
        "name": "PCI-DSS",
        "description": "Payment card and bank data (cardholder data + sensitive authentication data).",
        "types": ["credit_card", "card_expiry", "cvv", "bank_account", "iban"],
    },
    {
        "id": "gdpr",
        "regime": "gdpr",
        "recall_floor": 0.9,
        "name": "GDPR personal data",
        "description": "Data that identifies a natural person directly or indirectly.",
        "types": [
            "person_name", "email", "phone", "address", "date_of_birth",
            "national_id", "passport", "ip_address", "device_id", "geolocation",
        ],
    },
    {
        "id": "hipaa",
        "regime": "hipaa",
        "recall_floor": 0.95,
        "name": "HIPAA PHI",
        "description": "Protected health information: health details plus the identifiers HIPAA counts.",
        "types": [
            "medical_record_number", "health_plan_id", "diagnosis", "medication",
            "provider_name", "visit_date", "person_name", "address", "date_of_birth",
            "phone", "email", "national_id", "ip_address", "license_plate", "device_id",
        ],
    },
    {
        "id": "ccpa",
        "regime": "ccpa",
        "recall_floor": 0.9,
        "name": "CCPA personal information",
        "description": "Identifiers and financial data the California Consumer Privacy Act covers.",
        "types": [
            "person_name", "email", "phone", "address", "national_id", "passport",
            "license_plate", "ip_address", "device_id", "username", "bank_account",
            "credit_card", "geolocation",
        ],
    },
]

BUILTIN_REGIME_LABELS: dict[str, str] = {b["regime"]: b["name"].split(" ")[0] for b in BUILTIN_BUNDLES}
_BUNDLES_BY_ID: dict[str, dict[str, Any]] = {b["id"]: b for b in BUILTIN_BUNDLES}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def normalise_tag(value: Any) -> str:
    return _norm(value)[:MAX_TAG_CHARS]


def regime_label(tag: str) -> str:
    return BUILTIN_REGIME_LABELS.get(tag) or tag.replace("_", " ")


def list_bundles() -> list[dict[str, Any]]:
    """The built-in bundles with each type's starting definition."""
    return [
        {
            "id": b["id"],
            "regime": b["regime"],
            "regime_label": regime_label(b["regime"]),
            "name": b["name"],
            "description": b["description"],
            "recall_floor": b.get("recall_floor"),
            "types": [{"type": t, "definition": type_definition(t)} for t in b["types"]],
        }
        for b in BUILTIN_BUNDLES
    ]


def normalise_type_regimes(raw: Any) -> dict[str, list[str]]:
    """``{type: [regime tags]}`` with names normalised, tags de-duplicated
    (order kept) and types without tags dropped."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, tags in raw.items():
        name = _normalise_type(key)
        if not name or not isinstance(tags, (list, tuple)):
            continue
        clean: list[str] = []
        for tag in tags:
            t = normalise_tag(tag)
            if t and t not in clean:
                clean.append(t)
        if clean:
            out[name] = clean[:MAX_TAGS_PER_TYPE]
    return out


def compliance_state(runtime_config: Any) -> dict[str, Any]:
    """The project's normalised compliance block."""
    cfg = runtime_config if isinstance(runtime_config, dict) else {}
    raw = cfg.get(COMPLIANCE_KEY) if isinstance(cfg.get(COMPLIANCE_KEY), dict) else {}
    type_regimes = normalise_type_regimes(raw.get("type_regimes"))
    bundles = [b for b in (raw.get("bundles") or []) if b in _BUNDLES_BY_ID]
    return {
        "bundles": list(dict.fromkeys(bundles)),
        "custom": _custom_tags(type_regimes),
        "type_regimes": type_regimes,
    }


def _custom_tags(type_regimes: dict[str, list[str]]) -> list[str]:
    tags: list[str] = []
    for regimes in type_regimes.values():
        for tag in regimes:
            if tag not in BUILTIN_REGIME_LABELS and tag not in tags:
                tags.append(tag)
    return sorted(tags)


def entity_types(state: dict[str, Any]) -> list[str]:
    return list(state.get("type_regimes") or {})


def suggested_recall_floors(runtime_config: Any) -> dict[str, dict[str, Any]]:
    """Suggested minimum recalls for the project's regime-tagged types:
    ``{type: {floor, regimes}}`` — the highest ``recall_floor`` among the
    built-in regimes the type is tagged with (custom tags carry none). A
    suggestion only: the tuner reads ``runtime_config["span_recall_floors"]``
    and nothing here writes it — the user accepts a suggestion in the UI."""
    floors = {b["regime"]: float(b["recall_floor"]) for b in BUILTIN_BUNDLES if b.get("recall_floor") is not None}
    out: dict[str, dict[str, Any]] = {}
    for ent_type, regimes in compliance_state(runtime_config)["type_regimes"].items():
        tagged = [r for r in regimes if r in floors]
        if tagged:
            out[ent_type] = {"floor": max(floors[r] for r in tagged), "regimes": tagged}
    return out


def apply_bundles(
    runtime_config: Any,
    bundle_ids: list[str],
    remove_bundle_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pure: apply (and optionally remove) bundles on a runtime_config.

    Returns ``(new_runtime_config, result)``. Definitions are added only for
    types that have none — a user's own wording always wins. Removing a
    bundle takes its regime tag off its types (a type left with no tag
    leaves the map); definitions stay, they are the project's convention.
    Raises ``ValueError`` on an unknown bundle id."""
    unknown = [b for b in list(bundle_ids) + list(remove_bundle_ids or []) if b not in _BUNDLES_BY_ID]
    if unknown:
        raise ValueError(f"Unknown compliance bundle(s): {', '.join(unknown)}")
    cfg = dict(runtime_config) if isinstance(runtime_config, dict) else {}
    state = compliance_state(cfg)
    definitions = normalise_definitions(cfg.get(DEFINITIONS_KEY))
    type_regimes = {t: list(r) for t, r in state["type_regimes"].items()}
    applied = list(state["bundles"])
    added_definitions: list[str] = []
    kept_definitions: list[str] = []

    for bundle_id in remove_bundle_ids or []:
        bundle = _BUNDLES_BY_ID[bundle_id]
        for t in bundle["types"]:
            if t in type_regimes:
                type_regimes[t] = [r for r in type_regimes[t] if r != bundle["regime"]]
                if not type_regimes[t]:
                    del type_regimes[t]
        applied = [b for b in applied if b != bundle_id]

    for bundle_id in bundle_ids:
        bundle = _BUNDLES_BY_ID[bundle_id]
        for t in bundle["types"]:
            regimes = type_regimes.setdefault(t, [])
            if bundle["regime"] not in regimes:
                regimes.append(bundle["regime"])
            if t in definitions:
                if t not in kept_definitions:
                    kept_definitions.append(t)
            else:
                text = type_definition(t)
                if text:
                    definitions[t] = text
                    added_definitions.append(t)
        if bundle_id not in applied:
            applied.append(bundle_id)

    new_state = {"bundles": applied, "custom": _custom_tags(type_regimes), "type_regimes": type_regimes}
    cfg[COMPLIANCE_KEY] = new_state
    cfg[DEFINITIONS_KEY] = definitions
    return cfg, {
        **new_state,
        "entity_types": entity_types(new_state),
        "definitions_added": added_definitions,
        "definitions_kept": kept_definitions,
    }


def set_type_regimes(runtime_config: Any, type_regimes: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pure: replace the type→regimes map (the user's edits, custom tags
    included). Applied bundle ids stay as recorded."""
    cfg = dict(runtime_config) if isinstance(runtime_config, dict) else {}
    state = compliance_state(cfg)
    mapping = normalise_type_regimes(type_regimes)
    new_state = {"bundles": state["bundles"], "custom": _custom_tags(mapping), "type_regimes": mapping}
    cfg[COMPLIANCE_KEY] = new_state
    return cfg, {**new_state, "entity_types": entity_types(new_state)}


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def regime_scores(per_class: Any, type_regimes: Any) -> list[dict[str, Any]]:
    """Micro P / R / F1 per regime from a span evaluation's per-type
    tp / fp / fn. A regime type the evaluation never saw (no gold and no
    predicted spans) is listed in ``unscored_types``. Weakest recall first
    (unscored regimes last)."""
    if not isinstance(per_class, dict):
        return []
    mapping = normalise_type_regimes(type_regimes)
    regimes: dict[str, list[str]] = {}
    for t, tags in mapping.items():
        for tag in tags:
            regimes.setdefault(tag, []).append(t)
    out: list[dict[str, Any]] = []
    for tag, types in regimes.items():
        tp = fp = fn = 0
        scored: list[str] = []
        unscored: list[str] = []
        for t in types:
            row = per_class.get(t)
            if not isinstance(row, dict) or not any(isinstance(row.get(k), (int, float)) for k in ("tp", "fp", "fn")):
                unscored.append(t)
                continue
            tp += int(row.get("tp") or 0)
            fp += int(row.get("fp") or 0)
            fn += int(row.get("fn") or 0)
            scored.append(t)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision is not None and recall is not None and (precision + recall) > 0
            else (0.0 if precision is not None and recall is not None else None)
        )
        out.append({
            "regime": tag,
            "label": regime_label(tag),
            "custom": tag not in BUILTIN_REGIME_LABELS,
            "types": scored,
            "unscored_types": unscored,
            "tp": tp, "fp": fp, "fn": fn,
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    out.sort(key=lambda r: (r["recall"] is None, r["recall"] if r["recall"] is not None else 0.0, r["regime"]))
    return out
