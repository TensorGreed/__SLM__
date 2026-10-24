"""Declared output schema for typed JSON-record extraction + typed scalar compare.

A project on the ``json-extraction`` recipe may declare what its JSON record
looks like (a JSON-Schema subset)::

    {"required": ["damage", "injuries_fatal"],
     "properties": {
        "damage": {"type": ["string", "null"], "enum": ["destroyed", "minor", null]},
        "injuries_fatal": {"type": ["integer", "null"]}}}

It is stored on ``Project.selected_recipe["output_schema"]``, merged into the
prepared manifest's ``output_schema`` (``dataset_service._recipe_output_schema``)
and read by ``StructuredExtractionHandler`` (field list, enums, types).

This module also holds the typed scalar comparison the ``field_match`` scorer
uses, so the rules live in one place:

* NULL class: JSON null, a missing key, ``"null"`` and ``""`` (stripped,
  case-insensitive). ``"none"`` is NOT null (it is a legitimate enum value).
* numbers compare numerically (``1250.0`` == ``"1,250"``);
* booleans: ``true`` == ``"true"`` (casefold);
* strings: casefold, strip, collapse whitespace, drop one trailing period.
  Articles and internal punctuation are kept (``"the destroyed"`` != ``"destroyed"``).
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SUPPORTED_TYPES = ("string", "integer", "number", "boolean", "null", "object", "array")
_SCALAR_NUMERIC = {"integer", "number"}
_MAX_PROPERTIES = 200
_MAX_ENUM = 500

MISSING: Any = object()


# ── Declaration model ────────────────────────────────────────────────


class OutputProperty(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | list[str] | None = None
    enum: list[Any] | None = None

    @field_validator("type")
    @classmethod
    def _check_type(cls, value: str | list[str] | None) -> str | list[str] | None:
        if value is None:
            return None
        items = [value] if isinstance(value, str) else list(value)
        if not items:
            raise ValueError("type must not be an empty list")
        bad = [t for t in items if t not in SUPPORTED_TYPES]
        if bad:
            raise ValueError(
                f"unsupported type {bad[0]!r}; use one of: {', '.join(SUPPORTED_TYPES)}"
            )
        return value

    @field_validator("enum")
    @classmethod
    def _check_enum(cls, value: list[Any] | None) -> list[Any] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("enum must not be empty")
        if len(value) > _MAX_ENUM:
            raise ValueError(f"enum has more than {_MAX_ENUM} values")
        for item in value:
            if isinstance(item, (dict, list)):
                raise ValueError("enum values must be strings, numbers, booleans or null")
        return value


class OutputSchemaDeclaration(BaseModel):
    """What a project may supply. ``scoring_mode`` is accepted but ignored on
    save: the recipe owns it."""

    model_config = ConfigDict(extra="ignore")

    scoring_mode: str | None = None
    required: list[str] = Field(default_factory=list)
    properties: dict[str, OutputProperty] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "OutputSchemaDeclaration":
        if not self.properties:
            raise ValueError("properties must declare at least one field")
        if len(self.properties) > _MAX_PROPERTIES:
            raise ValueError(f"properties has more than {_MAX_PROPERTIES} fields")
        unknown = [name for name in self.required if name not in self.properties]
        if unknown:
            raise ValueError(f"required names a field that is not in properties: {unknown[0]!r}")
        return self


def validate_output_schema(raw: Any) -> dict[str, Any]:
    """Validate a user-supplied schema and return its stored form
    (``{"required": [...], "properties": {...}}``). Raises ``ValueError``
    with a message that is safe to show the user."""
    if not isinstance(raw, dict):
        raise ValueError("output schema must be a JSON object with a 'properties' map")
    try:
        decl = OutputSchemaDeclaration.model_validate(raw)
    except ValidationError as exc:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            msg = str(err.get("msg", "invalid")).removeprefix("Value error, ")
            parts.append(f"{loc}: {msg}" if loc else msg)
        raise ValueError("Invalid output schema — " + "; ".join(parts)) from exc
    return {
        "required": list(decl.required),
        "properties": {
            name: {
                **({"type": prop.type} if prop.type is not None else {}),
                **({"enum": list(prop.enum)} if prop.enum is not None else {}),
            }
            for name, prop in decl.properties.items()
        },
    }


def merge_manifest_output_schema(
    scoring_mode: str,
    required_default: list[str],
    declared: Any,
) -> dict[str, Any]:
    """Recipe scoring mode + the project's declared properties/required."""
    merged: dict[str, Any] = {"scoring_mode": scoring_mode, "required": list(required_default)}
    if isinstance(declared, dict) and isinstance(declared.get("properties"), dict) and declared["properties"]:
        merged["properties"] = declared["properties"]
        if isinstance(declared.get("required"), list) and declared["required"]:
            merged["required"] = list(declared["required"])
    return merged


def normalize_property(raw: Any) -> dict[str, Any]:
    """Handler-side tolerant normalisation of one manifest property:
    ``{"types": [...], "enum": [...] | None}``."""
    types: list[str] = []
    enum: list[Any] | None = None
    if isinstance(raw, dict):
        t = raw.get("type")
        if isinstance(t, str):
            types = [t]
        elif isinstance(t, list):
            types = [x for x in t if isinstance(x, str)]
        e = raw.get("enum")
        if isinstance(e, list) and e:
            enum = list(e)
    return {"types": types, "enum": enum}


# ── Typed scalar comparison ──────────────────────────────────────────

_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")


def is_null(value: Any) -> bool:
    if value is MISSING or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("null", "")
    return False


def to_number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        try:
            number = Decimal(repr(value))
        except InvalidOperation:
            return None
        return number if number.is_finite() else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _THOUSANDS.match(text):
            text = text.replace(",", "")
        try:
            number = Decimal(text)
        except InvalidOperation:
            return None
        return number if number.is_finite() else None
    return None


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().casefold()
        if text == "true":
            return True
        if text == "false":
            return False
    return None


def normalize_string(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.casefold().split())
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text


def is_container(value: Any) -> bool:
    return isinstance(value, (dict, list))


def text_of(value: Any) -> str:
    """String form fed to token-F1 (containers stringified stably)."""
    if is_container(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return "" if value is None or value is MISSING else str(value)


def scalar_match(pred: Any, gold: Any, types: list[str] | None = None) -> bool:
    """Typed exact match for two non-container values (missing counts as NULL)."""
    types = types or []
    p_null, g_null = is_null(pred), is_null(gold)
    if p_null or g_null:
        return p_null and g_null
    if "boolean" in types or isinstance(pred, bool) or isinstance(gold, bool):
        pb, gb = to_bool(pred), to_bool(gold)
        if pb is not None and gb is not None:
            return pb == gb
        if "boolean" in types:
            return False
    pn, gn = to_number(pred), to_number(gold)
    if set(types) & _SCALAR_NUMERIC:
        return pn is not None and gn is not None and pn == gn
    if pn is not None and gn is not None:
        return pn == gn
    return normalize_string(pred) == normalize_string(gold)


def value_match(pred: Any, gold: Any, types: list[str] | None = None) -> bool:
    """Typed exact match. Containers keep the pre-typed behaviour (SQuAD
    normalisation over a stable ``json.dumps(sort_keys=True)``)."""
    if is_container(pred) or is_container(gold):
        if is_null(pred) or is_null(gold):
            return is_null(pred) and is_null(gold)
        from app.services.evaluation_service import exact_match

        return bool(exact_match(text_of(pred), text_of(gold)))
    return scalar_match(pred, gold, types)


def type_ok(value: Any, types: list[str]) -> bool:
    """Does a non-NULL value satisfy at least one declared type?"""
    if not types:
        return True
    for t in types:
        if t == "string" and isinstance(value, str):
            return True
        if t == "boolean" and to_bool(value) is not None:
            return True
        if t == "number" and to_number(value) is not None:
            return True
        if t == "integer":
            number = to_number(value)
            if number is not None and number == number.to_integral_value():
                return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
    return False


def enum_contains(value: Any, enum: list[Any], types: list[str] | None = None) -> bool:
    """Non-NULL ``value`` among the enum's non-null members (typed compare)."""
    return any(not is_null(e) and scalar_match(value, e, types) for e in enum)
