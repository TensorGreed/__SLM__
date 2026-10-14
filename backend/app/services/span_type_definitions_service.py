"""Entity-type definitions: one labelling convention for every model that
labels the project's spans.

Two labellers can both be "right" and still disagree: is a date in a
clinical note a date of birth or a visit date, is a GB29 NWBK… string a
bank account or an IBAN, is "Dr. Chen" a person name or a provider name?
Without a written convention each model picks its own, and the gap shows
up as model error when one model's labels train and another's score. A
project keeps its convention as ``runtime_config["span_type_definitions"]``
(``{type: one-line definition}``); the span generator (training labels)
and the answer-key relabeller both put the definitions for the types they
label into their prompt, so both follow the same rules.

General to any span project: the definitions are the project's own text;
``suggest_definitions`` only offers starting wording for common type
names, which the user edits.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

DEFINITIONS_KEY = "span_type_definitions"
MAX_TYPES = 200
MAX_DEFINITION_CHARS = 400

# Starting wording for common type names — a suggestion the user edits,
# never applied silently. Each line says what the type is AND what it is
# not, because the disagreements are at the boundaries.
_SUGGESTIONS: dict[str, str] = {
    "date_of_birth": "a person's birth date, only when the text says or clearly implies it is the birth date (DOB, born on, date of birth); any other date is not a date_of_birth",
    "visit_date": "the date of an appointment, visit, admission, procedure or discharge; not a birth date, not a billing or due date",
    "bank_account": "a domestic bank account or routing/sort-code + account number; an IBAN is iban, not bank_account",
    "iban": "an International Bank Account Number (two letters, two check digits, then the account, often grouped in fours)",
    "person_name": "a private individual's name, including in greetings and signatures; a clinician or other care provider named in their professional role is provider_name",
    "provider_name": "a clinician, care provider or healthcare organisation named in that role (Dr. Chen, Nurse Patel, St. Mary's Clinic); the patient or customer is person_name",
    "national_id": "a government-issued personal identifier (SSN, National Insurance number, national ID card number); not a passport number, not a tax or employee id",
    "passport": "a passport number; not other government ids",
    "credit_card": "a payment card number (full or masked with visible digits); not the expiry date or the CVV",
    "card_expiry": "a payment card's expiry date (MM/YY or MM/YYYY next to a card)",
    "cvv": "a card security code (3–4 digits labelled CVV/CVC/security code)",
    "phone": "a telephone number in any format",
    "email": "an email address",
    "address": "a postal or street address (a street line, optionally with city/postcode); a city or country alone is not an address",
    "medical_record_number": "a patient's medical record number (MRN) or hospital number",
    "health_plan_id": "a health insurance member, policy or group number",
    "diagnosis": "a named medical condition or diagnosis code; not a symptom description",
    "medication": "a named drug or medication (with or without dose); not a drug class",
    "employee_id": "an organisation's employee or staff identifier",
    "api_key": "a secret API key or access token",
    "ticket_id": "an internal ticket, case or incident reference",
    "ip_address": "an IPv4 or IPv6 address",
    "license_plate": "a vehicle registration plate",
}


def _normalise_type(name: Any) -> str:
    return re.sub(r"\s+", "_", str(name or "").strip().lower())


def normalise_definitions(raw: Any) -> dict[str, str]:
    """``{type: definition}`` with types lower-cased / underscored, empty
    entries dropped, lengths capped."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in list(raw.items())[:MAX_TYPES]:
        name = _normalise_type(key)
        text = " ".join(str(value or "").split())[:MAX_DEFINITION_CHARS]
        if name and text:
            out[name] = text
    return out


def project_definitions(project: Any) -> dict[str, str]:
    cfg = getattr(project, "runtime_config", None)
    return normalise_definitions(cfg.get(DEFINITIONS_KEY) if isinstance(cfg, dict) else None)


async def load_definitions(db: AsyncSession | None, project_id: int) -> dict[str, str]:
    """The project's definitions; ``{}`` without a session or project."""
    if db is None:
        return {}
    from app.models.project import Project

    project = await db.get(Project, project_id)
    return project_definitions(project) if project is not None else {}


def definitions_clause(types: list[str] | None, definitions: dict[str, str]) -> str:
    """The prompt block: one line per labelled type that has a definition,
    in the order of ``types`` (all definitions when ``types`` is empty).
    Empty string when there is nothing to say."""
    if not definitions:
        return ""
    names = [_normalise_type(t) for t in (types or [])] or list(definitions)
    lines = [f"- {name}: {definitions[name]}" for name in names if name in definitions]
    if not lines:
        return ""
    return "Labelling conventions (follow these exactly; where two types could apply, these decide):\n" + "\n".join(lines) + "\n"


def suggest_definitions(types: list[str]) -> dict[str, str]:
    """Starting wording for the given type names (only the ones a
    suggestion exists for)."""
    out: dict[str, str] = {}
    for t in types:
        name = _normalise_type(t)
        if name in _SUGGESTIONS:
            out[name] = _SUGGESTIONS[name]
    return out
