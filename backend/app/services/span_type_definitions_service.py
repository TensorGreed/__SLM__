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

import hashlib
import json
import re
from collections import Counter
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

DEFINITIONS_KEY = "span_type_definitions"
# Project flag: the texts come from scans (OCR). Both labellers get
# SCAN_TOLERANT_CLAUSE, and evaluation / redaction tidy predicted spans
# (``train_token_classification.ocr_tidy_spans``).
SCAN_TOLERANT_KEY = "span_scan_tolerant"
SCAN_TOLERANT_CLAUSE = (
    "The texts may come from scanned documents (OCR): line breaks became spaces and stray marks "
    "(‘ ~ _ « : |) can sit inside or next to a value. A value broken by spaces or stray marks is still "
    "ONE entity — label it from its first to its last character (e.g. \"2024-02- 10\", "
    "\"US- Z1Y2X3W4\", \"Sarah - Jenkins\"); never include a stray mark at its start or end, nor a "
    "label word in front of it (\"Account number:\", \"SSN:\", \"MRN:\", \"Type\"). A prefix that is part "
    "of the identifier itself — joined to it by a hyphen or with no space (\"ACCT-3637328949\", "
    "\"EMP-004521\", \"PT-30021\") — stays in the entity.\n"
)
MAX_TYPES = 200
MAX_DEFINITION_CHARS = 400

# Starting wording for common type names — a suggestion the user edits,
# never applied silently. Each line says what the type is AND what it is
# not, because the disagreements are at the boundaries — including the
# span's own boundaries: on project 31 every remaining person_name
# disagreement was a title included by one labeller and not the other.
_SUGGESTIONS: dict[str, str] = {
    "date_of_birth": "a person's birth date, only when the text says or clearly implies it is the birth date (DOB, born on, date of birth); any other date is not a date_of_birth",
    "visit_date": "the date a patient was or will be seen for care — an appointment, visit, date of service, admission, procedure or discharge; not a birth date, and not an order, purchase, transaction, ticket, shipping, billing, due, print or 'document generated' date (those are not labelled)",
    "bank_account": "a domestic bank account number — the account itself, each number its own span, never the word 'account' or the label; a routing number, ABA, SWIFT / BIC or sort code identifies the bank, not the account, and is not labelled; an IBAN is iban, a card's number or last four digits are credit_card",
    "iban": "an International Bank Account Number (two letters, two check digits, then the account, often grouped in fours)",
    "person_name": "a private individual's name, including in greetings and signatures — the name only, without a title (Mr., Ms., Dr.); a clinician or other care provider named in their professional role is provider_name",
    "provider_name": "a clinician, care provider or healthcare organisation named in that role — a person's name without a title, like person_name (\"Chen\" in \"Dr. Chen\", \"Patel\" in \"Nurse Patel\"), or the organisation's name (St. Mary's Clinic); the patient or customer is person_name; a shop, bank, insurer's or other company's name is not a provider_name (not labelled)",
    "national_id": "a government-issued personal identifier (SSN, National Insurance number, national ID card number); not a passport number, not a tax or employee id",
    "passport": "a passport number; not other government ids",
    "credit_card": "a payment card number, full or masked with visible digits, including a card's last four digits when the text says so ('card ending in 4242'); not the expiry date or the CVV",
    "card_expiry": "a payment card's expiry date (MM/YY or MM/YYYY next to a card)",
    "cvv": "a card security code: the 3–4 digits only, never the word CVV / CVC / security code itself",
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


def project_scan_tolerant(project: Any) -> bool:
    cfg = getattr(project, "runtime_config", None)
    return bool(isinstance(cfg, dict) and cfg.get(SCAN_TOLERANT_KEY))


async def load_scan_tolerant(db: AsyncSession | None, project_id: int) -> bool:
    if db is None:
        return False
    from app.models.project import Project

    project = await db.get(Project, project_id)
    return project_scan_tolerant(project) if project is not None else False


# ── Conventions version ─────────────────────────────────────────────────
#
# Every labeller stamps the rows it writes with CONVENTIONS_VERSION_KEY =
# a short hash of the definitions + the scan flag it labelled under. A
# convention only reaches the rows labelled after it changed, so the review
# queue compares each group's stamps with the project's current version
# (``conventions_status``) and offers to relabel stale groups.

CONVENTIONS_VERSION_KEY = "conventions_version"


def conventions_version(definitions: dict[str, str] | None, scan_tolerant: bool = False) -> str | None:
    """Short stable hash of what the labellers are told; None when the
    project has no conventions at all (nothing can be stale)."""
    defs = normalise_definitions(definitions)
    if not defs and not scan_tolerant:
        return None
    # The clause's own wording is part of what the labellers are told, so a
    # change to it makes rows labelled under the old wording stale too.
    payload = json.dumps(
        {"definitions": defs, "scan_tolerant": bool(scan_tolerant), "scan_clause": SCAN_TOLERANT_CLAUSE if scan_tolerant else None},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def project_conventions_version(project: Any) -> str | None:
    return conventions_version(project_definitions(project), project_scan_tolerant(project))


async def load_conventions_version(db: AsyncSession | None, project_id: int) -> str | None:
    if db is None:
        return None
    from app.models.project import Project

    project = await db.get(Project, project_id)
    return project_conventions_version(project) if project is not None else None


_LABELLER_SOURCE_PREFIXES = ("passage_labels:", "document_labels:")


def row_labeller(row: dict[str, Any]) -> str | None:
    """Who last labelled a span row: ``relabelled_by`` / ``labelled_by``,
    else the backend in a ``passage_labels:<backend>`` source."""
    for key in ("relabelled_by", "labelled_by"):
        if isinstance(row.get(key), str) and row[key]:
            return row[key]
    source = row.get("source")
    if isinstance(source, str):
        for prefix in _LABELLER_SOURCE_PREFIXES:
            if source.startswith(prefix) and len(source) > len(prefix):
                return source[len(prefix):]
    return None


def relabel_pin(labeller: str | None) -> str | None:
    """The backend pin the relabel endpoint takes for a labeller label:
    a cloud pin as it is, any Ollama model → ``ollama``; else None."""
    if not labeller:
        return None
    if labeller.startswith("cloud:") and labeller.count(":") >= 2:
        return labeller
    if labeller == "ollama" or labeller.startswith("ollama:"):
        return "ollama"
    return None


def conventions_status(rows: list[dict[str, Any]], current: str | None) -> dict[str, Any] | None:
    """For one group of rows: how many span rows were labelled under the
    current conventions, under an older version (``stale_rows``) or before
    versions were stamped (``untracked_rows``), and the pin of the most
    common labeller. None when there is nothing to compare (no
    conventions, no span rows)."""
    if current is None:
        return None
    span_rows = [r for r in rows if isinstance(r, dict) and isinstance(r.get("entities"), list)]
    if not span_rows:
        return None
    stamps = [r.get(CONVENTIONS_VERSION_KEY) for r in span_rows]
    current_rows = sum(1 for s in stamps if s == current)
    untracked = sum(1 for s in stamps if not s)
    labellers = Counter(l for l in (row_labeller(r) for r in span_rows) if l)
    labelled_by = labellers.most_common(1)[0][0] if labellers else None
    return {
        "current": current,
        "span_rows": len(span_rows),
        "current_rows": current_rows,
        "stale_rows": len(span_rows) - current_rows - untracked,
        "untracked_rows": untracked,
        "is_stale": current_rows < len(span_rows),
        "labelled_by": labelled_by,
        "relabel_pin": relabel_pin(labelled_by),
    }


def definitions_clause(
    types: list[str] | None, definitions: dict[str, str], *, scan_tolerant: bool = False,
) -> str:
    """The prompt block: one line per labelled type that has a definition,
    in the order of ``types`` (all definitions when ``types`` is empty),
    plus ``SCAN_TOLERANT_CLAUSE`` when the project's texts come from scans.
    Empty string when there is nothing to say."""
    block = ""
    if definitions:
        names = [_normalise_type(t) for t in (types or [])] or list(definitions)
        lines = [f"- {name}: {definitions[name]}" for name in names if name in definitions]
        if lines:
            block = "Labelling conventions (follow these exactly; where two types could apply, these decide):\n" + "\n".join(lines) + "\n"
    if scan_tolerant:
        block += SCAN_TOLERANT_CLAUSE
    return block


def suggest_definitions(types: list[str]) -> dict[str, str]:
    """Starting wording for the given type names (only the ones a
    suggestion exists for)."""
    out: dict[str, str] = {}
    for t in types:
        name = _normalise_type(t)
        if name in _SUGGESTIONS:
            out[name] = _SUGGESTIONS[name]
    return out
