"""Seed documents: write sample documents for a domain when the user has
no corpus yet.

A span-extraction (or any documents-first) project needs text to work
from. When the user has none — "detect sensitive data in support tickets
and clinical notes", but no tickets or notes they can share — this flow
asks a generation model (local Ollama first, a cloud model only when
pinned) to write realistic sample documents in the genres the user
names, with the content they must contain, and ingests them as ONE text
document (``seed-<n>.txt``) that then goes through the normal pipeline:
parse → clean → passages → the span synthesiser rewrites and labels
them. Everything downstream is the same as for uploaded documents.

The texts are invented: every person, number and identifier is made up
but realistic in format. The model is told so, and the ingested document
is marked ``source="seed_documents"`` so provenance stays visible.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.synth_backends.base import SynthBackendError

_LOG = logging.getLogger(__name__)

# Genres a business corpus is made of. The user's brief picks or replaces
# them; "kinds" (what the documents must contain) is the user's.
DEFAULT_GENRES: tuple[str, ...] = (
    "customer support ticket",
    "order confirmation or billing email",
    "internal memo or HR email",
    "chat transcript between a customer and an agent",
    "incident report or log excerpt",
    "form letter or formal notice",
)
DEFAULT_LOCALES: tuple[str, ...] = ("US", "UK", "Germany", "Canada", "India", "France", "Australia")
STYLES: tuple[str, ...] = ("formal", "hurried", "friendly", "terse", "legalistic", "chatty")
MIN_COUNT = 1
MAX_COUNT = 400
MIN_WORDS = 60
MAX_WORDS = 400
# Shorter than this and the model answered with a refusal or a stub.
MIN_ACCEPTED_CHARS = 200

_SYSTEM_PROMPT = (
    "You write realistic sample documents for software testing. Everything in them is invented: "
    "people, companies, addresses, numbers and identifiers must be made up but realistic in format. "
    "Never reuse a real person's details. Output only the document text — no title, no markdown, "
    "no commentary."
)

ProgressFn = Callable[[float, str], Awaitable[None]]


def build_document_prompt(
    genre: str,
    kinds: str,
    *,
    words: int,
    locale: str,
    style: str,
    domain_hint: str = "",
) -> str:
    hint = f" The organisation: {domain_hint.strip()}." if domain_hint.strip() else ""
    must = f" It must naturally contain: {kinds.strip()}." if kinds.strip() else ""
    return (
        f"Write one realistic {genre.strip()} ({locale} conventions), about {words} words, as plain prose "
        f"or a short message — no markdown, no headings, no bullet lists.{hint}{must} "
        f"Style: {style}. All details are invented but realistic in format "
        "(e.g. a card number that looks real, a plausible record number). Vary the surface forms: "
        "numbers with spaces, dashes or none; identifiers in the locale's own formats; a person's "
        "full name inside a greeting or signature as well as in the body. Output only the text."
    )


# ── Page furniture (layout="paged") ──────────────────────────────────────
#
# Real documents arrive with page headers and footers — "Page 3 of 11",
# a letterhead, a confidentiality line, a print date. A model trained only
# on clean prose tags them (on project 31, "Page n of 11" lines between
# texts dropped precision from 0.79 to 0.65). The paged layout splits each
# document into pages and inserts invented furniture lines by code (not by
# the model, so they are always there and always varied). The bracketed
# "[3/11]" footer is deliberately absent: it is the held-out style the
# re-measurement uses.

LAYOUTS = ("plain", "paged")
DEFAULT_WORDS_PER_PAGE = 110
_ORGS = (
    "Northwind Health Partners", "Bluebell Mutual", "Harbor & Finch LLP", "Crestline Payments",
    "Meridian Clinic Group", "Atlas Freight Services", "Juniper Benefits Ltd", "Orchard Valley Bank",
)
_HEADERS = (
    "{org}", "{org} — {genre}", "CONFIDENTIAL", "{org} | Internal use only", "{genre} · {org}",
    "Strictly private and confidential", "{ORG}",
)
_FOOTERS = (
    "Page {i} of {n}", "- {i} -", "{i} / {n}", "Page {i}", "p. {i}",
    "{org} · Confidential · Page {i} of {n}", "Printed {date} · Page {i} of {n}",
    "Continued on next page", "Document generated {date}",
)


def _furniture_date(rng: random.Random) -> str:
    y, m, d = rng.randint(2019, 2026), rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice((f"{y}-{m:02d}-{d:02d}", f"{d:02d}/{m:02d}/{y}", f"{d} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]} {y}"))


def paginate_document(
    text: str, rng: random.Random, *, genre: str = "document", words_per_page: int = DEFAULT_WORDS_PER_PAGE,
) -> str:
    """Split ``text`` into pages of about ``words_per_page`` words (at a
    sentence end where possible) and put a header line above and a footer
    line below each page, each drawn from varied templates. Returns the
    text with the furniture on its own lines."""
    words = text.split()
    if not words:
        return text
    pages: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(current) >= words_per_page and word.endswith((".", "!", "?")):
            pages.append(" ".join(current))
            current = []
        elif len(current) >= int(words_per_page * 1.5):
            pages.append(" ".join(current))
            current = []
    if current:
        pages.append(" ".join(current))
    n = len(pages)
    org = rng.choice(_ORGS)
    header_tpl = rng.choice(_HEADERS)
    footer_tpl = rng.choice(_FOOTERS)
    date = _furniture_date(rng)
    title = genre.strip().split(" or ")[0][:60].strip().capitalize() or "Document"
    out: list[str] = []
    for i, page in enumerate(pages, start=1):
        fields = {"org": org, "ORG": org.upper(), "genre": title, "i": i, "n": n, "date": date}
        if i > 1 or rng.random() < 0.7:
            out.append(header_tpl.format(**fields))
        out.append(page)
        out.append(footer_tpl.format(**fields))
    return "\n".join(out)


def normalise_document(text: str) -> str:
    """One paragraph per document: collapse whitespace, strip fences."""
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", (text or "").strip(), flags=re.IGNORECASE)
    return " ".join(text.split())


@dataclass
class SeedDocumentsSummary:
    project_id: int
    requested: int
    written: int
    failed: int
    backend: str
    filename: str | None = None
    document_id: int | None = None
    passages: int | None = None
    genres: list[str] = field(default_factory=list)
    layout: str = "plain"
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "requested": self.requested,
            "written": self.written,
            "failed": self.failed,
            "backend": self.backend,
            "filename": self.filename,
            "document_id": self.document_id,
            "passages": self.passages,
            "genres": list(self.genres),
            "layout": self.layout,
            "warnings": list(self.warnings),
        }


async def run_seed_documents(
    db: AsyncSession,
    project_id: int,
    *,
    kinds: str = "",
    genres: list[str] | None = None,
    count: int = 60,
    words: int = 160,
    backend: str | None = None,
    domain_hint: str = "",
    chunk_size: int = 900,
    chunk_overlap: int = 60,
    layout: str = "plain",
    words_per_page: int = DEFAULT_WORDS_PER_PAGE,
    seed: int = 7,
    progress: ProgressFn | None = None,
    backend_override: Any = None,
) -> SeedDocumentsSummary:
    """Write ``count`` sample documents spread over ``genres`` (default
    ``DEFAULT_GENRES``), each containing ``kinds``, ingest them as one
    text file, parse and clean it into passages. ``layout="paged"`` splits
    each document into pages with header / footer lines
    (``paginate_document``). Raises when nothing was written (a silent
    "Done" with no documents misleads)."""
    from app.services.cleaning_service import clean_document
    from app.services.ingestion_service import ingest_file, process_document
    from app.services.synth_backends.base import pick_backend

    count = max(MIN_COUNT, min(MAX_COUNT, int(count)))
    words = max(MIN_WORDS, min(MAX_WORDS, int(words)))
    chosen_genres = [g.strip() for g in (genres or []) if isinstance(g, str) and g.strip()] or list(DEFAULT_GENRES)
    layout = layout if layout in LAYOUTS else "plain"

    async def report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    llm = backend_override or pick_backend(backend)
    backend_label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "backend")
    rng = random.Random(seed)
    texts: list[str] = []
    failed = 0
    for index in range(count):
        genre = chosen_genres[index % len(chosen_genres)]
        await report(0.02 + 0.85 * index / count, f"Writing document {index + 1}/{count}: {genre} ({backend_label})")
        prompt = build_document_prompt(
            genre, kinds, words=words, locale=rng.choice(DEFAULT_LOCALES), style=rng.choice(STYLES), domain_hint=domain_hint,
        )
        try:
            raw = await llm.complete(prompt, system_prompt=_SYSTEM_PROMPT, max_tokens=words * 3 + 200, temperature=0.9)
        except SynthBackendError as exc:
            failed += 1
            _LOG.warning("seed_documents %s/%s failed: %s", index + 1, count, exc)
            continue
        text = normalise_document(str(raw or ""))
        if len(text) < MIN_ACCEPTED_CHARS:
            failed += 1
            continue
        if layout == "paged":
            text = paginate_document(text, rng, genre=genre, words_per_page=words_per_page)
        texts.append(text)

    summary = SeedDocumentsSummary(
        project_id=project_id, requested=count, written=len(texts), failed=failed, backend=backend_label,
        genres=chosen_genres, layout=layout,
    )
    if not texts:
        raise RuntimeError(
            f"The generation model ({backend_label}) wrote no usable documents in {count} attempts — "
            "pick another model under Synthetic → backends, or check that it is reachable."
        )
    if failed:
        summary.warnings.append(f"{failed} of {count} documents were empty or failed and were skipped.")

    await report(0.9, f"Ingesting {len(texts)} documents")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"seed-documents-{stamp}.txt"
    doc = await ingest_file(
        db, project_id, filename, ("\n\n".join(texts) + "\n").encode("utf-8"), source="seed_documents",
    )
    await process_document(db, project_id, doc.id)
    await report(0.95, "Cleaning into passages")
    # The documents are invented by construction, so cleaning must NOT
    # redact the "PII" in them — a span project needs it in the text, and a
    # redacted seed corpus taught the span synthesiser to write
    # "[REDACTED_EMAIL]".
    cleaned = await clean_document(
        db, project_id, doc.id, chunk_size=chunk_size, chunk_overlap=chunk_overlap, redact=False,
    )
    summary.filename = filename
    summary.document_id = int(doc.id)
    chunk_count = cleaned.get("chunk_count") if isinstance(cleaned, dict) else None
    summary.passages = int(chunk_count) if isinstance(chunk_count, int) else None
    await report(1.0, f"{len(texts)} documents written and cleaned")
    return summary
