"""Serving gateway core: keys, model ids, retrieval injection, guided JSON,
upstream calls, named tools and settings (phase G11).

``/v1/chat/completions`` answers with a project's trained run (in-process via
``local_chat_service``) or forwards to an OpenAI-compatible server (vLLM,
Ollama, ...). Retrieval is injected exactly as passage-prompt training does it
(``document_preamble`` as the system message) and sources come back as
``citations``. Prompt and reply text are never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from jsonschema import Draft202012Validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import RawDocument
from app.models.experiment import Experiment
from app.models.gateway import GatewayApiKey
from app.models.project import Project
from app.services import gateway_tools_service as gts
from app.services import local_chat_service
from app.services import servable_run_service as srs

KEY_PREFIX = "bsk_"
UPSTREAMS = ("in_process", "openai_compatible")
UPSTREAM_KINDS = ("vllm", "ollama", "openai")
DEFAULT_MAX_TOKENS = 512
UPSTREAM_TIMEOUT_S = 300.0
# Tests inject an ``httpx.MockTransport`` here.
_HTTP_TRANSPORT: httpx.AsyncBaseTransport | None = None


# ---------------------------------------------------------------- errors


class GatewayError(Exception):
    """An OpenAI-envelope error: ``{"error": {"message", "type", "code"}}``."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str = "invalid_request",
        type_: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.type = type_ or ("invalid_request_error" if status_code < 500 else "server_error")
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "code": self.code}}


# ---------------------------------------------------------------- keys


@dataclass(frozen=True)
class KeyInfo:
    id: int
    prefix: str
    project_ids: tuple[int, ...]
    rate_limit_per_minute: int


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def serialize_key(row: GatewayApiKey) -> dict[str, Any]:
    def _iso(v: datetime | None) -> str | None:
        return v.isoformat() if v else None

    return {
        "id": row.id,
        "name": row.name,
        "key_prefix": row.key_prefix,
        "project_ids": [int(p) for p in (row.project_ids or [])],
        "rate_limit_per_minute": row.rate_limit_per_minute,
        "is_active": bool(row.is_active),
        "created_at": _iso(row.created_at),
        "last_used_at": _iso(row.last_used_at),
        "revoked_at": _iso(row.revoked_at),
    }


async def create_key(
    db: AsyncSession,
    *,
    name: str,
    project_ids: list[int],
    rate_limit_per_minute: int | None = None,
    created_by: str | None = None,
) -> tuple[str, GatewayApiKey]:
    name = str(name or "").strip()
    if not name:
        raise ValueError("name is required")
    ids = sorted({int(p) for p in project_ids or [] if not isinstance(p, bool)})
    if not ids:
        raise ValueError("project_ids must list at least one project")
    rate = 60 if rate_limit_per_minute is None else int(rate_limit_per_minute)
    if not 1 <= rate <= 100_000:
        raise ValueError("rate_limit_per_minute must be between 1 and 100000")
    found = set((await db.execute(select(Project.id).where(Project.id.in_(ids)))).scalars().all())
    missing = [p for p in ids if p not in found]
    if missing:
        raise ValueError(f"project(s) not found: {missing}")
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    row = GatewayApiKey(
        name=name[:128],
        key_hash=hash_key(raw),
        key_prefix=raw[:12],
        project_ids=ids,
        rate_limit_per_minute=rate,
        is_active=True,
        created_by=created_by,
    )
    db.add(row)
    await db.flush()
    return raw, row


async def authenticate(db: AsyncSession, raw: str | None) -> KeyInfo:
    if not raw:
        raise GatewayError(
            401, "Missing API key. Send 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
            code="invalid_api_key", type_="authentication_error",
        )
    row = (
        await db.execute(select(GatewayApiKey).where(GatewayApiKey.key_hash == hash_key(raw.strip())))
    ).scalar_one_or_none()
    if row is None or not row.is_active or row.revoked_at is not None:
        raise GatewayError(
            401, "Invalid or revoked API key.", code="invalid_api_key", type_="authentication_error"
        )
    now = datetime.now(timezone.utc)
    last = row.last_used_at
    if last is not None and last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    if last is None or (now - last).total_seconds() > 30:
        row.last_used_at = now
        await db.commit()
    return KeyInfo(
        id=row.id,
        prefix=row.key_prefix,
        project_ids=tuple(int(p) for p in (row.project_ids or [])),
        rate_limit_per_minute=int(row.rate_limit_per_minute or 60),
    )


async def purge_project(db: AsyncSession, project_id: int) -> None:
    """Project deleted: drop its id from every key's scope (revoking keys left
    with no project) and delete its request-log rows."""
    from sqlalchemy import delete as sa_delete

    from app.models.gateway import GatewayRequestLog

    now = datetime.now(timezone.utc)
    rows = (await db.execute(select(GatewayApiKey))).scalars().all()
    for row in rows:
        ids = [int(p) for p in (row.project_ids or [])]
        if project_id not in ids:
            continue
        remaining = [p for p in ids if p != project_id]
        row.project_ids = remaining
        if not remaining and row.revoked_at is None:
            row.is_active = False
            row.revoked_at = now
    await db.execute(sa_delete(GatewayRequestLog).where(GatewayRequestLog.project_id == project_id))
    await db.flush()


_RATE_LOCK = threading.Lock()
_RATE_WINDOWS: dict[int, deque[float]] = {}


def check_rate_limit(key: KeyInfo, *, now: float | None = None) -> None:
    """Sliding 60 s window per key id; raises a 429 with ``Retry-After``."""
    t = time.monotonic() if now is None else now
    with _RATE_LOCK:
        window = _RATE_WINDOWS.setdefault(key.id, deque())
        while window and t - window[0] >= 60.0:
            window.popleft()
        if len(window) >= key.rate_limit_per_minute:
            retry = max(1, int(60.0 - (t - window[0])) + 1)
            raise GatewayError(
                429,
                f"Rate limit exceeded ({key.rate_limit_per_minute} requests per minute for this key).",
                code="rate_limit_exceeded", type_="rate_limit_error",
                headers={"Retry-After": str(retry)},
            )
        window.append(t)


def reset_rate_limits() -> None:
    with _RATE_LOCK:
        _RATE_WINDOWS.clear()


# ---------------------------------------------------------------- slugs / models


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")[:48].strip("-")


async def all_slugs(db: AsyncSession) -> dict[int, str]:
    """``{project_id: slug}``: the slug of the name; the oldest project keeps it,
    a later project with the same (or an empty) slug gets ``-<id>`` appended
    (and a counter if even that is taken)."""
    rows = (await db.execute(select(Project.id, Project.name).order_by(Project.id.asc()))).all()
    taken: set[str] = set()
    out: dict[int, str] = {}
    for pid, name in rows:
        base = slugify(name)
        slug = base
        if not slug or slug in taken:
            slug = f"{base}-{pid}" if base else f"p{pid}"
            n = 2
            while slug in taken:
                slug = f"{base}-{pid}-{n}" if base else f"p{pid}-{n}"
                n += 1
        taken.add(slug)
        out[int(pid)] = slug
    return out


async def project_slug(db: AsyncSession, project: Project) -> str:
    return (await all_slugs(db)).get(project.id) or slugify(project.name) or f"p{project.id}"


def gateway_config(project: Project) -> dict[str, Any]:
    cfg = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    raw = cfg.get("gateway")
    return raw if isinstance(raw, dict) else {}


def guided_schema(project: Project) -> dict[str, Any] | None:
    cfg = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    hint = cfg.get("guided_json")
    schema = hint.get("schema") if isinstance(hint, dict) else None
    return schema if isinstance(schema, dict) else None


def _passage_trained(exp: Experiment | None) -> bool:
    from app.services import passage_prompts_service

    return exp is not None and passage_prompts_service.is_enabled(exp.config)


async def served_run(db: AsyncSession, project: Project, runs: list[Experiment] | None = None) -> Experiment | None:
    """The run ``<slug>`` resolves to: ``default_run_id`` when set (and servable),
    else the latest servable run."""
    runs = runs if runs is not None else await srs.list_servable_runs(db, project.id)
    default_id = gateway_config(project).get("default_run_id")
    if default_id is not None:
        for exp in runs:
            if exp.id == default_id:
                return exp
        raise GatewayError(
            400, f"default_run_id #{default_id} is not a servable run of this project.", code="no_servable_run"
        )
    return runs[-1] if runs else None


@dataclass
class ResolvedModel:
    requested: str
    canonical: str
    project: Project
    slug: str
    kind: str  # "served" | "run" | "base"
    experiment: Experiment | None
    base_model: str
    model_ref: str
    retrieval_default: bool
    upstream_name: str = ""
    config: dict[str, Any] = field(default_factory=dict)


def split_model_id(model_id: str) -> tuple[str, str]:
    head, _, tail = str(model_id or "").strip().partition(":")
    return head, tail


async def resolve_model(db: AsyncSession, key: KeyInfo, model_id: str) -> ResolvedModel:
    from app.services.rag_project_service import is_rag_first

    requested = str(model_id or "").strip()
    if not requested:
        raise GatewayError(400, "'model' is required.", code="invalid_request")
    head, tail = split_model_id(requested)
    slugs = await all_slugs(db)
    by_slug = {s: pid for pid, s in slugs.items()}
    pid = by_slug.get(head)
    if pid is None:
        alias = re.fullmatch(r"p(\d+)", head)
        if alias and int(alias.group(1)) in slugs:
            pid = int(alias.group(1))
    if pid is None:
        raise GatewayError(404, f"The model '{requested}' does not exist.", code="model_not_found")
    if pid not in key.project_ids:
        raise GatewayError(403, f"This API key has no access to the project behind '{requested}'.", code="project_not_in_scope")
    project = await db.get(Project, pid)
    if project is None:
        raise GatewayError(404, f"The model '{requested}' does not exist.", code="model_not_found")
    slug = slugs[pid]
    cfg = gateway_config(project)
    base = str(project.base_model_name or "").strip()
    exp: Experiment | None = None
    retrieval_default = False
    if tail == "base":
        if not base:
            raise GatewayError(400, "This project has no base model.", code="no_base_model")
        kind, canonical, model_ref, base_model = "base", f"{slug}:base", base, base
    elif tail.startswith("run-") and tail[4:].isdigit():
        exp = await db.get(Experiment, int(tail[4:]))
        if exp is None or exp.project_id != pid:
            raise GatewayError(404, f"Run #{tail[4:]} not found in this project.", code="model_not_found")
        problem = srs.run_problem(exp)
        if problem:
            raise GatewayError(400, problem, code="run_not_servable")
        kind, canonical = "run", f"{slug}:run-{exp.id}"
        model_ref, base_model = srs.run_model_ref(exp), exp.base_model
        retrieval_default = _passage_trained(exp)
    elif tail == "":
        exp = await served_run(db, project)
        if exp is not None:
            kind, canonical = "served", f"{slug}:run-{exp.id}"
            model_ref, base_model = srs.run_model_ref(exp), exp.base_model
            retrieval_default = _passage_trained(exp)
        elif is_rag_first(project) and base:
            kind, canonical, model_ref, base_model = "served", f"{slug}:base", base, base
            retrieval_default = True
        else:
            raise GatewayError(
                400, f"Project '{slug}' has no completed run with saved weights to serve.", code="no_servable_run"
            )
    else:
        raise GatewayError(404, f"The model '{requested}' does not exist.", code="model_not_found")
    umap = cfg.get("upstream_model_map") if isinstance(cfg.get("upstream_model_map"), dict) else {}
    upstream_name = str(umap.get(requested) or umap.get(canonical) or canonical)
    return ResolvedModel(
        requested=requested, canonical=canonical, project=project, slug=slug, kind=kind, experiment=exp,
        base_model=base_model, model_ref=model_ref, retrieval_default=retrieval_default,
        upstream_name=upstream_name, config=cfg,
    )


async def resolve_project(db: AsyncSession, key: KeyInfo, ref: str) -> tuple[Project, str]:
    ref = str(ref or "").strip()
    slugs = await all_slugs(db)
    by_slug = {s: pid for pid, s in slugs.items()}
    pid = by_slug.get(ref)
    if pid is None:
        alias = re.fullmatch(r"p(\d+)", ref)
        if alias and int(alias.group(1)) in slugs:
            pid = int(alias.group(1))
    if pid is None:
        raise GatewayError(404, f"The project '{ref}' does not exist.", code="project_not_found")
    if pid not in key.project_ids:
        raise GatewayError(403, f"This API key has no access to project '{ref}'.", code="project_not_in_scope")
    project = await db.get(Project, pid)
    if project is None:
        raise GatewayError(404, f"The project '{ref}' does not exist.", code="project_not_found")
    return project, slugs[pid]


async def list_models(db: AsyncSession, key: KeyInfo) -> list[dict[str, Any]]:
    from app.services.rag_project_service import is_rag_first

    slugs = await all_slugs(db)
    out: list[dict[str, Any]] = []
    for pid in key.project_ids:
        project = await db.get(Project, pid)
        if project is None:
            continue
        slug = slugs[pid]
        runs = await srs.list_servable_runs(db, pid)
        try:
            default = await served_run(db, project, runs)
        except GatewayError:
            default = None
        base = str(project.base_model_name or "")
        has_schema = guided_schema(project) is not None
        created = int(project.created_at.timestamp()) if getattr(project, "created_at", None) else int(time.time())

        def _entry(mid: str, exp: Experiment | None, retrieval: bool, ts: int) -> dict[str, Any]:
            return {
                "id": mid, "object": "model", "created": ts, "owned_by": "brewslm",
                "brewslm": {
                    "project_id": pid, "project_name": project.name, "run_id": exp.id if exp else None,
                    "base_model": exp.base_model if exp else base, "retrieval_default": retrieval,
                    "guided_json": has_schema,
                },
            }

        def _ts(exp: Experiment) -> int:
            return int(exp.created_at.timestamp()) if exp.created_at else created

        rag_first_default = default is None and is_rag_first(project) and bool(base)
        out.append(_entry(slug, default, _passage_trained(default) or rag_first_default, _ts(default) if default else created))
        if base:
            out.append(_entry(f"{slug}:base", None, False, created))
        for exp in runs:
            out.append(_entry(f"{slug}:run-{exp.id}", exp, _passage_trained(exp), _ts(exp)))
    return out


# ---------------------------------------------------------------- retrieval


def _clean_text(text: Any, limit: int) -> str:
    return str(text or "")[:limit]


async def _source_urls(db: AsyncSession, doc_ids: set[Any]) -> dict[Any, str | None]:
    ids = {int(d) for d in doc_ids if isinstance(d, (int, str)) and str(d).isdigit()}
    if not ids:
        return {}
    rows = (await db.execute(select(RawDocument).where(RawDocument.id.in_(ids)))).scalars().all()
    out: dict[Any, str | None] = {}
    for doc in rows:
        meta = doc.metadata_ if isinstance(doc.metadata_, dict) else {}
        out[doc.id] = doc.source_url or meta.get("source_url") or None
    return out


async def retrieve_passages(
    db: AsyncSession, project: Project, query: str, *, k: int | None = None
) -> dict[str, Any]:
    """Document-passage retrieval over the project's index with the project's
    retrieval setting. Never raises on a missing index:
    ``{applied, reason?, retrieval, hits, citations, preamble}``."""
    from app.services import auto_rag_service as ars
    base = ars.retrieval_settings(project)
    retrieval = ars.normalize_retrieval({**base, "k": k if k is not None else base["k"]})
    info = {
        "applied": False, "k": retrieval["k"], "reranker": retrieval["reranker"], "pool": retrieval["pool"],
        "chunk_size": retrieval["chunk_size"], "source": "request" if k is not None else ars.retrieval_source(project),
        "count": 0,
    }
    result: dict[str, Any] = {"applied": False, "retrieval": info, "hits": [], "citations": [], "preamble": None}
    if not query.strip():
        info["reason"] = "no_user_message_to_query"
        return result
    try:
        hits = await asyncio.to_thread(ars.retrieve_document_passages, project.id, query, retrieval)
    except ars.AutoRagUnavailable as exc:
        info["reason"] = f"no_index: {str(exc)[:200]}"
        return result
    except Exception as exc:  # noqa: BLE001 - retrieval is best-effort; never a 500
        info["reason"] = f"retrieval_failed: {type(exc).__name__}"
        return result
    if not hits:
        info["reason"] = "no_passages_matched"
        return result
    urls = await _source_urls(db, {(h.get("payload") or {}).get("source_document_id") for h in hits})
    citations = []
    for n, hit in enumerate(hits, start=1):
        payload = hit.get("payload") or {}
        doc_id = payload.get("source_document_id")
        citations.append(
            {
                "n": n,
                "passage_id": payload.get("id") if payload.get("id") is not None else hit.get("row_id"),
                "document": payload.get("source_doc"),
                "source_document_id": doc_id,
                "source_url": urls.get(int(doc_id)) if isinstance(doc_id, (int, str)) and str(doc_id).isdigit() else None,
                "snippet": _clean_text(payload.get("text"), 300),
                "score": hit.get("score"),
            }
        )
    info["applied"] = True
    info["count"] = len(hits)
    result.update(applied=True, hits=hits, citations=citations, preamble=ars.document_preamble(hits))
    return result


# ---------------------------------------------------------------- request handling


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
    return ""


def normalize_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise GatewayError(400, "'messages' must be a non-empty list.")
    out: list[dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            raise GatewayError(400, "Each message must be an object with 'role' and 'content'.")
        role = str(m.get("role") or "").strip().lower()
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            raise GatewayError(400, f"Unsupported message role '{role}'.")
        text = _text_of(m.get("content")).strip()
        if text:
            out.append({"role": role, "content": text})
    if not any(m["role"] == "user" for m in out):
        raise GatewayError(400, "At least one non-empty user message is required.")
    return out


def with_preamble(messages: list[dict[str, str]], preamble: str) -> list[dict[str, str]]:
    """ONE system message: the retrieval preamble first, then the client's system text."""
    systems = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    content = preamble + ("\n\n" + "\n\n".join(systems) if systems else "")
    return [{"role": "system", "content": content}, *rest]


def _validate_reply(schema: dict[str, Any], content: str, *, mode: str = "validated_only", via: str | None = None) -> dict[str, Any]:
    extra = {"via": via} if via else {}
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except ValueError as exc:
        return {"mode": mode, **extra, "valid": False, "errors": [f"reply is not valid JSON: {str(exc)[:120]}"]}
    errs = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: list(e.absolute_path))
    msgs = [(("/".join(str(p) for p in e.absolute_path) + ": ") if e.absolute_path else "") + e.message[:200] for e in errs[:5]]
    return {"mode": mode, **extra, "valid": not errs, "errors": msgs}


async def _call_upstream(
    cfg: dict[str, Any], model_name: str, messages: list[dict[str, str]], body: dict[str, Any],
    *, max_tokens: int, temperature: float, schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise GatewayError(502, "The project's upstream has no base_url configured.", code="upstream_not_configured")
    url = base_url + ("/chat/completions" if base_url.endswith("/v1") else "/v1/chat/completions")
    payload: dict[str, Any] = {
        "model": model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False,
    }
    for key in ("top_p", "stop", "seed", "presence_penalty", "frequency_penalty"):
        if body.get(key) is not None:
            payload[key] = body[key]
    guided: dict[str, Any] | None = None
    client_structured = body.get("response_format") is not None or body.get("guided_json") is not None
    if client_structured:
        for key in ("response_format", "guided_json"):
            if body.get(key) is not None:
                payload[key] = body[key]
        guided = {"mode": "client", "via": "guided_json" if body.get("guided_json") is not None else "response_format"}
    elif schema is not None:
        # One portable request shape for every upstream kind: Ollama's /v1 ignores
        # ``format`` and recent vLLM releases dropped ``guided_json``; both honour
        # ``response_format`` json_schema.
        payload["response_format"] = {
            "type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True},
        }
        guided = {"mode": "enforced", "via": "response_format"}
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S, transport=_HTTP_TRANSPORT) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise GatewayError(502, f"Upstream request failed: {type(exc).__name__}", code="upstream_error") from exc
    if resp.status_code >= 400:
        raise GatewayError(502, f"upstream returned HTTP {resp.status_code}", code="upstream_error")
    try:
        data = resp.json()
        choice = data["choices"][0]
        content = _text_of((choice.get("message") or {}).get("content"))
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise GatewayError(502, "Upstream returned a malformed chat completion.", code="upstream_error") from exc
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return (
        {"reply": content, "usage": usage, "finish_reason": choice.get("finish_reason") or "stop"},
        guided,
    )


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


async def execute_chat(
    db: AsyncSession, key: KeyInfo, body: dict[str, Any], *, resolved: ResolvedModel | None = None
) -> dict[str, Any]:
    """Run one chat completion; returns the full OpenAI-shaped response dict
    (plus ``citations`` / ``brewslm``) and logs a telemetry row."""
    from app.database import async_session_factory
    from app.services.served_model_telemetry_service import record_gateway_request

    started = time.perf_counter()
    request_id = "chatcmpl-" + uuid.uuid4().hex
    state: dict[str, Any] = {"status": 200, "code": None, "usage": {}, "retrieval": False, "guided": None}
    try:
        resolved = resolved or await resolve_model(db, key, body.get("model"))
        result = await _execute(db, resolved, body, request_id, state)
        return result
    except GatewayError as exc:
        state["status"], state["code"] = exc.status_code, exc.code
        raise
    except Exception:
        state["status"], state["code"] = 500, "internal_error"
        raise
    finally:
        if resolved is not None:
            try:
                async with async_session_factory() as log_db:
                    await record_gateway_request(
                        log_db, project_id=resolved.project.id, key_id=key.id, model_id=resolved.requested,
                        run_id=resolved.experiment.id if resolved.experiment else None,
                        latency_ms=(time.perf_counter() - started) * 1000.0, status_code=state["status"],
                        error_code=state["code"], input_tokens=_int_or_none(state["usage"].get("prompt_tokens")),
                        output_tokens=_int_or_none(state["usage"].get("completion_tokens")),
                        retrieval_applied=state["retrieval"], guided_json_mode=(state["guided"] or {}).get("mode"),
                        upstream=str(resolved.config.get("upstream") or "in_process"), request_id=request_id,
                    )
            except Exception:  # noqa: BLE001
                pass


async def _execute(
    db: AsyncSession, resolved: ResolvedModel, body: dict[str, Any], request_id: str, state: dict[str, Any]
) -> dict[str, Any]:
    ext = body.get("brewslm") if isinstance(body.get("brewslm"), dict) else {}
    messages = normalize_messages(body.get("messages"))
    if body.get("n") not in (None, 1):
        raise GatewayError(400, "Only n=1 is supported.")
    raw_max = body.get("max_tokens", body.get("max_completion_tokens"))
    max_tokens = max(1, min(4096, _int_or_none(raw_max) or DEFAULT_MAX_TOKENS))
    try:
        temperature = float(body["temperature"]) if body.get("temperature") is not None else 0.0
    except (TypeError, ValueError):
        raise GatewayError(400, "'temperature' must be a number.")
    project = resolved.project

    # retrieval
    want = ext.get("retrieval")
    do_retrieval = resolved.retrieval_default if want is None else bool(want)
    retrieval_info: dict[str, Any] = {"applied": False, "reason": "disabled"}
    citations: list[dict[str, Any]] = []
    if do_retrieval:
        k = _int_or_none(ext.get("k"))
        k = max(1, min(10, k)) if k is not None else None
        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        got = await retrieve_passages(db, project, query, k=k)
        retrieval_info = got["retrieval"]
        if got["applied"]:
            messages = with_preamble(messages, got["preamble"])
            citations = got["citations"]
    state["retrieval"] = bool(retrieval_info.get("applied"))

    # guided JSON
    schema = guided_schema(project) if ext.get("guided_json") is not False else None
    client_structured = body.get("response_format") is not None or body.get("guided_json") is not None
    in_process = str(resolved.config.get("upstream") or "in_process") != "openai_compatible"
    guided: dict[str, Any] | None = None

    if in_process:
        if client_structured:
            rf = body.get("response_format")
            cs = ((rf or {}).get("json_schema") or {}).get("schema") if isinstance(rf, dict) else None
            cs = cs if isinstance(cs, dict) else (body.get("guided_json") if isinstance(body.get("guided_json"), dict) else None)
            schema = cs
        try:
            out = await asyncio.to_thread(
                local_chat_service.generate_reply, resolved.model_ref, messages,
                max_tokens=max_tokens, temperature=temperature, base_model_hint=resolved.base_model,
            )
        except Exception as exc:  # noqa: BLE001
            raise GatewayError(500, f"Model runtime failed: {type(exc).__name__}: {str(exc)[:200]}", code="runtime_error") from exc
        if schema is not None:
            guided = _validate_reply(schema, str(out.get("reply") or ""))
    else:
        out, guided = await _call_upstream(
            resolved.config, resolved.upstream_name, messages, body,
            max_tokens=max_tokens, temperature=temperature, schema=schema,
        )
        if guided is not None and guided.get("mode") == "enforced" and schema is not None:
            guided = _validate_reply(schema, str(out.get("reply") or ""), mode="enforced", via=guided.get("via"))
    state["guided"] = guided
    content = str(out.get("reply") or "")
    usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
    pt, ct = _int_or_none(usage.get("prompt_tokens")), _int_or_none(usage.get("completion_tokens"))
    usage_out = {"prompt_tokens": pt or 0, "completion_tokens": ct or 0, "total_tokens": (pt or 0) + (ct or 0)}
    state["usage"] = usage_out
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resolved.requested,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": str(out.get("finish_reason") or "stop"),
            }
        ],
        "usage": usage_out,
        "citations": citations,
        "brewslm": {
            "project_id": project.id,
            "run_id": resolved.experiment.id if resolved.experiment else None,
            "upstream": "in_process" if in_process else "openai_compatible",
            "retrieval": retrieval_info,
            "guided_json": guided,
        },
    }


def sse_events(result: dict[str, Any]) -> list[str]:
    """The result as an SSE stream: one content chunk, a finish chunk, ``[DONE]``."""
    base = {"id": result["id"], "object": "chat.completion.chunk", "created": result["created"], "model": result["model"]}
    choice = result["choices"][0]
    first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": choice["message"]["content"]}, "finish_reason": None}]}
    last = {
        **base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
        "usage": result["usage"], "citations": result["citations"], "brewslm": result["brewslm"],
    }
    return [f"data: {json.dumps(first)}\n\n", f"data: {json.dumps(last)}\n\n", "data: [DONE]\n\n"]


# ---------------------------------------------------------------- tools


async def _tool_entries(db: AsyncSession, key: KeyInfo) -> list[dict[str, Any]]:
    """Internal tool list: each entry carries the raw tool name and project id so
    a call resolves by the exact exposed name (never by parsing a suffix)."""
    import logging

    entries: list[tuple[str, str, int, dict[str, Any]]] = []
    slugs = await all_slugs(db)
    for pid in key.project_ids:
        project = await db.get(Project, pid)
        if project is None:
            continue
        for tool in gateway_config(project).get("tools") or []:
            if isinstance(tool, dict) and tool.get("name"):
                entries.append((slugs[pid], str(tool["name"]), pid, tool))
    counts: dict[str, int] = {}
    for _, name, _, _ in entries:
        counts[name] = counts.get(name, 0) + 1
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slug, name, pid, tool in entries:
        exposed = name if counts[name] == 1 else f"{name}__{slug}"
        if exposed in seen:
            logging.getLogger(__name__).warning("gateway tool name collision on '%s'; skipping a duplicate", exposed)
            continue
        seen.add(exposed)
        out.append(
            {
                "name": exposed,
                "description": tool.get("description") or "",
                "input_schema": tool.get("input_schema") or {"type": "object"},
                "project": slug,
                "model": slug + str(tool.get("model") or ""),
                "_raw_name": name,
                "_project_id": pid,
            }
        )
    return out


async def list_tools(db: AsyncSession, key: KeyInfo) -> list[dict[str, Any]]:
    return [{k: v for k, v in e.items() if not k.startswith("_")} for e in await _tool_entries(db, key)]


async def _adapter_config(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """The project's dataset-adapter config (what prep reads), for recipe prompt builders."""
    from app.services.dataset_service import resolve_project_dataset_adapter_preference

    try:
        pref = await resolve_project_dataset_adapter_preference(db, project_id)
    except ValueError:
        return {}
    cfg = pref.get("adapter_config") if isinstance(pref, dict) else None
    return dict(cfg) if isinstance(cfg, dict) else {}


async def call_tool(db: AsyncSession, key: KeyInfo, name: str, arguments: Any) -> dict[str, Any]:
    entry = next((e for e in await _tool_entries(db, key) if e["name"] == name), None)
    if entry is None:
        raise GatewayError(404, f"Tool '{name}' not found.", code="tool_not_found")
    project, _slug = await resolve_project(db, key, entry["project"])
    tool = next((t for t in gateway_config(project).get("tools") or [] if t.get("name") == entry["_raw_name"]), None)
    if tool is None:
        raise GatewayError(404, f"Tool '{name}' not found.", code="tool_not_found")
    errors = gts.argument_errors(tool.get("input_schema") or {"type": "object"}, arguments)
    if errors:
        raise GatewayError(400, "Invalid arguments: " + "; ".join(errors), code="invalid_arguments")
    adapter_config = None
    if str(tool.get("prompt_template") or "").strip() == gts.RECIPE_TEMPLATE:
        adapter_config = await _adapter_config(db, project.id)
    try:
        prompt = await asyncio.to_thread(gts.build_tool_prompt, project, tool, arguments, adapter_config)
    except ValueError as exc:
        raise GatewayError(400, str(exc), code="invalid_arguments") from exc
    messages = []
    if tool.get("system_prompt"):
        messages.append({"role": "system", "content": tool["system_prompt"]})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {"model": entry["model"], "messages": messages}
    if tool.get("max_tokens"):
        body["max_tokens"] = tool["max_tokens"]
    if tool.get("retrieval") is not None:
        body["brewslm"] = {"retrieval": bool(tool["retrieval"])}
    result = await execute_chat(db, key, body)
    content = result["choices"][0]["message"]["content"]
    parsed = None
    try:
        parsed = json.loads(content)
    except ValueError:
        pass
    return {
        "content": content, "model": entry["model"], "citations": result["citations"],
        "brewslm": result["brewslm"], "parsed_json": parsed,
    }


# ---------------------------------------------------------------- settings


class GatewaySettingsError(ValueError):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        super().__init__("; ".join(e["message"] for e in errors) or "invalid gateway settings")
        self.errors = errors


def _defaults() -> dict[str, Any]:
    return {
        "upstream": "in_process", "base_url": None, "upstream_kind": None,
        "upstream_model_map": {}, "default_run_id": None, "tools": [],
    }


def read_settings(project: Project) -> dict[str, Any]:
    out = _defaults()
    cfg = gateway_config(project)
    for k in out:
        if k in cfg and cfg[k] is not None:
            out[k] = cfg[k]
    return out


async def update_settings(db: AsyncSession, project: Project, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge ``patch`` (partial is fine) into the stored settings; raises
    ``GatewaySettingsError`` with per-field messages."""
    merged = read_settings(project)
    for k in _defaults():
        if k in patch:
            merged[k] = patch[k]
    errors: list[dict[str, str]] = []
    if merged["upstream"] not in UPSTREAMS:
        errors.append({"field": "upstream", "message": f"upstream must be one of {list(UPSTREAMS)}"})
    kind = merged.get("upstream_kind")
    if kind is not None and kind not in UPSTREAM_KINDS:
        errors.append({"field": "upstream_kind", "message": f"upstream_kind must be one of {list(UPSTREAM_KINDS)}"})
    base_url = merged.get("base_url")
    if base_url is not None and (not isinstance(base_url, str) or not re.match(r"^https?://\S+$", base_url.strip())):
        errors.append({"field": "base_url", "message": "base_url must be an http(s) URL"})
    if merged["upstream"] == "openai_compatible" and not base_url:
        errors.append({"field": "base_url", "message": "base_url is required for an openai_compatible upstream"})
    umap = merged.get("upstream_model_map")
    if not isinstance(umap, dict) or not all(isinstance(a, str) and isinstance(b, str) for a, b in umap.items()):
        errors.append({"field": "upstream_model_map", "message": "upstream_model_map must map model ids to names (strings)"})
        umap = {}
    default_run = merged.get("default_run_id")
    if default_run is not None:
        runs = {e.id for e in await srs.list_servable_runs(db, project.id)}
        if isinstance(default_run, bool) or not isinstance(default_run, int) or default_run not in runs:
            errors.append({"field": "default_run_id", "message": "default_run_id must be a completed run of this project with saved weights"})
    tools, tool_errors = gts.normalize_tools(merged.get("tools"), project)
    errors.extend(tool_errors)
    if errors:
        raise GatewaySettingsError(errors)
    cfg = dict(project.runtime_config or {})
    cfg["gateway"] = {
        "upstream": merged["upstream"],
        "base_url": base_url.strip().rstrip("/") if isinstance(base_url, str) and base_url.strip() else None,
        "upstream_kind": kind, "upstream_model_map": umap, "default_run_id": default_run, "tools": tools,
    }
    project.runtime_config = cfg
    await db.flush()
    return read_settings(project)


async def settings_view(db: AsyncSession, project: Project) -> dict[str, Any]:
    slug = await project_slug(db, project)
    runs = await srs.list_servable_runs(db, project.id)  # ascending by id
    try:
        resolved = await served_run(db, project, runs)
    except GatewayError:
        resolved = None
    from app.services.rag_project_service import is_rag_first

    if resolved is not None:
        resolves_to: dict[str, Any] = {"kind": "run", "run_id": resolved.id}
    elif is_rag_first(project) and project.base_model_name:
        resolves_to = {"kind": "base_with_retrieval", "run_id": None}
    else:
        resolves_to = {"kind": "none", "run_id": None}
    ids = [slug] + ([f"{slug}:base"] if project.base_model_name else []) + [f"{slug}:run-{e.id}" for e in runs]
    adapter_flags = await asyncio.to_thread(lambda: [srs.run_adapter_config(e) is not None for e in runs])
    return {
        "settings": read_settings(project),
        "slug": slug,
        "model_ids": ids,
        "base_model": project.base_model_name or "",
        "default_run_resolved": resolved.id if resolved else None,
        "default_resolves_to": resolves_to,
        "guided_json_available": guided_schema(project) is not None,
        "servable_runs": [
            {
                "id": e.id, "name": e.name, "base_model": e.base_model,
                "passage_trained": _passage_trained(e), "adapter": adapter_flags[i],
            }
            for i, e in enumerate(runs)
        ],
    }


async def build_snippets(db: AsyncSession, project: Project) -> dict[str, Any]:
    import sys
    from pathlib import Path

    view = await settings_view(db, project)
    url = settings.GATEWAY_PUBLIC_URL.rstrip("/")
    model = view["slug"]
    ask = "Hello"
    curl = (
        f"curl -sS {url}/v1/chat/completions \\\n"
        f"  -H \"Authorization: Bearer ${{BREWSLM_API_KEY}}\" \\\n"
        f"  -H \"Content-Type: application/json\" \\\n"
        f"  -d '{json.dumps({'model': model, 'messages': [{'role': 'user', 'content': ask}]})}'"
    )
    py = (
        "import os\nfrom openai import OpenAI\n\n"
        f"client = OpenAI(base_url=\"{url}/v1\", api_key=os.environ[\"BREWSLM_API_KEY\"])\n"
        "resp = client.chat.completions.create(\n"
        f"    model=\"{model}\",\n"
        f"    messages=[{{\"role\": \"user\", \"content\": \"{ask}\"}}],\n"
        "    extra_body={\"brewslm\": {\"retrieval\": True}},\n"
        ")\nprint(resp.choices[0].message.content)\n"
    )
    backend_dir = str(Path(__file__).resolve().parents[2])
    return {
        "gateway_url": url,
        "model_ids": view["model_ids"],
        "mcp_json": {
            "mcpServers": {
                "brewslm": {
                    "type": "stdio", "command": sys.executable, "args": ["-m", "app.mcp_server"], "cwd": backend_dir,
                    "env": {"BREWSLM_GATEWAY_URL": url, "BREWSLM_API_KEY": "${BREWSLM_API_KEY}"},
                }
            }
        },
        "curl": curl,
        "openai_python": py,
    }
