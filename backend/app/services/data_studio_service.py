"""Data Studio overview intelligence.

The Data Studio is an additive UX layer over the existing pipeline.
This service keeps the first slice deliberately deterministic: it
summarizes project data state, computes simple readiness issues, and
returns action targets the frontend can route to existing panels.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.project import Project
from app.services.dataset_service import (
    preview_project_data_adapter,
    resolve_project_dataset_adapter_preference,
)
from app.services.domain_pack_service import get_domain_pack
from app.services.domain_profile_service import get_domain_profile
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.services.recipe_service import get_recipe
from app.services.synth_review_queue_service import list_review_queue


IssueSeverity = Literal["blocker", "warning", "info"]
OverviewVerdict = Literal["blocked", "needs_work", "ready"]
SourcesVerdict = Literal["empty", "attention", "healthy"]
MappingVerdict = Literal["empty", "attention", "ready"]
DomainVerdict = Literal["unknown", "attention", "confirmed"]

_MAPPING_SOURCE_PRIORITY: tuple[DatasetType, ...] = (
    DatasetType.RAW,
    DatasetType.GOLD_DEV,
    DatasetType.SYNTHETIC,
    DatasetType.CLEANED,
    DatasetType.GOLD_TEST,
    DatasetType.TRAIN,
    DatasetType.VALIDATION,
    DatasetType.TEST,
)

_DOMAIN_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "support_faq",
        "label": "Support FAQ",
        "aliases": ("support", "customer_support", "faq", "ticket"),
        "keywords": (
            "refund",
            "password reset",
            "ticket",
            "support",
            "account",
            "login",
            "subscription",
            "cancel",
            "agent",
            "escalate",
            "customer",
        ),
        "field_markers": ("question", "answer", "faq", "ticket", "intent"),
        "recipes": ("qa-sft", "classification"),
        "actions": (
            "Add customer phrasing variants for the top support topics.",
            "Add escalation and refusal examples for account-access requests.",
            "Review answers that sound vague or omit exact next steps.",
        ),
        "risks": (
            "Support data often contains personal account details; run PII checks before training.",
            "Account, billing, and cancellation answers should include escalation boundaries.",
        ),
    },
    {
        "id": "policy_qa",
        "label": "Policy Q&A",
        "aliases": ("policy", "handbook", "procedure"),
        "keywords": (
            "policy",
            "procedure",
            "eligibility",
            "exception",
            "guideline",
            "handbook",
            "benefit",
            "leave",
            "compliance",
            "covered",
        ),
        "field_markers": ("policy", "question", "answer", "section", "context"),
        "recipes": ("qa-sft", "summarization"),
        "actions": (
            "Add edge cases and exceptions from the policy source.",
            "Add insufficient-information examples for questions not covered by policy.",
            "Keep source sections available if the policy changes often.",
        ),
        "risks": (
            "Policy answers can become stale; consider RAG if the source changes frequently.",
            "Overconfident answers should be reviewed when policy text is ambiguous.",
        ),
    },
    {
        "id": "pii_pci_detection",
        "label": "PII/PCI Detection",
        "aliases": ("pii", "pci", "redaction", "privacy"),
        "keywords": (
            "pii",
            "pci",
            "ssn",
            "social security",
            "credit card",
            "card number",
            "cvv",
            "email address",
            "phone number",
            "redact",
            "personal data",
        ),
        "field_markers": ("label", "entity", "pii", "pci", "redaction", "category"),
        "recipes": ("classification", "span-extraction"),
        "actions": (
            "Balance entity classes and add hard negatives that only look sensitive.",
            "Add examples for safe strings such as order IDs or public phone numbers.",
            "Review redaction policy before using raw production data.",
        ),
        "risks": (
            "Sensitive values should be masked or synthesized before training.",
            "False positives and false negatives both matter; keep a gold test set.",
        ),
    },
    {
        "id": "security_alert_triage",
        "label": "Security Alert Triage",
        "aliases": ("security", "alert", "soc", "incident"),
        "keywords": (
            "alert",
            "incident",
            "severity",
            "phishing",
            "malware",
            "firewall",
            "cve",
            "vulnerability",
            "soc",
            "anomaly",
            "login attempt",
        ),
        "field_markers": ("severity", "alert", "incident", "label", "risk"),
        "recipes": ("classification", "generic-sft"),
        "actions": (
            "Add label-balanced examples for each alert severity.",
            "Include benign false-positive examples and escalation rules.",
            "Keep incident-response language consistent across labels.",
        ),
        "risks": (
            "Security triage needs strong false-negative review.",
            "Operational actions should be gated by human approval.",
        ),
    },
    {
        "id": "legal_contracts",
        "label": "Legal Clauses",
        "aliases": ("legal", "contract", "clause"),
        "keywords": (
            "contract",
            "clause",
            "agreement",
            "liability",
            "indemnity",
            "termination",
            "governing law",
            "party",
            "warranty",
        ),
        "field_markers": ("clause", "contract", "section", "question", "answer"),
        "recipes": ("qa-sft", "span-extraction", "summarization"),
        "actions": (
            "Add source-backed examples with clause references.",
            "Add examples for exceptions, unknowns, and human-review handoff.",
            "Keep jurisdiction and document-type metadata visible.",
        ),
        "risks": (
            "Do not present generated output as legal advice.",
            "Citation and provenance checks should be required for high-risk answers.",
        ),
    },
    {
        "id": "finance_support",
        "label": "Finance Support",
        "aliases": ("finance", "billing", "invoice", "payment"),
        "keywords": (
            "invoice",
            "billing",
            "payment",
            "charge",
            "refund",
            "transaction",
            "tax",
            "statement",
            "account balance",
        ),
        "field_markers": ("invoice", "amount", "currency", "payment", "label"),
        "recipes": ("qa-sft", "classification"),
        "actions": (
            "Add numeric edge cases and refund/cancellation variants.",
            "Review examples that mention transactions or account balances.",
            "Keep human approval for money-moving actions.",
        ),
        "risks": (
            "Financial outputs should not be framed as investment or tax advice.",
            "Numeric accuracy needs explicit evaluation coverage.",
        ),
    },
    {
        "id": "code_review",
        "label": "Code Review",
        "aliases": ("code", "review", "diff"),
        "keywords": (
            "code",
            "diff",
            "pull request",
            "function",
            "bug",
            "typescript",
            "python",
            "stack trace",
            "test failure",
        ),
        "field_markers": ("code", "diff", "patch", "review", "file"),
        "recipes": ("code-review", "generic-sft"),
        "actions": (
            "Add examples with both defects and acceptable code.",
            "Include concise rationale for each review finding.",
            "Balance style feedback against correctness and security issues.",
        ),
        "risks": (
            "Review data can overfit to style-only comments without defect examples.",
            "Security-sensitive code suggestions should be evaluated separately.",
        ),
    },
    {
        "id": "customer_sentiment",
        "label": "Customer Sentiment",
        "aliases": ("sentiment", "review", "rating"),
        "keywords": (
            "sentiment",
            "positive",
            "negative",
            "neutral",
            "complaint",
            "review",
            "rating",
            "angry",
            "happy",
        ),
        "field_markers": ("sentiment", "rating", "label", "review"),
        "recipes": ("classification",),
        "actions": (
            "Balance sentiment labels and add ambiguous neutral examples.",
            "Include short and long customer text examples.",
            "Review sarcasm, mixed sentiment, and low-context rows.",
        ),
        "risks": (
            "Class imbalance can make sentiment models look good while missing minority labels.",
        ),
    },
)


def _sum_counts(counter: Counter[str], *types: DatasetType) -> int:
    return sum(int(counter.get(t.value, 0)) for t in types)


def _issue(
    issue_id: str,
    severity: IssueSeverity,
    title: str,
    message: str,
    *,
    action_label: str,
    target_tab: str,
) -> dict[str, str]:
    return {
        "id": issue_id,
        "severity": severity,
        "title": title,
        "message": message,
        "action_label": action_label,
        "target_tab": target_tab,
    }


def _flatten_text_values(value: Any, out: list[str], *, limit: int = 80) -> None:
    if len(out) >= limit:
        return
    if value is None:
        return
    if isinstance(value, str):
        token = value.strip()
        if token:
            out.append(token[:500])
        return
    if isinstance(value, (int, float, bool)):
        out.append(str(value))
        return
    if isinstance(value, dict):
        for item in value.values():
            _flatten_text_values(item, out, limit=limit)
            if len(out) >= limit:
                break
        return
    if isinstance(value, (list, tuple, set)):
        for item in list(value):
            _flatten_text_values(item, out, limit=limit)
            if len(out) >= limit:
                break


def _preview_texts_and_fields(preview: dict[str, Any]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    preview_rows = preview.get("preview_rows")
    if isinstance(preview_rows, list):
        for row in preview_rows:
            if not isinstance(row, dict):
                continue
            _flatten_text_values(row.get("raw"), texts)
            _flatten_text_values(row.get("mapped"), texts)
            if len(texts) >= 80:
                break

    raw_field_frequency = preview.get("raw_field_frequency")
    fields = []
    if isinstance(raw_field_frequency, dict):
        fields = [
            str(key).strip()
            for key in raw_field_frequency.keys()
            if str(key).strip()
        ]
    return texts, fields


def _has_any_field(fields: list[str], markers: tuple[str, ...]) -> bool:
    normalized = [field.lower() for field in fields]
    return any(marker in field for field in normalized for marker in markers)


def _has_field_pair(
    fields: list[str],
    left_markers: tuple[str, ...],
    right_markers: tuple[str, ...],
) -> bool:
    return _has_any_field(fields, left_markers) and _has_any_field(fields, right_markers)


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _runtime_text(
    runtime: dict[str, Any],
    applied: dict[str, Any],
) -> str:
    parts = [
        runtime.get("domain_profile_applied"),
        runtime.get("domain_pack_applied"),
        applied.get("profile_display_name"),
        applied.get("pack_display_name"),
    ]
    return " ".join(str(part or "").lower() for part in parts)


def _is_generic_runtime(runtime: dict[str, Any], applied: dict[str, Any]) -> bool:
    text = _runtime_text(runtime, applied)
    if not text.strip():
        return True
    return any(token in text for token in ("generic", "general", "default", "fallback"))


def _runtime_matches_domain(
    domain_id: str,
    runtime: dict[str, Any],
    applied: dict[str, Any],
) -> bool:
    runtime_blob = _runtime_text(runtime, applied)
    for definition in _DOMAIN_DEFINITIONS:
        if definition["id"] != domain_id:
            continue
        return any(str(alias).lower() in runtime_blob for alias in definition["aliases"])
    return False


def _score_domain_candidates(
    *,
    texts: list[str],
    fields: list[str],
    inferred_task_profiles: list[str],
) -> list[dict[str, Any]]:
    text_blob = "\n".join(texts).lower()
    normalized_profiles = {
        str(profile or "").strip().lower().replace("-", "_")
        for profile in inferred_task_profiles
        if str(profile or "").strip()
    }
    scored: list[dict[str, Any]] = []
    for definition in _DOMAIN_DEFINITIONS:
        keywords = tuple(str(item).lower() for item in definition["keywords"])
        field_markers = tuple(str(item).lower() for item in definition["field_markers"])
        matched_terms = [term for term in keywords if term in text_blob]
        matched_fields = [
            field for field in fields
            if any(marker in field.lower() for marker in field_markers)
        ]

        score = 0.0
        score += min(0.62, 0.12 * len(matched_terms))
        score += min(0.18, 0.06 * len(set(matched_fields)))

        if _has_field_pair(fields, ("question", "prompt", "query"), ("answer", "response", "output")):
            if definition["id"] in {"support_faq", "policy_qa", "legal_contracts"}:
                score += 0.14
        if _has_any_field(fields, ("label", "class", "category")):
            if definition["id"] in {"pii_pci_detection", "security_alert_triage", "customer_sentiment"}:
                score += 0.12
        if "classification" in normalized_profiles:
            if definition["id"] in {"pii_pci_detection", "security_alert_triage", "customer_sentiment", "support_faq"}:
                score += 0.06
        if "qa" in normalized_profiles or "rag_qa" in normalized_profiles:
            if definition["id"] in {"support_faq", "policy_qa", "legal_contracts", "finance_support"}:
                score += 0.06

        score = round(min(1.0, score), 4)
        evidence: list[dict[str, Any]] = []
        signals: list[str] = []
        if matched_fields:
            top_fields = sorted(set(matched_fields))[:6]
            evidence.append({
                "id": "field_signals",
                "title": "Column signals",
                "message": f"Fields match this domain: {', '.join(top_fields)}.",
                "score": round(min(1.0, 0.25 + (0.08 * len(top_fields))), 4),
            })
            signals.append(f"columns:{','.join(top_fields)}")
        if matched_terms:
            top_terms = matched_terms[:8]
            evidence.append({
                "id": "term_signals",
                "title": "Content signals",
                "message": f"Sampled rows mention: {', '.join(top_terms)}.",
                "score": round(min(1.0, 0.25 + (0.08 * len(top_terms))), 4),
            })
            signals.append(f"terms:{','.join(top_terms)}")
        if _has_field_pair(fields, ("question", "prompt", "query"), ("answer", "response", "output")):
            evidence.append({
                "id": "qa_shape",
                "title": "Q&A row shape",
                "message": "Fields look like question/answer or prompt/response pairs.",
                "score": 0.74,
            })
            signals.append("row_shape:qa_pair")
        if _has_any_field(fields, ("label", "class", "category")):
            signals.append("row_shape:labeled_examples")

        scored.append({
            "id": definition["id"],
            "label": definition["label"],
            "confidence": score,
            "matched_keywords": matched_terms[:12],
            "matched_fields": sorted(set(matched_fields))[:12],
            "evidence": evidence,
            "signals": signals,
            "actions": list(definition["actions"]),
            "risks": list(definition["risks"]),
            "recommended_recipes": list(definition["recipes"]),
        })

    scored.sort(key=lambda item: (-float(item["confidence"]), str(item["id"])))
    return scored


def _runtime_detection_fallback(
    *,
    runtime: dict[str, Any],
    applied: dict[str, Any],
) -> dict[str, Any]:
    runtime_blob = _runtime_text(runtime, applied)
    for definition in _DOMAIN_DEFINITIONS:
        if any(str(alias).lower() in runtime_blob for alias in definition["aliases"]):
            source = str(runtime.get("domain_profile_source") or runtime.get("domain_pack_source") or "runtime")
            confidence = 0.84 if source == "project" else 0.68
            return {
                "id": definition["id"],
                "label": definition["label"],
                "confidence": confidence,
                "confidence_label": _confidence_label(confidence),
                "source": "applied_runtime",
                "summary": "Using the applied domain runtime because no stronger source sample was available.",
                "matched_keywords": [],
                "matched_fields": [],
                "evidence": [
                    {
                        "id": "runtime_applied",
                        "title": "Applied domain runtime",
                        "message": f"Project runtime is using {applied.get('profile_display_name') or runtime.get('domain_profile_applied') or 'a domain profile'}.",
                        "score": confidence,
                    }
                ],
                "signals": [f"runtime:{runtime.get('domain_profile_applied') or runtime.get('domain_pack_applied') or 'domain'}"],
                "actions": list(definition["actions"]),
                "risks": list(definition["risks"]),
                "recommended_recipes": list(definition["recipes"]),
            }

    confidence = 0.25
    return {
        "id": "generic_domain",
        "label": "Generic Domain",
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "source": "runtime_default",
        "summary": "No specific domain has been detected yet.",
        "matched_keywords": [],
        "matched_fields": [],
        "evidence": [
            {
                "id": "generic_runtime",
                "title": "Generic runtime",
                "message": "The project is using the generic domain defaults.",
                "score": confidence,
            }
        ],
        "signals": ["runtime:generic"],
        "actions": [
            "Add representative source rows so BrewSLM can infer the domain.",
            "Assign a domain profile or pack if you already know the use case.",
        ],
        "risks": [
            "Generic defaults may miss domain-specific quality, safety, and coverage checks.",
        ],
        "recommended_recipes": [],
    }


async def _domain_applied_summary(
    db: AsyncSession,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    profile_id = str(runtime.get("domain_profile_applied") or "").strip()
    pack_id = str(runtime.get("domain_pack_applied") or "").strip()
    profile = await get_domain_profile(db, profile_id) if profile_id else None
    pack = await get_domain_pack(db, pack_id) if pack_id else None
    effective_contract = runtime.get("effective_contract")
    profile_display_name = (
        profile.display_name
        if profile is not None
        else (
            effective_contract.get("display_name")
            if isinstance(effective_contract, dict)
            else None
        )
    )
    return {
        "profile_id": profile_id or None,
        "profile_source": runtime.get("domain_profile_source"),
        "profile_display_name": profile_display_name,
        "profile_version": profile.version if profile is not None else None,
        "pack_id": pack_id or None,
        "pack_source": runtime.get("domain_pack_source"),
        "pack_display_name": pack.display_name if pack is not None else None,
        "pack_version": pack.version if pack is not None else None,
        "pack_default_profile_id": runtime.get("pack_default_profile_id"),
    }


def _domain_actions(candidate: dict[str, Any]) -> list[dict[str, str]]:
    actions = []
    targets = ("synthetic", "goldset", "dataprep")
    for index, action in enumerate(list(candidate.get("actions") or [])[:4]):
        actions.append({
            "id": f"domain_action_{index + 1}",
            "label": str(action),
            "target_tab": targets[index % len(targets)],
        })
    return actions


def _domain_risks(candidate: dict[str, Any]) -> list[dict[str, str]]:
    risks = []
    for index, risk in enumerate(list(candidate.get("risks") or [])[:4]):
        risks.append({
            "id": f"domain_risk_{index + 1}",
            "severity": "warning" if index == 0 else "info",
            "title": "Domain risk" if index == 0 else "Domain note",
            "message": str(risk),
        })
    return risks


async def build_data_studio_domain_detection(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return deterministic domain detection and applied runtime evidence."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    runtime = await resolve_project_domain_runtime(db, project_id)
    applied = await _domain_applied_summary(db, runtime)
    recipe_payload = _recipe_payload(project)

    issues: list[dict[str, str]] = []
    source = await _select_mapping_source(db, project_id)
    preview: dict[str, Any] = {}
    source_payload: dict[str, Any] | None = None
    if source is not None:
        dataset_type = source["dataset_type"]
        try:
            preview = await preview_project_data_adapter(
                db=db,
                project_id=project_id,
                dataset_type=dataset_type,
                sample_size=120,
                adapter_id="auto",
                document_id=source.get("document_id"),
                preview_limit=12,
            )
            source_payload = {
                **source,
                "dataset_type": dataset_type.value,
                "sampled_records": int(preview.get("sampled_records") or 0),
            }
        except Exception as exc:  # noqa: BLE001
            issues.append(
                _issue(
                    "domain_sample_failed",
                    "warning",
                    "Could not inspect source sample",
                    str(exc)[:240],
                    action_label="Inspect sources",
                    target_tab="data",
                )
            )

    texts, fields = _preview_texts_and_fields(preview)
    inferred_profiles = [
        str(item)
        for item in list(preview.get("inferred_task_profiles") or [])
        if str(item).strip()
    ]
    candidates = _score_domain_candidates(
        texts=texts,
        fields=fields,
        inferred_task_profiles=inferred_profiles,
    )
    top_candidate = candidates[0] if candidates else {}
    if float(top_candidate.get("confidence") or 0.0) >= 0.3:
        confidence = float(top_candidate["confidence"])
        detected = {
            **top_candidate,
            "confidence_label": _confidence_label(confidence),
            "source": "sampled_data",
            "summary": f"Detected from {len(fields)} field(s) and {len(texts)} sampled text signal(s).",
        }
    else:
        detected = _runtime_detection_fallback(runtime=runtime, applied=applied)

    no_source = source is None or int((source_payload or {}).get("sampled_records") or 0) <= 0
    runtime_is_generic = _is_generic_runtime(runtime, applied)
    runtime_matches = _runtime_matches_domain(str(detected.get("id") or ""), runtime, applied)
    confidence = float(detected.get("confidence") or 0.0)

    if no_source:
        issues.append(
            _issue(
                "domain_needs_source_evidence",
                "warning" if runtime_is_generic else "info",
                "Domain evidence is limited",
                "Add source rows so BrewSLM can confirm the training domain from real examples.",
                action_label="Add sources",
                target_tab="data",
            )
        )

    if confidence < 0.45:
        issues.append(
            _issue(
                "low_domain_confidence",
                "info",
                "Low domain confidence",
                "The current sample does not contain enough domain-specific evidence yet.",
                action_label="Add representative rows",
                target_tab="data",
            )
        )

    if (
        str(detected.get("id") or "") != "generic_domain"
        and confidence >= 0.45
        and runtime_is_generic
    ):
        issues.append(
            _issue(
                "domain_candidate_not_applied",
                "warning",
                "Specific domain not applied",
                f"Sampled rows look like {detected.get('label')}, but the project is still using generic domain defaults.",
                action_label="Review domain settings",
                target_tab="data",
            )
        )

    if (
        str(detected.get("id") or "") != "generic_domain"
        and confidence >= 0.65
        and not runtime_is_generic
        and not runtime_matches
    ):
        issues.append(
            _issue(
                "domain_runtime_mismatch",
                "warning",
                "Applied domain may not match data",
                f"Sampled rows look like {detected.get('label')}, but the applied profile/pack points elsewhere.",
                action_label="Review domain settings",
                target_tab="data",
            )
        )

    if confidence >= 0.65 and (runtime_matches or not runtime_is_generic):
        verdict: DomainVerdict = "confirmed"
    elif issues:
        verdict = "attention"
    else:
        verdict = "unknown"

    return {
        "project_id": project_id,
        "verdict": verdict,
        "detected_domain": {
            "id": detected.get("id"),
            "label": detected.get("label"),
            "confidence": round(confidence, 4),
            "confidence_label": detected.get("confidence_label") or _confidence_label(confidence),
            "source": detected.get("source"),
            "summary": detected.get("summary"),
            "matched_keywords": list(detected.get("matched_keywords") or []),
            "matched_fields": list(detected.get("matched_fields") or []),
            "recommended_recipes": list(detected.get("recommended_recipes") or []),
        },
        "applied": applied,
        "recipe": recipe_payload,
        "source": source_payload,
        "evidence": list(detected.get("evidence") or []),
        "suggested_actions": _domain_actions(detected),
        "risks": _domain_risks(detected),
        "issues": issues,
        "power_details": {
            "signals": list(detected.get("signals") or []),
            "candidate_domains": candidates[:5],
            "runtime": {
                "domain_profile_applied": runtime.get("domain_profile_applied"),
                "domain_profile_source": runtime.get("domain_profile_source"),
                "domain_pack_applied": runtime.get("domain_pack_applied"),
                "domain_pack_source": runtime.get("domain_pack_source"),
                "pack_default_profile_id": runtime.get("pack_default_profile_id"),
            },
            "raw_fields": fields,
            "inferred_task_profiles": inferred_profiles,
        },
    }


def _primary_action(
    issues: list[dict[str, str]],
    *,
    prepared_rows: int,
) -> dict[str, str]:
    if issues:
        first = issues[0]
        return {
            "label": first["action_label"],
            "target_tab": first["target_tab"],
            "reason": first["title"],
        }
    if prepared_rows > 0:
        return {
            "label": "Open training",
            "target_tab": "training",
            "reason": "A prepared dataset is available.",
        }
    return {
        "label": "Prepare dataset",
        "target_tab": "dataprep",
        "reason": "Data looks usable; create train/validation/test splits next.",
    }


def _recipe_payload(project: Project) -> dict[str, Any] | None:
    selected_recipe = project.selected_recipe if isinstance(project.selected_recipe, dict) else {}
    recipe_id = str(selected_recipe.get("recipe_id") or "").strip()
    recipe = get_recipe(recipe_id) if recipe_id else None
    if recipe is not None:
        return {
            "id": recipe.id,
            "name": recipe.name,
            "task_profile": recipe.task_profile,
            "adapter_id": recipe.adapter_id,
            "default_input_column": recipe.default_input_column,
            "default_output_column": recipe.default_output_column,
        }
    if recipe_id:
        return {
            "id": recipe_id,
            "name": str(selected_recipe.get("name") or recipe_id),
            "task_profile": str(selected_recipe.get("task_profile") or ""),
            "adapter_id": str(selected_recipe.get("adapter_id") or ""),
            "default_input_column": str(selected_recipe.get("default_input_column") or ""),
            "default_output_column": str(selected_recipe.get("default_output_column") or ""),
        }
    return None


def _issue_status(issues: list[dict[str, str]], *, empty: bool = False) -> MappingVerdict:
    if empty:
        return "empty"
    if any(item["severity"] in {"blocker", "warning"} for item in issues):
        return "attention"
    return "ready"


async def build_data_studio_overview(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a project-level Data Studio readiness summary."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    datasets = list(datasets_result.scalars().all())
    dataset_counts: Counter[str] = Counter()
    for dataset in datasets:
        dataset_counts[dataset.dataset_type.value] += int(dataset.record_count or 0)

    doc_status_result = await db.execute(
        select(RawDocument.status, func.count(RawDocument.id))
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
        .group_by(RawDocument.status)
    )
    document_counts = {
        (status.value if isinstance(status, DocumentStatus) else str(status)): int(count or 0)
        for status, count in doc_status_result.all()
    }

    review_queue = await list_review_queue(db, project_id)
    synthetic_pending = int(review_queue.get("total_pending") or 0)
    synthetic_accepted = int(review_queue.get("total_accepted") or 0)

    raw_rows = _sum_counts(dataset_counts, DatasetType.RAW)
    cleaned_rows = _sum_counts(dataset_counts, DatasetType.CLEANED)
    gold_rows = _sum_counts(dataset_counts, DatasetType.GOLD_DEV, DatasetType.GOLD_TEST)
    prepared_rows = _sum_counts(
        dataset_counts,
        DatasetType.TRAIN,
        DatasetType.VALIDATION,
        DatasetType.TEST,
    )
    default_trainable_rows = cleaned_rows + gold_rows + synthetic_accepted
    trainable_rows = default_trainable_rows if default_trainable_rows > 0 else raw_rows

    recipe_payload = _recipe_payload(project)

    try:
        domain_runtime = await resolve_project_domain_runtime(db, project_id)
    except ValueError:
        domain_runtime = {}
    effective_contract = domain_runtime.get("effective_contract")
    domain_payload = {
        "profile_id": domain_runtime.get("domain_profile_applied"),
        "profile_source": domain_runtime.get("domain_profile_source"),
        "pack_id": domain_runtime.get("domain_pack_applied"),
        "pack_source": domain_runtime.get("domain_pack_source"),
        "display_name": (
            effective_contract.get("display_name")
            if isinstance(effective_contract, dict)
            else None
        ),
    }

    issues: list[dict[str, str]] = []
    if recipe_payload is None:
        issues.append(
            _issue(
                "missing_recipe",
                "blocker",
                "Recipe not selected",
                "Pick a task recipe so BrewSLM knows the training shape and validation rules.",
                action_label="Choose recipe",
                target_tab="data",
            )
        )

    if trainable_rows <= 0:
        issues.append(
            _issue(
                "no_trainable_rows",
                "blocker",
                "No trainable rows yet",
                "Import data, create gold rows, or accept reviewed synthetic rows before preparing a dataset.",
                action_label="Add sources",
                target_tab="data",
            )
        )
    elif trainable_rows < 20:
        issues.append(
            _issue(
                "low_trainable_rows",
                "warning",
                "Very small training set",
                f"{trainable_rows} trainable row(s) is enough to inspect the flow, but most useful SFT runs need more examples.",
                action_label="Add or generate rows",
                target_tab="synthetic",
            )
        )

    if synthetic_pending > 0:
        issues.append(
            _issue(
                "pending_synthetic_rows",
                "warning",
                "Synthetic rows pending review",
                f"{synthetic_pending} generated row(s) are gated out of training until accepted.",
                action_label="Review synthetic rows",
                target_tab="synthetic",
            )
        )

    if trainable_rows > 0 and prepared_rows <= 0:
        issues.append(
            _issue(
                "dataset_not_prepared",
                "warning",
                "Training dataset not prepared",
                "Create train, validation, and test splits before launching a training run.",
                action_label="Prepare dataset",
                target_tab="dataprep",
            )
        )

    if document_counts.get(DocumentStatus.ERROR.value, 0) > 0:
        issues.append(
            _issue(
                "source_errors",
                "warning",
                "Some sources failed ingestion",
                f"{document_counts[DocumentStatus.ERROR.value]} source document(s) need attention.",
                action_label="Inspect sources",
                target_tab="data",
            )
        )

    blocker_count = sum(1 for item in issues if item["severity"] == "blocker")
    warning_count = sum(1 for item in issues if item["severity"] == "warning")
    if blocker_count:
        verdict: OverviewVerdict = "blocked"
    elif warning_count:
        verdict = "needs_work"
    else:
        verdict = "ready"

    return {
        "project_id": project_id,
        "verdict": verdict,
        "recipe": recipe_payload,
        "domain": domain_payload,
        "row_counts": {
            "trainable": trainable_rows,
            "raw": raw_rows,
            "cleaned": cleaned_rows,
            "gold": gold_rows,
            "synthetic_total": synthetic_pending + synthetic_accepted,
            "synthetic_pending": synthetic_pending,
            "synthetic_accepted": synthetic_accepted,
            "prepared": prepared_rows,
            "train": int(dataset_counts.get(DatasetType.TRAIN.value, 0)),
            "validation": int(dataset_counts.get(DatasetType.VALIDATION.value, 0)),
            "test": int(dataset_counts.get(DatasetType.TEST.value, 0)),
        },
        "source_summary": {
            "dataset_count": len(datasets),
            "documents_total": sum(document_counts.values()),
            "documents_accepted": int(document_counts.get(DocumentStatus.ACCEPTED.value, 0)),
            "documents_processing": int(document_counts.get(DocumentStatus.PROCESSING.value, 0)),
            "documents_pending": int(document_counts.get(DocumentStatus.PENDING.value, 0)),
            "documents_error": int(document_counts.get(DocumentStatus.ERROR.value, 0)),
        },
        "issues": issues,
        "primary_action": _primary_action(issues, prepared_rows=prepared_rows),
    }


async def build_data_studio_sources(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return source health and recent source rows for Data Studio."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    datasets = list(datasets_result.scalars().all())

    groups_by_type: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for dataset in datasets:
        type_key = dataset.dataset_type.value
        group = groups_by_type.setdefault(
            type_key,
            {
                "dataset_type": type_key,
                "dataset_count": 0,
                "row_count": 0,
                "locked_count": 0,
                "with_file_count": 0,
            },
        )
        group["dataset_count"] += 1
        group["row_count"] += int(dataset.record_count or 0)
        group["locked_count"] += 1 if dataset.is_locked else 0
        group["with_file_count"] += 1 if dataset.file_path else 0
        total_rows += int(dataset.record_count or 0)

    docs_result = await db.execute(
        select(RawDocument, Dataset)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
        .order_by(RawDocument.ingested_at.desc())
    )
    doc_rows = list(docs_result.all())

    status_counts: Counter[str] = Counter()
    recent_documents: list[dict[str, Any]] = []
    for doc, dataset in doc_rows:
        status = doc.status.value if isinstance(doc.status, DocumentStatus) else str(doc.status)
        status_counts[status] += 1
        if len(recent_documents) >= 8:
            continue
        recent_documents.append({
            "id": doc.id,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "dataset_type": dataset.dataset_type.value,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "status": status,
            "source": doc.source or "upload",
            "sensitivity": doc.sensitivity or "internal",
            "file_size_bytes": int(doc.file_size_bytes or 0),
            "chunk_count": int(doc.chunk_count or 0),
            "quality_score": doc.quality_score,
            "ingested_at": doc.ingested_at.isoformat() if doc.ingested_at else None,
        })

    issues: list[dict[str, str]] = []
    if not datasets and not doc_rows:
        issues.append(
            _issue(
                "no_sources",
                "blocker",
                "No sources connected",
                "Add a local file, remote dataset, or project template to start building training data.",
                action_label="Add sources",
                target_tab="data",
            )
        )

    error_count = int(status_counts.get(DocumentStatus.ERROR.value, 0))
    if error_count:
        issues.append(
            _issue(
                "source_errors",
                "warning",
                "Source import errors",
                f"{error_count} source document(s) failed ingestion and need attention.",
                action_label="Inspect failed sources",
                target_tab="data",
            )
        )

    in_flight_count = int(
        status_counts.get(DocumentStatus.PENDING.value, 0)
        + status_counts.get(DocumentStatus.PROCESSING.value, 0)
    )
    if in_flight_count:
        issues.append(
            _issue(
                "sources_in_progress",
                "info",
                "Sources still processing",
                f"{in_flight_count} source document(s) are pending or processing.",
                action_label="Refresh sources",
                target_tab="data",
            )
        )

    empty_dataset_count = sum(1 for dataset in datasets if int(dataset.record_count or 0) <= 0)
    if datasets and empty_dataset_count == len(datasets):
        issues.append(
            _issue(
                "empty_datasets",
                "warning",
                "Datasets have no rows yet",
                "The project has dataset records, but no counted rows are available for training.",
                action_label="Inspect sources",
                target_tab="data",
            )
        )

    if not datasets and not doc_rows:
        verdict: SourcesVerdict = "empty"
    elif error_count or (datasets and empty_dataset_count == len(datasets)):
        verdict = "attention"
    else:
        verdict = "healthy"

    dataset_groups = sorted(
        groups_by_type.values(),
        key=lambda item: (str(item["dataset_type"]), -int(item["row_count"])),
    )

    return {
        "project_id": project_id,
        "verdict": verdict,
        "totals": {
            "dataset_count": len(datasets),
            "document_count": len(doc_rows),
            "row_count": total_rows,
            "accepted_documents": int(status_counts.get(DocumentStatus.ACCEPTED.value, 0)),
            "pending_documents": int(status_counts.get(DocumentStatus.PENDING.value, 0)),
            "processing_documents": int(status_counts.get(DocumentStatus.PROCESSING.value, 0)),
            "error_documents": error_count,
            "rejected_documents": int(status_counts.get(DocumentStatus.REJECTED.value, 0)),
        },
        "dataset_groups": dataset_groups,
        "recent_documents": recent_documents,
        "issues": issues,
    }


async def _select_mapping_source(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any] | None:
    raw_docs_result = await db.execute(
        select(RawDocument, Dataset)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
            RawDocument.status == DocumentStatus.ACCEPTED,
        )
        .order_by(RawDocument.ingested_at.desc())
    )
    raw_doc_rows = list(raw_docs_result.all())
    if raw_doc_rows:
        doc, dataset = raw_doc_rows[0]
        return {
            "dataset_type": DatasetType.RAW,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "document_id": doc.id,
            "document_name": doc.filename,
            "document_count": len(raw_doc_rows),
            "row_count": int(dataset.record_count or 0),
        }

    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    datasets = list(datasets_result.scalars().all())
    for dataset_type in _MAPPING_SOURCE_PRIORITY[1:]:
        candidates = [
            dataset
            for dataset in datasets
            if dataset.dataset_type == dataset_type
            and bool(str(dataset.file_path or "").strip())
            and int(dataset.record_count or 0) > 0
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item.updated_at, reverse=True)
        dataset = candidates[0]
        return {
            "dataset_type": dataset.dataset_type,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "document_id": None,
            "document_name": None,
            "document_count": 0,
            "row_count": int(dataset.record_count or 0),
        }
    return None


def _coverage_rows(conformance_report: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = conformance_report.get("required_field_coverage")
    if not isinstance(coverage, dict):
        return []
    rows: list[dict[str, Any]] = []
    for field, stats in coverage.items():
        if not isinstance(stats, dict):
            stats = {}
        rows.append({
            "field": str(field),
            "present": int(stats.get("present") or 0),
            "missing": int(stats.get("missing") or 0),
            "ratio": float(stats.get("ratio") or 0.0),
        })
    rows.sort(key=lambda item: (float(item["ratio"]), str(item["field"])))
    return rows


def _compact_preview_rows(preview_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(preview_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in preview_rows[:3]:
        if not isinstance(row, dict):
            continue
        rows.append({
            "index": int(row.get("index") or 0),
            "raw": row.get("raw") if isinstance(row.get("raw"), dict) else {},
            "mapped": row.get("mapped") if isinstance(row.get("mapped"), dict) else {},
        })
    return rows


def _empty_mapping_payload(
    *,
    project_id: int,
    verdict: MappingVerdict,
    recipe_payload: dict[str, Any] | None,
    preference_source: str,
    preference_adapter_id: str,
    preference_task_profile: str,
    field_mapping: dict[str, str],
    adapter_config: dict[str, Any],
    effective_source: str,
    effective_adapter_id: str,
    effective_task_profile: str | None,
    source: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "verdict": verdict,
        "recipe": recipe_payload,
        "preference": {
            "source": preference_source,
            "adapter_id": preference_adapter_id or "default-canonical",
            "task_profile": preference_task_profile or None,
            "field_mapping": field_mapping,
            "field_mapping_count": len(field_mapping),
        },
        "effective_mapping": {
            "source": effective_source,
            "adapter_id": effective_adapter_id,
            "task_profile": effective_task_profile,
            "adapter_config": adapter_config,
            "field_mapping": field_mapping,
        },
        "source": source,
        "summary": {
            "sampled_records": 0,
            "mapped_records": 0,
            "dropped_records": 0,
            "error_count": 0,
            "mapping_success_rate": 0.0,
            "contract_pass": False,
            "required_fields": [],
            "required_fields_below_100": [],
            "required_field_coverage": [],
        },
        "preview_rows": [],
        "diagnostics": {},
        "issues": issues,
    }


async def build_data_studio_mapping_preview(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a recipe-aware adapter/schema preview for Data Studio."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    recipe_payload = _recipe_payload(project)
    preference = await resolve_project_dataset_adapter_preference(db, project_id)
    preference_source = str(preference.get("source") or "default")
    field_mapping = dict(preference.get("field_mapping") or {})
    adapter_config = dict(preference.get("adapter_config") or {})

    recipe_adapter_id = str((recipe_payload or {}).get("adapter_id") or "").strip()
    recipe_task_profile = str((recipe_payload or {}).get("task_profile") or "").strip()
    preference_adapter_id = str(preference.get("adapter_id") or "").strip()
    preference_task_profile = str(preference.get("task_profile") or "").strip()

    if preference_source in {"project", "domain_pack"}:
        effective_adapter_id = preference_adapter_id or recipe_adapter_id or "default-canonical"
        effective_task_profile = preference_task_profile or recipe_task_profile or None
        effective_source = preference_source
    else:
        effective_adapter_id = recipe_adapter_id or preference_adapter_id or "default-canonical"
        effective_task_profile = recipe_task_profile or preference_task_profile or None
        effective_source = "recipe" if recipe_adapter_id else preference_source

    issues: list[dict[str, str]] = []
    if recipe_payload is None:
        issues.append(
            _issue(
                "missing_recipe",
                "warning",
                "Recipe not selected",
                "Pick a recipe to validate the mapping against the task shape you plan to train.",
                action_label="Choose recipe",
                target_tab="data",
            )
        )

    source = await _select_mapping_source(db, project_id)
    if source is None:
        issues.append(
            _issue(
                "no_mapping_source",
                "blocker",
                "No previewable rows",
                "Add an accepted raw document or a row-backed dataset before checking schema mapping.",
                action_label="Add sources",
                target_tab="data",
            )
        )
        return _empty_mapping_payload(
            project_id=project_id,
            verdict="empty",
            recipe_payload=recipe_payload,
            preference_source=preference_source,
            preference_adapter_id=preference_adapter_id,
            preference_task_profile=preference_task_profile,
            field_mapping=field_mapping,
            adapter_config=adapter_config,
            effective_source=effective_source,
            effective_adapter_id=effective_adapter_id,
            effective_task_profile=effective_task_profile,
            source=None,
            issues=issues,
        )

    dataset_type = source["dataset_type"]
    try:
        preview = await preview_project_data_adapter(
            db=db,
            project_id=project_id,
            dataset_type=dataset_type,
            sample_size=100,
            adapter_id=effective_adapter_id,
            adapter_config=adapter_config,
            field_mapping=field_mapping,
            task_profile=effective_task_profile,
            document_id=source.get("document_id"),
            preview_limit=3,
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "mapping_preview_failed",
                "warning",
                "Mapping preview could not run",
                str(exc)[:240],
                action_label="Open adapter preview",
                target_tab="dataprep",
            )
        )
        source_payload = {
            **source,
            "dataset_type": dataset_type.value,
        }
        return _empty_mapping_payload(
            project_id=project_id,
            verdict="attention",
            recipe_payload=recipe_payload,
            preference_source=preference_source,
            preference_adapter_id=preference_adapter_id,
            preference_task_profile=preference_task_profile,
            field_mapping=field_mapping,
            adapter_config=adapter_config,
            effective_source=effective_source,
            effective_adapter_id=effective_adapter_id,
            effective_task_profile=effective_task_profile,
            source=source_payload,
            issues=issues,
        )

    conformance_report = (
        preview.get("conformance_report")
        if isinstance(preview.get("conformance_report"), dict)
        else {}
    )
    sampled_records = int(preview.get("sampled_records") or 0)
    mapped_records = int(preview.get("mapped_records") or 0)
    dropped_records = int(preview.get("dropped_records") or 0)
    error_count = int(preview.get("error_count") or 0)
    required_fields = [
        str(item)
        for item in list(conformance_report.get("required_fields") or [])
        if str(item).strip()
    ]
    required_fields_below_100 = [
        str(item)
        for item in list(conformance_report.get("required_fields_below_100") or [])
        if str(item).strip()
    ]
    mapping_success_rate = float(conformance_report.get("mapping_success_rate") or 0.0)
    contract_pass = bool(conformance_report.get("contract_pass"))

    if sampled_records <= 0:
        issues.append(
            _issue(
                "no_sampled_rows",
                "blocker",
                "Source has no readable rows",
                "The selected source exists, but BrewSLM could not read sample rows from it.",
                action_label="Inspect sources",
                target_tab="data",
            )
        )
    elif mapped_records <= 0:
        issues.append(
            _issue(
                "no_mapped_rows",
                "blocker",
                "No rows mapped to the recipe shape",
                "The active adapter could not turn sampled rows into canonical training records.",
                action_label="Open adapter preview",
                target_tab="dataprep",
            )
        )
    elif required_fields_below_100:
        issues.append(
            _issue(
                "required_fields_missing",
                "warning",
                "Required fields are incomplete",
                f"Missing coverage for: {', '.join(required_fields_below_100)}.",
                action_label="Review mapping",
                target_tab="dataprep",
            )
        )

    if dropped_records > 0 or error_count > 0:
        issues.append(
            _issue(
                "mapping_drops",
                "warning",
                "Some rows dropped during mapping",
                f"{dropped_records} sampled row(s) dropped and {error_count} adapter error(s) were reported.",
                action_label="Open adapter preview",
                target_tab="dataprep",
            )
        )

    if preview.get("task_profile_compatible") is False:
        issues.append(
            _issue(
                "task_profile_mismatch",
                "warning",
                "Task profile does not match adapter",
                "The requested task profile is not declared by the resolved adapter.",
                action_label="Review adapter",
                target_tab="dataprep",
            )
        )

    if (
        recipe_adapter_id
        and preference_source in {"project", "domain_pack"}
        and preference_adapter_id
        and preference_adapter_id != recipe_adapter_id
    ):
        issues.append(
            _issue(
                "adapter_differs_from_recipe",
                "info",
                "Adapter preset differs from recipe default",
                f"Using {preference_adapter_id} from {preference_source}; the recipe default is {recipe_adapter_id}.",
                action_label="Review adapter",
                target_tab="dataprep",
            )
        )

    return {
        "project_id": project_id,
        "verdict": _issue_status(issues),
        "recipe": recipe_payload,
        "preference": {
            "source": preference_source,
            "adapter_id": preference_adapter_id or "default-canonical",
            "task_profile": preference_task_profile or None,
            "field_mapping": field_mapping,
            "field_mapping_count": len(field_mapping),
        },
        "effective_mapping": {
            "source": effective_source,
            "adapter_id": str(preview.get("resolved_adapter_id") or effective_adapter_id),
            "requested_adapter_id": str(preview.get("requested_adapter_id") or effective_adapter_id),
            "task_profile": str(preview.get("resolved_task_profile") or effective_task_profile or ""),
            "requested_task_profile": preview.get("requested_task_profile"),
            "adapter_config": adapter_config,
            "field_mapping": field_mapping,
            "auto_apply": preview.get("auto_apply") if isinstance(preview.get("auto_apply"), dict) else {},
        },
        "source": {
            **source,
            "dataset_type": dataset_type.value,
        },
        "summary": {
            "sampled_records": sampled_records,
            "mapped_records": mapped_records,
            "dropped_records": dropped_records,
            "error_count": error_count,
            "mapping_success_rate": mapping_success_rate,
            "contract_pass": contract_pass,
            "required_fields": required_fields,
            "required_fields_below_100": required_fields_below_100,
            "required_field_coverage": _coverage_rows(conformance_report),
        },
        "preview_rows": _compact_preview_rows(preview.get("preview_rows")),
        "diagnostics": {
            "adapter_contract": preview.get("adapter_contract") if isinstance(preview.get("adapter_contract"), dict) else {},
            "validation_report": preview.get("validation_report") if isinstance(preview.get("validation_report"), dict) else {},
            "detection_scores": preview.get("detection_scores") if isinstance(preview.get("detection_scores"), dict) else {},
            "auto_fix_suggestions": preview.get("auto_fix_suggestions") if isinstance(preview.get("auto_fix_suggestions"), list) else [],
            "compatibility_warnings": preview.get("compatibility_warnings") if isinstance(preview.get("compatibility_warnings"), list) else [],
            "inferred_task_profiles": preview.get("inferred_task_profiles") if isinstance(preview.get("inferred_task_profiles"), list) else [],
        },
        "issues": issues,
    }
