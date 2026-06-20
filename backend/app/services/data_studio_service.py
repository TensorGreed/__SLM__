"""Data Studio overview intelligence.

The Data Studio is an additive UX layer over the existing pipeline.
This service keeps the first slice deliberately deterministic: it
summarizes project data state, computes simple readiness issues, and
returns action targets the frontend can route to existing panels.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DatasetVersion, DocumentStatus, RawDocument
from app.models.gold_set_annotation import (
    GoldSetReviewerQueue,
    GoldSetReviewerQueueStatus,
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)
from app.models.label_job import LabelJob, LabelRow
from app.models.project import Project
from app.schemas.domain_pack import DomainPackContract
from app.schemas.domain_profile import DomainProfileContract
from app.services.dataset_service import (
    preview_project_data_adapter,
    resolve_project_dataset_adapter_preference,
)
from app.services.data_adapter_service import resolve_data_adapter_contract
from app.services.domain_pack_service import create_domain_pack, get_domain_pack
from app.services.domain_profile_service import create_domain_profile, get_domain_profile
from app.services.domain_runtime_service import resolve_project_domain_runtime
from app.services.recipe_service import get_recipe
from app.services.synth_backends import BACKEND_REGISTRY
from app.services.synth_playbook_service import available_playbooks_for_recipe
from app.services.synth_playbooks import list_playbooks
from app.services.synthetic_service import call_teacher_model
from app.services.synth_review_queue_service import list_review_queue


IssueSeverity = Literal["blocker", "warning", "info"]
OverviewVerdict = Literal["blocked", "needs_work", "ready"]
SourcesVerdict = Literal["empty", "attention", "healthy"]
MappingVerdict = Literal["empty", "attention", "ready"]
DomainVerdict = Literal["unknown", "attention", "confirmed"]
GoldSetVerdict = Literal["empty", "attention", "ready"]
SyntheticPlaybookVerdict = Literal["empty", "attention", "ready"]
SyntheticRecommendationVerdict = Literal["empty", "attention", "ready"]
SyntheticQualityVerdict = Literal["empty", "attention", "ready"]
ReviewQueueVerdict = Literal["empty", "attention", "ready"]
PrepareDatasetVerdict = Literal["blocked", "attention", "ready"]
DatasetVersionVerdict = Literal["empty", "attention", "ready"]
QualitySafetyVerdict = Literal["blocked", "attention", "ready"]
CoachVerdict = Literal["blocked", "attention", "ready"]
AssistFocus = Literal["mapping", "domain"]
AssistProvider = Literal["ollama", "openai_compatible"]

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


_DOMAIN_SETUP_MIN_CONFIDENCE = 0.45


def _domain_setup_slug(value: Any) -> str:
    token = str(value or "").strip().lower().replace("_", "-")
    token = re.sub(r"[^a-z0-9-]+", "-", token)
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or "detected-domain"


def _clone_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _policy_qa_profile_payload(profile_id: str) -> dict[str, Any]:
    return {
        "$schema": "slm.domain-profile/v1",
        "profile_id": profile_id,
        "version": "1.0.0",
        "display_name": "Policy Q&A Profile",
        "description": (
            "Draft profile for source-backed policy question answering with "
            "exceptions, unknown-answer behavior, and review gates."
        ),
        "owner": "workspace",
        "status": "draft",
        "tasks": [
            {
                "task_id": "policy-qa",
                "output_mode": "text",
                "required_fields": ["question", "answer"],
                "optional_fields": [
                    "context",
                    "policy_section",
                    "source",
                    "effective_date",
                    "exception_type",
                    "rationale",
                ],
            }
        ],
        "canonical_schema": {
            "required": ["input_text", "target_text"],
            "aliases": {
                "input_text": [
                    "question",
                    "prompt",
                    "policy_question",
                    "employee_question",
                    "customer_question",
                ],
                "target_text": ["answer", "response", "policy_answer", "expected_answer"],
                "context": [
                    "context",
                    "policy_text",
                    "section_text",
                    "source_excerpt",
                    "passage",
                ],
                "metadata": [
                    "policy_section",
                    "source",
                    "effective_date",
                    "exception_type",
                    "jurisdiction",
                    "audience",
                ],
            },
        },
        "normalization": {
            "trim_whitespace": True,
            "drop_empty_records": True,
            "dedupe": {"enabled": True, "method": "hash(input_text,target_text,context)"},
            "pii_redaction": {"enabled": False, "policy": "review_before_training"},
        },
        "data_quality": {
            "min_records": 300,
            "max_null_ratio": 0.08,
            "max_duplicate_ratio": 0.15,
            "required_coverage": {
                "input_text": 0.99,
                "target_text": 0.99,
                "context": 0.7,
            },
        },
        "dataset_split": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
            "stratify_by": ["policy_section", "exception_type"],
            "seed": 42,
            "leakage_checks": ["exact_text_overlap", "policy_section_overlap"],
        },
        "training_defaults": {
            "training_mode": "sft",
            "chat_template": "llama3",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.0002,
            "use_lora": True,
        },
        "evaluation": {
            "metrics": [
                {"metric_id": "exact_match", "weight": 0.2, "threshold": 0.5},
                {"metric_id": "f1", "weight": 0.25, "threshold": 0.68},
                {"metric_id": "source_coverage", "weight": 0.25, "threshold": 0.75},
                {"metric_id": "unknown_answer_pass_rate", "weight": 0.15, "threshold": 0.8},
                {"metric_id": "safety_pass_rate", "weight": 0.15, "threshold": 0.92},
            ],
            "required_metrics_for_promotion": [
                "f1",
                "source_coverage",
                "safety_pass_rate",
            ],
        },
        "tools": {
            "retrieval": {"enabled": False, "adapter": None},
            "function_calling": {"enabled": False, "adapter": None},
            "required_secrets": [],
        },
        "registry_gates": {
            "to_staging": {"min_metrics": {"f1": 0.68, "source_coverage": 0.75}},
            "to_production": {
                "min_metrics": {
                    "f1": 0.72,
                    "source_coverage": 0.82,
                    "safety_pass_rate": 0.94,
                },
                "max_regression_vs_prod": {"f1": 0.03},
            },
        },
        "audit": {
            "require_human_approval_for_production": True,
            "notes_required_on_force_promotion": True,
        },
    }


def _pii_pci_profile_payload(profile_id: str) -> dict[str, Any]:
    return {
        "$schema": "slm.domain-profile/v1",
        "profile_id": profile_id,
        "version": "1.0.0",
        "display_name": "PII/PCI Detection Profile",
        "description": (
            "Draft profile for sensitive-data detection, masking review, "
            "and false-negative focused evaluation."
        ),
        "owner": "workspace",
        "status": "draft",
        "tasks": [
            {
                "task_id": "pii-pci-detection",
                "output_mode": "label",
                "required_fields": ["text", "label"],
                "optional_fields": ["entity", "category", "span", "redacted_text", "rationale"],
            }
        ],
        "canonical_schema": {
            "required": ["input_text", "target_text"],
            "aliases": {
                "input_text": ["text", "input", "message", "document", "raw_text"],
                "target_text": ["label", "category", "entity", "pii_type", "pci_type"],
                "context": ["span", "surrounding_text", "example_context"],
                "metadata": ["redacted_text", "policy", "source", "risk_level"],
            },
        },
        "normalization": {
            "trim_whitespace": True,
            "drop_empty_records": True,
            "dedupe": {"enabled": True, "method": "hash(input_text,target_text)"},
            "pii_redaction": {"enabled": True, "policy": "mask_training_values"},
        },
        "data_quality": {
            "min_records": 500,
            "max_null_ratio": 0.04,
            "max_duplicate_ratio": 0.12,
            "required_coverage": {"input_text": 0.99, "target_text": 0.99},
        },
        "dataset_split": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
            "stratify_by": ["label", "entity"],
            "seed": 42,
            "leakage_checks": ["exact_text_overlap", "sensitive_value_overlap"],
        },
        "training_defaults": {
            "training_mode": "sft",
            "chat_template": "llama3",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.0002,
            "use_lora": True,
        },
        "evaluation": {
            "metrics": [
                {"metric_id": "precision", "weight": 0.22, "threshold": 0.9},
                {"metric_id": "recall", "weight": 0.3, "threshold": 0.92},
                {"metric_id": "f1", "weight": 0.24, "threshold": 0.9},
                {"metric_id": "false_negative_rate", "weight": 0.14, "threshold": 0.04},
                {"metric_id": "safety_pass_rate", "weight": 0.1, "threshold": 0.95},
            ],
            "required_metrics_for_promotion": ["recall", "f1", "safety_pass_rate"],
        },
        "tools": {
            "retrieval": {"enabled": False, "adapter": None},
            "function_calling": {"enabled": False, "adapter": None},
            "required_secrets": [],
        },
        "registry_gates": {
            "to_staging": {"min_metrics": {"recall": 0.9, "f1": 0.88}},
            "to_production": {
                "min_metrics": {"recall": 0.94, "f1": 0.9, "safety_pass_rate": 0.96},
                "max_regression_vs_prod": {"recall": 0.02, "f1": 0.02},
            },
        },
        "audit": {
            "require_human_approval_for_production": True,
            "notes_required_on_force_promotion": True,
        },
    }


def _generic_domain_profile_payload(candidate: dict[str, Any], profile_id: str) -> dict[str, Any]:
    domain_id = str(candidate.get("id") or "detected_domain")
    label = str(candidate.get("label") or "Detected Domain")
    recipes = {str(item) for item in candidate.get("recommended_recipes") or []}
    matched_fields = [
        str(field)
        for field in list(candidate.get("matched_fields") or [])[:8]
        if str(field).strip()
    ]
    classification_like = "classification" in recipes or "span-extraction" in recipes
    if classification_like:
        task_id = f"{_domain_setup_slug(domain_id)}-classification"
        output_mode = "label"
        required_fields = ["input", "label"]
        optional_fields = ["category", "rationale", "source", "difficulty"]
        metrics = [
            {"metric_id": "accuracy", "weight": 0.25, "threshold": 0.75},
            {"metric_id": "macro_f1", "weight": 0.35, "threshold": 0.7},
            {"metric_id": "class_balance", "weight": 0.15, "threshold": 0.8},
            {"metric_id": "safety_pass_rate", "weight": 0.25, "threshold": 0.9},
        ]
        required_metrics = ["macro_f1", "safety_pass_rate"]
        coverage = {"input_text": 0.99, "target_text": 0.98}
        stratify_by = ["label", "category"]
    else:
        task_id = f"{_domain_setup_slug(domain_id)}-qa"
        output_mode = "text"
        required_fields = ["question", "answer"]
        optional_fields = ["context", "source", "rationale", "difficulty"]
        metrics = [
            {"metric_id": "exact_match", "weight": 0.2, "threshold": 0.5},
            {"metric_id": "f1", "weight": 0.32, "threshold": 0.65},
            {"metric_id": "llm_judge_pass_rate", "weight": 0.28, "threshold": 0.75},
            {"metric_id": "safety_pass_rate", "weight": 0.2, "threshold": 0.9},
        ]
        required_metrics = ["f1", "llm_judge_pass_rate", "safety_pass_rate"]
        coverage = {"input_text": 0.99, "target_text": 0.99}
        stratify_by = ["source", "difficulty"]

    return {
        "$schema": "slm.domain-profile/v1",
        "profile_id": profile_id,
        "version": "1.0.0",
        "display_name": f"{label} Profile",
        "description": f"Draft profile generated from Data Studio detection for {label} projects.",
        "owner": "workspace",
        "status": "draft",
        "tasks": [
            {
                "task_id": task_id,
                "output_mode": output_mode,
                "required_fields": required_fields,
                "optional_fields": optional_fields,
            }
        ],
        "canonical_schema": {
            "required": ["input_text", "target_text"],
            "aliases": {
                "input_text": ["question", "prompt", "input", "text", "message"],
                "target_text": ["answer", "response", "output", "label", "completion"],
                "context": ["context", "passage", "document", "source_excerpt"],
                "metadata": ["source", "category", "labels", *matched_fields],
            },
        },
        "normalization": {
            "trim_whitespace": True,
            "drop_empty_records": True,
            "dedupe": {"enabled": True, "method": "hash(input_text,target_text)"},
            "pii_redaction": {"enabled": False, "policy": "review_before_training"},
        },
        "data_quality": {
            "min_records": 300,
            "max_null_ratio": 0.08,
            "max_duplicate_ratio": 0.15,
            "required_coverage": coverage,
        },
        "dataset_split": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
            "stratify_by": stratify_by,
            "seed": 42,
            "leakage_checks": ["exact_text_overlap"],
        },
        "training_defaults": {
            "training_mode": "sft",
            "chat_template": "llama3",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.0002,
            "use_lora": True,
        },
        "evaluation": {
            "metrics": metrics,
            "required_metrics_for_promotion": required_metrics,
        },
        "tools": {
            "retrieval": {"enabled": False, "adapter": None},
            "function_calling": {"enabled": False, "adapter": None},
            "required_secrets": [],
        },
        "registry_gates": {
            "to_staging": {"min_metrics": {"f1": 0.65}},
            "to_production": {
                "min_metrics": {"f1": 0.7, "safety_pass_rate": 0.92},
                "max_regression_vs_prod": {"f1": 0.03},
            },
        },
        "audit": {
            "require_human_approval_for_production": True,
            "notes_required_on_force_promotion": True,
        },
    }


def _domain_setup_profile_payload(candidate: dict[str, Any], profile_id: str) -> dict[str, Any]:
    domain_id = str(candidate.get("id") or "")
    if domain_id == "policy_qa":
        return _policy_qa_profile_payload(profile_id)
    if domain_id == "pii_pci_detection":
        return _pii_pci_profile_payload(profile_id)
    return _generic_domain_profile_payload(candidate, profile_id)


def _domain_setup_pack_payload(
    candidate: dict[str, Any],
    *,
    pack_id: str,
    profile_payload: dict[str, Any],
) -> dict[str, Any]:
    label = str(candidate.get("label") or "Detected Domain")
    slug = _domain_setup_slug(candidate.get("id") or label)
    return {
        "$schema": "slm.domain-pack/v1",
        "pack_id": pack_id,
        "version": "1.0.0",
        "display_name": f"{label} Pack",
        "description": (
            f"Draft domain pack generated from Data Studio detection for {label}. "
            "Review before assigning to the project."
        ),
        "owner": "workspace",
        "status": "draft",
        "default_profile_id": profile_payload.get("profile_id"),
        "tags": [slug, "data-studio", "domain-detection"],
        "hooks": {
            "normalizer": {"id": "default-normalizer", "config": {}},
            "validator": {"id": "default-validator", "config": {}},
            "evaluator": {"id": "default-evaluator", "config": {}},
        },
        "overlay": {
            "dataset_split": _clone_jsonable(profile_payload.get("dataset_split") or {}),
            "training_defaults": _clone_jsonable(profile_payload.get("training_defaults") or {}),
            "data_quality": _clone_jsonable(profile_payload.get("data_quality") or {}),
            "normalization": _clone_jsonable(profile_payload.get("normalization") or {}),
            "tools": _clone_jsonable(profile_payload.get("tools") or {}),
            "evaluation": _clone_jsonable(profile_payload.get("evaluation") or {}),
            "registry_gates": _clone_jsonable(profile_payload.get("registry_gates") or {}),
            "audit": _clone_jsonable(profile_payload.get("audit") or {}),
        },
    }


def _domain_setup_contract_payloads(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    slug = _domain_setup_slug(candidate.get("id") or candidate.get("label"))
    profile_id = f"{slug}-profile-v1"
    pack_id = f"{slug}-pack-v1"
    profile_contract = DomainProfileContract.model_validate(
        _domain_setup_profile_payload(candidate, profile_id)
    )
    profile_payload = profile_contract.model_dump(by_alias=True, exclude_none=True)
    pack_contract = DomainPackContract.model_validate(
        _domain_setup_pack_payload(
            candidate,
            pack_id=pack_id,
            profile_payload=profile_payload,
        )
    )
    pack_payload = pack_contract.model_dump(by_alias=True, exclude_none=True)
    return profile_payload, pack_payload


def _domain_setup_guidance(candidate: dict[str, Any]) -> list[dict[str, str]]:
    domain_id = str(candidate.get("id") or "")
    if domain_id == "policy_qa":
        return [
            {
                "id": "task_shape",
                "title": "Task shape",
                "recommendation": "Use Policy Q&A with question, answer, context, and policy-section metadata.",
                "why": "Policy assistants need grounded answers and enough metadata to catch stale or exception-heavy rules.",
            },
            {
                "id": "unknowns",
                "title": "Unknown-answer behavior",
                "recommendation": "Include examples where the policy does not answer the question.",
                "why": "This reduces confident answers when the handbook or policy source is incomplete.",
            },
            {
                "id": "source_gates",
                "title": "Evaluation gates",
                "recommendation": "Track F1, source coverage, unknown-answer pass rate, and safety pass rate.",
                "why": "Good policy models should be correct, source-backed, and cautious around ambiguity.",
            },
            {
                "id": "activation",
                "title": "Activation",
                "recommendation": "Create drafts, review them, then assign the pack/profile from the Domain managers.",
                "why": "Data Studio previews the setup but does not switch project defaults without you.",
            },
        ]
    if domain_id == "pii_pci_detection":
        return [
            {
                "id": "task_shape",
                "title": "Task shape",
                "recommendation": "Use text plus sensitive-label/entity fields with redacted examples where possible.",
                "why": "PII/PCI training benefits from explicit classes and safe handling of sensitive values.",
            },
            {
                "id": "false_negatives",
                "title": "Risk focus",
                "recommendation": "Gate promotion on recall, F1, and safety pass rate.",
                "why": "False negatives can leak sensitive data, so recall needs more weight than broad accuracy.",
            },
            {
                "id": "masking",
                "title": "Normalization",
                "recommendation": "Enable masking before training and review raw production examples carefully.",
                "why": "The model should learn patterns without storing real account or card data.",
            },
            {
                "id": "activation",
                "title": "Activation",
                "recommendation": "Create drafts, review them, then assign the pack/profile from the Domain managers.",
                "why": "Data Studio previews the setup but does not switch project defaults without you.",
            },
        ]
    label = str(candidate.get("label") or "this domain")
    return [
        {
            "id": "task_shape",
            "title": "Task shape",
            "recommendation": f"Use a {label} profile aligned to the detected recipe and fields.",
            "why": "Domain profiles make required fields, splits, metrics, and review gates explicit.",
        },
        {
            "id": "coverage",
            "title": "Coverage",
            "recommendation": "Review required-field coverage, label balance, and representative edge cases.",
            "why": "A domain setup is most useful when it encodes the examples the model must handle reliably.",
        },
        {
            "id": "evaluation",
            "title": "Evaluation",
            "recommendation": "Set promotion gates before training so wins and regressions are measurable.",
            "why": "Power users can tune metrics in the draft profile before assigning it.",
        },
        {
            "id": "activation",
            "title": "Activation",
            "recommendation": "Create drafts, review them, then assign the pack/profile from the Domain managers.",
            "why": "Data Studio previews the setup but does not switch project defaults without you.",
        },
    ]


async def _domain_setup_preview(
    db: AsyncSession,
    *,
    detected: dict[str, Any],
    runtime_is_generic: bool,
    runtime_matches: bool,
) -> dict[str, Any] | None:
    domain_id = str(detected.get("id") or "")
    confidence = float(detected.get("confidence") or 0.0)
    if domain_id == "generic_domain" or confidence < _DOMAIN_SETUP_MIN_CONFIDENCE:
        return None

    profile_payload, pack_payload = _domain_setup_contract_payloads(detected)
    profile_id = str(profile_payload.get("profile_id") or "")
    pack_id = str(pack_payload.get("pack_id") or "")
    profile = await get_domain_profile(db, profile_id)
    pack = await get_domain_pack(db, pack_id)
    profile_exists = profile is not None
    pack_exists = pack is not None
    label = str(detected.get("label") or "the detected domain")

    if profile_exists and pack_exists:
        reason = (
            f"BrewSLM already has a {label} profile and pack draft. Review "
            "or assign them from the Domain managers."
        )
    elif runtime_is_generic:
        reason = (
            f"Sampled rows look like {label}, but the project is still using "
            "generic domain defaults."
        )
    elif not runtime_matches:
        reason = (
            f"Sampled rows look like {label}, but the applied domain setup "
            "appears to point elsewhere."
        )
    else:
        reason = f"BrewSLM can prepare a draft {label} domain setup for review."

    return {
        "available": True,
        "recommended": bool(runtime_is_generic or not runtime_matches),
        "reason": reason,
        "read_only": True,
        "requires_confirmation": True,
        "create_mode": "create_missing_drafts",
        "detected_domain_id": domain_id,
        "detected_domain_label": label,
        "profile_id": profile_id,
        "pack_id": pack_id,
        "profile_exists": profile_exists,
        "pack_exists": pack_exists,
        "profile_status": getattr(getattr(profile, "status", None), "value", None) if profile else None,
        "pack_status": getattr(getattr(pack, "status", None), "value", None) if pack else None,
        "can_create_profile": not profile_exists,
        "can_create_pack": not pack_exists,
        "guidance": _domain_setup_guidance(detected),
        "choices": [
            {
                "id": "use_existing",
                "label": "Use existing setup",
                "target": "domain",
                "detail": "Open the Domain controls to assign or inspect an existing profile and pack.",
            },
            {
                "id": "create_drafts",
                "label": "Create draft setup",
                "target": "create_drafts",
                "detail": "Create only the missing draft profile/pack records; do not assign them automatically.",
            },
            {
                "id": "power_user_managers",
                "label": "Open pack/profile managers",
                "target": "domain-packs",
                "detail": "Review the full JSON contracts, hook defaults, overlays, and promotion gates.",
            },
        ],
        "profile_contract": profile_payload,
        "pack_contract": pack_payload,
    }


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

    domain_setup = await _domain_setup_preview(
        db,
        detected=detected,
        runtime_is_generic=runtime_is_generic,
        runtime_matches=runtime_matches,
    )

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
        "domain_setup": domain_setup,
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


async def create_data_studio_domain_setup_from_detection(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Create missing draft domain profile/pack records from detection preview."""

    detection = await build_data_studio_domain_detection(db, project_id)
    setup = detection.get("domain_setup")
    if not isinstance(setup, dict) or not setup.get("available"):
        raise ValueError("No domain setup draft is available for the current detection.")

    profile_payload = setup.get("profile_contract")
    pack_payload = setup.get("pack_contract")
    if not isinstance(profile_payload, dict) or not isinstance(pack_payload, dict):
        raise ValueError("Domain setup draft is missing profile or pack contracts.")

    profile_contract = DomainProfileContract.model_validate(profile_payload)
    pack_contract = DomainPackContract.model_validate(pack_payload)

    existing_profile = await get_domain_profile(db, profile_contract.profile_id)
    existing_pack = await get_domain_pack(db, pack_contract.pack_id)

    created_profile = False
    created_pack = False
    if existing_profile is None:
        existing_profile = await create_domain_profile(db, profile_contract)
        created_profile = True
    if existing_pack is None:
        existing_pack = await create_domain_pack(db, pack_contract)
        created_pack = True

    return {
        "status": "created" if created_profile or created_pack else "already_exists",
        "project_id": project_id,
        "detected_domain_id": setup.get("detected_domain_id"),
        "detected_domain_label": setup.get("detected_domain_label"),
        "created_profile": created_profile,
        "created_pack": created_pack,
        "assigned_to_project": False,
        "profile": {
            "profile_id": existing_profile.profile_id,
            "display_name": existing_profile.display_name,
            "status": getattr(existing_profile.status, "value", existing_profile.status),
            "version": existing_profile.version,
        },
        "pack": {
            "pack_id": existing_pack.pack_id,
            "display_name": existing_pack.display_name,
            "status": getattr(existing_pack.status, "value", existing_pack.status),
            "version": existing_pack.version,
            "default_profile_id": existing_pack.default_profile_id,
        },
        "next_targets": ["domain", "domain-packs", "domain-profiles"],
    }


_GOLD_SET_DATASET_TYPES = {DatasetType.GOLD_DEV, DatasetType.GOLD_TEST}
_GOLD_SET_MIN_STARTER_ROWS = 5


def _enum_value(value: Any) -> str:
    token = getattr(value, "value", value)
    return str(token or "").strip().lower()


def _load_jsonl_dicts(path: str | None, *, limit: int = 5000) -> list[dict[str, Any]]:
    token = str(path or "").strip()
    if not token:
        return []
    file_path = Path(token)
    if not file_path.exists() or not file_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= limit:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _field_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0
    return True


def _field_counter(payloads: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            field = str(key or "").strip()
            if field and _field_has_value(value):
                counter[field] += 1
    return counter


_QUALITY_SCAN_SOURCE_TYPES: tuple[DatasetType, ...] = (
    DatasetType.RAW,
    DatasetType.CLEANED,
    DatasetType.GOLD_DEV,
    DatasetType.GOLD_TEST,
    DatasetType.SYNTHETIC,
    DatasetType.TRAIN,
    DatasetType.VALIDATION,
    DatasetType.TEST,
)

_PII_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PII_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_PII_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PCI_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_PCI_CVV_RE = re.compile(r"\b(?:cvv|cvc|security code)\s*[:#-]?\s*\d{3,4}\b", re.IGNORECASE)
_LOW_QUALITY_PLACEHOLDERS = {
    "n/a",
    "na",
    "none",
    "null",
    "todo",
    "tbd",
    "test",
    "lorem ipsum",
}


def _quality_scan_text(row: dict[str, Any]) -> str:
    values: list[str] = []
    _flatten_text_values(row, values, limit=120)
    return " ".join(values).strip()


def _quality_text_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())[:4000]


def _quality_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token
    }


def _luhn_valid(digits: str) -> bool:
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    reverse_digits = list(map(int, reversed(digits)))
    for index, digit in enumerate(reverse_digits):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _pii_pci_signal_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if _PII_EMAIL_RE.search(text):
        counts["email"] += 1
    if _PII_PHONE_RE.search(text):
        counts["phone"] += 1
    if _PII_SSN_RE.search(text):
        counts["ssn"] += 1
    if _PCI_CVV_RE.search(text):
        counts["cvv"] += 1
    for match in _PCI_CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if _luhn_valid(digits):
            counts["credit_card"] += 1
            break
    return counts


def _quality_redact_text(text: Any, *, limit: int = 260) -> str:
    value = str(text or "")
    value = _PII_EMAIL_RE.sub("[EMAIL]", value)
    value = _PII_SSN_RE.sub("[SSN]", value)
    value = _PCI_CVV_RE.sub("[CVV]", value)

    def _redact_card(match: re.Match[str]) -> str:
        token = match.group(0)
        digits = re.sub(r"\D", "", token)
        return "[CARD]" if _luhn_valid(digits) else token

    value = _PCI_CARD_RE.sub(_redact_card, value)
    value = _PII_PHONE_RE.sub("[PHONE]", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        return f"{value[:limit].rstrip()}..."
    return value


def _quality_redacted_field_value(field: str, value: Any) -> str:
    lowered = field.lower()
    if any(token in lowered for token in ("password", "secret", "token", "ssn", "credit", "card", "cvv", "cvc")):
        return "[REDACTED]"
    if isinstance(value, (dict, list, tuple, set)):
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            serialized = str(value)
    else:
        serialized = "" if value is None else str(value)
    redacted = _quality_redact_text(serialized, limit=140)
    return redacted if redacted else "(empty)"


def _quality_row_preview(
    item: dict[str, Any],
    *,
    reason: str = "",
) -> dict[str, Any]:
    row = item.get("row") if isinstance(item.get("row"), dict) else {}
    file_path = str(item.get("file_path") or "").strip()
    fields: list[dict[str, str]] = []
    for key, value in list(row.items())[:8]:
        field = str(key or "").strip()
        if not field:
            continue
        fields.append({
            "field": field,
            "value": _quality_redacted_field_value(field, value),
        })
        if len(fields) >= 5:
            break
    return {
        "source": str(item.get("source") or "Project sample"),
        "source_type": str(item.get("source_type") or "sample"),
        "target_tab": str(item.get("target_tab") or "data"),
        "row_index": int(item.get("row_index") or 0),
        "file_name": Path(file_path).name if file_path else None,
        "redacted_text": _quality_redact_text(_quality_scan_text(row)),
        "fields": fields,
        "reason": reason,
    }


def _quality_source_count_rows(
    counts: Counter[str] | None,
    *,
    fallback_source: str,
    fallback_count: int,
    target_tab: str,
) -> list[dict[str, Any]]:
    source_counts = Counter(counts or {})
    if not source_counts and fallback_source:
        source_counts[fallback_source] = max(0, int(fallback_count or 0))
    rows = [
        {
            "source": str(source),
            "count": int(count),
            "target_tab": target_tab,
        }
        for source, count in source_counts.items()
        if int(count or 0) > 0
    ]
    rows.sort(key=lambda item: (-int(item["count"]), str(item["source"])))
    return rows[:8]


def _quality_check_drilldown(
    check: dict[str, Any],
    *,
    rows: list[dict[str, Any]] | None = None,
    source_counts: Counter[str] | None = None,
    empty_message: str | None = None,
) -> dict[str, Any]:
    target_tab = str(check.get("target_tab") or "data")
    action_label = str(check.get("action_label") or "Open workflow")
    count = int(check.get("count") or 0)
    preview_rows = rows or []
    if not source_counts and preview_rows:
        source_counts = Counter(str(item.get("source") or "Project sample") for item in preview_rows)
    source_count_rows = _quality_source_count_rows(
        source_counts,
        fallback_source=str(check.get("source") or "Project sample"),
        fallback_count=count,
        target_tab=target_tab,
    )
    return {
        "read_only": True,
        "redacted": True,
        "total_affected": count,
        "source_counts": source_count_rows,
        "rows": [
            _quality_row_preview(item, reason=str(check.get("label") or "Quality and safety finding"))
            for item in preview_rows[:5]
        ],
        "action": {
            "label": action_label,
            "target_tab": target_tab,
            "workflow_owner": str(check.get("workflow_owner") or "Data Studio"),
            "requires_confirmation": True,
            "description": (
                f"Open {check.get('workflow_owner') or 'the destination workflow'} for "
                f"'{action_label}'. Data Studio only previews this finding."
            ),
        },
        "empty_message": empty_message or "No affected row sample is available for this check yet.",
    }


def _low_quality_reason(text: str) -> str | None:
    normalized = _quality_text_fingerprint(text)
    if not normalized:
        return "empty"
    if normalized in _LOW_QUALITY_PLACEHOLDERS:
        return "placeholder"
    if len(normalized) < 12:
        return "too_short"
    words = re.findall(r"[A-Za-z0-9]+", normalized)
    if len(words) < 3:
        return "too_few_words"
    if re.search(r"([!?.,])\1{5,}", normalized):
        return "repeated_punctuation"
    if len(set(normalized.replace(" ", ""))) <= 3 and len(normalized) > 10:
        return "repeated_characters"
    return None


def _quality_required_missing_count(
    row: dict[str, Any],
    required_fields: list[str],
) -> int:
    missing = 0
    for field in required_fields:
        value = row.get(field)
        if not _field_has_value(value):
            missing += 1
    return missing


async def _quality_source_rows(
    db: AsyncSession,
    project_id: int,
    *,
    limit: int = 500,
) -> list[dict[str, Any]]:
    datasets_result = await db.execute(
        select(Dataset)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(list(_QUALITY_SCAN_SOURCE_TYPES)),
        )
        .order_by(Dataset.updated_at.desc(), Dataset.id.asc())
    )
    datasets = list(datasets_result.scalars().all())
    datasets_by_id = {int(dataset.id): dataset for dataset in datasets}

    raw_docs_result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            RawDocument.status == DocumentStatus.ACCEPTED,
        )
        .order_by(RawDocument.ingested_at.desc(), RawDocument.id.asc())
    )
    raw_docs = list(raw_docs_result.scalars().all())

    scan_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def add_rows(
        rows: list[dict[str, Any]],
        *,
        source: str,
        source_type: str,
        target_tab: str,
        file_path: str,
    ) -> None:
        for row_index, row in enumerate(rows):
            if len(scan_rows) >= limit:
                return
            scan_rows.append({
                "row": row,
                "source": source,
                "source_type": source_type,
                "target_tab": target_tab,
                "file_path": file_path,
                "row_index": row_index,
            })

    for doc in raw_docs:
        token = str(doc.file_path or "").strip()
        if not token or token in seen_paths:
            continue
        seen_paths.add(token)
        dataset = datasets_by_id.get(int(doc.dataset_id))
        dataset_type = dataset.dataset_type.value if dataset is not None else DatasetType.RAW.value
        add_rows(
            _load_jsonl_dicts(token, limit=max(1, limit - len(scan_rows))),
            source=doc.filename or "Raw document",
            source_type=dataset_type,
            target_tab="data",
            file_path=token,
        )
        if len(scan_rows) >= limit:
            return scan_rows

    target_by_type = {
        DatasetType.RAW: "data",
        DatasetType.CLEANED: "data",
        DatasetType.GOLD_DEV: "goldset",
        DatasetType.GOLD_TEST: "goldset",
        DatasetType.SYNTHETIC: "synthetic",
        DatasetType.TRAIN: "dataprep",
        DatasetType.VALIDATION: "dataprep",
        DatasetType.TEST: "dataprep",
    }
    for dataset in datasets:
        token = str(dataset.file_path or "").strip()
        if not token or token in seen_paths:
            continue
        seen_paths.add(token)
        add_rows(
            _load_jsonl_dicts(token, limit=max(1, limit - len(scan_rows))),
            source=dataset.name or dataset.dataset_type.value,
            source_type=dataset.dataset_type.value,
            target_tab=target_by_type.get(dataset.dataset_type, "data"),
            file_path=token,
        )
        if len(scan_rows) >= limit:
            return scan_rows
    return scan_rows


def _coverage_from_counter(
    counter: Counter[str],
    *,
    total: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if total <= 0:
        return []
    rows = [
        {
            "field": field,
            "present": int(count),
            "missing": max(0, int(total) - int(count)),
            "ratio": round(min(1.0, max(0.0, float(count) / float(total))), 4),
        }
        for field, count in counter.items()
    ]
    rows.sort(key=lambda item: (-float(item["ratio"]), str(item["field"])))
    return rows[:limit]


def _gold_coverage_payload(
    *,
    inputs: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    total = max(len(inputs), len(expected), len(labels))
    input_counter = _field_counter(inputs)
    expected_counter = _field_counter(expected)
    label_counter = _field_counter(labels)
    return {
        "source_rows": total,
        "input_fields": _coverage_from_counter(input_counter, total=total),
        "expected_fields": _coverage_from_counter(expected_counter, total=total),
        "label_fields": _coverage_from_counter(label_counter, total=total),
        "field_counts": {
            "input": len(input_counter),
            "expected": len(expected_counter),
            "labels": len(label_counter),
        },
    }


def _legacy_gold_payloads(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for entry in entries:
        inputs.append({"question": entry.get("question")})
        expected.append({"answer": entry.get("answer")})
        labels.append({
            "difficulty": entry.get("difficulty"),
            "criticality": entry.get("criticality"),
            "is_hallucination_trap": entry.get("is_hallucination_trap"),
        })
    return inputs, expected, labels


def _snippet(value: Any, *, limit: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, default=str)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def _gold_row_status_counts(rows: list[GoldSetRow]) -> dict[str, int]:
    counts = {status.value: 0 for status in GoldSetRowStatus}
    for row in rows:
        key = _enum_value(row.status) or GoldSetRowStatus.PENDING.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _gold_queue_status_counts(entries: list[GoldSetReviewerQueue]) -> dict[str, int]:
    counts = {status.value: 0 for status in GoldSetReviewerQueueStatus}
    for entry in entries:
        key = _enum_value(entry.status) or GoldSetReviewerQueueStatus.PENDING.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _gold_version_payload(version: GoldSetVersion | None) -> dict[str, Any] | None:
    if version is None:
        return None
    return {
        "id": int(version.id),
        "version": int(version.version),
        "status": _enum_value(version.status),
        "locked_at": version.locked_at.isoformat() if version.locked_at else None,
        "created_at": version.created_at.isoformat() if version.created_at else None,
    }


def _gold_versions_summary(versions: list[GoldSetVersion]) -> dict[str, Any]:
    if not versions:
        return {
            "count": 0,
            "draft_count": 0,
            "locked_count": 0,
            "latest": None,
            "active_draft": None,
            "latest_locked": None,
        }
    ordered = sorted(versions, key=lambda item: (int(item.version), int(item.id)))
    drafts = [
        version for version in ordered
        if _enum_value(version.status) == GoldSetVersionStatus.DRAFT.value
    ]
    locked = [
        version for version in ordered
        if _enum_value(version.status) == GoldSetVersionStatus.LOCKED.value
    ]
    return {
        "count": len(ordered),
        "draft_count": len(drafts),
        "locked_count": len(locked),
        "latest": _gold_version_payload(ordered[-1]),
        "active_draft": _gold_version_payload(drafts[-1] if drafts else None),
        "latest_locked": _gold_version_payload(locked[-1] if locked else None),
    }


def _gold_validation_status(
    *,
    example_count: int,
    trusted_examples: int,
    review_needed: int,
    has_locked_state: bool,
) -> str:
    if example_count <= 0:
        return "empty"
    if review_needed > 0:
        return "needs_review"
    if trusted_examples < _GOLD_SET_MIN_STARTER_ROWS:
        return "thin"
    if has_locked_state:
        return "locked"
    return "ready"


def _trusted_gold_samples(
    *,
    dataset: Dataset,
    rows: list[GoldSetRow],
    legacy_entries: list[dict[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    approved_rows = [
        row for row in rows
        if _enum_value(row.status) == GoldSetRowStatus.APPROVED.value
    ]
    for row in approved_rows[:limit]:
        samples.append({
            "dataset_id": int(dataset.id),
            "dataset_name": dataset.name,
            "source": "workbench_row",
            "status": _enum_value(row.status),
            "input_preview": _snippet(row.input),
            "expected_preview": _snippet(row.expected),
        })
    if samples:
        return samples
    for entry in legacy_entries[:limit]:
        samples.append({
            "dataset_id": int(dataset.id),
            "dataset_name": dataset.name,
            "source": "gold_dataset_entry",
            "status": "trusted",
            "input_preview": _snippet(entry.get("question")),
            "expected_preview": _snippet(entry.get("answer")),
        })
    return samples


async def build_data_studio_gold_set_workbench(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a read-only Gold Set workbench summary for Data Studio."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    datasets_result = await db.execute(
        select(Dataset)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(_GOLD_SET_DATASET_TYPES),
        )
        .order_by(Dataset.dataset_type.asc(), Dataset.id.asc())
    )
    gold_datasets = list(datasets_result.scalars().all())
    gold_ids = [int(dataset.id) for dataset in gold_datasets]

    rows_by_gold: dict[int, list[GoldSetRow]] = {gold_id: [] for gold_id in gold_ids}
    versions_by_gold: dict[int, list[GoldSetVersion]] = {gold_id: [] for gold_id in gold_ids}
    queue_by_gold: dict[int, list[GoldSetReviewerQueue]] = {gold_id: [] for gold_id in gold_ids}

    if gold_ids:
        rows_result = await db.execute(
            select(GoldSetRow)
            .where(GoldSetRow.gold_set_id.in_(gold_ids))
            .order_by(GoldSetRow.gold_set_id.asc(), GoldSetRow.id.asc())
        )
        for row in rows_result.scalars().all():
            rows_by_gold.setdefault(int(row.gold_set_id), []).append(row)

        versions_result = await db.execute(
            select(GoldSetVersion)
            .where(GoldSetVersion.gold_set_id.in_(gold_ids))
            .order_by(GoldSetVersion.gold_set_id.asc(), GoldSetVersion.version.asc())
        )
        for version in versions_result.scalars().all():
            versions_by_gold.setdefault(int(version.gold_set_id), []).append(version)

        queue_result = await db.execute(
            select(GoldSetReviewerQueue)
            .where(GoldSetReviewerQueue.gold_set_id.in_(gold_ids))
            .order_by(GoldSetReviewerQueue.gold_set_id.asc(), GoldSetReviewerQueue.id.asc())
        )
        for entry in queue_result.scalars().all():
            queue_by_gold.setdefault(int(entry.gold_set_id), []).append(entry)

    dataset_payloads: list[dict[str, Any]] = []
    trusted_samples: list[dict[str, Any]] = []
    aggregate_inputs: list[dict[str, Any]] = []
    aggregate_expected: list[dict[str, Any]] = []
    aggregate_labels: list[dict[str, Any]] = []
    totals = {
        "gold_set_count": len(gold_datasets),
        "example_count": 0,
        "trusted_examples": 0,
        "review_needed": 0,
        "approved_rows": 0,
        "pending_rows": 0,
        "in_review_rows": 0,
        "changes_requested_rows": 0,
        "rejected_rows": 0,
        "queue_pending": 0,
        "queue_in_progress": 0,
        "locked_gold_sets": 0,
        "draft_versions": 0,
        "locked_versions": 0,
    }

    for dataset in gold_datasets:
        gold_id = int(dataset.id)
        rows = rows_by_gold.get(gold_id, [])
        versions = versions_by_gold.get(gold_id, [])
        queue_entries = queue_by_gold.get(gold_id, [])
        row_counts = _gold_row_status_counts(rows)
        queue_counts = _gold_queue_status_counts(queue_entries)
        version_summary = _gold_versions_summary(versions)
        legacy_entries = _load_jsonl_dicts(dataset.file_path)

        if rows:
            inputs = [row.input or {} for row in rows]
            expected = [row.expected or {} for row in rows]
            labels = [row.labels or {} for row in rows]
            coverage_source = "workbench_rows"
            example_count = len(rows)
        else:
            inputs, expected, labels = _legacy_gold_payloads(legacy_entries)
            coverage_source = "gold_dataset_entries" if legacy_entries else "dataset_record_count"
            example_count = len(legacy_entries) or int(dataset.record_count or 0)

        aggregate_inputs.extend(inputs)
        aggregate_expected.extend(expected)
        aggregate_labels.extend(labels)

        approved = int(row_counts.get(GoldSetRowStatus.APPROVED.value, 0))
        pending = int(row_counts.get(GoldSetRowStatus.PENDING.value, 0))
        in_review = int(row_counts.get(GoldSetRowStatus.IN_REVIEW.value, 0))
        changes_requested = int(row_counts.get(GoldSetRowStatus.CHANGES_REQUESTED.value, 0))
        rejected = int(row_counts.get(GoldSetRowStatus.REJECTED.value, 0))
        row_review_need = pending + in_review + changes_requested
        queue_review_need = int(queue_counts.get(GoldSetReviewerQueueStatus.PENDING.value, 0)) + int(
            queue_counts.get(GoldSetReviewerQueueStatus.IN_PROGRESS.value, 0)
        )
        review_needed = max(row_review_need, queue_review_need)
        trusted_examples = approved if rows else example_count
        has_locked_state = bool(dataset.is_locked) or int(version_summary["locked_count"]) > 0
        validation_status = _gold_validation_status(
            example_count=example_count,
            trusted_examples=trusted_examples,
            review_needed=review_needed,
            has_locked_state=has_locked_state,
        )

        totals["example_count"] += example_count
        totals["trusted_examples"] += trusted_examples
        totals["review_needed"] += review_needed
        totals["approved_rows"] += approved
        totals["pending_rows"] += pending
        totals["in_review_rows"] += in_review
        totals["changes_requested_rows"] += changes_requested
        totals["rejected_rows"] += rejected
        totals["queue_pending"] += int(queue_counts.get(GoldSetReviewerQueueStatus.PENDING.value, 0))
        totals["queue_in_progress"] += int(queue_counts.get(GoldSetReviewerQueueStatus.IN_PROGRESS.value, 0))
        totals["locked_gold_sets"] += 1 if bool(dataset.is_locked) else 0
        totals["draft_versions"] += int(version_summary["draft_count"])
        totals["locked_versions"] += int(version_summary["locked_count"])

        trusted_samples.extend(
            _trusted_gold_samples(
                dataset=dataset,
                rows=rows,
                legacy_entries=legacy_entries,
                limit=max(0, 4 - len(trusted_samples)),
            )
        )

        dataset_payloads.append({
            "id": gold_id,
            "name": dataset.name,
            "dataset_type": dataset.dataset_type.value,
            "record_count": int(dataset.record_count or 0),
            "example_count": example_count,
            "trusted_examples": trusted_examples,
            "review_needed": review_needed,
            "is_locked": bool(dataset.is_locked),
            "validation_status": validation_status,
            "coverage_source": coverage_source,
            "row_status_counts": row_counts,
            "queue_status_counts": queue_counts,
            "versions": version_summary,
            "coverage": _gold_coverage_payload(
                inputs=inputs,
                expected=expected,
                labels=labels,
            ),
            "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
        })

    issues: list[dict[str, str]] = []
    if not gold_datasets:
        issues.append(
            _issue(
                "no_gold_sets",
                "blocker",
                "No gold set yet",
                "Create a small trusted gold set before relying on evaluations or regression checks.",
                action_label="Open Gold Set",
                target_tab="goldset",
            )
        )
    elif totals["example_count"] <= 0:
        issues.append(
            _issue(
                "no_gold_examples",
                "blocker",
                "Gold set has no examples",
                "Add trusted Q&A pairs or sample rows into the Gold Set workbench.",
                action_label="Add gold examples",
                target_tab="goldset",
            )
        )
    elif totals["trusted_examples"] < _GOLD_SET_MIN_STARTER_ROWS:
        issues.append(
            _issue(
                "thin_gold_set",
                "warning",
                "Gold set is thin",
                f"{totals['trusted_examples']} trusted example(s) are available; add at least {_GOLD_SET_MIN_STARTER_ROWS} to start evaluating reliably.",
                action_label="Grow Gold Set",
                target_tab="goldset",
            )
        )

    if totals["review_needed"] > 0:
        issues.append(
            _issue(
                "gold_rows_need_review",
                "warning",
                "Gold rows need review",
                f"{totals['review_needed']} gold row(s) are pending, in review, or waiting on changes.",
                action_label="Review Gold Set",
                target_tab="goldset",
            )
        )

    aggregate_coverage = _gold_coverage_payload(
        inputs=aggregate_inputs,
        expected=aggregate_expected,
        labels=aggregate_labels,
    )
    if totals["example_count"] > 0 and not aggregate_coverage["expected_fields"]:
        issues.append(
            _issue(
                "gold_expected_fields_missing",
                "warning",
                "Expected-answer fields are not visible",
                "Gold examples should include expected outputs so evaluations can score model responses.",
                action_label="Review fields",
                target_tab="goldset",
            )
        )

    if totals["example_count"] > 0 and not aggregate_coverage["label_fields"]:
        issues.append(
            _issue(
                "gold_label_metadata_missing",
                "info",
                "Label metadata is light",
                "Difficulty, category, reviewer, or risk labels make gold-set coverage easier to audit.",
                action_label="Add labels",
                target_tab="goldset",
            )
        )

    if not gold_datasets or totals["example_count"] <= 0:
        verdict: GoldSetVerdict = "empty"
    elif any(item["severity"] in {"blocker", "warning"} for item in issues):
        verdict = "attention"
    else:
        verdict = "ready"

    validation_status = (
        "empty"
        if verdict == "empty"
        else (
            "needs_review"
            if totals["review_needed"] > 0
            else ("ready" if totals["trusted_examples"] >= _GOLD_SET_MIN_STARTER_ROWS else "thin")
        )
    )

    return {
        "project_id": project_id,
        "verdict": verdict,
        "read_only": True,
        "minimum_recommended_examples": _GOLD_SET_MIN_STARTER_ROWS,
        "validation": {
            "status": validation_status,
            "trusted_examples": totals["trusted_examples"],
            "review_needed": totals["review_needed"],
            "locked_gold_sets": totals["locked_gold_sets"],
            "locked_versions": totals["locked_versions"],
        },
        "totals": totals,
        "datasets": dataset_payloads,
        "trusted_examples": trusted_samples[:4],
        "coverage": aggregate_coverage,
        "issues": issues,
        "entry_point": {
            "label": "Open Gold Set workflow",
            "target_tab": "goldset",
            "reason": "Use the existing Gold Set panel to add, review, sample, or lock trusted examples.",
        },
    }


def _playbook_mode_label(mode: str) -> str:
    labels = {
        "positives_paraphrase": "Paraphrase positives",
        "hard_negatives": "Hard negatives",
        "class_balance_fill": "Balance class distribution",
        "edge_cases": "Edge cases",
        "refusals": "Refusals",
        "format_robustness": "Format robustness",
        "cluster_targeted": "Target a failure cluster",
    }
    return labels.get(mode, mode.replace("_", " ").title())


def _playbook_payloads(playbooks: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in playbooks:
        recipe_id = str(item.get("recipe_id") or "").strip()
        mode = str(item.get("mode") or "").strip()
        if not recipe_id or not mode:
            continue
        rows.append({
            "recipe_id": recipe_id,
            "mode": mode,
            "label": _playbook_mode_label(mode),
        })
    rows.sort(key=lambda row: (row["recipe_id"], row["mode"]))
    return rows


def _backend_payloads() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, cls in enumerate(BACKEND_REGISTRY):
        name = str(getattr(cls, "name", cls.__name__)).strip() or cls.__name__
        try:
            available = bool(cls.is_available())
        except Exception:  # noqa: BLE001
            available = False
        description = name
        if available:
            try:
                description = str(cls().describe())
            except Exception:  # noqa: BLE001
                description = name
        entries.append({
            "name": name,
            "available": available,
            "describe": description,
            "is_default": index == 0 and name == "ollama",
            "is_local": name == "ollama",
            "paid_required": name == "nemo",
        })
    return entries


def _backend_named(backends: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for backend in backends:
        if str(backend.get("name") or "") == name:
            return backend
    return None


def _gold_file_backed_rows(datasets: list[Dataset]) -> int:
    total = 0
    for dataset in datasets:
        if dataset.dataset_type not in _GOLD_SET_DATASET_TYPES:
            continue
        rows = _load_jsonl_dicts(dataset.file_path)
        total += len(rows)
    return total


def _review_queue_summary(queue: dict[str, Any]) -> dict[str, Any]:
    groups = queue.get("groups") if isinstance(queue.get("groups"), list) else []
    accepted_groups = (
        queue.get("accepted_groups")
        if isinstance(queue.get("accepted_groups"), list)
        else []
    )
    top_pending = sorted(
        [
            {
                "synth_source": str(group.get("synth_source") or ""),
                "count": int(group.get("count") or 0),
                "truncated": bool(group.get("truncated")),
            }
            for group in groups
            if isinstance(group, dict)
        ],
        key=lambda item: (-int(item["count"]), item["synth_source"]),
    )[:4]
    top_accepted = sorted(
        [
            {
                "synth_source": str(group.get("synth_source") or ""),
                "count": int(group.get("count") or 0),
                "truncated": bool(group.get("truncated")),
            }
            for group in accepted_groups
            if isinstance(group, dict)
        ],
        key=lambda item: (-int(item["count"]), item["synth_source"]),
    )[:4]
    return {
        "dataset_id": queue.get("dataset_id"),
        "total_rows": int(queue.get("total_rows") or 0),
        "total_pending": int(queue.get("total_pending") or 0),
        "total_accepted": int(queue.get("total_accepted") or 0),
        "pending_group_count": len(groups),
        "accepted_group_count": len(accepted_groups),
        "top_pending_groups": top_pending,
        "top_accepted_groups": top_accepted,
    }


def _prerequisite(
    prereq_id: str,
    label: str,
    status: str,
    message: str,
    *,
    target_tab: str,
) -> dict[str, str]:
    return {
        "id": prereq_id,
        "label": label,
        "status": status,
        "message": message,
        "target_tab": target_tab,
    }


def _domain_definition(domain_id: str) -> dict[str, Any] | None:
    normalized = str(domain_id or "").strip()
    for definition in _DOMAIN_DEFINITIONS:
        if str(definition.get("id") or "") == normalized:
            return definition
    return None


def _domain_id_from_applied(applied: dict[str, Any]) -> str | None:
    applied_text = " ".join(
        str(value or "").lower()
        for value in (
            applied.get("profile_id"),
            applied.get("profile_display_name"),
            applied.get("pack_id"),
            applied.get("pack_display_name"),
            applied.get("pack_default_profile_id"),
        )
    )
    if not applied_text.strip():
        return None
    for definition in _DOMAIN_DEFINITIONS:
        domain_id = str(definition.get("id") or "")
        aliases = [domain_id.replace("_", "-"), domain_id, *list(definition.get("aliases") or [])]
        if any(str(alias).lower().replace("_", "-") in applied_text.replace("_", "-") for alias in aliases):
            return domain_id
    return None


def _synthetic_library_domain_candidates(domain_detection: dict[str, Any]) -> list[dict[str, Any]]:
    detected = (
        domain_detection.get("detected_domain")
        if isinstance(domain_detection.get("detected_domain"), dict)
        else {}
    )
    applied = (
        domain_detection.get("applied")
        if isinstance(domain_detection.get("applied"), dict)
        else {}
    )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    detected_id = str(detected.get("id") or "").strip()
    detected_confidence = float(detected.get("confidence") or 0.0)
    if detected_id and detected_id != "generic_domain" and detected_confidence >= 0.3:
        candidates.append({
            "domain_id": detected_id,
            "domain_label": str(detected.get("label") or detected_id.replace("_", " ").title()),
            "source": "detected",
            "confidence": round(detected_confidence, 4),
        })
        seen.add(detected_id)

    applied_id = _domain_id_from_applied(applied)
    if applied_id and applied_id not in seen:
        definition = _domain_definition(applied_id) or {}
        candidates.append({
            "domain_id": applied_id,
            "domain_label": str(definition.get("label") or applied_id.replace("_", " ").title()),
            "source": "applied",
            "confidence": 0.78,
        })
        seen.add(applied_id)

    if not candidates:
        candidates.append({
            "domain_id": "generic_domain",
            "domain_label": "Generic Domain",
            "source": "fallback",
            "confidence": max(0.0, min(1.0, detected_confidence)),
        })
    return candidates[:2]


def _synthetic_required_fields(recipe_payload: dict[str, Any] | None, domain_id: str) -> list[str]:
    if recipe_payload:
        input_column = str(recipe_payload.get("default_input_column") or "input").strip() or "input"
        output_column = str(recipe_payload.get("default_output_column") or "output").strip() or "output"
        fields = [input_column, output_column]
        recipe_id = str(recipe_payload.get("id") or "")
        if recipe_id == "qa-sft" and domain_id in {"policy_qa", "legal_contracts", "finance_support"}:
            fields.append("context")
        deduped: list[str] = []
        for field in fields:
            if field not in deduped:
                deduped.append(field)
        return deduped
    return ["input", "expected"]


def _synthetic_expected_output_shape(recipe_payload: dict[str, Any] | None, required_fields: list[str]) -> dict[str, Any]:
    recipe_id = str((recipe_payload or {}).get("id") or "unknown")
    payload_fields = list(required_fields)
    for field in ("synth_source", "synth_confidence", "review_status"):
        if field not in payload_fields:
            payload_fields.append(field)
    return {
        "format": "jsonl",
        "recipe_id": recipe_id,
        "payload_fields": payload_fields,
        "review_status": "pending",
        "notes": [
            "Rows are generated in the existing Synthetic workflow, not in Data Studio.",
            "Generated rows enter review before they can affect prepared datasets.",
        ],
    }


def _synthetic_prompt_focus(domain_id: str, strategy: dict[str, Any]) -> list[str]:
    focus = [
        str(strategy.get("domain_reason") or ""),
        "Preserve the active recipe's canonical input/output fields.",
        "Prefer local Ollama generation unless the user explicitly selects another backend.",
    ]
    if domain_id == "pii_pci_detection":
        focus.append("Use synthetic sensitive-looking examples; avoid leaking real secrets.")
    elif domain_id in {"policy_qa", "legal_contracts", "finance_support"}:
        focus.append("Keep citations, policy boundaries, and insufficient-information cases reviewable.")
    elif domain_id == "support_faq":
        focus.append("Vary customer wording while keeping the answer intent and escalation boundary stable.")
    return [item for item in focus if item][:5]


def _synthetic_review_gates(domain_id: str, pending_synth: int) -> list[str]:
    gates = [
        "Human review is required before generated rows enter prepared datasets.",
        "Reject rows that do not match the active recipe shape.",
    ]
    if pending_synth > 0:
        gates.append("Clear the current synthetic review queue before generating a large new batch.")
    if domain_id == "pii_pci_detection":
        gates.append("Review synthetic sensitive values and false-negative coverage before SFT.")
    elif domain_id in {"policy_qa", "legal_contracts", "finance_support"}:
        gates.append("Review answers for overconfidence, missing context, and stale policy language.")
    elif domain_id == "support_faq":
        gates.append("Review account, billing, and cancellation answers for escalation boundaries.")
    return gates[:5]


def _synthetic_library_prerequisites(
    *,
    recipe_payload: dict[str, Any] | None,
    recipe_compatible: bool,
    mode_available: bool,
    mapping_verdict: str,
    missing_fields: list[str],
    file_backed_gold_rows: int,
    gold_rows: int,
    ollama_backend: dict[str, Any] | None,
    pending_synth: int,
) -> list[dict[str, str]]:
    recipe_id = str((recipe_payload or {}).get("id") or "")
    if not recipe_payload:
        recipe_status = "missing"
        recipe_message = "Choose a recipe before using a domain-specific synthetic library."
    elif recipe_compatible:
        recipe_status = "met"
        recipe_message = f"{recipe_id} matches this domain library."
    else:
        recipe_status = "attention"
        recipe_message = f"{recipe_id} can still generate rows, but this domain usually fits another recipe."

    if not recipe_payload:
        mode_status = "missing"
        mode_message = "Compatible playbook modes appear after a recipe is selected."
    elif mode_available:
        mode_status = "met"
        mode_message = "At least one curated playbook mode is registered for the active recipe."
    else:
        mode_status = "missing"
        mode_message = "No registered playbook mode currently matches this domain strategy and recipe."

    if not recipe_payload or mapping_verdict == "empty":
        mapping_status = "attention"
        mapping_message = "Run mapping preview with source rows before generating at scale."
    elif missing_fields:
        mapping_status = "missing"
        mapping_message = f"Required mapping fields need review: {', '.join(missing_fields[:4])}."
    else:
        mapping_status = "met"
        mapping_message = "Required recipe fields look ready in the current mapping preview."

    if file_backed_gold_rows > 0:
        gold_status = "met"
        gold_message = f"{file_backed_gold_rows} file-backed Gold Set row(s) can anchor generation."
    elif gold_rows > 0:
        gold_status = "attention"
        gold_message = "Gold examples exist, but current playbooks need file-backed Gold Set rows."
    else:
        gold_status = "missing"
        gold_message = "Add trusted Gold Set examples before running this library."

    if bool((ollama_backend or {}).get("available")):
        ollama_status = "met"
        ollama_message = f"Local Ollama is ready: {(ollama_backend or {}).get('describe') or 'ollama'}."
    else:
        ollama_status = "attention"
        ollama_message = "Ollama is the free local default; start it before generating."

    review_status = "met" if pending_synth <= 0 else "attention"
    review_message = (
        "No pending synthetic review gate is active."
        if pending_synth <= 0
        else f"{pending_synth} synthetic row(s) are already pending review."
    )

    return [
        _prerequisite("recipe", "Recipe compatibility", recipe_status, recipe_message, target_tab="data"),
        _prerequisite("playbook_mode", "Playbook mode", mode_status, mode_message, target_tab="synthetic"),
        _prerequisite("mapping", "Required fields", mapping_status, mapping_message, target_tab="dataprep"),
        _prerequisite("gold_examples", "Gold anchors", gold_status, gold_message, target_tab="goldset"),
        _prerequisite("local_ollama", "Local Ollama", ollama_status, ollama_message, target_tab="synthetic"),
        _prerequisite("review_gate", "Review gate", review_status, review_message, target_tab="synthetic"),
    ]


def _synthetic_domain_playbook_libraries(
    *,
    domain_detection: dict[str, Any],
    mapping_preview: dict[str, Any],
    recipe_payload: dict[str, Any] | None,
    compatible_modes: set[str],
    ollama_backend: dict[str, Any] | None,
    gold_rows: int,
    file_backed_gold_rows: int,
    pending_synth: int,
) -> dict[str, Any]:
    mapping_summary = (
        mapping_preview.get("summary")
        if isinstance(mapping_preview.get("summary"), dict)
        else {}
    )
    mapping_gaps = [
        str(item)
        for item in list(mapping_summary.get("required_fields_below_100") or [])
        if str(item).strip()
    ]
    mapping_verdict = str(mapping_preview.get("verdict") or "empty")
    recipe_id = str((recipe_payload or {}).get("id") or "")
    recipe_label = str((recipe_payload or {}).get("name") or recipe_id or "No recipe")
    libraries: list[dict[str, Any]] = []

    for candidate in _synthetic_library_domain_candidates(domain_detection):
        domain_id = str(candidate["domain_id"])
        definition = _domain_definition(domain_id) or {}
        strategies = _DOMAIN_SYNTHETIC_STRATEGIES.get(domain_id) or [
            {
                "id": "baseline_variants",
                "title": "Generate baseline variants after domain confirmation",
                "strategy": "positive paraphrase",
                "desired_modes": ("positives_paraphrase",),
                "domain_reason": "Synthetic rows are safer when the domain and recipe are confirmed first.",
            }
        ]
        recommended_recipes = [str(item) for item in list(definition.get("recipes") or []) if str(item).strip()]
        recipe_compatible = bool(recipe_id and (not recommended_recipes or recipe_id in recommended_recipes))
        playbooks: list[dict[str, Any]] = []
        desired_modes_seen: set[str] = set()
        missing_modes_seen: set[str] = set()
        compatible_desired_seen: set[str] = set()

        for strategy in strategies[:4]:
            desired_modes = tuple(str(mode) for mode in strategy.get("desired_modes") or ())
            desired_modes_seen.update(mode for mode in desired_modes if mode)
            mode, mode_available = _pick_playbook_mode(desired_modes, compatible_modes)
            if not mode:
                mode = desired_modes[0] if desired_modes else "positives_paraphrase"
            if mode_available:
                compatible_desired_seen.add(mode)
            else:
                missing_modes_seen.update(mode for mode in desired_modes if mode and mode not in compatible_modes)

            required_fields = _synthetic_required_fields(recipe_payload, domain_id)
            missing_fields = [field for field in required_fields if field in mapping_gaps]
            if mapping_gaps and not missing_fields:
                missing_fields = mapping_gaps[:4]
            expected_shape = _synthetic_expected_output_shape(recipe_payload, required_fields)
            review_gates = _synthetic_review_gates(domain_id, pending_synth)
            prerequisites = _synthetic_library_prerequisites(
                recipe_payload=recipe_payload,
                recipe_compatible=recipe_compatible,
                mode_available=mode_available,
                mapping_verdict=mapping_verdict,
                missing_fields=missing_fields,
                file_backed_gold_rows=file_backed_gold_rows,
                gold_rows=gold_rows,
                ollama_backend=ollama_backend,
                pending_synth=pending_synth,
            )
            blocker_count = sum(1 for item in prerequisites if item["status"] == "missing")
            warning_count = sum(1 for item in prerequisites if item["status"] == "attention")
            if blocker_count:
                readiness = "blocked"
                readiness_reason = "Recipe, playbook mode, or Gold Set prerequisites need setup first."
            elif warning_count:
                readiness = "attention"
                readiness_reason = "The library can be reviewed, but setup or review gates need attention."
            else:
                readiness = "ready"
                readiness_reason = "Recipe, local backend, Gold anchors, mapping, and review gates look ready."

            playbooks.append({
                "id": f"{domain_id}:{strategy.get('id') or mode}",
                "title": str(strategy.get("title") or _playbook_mode_label(mode)),
                "strategy": str(strategy.get("strategy") or mode.replace("_", " ")),
                "mode": mode,
                "mode_label": _playbook_mode_label(mode),
                "mode_available": mode_available,
                "recipe_id": recipe_id or None,
                "recipe_compatible": recipe_compatible,
                "required_fields": required_fields,
                "missing_fields": missing_fields,
                "expected_output_shape": expected_shape,
                "prompt_focus": _synthetic_prompt_focus(domain_id, strategy),
                "review_gates": review_gates,
                "prerequisites": prerequisites,
                "readiness": readiness,
                "readiness_reason": readiness_reason,
                "generation_path": {
                    "backend": "ollama",
                    "available": bool((ollama_backend or {}).get("available")),
                    "describe": str((ollama_backend or {}).get("describe") or "ollama"),
                    "local_default": True,
                    "paid_required": False,
                },
                "generation_action": {
                    "label": "Open Synthetic workflow",
                    "target_tab": "synthetic",
                    "requires_confirmation": True,
                    "description": "Run this library from the existing Synthetic workflow; Data Studio only previews the plan.",
                },
            })

        library_blocked = any(item["readiness"] == "blocked" for item in playbooks)
        library_attention = any(item["readiness"] == "attention" for item in playbooks)
        if library_blocked:
            status = "blocked"
        elif library_attention:
            status = "attention"
        else:
            status = "ready"
        summary = (
            f"{candidate['domain_label']} library uses local Ollama by default and keeps generated rows behind review."
        )
        if recommended_recipes and recipe_id and recipe_id not in recommended_recipes:
            summary = (
                f"{candidate['domain_label']} usually pairs with {', '.join(recommended_recipes)}; "
                f"review compatibility before using {recipe_id}."
            )

        libraries.append({
            "id": f"{domain_id}-{candidate['source']}",
            "domain_id": domain_id,
            "domain_label": candidate["domain_label"],
            "source": candidate["source"],
            "confidence": candidate["confidence"],
            "status": status,
            "summary": summary,
            "local_first": True,
            "active_recipe_id": recipe_id or None,
            "active_recipe_label": recipe_label,
            "recommended_recipes": recommended_recipes,
            "recipe_compatible": recipe_compatible,
            "desired_modes": sorted(desired_modes_seen),
            "compatible_modes": sorted(compatible_desired_seen),
            "missing_modes": sorted(missing_modes_seen),
            "review_gates": _synthetic_review_gates(domain_id, pending_synth),
            "playbooks": playbooks,
        })

    ready_count = sum(1 for item in libraries if item["status"] == "ready")
    attention_count = sum(1 for item in libraries if item["status"] == "attention")
    blocked_count = sum(1 for item in libraries if item["status"] == "blocked")
    detected = (
        domain_detection.get("detected_domain")
        if isinstance(domain_detection.get("detected_domain"), dict)
        else {}
    )
    applied = (
        domain_detection.get("applied")
        if isinstance(domain_detection.get("applied"), dict)
        else {}
    )
    return {
        "read_only": True,
        "local_first": True,
        "default_backend": "ollama",
        "ollama_ready": bool((ollama_backend or {}).get("available")),
        "library_count": len(libraries),
        "ready_count": ready_count,
        "attention_count": attention_count,
        "blocked_count": blocked_count,
        "detected_domain": {
            "id": detected.get("id"),
            "label": detected.get("label"),
            "confidence": detected.get("confidence"),
            "source": detected.get("source"),
        },
        "applied_domain": {
            "profile_id": applied.get("profile_id"),
            "pack_id": applied.get("pack_id"),
            "display_name": applied.get("profile_display_name") or applied.get("pack_display_name"),
        },
        "libraries": libraries,
        "entry_point": {
            "label": "Open Synthetic workflow",
            "target_tab": "synthetic",
            "reason": "Run domain-specific playbooks in the existing Synthetic tab; generation requires user confirmation there.",
            "requires_confirmation": True,
        },
    }


async def build_data_studio_synthetic_playbook_center(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a read-only Synthetic Playbook Center summary."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    recipe_payload = _recipe_payload(project)
    recipe_id = str((recipe_payload or {}).get("id") or "").strip()
    full_catalog = _playbook_payloads(list_playbooks())
    compatible_catalog = (
        _playbook_payloads(available_playbooks_for_recipe(recipe_id))
        if recipe_id
        else []
    )
    preview_catalog = compatible_catalog if recipe_id else full_catalog
    domain_detection = await build_data_studio_domain_detection(db, project_id)
    mapping_preview = await build_data_studio_mapping_preview(db, project_id)

    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    datasets = list(datasets_result.scalars().all())
    gold_rows = sum(
        int(dataset.record_count or 0)
        for dataset in datasets
        if dataset.dataset_type in _GOLD_SET_DATASET_TYPES
    )
    file_backed_gold_rows = _gold_file_backed_rows(datasets)

    backends = _backend_payloads()
    ollama_backend = _backend_named(backends, "ollama")
    available_backends = [backend for backend in backends if bool(backend.get("available"))]
    queue = _review_queue_summary(await list_review_queue(db, project_id))

    prerequisites: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []

    if recipe_payload is None:
        prerequisites.append(
            _prerequisite(
                "recipe",
                "Recipe selected",
                "missing",
                "Pick a recipe so BrewSLM can show compatible synthetic playbooks.",
                target_tab="data",
            )
        )
        issues.append(
            _issue(
                "synthetic_recipe_missing",
                "blocker",
                "Recipe not selected",
                "Synthetic playbooks are recipe-aware; choose a recipe before generating rows.",
                action_label="Choose recipe",
                target_tab="data",
            )
        )
    else:
        prerequisites.append(
            _prerequisite(
                "recipe",
                "Recipe selected",
                "met",
                f"{recipe_payload.get('name') or recipe_id} is active.",
                target_tab="data",
            )
        )

    if recipe_id and not compatible_catalog:
        prerequisites.append(
            _prerequisite(
                "compatible_playbooks",
                "Compatible playbooks",
                "missing",
                f"No synthetic playbooks are registered for {recipe_id}.",
                target_tab="synthetic",
            )
        )
        issues.append(
            _issue(
                "synthetic_no_compatible_playbooks",
                "blocker",
                "No compatible playbooks",
                f"The active recipe '{recipe_id}' does not have a registered synthetic playbook.",
                action_label="Open Synthetic",
                target_tab="synthetic",
            )
        )
    else:
        prerequisites.append(
            _prerequisite(
                "compatible_playbooks",
                "Compatible playbooks",
                "met" if recipe_id else "attention",
                (
                    f"{len(compatible_catalog)} playbook mode(s) match the active recipe."
                    if recipe_id
                    else f"{len(full_catalog)} playbook mode(s) are available after a recipe is selected."
                ),
                target_tab="synthetic",
            )
        )

    if file_backed_gold_rows <= 0:
        prerequisites.append(
            _prerequisite(
                "gold_examples",
                "Gold examples",
                "missing" if gold_rows <= 0 else "attention",
                (
                    "Add file-backed Gold Set rows before running current playbooks."
                    if gold_rows <= 0
                    else "Gold examples exist, but current playbooks read from Gold Set JSONL files."
                ),
                target_tab="goldset",
            )
        )
        issues.append(
            _issue(
                "synthetic_gold_rows_missing",
                "warning",
                "Gold rows are not ready for playbooks",
                "Current playbooks generate from Gold Set rows; add or import trusted gold examples first.",
                action_label="Open Gold Set",
                target_tab="goldset",
            )
        )
    else:
        prerequisites.append(
            _prerequisite(
                "gold_examples",
                "Gold examples",
                "met",
                f"{file_backed_gold_rows} file-backed gold row(s) can seed playbook generation.",
                target_tab="goldset",
            )
        )

    if bool((ollama_backend or {}).get("available")):
        prerequisites.append(
            _prerequisite(
                "local_ollama",
                "Local Ollama",
                "met",
                f"Local default backend is ready: {ollama_backend.get('describe')}.",
                target_tab="synthetic",
            )
        )
    else:
        prerequisites.append(
            _prerequisite(
                "local_ollama",
                "Local Ollama",
                "attention",
                "Ollama is the free local default; start Ollama or pull a local model before generating.",
                target_tab="synthetic",
            )
        )
        issues.append(
            _issue(
                "synthetic_ollama_unavailable",
                "warning",
                "Local Ollama is not ready",
                "BrewSLM can still show playbooks, but local free generation needs a reachable Ollama model.",
                action_label="Open Synthetic",
                target_tab="synthetic",
            )
        )

    if not available_backends:
        issues.append(
            _issue(
                "synthetic_no_backend_available",
                "warning",
                "No synthetic backend is available",
                "Install or start Ollama for the default free local path, or configure an OpenAI-compatible endpoint.",
                action_label="Open Synthetic",
                target_tab="synthetic",
            )
        )

    if queue["total_pending"] > 0:
        issues.append(
            _issue(
                "synthetic_rows_pending_review",
                "warning",
                "Synthetic rows pending review",
                f"{queue['total_pending']} synthetic row(s) must be accepted before they enter dataset prep.",
                action_label="Review synthetic rows",
                target_tab="synthetic",
            )
        )

    if recipe_payload is None and not full_catalog:
        verdict: SyntheticPlaybookVerdict = "empty"
    elif any(item["severity"] in {"blocker", "warning"} for item in issues):
        verdict = "attention"
    else:
        verdict = "ready"

    supported_recipes = sorted({item["recipe_id"] for item in full_catalog})
    compatible_modes = sorted({item["mode"] for item in compatible_catalog})
    domain_libraries = _synthetic_domain_playbook_libraries(
        domain_detection=domain_detection,
        mapping_preview=mapping_preview,
        recipe_payload=recipe_payload,
        compatible_modes=set(compatible_modes),
        ollama_backend=ollama_backend,
        gold_rows=gold_rows,
        file_backed_gold_rows=file_backed_gold_rows,
        pending_synth=int(queue["total_pending"]),
    )

    return {
        "project_id": project_id,
        "verdict": verdict,
        "read_only": True,
        "recipe": recipe_payload,
        "catalog": {
            "total_playbooks": len(full_catalog),
            "compatible_playbooks": len(compatible_catalog),
            "preview_playbooks": preview_catalog[:8],
            "supported_recipes": supported_recipes,
            "compatible_modes": compatible_modes,
        },
        "backends": backends,
        "recommended_backend": {
            "name": "ollama",
            "available": bool((ollama_backend or {}).get("available")),
            "describe": str((ollama_backend or {}).get("describe") or "ollama"),
            "local_default": True,
            "paid_required": False,
        },
        "domain_libraries": domain_libraries,
        "prerequisites": prerequisites,
        "review_queue": queue,
        "issues": issues,
        "entry_point": {
            "label": "Open Synthetic workflow",
            "target_tab": "synthetic",
            "reason": "Use the existing Synthetic tab and PlaybookPickerPanel to generate, review, and accept rows.",
        },
    }


_DOMAIN_SYNTHETIC_STRATEGIES: dict[str, list[dict[str, Any]]] = {
    "support_faq": [
        {
            "id": "customer_phrasing",
            "title": "Generate customer phrasing variants",
            "strategy": "positive paraphrase",
            "desired_modes": ("positives_paraphrase",),
            "domain_reason": "Support assistants need to recognize the same intent across messy customer wording.",
        },
        {
            "id": "escalation_boundaries",
            "title": "Add escalation and boundary examples",
            "strategy": "hard negatives",
            "desired_modes": ("hard_negatives", "cluster_targeted"),
            "domain_reason": "Account, billing, and cancellation answers need clear handoff boundaries.",
        },
    ],
    "policy_qa": [
        {
            "id": "exceptions",
            "title": "Cover policy exceptions and edge cases",
            "strategy": "edge cases",
            "desired_modes": ("cluster_targeted", "positives_paraphrase"),
            "domain_reason": "Policy Q&A quality depends on edge cases, exceptions, and insufficient-information behavior.",
        },
    ],
    "pii_pci_detection": [
        {
            "id": "hard_negatives",
            "title": "Generate privacy hard negatives",
            "strategy": "hard negatives",
            "desired_modes": ("hard_negatives", "class_balance_fill"),
            "domain_reason": "PII/PCI detectors need examples that look sensitive but should not be redacted.",
        },
        {
            "id": "class_balance",
            "title": "Balance sensitive entity classes",
            "strategy": "class balance",
            "desired_modes": ("class_balance_fill", "hard_negatives"),
            "domain_reason": "False negatives matter for sensitive data; minority entity classes need explicit coverage.",
        },
    ],
    "security_alert_triage": [
        {
            "id": "severity_balance",
            "title": "Balance alert severity examples",
            "strategy": "class balance",
            "desired_modes": ("class_balance_fill", "hard_negatives"),
            "domain_reason": "Security triage models need enough coverage for rare but high-impact severities.",
        },
    ],
    "legal_contracts": [
        {
            "id": "unknowns_and_handoff",
            "title": "Add legal unknown and handoff cases",
            "strategy": "edge cases",
            "desired_modes": ("cluster_targeted", "positives_paraphrase"),
            "domain_reason": "Legal clause assistants should avoid overconfident answers when source text is ambiguous.",
        },
    ],
    "finance_support": [
        {
            "id": "numeric_edges",
            "title": "Generate finance numeric edge cases",
            "strategy": "hard negatives",
            "desired_modes": ("hard_negatives", "cluster_targeted"),
            "domain_reason": "Finance support data needs numeric and money-moving boundaries covered before training.",
        },
    ],
    "code_review": [
        {
            "id": "defect_hard_negatives",
            "title": "Generate code-review hard negatives",
            "strategy": "hard negatives",
            "desired_modes": ("hard_negatives", "cluster_targeted"),
            "domain_reason": "Code-review SLMs need examples that distinguish real defects from style-only comments.",
        },
    ],
    "customer_sentiment": [
        {
            "id": "label_balance",
            "title": "Balance sentiment classes",
            "strategy": "class balance",
            "desired_modes": ("class_balance_fill", "hard_negatives"),
            "domain_reason": "Sentiment datasets often overrepresent common labels and miss ambiguous neutral examples.",
        },
    ],
}


def _pick_playbook_mode(
    desired_modes: tuple[str, ...],
    compatible_modes: set[str],
) -> tuple[str | None, bool]:
    for mode in desired_modes:
        if mode in compatible_modes:
            return mode, True
    if compatible_modes:
        return sorted(compatible_modes)[0], False
    return None, False


def _synthetic_recommendation(
    *,
    rec_id: str,
    title: str,
    strategy: str,
    priority: str,
    target_tab: str,
    action_label: str,
    rationale: str,
    domain_reason: str,
    evidence: list[str],
    confidence: float,
    playbook_mode: str | None,
    playbook_available: bool,
    local_ollama: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": rec_id,
        "title": title,
        "strategy": strategy,
        "priority": priority,
        "target_tab": target_tab,
        "action_label": action_label,
        "rationale": rationale,
        "domain_reason": domain_reason,
        "evidence": [item for item in evidence if item][:6],
        "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
        "playbook_mode": playbook_mode,
        "playbook_available": playbook_available,
        "requires_user_confirmation": True,
        "generation_path": {
            "backend": "ollama",
            "available": bool(local_ollama.get("available")),
            "describe": str(local_ollama.get("describe") or "ollama"),
            "local_default": True,
            "paid_required": False,
        },
    }


def _domain_signal_evidence(domain: dict[str, Any]) -> list[str]:
    detected = domain.get("detected_domain") if isinstance(domain.get("detected_domain"), dict) else {}
    evidence = []
    label = str(detected.get("label") or "Unknown domain")
    confidence = float(detected.get("confidence") or 0.0)
    evidence.append(f"Detected domain: {label} ({round(confidence * 100)}% confidence).")
    keywords = list(detected.get("matched_keywords") or [])
    fields = list(detected.get("matched_fields") or [])
    if keywords:
        evidence.append(f"Domain keywords: {', '.join(str(item) for item in keywords[:6])}.")
    if fields:
        evidence.append(f"Domain fields: {', '.join(str(item) for item in fields[:6])}.")
    for item in list(domain.get("evidence") or [])[:2]:
        if isinstance(item, dict) and item.get("message"):
            evidence.append(str(item["message"]))
    return evidence


def _mapping_gap_evidence(mapping: dict[str, Any]) -> tuple[list[str], list[str]]:
    summary = mapping.get("summary") if isinstance(mapping.get("summary"), dict) else {}
    gaps = [
        str(item)
        for item in list(summary.get("required_fields_below_100") or [])
        if str(item).strip()
    ]
    evidence = []
    if gaps:
        evidence.append(f"Mapping coverage is incomplete for: {', '.join(gaps)}.")
    sampled = int(summary.get("sampled_records") or 0)
    mapped = int(summary.get("mapped_records") or 0)
    if sampled > 0:
        evidence.append(f"Mapping preview mapped {mapped}/{sampled} sampled row(s).")
    return gaps, evidence


async def build_data_studio_synthetic_recommendations(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return deterministic, domain-aware synthetic data recommendations."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    domain = await build_data_studio_domain_detection(db, project_id)
    mapping = await build_data_studio_mapping_preview(db, project_id)
    gold = await build_data_studio_gold_set_workbench(db, project_id)
    playbooks = await build_data_studio_synthetic_playbook_center(db, project_id)

    detected = domain.get("detected_domain") if isinstance(domain.get("detected_domain"), dict) else {}
    domain_id = str(detected.get("id") or "generic_domain")
    domain_label = str(detected.get("label") or "Generic Domain")
    domain_confidence = float(detected.get("confidence") or 0.0)
    recipe = playbooks.get("recipe") if isinstance(playbooks.get("recipe"), dict) else None
    compatible_modes = {
        str(mode)
        for mode in list((playbooks.get("catalog") or {}).get("compatible_modes") or [])
        if str(mode).strip()
    }
    local_ollama = (
        playbooks.get("recommended_backend")
        if isinstance(playbooks.get("recommended_backend"), dict)
        else {"name": "ollama", "available": False, "describe": "ollama"}
    )
    queue = playbooks.get("review_queue") if isinstance(playbooks.get("review_queue"), dict) else {}
    gold_validation = gold.get("validation") if isinstance(gold.get("validation"), dict) else {}
    gold_coverage = gold.get("coverage") if isinstance(gold.get("coverage"), dict) else {}
    mapping_gaps, mapping_evidence = _mapping_gap_evidence(mapping)
    domain_evidence = _domain_signal_evidence(domain)

    recommendations: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    if recipe is None:
        issues.append(
            _issue(
                "synthetic_recommendation_recipe_missing",
                "blocker",
                "Recipe needed before recommending playbooks",
                "Pick a recipe so recommendations can target compatible synthetic strategies.",
                action_label="Choose recipe",
                target_tab="data",
            )
        )
        recommendations.append(
            _synthetic_recommendation(
                rec_id="setup_recipe_for_synthetic_recommendations",
                title="Choose a recipe before generating synthetic data",
                strategy="setup",
                priority="high",
                target_tab="data",
                action_label="Choose recipe",
                rationale="Synthetic playbooks are recipe-aware and should match the training contract.",
                domain_reason=f"{domain_label} recommendations become more precise after the training recipe is known.",
                evidence=domain_evidence,
                confidence=0.82,
                playbook_mode=None,
                playbook_available=False,
                local_ollama=local_ollama,
            )
        )

    if mapping_gaps:
        issues.append(
            _issue(
                "synthetic_recommendation_mapping_gaps",
                "warning",
                "Mapping gaps can pollute synthetic prompts",
                "Fix required-field coverage before generating synthetic rows from this dataset shape.",
                action_label="Review mapping",
                target_tab="dataprep",
            )
        )
        recommendations.append(
            _synthetic_recommendation(
                rec_id="fix_mapping_before_synthetic_generation",
                title="Fix mapping coverage before generating rows",
                strategy="data prep",
                priority="high",
                target_tab="dataprep",
                action_label="Review mapping",
                rationale="Synthetic examples should mirror the canonical row shape that training will consume.",
                domain_reason=f"{domain_label} examples are only useful if required inputs and outputs are mapped consistently.",
                evidence=mapping_evidence + domain_evidence,
                confidence=0.86,
                playbook_mode=None,
                playbook_available=False,
                local_ollama=local_ollama,
            )
        )

    trusted_examples = int(gold_validation.get("trusted_examples") or 0)
    review_needed = int(gold_validation.get("review_needed") or 0)
    min_examples = int(gold.get("minimum_recommended_examples") or _GOLD_SET_MIN_STARTER_ROWS)
    label_field_count = int((gold_coverage.get("field_counts") or {}).get("labels") or 0)
    if trusted_examples < min_examples or review_needed > 0:
        issues.append(
            _issue(
                "synthetic_recommendation_gold_set_needs_work",
                "warning",
                "Gold Set should be strengthened first",
                "Current playbooks use trusted Gold Set examples as anchors for generation.",
                action_label="Review Gold Set",
                target_tab="goldset",
            )
        )
        evidence = [
            f"{trusted_examples}/{min_examples} recommended trusted Gold Set example(s) are ready.",
            f"{review_needed} Gold Set row(s) still need review.",
        ]
        if label_field_count <= 0:
            evidence.append("Gold Set label metadata is not visible yet.")
        recommendations.append(
            _synthetic_recommendation(
                rec_id="strengthen_gold_set_before_synthetic_generation",
                title="Strengthen Gold Set anchors before generation",
                strategy="gold coverage",
                priority="medium",
                target_tab="goldset",
                action_label="Review Gold Set",
                rationale="Better trusted anchors reduce synthetic drift and make generated rows easier to review.",
                domain_reason=f"{domain_label} synthetic data should preserve domain-specific labels, boundaries, and expected answers.",
                evidence=evidence + domain_evidence,
                confidence=0.78,
                playbook_mode=None,
                playbook_available=False,
                local_ollama=local_ollama,
            )
        )

    pending_synth = int(queue.get("total_pending") or 0)
    if pending_synth > 0:
        issues.append(
            _issue(
                "synthetic_recommendation_pending_review_queue",
                "warning",
                "Review pending synthetic rows",
                "Pending synthetic rows are gated out of training until accepted.",
                action_label="Review synthetic rows",
                target_tab="synthetic",
            )
        )
        groups = [
            f"{group.get('synth_source')} ({group.get('count')})"
            for group in list(queue.get("top_pending_groups") or [])[:3]
            if isinstance(group, dict)
        ]
        recommendations.append(
            _synthetic_recommendation(
                rec_id="review_pending_synthetic_before_more_generation",
                title="Review pending synthetic rows before generating more",
                strategy="review queue",
                priority="high",
                target_tab="synthetic",
                action_label="Review queue",
                rationale="Reviewing existing synthetic rows prevents low-quality or duplicate rows from piling up.",
                domain_reason=f"{domain_label} quality depends on accepted examples matching the domain policy and labels.",
                evidence=[f"{pending_synth} synthetic row(s) are pending review."] + groups,
                confidence=0.9,
                playbook_mode=None,
                playbook_available=False,
                local_ollama=local_ollama,
            )
        )

    if not bool(local_ollama.get("available")):
        issues.append(
            _issue(
                "synthetic_recommendation_ollama_unavailable",
                "warning",
                "Local Ollama is not ready",
                "Start Ollama or pull a local model before using the default free generation path.",
                action_label="Open Synthetic",
                target_tab="synthetic",
            )
        )
        recommendations.append(
            _synthetic_recommendation(
                rec_id="start_local_ollama_for_synthetic_generation",
                title="Start local Ollama for free generation",
                strategy="backend setup",
                priority="medium",
                target_tab="synthetic",
                action_label="Open Synthetic",
                rationale="BrewSLM defaults synthetic generation to a local Ollama-compatible backend.",
                domain_reason=f"{domain_label} recommendations can be executed locally once an Ollama model is reachable.",
                evidence=["Recommended backend is local Ollama.", "No paid backend is required by Data Studio."],
                confidence=0.8,
                playbook_mode=None,
                playbook_available=False,
                local_ollama=local_ollama,
            )
        )

    if domain_confidence < 0.45:
        issues.append(
            _issue(
                "synthetic_recommendation_domain_confidence_low",
                "info",
                "Domain signal is weak",
                "Add representative source or Gold Set rows before relying on domain-specific synthetic recommendations.",
                action_label="Add representative rows",
                target_tab="data",
            )
        )

    strategies = _DOMAIN_SYNTHETIC_STRATEGIES.get(domain_id) or [
        {
            "id": "baseline_variants",
            "title": "Generate baseline variants after domain confirmation",
            "strategy": "positive paraphrase",
            "desired_modes": ("positives_paraphrase",),
            "domain_reason": "Synthetic rows are safer when the domain and recipe are confirmed first.",
        }
    ]
    if recipe is not None and domain_confidence >= 0.3:
        for strategy in strategies[:3]:
            desired_modes = tuple(str(mode) for mode in strategy.get("desired_modes") or ())
            mode, mode_available = _pick_playbook_mode(desired_modes, compatible_modes)
            evidence = domain_evidence + [
                f"Compatible playbook modes: {', '.join(sorted(compatible_modes)) or 'none'}.",
                f"Gold Set trusted examples: {trusted_examples}.",
            ]
            recommendations.append(
                _synthetic_recommendation(
                    rec_id=f"domain_{domain_id}_{strategy['id']}",
                    title=str(strategy["title"]),
                    strategy=str(strategy["strategy"]),
                    priority="medium" if mode_available else "low",
                    target_tab="synthetic",
                    action_label="Open Synthetic",
                    rationale="This strategy follows from deterministic domain signals and current dataset readiness.",
                    domain_reason=str(strategy["domain_reason"]),
                    evidence=evidence,
                    confidence=max(0.45, min(0.95, domain_confidence)),
                    playbook_mode=mode,
                    playbook_available=mode_available,
                    local_ollama=local_ollama,
                )
            )

    seen: set[str] = set()
    unique_recommendations = []
    priority_order = {"high": 0, "medium": 1, "low": 2}
    for item in sorted(
        recommendations,
        key=lambda rec: (
            priority_order.get(str(rec.get("priority")), 9),
            -float(rec.get("confidence") or 0.0),
            str(rec.get("id")),
        ),
    ):
        rec_id = str(item.get("id") or "")
        if rec_id in seen:
            continue
        seen.add(rec_id)
        unique_recommendations.append(item)
        if len(unique_recommendations) >= 8:
            break

    if not unique_recommendations:
        verdict: SyntheticRecommendationVerdict = "empty"
    elif any(item["severity"] in {"blocker", "warning"} for item in issues):
        verdict = "attention"
    else:
        verdict = "ready"

    return {
        "project_id": project_id,
        "verdict": verdict,
        "read_only": True,
        "auto_apply": False,
        "source_of_truth": "deterministic_data_studio_checks",
        "domain": {
            "id": domain_id,
            "label": domain_label,
            "confidence": round(domain_confidence, 4),
            "source": detected.get("source"),
        },
        "recipe": recipe,
        "signals": {
            "mapping_verdict": mapping.get("verdict"),
            "mapping_required_gaps": mapping_gaps,
            "gold_trusted_examples": trusted_examples,
            "gold_review_needed": review_needed,
            "gold_label_field_count": label_field_count,
            "synthetic_pending": pending_synth,
            "synthetic_accepted": int(queue.get("total_accepted") or 0),
            "compatible_playbook_modes": sorted(compatible_modes),
            "ollama_available": bool(local_ollama.get("available")),
        },
        "recommendations": unique_recommendations,
        "issues": issues,
        "entry_points": [
            {
                "label": "Open Synthetic workflow",
                "target_tab": "synthetic",
                "reason": "Run playbooks, generate rows, and review synthetic output in the existing Synthetic tab.",
            },
            {
                "label": "Open Gold Set workflow",
                "target_tab": "goldset",
                "reason": "Improve trusted anchors before generating more rows.",
            },
            {
                "label": "Open Data Prep",
                "target_tab": "dataprep",
                "reason": "Fix schema mapping before synthetic data mirrors the wrong shape.",
            },
        ],
        "power_details": {
            "domain_detection": {
                "verdict": domain.get("verdict"),
                "evidence": domain.get("evidence", []),
                "issues": domain.get("issues", []),
            },
            "mapping": {
                "verdict": mapping.get("verdict"),
                "summary": mapping.get("summary"),
                "issues": mapping.get("issues", []),
            },
            "gold_set": {
                "verdict": gold.get("verdict"),
                "validation": gold.get("validation"),
                "coverage": gold.get("coverage"),
            },
            "synthetic_playbooks": {
                "verdict": playbooks.get("verdict"),
                "catalog": playbooks.get("catalog"),
                "review_queue": playbooks.get("review_queue"),
                "issues": playbooks.get("issues", []),
            },
        },
    }


def _synthetic_quality_status(row: dict[str, Any]) -> str:
    review_status = str(row.get("review_status") or "").strip().lower()
    if review_status:
        return review_status
    legacy_status = str(row.get("status") or "").strip().lower()
    if legacy_status in {"pending", "accepted", "rejected"}:
        return legacy_status
    return "accepted"


def _synthetic_quality_source(row: dict[str, Any]) -> str:
    source = str(row.get("synth_source") or "").strip()
    if source:
        return source
    generator = str(row.get("generator") or row.get("source") or "").strip()
    return generator or "manual_synthetic"


def _synthetic_quality_confidence(row: dict[str, Any]) -> float | None:
    value = row.get("synth_confidence")
    if value is None:
        return None
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return None


_SYNTHETIC_QUALITY_METADATA_FIELDS = {
    "id",
    "row_id",
    "uuid",
    "source",
    "synth_source",
    "generator",
    "provider",
    "model",
    "synth_confidence",
    "confidence",
    "score",
    "review_status",
    "status",
    "created_at",
    "updated_at",
    "metadata",
}


_SYNTHETIC_QUALITY_CONTENT_FIELDS = (
    "instruction",
    "input",
    "question",
    "prompt",
    "query",
    "context",
    "answer",
    "response",
    "output",
    "completion",
    "expected_answer",
    "policy_answer",
    "text",
    "label",
    "category",
    "class",
)


_SYNTHETIC_QUALITY_PRIMARY_TEXT_FIELDS = (
    "instruction",
    "input",
    "question",
    "prompt",
    "query",
    "context",
    "answer",
    "response",
    "output",
    "completion",
    "expected_answer",
    "policy_answer",
    "text",
)


def _synthetic_quality_trainable_text(row: dict[str, Any], required_fields: list[str]) -> str:
    values: list[str] = []

    for field in required_fields:
        value = row.get(field)
        if _field_has_value(value):
            _flatten_text_values(value, values, limit=20)

    if not values:
        for field in _SYNTHETIC_QUALITY_CONTENT_FIELDS:
            value = row.get(field)
            if _field_has_value(value):
                _flatten_text_values(value, values, limit=20)

    if not values:
        for field, value in row.items():
            if field in _SYNTHETIC_QUALITY_METADATA_FIELDS or not _field_has_value(value):
                continue
            _flatten_text_values(value, values, limit=20)

    return " ".join(values).strip()


def _synthetic_quality_primary_text(row: dict[str, Any]) -> str:
    values: list[str] = []
    for field in _SYNTHETIC_QUALITY_PRIMARY_TEXT_FIELDS:
        value = row.get(field)
        if _field_has_value(value):
            _flatten_text_values(value, values, limit=20)
    return " ".join(values).strip()


def _synthetic_quality_rows_from_datasets(datasets: list[Dataset], project_id: int, *, limit: int = 1200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    candidate_paths: list[tuple[str, str]] = []
    for dataset in datasets:
        token = str(dataset.file_path or "").strip()
        if token:
            candidate_paths.append((token, dataset.name or "Synthetic"))
    canonical_path = settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"
    candidate_paths.append((str(canonical_path), "Synthetic"))

    for file_path, dataset_name in candidate_paths:
        if len(rows) >= limit or not file_path or file_path in seen_paths:
            continue
        seen_paths.add(file_path)
        for row_index, row in enumerate(_load_jsonl_dicts(file_path, limit=max(1, limit - len(rows)))):
            rows.append({
                "row": row,
                "source": _synthetic_quality_source(row),
                "dataset_name": dataset_name,
                "status": _synthetic_quality_status(row),
                "confidence": _synthetic_quality_confidence(row),
                "file_path": file_path,
                "row_index": row_index,
                "text": _quality_scan_text(row),
            })
            if len(rows) >= limit:
                break
    return rows


def _synthetic_quality_gold_texts(gold: dict[str, Any], datasets: list[Dataset]) -> list[str]:
    texts: list[str] = []
    for sample in list(gold.get("trusted_examples") or []):
        if not isinstance(sample, dict):
            continue
        for key in ("input_preview", "expected_preview"):
            token = str(sample.get(key) or "").strip()
            if token:
                texts.append(token)
    for dataset in datasets:
        if dataset.dataset_type not in _GOLD_SET_DATASET_TYPES:
            continue
        for row in _load_jsonl_dicts(dataset.file_path, limit=1000):
            text = _quality_scan_text(row)
            if text:
                texts.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        fingerprint = _quality_text_fingerprint(text)
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(text)
        if len(deduped) >= 1000:
            break
    return deduped


def _synthetic_quality_similarity(left: str, right: str) -> float:
    left_tokens = _quality_tokens(left)
    right_tokens = _quality_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _synthetic_quality_best_gold_similarity(text: str, gold_texts: list[str]) -> float:
    if not text or not gold_texts:
        return 0.0
    best = 0.0
    for gold_text in gold_texts[:500]:
        score = _synthetic_quality_similarity(text, gold_text)
        if score > best:
            best = score
        if best >= 0.98:
            break
    return round(best, 4)


def _synthetic_quality_finding(
    finding_id: str,
    label: str,
    severity: IssueSeverity,
    status: str,
    message: str,
    *,
    count: int,
    target_tab: str,
    owner: str,
    evidence: list[str] | None = None,
    action_label: str | None = None,
) -> dict[str, Any]:
    return {
        "id": finding_id,
        "label": label,
        "severity": severity,
        "status": status,
        "message": message,
        "count": int(count),
        "target_tab": target_tab,
        "workflow_owner": owner,
        "evidence": list(evidence or [])[:6],
        "action_label": action_label or "Open workflow",
    }


def _synthetic_quality_source_groups(
    rows: list[dict[str, Any]],
    *,
    duplicate_indices: set[int],
    missing_by_index: dict[int, list[str]],
    gold_similarity_by_index: dict[int, float],
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(rows):
        source = str(item.get("source") or "manual_synthetic")
        bucket = buckets.setdefault(source, {
            "key": _domain_setup_slug(source),
            "source": source,
            "count": 0,
            "pending": 0,
            "accepted": 0,
            "rejected": 0,
            "other_status": 0,
            "low_confidence": 0,
            "unknown_confidence": 0,
            "missing_required": 0,
            "duplicate_signal_count": 0,
            "confidence_total": 0.0,
            "confidence_count": 0,
            "gold_similarity_total": 0.0,
            "target_tab": "synthetic",
        })
        bucket["count"] += 1
        status = str(item.get("status") or "accepted")
        if status in {"pending", "accepted", "rejected"}:
            bucket[status] += 1
        else:
            bucket["other_status"] += 1
        confidence = item.get("confidence")
        if confidence is None:
            bucket["unknown_confidence"] += 1
        else:
            bucket["confidence_total"] += float(confidence)
            bucket["confidence_count"] += 1
            if float(confidence) < 0.65:
                bucket["low_confidence"] += 1
        if missing_by_index.get(index):
            bucket["missing_required"] += 1
        if index in duplicate_indices:
            bucket["duplicate_signal_count"] += 1
        bucket["gold_similarity_total"] += float(gold_similarity_by_index.get(index) or 0.0)

    groups: list[dict[str, Any]] = []
    for bucket in buckets.values():
        confidence_count = int(bucket.pop("confidence_count") or 0)
        confidence_total = float(bucket.pop("confidence_total") or 0.0)
        gold_similarity_total = float(bucket.pop("gold_similarity_total") or 0.0)
        count = max(1, int(bucket.get("count") or 0))
        bucket["avg_confidence"] = round(confidence_total / confidence_count, 4) if confidence_count else None
        bucket["avg_gold_similarity"] = round(gold_similarity_total / count, 4)
        groups.append(bucket)
    groups.sort(
        key=lambda item: (
            -int(item.get("pending") or 0),
            -int(item.get("missing_required") or 0),
            -int(item.get("low_confidence") or 0),
            -int(item.get("count") or 0),
            str(item.get("source") or ""),
        )
    )
    return groups[:12]


async def build_data_studio_synthetic_quality_analytics(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return read-only deterministic synthetic quality analytics."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    domain = await build_data_studio_domain_detection(db, project_id)
    mapping = await build_data_studio_mapping_preview(db, project_id)
    gold = await build_data_studio_gold_set_workbench(db, project_id)
    review_queue = _review_queue_summary(await list_review_queue(db, project_id))

    datasets_result = await db.execute(
        select(Dataset)
        .where(Dataset.project_id == project_id)
        .order_by(Dataset.updated_at.desc(), Dataset.id.asc())
    )
    datasets = list(datasets_result.scalars().all())
    synthetic_datasets = [dataset for dataset in datasets if dataset.dataset_type == DatasetType.SYNTHETIC]
    rows = _synthetic_quality_rows_from_datasets(synthetic_datasets, project_id)

    detected = domain.get("detected_domain") if isinstance(domain.get("detected_domain"), dict) else {}
    domain_id = str(detected.get("id") or "generic_domain")
    domain_label = str(detected.get("label") or "Generic Domain")
    domain_confidence = float(detected.get("confidence") or 0.0)
    recipe_payload = _recipe_payload(project)
    required_fields = _synthetic_required_fields(recipe_payload, domain_id) if recipe_payload else []
    gold_texts = _synthetic_quality_gold_texts(gold, datasets)

    status_counts: Counter[str] = Counter()
    confidence_buckets: Counter[str] = Counter()
    confidence_total = 0.0
    confidence_count = 0
    missing_by_index: dict[int, list[str]] = {}
    low_quality_by_index: dict[int, str] = {}
    fingerprints: dict[str, list[int]] = {}
    gold_similarity_by_index: dict[int, float] = {}
    high_gold_similarity = 0
    low_gold_similarity = 0

    for index, item in enumerate(rows):
        row = item.get("row") if isinstance(item.get("row"), dict) else {}
        status_counts[str(item.get("status") or "accepted")] += 1
        confidence = item.get("confidence")
        if confidence is None:
            confidence_buckets["unknown"] += 1
        else:
            parsed_confidence = float(confidence)
            confidence_total += parsed_confidence
            confidence_count += 1
            if parsed_confidence >= 0.85:
                confidence_buckets["high"] += 1
            elif parsed_confidence >= 0.65:
                confidence_buckets["medium"] += 1
            else:
                confidence_buckets["low"] += 1

        missing_fields = [
            field
            for field in required_fields
            if not _field_has_value(row.get(field))
        ]
        if missing_fields:
            missing_by_index[index] = missing_fields

        text = _synthetic_quality_trainable_text(row, required_fields) or str(item.get("text") or "")
        primary_text = _synthetic_quality_primary_text(row) or text
        low_quality_reason = _low_quality_reason(primary_text)
        if low_quality_reason:
            low_quality_by_index[index] = low_quality_reason

        fingerprint = _quality_text_fingerprint(text)
        if fingerprint:
            fingerprints.setdefault(fingerprint, []).append(index)

        similarity = _synthetic_quality_best_gold_similarity(text, gold_texts)
        gold_similarity_by_index[index] = similarity
        if gold_texts and similarity >= 0.82:
            high_gold_similarity += 1
        elif gold_texts and similarity < 0.12:
            low_gold_similarity += 1

    duplicate_indices: set[int] = set()
    exact_duplicate_count = 0
    for indices in fingerprints.values():
        if len(indices) <= 1:
            continue
        exact_duplicate_count += len(indices) - 1
        duplicate_indices.update(indices)

    near_duplicate_pairs = 0
    comparable_limit = min(len(rows), 220)
    for left in range(comparable_limit):
        left_text = str(rows[left].get("text") or "")
        if not left_text:
            continue
        for right in range(left + 1, comparable_limit):
            if left in duplicate_indices and right in duplicate_indices:
                continue
            right_text = str(rows[right].get("text") or "")
            if not right_text:
                continue
            similarity = _synthetic_quality_similarity(left_text, right_text)
            if similarity >= 0.9:
                near_duplicate_pairs += 1
                duplicate_indices.update({left, right})

    total_rows = len(rows)
    pending_count = int(status_counts.get("pending", 0))
    accepted_count = int(status_counts.get("accepted", 0))
    rejected_count = int(status_counts.get("rejected", 0))
    low_confidence_count = int(confidence_buckets.get("low", 0))
    unknown_confidence_count = int(confidence_buckets.get("unknown", 0))
    missing_required_count = len(missing_by_index)
    low_quality_count = len(low_quality_by_index)
    duplicate_signal_count = len(duplicate_indices)
    avg_confidence = round(confidence_total / confidence_count, 4) if confidence_count else None
    avg_gold_similarity = (
        round(sum(gold_similarity_by_index.values()) / max(1, total_rows), 4)
        if total_rows
        else 0.0
    )

    findings: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    if total_rows <= 0:
        issues.append(
            _issue(
                "synthetic_quality_no_rows",
                "info",
                "No synthetic rows yet",
                "Generate synthetic rows before synthetic quality analytics can score sources and review outcomes.",
                action_label="Open Synthetic",
                target_tab="synthetic",
            )
        )
    if pending_count > 0:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_pending_review",
            "Pending synthetic review",
            "warning",
            "attention",
            f"{pending_count} synthetic row(s) are pending review before they can enter prepared datasets.",
            count=pending_count,
            target_tab="synthetic",
            owner="Synthetic Review",
            evidence=[
                f"{review_queue.get('pending_group_count', 0)} pending source group(s).",
                "Accept or reject rows in the existing Synthetic workflow.",
            ],
            action_label="Review synthetic rows",
        ))
    if missing_required_count > 0:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_missing_required_fields",
            "Missing required fields",
            "blocker",
            "blocked",
            f"{missing_required_count} synthetic row(s) are missing required recipe fields.",
            count=missing_required_count,
            target_tab="dataprep",
            owner="Data Prep",
            evidence=[f"Required fields: {', '.join(required_fields) or 'recipe not selected'}."],
            action_label="Review mapping",
        ))
    if duplicate_signal_count > 0:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_duplicates",
            "Duplicate or near-duplicate synthetic rows",
            "warning",
            "attention",
            f"Found {exact_duplicate_count} exact duplicate row(s) and {near_duplicate_pairs} near-duplicate pair(s).",
            count=duplicate_signal_count,
            target_tab="quality-safety",
            owner="Quality & Safety",
            evidence=["Repeated synthetic rows can inflate training volume without adding learning signal."],
            action_label="Open Quality & Safety",
        ))
    if low_confidence_count > 0 or unknown_confidence_count > 0:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_confidence_needs_review",
            "Synthetic confidence needs review",
            "warning",
            "attention",
            f"{low_confidence_count} row(s) have low confidence and {unknown_confidence_count} row(s) have no confidence score.",
            count=low_confidence_count + unknown_confidence_count,
            target_tab="synthetic",
            owner="Synthetic Review",
            evidence=["Low-confidence synthetic rows should be reviewed before SFT."],
            action_label="Review synthetic rows",
        ))
    if low_quality_count > 0:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_low_quality_text",
            "Low-quality synthetic text",
            "warning",
            "attention",
            f"{low_quality_count} synthetic row(s) look empty, placeholder-like, or too short.",
            count=low_quality_count,
            target_tab="synthetic",
            owner="Synthetic Review",
            evidence=sorted(set(low_quality_by_index.values()))[:4],
            action_label="Review synthetic rows",
        ))
    if total_rows > 0 and not gold_texts:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_gold_missing",
            "Gold Set anchors missing",
            "warning",
            "attention",
            "Synthetic rows cannot be compared with trusted Gold Set anchors yet.",
            count=total_rows,
            target_tab="goldset",
            owner="Gold Set",
            evidence=["Add trusted examples to estimate whether synthetic rows are close to the target domain."],
            action_label="Open Gold Set",
        ))
    elif low_gold_similarity > 0:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_gold_similarity_low",
            "Low Gold Set similarity",
            "warning",
            "attention",
            f"{low_gold_similarity} synthetic row(s) are far from trusted Gold Set wording.",
            count=low_gold_similarity,
            target_tab="goldset",
            owner="Gold Set",
            evidence=["Very low similarity can indicate domain drift or under-anchored generation."],
            action_label="Review Gold Set",
        ))
    if high_gold_similarity > 0:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_gold_similarity_high",
            "High Gold Set overlap",
            "info",
            "attention",
            f"{high_gold_similarity} synthetic row(s) are very close to trusted Gold Set examples.",
            count=high_gold_similarity,
            target_tab="synthetic",
            owner="Synthetic Review",
            evidence=["High overlap can be useful anchoring, but exact copies add less coverage."],
            action_label="Review synthetic rows",
        ))
    if domain_confidence < 0.45 and total_rows > 0:
        findings.append(_synthetic_quality_finding(
            "synthetic_quality_domain_weak",
            "Domain signal is weak",
            "info",
            "attention",
            "Synthetic analytics are less useful until the training domain is confirmed.",
            count=total_rows,
            target_tab="domain",
            owner="Domain Managers",
            evidence=[f"Detected domain confidence is {round(domain_confidence * 100)}%."],
            action_label="Review domain",
        ))

    for finding in findings:
        if finding.get("severity") in {"blocker", "warning", "info"}:
            issues.append(
                _issue(
                    str(finding["id"]),
                    str(finding["severity"]),
                    str(finding["label"]),
                    str(finding["message"]),
                    action_label=str(finding.get("action_label") or "Open workflow"),
                    target_tab=str(finding.get("target_tab") or "synthetic"),
                )
            )

    if total_rows <= 0:
        verdict: SyntheticQualityVerdict = "empty"
    elif any(item.get("severity") in {"blocker", "warning"} for item in findings):
        verdict = "attention"
    else:
        verdict = "ready"

    source_groups = _synthetic_quality_source_groups(
        rows,
        duplicate_indices=duplicate_indices,
        missing_by_index=missing_by_index,
        gold_similarity_by_index=gold_similarity_by_index,
    )
    status_groups = [
        {
            "status": status,
            "label": label,
            "count": int(status_counts.get(status, 0)),
            "target_tab": "synthetic",
        }
        for status, label in [
            ("pending", "Pending review"),
            ("accepted", "Accepted"),
            ("rejected", "Rejected"),
        ]
    ]
    status_groups.extend(
        {
            "status": status,
            "label": status.replace("_", " ").title(),
            "count": int(count),
            "target_tab": "synthetic",
        }
        for status, count in status_counts.items()
        if status not in {"pending", "accepted", "rejected"}
    )

    preview_rows = []
    for index in sorted(set(list(missing_by_index.keys()) + list(duplicate_indices) + list(low_quality_by_index.keys())))[:5]:
        if 0 <= index < len(rows):
            preview_rows.append(
                _quality_row_preview(
                    {
                        "row": rows[index].get("row"),
                        "source": rows[index].get("source"),
                        "source_type": "synthetic",
                        "target_tab": "synthetic",
                        "file_path": rows[index].get("file_path"),
                        "row_index": rows[index].get("row_index"),
                    },
                    reason="Synthetic quality analytics preview",
                )
            )

    return {
        "project_id": project_id,
        "verdict": verdict,
        "read_only": True,
        "auto_apply": False,
        "source_of_truth": "deterministic_synthetic_quality_checks",
        "domain": {
            "id": domain_id,
            "label": domain_label,
            "confidence": round(domain_confidence, 4),
            "source": detected.get("source"),
        },
        "recipe": recipe_payload,
        "summary": {
            "total_rows": total_rows,
            "pending_rows": pending_count,
            "accepted_rows": accepted_count,
            "rejected_rows": rejected_count,
            "source_count": len(source_groups),
            "avg_confidence": avg_confidence,
            "low_confidence_rows": low_confidence_count,
            "unknown_confidence_rows": unknown_confidence_count,
            "missing_required_rows": missing_required_count,
            "duplicate_signal_rows": duplicate_signal_count,
            "low_quality_rows": low_quality_count,
            "avg_gold_similarity": avg_gold_similarity,
            "high_gold_similarity_rows": high_gold_similarity,
            "low_gold_similarity_rows": low_gold_similarity,
            "gold_anchor_rows": len(gold_texts),
        },
        "quality_bands": {
            "confidence": {
                "high": int(confidence_buckets.get("high", 0)),
                "medium": int(confidence_buckets.get("medium", 0)),
                "low": low_confidence_count,
                "unknown": unknown_confidence_count,
                "average": avg_confidence,
            },
            "duplicates": {
                "exact_duplicate_rows": exact_duplicate_count,
                "near_duplicate_pairs": near_duplicate_pairs,
                "affected_rows": duplicate_signal_count,
                "ratio": round(duplicate_signal_count / max(1, total_rows), 4) if total_rows else 0.0,
            },
            "required_fields": {
                "required_fields": required_fields,
                "missing_rows": missing_required_count,
                "ratio": round(missing_required_count / max(1, total_rows), 4) if total_rows else 0.0,
            },
            "gold_similarity": {
                "average": avg_gold_similarity,
                "high_overlap_rows": high_gold_similarity,
                "low_similarity_rows": low_gold_similarity,
                "gold_anchor_rows": len(gold_texts),
            },
        },
        "review_outcomes": {
            "total_pending": int(review_queue.get("total_pending") or 0),
            "total_accepted": int(review_queue.get("total_accepted") or 0),
            "top_pending_groups": list(review_queue.get("top_pending_groups") or []),
            "top_accepted_groups": list(review_queue.get("top_accepted_groups") or []),
        },
        "source_groups": source_groups,
        "status_groups": status_groups,
        "domain_groups": [
            {
                "domain_id": domain_id,
                "domain_label": domain_label,
                "confidence": round(domain_confidence, 4),
                "synthetic_rows": total_rows,
                "pending_rows": pending_count,
                "accepted_rows": accepted_count,
                "source": detected.get("source"),
                "target_tab": "domain",
            }
        ],
        "findings": findings,
        "preview_rows": preview_rows,
        "issues": issues,
        "entry_points": [
            {
                "label": "Open Synthetic review",
                "target_tab": "synthetic",
                "reason": "Accept, reject, or inspect synthetic rows in the existing Synthetic workflow.",
            },
            {
                "label": "Open Review Queue",
                "target_tab": "review-queue",
                "reason": "Use Data Studio review triage for synthetic, Gold Set, and annotation review work.",
            },
            {
                "label": "Open Gold Set",
                "target_tab": "goldset",
                "reason": "Strengthen trusted anchors used for similarity and review decisions.",
            },
            {
                "label": "Open Data Prep",
                "target_tab": "dataprep",
                "reason": "Fix missing required fields or mapping shape before preparing datasets.",
            },
            {
                "label": "Open Quality & Safety",
                "target_tab": "quality-safety",
                "reason": "Inspect duplicate, missing-field, and low-quality deterministic checks.",
            },
        ],
        "assist": {
            "available": True,
            "read_only": True,
            "status": "not_invoked",
            "default_provider": "ollama",
            "supported_providers": ["ollama", "openai_compatible"],
            "purpose": "explanations_only",
            "message": "Synthetic quality analytics are deterministic by default; LLM assist may be used only to explain findings.",
        },
        "power_details": {
            "required_fields": required_fields,
            "status_counts": dict(status_counts),
            "confidence_buckets": dict(confidence_buckets),
            "mapping_verdict": mapping.get("verdict"),
            "gold_validation": gold.get("validation"),
            "synthetic_dataset_ids": [int(dataset.id) for dataset in synthetic_datasets],
        },
    }


def _review_triage_item(
    *,
    item_id: str,
    title: str,
    priority: str,
    count: int,
    message: str,
    action_label: str,
    target_tab: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": title,
        "priority": priority,
        "count": int(count),
        "message": message,
        "action_label": action_label,
        "target_tab": target_tab,
        "requires_user_confirmation": True,
        "evidence": [item for item in evidence if item][:5],
    }


def _review_group(
    *,
    key: str,
    label: str,
    kind: str,
    status: str,
    count: int,
    target_tab: str,
    synth_source: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": key,
        "label": label,
        "kind": kind,
        "status": status,
        "count": int(count),
        "target_tab": target_tab,
    }
    # Only the actionable synthetic-pending groups carry the raw
    # ``synth_source`` key — the Data Studio panel passes it back to the
    # bulk-update-by-source endpoint for one-click accept/reject-all.
    if synth_source is not None:
        entry["synth_source"] = synth_source
    return entry


async def _annotation_review_summary(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    jobs_result = await db.execute(
        select(LabelJob)
        .where(LabelJob.project_id == project_id)
        .order_by(LabelJob.updated_at.desc(), LabelJob.id.desc())
    )
    jobs = list(jobs_result.scalars().all())
    job_ids = [int(job.id) for job in jobs]
    rows_by_job: dict[int, list[LabelRow]] = {job_id: [] for job_id in job_ids}

    if job_ids:
        rows_result = await db.execute(
            select(LabelRow)
            .where(LabelRow.job_id.in_(job_ids))
            .order_by(LabelRow.job_id.asc(), LabelRow.id.asc())
        )
        for row in rows_result.scalars().all():
            rows_by_job.setdefault(int(row.job_id), []).append(row)

    totals = {
        "job_count": len(jobs),
        "active_jobs": 0,
        "paused_jobs": 0,
        "completed_jobs": 0,
        "total_rows": 0,
        "assigned": 0,
        "unlabeled": 0,
        "review_needed": 0,
        "labeled": 0,
        "labeled_unpromoted": 0,
        "promoted": 0,
    }
    job_payloads: list[dict[str, Any]] = []

    for job in jobs:
        rows = rows_by_job.get(int(job.id), [])
        total = len(rows)
        labeled = sum(1 for row in rows if row.labeled_at is not None)
        assigned = sum(
            1
            for row in rows
            if row.assigned_to is not None and row.labeled_at is None
        )
        promoted = sum(1 for row in rows if row.promoted_at is not None)
        unlabeled = max(0, total - labeled - assigned)
        labeled_unpromoted = max(0, labeled - promoted)
        status = str(job.status or "active").strip().lower()
        review_needed = assigned + (unlabeled if status == "active" else 0)

        if status == "active":
            totals["active_jobs"] += 1
        elif status == "paused":
            totals["paused_jobs"] += 1
        elif status == "completed":
            totals["completed_jobs"] += 1

        totals["total_rows"] += total
        totals["assigned"] += assigned
        totals["unlabeled"] += unlabeled
        totals["review_needed"] += review_needed
        totals["labeled"] += labeled
        totals["labeled_unpromoted"] += labeled_unpromoted
        totals["promoted"] += promoted

        job_payloads.append({
            "id": int(job.id),
            "name": job.name,
            "label_type": job.label_type,
            "status": status,
            "target_rows": job.target_rows,
            "total": total,
            "assigned": assigned,
            "unlabeled": unlabeled,
            "labeled": labeled,
            "labeled_unpromoted": labeled_unpromoted,
            "promoted": promoted,
            "review_needed": review_needed,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        })

    job_payloads.sort(
        key=lambda item: (
            -int(item.get("labeled_unpromoted") or 0),
            -int(item.get("review_needed") or 0),
            str(item.get("name") or ""),
        )
    )
    return {
        "totals": totals,
        "jobs": job_payloads[:8],
    }


def _status_group(
    *,
    status: str,
    label: str,
    count: int,
    target_tab: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "label": label,
        "count": int(count),
        "target_tab": target_tab,
        "kind": kind,
    }


async def build_data_studio_review_queue(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a read-only cross-workflow review queue summary."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    domain = await build_data_studio_domain_detection(db, project_id)
    detected = domain.get("detected_domain") if isinstance(domain.get("detected_domain"), dict) else {}
    gold = await build_data_studio_gold_set_workbench(db, project_id)
    synthetic_queue_raw = await list_review_queue(db, project_id)
    synthetic = _review_queue_summary(synthetic_queue_raw)
    annotation = await _annotation_review_summary(db, project_id)

    gold_totals = gold.get("totals") if isinstance(gold.get("totals"), dict) else {}
    annotation_totals = (
        annotation.get("totals") if isinstance(annotation.get("totals"), dict) else {}
    )
    synthetic_pending = int(synthetic.get("total_pending") or 0)
    synthetic_accepted = int(synthetic.get("total_accepted") or 0)
    gold_review_needed = int(gold_totals.get("review_needed") or 0)
    gold_trusted = int(gold_totals.get("trusted_examples") or 0)
    annotation_assigned = int(annotation_totals.get("assigned") or 0)
    annotation_unlabeled = int(annotation_totals.get("unlabeled") or 0)
    annotation_review_needed = int(annotation_totals.get("review_needed") or 0)
    annotation_labeled_unpromoted = int(annotation_totals.get("labeled_unpromoted") or 0)
    annotation_promoted = int(annotation_totals.get("promoted") or 0)
    open_review_items = (
        synthetic_pending
        + gold_review_needed
        + annotation_review_needed
        + annotation_labeled_unpromoted
    )
    accepted_or_promoted = synthetic_accepted + gold_trusted + annotation_promoted

    issues: list[dict[str, str]] = []
    triage: list[dict[str, Any]] = []

    if synthetic_pending > 0:
        issues.append(
            _issue(
                "review_queue_synthetic_pending",
                "warning",
                "Synthetic rows need review",
                f"{synthetic_pending} synthetic row(s) are pending accept/reject before dataset prep can use them.",
                action_label="Review synthetic rows",
                target_tab="synthetic",
            )
        )
        top_sources = [
            f"{group.get('synth_source')} ({group.get('count')})"
            for group in list(synthetic.get("top_pending_groups") or [])[:3]
            if isinstance(group, dict)
        ]
        triage.append(
            _review_triage_item(
                item_id="review_pending_synthetic_rows",
                title="Review pending synthetic rows",
                priority="high",
                count=synthetic_pending,
                message="Accept good rows or reject weak rows before they enter the next prepared dataset.",
                action_label="Open Synthetic review",
                target_tab="synthetic",
                evidence=top_sources,
            )
        )

    if gold_review_needed > 0:
        issues.append(
            _issue(
                "review_queue_gold_needs_review",
                "warning",
                "Gold Set rows need review",
                f"{gold_review_needed} Gold Set row(s) are pending, in review, or waiting on changes.",
                action_label="Review Gold Set",
                target_tab="goldset",
            )
        )
        triage.append(
            _review_triage_item(
                item_id="review_gold_set_rows",
                title="Review Gold Set rows",
                priority="high",
                count=gold_review_needed,
                message="Trusted evaluation examples should be reviewed before training and eval decisions depend on them.",
                action_label="Open Gold Set",
                target_tab="goldset",
                evidence=[
                    f"{gold_trusted} trusted Gold Set example(s) are ready.",
                    f"{int(gold_totals.get('queue_pending') or 0)} assigned queue item(s) are pending.",
                ],
            )
        )

    if annotation_labeled_unpromoted > 0:
        issues.append(
            _issue(
                "review_queue_annotation_labeled_unpromoted",
                "warning",
                "Labeled annotation rows are not promoted",
                f"{annotation_labeled_unpromoted} labeled annotation row(s) are waiting to be promoted into a training dataset.",
                action_label="Open Annotation",
                target_tab="annotate",
            )
        )
        triage.append(
            _review_triage_item(
                item_id="promote_labeled_annotation_rows",
                title="Promote labeled annotation rows",
                priority="medium",
                count=annotation_labeled_unpromoted,
                message="Labels remain advisory until the annotation workflow promotes them into synthetic or gold data.",
                action_label="Open Annotation",
                target_tab="annotate",
                evidence=[
                    f"{int(annotation_totals.get('labeled') or 0)} labeled row(s).",
                    f"{annotation_promoted} promoted row(s).",
                ],
            )
        )

    if annotation_review_needed > 0:
        issues.append(
            _issue(
                "review_queue_annotation_work_open",
                "warning",
                "Annotation jobs have open review work",
                f"{annotation_review_needed} annotation row(s) are assigned or waiting for labels.",
                action_label="Continue annotation",
                target_tab="annotate",
            )
        )
        triage.append(
            _review_triage_item(
                item_id="continue_annotation_review",
                title="Continue annotation review",
                priority="medium",
                count=annotation_review_needed,
                message="Finish assigned and unlabeled rows so the labels can be reviewed and promoted.",
                action_label="Open Annotation",
                target_tab="annotate",
                evidence=[
                    f"{annotation_assigned} assigned row(s).",
                    f"{annotation_unlabeled} unassigned row(s).",
                ],
            )
        )

    if open_review_items <= 0 and accepted_or_promoted > 0:
        triage.append(
            _review_triage_item(
                item_id="review_gates_clear",
                title="Review gates are clear",
                priority="low",
                count=accepted_or_promoted,
                message="No open review queues are blocking the next dataset prep step.",
                action_label="Open Data Prep",
                target_tab="dataprep",
                evidence=[
                    f"{synthetic_accepted} accepted synthetic row(s).",
                    f"{gold_trusted} trusted Gold Set example(s).",
                    f"{annotation_promoted} promoted annotation row(s).",
                ],
            )
        )

    if open_review_items <= 0 and accepted_or_promoted <= 0:
        issues.append(
            _issue(
                "review_queue_no_review_sources",
                "info",
                "No review queue yet",
                "Generate synthetic rows, add Gold Set examples, or create annotation jobs to start review flow.",
                action_label="Open Synthetic",
                target_tab="synthetic",
            )
        )
        triage.append(
            _review_triage_item(
                item_id="create_review_source",
                title="Create a review source",
                priority="low",
                count=0,
                message="Review queues appear after synthetic generation, Gold Set sampling, or annotation seeding.",
                action_label="Open Synthetic",
                target_tab="synthetic",
                evidence=[],
            )
        )

    by_source: list[dict[str, Any]] = []
    for group in list(synthetic_queue_raw.get("groups") or []):
        if not isinstance(group, dict):
            continue
        by_source.append(
            _review_group(
                key=f"synthetic:pending:{group.get('synth_source')}",
                label=str(group.get("synth_source") or "Synthetic"),
                kind="synthetic",
                status="pending",
                count=int(group.get("count") or 0),
                target_tab="synthetic",
                synth_source=str(group.get("synth_source") or ""),
            )
        )
    for group in list(synthetic_queue_raw.get("accepted_groups") or []):
        if not isinstance(group, dict):
            continue
        by_source.append(
            _review_group(
                key=f"synthetic:accepted:{group.get('synth_source')}",
                label=str(group.get("synth_source") or "Synthetic"),
                kind="synthetic",
                status="accepted",
                count=int(group.get("count") or 0),
                target_tab="synthetic",
            )
        )
    for dataset in list(gold.get("datasets") or []):
        if not isinstance(dataset, dict):
            continue
        review_needed = int(dataset.get("review_needed") or 0)
        trusted = int(dataset.get("trusted_examples") or 0)
        if review_needed > 0:
            by_source.append(
                _review_group(
                    key=f"gold:{dataset.get('id')}:review",
                    label=str(dataset.get("name") or "Gold Set"),
                    kind="gold_set",
                    status="needs_review",
                    count=review_needed,
                    target_tab="goldset",
                )
            )
        if trusted > 0:
            by_source.append(
                _review_group(
                    key=f"gold:{dataset.get('id')}:trusted",
                    label=str(dataset.get("name") or "Gold Set"),
                    kind="gold_set",
                    status="trusted",
                    count=trusted,
                    target_tab="goldset",
                )
            )
    for job in list(annotation.get("jobs") or []):
        if not isinstance(job, dict):
            continue
        review_needed = int(job.get("review_needed") or 0)
        labeled_unpromoted = int(job.get("labeled_unpromoted") or 0)
        promoted = int(job.get("promoted") or 0)
        if review_needed > 0:
            by_source.append(
                _review_group(
                    key=f"annotation:{job.get('id')}:review",
                    label=str(job.get("name") or "Annotation job"),
                    kind="annotation",
                    status="needs_labeling",
                    count=review_needed,
                    target_tab="annotate",
                )
            )
        if labeled_unpromoted > 0:
            by_source.append(
                _review_group(
                    key=f"annotation:{job.get('id')}:promotion",
                    label=str(job.get("name") or "Annotation job"),
                    kind="annotation",
                    status="needs_promotion",
                    count=labeled_unpromoted,
                    target_tab="annotate",
                )
            )
        if promoted > 0:
            by_source.append(
                _review_group(
                    key=f"annotation:{job.get('id')}:promoted",
                    label=str(job.get("name") or "Annotation job"),
                    kind="annotation",
                    status="promoted",
                    count=promoted,
                    target_tab="annotate",
                )
            )

    by_source.sort(key=lambda item: (-int(item["count"]), item["kind"], item["label"]))
    by_status = [
        _status_group(
            status="synthetic_pending",
            label="Synthetic pending review",
            count=synthetic_pending,
            target_tab="synthetic",
            kind="synthetic",
        ),
        _status_group(
            status="synthetic_accepted",
            label="Synthetic accepted",
            count=synthetic_accepted,
            target_tab="synthetic",
            kind="synthetic",
        ),
        _status_group(
            status="gold_review_needed",
            label="Gold Set review needed",
            count=gold_review_needed,
            target_tab="goldset",
            kind="gold_set",
        ),
        _status_group(
            status="gold_trusted",
            label="Gold Set trusted",
            count=gold_trusted,
            target_tab="goldset",
            kind="gold_set",
        ),
        _status_group(
            status="annotation_review_needed",
            label="Annotation review needed",
            count=annotation_review_needed,
            target_tab="annotate",
            kind="annotation",
        ),
        _status_group(
            status="annotation_needs_promotion",
            label="Annotation needs promotion",
            count=annotation_labeled_unpromoted,
            target_tab="annotate",
            kind="annotation",
        ),
        _status_group(
            status="annotation_promoted",
            label="Annotation promoted",
            count=annotation_promoted,
            target_tab="annotate",
            kind="annotation",
        ),
    ]

    if open_review_items <= 0 and accepted_or_promoted <= 0:
        verdict: ReviewQueueVerdict = "empty"
    elif any(item["severity"] in {"blocker", "warning"} for item in issues):
        verdict = "attention"
    else:
        verdict = "ready"

    return {
        "project_id": project_id,
        "verdict": verdict,
        "read_only": True,
        "auto_apply": False,
        "source_of_truth": "deterministic_data_studio_checks",
        "domain": {
            "id": str(detected.get("id") or "generic_domain"),
            "label": str(detected.get("label") or "Generic Domain"),
            "confidence": round(float(detected.get("confidence") or 0.0), 4),
            "source": detected.get("source"),
        },
        "totals": {
            "open_review_items": open_review_items,
            "accepted_or_promoted": accepted_or_promoted,
            "synthetic_pending": synthetic_pending,
            "synthetic_accepted": synthetic_accepted,
            "gold_review_needed": gold_review_needed,
            "gold_trusted_examples": gold_trusted,
            "annotation_jobs": int(annotation_totals.get("job_count") or 0),
            "annotation_review_needed": annotation_review_needed,
            "annotation_labeled": int(annotation_totals.get("labeled") or 0),
            "annotation_labeled_unpromoted": annotation_labeled_unpromoted,
            "annotation_promoted": annotation_promoted,
        },
        "synthetic": synthetic,
        "gold_set": {
            "validation": gold.get("validation"),
            "totals": gold_totals,
            "datasets": list(gold.get("datasets") or [])[:4],
        },
        "annotation": annotation,
        "triage": sorted(
            triage,
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2}.get(str(item.get("priority")), 9),
                -int(item.get("count") or 0),
                str(item.get("id") or ""),
            ),
        )[:6],
        "groupings": {
            "by_source": by_source[:12],
            "by_status": by_status,
            "by_domain": [
                {
                    "domain_id": str(detected.get("id") or "generic_domain"),
                    "domain_label": str(detected.get("label") or "Generic Domain"),
                    "confidence": round(float(detected.get("confidence") or 0.0), 4),
                    "open_review_items": open_review_items,
                    "accepted_or_promoted": accepted_or_promoted,
                    "source": detected.get("source"),
                }
            ],
        },
        "issues": issues,
        "entry_points": [
            {
                "label": "Open Synthetic review",
                "target_tab": "synthetic",
                "reason": "Accept or reject pending synthetic rows in the existing Synthetic tab.",
            },
            {
                "label": "Open Gold Set review",
                "target_tab": "goldset",
                "reason": "Review trusted examples and Gold Set workbench rows in the existing Gold Set workflow.",
            },
            {
                "label": "Open Annotation workspace",
                "target_tab": "annotate",
                "reason": "Label, skip, or promote annotation rows in the existing Annotation workspace.",
            },
            {
                "label": "Open Eval active learning",
                "target_tab": "eval",
                "reason": "Review failed eval rows that can be promoted into training data.",
            },
        ],
        "power_details": {
            "domain_detection": {
                "verdict": domain.get("verdict"),
                "evidence": domain.get("evidence", []),
            },
            "synthetic_review_queue": synthetic,
            "gold_set_validation": gold.get("validation"),
            "annotation_totals": annotation_totals,
        },
    }


def _assist_default_endpoint(provider: str, api_url: str | None) -> str:
    explicit = str(api_url or "").strip()
    if explicit:
        return explicit
    if provider == "ollama":
        return "http://localhost:11434/v1/chat/completions"
    return str(settings.TEACHER_MODEL_API_URL or "").strip()


def _redact_endpoint(endpoint: str) -> str:
    token = str(endpoint or "").strip()
    if not token:
        return ""
    return token.split("?", 1)[0]


def _jsonable_compact(value: Any, *, max_chars: int = 16000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, indent=2)
    except TypeError:
        text = json.dumps(str(value), ensure_ascii=True)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...<truncated>"


def _assist_context_for_prompt(focus: AssistFocus, context: dict[str, Any]) -> dict[str, Any]:
    if focus == "mapping":
        return {
            "recipe": context.get("recipe"),
            "source": context.get("source"),
            "effective_mapping": context.get("effective_mapping"),
            "mapping_summary": context.get("summary"),
            "issues": context.get("issues"),
            "preview_rows": context.get("preview_rows"),
            "deterministic_suggestions": (
                context.get("diagnostics", {}).get("auto_fix_suggestions", [])
                if isinstance(context.get("diagnostics"), dict)
                else []
            ),
        }
    return {
        "recipe": context.get("recipe"),
        "source": context.get("source"),
        "detected_domain": context.get("detected_domain"),
        "applied_domain": context.get("applied"),
        "evidence": context.get("evidence"),
        "issues": context.get("issues"),
        "suggested_actions": context.get("suggested_actions"),
        "risks": context.get("risks"),
        "power_details": context.get("power_details"),
    }


def _build_assist_prompt(focus: AssistFocus, context: dict[str, Any]) -> str:
    focus_label = "schema mapping" if focus == "mapping" else "domain detection"
    return (
        "You are BrewSLM Data Studio LLM assist. Deterministic checks are the "
        "source of truth. Do not claim that you changed data, saved mappings, "
        "assigned a domain, generated rows, or applied any setting. Produce "
        "reviewable suggestions only.\n\n"
        f"Focus: {focus_label}\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "summary": "short beginner-friendly explanation",\n'
        '  "suggestions": [\n'
        "    {\n"
        '      "id": "stable-id",\n'
        '      "type": "mapping|domain|coverage|risk|synthetic|review",\n'
        '      "title": "short title",\n'
        '      "confidence": 0.0,\n'
        '      "rationale": "why this suggestion follows from the evidence",\n'
        '      "evidence": ["specific signal from the context"],\n'
        '      "suggested_field_mapping": {"canonical_field": "raw_field"},\n'
        '      "target_tab": "data|dataprep|goldset|synthetic",\n'
        '      "requires_user_confirmation": true\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Keep confidence between 0 and 1.\n"
        "- Omit suggested_field_mapping unless the focus is mapping and the mapping is explicit.\n"
        "- Include concrete evidence; avoid generic advice.\n"
        "- Never auto-apply anything.\n\n"
        "Deterministic Data Studio context:\n"
        f"{_jsonable_compact(_assist_context_for_prompt(focus, context))}"
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [match.group(1).strip() for match in _JSON_FENCE_RE.finditer(raw)]
    candidates.append(raw)
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _safe_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, parsed)), 4)


def _normalize_assist_suggestions(raw: Any, *, focus: AssistFocus) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    suggestions: list[dict[str, Any]] = []
    for index, item in enumerate(raw[:6]):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        rationale = str(item.get("rationale") or item.get("reason") or "").strip()
        if not title and not rationale:
            continue
        evidence_raw = item.get("evidence")
        evidence = [
            str(part).strip()
            for part in (evidence_raw if isinstance(evidence_raw, list) else [])
            if str(part).strip()
        ]
        suggestion: dict[str, Any] = {
            "id": str(item.get("id") or f"{focus}_assist_{index + 1}").strip(),
            "type": str(item.get("type") or focus).strip() or focus,
            "title": title or f"{focus.title()} suggestion",
            "confidence": _safe_confidence(item.get("confidence")),
            "rationale": rationale,
            "evidence": evidence[:6],
            "target_tab": str(item.get("target_tab") or ("dataprep" if focus == "mapping" else "data")).strip(),
            "requires_user_confirmation": True,
        }
        mapping = item.get("suggested_field_mapping")
        if focus == "mapping" and isinstance(mapping, dict):
            cleaned_mapping = {
                str(key).strip(): str(value).strip()
                for key, value in mapping.items()
                if str(key).strip() and str(value).strip()
            }
            if cleaned_mapping:
                suggestion["suggested_field_mapping"] = cleaned_mapping
        suggestions.append(suggestion)
    return suggestions


def _assist_unavailable_payload(
    *,
    project_id: int,
    focus: AssistFocus,
    provider: AssistProvider,
    endpoint: str,
    model_name: str,
    message: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "focus": focus,
        "status": "unavailable",
        "provider": {
            "provider": provider,
            "api_url": _redact_endpoint(endpoint),
            "model_name": model_name,
            "api_key_configured": False,
        },
        "source_of_truth": "deterministic_data_studio_checks",
        "auto_apply": False,
        "summary": message,
        "suggestions": [],
        "deterministic_context": {
            "verdict": context.get("verdict"),
            "issues": context.get("issues", []),
        },
        "warnings": [
            "LLM assist is optional. Deterministic Data Studio checks remain available.",
        ],
    }


async def build_data_studio_llm_assist(
    db: AsyncSession,
    project_id: int,
    *,
    focus: AssistFocus,
    provider: AssistProvider = "ollama",
    api_url: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run optional LLM assist over deterministic Data Studio context."""

    if focus == "mapping":
        context = await build_data_studio_mapping_preview(db, project_id)
    elif focus == "domain":
        context = await build_data_studio_domain_detection(db, project_id)
    else:
        raise ValueError(f"Unsupported assist focus '{focus}'")

    normalized_provider = "ollama" if provider == "ollama" else "openai_compatible"
    endpoint = _assist_default_endpoint(normalized_provider, api_url)
    resolved_model = str(model_name or "").strip() or "llama3"
    if not endpoint:
        return _assist_unavailable_payload(
            project_id=project_id,
            focus=focus,
            provider=normalized_provider,
            endpoint=endpoint,
            model_name=resolved_model,
            message="No LLM endpoint is configured for Data Studio assist.",
            context=context,
        )

    prompt = _build_assist_prompt(focus, context)
    system_prompt = (
        "You are a careful data-preparation assistant for BrewSLM. "
        "Return compact JSON only. Do not mutate data or imply that "
        "changes were applied."
    )
    try:
        result = await call_teacher_model(
            prompt=prompt,
            system_prompt=system_prompt,
            api_url=endpoint,
            api_key=str(api_key or ""),
            model_name=resolved_model,
            temperature=0.2,
            max_tokens=1800,
            force_json=True,
        )
    except Exception as exc:  # noqa: BLE001
        return _assist_unavailable_payload(
            project_id=project_id,
            focus=focus,
            provider=normalized_provider,
            endpoint=endpoint,
            model_name=resolved_model,
            message=f"LLM assist is unavailable: {str(exc)[:240]}",
            context=context,
        )

    content = str(result.get("content") or "")
    parsed = _extract_json_object(content)
    if not parsed:
        return {
            "project_id": project_id,
            "focus": focus,
            "status": "invalid_response",
            "provider": {
                "provider": normalized_provider,
                "api_url": _redact_endpoint(endpoint),
                "model_name": str(result.get("model") or resolved_model),
                "api_key_configured": bool(str(api_key or "").strip()),
            },
            "source_of_truth": "deterministic_data_studio_checks",
            "auto_apply": False,
            "summary": "The assistant responded, but not in the expected JSON format.",
            "suggestions": [],
            "deterministic_context": {
                "verdict": context.get("verdict"),
                "issues": context.get("issues", []),
            },
            "warnings": [
                "No changes were applied. Rerun assist or inspect deterministic checks.",
            ],
        }

    suggestions = _normalize_assist_suggestions(parsed.get("suggestions"), focus=focus)
    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        summary = "The assistant produced reviewable suggestions from the deterministic Data Studio context."

    return {
        "project_id": project_id,
        "focus": focus,
        "status": "ok",
        "provider": {
            "provider": normalized_provider,
            "api_url": _redact_endpoint(endpoint),
            "model_name": str(result.get("model") or resolved_model),
            "api_key_configured": bool(str(api_key or "").strip()),
            "tokens_used": int(result.get("tokens_used") or 0),
        },
        "source_of_truth": "deterministic_data_studio_checks",
        "auto_apply": False,
        "summary": summary,
        "suggestions": suggestions,
        "deterministic_context": {
            "verdict": context.get("verdict"),
            "issues": context.get("issues", []),
        },
        "warnings": [
            "LLM suggestions are advisory and require user confirmation.",
            "Deterministic Data Studio checks remain the source of truth.",
        ],
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


def _mapping_template_slug(value: Any) -> str:
    token = str(value or "").strip().lower().replace("_", "-")
    token = re.sub(r"[^a-z0-9-]+", "-", token)
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or "mapping-template"


def _mapping_detected_field_rows(raw_field_frequency: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"field": str(field), "count": int(count or 0)}
        for field, count in raw_field_frequency.items()
        if str(field).strip()
    ]
    rows.sort(key=lambda item: (-int(item["count"]), str(item["field"])))
    return rows[:16]


def _mapping_schema_hint(contract: dict[str, Any]) -> dict[str, Any]:
    schema_hint = contract.get("schema_hint")
    return schema_hint if isinstance(schema_hint, dict) else {}


def _mapping_output_fields(schema_hint: dict[str, Any], recipe_payload: dict[str, Any] | None) -> list[str]:
    output_shape = schema_hint.get("output_shape")
    fields = [
        str(field)
        for field, required in (output_shape.items() if isinstance(output_shape, dict) else [])
        if str(field).strip() and str(required or "").lower() == "required"
    ]
    task_profile = str((recipe_payload or {}).get("task_profile") or "").strip().lower()
    if not fields:
        if task_profile == "classification":
            fields = ["text", "label"]
        elif task_profile in {"rag_qa"}:
            fields = ["question", "context", "answer"]
        elif task_profile in {"summarization", "seq2seq"}:
            fields = ["source_text", "target_text"]
        else:
            fields = ["question", "answer"]
    if "source_text" in fields and "text" not in fields and task_profile == "classification":
        fields.insert(0, "text")
    deduped: list[str] = []
    for field in fields:
        if field not in deduped:
            deduped.append(field)
    return deduped


def _mapping_field_role(field: str, recipe_payload: dict[str, Any] | None) -> str:
    token = field.lower()
    task_profile = str((recipe_payload or {}).get("task_profile") or "").strip().lower()
    if token in {"answer", "target", "target_text", "output", "completion", "expected", "label", "class", "category"}:
        return "output"
    if token in {"context", "source", "source_text", "passage", "document"} and task_profile == "rag_qa":
        return "context"
    if token in {"context", "metadata"}:
        return "context"
    return "input"


def _mapping_recipe_default_source(field: str, recipe_payload: dict[str, Any] | None) -> str:
    if not recipe_payload:
        return field
    role = _mapping_field_role(field, recipe_payload)
    if role == "output":
        return str(recipe_payload.get("default_output_column") or field).strip() or field
    if role == "context":
        return "context"
    return str(recipe_payload.get("default_input_column") or field).strip() or field


def _mapping_aliases_for_field(
    field: str,
    schema_hint: dict[str, Any],
    domain_aliases: dict[str, list[str]] | None = None,
) -> list[str]:
    aliases: list[str] = []

    def add(value: Any) -> None:
        token = str(value or "").strip()
        if token and token not in aliases:
            aliases.append(token)

    input_candidates = schema_hint.get("input_candidates")
    if isinstance(input_candidates, dict):
        for value in list(input_candidates.get(field) or []):
            add(value)
    for value in (domain_aliases or {}).get(field, []):
        add(value)
    if field in {"text", "source_text", "question"}:
        for value in (domain_aliases or {}).get("input_text", []):
            add(value)
    if field in {"answer", "target_text", "label"}:
        for value in (domain_aliases or {}).get("target_text", []):
            add(value)
    if field == "context":
        for value in (domain_aliases or {}).get("context", []):
            add(value)
    add(field)
    return aliases


def _mapping_domain_aliases(contract: dict[str, Any]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}

    def add(field: Any, value: Any) -> None:
        key = str(field or "").strip()
        token = str(value or "").strip()
        if not key or not token:
            return
        bucket = aliases.setdefault(key, [])
        if token not in bucket:
            bucket.append(token)

    canonical_schema = contract.get("canonical_schema")
    if isinstance(canonical_schema, dict):
        raw_aliases = canonical_schema.get("aliases")
        if isinstance(raw_aliases, dict):
            for field, values in raw_aliases.items():
                add(field, field)
                for value in values if isinstance(values, list) else []:
                    add(field, value)
        for field in list(canonical_schema.get("required") or []):
            add(field, field)
    return aliases


def _mapping_pick_source(
    field: str,
    *,
    preferred: str,
    schema_hint: dict[str, Any],
    raw_field_frequency: dict[str, Any],
    domain_aliases: dict[str, list[str]] | None = None,
) -> tuple[str, list[str], list[str]]:
    raw_fields = {str(field) for field in raw_field_frequency.keys()}
    aliases = _mapping_aliases_for_field(field, schema_hint, domain_aliases)
    candidates: list[str] = []
    for value in [preferred, *aliases]:
        token = str(value or "").strip()
        if token and token not in candidates:
            candidates.append(token)
    detected = [candidate for candidate in candidates if candidate in raw_fields]
    if detected:
        detected.sort(key=lambda item: (-int(raw_field_frequency.get(item) or 0), candidates.index(item)))
        return detected[0], detected, candidates
    return str(preferred or (candidates[0] if candidates else field)).strip() or field, detected, candidates


def _mapping_template_from_fields(
    *,
    template_id: str,
    label: str,
    description: str,
    source: str,
    fields: list[dict[str, Any]],
    current_mapping: dict[str, str],
    raw_field_frequency: dict[str, Any],
    adapter_id: str,
    task_profile: str | None,
) -> dict[str, Any]:
    field_rows: list[dict[str, Any]] = []
    field_mapping: dict[str, str] = {}
    status_counts: Counter[str] = Counter()
    for field in fields:
        canonical = str(field.get("canonical_field") or "").strip()
        recommended = str(field.get("source_field") or "").strip()
        required = bool(field.get("required", True))
        candidates = [
            str(item)
            for item in list(field.get("candidates") or [])
            if str(item).strip()
        ]
        detected_candidates = [
            str(item)
            for item in list(field.get("detected_candidates") or [])
            if str(item).strip()
        ]
        current_source = str(current_mapping.get(canonical) or "").strip()
        if recommended:
            field_mapping[canonical] = recommended
        if current_source and current_source == recommended:
            status = "applied"
        elif required and recommended not in raw_field_frequency and not detected_candidates:
            status = "missing"
        elif len(detected_candidates) > 1 and not current_source:
            status = "ambiguous"
        else:
            status = "available"
        status_counts[status] += 1
        field_rows.append({
            "canonical_field": canonical,
            "recommended_source": recommended,
            "current_source": current_source or None,
            "status": status,
            "required": required,
            "detected_candidates": detected_candidates[:6],
            "candidate_sources": candidates[:8],
            "note": str(field.get("note") or ""),
        })

    total = max(1, len(field_rows))
    ready_count = int(status_counts.get("applied", 0) + status_counts.get("available", 0))
    score = max(
        0.0,
        min(
            1.0,
            (ready_count / total)
            - (0.22 * int(status_counts.get("missing", 0)) / total)
            - (0.12 * int(status_counts.get("ambiguous", 0)) / total),
        ),
    )
    if int(status_counts.get("missing", 0)):
        status = "missing"
    elif int(status_counts.get("ambiguous", 0)):
        status = "attention"
    else:
        status = "ready"
    return {
        "id": template_id,
        "label": label,
        "description": description,
        "source": source,
        "status": status,
        "recommended": False,
        "confidence": round(score, 4),
        "adapter_id": adapter_id,
        "task_profile": task_profile,
        "field_mapping": field_mapping,
        "fields": field_rows,
        "summary": {
            "total_fields": len(field_rows),
            "applied_count": int(status_counts.get("applied", 0)),
            "available_count": int(status_counts.get("available", 0)),
            "missing_count": int(status_counts.get("missing", 0)),
            "ambiguous_count": int(status_counts.get("ambiguous", 0)),
        },
        "apply_action": {
            "label": "Open Data Prep to apply",
            "target_tab": "dataprep",
            "requires_confirmation": True,
            "description": "Review and save/apply this mapping in Data Prep. Data Studio does not mutate project mapping.",
        },
    }


def _mapping_build_templates(
    *,
    recipe_payload: dict[str, Any] | None,
    effective_adapter_id: str,
    effective_task_profile: str | None,
    field_mapping: dict[str, str],
    adapter_contract: dict[str, Any],
    raw_field_frequency: dict[str, Any],
    auto_apply: dict[str, Any] | None = None,
    domain_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema_hint = _mapping_schema_hint(adapter_contract)
    domain_aliases = _mapping_domain_aliases(domain_contract or {}) if isinstance(domain_contract, dict) else {}
    output_fields = _mapping_output_fields(schema_hint, recipe_payload)
    templates: list[dict[str, Any]] = []

    def build_fields(source_kind: str, aliases: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for field in output_fields:
            preferred = _mapping_recipe_default_source(field, recipe_payload)
            source, detected, candidates = _mapping_pick_source(
                field,
                preferred=preferred,
                schema_hint=schema_hint,
                raw_field_frequency=raw_field_frequency,
                domain_aliases=aliases,
            )
            rows.append({
                "canonical_field": field,
                "source_field": source,
                "required": True,
                "detected_candidates": detected,
                "candidates": candidates,
                "note": source_kind,
            })
        return rows

    if recipe_payload:
        templates.append(
            _mapping_template_from_fields(
                template_id=f"recipe-{_mapping_template_slug(recipe_payload.get('id'))}",
                label=f"{recipe_payload.get('name') or recipe_payload.get('id')} recipe defaults",
                description="Uses the selected recipe's expected input/output columns as the starter mapping.",
                source="recipe",
                fields=build_fields("recipe-defaults"),
                current_mapping=field_mapping,
                raw_field_frequency=raw_field_frequency,
                adapter_id=effective_adapter_id,
                task_profile=effective_task_profile,
            )
        )

    if schema_hint:
        templates.append(
            _mapping_template_from_fields(
                template_id=f"adapter-{_mapping_template_slug(effective_adapter_id)}",
                label=f"{effective_adapter_id} adapter aliases",
                description="Uses the adapter's built-in aliases and detected source fields to propose field mappings.",
                source="adapter",
                fields=build_fields("adapter-aliases"),
                current_mapping=field_mapping,
                raw_field_frequency=raw_field_frequency,
                adapter_id=effective_adapter_id,
                task_profile=effective_task_profile,
            )
        )

    if domain_aliases:
        templates.append(
            _mapping_template_from_fields(
                template_id="domain-applied-contract",
                label="Applied domain contract",
                description="Uses aliases from the applied Domain Profile/Pack contract when available.",
                source="domain",
                fields=build_fields("domain-contract", aliases=domain_aliases),
                current_mapping=field_mapping,
                raw_field_frequency=raw_field_frequency,
                adapter_id=effective_adapter_id,
                task_profile=effective_task_profile,
            )
        )

    suggested_mapping = (
        auto_apply.get("suggested_field_mapping")
        if isinstance(auto_apply, dict) and isinstance(auto_apply.get("suggested_field_mapping"), dict)
        else {}
    )
    if suggested_mapping:
        fields = []
        for canonical, source in suggested_mapping.items():
            canonical_field = str(canonical or "").strip()
            source_field = str(source or "").strip()
            if not canonical_field or not source_field:
                continue
            fields.append({
                "canonical_field": canonical_field,
                "source_field": source_field,
                "required": True,
                "detected_candidates": [source_field] if source_field in raw_field_frequency else [],
                "candidates": [source_field],
                "note": "auto-fix",
            })
        if fields:
            templates.append(
                _mapping_template_from_fields(
                    template_id="auto-fix-high-confidence",
                    label="High-confidence detected mapping",
                    description="Uses high-confidence field-mapping suggestions from the current adapter preview.",
                    source="auto_fix",
                    fields=fields,
                    current_mapping=field_mapping,
                    raw_field_frequency=raw_field_frequency,
                    adapter_id=effective_adapter_id,
                    task_profile=effective_task_profile,
                )
            )

    if templates:
        best = max(
            templates,
            key=lambda item: (
                float(item.get("confidence") or 0.0),
                -int(item.get("summary", {}).get("missing_count") or 0),
                -int(item.get("summary", {}).get("ambiguous_count") or 0),
            ),
        )
        best["recommended"] = True

    missing_count = sum(int(item.get("summary", {}).get("missing_count") or 0) for item in templates)
    ambiguous_count = sum(int(item.get("summary", {}).get("ambiguous_count") or 0) for item in templates)
    recommended = next((item for item in templates if item.get("recommended")), None)
    return {
        "read_only": True,
        "template_count": len(templates),
        "recommended_template_id": recommended.get("id") if recommended else None,
        "detected_fields": _mapping_detected_field_rows(raw_field_frequency),
        "missing_field_count": missing_count,
        "ambiguous_field_count": ambiguous_count,
        "templates": templates,
        "entry_points": [
            {
                "label": "Open Data Prep",
                "target_tab": "dataprep",
                "reason": "Save or apply mapping templates only after reviewing them in the existing Data Prep workflow.",
                "requires_confirmation": True,
            }
        ],
    }


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
    mapping_templates: dict[str, Any] | None = None,
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
        "mapping_templates": mapping_templates or {
            "read_only": True,
            "template_count": 0,
            "recommended_template_id": None,
            "detected_fields": [],
            "missing_field_count": 0,
            "ambiguous_field_count": 0,
            "templates": [],
            "entry_points": [],
        },
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

    try:
        domain_runtime = await resolve_project_domain_runtime(db, project_id)
    except ValueError:
        domain_runtime = {}
    domain_contract = (
        domain_runtime.get("effective_contract")
        if isinstance(domain_runtime.get("effective_contract"), dict)
        else {}
    )
    base_adapter_contract = resolve_data_adapter_contract(effective_adapter_id)
    empty_mapping_templates = _mapping_build_templates(
        recipe_payload=recipe_payload,
        effective_adapter_id=effective_adapter_id,
        effective_task_profile=effective_task_profile,
        field_mapping=field_mapping,
        adapter_contract=base_adapter_contract,
        raw_field_frequency={},
        auto_apply={},
        domain_contract=domain_contract,
    )

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
            mapping_templates=empty_mapping_templates,
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
            mapping_templates=empty_mapping_templates,
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

    resolved_adapter_id = str(preview.get("resolved_adapter_id") or effective_adapter_id)
    resolved_task_profile = str(preview.get("resolved_task_profile") or effective_task_profile or "")
    resolved_contract = resolve_data_adapter_contract(resolved_adapter_id)
    raw_field_frequency = (
        preview.get("raw_field_frequency")
        if isinstance(preview.get("raw_field_frequency"), dict)
        else {}
    )
    auto_apply = preview.get("auto_apply") if isinstance(preview.get("auto_apply"), dict) else {}
    mapping_templates = _mapping_build_templates(
        recipe_payload=recipe_payload,
        effective_adapter_id=resolved_adapter_id,
        effective_task_profile=resolved_task_profile or effective_task_profile,
        field_mapping=field_mapping,
        adapter_contract=resolved_contract,
        raw_field_frequency=raw_field_frequency,
        auto_apply=auto_apply,
        domain_contract=domain_contract,
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
            "adapter_id": resolved_adapter_id,
            "requested_adapter_id": str(preview.get("requested_adapter_id") or effective_adapter_id),
            "task_profile": resolved_task_profile,
            "requested_task_profile": preview.get("requested_task_profile"),
            "adapter_config": adapter_config,
            "field_mapping": field_mapping,
            "auto_apply": auto_apply,
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
            "raw_field_frequency": raw_field_frequency,
        },
        "mapping_templates": mapping_templates,
        "issues": issues,
    }


_PREPARED_SPLIT_SPECS: tuple[tuple[str, str, DatasetType], ...] = (
    ("train", "Train", DatasetType.TRAIN),
    ("validation", "Validation", DatasetType.VALIDATION),
    ("test", "Test", DatasetType.TEST),
)


def _manifest_split_key(split_key: str) -> str:
    return "val" if split_key == "validation" else split_key


def _prepare_check(
    check_id: str,
    label: str,
    status: str,
    message: str,
    *,
    target_tab: str,
) -> dict[str, str]:
    return {
        "id": check_id,
        "label": label,
        "status": status,
        "message": message,
        "target_tab": target_tab,
    }


def _prepared_manifest_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "manifest.json"


# Epic E — exportable prepared split keys + their on-disk JSONL filenames.
# (The validation split is stored as ``val.jsonl``.)
PREPARED_SPLIT_FILENAMES: dict[str, str] = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "validation": "val.jsonl",
    "test": "test.jsonl",
}


def resolve_prepared_split_path(project_id: int, split: str) -> Path | None:
    """Return the on-disk JSONL path for a prepared split (``train`` / ``val`` /
    ``test``), or ``None`` for an unknown split. Pure — existence is the
    caller's check, so the export endpoint can 404 distinctly on
    unknown-split vs not-prepared-yet."""
    filename = PREPARED_SPLIT_FILENAMES.get(str(split or "").strip().lower())
    if filename is None:
        return None
    return settings.DATA_DIR / "projects" / str(project_id) / "prepared" / filename


# Above this many distinct train labels we assume the field isn't a
# classification label (free-text target / regression / generation), so
# per-class coverage warnings would be noise — we mark the report
# not-applicable instead.
_SPLIT_COVERAGE_MAX_CLASSES = 40


def _read_prepared_split_rows(project_id: int, split: str) -> list[dict[str, Any]] | None:
    """Read a prepared split's JSONL rows, or ``None`` if it isn't on disk."""
    path = resolve_prepared_split_path(project_id, split)
    if path is None or not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def build_split_class_coverage(project_id: int) -> dict[str, Any]:
    """Per-class coverage across the prepared TRAIN/VAL/TEST splits, with
    plain-language warnings (Epic E).

    The honesty win: a stratified split *should* put every class in every
    split, but a class with very few examples can land train-only — so the
    val/test pass-rate silently never measures it. This reads the prepared
    JSONL on disk, counts labels per split, and warns in plain language
    ("your val set has no `billing` examples — train has 42") so the user
    fixes coverage before training on a blind eval.

    Pure (file-only). ``applicable=False`` when nothing is prepared, the rows
    carry no label field, or the label looks free-text (too many classes)."""
    label_field = "label"
    manifest, _meta = _read_prepared_manifest(project_id)
    field_mapping = manifest.get("field_mapping")
    if isinstance(field_mapping, dict):
        mapped = field_mapping.get("label_field")
        if isinstance(mapped, str) and mapped.strip():
            label_field = mapped.strip()

    splits: dict[str, dict[str, Any]] = {}
    any_prepared = False
    for split in ("train", "val", "test"):
        rows = _read_prepared_split_rows(project_id, split)
        if rows is None:
            splits[split] = {"prepared": False, "total": 0, "by_label": {}}
            continue
        any_prepared = True
        counts: dict[str, int] = {}
        labelled = 0
        for row in rows:
            if label_field in row and row.get(label_field) is not None:
                counts[str(row.get(label_field))] = counts.get(str(row.get(label_field)), 0) + 1
                labelled += 1
        splits[split] = {
            "prepared": True,
            "total": len(rows),
            "labelled": labelled,
            "by_label": counts,
        }

    train = splits.get("train", {})
    train_labels = train.get("by_label", {}) if isinstance(train, dict) else {}

    if not any_prepared:
        return {"applicable": False, "reason": "not_prepared", "label_field": label_field, "splits": splits}
    if not train_labels:
        return {"applicable": False, "reason": "no_label_field", "label_field": label_field, "splits": splits}
    if len(train_labels) > _SPLIT_COVERAGE_MAX_CLASSES:
        return {"applicable": False, "reason": "free_text_label", "label_field": label_field, "splits": splits}

    warnings: list[dict[str, Any]] = []
    for split in ("val", "test"):
        s = splits.get(split, {})
        if not s.get("prepared") or int(s.get("total") or 0) == 0:
            continue  # an empty/unprepared eval split is a different problem
        present = s.get("by_label", {})
        for label, train_count in sorted(train_labels.items(), key=lambda kv: -kv[1]):
            if label not in present:
                warnings.append({
                    "severity": "warning",
                    "split": split,
                    "label": label,
                    "train_count": int(train_count),
                    "message": (
                        f"Your {split} set has no “{label}” examples — train has "
                        f"{int(train_count)}. That class's quality is never measured at eval."
                    ),
                })

    return {
        "applicable": True,
        "label_field": label_field,
        "class_count": len(train_labels),
        "splits": splits,
        "warnings": warnings,
    }


async def get_prepared_version_preview(
    db: AsyncSession, project_id: int
) -> dict[str, Any]:
    """Answer the review queue's "what version will include this?" — the next
    prepared dataset version number + the row breakdown the next Prepare will
    snapshot (Epic E). Closes the "where do my accepted synthetic rows show up?"
    gap: a reviewer sees that accepting a row stages it for ``v{next}``.

    Accepted = ``review_status`` accepted OR absent (legacy pre-review rows) —
    matching ``list_review_queue``'s accepted definition, so the count agrees
    with the queue surface."""
    synth_path = (
        settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"
    )
    accepted = 0
    pending = 0
    if synth_path.exists():
        with synth_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                status = row.get("review_status")
                if status == "pending":
                    pending += 1
                elif status in (None, "accepted"):
                    accepted += 1

    prepared_ids = (
        await db.execute(
            select(Dataset.id).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type.in_(
                    [DatasetType.TRAIN, DatasetType.VALIDATION, DatasetType.TEST]
                ),
            )
        )
    ).scalars().all()
    max_version = None
    if prepared_ids:
        max_version = (
            await db.execute(
                select(func.max(DatasetVersion.version)).where(
                    DatasetVersion.dataset_id.in_(prepared_ids)
                )
            )
        ).scalar()

    async def _record_count(*types: DatasetType) -> int:
        total = (
            await db.execute(
                select(func.coalesce(func.sum(Dataset.record_count), 0)).where(
                    Dataset.project_id == project_id,
                    Dataset.dataset_type.in_(types),
                )
            )
        ).scalar()
        return int(total or 0)

    gold = await _record_count(DatasetType.GOLD_DEV, DatasetType.GOLD_TEST)
    cleaned = await _record_count(DatasetType.CLEANED)

    return {
        "project_id": int(project_id),
        "next_version": int(max_version or 0) + 1,
        "has_existing_versions": max_version is not None,
        "staged": {
            "synthetic_accepted": accepted,
            "synthetic_pending": pending,
            "gold": gold,
            "cleaned": cleaned,
        },
        # The accepted-now rows + gold + cleaned that a Prepare would draw from
        # (pending rows are excluded until accepted — that's the point of review).
        "trainable_total": accepted + gold + cleaned,
    }


def _read_prepared_manifest(project_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _prepared_manifest_path(project_id)
    meta: dict[str, Any] = {
        "exists": path.exists(),
        "readable": False,
        "path": str(path),
        "error": None,
    }
    if not path.exists():
        return {}, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)[:240]
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "Prepared manifest is not a JSON object."
        return {}, meta
    meta["readable"] = True
    return payload, meta


def _dataset_version_summary(version: DatasetVersion | None) -> dict[str, Any] | None:
    if version is None:
        return None
    return {
        "id": int(version.id),
        "version": int(version.version or 0),
        "record_count": int(version.record_count or 0),
        "file_path": version.file_path,
        "created_at": version.created_at.isoformat() if version.created_at else None,
        "manifest": version.manifest if isinstance(version.manifest, dict) else {},
    }


def _file_exists(path: str | None) -> bool:
    token = str(path or "").strip()
    if not token:
        return False
    try:
        return Path(token).exists()
    except OSError:
        return False


def _prepared_split_summary(
    *,
    split_key: str,
    label: str,
    dataset_type: DatasetType,
    dataset: Dataset | None,
    versions: list[DatasetVersion],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_key = _manifest_split_key(split_key)
    manifest_splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    manifest_file_paths = (
        manifest.get("file_paths") if isinstance(manifest.get("file_paths"), dict) else {}
    )
    manifest_versions = (
        manifest.get("dataset_versions")
        if isinstance(manifest.get("dataset_versions"), dict)
        else {}
    )
    latest_version = versions[-1] if versions else None
    dataset_file_path = str(getattr(dataset, "file_path", "") or "")
    manifest_file_path = str(manifest_file_paths.get(manifest_key) or "")
    file_path = dataset_file_path or manifest_file_path
    row_count = int(getattr(dataset, "record_count", 0) or 0)
    manifest_count = int(manifest_splits.get(manifest_key) or 0)
    manifest_version = manifest_versions.get(manifest_key)
    try:
        manifest_version_int = int(manifest_version) if manifest_version is not None else None
    except (TypeError, ValueError):
        manifest_version_int = None

    return {
        "key": split_key,
        "manifest_key": manifest_key,
        "label": label,
        "dataset_type": dataset_type.value,
        "dataset_id": int(dataset.id) if dataset is not None else None,
        "exists": dataset is not None,
        "row_count": row_count,
        "file_path": file_path,
        "file_exists": _file_exists(file_path),
        "manifest_count": manifest_count,
        "manifest_version": manifest_version_int,
        "version_count": len(versions),
        "latest_version": _dataset_version_summary(latest_version),
    }


def _prepare_review_blocker(
    blocker_id: str,
    label: str,
    count: int,
    message: str,
    *,
    severity: IssueSeverity,
    target_tab: str,
) -> dict[str, Any]:
    return {
        "id": blocker_id,
        "label": label,
        "count": int(count),
        "severity": severity,
        "message": message,
        "target_tab": target_tab,
    }


async def build_data_studio_prepare_dataset(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a read-only dataset preparation readiness summary."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    overview = await build_data_studio_overview(db, project_id)
    mapping = await build_data_studio_mapping_preview(db, project_id)
    gold = await build_data_studio_gold_set_workbench(db, project_id)
    synthetic_queue = _review_queue_summary(await list_review_queue(db, project_id))
    annotation = await _annotation_review_summary(db, project_id)

    datasets_result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    datasets = list(datasets_result.scalars().all())
    datasets_by_type: dict[DatasetType, Dataset] = {}
    for dataset in sorted(datasets, key=lambda item: item.updated_at, reverse=True):
        datasets_by_type.setdefault(dataset.dataset_type, dataset)

    prepared_dataset_ids = [
        int(dataset.id)
        for dataset in datasets
        if dataset.dataset_type in {DatasetType.TRAIN, DatasetType.VALIDATION, DatasetType.TEST}
    ]
    versions_by_dataset: dict[int, list[DatasetVersion]] = {
        dataset_id: [] for dataset_id in prepared_dataset_ids
    }
    if prepared_dataset_ids:
        versions_result = await db.execute(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id.in_(prepared_dataset_ids))
            .order_by(DatasetVersion.dataset_id.asc(), DatasetVersion.version.asc())
        )
        for version in versions_result.scalars().all():
            versions_by_dataset.setdefault(int(version.dataset_id), []).append(version)

    manifest, manifest_meta = _read_prepared_manifest(project_id)
    split_items = [
        _prepared_split_summary(
            split_key=split_key,
            label=label,
            dataset_type=dataset_type,
            dataset=datasets_by_type.get(dataset_type),
            versions=versions_by_dataset.get(int(datasets_by_type[dataset_type].id), [])
            if dataset_type in datasets_by_type
            else [],
            manifest=manifest,
        )
        for split_key, label, dataset_type in _PREPARED_SPLIT_SPECS
    ]
    prepared_total_rows = sum(int(item["row_count"] or 0) for item in split_items)
    all_splits_have_rows = all(int(item["row_count"] or 0) > 0 for item in split_items)
    any_split_has_rows = any(int(item["row_count"] or 0) > 0 for item in split_items)
    if all_splits_have_rows:
        split_status = "ready"
    elif any_split_has_rows:
        split_status = "partial"
    else:
        split_status = "missing"

    missing_dataset_versions = [
        str(item["key"])
        for item in split_items
        if int(item["row_count"] or 0) > 0 and int(item["version_count"] or 0) <= 0
    ]
    missing_manifest_versions = [
        str(item["key"])
        for item in split_items
        if int(item["row_count"] or 0) > 0 and item.get("manifest_version") is None
    ]

    manifest_total_entries = int(manifest.get("total_entries") or 0)
    if bool(manifest_meta.get("readable")) and prepared_total_rows > 0:
        manifest_status = (
            "ready"
            if not missing_dataset_versions and not missing_manifest_versions
            else "attention"
        )
    elif prepared_total_rows > 0 or bool(manifest_meta.get("exists")):
        manifest_status = "attention"
    else:
        manifest_status = "missing"

    recipe_payload = overview.get("recipe") if isinstance(overview.get("recipe"), dict) else None
    row_counts = overview.get("row_counts") if isinstance(overview.get("row_counts"), dict) else {}
    mapping_summary = mapping.get("summary") if isinstance(mapping.get("summary"), dict) else {}
    effective_mapping = (
        mapping.get("effective_mapping") if isinstance(mapping.get("effective_mapping"), dict) else {}
    )
    mapping_source = mapping.get("source") if isinstance(mapping.get("source"), dict) else None
    mapping_contract_pass = bool(mapping_summary.get("contract_pass"))
    required_gaps = [
        str(item)
        for item in list(mapping_summary.get("required_fields_below_100") or [])
        if str(item).strip()
    ]

    trainable_rows = int(row_counts.get("trainable") or 0)
    gold_totals = gold.get("totals") if isinstance(gold.get("totals"), dict) else {}
    annotation_totals = (
        annotation.get("totals") if isinstance(annotation.get("totals"), dict) else {}
    )
    synthetic_pending = int(synthetic_queue.get("total_pending") or 0)
    synthetic_accepted = int(synthetic_queue.get("total_accepted") or 0)
    gold_review_needed = int(gold_totals.get("review_needed") or 0)
    gold_trusted = int(gold_totals.get("trusted_examples") or 0)
    annotation_review_needed = int(annotation_totals.get("review_needed") or 0)
    annotation_labeled_unpromoted = int(annotation_totals.get("labeled_unpromoted") or 0)

    issues: list[dict[str, str]] = []
    review_blockers: list[dict[str, Any]] = []

    if recipe_payload is None:
        issues.append(
            _issue(
                "prepare_missing_recipe",
                "blocker",
                "Recipe not selected",
                "Pick a recipe before preparing splits so BrewSLM knows the training shape.",
                action_label="Choose recipe",
                target_tab="data",
            )
        )

    if trainable_rows <= 0:
        issues.append(
            _issue(
                "prepare_no_trainable_rows",
                "blocker",
                "No trainable rows",
                "Add sources, create Gold Set examples, or accept synthetic rows before preparing a dataset.",
                action_label="Add sources",
                target_tab="data",
            )
        )

    if mapping_source is None:
        issues.append(
            _issue(
                "prepare_no_mapping_source",
                "blocker",
                "No mapping source",
                "Add a previewable source so the adapter contract can be checked before splitting.",
                action_label="Add sources",
                target_tab="data",
            )
        )
    elif not mapping_contract_pass:
        gap_text = f" Missing fields: {', '.join(required_gaps)}." if required_gaps else ""
        issues.append(
            _issue(
                "prepare_mapping_contract_not_ready",
                "blocker",
                "Mapping contract is not ready",
                f"Review the adapter preview before preparing train/validation/test files.{gap_text}",
                action_label="Review mapping",
                target_tab="dataprep",
            )
        )

    if trainable_rows > 0 and trainable_rows < 20:
        issues.append(
            _issue(
                "prepare_low_trainable_rows",
                "warning",
                "Very small training set",
                f"{trainable_rows} trainable row(s) can test the flow, but useful SFT usually needs more examples.",
                action_label="Add or generate rows",
                target_tab="synthetic",
            )
        )

    if synthetic_pending > 0:
        review_blockers.append(
            _prepare_review_blocker(
                "synthetic_pending_review",
                "Synthetic rows pending review",
                synthetic_pending,
                "Pending generated rows are excluded from dataset prep until accepted.",
                severity="warning",
                target_tab="synthetic",
            )
        )
        issues.append(
            _issue(
                "prepare_synthetic_pending_review",
                "warning",
                "Synthetic rows pending review",
                f"{synthetic_pending} generated row(s) will stay out of prepared splits until accepted.",
                action_label="Review synthetic rows",
                target_tab="synthetic",
            )
        )

    if gold_review_needed > 0:
        review_blockers.append(
            _prepare_review_blocker(
                "gold_needs_review",
                "Gold Set rows need review",
                gold_review_needed,
                "Gold Set examples are more valuable after approval or lock review.",
                severity="warning",
                target_tab="goldset",
            )
        )
        issues.append(
            _issue(
                "prepare_gold_needs_review",
                "warning",
                "Gold Set review is open",
                f"{gold_review_needed} Gold Set row(s) still need review.",
                action_label="Open Gold Set",
                target_tab="goldset",
            )
        )

    if annotation_labeled_unpromoted > 0:
        review_blockers.append(
            _prepare_review_blocker(
                "annotation_labeled_unpromoted",
                "Annotation labels need promotion",
                annotation_labeled_unpromoted,
                "Labeled annotation rows are not included downstream until promoted by the annotation workflow.",
                severity="warning",
                target_tab="annotate",
            )
        )
        issues.append(
            _issue(
                "prepare_annotation_labeled_unpromoted",
                "warning",
                "Annotation labels are not promoted",
                f"{annotation_labeled_unpromoted} labeled annotation row(s) are waiting for promotion.",
                action_label="Open Annotation",
                target_tab="annotate",
            )
        )

    if annotation_review_needed > 0:
        review_blockers.append(
            _prepare_review_blocker(
                "annotation_review_open",
                "Annotation review work is open",
                annotation_review_needed,
                "Assigned or unlabeled annotation rows are still in review.",
                severity="warning",
                target_tab="annotate",
            )
        )
        issues.append(
            _issue(
                "prepare_annotation_review_open",
                "warning",
                "Annotation review is open",
                f"{annotation_review_needed} annotation row(s) are assigned or waiting for labels.",
                action_label="Continue annotation",
                target_tab="annotate",
            )
        )

    has_blockers = any(item["severity"] == "blocker" for item in issues)
    if not has_blockers and split_status == "missing":
        issues.append(
            _issue(
                "prepare_splits_missing",
                "warning",
                "Prepared splits are missing",
                "Open Dataset Prep to create train, validation, and test files after confirming the split settings.",
                action_label="Open Dataset Prep",
                target_tab="dataprep",
            )
        )
    elif split_status == "partial":
        missing_splits = [
            str(item["label"])
            for item in split_items
            if int(item["row_count"] or 0) <= 0
        ]
        issues.append(
            _issue(
                "prepare_splits_partial",
                "warning",
                "Prepared splits are incomplete",
                f"Missing rows for: {', '.join(missing_splits)}.",
                action_label="Open Dataset Prep",
                target_tab="dataprep",
            )
        )

    if prepared_total_rows > 0 and not bool(manifest_meta.get("readable")):
        issues.append(
            _issue(
                "prepare_manifest_missing",
                "warning",
                "Prepared manifest is missing or unreadable",
                "Re-run Dataset Prep after confirming split settings so the manifest matches the split files.",
                action_label="Open Dataset Prep",
                target_tab="dataprep",
            )
        )

    if missing_dataset_versions:
        issues.append(
            _issue(
                "prepare_dataset_versions_missing",
                "warning",
                "Prepared split versions are missing",
                f"Missing DatasetVersion rows for: {', '.join(missing_dataset_versions)}.",
                action_label="Open Dataset Prep",
                target_tab="dataprep",
            )
        )

    if missing_manifest_versions:
        issues.append(
            _issue(
                "prepare_manifest_versions_missing",
                "warning",
                "Manifest version references are incomplete",
                f"Manifest has no dataset version reference for: {', '.join(missing_manifest_versions)}.",
                action_label="Open Dataset Prep",
                target_tab="dataprep",
            )
        )

    blocker_count = sum(1 for item in issues if item["severity"] == "blocker")
    warning_count = sum(1 for item in issues if item["severity"] == "warning")
    if blocker_count:
        verdict: PrepareDatasetVerdict = "blocked"
    elif warning_count:
        verdict = "attention"
    else:
        verdict = "ready"

    if recipe_payload is None:
        recipe_status = "missing"
        recipe_message = "Choose a recipe to make split and adapter checks recipe-aware."
    else:
        recipe_status = "met"
        recipe_message = f"{recipe_payload.get('name') or recipe_payload.get('id')} is selected."

    if mapping_source is None:
        mapping_status = "missing"
        mapping_message = "No previewable rows are available for adapter contract checks."
    elif recipe_payload is None:
        mapping_status = "attention"
        mapping_message = "Mapping can be previewed, but it is not tied to a selected recipe yet."
    elif mapping_contract_pass:
        mapping_status = "met"
        mapping_message = "Adapter mapping passes the required field contract for the selected recipe."
    else:
        mapping_status = "attention"
        mapping_message = "Adapter mapping needs review before creating prepared split files."

    review_status = "met" if not review_blockers else "attention"
    if synthetic_pending > 0:
        review_target_tab = "synthetic"
    elif gold_review_needed > 0:
        review_target_tab = "goldset"
    elif annotation_review_needed > 0 or annotation_labeled_unpromoted > 0:
        review_target_tab = "annotate"
    else:
        review_target_tab = "dataprep"
    split_check_status = "met" if split_status == "ready" else split_status
    manifest_check_status = (
        "met"
        if manifest_status == "ready"
        else ("missing" if manifest_status == "missing" else "attention")
    )

    checks = [
        _prepare_check(
            "recipe",
            "Recipe readiness",
            recipe_status,
            recipe_message,
            target_tab="data",
        ),
        _prepare_check(
            "mapping_contract",
            "Mapping contract",
            mapping_status,
            mapping_message,
            target_tab="dataprep",
        ),
        _prepare_check(
            "trainable_rows",
            "Trainable rows",
            "met" if trainable_rows > 0 else "missing",
            f"{trainable_rows} row(s) are currently eligible for preparation.",
            target_tab="data",
        ),
        _prepare_check(
            "review_gates",
            "Review gates",
            review_status,
            "Review queues are clear." if not review_blockers else "Some reviewed data will be excluded or needs attention.",
            target_tab=review_target_tab,
        ),
        _prepare_check(
            "split_files",
            "Prepared split files",
            split_check_status,
            (
                "Train, validation, and test splits are present."
                if split_status == "ready"
                else "Open Dataset Prep to create or refresh prepared split files."
            ),
            target_tab="dataprep",
        ),
        _prepare_check(
            "manifest_versions",
            "Manifest and versions",
            manifest_check_status,
            (
                "Prepared manifest and DatasetVersion rows are aligned."
                if manifest_status == "ready"
                else "Prepared manifest/version records will be created when Dataset Prep runs."
            ),
            target_tab="dataprep",
        ),
    ]

    included_types_raw = manifest.get("included_types")
    included_source_types = [
        str(item)
        for item in (included_types_raw if isinstance(included_types_raw, list) else [])
        if str(item).strip()
    ]

    return {
        "project_id": project_id,
        "verdict": verdict,
        "can_prepare": blocker_count == 0,
        "read_only": True,
        "auto_apply": False,
        "source_of_truth": "deterministic_data_studio_checks",
        "recipe": {
            "status": recipe_status,
            "selected": recipe_payload,
            "message": recipe_message,
        },
        "mapping": {
            "status": mapping_status,
            "message": mapping_message,
            "verdict": mapping.get("verdict"),
            "contract_pass": mapping_contract_pass,
            "source": mapping_source,
            "adapter_id": effective_mapping.get("adapter_id"),
            "task_profile": effective_mapping.get("task_profile"),
            "mapping_success_rate": float(mapping_summary.get("mapping_success_rate") or 0.0),
            "sampled_records": int(mapping_summary.get("sampled_records") or 0),
            "mapped_records": int(mapping_summary.get("mapped_records") or 0),
            "required_fields": list(mapping_summary.get("required_fields") or []),
            "required_fields_below_100": required_gaps,
        },
        "splits": {
            "status": split_status,
            "total_prepared_rows": prepared_total_rows,
            "required_splits": [item[0] for item in _PREPARED_SPLIT_SPECS],
            "items": split_items,
        },
        "manifest": {
            "status": manifest_status,
            "exists": bool(manifest_meta.get("exists")),
            "readable": bool(manifest_meta.get("readable")),
            "path": manifest_meta.get("path"),
            "error": manifest_meta.get("error"),
            "created_at": manifest.get("created_at"),
            "total_entries": manifest_total_entries,
            "splits": manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {},
            "ratios": manifest.get("ratios") if isinstance(manifest.get("ratios"), dict) else {},
            "included_types": included_source_types,
            "adapter_id": manifest.get("adapter_id"),
            "task_profile": manifest.get("task_profile"),
            "dataset_versions": (
                manifest.get("dataset_versions")
                if isinstance(manifest.get("dataset_versions"), dict)
                else {}
            ),
            "missing_dataset_version_splits": missing_dataset_versions,
            "missing_manifest_version_splits": missing_manifest_versions,
        },
        "inclusion": {
            "trainable_rows": trainable_rows,
            "raw_rows": int(row_counts.get("raw") or 0),
            "cleaned_rows": int(row_counts.get("cleaned") or 0),
            "gold_rows": int(row_counts.get("gold") or 0),
            "synthetic_total": int(row_counts.get("synthetic_total") or 0),
            "synthetic_pending": synthetic_pending,
            "synthetic_accepted": synthetic_accepted,
            "synthetic_pending_excluded": synthetic_pending > 0,
            "gold_trusted_examples": gold_trusted,
            "gold_review_needed": gold_review_needed,
            "included_source_types": included_source_types,
        },
        "review_blockers": review_blockers,
        "checks": checks,
        "issues": issues,
        "entry_point": {
            "label": "Open Dataset Prep",
            "target_tab": "dataprep",
            "reason": "Confirm adapter and split settings before writing prepared files.",
            "requires_confirmation": True,
        },
        "power_details": {
            "overview_issues": overview.get("issues") if isinstance(overview.get("issues"), list) else [],
            "mapping_issues": mapping.get("issues") if isinstance(mapping.get("issues"), list) else [],
            "gold_validation": gold.get("validation") if isinstance(gold.get("validation"), dict) else {},
            "synthetic_review_queue": synthetic_queue,
            "annotation_totals": annotation_totals,
            "manifest": manifest if bool(manifest_meta.get("readable")) else {},
        },
    }


def _quality_check(
    check_id: str,
    label: str,
    category: str,
    status: str,
    severity: IssueSeverity,
    message: str,
    *,
    count: int,
    target_tab: str,
    workflow_owner: str,
    source: str,
    domain_id: str,
    domain_label: str,
    evidence: list[str] | None = None,
    action_label: str | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "category": category,
        "status": status,
        "severity": severity,
        "message": message,
        "count": int(count),
        "target_tab": target_tab,
        "workflow_owner": workflow_owner,
        "source": source,
        "domain_id": domain_id,
        "domain_label": domain_label,
        "evidence": list(evidence or []),
        "action_label": action_label or "Open workflow",
    }


def _quality_issue_from_check(check: dict[str, Any]) -> dict[str, str]:
    return _issue(
        str(check.get("id") or "quality_safety_check"),
        check.get("severity") if check.get("severity") in {"blocker", "warning", "info"} else "info",
        str(check.get("label") or "Quality and safety check"),
        str(check.get("message") or ""),
        action_label=str(check.get("action_label") or "Open workflow"),
        target_tab=str(check.get("target_tab") or "data"),
    )


def _quality_group_rows(
    groups: dict[str, Counter[str]],
    targets: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, counts in groups.items():
        blocker_count = int(counts.get("blocker", 0))
        warning_count = int(counts.get("warning", 0))
        info_count = int(counts.get("info", 0))
        total = blocker_count + warning_count + info_count
        rows.append({
            "key": _domain_setup_slug(label),
            "label": label,
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "total": total,
            "target_tab": targets.get(label, "data"),
        })
    rows.sort(key=lambda item: (-int(item["blocker_count"]), -int(item["warning_count"]), -int(item["total"]), str(item["label"])))
    return rows


def _quality_top_source(counter: Counter[str], fallback: str = "Project sample") -> str:
    if not counter:
        return fallback
    return counter.most_common(1)[0][0]


def _quality_split_fingerprints(rows: list[dict[str, Any]]) -> set[str]:
    fingerprints: set[str] = set()
    for row in rows:
        text = _quality_scan_text(row)
        fingerprint = _quality_text_fingerprint(text)
        if fingerprint:
            fingerprints.add(fingerprint)
    return fingerprints


def _quality_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _quality_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "required"}
    return bool(value)


def _quality_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quality_contract_extra_checks(contract: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[Any] = [
        contract.get("quality_checks"),
        contract.get("recommended_quality_checks"),
        contract.get("recommended_checks"),
    ]
    for key in ("data_quality", "safety", "privacy", "review_gates"):
        section = contract.get(key)
        if isinstance(section, dict):
            containers.extend([
                section.get("quality_checks"),
                section.get("recommended_quality_checks"),
                section.get("recommended_checks"),
                section.get("checks"),
            ])
    checks: list[dict[str, Any]] = []
    for container in containers:
        for item in _quality_list(container):
            if isinstance(item, dict):
                checks.append(item)
            elif isinstance(item, str) and item.strip():
                checks.append({"id": _domain_setup_slug(item), "label": item, "type": "note"})
    return checks


def _quality_domain_contract_context(
    runtime: dict[str, Any],
    *,
    profile_contract: dict[str, Any] | None = None,
    pack_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = runtime.get("effective_contract")
    if not isinstance(contract, dict):
        contract = {}
    profile_id = str(runtime.get("domain_profile_applied") or "").strip()
    pack_id = str(runtime.get("domain_pack_applied") or "").strip()
    profile_source = str(runtime.get("domain_profile_source") or "").strip()
    pack_source = str(runtime.get("domain_pack_source") or "").strip()
    pack_overlay = runtime.get("pack_overlay") if isinstance(runtime.get("pack_overlay"), dict) else {}
    explicit_checks: list[dict[str, Any]] = []
    seen_check_ids: set[str] = set()
    for source_contract in [profile_contract, pack_overlay, pack_contract, contract]:
        if not isinstance(source_contract, dict):
            continue
        for check in _quality_contract_extra_checks(source_contract):
            check_key = str(check.get("id") or check.get("label") or check.get("title") or check)
            if check_key in seen_check_ids:
                continue
            seen_check_ids.add(check_key)
            explicit_checks.append(check)
    non_generic_profile = bool(profile_id and profile_id != "generic-domain-v1")
    non_generic_pack = bool(pack_id and pack_id != "general-pack-v1")
    available = bool(non_generic_profile or non_generic_pack or explicit_checks)
    return {
        "available": available,
        "preview_only": True,
        "applied_profile_id": profile_id or None,
        "applied_profile_source": profile_source or None,
        "applied_pack_id": pack_id or None,
        "applied_pack_source": pack_source or None,
        "contract": contract,
        "explicit_checks": explicit_checks,
    }


def _quality_contract_aliases(contract: dict[str, Any]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}

    def add_alias(field: Any, alias: Any) -> None:
        key = str(field or "").strip()
        token = str(alias or "").strip()
        if not key or not token:
            return
        bucket = aliases.setdefault(key, [])
        if token not in bucket:
            bucket.append(token)

    canonical_schema = contract.get("canonical_schema")
    if isinstance(canonical_schema, dict):
        raw_aliases = canonical_schema.get("aliases")
        if isinstance(raw_aliases, dict):
            for field, values in raw_aliases.items():
                add_alias(field, field)
                for alias in _quality_list(values):
                    add_alias(field, alias)
        for field in _quality_list(canonical_schema.get("required")):
            add_alias(field, field)

    for task in _quality_list(contract.get("tasks")):
        if not isinstance(task, dict):
            continue
        for field in _quality_list(task.get("required_fields")) + _quality_list(task.get("optional_fields")):
            add_alias(field, field)
    return aliases


def _quality_contract_field_candidates(
    field: str,
    aliases: dict[str, list[str]],
) -> list[str]:
    candidates: list[str] = []
    for item in [field, *aliases.get(field, [])]:
        token = str(item or "").strip()
        if token and token not in candidates:
            candidates.append(token)
    return candidates or [field]


def _quality_contract_field_value(
    row: dict[str, Any],
    field: str,
    aliases: dict[str, list[str]],
) -> Any:
    candidates = _quality_contract_field_candidates(field, aliases)
    lowered = {str(key).lower(): key for key in row.keys()}
    for candidate in candidates:
        if candidate in row:
            return row.get(candidate)
        match = lowered.get(candidate.lower())
        if match is not None:
            return row.get(match)
    return None


def _quality_contract_text_for_field(
    row: dict[str, Any],
    field: str | None,
    aliases: dict[str, list[str]],
) -> str:
    if field:
        value = _quality_contract_field_value(row, field, aliases)
        values: list[str] = []
        _flatten_text_values(value, values, limit=20)
        return " ".join(values).strip()
    return _quality_scan_text(row)


def _quality_contract_field_coverage(
    scan_rows: list[dict[str, Any]],
    field: str,
    aliases: dict[str, list[str]],
) -> dict[str, Any]:
    present = 0
    missing = 0
    missing_sources: Counter[str] = Counter()
    for item in scan_rows:
        row = item.get("row") if isinstance(item.get("row"), dict) else {}
        if _field_has_value(_quality_contract_field_value(row, field, aliases)):
            present += 1
        else:
            missing += 1
            missing_sources[str(item.get("source") or "Project sample")] += 1
    total = present + missing
    return {
        "field": field,
        "present": present,
        "missing": missing,
        "ratio": (present / total) if total else 0.0,
        "missing_sources": missing_sources,
    }


def _quality_domain_check_id(prefix: str, value: Any) -> str:
    return f"domain_authored_{prefix}_{_domain_setup_slug(value)}"[:160]


def _quality_domain_severity(value: Any, default: IssueSeverity = "warning") -> IssueSeverity:
    token = str(value or default).strip().lower()
    if token in {"blocker", "warning", "info"}:
        return token  # type: ignore[return-value]
    if token in {"error", "critical", "high"}:
        return "blocker"
    if token in {"medium", "warn"}:
        return "warning"
    return default


def _quality_domain_status(severity: IssueSeverity, count: int) -> str:
    if count <= 0:
        return "ready"
    return "blocked" if severity == "blocker" else "attention"


def _quality_domain_authored_checks(
    *,
    domain_context: dict[str, Any],
    scan_rows: list[dict[str, Any]],
    row_texts: list[str],
    pii_total: int,
    duplicate_signal_count: int,
    leakage_overlap_count: int,
    synthetic_pending: int,
    gold_review_needed: int,
    annotation_review_needed: int,
    annotation_unpromoted: int,
    domain_id: str,
    domain_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = domain_context.get("contract") if isinstance(domain_context.get("contract"), dict) else {}
    explicit_checks = (
        domain_context.get("explicit_checks")
        if isinstance(domain_context.get("explicit_checks"), list)
        else []
    )
    aliases = _quality_contract_aliases(contract)
    data_quality = contract.get("data_quality") if isinstance(contract.get("data_quality"), dict) else {}
    normalization = contract.get("normalization") if isinstance(contract.get("normalization"), dict) else {}
    dataset_split = contract.get("dataset_split") if isinstance(contract.get("dataset_split"), dict) else {}
    audit = contract.get("audit") if isinstance(contract.get("audit"), dict) else {}

    if not domain_context.get("available"):
        return [], {
            "available": False,
            "preview_only": True,
            "applied_profile_id": domain_context.get("applied_profile_id"),
            "applied_profile_source": domain_context.get("applied_profile_source"),
            "applied_pack_id": domain_context.get("applied_pack_id"),
            "applied_pack_source": domain_context.get("applied_pack_source"),
            "check_count": 0,
            "failing_count": 0,
            "blocker_count": 0,
            "warning_count": 0,
            "ready_count": 0,
            "supported_sources": [],
        }

    checks: list[dict[str, Any]] = []

    def append_check(check: dict[str, Any]) -> None:
        check["domain_authored"] = True
        check["read_only_preview"] = True
        checks.append(check)

    total_rows = len(scan_rows)
    min_records = data_quality.get("min_records")
    if isinstance(min_records, int) and min_records > 0:
        missing_records = max(0, min_records - total_rows)
        severity: IssueSeverity = "warning"
        append_check(
            _quality_check(
                "domain_authored_min_records",
                "Domain minimum row target",
                "domain-authored",
                _quality_domain_status(severity, missing_records),
                severity if missing_records else "info",
                (
                    f"Applied domain contract recommends at least {min_records} row(s); "
                    f"{total_rows} row(s) are available in the scan sample."
                ),
                count=missing_records,
                target_tab="data" if total_rows <= 0 else "synthetic",
                workflow_owner="Domain Managers",
                source="Applied domain contract",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=["This is a readiness target, not an automatic mutation."],
                action_label="Review domain setup",
            )
        )

    required_fields: dict[str, float] = {}
    canonical_schema = contract.get("canonical_schema") if isinstance(contract.get("canonical_schema"), dict) else {}
    for field in _quality_list(canonical_schema.get("required")):
        token = str(field or "").strip()
        if token:
            required_fields.setdefault(token, 1.0)
    required_coverage = data_quality.get("required_coverage")
    if isinstance(required_coverage, dict):
        for field, threshold in required_coverage.items():
            token = str(field or "").strip()
            if token:
                required_fields[token] = max(required_fields.get(token, 0.0), _quality_float(threshold, 1.0))
    for task in _quality_list(contract.get("tasks")):
        if not isinstance(task, dict):
            continue
        for field in _quality_list(task.get("required_fields")):
            token = str(field or "").strip()
            if token:
                required_fields.setdefault(token, 1.0)

    coverage_gaps: list[str] = []
    missing_sources: Counter[str] = Counter()
    for field, threshold in required_fields.items():
        coverage = _quality_contract_field_coverage(scan_rows, field, aliases)
        ratio = float(coverage.get("ratio") or 0.0)
        missing = int(coverage.get("missing") or 0)
        if total_rows <= 0 or ratio < threshold:
            coverage_gaps.append(f"{field}: {round(ratio * 100)}% < {round(threshold * 100)}%")
            missing_sources.update(coverage.get("missing_sources") or Counter())
    if coverage_gaps:
        append_check(
            _quality_check(
                "domain_authored_required_coverage",
                "Domain-required field coverage",
                "domain-authored",
                "attention",
                "warning",
                "Applied domain contract requires stronger field coverage before training.",
                count=len(coverage_gaps),
                target_tab="dataprep",
                workflow_owner="Domain Managers",
                source=_quality_top_source(missing_sources, "Applied domain contract"),
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=coverage_gaps[:5],
                action_label="Review mapping",
            )
        )
    elif required_fields:
        append_check(
            _quality_check(
                "domain_authored_required_coverage_clear",
                "Domain-required fields covered",
                "domain-authored",
                "ready",
                "info",
                "Applied domain-required fields meet configured coverage thresholds in the scanned sample.",
                count=0,
                target_tab="dataprep",
                workflow_owner="Domain Managers",
                source="Applied domain contract",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=list(required_fields.keys())[:5],
                action_label="Review mapping",
            )
        )

    max_duplicate_ratio = data_quality.get("max_duplicate_ratio")
    if max_duplicate_ratio is not None and row_texts:
        threshold = _quality_float(max_duplicate_ratio, 1.0)
        duplicate_ratio = duplicate_signal_count / max(1, len(row_texts))
        excess = max(0, duplicate_signal_count - int(threshold * len(row_texts)))
        if duplicate_ratio > threshold:
            append_check(
                _quality_check(
                    "domain_authored_duplicate_ratio",
                    "Domain duplicate ratio",
                    "domain-authored",
                    "attention",
                    "warning",
                    f"Duplicate signal ratio is {round(duplicate_ratio * 100)}%; domain threshold is {round(threshold * 100)}%.",
                    count=max(1, excess),
                    target_tab="dataprep",
                    workflow_owner="Domain Managers",
                    source="Applied domain contract",
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=["Tune dedupe or refresh splits before relying on evaluation metrics."],
                    action_label="Open Data Prep",
                )
            )

    leakage_checks = [
        str(item)
        for item in _quality_list(dataset_split.get("leakage_checks"))
        if str(item).strip()
    ]
    if leakage_checks:
        severity = "blocker" if leakage_overlap_count else "info"
        append_check(
            _quality_check(
                "domain_authored_leakage_gates",
                "Domain leakage gates",
                "domain-authored",
                _quality_domain_status(severity, leakage_overlap_count),
                severity,
                (
                    f"Applied domain contract asks for leakage checks: {', '.join(leakage_checks[:4])}."
                    if leakage_overlap_count <= 0
                    else "Applied domain leakage gates found overlap in prepared splits."
                ),
                count=leakage_overlap_count,
                target_tab="dataprep",
                workflow_owner="Domain Managers",
                source="Applied domain contract",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=leakage_checks[:5],
                action_label="Open Data Prep",
            )
        )

    for section_key in ("data_quality", "safety", "privacy"):
        section = contract.get(section_key)
        if not isinstance(section, dict):
            continue
        forbidden_phrases = [
            str(item.get("phrase") if isinstance(item, dict) else item).strip()
            for item in _quality_list(section.get("forbidden_phrases"))
            if str(item.get("phrase") if isinstance(item, dict) else item).strip()
        ]
        if forbidden_phrases:
            match_count = 0
            match_sources: Counter[str] = Counter()
            lowered_phrases = [phrase.lower() for phrase in forbidden_phrases]
            for item in scan_rows:
                text = _quality_scan_text(item.get("row") if isinstance(item.get("row"), dict) else {}).lower()
                if any(phrase in text for phrase in lowered_phrases):
                    match_count += 1
                    match_sources[str(item.get("source") or "Project sample")] += 1
            if match_count:
                append_check(
                    _quality_check(
                        _quality_domain_check_id("forbidden_phrases", section_key),
                        "Domain forbidden phrases",
                        "domain-authored",
                        "attention",
                        "warning",
                        f"Applied domain contract found {match_count} row(s) with forbidden phrase matches.",
                        count=match_count,
                        target_tab="data",
                        workflow_owner="Domain Managers",
                        source=_quality_top_source(match_sources),
                        domain_id=domain_id,
                        domain_label=domain_label,
                        evidence=["Forbidden phrase values are not echoed in this preview."],
                        action_label="Inspect sources",
                    )
                )

    pii_redaction = normalization.get("pii_redaction")
    pii_redaction_enabled = isinstance(pii_redaction, dict) and _quality_bool(pii_redaction.get("enabled"))
    privacy_rules = contract.get("privacy") if isinstance(contract.get("privacy"), dict) else {}
    privacy_required = pii_redaction_enabled or _quality_bool(privacy_rules.get("pii_redaction_required"))
    if privacy_required:
        redaction_fields = {
            "redacted_text",
            "masked_text",
            "redaction",
            "redaction_policy",
            "pii_masked",
        }
        has_redaction_field = any(
            field.lower() in redaction_fields
            for item in scan_rows
            for field in ((item.get("row") if isinstance(item.get("row"), dict) else {}).keys())
        )
        if pii_total and not has_redaction_field:
            append_check(
                _quality_check(
                    "domain_authored_privacy_redaction",
                    "Domain privacy redaction gate",
                    "domain-authored",
                    "blocked",
                    "blocker",
                    "Applied domain contract requires PII redaction, but sensitive patterns appear without redaction fields.",
                    count=pii_total,
                    target_tab="data",
                    workflow_owner="Domain Managers",
                    source="Applied domain contract",
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=["Values are redacted from this preview; inspect sources before preparing datasets."],
                    action_label="Inspect sources",
                )
            )
        else:
            append_check(
                _quality_check(
                    "domain_authored_privacy_redaction_ready",
                    "Domain privacy redaction gate",
                    "domain-authored",
                    "ready",
                    "info",
                    "Applied domain privacy redaction gate is configured; no blocking unredacted PII signal was found.",
                    count=0,
                    target_tab="domain",
                    workflow_owner="Domain Managers",
                    source="Applied domain contract",
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=[],
                    action_label="Open Domain Managers",
                )
            )

    review_gate_count = synthetic_pending + gold_review_needed + annotation_review_needed + annotation_unpromoted
    review_gates = data_quality.get("review_gates") or contract.get("review_gates")
    human_review_required = (
        _quality_bool(audit.get("require_human_approval_for_production"))
        or _quality_bool(data_quality.get("human_review_required"))
        or bool(review_gates)
    )
    if human_review_required:
        append_check(
            _quality_check(
                "domain_authored_review_gate",
                "Domain review gate",
                "domain-authored",
                _quality_domain_status("warning", review_gate_count),
                "warning" if review_gate_count else "info",
                (
                    f"Applied domain contract requires review gates; {review_gate_count} review item(s) are still open."
                    if review_gate_count
                    else "Applied domain contract requires review gates, and current review queues are clear."
                ),
                count=review_gate_count,
                target_tab="synthetic" if synthetic_pending else ("goldset" if gold_review_needed else "annotate"),
                workflow_owner="Domain Managers",
                source="Applied domain contract",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[
                    f"Synthetic pending: {synthetic_pending}.",
                    f"Gold Set review needed: {gold_review_needed}.",
                    f"Annotation review/promotion: {annotation_review_needed + annotation_unpromoted}.",
                ],
                action_label="Open Review",
            )
        )

    citation_required = _quality_bool(data_quality.get("citation_required")) or _quality_bool(contract.get("citation_required"))
    context_required = _quality_bool(data_quality.get("context_required")) or _quality_bool(contract.get("context_required"))
    citation_fields = [
        str(item).strip()
        for item in _quality_list(data_quality.get("citation_fields") or contract.get("citation_fields"))
        if str(item).strip()
    ] or ["citation", "source", "reference", "policy_section"]
    context_fields = [
        str(item).strip()
        for item in _quality_list(data_quality.get("context_fields") or contract.get("context_fields"))
        if str(item).strip()
    ] or ["context", "source_excerpt", "passage", "policy_text"]
    expectations: list[tuple[str, list[str]]] = []
    if citation_required:
        expectations.append(("citation", citation_fields))
    if context_required:
        expectations.append(("context", context_fields))
    for expected_label, expected_fields in expectations:
        present = 0
        missing = 0
        missing_sources: Counter[str] = Counter()
        for item in scan_rows:
            row = item.get("row") if isinstance(item.get("row"), dict) else {}
            if any(_field_has_value(_quality_contract_field_value(row, field, aliases)) for field in expected_fields):
                present += 1
            else:
                missing += 1
                missing_sources[str(item.get("source") or "Project sample")] += 1
        append_check(
            _quality_check(
                f"domain_authored_{expected_label}_gate",
                f"Domain {expected_label} expectation",
                "domain-authored",
                _quality_domain_status("warning", missing),
                "warning" if missing else "info",
                (
                    f"Applied domain contract expects {expected_label} fields; {missing} scanned row(s) are missing them."
                    if missing
                    else f"Applied domain {expected_label} expectation is satisfied in the scanned sample."
                ),
                count=missing,
                target_tab="dataprep",
                workflow_owner="Domain Managers",
                source=_quality_top_source(missing_sources, "Applied domain contract"),
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=expected_fields[:5],
                action_label="Review mapping",
            )
        )

    for index, check in enumerate(explicit_checks):
        if not isinstance(check, dict):
            continue
        check_type = str(check.get("type") or check.get("kind") or "").strip().lower().replace("-", "_")
        label = str(check.get("label") or check.get("title") or check.get("id") or f"Domain check {index + 1}")
        severity = _quality_domain_severity(check.get("severity"), "warning")
        target_tab = str(check.get("target_tab") or check.get("target") or "domain")
        action_label = str(check.get("action_label") or "Open Domain Managers")

        if check_type in {"required_field", "required_fields", "field_coverage"}:
            fields = [
                str(item).strip()
                for item in _quality_list(check.get("fields") or check.get("field"))
                if str(item).strip()
            ]
            min_coverage = _quality_float(check.get("min_coverage") or check.get("threshold"), 1.0)
            gaps: list[str] = []
            missing_sources: Counter[str] = Counter()
            for field in fields:
                coverage = _quality_contract_field_coverage(scan_rows, field, aliases)
                if float(coverage.get("ratio") or 0.0) < min_coverage:
                    gaps.append(f"{field}: {round(float(coverage.get('ratio') or 0.0) * 100)}%")
                    missing_sources.update(coverage.get("missing_sources") or Counter())
            append_check(
                _quality_check(
                    _quality_domain_check_id("explicit_field", check.get("id") or label),
                    label,
                    "domain-authored",
                    _quality_domain_status(severity, len(gaps)),
                    severity if gaps else "info",
                    str(check.get("message") or ("Domain-authored field coverage check found gaps." if gaps else "Domain-authored field coverage check is clear.")),
                    count=len(gaps),
                    target_tab=target_tab if gaps else "domain",
                    workflow_owner="Domain Managers",
                    source=_quality_top_source(missing_sources, "Applied domain contract"),
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=gaps[:5],
                    action_label=action_label,
                )
            )
            continue

        if check_type in {"regex", "pattern", "regex_pattern"} or check.get("pattern"):
            pattern = str(check.get("pattern") or "").strip()
            if not pattern:
                continue
            mode = str(check.get("mode") or check.get("expectation") or "forbid").strip().lower()
            field = str(check.get("field") or "").strip() or None
            flags = 0 if _quality_bool(check.get("case_sensitive")) else re.IGNORECASE
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                append_check(
                    _quality_check(
                        _quality_domain_check_id("invalid_regex", check.get("id") or label),
                        f"{label} regex invalid",
                        "domain-authored",
                        "attention",
                        "warning",
                        f"Domain-authored regex could not compile: {str(exc)[:160]}",
                        count=1,
                        target_tab="domain",
                        workflow_owner="Domain Managers",
                        source="Applied domain contract",
                        domain_id=domain_id,
                        domain_label=domain_label,
                        evidence=[],
                        action_label="Open Domain Managers",
                    )
                )
                continue
            match_count = 0
            source_counts: Counter[str] = Counter()
            for item in scan_rows:
                row = item.get("row") if isinstance(item.get("row"), dict) else {}
                text = _quality_contract_text_for_field(row, field, aliases)
                matched = bool(compiled.search(text))
                failed = not matched if mode in {"require", "required", "must_match"} else matched
                if failed:
                    match_count += 1
                    source_counts[str(item.get("source") or "Project sample")] += 1
            append_check(
                _quality_check(
                    _quality_domain_check_id("explicit_regex", check.get("id") or label),
                    label,
                    "domain-authored",
                    _quality_domain_status(severity, match_count),
                    severity if match_count else "info",
                    str(check.get("message") or f"Domain-authored regex check {'found matches' if mode not in {'require', 'required', 'must_match'} else 'found missing matches'}."),
                    count=match_count,
                    target_tab=target_tab if match_count else "domain",
                    workflow_owner="Domain Managers",
                    source=_quality_top_source(source_counts, "Applied domain contract"),
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=["Regex evidence is summarized without echoing matched values."],
                    action_label=action_label,
                )
            )
            continue

        if check_type in {"forbidden_phrase", "forbidden_phrases"} or check.get("phrases"):
            phrases = [
                str(item).strip().lower()
                for item in _quality_list(check.get("phrases") or check.get("phrase"))
                if str(item).strip()
            ]
            match_count = 0
            source_counts: Counter[str] = Counter()
            for item in scan_rows:
                row = item.get("row") if isinstance(item.get("row"), dict) else {}
                text = _quality_scan_text(row).lower()
                if any(phrase in text for phrase in phrases):
                    match_count += 1
                    source_counts[str(item.get("source") or "Project sample")] += 1
            append_check(
                _quality_check(
                    _quality_domain_check_id("explicit_phrase", check.get("id") or label),
                    label,
                    "domain-authored",
                    _quality_domain_status(severity, match_count),
                    severity if match_count else "info",
                    str(check.get("message") or "Domain-authored forbidden phrase check completed."),
                    count=match_count,
                    target_tab=target_tab if match_count else "domain",
                    workflow_owner="Domain Managers",
                    source=_quality_top_source(source_counts, "Applied domain contract"),
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=["Forbidden phrase values are not echoed in this preview."],
                    action_label=action_label,
                )
            )

    blocker_count = sum(1 for check in checks if check.get("severity") == "blocker")
    warning_count = sum(1 for check in checks if check.get("severity") == "warning")
    ready_count = sum(1 for check in checks if check.get("status") == "ready")
    supported_sources = [
        source
        for source in [
            "profile:data_quality",
            "profile:normalization",
            "profile:dataset_split",
            "profile:audit",
            "contract:quality_checks" if explicit_checks else "",
        ]
        if source
    ]
    return checks, {
        "available": True,
        "preview_only": True,
        "applied_profile_id": domain_context.get("applied_profile_id"),
        "applied_profile_source": domain_context.get("applied_profile_source"),
        "applied_pack_id": domain_context.get("applied_pack_id"),
        "applied_pack_source": domain_context.get("applied_pack_source"),
        "check_count": len(checks),
        "failing_count": sum(1 for check in checks if check.get("status") != "ready"),
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "ready_count": ready_count,
        "supported_sources": supported_sources,
    }


async def build_data_studio_quality_safety(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return deterministic read-only data quality and safety scans."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    mapping = await build_data_studio_mapping_preview(db, project_id)
    domain = await build_data_studio_domain_detection(db, project_id)
    review_queue = await build_data_studio_review_queue(db, project_id)
    prepare_dataset = await build_data_studio_prepare_dataset(db, project_id)
    try:
        domain_runtime = await resolve_project_domain_runtime(db, project_id)
    except ValueError:
        domain_runtime = {}
    profile_contract: dict[str, Any] | None = None
    pack_contract: dict[str, Any] | None = None
    profile_id = str(domain_runtime.get("domain_profile_applied") or "").strip()
    pack_id = str(domain_runtime.get("domain_pack_applied") or "").strip()
    if profile_id:
        profile = await get_domain_profile(db, profile_id)
        if profile is not None and isinstance(profile.contract, dict):
            profile_contract = profile.contract
    if pack_id:
        pack = await get_domain_pack(db, pack_id)
        if pack is not None and isinstance(pack.contract, dict):
            pack_contract = pack.contract
    domain_contract_context = _quality_domain_contract_context(
        domain_runtime,
        profile_contract=profile_contract,
        pack_contract=pack_contract,
    )

    detected_domain = domain.get("detected_domain") if isinstance(domain.get("detected_domain"), dict) else {}
    domain_id = str(detected_domain.get("id") or "generic_domain")
    domain_label = str(detected_domain.get("label") or "Generic Domain")
    domain_confidence = float(detected_domain.get("confidence") or 0.0)

    scan_rows = await _quality_source_rows(db, project_id, limit=500)
    if not scan_rows:
        for preview_row in list(mapping.get("preview_rows") or []):
            if not isinstance(preview_row, dict):
                continue
            raw = preview_row.get("raw") if isinstance(preview_row.get("raw"), dict) else {}
            mapped = preview_row.get("mapped") if isinstance(preview_row.get("mapped"), dict) else {}
            scan_rows.append({
                "row": {**raw, **mapped},
                "source": "Mapping preview",
                "source_type": "preview",
                "target_tab": "dataprep",
                "file_path": "",
                "row_index": int(preview_row.get("index") or len(scan_rows)),
            })

    row_texts: list[str] = []
    row_fingerprints: list[str] = []
    row_sources: list[str] = []
    row_targets: list[str] = []
    row_token_sets: list[set[str]] = []
    row_items: list[dict[str, Any]] = []
    field_counter: Counter[str] = Counter()
    pii_counts: Counter[str] = Counter()
    pii_by_source: Counter[str] = Counter()
    pii_preview_items: list[dict[str, Any]] = []
    low_quality_reasons: Counter[str] = Counter()
    low_quality_by_source: Counter[str] = Counter()
    low_quality_preview_items: list[dict[str, Any]] = []
    synthetic_pending_preview_items: list[dict[str, Any]] = []
    prepared_pending_synthetic_preview_items: list[dict[str, Any]] = []
    prepared_pending_synthetic = 0

    for item in scan_rows:
        row = item.get("row") if isinstance(item.get("row"), dict) else {}
        source = str(item.get("source") or "Project sample")
        source_type = str(item.get("source_type") or "")
        target_tab = str(item.get("target_tab") or "data")
        text = _quality_scan_text(row)
        fingerprint = _quality_text_fingerprint(text)
        if text:
            row_texts.append(text)
            row_fingerprints.append(fingerprint)
            row_sources.append(source)
            row_targets.append(target_tab)
            row_token_sets.append(_quality_tokens(text))
            row_items.append(item)
        for field in row.keys():
            field_counter[str(field)] += 1

        row_pii_counts = _pii_pci_signal_counts(text)
        if row_pii_counts:
            pii_counts.update(row_pii_counts)
            pii_by_source[source] += sum(row_pii_counts.values())
            pii_preview_items.append(item)

        low_quality_reason = _low_quality_reason(text)
        if low_quality_reason is not None:
            low_quality_reasons[low_quality_reason] += 1
            low_quality_by_source[source] += 1
            low_quality_preview_items.append(item)

        status = str(row.get("review_status") or row.get("status") or "").strip().lower()
        if source_type == DatasetType.SYNTHETIC.value and status == "pending":
            synthetic_pending_preview_items.append(item)
        if (
            source_type in {DatasetType.TRAIN.value, DatasetType.VALIDATION.value, DatasetType.TEST.value}
            and str(row.get("synth_source") or "").strip()
            and status == "pending"
        ):
            prepared_pending_synthetic += 1
            prepared_pending_synthetic_preview_items.append(item)

    low_quality_docs_result = await db.execute(
        select(func.count(RawDocument.id))
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            RawDocument.quality_score.is_not(None),
            RawDocument.quality_score < 0.45,
        )
    )
    low_quality_document_count = int(low_quality_docs_result.scalar_one() or 0)

    exact_counter: Counter[str] = Counter(
        fingerprint for fingerprint in row_fingerprints if fingerprint
    )
    exact_indices: dict[str, list[int]] = {}
    for index, fingerprint in enumerate(row_fingerprints):
        if fingerprint:
            exact_indices.setdefault(fingerprint, []).append(index)
    duplicate_count = sum(count - 1 for count in exact_counter.values() if count > 1)
    duplicate_by_source: Counter[str] = Counter()
    duplicate_preview_indices: set[int] = set()
    seen_exact: set[str] = set()
    for fingerprint, source in zip(row_fingerprints, row_sources, strict=False):
        if not fingerprint:
            continue
        if exact_counter[fingerprint] > 1:
            duplicate_preview_indices.update(exact_indices.get(fingerprint, [])[:4])
            if fingerprint in seen_exact:
                duplicate_by_source[source] += 1
            else:
                seen_exact.add(fingerprint)

    near_duplicate_pairs = 0
    near_duplicate_by_source: Counter[str] = Counter()
    compare_limit = min(len(row_token_sets), 160)
    for left in range(compare_limit):
        left_tokens = row_token_sets[left]
        if len(left_tokens) < 5:
            continue
        for right in range(left + 1, compare_limit):
            if row_fingerprints[left] == row_fingerprints[right]:
                continue
            right_tokens = row_token_sets[right]
            if len(right_tokens) < 5:
                continue
            union = left_tokens | right_tokens
            if not union:
                continue
            similarity = len(left_tokens & right_tokens) / len(union)
            if similarity >= 0.88:
                near_duplicate_pairs += 1
                near_duplicate_by_source[row_sources[right]] += 1
                duplicate_preview_indices.update({left, right})
    duplicate_preview_items = [
        row_items[index]
        for index in sorted(duplicate_preview_indices)
        if 0 <= index < len(row_items)
    ][:5]

    mapping_summary = mapping.get("summary") if isinstance(mapping.get("summary"), dict) else {}
    mapping_preview_rows = mapping.get("preview_rows") if isinstance(mapping.get("preview_rows"), list) else []
    required_fields = [
        str(item)
        for item in list(mapping_summary.get("required_fields") or [])
        if str(item).strip()
    ]
    effective_mapping = (
        mapping.get("effective_mapping") if isinstance(mapping.get("effective_mapping"), dict) else {}
    )
    field_mapping = (
        effective_mapping.get("field_mapping")
        if isinstance(effective_mapping.get("field_mapping"), dict)
        else {}
    )
    required_gaps = [
        str(item)
        for item in list(mapping_summary.get("required_fields_below_100") or [])
        if str(item).strip()
    ]
    required_missing_count = 0
    required_missing_preview_items: list[dict[str, Any]] = []
    for preview_row in mapping_preview_rows:
        if not isinstance(preview_row, dict):
            continue
        mapped = preview_row.get("mapped") if isinstance(preview_row.get("mapped"), dict) else {}
        missing_in_row = _quality_required_missing_count(mapped, required_fields)
        required_missing_count += missing_in_row
        if missing_in_row:
            raw = preview_row.get("raw") if isinstance(preview_row.get("raw"), dict) else {}
            required_missing_preview_items.append({
                "row": {**raw, **mapped},
                "source": "Mapping preview",
                "source_type": "preview",
                "target_tab": "dataprep",
                "file_path": "",
                "row_index": int(preview_row.get("index") or len(required_missing_preview_items)),
            })
    for coverage in list(mapping_summary.get("required_field_coverage") or []):
        if isinstance(coverage, dict):
            required_missing_count += int(coverage.get("missing") or 0)
    for item in scan_rows:
        if str(item.get("source_type") or "") not in {DatasetType.RAW.value, "preview"}:
            continue
        row = item.get("row") if isinstance(item.get("row"), dict) else {}
        missing_in_source_row = 0
        for field in required_fields:
            source_field = str(field_mapping.get(field) or field)
            if not _field_has_value(row.get(source_field)):
                required_missing_count += 1
                missing_in_source_row += 1
        if missing_in_source_row:
            required_missing_preview_items.append(item)

    prepare_splits = prepare_dataset.get("splits") if isinstance(prepare_dataset.get("splits"), dict) else {}
    split_items = prepare_splits.get("items") if isinstance(prepare_splits.get("items"), list) else []
    split_fingerprints: dict[str, set[str]] = {}
    split_row_counts: dict[str, int] = {}
    split_preview_by_fingerprint: dict[str, dict[str, dict[str, Any]]] = {}
    for split in split_items:
        if not isinstance(split, dict):
            continue
        split_key = str(split.get("key") or "")
        rows = _load_jsonl_dicts(str(split.get("file_path") or ""), limit=2000)
        split_fingerprints[split_key] = _quality_split_fingerprints(rows)
        split_row_counts[split_key] = len(rows)
        split_preview_by_fingerprint.setdefault(split_key, {})
        for row_index, row in enumerate(rows):
            fingerprint = _quality_text_fingerprint(_quality_scan_text(row))
            if not fingerprint or fingerprint in split_preview_by_fingerprint[split_key]:
                continue
            split_preview_by_fingerprint[split_key][fingerprint] = {
                "row": row,
                "source": f"{split_key or 'prepared'} split",
                "source_type": split_key or "prepared",
                "target_tab": "dataprep",
                "file_path": str(split.get("file_path") or ""),
                "row_index": row_index,
            }

    leakage_pairs: list[str] = []
    leakage_overlap_count = 0
    leakage_by_source: Counter[str] = Counter()
    leakage_preview_items: list[dict[str, Any]] = []
    for left_key, right_key in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_fingerprints.get(left_key, set()) & split_fingerprints.get(right_key, set())
        if overlap:
            leakage_overlap_count += len(overlap)
            leakage_pairs.append(f"{left_key}/{right_key}: {len(overlap)} overlapping row(s)")
            leakage_by_source[f"{left_key} split"] += len(overlap)
            leakage_by_source[f"{right_key} split"] += len(overlap)
            for fingerprint in list(overlap)[:3]:
                for split_key in (left_key, right_key):
                    preview_item = split_preview_by_fingerprint.get(split_key, {}).get(fingerprint)
                    if preview_item:
                        leakage_preview_items.append(preview_item)
    leakage_preview_items = leakage_preview_items[:5]

    review_totals = review_queue.get("totals") if isinstance(review_queue.get("totals"), dict) else {}
    synthetic_pending = int(review_totals.get("synthetic_pending") or 0)
    gold_review_needed = int(review_totals.get("gold_review_needed") or 0)
    annotation_review_needed = int(review_totals.get("annotation_review_needed") or 0)
    annotation_unpromoted = int(review_totals.get("annotation_labeled_unpromoted") or 0)

    checks: list[dict[str, Any]] = []
    source_groups: dict[str, Counter[str]] = {}
    owner_groups: dict[str, Counter[str]] = {}
    domain_groups: dict[str, Counter[str]] = {}
    source_targets: dict[str, str] = {}
    owner_targets: dict[str, str] = {}
    domain_targets: dict[str, str] = {}

    def preview_items_for_check(check: dict[str, Any]) -> list[dict[str, Any]]:
        fingerprint = " ".join([
            str(check.get("id") or ""),
            str(check.get("category") or ""),
            str(check.get("label") or ""),
        ]).lower()
        if "pii" in fingerprint or "pci" in fingerprint or "privacy" in fingerprint or "redaction" in fingerprint:
            return pii_preview_items[:5]
        if "duplicate" in fingerprint:
            return duplicate_preview_items[:5]
        if "leakage" in fingerprint or "split" in fingerprint:
            return leakage_preview_items[:5]
        if "required" in fingerprint or "field" in fingerprint or "mapping" in fingerprint or "context" in fingerprint or "citation" in fingerprint:
            return (required_missing_preview_items or scan_rows)[:5]
        if "low-quality" in fingerprint or "low_quality" in fingerprint or "quality row" in fingerprint:
            return low_quality_preview_items[:5]
        if "synthetic" in fingerprint or "review" in fingerprint or "gate" in fingerprint:
            return (synthetic_pending_preview_items + prepared_pending_synthetic_preview_items)[:5]
        if "domain" in fingerprint:
            return scan_rows[:5]
        if str(check.get("status") or "") == "ready":
            return scan_rows[:3]
        return scan_rows[:5]

    def add_check(check: dict[str, Any], source_counts: Counter[str] | None = None) -> None:
        if not isinstance(check.get("drilldown"), dict):
            check["drilldown"] = _quality_check_drilldown(
                check,
                rows=preview_items_for_check(check),
                source_counts=source_counts,
            )
        checks.append(check)
        severity = str(check.get("severity") or "info")
        severity_count = int(check.get("count") or 0)
        if severity_count <= 0:
            severity_count = 1
        owner = str(check.get("workflow_owner") or "Data Studio")
        domain_key = str(check.get("domain_label") or domain_label)
        owner_groups.setdefault(owner, Counter())[severity] += severity_count
        owner_targets.setdefault(owner, str(check.get("target_tab") or "data"))
        domain_groups.setdefault(domain_key, Counter())[severity] += severity_count
        domain_targets.setdefault(domain_key, str(check.get("target_tab") or "domain"))
        if source_counts:
            for source, count in source_counts.items():
                source_groups.setdefault(source, Counter())[severity] += int(count or 1)
                source_targets.setdefault(source, str(check.get("target_tab") or "data"))
        else:
            source = str(check.get("source") or "Project sample")
            source_groups.setdefault(source, Counter())[severity] += severity_count
            source_targets.setdefault(source, str(check.get("target_tab") or "data"))

    sensitive_strong_count = int(pii_counts.get("ssn", 0) + pii_counts.get("credit_card", 0) + pii_counts.get("cvv", 0))
    pii_total = sum(pii_counts.values())
    if pii_total:
        severity: IssueSeverity = "blocker" if sensitive_strong_count else "warning"
        status = "blocked" if severity == "blocker" else "attention"
        detected_types = ", ".join(sorted(pii_counts.keys()))
        add_check(
            _quality_check(
                "pii_pci_sensitive_values",
                "PII/PCI patterns detected",
                "safety",
                status,
                severity,
                f"Found {pii_total} deterministic sensitive-data signal(s): {detected_types}. Values are not shown in Data Studio.",
                count=pii_total,
                target_tab="data",
                workflow_owner="Source Ingestion",
                source=_quality_top_source(pii_by_source),
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[
                    "Regex checks cover email, phone, SSN-like, Luhn-valid payment card, and CVV-like patterns.",
                    "Mask, remove, or synthesize sensitive values before preparing training data.",
                ],
                action_label="Inspect sources",
            ),
            pii_by_source,
        )
    else:
        add_check(
            _quality_check(
                "pii_pci_no_patterns",
                "No PII/PCI regex hits",
                "safety",
                "ready",
                "info",
                "Deterministic regex checks did not find common PII/PCI patterns in the scanned sample.",
                count=0,
                target_tab="data",
                workflow_owner="Source Ingestion",
                source="Project sample",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=["This does not replace human review for regulated data."],
                action_label="Open sources",
            )
        )

    duplicate_signal_count = duplicate_count + near_duplicate_pairs
    duplicate_sources = duplicate_by_source + near_duplicate_by_source
    if duplicate_signal_count:
        add_check(
            _quality_check(
                "duplicate_or_near_duplicate_rows",
                "Duplicate or near-duplicate rows",
                "quality",
                "attention",
                "warning",
                f"Found {duplicate_count} exact duplicate row(s) and {near_duplicate_pairs} near-duplicate pair(s) in the scanned sample.",
                count=duplicate_signal_count,
                target_tab="dataprep",
                workflow_owner="Data Prep",
                source=_quality_top_source(duplicate_sources),
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=["Deduplicate before splitting so repeated examples do not inflate evaluation confidence."],
                action_label="Open Data Prep",
            ),
            duplicate_sources,
        )
    else:
        add_check(
            _quality_check(
                "duplicate_rows_clear",
                "Duplicate scan clear",
                "quality",
                "ready",
                "info",
                "No exact or high-overlap near duplicates were found in the scanned sample.",
                count=0,
                target_tab="dataprep",
                workflow_owner="Data Prep",
                source="Project sample",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[],
                action_label="Open Data Prep",
            )
        )

    if required_gaps or required_missing_count:
        gap_text = ", ".join(required_gaps) if required_gaps else "required fields"
        add_check(
            _quality_check(
                "required_fields_missing",
                "Missing required fields",
                "quality",
                "attention",
                "warning",
                f"Mapping coverage is incomplete for {gap_text}.",
                count=max(required_missing_count, len(required_gaps)),
                target_tab="dataprep",
                workflow_owner="Data Prep",
                source="Mapping preview",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[
                    f"{int(mapping_summary.get('sampled_records') or 0)} row(s) sampled for mapping.",
                    f"{int(mapping_summary.get('mapped_records') or 0)} row(s) mapped.",
                ],
                action_label="Review mapping",
            )
        )
    else:
        add_check(
            _quality_check(
                "required_fields_present",
                "Required fields covered",
                "quality",
                "ready",
                "info",
                "Required recipe fields are present in the mapping preview.",
                count=0,
                target_tab="dataprep",
                workflow_owner="Data Prep",
                source="Mapping preview",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[],
                action_label="Review mapping",
            )
        )

    if leakage_overlap_count:
        add_check(
            _quality_check(
                "train_validation_test_leakage",
                "Train/validation/test leakage risk",
                "leakage",
                "blocked",
                "blocker",
                f"Found {leakage_overlap_count} overlapping row fingerprint(s) across prepared splits.",
                count=leakage_overlap_count,
                target_tab="dataprep",
                workflow_owner="Data Prep",
                source="Prepared splits",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=leakage_pairs[:4],
                action_label="Refresh splits",
            ),
            leakage_by_source,
        )
    elif any(split_row_counts.values()):
        add_check(
            _quality_check(
                "split_leakage_clear",
                "Split leakage scan clear",
                "leakage",
                "ready",
                "info",
                "No identical row fingerprints were found across prepared train/validation/test files.",
                count=0,
                target_tab="dataprep",
                workflow_owner="Data Prep",
                source="Prepared splits",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[],
                action_label="Open Data Prep",
            )
        )
    else:
        add_check(
            _quality_check(
                "split_leakage_waiting_for_splits",
                "Split leakage scan waiting",
                "leakage",
                "ready",
                "info",
                "Leakage checks will run after Dataset Prep creates train, validation, and test files.",
                count=0,
                target_tab="dataprep",
                workflow_owner="Data Prep",
                source="Prepared splits",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[],
                action_label="Open Data Prep",
            )
        )

    low_quality_total = sum(low_quality_reasons.values()) + low_quality_document_count
    if low_quality_total:
        reason_text = ", ".join(f"{reason.replace('_', ' ')}: {count}" for reason, count in low_quality_reasons.most_common(3))
        evidence = [reason_text] if reason_text else []
        if low_quality_document_count:
            evidence.append(f"{low_quality_document_count} source document(s) have quality_score below 0.45.")
        add_check(
            _quality_check(
                "low_quality_rows",
                "Low-quality rows",
                "quality",
                "attention",
                "warning",
                f"Found {low_quality_total} low-quality row or document signal(s).",
                count=low_quality_total,
                target_tab="data",
                workflow_owner="Source Ingestion",
                source=_quality_top_source(low_quality_by_source),
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=evidence,
                action_label="Inspect sources",
            ),
            low_quality_by_source,
        )
    else:
        add_check(
            _quality_check(
                "low_quality_rows_clear",
                "Low-quality row scan clear",
                "quality",
                "ready",
                "info",
                "No empty, placeholder, very short, or repeated-character rows were found in the scanned sample.",
                count=0,
                target_tab="data",
                workflow_owner="Source Ingestion",
                source="Project sample",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[],
                action_label="Open sources",
            )
        )

    synthetic_contamination_count = synthetic_pending + prepared_pending_synthetic
    if synthetic_contamination_count:
        severity = "blocker" if prepared_pending_synthetic else "warning"
        synthetic_contamination_sources: Counter[str] = Counter()
        if synthetic_pending:
            synthetic_contamination_sources["Synthetic review queue"] = synthetic_pending
        for item in synthetic_pending_preview_items + prepared_pending_synthetic_preview_items:
            synthetic_contamination_sources[str(item.get("source") or "Synthetic review queue")] += 1
        add_check(
            _quality_check(
                "synthetic_review_contamination",
                "Synthetic review contamination",
                "review",
                "blocked" if severity == "blocker" else "attention",
                severity,
                (
                    f"{synthetic_pending} synthetic row(s) are pending review"
                    f" and {prepared_pending_synthetic} pending synthetic row(s) appear in prepared splits."
                ),
                count=synthetic_contamination_count,
                target_tab="synthetic",
                workflow_owner="Review",
                source="Synthetic review queue",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=["Accepted synthetic rows can be used downstream; pending rows should stay gated."],
                action_label="Review synthetic rows",
            ),
            synthetic_contamination_sources,
        )
    else:
        add_check(
            _quality_check(
                "synthetic_review_clear",
                "Synthetic review gate clear",
                "review",
                "ready",
                "info",
                "No pending synthetic rows were found in the review queue or prepared split sample.",
                count=0,
                target_tab="synthetic",
                workflow_owner="Review",
                source="Synthetic review queue",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[],
                action_label="Open Synthetic",
            )
        )

    if gold_review_needed or annotation_review_needed or annotation_unpromoted:
        review_count = gold_review_needed + annotation_review_needed + annotation_unpromoted
        add_check(
            _quality_check(
                "human_review_items_open",
                "Human review items open",
                "review",
                "attention",
                "warning",
                f"{review_count} Gold Set or annotation review item(s) still need attention.",
                count=review_count,
                target_tab="annotate" if annotation_review_needed or annotation_unpromoted else "goldset",
                workflow_owner="Review",
                source="Review Queue",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[
                    f"Gold Set review needed: {gold_review_needed}.",
                    f"Annotation review needed: {annotation_review_needed}.",
                    f"Annotation labels not promoted: {annotation_unpromoted}.",
                ],
                action_label="Open Review",
            )
        )

    domain_field_names = {
        field.lower()
        for field in field_counter.keys()
        if field.lower() not in {"synth_source", "review_status", "generated_at", "model"}
    }
    if domain_confidence >= _DOMAIN_SETUP_MIN_CONFIDENCE and domain_id == "policy_qa":
        has_policy_context = any(
            marker in field
            for field in domain_field_names
            for marker in ("context", "source", "section", "policy", "citation", "reference")
        )
        if not has_policy_context:
            add_check(
                _quality_check(
                    "domain_policy_context_missing",
                    "Policy context missing",
                    "domain",
                    "attention",
                    "warning",
                    "Policy Q&A was detected, but scanned fields do not expose source section, context, or citation signals.",
                    count=1,
                    target_tab="domain",
                    workflow_owner="Domain Managers",
                    source="Domain detection",
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=["Policy answers are safer when training rows preserve the governing policy context."],
                    action_label="Open Domain Managers",
                )
            )
        else:
            add_check(
                _quality_check(
                    "domain_policy_context_present",
                    "Policy context present",
                    "domain",
                    "ready",
                    "info",
                    "Policy-specific fields include context, source, section, or citation signals.",
                    count=0,
                    target_tab="domain",
                    workflow_owner="Domain Managers",
                    source="Domain detection",
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=[],
                    action_label="Open Domain Managers",
                )
            )
    elif domain_confidence >= _DOMAIN_SETUP_MIN_CONFIDENCE and domain_id == "support_faq" and pii_total:
        add_check(
            _quality_check(
                "domain_support_privacy_review",
                "Support privacy review",
                "domain",
                "attention",
                "warning",
                "Support FAQ data often contains account details; review detected PII/PCI signals before training.",
                count=pii_total,
                target_tab="domain",
                workflow_owner="Domain Managers",
                source="Domain detection",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=["Consider redaction, synthetic placeholders, and escalation examples."],
                action_label="Open Domain Managers",
            )
        )
    elif domain_confidence >= _DOMAIN_SETUP_MIN_CONFIDENCE and domain_id == "pii_pci_detection":
        has_label_field = any(
            marker in field
            for field in domain_field_names
            for marker in ("label", "entity", "redaction", "pii", "pci", "category")
        )
        if not has_label_field:
            add_check(
                _quality_check(
                    "domain_pii_labels_missing",
                    "PII/PCI label fields missing",
                    "domain",
                    "attention",
                    "warning",
                    "PII/PCI Detection was detected, but scanned fields do not expose entity, label, or redaction targets.",
                    count=1,
                    target_tab="domain",
                    workflow_owner="Domain Managers",
                    source="Domain detection",
                    domain_id=domain_id,
                    domain_label=domain_label,
                    evidence=["Detection models need explicit labels or spans, not just raw sensitive text."],
                    action_label="Open Domain Managers",
                )
            )
    elif domain_confidence < _DOMAIN_SETUP_MIN_CONFIDENCE:
        add_check(
            _quality_check(
                "domain_specific_checks_waiting",
                "Domain-specific checks waiting",
                "domain",
                "ready",
                "info",
                "Domain-specific checks become more precise after BrewSLM sees stronger domain evidence or an applied domain pack.",
                count=0,
                target_tab="domain",
                workflow_owner="Domain Managers",
                source="Domain detection",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[],
                action_label="Open Domain Managers",
            )
        )

    domain_authored_checks, domain_authored_summary = _quality_domain_authored_checks(
        domain_context=domain_contract_context,
        scan_rows=scan_rows,
        row_texts=row_texts,
        pii_total=pii_total,
        duplicate_signal_count=duplicate_signal_count,
        leakage_overlap_count=leakage_overlap_count,
        synthetic_pending=synthetic_pending,
        gold_review_needed=gold_review_needed,
        annotation_review_needed=annotation_review_needed,
        annotation_unpromoted=annotation_unpromoted,
        domain_id=domain_id,
        domain_label=domain_label,
    )
    for check in domain_authored_checks:
        add_check(check)

    if not scan_rows:
        add_check(
            _quality_check(
                "quality_scan_no_rows",
                "No rows available for scanning",
                "quality",
                "blocked",
                "blocker",
                "Add source rows before BrewSLM can run deterministic quality and safety scans.",
                count=1,
                target_tab="data",
                workflow_owner="Source Ingestion",
                source="Project sample",
                domain_id=domain_id,
                domain_label=domain_label,
                evidence=[],
                action_label="Add sources",
            )
        )

    issues = [
        _quality_issue_from_check(check)
        for check in checks
        if check.get("severity") in {"blocker", "warning"}
    ]
    blocker_count = sum(1 for item in issues if item["severity"] == "blocker")
    warning_count = sum(1 for item in issues if item["severity"] == "warning")
    info_count = sum(1 for check in checks if check.get("severity") == "info")
    if blocker_count:
        verdict: QualitySafetyVerdict = "blocked"
    elif warning_count:
        verdict = "attention"
    else:
        verdict = "ready"

    status_groups = [
        {
            "status": "blocked",
            "label": "Blockers",
            "count": blocker_count,
            "target_tab": "dataprep" if leakage_overlap_count else "data",
        },
        {
            "status": "attention",
            "label": "Warnings",
            "count": warning_count,
            "target_tab": "synthetic" if synthetic_pending else "dataprep",
        },
        {
            "status": "ready",
            "label": "Ready checks",
            "count": info_count,
            "target_tab": "data",
        },
    ]

    entry_points = [
        {
            "label": "Open Source Ingestion",
            "target_tab": "data",
            "reason": "Inspect source rows and ingestion quality signals.",
            "requires_confirmation": True,
        },
        {
            "label": "Open Data Prep",
            "target_tab": "dataprep",
            "reason": "Review mapping, dedupe, split, and leakage checks.",
            "requires_confirmation": True,
        },
        {
            "label": "Open Gold Set",
            "target_tab": "goldset",
            "reason": "Review trusted examples and field coverage before training decisions depend on them.",
            "requires_confirmation": True,
        },
        {
            "label": "Open Review",
            "target_tab": "synthetic" if synthetic_pending else "annotate",
            "reason": "Clear synthetic, Gold Set, or annotation review items before preparation.",
            "requires_confirmation": True,
        },
        {
            "label": "Open Domain Managers",
            "target_tab": "domain",
            "reason": "Tune domain-specific safety and policy checks.",
            "requires_confirmation": True,
        },
    ]

    return {
        "project_id": project_id,
        "verdict": verdict,
        "read_only": True,
        "auto_apply": False,
        "source_of_truth": "deterministic_data_studio_checks",
        "summary": {
            "scanned_rows": len(scan_rows),
            "sampled_rows": len(row_texts),
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "pii_pci_signal_count": pii_total,
            "duplicate_signal_count": duplicate_signal_count,
            "leakage_overlap_count": leakage_overlap_count,
            "low_quality_signal_count": low_quality_total,
            "pending_review_count": synthetic_pending + gold_review_needed + annotation_review_needed + annotation_unpromoted,
            "domain_signal_count": 1 if domain_confidence >= _DOMAIN_SETUP_MIN_CONFIDENCE else 0,
            "domain_authored_check_count": int(domain_authored_summary.get("check_count") or 0),
            "domain_authored_warning_count": int(domain_authored_summary.get("warning_count") or 0),
            "domain_authored_blocker_count": int(domain_authored_summary.get("blocker_count") or 0),
        },
        "domain": {
            "id": domain_id,
            "label": domain_label,
            "confidence": round(domain_confidence, 4),
            "source": detected_domain.get("source"),
        },
        "domain_authored": domain_authored_summary,
        "checks": checks,
        "findings_by_source": _quality_group_rows(source_groups, source_targets),
        "findings_by_status": status_groups,
        "findings_by_domain": _quality_group_rows(domain_groups, domain_targets),
        "findings_by_owner": _quality_group_rows(owner_groups, owner_targets),
        "issues": issues,
        "entry_points": entry_points,
        "assist": {
            "available": True,
            "default_provider": "ollama",
            "openai_compatible_supported": True,
            "purpose": "explanations_only",
            "auto_apply": False,
            "target_tab": "assist",
        },
        "power_details": {
            "pii_pci_counts": dict(pii_counts),
            "low_quality_reasons": dict(low_quality_reasons),
            "required_fields": required_fields,
            "required_fields_below_100": required_gaps,
            "split_row_counts": split_row_counts,
            "domain_evidence": domain.get("evidence") if isinstance(domain.get("evidence"), list) else [],
            "domain_authored_check_ids": [check.get("id") for check in domain_authored_checks],
            "review_totals": review_totals,
        },
    }


def _prepared_dataset_type_order(dataset_type: DatasetType) -> int:
    order = {
        DatasetType.TRAIN: 0,
        DatasetType.VALIDATION: 1,
        DatasetType.TEST: 2,
    }
    return order.get(dataset_type, 99)


def _version_payload(version: DatasetVersion) -> dict[str, Any]:
    manifest = version.manifest if isinstance(version.manifest, dict) else {}
    split = str(manifest.get("split") or "").strip()
    count = manifest.get("count")
    return {
        "id": int(version.id),
        "version": int(version.version or 0),
        "record_count": int(version.record_count or 0),
        "file_path": version.file_path,
        "file_exists": _file_exists(version.file_path),
        "created_at": version.created_at.isoformat() if version.created_at else None,
        "manifest_split": split or None,
        "manifest_count": int(count) if isinstance(count, int) else None,
        "manifest": manifest,
    }


def _dataset_version_history_payload(
    dataset: Dataset,
    versions: list[DatasetVersion],
) -> dict[str, Any]:
    ordered_versions = sorted(versions, key=lambda item: (int(item.version or 0), int(item.id)), reverse=True)
    return {
        "dataset_id": int(dataset.id),
        "dataset_name": dataset.name,
        "dataset_type": dataset.dataset_type.value,
        "row_count": int(dataset.record_count or 0),
        "file_path": dataset.file_path or "",
        "file_exists": _file_exists(dataset.file_path),
        "is_locked": bool(dataset.is_locked),
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
        "version_count": len(ordered_versions),
        "latest_version": _version_payload(ordered_versions[0]) if ordered_versions else None,
        "versions": [_version_payload(version) for version in ordered_versions[:8]],
    }


def _dataset_version_artifact_payload(
    *,
    split_key: str,
    label: str,
    dataset_type: DatasetType,
    dataset: Dataset | None,
    versions: list[DatasetVersion],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_key = _manifest_split_key(split_key)
    manifest_splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    manifest_versions = (
        manifest.get("dataset_versions")
        if isinstance(manifest.get("dataset_versions"), dict)
        else {}
    )
    manifest_file_paths = (
        manifest.get("file_paths") if isinstance(manifest.get("file_paths"), dict) else {}
    )
    manifest_file_hashes = (
        manifest.get("file_hashes") if isinstance(manifest.get("file_hashes"), dict) else {}
    )
    ordered_versions = sorted(versions, key=lambda item: (int(item.version or 0), int(item.id)))
    latest_version = ordered_versions[-1] if ordered_versions else None
    manifest_version_raw = manifest_versions.get(manifest_key)
    try:
        manifest_version = int(manifest_version_raw) if manifest_version_raw is not None else None
    except (TypeError, ValueError):
        manifest_version = None
    manifest_count = int(manifest_splits.get(manifest_key) or 0)
    dataset_count = int(getattr(dataset, "record_count", 0) or 0)
    latest_count = int(getattr(latest_version, "record_count", 0) or 0)
    file_path = str(getattr(dataset, "file_path", "") or manifest_file_paths.get(manifest_key) or "")
    latest_version_number = int(latest_version.version or 0) if latest_version is not None else None
    version_matches_manifest = (
        manifest_version is not None
        and latest_version_number is not None
        and manifest_version == latest_version_number
    )
    if manifest_count <= 0:
        row_count_matches_manifest = dataset_count <= 0 and latest_count <= 0
    else:
        row_count_matches_manifest = (
            dataset_count == manifest_count
            and (latest_version is None or latest_count == manifest_count)
        )

    return {
        "key": split_key,
        "manifest_key": manifest_key,
        "label": label,
        "dataset_type": dataset_type.value,
        "dataset_id": int(dataset.id) if dataset is not None else None,
        "dataset_name": dataset.name if dataset is not None else None,
        "row_count": dataset_count,
        "file_path": file_path,
        "file_exists": _file_exists(file_path),
        "version_count": len(ordered_versions),
        "latest_version": _version_payload(latest_version) if latest_version is not None else None,
        "latest_version_number": latest_version_number,
        "manifest_count": manifest_count,
        "manifest_version": manifest_version,
        "manifest_file_path": str(manifest_file_paths.get(manifest_key) or ""),
        "manifest_file_hash": str(manifest_file_hashes.get(manifest_key) or ""),
        "version_matches_manifest": version_matches_manifest,
        "row_count_matches_manifest": row_count_matches_manifest,
    }


def _dataset_version_signal(
    signal_id: str,
    label: str,
    status: str,
    message: str,
    *,
    target_tab: str,
) -> dict[str, str]:
    return {
        "id": signal_id,
        "label": label,
        "status": status,
        "message": message,
        "target_tab": target_tab,
    }


async def build_data_studio_dataset_versions(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a read-only prepared dataset version summary for Data Studio."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    datasets_result = await db.execute(
        select(Dataset)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [DatasetType.TRAIN, DatasetType.VALIDATION, DatasetType.TEST]
            ),
        )
        .order_by(Dataset.dataset_type.asc(), Dataset.updated_at.desc(), Dataset.id.asc())
    )
    prepared_datasets = list(datasets_result.scalars().all())
    prepared_dataset_ids = [int(dataset.id) for dataset in prepared_datasets]
    versions_by_dataset: dict[int, list[DatasetVersion]] = {
        dataset_id: [] for dataset_id in prepared_dataset_ids
    }
    if prepared_dataset_ids:
        versions_result = await db.execute(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id.in_(prepared_dataset_ids))
            .order_by(DatasetVersion.dataset_id.asc(), DatasetVersion.version.asc(), DatasetVersion.id.asc())
        )
        for version in versions_result.scalars().all():
            versions_by_dataset.setdefault(int(version.dataset_id), []).append(version)

    datasets_by_type: dict[DatasetType, Dataset] = {}
    for dataset in sorted(prepared_datasets, key=lambda item: item.updated_at, reverse=True):
        datasets_by_type.setdefault(dataset.dataset_type, dataset)

    manifest, manifest_meta = _read_prepared_manifest(project_id)
    artifacts = [
        _dataset_version_artifact_payload(
            split_key=split_key,
            label=label,
            dataset_type=dataset_type,
            dataset=datasets_by_type.get(dataset_type),
            versions=versions_by_dataset.get(int(datasets_by_type[dataset_type].id), [])
            if dataset_type in datasets_by_type
            else [],
            manifest=manifest,
        )
        for split_key, label, dataset_type in _PREPARED_SPLIT_SPECS
    ]
    history = [
        _dataset_version_history_payload(dataset, versions_by_dataset.get(int(dataset.id), []))
        for dataset in sorted(
            prepared_datasets,
            key=lambda item: (_prepared_dataset_type_order(item.dataset_type), int(item.id)),
        )
    ]

    total_version_count = sum(int(item.get("version_count") or 0) for item in history)
    latest_total_rows = sum(int(item.get("row_count") or 0) for item in artifacts)
    all_required_splits = all(int(item.get("row_count") or 0) > 0 for item in artifacts)
    all_versions_present = all(int(item.get("version_count") or 0) > 0 for item in artifacts)
    all_files_present = all(bool(item.get("file_exists")) for item in artifacts if int(item.get("row_count") or 0) > 0)
    all_manifest_refs_match = all(bool(item.get("version_matches_manifest")) for item in artifacts)
    all_counts_match = all(bool(item.get("row_count_matches_manifest")) for item in artifacts)
    file_hashes_present = all(bool(item.get("manifest_file_hash")) for item in artifacts)
    manifest_readable = bool(manifest_meta.get("readable"))
    manifest_exists = bool(manifest_meta.get("exists"))

    latest_created_at_values = [
        str(version.get("created_at"))
        for item in history
        for version in [item.get("latest_version")]
        if isinstance(version, dict) and version.get("created_at")
    ]
    latest_created_at = max(latest_created_at_values) if latest_created_at_values else None

    recipe_payload = _recipe_payload(project)
    try:
        runtime = await resolve_project_domain_runtime(db, project_id)
        domain_payload = await _domain_applied_summary(db, runtime)
    except ValueError:
        runtime = {}
        domain_payload = {}

    included_types_raw = manifest.get("included_types")
    included_source_types = [
        str(item)
        for item in (included_types_raw if isinstance(included_types_raw, list) else [])
        if str(item).strip()
    ]
    manifest_dataset_versions = (
        manifest.get("dataset_versions")
        if isinstance(manifest.get("dataset_versions"), dict)
        else {}
    )
    manifest_file_hashes = (
        manifest.get("file_hashes") if isinstance(manifest.get("file_hashes"), dict) else {}
    )

    issues: list[dict[str, str]] = []
    if not prepared_datasets and not manifest_exists:
        issues.append(
            _issue(
                "dataset_versions_empty",
                "info",
                "No prepared dataset versions yet",
                "Run Dataset Prep to create versioned train, validation, and test artifacts.",
                action_label="Open Dataset Prep",
                target_tab="dataprep",
            )
        )
    elif not manifest_readable:
        issues.append(
            _issue(
                "dataset_versions_manifest_missing",
                "warning",
                "Prepared manifest is missing or unreadable",
                "Re-run Dataset Prep so version rows and split files have a reproducible manifest.",
                action_label="Open Dataset Prep",
                target_tab="dataprep",
            )
        )

    missing_artifacts = [
        str(item.get("label"))
        for item in artifacts
        if int(item.get("row_count") or 0) <= 0
    ]
    if prepared_datasets and missing_artifacts:
        issues.append(
            _issue(
                "dataset_versions_missing_split_artifacts",
                "warning",
                "Prepared split artifacts are incomplete",
                f"Missing prepared rows for: {', '.join(missing_artifacts)}.",
                action_label="Refresh Dataset Prep",
                target_tab="dataprep",
            )
        )

    missing_versions = [
        str(item.get("label"))
        for item in artifacts
        if int(item.get("row_count") or 0) > 0 and int(item.get("version_count") or 0) <= 0
    ]
    if missing_versions:
        issues.append(
            _issue(
                "dataset_versions_missing_rows",
                "warning",
                "Prepared datasets are not versioned",
                f"Missing DatasetVersion rows for: {', '.join(missing_versions)}.",
                action_label="Refresh Dataset Prep",
                target_tab="dataprep",
            )
        )

    stale_versions = [
        str(item.get("label"))
        for item in artifacts
        if int(item.get("row_count") or 0) > 0
        and manifest_readable
        and not bool(item.get("version_matches_manifest"))
    ]
    if stale_versions:
        issues.append(
            _issue(
                "dataset_versions_manifest_mismatch",
                "warning",
                "Manifest version references are stale",
                f"Latest version does not match manifest for: {', '.join(stale_versions)}.",
                action_label="Refresh Dataset Prep",
                target_tab="dataprep",
            )
        )

    count_mismatches = [
        str(item.get("label"))
        for item in artifacts
        if int(item.get("row_count") or 0) > 0
        and manifest_readable
        and not bool(item.get("row_count_matches_manifest"))
    ]
    if count_mismatches:
        issues.append(
            _issue(
                "dataset_versions_count_mismatch",
                "warning",
                "Manifest counts do not match artifacts",
                f"Row counts differ for: {', '.join(count_mismatches)}.",
                action_label="Refresh Dataset Prep",
                target_tab="dataprep",
            )
        )

    missing_files = [
        str(item.get("label"))
        for item in artifacts
        if int(item.get("row_count") or 0) > 0 and not bool(item.get("file_exists"))
    ]
    if missing_files:
        issues.append(
            _issue(
                "dataset_versions_files_missing",
                "warning",
                "Prepared files are missing",
                f"Expected prepared JSONL files were not found for: {', '.join(missing_files)}.",
                action_label="Refresh Dataset Prep",
                target_tab="dataprep",
            )
        )

    if manifest_readable and not file_hashes_present:
        issues.append(
            _issue(
                "dataset_versions_hashes_missing",
                "info",
                "Manifest hashes are incomplete",
                "File hashes help confirm that split files have not drifted since preparation.",
                action_label="Open Dataset Prep",
                target_tab="dataprep",
            )
        )

    if recipe_payload is None:
        issues.append(
            _issue(
                "dataset_versions_recipe_missing",
                "info",
                "Recipe context is missing",
                "A selected recipe makes version reuse easier to interpret for training and evaluation.",
                action_label="Choose recipe",
                target_tab="data",
            )
        )

    training_ready = (
        manifest_readable
        and all_required_splits
        and all_versions_present
        and all_files_present
        and all_manifest_refs_match
        and all_counts_match
        and int(artifacts[0].get("row_count") or 0) > 0
    )
    eval_ready = (
        manifest_readable
        and all_versions_present
        and all_files_present
        and int(artifacts[1].get("row_count") or 0) > 0
        and int(artifacts[2].get("row_count") or 0) > 0
    )
    any_versions = total_version_count > 0 or manifest_exists or bool(prepared_datasets)

    if not any_versions:
        verdict: DatasetVersionVerdict = "empty"
    elif any(item["severity"] == "warning" for item in issues) or not training_ready:
        verdict = "attention"
    else:
        verdict = "ready"

    reproducibility = [
        _dataset_version_signal(
            "manifest",
            "Prepared manifest",
            "met" if manifest_readable else ("attention" if manifest_exists else "missing"),
            "Prepared manifest is readable." if manifest_readable else "Create or refresh the prepared manifest in Dataset Prep.",
            target_tab="dataprep",
        ),
        _dataset_version_signal(
            "split_artifacts",
            "Split artifacts",
            "met" if all_required_splits else ("attention" if prepared_datasets else "missing"),
            "Train, validation, and test artifacts have rows." if all_required_splits else "Prepared train/validation/test artifacts are incomplete.",
            target_tab="dataprep",
        ),
        _dataset_version_signal(
            "version_refs",
            "Manifest version refs",
            "met" if all_manifest_refs_match and all_versions_present else ("attention" if total_version_count > 0 else "missing"),
            "Latest versions match manifest references." if all_manifest_refs_match and all_versions_present else "Refresh Dataset Prep so manifest references latest DatasetVersion rows.",
            target_tab="dataprep",
        ),
        _dataset_version_signal(
            "row_counts",
            "Row count alignment",
            "met" if all_counts_match and all_required_splits else ("attention" if any_versions else "missing"),
            "Artifact counts match manifest counts." if all_counts_match and all_required_splits else "Manifest counts and artifact counts need review.",
            target_tab="dataprep",
        ),
        _dataset_version_signal(
            "file_hashes",
            "File hashes",
            "met" if file_hashes_present else ("attention" if manifest_readable else "missing"),
            "Manifest includes split file hashes." if file_hashes_present else "File hashes are missing or incomplete in the manifest.",
            target_tab="dataprep",
        ),
        _dataset_version_signal(
            "source_inclusion",
            "Source inclusion",
            "met" if included_source_types else ("attention" if manifest_readable else "missing"),
            (
                f"Manifest records source types: {', '.join(included_source_types)}."
                if included_source_types
                else "Manifest does not record source inclusion."
            ),
            target_tab="dataprep",
        ),
    ]

    return {
        "project_id": project_id,
        "verdict": verdict,
        "read_only": True,
        "auto_apply": False,
        "source_of_truth": "deterministic_data_studio_checks",
        "summary": {
            "prepared_dataset_count": len(prepared_datasets),
            "total_version_count": total_version_count,
            "latest_total_rows": latest_total_rows,
            "latest_created_at": latest_created_at,
            "manifest_exists": manifest_exists,
            "manifest_readable": manifest_readable,
            "manifest_version_ref_count": len(manifest_dataset_versions),
            "training_reuse_ready": training_ready,
            "eval_reuse_ready": eval_ready,
        },
        "latest_artifacts": artifacts,
        "version_history": history,
        "manifest": {
            "exists": manifest_exists,
            "readable": manifest_readable,
            "path": manifest_meta.get("path"),
            "error": manifest_meta.get("error"),
            "created_at": manifest.get("created_at"),
            "seed": manifest.get("seed"),
            "total_entries": int(manifest.get("total_entries") or 0),
            "splits": manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {},
            "ratios": manifest.get("ratios") if isinstance(manifest.get("ratios"), dict) else {},
            "file_hashes": manifest_file_hashes,
            "dataset_versions": manifest_dataset_versions,
            "included_types": included_source_types,
            "chat_template": manifest.get("chat_template"),
            "adapter_id": manifest.get("adapter_id"),
            "task_profile": manifest.get("task_profile"),
        },
        "source_context": {
            "recipe": recipe_payload,
            "domain": domain_payload,
            "domain_runtime": {
                "domain_profile_applied": runtime.get("domain_profile_applied"),
                "domain_profile_source": runtime.get("domain_profile_source"),
                "domain_pack_applied": runtime.get("domain_pack_applied"),
                "domain_pack_source": runtime.get("domain_pack_source"),
            },
            "adapter_id": manifest.get("adapter_id"),
            "task_profile": manifest.get("task_profile"),
            "included_source_types": included_source_types,
        },
        "reuse_readiness": {
            "training": {
                "status": "ready" if training_ready else ("attention" if any_versions else "missing"),
                "target_tab": "training",
                "message": (
                    "Prepared train/validation/test versions are reusable for training."
                    if training_ready
                    else "Refresh prepared versions before treating this dataset as reusable for training."
                ),
            },
            "evaluation": {
                "status": "ready" if eval_ready else ("attention" if any_versions else "missing"),
                "target_tab": "eval",
                "message": (
                    "Validation and test artifacts are available for evaluation."
                    if eval_ready
                    else "Prepare validation and test artifacts before relying on evaluation reuse."
                ),
            },
        },
        "reproducibility": reproducibility,
        "issues": issues,
        "entry_points": [
            {
                "label": "Open Dataset Prep",
                "target_tab": "dataprep",
                "reason": "Create or refresh prepared dataset versions.",
                "requires_confirmation": True,
            },
            {
                "label": "Open Training",
                "target_tab": "training",
                "reason": "Use prepared split versions for training runs.",
                "requires_confirmation": False,
            },
            {
                "label": "Open Eval",
                "target_tab": "eval",
                "reason": "Use validation/test artifacts for evaluation.",
                "requires_confirmation": False,
            },
        ],
        "power_details": {
            "manifest": manifest if manifest_readable else {},
            "prepared_dataset_ids": prepared_dataset_ids,
            "runtime": runtime,
        },
    }


_COACH_SECTION_ORDER: dict[str, int] = {
    "overview": 0,
    "sources": 1,
    "mapping": 2,
    "domain": 3,
    "quality_safety": 4,
    "gold_set": 5,
    "synthetic_playbooks": 6,
    "synthetic_recommendations": 7,
    "synthetic_quality": 8,
    "review_queue": 9,
    "prepare_dataset": 10,
    "dataset_versions": 11,
}

_COACH_CONFIRMATION_TARGETS = {
    "data",
    "dataprep",
    "synthetic",
    "goldset",
    "annotate",
    "domain",
    "domain-packs",
    "domain-profiles",
    "training",
    "eval",
}


def _coach_priority(severity: str) -> str:
    if severity == "blocker":
        return "high"
    if severity == "warning":
        return "medium"
    return "low"


def _coach_severity_rank(severity: str) -> int:
    return {"blocker": 0, "warning": 1, "info": 2}.get(severity, 3)


def _coach_issue(
    *,
    section_id: str,
    section_label: str,
    issue: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    severity = str(issue.get("severity") or "info")
    if severity not in {"blocker", "warning", "info"}:
        severity = "info"
    target_tab = str(issue.get("target_tab") or "data")
    return {
        "id": f"{section_id}:{issue.get('id') or index}",
        "section_id": section_id,
        "section_label": section_label,
        "severity": severity,
        "priority": _coach_priority(severity),
        "title": str(issue.get("title") or section_label),
        "message": str(issue.get("message") or ""),
        "action_label": str(issue.get("action_label") or "Open"),
        "target_tab": target_tab,
        "requires_user_confirmation": target_tab in _COACH_CONFIRMATION_TARGETS,
        "sort": [
            _coach_severity_rank(severity),
            _COACH_SECTION_ORDER.get(section_id, 99),
            index,
        ],
    }


def _coach_status(
    *,
    verdict: Any,
    issues: list[dict[str, Any]],
) -> str:
    if any(str(item.get("severity") or "") == "blocker" for item in issues):
        return "blocked"
    if any(str(item.get("severity") or "") == "warning" for item in issues):
        return "attention"
    verdict_token = str(verdict or "").strip().lower()
    if verdict_token == "empty":
        return "empty"
    if verdict_token in {"attention", "needs_work", "unknown"}:
        return "attention"
    return "ready"


def _coach_section(
    *,
    section_id: str,
    label: str,
    verdict: Any,
    issues: list[dict[str, Any]],
    target_tab: str,
    action_label: str,
    ready_message: str,
    empty_message: str | None = None,
) -> dict[str, Any]:
    status = _coach_status(verdict=verdict, issues=issues)
    blocker_count = sum(1 for item in issues if str(item.get("severity") or "") == "blocker")
    warning_count = sum(1 for item in issues if str(item.get("severity") or "") == "warning")
    info_count = sum(1 for item in issues if str(item.get("severity") or "") == "info")
    if issues:
        first_issue = sorted(
            issues,
            key=lambda item: _coach_severity_rank(str(item.get("severity") or "info")),
        )[0]
        message = str(first_issue.get("title") or ready_message)
    elif status == "empty" and empty_message:
        message = empty_message
    else:
        message = ready_message
    return {
        "id": section_id,
        "label": label,
        "status": status,
        "verdict": str(verdict or ""),
        "target_tab": target_tab,
        "action_label": action_label,
        "message": message,
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "info_count": info_count,
    }


def _coach_next_action_from_entry(
    *,
    action_id: str,
    title: str,
    message: str,
    action_label: str,
    target_tab: str,
    priority: str = "low",
    section_id: str = "overview",
    section_label: str = "Data Studio",
    requires_user_confirmation: bool = False,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "section_id": section_id,
        "section_label": section_label,
        "severity": "info",
        "priority": priority,
        "title": title,
        "message": message,
        "action_label": action_label,
        "target_tab": target_tab,
        "requires_user_confirmation": requires_user_confirmation,
    }


def _coach_public_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in action.items()
        if key != "sort"
    }


async def build_data_studio_coach_rail(
    db: AsyncSession,
    project_id: int,
) -> dict[str, Any]:
    """Return a read-only cross-section coach rail for Data Studio."""

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    overview = await build_data_studio_overview(db, project_id)
    sources = await build_data_studio_sources(db, project_id)
    mapping = await build_data_studio_mapping_preview(db, project_id)
    domain = await build_data_studio_domain_detection(db, project_id)
    quality_safety = await build_data_studio_quality_safety(db, project_id)
    gold = await build_data_studio_gold_set_workbench(db, project_id)
    synthetic_playbooks = await build_data_studio_synthetic_playbook_center(db, project_id)
    synthetic_recommendations = await build_data_studio_synthetic_recommendations(db, project_id)
    synthetic_quality = await build_data_studio_synthetic_quality_analytics(db, project_id)
    review_queue = await build_data_studio_review_queue(db, project_id)
    prepare_dataset = await build_data_studio_prepare_dataset(db, project_id)
    dataset_versions = await build_data_studio_dataset_versions(db, project_id)

    section_specs = [
        {
            "id": "overview",
            "label": "Overview",
            "payload": overview,
            "target_tab": "data",
            "action_label": "Open Data",
            "ready_message": "Project readiness looks clear.",
            "empty_message": None,
        },
        {
            "id": "sources",
            "label": "Sources",
            "payload": sources,
            "target_tab": "data",
            "action_label": "Open Sources",
            "ready_message": "Sources are connected and readable.",
            "empty_message": "No source data has been added yet.",
        },
        {
            "id": "mapping",
            "label": "Mapping",
            "payload": mapping,
            "target_tab": "dataprep",
            "action_label": "Review Mapping",
            "ready_message": "Schema mapping is aligned with the recipe.",
            "empty_message": "Mapping needs previewable source rows.",
        },
        {
            "id": "domain",
            "label": "Domain",
            "payload": domain,
            "target_tab": "domain",
            "action_label": "Review Domain",
            "ready_message": "Domain signals are confirmed or low risk.",
            "empty_message": "Domain evidence is still limited.",
        },
        {
            "id": "quality_safety",
            "label": "Quality & Safety",
            "payload": quality_safety,
            "target_tab": "dataprep",
            "action_label": "Review Quality",
            "ready_message": "Quality and safety scans are clear.",
            "empty_message": None,
        },
        {
            "id": "gold_set",
            "label": "Gold Set",
            "payload": gold,
            "target_tab": "goldset",
            "action_label": "Open Gold Set",
            "ready_message": "Trusted Gold Set examples are ready.",
            "empty_message": "Gold Set examples are not ready yet.",
        },
        {
            "id": "synthetic_playbooks",
            "label": "Synthetic Playbooks",
            "payload": synthetic_playbooks,
            "target_tab": "synthetic",
            "action_label": "Open Synthetic",
            "ready_message": "Synthetic playbook prerequisites are ready.",
            "empty_message": "Synthetic playbooks need recipe context.",
        },
        {
            "id": "synthetic_recommendations",
            "label": "Synthetic Recommendations",
            "payload": synthetic_recommendations,
            "target_tab": "synthetic",
            "action_label": "Open Recommendations",
            "ready_message": "Synthetic recommendations are available.",
            "empty_message": "Synthetic recommendations need more setup.",
        },
        {
            "id": "synthetic_quality",
            "label": "Synthetic Quality",
            "payload": synthetic_quality,
            "target_tab": "synthetic",
            "action_label": "Review Synthetic Quality",
            "ready_message": "Synthetic quality analytics look clear.",
            "empty_message": "Synthetic quality analytics need generated rows.",
        },
        {
            "id": "review_queue",
            "label": "Review Queue",
            "payload": review_queue,
            "target_tab": "synthetic",
            "action_label": "Open Review",
            "ready_message": "Review gates are clear.",
            "empty_message": "No review queue is active.",
        },
        {
            "id": "prepare_dataset",
            "label": "Prepare Dataset",
            "payload": prepare_dataset,
            "target_tab": "dataprep",
            "action_label": "Open Dataset Prep",
            "ready_message": "Dataset preparation checks are aligned.",
            "empty_message": None,
        },
        {
            "id": "dataset_versions",
            "label": "Dataset Versions",
            "payload": dataset_versions,
            "target_tab": "dataprep",
            "action_label": "Open Versions",
            "ready_message": "Prepared versions are reusable.",
            "empty_message": "Prepared dataset versions are not available yet.",
        },
    ]

    coach_issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for section in section_specs:
        payload = section["payload"]
        issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
        section_id = str(section["id"])
        section_label = str(section["label"])
        for index, issue in enumerate(issues):
            if isinstance(issue, dict):
                coach_issues.append(
                    _coach_issue(
                        section_id=section_id,
                        section_label=section_label,
                        issue=issue,
                        index=index,
                    )
                )
        checks.append(
            _coach_section(
                section_id=section_id,
                label=section_label,
                verdict=payload.get("verdict"),
                issues=[item for item in issues if isinstance(item, dict)],
                target_tab=str(section["target_tab"]),
                action_label=str(section["action_label"]),
                ready_message=str(section["ready_message"]),
                empty_message=section.get("empty_message"),
            )
        )

    coach_issues.sort(key=lambda item: tuple(item.get("sort") or [99, 99, 99]))
    blocker_count = sum(1 for item in coach_issues if item["severity"] == "blocker")
    warning_count = sum(1 for item in coach_issues if item["severity"] == "warning")
    info_count = sum(1 for item in coach_issues if item["severity"] == "info")
    empty_section_count = sum(1 for item in checks if item["status"] == "empty")
    ready_section_count = sum(1 for item in checks if item["status"] == "ready")

    if blocker_count:
        verdict: CoachVerdict = "blocked"
    elif warning_count or empty_section_count:
        verdict = "attention"
    else:
        verdict = "ready"

    actionable_issues = [
        issue
        for issue in coach_issues
        if issue["severity"] in {"blocker", "warning"}
    ]
    if actionable_issues:
        next_action = _coach_public_action(actionable_issues[0])
    else:
        version_reuse = dataset_versions.get("reuse_readiness")
        training_reuse = (
            version_reuse.get("training")
            if isinstance(version_reuse, dict) and isinstance(version_reuse.get("training"), dict)
            else {}
        )
        if str(training_reuse.get("status") or "") == "ready":
            next_action = _coach_next_action_from_entry(
                action_id="coach_open_training",
                title="Launch training from the prepared dataset",
                message=str(training_reuse.get("message") or "Prepared versions are ready for training."),
                action_label="Open Training",
                target_tab="training",
                priority="medium",
                section_id="dataset_versions",
                section_label="Dataset Versions",
                requires_user_confirmation=True,
            )
        else:
            entry = prepare_dataset.get("entry_point") if isinstance(prepare_dataset.get("entry_point"), dict) else {}
            next_action = _coach_next_action_from_entry(
                action_id="coach_open_dataset_prep",
                title="Prepare a dataset version",
                message=str(entry.get("reason") or "Create or refresh prepared dataset versions."),
                action_label=str(entry.get("label") or "Open Dataset Prep"),
                target_tab=str(entry.get("target_tab") or "dataprep"),
                priority="medium",
                section_id="prepare_dataset",
                section_label="Prepare Dataset",
                requires_user_confirmation=True,
            )

    next_steps = [_coach_public_action(item) for item in actionable_issues[:5]]
    if not next_steps:
        next_steps = [next_action]
    elif next_action["id"] not in {item["id"] for item in next_steps}:
        next_steps.insert(0, next_action)
        next_steps = next_steps[:5]

    return {
        "project_id": project_id,
        "verdict": verdict,
        "read_only": True,
        "auto_apply": False,
        "source_of_truth": "deterministic_data_studio_checks",
        "summary": {
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "section_count": len(checks),
            "ready_section_count": ready_section_count,
            "empty_section_count": empty_section_count,
            "next_action_target": next_action.get("target_tab"),
        },
        "next_action": next_action,
        "next_steps": next_steps,
        "checks": checks,
        "issues": [_coach_public_action(item) for item in coach_issues[:30]],
        "entry_points": [
            {
                "label": "Open Data Prep",
                "target_tab": "dataprep",
                "reason": "Review mapping, split, manifest, and version checks.",
                "requires_confirmation": True,
            },
            {
                "label": "Open Synthetic",
                "target_tab": "synthetic",
                "reason": "Generate or review synthetic rows.",
                "requires_confirmation": True,
            },
            {
                "label": "Open Training",
                "target_tab": "training",
                "reason": "Use prepared dataset versions for training.",
                "requires_confirmation": True,
            },
        ],
        "power_details": {
            "section_verdicts": {
                str(section["id"]): str(section["payload"].get("verdict") or "")
                for section in section_specs
            },
            "overview_primary_action": overview.get("primary_action"),
            "prepare_can_prepare": prepare_dataset.get("can_prepare"),
            "training_reuse_ready": (dataset_versions.get("summary") or {}).get("training_reuse_ready")
            if isinstance(dataset_versions.get("summary"), dict)
            else None,
        },
    }
