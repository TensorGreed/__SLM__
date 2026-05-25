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
ReviewQueueVerdict = Literal["empty", "attention", "ready"]
PrepareDatasetVerdict = Literal["blocked", "attention", "ready"]
DatasetVersionVerdict = Literal["empty", "attention", "ready"]
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
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "kind": kind,
        "status": status,
        "count": int(count),
        "target_tab": target_tab,
    }


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
