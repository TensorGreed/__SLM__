"""Domain blueprint analysis, validation, versioning, and project-apply service."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.domain_blueprint import DomainBlueprintRevision, DomainBlueprintStatus
from app.models.project import Project
from app.schemas.domain_blueprint import (
    DomainBlueprintAnalyzeRequest,
    DomainBlueprintAnalyzeResponse,
    DomainBlueprintContract,
    DomainBlueprintDiffItem,
    DomainBlueprintDiffResponse,
    DomainBlueprintGlossaryHelpResponse,
    DomainBlueprintGuidance,
    DomainBlueprintRevisionResponse,
    DomainBlueprintValidationIssue,
    DomainBlueprintValidationResult,
    GlossaryEntry,
    SuccessMetric,
)
from app.services.artifact_registry_service import publish_artifact
from app.services.synthetic_service import call_teacher_model


TASK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "structured_extraction": (
        "extract",
        "extraction",
        "field",
        "json",
        "invoice",
        "contract",
        "entity",
        "schema",
    ),
    "classification": (
        "classify",
        "classification",
        "label",
        "category",
        "sentiment",
        "route",
        "triage",
    ),
    "summarization": (
        "summarize",
        "summary",
        "brief",
        "digest",
        "recap",
        "tl;dr",
    ),
    "qa": (
        "q&a",
        "question",
        "answer",
        "faq",
        "helpdesk",
        "assistant",
        "support",
    ),
    "rag_qa": (
        "grounded",
        "retrieval",
        "rag",
        "context window",
        "source citation",
        "citation",
    ),
    "tool_calling": (
        "tool",
        "function call",
        "function-calling",
        "api call",
        "action schema",
    ),
    "instruction_sft": ("assistant", "instruction", "chat", "general"),
}

TASK_DEFAULT_METRICS: dict[str, list[SuccessMetric]] = {
    "structured_extraction": [
        SuccessMetric(
            metric_id="exact_match",
            label="Field Exact Match",
            target=">= 0.85",
            why_it_matters="Verifies extraction fidelity per required field.",
        ),
        SuccessMetric(
            metric_id="json_valid_rate",
            label="JSON Validity Rate",
            target=">= 0.98",
            why_it_matters="Ensures responses are machine-consumable.",
        ),
    ],
    "classification": [
        SuccessMetric(
            metric_id="macro_f1",
            label="Macro F1",
            target=">= 0.80",
            why_it_matters="Balances precision/recall across classes.",
        ),
        SuccessMetric(
            metric_id="accuracy",
            label="Accuracy",
            target=">= 0.85",
            why_it_matters="Simple global quality indicator for label correctness.",
        ),
    ],
    "summarization": [
        SuccessMetric(
            metric_id="faithfulness",
            label="Faithfulness",
            target=">= 0.85",
            why_it_matters="Prevents hallucinated summary content.",
        ),
        SuccessMetric(
            metric_id="brevity_score",
            label="Brevity Score",
            target=">= 0.75",
            why_it_matters="Keeps summaries concise and useful.",
        ),
    ],
    "qa": [
        SuccessMetric(
            metric_id="answer_correctness",
            label="Answer Correctness",
            target=">= 0.82",
            why_it_matters="Tracks direct answer accuracy on gold examples.",
        ),
        SuccessMetric(
            metric_id="hallucination_rate",
            label="Hallucination Rate",
            target="<= 0.10",
            why_it_matters="Controls unsafe fabricated responses.",
        ),
    ],
    "rag_qa": [
        SuccessMetric(
            metric_id="citation_precision",
            label="Citation Precision",
            target=">= 0.80",
            why_it_matters="Measures whether retrieved evidence supports answers.",
        ),
        SuccessMetric(
            metric_id="grounded_answer_rate",
            label="Grounded Answer Rate",
            target=">= 0.85",
            why_it_matters="Reduces unsupported claims.",
        ),
    ],
    "tool_calling": [
        SuccessMetric(
            metric_id="tool_call_valid_rate",
            label="Tool Call Valid Rate",
            target=">= 0.95",
            why_it_matters="Ensures generated calls match schema and arguments.",
        ),
        SuccessMetric(
            metric_id="tool_selection_accuracy",
            label="Tool Selection Accuracy",
            target=">= 0.85",
            why_it_matters="Measures whether the right tool is chosen per intent.",
        ),
    ],
    "instruction_sft": [
        SuccessMetric(
            metric_id="helpfulness_score",
            label="Helpfulness",
            target=">= 0.80",
            why_it_matters="Captures practical utility for end users.",
        ),
        SuccessMetric(
            metric_id="safety_violation_rate",
            label="Safety Violation Rate",
            target="<= 0.05",
            why_it_matters="Prevents unsafe or policy-violating outputs.",
        ),
    ],
}

BUILTIN_GLOSSARY: dict[str, tuple[str, str]] = {
    "task family": (
        "The category of behavior your model is trained for, like extraction or classification.",
        "general",
    ),
    "input modality": (
        "The kind of input the model expects, such as text, json, or multimodal content.",
        "general",
    ),
    "output schema": (
        "A structured contract that defines how model outputs should be formatted.",
        "general",
    ),
    "hallucination": (
        "A response that sounds plausible but is not supported by source data.",
        "safety",
    ),
    "grounded answer": (
        "An answer that is directly supported by provided evidence.",
        "evaluation",
    ),
    "gold set": (
        "A trusted reference dataset used for evaluation and regression checks.",
        "evaluation",
    ),
    "latency": (
        "How long the model takes to respond after receiving input.",
        "deployment",
    ),
    "throughput": (
        "How many requests can be served per unit time.",
        "deployment",
    ),
    "lora": (
        "A parameter-efficient training method that updates small adapter layers.",
        "training",
    ),
    "qlora": (
        "A LoRA approach that uses quantized base weights to reduce memory usage.",
        "training",
    ),
    "preflight": (
        "A validation step that checks compatibility before training or deployment.",
        "operations",
    ),
    "adapter": (
        "A mapping layer that converts source data into training-ready fields.",
        "data",
    ),
    "domain pack": (
        "A bundle of defaults and policy overlays for a domain use case.",
        "domain",
    ),
    "domain profile": (
        "A typed configuration profile for runtime and evaluation behavior.",
        "domain",
    ),
}

DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "legal": ("legal", "contract", "compliance", "law", "regulation"),
    "healthcare": ("healthcare", "patient", "clinical", "hipaa", "medical"),
    "finance": ("finance", "bank", "trading", "invoice", "ledger", "risk"),
    "support": ("support", "ticket", "faq", "customer", "helpdesk"),
}

PERSONA_HINTS: dict[str, str] = {
    "support": "Support agents and end customers",
    "healthcare": "Clinical and operations teams",
    "legal": "Legal analysts and contract reviewers",
    "finance": "Finance analysts and operations stakeholders",
}

TARGET_PROFILE_HINTS: dict[str, str] = {
    "mobile": "mobile_cpu",
    "phone": "mobile_cpu",
    "browser": "browser_webgpu",
    "edge": "edge_gpu",
    "laptop": "edge_gpu",
    "server": "vllm_server",
    "cloud": "vllm_server",
}


@dataclass
class DomainBlueprintValidationError(Exception):
    validation: DomainBlueprintValidationResult

    def __str__(self) -> str:
        if self.validation.errors:
            return self.validation.errors[0].message
        return "Domain blueprint validation failed."


def _project_blueprints_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "blueprints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_token(value: Any) -> str:
    return _normalize_text(value).lower().replace("-", "_").replace(" ", "_")


def _infer_task_family(brief: str, task_hint: str | None = None) -> tuple[str, list[str]]:
    hint = _normalized_token(task_hint)
    if hint and hint != "none":
        return hint, [f"task_family_hint={hint}"]

    token = brief.lower()
    best_key = "instruction_sft"
    best_score = -1
    reasons: list[str] = []
    for key, keywords in TASK_KEYWORDS.items():
        score = 0
        matched: list[str] = []
        for keyword in keywords:
            if keyword in token:
                score += 1
                matched.append(keyword)
        if score > best_score:
            best_key = key
            best_score = score
            reasons = matched

    if best_score <= 0 and ("extract" in token and "json" in token):
        return "structured_extraction", ["extract+json heuristic"]
    return best_key, [f"matched:{item}" for item in reasons] if reasons else ["default:instruction_sft"]


def _infer_domain_name(brief: str, explicit_domain_name: str | None = None) -> str:
    explicit = _normalize_text(explicit_domain_name)
    if explicit:
        return explicit
    lower = brief.lower()
    for domain, keywords in DOMAIN_HINTS.items():
        if any(keyword in lower for keyword in keywords):
            return domain.title()
    match = re.search(r"\bfor\s+([a-z0-9][a-z0-9 /\-]{2,48})", lower)
    if match:
        candidate = match.group(1).strip(" .,:;")
        if candidate:
            return candidate.title()
    return "General Domain"


def _infer_persona(brief: str, explicit_persona: str | None = None) -> str:
    explicit = _normalize_text(explicit_persona)
    if explicit:
        return explicit
    lower = brief.lower()
    for domain, persona in PERSONA_HINTS.items():
        if domain in lower:
            return persona
    if "customer" in lower:
        return "Customer-facing users"
    return "Domain practitioners"


def _infer_input_modality(
    brief: str,
    sample_inputs: list[str],
    hint: str | None = None,
) -> str:
    hint_token = _normalized_token(hint)
    if hint_token and hint_token != "none":
        return hint_token

    lower = brief.lower()
    if any(keyword in lower for keyword in ("image", "vision", "screenshot")):
        return "image"
    if any(keyword in lower for keyword in ("audio", "transcript", "speech")):
        return "audio_text"

    sample_blob = "\n".join(sample_inputs).strip()
    if sample_blob.startswith("{") or sample_blob.startswith("["):
        return "json"
    if "\t" in sample_blob or "," in sample_blob:
        return "tabular_text"
    return "text"


def _safe_json_parse(value: str) -> Any:
    token = _normalize_text(value)
    if not token:
        return None
    try:
        return json.loads(token)
    except json.JSONDecodeError:
        return None


def _parse_output_examples(sample_outputs: list[str]) -> list[Any]:
    examples: list[Any] = []
    for raw in sample_outputs[:5]:
        parsed = _safe_json_parse(raw)
        examples.append(parsed if parsed is not None else raw.strip())
    return [row for row in examples if row not in (None, "")]


def _infer_output_schema(
    task_family: str,
    explicit_schema: dict[str, Any] | None,
    output_examples: list[Any],
) -> dict[str, Any]:
    if isinstance(explicit_schema, dict) and explicit_schema:
        return dict(explicit_schema)

    if output_examples and isinstance(output_examples[0], dict):
        first = dict(output_examples[0])
        inferred = {str(key): type(value).__name__ for key, value in first.items()}
        return {
            "type": "object",
            "properties": inferred,
            "required": list(inferred.keys()),
        }

    if task_family == "structured_extraction":
        return {
            "type": "object",
            "properties": {
                "fields": "object",
                "confidence": "number",
            },
            "required": ["fields"],
        }
    if task_family == "classification":
        return {
            "type": "object",
            "properties": {
                "label": "string",
                "confidence": "number",
            },
            "required": ["label"],
        }
    if task_family in {"qa", "rag_qa", "instruction_sft", "summarization"}:
        return {
            "type": "object",
            "properties": {"answer": "string"},
            "required": ["answer"],
        }
    return {}


def _infer_deployment_constraints(brief: str, deployment_target: str | None) -> dict[str, Any]:
    target = _normalize_text(deployment_target).lower()
    if not target:
        lowered = brief.lower()
        for key, profile_id in TARGET_PROFILE_HINTS.items():
            if key in lowered:
                target = key
                break

    target_profile_id = "vllm_server"
    if target:
        for key, profile_id in TARGET_PROFILE_HINTS.items():
            if key in target:
                target_profile_id = profile_id
                break

    lower = brief.lower()
    offline_required = any(keyword in lower for keyword in ("offline", "air-gapped", "air gapped"))
    low_latency = any(keyword in lower for keyword in ("realtime", "real-time", "low latency", "instant"))
    return {
        "target_profile_id": target_profile_id,
        "offline_required": offline_required,
        "requires_cloud_inference": target_profile_id == "vllm_server" and not offline_required,
        "latency_priority": "high" if low_latency else "normal",
    }


def _infer_safety_notes(brief: str, explicit_notes: list[str], risk_constraints: list[str]) -> list[str]:
    notes: list[str] = []
    for raw in [*explicit_notes, *risk_constraints]:
        token = _normalize_text(raw)
        if token:
            notes.append(token)

    lower = brief.lower()
    if "hipaa" in lower or "patient" in lower:
        notes.append("Handle PHI carefully and enforce HIPAA-safe data handling.")
    if "contract" in lower or "legal" in lower:
        notes.append("Avoid legal advice language without clear disclaimer policy.")
    if "finance" in lower or "bank" in lower:
        notes.append("Add controls for financial misinformation and compliance checks.")
    if "pii" in lower:
        notes.append("Apply PII redaction/guardrails before training and inference.")
    return _dedupe_preserve_order(notes)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in values:
        token = _normalize_text(raw)
        if not token:
            continue
        marker = token.lower()
        if marker in seen:
            continue
        seen.add(marker)
        cleaned.append(token)
    return cleaned


def _infer_success_metrics(task_family: str, explicit_metrics: list[str]) -> list[SuccessMetric]:
    metrics = list(TASK_DEFAULT_METRICS.get(task_family) or TASK_DEFAULT_METRICS["instruction_sft"])
    for raw in explicit_metrics:
        token = _normalized_token(raw)
        if not token:
            continue
        metrics.append(
            SuccessMetric(
                metric_id=token,
                label=_normalize_text(raw),
                target="",
                why_it_matters="User-requested success metric.",
            )
        )

    deduped: list[SuccessMetric] = []
    seen: set[str] = set()
    for metric in metrics:
        marker = metric.metric_id.lower()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(metric)
    return deduped[:8]


def _infer_glossary(brief: str, task_family: str) -> list[GlossaryEntry]:
    token = brief.lower()
    rows: list[GlossaryEntry] = []
    for term, (plain_language, category) in BUILTIN_GLOSSARY.items():
        if term in token or category in token:
            rows.append(
                GlossaryEntry(
                    term=term,
                    plain_language=plain_language,
                    category=category,
                )
            )
    rows.extend(
        [
            GlossaryEntry(
                term="task_family",
                plain_language=f"Selected task family is `{task_family}` based on your brief.",
                category="analysis",
            ),
            GlossaryEntry(
                term="confidence_score",
                plain_language="Confidence indicates how complete and unambiguous the brief appears.",
                category="analysis",
            ),
        ]
    )

    deduped: list[GlossaryEntry] = []
    seen: set[str] = set()
    for row in rows:
        marker = row.term.lower()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(row)
    return deduped[:12]


def _infer_unresolved_assumptions(
    *,
    sample_inputs: list[str],
    sample_outputs: list[str],
    deployment_target: str | None,
    safety_notes: list[str],
) -> list[str]:
    assumptions: list[str] = []
    if not sample_inputs:
        assumptions.append("No sample inputs were provided; mapping quality is still uncertain.")
    if not sample_outputs:
        assumptions.append("No sample outputs were provided; output contract may need refinement.")
    if not _normalize_text(deployment_target):
        assumptions.append("Deployment target was inferred; confirm runtime target constraints.")
    if not safety_notes:
        assumptions.append("Safety/compliance constraints are minimal; add policy-specific guardrails.")
    return assumptions


def _confidence_score(
    *,
    sample_inputs: list[str],
    sample_outputs: list[str],
    unresolved_assumptions: list[str],
    explicit_schema: dict[str, Any],
) -> float:
    score = 0.35
    if sample_inputs:
        score += 0.2
    if sample_outputs:
        score += 0.2
    if explicit_schema:
        score += 0.15
    score -= min(0.25, len(unresolved_assumptions) * 0.05)
    return max(0.1, min(0.98, round(score, 3)))


def validate_domain_blueprint(blueprint: DomainBlueprintContract) -> DomainBlueprintValidationResult:
    errors: list[DomainBlueprintValidationIssue] = []
    warnings: list[DomainBlueprintValidationIssue] = []

    if not _normalize_text(blueprint.domain_name):
        errors.append(
            DomainBlueprintValidationIssue(
                code="DOMAIN_NAME_REQUIRED",
                field="domain_name",
                message="Domain name is required.",
                actionable_fix="Provide a short domain name, such as 'Legal' or 'Customer Support'.",
            )
        )
    if not _normalize_text(blueprint.problem_statement):
        errors.append(
            DomainBlueprintValidationIssue(
                code="PROBLEM_STATEMENT_REQUIRED",
                field="problem_statement",
                message="Problem statement is required.",
                actionable_fix="Describe the business problem in one or two sentences.",
            )
        )
    if not _normalize_text(blueprint.target_user_persona):
        errors.append(
            DomainBlueprintValidationIssue(
                code="TARGET_PERSONA_REQUIRED",
                field="target_user_persona",
                message="Target user persona is required.",
                actionable_fix="Specify who will use the model output.",
            )
        )

    if blueprint.task_family in {"structured_extraction", "classification"} and not blueprint.expected_output_schema:
        errors.append(
            DomainBlueprintValidationIssue(
                code="OUTPUT_SCHEMA_REQUIRED",
                field="expected_output_schema",
                message=f"`{blueprint.task_family}` requires an explicit output schema.",
                actionable_fix="Provide a JSON object schema with required fields.",
            )
        )

    constraints = dict(blueprint.deployment_target_constraints or {})
    if bool(constraints.get("offline_required")) and bool(constraints.get("requires_cloud_inference")):
        errors.append(
            DomainBlueprintValidationIssue(
                code="DEPLOYMENT_CONSTRAINT_CONFLICT",
                field="deployment_target_constraints",
                message="Blueprint requires both offline execution and cloud inference.",
                actionable_fix="Set either offline-only or cloud-required, not both.",
            )
        )

    if float(blueprint.confidence_score) < 0.45:
        warnings.append(
            DomainBlueprintValidationIssue(
                code="LOW_CONFIDENCE_BLUEPRINT",
                field="confidence_score",
                message="Blueprint confidence is low due to missing context.",
                actionable_fix="Add sample inputs/outputs and explicit deployment constraints.",
            )
        )

    if blueprint.unresolved_assumptions:
        warnings.append(
            DomainBlueprintValidationIssue(
                code="UNRESOLVED_ASSUMPTIONS",
                field="unresolved_assumptions",
                message="Blueprint still has unresolved assumptions.",
                actionable_fix="Review assumptions and resolve them before training.",
            )
        )

    return DomainBlueprintValidationResult(ok=not errors, errors=errors, warnings=warnings)


def _build_guidance(
    blueprint: DomainBlueprintContract,
    validation: DomainBlueprintValidationResult,
    task_reasons: list[str],
) -> DomainBlueprintGuidance:
    unresolved_questions = list(blueprint.unresolved_assumptions or [])
    recommended_next_actions = [
        "Review inferred task family and output schema before saving.",
        "Attach 5-20 representative sample rows in Dataset Prep Adapter Lab.",
        "Run training preflight after applying blueprint defaults.",
    ]
    inferred_fields = [
        {"field": "task_family", "reason": ", ".join(task_reasons)},
        {"field": "input_modality", "reason": f"inferred from brief/examples ({blueprint.input_modality})"},
        {
            "field": "deployment_target_constraints.target_profile_id",
            "reason": str(blueprint.deployment_target_constraints.get("target_profile_id", "vllm_server")),
        },
    ]
    return DomainBlueprintGuidance(
        unresolved_questions=unresolved_questions,
        recommended_next_actions=_dedupe_preserve_order(recommended_next_actions),
        warnings=[issue.message for issue in validation.warnings],
        assumptions=list(blueprint.unresolved_assumptions or []),
        inferred_fields=inferred_fields,
    )


async def _llm_enrich_blueprint(
    req: DomainBlueprintAnalyzeRequest,
    blueprint: DomainBlueprintContract,
) -> tuple[DomainBlueprintContract, dict[str, Any]]:
    if not req.llm_enrich:
        return blueprint, {"enabled": False, "applied": False, "reason": "llm_enrich=false"}

    if not bool(getattr(settings, "DOMAIN_BLUEPRINT_ENABLE_LLM_ENRICHMENT", False)):
        return blueprint, {"enabled": False, "applied": False, "reason": "setting_disabled"}
    if not settings.TEACHER_MODEL_API_URL:
        return blueprint, {"enabled": False, "applied": False, "reason": "teacher_model_not_configured"}

    prompt = (
        "You are assisting with domain blueprint enrichment. "
        "Return strict JSON with keys: extra_glossary (list), extra_metrics (list), assumptions (list). "
        "Do not include markdown.\n\n"
        f"Brief:\n{req.brief_text}\n\n"
        f"Current blueprint:\n{json.dumps(blueprint.model_dump(mode='json'), ensure_ascii=True)}"
    )

    try:
        response = await call_teacher_model(prompt=prompt, max_tokens=800, temperature=0.1)
    except Exception as e:
        return blueprint, {"enabled": True, "applied": False, "reason": f"llm_call_failed:{e}"}

    content = str(response.get("content") or "").strip()
    if not content:
        return blueprint, {"enabled": True, "applied": False, "reason": "empty_response"}
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "", 1).strip()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return blueprint, {"enabled": True, "applied": False, "reason": "invalid_json_response"}

    if not isinstance(payload, dict):
        return blueprint, {"enabled": True, "applied": False, "reason": "invalid_payload_type"}

    glossary = list(blueprint.glossary)
    for raw in list(payload.get("extra_glossary") or []):
        if not isinstance(raw, dict):
            continue
        term = _normalize_text(raw.get("term"))
        plain = _normalize_text(raw.get("plain_language"))
        if not term or not plain:
            continue
        glossary.append(
            GlossaryEntry(
                term=term,
                plain_language=plain,
                category=_normalize_text(raw.get("category")) or "analysis",
                example=_normalize_text(raw.get("example")) or None,
            )
        )
    deduped_glossary: list[GlossaryEntry] = []
    seen_terms: set[str] = set()
    for row in glossary:
        key = row.term.lower()
        if key in seen_terms:
            continue
        seen_terms.add(key)
        deduped_glossary.append(row)

    metrics = list(blueprint.success_metrics)
    for raw in list(payload.get("extra_metrics") or []):
        if not isinstance(raw, dict):
            continue
        metric_id = _normalized_token(raw.get("metric_id") or raw.get("id"))
        label = _normalize_text(raw.get("label"))
        if not metric_id or not label:
            continue
        metrics.append(
            SuccessMetric(
                metric_id=metric_id,
                label=label,
                target=_normalize_text(raw.get("target")),
                why_it_matters=_normalize_text(raw.get("why_it_matters") or raw.get("reason")),
            )
        )
    deduped_metrics: list[SuccessMetric] = []
    seen_metrics: set[str] = set()
    for row in metrics:
        key = row.metric_id.lower()
        if key in seen_metrics:
            continue
        seen_metrics.add(key)
        deduped_metrics.append(row)

    assumptions = _dedupe_preserve_order(
        list(blueprint.unresolved_assumptions)
        + [str(item) for item in list(payload.get("assumptions") or [])]
    )

    enriched = blueprint.model_copy(
        update={
            "glossary": deduped_glossary[:16],
            "success_metrics": deduped_metrics[:10],
            "unresolved_assumptions": assumptions[:10],
        }
    )
    return enriched, {"enabled": True, "applied": True, "reason": "enriched", "model": response.get("model")}


async def analyze_domain_brief(
    req: DomainBlueprintAnalyzeRequest,
    *,
    project_id: int | None = None,
) -> DomainBlueprintAnalyzeResponse:
    brief = _normalize_text(req.brief_text)
    task_family, task_reasons = _infer_task_family(brief, req.task_family_hint)
    output_examples = _parse_output_examples(req.sample_outputs)
    output_schema = _infer_output_schema(task_family, req.expected_output_schema, output_examples)
    safety_notes = _infer_safety_notes(brief, req.safety_compliance_notes, req.risk_constraints)
    unresolved_assumptions = _infer_unresolved_assumptions(
        sample_inputs=req.sample_inputs,
        sample_outputs=req.sample_outputs,
        deployment_target=req.deployment_target,
        safety_notes=safety_notes,
    )
    confidence = _confidence_score(
        sample_inputs=req.sample_inputs,
        sample_outputs=req.sample_outputs,
        unresolved_assumptions=unresolved_assumptions,
        explicit_schema=req.expected_output_schema,
    )

    blueprint = DomainBlueprintContract(
        domain_name=_infer_domain_name(brief, req.domain_name),
        problem_statement=_normalize_text(req.problem_statement) or brief,
        target_user_persona=_infer_persona(brief, req.target_user_persona),
        task_family=task_family,
        input_modality=_infer_input_modality(brief, req.sample_inputs, req.input_modality_hint),
        expected_output_schema=output_schema,
        expected_output_examples=output_examples,
        safety_compliance_notes=safety_notes,
        deployment_target_constraints=_infer_deployment_constraints(brief, req.deployment_target),
        success_metrics=_infer_success_metrics(task_family, req.success_metrics),
        glossary=_infer_glossary(brief, task_family),
        confidence_score=confidence,
        unresolved_assumptions=unresolved_assumptions,
    )

    enriched_blueprint, llm_meta = await _llm_enrich_blueprint(req, blueprint)
    validation = validate_domain_blueprint(enriched_blueprint)
    guidance = _build_guidance(enriched_blueprint, validation, task_reasons)

    return DomainBlueprintAnalyzeResponse(
        project_id=project_id,
        blueprint=enriched_blueprint,
        validation=validation,
        guidance=guidance,
        llm_enrichment=llm_meta,
    )


async def _next_blueprint_version(db: AsyncSession, project_id: int) -> int:
    result = await db.execute(
        select(DomainBlueprintRevision.version)
        .where(DomainBlueprintRevision.project_id == project_id)
        .order_by(DomainBlueprintRevision.version.desc(), DomainBlueprintRevision.id.desc())
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    return int(latest or 0) + 1


async def _get_project_or_none(db: AsyncSession, project_id: int) -> Project | None:
    result = await db.execute(select(Project).where(Project.id == project_id))
    return result.scalar_one_or_none()


def _blueprint_to_record_kwargs(blueprint: DomainBlueprintContract) -> dict[str, Any]:
    return {
        "domain_name": blueprint.domain_name,
        "problem_statement": blueprint.problem_statement,
        "target_user_persona": blueprint.target_user_persona,
        "task_family": blueprint.task_family,
        "input_modality": blueprint.input_modality,
        "expected_output_schema": blueprint.expected_output_schema,
        "expected_output_examples": blueprint.expected_output_examples,
        "safety_compliance_notes": blueprint.safety_compliance_notes,
        "deployment_target_constraints": blueprint.deployment_target_constraints,
        "success_metrics": [row.model_dump(mode="json") for row in blueprint.success_metrics],
        "glossary": [row.model_dump(mode="json") for row in blueprint.glossary],
        "confidence_score": float(blueprint.confidence_score),
        "unresolved_assumptions": list(blueprint.unresolved_assumptions),
    }


def _record_to_blueprint(record: DomainBlueprintRevision) -> DomainBlueprintContract:
    return DomainBlueprintContract(
        domain_name=str(record.domain_name or ""),
        problem_statement=str(record.problem_statement or ""),
        target_user_persona=str(record.target_user_persona or ""),
        task_family=str(record.task_family or "instruction_sft"),
        input_modality=str(record.input_modality or "text"),
        expected_output_schema=dict(record.expected_output_schema or {}),
        expected_output_examples=list(record.expected_output_examples or []),
        safety_compliance_notes=[str(item) for item in list(record.safety_compliance_notes or [])],
        deployment_target_constraints=dict(record.deployment_target_constraints or {}),
        success_metrics=[
            SuccessMetric.model_validate(item)
            for item in list(record.success_metrics or [])
            if isinstance(item, dict)
        ],
        glossary=[
            GlossaryEntry.model_validate(item)
            for item in list(record.glossary or [])
            if isinstance(item, dict)
        ],
        confidence_score=float(record.confidence_score or 0.0),
        unresolved_assumptions=[str(item) for item in list(record.unresolved_assumptions or [])],
    )


def serialize_domain_blueprint_revision(record: DomainBlueprintRevision) -> DomainBlueprintRevisionResponse:
    return DomainBlueprintRevisionResponse(
        id=record.id,
        project_id=record.project_id,
        version=record.version,
        status=record.status.value,
        source=str(record.source or "manual"),
        brief_text=str(record.brief_text or ""),
        blueprint=_record_to_blueprint(record),
        analysis_metadata=dict(record.analysis_metadata or {}),
        created_by_user_id=record.created_by_user_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


async def save_domain_blueprint_revision(
    db: AsyncSession,
    *,
    project_id: int,
    blueprint: DomainBlueprintContract,
    source: str = "manual",
    brief_text: str = "",
    analysis_metadata: dict[str, Any] | None = None,
    created_by_user_id: int | None = None,
    status: DomainBlueprintStatus = DomainBlueprintStatus.DRAFT,
) -> DomainBlueprintRevision:
    project = await _get_project_or_none(db, project_id)
    if project is None:
        raise ValueError("Project not found")

    validation = validate_domain_blueprint(blueprint)
    if not validation.ok:
        raise DomainBlueprintValidationError(validation)

    version = await _next_blueprint_version(db, project_id)
    record = DomainBlueprintRevision(
        project_id=project_id,
        version=version,
        status=status,
        source=_normalize_text(source) or "manual",
        brief_text=_normalize_text(brief_text),
        analysis_metadata=dict(analysis_metadata or {}),
        created_by_user_id=created_by_user_id,
        **_blueprint_to_record_kwargs(blueprint),
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)

    blueprint_path = _project_blueprints_dir(project_id) / f"domain_blueprint_v{record.version}.json"
    snapshot_payload = {
        "project_id": project_id,
        "version": record.version,
        "status": record.status.value,
        "source": record.source,
        "brief_text": record.brief_text,
        "blueprint": serialize_domain_blueprint_revision(record).blueprint.model_dump(mode="json"),
        "analysis_metadata": dict(record.analysis_metadata or {}),
    }
    blueprint_path.write_text(
        json.dumps(snapshot_payload, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )

    await publish_artifact(
        db=db,
        project_id=project_id,
        artifact_key="domain.blueprint",
        uri=str(blueprint_path),
        schema_ref="slm.domain_blueprint/v1",
        producer_stage="onboarding",
        metadata={
            "version": record.version,
            "status": record.status.value,
            "source": record.source,
        },
    )
    return record


async def list_domain_blueprint_revisions(db: AsyncSession, project_id: int) -> list[DomainBlueprintRevision]:
    result = await db.execute(
        select(DomainBlueprintRevision)
        .where(DomainBlueprintRevision.project_id == project_id)
        .order_by(DomainBlueprintRevision.version.desc(), DomainBlueprintRevision.id.desc())
    )
    return list(result.scalars().all())


async def get_domain_blueprint_revision(
    db: AsyncSession,
    *,
    project_id: int,
    version: int,
) -> DomainBlueprintRevision | None:
    result = await db.execute(
        select(DomainBlueprintRevision).where(
            DomainBlueprintRevision.project_id == project_id,
            DomainBlueprintRevision.version == version,
        )
    )
    return result.scalar_one_or_none()


async def get_latest_domain_blueprint_revision(
    db: AsyncSession,
    *,
    project_id: int,
) -> DomainBlueprintRevision | None:
    result = await db.execute(
        select(DomainBlueprintRevision)
        .where(DomainBlueprintRevision.project_id == project_id)
        .order_by(DomainBlueprintRevision.version.desc(), DomainBlueprintRevision.id.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def apply_domain_blueprint_revision(
    db: AsyncSession,
    *,
    project_id: int,
    version: int,
    adopt_project_description: bool = True,
    adopt_target_profile: bool = True,
    set_beginner_mode: bool = True,
) -> tuple[Project, DomainBlueprintRevision]:
    project = await _get_project_or_none(db, project_id)
    if project is None:
        raise ValueError("Project not found")

    revision = await get_domain_blueprint_revision(db, project_id=project_id, version=version)
    if revision is None:
        raise ValueError(f"Domain blueprint version {version} not found")

    active_rows = await list_domain_blueprint_revisions(db, project_id)
    for row in active_rows:
        if row.status == DomainBlueprintStatus.ACTIVE:
            row.status = DomainBlueprintStatus.ARCHIVED

    revision.status = DomainBlueprintStatus.ACTIVE
    project.active_domain_blueprint_version = revision.version
    if set_beginner_mode:
        project.beginner_mode = True
    if adopt_project_description and _normalize_text(revision.problem_statement):
        project.description = _normalize_text(revision.problem_statement)
    if adopt_target_profile:
        constraints = dict(revision.deployment_target_constraints or {})
        target_profile_id = _normalize_text(constraints.get("target_profile_id"))
        if target_profile_id:
            project.target_profile_id = target_profile_id

    await db.flush()
    await db.refresh(project)
    await db.refresh(revision)

    await publish_artifact(
        db=db,
        project_id=project_id,
        artifact_key="domain.blueprint.applied",
        schema_ref="slm.domain_blueprint/v1",
        producer_stage="onboarding",
        metadata={
            "version": revision.version,
            "adopt_project_description": bool(adopt_project_description),
            "adopt_target_profile": bool(adopt_target_profile),
            "set_beginner_mode": bool(set_beginner_mode),
        },
    )
    return project, revision


def _flatten_blueprint_for_diff(blueprint: DomainBlueprintContract) -> dict[str, Any]:
    return {
        "domain_name": blueprint.domain_name,
        "problem_statement": blueprint.problem_statement,
        "target_user_persona": blueprint.target_user_persona,
        "task_family": blueprint.task_family,
        "input_modality": blueprint.input_modality,
        "expected_output_schema": blueprint.expected_output_schema,
        "expected_output_examples": blueprint.expected_output_examples,
        "safety_compliance_notes": blueprint.safety_compliance_notes,
        "deployment_target_constraints": blueprint.deployment_target_constraints,
        "success_metrics": [row.model_dump(mode="json") for row in blueprint.success_metrics],
        "glossary": [row.model_dump(mode="json") for row in blueprint.glossary],
        "confidence_score": blueprint.confidence_score,
        "unresolved_assumptions": blueprint.unresolved_assumptions,
    }


async def diff_domain_blueprint_revisions(
    db: AsyncSession,
    *,
    project_id: int,
    from_version: int,
    to_version: int,
) -> DomainBlueprintDiffResponse:
    from_row = await get_domain_blueprint_revision(db, project_id=project_id, version=from_version)
    to_row = await get_domain_blueprint_revision(db, project_id=project_id, version=to_version)
    if from_row is None or to_row is None:
        raise ValueError("One or both blueprint versions were not found")

    from_payload = _flatten_blueprint_for_diff(_record_to_blueprint(from_row))
    to_payload = _flatten_blueprint_for_diff(_record_to_blueprint(to_row))
    changed: list[DomainBlueprintDiffItem] = []
    for key in sorted(set(from_payload.keys()) | set(to_payload.keys())):
        if from_payload.get(key) == to_payload.get(key):
            continue
        changed.append(
            DomainBlueprintDiffItem(field=key, before=from_payload.get(key), after=to_payload.get(key))
        )

    return DomainBlueprintDiffResponse(
        project_id=project_id,
        from_version=from_version,
        to_version=to_version,
        changed_fields=changed,
    )


def glossary_help(
    *,
    term_query: str = "",
    project_id: int | None = None,
    latest_blueprint: DomainBlueprintRevision | None = None,
) -> DomainBlueprintGlossaryHelpResponse:
    query = _normalize_text(term_query).lower()
    entries: list[GlossaryEntry] = [
        GlossaryEntry(term=term, plain_language=plain, category=category)
        for term, (plain, category) in BUILTIN_GLOSSARY.items()
    ]
    if latest_blueprint is not None:
        for raw in list(latest_blueprint.glossary or []):
            if not isinstance(raw, dict):
                continue
            try:
                entries.append(GlossaryEntry.model_validate(raw))
            except Exception:
                continue

    deduped: list[GlossaryEntry] = []
    seen: set[str] = set()
    for entry in entries:
        key = entry.term.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    if query:
        filtered = [
            row
            for row in deduped
            if query in row.term.lower() or query in row.plain_language.lower() or query in row.category.lower()
        ]
    else:
        filtered = deduped

    return DomainBlueprintGlossaryHelpResponse(
        project_id=project_id,
        term_query=query,
        count=len(filtered),
        entries=filtered[:50],
    )
