"""Canonical reason-code taxonomy (priority.md P33, Wave G).

Stable enum of ``reason_code`` values that ``severity in {error, critical}``
:class:`RunEvent` rows must set. Grouped by stage. Adding a new code is a
small change here + a service hook that emits it; emitting a code that is
**not** in this file is rejected at the service boundary
(:func:`run_event_service.emit_event`) with ``invalid_reason_code:<value>``.

This is the "lint rule" from priority.md spelled in code: a single
authoritative list is the easiest thing to grep, code-review, and
update. P36's failure-analysis surface and P34's support bundle both
read these strings, so the taxonomy doubles as the operator-facing
vocabulary.

Extending the taxonomy:
1. Add the constant + a one-line docstring below.
2. Add it to the ``REASON_CODES_BY_STAGE`` map under the correct stage.
3. Hook the emitting service to set it on the matching error path.
"""

from __future__ import annotations

from app.models.run_event import (
    STAGE_ADAPTER,
    STAGE_AUTOPILOT,
    STAGE_CLEANING,
    STAGE_DEPLOYMENT,
    STAGE_EVAL,
    STAGE_EXPORT,
    STAGE_INGESTION,
    STAGE_SYSTEM,
    STAGE_TRAINING,
)


# -- ingestion ----------------------------------------------------------
INGEST_UNSUPPORTED_FORMAT = "ingest_unsupported_format"
"""File extension not in the configured supported set."""
INGEST_IO_ERROR = "ingest_io_error"
"""Disk write / read failure during ingest."""
INGEST_VALIDATION_FAILED = "ingest_validation_failed"
"""Ingested document failed schema or content validation."""
DATASET_IMPORT_RUN = "dataset_import_run"
"""Generic dataset-import pipeline (Phase A–F) committed rows to the
project's synthetic dataset. Severity is ``info``; the payload carries
source, locator, mapper, accepted/rejected counts, written_path, and
config_id when re-run from a saved mapping."""
DATASET_IMPORT_FAILED = "dataset_import_failed"
"""Generic dataset-import pipeline raised before any rows were written.
Severity is ``error``. Payload carries source, locator, mapper, and
the error message."""
ANNOTATION_JOB_CREATED = "annotation_job_created"
"""Annotation label-job created (Story 1.1). Severity is ``info``;
payload carries job_id, name, label_type, target_rows."""
ANNOTATION_LABEL_SUBMITTED = "annotation_label_submitted"
"""Reviewer submitted a label for one row in an annotation job.
Severity is ``info``; payload carries job_id, row_id, user_id,
label_type, and (optionally) the submitted label_payload."""

# -- cleaning -----------------------------------------------------------
CLEANING_OUTLIER_THRESHOLD_EXCEEDED = "cleaning_outlier_threshold_exceeded"
"""Outlier removal removed more rows than the configured threshold allowed."""
CLEANING_PII_BLOCK = "cleaning_pii_block"
"""PII / safety scan blocked the dataset from advancing."""

# -- adapter ------------------------------------------------------------
ADAPTER_SCHEMA_MISMATCH = "adapter_schema_mismatch"
"""Adapter could not match its declared input/output schema to the data."""
ADAPTER_FIELD_RESOLUTION_FAILED = "adapter_field_resolution_failed"
"""Adapter could not resolve a field mapping (missing column, wrong type)."""

# -- training -----------------------------------------------------------
TRAINING_DISPATCH_ERROR = "training_dispatch_error"
"""Failure dispatching the training run to the runtime backend."""
TRAINING_RUNTIME_ERROR = "training_runtime_error"
"""Generic runtime failure from inside the training loop."""
TRAINING_OOM = "training_oom"
"""GPU out-of-memory during training."""
TRAINING_TIMEOUT = "training_timeout"
"""Training run exceeded the configured wallclock budget."""
TRAINING_CANCELLED = "training_cancelled"
"""Training run was cancelled (operator action or upstream signal)."""

# -- eval ---------------------------------------------------------------
EVAL_RUNTIME_ERROR = "eval_runtime_error"
"""Generic failure inside the evaluation runner."""
EVAL_DATASET_MISSING = "eval_dataset_missing"
"""Eval pack referenced a dataset that no longer exists."""
EVAL_JUDGE_UNAVAILABLE = "eval_judge_unavailable"
"""LLM judge call failed (provider down / quota exhausted / config error)."""

# -- export -------------------------------------------------------------
EXPORT_RUN_FAILED = "export_run_failed"
"""Generic export failure (artifact build, manifest write, smoke check)."""
EXPORT_ARTIFACT_MISSING = "export_artifact_missing"
"""A required model / tokenizer artifact was missing at export time."""
EXPORT_QUANTIZATION_FAILED = "export_quantization_failed"
"""Quantization step exited non-zero or produced an invalid artifact."""

# -- deployment ---------------------------------------------------------
DEPLOYMENT_SMOKE_FAILED = "deployment_smoke_failed"
"""Post-deploy smoke check failed against the live endpoint."""
DEPLOYMENT_PROMOTE_BLOCKED = "deployment_promote_blocked"
"""Promote refused (status not promotable / readiness gate failed)."""
DEPLOYMENT_ROLLBACK_NO_PREDECESSOR = "deployment_rollback_no_predecessor"
"""Rollback refused because no superseded predecessor exists."""
DEPLOYMENT_DRIFT_DETECTED = "deployment_drift_detected"
"""Drift check found pass-rate delta beyond tolerance vs baseline."""

# -- autopilot ----------------------------------------------------------
AUTOPILOT_REPAIR_BLOCKED = "autopilot_repair_blocked"
"""Autopilot repair refused — strict mode or unsafe action."""
AUTOPILOT_STRICT_MODE_REFUSED = "autopilot_strict_mode_refused"
"""Strict mode rejected an auto-repair that would otherwise have applied."""
AUTOPILOT_NO_SAFE_PLAN = "autopilot_no_safe_plan"
"""Autopilot couldn't construct a safe plan from the current intent."""

# -- system -------------------------------------------------------------
SYSTEM_DB_ERROR = "system_db_error"
"""Unexpected database error from a service-internal write path."""
SYSTEM_CONFIG_INVALID = "system_config_invalid"
"""A required configuration value was missing or malformed at runtime."""
EXTENSION_LOAD_FAILED = "extension_load_failed"
"""Plugin module import / register hook raised (priority.md P37)."""
EXTENSION_CONTRACT_INVALID = "extension_contract_invalid"
"""Plugin module failed one or more contract checks (priority.md P37)."""


REASON_CODES_BY_STAGE: dict[str, frozenset[str]] = {
    STAGE_INGESTION: frozenset({
        INGEST_UNSUPPORTED_FORMAT,
        INGEST_IO_ERROR,
        INGEST_VALIDATION_FAILED,
        DATASET_IMPORT_RUN,
        DATASET_IMPORT_FAILED,
        ANNOTATION_JOB_CREATED,
        ANNOTATION_LABEL_SUBMITTED,
    }),
    STAGE_CLEANING: frozenset({
        CLEANING_OUTLIER_THRESHOLD_EXCEEDED,
        CLEANING_PII_BLOCK,
    }),
    STAGE_ADAPTER: frozenset({
        ADAPTER_SCHEMA_MISMATCH,
        ADAPTER_FIELD_RESOLUTION_FAILED,
    }),
    STAGE_TRAINING: frozenset({
        TRAINING_DISPATCH_ERROR,
        TRAINING_RUNTIME_ERROR,
        TRAINING_OOM,
        TRAINING_TIMEOUT,
        TRAINING_CANCELLED,
    }),
    STAGE_EVAL: frozenset({
        EVAL_RUNTIME_ERROR,
        EVAL_DATASET_MISSING,
        EVAL_JUDGE_UNAVAILABLE,
    }),
    STAGE_EXPORT: frozenset({
        EXPORT_RUN_FAILED,
        EXPORT_ARTIFACT_MISSING,
        EXPORT_QUANTIZATION_FAILED,
    }),
    STAGE_DEPLOYMENT: frozenset({
        DEPLOYMENT_SMOKE_FAILED,
        DEPLOYMENT_PROMOTE_BLOCKED,
        DEPLOYMENT_ROLLBACK_NO_PREDECESSOR,
        DEPLOYMENT_DRIFT_DETECTED,
    }),
    STAGE_AUTOPILOT: frozenset({
        AUTOPILOT_REPAIR_BLOCKED,
        AUTOPILOT_STRICT_MODE_REFUSED,
        AUTOPILOT_NO_SAFE_PLAN,
    }),
    STAGE_SYSTEM: frozenset({
        SYSTEM_DB_ERROR,
        SYSTEM_CONFIG_INVALID,
        EXTENSION_LOAD_FAILED,
        EXTENSION_CONTRACT_INVALID,
    }),
}


KNOWN_REASON_CODES: frozenset[str] = frozenset(
    code for codes in REASON_CODES_BY_STAGE.values() for code in codes
)


def is_known_reason_code(code: str | None) -> bool:
    return code is not None and code in KNOWN_REASON_CODES


def is_valid_for_stage(stage: str, code: str | None) -> bool:
    """Return True iff ``code`` is registered for ``stage``.

    Used as a soft check at emit time — we *warn* on stage/code mismatch
    rather than reject, since cross-stage codes are rare but legitimate
    (e.g. a system-stage event emitted from a deployment service).
    """
    if code is None:
        return True
    bucket = REASON_CODES_BY_STAGE.get(stage)
    if bucket is None:
        return False
    return code in bucket
