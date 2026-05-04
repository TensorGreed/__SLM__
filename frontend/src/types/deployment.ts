/**
 * TypeScript shapes for the Wave F deployment plane (priority.md P25–P28).
 *
 * Mirrors:
 * - backend/app/services/deployment_version_service.py    (P25)
 * - backend/app/services/served_model_telemetry_service.py (P26)
 * - backend/app/services/deployment_drift_service.py       (P27)
 * - backend/app/services/deployment_score_service.py       (P28)
 *
 * The frontend only models the fields it actually consumes — additions
 * to the backend payload land here when a UI surface needs them.
 */

export type DeploymentVersionStatus =
    | 'pending'
    | 'promoted'
    | 'rejected'
    | 'rolled_back'
    | 'superseded';

export interface DeploymentVersion {
    id: number;
    project_id: number;
    export_id: number;
    registry_entry_id: number | null;
    version: number;
    target_id: string;
    target_kind: string | null;
    endpoint_name: string | null;
    endpoint_handle: string | null;
    region: string | null;
    instance_type: string | null;
    status: DeploymentVersionStatus;
    plan_payload: Record<string, unknown>;
    promoted_reason: string | null;
    rejected_reason: string | null;
    rolled_back_reason: string | null;
    rolled_back_to_id: number | null;
    actor: string;
    created_at: string | null;
    promoted_at: string | null;
    rejected_at: string | null;
    rolled_back_at: string | null;
    superseded_at: string | null;
}

export interface DeploymentRollbackAuditRow {
    id: number;
    deployment_version_id: number;
    project_id: number;
    sequence: number;
    action: 'promote' | 'reject' | 'rollback';
    reason: string | null;
    actor: string;
    status_after: string | null;
    rolled_back_to_id: number | null;
    payload: Record<string, unknown>;
    created_at: string | null;
}

export interface DeploymentVersionListResponse {
    project_id: number;
    deployment_versions: DeploymentVersion[];
}

export interface DeploymentVersionDetailResponse {
    deployment_version: DeploymentVersion;
    audit: DeploymentRollbackAuditRow[];
}

// -- Telemetry (P26) ---------------------------------------------------

export interface TelemetryAggregate {
    deployment_version_id: number;
    window_start: string;
    window_end: string;
    window_seconds: number;
    sample_count: number;
    request_volume: {
        total: number;
        per_second: number;
        per_minute: number;
    };
    latency_ms: {
        p50: number;
        p95: number;
        p99: number;
        min: number;
        max: number;
        mean: number;
    };
    errors: {
        count: number;
        rate: number;
    };
    tokens: {
        input_total: number;
        output_total: number;
        input_per_second: number;
        output_per_second: number;
        total_per_second: number;
    };
}

export interface TelemetrySample {
    id: number;
    ts: string | null;
    latency_ms: number;
    success: boolean;
    status_code: number | null;
    error_code: string | null;
    input_tokens: number | null;
    output_tokens: number | null;
    request_id: string | null;
}

export interface TelemetrySamplesResponse {
    deployment_version_id: number;
    limit: number;
    samples: TelemetrySample[];
}

// -- Drift check (P27) -------------------------------------------------

export interface DeploymentDriftCheck {
    id: number;
    deployment_version_id: number;
    project_id: number;
    gold_set_id: number | null;
    gold_set_version_id: number | null;
    baseline_experiment_id: number | null;
    baseline_eval_result_id: number | null;
    eval_type: string;
    baseline_pass_rate: number | null;
    current_pass_rate: number;
    delta: number | null;
    tolerance: number;
    drift_detected: boolean;
    samples_evaluated: number;
    samples_failed: number;
    samples_skipped: number;
    mode: string;
    notes: string | null;
    actor: string;
    per_row_results: Array<{
        row_id: number;
        match: boolean;
        expected: unknown;
        prediction: unknown;
        error: string | null;
    }>;
    summary: Record<string, unknown>;
    created_at: string | null;
}

export interface DeploymentDriftHistoryResponse {
    deployment_version_id: number;
    limit: number;
    drift_checks: DeploymentDriftCheck[];
}

// -- Deployability score (P28) -----------------------------------------

export type ScoreProvenance = 'measured' | 'estimated' | 'mixed';
export type ScoreConfidenceBand = 'low' | 'medium' | 'high';

export interface ScoreComponentSignal {
    key: string;
    value: unknown;
    ok: boolean;
}

export interface ScoreComponent {
    name: string;
    score: number | null;
    weight: number;
    weight_normalised?: number;
    provenance: 'measured' | 'estimated';
    confidence: number;
    signals: ScoreComponentSignal[];
    summary: string;
}

export interface DeploymentScore {
    id: number;
    deployment_version_id: number;
    project_id: number;
    overall_score: number;
    confidence: number;
    confidence_band: ScoreConfidenceBand;
    provenance: ScoreProvenance;
    components: ScoreComponent[];
    signals_summary: {
        deployment_version_id?: number;
        target_id?: string;
        telemetry_sample_count?: number;
        drift_check_id?: number | null;
        drift_detected?: boolean | null;
        components_present?: string[];
        components_missing?: string[];
    } & Record<string, unknown>;
    notes: string | null;
    actor: string;
    created_at: string | null;
}

export interface DeploymentScoreHistoryResponse {
    deployment_version_id: number;
    limit: number;
    scores: DeploymentScore[];
}
