/**
 * TypeScript shapes for the Wave G observability plane (priority.md
 * P31–P34 backend, P36 frontend).
 *
 * Mirrors:
 * - backend/app/services/run_event_service.py            (P31)
 * - backend/app/services/timeline_service.py             (P32)
 * - backend/app/services/run_event_clustering_service.py (P33)
 * - backend/app/services/support_bundle_service.py       (P34)
 *
 * Frontend models only the fields it consumes; additions to the
 * backend payload land here when a UI surface needs them.
 */

// -- shared enums ------------------------------------------------------

export type Severity = 'info' | 'warning' | 'error' | 'critical';

export type Stage =
    | 'ingestion'
    | 'cleaning'
    | 'adapter'
    | 'training'
    | 'eval'
    | 'export'
    | 'deployment'
    | 'autopilot'
    | 'system';

export const KNOWN_STAGES: Stage[] = [
    'ingestion',
    'cleaning',
    'adapter',
    'training',
    'eval',
    'export',
    'deployment',
    'autopilot',
    'system',
];

export const KNOWN_SEVERITIES: Severity[] = [
    'info',
    'warning',
    'error',
    'critical',
];

// -- P31 RunEvent ------------------------------------------------------

export interface RunEvent {
    id: number;
    project_id: number;
    run_id: string;
    parent_run_id: string | null;
    stage: Stage | string;
    severity: Severity | string;
    reason_code: string | null;
    actor: string;
    summary: string | null;
    payload: Record<string, unknown>;
    ts: string | null;
    created_at: string | null;
}

export interface RunEventsResponse {
    project_id: number;
    limit: number;
    events: RunEvent[];
}

export interface RunEventsForRunResponse {
    run_id: string;
    limit: number;
    events: RunEvent[];
}

// -- P32 Timeline ------------------------------------------------------

export interface TimelineNode {
    run_id: string;
    parent_run_id: string | null;
    is_orphan: boolean;
    stage: Stage | string;
    stages_present: string[];
    summary: string | null;
    actor: string;
    first_ts: string | null;
    last_ts: string | null;
    duration_seconds: number | null;
    event_count: number;
    severity_counts: Record<string, number>;
    highest_severity: Severity | string;
    latest_reason_code: string | null;
    children: TimelineNode[];
}

export interface TimelineResponse {
    project_id: number;
    window_start: string | null;
    window_end: string | null;
    total_events: number;
    total_runs: number;
    orphaned_count: number;
    truncated: boolean;
    tree: TimelineNode[];
    anchor_run_id?: string;
    anchor_present?: boolean;
}

export interface TimelineFilters {
    since?: string;
    until?: string;
    stage?: Stage | '';
    severity?: Severity | '';
    run_id?: string;
    limit?: number;
}

// -- P33 Failure clusters ---------------------------------------------

export interface FailureCluster {
    id: number;
    project_id: number;
    stage: string;
    reason_code: string;
    signature: string;
    failure_count: number;
    first_seen_at: string | null;
    last_seen_at: string | null;
    exemplar_event_ids: number[];
    exemplar_summaries: string[];
    exemplar_run_ids: string[];
    last_computed_at: string | null;
}

export interface FailureClusterListResponse {
    project_id: number;
    limit: number;
    clusters: FailureCluster[];
}

export interface FailureClusterRecomputeResponse {
    project_id: number;
    window_start: string | null;
    window_end: string | null;
    events_considered: number;
    events_skipped_no_reason_code: number;
    clusters_total: number;
    clusters_created: number;
    clusters_updated: number;
    computed_at: string;
}

// -- P34 Support bundles ----------------------------------------------

export interface RedactionStats {
    total: number;
    by_reason: Record<string, number>;
}

export interface SupportBundleMetadata {
    bundle_uid: string;
    project_id: number;
    size_bytes: number;
    sha256: string | null;
    section_counts: Record<string, number>;
    redactions_applied: Record<string, RedactionStats>;
    expires_at: string | null;
    created_at: string | null;
    download_url: string;
    download_token: string;
    actor: string;
}

export interface SupportBundleListItem {
    bundle_uid: string;
    size_bytes: number;
    sha256: string | null;
    section_counts: Record<string, number>;
    redactions_applied: Record<string, RedactionStats>;
    actor: string;
    created_at: string | null;
    expires_at: string | null;
}

export interface SupportBundleListResponse {
    project_id: number;
    limit: number;
    bundles: SupportBundleListItem[];
}
