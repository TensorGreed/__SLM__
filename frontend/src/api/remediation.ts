/**
 * Typed wrapper for the remediation tracking API (E2).
 *
 * Posts one event per user click on a suggested-action button so the
 * post-eval pipeline can later stamp it with the pass-rate lift.
 * Best-effort — the caller fires this in a fire-and-forget pattern;
 * a failed POST must NOT block the user's navigation.
 */

import api from './client';


export type RemediationOutcome =
    | 'clicked'
    | 'dismissed'
    | 'applied'
    | 'ignored';


export interface RemediationEventPayload {
    /** ``synth_augment`` / ``synth_balance`` / ``synth_diversify`` /
     *  ``fix_gold_rows`` from the forecast panel, or ``cluster_fix``
     *  from a failure-cluster card. Free-form so future kinds land
     *  without an API contract change. */
    kind: string;
    params?: unknown;
    outcome?: RemediationOutcome;
}


export interface RemediationEventResponse {
    id: number;
    project_id: number;
    action_kind: string;
    params_hash: string;
    outcome: RemediationOutcome;
    observed_at: string;
}


/** Record a remediation-action event. Fire-and-forget — the panel
 *  shouldn't block its navigation on this POST. The promise resolves
 *  with the persisted event but callers can ignore the result. */
export async function recordRemediationEvent(
    projectId: number,
    payload: RemediationEventPayload,
): Promise<RemediationEventResponse | null> {
    try {
        const resp = await api.post(
            `/projects/${projectId}/remediation/events`,
            payload,
        );
        return resp.data as RemediationEventResponse;
    } catch {
        // Telemetry must never block the user's flow. Swallow.
        return null;
    }
}


// ---------------------------------------------------------------------------
// Code-remediation settings (the `code-remediation` recipe)
//
//   GET /projects/{id}/remediation-config
//   PUT /projects/{id}/remediation-config
// ---------------------------------------------------------------------------
export type RemediationDetector =
    | { kind: 'project'; project_id: number; experiment_id: number | null }
    | { kind: 'command'; check_id: string };

export interface RemediationConfig {
    detector: RemediationDetector | null;
    /** null = every check that applies to the language. */
    checks: string[] | null;
    judge: boolean | null;
    retrieve_passages: boolean | null;
    top_k: number | null;
}

export interface RemediationEffective {
    detector?: RemediationDetector | null;
    checks?: string[] | null;
    judge: boolean;
    retrieve_passages: boolean;
    top_k: number;
}

export interface AvailableDetector {
    project_id: number;
    project_name: string;
    experiment_id: number;
    run_name: string | null;
    completed_at: string | null;
}

export interface RemediationConfigResponse {
    config: RemediationConfig;
    effective: RemediationEffective;
    /** "<project name> · run <id>" for the configured project detector. */
    detector_label: string | null;
    available_detectors: AvailableDetector[];
}

export async function getRemediationConfig(projectId: number): Promise<RemediationConfigResponse> {
    const res = await api.get<RemediationConfigResponse>(`/projects/${projectId}/remediation-config`);
    return res.data;
}

export async function putRemediationConfig(
    projectId: number,
    config: RemediationConfig,
): Promise<RemediationConfigResponse> {
    const res = await api.put<RemediationConfigResponse>(`/projects/${projectId}/remediation-config`, config);
    return res.data;
}

/** The picker label for a detector run, so the user can correlate it. */
export function detectorOptionLabel(d: AvailableDetector): string {
    return `${d.project_name} · run ${d.experiment_id}`;
}
