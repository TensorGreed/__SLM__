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
