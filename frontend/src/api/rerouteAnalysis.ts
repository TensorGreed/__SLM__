/**
 * USER-SUCCESS Epic 7 Phase 7c — typed client for the post-eval
 * decision engine (Phase 7a backend) + the RAG reroute endpoint
 * (Phase 7b backend).
 *
 * Two endpoints:
 *   * GET  /api/projects/{id}/evaluation/{eval_id}/reroute-analysis
 *   * POST /api/projects/{id}/reroute-to-rag
 *
 * Shapes mirror the backend RerouteAnalysis / ProjectRerouteToRagResponse.
 * Keep these in lockstep with
 * backend/app/services/post_eval_decision_engine_service.py +
 * backend/app/schemas/project.py.
 */

import api from './client';


export type RerouteRecommendationKind =
    | 'try_rag'
    | 'try_prompt_engineering'
    | 'expand_data'
    | 'stay_the_course';


export interface RerouteSignal {
    id: string;
    fired: boolean;
    detail: string;
    evidence: Record<string, unknown>;
}


export interface RerouteRecommendation {
    kind: RerouteRecommendationKind;
    confidence: number;
    rationale: string;
}


export interface RerouteAnalysis {
    eval_result_id: number;
    project_id: number;
    pass_rate: number | null;
    signals: RerouteSignal[];
    recommendation: RerouteRecommendation;
    computed_at: string;
}


export interface RerouteToRagResponse {
    new_project_id: number;
    new_project_name: string;
    source_project_id: number;
    clone_report: Record<string, unknown> | null;
}


export async function fetchRerouteAnalysis(
    projectId: number,
    evalResultId: number,
    options: { refresh?: boolean } = {},
): Promise<RerouteAnalysis> {
    const params = options.refresh ? '?refresh=true' : '';
    const resp = await api.get<RerouteAnalysis>(
        `/projects/${projectId}/evaluation/${evalResultId}/reroute-analysis${params}`,
    );
    return resp.data;
}


export async function rerouteToRag(
    projectId: number,
    nameSuffix?: string,
): Promise<RerouteToRagResponse> {
    const body: Record<string, unknown> = {};
    if (nameSuffix !== undefined) {
        body.name_suffix = nameSuffix;
    }
    const resp = await api.post<RerouteToRagResponse>(
        `/projects/${projectId}/reroute-to-rag`,
        body,
    );
    return resp.data;
}


// Hardening Phase H1 — async-job variant. Returns the Job stub
// (HTTP 202). The user keeps working; the notification bell shows
// progress and surfaces the "Open" deep-link when the clone is done.
import type { Job } from './jobs';

export async function rerouteToRagAsync(
    projectId: number,
    nameSuffix?: string,
): Promise<Job> {
    const body: Record<string, unknown> = {};
    if (nameSuffix !== undefined) {
        body.name_suffix = nameSuffix;
    }
    const resp = await api.post<Job>(
        `/projects/${projectId}/reroute-to-rag?async_job=true`,
        body,
    );
    return resp.data;
}
