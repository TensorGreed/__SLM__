/**
 * Typed client for the recipe-aware eval-pack scaffolder (E5).
 *
 * GET → recipe-derived draft (not persisted).
 * POST → persist edited draft + flip evaluation_preferred_pack_id to
 * ``evalpack.project.scaffolded`` so the resolver picks it up.
 */

import api from './client';


export interface ScaffoldGate {
    gate_id: string;
    metric_id: string;
    operator: string;
    threshold: number;
    required: boolean;
}


export interface ScaffoldTaskSpec {
    task_profile: string;
    display_name: string;
    description?: string;
    required_metric_ids: string[];
    gates: ScaffoldGate[];
}


export interface ScaffoldDraftPack {
    pack_id: string;
    display_name: string;
    description: string;
    version: string;
    owner: string;
    tags: string[];
    default_task_profile: string;
    task_specs: ScaffoldTaskSpec[];
    gates: ScaffoldGate[];
}


export interface ScaffoldResponse {
    project_id: number;
    recipe_id: string;
    gold_set_summary: { row_count: number; dataset_types_seen: string[] };
    draft_pack: ScaffoldDraftPack;
}


export interface SaveScaffoldResponse {
    project_id: number;
    preferred_pack_id: string;
    scaffolded_pack: ScaffoldDraftPack;
}


export async function fetchPackScaffold(projectId: number): Promise<ScaffoldResponse> {
    const resp = await api.get(`/projects/${projectId}/evaluation/pack-scaffold`);
    return resp.data as ScaffoldResponse;
}


export async function savePackScaffold(
    projectId: number,
    draft: ScaffoldDraftPack,
): Promise<SaveScaffoldResponse> {
    const resp = await api.post(
        `/projects/${projectId}/evaluation/pack-scaffold`,
        { draft_pack: draft },
    );
    return resp.data as SaveScaffoldResponse;
}


/** Gap-#5 slice 1/2 — gate-options catalog the scaffold editor reads
 *  to populate its metric + operator dropdowns. Recipe-aware: the
 *  catalog's ``recommended`` flag mirrors the scaffolder's
 *  required_metric_ids for the project's selected recipe. */
export interface GateOperatorOption {
    value: string;
    label: string;
}


export interface GateMetricOption {
    metric_id: string;
    label: string;
    description: string;
    expected_range: [number, number] | number[];
    default_operator: string;
    recommended: boolean;
}


export interface GateOptionsResponse {
    operators: GateOperatorOption[];
    metrics: GateMetricOption[];
    recipe_id: string | null;
}


export async function fetchGateOptions(projectId: number): Promise<GateOptionsResponse> {
    const resp = await api.get(`/projects/${projectId}/evaluation/gate-options`);
    return resp.data as GateOptionsResponse;
}
