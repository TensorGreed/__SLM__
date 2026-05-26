/**
 * Typed client for the per-project archetype comparison
 * (USER-SUCCESS Epic 8 Phase 8b). Shapes mirror
 * `backend/app/services/archetype_service.ProjectArchetypeComparison`
 * exactly — keep them in lockstep.
 */

import api from './client';


export type FeatureStatus = 'ok' | 'below' | 'above' | 'missing';
export type ComparisonSummary =
    | 'healthy'
    | 'below_cohort'
    | 'above_cohort'
    | 'mixed';


export interface FeatureDistribution {
    feature_id: string;
    label: string;
    n_projects: number;
    p25: number | null;
    p50: number | null;
    p75: number | null;
    mean: number | null;
    min: number | null;
    max: number | null;
    unit: string;
}


export interface CohortMember {
    id: number;
    name: string;
    source: 'user' | 'template';
    pass_rate: number | null;
}


export interface RecipeArchetype {
    recipe_id: string;
    n_passing_projects: number;
    n_user_projects: number;
    n_template_seeds: number;
    computed_at: string;
    features: FeatureDistribution[];
    cohort_provenance: CohortMember[];
}


export interface FeatureComparison {
    feature_id: string;
    label: string;
    unit: string;
    your_value: number | null;
    archetype_p25: number | null;
    archetype_p50: number | null;
    archetype_p75: number | null;
    status: FeatureStatus;
    suggestion: string | null;
    suggested_action: {
        kind: 'run_playbook' | 'navigate';
        params: Record<string, unknown>;
    } | null;
}


export interface ProjectArchetypeComparison {
    project_id: number;
    recipe_id: string;
    archetype: RecipeArchetype;
    features: FeatureComparison[];
    summary: ComparisonSummary;
}


export async function fetchProjectArchetypeComparison(
    projectId: number,
): Promise<ProjectArchetypeComparison> {
    const resp = await api.get<ProjectArchetypeComparison>(
        `/projects/${projectId}/archetype-comparison`,
    );
    return resp.data;
}
