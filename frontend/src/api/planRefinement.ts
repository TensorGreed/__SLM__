import api from './client';

export interface PlanRefinementSignal {
    id: string;
    severity: string;
    headline: string;
    target_tab?: string | null;
}

export interface CloudSafeProfile {
    recipe_id: string | null;
    task_profile: string | null;
    base_model_name: string | null;
    target_profile_id: string | null;
    labelled_row_count: number;
    label_distribution_shape: {
        num_classes: number;
        min_class_count: number;
        max_class_count: number;
        imbalance_ratio: number;
        classes_below_floor: number;
    } | null;
    truncation_risk: string | null;
    tokenizer_oov: string | null;
    archetype_below_band_features: string[];
    forecast_verdict: string | null;
}

export interface PlanRefinement {
    project_id: number;
    plan: {
        recipe_id: string | null;
        task_profile: string | null;
        base_model_name: string | null;
        target_profile_id: string | null;
    };
    cloud_safe_profile: CloudSafeProfile;
    plan_health: { verdict: 'ready' | 'attention' | 'mismatch' | string; signals: PlanRefinementSignal[] };
    privacy: { cloud_sharing: string; note: string };
    cloud_refinement: { available: boolean; supported_providers: string[]; reason: string };
}

/** Phase 1 — deterministic plan-refinement report (no cloud call). */
export async function getPlanRefinement(projectId: number): Promise<PlanRefinement> {
    const resp = await api.get(`/projects/${projectId}/refine-plan`);
    return resp.data as PlanRefinement;
}
