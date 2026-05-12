/**
 * Demo project shapes — newbie UX Phase 3.
 * Mirrors backend/app/services/demo_project_service.py.
 */

export interface DemoArchetype {
    slug: string;
    name: string;
    headline: string;
    description: string;
    task_profile: string;
    target_profile: string;
    suggested_brief: string;
}

export interface DemoCatalogResponse {
    archetypes: DemoArchetype[];
}

export interface DemoSeedResponse {
    summary: {
        slug: string;
        created: boolean;
        project_id: number;
        project_name?: string;
        source_dataset_id?: number;
        source_row_count?: number;
        gold_set_id?: number;
        gold_version_id?: number;
        gold_row_count?: number;
        suggested_brief?: string;
    };
    project: {
        id: number;
        name: string;
        description: string | null;
        status: string | null;
        beginner_mode: boolean;
        target_profile_id: string | null;
        training_preferred_plan_profile: string | null;
        evaluation_preferred_pack_id: string | null;
    };
}
