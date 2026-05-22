/**
 * Typed wrappers for the project-template catalog + instantiation API.
 *
 * Endpoints:
 *   GET  /api/project-templates                       — list
 *   GET  /api/project-templates/{slug}                — single
 *   POST /api/project-templates/{slug}/instantiate    — clone into a new Project
 */

import api from './client';
import type { Project } from '../types';

export interface ProjectTemplateSummary {
    slug: string;
    name: string;
    headline: string;
    description: string;
    icon: string;
    recipe_id: string | null;
    task_profile: string;
    target_profile: string;
    training_preferred_plan_profile: string;
    evaluation_preferred_pack_id: string | null;
    minimum_dataset_size: number;
    recommended_base_models: string[];
    labels: string[];
    suggested_brief: string;
    template_version: string;
    dataset_input_field: string;
    dataset_output_field: string;
}

export interface ProjectTemplateListResponse {
    templates: ProjectTemplateSummary[];
    count: number;
}

export async function listProjectTemplates(): Promise<ProjectTemplateListResponse> {
    const res = await api.get<ProjectTemplateListResponse>('/project-templates');
    return res.data;
}

export async function getProjectTemplate(
    slug: string,
): Promise<ProjectTemplateSummary> {
    const res = await api.get<ProjectTemplateSummary>(
        `/project-templates/${encodeURIComponent(slug)}`,
    );
    return res.data;
}

export async function instantiateProjectTemplate(
    slug: string,
    projectName: string | null,
): Promise<Project> {
    const body = projectName ? { project_name: projectName } : {};
    const res = await api.post<Project>(
        `/project-templates/${encodeURIComponent(slug)}/instantiate`,
        body,
    );
    return res.data;
}
