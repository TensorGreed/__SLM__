/**
 * Typed wrapper for the unified task-shape detector (Wave 2b).
 *
 * GET  /api/projects/{id}/task-shape          — rank shapes for the project's data
 * POST /api/projects/{id}/task-shape/confirm  — persist the choice (what prep,
 *                                               training and eval read)
 * The import wizard gets the same detection inline from /dataset-import/introspect.
 */

import api from './client';

export interface TaskShapeCandidate {
    task_profile: string;
    label: string;
    description: string;
    adapter_id: string;
    recipe_id: string;
    mapper_id: string | null;
    field_map: Record<string, unknown>;
    confidence: number;
    map_rate: number | null;
    rationale: string[];
    source: 'columns' | 'row_fit' | 'fallback' | string;
}

export interface TaskShapeDetection {
    top: TaskShapeCandidate;
    candidates: TaskShapeCandidate[];
    needs_confirmation: boolean;
    confirm_threshold: number;
    rows_examined: number;
}

export interface TaskShapeCatalogEntry {
    task_profile: string;
    label: string;
    description: string;
}

export interface ProjectTaskShape {
    project_id: number;
    confirmed: { task_profile: string; label: string; adapter_id: string | null } | null;
    detection: TaskShapeDetection | null;
    catalog: TaskShapeCatalogEntry[];
}

export interface ConfirmTaskShapeResult {
    task_profile: string;
    label: string;
    adapter_id: string;
    recipe_id: string;
}

export async function fetchProjectTaskShape(
    projectId: number,
    intent?: string,
): Promise<ProjectTaskShape> {
    const res = await api.get<ProjectTaskShape>(
        `/projects/${projectId}/task-shape`,
        intent ? { params: { intent } } : undefined,
    );
    return res.data;
}

export async function confirmTaskShape(
    projectId: number,
    taskProfile: string,
): Promise<ConfirmTaskShapeResult> {
    const res = await api.post<ConfirmTaskShapeResult>(
        `/projects/${projectId}/task-shape/confirm`,
        { task_profile: taskProfile },
    );
    return res.data;
}
