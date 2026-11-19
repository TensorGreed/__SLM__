/**
 * Typed wrapper for the per-project automatic lift check switch.
 *
 *   GET /projects/{id}/settings/auto-lift-eval
 *   PUT /projects/{id}/settings/auto-lift-eval   { enabled: true | false | null }
 *
 * `enabled` is the project's own setting (null = follow the default, which is on). The server-wide
 * `AUTO_LIFT_EVAL_ENABLED` switch wins when it is off: `global_enabled` false, `effective` false.
 */

import api from './client';

export interface AutoLiftSetting {
    enabled: boolean | null;
    global_enabled: boolean;
    effective: boolean;
    reason: 'disabled_globally' | 'disabled_for_project' | null;
}

export async function getAutoLiftSetting(projectId: number): Promise<AutoLiftSetting> {
    const res = await api.get<AutoLiftSetting>(`/projects/${projectId}/settings/auto-lift-eval`);
    return res.data;
}

export async function putAutoLiftSetting(projectId: number, enabled: boolean | null): Promise<AutoLiftSetting> {
    const res = await api.put<AutoLiftSetting>(`/projects/${projectId}/settings/auto-lift-eval`, { enabled });
    return res.data;
}
