/**
 * Opt-in "null counts as absent" JSON scoring (backend `null_scoring`).
 *
 * When on, a predicted `"key": null` counts the same as the key being left out when a JSON answer is
 * compared to its answer key (extraction fields, tool-call arguments). Off by default. Saved per project,
 * and a held-out eval run may override it (`HeldoutEvalRunRequest.null_as_absent`).
 *   GET /projects/{id}/settings/null-as-absent
 *   PUT /projects/{id}/settings/null-as-absent   { enabled: true | false | null }
 * An eval scored with it on records `metrics.json_scoring = {null_as_absent: true, source}`.
 */

import api from './client';

export interface NullAsAbsentSetting {
    enabled: boolean;
}

/** `metrics.json_scoring` on an eval result (only when the rule was on). */
export interface JsonScoringRecord {
    null_as_absent?: boolean;
    source?: 'project' | 'run';
}

export async function fetchNullAsAbsent(projectId: number): Promise<NullAsAbsentSetting> {
    const res = await api.get<NullAsAbsentSetting>(`/projects/${projectId}/settings/null-as-absent`);
    return res.data;
}

export async function saveNullAsAbsent(projectId: number, enabled: boolean | null): Promise<NullAsAbsentSetting> {
    const res = await api.put<NullAsAbsentSetting>(`/projects/${projectId}/settings/null-as-absent`, { enabled });
    return res.data;
}

/** Whether an eval result was scored with the rule on. */
export function scoredNullAsAbsent(metrics: { json_scoring?: JsonScoringRecord | null } | null | undefined): boolean {
    return metrics?.json_scoring?.null_as_absent === true;
}
