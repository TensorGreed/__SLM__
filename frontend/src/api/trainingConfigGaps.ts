/**
 * Training Config Gaps API client — Coach-stage-2 phases 1 + 2.
 *
 * Phase 1: read-only gap report (`fetchTrainingConfigGaps`).
 * Phase 2: preview + apply for one-click patches (`previewPatch` /
 * `applyPatch`) and the persisted overrides read (`fetchOverrides`).
 * The payload mirrors data-health's shape so the panel rendering can
 * share the same severity vocab.
 */

import api from './client';

export type GapSeverity = 'ok' | 'warn' | 'block';

export interface GapSuggestedAction {
    kind?: string;
    label?: string;
    target?: string;
    params?: Record<string, unknown>;
}

export interface GapSignal {
    id: string;
    severity: GapSeverity;
    headline: string;
    plain_english: string;
    why_it_matters: string;
    suggested_action: GapSuggestedAction | null;
    context: Record<string, unknown>;
    /**
     * Phase 2 — when set, the signal can be one-click resolved by
     * posting `{signal_id: signal.id}` to `/patch/preview` then
     * `/patch/apply`. ``null`` = no safe patch available.
     */
    apply_patch_kind?: string | null;
}

export interface GapGroup {
    id: string;
    title: string;
    subtitle: string;
    signals: GapSignal[];
}

export interface TrainingConfigGapReport {
    project_id: number;
    computed_at: string;
    overall: GapSeverity;
    severity_summary: { ok: number; warn: number; block: number };
    total_signals: number;
    groups: GapGroup[];
}

export async function fetchTrainingConfigGaps(
    projectId: number,
): Promise<TrainingConfigGapReport> {
    const res = await api.get<TrainingConfigGapReport>(
        `/projects/${projectId}/training-config-gaps`,
    );
    return res.data;
}

// ── Phase 2: patch engine ───────────────────────────────────────────

/** The before → after diff a patch *would* apply. */
export interface TrainingConfigPatchPreview {
    project_id: number;
    signal_id: string;
    patch_kind: string;
    patch_label: string;
    plain_english: string;
    patch: Record<string, number>;
    before: Record<string, number>;
    after: Record<string, number>;
    safe_to_apply: boolean;
}

/** The result of an apply call — the preview shape plus the persisted
 * overrides block, so the caller can pipe it into TrainingPanel. */
export interface TrainingConfigPatchResult
    extends TrainingConfigPatchPreview {
    applied: true;
    overrides_after: Record<string, number>;
}

/** The persisted overrides block — what TrainingPanel reads on mount
 * so the visible form matches the gap scanner's effective config. */
export interface TrainingConfigOverridesResponse {
    project_id: number;
    overrides: Record<string, number>;
}

export async function fetchOverrides(
    projectId: number,
): Promise<TrainingConfigOverridesResponse> {
    const res = await api.get<TrainingConfigOverridesResponse>(
        `/projects/${projectId}/training-config-gaps/overrides`,
    );
    return res.data;
}

export async function previewPatch(
    projectId: number,
    signalId: string,
): Promise<TrainingConfigPatchPreview> {
    const res = await api.post<TrainingConfigPatchPreview>(
        `/projects/${projectId}/training-config-gaps/patch/preview`,
        { signal_id: signalId },
    );
    return res.data;
}

export async function applyPatch(
    projectId: number,
    signalId: string,
): Promise<TrainingConfigPatchResult> {
    const res = await api.post<TrainingConfigPatchResult>(
        `/projects/${projectId}/training-config-gaps/patch/apply`,
        { signal_id: signalId },
    );
    return res.data;
}

/** DOM event name the panel dispatches after a successful apply so the
 * TrainingPanel can update its form state without a page reload. */
export const TRAINING_OVERRIDES_APPLIED_EVENT =
    'brewslm:training-overrides-applied';

/** Detail payload carried on the DOM event. The TrainingPanel listener
 * pipes `overrides` directly into `applySuggestedConfig`. */
export interface TrainingOverridesAppliedDetail {
    projectId: number;
    overrides: Record<string, number>;
}
