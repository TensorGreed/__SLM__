/**
 * Eval Gaps API client — Coach-stage-2 phases 3 + 5.
 *
 * Phase 3: read-only gap report.
 * Phase 5: preview + apply for the two patch kinds
 * (regression_baseline_promote_last_green, label_kl_rebalance_eval).
 *
 * Shares the gap-payload shape with the training-config gap panel so
 * the panel rendering can reuse the same severity vocab.
 */

import api from './client';

import type { GapSeverity, GapSuggestedAction } from './trainingConfigGaps';

export type EvalGapSeverity = GapSeverity;

export interface EvalGapSignal {
    id: string;
    severity: EvalGapSeverity;
    headline: string;
    plain_english: string;
    why_it_matters: string;
    suggested_action: GapSuggestedAction | null;
    context: Record<string, unknown>;
    /**
     * Phase 5 — when set, the signal can be one-click resolved by
     * posting `{signal_id: signal.id}` to `/patch/preview` then
     * `/patch/apply`. ``null`` = no safe patch available yet.
     */
    apply_patch_kind?: string | null;
}

export interface EvalGapGroup {
    id: string;
    title: string;
    subtitle: string;
    signals: EvalGapSignal[];
}

export interface EvalGapReport {
    project_id: number;
    computed_at: string;
    overall: EvalGapSeverity;
    severity_summary: { ok: number; warn: number; block: number };
    total_signals: number;
    groups: EvalGapGroup[];
}

export async function fetchEvalGaps(
    projectId: number,
): Promise<EvalGapReport> {
    const res = await api.get<EvalGapReport>(
        `/projects/${projectId}/eval-gaps`,
    );
    return res.data;
}

// ── Phase 5: patch engine ───────────────────────────────────────────

/**
 * Baseline-promote preview shape — picks the most recent green run's
 * best checkpoint and projects its promote.
 */
export interface BaselinePromotePreview {
    project_id: number;
    signal_id: string;
    patch_kind: 'regression_baseline_promote_last_green';
    patch_label: string;
    plain_english: string;
    before: {
        promoted_checkpoint_id: number | null;
        promoted_experiment_id: number | null;
        promoted_step: number | null;
    };
    after: {
        promoted_checkpoint_id: number;
        promoted_experiment_id: number;
        promoted_step: number;
    };
    candidate: {
        experiment_id: number;
        experiment_name: string;
        checkpoint_id: number;
        checkpoint_step: number;
        checkpoint_is_best: boolean;
        pass_rate: number;
    };
    safe_to_apply: boolean;
}

/**
 * Label-KL rebalance preview shape — projects the per-class trim.
 */
export interface LabelKlRebalancePreview {
    project_id: number;
    signal_id: string;
    patch_kind: 'label_kl_rebalance_eval';
    patch_label: string;
    plain_english: string;
    before: { counts: Record<string, number>; kl_nats: number };
    after: { counts: Record<string, number>; kl_nats: number };
    rows_to_drop: number;
    gold_dev_path: string | null;
    safe_to_apply: boolean;
    skipped_reason: string | null;
}

export type EvalGapPatchPreview =
    | BaselinePromotePreview
    | LabelKlRebalancePreview;

export type EvalGapPatchResult = EvalGapPatchPreview & {
    applied: true;
    rows_after?: number;
};

export async function previewEvalPatch(
    projectId: number,
    signalId: string,
): Promise<EvalGapPatchPreview> {
    const res = await api.post<EvalGapPatchPreview>(
        `/projects/${projectId}/eval-gaps/patch/preview`,
        { signal_id: signalId },
    );
    return res.data;
}

export async function applyEvalPatch(
    projectId: number,
    signalId: string,
): Promise<EvalGapPatchResult> {
    const res = await api.post<EvalGapPatchResult>(
        `/projects/${projectId}/eval-gaps/patch/apply`,
        { signal_id: signalId },
    );
    return res.data;
}
