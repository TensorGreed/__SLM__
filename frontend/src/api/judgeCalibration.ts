/**
 * Judge calibration (V2): blind labels → per-judge trust numbers for the review-assist judges.
 * Backend: `/projects/{id}/synthetic/calibration/*` (see `judge_calibration_service.py`).
 */

import api from './client';
import type { Job } from './jobs';

export type CalibrationTier = 'human' | 'reference_model';
export type CalibrationLabelSource = CalibrationTier | 'both';
export type CalibrationLabel = 'pass' | 'fail';

export interface CalibrationSettings {
    label_source: CalibrationLabelSource;
    default: CalibrationLabelSource;
    options: CalibrationLabelSource[];
}

/** One row as a judge reads it — no verdict, consensus, review status or provenance. */
export interface BlindRow {
    row_id: number;
    kind: string;
    content: string;
    fields: Record<string, unknown> | null;
    context_truncated: boolean;
}

export interface CalibrationTierProgress {
    labelled: number;
    stale: number;
    /** Rows offered and skipped (not labels; never re-offered in this tier). */
    skipped: number;
    pass: number;
    fail: number;
    labellers: string[];
    strata: Record<string, { population: number; labelled: number; skipped: number }>;
}

export interface CalibrationProgress {
    population: number;
    tiers: Record<CalibrationTier, CalibrationTierProgress>;
}

export interface CalibrationRate {
    value: number | null;
    low: number | null;
    high: number | null;
    n: number;
    n_eff: number;
}

export interface CalibrationJudge {
    judge: string;
    n: number;
    n_eff: number;
    label_fail: number;
    label_pass: number;
    counts: { tp: number; fp: number; fn: number; tn: number; unsure_label_fail: number; unsure_label_pass: number };
    fail_recall: CalibrationRate;
    pass_precision: CalibrationRate;
    unsure_rate: number | null;
    kappa: number | null;
    insufficient: boolean;
    insufficient_reasons: string[];
    unparseable: number;
    rubric_shas: Record<string, number>;
    rubric_mixed: boolean;
    self: boolean;
    self_rows: number;
}

export interface CalibrationTierReport {
    tier: CalibrationTier;
    labelled: number;
    stale: number;
    skipped: number;
    labellers: string[];
    population: number;
    coverage: number | null;
    strata: Record<string, { population: number; labelled: number; skipped: number; skip_share: number; weight: number }>;
    /** Strata where skips are >= 20% of the rows offered; `skip_note` explains why that matters. */
    skip_warning_strata: string[];
    skip_note: string | null;
    judges: CalibrationJudge[];
}

export interface CalibrationReport {
    label_source: CalibrationLabelSource;
    tier: CalibrationLabelSource;
    min_class_labels: number;
    min_total_labels: number;
    reports: CalibrationTierReport[];
}

const base = (projectId: number) => `/projects/${projectId}/synthetic/calibration`;

export async function getCalibrationSettings(projectId: number): Promise<CalibrationSettings> {
    return (await api.get(`${base(projectId)}/settings`)).data;
}

export async function putCalibrationSettings(projectId: number, labelSource: CalibrationLabelSource): Promise<CalibrationSettings> {
    return (await api.put(`${base(projectId)}/settings`, { label_source: labelSource })).data;
}

export async function getCalibrationSample(projectId: number, n = 20, tier: CalibrationTier = 'human'): Promise<BlindRow[]> {
    const data = (await api.get(`${base(projectId)}/sample`, { params: { n, tier } })).data;
    return Array.isArray(data?.rows) ? data.rows : [];
}

export async function postCalibrationLabel(
    projectId: number, rowId: number, label: CalibrationLabel, reason?: string,
): Promise<void> {
    await api.post(`${base(projectId)}/labels`, {
        row_id: rowId, tier: 'human', label, ...(reason?.trim() ? { reason: reason.trim() } : {}),
    });
}

/** Record that this row was shown and skipped (not a label; it is not offered again). */
export async function postCalibrationSkip(projectId: number, rowId: number, reason?: string): Promise<void> {
    await api.post(`${base(projectId)}/skips`, {
        row_id: rowId, tier: 'human', ...(reason?.trim() ? { reason: reason.trim() } : {}),
    });
}

export async function getCalibrationProgress(projectId: number): Promise<CalibrationProgress> {
    return (await api.get(`${base(projectId)}/labels`)).data;
}

export async function getCalibrationReport(projectId: number, tier?: CalibrationLabelSource): Promise<CalibrationReport> {
    return (await api.get(`${base(projectId)}/report`, { params: tier ? { tier } : {} })).data;
}

export interface ReferenceLabellingArgs {
    backend: string | null;
    n?: number;
    rubricText?: string;
}

/** Start the `judge_calibration_reference` Job (a strong model labels a blind sample). */
export async function startReferenceLabelling(
    projectId: number, args: ReferenceLabellingArgs,
): Promise<Job & { budget_warning?: unknown }> {
    const resp = await api.post(`${base(projectId)}/reference-labelling`, {
        backend: args.backend,
        n: args.n ?? 20,
        ...(args.rubricText?.trim() ? { rubric_text: args.rubricText.trim() } : {}),
    }, { params: { async_job: true } });
    return resp.data;
}
