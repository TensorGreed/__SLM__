/**
 * Eval-judge calibration (Judge calibration V4): blind labels on judged eval answers → per-judge trust.
 * Backend: `/projects/{id}/eval-judge/calibration/*` (`eval_judge_calibration_service.py`). The label
 * source is the same project setting as the review-queue judges (`judgeCalibration.ts` settings calls).
 */

import api from './client';
import type { Job } from './jobs';
import type { CalibrationLabelSource, CalibrationRate, CalibrationTier } from './judgeCalibration';

/** `verdict` = the answer judge (correct / partial / wrong); `rubric` = the plan rubric judge (pass / fail). */
export type EvalJudgeScale = 'verdict' | 'rubric';

export const SCALE_LABELS: Record<EvalJudgeScale, string[]> = {
    verdict: ['correct', 'partial', 'wrong'],
    rubric: ['pass', 'fail'],
};

/** One judged answer as the labeller sees it: no verdict, score, reason or judge name. */
export interface BlindAnswer {
    content_key: string;
    scale: EvalJudgeScale;
    question: string;
    reference: string;
    prediction: string;
    /** Rubric rows: the passages the reference model also gets (same budget). */
    passages?: { id: string; text: string; source?: string }[];
    passages_truncated?: boolean;
}

export interface EvalJudgeTrust {
    status: 'calibrated_ok' | 'below_floor' | 'insufficient' | 'uncalibrated';
    judge: string | null;
    scale: EvalJudgeScale;
    tier: CalibrationTier | null;
    n: number;
    /** Exact agreement with the labels (all classes). */
    agreement: number | null;
    /** Agreement on correct vs not (the floor is checked on its upper bound). */
    strict_agreement: number | null;
    ci: [number, number] | null;
    too_strict: number | null;
    too_lenient: number | null;
    flags: string[];
    reasons?: string[];
    label_source: CalibrationLabelSource;
    floor: number;
    too_strict_ceiling: number;
    too_lenient_ceiling: number;
    min_labels: number;
}

export interface EvalJudgeTrustSummary {
    verdict: EvalJudgeTrust;
    rubric: EvalJudgeTrust | null;
}

export interface EvalJudgeEntry {
    judge: string;
    scale: EvalJudgeScale;
    n: number;
    n_eff: number;
    by_label: Record<string, number>;
    exact_agreement: CalibrationRate;
    strict_agreement: CalibrationRate;
    pass_precision: CalibrationRate;
    pass_recall: CalibrationRate;
    wrong_recall: CalibrationRate;
    too_strict: CalibrationRate;
    too_lenient: CalibrationRate;
    kappa: number | null;
    kappa_kind: 'linear_weighted' | 'cohen';
    insufficient: boolean;
    insufficient_reasons: string[];
    population: number;
    skipped: number;
    skip_note: string | null;
    tasks: Record<string, number>;
    self: boolean;
    self_rows: number;
}

export interface EvalJudgeTierReport {
    tier: CalibrationTier;
    scale: EvalJudgeScale;
    labelled: number;
    skipped: number;
    labellers: string[];
    population: number;
    judges: EvalJudgeEntry[];
}

export interface EvalJudgeReport {
    label_source: CalibrationLabelSource;
    tier: CalibrationLabelSource;
    scale: EvalJudgeScale;
    labels: string[];
    reports: EvalJudgeTierReport[];
    trust: Record<string, EvalJudgeTrust>;
}

export interface EvalJudgeScaleProgress {
    population: number;
    labels: string[];
    judges: { judge: string; rows: number; by_verdict: Record<string, number> }[];
    tiers: Record<CalibrationTier, { labelled: number; skipped: number; by_label: Record<string, number>; labellers: string[] }>;
}

export interface EvalJudgeProgress {
    scales: Record<EvalJudgeScale, EvalJudgeScaleProgress>;
}

const base = (projectId: number) => `/projects/${projectId}/eval-judge/calibration`;

export async function getEvalJudgeProgress(projectId: number): Promise<EvalJudgeProgress> {
    return (await api.get(`${base(projectId)}/labels`)).data;
}

export async function getEvalJudgeSample(
    projectId: number, scale: EvalJudgeScale, judge: string | null, n = 20,
): Promise<BlindAnswer[]> {
    const params: Record<string, unknown> = { scale, n, tier: 'human' };
    if (judge) params.judge = judge;
    const data = (await api.get(`${base(projectId)}/sample`, { params })).data;
    return Array.isArray(data?.rows) ? data.rows : [];
}

export async function postEvalJudgeLabel(
    projectId: number, scale: EvalJudgeScale, contentKey: string, label: string, judge: string | null, reason?: string,
): Promise<void> {
    await api.post(`${base(projectId)}/labels`, {
        content_key: contentKey, scale, tier: 'human', label,
        ...(judge ? { judge } : {}),
        ...(reason?.trim() ? { reason: reason.trim() } : {}),
    });
}

export async function postEvalJudgeSkip(projectId: number, scale: EvalJudgeScale, contentKey: string, reason?: string): Promise<void> {
    await api.post(`${base(projectId)}/skips`, {
        content_key: contentKey, scale, tier: 'human', ...(reason?.trim() ? { reason: reason.trim() } : {}),
    });
}

export async function getEvalJudgeReport(projectId: number, scale: EvalJudgeScale): Promise<EvalJudgeReport> {
    return (await api.get(`${base(projectId)}/report`, { params: { scale } })).data;
}

/** Start the `eval_judge_calibration_reference` Job (a strong model labels a blind sample). */
export async function startEvalJudgeReference(
    projectId: number, args: { backend: string | null; scale: EvalJudgeScale; judge: string | null; n?: number },
): Promise<Job & { budget_warning?: unknown }> {
    const resp = await api.post(`${base(projectId)}/reference-labelling`, {
        backend: args.backend, scale: args.scale, n: args.n ?? 20, ...(args.judge ? { judge: args.judge } : {}),
    }, { params: { async_job: true } });
    return resp.data;
}

function pct(v: number | null | undefined): string {
    return typeof v === 'number' ? `${Math.round(v * 100)}%` : 'n/a';
}

function sourceName(t: EvalJudgeTrust): string {
    if (t.tier === 'human') return 'human';
    if (t.tier === 'reference_model') return 'reference model';
    return t.label_source === 'both' ? 'human or reference model' : t.label_source === 'reference_model' ? 'reference model' : 'human';
}

/** The one-line trust text shown next to the judge (settings card and eval summary). */
export function evalJudgeTrustText(t: EvalJudgeTrust, what = 'answers'): string {
    const n = t.min_labels || 20;
    if (t.status === 'uncalibrated') return `judge not calibrated: label ${n} ${what}`;
    if (t.status === 'insufficient') {
        return `judge not calibrated yet vs ${sourceName(t)}: ${t.n} labelled, needs at least ${n} with both kinds of verdict`;
    }
    const pos = t.scale === 'rubric' ? 'pass vs fail' : 'correct vs not';
    const numbers = `agrees on ${pos} ${pct(t.strict_agreement)} (n ${t.n}; exact ${pct(t.agreement)}); `
        + `too strict ${pct(t.too_strict)}, too lenient ${pct(t.too_lenient)}`;
    if (t.status === 'below_floor') {
        const why = t.flags.map((f) => (
            f === 'too_strict' ? `too strict over the ${pct(t.too_strict_ceiling)} limit`
                : f === 'too_lenient' ? `too lenient over the ${pct(t.too_lenient_ceiling)} limit`
                    : `agreement below the ${pct(t.floor)} floor`
        )).join(', ');
        return `judge below the trust floor vs ${sourceName(t)} (${why}): ${numbers}`;
    }
    return `judge calibrated vs ${sourceName(t)}: ${numbers}`;
}

export const TRUST_BADGE: Record<EvalJudgeTrust['status'], string> = {
    calibrated_ok: 'calibrated',
    below_floor: 'below floor',
    insufficient: 'too few labels',
    uncalibrated: 'not calibrated',
};
