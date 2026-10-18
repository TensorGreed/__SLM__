/**
 * "Did SFT help?" lift summary panel (Theme 8 Epic 4).
 *
 * Compares the project's most recent baseline experiment (created
 * by the Quickstart 'Baseline (untrained)' tile) with its most
 * recent trained experiment, and shows per-metric lift bars +
 * gate status against `evalpack.general.default`. No new judge
 * calls — pulls from existing EvalResult rows.
 *
 * Soft-fallback when prereqs aren't met:
 *   - status='no_baseline' → "run baseline first"
 *   - status='no_trained'  → "run training first"
 *   - status='no_overlap'  → "metric keys don't overlap"
 * Each renders a small explanatory card with a CTA, not an error.
 */

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { AlertTriangle, Check, XCircle } from 'lucide-react';
import LoadingState from '../shared/LoadingState';
import './SftLiftPanel.css';

import {
    fetchSftLiftSummary,
    type SftLiftSummary,
    type SftLiftMetricRow,
    type SftLiftGateRow,
    type SftLiftGateStatus,
} from '../../api/sftLift';
import { isUnproven, metricDisplayName, shortEvidenceText } from './liftEvidence';
import { resultsPath } from '../../utils/workspaceRoutes';

interface SftLiftPanelProps {
    projectId: number;
    /** Triggers a refetch when this changes (e.g. eval just re-ran). */
    refreshToken?: unknown;
    /** Compare this run (vs its base model's baseline) instead of the
     * project's latest trained run. */
    experimentId?: number | null;
}

function errorDetail(err: unknown, fallback: string): string {
    const d = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    return typeof d === 'string' && d ? d : fallback;
}

function formatValue(v: number | null | undefined): string {
    if (v === null || v === undefined || !Number.isFinite(v)) return '—';
    return v.toFixed(2);
}

function formatRelativeDelta(pct: number | null): string {
    if (pct === null || !Number.isFinite(pct)) return 'new';
    const sign = pct > 0 ? '+' : '';
    return `${sign}${Math.round(pct)}%`;
}

function formatAbsDelta(d: number): string {
    const sign = d >= 0 ? '+' : '';
    return `${sign}${d.toFixed(2)}`;
}

function GateBadge({ status }: { status: SftLiftGateStatus }) {
    const labels: Record<SftLiftGateStatus, { label: string; tone: string; icon: ReactNode }> = {
        cleared: { label: 'pass/fail rules passed after training', tone: 'success', icon: <Check className="u-icon" aria-hidden="true" /> },
        still_failing: { label: 'still below threshold', tone: 'error', icon: <XCircle className="u-icon" aria-hidden="true" /> },
        regressed: { label: 'regressed — baseline passed, trained fails', tone: 'warning', icon: <AlertTriangle className="u-icon" aria-hidden="true" /> },
        always_passed: { label: 'already passing pre-SFT', tone: 'success', icon: <Check className="u-icon" aria-hidden="true" /> },
        incomplete: { label: 'incomplete', tone: 'info', icon: null },
    };
    const cfg = labels[status] ?? labels.incomplete;
    return (
        <span
            className={`badge badge-${cfg.tone}`}
            data-testid={`sft-lift-gate-${status}`}
        >
            {cfg.icon}{cfg.icon ? ' ' : ''}{cfg.label}
        </span>
    );
}

function directionTone(direction: string | undefined, unproven = false): string {
    if (unproven) return 'neutral';
    if (direction === 'improved') return 'good';
    if (direction === 'regressed') return 'bad';
    return 'neutral';
}

function MetricLiftRow({ row }: { row: SftLiftMetricRow }) {
    // Render a tiny bar going from baseline width → trained width
    // (both normalized 0..1 since most eval metrics are bounded).
    const baselinePct = Math.max(0, Math.min(1, row.baseline_value)) * 100;
    const trainedPct = Math.max(0, Math.min(1, row.trained_value)) * 100;
    // A delta the rows can't back up (within noise / too few rows) is shown
    // in the neutral colour, with the row split next to it.
    const evidence = row.evidence ?? null;
    const unproven = isUnproven(evidence);
    const tone = directionTone(row.direction, unproven);
    const bucket = (pct: number) => Math.round(pct / 10) * 10;
    return (
        <div
            data-testid={`sft-lift-row-${row.metric_id}`}
            className="sft-lift__row"
        >
            <div className="sft-lift__metric">
                {metricDisplayName(row.metric_id)}
            </div>
            <div>
                <div className="sft-lift__values">
                    <code data-testid={`sft-lift-baseline-${row.metric_id}`}>
                        {formatValue(row.baseline_value)}
                    </code>
                    <div aria-hidden="true" className="sft-lift__bar">
                        <div className={`sft-lift__bar-fill sft-lift__bar-fill--base sft-lift__w-${bucket(baselinePct)}`} />
                        <div className={`sft-lift__bar-fill sft-lift__bar-fill--trained sft-lift__tone--${tone} sft-lift__w-${bucket(trainedPct)}`} />
                    </div>
                    <code data-testid={`sft-lift-trained-${row.metric_id}`}>
                        {formatValue(row.trained_value)}
                    </code>
                    <span
                        data-testid={`sft-lift-delta-${row.metric_id}`}
                        className={`sft-lift__delta sft-lift__text--${tone}`}
                    >
                        {formatAbsDelta(row.absolute_delta)} (
                        {formatRelativeDelta(row.relative_delta_pct)})
                    </span>
                </div>
                {evidence && (
                    <div
                        data-testid={`sft-lift-evidence-${row.metric_id}`}
                        data-verdict={evidence.verdict}
                        className={`sft-lift__evidence${unproven ? ' sft-lift__evidence--unproven' : ''}`}
                    >
                        {shortEvidenceText(evidence)}
                    </div>
                )}
            </div>
        </div>
    );
}

function GateRow({ row }: { row: SftLiftGateRow }) {
    return (
        <div
            data-testid={`sft-lift-gate-row-${row.gate_id}`}
            className="sft-lift__gate-row"
        >
            <GateBadge status={row.status} />
            <code className="u-mono">
                {metricDisplayName(row.metric_id)}
            </code>
            <span className="u-text-secondary">
                {row.operator === 'lte' ? '≤' : '≥'} {row.threshold.toFixed(2)}
            </span>
            <span className="u-text-secondary">
                · {formatValue(row.baseline_value)} → {formatValue(row.trained_value)}
                {row.delta_to_threshold !== null && (
                    <> · {formatAbsDelta(row.delta_to_threshold)} to threshold</>
                )}
            </span>
        </div>
    );
}

export default function SftLiftPanel({
    projectId,
    refreshToken,
    experimentId,
}: SftLiftPanelProps) {
    const [summary, setSummary] = useState<SftLiftSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string>('');

    const load = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const res = await fetchSftLiftSummary(projectId, experimentId);
            setSummary(res);
        } catch (err) {
            setError(errorDetail(err, 'Failed to load lift summary.'));
        } finally {
            setLoading(false);
        }
    }, [projectId, experimentId]);

    useEffect(() => {
        load();
    }, [load, refreshToken]);

    if (loading) {
        return (
            <section
                className="card u-p-md"
                data-testid="sft-lift-panel-loading"
            >
                <LoadingState label="Computing baseline → trained lift…" lines={2} />
            </section>
        );
    }

    if (error) {
        return (
            <section
                className="card u-p-md sft-lift__error"
                role="alert"
                data-testid="sft-lift-panel-error"
            >
                {error}
            </section>
        );
    }

    // Defensive: malformed / empty responses (e.g. tests that mock
    // unrelated endpoints + leave this one returning `{}`) shouldn't
    // crash. Require the status discriminator + the array fields
    // before attempting render.
    if (
        !summary
        || !summary.status
        || !Array.isArray(summary.metric_lifts)
        || !Array.isArray(summary.gate_status)
    ) {
        return null;
    }

    if (summary.status === 'no_baseline') {
        return (
            <section
                className="card u-p-md"
                data-testid="sft-lift-panel-no-baseline"
            >
                <h4 className="u-m0">Did SFT help? — no baseline yet</h4>
                <p className="u-hint u-mt-xs u-mb0">
                    {summary.message}
                </p>
            </section>
        );
    }

    if (summary.status === 'no_trained') {
        return (
            <section
                className="card u-p-md"
                data-testid="sft-lift-panel-no-trained"
            >
                <h4 className="u-m0">Did SFT help? — no trained eval yet</h4>
                <p className="u-hint u-mt-xs u-mb0">
                    {summary.message}
                </p>
            </section>
        );
    }

    if (summary.status === 'no_overlap') {
        return (
            <section
                className="card u-p-md"
                data-testid="sft-lift-panel-no-overlap"
            >
                <h4 className="u-m0">Did SFT help? — metrics didn't overlap</h4>
                <p className="u-hint u-mt-xs u-mb0">
                    {summary.message}
                </p>
            </section>
        );
    }

    const { baseline, trained, metric_lifts, gate_status } = summary;
    const headline = metric_lifts.find((m) => m.is_headline);
    const clearedCount = gate_status.filter((g) => g.status === 'cleared').length;
    const stillFailingCount = gate_status.filter(
        (g) => g.status === 'still_failing',
    ).length;
    const regressedCount = gate_status.filter(
        (g) => g.status === 'regressed',
    ).length;

    return (
        <section
            className="card u-p-md u-stack"
            data-testid="sft-lift-panel"
        >
            <div>
                <h4 className="u-m0">Did SFT help?</h4>
                <p className="u-hint u-mt-xs u-mb0">
                    Baseline ·{' '}
                    <strong data-testid="sft-lift-baseline-name">
                        {baseline?.experiment_name}
                    </strong>{' '}
                    → trained ·{' '}
                    <strong data-testid="sft-lift-trained-name">
                        {trained?.experiment_name}
                    </strong>
                    {headline && (
                        <>
                            {' '}— headline{' '}
                            <code>{metricDisplayName(headline.metric_id)}</code>{' '}
                            <code>{formatValue(headline.baseline_value)}</code> →{' '}
                            <code>{formatValue(headline.trained_value)}</code>{' '}
                            <strong
                                data-testid="sft-lift-headline-delta"
                                className={`sft-lift__text--${directionTone(headline.direction)}`}
                            >
                                {formatAbsDelta(headline.absolute_delta)} (
                                {formatRelativeDelta(headline.relative_delta_pct)})
                            </strong>
                        </>
                    )}
                </p>
            </div>

            {trained?.experiment_id != null && (
                <a
                    href={resultsPath(projectId, 'try', `run=${trained.experiment_id}`)}
                    data-testid="sft-lift-chat-link"
                >
                    Chat with run #{trained.experiment_id} →
                </a>
            )}

            <div data-testid="sft-lift-metric-rows">
                {metric_lifts.map((row) => (
                    <MetricLiftRow key={row.metric_id} row={row} />
                ))}
            </div>

            {gate_status.length > 0 && (
                <div data-testid="sft-lift-gate-rows">
                    <h5 className="u-m0 u-mb-xs">
                        Pass/fail rules from{' '}
                        <code>{summary.eval_pack_id || 'evalpack.general.default'}</code>
                        {' '}·{' '}
                        <span data-testid="sft-lift-gate-summary">
                            {clearedCount} cleared, {stillFailingCount} still failing
                            {regressedCount > 0 && `, ${regressedCount} regressed`}
                        </span>
                    </h5>
                    {gate_status.map((row) => (
                        <GateRow key={row.gate_id} row={row} />
                    ))}
                </div>
            )}
        </section>
    );
}
