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

import { useCallback, useEffect, useState } from 'react';

import {
    fetchSftLiftSummary,
    type SftLiftSummary,
    type SftLiftMetricRow,
    type SftLiftGateRow,
    type SftLiftGateStatus,
} from '../../api/sftLift';

interface SftLiftPanelProps {
    projectId: number;
    /** Triggers a refetch when this changes (e.g. eval just re-ran). */
    refreshToken?: unknown;
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
    const labels: Record<SftLiftGateStatus, { label: string; tone: string }> = {
        cleared: { label: '✓ gate cleared by training', tone: 'success' },
        still_failing: { label: '❌ still below threshold', tone: 'error' },
        regressed: { label: '⚠ regressed — baseline passed, trained fails', tone: 'warning' },
        always_passed: { label: '✓ already passing pre-SFT', tone: 'success' },
        incomplete: { label: 'incomplete', tone: 'info' },
    };
    const cfg = labels[status] ?? labels.incomplete;
    return (
        <span
            className={`badge badge-${cfg.tone}`}
            data-testid={`sft-lift-gate-${status}`}
        >
            {cfg.label}
        </span>
    );
}

function MetricLiftRow({ row }: { row: SftLiftMetricRow }) {
    // Render a tiny bar going from baseline width → trained width
    // (both normalized 0..1 since most eval metrics are bounded).
    const baselinePct = Math.max(0, Math.min(1, row.baseline_value)) * 100;
    const trainedPct = Math.max(0, Math.min(1, row.trained_value)) * 100;
    const directionColor =
        row.direction === 'improved'
            ? 'var(--color-success)'
            : row.direction === 'regressed'
                ? 'var(--color-error)'
                : 'var(--text-secondary)';
    return (
        <div
            data-testid={`sft-lift-row-${row.metric_id}`}
            style={{
                display: 'grid',
                gridTemplateColumns: 'minmax(110px, auto) 1fr',
                gap: 'var(--space-md)',
                alignItems: 'center',
                padding: 'var(--space-xs) 0',
            }}
        >
            <div
                style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                }}
            >
                {row.metric_id}
            </div>
            <div>
                <div
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 'var(--space-sm)',
                        fontSize: '0.85rem',
                    }}
                >
                    <code data-testid={`sft-lift-baseline-${row.metric_id}`}>
                        {formatValue(row.baseline_value)}
                    </code>
                    <div
                        aria-hidden="true"
                        style={{
                            position: 'relative',
                            flex: 1,
                            height: 8,
                            borderRadius: 4,
                            background: 'var(--bg-subtle)',
                            overflow: 'hidden',
                        }}
                    >
                        <div
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: `${baselinePct}%`,
                                height: '100%',
                                background: 'var(--gray-400)',
                            }}
                        />
                        <div
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: `${trainedPct}%`,
                                height: '100%',
                                background: directionColor,
                                opacity: 0.6,
                                mixBlendMode: 'multiply',
                            }}
                        />
                    </div>
                    <code data-testid={`sft-lift-trained-${row.metric_id}`}>
                        {formatValue(row.trained_value)}
                    </code>
                    <span
                        data-testid={`sft-lift-delta-${row.metric_id}`}
                        style={{
                            color: directionColor,
                            fontSize: '0.85rem',
                            fontWeight: 600,
                            minWidth: 110,
                            textAlign: 'right',
                        }}
                    >
                        {formatAbsDelta(row.absolute_delta)} (
                        {formatRelativeDelta(row.relative_delta_pct)})
                    </span>
                </div>
            </div>
        </div>
    );
}

function GateRow({ row }: { row: SftLiftGateRow }) {
    return (
        <div
            data-testid={`sft-lift-gate-row-${row.gate_id}`}
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-sm)',
                padding: 'var(--space-xs) 0',
                fontSize: '0.85rem',
            }}
        >
            <GateBadge status={row.status} />
            <code style={{ fontFamily: 'var(--font-mono)' }}>
                {row.metric_id}
            </code>
            <span style={{ color: 'var(--text-secondary)' }}>
                {row.operator === 'lte' ? '≤' : '≥'} {row.threshold.toFixed(2)}
            </span>
            <span style={{ color: 'var(--text-secondary)' }}>
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
}: SftLiftPanelProps) {
    const [summary, setSummary] = useState<SftLiftSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string>('');

    const load = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const res = await fetchSftLiftSummary(projectId);
            setSummary(res);
        } catch (err) {
            setError(errorDetail(err, 'Failed to load lift summary.'));
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        load();
    }, [load, refreshToken]);

    if (loading) {
        return (
            <section
                className="card"
                data-testid="sft-lift-panel-loading"
                style={{ padding: 'var(--space-md)' }}
            >
                <div style={{ color: 'var(--text-secondary)' }}>
                    Computing baseline → trained lift…
                </div>
            </section>
        );
    }

    if (error) {
        return (
            <section
                className="card"
                role="alert"
                data-testid="sft-lift-panel-error"
                style={{
                    padding: 'var(--space-md)',
                    background: 'var(--color-error-bg)',
                    color: 'var(--color-error)',
                }}
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
                className="card"
                data-testid="sft-lift-panel-no-baseline"
                style={{ padding: 'var(--space-md)' }}
            >
                <h4 style={{ margin: 0 }}>Did SFT help? — no baseline yet</h4>
                <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0' }}>
                    {summary.message}
                </p>
            </section>
        );
    }

    if (summary.status === 'no_trained') {
        return (
            <section
                className="card"
                data-testid="sft-lift-panel-no-trained"
                style={{ padding: 'var(--space-md)' }}
            >
                <h4 style={{ margin: 0 }}>Did SFT help? — no trained eval yet</h4>
                <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0' }}>
                    {summary.message}
                </p>
            </section>
        );
    }

    if (summary.status === 'no_overlap') {
        return (
            <section
                className="card"
                data-testid="sft-lift-panel-no-overlap"
                style={{ padding: 'var(--space-md)' }}
            >
                <h4 style={{ margin: 0 }}>Did SFT help? — metrics didn't overlap</h4>
                <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0' }}>
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
            className="card"
            data-testid="sft-lift-panel"
            style={{
                padding: 'var(--space-md)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-md)',
            }}
        >
            <div>
                <h4 style={{ margin: 0 }}>Did SFT help?</h4>
                <p
                    style={{
                        margin: '4px 0 0',
                        color: 'var(--text-secondary)',
                        fontSize: '0.85rem',
                    }}
                >
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
                            <code>{headline.metric_id}</code>{' '}
                            <code>{formatValue(headline.baseline_value)}</code> →{' '}
                            <code>{formatValue(headline.trained_value)}</code>{' '}
                            <strong
                                data-testid="sft-lift-headline-delta"
                                style={{
                                    color:
                                        headline.direction === 'improved'
                                            ? 'var(--color-success)'
                                            : headline.direction === 'regressed'
                                                ? 'var(--color-error)'
                                                : 'var(--text-secondary)',
                                }}
                            >
                                {formatAbsDelta(headline.absolute_delta)} (
                                {formatRelativeDelta(headline.relative_delta_pct)})
                            </strong>
                        </>
                    )}
                </p>
            </div>

            <div data-testid="sft-lift-metric-rows">
                {metric_lifts.map((row) => (
                    <MetricLiftRow key={row.metric_id} row={row} />
                ))}
            </div>

            {gate_status.length > 0 && (
                <div data-testid="sft-lift-gate-rows">
                    <h5 style={{ margin: '0 0 var(--space-xs)' }}>
                        Gate status against{' '}
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
