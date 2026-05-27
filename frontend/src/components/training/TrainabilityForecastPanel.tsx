/**
 * TrainabilityForecastPanel — USER-SUCCESS Epic 1.
 *
 * Renders the pre-training forecast for a project: an overall
 * verdict badge, a confidence-band headline, and a per-signal list
 * with one-click suggested actions. Mounted above the Run Preflight
 * button on the Training Config page so the user sees the forecast
 * *before* committing to a training run.
 *
 * Advisory only — the panel never blocks training. When the forecast
 * is amber/red the Train button's label shifts to "Train anyway"
 * elsewhere in the surface; this panel only reports.
 */

import { useCallback, useEffect, useState } from 'react';

import type {
    ForecastResult,
    ForecastSeverity,
    ForecastSignal,
    ForecastSnapshot,
    ForecastVerdict,
    SuggestedActionKind,
} from '../../api/trainabilityForecast';
import {
    fetchTrainingForecast,
    fetchTrainingForecastHistory,
} from '../../api/trainabilityForecast';
import TrainAnywayButton from './TrainAnywayButton';
import './TrainabilityForecastPanel.css';

interface Props {
    projectId: number;
    /** Optional callback fired when a suggested-action button is clicked. */
    onActionClicked?: (kind: SuggestedActionKind, params: Record<string, unknown>) => void;
}

const VERDICT_LABELS: Record<ForecastResult['overall'], { label: string; tone: string }> = {
    likely_pass: { label: 'Likely to pass gates', tone: 'ok' },
    borderline: { label: 'Borderline — could go either way', tone: 'warn' },
    likely_fail: { label: 'Likely to fall short of gates', tone: 'block' },
};

const SEVERITY_ICON: Record<ForecastSeverity, string> = {
    ok: '✓',
    warn: '!',
    block: '✕',
};

const SUGGESTED_ACTION_LABEL: Record<SuggestedActionKind, string> = {
    synth_augment: 'Generate more training rows',
    synth_balance: 'Balance class distribution',
    synth_diversify: 'Diversify gold set',
    fix_gold_rows: 'Fix invalid gold rows',
};

export default function TrainabilityForecastPanel({ projectId, onActionClicked }: Props) {
    const [result, setResult] = useState<ForecastResult | null>(null);
    const [history, setHistory] = useState<ForecastSnapshot[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [refreshing, setRefreshing] = useState(false);

    const load = useCallback(
        async (force = false) => {
            setError(null);
            if (force) {
                setRefreshing(true);
            } else {
                setLoading(true);
            }
            try {
                const data = await fetchTrainingForecast(projectId, { refresh: force });
                setResult(data);
                // History is best-effort — the panel still renders the
                // live forecast even if the snapshot endpoint hiccups.
                // Fire after the live read so a slow history fetch
                // doesn't block the verdict badge from appearing.
                try {
                    const histResp = await fetchTrainingForecastHistory(projectId, { limit: 10 });
                    // Defensive: tests + older backends may not return
                    // the snapshot list. We treat anything that isn't
                    // an array as "no history yet" — same UX as a
                    // freshly-instantiated project.
                    setHistory(Array.isArray(histResp?.snapshots) ? histResp.snapshots : []);
                } catch {
                    setHistory([]);
                }
            } catch (err: any) {
                // 400 (no recipe) is expected during the first dataset import — keep the
                // panel quiet rather than alarming. Other errors surface.
                const status = err?.response?.status;
                if (status === 400) {
                    setResult(null);
                    setError(null);
                } else {
                    setError(err?.response?.data?.detail || err?.message || 'Failed to load forecast');
                }
            } finally {
                setLoading(false);
                setRefreshing(false);
            }
        },
        [projectId],
    );

    useEffect(() => {
        load(false);
    }, [load]);

    if (loading && !result) {
        return (
            <section className="trainability-forecast" aria-busy="true" data-testid="trainability-forecast-loading">
                <p className="trainability-forecast__loading">Computing trainability forecast…</p>
            </section>
        );
    }

    if (error) {
        return (
            <section className="trainability-forecast trainability-forecast--error" data-testid="trainability-forecast-error">
                <p className="trainability-forecast__error">{error}</p>
                <button
                    type="button"
                    className="trainability-forecast__retry"
                    onClick={() => load(true)}
                >
                    Retry
                </button>
            </section>
        );
    }

    if (!result) {
        // Quiet skip — recipe not selected yet, or forecast unavailable.
        return null;
    }

    const verdict = VERDICT_LABELS[result.overall];

    return (
        <section
            className={`trainability-forecast trainability-forecast--${verdict.tone}`}
            aria-label="Trainability forecast"
            data-testid="trainability-forecast"
        >
            <header className="trainability-forecast__head">
                <div className="trainability-forecast__head-main">
                    <span className={`trainability-forecast__verdict-badge trainability-forecast__verdict-badge--${verdict.tone}`}>
                        {verdict.label}
                    </span>
                    <span className="trainability-forecast__confidence">
                        Predicted gate-pass: <strong>~{result.confidence_pct}%</strong>
                    </span>
                </div>
                <button
                    type="button"
                    className="trainability-forecast__refresh"
                    onClick={() => load(true)}
                    disabled={refreshing}
                    aria-label="Refresh forecast"
                >
                    {refreshing ? 'Refreshing…' : 'Refresh'}
                </button>
            </header>
            <ForecastHistoryStrip history={history} />
            <ul className="trainability-forecast__signals">
                {result.signals.map((signal) => (
                    <SignalRow
                        key={signal.id}
                        signal={signal}
                        onActionClicked={onActionClicked}
                    />
                ))}
            </ul>
            <TrainAnywayButton
                verdict={result.overall}
                confidencePct={result.confidence_pct}
            />
            {result.cache_hit && (
                <p className="trainability-forecast__cache-note">
                    Cached result — click Refresh to recompute.
                </p>
            )}
        </section>
    );
}

interface SignalRowProps {
    signal: ForecastSignal;
    onActionClicked?: (kind: SuggestedActionKind, params: Record<string, unknown>) => void;
}

/** Sparkline + last-3 verdict-delta strip rendered above the signal
 *  list (T2). Hidden when there's fewer than 2 snapshots — a single
 *  point isn't a trend. The sparkline uses confidence_pct on the y-axis;
 *  each point's tooltip shows the verdict + signal severities at that
 *  snapshot so the user can inspect what changed without leaving the
 *  page. Snapshots arrive newest-first from the API; we reverse for
 *  the chart so time reads left-to-right.
 */
function ForecastHistoryStrip({ history }: { history: ForecastSnapshot[] }) {
    if (history.length < 2) return null;

    const chronological = [...history].reverse();
    const width = 220;
    const height = 36;
    const padding = 4;
    const innerW = width - 2 * padding;
    const innerH = height - 2 * padding;

    // y-axis is fixed [0, 100] so the line shape is comparable across
    // sessions — a sparkline that auto-fits its y-range would tell you
    // the wrong story ("went up 2%" vs "went up 30%" look the same).
    const xFor = (idx: number) =>
        padding + (chronological.length === 1
            ? innerW / 2
            : (idx / (chronological.length - 1)) * innerW);
    const yFor = (pct: number) => padding + innerH - (pct / 100) * innerH;

    const linePoints = chronological
        .map((snap, idx) => `${xFor(idx).toFixed(1)},${yFor(snap.confidence_pct).toFixed(1)}`)
        .join(' ');

    // Verdict-delta strip: render the last three snapshots' deltas
    // newest-first ("now ← prior ← prior-1"). Each chip shows the
    // confidence delta + an arrow capturing direction.
    const recent = history.slice(0, 4); // up to 3 deltas across 4 points
    const deltas = recent
        .slice(0, recent.length - 1)
        .map((curr, idx) => {
            const prev = recent[idx + 1];
            return {
                key: `${curr.id}-${prev.id}`,
                delta: curr.confidence_pct - prev.confidence_pct,
                verdict: curr.overall as ForecastVerdict,
                priorVerdict: prev.overall as ForecastVerdict,
            };
        });

    return (
        <div
            className="trainability-forecast__history"
            data-testid="trainability-forecast-history"
        >
            <svg
                className="trainability-forecast__sparkline"
                viewBox={`0 0 ${width} ${height}`}
                width={width}
                height={height}
                role="img"
                aria-label={`Confidence trend over the last ${chronological.length} snapshots`}
                data-testid="trainability-forecast-sparkline"
            >
                <polyline
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    points={linePoints}
                />
                {chronological.map((snap, idx) => (
                    <circle
                        key={snap.id}
                        cx={xFor(idx)}
                        cy={yFor(snap.confidence_pct)}
                        r={idx === chronological.length - 1 ? 3 : 2}
                        className={`trainability-forecast__sparkline-dot trainability-forecast__sparkline-dot--${snap.overall}`}
                        data-testid={`trainability-forecast-sparkline-dot-${idx}`}
                    >
                        <title>
                            {`${new Date(snap.computed_at).toLocaleString()} · `}
                            {`${snap.overall.replace('_', ' ')} @ ${snap.confidence_pct}%`}
                            {snap.signals.length > 0
                                ? '\n' + snap.signals
                                    .map((s) => `${s.severity}: ${s.id}`)
                                    .join('\n')
                                : ''}
                        </title>
                    </circle>
                ))}
            </svg>
            <ul
                className="trainability-forecast__deltas"
                data-testid="trainability-forecast-deltas"
            >
                {deltas.map((d) => {
                    const arrow = d.delta > 0 ? '▲' : d.delta < 0 ? '▼' : '·';
                    const tone = d.delta > 0 ? 'up' : d.delta < 0 ? 'down' : 'flat';
                    return (
                        <li
                            key={d.key}
                            className={`trainability-forecast__delta trainability-forecast__delta--${tone}`}
                            data-testid="trainability-forecast-delta-chip"
                        >
                            <span className="trainability-forecast__delta-arrow" aria-hidden="true">
                                {arrow}
                            </span>
                            <span>
                                {d.delta > 0 ? '+' : ''}
                                {d.delta}%
                            </span>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}

function SignalRow({ signal, onActionClicked }: SignalRowProps) {
    const action = signal.suggested_action;
    return (
        <li
            className={`trainability-forecast__signal trainability-forecast__signal--${signal.severity}`}
            data-testid={`trainability-forecast-signal-${signal.id}`}
            data-severity={signal.severity}
        >
            <span
                className={`trainability-forecast__signal-icon trainability-forecast__signal-icon--${signal.severity}`}
                aria-hidden="true"
            >
                {SEVERITY_ICON[signal.severity]}
            </span>
            <div className="trainability-forecast__signal-body">
                <p className="trainability-forecast__signal-headline">{signal.headline}</p>
                {signal.detail && (
                    <p className="trainability-forecast__signal-detail">{signal.detail}</p>
                )}
            </div>
            {action && onActionClicked && (
                <button
                    type="button"
                    className="trainability-forecast__action"
                    onClick={() => onActionClicked(action.kind, action.params)}
                >
                    {SUGGESTED_ACTION_LABEL[action.kind] || 'Take action'}
                </button>
            )}
        </li>
    );
}
