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
    SuggestedActionKind,
} from '../../api/trainabilityForecast';
import { fetchTrainingForecast } from '../../api/trainabilityForecast';
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
