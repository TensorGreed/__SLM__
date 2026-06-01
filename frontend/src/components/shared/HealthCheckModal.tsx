/**
 * HealthCheckModal — Diagnostics Intervention C.
 *
 * Renders the smoke-test result as a checklist. On open, fires the
 * smoke-test POST + shows a spinner; on result, renders one row per
 * check with an ok/warn/fail/skip glyph + the check's message +
 * (collapsed by default) the ErrorPanel for any failures.
 *
 * The point: a user wondering "is anything broken?" gets a definitive
 * answer in <3 seconds, with actionable next steps for every red row.
 */

import { useCallback, useEffect, useState } from 'react';

import { parseErrorEnvelope } from '../../api/errors';
import type { ErrorEnvelope } from '../../api/errors';
import type { SmokeCheckResult, SmokeStatus, SmokeTestSummary } from '../../api/smokeTest';
import { runSmokeTest } from '../../api/smokeTest';
import ErrorPanel from './ErrorPanel';
import './HealthCheckModal.css';


interface HealthCheckModalProps {
    projectId: number;
    onClose: () => void;
}


// Display labels for the check ids. Keep the keys aligned with the
// backend's ``SmokeCheckResult.name`` strings; new checks added on
// the backend show up with their raw name until a label is added here.
const CHECK_LABELS: Record<string, string> = {
    project_exists: 'Project accessible',
    recipe_applied: 'Recipe applied',
    gold_set: 'Gold set seeded',
    data_health: 'Data Health Report',
    trainability_forecast: 'Trainability forecast',
    synth_catalog: 'Synth playbook catalog',
    synth_backend: 'Synth backend reachable',
    prepared_splits: 'Labelled corpus',
    experiments_accessible: 'Experiments table',
};


const STATUS_GLYPH: Record<SmokeStatus, string> = {
    ok: '✓',
    warn: '⚠',
    fail: '✗',
    skip: '·',
};


const OVERALL_HEADLINE: Record<SmokeStatus, string> = {
    ok: 'All clear.',
    warn: 'Working with warnings.',
    fail: 'Blocking issues detected.',
    skip: 'Inconclusive.',
};


export default function HealthCheckModal({ projectId, onClose }: HealthCheckModalProps) {
    const [summary, setSummary] = useState<SmokeTestSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<ErrorEnvelope | null>(null);

    const fetchSummary = useCallback(async () => {
        setLoading(true);
        setLoadError(null);
        setSummary(null);
        try {
            const result = await runSmokeTest(projectId);
            setSummary(result);
        } catch (err) {
            setLoadError(parseErrorEnvelope(err));
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetchSummary();
    }, [fetchSummary]);

    return (
        <div
            className="health-check-modal-backdrop"
            data-testid="health-check-modal-backdrop"
            onClick={onClose}
        >
            <div
                className="health-check-modal"
                data-testid="health-check-modal"
                role="dialog"
                aria-modal="true"
                aria-labelledby="health-check-modal-title"
                onClick={(e) => e.stopPropagation()}
            >
                <header className="health-check-modal__head">
                    <h3
                        id="health-check-modal-title"
                        className="health-check-modal__title"
                    >
                        Project health check
                    </h3>
                    <button
                        type="button"
                        className="health-check-modal__close"
                        onClick={onClose}
                        aria-label="Close health check"
                        data-testid="health-check-modal-close"
                    >
                        ×
                    </button>
                </header>

                {loading && (
                    <div
                        className="health-check-modal__loading"
                        data-testid="health-check-modal-loading"
                    >
                        <span className="health-check-modal__spinner" aria-hidden="true">…</span>
                        Running smoke checks across project surfaces (read-only,
                        ≤3s)…
                    </div>
                )}

                {!loading && loadError && (
                    <div className="health-check-modal__load-error">
                        <ErrorPanel
                            envelope={loadError}
                            testIdPrefix="health-check-modal-load-error"
                            actions={
                                <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={() => void fetchSummary()}
                                    data-testid="health-check-modal-retry"
                                >
                                    Retry
                                </button>
                            }
                        />
                    </div>
                )}

                {!loading && summary && (
                    <>
                        <Summary summary={summary} />
                        <ul
                            className="health-check-modal__checks"
                            data-testid="health-check-modal-checks"
                        >
                            {summary.checks.map((check) => (
                                <CheckRow key={check.name} check={check} />
                            ))}
                        </ul>
                    </>
                )}

                <footer className="health-check-modal__foot">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void fetchSummary()}
                        disabled={loading}
                        data-testid="health-check-modal-rerun"
                    >
                        {loading ? 'Running…' : '⟳ Re-run'}
                    </button>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={onClose}
                        data-testid="health-check-modal-done"
                    >
                        Done
                    </button>
                </footer>
            </div>
        </div>
    );
}


function Summary({ summary }: { summary: SmokeTestSummary }) {
    return (
        <div
            className={`health-check-modal__summary health-check-modal__summary--${summary.overall}`}
            data-testid="health-check-modal-summary"
            data-overall={summary.overall}
        >
            <span
                className={`health-check-modal__overall-badge health-check-modal__overall-badge--${summary.overall}`}
                data-testid="health-check-modal-overall-badge"
            >
                {STATUS_GLYPH[summary.overall]} {summary.overall.toUpperCase()}
            </span>
            <span className="health-check-modal__overall-headline">
                {OVERALL_HEADLINE[summary.overall]}
            </span>
            <span className="health-check-modal__overall-counts">
                {summary.counts.ok} ok · {summary.counts.warn} warn
                {' · '}{summary.counts.fail} fail · {summary.counts.skip} skip
                {' · '}{summary.elapsedMs}ms
            </span>
        </div>
    );
}


function CheckRow({ check }: { check: SmokeCheckResult }) {
    const label = CHECK_LABELS[check.name] || check.name;
    return (
        <li
            className={`health-check-modal__check health-check-modal__check--${check.status}`}
            data-testid={`health-check-modal-check-${check.name}`}
            data-status={check.status}
        >
            <span
                className={`health-check-modal__check-glyph health-check-modal__check-glyph--${check.status}`}
                aria-label={check.status}
            >
                {STATUS_GLYPH[check.status]}
            </span>
            <div className="health-check-modal__check-body">
                <div className="health-check-modal__check-head">
                    <span className="health-check-modal__check-label">{label}</span>
                    <span className="health-check-modal__check-elapsed">
                        {check.elapsedMs}ms
                    </span>
                </div>
                <p className="health-check-modal__check-message">{check.message}</p>
                {check.remediation && (
                    <p
                        className="health-check-modal__check-remediation"
                        data-testid={`health-check-modal-remediation-${check.name}`}
                    >
                        {check.remediation}
                    </p>
                )}
                {check.envelope && (
                    <div className="health-check-modal__check-envelope">
                        <ErrorPanel
                            envelope={check.envelope}
                            testIdPrefix={`health-check-modal-envelope-${check.name}`}
                        />
                    </div>
                )}
            </div>
        </li>
    );
}
