/**
 * DeployabilityScoreCard — readiness report (priority.md P30, P28).
 *
 * Renders the headline 0..1 score plus per-component provenance / score
 * breakdown so an operator can see *why* the score is where it is.
 *
 * - Overall score with provenance + confidence badge.
 * - Per-component cards: name, score (or "no signal"), weight,
 *   provenance, signals list, free-text summary.
 * - "Recompute" button POSTs to /deployments/{id}/score/compute.
 * - Surfaces score_not_found (404) as a clean empty state with a
 *   one-click "Compute now" call-to-action.
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import type { DeploymentScore } from '../../types/deployment';

interface Props {
    deploymentVersionId: number;
    /**
     * Bump from the parent to force a re-fetch even when
     * ``deploymentVersionId`` is unchanged — e.g. after a promote, the
     * same dv is now live and its score should be recomputed.
     */
    refreshKey?: number;
}

interface ApiErrorShape {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
}

function extractErrorMessage(err: unknown, fallback = 'Score request failed.'): string {
    const e = err as ApiErrorShape;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    return e?.message || fallback;
}

function provenanceBadgeClass(provenance: string): string {
    if (provenance === 'measured') return 'badge badge-success';
    if (provenance === 'mixed') return 'badge badge-warning';
    return 'badge badge-info';
}

function bandBadgeClass(band: string): string {
    if (band === 'high') return 'badge badge-success';
    if (band === 'medium') return 'badge badge-warning';
    return 'badge badge-danger';
}

function formatScore(score: number | null): string {
    if (score == null || Number.isNaN(score)) return '—';
    return score.toFixed(2);
}

export default function DeployabilityScoreCard({
    deploymentVersionId,
    refreshKey = 0,
}: Props) {
    const [score, setScore] = useState<DeploymentScore | null>(null);
    const [loading, setLoading] = useState(false);
    const [computing, setComputing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [notFound, setNotFound] = useState(false);

    const fetchLatest = useCallback(async () => {
        setLoading(true);
        setError(null);
        setNotFound(false);
        try {
            const response = await api.get<DeploymentScore>(
                `/deployments/${deploymentVersionId}/score`,
            );
            setScore(response.data);
        } catch (err) {
            const e = err as ApiErrorShape;
            if (e?.response?.status === 404) {
                setNotFound(true);
                setScore(null);
                return;
            }
            setError(extractErrorMessage(err, 'Failed to load score.'));
        } finally {
            setLoading(false);
        }
    }, [deploymentVersionId]);

    const computeNow = useCallback(async () => {
        setComputing(true);
        setError(null);
        try {
            const response = await api.post<DeploymentScore>(
                `/deployments/${deploymentVersionId}/score/compute`,
                {},
            );
            setScore(response.data);
            setNotFound(false);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to compute score.'));
        } finally {
            setComputing(false);
        }
    }, [deploymentVersionId]);

    useEffect(() => {
        void fetchLatest();
        // refreshKey is a deliberate dependency so the parent can force
        // a re-fetch after a promote / reject / rollback even when the
        // deployment id hasn't changed.
    }, [fetchLatest, refreshKey]);

    if (loading && !score) {
        return (
            <div className="card deployment-score-card">
                <div className="dim">Loading deployability score…</div>
            </div>
        );
    }

    if (notFound) {
        return (
            <div className="card deployment-score-card">
                <div className="deployment-score-empty">
                    <div className="dim">
                        No score has been computed for this deployment yet.
                    </div>
                    <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        disabled={computing}
                        onClick={() => void computeNow()}
                    >
                        {computing ? 'Computing…' : 'Compute now'}
                    </button>
                </div>
                {error && (
                    <div className="deployment-status is-error" role="alert">
                        {error}
                    </div>
                )}
            </div>
        );
    }

    if (!score) {
        return (
            <div className="card deployment-score-card">
                {error && (
                    <div className="deployment-status is-error" role="alert">
                        {error}
                    </div>
                )}
            </div>
        );
    }

    const overall = formatScore(score.overall_score);
    return (
        <div className="card deployment-score-card">
            <div className="deployment-score-header">
                <div>
                    <h3 style={{ marginTop: 0 }}>Deployability score</h3>
                    <div className="deployment-score-headline">
                        <span className="score-value">{overall}</span>
                        <span className={provenanceBadgeClass(score.provenance)}>
                            {score.provenance}
                        </span>
                        <span className={bandBadgeClass(score.confidence_band)}>
                            confidence: {score.confidence_band}
                        </span>
                    </div>
                </div>
                <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={computing}
                    onClick={() => void computeNow()}
                >
                    {computing ? 'Recomputing…' : 'Recompute'}
                </button>
            </div>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            <div className="deployment-score-components">
                {score.components.map((component) => (
                    <div
                        key={component.name}
                        className={`deployment-score-component ${
                            component.score == null ? 'is-missing' : ''
                        }`}
                    >
                        <div className="component-head">
                            <span className="component-name">{component.name}</span>
                            <span className={provenanceBadgeClass(component.provenance)}>
                                {component.provenance}
                            </span>
                            <span className="component-score">
                                {component.score == null
                                    ? 'no signal'
                                    : formatScore(component.score)}
                            </span>
                        </div>
                        <div className="component-summary">{component.summary}</div>
                        {component.signals.length > 0 && (
                            <ul className="component-signals">
                                {component.signals.map((signal) => (
                                    <li
                                        key={signal.key}
                                        className={signal.ok ? 'is-ok' : 'is-warn'}
                                    >
                                        <code>{signal.key}</code>
                                        <span> = </span>
                                        <code>{String(signal.value)}</code>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
