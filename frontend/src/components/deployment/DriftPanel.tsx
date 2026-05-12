/**
 * DriftPanel — drift verdict + history for a deployment version
 * (priority.md P30, P27).
 *
 * Shows the most recent drift check (delta vs baseline, drift badge,
 * sample counts, mode) plus a tabular history. Includes a small launcher
 * for offline drift checks: pick a gold-set id, paste predictions, hit
 * "Run drift check". Live-URL drift mode is intentionally left to the
 * CLI (P29) — the frontend launcher exists for the offline replay path.
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import EmptyState from '../shared/EmptyState';
import type {
    DeploymentDriftCheck,
    DeploymentDriftHistoryResponse,
} from '../../types/deployment';

interface Props {
    deploymentVersionId: number;
    /** Bump from the parent to force a re-fetch of drift history. */
    refreshKey?: number;
}

interface ApiErrorShape {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
}

function extractErrorMessage(err: unknown, fallback = 'Drift request failed.'): string {
    const e = err as ApiErrorShape;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    return e?.message || fallback;
}

function formatTs(value: string | null): string {
    if (!value) return '—';
    try {
        const d = new Date(value);
        if (Number.isNaN(d.getTime())) return value;
        return d.toLocaleString();
    } catch {
        return value;
    }
}

function driftBadgeClass(check: DeploymentDriftCheck): string {
    if (check.delta == null) return 'badge badge-info';
    return check.drift_detected ? 'badge badge-danger' : 'badge badge-success';
}

function driftBadgeLabel(check: DeploymentDriftCheck): string {
    if (check.delta == null) return 'no baseline';
    return check.drift_detected ? 'drift' : 'within tolerance';
}

export default function DriftPanel({
    deploymentVersionId,
    refreshKey = 0,
}: Props) {
    const [history, setHistory] = useState<DeploymentDriftCheck[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [goldSetId, setGoldSetId] = useState('');
    const [tolerance, setTolerance] = useState('0.05');
    const [predictionsText, setPredictionsText] = useState('');
    const [running, setRunning] = useState(false);

    const fetchHistory = useCallback(async () => {
        setLoading(true);
        try {
            const response = await api.get<DeploymentDriftHistoryResponse>(
                `/deployments/${deploymentVersionId}/drift/checks`,
            );
            setHistory(response.data.drift_checks || []);
            setError(null);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to load drift history.'));
        } finally {
            setLoading(false);
        }
    }, [deploymentVersionId]);

    useEffect(() => {
        void fetchHistory();
        // refreshKey is a deliberate dependency so the parent can force
        // a re-fetch after a promote/reject/rollback.
    }, [fetchHistory, refreshKey]);

    const runDriftCheck = useCallback(async () => {
        const gold = Number.parseInt(goldSetId, 10);
        if (!Number.isFinite(gold) || gold <= 0) {
            setError('Enter a gold-set id (an integer Dataset id).');
            return;
        }
        const predictionsRaw = predictionsText.trim();
        if (!predictionsRaw) {
            setError('Paste predictions JSON to run an offline drift check.');
            return;
        }
        let predictions: unknown;
        try {
            predictions = JSON.parse(predictionsRaw);
        } catch (err) {
            setError(`Predictions JSON is not valid: ${(err as Error).message}`);
            return;
        }
        const list = Array.isArray(predictions)
            ? predictions
            : (predictions as { predictions?: unknown[] })?.predictions;
        if (!Array.isArray(list)) {
            setError('Predictions must be a JSON array or {"predictions": [...]}');
            return;
        }
        setRunning(true);
        setError(null);
        try {
            await api.post(`/deployments/${deploymentVersionId}/drift/check`, {
                gold_set_id: gold,
                tolerance: Number.parseFloat(tolerance) || 0.05,
                predictions: list,
            });
            await fetchHistory();
        } catch (err) {
            setError(extractErrorMessage(err, 'Drift check failed.'));
        } finally {
            setRunning(false);
        }
    }, [
        deploymentVersionId,
        fetchHistory,
        goldSetId,
        predictionsText,
        tolerance,
    ]);

    const latest = history[0] || null;

    return (
        <div className="card deployment-drift-card">
            <div className="deployment-section-header">
                <div>
                    <h3 style={{ marginTop: 0 }}>Drift</h3>
                    <div className="dim">
                        Replays gold-eval against this deployment and compares to the
                        training-time baseline.
                    </div>
                </div>
                {latest && (
                    <span className={driftBadgeClass(latest)} aria-label="drift status">
                        {driftBadgeLabel(latest)}
                    </span>
                )}
            </div>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            {latest && (
                <div className="deployment-drift-summary">
                    <div className="kv">
                        <span className="dim">baseline</span>
                        <span>
                            {latest.baseline_pass_rate != null
                                ? latest.baseline_pass_rate.toFixed(3)
                                : '—'}
                        </span>
                    </div>
                    <div className="kv">
                        <span className="dim">current</span>
                        <span>{latest.current_pass_rate.toFixed(3)}</span>
                    </div>
                    <div className="kv">
                        <span className="dim">Δ</span>
                        <span>
                            {latest.delta != null
                                ? (latest.delta >= 0 ? '+' : '') + latest.delta.toFixed(3)
                                : '—'}
                        </span>
                    </div>
                    <div className="kv">
                        <span className="dim">tolerance</span>
                        <span>±{latest.tolerance.toFixed(2)}</span>
                    </div>
                    <div className="kv">
                        <span className="dim">samples</span>
                        <span>
                            {latest.samples_evaluated} ok / {latest.samples_failed} fail /{' '}
                            {latest.samples_skipped} skip
                        </span>
                    </div>
                    <div className="kv">
                        <span className="dim">mode</span>
                        <span>{latest.mode}</span>
                    </div>
                </div>
            )}

            <details className="deployment-drift-launcher">
                <summary>Run a drift check</summary>
                <div className="drift-launcher-form">
                    <label>
                        <span>Gold-set id</span>
                        <input
                            type="number"
                            value={goldSetId}
                            onChange={(e) => setGoldSetId(e.target.value)}
                            aria-label="Gold-set id"
                        />
                    </label>
                    <label>
                        <span>Tolerance</span>
                        <input
                            type="text"
                            value={tolerance}
                            onChange={(e) => setTolerance(e.target.value)}
                            aria-label="Tolerance"
                        />
                    </label>
                    <label className="drift-predictions">
                        <span>Predictions JSON (offline mode)</span>
                        <textarea
                            value={predictionsText}
                            onChange={(e) => setPredictionsText(e.target.value)}
                            placeholder='[{"row_id": 1, "prediction": "yes"}]'
                            aria-label="Predictions JSON"
                            rows={5}
                        />
                    </label>
                    <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        disabled={running}
                        onClick={() => void runDriftCheck()}
                    >
                        {running ? 'Running…' : 'Run drift check'}
                    </button>
                </div>
            </details>

            <div className="deployment-drift-history">
                <h4>History</h4>
                {loading && !history.length && <div className="dim">Loading drift history…</div>}
                {!loading && !history.length && (
                    <EmptyState
                        title="No drift checks yet"
                        description="A drift check re-runs your gold-set eval against the live endpoint and compares to the promote-time baseline. Run one weekly or after any infra change."
                        docsHref="http://localhost:3001/docs/deployment/drift-checks"
                    />
                )}
                {history.length > 0 && (
                    <table className="deployment-table" aria-label="Drift history">
                        <thead>
                            <tr>
                                <th>When</th>
                                <th>Δ</th>
                                <th>Verdict</th>
                                <th>Current</th>
                                <th>Baseline</th>
                                <th>Samples</th>
                                <th>Mode</th>
                            </tr>
                        </thead>
                        <tbody>
                            {history.map((row) => (
                                <tr key={row.id}>
                                    <td>{formatTs(row.created_at)}</td>
                                    <td>
                                        {row.delta != null
                                            ? (row.delta >= 0 ? '+' : '') + row.delta.toFixed(3)
                                            : '—'}
                                    </td>
                                    <td>
                                        <span className={driftBadgeClass(row)}>
                                            {driftBadgeLabel(row)}
                                        </span>
                                    </td>
                                    <td>{row.current_pass_rate.toFixed(3)}</td>
                                    <td>
                                        {row.baseline_pass_rate != null
                                            ? row.baseline_pass_rate.toFixed(3)
                                            : '—'}
                                    </td>
                                    <td>
                                        {row.samples_evaluated}/{row.samples_failed}/
                                        {row.samples_skipped}
                                    </td>
                                    <td>{row.mode}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}
