/**
 * CheckpointsPanel — P20 Checkpoints side panel.
 *
 * Wraps three Wave D backend surfaces in one operator-friendly card:
 *   - GET .../runs/{id}/checkpoints (P16 list)
 *   - POST .../runs/{id}/checkpoints/{step}/promote (P16 promotion)
 *   - POST .../runs/{id}/resume-from/{step} (P16 fork from a step)
 *   - POST .../runs/{id}/pause + .../resume (P17)
 *
 * Pause/Resume buttons live at the top — visible only when the run's
 * status makes them actionable (RUNNING → Pause; PAUSED → Resume).
 *
 * The list re-fetches after every successful mutation so the operator
 * sees `is_best` flip / `promoted_at` stamp / new fork experiment id
 * without a manual refresh.
 *
 * Props are minimal — projectId + experiment summary + an optional
 * ``onForked`` callback the parent can use to navigate to the new
 * forked experiment after a resume-from-step click.
 */

import { useCallback, useEffect, useState } from 'react';
import api from '../../api/client';

interface ExperimentSummary {
    id: number;
    name: string;
    status: string;
}

interface CheckpointRow {
    id: number;
    run_id: number;
    experiment_id: number;
    step: number;
    epoch: number;
    loss: number | null;
    train_loss: number | null;
    eval_loss: number | null;
    metrics_blob: Record<string, unknown>;
    path: string;
    is_best: boolean;
    promoted_at: string | null;
    created_at: string | null;
}

interface CheckpointsResponse {
    project_id: number;
    experiment_id: number;
    run_status: string;
    count: number;
    checkpoints: CheckpointRow[];
}

interface CheckpointsPanelProps {
    projectId: number;
    experiment: ExperimentSummary;
    /** Called after a successful resume-from-step fork with the new run id. */
    onForked?: (newExperimentId: number) => void;
    /** Called after a successful pause/resume to let the parent refresh state. */
    onLifecycleChange?: () => void;
}

function errorDetail(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

function formatLoss(value: number | null): string {
    if (value === null || !Number.isFinite(value)) return '—';
    return value.toFixed(4);
}

function formatTimestamp(value: string | null): string {
    if (!value) return '—';
    const parsed = Date.parse(value);
    if (!Number.isFinite(parsed)) return value;
    const date = new Date(parsed);
    return date.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}

export default function CheckpointsPanel({
    projectId,
    experiment,
    onForked,
    onLifecycleChange,
}: CheckpointsPanelProps) {
    const [data, setData] = useState<CheckpointsResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [actionInFlight, setActionInFlight] = useState<string | null>(null);

    const fetchList = useCallback(async () => {
        setIsLoading(true);
        setErrorMessage(null);
        try {
            const response = await api.get<CheckpointsResponse>(
                `/projects/${projectId}/training/runs/${experiment.id}/checkpoints`,
            );
            setData(response.data);
        } catch (err) {
            setErrorMessage(errorDetail(err, 'Failed to load checkpoints.'));
        } finally {
            setIsLoading(false);
        }
    }, [projectId, experiment.id]);

    useEffect(() => {
        void fetchList();
    }, [fetchList]);

    const handlePause = useCallback(async () => {
        setActionInFlight('pause');
        setErrorMessage(null);
        try {
            await api.post(
                `/projects/${projectId}/training/runs/${experiment.id}/pause`,
            );
            onLifecycleChange?.();
            await fetchList();
        } catch (err) {
            setErrorMessage(errorDetail(err, 'Pause request failed.'));
        } finally {
            setActionInFlight(null);
        }
    }, [projectId, experiment.id, onLifecycleChange, fetchList]);

    const handleResume = useCallback(async () => {
        setActionInFlight('resume');
        setErrorMessage(null);
        try {
            await api.post(
                `/projects/${projectId}/training/runs/${experiment.id}/resume`,
            );
            onLifecycleChange?.();
            await fetchList();
        } catch (err) {
            setErrorMessage(errorDetail(err, 'Resume request failed.'));
        } finally {
            setActionInFlight(null);
        }
    }, [projectId, experiment.id, onLifecycleChange, fetchList]);

    const handlePromote = useCallback(
        async (step: number) => {
            setActionInFlight(`promote-${step}`);
            setErrorMessage(null);
            try {
                await api.post(
                    `/projects/${projectId}/training/runs/${experiment.id}/checkpoints/${step}/promote`,
                );
                await fetchList();
            } catch (err) {
                setErrorMessage(errorDetail(err, `Promote step ${step} failed.`));
            } finally {
                setActionInFlight(null);
            }
        },
        [projectId, experiment.id, fetchList],
    );

    const handleResumeFrom = useCallback(
        async (step: number) => {
            setActionInFlight(`resume-from-${step}`);
            setErrorMessage(null);
            try {
                const response = await api.post<{ experiment_id?: number }>(
                    `/projects/${projectId}/training/runs/${experiment.id}/resume-from/${step}`,
                    {},
                );
                const newId = Number(response.data?.experiment_id);
                if (Number.isFinite(newId) && newId > 0 && onForked) {
                    onForked(newId);
                }
                await fetchList();
            } catch (err) {
                setErrorMessage(errorDetail(err, `Fork from step ${step} failed.`));
            } finally {
                setActionInFlight(null);
            }
        },
        [projectId, experiment.id, fetchList, onForked],
    );

    const status = experiment.status?.toLowerCase() ?? '';
    const canPause = status === 'running';
    const canResume = status === 'paused';
    const checkpoints = data?.checkpoints ?? [];

    return (
        <div className="card checkpoints-panel">
            <div className="checkpoints-panel__head">
                <h4 className="checkpoints-panel__title">Checkpoints</h4>
                <div className="checkpoints-panel__actions">
                    {canPause && (
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => void handlePause()}
                            disabled={actionInFlight === 'pause'}
                        >
                            {actionInFlight === 'pause' ? 'Pausing…' : 'Pause'}
                        </button>
                    )}
                    {canResume && (
                        <button
                            className="btn btn-primary btn-sm"
                            onClick={() => void handleResume()}
                            disabled={actionInFlight === 'resume'}
                        >
                            {actionInFlight === 'resume' ? 'Resuming…' : 'Resume'}
                        </button>
                    )}
                    <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => void fetchList()}
                        disabled={isLoading}
                    >
                        {isLoading ? 'Refreshing…' : 'Refresh'}
                    </button>
                </div>
            </div>

            {errorMessage && (
                <div className="checkpoints-panel__error">{errorMessage}</div>
            )}

            <div className="checkpoints-panel__meta">
                <span>Run status: <strong>{data?.run_status ?? status ?? '—'}</strong></span>
                <span>Checkpoints: <strong>{checkpoints.length}</strong></span>
            </div>

            {checkpoints.length === 0 ? (
                <div className="checkpoints-panel__empty">
                    No checkpoints yet. They'll appear here as training writes them.
                </div>
            ) : (
                <table className="checkpoints-panel__table">
                    <thead>
                        <tr>
                            <th>Step</th>
                            <th>Epoch</th>
                            <th>Loss</th>
                            <th>Best</th>
                            <th>Promoted</th>
                            <th>Path</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {checkpoints.map((row) => (
                            <tr key={row.id} className={row.is_best ? 'is-best' : ''}>
                                <td>{row.step}</td>
                                <td>{row.epoch}</td>
                                <td>{formatLoss(row.loss)}</td>
                                <td>
                                    {row.is_best ? (
                                        <span className="badge badge-success">best</span>
                                    ) : (
                                        '—'
                                    )}
                                </td>
                                <td>{formatTimestamp(row.promoted_at)}</td>
                                <td className="checkpoints-panel__path" title={row.path}>
                                    {row.path}
                                </td>
                                <td>
                                    <div className="checkpoints-panel__row-actions">
                                        <button
                                            className="btn btn-secondary btn-xs"
                                            onClick={() => void handlePromote(row.step)}
                                            disabled={actionInFlight === `promote-${row.step}`}
                                        >
                                            {actionInFlight === `promote-${row.step}`
                                                ? 'Promoting…'
                                                : 'Promote'}
                                        </button>
                                        <button
                                            className="btn btn-secondary btn-xs"
                                            onClick={() => void handleResumeFrom(row.step)}
                                            disabled={actionInFlight === `resume-from-${row.step}`}
                                        >
                                            {actionInFlight === `resume-from-${row.step}`
                                                ? 'Forking…'
                                                : 'Resume from…'}
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}
