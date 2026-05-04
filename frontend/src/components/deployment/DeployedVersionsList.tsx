/**
 * DeployedVersionsList — served-versions table for the deployment page
 * (priority.md P30).
 *
 * Renders deployment versions for a project, with status-aware row
 * actions:
 *
 * - PENDING        → Promote / Reject buttons
 * - PROMOTED       → Rollback button (one-click) — surfaces the
 *                    `no_promoted_predecessor` 409 inline so the user
 *                    knows there's nothing to roll back to.
 * - others         → action-less (read-only history rows).
 *
 * Action POSTs hit P25 endpoints:
 *   POST /deployments/{id}/{promote|reject|rollback}
 *
 * The component is fully controlled — its parent owns the version list
 * and re-fetches via `onRefresh` after a successful action so the row
 * statuses stay consistent across the header KPIs and drift/score
 * cards on the same page.
 */

import { useCallback, useState } from 'react';

import api from '../../api/client';
import type {
    DeploymentVersion,
    DeploymentVersionStatus,
} from '../../types/deployment';

interface Props {
    versions: DeploymentVersion[];
    selectedDeploymentId: number | null;
    onSelect: (deploymentId: number) => void;
    onRefresh: () => Promise<void> | void;
}

interface ApiErrorShape {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
}

function extractErrorMessage(err: unknown, fallback = 'Action failed.'): string {
    const e = err as ApiErrorShape;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    return e?.message || fallback;
}

function statusBadgeClass(status: DeploymentVersionStatus): string {
    switch (status) {
        case 'promoted':
            return 'badge badge-success';
        case 'pending':
            return 'badge badge-warning';
        case 'rejected':
            return 'badge badge-danger';
        case 'rolled_back':
            return 'badge badge-danger';
        case 'superseded':
        default:
            return 'badge badge-info';
    }
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

export default function DeployedVersionsList({
    versions,
    selectedDeploymentId,
    onSelect,
    onRefresh,
}: Props) {
    const [busyDeploymentId, setBusyDeploymentId] = useState<number | null>(null);
    const [actionError, setActionError] = useState<string | null>(null);

    const callAction = useCallback(
        async (deploymentId: number, action: 'promote' | 'reject' | 'rollback') => {
            const reason = window.prompt(
                `Optional reason for ${action} (Cancel to abort):`,
                '',
            );
            if (reason === null) return; // user cancelled
            setBusyDeploymentId(deploymentId);
            setActionError(null);
            try {
                await api.post(`/deployments/${deploymentId}/${action}`, {
                    reason: reason.trim() || undefined,
                });
                await onRefresh();
            } catch (err) {
                setActionError(extractErrorMessage(err, `${action} failed.`));
            } finally {
                setBusyDeploymentId(null);
            }
        },
        [onRefresh],
    );

    if (!versions.length) {
        return (
            <div className="deployment-empty" role="status">
                No deployment versions yet. Run a non-dry-run{' '}
                <code>deploy-as-api/execute</code> to record one.
            </div>
        );
    }

    return (
        <div className="deployed-versions">
            {actionError && (
                <div className="deployment-status is-error" role="alert">
                    {actionError}
                </div>
            )}
            <table className="deployment-table" aria-label="Deployment versions">
                <thead>
                    <tr>
                        <th>Version</th>
                        <th>Target</th>
                        <th>Status</th>
                        <th>Endpoint</th>
                        <th>Created</th>
                        <th>Actor</th>
                        <th aria-label="Row actions" />
                    </tr>
                </thead>
                <tbody>
                    {versions.map((dv) => {
                        const isSelected = dv.id === selectedDeploymentId;
                        const isBusy = busyDeploymentId === dv.id;
                        return (
                            <tr
                                key={dv.id}
                                className={isSelected ? 'is-selected' : ''}
                            >
                                <td>
                                    <button
                                        type="button"
                                        className="deployment-version-button"
                                        onClick={() => onSelect(dv.id)}
                                        aria-label={`Select deployment version ${dv.version}`}
                                    >
                                        v{dv.version} <span className="dim">#{dv.id}</span>
                                    </button>
                                </td>
                                <td>
                                    <code>{dv.target_id}</code>
                                    {dv.target_kind && (
                                        <span className="dim"> ({dv.target_kind})</span>
                                    )}
                                </td>
                                <td>
                                    <span className={statusBadgeClass(dv.status)}>
                                        {dv.status}
                                    </span>
                                </td>
                                <td>{dv.endpoint_name || '—'}</td>
                                <td>{formatTs(dv.created_at)}</td>
                                <td>{dv.actor}</td>
                                <td className="row-actions">
                                    {dv.status === 'pending' && (
                                        <>
                                            <button
                                                type="button"
                                                className="btn btn-primary btn-sm"
                                                disabled={isBusy}
                                                onClick={() => void callAction(dv.id, 'promote')}
                                            >
                                                {isBusy ? '…' : 'Promote'}
                                            </button>
                                            <button
                                                type="button"
                                                className="btn btn-secondary btn-sm"
                                                disabled={isBusy}
                                                onClick={() => void callAction(dv.id, 'reject')}
                                            >
                                                Reject
                                            </button>
                                        </>
                                    )}
                                    {dv.status === 'promoted' && (
                                        <button
                                            type="button"
                                            className="btn btn-warning btn-sm"
                                            disabled={isBusy}
                                            onClick={() => void callAction(dv.id, 'rollback')}
                                        >
                                            {isBusy ? '…' : 'Rollback'}
                                        </button>
                                    )}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
