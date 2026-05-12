/**
 * FailureClusterList — P33 cluster table + recompute (priority.md P36).
 *
 * Table of clusters ordered by failure_count DESC. Each row carries a
 * severity-style danger badge with the count, the (stage, reason_code,
 * signature) triple, last-seen timestamp, and an expandable exemplars
 * block with a "View events" deep link per exemplar (uses the parent's
 * ``onSelectRun`` to feed the EventDrilldownDrawer — exemplars are
 * RunEvents, not run_ids, so we deep-link by event_id and the parent
 * resolves the run via the per-event lookup).
 *
 * The recompute button POSTs to ``/failure-clusters/recompute`` and
 * re-fetches the list on success, surfacing the compute summary
 * (events_considered, clusters_created/updated) inline.
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import type {
    FailureCluster,
    FailureClusterListResponse,
    FailureClusterRecomputeResponse,
} from '../../types/observability';
import CommandSnippet from '../shared/CommandSnippet';
import EmptyState from '../shared/EmptyState';

interface Props {
    projectId: number;
    /**
     * Called when an operator clicks an exemplar's "View events" link.
     * Receives the exemplar's ``run_id`` so the parent can deep-link
     * straight into the per-run event drilldown drawer.
     */
    onSelectRun: (runId: string) => void;
    refreshKey?: number;
}

interface ApiErrorShape {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
}

function extractErrorMessage(err: unknown, fallback = 'Request failed.'): string {
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

interface ClusterRowProps {
    cluster: FailureCluster;
    onSelectRun: (runId: string) => void;
}

function ClusterRow({ cluster, onSelectRun }: ClusterRowProps) {
    const [open, setOpen] = useState(false);
    const exemplars = cluster.exemplar_event_ids || [];
    const summaries = cluster.exemplar_summaries || [];
    const runIds = cluster.exemplar_run_ids || [];

    return (
        <li className="failure-cluster-row">
            <div className="failure-cluster-head">
                <span className="badge badge-danger">
                    {cluster.failure_count}×
                </span>
                <span className="failure-cluster-stage">{cluster.stage}</span>
                <code className="failure-cluster-reason">
                    {cluster.reason_code}
                </code>
                <span className="dim">sig {cluster.signature}</span>
                <span className="dim failure-cluster-last">
                    last {formatTs(cluster.last_seen_at)}
                </span>
                <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setOpen((o) => !o)}
                    aria-expanded={open ? 'true' : 'false'}
                    aria-label={
                        open ? 'Collapse exemplars' : 'Expand exemplars'
                    }
                >
                    {open
                        ? `Hide exemplars (${exemplars.length})`
                        : `Show exemplars (${exemplars.length})`}
                </button>
            </div>
            {open && (
                <ol className="failure-cluster-exemplars">
                    {exemplars.map((event_id, idx) => {
                        const exemplarRunId = runIds[idx] || '';
                        return (
                            <li key={event_id} className="failure-cluster-exemplar">
                                <button
                                    type="button"
                                    className="failure-cluster-exemplar-link"
                                    onClick={() => {
                                        if (exemplarRunId) {
                                            onSelectRun(exemplarRunId);
                                        }
                                    }}
                                    disabled={!exemplarRunId}
                                    aria-label={`Open events for cluster exemplar ${event_id}`}
                                >
                                    event #{event_id}
                                    {exemplarRunId && (
                                        <span className="dim">
                                            {' '}
                                            ({exemplarRunId})
                                        </span>
                                    )}
                                </button>
                                <span className="dim">
                                    {summaries[idx] || '(no summary)'}
                                </span>
                            </li>
                        );
                    })}
                </ol>
            )}
        </li>
    );
}

export default function FailureClusterList({
    projectId,
    onSelectRun,
    refreshKey = 0,
}: Props) {
    const [clusters, setClusters] = useState<FailureCluster[]>([]);
    const [loading, setLoading] = useState(false);
    const [recomputing, setRecomputing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [recomputeNote, setRecomputeNote] = useState<string | null>(null);

    const fetchClusters = useCallback(async () => {
        setLoading(true);
        try {
            const response = await api.get<FailureClusterListResponse>(
                `/projects/${projectId}/failure-clusters`,
            );
            setClusters(response.data.clusters || []);
            setError(null);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to load clusters.'));
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetchClusters();
    }, [fetchClusters, refreshKey]);

    const recompute = useCallback(async () => {
        setRecomputing(true);
        setError(null);
        setRecomputeNote(null);
        try {
            const response = await api.post<FailureClusterRecomputeResponse>(
                `/projects/${projectId}/failure-clusters/recompute`,
                {},
            );
            const body = response.data;
            setRecomputeNote(
                `Recompute scanned ${body.events_considered} event(s); `
                + `${body.clusters_created} created, ${body.clusters_updated} updated `
                + `(${body.clusters_total} total).`,
            );
            await fetchClusters();
        } catch (err) {
            setError(extractErrorMessage(err, 'Recompute failed.'));
        } finally {
            setRecomputing(false);
        }
    }, [fetchClusters, projectId]);

    return (
        <div className="failure-cluster-list">
            <div className="failure-cluster-header">
                <h3 className="observability-heading">Failure clusters</h3>
                <div className="failure-cluster-header-actions">
                    <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => void recompute()}
                        disabled={recomputing}
                    >
                        {recomputing ? 'Recomputing…' : 'Recompute'}
                    </button>
                    <CommandSnippet
                        cli={`brewslm logs clusters --project ${projectId}`}
                        api={{
                            method: 'POST',
                            path: `/projects/${projectId}/failure-clusters/recompute`,
                            body: {},
                        }}
                    />
                </div>
            </div>
            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}
            {recomputeNote && (
                <div className="deployment-status is-info" role="status">
                    {recomputeNote}
                </div>
            )}
            {loading && !clusters.length && (
                <div className="dim">Loading clusters…</div>
            )}
            {!loading && !clusters.length && !error && (
                <EmptyState
                    title="No failure clusters yet"
                    description="Failure clusters group recent error events by reason code + signature so you can spot patterns. Click Recompute to fold any error events in the run log into clusters."
                    primary={{ label: 'Recompute now', onClick: () => void recompute() }}
                    docsHref="http://localhost:3001/docs/observability/failure-clusters"
                />
            )}
            {clusters.length > 0 && (
                <ul className="failure-cluster-rows" aria-label="Failure clusters">
                    {clusters.map((cluster) => (
                        <ClusterRow
                            key={cluster.id}
                            cluster={cluster}
                            onSelectRun={onSelectRun}
                        />
                    ))}
                </ul>
            )}
        </div>
    );
}
