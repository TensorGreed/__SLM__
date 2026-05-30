/**
 * ProjectObservabilityPage — Run Timeline + Failure Analysis +
 * Support Bundle (priority.md P36, Wave G).
 *
 * Single-page surface assembling the four Wave G consumer cards on
 * top of P31 RunEvents:
 *
 *   - Filters (P32 timeline filters: stage / severity / since / until /
 *     run_id anchor / limit).
 *   - Timeline tree (P32) with recursive expand/collapse.
 *   - Failure cluster list (P33) with exemplar deep-links.
 *   - Support bundle card (P34) for ops handoff.
 *   - Per-run drill-in drawer (P31's `/run-events/run/{id}`) shared
 *     by both the timeline and the cluster exemplars.
 *
 * Endpoints used:
 *   GET /api/projects/{id}/timeline
 *   GET /api/projects/{id}/failure-clusters
 *   POST /api/projects/{id}/failure-clusters/recompute
 *   GET /api/projects/{id}/support-bundles
 *   POST /api/projects/{id}/support-bundle
 *   GET /api/run-events/run/{run_id}
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useOutletContext, useParams } from 'react-router-dom';

import api from '../api/client';
import EventDrilldownDrawer from '../components/observability/EventDrilldownDrawer';
import FailureClusterList from '../components/observability/FailureClusterList';
import RunTimelineFilters from '../components/observability/RunTimelineFilters';
import SupportBundleCard from '../components/observability/SupportBundleCard';
import TimelineTree from '../components/observability/TimelineTree';
import type {
    TimelineFilters,
    TimelineResponse,
} from '../types/observability';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

import './ProjectObservabilityPage.css';

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

const DEFAULT_FILTERS: TimelineFilters = {
    stage: '',
    severity: '',
    run_id: '',
    since: '',
    until: '',
    limit: 500,
};

function buildTimelineParams(
    filters: TimelineFilters,
): Record<string, string | number> {
    const params: Record<string, string | number> = {};
    if (filters.stage) params.stage = filters.stage;
    if (filters.severity) params.severity = filters.severity;
    if (filters.run_id) params.run_id = filters.run_id;
    if (filters.since) params.since = filters.since;
    if (filters.until) params.until = filters.until;
    if (filters.limit) params.limit = filters.limit;
    return params;
}

export default function ProjectObservabilityPage() {
    const params = useParams();
    const workspace = useOutletContext<ProjectWorkspaceContextValue | null>();
    const routeProjectId = params.id ? Number.parseInt(params.id, 10) : null;
    const projectId =
        workspace?.projectId
        ?? (Number.isFinite(routeProjectId) ? routeProjectId : null);

    const [filters, setFilters] = useState<TimelineFilters>(DEFAULT_FILTERS);
    const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

    const fetchTimeline = useCallback(async () => {
        if (projectId == null) return;
        setLoading(true);
        try {
            const response = await api.get<TimelineResponse>(
                `/projects/${projectId}/timeline`,
                { params: buildTimelineParams(filters) },
            );
            setTimeline(response.data);
            setError(null);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to load timeline.'));
        } finally {
            setLoading(false);
        }
    }, [filters, projectId]);

    useEffect(() => {
        void fetchTimeline();
    }, [fetchTimeline]);

    const handleSelectRun = useCallback((runId: string) => {
        setSelectedRunId(runId);
    }, []);

    const handleCloseDrawer = useCallback(() => {
        setSelectedRunId(null);
    }, []);

    const summaryBadges = useMemo(() => {
        if (!timeline) return null;
        return (
            <div className="observability-summary-counts">
                <span className="badge badge-info">
                    {timeline.total_runs} run(s)
                </span>
                <span className="badge badge-info">
                    {timeline.total_events} event(s)
                </span>
                {timeline.orphaned_count > 0 && (
                    <span className="badge badge-warning">
                        {timeline.orphaned_count} orphaned
                    </span>
                )}
                {timeline.truncated && (
                    <span className="badge badge-warning">truncated</span>
                )}
            </div>
        );
    }, [timeline]);

    if (projectId == null) {
        return (
            <div className="workspace-page observability-page">
                <div className="deployment-status is-error" role="alert">
                    Project context is not available.
                </div>
            </div>
        );
    }

    return (
        <div className="workspace-page observability-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Observability</h2>
                    <p className="workspace-page-subtitle">
                        Tree-ordered run timeline (P32), clustered failures (P33),
                        and one-click support bundles with redaction (P34) — all
                        on top of the canonical RunEvent log (P31).
                    </p>
                </div>
                {summaryBadges}
            </section>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            <section className="card">
                <RunTimelineFilters
                    value={filters}
                    onChange={setFilters}
                    onRefresh={() => void fetchTimeline()}
                    loading={loading}
                    truncated={Boolean(timeline?.truncated)}
                />
                <TimelineTree
                    tree={timeline?.tree || []}
                    selectedRunId={selectedRunId}
                    onSelectRun={handleSelectRun}
                />
            </section>

            <section className="card" id="failure-clusters">
                <FailureClusterList
                    projectId={projectId}
                    onSelectRun={handleSelectRun}
                />
            </section>

            <section>
                <SupportBundleCard projectId={projectId} />
            </section>

            <EventDrilldownDrawer
                runId={selectedRunId}
                onClose={handleCloseDrawer}
            />
        </div>
    );
}
