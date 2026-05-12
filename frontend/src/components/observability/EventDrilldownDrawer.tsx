/**
 * EventDrilldownDrawer — per-run event stream (priority.md P36, P31).
 *
 * Slide-in panel that fetches every event for a single ``run_id``
 * via ``GET /api/run-events/run/{run_id}``. Used as a drill-in from
 * the TimelineTree's "Open" link and from the FailureClusterList
 * exemplars. Empty / loading / error states are explicit so the
 * surface never silently shows "nothing".
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import EmptyState from '../shared/EmptyState';
import type {
    RunEvent,
    RunEventsForRunResponse,
} from '../../types/observability';

interface Props {
    runId: string | null;
    onClose: () => void;
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

function severityBadgeClass(severity: string): string {
    if (severity === 'critical' || severity === 'error') return 'badge badge-danger';
    if (severity === 'warning') return 'badge badge-warning';
    return 'badge badge-info';
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

export default function EventDrilldownDrawer({ runId, onClose }: Props) {
    const [events, setEvents] = useState<RunEvent[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchEvents = useCallback(async () => {
        if (!runId) return;
        setLoading(true);
        setError(null);
        try {
            const response = await api.get<RunEventsForRunResponse>(
                `/run-events/run/${encodeURIComponent(runId)}`,
            );
            setEvents(response.data.events || []);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to load events.'));
        } finally {
            setLoading(false);
        }
    }, [runId]);

    useEffect(() => {
        if (runId) {
            void fetchEvents();
        } else {
            setEvents([]);
            setError(null);
        }
    }, [fetchEvents, runId]);

    if (!runId) return null;

    return (
        <aside
            className="event-drilldown-drawer"
            role="dialog"
            aria-label={`Events for run ${runId}`}
        >
            <header className="event-drilldown-header">
                <div>
                    <div className="dim">run_id</div>
                    <h3>{runId}</h3>
                </div>
                <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={onClose}
                    aria-label="Close drilldown"
                >
                    Close
                </button>
            </header>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            {loading && !events.length && (
                <div className="dim">Loading events…</div>
            )}

            {!loading && !events.length && !error && (
                <EmptyState
                    title="No events for this run"
                    description="Every pipeline stage emits canonical RunEvents to this timeline. If nothing's here, the run probably finished outside the current window or was started before observability went live."
                    docsHref="http://localhost:3001/docs/observability/run-events"
                />
            )}

            <ol className="event-drilldown-list">
                {events.map((event) => (
                    <li key={event.id} className="event-drilldown-item">
                        <div className="event-drilldown-head">
                            <span className={severityBadgeClass(String(event.severity))}>
                                {String(event.severity)}
                            </span>
                            <span className="event-drilldown-stage">
                                {event.stage}
                            </span>
                            <span className="event-drilldown-ts dim">
                                {formatTs(event.ts)}
                            </span>
                            <span className="event-drilldown-actor dim">
                                {event.actor}
                            </span>
                            {event.reason_code && (
                                <code className="timeline-reason-code">
                                    {event.reason_code}
                                </code>
                            )}
                        </div>
                        <div className="event-drilldown-summary">
                            {event.summary || '(no summary)'}
                        </div>
                        {Object.keys(event.payload || {}).length > 0 && (
                            <details className="event-drilldown-payload">
                                <summary>payload</summary>
                                <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                            </details>
                        )}
                    </li>
                ))}
            </ol>
        </aside>
    );
}
