import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    ClipboardCheck,
    ExternalLink,
    ListChecks,
    RefreshCw,
    ShieldCheck,
    UserCheck,
} from 'lucide-react';

import {
    getDataStudioReviewQueue,
} from '../../api/dataStudio';
import type {
    DataStudioReviewQueue,
    DataStudioReviewQueueTriageItem,
} from '../../api/dataStudio';
import './DataStudioReviewQueuePanel.css';

interface DataStudioReviewQueuePanelProps {
    projectId: number;
    onOpenTarget: (target: string) => void;
}

const REVIEW_VERDICT_COPY: Record<DataStudioReviewQueue['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No queue',
        detail: 'Create synthetic, Gold Set, or annotation review work to start a queue.',
    },
    attention: {
        label: 'Needs review',
        detail: 'Review work is open across synthetic, Gold Set, or annotation workflows.',
    },
    ready: {
        label: 'Clear',
        detail: 'Review gates are clear and accepted or promoted examples are ready downstream.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatPercent(value: number | undefined): string {
    const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
    return `${Math.round(normalized * 100)}%`;
}

function labelForToken(value: string | undefined | null): string {
    if (!value) return 'Unknown';
    return value.replace(/_/g, ' ');
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function issueIcon(severity: string) {
    if (severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

function priorityClass(priority: string): string {
    if (priority === 'high' || priority === 'medium' || priority === 'low') {
        return priority;
    }
    return 'low';
}

function TriageCard({
    item,
    onOpenTarget,
}: {
    item: DataStudioReviewQueueTriageItem;
    onOpenTarget: (target: string) => void;
}) {
    return (
        <article className={`data-studio-review__triage-card data-studio-review__triage-card--${priorityClass(item.priority)}`}>
            <div className="data-studio-review__triage-head">
                <div>
                    <strong>{item.title}</strong>
                    <small>{labelForToken(item.priority)} priority</small>
                </div>
                <span>{formatNumber(item.count)}</span>
            </div>
            <p>{item.message}</p>
            {item.evidence.length > 0 ? (
                <ul>
                    {item.evidence.slice(0, 3).map((evidence) => (
                        <li key={evidence}>{evidence}</li>
                    ))}
                </ul>
            ) : null}
            <button type="button" className="btn btn-secondary" onClick={() => onOpenTarget(item.target_tab)}>
                <ExternalLink size={15} aria-hidden="true" />
                {item.action_label}
            </button>
        </article>
    );
}

export default function DataStudioReviewQueuePanel({
    projectId,
    onOpenTarget,
}: DataStudioReviewQueuePanelProps) {
    const [queue, setQueue] = useState<DataStudioReviewQueue | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadQueue = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioReviewQueue(projectId);
            setQueue(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Data Studio review queue.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadQueue();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topIssues = useMemo(
        () => queue?.issues.slice(0, 4) ?? [],
        [queue],
    );
    const topTriage = useMemo(
        () => queue?.triage.slice(0, 4) ?? [],
        [queue],
    );
    const sourceGroups = useMemo(
        () => queue?.groupings.by_source.slice(0, 6) ?? [],
        [queue],
    );
    const statusGroups = useMemo(
        () => (queue?.groupings.by_status ?? []).filter((group) => group.count > 0).slice(0, 7),
        [queue],
    );

    if (loading && !queue) {
        return (
            <section className="data-studio-review data-studio-review--loading">
                <span>Loading review queue...</span>
            </section>
        );
    }

    if (error && !queue) {
        return (
            <section className="data-studio-review data-studio-review--error">
                <div>
                    <h3>Review Queue</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadQueue()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!queue) {
        return null;
    }

    const verdict = REVIEW_VERDICT_COPY[queue.verdict];

    return (
        <section
            className={`data-studio-review data-studio-review--${queue.verdict}`}
            data-testid="data-studio-review-queue"
        >
            <div className="data-studio-review__header">
                <div>
                    <p className="data-studio-review__eyebrow">Review</p>
                    <h3>Review Queue</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-review__actions">
                    <span className={`data-studio-review__verdict data-studio-review__verdict--${queue.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-review__refresh"
                        onClick={() => void loadQueue()}
                        aria-label="Refresh Data Studio review queue"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-review__metrics" aria-label="Review queue metrics">
                <div className="data-studio-review__metric">
                    <ListChecks size={18} aria-hidden="true" />
                    <span>Open review</span>
                    <strong>{formatNumber(queue.totals.open_review_items)}</strong>
                </div>
                <div className="data-studio-review__metric">
                    <ClipboardCheck size={18} aria-hidden="true" />
                    <span>Accepted synthetic</span>
                    <strong>{formatNumber(queue.totals.synthetic_accepted)}</strong>
                </div>
                <div className="data-studio-review__metric">
                    <ShieldCheck size={18} aria-hidden="true" />
                    <span>Trusted gold</span>
                    <strong>{formatNumber(queue.totals.gold_trusted_examples)}</strong>
                </div>
                <div className="data-studio-review__metric">
                    <UserCheck size={18} aria-hidden="true" />
                    <span>Promoted labels</span>
                    <strong>{formatNumber(queue.totals.annotation_promoted)}</strong>
                </div>
            </div>

            <div className="data-studio-review__signals">
                <span>{queue.domain.label}</span>
                <span>{formatPercent(queue.domain.confidence)} domain confidence</span>
                <span>{formatNumber(queue.totals.synthetic_pending)} pending synthetic</span>
                <span>{formatNumber(queue.totals.annotation_labeled_unpromoted)} labels to promote</span>
            </div>

            <div className="data-studio-review__entrypoints">
                {queue.entry_points.slice(0, 4).map((entry) => (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        key={entry.target_tab}
                        onClick={() => onOpenTarget(entry.target_tab)}
                    >
                        <ExternalLink size={15} aria-hidden="true" />
                        {entry.label}
                    </button>
                ))}
            </div>

            <div className="data-studio-review__body">
                <div className="data-studio-review__triage">
                    <h4>What to review first</h4>
                    {topTriage.length > 0 ? (
                        <div className="data-studio-review__triage-list">
                            {topTriage.map((item) => (
                                <TriageCard item={item} key={item.id} onOpenTarget={onOpenTarget} />
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-review__empty">
                            No review actions are waiting right now.
                        </p>
                    )}
                </div>

                <div className="data-studio-review__groups">
                    <h4>Power grouping</h4>
                    {sourceGroups.length > 0 ? (
                        <div className="data-studio-review__source-list">
                            {sourceGroups.map((group) => (
                                <button
                                    type="button"
                                    className="data-studio-review__source"
                                    key={group.key}
                                    onClick={() => onOpenTarget(group.target_tab)}
                                >
                                    <span>
                                        <strong>{group.label}</strong>
                                        <small>
                                            {labelForToken(group.kind)}
                                            {' · '}
                                            {labelForToken(group.status)}
                                        </small>
                                    </span>
                                    <b>{formatNumber(group.count)}</b>
                                </button>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-review__empty">
                            Source groupings appear after reviewable rows exist.
                        </p>
                    )}

                    <div className="data-studio-review__status-list">
                        {statusGroups.map((group) => (
                            <button
                                type="button"
                                className="data-studio-review__status"
                                key={group.status}
                                onClick={() => onOpenTarget(group.target_tab)}
                            >
                                <span>{group.label}</span>
                                <strong>{formatNumber(group.count)}</strong>
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {topIssues.length > 0 ? (
                <ul className="data-studio-review__issues">
                    {topIssues.map((issue) => (
                        <li key={issue.id} className={`data-studio-review__issue data-studio-review__issue--${issue.severity}`}>
                            <span>{issueIcon(issue.severity)}</span>
                            <div>
                                <strong>{issue.title}</strong>
                                <small>{issue.message}</small>
                            </div>
                            <button type="button" className="btn btn-ghost" onClick={() => onOpenTarget(issue.target_tab)}>
                                {issue.action_label}
                            </button>
                        </li>
                    ))}
                </ul>
            ) : null}

            <details className="data-studio-review__details">
                <summary>Power details</summary>
                <pre>{compactJson(queue)}</pre>
            </details>
        </section>
    );
}
