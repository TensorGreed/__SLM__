/**
 * Overview panel summarizing data readiness with blocker / warning / info issue display.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    ClipboardCheck,
    Database,
    RefreshCw,
} from 'lucide-react';

import {
    getDataStudioOverview,
} from '../../api/dataStudio';
import type { DataStudioOverview } from '../../api/dataStudio';
import GoalLedgerCard from './GoalLedgerCard';
import './DataStudioOverviewPanel.css';

interface DataStudioOverviewPanelProps {
    projectId: number;
    onOpenTab?: (tabKey: string) => void;
}

const VERDICT_COPY: Record<DataStudioOverview['verdict'], { label: string; detail: string }> = {
    blocked: {
        label: 'Blocked',
        detail: 'Fix blockers before preparing training data.',
    },
    needs_work: {
        label: 'Needs work',
        detail: 'Usable, but a few data checks need attention.',
    },
    ready: {
        label: 'Ready',
        detail: 'Data is ready for the next training step.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function issueIcon(severity: string) {
    if (severity === 'blocker') {
        return <AlertTriangle size={16} aria-hidden="true" />;
    }
    if (severity === 'warning') {
        return <AlertTriangle size={16} aria-hidden="true" />;
    }
    return <CheckCircle2 size={16} aria-hidden="true" />;
}

export default function DataStudioOverviewPanel({
    projectId,
    onOpenTab,
}: DataStudioOverviewPanelProps) {
    const [overview, setOverview] = useState<DataStudioOverview | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadOverview = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioOverview(projectId);
            setOverview(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Data Studio overview.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadOverview();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topIssues = useMemo(() => overview?.issues.slice(0, 4) ?? [], [overview]);

    if (loading && !overview) {
        return (
            <section className="data-studio-overview data-studio-overview--loading">
                <span>Loading Data Studio overview...</span>
            </section>
        );
    }

    if (error && !overview) {
        return (
            <section className="data-studio-overview data-studio-overview--error">
                <div>
                    <h3>Data Studio Overview</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadOverview()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!overview) {
        return null;
    }

    const verdict = VERDICT_COPY[overview.verdict];
    const primaryAction = overview.primary_action;
    const domainName = overview.domain?.display_name
        || overview.domain?.profile_id
        || 'Generic domain';
    const recipeName = overview.recipe?.name || 'No recipe selected';

    return (
        <section
            className={`data-studio-overview data-studio-overview--${overview.verdict}`}
            data-testid="data-studio-overview"
        >
            {/* Arc H — single "% toward your stated goal" widget at
                the top of the overview. Renders the goal ledger
                (data_ready / gold_set / predicted_pass / eval_pass_rate)
                with Term-linked components so each row teaches the
                user via Academy deep-links. */}
            <GoalLedgerCard projectId={projectId} />

            <div className="data-studio-overview__header">
                <div>
                    <p className="data-studio-overview__eyebrow">Data Studio</p>
                    <h3>Overview readiness</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-overview__actions">
                    <span className={`data-studio-overview__verdict data-studio-overview__verdict--${overview.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-overview__refresh"
                        onClick={() => void loadOverview()}
                        aria-label="Refresh Data Studio overview"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-overview__metrics" aria-label="Data Studio readiness metrics">
                <div className="data-studio-overview__metric">
                    <Database size={18} aria-hidden="true" />
                    <span>Trainable rows</span>
                    <strong>{formatNumber(overview.row_counts.trainable)}</strong>
                </div>
                <div className="data-studio-overview__metric">
                    <ClipboardCheck size={18} aria-hidden="true" />
                    <span>Pending review</span>
                    <strong>{formatNumber(overview.row_counts.synthetic_pending)}</strong>
                </div>
                <div className="data-studio-overview__metric">
                    <span>Recipe</span>
                    <strong>{recipeName}</strong>
                </div>
                <div className="data-studio-overview__metric">
                    <span>Domain</span>
                    <strong>{domainName}</strong>
                </div>
            </div>

            <div className="data-studio-overview__body">
                <div className="data-studio-overview__checks">
                    <h4>Focus checks</h4>
                    {topIssues.length > 0 ? (
                        <ul>
                            {topIssues.map((issue) => (
                                <li key={issue.id} className={`data-studio-overview__issue data-studio-overview__issue--${issue.severity}`}>
                                    <span className="data-studio-overview__issue-icon">
                                        {issueIcon(issue.severity)}
                                    </span>
                                    <span>
                                        <strong>{issue.title}</strong>
                                        <small>{issue.message}</small>
                                    </span>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="data-studio-overview__empty">
                            No current blockers. Prepare or train from the active dataset version.
                        </p>
                    )}
                </div>

                <div className="data-studio-overview__next">
                    <h4>Next best action</h4>
                    <p>{primaryAction.reason}</p>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => onOpenTab?.(primaryAction.target_tab)}
                    >
                        {primaryAction.label}
                    </button>
                </div>
            </div>
        </section>
    );
}
