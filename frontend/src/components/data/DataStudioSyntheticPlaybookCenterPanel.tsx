import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    Bot,
    CheckCircle2,
    ExternalLink,
    Library,
    ListChecks,
    RefreshCw,
    Sparkles,
} from 'lucide-react';

import {
    getDataStudioSyntheticPlaybookCenter,
} from '../../api/dataStudio';
import type {
    DataStudioSyntheticPlaybookCenter,
    DataStudioSyntheticPrerequisite,
} from '../../api/dataStudio';
import './DataStudioSyntheticPlaybookCenterPanel.css';

interface DataStudioSyntheticPlaybookCenterPanelProps {
    projectId: number;
    onOpenSynthetic: () => void;
}

const SYNTHETIC_VERDICT_COPY: Record<DataStudioSyntheticPlaybookCenter['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No playbooks',
        detail: 'Select a recipe to unlock recipe-aware synthetic playbooks.',
    },
    attention: {
        label: 'Needs setup',
        detail: 'Synthetic playbooks are available, but prerequisites or review queues need attention.',
    },
    ready: {
        label: 'Ready',
        detail: 'Playbooks, local backend, and review gates are ready for synthetic expansion.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function labelForStatus(status: string | undefined): string {
    if (!status) return 'Unknown';
    return status.replace(/_/g, ' ');
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function prerequisiteIcon(item: DataStudioSyntheticPrerequisite) {
    if (item.status === 'met') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

function issueIcon(severity: string) {
    if (severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

export default function DataStudioSyntheticPlaybookCenterPanel({
    projectId,
    onOpenSynthetic,
}: DataStudioSyntheticPlaybookCenterPanelProps) {
    const [center, setCenter] = useState<DataStudioSyntheticPlaybookCenter | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadCenter = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioSyntheticPlaybookCenter(projectId);
            setCenter(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Synthetic Playbook Center.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadCenter();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topIssues = useMemo(
        () => center?.issues.slice(0, 3) ?? [],
        [center],
    );
    const topPlaybooks = useMemo(
        () => center?.catalog.preview_playbooks.slice(0, 4) ?? [],
        [center],
    );
    const topPendingGroups = useMemo(
        () => center?.review_queue.top_pending_groups.slice(0, 3) ?? [],
        [center],
    );

    if (loading && !center) {
        return (
            <section className="data-studio-synth data-studio-synth--loading">
                <span>Loading Synthetic Playbook Center...</span>
            </section>
        );
    }

    if (error && !center) {
        return (
            <section className="data-studio-synth data-studio-synth--error">
                <div>
                    <h3>Synthetic Playbook Center</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadCenter()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!center) {
        return null;
    }

    const verdict = SYNTHETIC_VERDICT_COPY[center.verdict];
    const recipeLabel = center.recipe?.name || center.recipe?.id || 'No recipe';
    const ollamaReady = center.recommended_backend.available;
    const backendLabel = ollamaReady ? center.recommended_backend.describe : 'Ollama not ready';

    return (
        <section
            className={`data-studio-synth data-studio-synth--${center.verdict}`}
            data-testid="data-studio-synth-playbooks"
        >
            <div className="data-studio-synth__header">
                <div>
                    <p className="data-studio-synth__eyebrow">Synthetic</p>
                    <h3>Synthetic Playbook Center</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-synth__actions">
                    <span className={`data-studio-synth__verdict data-studio-synth__verdict--${center.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-synth__refresh"
                        onClick={() => void loadCenter()}
                        aria-label="Refresh Synthetic Playbook Center"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-synth__metrics" aria-label="Synthetic playbook metrics">
                <div className="data-studio-synth__metric">
                    <Library size={18} aria-hidden="true" />
                    <span>Compatible playbooks</span>
                    <strong>
                        {formatNumber(center.catalog.compatible_playbooks)}
                        {' / '}
                        {formatNumber(center.catalog.total_playbooks)}
                    </strong>
                </div>
                <div className="data-studio-synth__metric">
                    <Bot size={18} aria-hidden="true" />
                    <span>Local default</span>
                    <strong>{backendLabel}</strong>
                </div>
                <div className="data-studio-synth__metric">
                    <ListChecks size={18} aria-hidden="true" />
                    <span>Pending review</span>
                    <strong>{formatNumber(center.review_queue.total_pending)}</strong>
                </div>
                <div className="data-studio-synth__metric">
                    <Sparkles size={18} aria-hidden="true" />
                    <span>Accepted synthetic</span>
                    <strong>{formatNumber(center.review_queue.total_accepted)}</strong>
                </div>
            </div>

            <div className="data-studio-synth__entry">
                <div>
                    <strong>{center.entry_point.label}</strong>
                    <small>
                        {recipeLabel}
                        {' · '}
                        {center.recommended_backend.paid_required ? 'paid backend' : 'free local default'}
                    </small>
                </div>
                <button type="button" className="btn btn-primary" onClick={onOpenSynthetic}>
                    <ExternalLink size={16} aria-hidden="true" />
                    Open Synthetic workflow
                </button>
            </div>

            <div className="data-studio-synth__body">
                <div className="data-studio-synth__playbooks">
                    <h4>Available playbooks</h4>
                    {topPlaybooks.length > 0 ? (
                        <div className="data-studio-synth__playbook-list">
                            {topPlaybooks.map((playbook) => (
                                <article
                                    className="data-studio-synth__playbook"
                                    key={`${playbook.recipe_id}-${playbook.mode}`}
                                >
                                    <strong>{playbook.label}</strong>
                                    <small>
                                        {playbook.recipe_id}
                                        {' · '}
                                        {labelForStatus(playbook.mode)}
                                    </small>
                                </article>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-synth__empty">
                            Pick a recipe to see compatible playbook modes.
                        </p>
                    )}

                    <div className="data-studio-synth__prerequisites">
                        {center.prerequisites.map((item) => (
                            <div
                                className={`data-studio-synth__prereq data-studio-synth__prereq--${item.status}`}
                                key={item.id}
                            >
                                <span>{prerequisiteIcon(item)}</span>
                                <div>
                                    <strong>{item.label}</strong>
                                    <small>{item.message}</small>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="data-studio-synth__review">
                    <h4>Review queue</h4>
                    {topPendingGroups.length > 0 ? (
                        <div className="data-studio-synth__queue-list">
                            {topPendingGroups.map((group) => (
                                <article className="data-studio-synth__queue-group" key={group.synth_source}>
                                    <strong>{group.synth_source}</strong>
                                    <small>
                                        {formatNumber(group.count)} pending
                                        {group.truncated ? ' · sample shown' : ''}
                                    </small>
                                </article>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-synth__empty">
                            No synthetic rows are waiting for review.
                        </p>
                    )}

                    {topIssues.length > 0 ? (
                        <ul className="data-studio-synth__issues">
                            {topIssues.map((issue) => (
                                <li key={issue.id} className={`data-studio-synth__issue data-studio-synth__issue--${issue.severity}`}>
                                    <span>{issueIcon(issue.severity)}</span>
                                    <div>
                                        <strong>{issue.title}</strong>
                                        <small>{issue.message}</small>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="data-studio-synth__empty">
                            Playbook prerequisites are clear.
                        </p>
                    )}
                </div>
            </div>

            <details className="data-studio-synth__details">
                <summary>Power details</summary>
                <pre>{compactJson(center)}</pre>
            </details>
        </section>
    );
}
