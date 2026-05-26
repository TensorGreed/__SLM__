/**
 * Panel with domain-aware synthetic-data expansion recommendations and setup guidance.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    Bot,
    CheckCircle2,
    ExternalLink,
    Lightbulb,
    RefreshCw,
    Route,
    ShieldCheck,
} from 'lucide-react';

import {
    getDataStudioSyntheticRecommendations,
} from '../../api/dataStudio';
import type {
    DataStudioSyntheticRecommendationItem,
    DataStudioSyntheticRecommendations,
} from '../../api/dataStudio';
import './DataStudioSyntheticRecommendationsPanel.css';

interface DataStudioSyntheticRecommendationsPanelProps {
    projectId: number;
    onOpenTab: (targetTab: string) => void;
}

const RECOMMENDATION_VERDICT_COPY: Record<DataStudioSyntheticRecommendations['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No advice',
        detail: 'Add domain evidence, a recipe, or Gold Set rows to unlock recommendations.',
    },
    attention: {
        label: 'Review advice',
        detail: 'BrewSLM found domain-aware synthetic actions and setup checks to review.',
    },
    ready: {
        label: 'Ready',
        detail: 'Domain-aware synthetic recommendations are ready to use in the existing workflow.',
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
    if (!value) return 'n/a';
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

function RecommendationCard({
    item,
    onOpenTab,
}: {
    item: DataStudioSyntheticRecommendationItem;
    onOpenTab: (targetTab: string) => void;
}) {
    return (
        <article className={`data-studio-synth-recs__card data-studio-synth-recs__card--${priorityClass(item.priority)}`}>
            <div className="data-studio-synth-recs__card-head">
                <div>
                    <strong>{item.title}</strong>
                    <small>
                        {labelForToken(item.strategy)}
                        {item.playbook_mode ? ` · ${labelForToken(item.playbook_mode)}` : ''}
                    </small>
                </div>
                <span>{formatPercent(item.confidence)}</span>
            </div>
            <p>{item.domain_reason}</p>
            <p>{item.rationale}</p>
            {item.evidence.length > 0 ? (
                <ul>
                    {item.evidence.slice(0, 4).map((evidence) => (
                        <li key={evidence}>{evidence}</li>
                    ))}
                </ul>
            ) : null}
            <div className="data-studio-synth-recs__card-foot">
                <span>
                    {item.generation_path.available ? item.generation_path.describe : 'Ollama setup needed'}
                    {' · '}
                    {item.generation_path.paid_required ? 'paid backend' : 'local default'}
                </span>
                <button type="button" className="btn btn-secondary" onClick={() => onOpenTab(item.target_tab)}>
                    <ExternalLink size={15} aria-hidden="true" />
                    {item.action_label}
                </button>
            </div>
        </article>
    );
}

export default function DataStudioSyntheticRecommendationsPanel({
    projectId,
    onOpenTab,
}: DataStudioSyntheticRecommendationsPanelProps) {
    const [recommendations, setRecommendations] = useState<DataStudioSyntheticRecommendations | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadRecommendations = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioSyntheticRecommendations(projectId);
            setRecommendations(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load synthetic recommendations.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadRecommendations();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topIssues = useMemo(
        () => recommendations?.issues.slice(0, 3) ?? [],
        [recommendations],
    );
    const topRecommendations = useMemo(
        () => recommendations?.recommendations.slice(0, 5) ?? [],
        [recommendations],
    );

    if (loading && !recommendations) {
        return (
            <section className="data-studio-synth-recs data-studio-synth-recs--loading">
                <span>Loading synthetic recommendations...</span>
            </section>
        );
    }

    if (error && !recommendations) {
        return (
            <section className="data-studio-synth-recs data-studio-synth-recs--error">
                <div>
                    <h3>Synthetic recommendations</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadRecommendations()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!recommendations) {
        return null;
    }

    const verdict = RECOMMENDATION_VERDICT_COPY[recommendations.verdict];
    const domainLabel = recommendations.domain.label || 'Generic Domain';
    const recipeLabel = recommendations.recipe?.name || recommendations.recipe?.id || 'No recipe';

    return (
        <section
            className={`data-studio-synth-recs data-studio-synth-recs--${recommendations.verdict}`}
            data-testid="data-studio-synth-recommendations"
        >
            <div className="data-studio-synth-recs__header">
                <div>
                    <p className="data-studio-synth-recs__eyebrow">Recommend</p>
                    <h3>Domain-aware synthetic recommendations</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-synth-recs__actions">
                    <span className={`data-studio-synth-recs__verdict data-studio-synth-recs__verdict--${recommendations.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-synth-recs__refresh"
                        onClick={() => void loadRecommendations()}
                        aria-label="Refresh synthetic recommendations"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-synth-recs__metrics" aria-label="Synthetic recommendation metrics">
                <div className="data-studio-synth-recs__metric">
                    <ShieldCheck size={18} aria-hidden="true" />
                    <span>Detected domain</span>
                    <strong>{domainLabel}</strong>
                </div>
                <div className="data-studio-synth-recs__metric">
                    <Route size={18} aria-hidden="true" />
                    <span>Recipe</span>
                    <strong>{recipeLabel}</strong>
                </div>
                <div className="data-studio-synth-recs__metric">
                    <Lightbulb size={18} aria-hidden="true" />
                    <span>Recommendations</span>
                    <strong>{formatNumber(recommendations.recommendations.length)}</strong>
                </div>
                <div className="data-studio-synth-recs__metric">
                    <Bot size={18} aria-hidden="true" />
                    <span>Local generation</span>
                    <strong>{recommendations.signals.ollama_available ? 'Ollama ready' : 'Ollama setup'}</strong>
                </div>
            </div>

            <div className="data-studio-synth-recs__signals">
                <span>{formatPercent(recommendations.domain.confidence)} domain confidence</span>
                <span>{formatNumber(recommendations.signals.gold_trusted_examples)} trusted gold</span>
                <span>{formatNumber(recommendations.signals.synthetic_pending)} pending synthetic</span>
                <span>
                    {recommendations.signals.compatible_playbook_modes.length} compatible mode
                    {recommendations.signals.compatible_playbook_modes.length === 1 ? '' : 's'}
                </span>
            </div>

            <div className="data-studio-synth-recs__body">
                <div className="data-studio-synth-recs__recommendations">
                    <h4>Recommended actions</h4>
                    {topRecommendations.length > 0 ? (
                        <div className="data-studio-synth-recs__cards">
                            {topRecommendations.map((item) => (
                                <RecommendationCard item={item} key={item.id} onOpenTab={onOpenTab} />
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-synth-recs__empty">
                            Recommendations appear after BrewSLM has enough deterministic context.
                        </p>
                    )}
                </div>

                <div className="data-studio-synth-recs__checks">
                    <h4>Checks behind the advice</h4>
                    {topIssues.length > 0 ? (
                        <ul className="data-studio-synth-recs__issues">
                            {topIssues.map((issue) => (
                                <li key={issue.id} className={`data-studio-synth-recs__issue data-studio-synth-recs__issue--${issue.severity}`}>
                                    <span>{issueIcon(issue.severity)}</span>
                                    <div>
                                        <strong>{issue.title}</strong>
                                        <small>{issue.message}</small>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="data-studio-synth-recs__empty">
                            No synthetic recommendation blockers are active.
                        </p>
                    )}

                    <div className="data-studio-synth-recs__entrypoints">
                        {recommendations.entry_points.slice(0, 3).map((entry) => (
                            <button
                                type="button"
                                className="btn btn-secondary"
                                key={entry.target_tab}
                                onClick={() => onOpenTab(entry.target_tab)}
                            >
                                <ExternalLink size={15} aria-hidden="true" />
                                {entry.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            <details className="data-studio-synth-recs__details">
                <summary>Power details</summary>
                <pre>{compactJson(recommendations)}</pre>
            </details>
        </section>
    );
}
