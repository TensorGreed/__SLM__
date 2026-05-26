/**
 * Panel for synthetic-row quality analytics with review gates and anchor consistency checks.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    BarChart3,
    CheckCircle2,
    ExternalLink,
    RefreshCw,
    ShieldCheck,
    Sparkles,
    Tags,
} from 'lucide-react';

import { getDataStudioSyntheticQualityAnalytics } from '../../api/dataStudio';
import type {
    DataStudioSyntheticQualityAnalytics,
    DataStudioSyntheticQualityFinding,
    DataStudioSyntheticQualitySourceGroup,
} from '../../api/dataStudio';
import './DataStudioSyntheticQualityPanel.css';

interface DataStudioSyntheticQualityPanelProps {
    projectId: number;
    onOpenTarget: (target: string) => void;
}

const SYNTHETIC_QUALITY_COPY: Record<DataStudioSyntheticQualityAnalytics['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No rows',
        detail: 'Generate or import synthetic rows before quality analytics can score them.',
    },
    attention: {
        label: 'Needs review',
        detail: 'Synthetic rows exist, but quality, review, or anchor checks need attention before SFT.',
    },
    ready: {
        label: 'Ready',
        detail: 'Synthetic quality checks look clear for the current reviewed rows.',
    },
};

function formatNumber(value: number | undefined | null): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatPercent(value: number | undefined | null): string {
    if (value === null || value === undefined) return 'n/a';
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

function findingIcon(finding: DataStudioSyntheticQualityFinding) {
    if (finding.severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

function FindingCard({
    finding,
    onOpenTarget,
}: {
    finding: DataStudioSyntheticQualityFinding;
    onOpenTarget: (target: string) => void;
}) {
    return (
        <article className={`data-studio-synth-quality__finding data-studio-synth-quality__finding--${finding.severity}`}>
            <span>{findingIcon(finding)}</span>
            <div>
                <strong>{finding.label}</strong>
                <p>{finding.message}</p>
                {finding.evidence.length > 0 ? (
                    <small>{finding.evidence.slice(0, 2).join(' · ')}</small>
                ) : null}
            </div>
            <button type="button" className="btn btn-secondary" onClick={() => onOpenTarget(finding.target_tab)}>
                <ExternalLink size={15} aria-hidden="true" />
                {finding.action_label}
            </button>
        </article>
    );
}

function SourceGroupCard({
    group,
    onOpenTarget,
}: {
    group: DataStudioSyntheticQualitySourceGroup;
    onOpenTarget: (target: string) => void;
}) {
    return (
        <article className="data-studio-synth-quality__source">
            <div className="data-studio-synth-quality__source-head">
                <strong>{group.source}</strong>
                <span>{formatNumber(group.count)} rows</span>
            </div>
            <div className="data-studio-synth-quality__chips">
                <span>{formatNumber(group.pending)} pending</span>
                <span>{formatNumber(group.accepted)} accepted</span>
                <span>{formatNumber(group.low_confidence)} low confidence</span>
                <span>{formatNumber(group.missing_required)} missing fields</span>
                <span>{formatPercent(group.avg_gold_similarity)} Gold similarity</span>
            </div>
            <button type="button" className="btn btn-secondary" onClick={() => onOpenTarget(group.target_tab)}>
                <ExternalLink size={15} aria-hidden="true" />
                Open source review
            </button>
        </article>
    );
}

export default function DataStudioSyntheticQualityPanel({
    projectId,
    onOpenTarget,
}: DataStudioSyntheticQualityPanelProps) {
    const [analytics, setAnalytics] = useState<DataStudioSyntheticQualityAnalytics | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadAnalytics = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioSyntheticQualityAnalytics(projectId);
            setAnalytics(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load synthetic quality analytics.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadAnalytics();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topFindings = useMemo(
        () => analytics?.findings.slice(0, 4) ?? [],
        [analytics],
    );
    const topSources = useMemo(
        () => analytics?.source_groups.slice(0, 4) ?? [],
        [analytics],
    );

    if (loading && !analytics) {
        return (
            <section className="data-studio-synth-quality data-studio-synth-quality--loading">
                <span>Loading synthetic quality analytics...</span>
            </section>
        );
    }

    if (error && !analytics) {
        return (
            <section className="data-studio-synth-quality data-studio-synth-quality--error">
                <div>
                    <h3>Synthetic quality analytics</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadAnalytics()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!analytics) {
        return null;
    }

    const verdict = SYNTHETIC_QUALITY_COPY[analytics.verdict];
    const recipeLabel = analytics.recipe?.name || analytics.recipe?.id || 'No recipe';

    return (
        <section
            className={`data-studio-synth-quality data-studio-synth-quality--${analytics.verdict}`}
            data-testid="data-studio-synthetic-quality"
        >
            <div className="data-studio-synth-quality__header">
                <div>
                    <p className="data-studio-synth-quality__eyebrow">Synthetic quality</p>
                    <h3>Synthetic quality analytics</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-synth-quality__actions">
                    <span className={`data-studio-synth-quality__verdict data-studio-synth-quality__verdict--${analytics.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-synth-quality__refresh"
                        onClick={() => void loadAnalytics()}
                        aria-label="Refresh synthetic quality analytics"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-synth-quality__metrics" aria-label="Synthetic quality metrics">
                <div className="data-studio-synth-quality__metric">
                    <Sparkles size={18} aria-hidden="true" />
                    <span>Total synthetic</span>
                    <strong>{formatNumber(analytics.summary.total_rows)}</strong>
                </div>
                <div className="data-studio-synth-quality__metric">
                    <BarChart3 size={18} aria-hidden="true" />
                    <span>Avg confidence</span>
                    <strong>{formatPercent(analytics.summary.avg_confidence)}</strong>
                </div>
                <div className="data-studio-synth-quality__metric">
                    <ShieldCheck size={18} aria-hidden="true" />
                    <span>Gold similarity</span>
                    <strong>{formatPercent(analytics.summary.avg_gold_similarity)}</strong>
                </div>
                <div className="data-studio-synth-quality__metric">
                    <Tags size={18} aria-hidden="true" />
                    <span>Domain</span>
                    <strong>{analytics.domain.label}</strong>
                </div>
            </div>

            <div className="data-studio-synth-quality__signals">
                <span>{recipeLabel}</span>
                <span>{formatNumber(analytics.summary.pending_rows)} pending</span>
                <span>{formatNumber(analytics.summary.accepted_rows)} accepted</span>
                <span>{formatNumber(analytics.summary.duplicate_signal_rows)} duplicate signals</span>
                <span>{formatNumber(analytics.summary.missing_required_rows)} missing required</span>
                <span>{analytics.assist.default_provider} explanations optional</span>
            </div>

            <div className="data-studio-synth-quality__body">
                <div className="data-studio-synth-quality__findings">
                    <h4>Fix or review</h4>
                    {topFindings.length > 0 ? (
                        <div className="data-studio-synth-quality__finding-list">
                            {topFindings.map((finding) => (
                                <FindingCard
                                    key={finding.id}
                                    finding={finding}
                                    onOpenTarget={onOpenTarget}
                                />
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-synth-quality__empty">
                            No synthetic quality findings need action.
                        </p>
                    )}
                </div>

                <div className="data-studio-synth-quality__sources">
                    <h4>By playbook/source</h4>
                    {topSources.length > 0 ? (
                        <div className="data-studio-synth-quality__source-list">
                            {topSources.map((source) => (
                                <SourceGroupCard
                                    key={source.key}
                                    group={source}
                                    onOpenTarget={onOpenTarget}
                                />
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-synth-quality__empty">
                            Source-level analytics appear after synthetic rows exist.
                        </p>
                    )}
                </div>
            </div>

            <div className="data-studio-synth-quality__bands">
                <div>
                    <strong>Confidence</strong>
                    <small>
                        {formatNumber(analytics.quality_bands.confidence.high)} high ·{' '}
                        {formatNumber(analytics.quality_bands.confidence.medium)} medium ·{' '}
                        {formatNumber(analytics.quality_bands.confidence.low)} low ·{' '}
                        {formatNumber(analytics.quality_bands.confidence.unknown)} unknown
                    </small>
                </div>
                <div>
                    <strong>Duplicates</strong>
                    <small>
                        {formatNumber(analytics.quality_bands.duplicates.exact_duplicate_rows)} exact ·{' '}
                        {formatNumber(analytics.quality_bands.duplicates.near_duplicate_pairs)} near pairs
                    </small>
                </div>
                <div>
                    <strong>Gold anchors</strong>
                    <small>
                        {formatNumber(analytics.quality_bands.gold_similarity.gold_anchor_rows)} anchors ·{' '}
                        {formatNumber(analytics.quality_bands.gold_similarity.low_similarity_rows)} low similarity
                    </small>
                </div>
                <div>
                    <strong>Status</strong>
                    <small>
                        {analytics.status_groups
                            .filter((group) => group.count > 0)
                            .map((group) => `${labelForToken(group.status)} ${group.count}`)
                            .join(' · ') || 'No status rows'}
                    </small>
                </div>
            </div>

            {analytics.preview_rows.length > 0 ? (
                <details className="data-studio-synth-quality__preview">
                    <summary>Redacted row previews</summary>
                    <div className="data-studio-synth-quality__preview-list">
                        {analytics.preview_rows.slice(0, 4).map((row) => (
                            <article key={`${row.source}:${row.row_index}`}>
                                <strong>{row.source}</strong>
                                <small>{row.redacted_text || row.reason}</small>
                            </article>
                        ))}
                    </div>
                </details>
            ) : null}

            <div className="data-studio-synth-quality__entrypoints">
                {analytics.entry_points.slice(0, 5).map((entry) => (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        key={`${entry.target_tab}:${entry.label}`}
                        onClick={() => onOpenTarget(entry.target_tab)}
                    >
                        <ExternalLink size={15} aria-hidden="true" />
                        {entry.label}
                    </button>
                ))}
            </div>

            <details className="data-studio-synth-quality__details">
                <summary>Power details</summary>
                <pre>{compactJson(analytics)}</pre>
            </details>
        </section>
    );
}
