import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    ClipboardCheck,
    Database,
    ExternalLink,
    FileText,
    GitBranch,
    RefreshCw,
    ShieldCheck,
} from 'lucide-react';

import { getDataStudioPrepareDataset } from '../../api/dataStudio';
import type {
    DataStudioIssue,
    DataStudioPrepareCheck,
    DataStudioPrepareDataset,
    DataStudioPrepareSplitItem,
} from '../../api/dataStudio';
import './DataStudioPrepareDatasetPanel.css';

interface DataStudioPrepareDatasetPanelProps {
    projectId: number;
    onOpenTarget: (target: string) => void;
}

const PREPARE_VERDICT_COPY: Record<DataStudioPrepareDataset['verdict'], { label: string; detail: string }> = {
    blocked: {
        label: 'Blocked',
        detail: 'Fix recipe, source, or mapping blockers before creating prepared split files.',
    },
    attention: {
        label: 'Check before prepare',
        detail: 'Core inputs can be inspected, but review or prepared-artifact checks still need attention.',
    },
    ready: {
        label: 'Ready',
        detail: 'Recipe, mapping, splits, manifest, and versions are aligned for downstream training.',
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

function statusClass(status: string): string {
    if (status === 'met' || status === 'ready') {
        return 'ready';
    }
    if (status === 'attention' || status === 'partial') {
        return 'attention';
    }
    return 'missing';
}

function statusIcon(status: string) {
    const normalized = statusClass(status);
    if (normalized === 'ready') {
        return <CheckCircle2 size={16} aria-hidden="true" />;
    }
    return <AlertTriangle size={16} aria-hidden="true" />;
}

function issueIcon(issue: DataStudioIssue) {
    if (issue.severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

function CheckRow({
    check,
    onOpenTarget,
}: {
    check: DataStudioPrepareCheck;
    onOpenTarget: (target: string) => void;
}) {
    return (
        <button
            type="button"
            className={`data-studio-prepare__check data-studio-prepare__check--${statusClass(check.status)}`}
            onClick={() => onOpenTarget(check.target_tab)}
        >
            <span>{statusIcon(check.status)}</span>
            <span>
                <strong>{check.label}</strong>
                <small>{check.message}</small>
            </span>
            <b>{labelForToken(check.status)}</b>
        </button>
    );
}

function SplitRow({ split }: { split: DataStudioPrepareSplitItem }) {
    const versionLabel = split.latest_version
        ? `v${split.latest_version.version}`
        : 'No version';
    return (
        <div className="data-studio-prepare__split">
            <div>
                <strong>{split.label}</strong>
                <small>
                    {split.file_exists ? 'File found' : 'No file'}
                    {' · '}
                    {versionLabel}
                </small>
            </div>
            <div>
                <span>{formatNumber(split.row_count)} rows</span>
                <small>{formatNumber(split.manifest_count)} in manifest</small>
            </div>
        </div>
    );
}

export default function DataStudioPrepareDatasetPanel({
    projectId,
    onOpenTarget,
}: DataStudioPrepareDatasetPanelProps) {
    const [prepare, setPrepare] = useState<DataStudioPrepareDataset | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadPrepare = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioPrepareDataset(projectId);
            setPrepare(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Data Studio prepare checks.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadPrepare();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topIssues = useMemo(
        () => prepare?.issues.slice(0, 5) ?? [],
        [prepare],
    );
    const openReviewCount = useMemo(
        () => prepare?.review_blockers.reduce((total, item) => total + Number(item.count || 0), 0) ?? 0,
        [prepare],
    );

    if (loading && !prepare) {
        return (
            <section className="data-studio-prepare data-studio-prepare--loading">
                <span>Loading prepare checks...</span>
            </section>
        );
    }

    if (error && !prepare) {
        return (
            <section className="data-studio-prepare data-studio-prepare--error">
                <div>
                    <h3>Prepare Dataset</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadPrepare()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!prepare) {
        return null;
    }

    const verdict = PREPARE_VERDICT_COPY[prepare.verdict];
    const selectedRecipe = prepare.recipe.selected;

    return (
        <section
            className={`data-studio-prepare data-studio-prepare--${prepare.verdict}`}
            data-testid="data-studio-prepare-dataset"
        >
            <div className="data-studio-prepare__header">
                <div>
                    <p className="data-studio-prepare__eyebrow">Data Prep</p>
                    <h3>Prepare Dataset</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-prepare__actions">
                    <span className={`data-studio-prepare__verdict data-studio-prepare__verdict--${prepare.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-prepare__refresh"
                        onClick={() => void loadPrepare()}
                        aria-label="Refresh Data Studio prepare checks"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-prepare__metrics" aria-label="Prepare dataset metrics">
                <div className="data-studio-prepare__metric">
                    <Database size={18} aria-hidden="true" />
                    <span>Trainable rows</span>
                    <strong>{formatNumber(prepare.inclusion.trainable_rows)}</strong>
                </div>
                <div className="data-studio-prepare__metric">
                    <ClipboardCheck size={18} aria-hidden="true" />
                    <span>Mapping contract</span>
                    <strong>{prepare.mapping.contract_pass ? 'Pass' : 'Needs review'}</strong>
                </div>
                <div className="data-studio-prepare__metric">
                    <GitBranch size={18} aria-hidden="true" />
                    <span>Prepared rows</span>
                    <strong>{formatNumber(prepare.splits.total_prepared_rows)}</strong>
                </div>
                <div className="data-studio-prepare__metric">
                    <ShieldCheck size={18} aria-hidden="true" />
                    <span>Review pending</span>
                    <strong>{formatNumber(openReviewCount)}</strong>
                </div>
            </div>

            <div className="data-studio-prepare__signals">
                <span>{selectedRecipe?.name || 'No recipe'}</span>
                <span>{labelForToken(prepare.splits.status)} splits</span>
                <span>{labelForToken(prepare.manifest.status)} manifest</span>
                <span>{formatPercent(prepare.mapping.mapping_success_rate)} mapping success</span>
                {prepare.read_only ? <span>Read-only check</span> : null}
            </div>

            <div className="data-studio-prepare__entrypoints">
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => onOpenTarget(prepare.entry_point.target_tab)}
                >
                    <ExternalLink size={15} aria-hidden="true" />
                    {prepare.entry_point.label}
                </button>
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => onOpenTarget('dataprep')}
                >
                    <FileText size={15} aria-hidden="true" />
                    Adapter preview
                </button>
            </div>

            <div className="data-studio-prepare__body">
                <div className="data-studio-prepare__checks">
                    <h4>Readiness checks</h4>
                    <div className="data-studio-prepare__check-list">
                        {prepare.checks.map((check) => (
                            <CheckRow check={check} key={check.id} onOpenTarget={onOpenTarget} />
                        ))}
                    </div>
                </div>

                <div className="data-studio-prepare__splits">
                    <h4>Split artifacts</h4>
                    <div className="data-studio-prepare__split-list">
                        {prepare.splits.items.map((split) => (
                            <SplitRow split={split} key={split.key} />
                        ))}
                    </div>
                </div>
            </div>

            <div className="data-studio-prepare__inclusion">
                <div>
                    <h4>Included data</h4>
                    <dl>
                        <div>
                            <dt>Cleaned</dt>
                            <dd>{formatNumber(prepare.inclusion.cleaned_rows)}</dd>
                        </div>
                        <div>
                            <dt>Gold</dt>
                            <dd>{formatNumber(prepare.inclusion.gold_rows)}</dd>
                        </div>
                        <div>
                            <dt>Accepted synthetic</dt>
                            <dd>{formatNumber(prepare.inclusion.synthetic_accepted)}</dd>
                        </div>
                        <div>
                            <dt>Pending synthetic</dt>
                            <dd>{formatNumber(prepare.inclusion.synthetic_pending)}</dd>
                        </div>
                    </dl>
                </div>
                <div>
                    <h4>Review blockers</h4>
                    {prepare.review_blockers.length > 0 ? (
                        <div className="data-studio-prepare__blocker-list">
                            {prepare.review_blockers.map((blocker) => (
                                <button
                                    type="button"
                                    className={`data-studio-prepare__blocker data-studio-prepare__blocker--${blocker.severity}`}
                                    key={blocker.id}
                                    onClick={() => onOpenTarget(blocker.target_tab)}
                                >
                                    <span>{issueIcon({ ...blocker, title: blocker.label, action_label: '', target_tab: blocker.target_tab })}</span>
                                    <span>
                                        <strong>{blocker.label}</strong>
                                        <small>{blocker.message}</small>
                                    </span>
                                    <b>{formatNumber(blocker.count)}</b>
                                </button>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-prepare__empty">Review gates are clear.</p>
                    )}
                </div>
            </div>

            {topIssues.length > 0 ? (
                <ul className="data-studio-prepare__issues">
                    {topIssues.map((issue) => (
                        <li key={issue.id} className={`data-studio-prepare__issue data-studio-prepare__issue--${issue.severity}`}>
                            <span>{issueIcon(issue)}</span>
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

            <details className="data-studio-prepare__details">
                <summary>Power details</summary>
                <pre>{compactJson(prepare)}</pre>
            </details>
        </section>
    );
}
