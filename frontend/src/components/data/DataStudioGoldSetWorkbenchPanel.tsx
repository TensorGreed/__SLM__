/**
 * Panel for gold-set coverage metrics, field alignment checks, and review-state assessment.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    ClipboardCheck,
    ExternalLink,
    ListChecks,
    Lock,
    RefreshCw,
} from 'lucide-react';

import {
    getDataStudioGoldSetWorkbench,
} from '../../api/dataStudio';
import type {
    DataStudioGoldSetFieldCoverage,
    DataStudioGoldSetWorkbench,
} from '../../api/dataStudio';
import Term from '../shared/Term';
import './DataStudioGoldSetWorkbenchPanel.css';

interface DataStudioGoldSetWorkbenchPanelProps {
    projectId: number;
    onOpenGoldSet: () => void;
}

const GOLD_VERDICT_COPY: Record<DataStudioGoldSetWorkbench['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No gold set',
        detail: 'Start a trusted reference set before relying on evals.',
    },
    attention: {
        label: 'Needs review',
        detail: 'Trusted examples exist, but coverage or review state needs attention.',
    },
    ready: {
        label: 'Ready',
        detail: 'Gold Set checks look ready for evaluation and regression tracking.',
    },
};

const DATASET_TYPE_LABELS: Record<string, string> = {
    gold_dev: 'Gold dev',
    gold_test: 'Gold test',
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatPercent(value: number | undefined): string {
    const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
    return `${Math.round(normalized * 100)}%`;
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function labelForDatasetType(type: string | undefined): string {
    if (!type) return 'Gold set';
    return DATASET_TYPE_LABELS[type] || type.replace(/_/g, ' ');
}

function labelForStatus(status: string | undefined): string {
    if (!status) return 'Unknown';
    return status.replace(/_/g, ' ');
}

function coverageLabel(fields: DataStudioGoldSetFieldCoverage[]): string {
    if (fields.length === 0) {
        return 'n/a';
    }
    const minRatio = fields.reduce((acc, item) => Math.min(acc, Number(item.ratio || 0)), 1);
    return formatPercent(minRatio);
}

function issueIcon(severity: string) {
    if (severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

export default function DataStudioGoldSetWorkbenchPanel({
    projectId,
    onOpenGoldSet,
}: DataStudioGoldSetWorkbenchPanelProps) {
    const [goldSet, setGoldSet] = useState<DataStudioGoldSetWorkbench | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadGoldSet = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioGoldSetWorkbench(projectId);
            setGoldSet(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Gold Set summary.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadGoldSet();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topIssues = useMemo(
        () => goldSet?.issues.slice(0, 3) ?? [],
        [goldSet],
    );
    const topDatasets = useMemo(
        () => goldSet?.datasets.slice(0, 2) ?? [],
        [goldSet],
    );
    const trustedSamples = useMemo(
        () => goldSet?.trusted_examples.slice(0, 3) ?? [],
        [goldSet],
    );

    if (loading && !goldSet) {
        return (
            <section className="data-studio-gold data-studio-gold--loading">
                <span>Loading Gold Set summary...</span>
            </section>
        );
    }

    if (error && !goldSet) {
        return (
            <section className="data-studio-gold data-studio-gold--error">
                <div>
                    <h3>Gold Set workbench</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadGoldSet()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!goldSet) {
        return null;
    }

    const verdict = GOLD_VERDICT_COPY[goldSet.verdict];
    const fieldCoverage = coverageLabel([
        ...goldSet.coverage.input_fields,
        ...goldSet.coverage.expected_fields,
    ]);
    const labelCoverage = coverageLabel(goldSet.coverage.label_fields);
    const lockedLabel = goldSet.validation.locked_gold_sets > 0 || goldSet.validation.locked_versions > 0
        ? 'Locked'
        : 'Draft';

    return (
        <section
            className={`data-studio-gold data-studio-gold--${goldSet.verdict}`}
            data-testid="data-studio-gold"
        >
            <div className="data-studio-gold__header">
                <div>
                    <p className="data-studio-gold__eyebrow">Gold Set</p>
                    <h3><Term id="gold_set" advanced /> workbench</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-gold__actions">
                    <span className={`data-studio-gold__verdict data-studio-gold__verdict--${goldSet.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-gold__refresh"
                        onClick={() => void loadGoldSet()}
                        aria-label="Refresh Gold Set summary"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-gold__metrics" aria-label="Gold Set metrics">
                <div className="data-studio-gold__metric">
                    <ClipboardCheck size={18} aria-hidden="true" />
                    <span>Trusted examples</span>
                    <strong>
                        {formatNumber(goldSet.validation.trusted_examples)}
                        {' / '}
                        {formatNumber(goldSet.minimum_recommended_examples)}
                    </strong>
                </div>
                <div className="data-studio-gold__metric">
                    <ListChecks size={18} aria-hidden="true" />
                    <span>Review needed</span>
                    <strong>{formatNumber(goldSet.validation.review_needed)}</strong>
                </div>
                <div className="data-studio-gold__metric">
                    <span>Field coverage</span>
                    <strong>{fieldCoverage}</strong>
                </div>
                <div className="data-studio-gold__metric">
                    <Lock size={18} aria-hidden="true" />
                    <span>Validation state</span>
                    <strong>{lockedLabel}</strong>
                </div>
            </div>

            <div className="data-studio-gold__entry">
                <div>
                    <strong>{goldSet.entry_point.label}</strong>
                    <small>{goldSet.entry_point.reason}</small>
                </div>
                <button type="button" className="btn btn-primary" onClick={onOpenGoldSet}>
                    <ExternalLink size={16} aria-hidden="true" />
                    Open Gold Set workflow
                </button>
            </div>

            <div className="data-studio-gold__body">
                <div className="data-studio-gold__datasets">
                    <h4>Trusted sets</h4>
                    {topDatasets.length > 0 ? (
                        <div className="data-studio-gold__dataset-list">
                            {topDatasets.map((dataset) => (
                                <article className="data-studio-gold__dataset" key={dataset.id}>
                                    <div>
                                        <strong>{dataset.name}</strong>
                                        <small>
                                            {labelForDatasetType(dataset.dataset_type)}
                                            {' · '}
                                            {labelForStatus(dataset.validation_status)}
                                        </small>
                                    </div>
                                    <div className="data-studio-gold__dataset-stats">
                                        <span>{formatNumber(dataset.trusted_examples)} trusted</span>
                                        <span>{formatNumber(dataset.review_needed)} review</span>
                                    </div>
                                    <div className="data-studio-gold__coverage-row">
                                        <span>Expected</span>
                                        <strong>{coverageLabel(dataset.coverage.expected_fields)}</strong>
                                    </div>
                                    <div className="data-studio-gold__coverage-row">
                                        <span>Labels</span>
                                        <strong>{coverageLabel(dataset.coverage.label_fields)}</strong>
                                    </div>
                                </article>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-gold__empty">
                            No Gold Set dataset has been created for this project yet.
                        </p>
                    )}

                    {topIssues.length > 0 ? (
                        <ul className="data-studio-gold__issues">
                            {topIssues.map((issue) => (
                                <li key={issue.id} className={`data-studio-gold__issue data-studio-gold__issue--${issue.severity}`}>
                                    <span>{issueIcon(issue.severity)}</span>
                                    <div>
                                        <strong>{issue.title}</strong>
                                        <small>{issue.message}</small>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="data-studio-gold__empty">
                            No Gold Set review issues are active.
                        </p>
                    )}
                </div>

                <div className="data-studio-gold__samples">
                    <h4>Coverage and examples</h4>
                    <div className="data-studio-gold__coverage-grid">
                        <div>
                            <span>Input fields</span>
                            <strong>{goldSet.coverage.field_counts.input}</strong>
                        </div>
                        <div>
                            <span>Expected fields</span>
                            <strong>{goldSet.coverage.field_counts.expected}</strong>
                        </div>
                        <div>
                            <span>Label fields</span>
                            <strong>
                                {goldSet.coverage.field_counts.labels}
                                {' · '}
                                {labelCoverage}
                            </strong>
                        </div>
                    </div>

                    {trustedSamples.length > 0 ? (
                        <div className="data-studio-gold__sample-list">
                            {trustedSamples.map((sample) => (
                                <article
                                    className="data-studio-gold__sample"
                                    key={`${sample.dataset_id}-${sample.source}-${sample.input_preview}`}
                                >
                                    <strong>{sample.dataset_name}</strong>
                                    <p>{sample.input_preview}</p>
                                    <small>{sample.expected_preview}</small>
                                </article>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-gold__empty">
                            Trusted example previews appear after gold rows are added.
                        </p>
                    )}
                </div>
            </div>

            <details className="data-studio-gold__details">
                <summary>Power details</summary>
                <pre>{compactJson(goldSet)}</pre>
            </details>
        </section>
    );
}
