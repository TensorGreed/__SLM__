import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    Code2,
    GitBranch,
    RefreshCw,
} from 'lucide-react';

import {
    getDataStudioMappingPreview,
} from '../../api/dataStudio';
import type {
    DataStudioMappingPreview,
    DataStudioRequiredFieldCoverage,
} from '../../api/dataStudio';
import './DataStudioMappingPreviewPanel.css';

interface DataStudioMappingPreviewPanelProps {
    projectId: number;
}

const DATASET_TYPE_LABELS: Record<string, string> = {
    raw: 'Raw',
    cleaned: 'Cleaned',
    gold_dev: 'Gold dev',
    gold_test: 'Gold test',
    synthetic: 'Synthetic',
    train: 'Train',
    validation: 'Validation',
    test: 'Test',
};

const MAPPING_VERDICT_COPY: Record<DataStudioMappingPreview['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No preview',
        detail: 'Add rows before checking how they map into the training shape.',
    },
    attention: {
        label: 'Needs attention',
        detail: 'The sample maps, but required fields or adapter settings need review.',
    },
    ready: {
        label: 'Ready',
        detail: 'Sampled rows match the active recipe mapping contract.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatPercent(value: number | undefined): string {
    const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
    return `${Math.round(normalized * 100)}%`;
}

function labelForDatasetType(type: string | undefined): string {
    if (!type) return 'No source';
    return DATASET_TYPE_LABELS[type] || type.replace(/_/g, ' ');
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

function requiredCoverageLabel(fields: DataStudioRequiredFieldCoverage[]): string {
    if (fields.length === 0) {
        return 'n/a';
    }
    const minRatio = fields.reduce((acc, item) => Math.min(acc, Number(item.ratio || 0)), 1);
    return formatPercent(minRatio);
}

export default function DataStudioMappingPreviewPanel({
    projectId,
}: DataStudioMappingPreviewPanelProps) {
    const [mapping, setMapping] = useState<DataStudioMappingPreview | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadMapping = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioMappingPreview(projectId);
            setMapping(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load schema mapping preview.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadMapping();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const requiredCoverage = useMemo(
        () => mapping?.summary.required_field_coverage ?? [],
        [mapping],
    );
    const topIssues = useMemo(
        () => mapping?.issues.slice(0, 4) ?? [],
        [mapping],
    );

    if (loading && !mapping) {
        return (
            <section className="data-studio-mapping data-studio-mapping--loading">
                <span>Loading schema mapping preview...</span>
            </section>
        );
    }

    if (error && !mapping) {
        return (
            <section className="data-studio-mapping data-studio-mapping--error">
                <div>
                    <h3>Schema mapping preview</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadMapping()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!mapping) {
        return null;
    }

    const verdict = MAPPING_VERDICT_COPY[mapping.verdict];
    const adapterId = mapping.effective_mapping.adapter_id || 'default-canonical';
    const taskProfile = mapping.effective_mapping.task_profile || mapping.recipe?.task_profile || 'auto';
    const sourceLabel = mapping.source
        ? `${labelForDatasetType(mapping.source.dataset_type)}${mapping.source.document_name ? ` · ${mapping.source.document_name}` : ''}`
        : 'No previewable source';
    const mappedLabel = `${formatNumber(mapping.summary.mapped_records)} / ${formatNumber(mapping.summary.sampled_records)} mapped`;
    const requiredLabel = requiredCoverageLabel(requiredCoverage);

    return (
        <section
            className={`data-studio-mapping data-studio-mapping--${mapping.verdict}`}
            data-testid="data-studio-mapping"
        >
            <div className="data-studio-mapping__header">
                <div>
                    <p className="data-studio-mapping__eyebrow">Mapping</p>
                    <h3>Schema mapping preview</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-mapping__actions">
                    <span className={`data-studio-mapping__verdict data-studio-mapping__verdict--${mapping.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-mapping__refresh"
                        onClick={() => void loadMapping()}
                        aria-label="Refresh schema mapping preview"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-mapping__metrics" aria-label="Schema mapping metrics">
                <div className="data-studio-mapping__metric">
                    <GitBranch size={18} aria-hidden="true" />
                    <span>Adapter</span>
                    <strong>{adapterId}</strong>
                </div>
                <div className="data-studio-mapping__metric">
                    <span>Task profile</span>
                    <strong>{taskProfile}</strong>
                </div>
                <div className="data-studio-mapping__metric">
                    <span>Mapped rows</span>
                    <strong>{mappedLabel}</strong>
                </div>
                <div className="data-studio-mapping__metric">
                    <span>Required coverage</span>
                    <strong>{requiredLabel}</strong>
                </div>
            </div>

            <div className="data-studio-mapping__source">
                <Code2 size={16} aria-hidden="true" />
                <span>{sourceLabel}</span>
                <small>
                    {mapping.effective_mapping.source} mapping
                    {mapping.preference.field_mapping_count > 0
                        ? ` · ${mapping.preference.field_mapping_count} saved override${mapping.preference.field_mapping_count === 1 ? '' : 's'}`
                        : ''}
                </small>
            </div>

            <div className="data-studio-mapping__body">
                <div className="data-studio-mapping__coverage">
                    <h4>Required fields</h4>
                    {requiredCoverage.length > 0 ? (
                        <div className="data-studio-mapping__coverage-list">
                            {requiredCoverage.map((field) => (
                                <div className="data-studio-mapping__coverage-row" key={field.field}>
                                    <div>
                                        <strong>{field.field}</strong>
                                        <small>
                                            {formatNumber(field.present)} present / {formatNumber(field.missing)} missing
                                        </small>
                                    </div>
                                    <span>{formatPercent(field.ratio)}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-mapping__empty">
                            Required-field coverage appears after BrewSLM can sample rows.
                        </p>
                    )}

                    {topIssues.length > 0 ? (
                        <ul className="data-studio-mapping__issues">
                            {topIssues.map((issue) => (
                                <li key={issue.id} className={`data-studio-mapping__issue data-studio-mapping__issue--${issue.severity}`}>
                                    <span>{issueIcon(issue.severity)}</span>
                                    <div>
                                        <strong>{issue.title}</strong>
                                        <small>{issue.message}</small>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="data-studio-mapping__empty">
                            Required fields are covered on the current sample.
                        </p>
                    )}
                </div>

                <div className="data-studio-mapping__preview">
                    <h4>Canonical preview</h4>
                    {mapping.preview_rows.length > 0 ? (
                        <div className="data-studio-mapping__preview-list">
                            {mapping.preview_rows.map((row) => (
                                <div className="data-studio-mapping__preview-row" key={row.index}>
                                    <span>Row {row.index + 1}</span>
                                    <pre>{compactJson(row.mapped)}</pre>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-mapping__empty">
                            Canonical rows will appear after the adapter maps a sample.
                        </p>
                    )}
                </div>
            </div>

            <details className="data-studio-mapping__details">
                <summary>Power details</summary>
                <pre>
                    {compactJson({
                        preference: mapping.preference,
                        effective_mapping: mapping.effective_mapping,
                        source: mapping.source,
                        diagnostics: mapping.diagnostics,
                    })}
                </pre>
            </details>
        </section>
    );
}
