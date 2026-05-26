/**
 * Panel showing sampled row mapping preview against the active recipe with required-field coverage.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    Code2,
    ExternalLink,
    GitBranch,
    RefreshCw,
} from 'lucide-react';

import {
    getDataStudioMappingPreview,
} from '../../api/dataStudio';
import type {
    DataStudioMappingPreview,
    DataStudioMappingTemplate,
    DataStudioRequiredFieldCoverage,
} from '../../api/dataStudio';
import './DataStudioMappingPreviewPanel.css';

interface DataStudioMappingPreviewPanelProps {
    projectId: number;
    onOpenTarget?: (target: string) => void;
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

function templateStatusLabel(status: string): string {
    if (status === 'ready') return 'Ready';
    if (status === 'missing') return 'Missing fields';
    return 'Needs review';
}

function templateSourceLabel(source: string): string {
    if (source === 'auto_fix') return 'Detected';
    if (source === 'recipe') return 'Recipe';
    if (source === 'adapter') return 'Adapter';
    if (source === 'domain') return 'Domain';
    return source.replace(/_/g, ' ');
}

function fieldStatusLabel(status: string): string {
    if (status === 'applied') return 'Applied';
    if (status === 'available') return 'Available';
    if (status === 'missing') return 'Missing';
    if (status === 'ambiguous') return 'Ambiguous';
    return status.replace(/_/g, ' ');
}

function MappingTemplateCard({
    template,
    onOpenTarget,
}: {
    template: DataStudioMappingTemplate;
    onOpenTarget?: (target: string) => void;
}) {
    const fieldSummary = [
        `${formatNumber(template.summary.applied_count)} applied`,
        `${formatNumber(template.summary.available_count)} available`,
        `${formatNumber(template.summary.missing_count)} missing`,
        `${formatNumber(template.summary.ambiguous_count)} ambiguous`,
    ].join(' · ');

    return (
        <article className={`data-studio-mapping__template data-studio-mapping__template--${template.status}`}>
            <div className="data-studio-mapping__template-head">
                <div>
                    <h5>{template.label}</h5>
                    <p>{template.description}</p>
                </div>
                <div className="data-studio-mapping__template-badges">
                    {template.recommended ? <span>Recommended</span> : null}
                    <span>{templateSourceLabel(template.source)}</span>
                    <span>{templateStatusLabel(template.status)}</span>
                </div>
            </div>
            <div className="data-studio-mapping__template-summary">
                <span>{fieldSummary}</span>
                <b>{formatPercent(template.confidence)} match</b>
            </div>
            {template.fields.length > 0 ? (
                <div className="data-studio-mapping__template-fields">
                    {template.fields.map((field) => (
                        <div
                            className={`data-studio-mapping__template-field data-studio-mapping__template-field--${field.status}`}
                            key={`${template.id}:${field.canonical_field}`}
                        >
                            <span>
                                <strong>{field.canonical_field}</strong>
                                <small>
                                    {field.current_source ? `Current: ${field.current_source}` : 'No saved override'}
                                </small>
                            </span>
                            <code>{field.recommended_source || 'No match'}</code>
                            <em>{fieldStatusLabel(field.status)}</em>
                        </div>
                    ))}
                </div>
            ) : (
                <p className="data-studio-mapping__empty">No field-level template details are available yet.</p>
            )}
            <div className="data-studio-mapping__template-action">
                <p>{template.apply_action.description}</p>
                {onOpenTarget ? (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => onOpenTarget(template.apply_action.target_tab)}
                    >
                        <ExternalLink size={15} aria-hidden="true" />
                        {template.apply_action.label}
                    </button>
                ) : null}
            </div>
        </article>
    );
}

export default function DataStudioMappingPreviewPanel({
    projectId,
    onOpenTarget,
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
    const templates = mapping?.mapping_templates;
    const recommendedTemplate = templates?.templates.find((template) => template.recommended) ?? null;

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

            <div className="data-studio-mapping__templates">
                <div className="data-studio-mapping__templates-head">
                    <div>
                        <h4>Mapping templates</h4>
                        <p>
                            Compare recipe, adapter, domain, and detected templates before saving mapping changes in Data Prep.
                        </p>
                    </div>
                    <div className="data-studio-mapping__template-metrics">
                        <span>{formatNumber(templates?.template_count)} templates</span>
                        <span>{formatNumber(templates?.detected_fields.length ?? 0)} detected fields</span>
                        <span>{templates?.read_only ? 'Read-only' : 'Can mutate'}</span>
                    </div>
                </div>

                {recommendedTemplate ? (
                    <p className="data-studio-mapping__template-guidance">
                        Recommended: <strong>{recommendedTemplate.label}</strong>
                        {' · '}
                        {formatNumber(recommendedTemplate.summary.missing_count)} missing
                        {' · '}
                        {formatNumber(recommendedTemplate.summary.ambiguous_count)} ambiguous
                    </p>
                ) : (
                    <p className="data-studio-mapping__template-guidance">
                        Template recommendations appear after a recipe, adapter, or domain contract is available.
                    </p>
                )}

                {templates?.detected_fields.length ? (
                    <div className="data-studio-mapping__detected-fields" aria-label="Detected source fields">
                        {templates.detected_fields.slice(0, 10).map((field) => (
                            <span key={field.field}>
                                {field.field}
                                <b>{formatNumber(field.count)}</b>
                            </span>
                        ))}
                    </div>
                ) : null}

                {templates?.templates.length ? (
                    <div className="data-studio-mapping__template-list">
                        {templates.templates.slice(0, 4).map((template) => (
                            <MappingTemplateCard
                                key={template.id}
                                template={template}
                                onOpenTarget={onOpenTarget}
                            />
                        ))}
                    </div>
                ) : (
                    <p className="data-studio-mapping__empty">
                        No mapping templates are available for the current recipe and source sample yet.
                    </p>
                )}
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
                        mapping_templates: mapping.mapping_templates,
                    })}
                </pre>
            </details>
        </section>
    );
}
