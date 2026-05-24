import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    Database,
    FileText,
    RefreshCw,
} from 'lucide-react';

import {
    getDataStudioSources,
} from '../../api/dataStudio';
import type {
    DataStudioDatasetGroup,
    DataStudioSources,
} from '../../api/dataStudio';
import './DataStudioSourcesSummaryPanel.css';

interface DataStudioSourcesSummaryPanelProps {
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

const SOURCE_VERDICT_COPY: Record<DataStudioSources['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No sources',
        detail: 'Add files or import a dataset to begin.',
    },
    attention: {
        label: 'Needs attention',
        detail: 'Some source records need review before training.',
    },
    healthy: {
        label: 'Healthy',
        detail: 'Sources are connected and ready for the next data step.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatBytes(bytes: number): string {
    if (!bytes) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function labelForDatasetType(type: string): string {
    return DATASET_TYPE_LABELS[type] || type.replace(/_/g, ' ');
}

function topDatasetGroups(groups: DataStudioDatasetGroup[]): DataStudioDatasetGroup[] {
    return [...groups]
        .sort((a, b) => {
            const rowDelta = Number(b.row_count || 0) - Number(a.row_count || 0);
            if (rowDelta !== 0) return rowDelta;
            return a.dataset_type.localeCompare(b.dataset_type);
        })
        .slice(0, 5);
}

export default function DataStudioSourcesSummaryPanel({
    projectId,
}: DataStudioSourcesSummaryPanelProps) {
    const [sources, setSources] = useState<DataStudioSources | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadSources = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioSources(projectId);
            setSources(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load source summary.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadSources();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const visibleGroups = useMemo(
        () => topDatasetGroups(sources?.dataset_groups ?? []),
        [sources],
    );

    if (loading && !sources) {
        return (
            <section className="data-studio-sources data-studio-sources--loading">
                <span>Loading source summary...</span>
            </section>
        );
    }

    if (error && !sources) {
        return (
            <section className="data-studio-sources data-studio-sources--error">
                <div>
                    <h3>Sources summary</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadSources()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!sources) {
        return null;
    }

    const verdict = SOURCE_VERDICT_COPY[sources.verdict];
    const hasRecentDocuments = sources.recent_documents.length > 0;

    return (
        <section
            className={`data-studio-sources data-studio-sources--${sources.verdict}`}
            data-testid="data-studio-sources"
        >
            <div className="data-studio-sources__header">
                <div>
                    <p className="data-studio-sources__eyebrow">Sources</p>
                    <h3>Source health</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-sources__actions">
                    <span className={`data-studio-sources__verdict data-studio-sources__verdict--${sources.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-sources__refresh"
                        onClick={() => void loadSources()}
                        aria-label="Refresh source summary"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-sources__metrics" aria-label="Source summary metrics">
                <div className="data-studio-sources__metric">
                    <Database size={18} aria-hidden="true" />
                    <span>Datasets</span>
                    <strong>{formatNumber(sources.totals.dataset_count)}</strong>
                </div>
                <div className="data-studio-sources__metric">
                    <FileText size={18} aria-hidden="true" />
                    <span>Source docs</span>
                    <strong>{formatNumber(sources.totals.document_count)}</strong>
                </div>
                <div className="data-studio-sources__metric">
                    <span>Total rows</span>
                    <strong>{formatNumber(sources.totals.row_count)}</strong>
                </div>
                <div className="data-studio-sources__metric">
                    <span>Problems</span>
                    <strong>{formatNumber(sources.totals.error_documents)}</strong>
                </div>
            </div>

            <div className="data-studio-sources__body">
                <div className="data-studio-sources__groups">
                    <h4>Dataset types</h4>
                    {visibleGroups.length > 0 ? (
                        <div className="data-studio-sources__group-list">
                            {visibleGroups.map((group) => (
                                <div className="data-studio-sources__group" key={group.dataset_type}>
                                    <span>{labelForDatasetType(group.dataset_type)}</span>
                                    <strong>{formatNumber(group.row_count)} rows</strong>
                                    <small>
                                        {formatNumber(group.dataset_count)} dataset{group.dataset_count === 1 ? '' : 's'}
                                    </small>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-sources__empty">
                            No dataset records yet. Use the import controls below to add sources.
                        </p>
                    )}
                </div>

                <div className="data-studio-sources__recent">
                    <h4>Recent source documents</h4>
                    {hasRecentDocuments ? (
                        <ul>
                            {sources.recent_documents.slice(0, 5).map((doc) => (
                                <li key={doc.id}>
                                    <span className={`data-studio-sources__status data-studio-sources__status--${doc.status}`}>
                                        {doc.status === 'accepted' ? (
                                            <CheckCircle2 size={14} aria-hidden="true" />
                                        ) : (
                                            <AlertTriangle size={14} aria-hidden="true" />
                                        )}
                                        {doc.status}
                                    </span>
                                    <span className="data-studio-sources__doc-main">
                                        <strong>{doc.filename}</strong>
                                        <small>
                                            {labelForDatasetType(doc.dataset_type)}
                                            {' · '}
                                            {doc.file_type || 'file'}
                                            {' · '}
                                            {formatBytes(doc.file_size_bytes)}
                                        </small>
                                    </span>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="data-studio-sources__empty">
                            Source documents will appear here after upload or remote import.
                        </p>
                    )}
                </div>
            </div>
        </section>
    );
}
