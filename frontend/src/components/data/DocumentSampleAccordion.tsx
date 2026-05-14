/**
 * Inline 10-random-rows preview for the Ingested Documents table.
 *
 * Rendered as an extra <tr> below its parent row when the user
 * expands a document. Fetches once on first expand + caches in
 * local state — collapsing keeps the rows around so a second
 * expand is instant. "Refresh sample" re-rolls the reservoir on
 * demand.
 *
 * The endpoint streams a reservoir sample (O(file_size) time,
 * O(n) memory) so even a 100K-row HF import comes back in well
 * under a second.
 */

import { useCallback, useEffect, useState } from 'react';
import api from '../../api/client';

interface DocumentSampleResponse {
    document_id: number;
    filename: string;
    rows: Array<Record<string, unknown>>;
    total_rows_scanned: number;
    source: 'raw' | 'chunks' | 'missing' | string;
    file_type?: string;
    note?: string;
}

interface DocumentSampleAccordionProps {
    projectId: number;
    documentId: number;
    /** Width of the parent table — wires <tr colSpan> so the
     *  expanded row spans every column. */
    colSpan: number;
}

function formatRow(row: Record<string, unknown>): string {
    try {
        return JSON.stringify(row, null, 2);
    } catch {
        return String(row);
    }
}

function extractErrorMessage(err: unknown): string {
    if (typeof err === 'object' && err !== null) {
        const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data
            ?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
        const message = (err as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) {
            return message;
        }
    }
    return 'Failed to load sample.';
}

export default function DocumentSampleAccordion({
    projectId,
    documentId,
    colSpan,
}: DocumentSampleAccordionProps) {
    const [sample, setSample] = useState<DocumentSampleResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string>('');

    const fetchSample = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const resp = await api.get<DocumentSampleResponse>(
                `/projects/${projectId}/ingestion/documents/${documentId}/sample`,
            );
            setSample(resp.data);
        } catch (err) {
            setError(extractErrorMessage(err));
        } finally {
            setLoading(false);
        }
    }, [projectId, documentId]);

    useEffect(() => {
        // Lazy-load on first expand; caller controls when this
        // component mounts via its expanded state.
        if (sample === null && !loading && !error) {
            void fetchSample();
        }
    }, [sample, loading, error, fetchSample]);

    return (
        <tr
            className="doc-sample-row"
            data-testid={`doc-sample-row-${documentId}`}
        >
            <td colSpan={colSpan} style={{ padding: 'var(--space-md)', background: 'var(--bg-secondary)' }}>
                <div
                    style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'baseline',
                        marginBottom: 'var(--space-sm)',
                    }}
                >
                    <strong style={{ fontSize: '0.85rem' }}>
                        Random sample (up to 10 rows)
                        {sample && sample.total_rows_scanned > 0 && (
                            <span
                                style={{
                                    fontWeight: 400,
                                    marginLeft: 8,
                                    color: 'var(--text-secondary)',
                                }}
                            >
                                from {sample.total_rows_scanned.toLocaleString()}{' '}
                                total{sample.source === 'chunks' ? ' (post-cleaning)' : ''}
                            </span>
                        )}
                    </strong>
                    <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => void fetchSample()}
                        disabled={loading}
                        data-testid={`refresh-sample-${documentId}`}
                    >
                        {loading ? 'Loading…' : 'Refresh sample'}
                    </button>
                </div>
                {error && (
                    <div className="error-banner" data-testid={`sample-error-${documentId}`}>
                        {error}
                    </div>
                )}
                {!error && sample && sample.note && sample.rows.length === 0 && (
                    <div
                        style={{
                            color: 'var(--text-secondary)',
                            fontSize: '0.9rem',
                            fontStyle: 'italic',
                        }}
                        data-testid={`sample-note-${documentId}`}
                    >
                        {sample.note}
                    </div>
                )}
                {!error && sample && sample.rows.length > 0 && (
                    <pre
                        style={{
                            background: 'var(--bg-primary, #0c0c0c)',
                            padding: 'var(--space-sm)',
                            borderRadius: 'var(--radius-sm)',
                            border: '1px solid var(--border-color)',
                            fontSize: '0.8rem',
                            maxHeight: 320,
                            overflow: 'auto',
                            margin: 0,
                        }}
                    >
                        {sample.rows.map(formatRow).join('\n\n')}
                    </pre>
                )}
                {!error && !sample && loading && (
                    <div
                        style={{
                            color: 'var(--text-secondary)',
                            fontSize: '0.9rem',
                            fontStyle: 'italic',
                        }}
                    >
                        Loading sample…
                    </div>
                )}
            </td>
        </tr>
    );
}
