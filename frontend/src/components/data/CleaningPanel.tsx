import { useState, useCallback, useEffect } from 'react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import './CleaningPanel.css';

interface CleaningPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

interface DocToClean {
    id: number;
    filename: string;
    status: string;
    quality_score: number | null;
    chunk_count: number;
}

interface CleaningResult {
    document_id: number;
    quality_score: number;
    pii_findings?: unknown[];
    toxicity_findings?: unknown[];
    chunk_count: number;
    original_chars: number;
    cleaned_chars: number;
}

interface CleaningBatchError {
    document_id: number;
    error: string;
}

interface CleaningBatchResponse {
    cleaned: number;
    errors: CleaningBatchError[];
    results: CleaningResult[];
}

function extractErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
        if (Array.isArray(detail)) {
            return detail
                .map((item) => {
                    if (typeof item === 'string') return item;
                    if (typeof item === 'object' && item !== null) {
                        const msg = (item as { msg?: unknown }).msg;
                        return typeof msg === 'string' ? msg : '';
                    }
                    return '';
                })
                .filter(Boolean)
                .join('; ');
        }
    }
    if (error instanceof Error) return error.message;
    return 'Operation failed';
}

export default function CleaningPanel({ projectId, onNextStep }: CleaningPanelProps) {
    const [documents, setDocuments] = useState<DocToClean[]>([]);
    const [chunkSize, setChunkSize] = useState(1000);
    const [redactPii, setRedactPii] = useState(true);
    const [redactToxicity, setRedactToxicity] = useState(false);
    const [cleaningResults, setCleaningResults] = useState<CleaningResult[]>([]);
    const [cleaningErrors, setCleaningErrors] = useState<CleaningBatchError[]>([]);
    const [isCleaning, setIsCleaning] = useState(false);
    const [cleaningStatus, setCleaningStatus] = useState('');

    const fetchDocs = useCallback(async () => {
        const res = await api.get(`/projects/${projectId}/ingestion/documents`);
        setDocuments(res.data);
    }, [projectId]);

    useEffect(() => {
        fetchDocs();
    }, [fetchDocs]);

    const handleCleanAll = async () => {
        const acceptedDocs = documents.filter((d) => d.status === 'accepted');
        const skippedDocs = documents.length - acceptedDocs.length;
        if (!acceptedDocs.length) {
            setCleaningStatus('No accepted documents found. Process pending documents in Ingestion first.');
            setCleaningErrors([]);
            setCleaningResults([]);
            return;
        }

        setIsCleaning(true);
        setCleaningStatus(`Cleaning ${acceptedDocs.length} document(s)...`);
        setCleaningErrors([]);
        try {
            const res = await api.post<CleaningBatchResponse>(`/projects/${projectId}/cleaning/clean-batch`, {
                document_ids: acceptedDocs.map((d) => d.id),
                chunk_size: chunkSize,
                redact_pii: redactPii,
                redact_toxicity: redactToxicity,
            });
            const cleaned = res.data.cleaned || 0;
            const errors = res.data.errors || [];
            setCleaningResults(res.data.results || []);
            setCleaningErrors(errors);

            let summary = `Cleaned ${cleaned}/${acceptedDocs.length} accepted document(s)`;
            if (errors.length > 0) {
                summary += ` with ${errors.length} error(s)`;
            }
            if (skippedDocs > 0) {
                summary += ` (${skippedDocs} non-accepted skipped)`;
            }
            setCleaningStatus(summary);
            await fetchDocs();
        } catch (err) {
            setCleaningStatus(`Cleaning failed: ${extractErrorMessage(err)}`);
            console.error('Cleaning failed', err);
        } finally {
            setIsCleaning(false);
        }
    };

    return (
        <div className="cleaning-panel animate-fade-in">
            <div className="card cleaning-config">
                <h3>Cleaning Configuration</h3>
                <div className="config-grid">
                    <div className="form-group">
                        <label className="form-label">Chunk Size (chars)</label>
                        <input className="input" type="number" value={chunkSize} onChange={e => setChunkSize(+e.target.value)} min={100} max={10000} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">
                            <input type="checkbox" checked={redactPii} onChange={e => setRedactPii(e.target.checked)} />
                            {' '}Redact PII & Secrets
                        </label>
                    </div>
                    <div className="form-group">
                        <label className="form-label">
                            <input
                                type="checkbox"
                                checked={redactToxicity}
                                onChange={e => setRedactToxicity(e.target.checked)}
                            />
                            {' '}Mask Toxic Language
                        </label>
                    </div>
                </div>
                <button className="btn btn-primary" onClick={handleCleanAll} disabled={!documents.length}>
                    {isCleaning ? 'Cleaning...' : '🧹 Clean All Documents'}
                </button>
                {cleaningStatus && (
                    <div style={{ marginTop: 'var(--space-sm)', fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
                        {cleaningStatus}
                    </div>
                )}
            </div>

            {cleaningResults.length > 0 && (
                <div className="card">
                    <h3>Cleaning Results</h3>
                    <div className="results-grid">
                        {cleaningResults.map((r, i) => (
                            <div key={i} className="result-card">
                                <div className="result-header">
                                    <span className="badge badge-success">Doc #{r.document_id}</span>
                                    <span className="quality-score">Quality: {(r.quality_score * 100).toFixed(0)}%</span>
                                </div>
                                <div className="result-stats">
                                    <div><strong>{r.chunk_count}</strong> chunks</div>
                                    <div><strong>{r.pii_findings?.length || 0}</strong> PII found</div>
                                    <div><strong>{r.toxicity_findings?.length || 0}</strong> toxicity spans</div>
                                    <div>{r.original_chars?.toLocaleString()} → {r.cleaned_chars?.toLocaleString()} chars</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {cleaningErrors.length > 0 && (
                <div className="card">
                    <h3>Cleaning Errors</h3>
                    <div className="results-grid">
                        {cleaningErrors.map((err) => (
                            <div key={`${err.document_id}-${err.error}`} className="result-card">
                                <div className="result-header">
                                    <span className="badge badge-error">Doc #{err.document_id}</span>
                                </div>
                                <div className="result-stats">
                                    <div>{err.error}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {onNextStep && (
                <StepFooter
                    currentStep="Data Cleaning"
                    nextStep="Gold Dataset"
                    nextStepIcon="🏆"
                    isComplete={cleaningResults.length > 0 && cleaningErrors.length === 0}
                    hint="Run cleaning on your documents to proceed"
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
