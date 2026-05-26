/**
 * Panel for async batch document cleaning with progress tracking and quality scoring.
 */

import { useState, useCallback, useEffect } from 'react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import CoachStrip from '../coach/CoachStrip';
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

interface CleaningTaskStatus {
    task_id: string;
    project_id: number;
    status: 'pending' | 'running' | 'completed' | 'failed' | string;
    total: number;
    completed: number;
    current_document_id: number | null;
    results: CleaningResult[];
    errors: CleaningBatchError[];
    error: string | null;
    started_at: string;
    updated_at: string;
    finished_at: string | null;
}

// Long-running cleaning jobs (e.g. 100K-row HF imports cleaning a
// multi-megabyte extracted text dump) blow past the dev proxy's
// 10-minute timeout, so we use the backgrounded variant:
//   POST .../clean-batch-async   → returns task_id (202)
//   GET  .../cleaning/tasks/{id} → progress + final results
// The request never stays open long enough for the proxy to sever it.
const TASK_POLL_INTERVAL_MS = 1500;

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
        setCleaningStatus(`Starting cleaning for ${acceptedDocs.length} document(s)…`);
        setCleaningErrors([]);
        setCleaningResults([]);
        try {
            // Start the background task — returns 202 within ms.
            const start = await api.post<CleaningTaskStatus>(
                `/projects/${projectId}/cleaning/clean-batch-async`,
                {
                    document_ids: acceptedDocs.map((d) => d.id),
                    chunk_size: chunkSize,
                    redact_pii: redactPii,
                    redact_toxicity: redactToxicity,
                },
            );
            const taskId = start.data.task_id;

            // Poll for progress. We update setCleaningStatus on every
            // poll so the user sees movement; on completion or failure
            // we materialize the final results + errors and stop.
            // eslint-disable-next-line no-constant-condition
            while (true) {
                await new Promise((resolve) =>
                    setTimeout(resolve, TASK_POLL_INTERVAL_MS),
                );
                const poll = await api.get<CleaningTaskStatus>(
                    `/projects/${projectId}/cleaning/tasks/${taskId}`,
                );
                const t = poll.data;

                if (t.status === 'running' || t.status === 'pending') {
                    const where =
                        t.current_document_id !== null
                            ? ` (currently #${t.current_document_id})`
                            : '';
                    setCleaningStatus(
                        `Cleaning ${t.completed}/${t.total} document(s)${where}…`,
                    );
                    continue;
                }

                // Terminal state: completed or failed.
                setCleaningResults(t.results || []);
                setCleaningErrors(t.errors || []);

                if (t.status === 'failed') {
                    setCleaningStatus(
                        `Cleaning failed: ${t.error ?? 'unknown error'}`,
                    );
                } else {
                    let summary = `Cleaned ${(t.results || []).length}/${acceptedDocs.length} accepted document(s)`;
                    if ((t.errors || []).length > 0) {
                        summary += ` with ${t.errors.length} error(s)`;
                    }
                    if (skippedDocs > 0) {
                        summary += ` (${skippedDocs} non-accepted skipped)`;
                    }
                    setCleaningStatus(summary);
                }
                await fetchDocs();
                break;
            }
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
                <CoachStrip projectId={projectId} stage="cleaning" />
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
