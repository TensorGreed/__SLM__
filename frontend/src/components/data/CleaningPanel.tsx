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

export default function CleaningPanel({ projectId, onNextStep }: CleaningPanelProps) {
    const [documents, setDocuments] = useState<DocToClean[]>([]);
    const [chunkSize, setChunkSize] = useState(1000);
    const [redactPii, setRedactPii] = useState(true);
    const [cleaningResults, setCleaningResults] = useState<any[]>([]);

    const fetchDocs = useCallback(async () => {
        const res = await api.get(`/projects/${projectId}/ingestion/documents`);
        setDocuments(res.data);
    }, [projectId]);

    useEffect(() => {
        fetchDocs();
    }, [fetchDocs]);

    const handleCleanAll = async () => {
        const pendingDocs = documents.filter(d => d.status === 'accepted' || d.status === 'pending');
        if (!pendingDocs.length) return;

        try {
            const res = await api.post(`/projects/${projectId}/cleaning/clean-batch`, {
                document_ids: pendingDocs.map(d => d.id),
                chunk_size: chunkSize,
                redact_pii: redactPii,
            });
            setCleaningResults(res.data.results || []);
            fetchDocs();
        } catch (err) {
            console.error('Cleaning failed', err);
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
                </div>
                <button className="btn btn-primary" onClick={handleCleanAll} disabled={!documents.length}>
                    🧹 Clean All Documents
                </button>
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
                                    <div>{r.original_chars?.toLocaleString()} → {r.cleaned_chars?.toLocaleString()} chars</div>
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
                    isComplete={cleaningResults.length > 0}
                    hint="Run cleaning on your documents to proceed"
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
