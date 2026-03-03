import { useState, useCallback } from 'react';
import type { RawDocument, DocumentStatus } from '../../types';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import './IngestionPanel.css';

interface IngestionPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

type SourceTab = 'upload' | 'huggingface' | 'kaggle' | 'url';

export default function IngestionPanel({ projectId, onNextStep }: IngestionPanelProps) {
    const [documents, setDocuments] = useState<RawDocument[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const [uploadProgress, setUploadProgress] = useState<string>('');
    const [loaded, setLoaded] = useState(false);

    // Remote import state
    const [activeTab, setActiveTab] = useState<SourceTab>('upload');
    const [remoteId, setRemoteId] = useState('');
    const [remoteSplit, setRemoteSplit] = useState('train');
    const [remoteMaxSamples, setRemoteMaxSamples] = useState('');
    const [isImporting, setIsImporting] = useState(false);
    const [importStatus, setImportStatus] = useState('');

    const fetchDocs = useCallback(async () => {
        setIsLoading(true);
        try {
            const res = await api.get(`/projects/${projectId}/ingestion/documents`);
            setDocuments(res.data);
        } catch (err) {
            console.error('Failed to fetch documents', err);
        } finally {
            setIsLoading(false);
            setLoaded(true);
        }
    }, [projectId]);

    // Load on first render
    if (!loaded && !isLoading) {
        fetchDocs();
    }

    const handleUpload = async (files: FileList | File[]) => {
        if (!files.length) return;
        setIsUploading(true);
        setUploadProgress(`Uploading ${files.length} file(s)...`);

        const formData = new FormData();
        for (const file of Array.from(files)) {
            formData.append('files', file);
        }

        try {
            const res = await api.post(`/projects/${projectId}/ingestion/upload-batch`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setUploadProgress(`Uploaded ${res.data.uploaded} file(s)${res.data.errors.length ? `, ${res.data.errors.length} error(s)` : ''}`);
            fetchDocs();
        } catch (err) {
            setUploadProgress('Upload failed');
            console.error(err);
        } finally {
            setIsUploading(false);
            setTimeout(() => setUploadProgress(''), 3000);
        }
    };

    const handleRemoteImport = async () => {
        if (!remoteId.trim()) return;
        setIsImporting(true);
        setImportStatus('Connecting to source...');
        try {
            const res = await api.post(`/projects/${projectId}/ingestion/import-remote`, {
                source_type: activeTab,
                identifier: remoteId,
                split: remoteSplit,
                max_samples: remoteMaxSamples ? parseInt(remoteMaxSamples) : null,
            });
            setImportStatus(`Imported ${res.data.samples_ingested} samples from ${res.data.source_type}:${res.data.identifier}`);
            setRemoteId('');
            fetchDocs();
        } catch (err) {
            setImportStatus('Import failed. Check identifier and try again.');
            console.error(err);
        } finally {
            setIsImporting(false);
            setTimeout(() => setImportStatus(''), 5000);
        }
    };

    const handleProcess = async (docId: number) => {
        try {
            await api.post(`/projects/${projectId}/ingestion/documents/${docId}/process`);
            fetchDocs();
        } catch (err) {
            console.error('Processing failed', err);
        }
    };

    const handleDelete = async (docId: number) => {
        if (!confirm('Delete this document?')) return;
        try {
            await api.delete(`/projects/${projectId}/ingestion/documents/${docId}`);
            setDocuments((prev) => prev.filter((d) => d.id !== docId));
        } catch (err) {
            console.error('Delete failed', err);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        handleUpload(e.dataTransfer.files);
    };

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            handleUpload(e.target.files);
            e.target.value = '';
        }
    };

    const statusBadgeClass = (status: DocumentStatus) => {
        switch (status) {
            case 'accepted': return 'badge-success';
            case 'processing': return 'badge-info';
            case 'pending': return 'badge-warning';
            case 'error': return 'badge-error';
            case 'rejected': return 'badge-error';
            default: return 'badge-info';
        }
    };

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const sourceTabConfig: Record<SourceTab, { icon: string; label: string; placeholder: string; help: string }> = {
        upload: { icon: '📤', label: 'Local Files', placeholder: '', help: '' },
        huggingface: { icon: '🤗', label: 'HuggingFace', placeholder: 'e.g. tatsu-lab/alpaca or squad', help: 'Enter a HuggingFace dataset ID. Public datasets are fetched instantly.' },
        kaggle: { icon: '📊', label: 'Kaggle', placeholder: 'e.g. username/dataset-name', help: 'Enter a Kaggle dataset slug. Requires KAGGLE_USERNAME and KAGGLE_KEY env vars on the server.' },
        url: { icon: '🔗', label: 'URL / S3 / GCS', placeholder: 'https://example.com/data.csv', help: 'Paste a direct link to a CSV, JSON, or JSONL file. Supports HTTP, S3, and GCS URIs.' },
    };

    return (
        <div className="ingestion-panel animate-fade-in">
            {/* Source Tabs */}
            <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 'var(--space-md)' }}>
                    {(Object.keys(sourceTabConfig) as SourceTab[]).map(tab => (
                        <button
                            key={tab}
                            className={`btn ${activeTab === tab ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setActiveTab(tab)}
                            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                        >
                            <span>{sourceTabConfig[tab].icon}</span>
                            {sourceTabConfig[tab].label}
                        </button>
                    ))}
                </div>

                {/* Local Upload Zone */}
                {activeTab === 'upload' && (
                    <div
                        className={`upload-zone glass-card ${dragOver ? 'drag-over' : ''}`}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                    >
                        <div className="upload-zone-content">
                            <div className="upload-icon">📤</div>
                            <h3 className="upload-title">Drop files here or click to browse</h3>
                            <p className="upload-subtitle">Supports PDF, DOCX, TXT, Markdown, CSV, JSON, JSONL</p>
                            <input
                                type="file"
                                multiple
                                accept=".pdf,.docx,.txt,.md,.csv,.markdown,.json,.jsonl"
                                onChange={handleFileInput}
                                className="upload-input"
                                id="file-upload"
                            />
                            <label htmlFor="file-upload" className="btn btn-primary upload-btn">
                                {isUploading ? 'Uploading...' : 'Choose Files'}
                            </label>
                        </div>
                        {uploadProgress && (
                            <div className="upload-progress">{uploadProgress}</div>
                        )}
                    </div>
                )}

                {/* Remote Import Form */}
                {activeTab !== 'upload' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', margin: 0 }}>
                            {sourceTabConfig[activeTab].help}
                        </p>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: 'var(--space-md)', alignItems: 'end' }}>
                            <div className="form-group" style={{ margin: 0 }}>
                                <label className="form-label">Dataset Identifier</label>
                                <input
                                    className="input"
                                    value={remoteId}
                                    onChange={e => setRemoteId(e.target.value)}
                                    placeholder={sourceTabConfig[activeTab].placeholder}
                                />
                            </div>
                            {activeTab === 'huggingface' && (
                                <div className="form-group" style={{ margin: 0 }}>
                                    <label className="form-label">Split</label>
                                    <select className="input" value={remoteSplit} onChange={e => setRemoteSplit(e.target.value)} style={{ width: 120 }}>
                                        <option value="train">train</option>
                                        <option value="test">test</option>
                                        <option value="validation">validation</option>
                                    </select>
                                </div>
                            )}
                            <div className="form-group" style={{ margin: 0 }}>
                                <label className="form-label">Max Samples</label>
                                <input
                                    className="input"
                                    type="number"
                                    value={remoteMaxSamples}
                                    onChange={e => setRemoteMaxSamples(e.target.value)}
                                    placeholder="All"
                                    style={{ width: 100 }}
                                />
                            </div>
                        </div>
                        <button
                            className="btn btn-primary"
                            onClick={handleRemoteImport}
                            disabled={isImporting || !remoteId.trim()}
                            style={{ alignSelf: 'flex-start' }}
                        >
                            {isImporting ? 'Importing...' : `Import from ${sourceTabConfig[activeTab].label}`}
                        </button>
                        {importStatus && (
                            <div style={{
                                padding: 'var(--space-sm) var(--space-md)',
                                borderRadius: 'var(--radius-md)',
                                background: importStatus.includes('failed') ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.1)',
                                color: importStatus.includes('failed') ? 'var(--color-error)' : 'var(--color-success)',
                                fontSize: 'var(--font-size-sm)',
                            }}>
                                {importStatus}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Document List */}
            <div className="docs-section">
                <div className="docs-header">
                    <h3 className="docs-title">Ingested Documents</h3>
                    <span className="docs-count badge badge-accent">{documents.length}</span>
                </div>

                {isLoading ? (
                    <div className="docs-loading">
                        {[1, 2, 3].map((i) => (
                            <div key={i} className="skeleton" style={{ height: 56, marginBottom: 8, borderRadius: 10 }} />
                        ))}
                    </div>
                ) : documents.length === 0 ? (
                    <div className="empty-state" style={{ padding: '2rem' }}>
                        <div className="empty-state-icon">📂</div>
                        <div className="empty-state-title">No documents yet</div>
                        <div className="empty-state-text">Upload files or import from HuggingFace, Kaggle, or a URL above.</div>
                    </div>
                ) : (
                    <div className="docs-table-container">
                        <table className="docs-table">
                            <thead>
                                <tr>
                                    <th>Filename</th>
                                    <th>Source</th>
                                    <th>Type</th>
                                    <th>Size</th>
                                    <th>Status</th>
                                    <th>Date</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {documents.map((doc) => (
                                    <tr key={doc.id} className="doc-row">
                                        <td className="doc-name">{doc.filename}</td>
                                        <td><span className="badge badge-info" style={{ fontSize: 10 }}>{(doc as any).source || 'upload'}</span></td>
                                        <td><span className="badge badge-accent">{doc.file_type}</span></td>
                                        <td className="doc-size">{formatSize(doc.file_size_bytes)}</td>
                                        <td><span className={`badge ${statusBadgeClass(doc.status)}`}>{doc.status}</span></td>
                                        <td className="doc-date">{new Date(doc.ingested_at).toLocaleDateString()}</td>
                                        <td className="doc-actions">
                                            {doc.status === 'pending' && (
                                                <button className="btn btn-ghost btn-sm" onClick={() => handleProcess(doc.id)} title="Process">
                                                    ⚙️
                                                </button>
                                            )}
                                            <button className="btn btn-ghost btn-sm" onClick={() => handleDelete(doc.id)} title="Delete">
                                                🗑️
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {onNextStep && (
                <StepFooter
                    currentStep="Data Ingestion"
                    nextStep="Data Cleaning"
                    nextStepIcon="🧹"
                    isComplete={documents.length > 0}
                    hint="Upload files or import a dataset to continue"
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
