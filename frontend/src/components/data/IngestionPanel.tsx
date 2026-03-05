import { useState, useCallback, useEffect } from 'react';
import type { RawDocument, DocumentStatus } from '../../types';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { TerminalConsole } from '../shared/TerminalConsole';
import './IngestionPanel.css';

interface IngestionPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

type SourceTab = 'upload' | 'huggingface' | 'kaggle' | 'url';

interface RemoteImportQueueResponse {
    status: string;
    report_path?: string;
    source_type: string;
    identifier: string;
    task_id?: string;
}

interface RemoteImportStatusResponse {
    status: 'running' | 'completed' | 'failed' | string;
    result?: {
        samples_ingested?: number;
        source_type?: string;
        identifier?: string;
    };
    error?: string;
}

function extractErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const detail = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Operation failed';
}

export default function IngestionPanel({ projectId, onNextStep }: IngestionPanelProps) {
    const [documents, setDocuments] = useState<RawDocument[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const [uploadProgress, setUploadProgress] = useState('');

    const [activeTab, setActiveTab] = useState<SourceTab>('upload');
    const [remoteId, setRemoteId] = useState('');
    const [remoteSplit, setRemoteSplit] = useState('train');
    const [remoteMaxSamples, setRemoteMaxSamples] = useState('');
    const [hfToken, setHfToken] = useState('');
    const [kaggleUsername, setKaggleUsername] = useState('');
    const [kaggleKey, setKaggleKey] = useState('');

    const [isImporting, setIsImporting] = useState(false);
    const [importStatus, setImportStatus] = useState('');
    const [importLogs, setImportLogs] = useState<string[]>([]);
    const [activeReportPath, setActiveReportPath] = useState<string | null>(null);

    const fetchDocs = useCallback(async () => {
        setIsLoading(true);
        try {
            const res = await api.get<RawDocument[]>(`/projects/${projectId}/ingestion/documents`);
            setDocuments(res.data);
        } catch (err) {
            console.error('Failed to fetch documents', err);
        } finally {
            setIsLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetchDocs();
    }, [fetchDocs]);

    useEffect(() => {
        if (!activeReportPath) {
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${protocol}://${window.location.host}/api/projects/${projectId}/ingestion/ws/logs`;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload.type === 'log' && typeof payload.text === 'string') {
                    setImportLogs((prev) => [...prev, payload.text]);
                }
            } catch (err) {
                console.error('Failed to parse ingestion log message', err);
            }
        };

        return () => {
            ws.close();
        };
    }, [activeReportPath, projectId]);

    useEffect(() => {
        if (!activeReportPath) {
            return;
        }

        const interval = window.setInterval(async () => {
            try {
                const statusRes = await api.get<RemoteImportStatusResponse>(
                    `/projects/${projectId}/ingestion/imports/status`,
                    { params: { report_path: activeReportPath } },
                );
                const status = statusRes.data;
                if (status.status === 'completed') {
                    const imported = status.result?.samples_ingested ?? 0;
                    const source = status.result?.source_type ?? activeTab;
                    const identifier = status.result?.identifier ?? remoteId;
                    setImportStatus(`Imported ${imported} samples from ${source}:${identifier}`);
                    setIsImporting(false);
                    setActiveReportPath(null);
                    void fetchDocs();
                } else if (status.status === 'failed') {
                    setImportStatus(status.error ? `Import failed: ${status.error}` : 'Import failed. Check logs and retry.');
                    setIsImporting(false);
                    setActiveReportPath(null);
                }
            } catch (err) {
                setImportStatus(`Import polling failed: ${extractErrorMessage(err)}`);
                setIsImporting(false);
                setActiveReportPath(null);
            }
        }, 2000);

        return () => {
            window.clearInterval(interval);
        };
    }, [activeReportPath, activeTab, fetchDocs, projectId, remoteId]);

    const handleUpload = async (files: FileList | File[]) => {
        if (!files.length) return;
        setIsUploading(true);
        setUploadProgress(`Uploading ${files.length} file(s)...`);

        const formData = new FormData();
        for (const file of Array.from(files)) {
            formData.append('files', file);
        }

        try {
            const res = await api.post<{ uploaded: number; errors: Array<{ error: string }> }>(
                `/projects/${projectId}/ingestion/upload-batch`,
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                },
            );
            const errorCount = res.data.errors.length;
            setUploadProgress(
                `Uploaded ${res.data.uploaded} file(s)${errorCount ? `, ${errorCount} error(s)` : ''}`,
            );
            void fetchDocs();
        } catch (err) {
            setUploadProgress(`Upload failed: ${extractErrorMessage(err)}`);
            console.error(err);
        } finally {
            setIsUploading(false);
            setTimeout(() => setUploadProgress(''), 3000);
        }
    };

    const handleRemoteImport = async () => {
        if (!remoteId.trim()) return;

        setIsImporting(true);
        setImportStatus('Queueing remote import job...');
        setImportLogs([]);
        setActiveReportPath(null);

        try {
            const payload: {
                source_type: SourceTab;
                identifier: string;
                split: string;
                max_samples: number | null;
                hf_token?: string;
                kaggle_username?: string;
                kaggle_key?: string;
            } = {
                source_type: activeTab,
                identifier: remoteId.trim(),
                split: remoteSplit,
                max_samples: remoteMaxSamples ? Number(remoteMaxSamples) : null,
            };

            if (activeTab === 'huggingface' && hfToken.trim()) {
                payload.hf_token = hfToken.trim();
            }
            if (activeTab === 'kaggle') {
                if (kaggleUsername.trim()) {
                    payload.kaggle_username = kaggleUsername.trim();
                }
                if (kaggleKey.trim()) {
                    payload.kaggle_key = kaggleKey.trim();
                }
            }

            const res = await api.post<RemoteImportQueueResponse>(
                `/projects/${projectId}/ingestion/import-remote/queue`,
                payload,
            );

            if (res.data.report_path) {
                setActiveReportPath(res.data.report_path);
                setImportStatus(`Import queued (${res.data.task_id ?? 'task'}). Streaming progress...`);
            } else {
                setImportStatus('Import queued but no report path returned.');
                setIsImporting(false);
            }
        } catch (err) {
            const message = extractErrorMessage(err);
            setImportStatus(`Import failed: ${message}`);
            setIsImporting(false);
            console.error(err);
        }
    };

    const handleProcess = async (docId: number) => {
        try {
            await api.post(`/projects/${projectId}/ingestion/documents/${docId}/process`);
            void fetchDocs();
        } catch (err) {
            console.error('Processing failed', err);
        }
    };

    const handleDelete = async (docId: number) => {
        if (!confirm('Delete this document?')) return;
        try {
            await api.delete(`/projects/${projectId}/ingestion/documents/${docId}`);
            setDocuments((prev) => prev.filter((doc) => doc.id !== docId));
        } catch (err) {
            console.error('Delete failed', err);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        void handleUpload(e.dataTransfer.files);
    };

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            void handleUpload(e.target.files);
            e.target.value = '';
        }
    };

    const statusBadgeClass = (status: DocumentStatus) => {
        switch (status) {
            case 'accepted':
                return 'badge-success';
            case 'processing':
                return 'badge-info';
            case 'pending':
                return 'badge-warning';
            case 'error':
                return 'badge-error';
            case 'rejected':
                return 'badge-error';
            default:
                return 'badge-info';
        }
    };

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const sourceTabConfig: Record<SourceTab, { icon: string; label: string; placeholder: string; help: string }> = {
        upload: {
            icon: '📤',
            label: 'Local Files',
            placeholder: '',
            help: '',
        },
        huggingface: {
            icon: '🤗',
            label: 'HuggingFace',
            placeholder: 'e.g. tatsu-lab/alpaca or squad',
            help: 'Enter a HuggingFace dataset ID. Add HF token below for gated/private datasets.',
        },
        kaggle: {
            icon: '📊',
            label: 'Kaggle',
            placeholder: 'e.g. username/dataset-name',
            help: 'Enter a Kaggle dataset slug. Add Kaggle username/key below if not configured on the server.',
        },
        url: {
            icon: '🔗',
            label: 'URL / S3 / GCS',
            placeholder: 'https://example.com/data.csv',
            help: 'Paste a direct link to a CSV, JSON, or JSONL file. Supports HTTP, S3, and GCS URIs.',
        },
    };

    return (
        <div className="ingestion-panel animate-fade-in">
            <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 'var(--space-md)' }}>
                    {(Object.keys(sourceTabConfig) as SourceTab[]).map((tab) => (
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

                {activeTab === 'upload' && (
                    <div
                        className={`upload-zone glass-card ${dragOver ? 'drag-over' : ''}`}
                        onDragOver={(e) => {
                            e.preventDefault();
                            setDragOver(true);
                        }}
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
                        {uploadProgress && <div className="upload-progress">{uploadProgress}</div>}
                    </div>
                )}

                {activeTab !== 'upload' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', margin: 0 }}>
                            {sourceTabConfig[activeTab].help}
                        </p>
                        <div
                            style={{
                                display: 'grid',
                                gridTemplateColumns: '1fr auto auto',
                                gap: 'var(--space-md)',
                                alignItems: 'end',
                            }}
                        >
                            <div className="form-group" style={{ margin: 0 }}>
                                <label className="form-label">Dataset Identifier</label>
                                <input
                                    className="input"
                                    value={remoteId}
                                    onChange={(e) => setRemoteId(e.target.value)}
                                    placeholder={sourceTabConfig[activeTab].placeholder}
                                />
                            </div>
                            {activeTab === 'huggingface' && (
                                <div className="form-group" style={{ margin: 0 }}>
                                    <label className="form-label">Split</label>
                                    <select
                                        className="input"
                                        value={remoteSplit}
                                        onChange={(e) => setRemoteSplit(e.target.value)}
                                        style={{ width: 120 }}
                                    >
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
                                    onChange={(e) => setRemoteMaxSamples(e.target.value)}
                                    placeholder="All"
                                    style={{ width: 100 }}
                                />
                            </div>
                        </div>

                        {activeTab === 'huggingface' && (
                            <div className="form-group" style={{ margin: 0 }}>
                                <label className="form-label">HuggingFace Token (optional)</label>
                                <input
                                    className="input"
                                    type="password"
                                    value={hfToken}
                                    onChange={(e) => setHfToken(e.target.value)}
                                    placeholder="hf_..."
                                />
                                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', marginTop: 4 }}>
                                    Used only for this import request. Not saved to project metadata.
                                </div>
                            </div>
                        )}

                        {activeTab === 'kaggle' && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-md)' }}>
                                <div className="form-group" style={{ margin: 0 }}>
                                    <label className="form-label">Kaggle Username (optional)</label>
                                    <input
                                        className="input"
                                        value={kaggleUsername}
                                        onChange={(e) => setKaggleUsername(e.target.value)}
                                        placeholder="your_kaggle_username"
                                    />
                                </div>
                                <div className="form-group" style={{ margin: 0 }}>
                                    <label className="form-label">Kaggle Key (optional)</label>
                                    <input
                                        className="input"
                                        type="password"
                                        value={kaggleKey}
                                        onChange={(e) => setKaggleKey(e.target.value)}
                                        placeholder="kaggle_api_key"
                                    />
                                </div>
                            </div>
                        )}

                        <button
                            className="btn btn-primary"
                            onClick={() => void handleRemoteImport()}
                            disabled={isImporting || !remoteId.trim()}
                            style={{ alignSelf: 'flex-start' }}
                        >
                            {isImporting
                                ? 'Importing...'
                                : `Import from ${sourceTabConfig[activeTab].label}`}
                        </button>

                        {importStatus && (
                            <div
                                style={{
                                    padding: 'var(--space-sm) var(--space-md)',
                                    borderRadius: 'var(--radius-md)',
                                    background: importStatus.toLowerCase().includes('failed')
                                        ? 'rgba(239,68,68,0.1)'
                                        : 'rgba(34,197,94,0.1)',
                                    color: importStatus.toLowerCase().includes('failed')
                                        ? 'var(--color-error)'
                                        : 'var(--color-success)',
                                    fontSize: 'var(--font-size-sm)',
                                }}
                            >
                                {importStatus}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {(isImporting || importLogs.length > 0) && (
                <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
                    <h3
                        style={{
                            fontSize: 'var(--font-size-md)',
                            fontWeight: 600,
                            marginBottom: 'var(--space-md)',
                        }}
                    >
                        Remote Import Progress
                    </h3>
                    <TerminalConsole logs={importLogs} height="260px" />
                </div>
            )}

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
                        <div className="empty-state-text">
                            Upload files or import from HuggingFace, Kaggle, or a URL above.
                        </div>
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
                                        <td>
                                            <span className="badge badge-info" style={{ fontSize: 10 }}>
                                                {doc.source || 'upload'}
                                            </span>
                                        </td>
                                        <td>
                                            <span className="badge badge-accent">{doc.file_type}</span>
                                        </td>
                                        <td className="doc-size">{formatSize(doc.file_size_bytes)}</td>
                                        <td>
                                            <span className={`badge ${statusBadgeClass(doc.status)}`}>{doc.status}</span>
                                        </td>
                                        <td className="doc-date">{new Date(doc.ingested_at).toLocaleDateString()}</td>
                                        <td className="doc-actions">
                                            {doc.status === 'pending' && (
                                                <button
                                                    className="btn btn-ghost btn-sm"
                                                    onClick={() => void handleProcess(doc.id)}
                                                    title="Process"
                                                >
                                                    ⚙️
                                                </button>
                                            )}
                                            <button
                                                className="btn btn-ghost btn-sm"
                                                onClick={() => void handleDelete(doc.id)}
                                                title="Delete"
                                            >
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
