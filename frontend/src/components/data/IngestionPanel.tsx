import { useState, useCallback } from 'react';
import type { RawDocument, DocumentStatus } from '../../types';
import api from '../../api/client';
import './IngestionPanel.css';

interface IngestionPanelProps {
    projectId: number;
}

export default function IngestionPanel({ projectId }: IngestionPanelProps) {
    const [documents, setDocuments] = useState<RawDocument[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const [uploadProgress, setUploadProgress] = useState<string>('');
    const [loaded, setLoaded] = useState(false);

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

    return (
        <div className="ingestion-panel animate-fade-in">
            {/* Upload Zone */}
            <div
                className={`upload-zone glass-card ${dragOver ? 'drag-over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
            >
                <div className="upload-zone-content">
                    <div className="upload-icon">📤</div>
                    <h3 className="upload-title">Drop files here or click to browse</h3>
                    <p className="upload-subtitle">Supports PDF, DOCX, TXT, Markdown, CSV</p>
                    <input
                        type="file"
                        multiple
                        accept=".pdf,.docx,.txt,.md,.csv,.markdown"
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
                        <div className="empty-state-text">Upload files above to start building your dataset.</div>
                    </div>
                ) : (
                    <div className="docs-table-container">
                        <table className="docs-table">
                            <thead>
                                <tr>
                                    <th>Filename</th>
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
        </div>
    );
}
