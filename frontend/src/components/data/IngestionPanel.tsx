import { useState, useCallback, useEffect } from 'react';
import type { RawDocument, DocumentStatus } from '../../types';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { TerminalConsole } from '../shared/TerminalConsole';
import { ReadinessPanel } from '../shared/ReadinessPanel';
import { buildWsUrl } from '../../utils/ws';
import EDADashboard from './EDADashboard';
import './IngestionPanel.css';

interface IngestionPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

type SourceTab = 'upload' | 'huggingface' | 'kaggle' | 'url';
type RemoteSourceTab = Exclude<SourceTab, 'upload'>;

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
    result_visible_in_api_db?: boolean;
    warning?: string;
    error?: string;
}

interface ProjectSecret {
    provider: string;
    key_name: string;
    value_hint: string;
}

interface ProjectSecretListResponse {
    secrets: ProjectSecret[];
}

interface AdapterCatalogResponse {
    default_adapter: string;
    adapters: Record<string, {
        description?: string;
        source?: string;
    }>;
}

function parseJsonObjectInput(raw: string): { value: Record<string, unknown>; error: string } {
    const text = raw.trim();
    if (!text) {
        return { value: {}, error: '' };
    }
    try {
        const parsed = JSON.parse(text);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            return { value: {}, error: 'JSON config must be an object (e.g. {"key":"value"}).' };
        }
        return { value: parsed as Record<string, unknown>, error: '' };
    } catch (error) {
        if (error instanceof Error) {
            return { value: {}, error: error.message };
        }
        return { value: {}, error: 'Invalid JSON.' };
    }
}

function extractErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
        if (Array.isArray(detail)) {
            const messages = detail
                .map((item) => {
                    if (typeof item === 'string') {
                        return item;
                    }
                    if (typeof item === 'object' && item !== null) {
                        const msg = (item as { msg?: unknown }).msg;
                        const loc = (item as { loc?: unknown }).loc;
                        const locText = Array.isArray(loc) ? loc.join('.') : '';
                        if (typeof msg === 'string' && msg.trim()) {
                            return locText ? `${locText}: ${msg}` : msg;
                        }
                    }
                    return '';
                })
                .filter((item) => item);
            if (messages.length > 0) {
                return messages.join('; ');
            }
        }
        if (typeof detail === 'object' && detail !== null) {
            const msg = (detail as { message?: unknown }).message;
            if (typeof msg === 'string' && msg.trim()) {
                return msg;
            }
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
    const [remoteConfig, setRemoteConfig] = useState('');
    const [remoteMaxSamples, setRemoteMaxSamples] = useState('');
    const [hfToken, setHfToken] = useState('');
    const [kaggleUsername, setKaggleUsername] = useState('');
    const [kaggleKey, setKaggleKey] = useState('');

    const [isInspecting, setIsInspecting] = useState(false);
    const [inspectionResult, setInspectionResult] = useState<{
        configs?: string[];
        splits?: string[];
        features?: Record<string, string>;
        error?: string;
        remediation?: string;
    } | null>(null);

    const [isImporting, setIsImporting] = useState(false);
    const [importStatus, setImportStatus] = useState('');
    const [importLogs, setImportLogs] = useState<string[]>([]);
    const [activeReportPath, setActiveReportPath] = useState<string | null>(null);
    const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
    const [adapterCatalog, setAdapterCatalog] = useState<AdapterCatalogResponse | null>(null);
    const [remoteAdapterId, setRemoteAdapterId] = useState('default-canonical');
    const [remoteAdapterConfigText, setRemoteAdapterConfigText] = useState('');

    const [savedHfTokenHint, setSavedHfTokenHint] = useState('');
    const [savedKaggleUserHint, setSavedKaggleUserHint] = useState('');
    const [savedKaggleKeyHint, setSavedKaggleKeyHint] = useState('');
    const [useSavedHfToken, setUseSavedHfToken] = useState(true);
    const [useSavedKaggleCreds, setUseSavedKaggleCreds] = useState(true);
    const [secretStatus, setSecretStatus] = useState('');

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

    const refreshSecrets = useCallback(async () => {
        try {
            const res = await api.get<ProjectSecretListResponse>(`/projects/${projectId}/secrets`);
            const secrets = res.data.secrets || [];
            const findHint = (provider: string, keyName: string) =>
                secrets.find((item) => item.provider === provider && item.key_name === keyName)?.value_hint || '';

            const hfHint = findHint('huggingface', 'token');
            const kaggleUserHint = findHint('kaggle', 'username');
            const kaggleKey = findHint('kaggle', 'key');
            setSavedHfTokenHint(hfHint);
            setSavedKaggleUserHint(kaggleUserHint);
            setSavedKaggleKeyHint(kaggleKey);
            setUseSavedHfToken(Boolean(hfHint));
            setUseSavedKaggleCreds(Boolean(kaggleUserHint && kaggleKey));
        } catch {
            setSavedHfTokenHint('');
            setSavedKaggleUserHint('');
            setSavedKaggleKeyHint('');
            setUseSavedHfToken(false);
            setUseSavedKaggleCreds(false);
        }
    }, [projectId]);

    const refreshAdapterCatalog = useCallback(async () => {
        try {
            const res = await api.get<AdapterCatalogResponse>(`/projects/${projectId}/dataset/adapters/catalog`);
            setAdapterCatalog(res.data);
            if (res.data?.default_adapter) {
                setRemoteAdapterId((prev) => prev || res.data.default_adapter);
            }
        } catch {
            setAdapterCatalog(null);
        }
    }, [projectId]);

    useEffect(() => {
        void fetchDocs();
        void refreshSecrets();
        void refreshAdapterCatalog();
    }, [fetchDocs, refreshSecrets, refreshAdapterCatalog]);

    useEffect(() => {
        if (!activeReportPath) {
            return;
        }

        const wsUrl = buildWsUrl(`/api/projects/${projectId}/ingestion/ws/logs`);
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
                    if (status.result_visible_in_api_db === false) {
                        setImportStatus(
                            status.warning
                                || 'Import completed in worker, but this API process cannot see the document. Verify API/worker DATABASE_URL alignment and restart both.',
                        );
                    } else {
                        setImportStatus(`Imported ${imported} samples from ${source}:${identifier}`);
                    }
                    setIsImporting(false);
                    setActiveReportPath(null);
                    setActiveTaskId(null);
                    void fetchDocs();
                } else if (status.status === 'failed') {
                    setImportStatus(status.error ? `Import failed: ${status.error}` : 'Import failed. Check logs and retry.');
                    setIsImporting(false);
                    setActiveReportPath(null);
                    setActiveTaskId(null);
                }
            } catch (err) {
                setImportStatus(`Import polling failed: ${extractErrorMessage(err)}`);
                setIsImporting(false);
                setActiveReportPath(null);
                setActiveTaskId(null);
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

    const upsertSecret = async (provider: string, keyName: string, value: string) => {
        await api.put(`/projects/${projectId}/secrets`, {
            provider,
            key_name: keyName,
            value,
        });
    };

    const handleSaveHfToken = async () => {
        if (!hfToken.trim()) {
            setSecretStatus('Enter a HuggingFace token to save.');
            return;
        }
        try {
            await upsertSecret('huggingface', 'token', hfToken.trim());
            setSecretStatus('Saved HuggingFace token to project secrets.');
            await refreshSecrets();
        } catch (err) {
            setSecretStatus(`Failed to save HuggingFace token: ${extractErrorMessage(err)}`);
        }
    };

    const handleSaveKaggleCredentials = async () => {
        if (!kaggleUsername.trim() || !kaggleKey.trim()) {
            setSecretStatus('Enter both Kaggle username and key to save.');
            return;
        }
        try {
            await Promise.all([
                upsertSecret('kaggle', 'username', kaggleUsername.trim()),
                upsertSecret('kaggle', 'key', kaggleKey.trim()),
            ]);
            setSecretStatus('Saved Kaggle credentials to project secrets.');
            await refreshSecrets();
        } catch (err) {
            setSecretStatus(`Failed to save Kaggle credentials: ${extractErrorMessage(err)}`);
        }
    };

    const handleCancelImport = async () => {
        if (!activeTaskId) {
            setImportStatus('No active import task to cancel.');
            return;
        }
        try {
            await api.post(`/projects/${projectId}/ingestion/imports/tasks/${activeTaskId}/cancel`);
            setImportStatus('Cancel requested. Worker may take a few seconds to stop.');
            setIsImporting(false);
            setActiveTaskId(null);
            setActiveReportPath(null);
        } catch (err) {
            setImportStatus(`Failed to cancel import: ${extractErrorMessage(err)}`);
        }
    };

    const handleInspect = async () => {
        if (!remoteId.trim()) return;
        setIsInspecting(true);
        setInspectionResult(null);
        setImportStatus('');

        try {
            const params: any = {
                project_id: projectId,
                source_type: activeTab,
                identifier: remoteId.trim(),
            };
            if (activeTab === 'huggingface') {
                if (hfToken.trim()) params.hf_token = hfToken.trim();
                else if (useSavedHfToken) params.use_saved_secrets = true;
            }

            const res = await api.get(`/projects/${projectId}/ingestion/import-remote/inspect`, { params });
            const data = res.data;
            setInspectionResult(data);
            
            if (data.configs && data.configs.length > 0) {
                setRemoteConfig(data.configs[0]);
            }
            if (data.splits && data.splits.length > 0) {
                setRemoteSplit(data.splits[0]);
            }
        } catch (err) {
            setImportStatus(`Inspection failed: ${extractErrorMessage(err)}`);
        } finally {
            setIsInspecting(false);
        }
    };

    const handleRemoteImport = async () => {
        if (!remoteId.trim()) return;
        if (activeTab === 'upload') return;

        setIsImporting(true);
        setImportStatus('Queueing remote import job...');
        setImportLogs([]);
        setActiveReportPath(null);
        setActiveTaskId(null);

        try {
            let parsedMaxSamples: number | null = null;
            if (remoteMaxSamples.trim()) {
                const parsed = Number(remoteMaxSamples);
                if (!Number.isFinite(parsed) || parsed <= 0) {
                    setImportStatus('Import failed: max samples must be a positive number or left blank.');
                    setIsImporting(false);
                    return;
                }
                parsedMaxSamples = Math.floor(parsed);
            }

            const remoteSource = activeTab as RemoteSourceTab;
            const parsedAdapterConfig = parseJsonObjectInput(remoteAdapterConfigText);
            if (parsedAdapterConfig.error) {
                setImportStatus(`Import failed: adapter config JSON error: ${parsedAdapterConfig.error}`);
                setIsImporting(false);
                return;
            }
            const payload: {
                source_type: RemoteSourceTab;
                identifier: string;
                split: string;
                config_name?: string | null;
                max_samples: number | null;
                adapter_id: string;
                adapter_config?: Record<string, unknown>;
                use_saved_secrets: boolean;
                hf_token?: string;
                kaggle_username?: string;
                kaggle_key?: string;
            } = {
                source_type: remoteSource,
                identifier: remoteId.trim(),
                split: remoteSource === 'huggingface' ? (remoteSplit || 'train') : 'train',
                config_name: remoteSource === 'huggingface' ? (remoteConfig || null) : null,
                max_samples: parsedMaxSamples,
                adapter_id: remoteAdapterId,
                use_saved_secrets:
                    remoteSource === 'huggingface'
                        ? useSavedHfToken
                        : remoteSource === 'kaggle'
                            ? useSavedKaggleCreds
                            : true,
            };
            if (Object.keys(parsedAdapterConfig.value).length > 0) {
                payload.adapter_config = parsedAdapterConfig.value;
            }

            if (remoteSource === 'huggingface' && hfToken.trim()) {
                payload.hf_token = hfToken.trim();
            }
            if (remoteSource === 'kaggle') {
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
                setActiveTaskId(res.data.task_id ?? null);
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
            <ReadinessPanel projectId={projectId} />
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
                    <div className="remote-import-panel">
                        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', margin: 0 }}>
                            {sourceTabConfig[activeTab].help}
                        </p>
                        <div className="remote-import-grid" style={{ gridTemplateColumns: '1fr auto', alignItems: 'end' }}>
                            <div className="form-group" style={{ margin: 0 }}>
                                <label className="form-label">Dataset Identifier</label>
                                <input
                                    className="input"
                                    value={remoteId}
                                    onChange={(e) => {
                                        setRemoteId(e.target.value);
                                        setInspectionResult(null);
                                    }}
                                    placeholder={sourceTabConfig[activeTab].placeholder}
                                />
                            </div>
                            <button
                                className={`btn ${inspectionResult ? 'btn-secondary' : 'btn-primary'}`}
                                onClick={() => void handleInspect()}
                                disabled={isInspecting || !remoteId.trim()}
                            >
                                {isInspecting ? 'Inspecting...' : (inspectionResult ? 'Re-Inspect' : 'Inspect Source')}
                            </button>
                        </div>

                        {inspectionResult?.error && (
                            <div className="alert alert-error" style={{ marginTop: 'var(--space-md)' }}>
                                <div style={{ fontWeight: 600 }}>{inspectionResult.error}</div>
                                {inspectionResult.remediation && (
                                    <div style={{ marginTop: 4, fontSize: '0.9em', opacity: 0.9 }}>
                                        💡 {inspectionResult.remediation}
                                    </div>
                                )}
                            </div>
                        )}

                        {inspectionResult && !inspectionResult.error && (
                            <div className="remote-import-step2 animate-slide-up" style={{ marginTop: 'var(--space-md)', padding: 'var(--space-md)', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-md)' }}>
                                <div className="remote-import-grid">
                                    {activeTab === 'huggingface' && inspectionResult.configs && inspectionResult.configs.length > 1 && (
                                        <div className="form-group" style={{ margin: 0 }}>
                                            <label className="form-label">Config / Subset</label>
                                            <select
                                                className="input"
                                                value={remoteConfig}
                                                onChange={(e) => setRemoteConfig(e.target.value)}
                                            >
                                                {inspectionResult.configs.map(c => <option key={c} value={c}>{c}</option>)}
                                            </select>
                                        </div>
                                    )}
                                    <div className="form-group" style={{ margin: 0 }}>
                                        <label className="form-label">Split</label>
                                        <select
                                            className="input"
                                            value={remoteSplit}
                                            onChange={(e) => setRemoteSplit(e.target.value)}
                                            style={{ width: 140 }}
                                        >
                                            {inspectionResult.splits?.map(s => <option key={s} value={s}>{s}</option>) || (
                                                <>
                                                    <option value="train">train</option>
                                                    <option value="test">test</option>
                                                    <option value="validation">validation</option>
                                                </>
                                            )}
                                        </select>
                                    </div>
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
                                    <div className="form-group" style={{ margin: 0 }}>
                                        <label className="form-label">Adapter</label>
                                        <select
                                            className="input"
                                            value={remoteAdapterId}
                                            onChange={(e) => setRemoteAdapterId(e.target.value)}
                                            style={{ minWidth: 180 }}
                                        >
                                            <option value="auto">auto-detect</option>
                                            {Object.keys(adapterCatalog?.adapters || {})
                                                .filter((adapterKey) => adapterKey !== 'auto')
                                                .map((adapterKey) => (
                                                    <option key={adapterKey} value={adapterKey}>
                                                        {adapterKey}
                                                    </option>
                                                ))}
                                        </select>
                                    </div>
                                </div>

                                {inspectionResult.features && Object.keys(inspectionResult.features).length > 0 && (
                                    <div style={{ marginTop: 'var(--space-md)', fontSize: 'var(--font-size-xs)' }}>
                                        <div style={{ color: 'var(--text-secondary)', marginBottom: 4 }}>Detected Schema:</div>
                                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                                            {Object.entries(inspectionResult.features).map(([name, type]) => (
                                                <span key={name} className="badge badge-outline" title={type}>{name}</span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                <div className="remote-import-adapter-box" style={{ marginTop: 'var(--space-md)' }}>
                                    <div className="remote-import-adapter-head">
                                        <strong>Adapter Mapping</strong>
                                        <span>Optional JSON overrides for adapter behavior.</span>
                                    </div>
                                    <textarea
                                        className="input remote-import-json"
                                        value={remoteAdapterConfigText}
                                        onChange={(e) => setRemoteAdapterConfigText(e.target.value)}
                                        placeholder='{"field_mapping":{"instruction":"question","response":"answer"}}'
                                        style={{ height: 60 }}
                                    />
                                </div>

                                <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center', marginTop: 'var(--space-lg)' }}>
                                    <button
                                        className="btn btn-primary"
                                        onClick={() => void handleRemoteImport()}
                                        disabled={isImporting}
                                    >
                                        {isImporting ? 'Importing...' : `Import Selected Data`}
                                    </button>
                                    {isImporting && activeTaskId && (
                                        <button
                                            className="btn btn-secondary"
                                            onClick={() => void handleCancelImport()}
                                        >
                                            Cancel Import
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}

                        {!inspectionResult && activeTab === 'huggingface' && (
                            <div className="form-group" style={{ margin: 0 }}>
                                <label className="form-label">HuggingFace Token (optional)</label>
                                <input
                                    className="input"
                                    type="password"
                                    value={hfToken}
                                    onChange={(e) => setHfToken(e.target.value)}
                                    placeholder="hf_..."
                                />
                                <div style={{ display: 'flex', gap: 'var(--space-sm)', marginTop: 8, flexWrap: 'wrap' }}>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 'var(--font-size-xs)' }}>
                                        <input
                                            type="checkbox"
                                            checked={useSavedHfToken}
                                            onChange={(e) => setUseSavedHfToken(e.target.checked)}
                                            disabled={!savedHfTokenHint}
                                        />
                                        Use saved token
                                    </label>
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => void handleSaveHfToken()}
                                        disabled={!hfToken.trim()}
                                    >
                                        Save Token
                                    </button>
                                </div>
                                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', marginTop: 4 }}>
                                    {savedHfTokenHint
                                        ? `Saved token: ${savedHfTokenHint} (auto-used when request token is blank).`
                                        : 'No saved token. Imported request token is used only for this run unless you click Save Token.'}
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
                                <div className="form-group" style={{ margin: 0, alignSelf: 'end' }}>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 'var(--font-size-xs)' }}>
                                        <input
                                            type="checkbox"
                                            checked={useSavedKaggleCreds}
                                            onChange={(e) => setUseSavedKaggleCreds(e.target.checked)}
                                            disabled={!(savedKaggleUserHint && savedKaggleKeyHint)}
                                        />
                                        Use saved credentials
                                    </label>
                                </div>
                                <div className="form-group" style={{ margin: 0, alignSelf: 'end' }}>
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => void handleSaveKaggleCredentials()}
                                        disabled={!kaggleUsername.trim() || !kaggleKey.trim()}
                                    >
                                        Save Credentials
                                    </button>
                                </div>
                                <div style={{ gridColumn: '1 / -1', fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)' }}>
                                    {savedKaggleUserHint && savedKaggleKeyHint
                                        ? `Saved Kaggle credentials: username ${savedKaggleUserHint}, key ${savedKaggleKeyHint}.`
                                        : 'No saved Kaggle credentials.'}
                                </div>
                            </div>
                        )}

                        <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center', flexWrap: 'wrap' }}>
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
                            {isImporting && activeTaskId && (
                                <button
                                    className="btn btn-secondary"
                                    onClick={() => void handleCancelImport()}
                                >
                                    Cancel Import
                                </button>
                            )}
                        </div>

                        {secretStatus && (
                            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-info)' }}>
                                {secretStatus}
                            </div>
                        )}

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

            {documents.length > 0 && (
                <EDADashboard projectId={projectId} />
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
