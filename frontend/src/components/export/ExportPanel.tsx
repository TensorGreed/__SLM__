import React, { useEffect, useState } from 'react';
import api from '../../api/client';
import './ExportPanel.css';

interface ExportPanelProps { projectId: number; }

interface ExperimentOption {
    id: number;
    name: string;
    status: string;
}

interface ExportRecord {
    id: number;
    export_format?: string;
    format?: string;
    quantization?: string | null;
    status: string;
    output_path?: string;
    file_size_bytes?: number | null;
    manifest?: Record<string, unknown>;
    created_at?: string;
    completed_at?: string | null;
}

interface DeploymentTargetSpec {
    target_id: string;
    kind: string;
    display_name: string;
    description: string;
    compatible: boolean;
    smoke_supported: boolean;
}

interface DeploymentTargetCatalogResponse {
    default_target_ids: string[];
    targets: DeploymentTargetSpec[];
}

interface ExportRunResponse {
    id: number;
    status: string;
    output_path?: string;
    file_size_bytes?: number | null;
    manifest?: Record<string, unknown>;
    deployment?: {
        summary?: {
            deployable_artifact?: boolean;
            local_smoke_passed?: boolean | null;
        };
    };
}

interface RegistryModel {
    id: number;
    experiment_id: number;
    name: string;
    version: string;
    stage: string;
    deployment_status: string;
    readiness?: {
        metrics?: {
            exact_match?: number | null;
            f1?: number | null;
            llm_judge_pass_rate?: number | null;
            safety_pass_rate?: number | null;
        };
    };
    deployment?: {
        endpoint_url?: string;
        environment?: string;
    };
}

interface RegistryListResponse {
    models: RegistryModel[];
}

function toErrorMessage(error: unknown): string {
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

export default function ExportPanel({ projectId }: ExportPanelProps) {
    const [experiments, setExperiments] = useState<ExperimentOption[]>([]);
    const [exportsList, setExportsList] = useState<ExportRecord[]>([]);
    const [registryModels, setRegistryModels] = useState<RegistryModel[]>([]);

    const [selectedExp, setSelectedExp] = useState('');
    const [format, setFormat] = useState('gguf');
    const [quantization, setQuantization] = useState('4-bit');
    const [runSmokeTests, setRunSmokeTests] = useState(true);

    const [deploymentTargets, setDeploymentTargets] = useState<DeploymentTargetSpec[]>([]);
    const [defaultDeploymentTargets, setDefaultDeploymentTargets] = useState<string[]>([]);
    const [selectedDeploymentTargets, setSelectedDeploymentTargets] = useState<string[]>([]);
    const [isLoadingTargets, setIsLoadingTargets] = useState(false);
    const [targetLoadError, setTargetLoadError] = useState('');

    const [isLoading, setIsLoading] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const [expandedIds, setExpandedIds] = useState<number[]>([]);
    const [copyState, setCopyState] = useState<Record<string, boolean>>({});

    const refreshAll = async () => {
        setIsLoading(true);
        try {
            const [expRes, exportRes, registryRes] = await Promise.all([
                api.get<ExperimentOption[]>(`/projects/${projectId}/training/experiments`),
                api.get<ExportRecord[]>(`/projects/${projectId}/export/list`),
                api.get<RegistryListResponse>(`/projects/${projectId}/registry/models`),
            ]);
            setExperiments(expRes.data || []);
            setExportsList(exportRes.data || []);
            setRegistryModels(registryRes.data.models || []);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        setSelectedExp('');
        setExpandedIds([]);
        setErrorMessage('');
        setStatusMessage('');
        setTargetLoadError('');
        void refreshAll();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    useEffect(() => {
        const loadDeploymentTargets = async () => {
            setIsLoadingTargets(true);
            setTargetLoadError('');
            try {
                const res = await api.get<DeploymentTargetCatalogResponse>(
                    `/projects/${projectId}/export/deployment-targets`,
                    {
                        params: { export_format: format },
                    },
                );
                const compatibleTargets = (res.data.targets || []).filter((target) => target.compatible);
                const defaultTargetIds = (res.data.default_target_ids || []).filter((targetId) =>
                    compatibleTargets.some((target) => target.target_id === targetId),
                );
                setDeploymentTargets(compatibleTargets);
                setDefaultDeploymentTargets(defaultTargetIds);
                setSelectedDeploymentTargets(defaultTargetIds);
            } catch (err) {
                setDeploymentTargets([]);
                setDefaultDeploymentTargets([]);
                setSelectedDeploymentTargets([]);
                setTargetLoadError(toErrorMessage(err));
            } finally {
                setIsLoadingTargets(false);
            }
        };

        void loadDeploymentTargets();
    }, [format, projectId]);

    const toggleDeploymentTarget = (targetId: string) => {
        setSelectedDeploymentTargets((prev) => (
            prev.includes(targetId)
                ? prev.filter((item) => item !== targetId)
                : [...prev, targetId]
        ));
    };

    const handleCreateExport = async () => {
        if (!selectedExp) return;
        setIsExporting(true);
        setErrorMessage('');
        setStatusMessage('Creating export...');
        try {
            const createRes = await api.post(`/projects/${projectId}/export/create`, {
                experiment_id: Number(selectedExp),
                export_format: format,
                quantization,
            });
            const runRes = await api.post<ExportRunResponse>(`/projects/${projectId}/export/${createRes.data.id}/run`, {
                deployment_targets: selectedDeploymentTargets,
                run_smoke_tests: runSmokeTests,
            });
            const runPayload: ExportRecord = {
                id: runRes.data.id,
                status: runRes.data.status,
                output_path: runRes.data.output_path,
                file_size_bytes: runRes.data.file_size_bytes,
                manifest: runRes.data.manifest,
                format,
                quantization,
                created_at: new Date().toISOString(),
            };
            setExportsList((prev) => [runPayload, ...prev]);
            const deployable = runRes.data.deployment?.summary?.deployable_artifact;
            const smokePassed = runRes.data.deployment?.summary?.local_smoke_passed;
            const summaryParts: string[] = ['Export run completed.'];
            if (typeof deployable === 'boolean') {
                summaryParts.push(`Artifact deployable: ${deployable ? 'yes' : 'no'}.`);
            }
            if (runSmokeTests && smokePassed !== undefined && smokePassed !== null) {
                summaryParts.push(`Smoke tests: ${smokePassed ? 'passed' : 'failed'}.`);
            }
            setStatusMessage(summaryParts.join(' '));
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        } finally {
            setIsExporting(false);
        }
    };

    const handleRegisterModel = async () => {
        if (!selectedExp) return;
        setErrorMessage('');
        setStatusMessage('Registering model...');
        try {
            await api.post(`/projects/${projectId}/registry/models/register`, {
                experiment_id: Number(selectedExp),
            });
            await refreshAll();
            setStatusMessage('Model registered in candidate stage.');
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        }
    };

    const handlePromote = async (modelId: number, targetStage: 'staging' | 'production') => {
        setErrorMessage('');
        setStatusMessage(`Promoting model ${modelId} to ${targetStage}...`);
        try {
            await api.post(`/projects/${projectId}/registry/models/${modelId}/promote`, {
                target_stage: targetStage,
                force: false,
            });
            await refreshAll();
            setStatusMessage(`Model ${modelId} promoted to ${targetStage}.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        }
    };

    const handleDeploy = async (modelId: number, environment: 'staging' | 'production') => {
        setErrorMessage('');
        setStatusMessage(`Marking model ${modelId} as deployed (${environment})...`);
        try {
            await api.post(`/projects/${projectId}/registry/models/${modelId}/deploy`, {
                environment,
            });
            await refreshAll();
            setStatusMessage(`Model ${modelId} marked deployed (${environment}).`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        }
    };

    const toggleExpand = (id: number) => {
        setExpandedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
    };

    const handleCopy = (key: string, text: string) => {
        navigator.clipboard.writeText(text);
        setCopyState((prev) => ({ ...prev, [key]: true }));
        setTimeout(() => setCopyState((prev) => ({ ...prev, [key]: false })), 1500);
    };

    const statusColor = (status: string) =>
        status === 'completed'
            ? 'badge-success'
            : status === 'in_progress' || status === 'running'
                ? 'badge-info'
                : status === 'failed'
                    ? 'badge-error'
                    : 'badge-warning';

    const stageColor = (stage: string) =>
        stage === 'production'
            ? 'badge-success'
            : stage === 'staging'
                ? 'badge-info'
                : stage === 'archived'
                    ? 'badge-warning'
                    : 'badge-accent';

    const formatBytes = (bytes?: number | null) => {
        if (!bytes) return '—';
        if (bytes === 0) return '0 B';
        const k = 1024;
        const units = ['B', 'KB', 'MB', 'GB'];
        const idx = Math.floor(Math.log(bytes) / Math.log(k));
        return `${(bytes / Math.pow(k, idx)).toFixed(2)} ${units[idx]}`;
    };

    const fmtMetric = (value?: number | null) => (value == null ? '—' : `${(value * 100).toFixed(1)}%`);

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-lg)' }}>Export and Registry</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 'var(--space-md)', marginBottom: 'var(--space-lg)' }}>
                    <div className="form-group">
                        <label className="form-label">Experiment</label>
                        <select className="input" value={selectedExp} onChange={(e) => setSelectedExp(e.target.value)}>
                            <option value="">Select...</option>
                            {experiments.map((exp) => (
                                <option key={exp.id} value={exp.id}>
                                    {exp.name} ({exp.status})
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Format</label>
                        <select className="input" value={format} onChange={(e) => setFormat(e.target.value)}>
                            <option value="gguf">GGUF (CPU)</option>
                            <option value="onnx">ONNX</option>
                            <option value="huggingface">HuggingFace</option>
                            <option value="tensorrt">TensorRT-LLM</option>
                            <option value="docker">Docker</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Quantization</label>
                        <select className="input" value={quantization} onChange={(e) => setQuantization(e.target.value)}>
                            <option value="none">None</option>
                            <option value="4-bit">4-bit</option>
                            <option value="8-bit">8-bit</option>
                        </select>
                    </div>
                </div>
                <div className="export-target-panel">
                    <div className="export-target-panel-header">
                        <div>
                            <div className="export-target-panel-title">Deployment Targets</div>
                            <div className="export-target-panel-subtitle">
                                {selectedDeploymentTargets.length} selected
                            </div>
                        </div>
                        <button
                            className="btn btn-secondary btn-sm"
                            type="button"
                            disabled={isLoadingTargets || defaultDeploymentTargets.length === 0}
                            onClick={() => setSelectedDeploymentTargets(defaultDeploymentTargets)}
                        >
                            Use Recommended
                        </button>
                    </div>
                    {isLoadingTargets && (
                        <div className="export-target-panel-note">Loading deployment target catalog...</div>
                    )}
                    {!isLoadingTargets && targetLoadError && (
                        <div className="export-target-panel-error">
                            Failed to load target catalog: {targetLoadError}. Export will use server defaults.
                        </div>
                    )}
                    {!isLoadingTargets && !targetLoadError && deploymentTargets.length === 0 && (
                        <div className="export-target-panel-note">
                            No compatible deployment targets available for this format.
                        </div>
                    )}
                    {!isLoadingTargets && !targetLoadError && deploymentTargets.length > 0 && (
                        <div className="export-target-grid">
                            {deploymentTargets.map((target) => {
                                const checked = selectedDeploymentTargets.includes(target.target_id);
                                return (
                                    <label
                                        key={target.target_id}
                                        className={`export-target-option ${checked ? 'selected' : ''}`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={checked}
                                            onChange={() => toggleDeploymentTarget(target.target_id)}
                                        />
                                        <div>
                                            <div className="export-target-name">
                                                {target.display_name}
                                                <span className="export-target-kind">{target.kind}</span>
                                            </div>
                                            <div className="export-target-description">{target.description}</div>
                                        </div>
                                    </label>
                                );
                            })}
                        </div>
                    )}
                    <label className="export-smoke-toggle">
                        <input
                            type="checkbox"
                            checked={runSmokeTests}
                            onChange={(e) => setRunSmokeTests(e.target.checked)}
                        />
                        <span>Run local smoke tests for selected runner targets</span>
                    </label>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button
                        className="btn btn-primary"
                        onClick={() => void handleCreateExport()}
                        disabled={!selectedExp || isExporting || isLoading || isLoadingTargets}
                    >
                        {isExporting ? 'Exporting...' : 'Run Export'}
                    </button>
                    <button className="btn btn-secondary" onClick={() => void handleRegisterModel()} disabled={!selectedExp || isLoading}>
                        Register Selected Model
                    </button>
                </div>
                {(errorMessage || statusMessage) && (
                    <div style={{ marginTop: 'var(--space-md)', color: errorMessage ? 'var(--color-error)' : 'var(--color-success)' }}>
                        {errorMessage || statusMessage}
                    </div>
                )}
            </div>

            {registryModels.length > 0 && (
                <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                    <div style={{ padding: 'var(--space-lg)', borderBottom: '1px solid var(--border-color)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, margin: 0 }}>Model Registry</h3>
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                        <table className="export-history-table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Version</th>
                                    <th>Stage</th>
                                    <th>Exact Match</th>
                                    <th>F1</th>
                                    <th>Judge Pass</th>
                                    <th>Safety Pass</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {registryModels.map((model) => (
                                    <tr key={model.id}>
                                        <td>{model.name}</td>
                                        <td>{model.version}</td>
                                        <td><span className={`badge ${stageColor(model.stage)}`}>{model.stage}</span></td>
                                        <td>{fmtMetric(model.readiness?.metrics?.exact_match)}</td>
                                        <td>{fmtMetric(model.readiness?.metrics?.f1)}</td>
                                        <td>{fmtMetric(model.readiness?.metrics?.llm_judge_pass_rate)}</td>
                                        <td>{fmtMetric(model.readiness?.metrics?.safety_pass_rate)}</td>
                                        <td>
                                            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                                <button className="btn btn-secondary btn-sm" onClick={() => void handlePromote(model.id, 'staging')}>
                                                    Promote Staging
                                                </button>
                                                <button className="btn btn-secondary btn-sm" onClick={() => void handlePromote(model.id, 'production')}>
                                                    Promote Prod
                                                </button>
                                                <button className="btn btn-secondary btn-sm" onClick={() => void handleDeploy(model.id, model.stage === 'production' ? 'production' : 'staging')}>
                                                    Mark Deployed
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {exportsList.length > 0 && (
                <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                    <div style={{ padding: 'var(--space-lg)', borderBottom: '1px solid var(--border-color)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, margin: 0 }}>Export History</h3>
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                        <table className="export-history-table">
                            <thead>
                                <tr>
                                    <th style={{ width: 40 }}></th>
                                    <th>Format</th>
                                    <th>Quantization</th>
                                    <th>Size</th>
                                    <th>Status</th>
                                    <th>Date</th>
                                </tr>
                            </thead>
                            <tbody>
                                {exportsList.map((exp) => {
                                    const isExpanded = expandedIds.includes(exp.id);
                                    const manifestJson = exp.manifest
                                        ? JSON.stringify(exp.manifest, null, 2)
                                        : '{\n  "status": "No manifest generated yet."\n}';

                                    return (
                                        <React.Fragment key={exp.id}>
                                            <tr className={`export-row ${isExpanded ? 'expanded' : ''}`} onClick={() => toggleExpand(exp.id)} style={{ cursor: 'pointer' }}>
                                                <td style={{ textAlign: 'center' }}><span className="expand-icon" style={{ display: 'inline-block', fontSize: 10 }}>▼</span></td>
                                                <td style={{ fontWeight: 600 }}>{String(exp.export_format || exp.format || 'unknown').toUpperCase()}</td>
                                                <td>{exp.quantization || 'None'}</td>
                                                <td>{formatBytes(exp.file_size_bytes)}</td>
                                                <td><span className={`badge ${statusColor(exp.status)}`}>{exp.status}</span></td>
                                                <td><span style={{ color: 'var(--text-tertiary)' }}>{exp.created_at ? new Date(exp.created_at).toLocaleString() : '—'}</span></td>
                                            </tr>
                                            {isExpanded && (
                                                <tr className="export-details-row">
                                                    <td></td>
                                                    <td colSpan={5}>
                                                        <div className="manifest-header">
                                                            <div className="manifest-title">Manifest and Output Path</div>
                                                            <div style={{ display: 'flex', gap: 8 }}>
                                                                <button
                                                                    className="btn btn-secondary btn-sm"
                                                                    onClick={(e) => { e.stopPropagation(); handleCopy(`path-${exp.id}`, exp.output_path || ''); }}
                                                                    disabled={!exp.output_path}
                                                                >
                                                                    {copyState[`path-${exp.id}`] ? 'Copied!' : 'Copy Path'}
                                                                </button>
                                                                <button
                                                                    className="btn btn-primary btn-sm"
                                                                    onClick={(e) => { e.stopPropagation(); handleCopy(`manifest-${exp.id}`, manifestJson); }}
                                                                >
                                                                    {copyState[`manifest-${exp.id}`] ? 'Copied!' : 'Copy Manifest'}
                                                                </button>
                                                            </div>
                                                        </div>

                                                        {exp.output_path && (
                                                            <div style={{ marginBottom: 16, fontSize: '0.8rem', color: 'var(--color-info)' }}>
                                                                <span style={{ color: 'var(--text-tertiary)' }}>Path:</span>{' '}
                                                                <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 4 }}>{exp.output_path}</code>
                                                            </div>
                                                        )}

                                                        <div className="manifest-viewer">
                                                            <pre><code>{manifestJson}</code></pre>
                                                        </div>
                                                    </td>
                                                </tr>
                                            )}
                                        </React.Fragment>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
