import React, { useEffect, useState } from 'react';
import api from '../../api/client';
import './ExportPanel.css';

interface ExportPanelProps { projectId: number; }

export default function ExportPanel({ projectId }: ExportPanelProps) {
    const [experiments, setExperiments] = useState<any[]>([]);
    const [exportsList, setExportsList] = useState<any[]>([]);
    const [selectedExp, setSelectedExp] = useState('');
    const [format, setFormat] = useState('gguf');
    const [quantization, setQuantization] = useState('4-bit');
    const [loaded, setLoaded] = useState(false);

    const [expandedIds, setExpandedIds] = useState<HTMLElement[] | any>([]); // Array of expanded export IDs
    const [copyState, setCopyState] = useState<Record<number, boolean>>({});

    useEffect(() => {
        setLoaded(false);
        setExperiments([]);
        setExportsList([]);
        setSelectedExp('');
        setExpandedIds([]);
    }, [projectId]);

    useEffect(() => {
        if (!loaded) {
            Promise.all([
                api.get(`/projects/${projectId}/training/experiments`),
                api.get(`/projects/${projectId}/export/list`),
            ]).then(([exps, expts]) => {
                setExperiments(exps.data);
                setExportsList(expts.data);
                setLoaded(true);
            });
        }
    }, [loaded, projectId]);

    const handleCreate = async () => {
        if (!selectedExp) return;
        const res = await api.post(`/projects/${projectId}/export/create`, {
            experiment_id: parseInt(selectedExp), export_format: format, quantization,
        });
        const run = await api.post(`/projects/${projectId}/export/${res.data.id}/run`);
        setExportsList(prev => [run.data, ...prev]);
    };

    const toggleExpand = (id: number) => {
        setExpandedIds((prev: number[]) => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
    };

    const handleCopy = (id: number, text: string) => {
        navigator.clipboard.writeText(text);
        setCopyState(prev => ({ ...prev, [id]: true }));
        setTimeout(() => setCopyState(prev => ({ ...prev, [id]: false })), 2000);
    };

    const statusColor = (s: string) => s === 'completed' ? 'badge-success' : s === 'in_progress' ? 'badge-info' : s === 'failed' ? 'badge-error' : 'badge-warning';

    const formatBytes = (bytes?: number) => {
        if (!bytes) return '—';
        if (bytes === 0) return '0 B';
        const k = 1024, sizes = ['B', 'KB', 'MB', 'GB'], i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-lg)' }}>🚀 Export to Production</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 'var(--space-md)', marginBottom: 'var(--space-lg)' }}>
                    <div className="form-group">
                        <label className="form-label">Experiment</label>
                        <select className="input" value={selectedExp} onChange={e => setSelectedExp(e.target.value)}>
                            <option value="">Select...</option>
                            {experiments.map(e => <option key={e.id} value={e.id}>{e.name}</option>)}
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Format</label>
                        <select className="input" value={format} onChange={e => setFormat(e.target.value)}>
                            <option value="gguf">GGUF (CPU)</option>
                            <option value="onnx">ONNX</option>
                            <option value="huggingface">HuggingFace</option>
                            <option value="tensorrt">TensorRT-LLM</option>
                            <option value="docker">Docker Container</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Quantization</label>
                        <select className="input" value={quantization} onChange={e => setQuantization(e.target.value)}>
                            <option value="none">None</option>
                            <option value="4-bit">4-bit</option>
                            <option value="8-bit">8-bit</option>
                        </select>
                    </div>
                </div>
                <button className="btn btn-primary" onClick={handleCreate} disabled={!selectedExp}>🚀 Export Model</button>
            </div>

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
                                {exportsList.map(exp => {
                                    const isExpanded = expandedIds.includes(exp.id);
                                    const manifestJson = exp.manifest ? JSON.stringify(exp.manifest, null, 2) : '{\n  "status": "No manifest generated yet."\n}';

                                    return (
                                        <React.Fragment key={exp.id}>
                                            <tr className={`export-row ${isExpanded ? 'expanded' : ''}`} onClick={() => toggleExpand(exp.id)} style={{ cursor: 'pointer' }}>
                                                <td style={{ textAlign: 'center' }}>
                                                    <span className="expand-icon" style={{ display: 'inline-block', fontSize: 10 }}>▼</span>
                                                </td>
                                                <td style={{ fontWeight: 600 }}>{exp.export_format.toUpperCase()}</td>
                                                <td>{exp.quantization || 'None'}</td>
                                                <td>{formatBytes(exp.file_size_bytes)}</td>
                                                <td><span className={`badge ${statusColor(exp.status)}`}>{exp.status}</span></td>
                                                <td><span style={{ color: 'var(--text-tertiary)' }}>{new Date(exp.created_at).toLocaleString()}</span></td>
                                            </tr>
                                            {isExpanded && (
                                                <tr className="export-details-row">
                                                    <td></td>
                                                    <td colSpan={5}>
                                                        <div className="manifest-header">
                                                            <div className="manifest-title">🗎 Build Manifest & Output path</div>
                                                            <div style={{ display: 'flex', gap: 8 }}>
                                                                <button
                                                                    className="btn btn-secondary btn-sm"
                                                                    onClick={(e) => { e.stopPropagation(); handleCopy(exp.id, exp.output_path || ''); }}
                                                                    disabled={!exp.output_path}
                                                                >
                                                                    {copyState[exp.id] ? 'Copied!' : 'Copy Path'}
                                                                </button>
                                                                <button
                                                                    className="btn btn-primary btn-sm"
                                                                    onClick={(e) => { e.stopPropagation(); handleCopy(exp.id + 9999, manifestJson); }}
                                                                >
                                                                    {copyState[exp.id + 9999] ? 'Copied!' : 'Copy Manifest'}
                                                                </button>
                                                            </div>
                                                        </div>

                                                        {exp.output_path && (
                                                            <div style={{ marginBottom: 16, fontSize: '0.8rem', color: 'var(--color-info)' }}>
                                                                <span style={{ color: 'var(--text-tertiary)' }}>Path:</span> <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 4 }}>{exp.output_path}</code>
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
