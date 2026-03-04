import { useEffect, useState } from 'react';
import api from '../../api/client';

interface ExportPanelProps { projectId: number; }

export default function ExportPanel({ projectId }: ExportPanelProps) {
    const [experiments, setExperiments] = useState<any[]>([]);
    const [exports, setExports] = useState<any[]>([]);
    const [selectedExp, setSelectedExp] = useState('');
    const [format, setFormat] = useState('gguf');
    const [quantization, setQuantization] = useState('4-bit');
    const [loaded, setLoaded] = useState(false);

    useEffect(() => {
        setLoaded(false);
        setExperiments([]);
        setExports([]);
        setSelectedExp('');
    }, [projectId]);

    useEffect(() => {
        if (!loaded) {
            Promise.all([
                api.get(`/projects/${projectId}/training/experiments`),
                api.get(`/projects/${projectId}/export/list`),
            ]).then(([exps, expts]) => {
                setExperiments(exps.data);
                setExports(expts.data);
                setLoaded(true);
            });
        }
    }, [loaded, projectId]);

    const handleCreate = async () => {
        if (!selectedExp) return;
        const res = await api.post(`/projects/${projectId}/export/create`, {
            experiment_id: parseInt(selectedExp), export_format: format, quantization,
        });
        // Run immediately
        const run = await api.post(`/projects/${projectId}/export/${res.data.id}/run`);
        setExports(prev => [run.data, ...prev]);
    };

    const statusColor = (s: string) => s === 'completed' ? 'badge-success' : s === 'in_progress' ? 'badge-info' : s === 'failed' ? 'badge-error' : 'badge-warning';

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

            {exports.length > 0 && (
                <div className="card">
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Export History</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {exports.map((e, i) => (
                            <div key={i} style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div style={{ fontWeight: 500 }}>{e.format?.toUpperCase()} Export</div>
                                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)' }}>{e.output_path || e.created_at}</div>
                                </div>
                                <span className={`badge ${statusColor(e.status)}`}>{e.status}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
