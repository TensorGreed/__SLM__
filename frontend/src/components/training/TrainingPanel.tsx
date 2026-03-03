import { useState } from 'react';
import api from '../../api/client';

interface TrainingPanelProps { projectId: number; }

export default function TrainingPanel({ projectId }: TrainingPanelProps) {
    const [experiments, setExperiments] = useState<any[]>([]);
    const [showCreate, setShowCreate] = useState(false);
    const [name, setName] = useState('');
    const [baseModel, setBaseModel] = useState('microsoft/phi-2');
    const [lr, setLr] = useState('2e-4');
    const [epochs, setEpochs] = useState(3);
    const [batchSize, setBatchSize] = useState(4);
    const [useLora, setUseLora] = useState(true);
    const [loaded, setLoaded] = useState(false);

    if (!loaded) {
        api.get(`/projects/${projectId}/training/experiments`).then(r => { setExperiments(r.data); setLoaded(true); });
    }

    const handleCreate = async () => {
        if (!name.trim()) return;
        const res = await api.post(`/projects/${projectId}/training/experiments`, {
            name, config: { base_model: baseModel, learning_rate: parseFloat(lr), num_epochs: epochs, batch_size: batchSize, use_lora: useLora },
        });
        setExperiments(prev => [res.data, ...prev]);
        setShowCreate(false); setName('');
    };

    const handleStart = async (expId: number) => {
        await api.post(`/projects/${projectId}/training/experiments/${expId}/start`);
        setLoaded(false);
    };

    const statusColor = (s: string) => s === 'completed' ? 'badge-success' : s === 'running' ? 'badge-info' : s === 'failed' ? 'badge-error' : 'badge-warning';

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-lg)' }}>
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600 }}>Training Experiments</h3>
                    <button className="btn btn-primary" onClick={() => setShowCreate(!showCreate)}>+ New Experiment</button>
                </div>

                {showCreate && (
                    <div style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-lg)', marginBottom: 'var(--space-lg)' }}>
                        <div className="form-group"><label className="form-label">Name</label><input className="input" value={name} onChange={e => setName(e.target.value)} placeholder="Experiment name" /></div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-md)' }}>
                            <div className="form-group"><label className="form-label">Base Model</label><input className="input" value={baseModel} onChange={e => setBaseModel(e.target.value)} /></div>
                            <div className="form-group"><label className="form-label">Learning Rate</label><input className="input" value={lr} onChange={e => setLr(e.target.value)} /></div>
                            <div className="form-group"><label className="form-label">Epochs</label><input className="input" type="number" value={epochs} onChange={e => setEpochs(+e.target.value)} /></div>
                            <div className="form-group"><label className="form-label">Batch Size</label><input className="input" type="number" value={batchSize} onChange={e => setBatchSize(+e.target.value)} /></div>
                        </div>
                        <label className="form-label" style={{ display: 'flex', gap: 8, alignItems: 'center', margin: '8px 0 16px' }}><input type="checkbox" checked={useLora} onChange={e => setUseLora(e.target.checked)} /> Use LoRA</label>
                        <div style={{ display: 'flex', gap: 8 }}><button className="btn btn-primary" onClick={handleCreate}>Create</button><button className="btn btn-secondary" onClick={() => setShowCreate(false)}>Cancel</button></div>
                    </div>
                )}

                {experiments.length === 0 ? (
                    <div className="empty-state"><div className="empty-state-icon">🔬</div><div className="empty-state-title">No experiments</div><div className="empty-state-text">Create a training experiment to start fine-tuning.</div></div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {experiments.map(exp => (
                            <div key={exp.id} style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div style={{ fontWeight: 600 }}>{exp.name}</div>
                                    <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>{exp.base_model} • {exp.training_mode}</div>
                                </div>
                                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                    <span className={`badge ${statusColor(exp.status)}`}>{exp.status}</span>
                                    {exp.status === 'pending' && <button className="btn btn-primary btn-sm" onClick={() => handleStart(exp.id)}>▶ Start</button>}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
