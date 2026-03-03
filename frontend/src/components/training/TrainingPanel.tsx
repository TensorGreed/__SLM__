import { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import api from '../../api/client';
import './TrainingPanel.css';

interface TrainingPanelProps { projectId: number; }

export default function TrainingPanel({ projectId }: TrainingPanelProps) {
    const [experiments, setExperiments] = useState<any[]>([]);
    const [showCreate, setShowCreate] = useState(false);
    const [activeExperiment, setActiveExperiment] = useState<any | null>(null);
    const [metrics, setMetrics] = useState<any[]>([]);
    const [loaded, setLoaded] = useState(false);

    // Form State
    const [name, setName] = useState('');
    const [baseModel, setBaseModel] = useState('microsoft/phi-2');
    const [chatTemplate, setChatTemplate] = useState('llama3');
    const [lr, setLr] = useState('2e-4');
    const [epochs, setEpochs] = useState(3);
    const [batchSize, setBatchSize] = useState(4);
    const [optimizer, setOptimizer] = useState('paged_adamw_8bit');

    // LoRA State
    const [useLora, setUseLora] = useState(true);
    const [loraR, setLoraR] = useState(16);
    const [loraAlpha, setLoraAlpha] = useState(32);
    const [targetModules, setTargetModules] = useState('q_proj, v_proj');

    // Compute State
    const [gradientCheckpointing, setGradientCheckpointing] = useState(true);

    if (!loaded) {
        api.get(`/projects/${projectId}/training/experiments`).then(r => { setExperiments(r.data); setLoaded(true); });
    }

    const handleCreate = async () => {
        if (!name.trim()) return;
        const config = {
            base_model: baseModel,
            chat_template: chatTemplate,
            learning_rate: parseFloat(lr),
            num_epochs: epochs,
            batch_size: batchSize,
            optimizer,
            use_lora: useLora,
            lora_rank: loraR,
            lora_alpha: loraAlpha,
            target_modules: targetModules.split(',').map(s => s.trim()),
            gradient_checkpointing: gradientCheckpointing,
        };
        const res = await api.post(`/projects/${projectId}/training/experiments`, { name, config });
        setExperiments(prev => [res.data, ...prev]);
        setShowCreate(false);
        setName('');
    };

    const handleStart = async (expId: number) => {
        await api.post(`/projects/${projectId}/training/experiments/${expId}/start`);
        const exp = experiments.find(e => e.id === expId);
        if (exp) {
            setActiveExperiment({ ...exp, status: 'running' });
            setMetrics([]);
        }
    };

    const viewDashboard = (exp: any) => {
        setActiveExperiment(exp);
        setMetrics([]);
    };

    // WebSocket for Live Metrics
    useEffect(() => {
        if (!activeExperiment || activeExperiment.status !== 'running') return;

        // In dev, assuming FastAPI is on port 8000
        const wsUrl = `ws://localhost:8000/api/projects/${projectId}/training/experiments/${activeExperiment.id}/metrics/ws`;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setMetrics(prev => [...prev.slice(-99), data]); // Keep last 100 points
            } catch (err) {
                console.error("WS Parse error", err);
            }
        };

        return () => {
            ws.close();
        };
    }, [activeExperiment, projectId]);

    const statusColor = (s: string) => s === 'completed' ? 'badge-success' : s === 'running' ? 'badge-info' : s === 'failed' ? 'badge-error' : 'badge-warning';

    if (activeExperiment) {
        const latestMetric = metrics[metrics.length - 1] || {};
        return (
            <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
                <div className="card">
                    <button className="btn btn-secondary btn-sm" style={{ marginBottom: 16 }} onClick={() => { setActiveExperiment(null); setLoaded(false); }}>← Back to Experiments</button>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-xl)' }}>
                        <div>
                            <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 600, margin: 0 }}>{activeExperiment.name}</h3>
                            <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>{activeExperiment.base_model} • {activeExperiment.training_mode}</div>
                        </div>
                        <span className={`badge ${statusColor(activeExperiment.status)}`} style={{ fontSize: 'var(--font-size-md)', padding: '6px 16px' }}>
                            {activeExperiment.status.toUpperCase()}
                        </span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-md)', marginBottom: 'var(--space-xl)' }}>
                        <div className="metric-box">
                            <div className="metric-label">Epoch</div>
                            <div className="metric-val">{latestMetric.epoch !== undefined ? latestMetric.epoch : '—'} / {activeExperiment.config?.num_epochs || 3}</div>
                        </div>
                        <div className="metric-box">
                            <div className="metric-label">Train Loss</div>
                            <div className="metric-val" style={{ color: 'var(--color-warning)' }}>{latestMetric.train_loss !== undefined ? latestMetric.train_loss.toFixed(4) : '—'}</div>
                        </div>
                        <div className="metric-box">
                            <div className="metric-label">Eval Loss</div>
                            <div className="metric-val" style={{ color: 'var(--color-success)' }}>{latestMetric.eval_loss !== undefined ? latestMetric.eval_loss.toFixed(4) : '—'}</div>
                        </div>
                        <div className="metric-box">
                            <div className="metric-label">GPU Util</div>
                            <div className="metric-val" style={{ color: 'var(--color-info)' }}>{latestMetric.gpu_utilization !== undefined ? latestMetric.gpu_utilization + '%' : '—'}</div>
                        </div>
                    </div>

                    <div style={{ height: 400, width: '100%', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-lg)' }}>
                        {metrics.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={metrics} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis dataKey="step" stroke="#888" tick={{ fontSize: 12 }} />
                                    <YAxis stroke="#888" tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
                                    <Tooltip contentStyle={{ backgroundColor: '#1a1b23', border: '1px solid #333', borderRadius: 8 }} />
                                    <Legend />
                                    <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#eab308" strokeWidth={2} dot={false} isAnimationActive={false} />
                                    <Line type="monotone" dataKey="eval_loss" name="Eval Loss" stroke="#22c55e" strokeWidth={2} dot={true} isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-tertiary)' }}>
                                {activeExperiment.status === 'running' ? 'Connecting to telemetry stream...' : 'No telemetry data available for this experiment.'}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-lg)' }}>
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600 }}>Training Experiments</h3>
                    <button className="btn btn-primary" onClick={() => setShowCreate(!showCreate)}>+ New Experiment</button>
                </div>

                {showCreate && (
                    <div style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-lg)', marginBottom: 'var(--space-lg)' }}>
                        <div className="form-group"><label className="form-label">Experiment Name</label><input className="input" value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Meta-Llama-3-8B-Instruct-v1" /></div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
                            {/* Basic Settings */}
                            <div>
                                <h4 style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-md)', textTransform: 'uppercase' }}>Basic HParams</h4>
                                <div className="form-group"><label className="form-label">Base Model</label><input className="input" value={baseModel} onChange={e => setBaseModel(e.target.value)} /></div>
                                <div className="form-group">
                                    <label className="form-label">Chat Template</label>
                                    <select className="input" value={chatTemplate} onChange={e => setChatTemplate(e.target.value)}>
                                        <option value="llama3">Llama-3</option>
                                        <option value="chatml">ChatML</option>
                                        <option value="zephyr">Zephyr</option>
                                        <option value="phi3">Phi-3</option>
                                    </select>
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                                    <div className="form-group"><label className="form-label">Epochs</label><input className="input" type="number" value={epochs} onChange={e => setEpochs(+e.target.value)} /></div>
                                    <div className="form-group"><label className="form-label">Batch Size</label><input className="input" type="number" value={batchSize} onChange={e => setBatchSize(+e.target.value)} /></div>
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                                    <div className="form-group"><label className="form-label">Learning Rate</label><input className="input" value={lr} onChange={e => setLr(e.target.value)} /></div>
                                    <div className="form-group">
                                        <label className="form-label">Optimizer</label>
                                        <select className="input" value={optimizer} onChange={e => setOptimizer(e.target.value)}>
                                            <option value="paged_adamw_8bit">Paged AdamW (8-bit)</option>
                                            <option value="adamw_torch">AdamW</option>
                                        </select>
                                    </div>
                                </div>
                            </div>

                            {/* Advanced / PEFT */}
                            <div>
                                <h4 style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-md)', textTransform: 'uppercase' }}>Advanced & PEFT</h4>
                                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                    <input type="checkbox" checked={useLora} onChange={e => setUseLora(e.target.checked)} />
                                    <label className="form-label" style={{ margin: 0 }}>Enable LoRA (PEFT)</label>
                                </div>
                                {useLora && (
                                    <div style={{ padding: 'var(--space-md)', background: 'rgba(255,255,255,0.03)', borderRadius: 8, marginBottom: 16 }}>
                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                                            <div className="form-group"><label className="form-label">Rank (r)</label><input className="input" type="number" value={loraR} onChange={e => setLoraR(+e.target.value)} /></div>
                                            <div className="form-group"><label className="form-label">Alpha</label><input className="input" type="number" value={loraAlpha} onChange={e => setLoraAlpha(+e.target.value)} /></div>
                                        </div>
                                        <div className="form-group">
                                            <label className="form-label">Target Modules (comma separated)</label>
                                            <input className="input" value={targetModules} onChange={e => setTargetModules(e.target.value)} />
                                        </div>
                                    </div>
                                )}
                                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                    <input type="checkbox" checked={gradientCheckpointing} onChange={e => setGradientCheckpointing(e.target.checked)} />
                                    <label className="form-label" style={{ margin: 0 }}>Use Gradient Checkpointing (Saves VRAM)</label>
                                </div>
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: 8, marginTop: 'var(--space-lg)', borderTop: '1px solid var(--border-color)', paddingTop: 'var(--space-lg)' }}>
                            <button className="btn btn-primary" onClick={handleCreate}>Create Experiment</button>
                            <button className="btn btn-secondary" onClick={() => setShowCreate(false)}>Cancel</button>
                        </div>
                    </div>
                )}

                {experiments.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">🔬</div>
                        <div className="empty-state-title">No experiments</div>
                        <div className="empty-state-text">Create a training experiment to start fine-tuning.</div>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {experiments.map(exp => (
                            <div key={exp.id} style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div style={{ fontWeight: 600 }}>{exp.name}</div>
                                    <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>{exp.base_model} • {exp.training_mode}</div>
                                </div>
                                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                                    <span className={`badge ${statusColor(exp.status)}`}>{exp.status}</span>
                                    {exp.status === 'pending' && <button className="btn btn-primary btn-sm" onClick={() => handleStart(exp.id)}>▶ Start</button>}
                                    <button className="btn btn-secondary btn-sm" onClick={() => viewDashboard(exp)}>📊 Dashboard</button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
