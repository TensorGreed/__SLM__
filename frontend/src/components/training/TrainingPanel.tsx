import { useEffect, useState } from 'react';

import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { TerminalConsole } from '../shared/TerminalConsole';
import ExperimentCompare from './ExperimentCompare';
import './TrainingPanel.css';

interface TrainingPanelProps {
  projectId: number;
  onNextStep?: () => void;
}

interface ExperimentConfig {
  num_epochs?: number;
  learning_rate?: number | string;
  batch_size?: number;
  use_lora?: boolean;
  lora_r?: number;
  target_modules?: string[];
}

interface Experiment {
  id: number;
  name: string;
  status: string;
  training_mode: string;
  base_model: string;
  config?: ExperimentConfig;
}

interface TrainingMetric {
  experiment_id?: number;
  epoch?: number;
  step?: number;
  train_loss?: number | null;
  eval_loss?: number | null;
  [key: string]: unknown;
}

export default function TrainingPanel({ projectId, onNextStep }: TrainingPanelProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [activeExperiment, setActiveExperiment] = useState<Experiment | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [selectedForCompare, setSelectedForCompare] = useState<number[]>([]);
  const [showCompare, setShowCompare] = useState(false);

  const [name, setName] = useState('');
  const [baseModel, setBaseModel] = useState('microsoft/phi-2');
  const [chatTemplate, setChatTemplate] = useState('llama3');
  const [lr, setLr] = useState('2e-4');
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [optimizer, setOptimizer] = useState('paged_adamw_8bit');
  const [useLora, setUseLora] = useState(true);
  const [loraR, setLoraR] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [targetModules, setTargetModules] = useState('q_proj, v_proj');
  const [gradientCheckpointing, setGradientCheckpointing] = useState(true);

  const statusColor = (status: string) =>
    status === 'completed'
      ? 'badge-success'
      : status === 'running'
        ? 'badge-info'
        : status === 'failed'
          ? 'badge-error'
          : 'badge-warning';

  const refreshExperiments = async () => {
    const res = await api.get(`/projects/${projectId}/training/experiments`);
    setExperiments(res.data || []);
    if (activeExperiment) {
      const latest = (res.data || []).find((e: Experiment) => e.id === activeExperiment.id);
      if (latest) {
        setActiveExperiment(latest);
      }
    }
  };

  useEffect(() => {
    setExperiments([]);
    setActiveExperiment(null);
    setMetrics([]);
    setTrainingLogs([]);
    setSelectedForCompare([]);
    setShowCompare(false);
    setShowCreate(false);
    refreshExperiments().catch((err) => console.error('Failed to load experiments', err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  useEffect(() => {
    if (!activeExperiment || activeExperiment.status !== 'running') {
      return;
    }
    const interval = window.setInterval(() => {
      api
        .get(`/projects/${projectId}/training/experiments/${activeExperiment.id}/status`)
        .then((res) => {
          const status = String(res.data?.status || '');
          if (!status) return;
          setActiveExperiment((prev) => (prev ? { ...prev, status } : prev));
          if (status !== 'running') {
            refreshExperiments().catch(() => undefined);
          }
        })
        .catch(() => undefined);
    }, 5000);

    return () => window.clearInterval(interval);
  }, [activeExperiment, projectId]);

  useEffect(() => {
    if (!activeExperiment || activeExperiment.status !== 'running') {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://${window.location.host}/api/projects/${projectId}/training/ws/${activeExperiment.id}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'init') {
          setMetrics(Array.isArray(data.metrics) ? data.metrics : []);
          return;
        }
        if (data.type === 'metric' && data.metric) {
          setMetrics((prev) => [...prev.slice(-199), data.metric]);
          return;
        }
        if (data.type === 'log' && data.text) {
          setTrainingLogs((prev) => [...prev.slice(-999), String(data.text)]);
          return;
        }
        if (data.type === 'status' && data.status) {
          setActiveExperiment((prev) => (prev ? { ...prev, status: String(data.status) } : prev));
          if (String(data.status) !== 'running') {
            refreshExperiments().catch(() => undefined);
          }
        }
      } catch (err) {
        console.error('WS parse error', err);
      }
    };

    ws.onerror = () => {
      console.error('Training websocket error');
    };

    return () => ws.close();
  }, [activeExperiment, projectId]);

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
      lora_r: loraR,
      lora_alpha: loraAlpha,
      target_modules: targetModules.split(',').map((s) => s.trim()),
      gradient_checkpointing: gradientCheckpointing,
    };
    const res = await api.post(`/projects/${projectId}/training/experiments`, { name, config });
    setExperiments((prev) => [res.data, ...prev]);
    setShowCreate(false);
    setName('');
  };

  const handleStart = async (experimentId: number) => {
    await api.post(`/projects/${projectId}/training/experiments/${experimentId}/start`);
    setExperiments((prev) =>
      prev.map((exp) => (exp.id === experimentId ? { ...exp, status: 'running' } : exp))
    );
    const exp = experiments.find((e) => e.id === experimentId);
    if (exp) {
      setActiveExperiment({ ...exp, status: 'running' });
      setMetrics([]);
      setTrainingLogs([]);
    }
  };

  const viewDashboard = (exp: Experiment) => {
    setActiveExperiment(exp);
    setMetrics([]);
    setTrainingLogs([]);
  };

  const toggleCompareSelection = (expId: number) => {
    setSelectedForCompare((prev) =>
      prev.includes(expId) ? prev.filter((id) => id !== expId) : [...prev, expId]
    );
  };

  if (showCompare && selectedForCompare.length > 1) {
    return (
      <ExperimentCompare
        experimentIds={selectedForCompare}
        onClose={() => setShowCompare(false)}
        projectId={projectId}
      />
    );
  }

  if (activeExperiment) {
    const latestMetric = metrics[metrics.length - 1] || {};
    const currentEpoch = latestMetric.epoch !== undefined ? latestMetric.epoch : '—';
    const currentTrainLoss = latestMetric.train_loss !== undefined ? latestMetric.train_loss : null;
    const currentEvalLoss = latestMetric.eval_loss !== undefined ? latestMetric.eval_loss : null;

    return (
      <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
        <div className="card">
          <button
            className="btn btn-secondary btn-sm"
            style={{ marginBottom: 16 }}
            onClick={() => {
              setActiveExperiment(null);
              refreshExperiments().catch(() => undefined);
            }}
          >
            ← Back to Experiments
          </button>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-xl)' }}>
            <div>
              <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 600, margin: 0 }}>{activeExperiment.name}</h3>
              <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                {activeExperiment.base_model} • {activeExperiment.training_mode}
              </div>
            </div>
            <span className={`badge ${statusColor(activeExperiment.status)}`} style={{ fontSize: 'var(--font-size-md)', padding: '6px 16px' }}>
              {activeExperiment.status.toUpperCase()}
            </span>
          </div>

          <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-md)', marginBottom: 'var(--space-xl)' }}>
            <div className="metric-box box-blue">
              <span className="mb-label">Current Epoch</span>
              <span className="mb-value">{currentEpoch} / {activeExperiment.config?.num_epochs || 3}</span>
            </div>
            <div className="metric-box box-green">
              <span className="mb-label">Training Loss</span>
              <span className="mb-value">{currentTrainLoss !== null ? Number(currentTrainLoss).toFixed(4) : '--'}</span>
            </div>
            <div className="metric-box box-purple">
              <span className="mb-label">Eval Loss</span>
              <span className="mb-value">{currentEvalLoss !== null ? Number(currentEvalLoss).toFixed(4) : '--'}</span>
            </div>
          </div>

          <TerminalConsole logs={trainingLogs} height="320px" />
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-lg)' }}>
          <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600 }}>Training Experiments</h3>
          <button className="btn btn-primary" onClick={() => setShowCreate((prev) => !prev)}>+ New Experiment</button>
        </div>

        {showCreate && (
          <div style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-lg)', marginBottom: 'var(--space-lg)' }}>
            <div className="form-group">
              <label className="form-label">Experiment Name</label>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. llama3-sft-v1" />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
              <div>
                <h4 style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-md)', textTransform: 'uppercase' }}>Basic HParams</h4>
                <div className="form-group">
                  <label className="form-label">Base Model</label>
                  <input className="input" value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />
                </div>
                <div className="form-group">
                  <label className="form-label">Chat Template</label>
                  <select className="input" value={chatTemplate} onChange={(e) => setChatTemplate(e.target.value)}>
                    <option value="llama3">Llama-3</option>
                    <option value="chatml">ChatML</option>
                    <option value="zephyr">Zephyr</option>
                    <option value="phi3">Phi-3</option>
                  </select>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div className="form-group">
                    <label className="form-label">Epochs</label>
                    <input className="input" type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value) || 1)} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Batch Size</label>
                    <input className="input" type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value) || 1)} />
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div className="form-group">
                    <label className="form-label">Learning Rate</label>
                    <input className="input" value={lr} onChange={(e) => setLr(e.target.value)} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Optimizer</label>
                    <select className="input" value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
                      <option value="paged_adamw_8bit">Paged AdamW (8-bit)</option>
                      <option value="adamw_torch">AdamW</option>
                    </select>
                  </div>
                </div>
              </div>

              <div>
                <h4 style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-md)', textTransform: 'uppercase' }}>Advanced & PEFT</h4>
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input type="checkbox" checked={useLora} onChange={(e) => setUseLora(e.target.checked)} />
                  <label className="form-label" style={{ margin: 0 }}>Enable LoRA</label>
                </div>
                {useLora && (
                  <div style={{ padding: 'var(--space-md)', background: 'rgba(255,255,255,0.03)', borderRadius: 8, marginBottom: 16 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div className="form-group">
                        <label className="form-label">Rank (r)</label>
                        <input className="input" type="number" value={loraR} onChange={(e) => setLoraR(Number(e.target.value) || 1)} />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Alpha</label>
                        <input className="input" type="number" value={loraAlpha} onChange={(e) => setLoraAlpha(Number(e.target.value) || 1)} />
                      </div>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Target Modules (comma-separated)</label>
                      <input className="input" value={targetModules} onChange={(e) => setTargetModules(e.target.value)} />
                    </div>
                  </div>
                )}
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input type="checkbox" checked={gradientCheckpointing} onChange={(e) => setGradientCheckpointing(e.target.checked)} />
                  <label className="form-label" style={{ margin: 0 }}>Use Gradient Checkpointing</label>
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
            {selectedForCompare.length > 1 && (
              <div style={{ padding: 'var(--space-sm) 0', display: 'flex', justifyContent: 'flex-end', borderBottom: '1px solid rgba(255,255,255,0.05)', marginBottom: 8 }}>
                <button className="btn btn-primary" onClick={() => setShowCompare(true)}>
                  Compare Selected ({selectedForCompare.length})
                </button>
              </div>
            )}
            {experiments.map((exp) => (
              <div
                key={exp.id}
                style={{
                  background: 'var(--bg-tertiary)',
                  borderRadius: 'var(--radius-md)',
                  padding: 'var(--space-md)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={selectedForCompare.includes(exp.id)}
                    onChange={() => toggleCompareSelection(exp.id)}
                    style={{ width: 16, height: 16, cursor: 'pointer' }}
                  />
                  <div>
                    <div style={{ fontWeight: 600 }}>{exp.name}</div>
                    <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
                      {exp.base_model} • {exp.training_mode}
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                  <span className={`badge ${statusColor(exp.status)}`}>{exp.status}</span>
                  {exp.status === 'pending' && (
                    <button className="btn btn-primary btn-sm" onClick={() => handleStart(exp.id)}>
                      Start
                    </button>
                  )}
                  <button className="btn btn-secondary btn-sm" onClick={() => viewDashboard(exp)}>
                    Dashboard
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {onNextStep && (
        <StepFooter
          currentStep="Training"
          nextStep="Compression"
          nextStepIcon="🗜️"
          isComplete={experiments.some((e) => e.status === 'completed')}
          hint="Start and complete an experiment to proceed"
          onNext={onNextStep}
        />
      )}
    </div>
  );
}
