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
  domain_pack_applied?: string | null;
  domain_pack_source?: string | null;
  domain_profile_applied?: string | null;
  domain_profile_source?: string | null;
  profile_training_defaults?: Record<string, unknown> | null;
  resolved_training_config?: Record<string, unknown> | null;
  profile_defaults_applied?: string[];
}

interface TrainingMetric {
  experiment_id?: number;
  epoch?: number;
  step?: number;
  train_loss?: number | null;
  eval_loss?: number | null;
  [key: string]: unknown;
}

interface TrainingEffectiveConfigResponse {
  domain_pack_applied?: string | null;
  domain_pack_source?: string | null;
  domain_profile_applied?: string | null;
  domain_profile_source?: string | null;
  profile_training_defaults?: Record<string, unknown> | null;
  resolved_training_config?: Record<string, unknown> | null;
  resolved_training_mode?: string;
  profile_defaults_applied?: string[];
}

type ConfigFieldKey =
  | 'chat_template'
  | 'learning_rate'
  | 'num_epochs'
  | 'batch_size'
  | 'optimizer'
  | 'use_lora'
  | 'lora_r'
  | 'lora_alpha'
  | 'target_modules'
  | 'gradient_checkpointing';

export default function TrainingPanel({ projectId, onNextStep }: TrainingPanelProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [activeExperiment, setActiveExperiment] = useState<Experiment | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [selectedForCompare, setSelectedForCompare] = useState<number[]>([]);
  const [showCompare, setShowCompare] = useState(false);
  const [taskState, setTaskState] = useState<string>('');
  const [trainingError, setTrainingError] = useState<string>('');

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
  const [useProfileDefaults, setUseProfileDefaults] = useState(true);
  const [touchedConfig, setTouchedConfig] = useState<Record<ConfigFieldKey, boolean>>({
    chat_template: false,
    learning_rate: false,
    num_epochs: false,
    batch_size: false,
    optimizer: false,
    use_lora: false,
    lora_r: false,
    lora_alpha: false,
    target_modules: false,
    gradient_checkpointing: false,
  });
  const [lastCreateSummary, setLastCreateSummary] = useState<{
    domainPackApplied: string | null;
    domainPackSource: string | null;
    domainProfileApplied: string | null;
    domainProfileSource: string | null;
    defaultsApplied: string[];
    profileDefaults: Record<string, unknown> | null;
    resolvedConfig: Record<string, unknown> | null;
  } | null>(null);
  const [effectivePreview, setEffectivePreview] = useState<TrainingEffectiveConfigResponse | null>(null);
  const [effectivePreviewLoading, setEffectivePreviewLoading] = useState(false);
  const [effectivePreviewError, setEffectivePreviewError] = useState('');

  const statusColor = (status: string) =>
    status === 'completed'
      ? 'badge-success'
      : status === 'running'
        ? 'badge-info'
        : status === 'failed'
          ? 'badge-error'
          : 'badge-warning';

  const buildTrainingConfigPayload = (): Record<string, unknown> => {
    const learningRate = Number.parseFloat(lr);
    const parsedTargetModules = targetModules
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);

    const config: Record<string, unknown> = {
      base_model: baseModel,
    };
    const includeField = (key: ConfigFieldKey): boolean => !useProfileDefaults || touchedConfig[key];
    if (includeField('chat_template')) config.chat_template = chatTemplate;
    if (includeField('learning_rate')) config.learning_rate = learningRate;
    if (includeField('num_epochs')) config.num_epochs = epochs;
    if (includeField('batch_size')) config.batch_size = batchSize;
    if (includeField('optimizer')) config.optimizer = optimizer;
    if (includeField('use_lora')) config.use_lora = useLora;
    if (includeField('lora_r')) config.lora_r = loraR;
    if (includeField('lora_alpha')) config.lora_alpha = loraAlpha;
    if (includeField('target_modules')) config.target_modules = parsedTargetModules;
    if (includeField('gradient_checkpointing')) config.gradient_checkpointing = gradientCheckpointing;
    return config;
  };

  const previewEffectiveConfig = async () => {
    setEffectivePreviewLoading(true);
    setEffectivePreviewError('');
    try {
      const config = buildTrainingConfigPayload();
      const res = await api.post<TrainingEffectiveConfigResponse>(
        `/projects/${projectId}/training/experiments/effective-config`,
        { config },
      );
      setEffectivePreview(res.data);
    } catch (err: any) {
      setEffectivePreview(null);
      setEffectivePreviewError(err?.response?.data?.detail || 'Failed to preview effective training config');
    } finally {
      setEffectivePreviewLoading(false);
    }
  };

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
    setTaskState('');
    setTrainingError('');
    setUseProfileDefaults(true);
    setTouchedConfig({
      chat_template: false,
      learning_rate: false,
      num_epochs: false,
      batch_size: false,
      optimizer: false,
      use_lora: false,
      lora_r: false,
      lora_alpha: false,
      target_modules: false,
      gradient_checkpointing: false,
    });
    setLastCreateSummary(null);
    setEffectivePreview(null);
    setEffectivePreviewError('');
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
          const nextTaskState = String(res.data?.task_status?.state || '').trim();
          if (nextTaskState) {
            setTaskState(nextTaskState);
          }
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

  useEffect(() => {
    if (!showCreate) {
      return;
    }
    void previewEffectiveConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showCreate, projectId]);

  const handleCreate = async () => {
    if (!name.trim()) return;
    setTrainingError('');

    const config = buildTrainingConfigPayload();

    try {
      const res = await api.post<Experiment>(`/projects/${projectId}/training/experiments`, { name, config });
      const created = res.data;
      setExperiments((prev) => [created, ...prev]);
      setLastCreateSummary({
        domainPackApplied: created.domain_pack_applied ?? null,
        domainPackSource: created.domain_pack_source ?? null,
        domainProfileApplied: created.domain_profile_applied ?? null,
        domainProfileSource: created.domain_profile_source ?? null,
        defaultsApplied: created.profile_defaults_applied || [],
        profileDefaults:
          created.profile_training_defaults && typeof created.profile_training_defaults === 'object'
            ? created.profile_training_defaults
            : null,
        resolvedConfig:
          created.resolved_training_config && typeof created.resolved_training_config === 'object'
            ? created.resolved_training_config
            : null,
      });
      setShowCreate(false);
      setName('');
      setTouchedConfig({
        chat_template: false,
        learning_rate: false,
        num_epochs: false,
        batch_size: false,
        optimizer: false,
        use_lora: false,
        lora_r: false,
        lora_alpha: false,
        target_modules: false,
        gradient_checkpointing: false,
      });
    } catch (err: any) {
      setTrainingError(err?.response?.data?.detail || 'Failed to create experiment');
    }
  };

  const handleStart = async (experimentId: number) => {
    setTrainingError('');
    try {
      const res = await api.post(`/projects/${projectId}/training/experiments/${experimentId}/start`);
      setExperiments((prev) =>
        prev.map((exp) => (exp.id === experimentId ? { ...exp, status: 'running' } : exp))
      );
      setTaskState(String(res.data?.task_id || '').trim() ? 'queued' : '');
      const exp = experiments.find((e) => e.id === experimentId);
      if (exp) {
        setActiveExperiment({ ...exp, status: 'running' });
        setMetrics([]);
        setTrainingLogs([]);
      }
    } catch (err: any) {
      setTrainingError(err?.response?.data?.detail || 'Failed to start training');
    }
  };

  const handleCancel = async (experimentId: number) => {
    setTrainingError('');
    try {
      await api.post(`/projects/${projectId}/training/experiments/${experimentId}/cancel`);
      setActiveExperiment((prev) => (prev ? { ...prev, status: 'cancelled' } : prev));
      setTaskState('cancel_requested');
      refreshExperiments().catch(() => undefined);
    } catch (err: any) {
      setTrainingError(err?.response?.data?.detail || 'Failed to cancel training');
    }
  };

  const viewDashboard = (exp: Experiment) => {
    setActiveExperiment(exp);
    setMetrics([]);
    setTrainingLogs([]);
    setTaskState('');
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
              setTaskState('');
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
              {activeExperiment.domain_pack_applied && (
                <div style={{ color: 'var(--text-tertiary)', fontSize: 'var(--font-size-xs)' }}>
                  Pack: {activeExperiment.domain_pack_applied}
                  {activeExperiment.domain_pack_source ? ` (${activeExperiment.domain_pack_source})` : ''}
                </div>
              )}
              {activeExperiment.domain_profile_applied && (
                <div style={{ color: 'var(--text-tertiary)', fontSize: 'var(--font-size-xs)' }}>
                  Profile: {activeExperiment.domain_profile_applied}
                  {activeExperiment.domain_profile_source ? ` (${activeExperiment.domain_profile_source})` : ''}
                </div>
              )}
              {taskState && (
                <div style={{ color: 'var(--text-tertiary)', fontSize: 'var(--font-size-xs)' }}>
                  Worker task state: {taskState}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {activeExperiment.status === 'running' && (
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => void handleCancel(activeExperiment.id)}
                >
                  Cancel
                </button>
              )}
              <span className={`badge ${statusColor(activeExperiment.status)}`} style={{ fontSize: 'var(--font-size-md)', padding: '6px 16px' }}>
                {activeExperiment.status.toUpperCase()}
              </span>
            </div>
          </div>

          {trainingError && (
            <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>
              {trainingError}
            </div>
          )}

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
            <div className="form-group" style={{ marginBottom: 'var(--space-md)' }}>
              <label className="form-label" style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                <input
                  type="checkbox"
                  checked={useProfileDefaults}
                  onChange={(e) => setUseProfileDefaults(e.target.checked)}
                />
                Use domain runtime defaults for untouched fields
              </label>
              <div className="form-hint">
                Base model is always sent. Other fields are only sent after you edit them.
              </div>
            </div>
            <div className="form-group" style={{ marginBottom: 'var(--space-md)' }}>
              <button
                className="btn btn-secondary"
                onClick={() => void previewEffectiveConfig()}
                disabled={effectivePreviewLoading}
              >
                {effectivePreviewLoading ? 'Resolving...' : 'Preview Effective Config'}
              </button>
            </div>
            {effectivePreviewError && (
              <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>
                {effectivePreviewError}
              </div>
            )}
            {effectivePreview && (
              <div className="resolved-defaults-panel">
                <div className="resolved-defaults-panel__title">Effective Config Preview (Pre-create)</div>
                <div className="resolved-defaults-panel__kv">
                  <span>Applied Pack</span>
                  <strong>
                    {effectivePreview.domain_pack_applied
                      ? `${effectivePreview.domain_pack_applied} (${effectivePreview.domain_pack_source || 'unknown'})`
                      : 'none'}
                  </strong>
                </div>
                <div className="resolved-defaults-panel__kv">
                  <span>Applied Profile</span>
                  <strong>
                    {effectivePreview.domain_profile_applied
                      ? `${effectivePreview.domain_profile_applied} (${effectivePreview.domain_profile_source || 'unknown'})`
                      : 'none'}
                  </strong>
                </div>
                <div className="resolved-defaults-panel__kv">
                  <span>Resolved Training Mode</span>
                  <strong>{effectivePreview.resolved_training_mode || 'sft'}</strong>
                </div>
                <div className="resolved-defaults-panel__kv">
                  <span>Runtime Fields Applied</span>
                  <strong>
                    {effectivePreview.profile_defaults_applied && effectivePreview.profile_defaults_applied.length > 0
                      ? effectivePreview.profile_defaults_applied.join(', ')
                      : 'none'}
                  </strong>
                </div>
                <div className="resolved-defaults-panel__grid">
                  <div>
                    <div className="resolved-defaults-panel__subtitle">Resolved Training Config</div>
                    <pre className="resolved-defaults-panel__json">
                      {JSON.stringify(effectivePreview.resolved_training_config || {}, null, 2)}
                    </pre>
                  </div>
                  <div>
                    <div className="resolved-defaults-panel__subtitle">Runtime Training Defaults</div>
                    <pre className="resolved-defaults-panel__json">
                      {JSON.stringify(effectivePreview.profile_training_defaults || {}, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
              <div>
                <h4 style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-md)', textTransform: 'uppercase' }}>Basic HParams</h4>
                <div className="form-group">
                  <label className="form-label">Base Model</label>
                  <input className="input" value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />
                </div>
                <div className="form-group">
                  <label className="form-label">Chat Template</label>
                  <select
                    className="input"
                    value={chatTemplate}
                    onChange={(e) => {
                      setChatTemplate(e.target.value);
                      setTouchedConfig((prev) => ({ ...prev, chat_template: true }));
                    }}
                  >
                    <option value="llama3">Llama-3</option>
                    <option value="chatml">ChatML</option>
                    <option value="zephyr">Zephyr</option>
                    <option value="phi3">Phi-3</option>
                  </select>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div className="form-group">
                    <label className="form-label">Epochs</label>
                    <input
                      className="input"
                      type="number"
                      value={epochs}
                      onChange={(e) => {
                        setEpochs(Number(e.target.value) || 1);
                        setTouchedConfig((prev) => ({ ...prev, num_epochs: true }));
                      }}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Batch Size</label>
                    <input
                      className="input"
                      type="number"
                      value={batchSize}
                      onChange={(e) => {
                        setBatchSize(Number(e.target.value) || 1);
                        setTouchedConfig((prev) => ({ ...prev, batch_size: true }));
                      }}
                    />
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div className="form-group">
                    <label className="form-label">Learning Rate</label>
                    <input
                      className="input"
                      value={lr}
                      onChange={(e) => {
                        setLr(e.target.value);
                        setTouchedConfig((prev) => ({ ...prev, learning_rate: true }));
                      }}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Optimizer</label>
                    <select
                      className="input"
                      value={optimizer}
                      onChange={(e) => {
                        setOptimizer(e.target.value);
                        setTouchedConfig((prev) => ({ ...prev, optimizer: true }));
                      }}
                    >
                      <option value="paged_adamw_8bit">Paged AdamW (8-bit)</option>
                      <option value="adamw_torch">AdamW</option>
                    </select>
                  </div>
                </div>
              </div>

              <div>
                <h4 style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-md)', textTransform: 'uppercase' }}>Advanced & PEFT</h4>
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={useLora}
                    onChange={(e) => {
                      setUseLora(e.target.checked);
                      setTouchedConfig((prev) => ({ ...prev, use_lora: true }));
                    }}
                  />
                  <label className="form-label" style={{ margin: 0 }}>Enable LoRA</label>
                </div>
                {useLora && (
                  <div style={{ padding: 'var(--space-md)', background: 'rgba(255,255,255,0.03)', borderRadius: 8, marginBottom: 16 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div className="form-group">
                        <label className="form-label">Rank (r)</label>
                        <input
                          className="input"
                          type="number"
                          value={loraR}
                          onChange={(e) => {
                            setLoraR(Number(e.target.value) || 1);
                            setTouchedConfig((prev) => ({ ...prev, lora_r: true }));
                          }}
                        />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Alpha</label>
                        <input
                          className="input"
                          type="number"
                          value={loraAlpha}
                          onChange={(e) => {
                            setLoraAlpha(Number(e.target.value) || 1);
                            setTouchedConfig((prev) => ({ ...prev, lora_alpha: true }));
                          }}
                        />
                      </div>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Target Modules (comma-separated)</label>
                      <input
                        className="input"
                        value={targetModules}
                        onChange={(e) => {
                          setTargetModules(e.target.value);
                          setTouchedConfig((prev) => ({ ...prev, target_modules: true }));
                        }}
                      />
                    </div>
                  </div>
                )}
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={gradientCheckpointing}
                    onChange={(e) => {
                      setGradientCheckpointing(e.target.checked);
                      setTouchedConfig((prev) => ({ ...prev, gradient_checkpointing: true }));
                    }}
                  />
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

        {trainingError && (
          <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>
            {trainingError}
          </div>
        )}

        {lastCreateSummary && (
          <div className="resolved-defaults-panel">
            <div className="resolved-defaults-panel__title">Resolved Defaults</div>
            <div className="resolved-defaults-panel__kv">
              <span>Applied Pack</span>
              <strong>
                {lastCreateSummary.domainPackApplied
                  ? `${lastCreateSummary.domainPackApplied} (${lastCreateSummary.domainPackSource || 'unknown'})`
                  : 'none'}
              </strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Applied Profile</span>
              <strong>
                {lastCreateSummary.domainProfileApplied
                  ? `${lastCreateSummary.domainProfileApplied} (${lastCreateSummary.domainProfileSource || 'unknown'})`
                  : 'none'}
              </strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Runtime Fields Applied</span>
              <strong>
                {lastCreateSummary.defaultsApplied.length > 0
                  ? lastCreateSummary.defaultsApplied.join(', ')
                  : 'none'}
              </strong>
            </div>
            <div className="resolved-defaults-panel__grid">
              <div>
                <div className="resolved-defaults-panel__subtitle">Resolved Training Config</div>
                <pre className="resolved-defaults-panel__json">
                  {JSON.stringify(lastCreateSummary.resolvedConfig || {}, null, 2)}
                </pre>
              </div>
              <div>
                <div className="resolved-defaults-panel__subtitle">Runtime Training Defaults</div>
                <pre className="resolved-defaults-panel__json">
                  {JSON.stringify(lastCreateSummary.profileDefaults || {}, null, 2)}
                </pre>
              </div>
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
                    {exp.domain_pack_applied && (
                      <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)' }}>
                        Pack: {exp.domain_pack_applied}
                        {exp.domain_pack_source ? ` (${exp.domain_pack_source})` : ''}
                      </div>
                    )}
                    {exp.domain_profile_applied && (
                      <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)' }}>
                        Profile: {exp.domain_profile_applied}
                        {exp.domain_profile_source ? ` (${exp.domain_profile_source})` : ''}
                      </div>
                    )}
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
