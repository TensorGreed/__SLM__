import { useEffect, useState } from 'react';

import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { TerminalConsole } from '../shared/TerminalConsole';
import ExperimentCompare from './ExperimentCompare';
import { buildWsUrl } from '../../utils/ws';
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

interface TrainingPreflightReport {
  ok: boolean;
  errors: string[];
  warnings: string[];
  hints?: string[];
  capability_summary?: Record<string, unknown>;
}

interface TrainingPreflightPreviewResponse extends TrainingEffectiveConfigResponse {
  preflight?: TrainingPreflightReport;
}

interface TrainingExperimentPreflightResponse {
  experiment_id: number;
  status: string;
  resolved_training_config?: Record<string, unknown> | null;
  preflight?: TrainingPreflightReport;
}

interface TrainingPlanChange {
  field: string;
  from?: unknown;
  to?: unknown;
  reason?: string;
}

interface TrainingPreflightPlanSuggestion {
  profile: string;
  title: string;
  description: string;
  config: Record<string, unknown>;
  changes: TrainingPlanChange[];
  estimated_vram_risk: string;
  estimated_vram_score: number;
  estimated_vram_note?: string | null;
  preflight: TrainingPreflightReport;
}

interface TrainingPreflightPlanReport {
  base_preflight?: TrainingPreflightReport;
  suggestions: TrainingPreflightPlanSuggestion[];
  recommended_profile?: string;
  profile_order?: string[];
}

interface TrainingPreflightPlanResponse extends TrainingEffectiveConfigResponse {
  plan?: TrainingPreflightPlanReport;
}

interface TrainingPreferencesResponse {
  project_id: number;
  preferred_plan_profile: string;
  profile_options?: string[];
  source?: string;
}

interface TrainingRuntimeSpec {
  runtime_id: string;
  label: string;
  description?: string;
  execution_backend?: string;
  required_dependencies?: string[];
  supports_task_tracking?: boolean;
  supports_cancellation?: boolean;
  is_builtin?: boolean;
}

interface TrainingRuntimeCatalogResponse {
  project_id: number;
  default_runtime_id?: string;
  runtime_count?: number;
  legacy_aliases?: Record<string, string>;
  runtimes?: TrainingRuntimeSpec[];
}

const METRIC_PREFIX = 'SLM_METRIC ';
const PLAN_PROFILE_STORAGE_PREFIX = 'slm-training-plan-profile';

function parseNumericField(text: string, key: string): number | null {
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(
    `[\"']${escapedKey}[\"']\\s*:\\s*[\"']?([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)[\"']?`,
  );
  const match = text.match(pattern);
  if (!match || !match[1]) {
    return null;
  }
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
}

function parseMetricFromLogLine(text: string, experimentId: number): TrainingMetric | null {
  const trimmed = String(text || '').trim();
  if (!trimmed) {
    return null;
  }

  const markerIndex = trimmed.indexOf(METRIC_PREFIX);
  if (markerIndex >= 0) {
    const payload = trimmed.slice(markerIndex + METRIC_PREFIX.length).trim();
    try {
      const parsed = JSON.parse(payload);
      if (parsed && typeof parsed === 'object') {
        const metric: TrainingMetric = { experiment_id: experimentId };
        const parsedStep = Number((parsed as Record<string, unknown>).step);
        const parsedEpoch = Number((parsed as Record<string, unknown>).epoch);
        const parsedTrainLoss = Number((parsed as Record<string, unknown>).train_loss);
        const parsedEvalLoss = Number((parsed as Record<string, unknown>).eval_loss);
        if (Number.isFinite(parsedStep)) metric.step = parsedStep;
        if (Number.isFinite(parsedEpoch)) metric.epoch = parsedEpoch;
        if (Number.isFinite(parsedTrainLoss)) metric.train_loss = parsedTrainLoss;
        if (Number.isFinite(parsedEvalLoss)) metric.eval_loss = parsedEvalLoss;
        if (
          metric.step !== undefined ||
          metric.epoch !== undefined ||
          metric.train_loss !== undefined ||
          metric.eval_loss !== undefined
        ) {
          return metric;
        }
      }
    } catch {
      // fall through to legacy trainer log parsing
    }
  }

  if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) {
    return null;
  }
  if (
    !trimmed.includes("'loss'") &&
    !trimmed.includes('"loss"') &&
    !trimmed.includes("'train_loss'") &&
    !trimmed.includes('"train_loss"') &&
    !trimmed.includes("'eval_loss'") &&
    !trimmed.includes('"eval_loss"')
  ) {
    return null;
  }

  const epoch = parseNumericField(trimmed, 'epoch');
  const step = parseNumericField(trimmed, 'step');
  const trainLoss = parseNumericField(trimmed, 'train_loss') ?? parseNumericField(trimmed, 'loss');
  const evalLoss = parseNumericField(trimmed, 'eval_loss');

  if (epoch === null && step === null && trainLoss === null && evalLoss === null) {
    return null;
  }
  return {
    experiment_id: experimentId,
    ...(epoch !== null ? { epoch } : {}),
    ...(step !== null ? { step } : {}),
    ...(trainLoss !== null ? { train_loss: trainLoss } : {}),
    ...(evalLoss !== null ? { eval_loss: evalLoss } : {}),
  };
}

type ConfigFieldKey =
  | 'training_runtime_id'
  | 'task_type'
  | 'trainer_backend'
  | 'chat_template'
  | 'learning_rate'
  | 'num_epochs'
  | 'batch_size'
  | 'gradient_accumulation_steps'
  | 'max_seq_length'
  | 'optimizer'
  | 'save_steps'
  | 'eval_steps'
  | 'sequence_packing'
  | 'use_lora'
  | 'lora_r'
  | 'lora_alpha'
  | 'target_modules'
  | 'fp16'
  | 'bf16'
  | 'flash_attention'
  | 'auto_oom_retry'
  | 'max_oom_retries'
  | 'oom_retry_seq_shrink'
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
  const [trainingRuntimeId, setTrainingRuntimeId] = useState('auto');
  const [taskType, setTaskType] = useState('causal_lm');
  const [trainerBackend, setTrainerBackend] = useState('auto');
  const [chatTemplate, setChatTemplate] = useState('llama3');
  const [lr, setLr] = useState('2e-4');
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState(4);
  const [maxSeqLength, setMaxSeqLength] = useState(2048);
  const [optimizer, setOptimizer] = useState('paged_adamw_8bit');
  const [saveSteps, setSaveSteps] = useState(100);
  const [evalSteps, setEvalSteps] = useState(100);
  const [sequencePacking, setSequencePacking] = useState(true);
  const [useLora, setUseLora] = useState(true);
  const [loraR, setLoraR] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [targetModules, setTargetModules] = useState('q_proj, v_proj');
  const [fp16, setFp16] = useState(false);
  const [bf16, setBf16] = useState(true);
  const [flashAttention, setFlashAttention] = useState(true);
  const [autoOomRetry, setAutoOomRetry] = useState(true);
  const [maxOomRetries, setMaxOomRetries] = useState(2);
  const [oomRetrySeqShrink, setOomRetrySeqShrink] = useState('0.75');
  const [gradientCheckpointing, setGradientCheckpointing] = useState(true);
  const [useProfileDefaults, setUseProfileDefaults] = useState(true);
  const [touchedConfig, setTouchedConfig] = useState<Record<ConfigFieldKey, boolean>>({
    training_runtime_id: false,
    task_type: false,
    trainer_backend: false,
    chat_template: false,
    learning_rate: false,
    num_epochs: false,
    batch_size: false,
    gradient_accumulation_steps: false,
    max_seq_length: false,
    optimizer: false,
    save_steps: false,
    eval_steps: false,
    sequence_packing: false,
    use_lora: false,
    lora_r: false,
    lora_alpha: false,
    target_modules: false,
    fp16: false,
    bf16: false,
    flash_attention: false,
    auto_oom_retry: false,
    max_oom_retries: false,
    oom_retry_seq_shrink: false,
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
  const [preflightPreview, setPreflightPreview] = useState<TrainingPreflightReport | null>(null);
  const [preflightPreviewLoading, setPreflightPreviewLoading] = useState(false);
  const [preflightPreviewError, setPreflightPreviewError] = useState('');
  const [preflightPlan, setPreflightPlan] = useState<TrainingPreflightPlanReport | null>(null);
  const [preflightPlanLoading, setPreflightPlanLoading] = useState(false);
  const [preflightPlanError, setPreflightPlanError] = useState('');
  const [preferredPlanProfile, setPreferredPlanProfile] = useState('balanced');
  const [trainingWarnings, setTrainingWarnings] = useState<string[]>([]);
  const [runtimeCatalog, setRuntimeCatalog] = useState<TrainingRuntimeCatalogResponse | null>(null);
  const [runtimeCatalogError, setRuntimeCatalogError] = useState('');

  const statusColor = (status: string) =>
    status === 'completed'
      ? 'badge-success'
      : status === 'running'
        ? 'badge-info'
        : status === 'failed'
          ? 'badge-error'
          : 'badge-warning';

  const activeExperimentKey = activeExperiment ? `${activeExperiment.id}:${activeExperiment.status}` : '';

  const buildTrainingConfigPayload = (): Record<string, unknown> => {
    const learningRate = Number.parseFloat(lr);
    const retryShrink = Number.parseFloat(oomRetrySeqShrink);
    const parsedTargetModules = targetModules
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);

    const config: Record<string, unknown> = {
      base_model: baseModel,
    };
    const includeField = (key: ConfigFieldKey): boolean => !useProfileDefaults || touchedConfig[key];
    if (includeField('training_runtime_id')) config.training_runtime_id = trainingRuntimeId;
    if (includeField('task_type')) config.task_type = taskType;
    if (includeField('trainer_backend')) config.trainer_backend = trainerBackend;
    if (includeField('chat_template')) config.chat_template = chatTemplate;
    if (includeField('learning_rate')) config.learning_rate = learningRate;
    if (includeField('num_epochs')) config.num_epochs = epochs;
    if (includeField('batch_size')) config.batch_size = batchSize;
    if (includeField('gradient_accumulation_steps')) config.gradient_accumulation_steps = gradientAccumulationSteps;
    if (includeField('max_seq_length')) config.max_seq_length = maxSeqLength;
    if (includeField('optimizer')) config.optimizer = optimizer;
    if (includeField('save_steps')) config.save_steps = saveSteps;
    if (includeField('eval_steps')) config.eval_steps = evalSteps;
    if (includeField('sequence_packing')) config.sequence_packing = sequencePacking;
    if (includeField('use_lora')) config.use_lora = useLora;
    if (includeField('lora_r')) config.lora_r = loraR;
    if (includeField('lora_alpha')) config.lora_alpha = loraAlpha;
    if (includeField('target_modules')) config.target_modules = parsedTargetModules;
    if (includeField('fp16')) config.fp16 = fp16;
    if (includeField('bf16')) config.bf16 = bf16;
    if (includeField('flash_attention')) config.flash_attention = flashAttention;
    if (includeField('auto_oom_retry')) config.auto_oom_retry = autoOomRetry;
    if (includeField('max_oom_retries')) config.max_oom_retries = maxOomRetries;
    if (includeField('oom_retry_seq_shrink') && Number.isFinite(retryShrink)) {
      config.oom_retry_seq_shrink = retryShrink;
    }
    if (includeField('gradient_checkpointing')) config.gradient_checkpointing = gradientCheckpointing;
    return config;
  };

  const applySuggestedConfig = (config: Record<string, unknown>) => {
    const parseNumber = (value: unknown, fallback: number): number => {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : fallback;
    };
    const parseBoolean = (value: unknown, fallback: boolean): boolean => {
      if (typeof value === 'boolean') return value;
      if (typeof value === 'number') return value !== 0;
      if (typeof value === 'string') {
        const token = value.trim().toLowerCase();
        if (['true', '1', 'yes', 'on'].includes(token)) return true;
        if (['false', '0', 'no', 'off', ''].includes(token)) return false;
      }
      return fallback;
    };
    const parseString = (value: unknown, fallback: string): string =>
      typeof value === 'string' && value.trim() ? value : fallback;

    if (typeof config.base_model === 'string' && config.base_model.trim()) setBaseModel(config.base_model);
    setTrainingRuntimeId(parseString(config.training_runtime_id, trainingRuntimeId));
    setTaskType(parseString(config.task_type, taskType));
    setTrainerBackend(parseString(config.trainer_backend, trainerBackend));
    setChatTemplate(parseString(config.chat_template, chatTemplate));
    setLr(String(config.learning_rate ?? lr));
    setEpochs(Math.max(1, parseNumber(config.num_epochs, epochs)));
    setBatchSize(Math.max(1, parseNumber(config.batch_size, batchSize)));
    setGradientAccumulationSteps(Math.max(1, parseNumber(config.gradient_accumulation_steps, gradientAccumulationSteps)));
    setMaxSeqLength(Math.max(128, parseNumber(config.max_seq_length, maxSeqLength)));
    setOptimizer(parseString(config.optimizer, optimizer));
    setSaveSteps(Math.max(1, parseNumber(config.save_steps, saveSteps)));
    setEvalSteps(Math.max(1, parseNumber(config.eval_steps, evalSteps)));
    setSequencePacking(parseBoolean(config.sequence_packing, sequencePacking));
    const nextUseLora = parseBoolean(config.use_lora, useLora);
    setUseLora(nextUseLora);
    setLoraR(Math.max(1, parseNumber(config.lora_r, loraR)));
    setLoraAlpha(Math.max(1, parseNumber(config.lora_alpha, loraAlpha)));
    if (Array.isArray(config.target_modules)) {
      setTargetModules(
        config.target_modules
          .map((item) => String(item).trim())
          .filter(Boolean)
          .join(', '),
      );
    }
    const nextFp16 = parseBoolean(config.fp16, fp16);
    const nextBf16 = parseBoolean(config.bf16, bf16);
    if (nextFp16 && nextBf16) {
      setFp16(false);
      setBf16(true);
    } else {
      setFp16(nextFp16);
      setBf16(nextBf16);
    }
    setFlashAttention(parseBoolean(config.flash_attention, flashAttention));
    setAutoOomRetry(parseBoolean(config.auto_oom_retry, autoOomRetry));
    setMaxOomRetries(Math.max(0, Math.min(5, parseNumber(config.max_oom_retries, maxOomRetries))));
    setOomRetrySeqShrink(String(config.oom_retry_seq_shrink ?? oomRetrySeqShrink));
    setGradientCheckpointing(parseBoolean(config.gradient_checkpointing, gradientCheckpointing));

    setUseProfileDefaults(false);
    setTouchedConfig((prev) => {
      const next = { ...prev };
      (Object.keys(next) as ConfigFieldKey[]).forEach((key) => {
        if (Object.prototype.hasOwnProperty.call(config, key)) {
          next[key] = true;
        }
      });
      return next;
    });
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

  const runPreflightPreview = async () => {
    setPreflightPreviewLoading(true);
    setPreflightPreviewError('');
    try {
      const config = buildTrainingConfigPayload();
      const res = await api.post<TrainingPreflightPreviewResponse>(
        `/projects/${projectId}/training/experiments/preflight`,
        { config },
      );
      setEffectivePreview({
        domain_pack_applied: res.data?.domain_pack_applied ?? null,
        domain_pack_source: res.data?.domain_pack_source ?? null,
        domain_profile_applied: res.data?.domain_profile_applied ?? null,
        domain_profile_source: res.data?.domain_profile_source ?? null,
        profile_training_defaults: res.data?.profile_training_defaults ?? null,
        resolved_training_config: res.data?.resolved_training_config ?? null,
        resolved_training_mode: res.data?.resolved_training_mode ?? 'sft',
        profile_defaults_applied: res.data?.profile_defaults_applied ?? [],
      });
      setPreflightPreview(res.data?.preflight || null);
    } catch (err: any) {
      setPreflightPreview(null);
      setPreflightPreviewError(err?.response?.data?.detail || 'Failed to run capability preflight');
    } finally {
      setPreflightPreviewLoading(false);
    }
  };

  const runPreflightPlan = async () => {
    setPreflightPlanLoading(true);
    setPreflightPlanError('');
    try {
      const config = buildTrainingConfigPayload();
      const res = await api.post<TrainingPreflightPlanResponse>(
        `/projects/${projectId}/training/experiments/preflight/plan`,
        { config },
      );
      setEffectivePreview({
        domain_pack_applied: res.data?.domain_pack_applied ?? null,
        domain_pack_source: res.data?.domain_pack_source ?? null,
        domain_profile_applied: res.data?.domain_profile_applied ?? null,
        domain_profile_source: res.data?.domain_profile_source ?? null,
        profile_training_defaults: res.data?.profile_training_defaults ?? null,
        resolved_training_config: res.data?.resolved_training_config ?? null,
        resolved_training_mode: res.data?.resolved_training_mode ?? 'sft',
        profile_defaults_applied: res.data?.profile_defaults_applied ?? [],
      });
      const plan = res.data?.plan || null;
      setPreflightPlan(plan);
      setPreflightPreview(plan?.base_preflight || null);
    } catch (err: any) {
      setPreflightPlan(null);
      setPreflightPlanError(err?.response?.data?.detail || 'Failed to generate preflight plan');
    } finally {
      setPreflightPlanLoading(false);
    }
  };

  const loadPreferredPlanProfile = async () => {
    try {
      const res = await api.get<TrainingPreferencesResponse>(`/projects/${projectId}/training/preferences`);
      const preferred = String(res.data?.preferred_plan_profile || '').trim().toLowerCase();
      if (preferred) {
        setPreferredPlanProfile(preferred);
        try {
          window.localStorage.setItem(`${PLAN_PROFILE_STORAGE_PREFIX}:${projectId}`, preferred);
        } catch {
          // no-op for storage failures
        }
        return;
      }
    } catch {
      // fallback to cached local value
    }
    try {
      const stored = window.localStorage.getItem(`${PLAN_PROFILE_STORAGE_PREFIX}:${projectId}`);
      const fallback = String(stored || '').trim().toLowerCase();
      setPreferredPlanProfile(fallback || 'balanced');
    } catch {
      setPreferredPlanProfile('balanced');
    }
  };

  const loadTrainingRuntimes = async () => {
    try {
      const res = await api.get<TrainingRuntimeCatalogResponse>(`/projects/${projectId}/training/runtimes`);
      setRuntimeCatalog(res.data || null);
      setRuntimeCatalogError('');
    } catch (err: any) {
      setRuntimeCatalog(null);
      setRuntimeCatalogError(err?.response?.data?.detail || 'Failed to load runtime catalog');
    }
  };

  const persistPreferredPlanProfile = async (profile: string) => {
    const normalized = String(profile || '').trim().toLowerCase();
    if (!normalized) return;
    setPreferredPlanProfile(normalized);
    try {
      await api.put<TrainingPreferencesResponse>(
        `/projects/${projectId}/training/preferences`,
        { preferred_plan_profile: normalized },
      );
    } catch {
      // Keep local fallback if backend persistence is unavailable.
    }
    try {
      window.localStorage.setItem(`${PLAN_PROFILE_STORAGE_PREFIX}:${projectId}`, normalized);
    } catch {
      // no-op for storage failures
    }
  };

  const applyPlanSuggestion = (suggestion: TrainingPreflightPlanSuggestion) => {
    applySuggestedConfig(suggestion.config || {});
    setPreflightPreview(suggestion.preflight || null);
    const warningItems = Array.isArray(suggestion.preflight?.warnings)
      ? suggestion.preflight.warnings.filter(Boolean)
      : [];
    setTrainingWarnings(warningItems);

    const profile = String(suggestion.profile || '').trim();
    if (profile) {
      void persistPreferredPlanProfile(profile);
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
    setTrainingRuntimeId('auto');
    setUseProfileDefaults(true);
    setTouchedConfig({
      training_runtime_id: false,
      task_type: false,
      trainer_backend: false,
      chat_template: false,
      learning_rate: false,
      num_epochs: false,
      batch_size: false,
      gradient_accumulation_steps: false,
      max_seq_length: false,
      optimizer: false,
      save_steps: false,
      eval_steps: false,
      sequence_packing: false,
      use_lora: false,
      lora_r: false,
      lora_alpha: false,
      target_modules: false,
      fp16: false,
      bf16: false,
      flash_attention: false,
      auto_oom_retry: false,
      max_oom_retries: false,
      oom_retry_seq_shrink: false,
      gradient_checkpointing: false,
    });
    setLastCreateSummary(null);
    setEffectivePreview(null);
    setEffectivePreviewError('');
    setPreflightPreview(null);
    setPreflightPreviewError('');
    setPreflightPlan(null);
    setPreflightPlanError('');
    setTrainingWarnings([]);
    setRuntimeCatalog(null);
    setRuntimeCatalogError('');
    void loadPreferredPlanProfile();
    void loadTrainingRuntimes();
    refreshExperiments().catch((err) => console.error('Failed to load experiments', err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  useEffect(() => {
    if (!activeExperiment || activeExperiment.status !== 'running') {
      return;
    }
    const experimentId = activeExperiment.id;
    const interval = window.setInterval(() => {
      api
        .get(`/projects/${projectId}/training/experiments/${experimentId}/status`)
        .then((res) => {
          const status = String(res.data?.status || '');
          if (!status) return;
          setActiveExperiment((prev) => {
            if (!prev || prev.id !== experimentId) return prev;
            if (prev.status === status) return prev;
            return { ...prev, status };
          });
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
  }, [activeExperimentKey, projectId]);

  useEffect(() => {
    if (!activeExperiment || activeExperiment.status !== 'running') {
      return;
    }
    const experimentId = activeExperiment.id;

    const wsUrl = buildWsUrl(`/api/projects/${projectId}/training/ws/${experimentId}`);
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'init') {
          setMetrics(Array.isArray(data.metrics) ? data.metrics : []);
          return;
        }
        if (data.type === 'metric' && data.metric) {
          setMetrics((prev) => {
            const nextMetric = data.metric as TrainingMetric;
            const last = prev[prev.length - 1];
            if (
              last &&
              last.step === nextMetric.step &&
              last.epoch === nextMetric.epoch &&
              last.train_loss === nextMetric.train_loss &&
              last.eval_loss === nextMetric.eval_loss
            ) {
              return prev;
            }
            return [...prev.slice(-199), nextMetric];
          });
          return;
        }
        if (data.type === 'log' && data.text) {
          const text = String(data.text);
          const metricFromLog = parseMetricFromLogLine(text, experimentId);
          if (metricFromLog) {
            setMetrics((prev) => {
              const last = prev[prev.length - 1];
              if (
                last &&
                last.step === metricFromLog.step &&
                last.epoch === metricFromLog.epoch &&
                last.train_loss === metricFromLog.train_loss &&
                last.eval_loss === metricFromLog.eval_loss
              ) {
                return prev;
              }
              return [...prev.slice(-199), metricFromLog];
            });
          }
          setTrainingLogs((prev) => [...prev.slice(-999), text]);
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
  }, [activeExperimentKey, projectId]);

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
    setTrainingWarnings([]);

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
      setTrainingRuntimeId('auto');
      setPreflightPreview(null);
      setPreflightPreviewError('');
      setPreflightPlan(null);
      setPreflightPlanError('');
      setTouchedConfig({
        training_runtime_id: false,
        task_type: false,
        trainer_backend: false,
        chat_template: false,
        learning_rate: false,
        num_epochs: false,
        batch_size: false,
        gradient_accumulation_steps: false,
        max_seq_length: false,
        optimizer: false,
        save_steps: false,
        eval_steps: false,
        sequence_packing: false,
        use_lora: false,
        lora_r: false,
        lora_alpha: false,
        target_modules: false,
        fp16: false,
        bf16: false,
        flash_attention: false,
        auto_oom_retry: false,
        max_oom_retries: false,
        oom_retry_seq_shrink: false,
        gradient_checkpointing: false,
      });
    } catch (err: any) {
      setTrainingError(err?.response?.data?.detail || 'Failed to create experiment');
    }
  };

  const handleStart = async (experimentId: number) => {
    setTrainingError('');
    setTrainingWarnings([]);
    try {
      const preflightRes = await api.get<TrainingExperimentPreflightResponse>(
        `/projects/${projectId}/training/experiments/${experimentId}/preflight`,
      );
      const preflight = preflightRes.data?.preflight;
      if (preflight && !preflight.ok) {
        const errors = Array.isArray(preflight.errors) ? preflight.errors.filter(Boolean) : [];
        const hints = Array.isArray(preflight.hints) ? preflight.hints.filter(Boolean) : [];
        const hintText = hints.length > 0 ? ` Fix hints: ${hints.slice(0, 2).join(' | ')}` : '';
        setTrainingError(
          errors.length > 0
            ? `Preflight failed: ${errors.join(' | ')}${hintText}`
            : 'Preflight failed due to incompatible configuration.',
        );
        return;
      }
      const warnings = Array.isArray(preflight?.warnings) ? preflight.warnings.filter(Boolean) : [];
      setTrainingWarnings(warnings);

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
    setTrainingWarnings([]);
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
    const totalEpochs = Number(activeExperiment.config?.num_epochs || 3);
    const epochValue = typeof latestMetric.epoch === 'number' ? Number(latestMetric.epoch) : null;
    const currentEpoch = epochValue !== null ? epochValue.toFixed(2) : '—';
    const currentStep = typeof latestMetric.step === 'number' ? Number(latestMetric.step) : null;
    const completedEpochs = epochValue !== null ? Math.floor(epochValue) : 0;
    let epochState = 'Waiting for first metric...';
    if (epochValue !== null) {
      if (completedEpochs >= totalEpochs) {
        epochState = `All ${totalEpochs} epochs completed`;
      } else {
        epochState = `Epoch ${Math.min(totalEpochs, completedEpochs + 1)} running`;
      }
    }
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
              setTrainingWarnings([]);
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
          {trainingWarnings.length > 0 && (
            <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-warning)', fontSize: 'var(--font-size-sm)' }}>
              Preflight warnings: {trainingWarnings.join(' | ')}
            </div>
          )}

          <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-md)', marginBottom: 'var(--space-xl)' }}>
            <div className="metric-box box-blue">
              <span className="mb-label">Current Epoch</span>
              <span className="mb-value">{currentEpoch} / {totalEpochs}</span>
              <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', marginTop: 6 }}>
                {epochState}
                {currentStep !== null ? ` • step ${currentStep}` : ''}
              </div>
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
          <button
            className="btn btn-primary"
            onClick={() => {
              setShowCreate((prev) => {
                const next = !prev;
                if (next) {
                  setPreflightPreview(null);
                  setPreflightPreviewError('');
                  setPreflightPlan(null);
                  setPreflightPlanError('');
                }
                return next;
              });
            }}
          >
            + New Experiment
          </button>
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
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => void previewEffectiveConfig()}
                  disabled={effectivePreviewLoading}
                >
                  {effectivePreviewLoading ? 'Resolving...' : 'Preview Effective Config'}
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => void runPreflightPreview()}
                  disabled={preflightPreviewLoading}
                >
                  {preflightPreviewLoading ? 'Checking...' : 'Run Capability Preflight'}
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => void runPreflightPlan()}
                  disabled={preflightPlanLoading}
                >
                  {preflightPlanLoading ? 'Planning...' : 'Run Preflight Plan'}
                </button>
              </div>
            </div>
            {effectivePreviewError && (
              <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>
                {effectivePreviewError}
              </div>
            )}
            {preflightPreviewError && (
              <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>
                {preflightPreviewError}
              </div>
            )}
            {preflightPlanError && (
              <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>
                {preflightPlanError}
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
            {preflightPreview && (
              <div
                className={`training-preflight-panel ${
                  preflightPreview.ok ? 'training-preflight-panel--ok' : 'training-preflight-panel--error'
                }`}
              >
                <div className="training-preflight-panel__title-row">
                  <strong>Capability Preflight</strong>
                  <span className={`badge ${preflightPreview.ok ? 'badge-success' : 'badge-error'}`}>
                    {preflightPreview.ok ? 'PASS' : 'BLOCKED'}
                  </span>
                </div>
                {preflightPreview.errors.length > 0 && (
                  <div className="training-preflight-panel__section">
                    <div className="training-preflight-panel__section-title">Blocking Issues</div>
                    <ul className="training-preflight-panel__list">
                      {preflightPreview.errors.map((item, idx) => (
                        <li key={`preflight-error-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {preflightPreview.warnings.length > 0 && (
                  <div className="training-preflight-panel__section">
                    <div className="training-preflight-panel__section-title">Warnings</div>
                    <ul className="training-preflight-panel__list">
                      {preflightPreview.warnings.map((item, idx) => (
                        <li key={`preflight-warning-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {Array.isArray(preflightPreview.hints) && preflightPreview.hints.length > 0 && (
                  <div className="training-preflight-panel__section">
                    <div className="training-preflight-panel__section-title">Fix Hints</div>
                    <ul className="training-preflight-panel__list">
                      {preflightPreview.hints.map((item, idx) => (
                        <li key={`preflight-hint-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="training-preflight-panel__section">
                  <div className="training-preflight-panel__section-title">Capability Summary</div>
                  <pre className="resolved-defaults-panel__json">
                    {JSON.stringify(preflightPreview.capability_summary || {}, null, 2)}
                  </pre>
                </div>
              </div>
            )}
            {preflightPlan && Array.isArray(preflightPlan.suggestions) && preflightPlan.suggestions.length > 0 && (
              <div className="training-plan-panel">
                <div className="training-plan-panel__head">
                  <div>
                    <strong>Preflight Plan Suggestions</strong>
                    <div className="training-plan-panel__hint">
                      Recommended: {preflightPlan.recommended_profile || 'balanced'}
                    </div>
                  </div>
                </div>
                <div className="training-plan-panel__grid">
                  {preflightPlan.suggestions.map((suggestion) => {
                    const isRecommended =
                      String(suggestion.profile || '').trim() === String(preflightPlan.recommended_profile || '').trim();
                    const isPreferred = String(suggestion.profile || '').trim() === preferredPlanProfile;
                    return (
                      <div
                        key={`training-plan-${suggestion.profile}`}
                        className={`training-plan-card ${
                          suggestion.preflight?.ok ? 'training-plan-card--ok' : 'training-plan-card--error'
                        } ${isRecommended ? 'training-plan-card--recommended' : ''}`}
                      >
                        <div className="training-plan-card__head">
                          <strong>{suggestion.title || suggestion.profile}</strong>
                          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                            {isPreferred && <span className="badge badge-info">Preferred</span>}
                            {isRecommended && <span className="badge badge-success">Recommended</span>}
                            <span className={`badge ${suggestion.preflight?.ok ? 'badge-success' : 'badge-error'}`}>
                              {suggestion.preflight?.ok ? 'PASS' : 'BLOCKED'}
                            </span>
                          </div>
                        </div>
                        <div className="training-plan-card__meta">
                          VRAM risk: <strong>{String(suggestion.estimated_vram_risk || 'unknown')}</strong>
                          {Number.isFinite(Number(suggestion.estimated_vram_score))
                            ? ` (score ${Number(suggestion.estimated_vram_score)})`
                            : ''}
                        </div>
                        {suggestion.estimated_vram_note && (
                          <div className="training-plan-card__meta">{suggestion.estimated_vram_note}</div>
                        )}
                        <div className="training-plan-card__desc">
                          {suggestion.description}
                        </div>
                        {Array.isArray(suggestion.changes) && suggestion.changes.length > 0 && (
                          <div className="training-plan-card__changes">
                            {suggestion.changes.slice(0, 8).map((change, idx) => (
                              <div key={`plan-change-${suggestion.profile}-${idx}`}>
                                <code>{change.field}</code>: {String(change.from)} → {String(change.to)}
                                {change.reason ? ` (${change.reason})` : ''}
                              </div>
                            ))}
                            {suggestion.changes.length > 8 && (
                              <div>+{suggestion.changes.length - 8} more changes</div>
                            )}
                          </div>
                        )}
                        {Array.isArray(suggestion.preflight?.errors) && suggestion.preflight.errors.length > 0 && (
                          <div className="training-plan-card__errors">
                            {suggestion.preflight.errors.slice(0, 3).join(' | ')}
                          </div>
                        )}
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => applyPlanSuggestion(suggestion)}
                        >
                          Apply Suggested Config
                        </button>
                      </div>
                    );
                  })}
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
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                  <div className="form-group">
                    <label className="form-label">Runtime</label>
                    <select
                      className="input"
                      value={trainingRuntimeId}
                      onChange={(e) => {
                        setTrainingRuntimeId(e.target.value);
                        setTouchedConfig((prev) => ({ ...prev, training_runtime_id: true }));
                      }}
                    >
                      <option value="auto">
                        Auto
                        {runtimeCatalog?.default_runtime_id ? ` (${runtimeCatalog.default_runtime_id})` : ''}
                      </option>
                      {(runtimeCatalog?.runtimes || []).map((runtime) => (
                        <option key={runtime.runtime_id} value={runtime.runtime_id}>
                          {runtime.label}
                          {runtime.execution_backend ? ` [${runtime.execution_backend}]` : ''}
                        </option>
                      ))}
                    </select>
                    {runtimeCatalogError && (
                      <div className="form-hint" style={{ color: 'var(--color-warning)' }}>
                        {runtimeCatalogError}
                      </div>
                    )}
                  </div>
                  <div className="form-group">
                    <label className="form-label">Task Type</label>
                    <select
                      className="input"
                      value={taskType}
                      onChange={(e) => {
                        setTaskType(e.target.value);
                        setTouchedConfig((prev) => ({ ...prev, task_type: true }));
                      }}
                    >
                      <option value="causal_lm">Causal LM</option>
                      <option value="seq2seq">Seq2Seq</option>
                      <option value="classification">Classification</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Trainer Backend</label>
                    <select
                      className="input"
                      value={trainerBackend}
                      onChange={(e) => {
                        setTrainerBackend(e.target.value);
                        setTouchedConfig((prev) => ({ ...prev, trainer_backend: true }));
                      }}
                    >
                      <option value="auto">Auto (HF Trainer)</option>
                      <option value="hf_trainer">HF Trainer</option>
                      <option value="trl_sft">TRL SFTTrainer</option>
                    </select>
                  </div>
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
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div className="form-group">
                    <label className="form-label">Grad Accum Steps</label>
                    <input
                      className="input"
                      type="number"
                      min={1}
                      value={gradientAccumulationSteps}
                      onChange={(e) => {
                        setGradientAccumulationSteps(Math.max(1, Number(e.target.value) || 1));
                        setTouchedConfig((prev) => ({ ...prev, gradient_accumulation_steps: true }));
                      }}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Max Seq Length</label>
                    <input
                      className="input"
                      type="number"
                      min={128}
                      value={maxSeqLength}
                      onChange={(e) => {
                        setMaxSeqLength(Math.max(128, Number(e.target.value) || 128));
                        setTouchedConfig((prev) => ({ ...prev, max_seq_length: true }));
                      }}
                    />
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div className="form-group">
                    <label className="form-label">Save Steps</label>
                    <input
                      className="input"
                      type="number"
                      min={1}
                      value={saveSteps}
                      onChange={(e) => {
                        setSaveSteps(Math.max(1, Number(e.target.value) || 1));
                        setTouchedConfig((prev) => ({ ...prev, save_steps: true }));
                      }}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Eval Steps</label>
                    <input
                      className="input"
                      type="number"
                      min={1}
                      value={evalSteps}
                      onChange={(e) => {
                        setEvalSteps(Math.max(1, Number(e.target.value) || 1));
                        setTouchedConfig((prev) => ({ ...prev, eval_steps: true }));
                      }}
                    />
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
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={sequencePacking}
                    onChange={(e) => {
                      setSequencePacking(e.target.checked);
                      setTouchedConfig((prev) => ({ ...prev, sequence_packing: true }));
                    }}
                  />
                  <label className="form-label" style={{ margin: 0 }}>Enable Sequence Packing</label>
                </div>
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={flashAttention}
                    onChange={(e) => {
                      setFlashAttention(e.target.checked);
                      setTouchedConfig((prev) => ({ ...prev, flash_attention: true }));
                    }}
                  />
                  <label className="form-label" style={{ margin: 0 }}>Enable Flash Attention</label>
                </div>
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={bf16}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      setBf16(checked);
                      setTouchedConfig((prev) => ({ ...prev, bf16: true }));
                      if (checked && fp16) {
                        setFp16(false);
                        setTouchedConfig((prev) => ({ ...prev, fp16: true }));
                      }
                    }}
                  />
                  <label className="form-label" style={{ margin: 0 }}>Use BF16</label>
                </div>
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={fp16}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      setFp16(checked);
                      setTouchedConfig((prev) => ({ ...prev, fp16: true }));
                      if (checked && bf16) {
                        setBf16(false);
                        setTouchedConfig((prev) => ({ ...prev, bf16: true }));
                      }
                    }}
                  />
                  <label className="form-label" style={{ margin: 0 }}>Use FP16</label>
                </div>
                <div className="form-group" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={autoOomRetry}
                    onChange={(e) => {
                      setAutoOomRetry(e.target.checked);
                      setTouchedConfig((prev) => ({ ...prev, auto_oom_retry: true }));
                    }}
                  />
                  <label className="form-label" style={{ margin: 0 }}>Auto OOM Retry Planner</label>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div className="form-group">
                    <label className="form-label">Max OOM Retries</label>
                    <input
                      className="input"
                      type="number"
                      min={0}
                      max={5}
                      value={maxOomRetries}
                      onChange={(e) => {
                        const v = Math.min(5, Math.max(0, Number(e.target.value) || 0));
                        setMaxOomRetries(v);
                        setTouchedConfig((prev) => ({ ...prev, max_oom_retries: true }));
                      }}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">OOM Seq Shrink</label>
                    <input
                      className="input"
                      value={oomRetrySeqShrink}
                      onChange={(e) => {
                        setOomRetrySeqShrink(e.target.value);
                        setTouchedConfig((prev) => ({ ...prev, oom_retry_seq_shrink: true }));
                      }}
                      placeholder="0.75"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 8, marginTop: 'var(--space-lg)', borderTop: '1px solid var(--border-color)', paddingTop: 'var(--space-lg)' }}>
              <button className="btn btn-primary" onClick={handleCreate}>Create Experiment</button>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setShowCreate(false);
                  setPreflightPreview(null);
                  setPreflightPreviewError('');
                  setPreflightPlan(null);
                  setPreflightPlanError('');
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {trainingError && (
          <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>
            {trainingError}
          </div>
        )}
        {trainingWarnings.length > 0 && (
          <div style={{ marginBottom: 'var(--space-md)', color: 'var(--color-warning)', fontSize: 'var(--font-size-sm)' }}>
            Preflight warnings: {trainingWarnings.join(' | ')}
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
