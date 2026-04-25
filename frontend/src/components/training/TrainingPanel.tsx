import { useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { TerminalConsole } from '../shared/TerminalConsole';
import { ReadinessPanel } from '../shared/ReadinessPanel';
import ExperimentCompare from './ExperimentCompare';
import HardwareRecommenderModal from './HardwareRecommenderModal';
import type { RecommendationResult } from './HardwareRecommenderModal';
import WhyThisPlanPanel from './WhyThisPlanPanel';
import CheckpointsPanel from './CheckpointsPanel';
import PreRunConfirmModal from './PreRunConfirmModal';
import { buildWsUrl } from '../../utils/ws';
import { loadWorkflowStagePrefill } from '../../utils/workflowGraphPrefill';
import './TrainingPanel.css';
import './WaveDPanels.css';

interface TrainingPanelProps {
  projectId: number;
  onNextStep?: () => void;
  title?: string;
  hideStepFooter?: boolean;
  hideCreateControls?: boolean;
  hideExperimentList?: boolean;
  forceCreateVisible?: boolean;
  setupMode?: 'essentials' | 'advanced';
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
  supported_modalities?: string[];
  declares_supported_modalities?: boolean;
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

interface ObservabilityLayerSummary {
  layer?: string;
  event_count?: number;
  avg_grad_norm?: number;
  max_grad_norm?: number;
  avg_update_ratio?: number;
}

interface ObservabilityTokenSummary {
  token?: string;
  count?: number;
}

interface TrainingObservabilitySummary {
  event_count?: number;
  gradient_anomaly_count?: number;
  gradient_anomaly_rate?: number;
  hallucination_signal_count?: number;
  hallucination_signal_rate?: number;
  step_min?: number | null;
  step_max?: number | null;
  top_layers?: ObservabilityLayerSummary[];
  top_attention_tokens?: ObservabilityTokenSummary[];
  last_event_at?: string | null;
  path?: string;
}

interface TrainingObservabilityResponse {
  summary?: TrainingObservabilitySummary;
  recent?: {
    count?: number;
    events?: Array<Record<string, unknown>>;
  };
}

interface VibeCheckOutput {
  prompt_id?: string;
  prompt?: string;
  reply?: string;
  provider?: string;
  model_name?: string;
  latency_ms?: number | null;
  error?: string | null;
}

interface VibeCheckSnapshot {
  step?: number;
  epoch?: number | null;
  progress?: number | null;
  train_loss?: number | null;
  eval_loss?: number | null;
  created_at?: string | null;
  outputs?: VibeCheckOutput[];
}

interface VibeCheckTimelineResponse {
  config?: {
    enabled?: boolean;
    interval_steps?: number;
    prompts?: string[];
    provider?: string;
    model_name?: string;
  };
  snapshot_count?: number;
  snapshots?: VibeCheckSnapshot[];
  latest_snapshot?: VibeCheckSnapshot | null;
}

interface TrainingRecipe {
  recipe_id: string;
  display_name: string;
  description?: string;
  category?: string;
  tags?: string[];
  required_fields?: string[];
}

interface TrainingRecipeCatalogResponse {
  project_id: number;
  recipe_count?: number;
  recipes?: TrainingRecipe[];
}

interface TrainingRecipeResolveResponse extends TrainingEffectiveConfigResponse {
  project_id: number;
  recipe?: TrainingRecipe & { config_patch?: Record<string, unknown> };
  recipe_missing_required_fields?: string[];
  recipe_config?: Record<string, unknown>;
  preflight?: TrainingPreflightReport;
}

interface ModelWizardRecommendation {
  model_id: string;
  family?: string;
  params_b?: number;
  estimated_min_vram_gb?: number;
  estimated_ideal_vram_gb?: number;
  introspection_estimated_min_vram_gb?: number | null;
  introspection_estimated_ideal_vram_gb?: number | null;
  architecture?: string;
  context_length?: number | null;
  license?: string | null;
  metadata_source?: string;
  supported_languages?: string[];
  strengths?: string[];
  caveats?: string[];
  match_reasons?: string[];
  match_score?: number;
  adaptive_bias?: number;
  suggested_defaults?: {
    task_type?: string;
    chat_template?: string;
    use_lora?: boolean;
    batch_size?: number;
    max_seq_length?: number;
  };
}

interface ModelWizardResponse {
  project_id: number;
  catalog_strategy?: string;
  request?: {
    target_device?: string;
    primary_language?: string;
    available_vram_gb?: number | null;
    task_profile?: string | null;
    top_k?: number;
  };
  recommendation_count?: number;
  recommendations?: ModelWizardRecommendation[];
  warnings?: string[];
  adaptive_ranking?: {
    enabled?: boolean;
    context_label?: string;
    global_apply_events?: number;
    context_apply_events?: number;
    boosted_model_count?: number;
  };
}

interface ModelBenchmarkRow {
  rank?: number;
  model_id?: string;
  params_b?: number;
  estimated_min_vram_gb?: number;
  estimated_quality_score?: number;
  estimated_accuracy_percent?: number;
  estimated_latency_ms?: number;
  estimated_throughput_tps?: number;
  fits_available_vram?: boolean | null;
  benchmark_mode?: string;
}

interface ModelBenchmarkTradeoffSummary {
  best_quality_model_id?: string;
  best_speed_model_id?: string;
  best_balance_model_id?: string;
}

interface ModelBenchmarkSweepResponse {
  project_id: number;
  run_id?: string;
  benchmark_mode?: string;
  model_count?: number;
  sampled_row_count?: number;
  sampled_avg_tokens?: number;
  sampled_total_tokens?: number;
  benchmark_window_minutes?: number;
  matrix?: ModelBenchmarkRow[];
  tradeoff_summary?: ModelBenchmarkTradeoffSummary;
  warnings?: string[];
}

interface ModelBenchmarkHistoryRun {
  run_id?: string;
  timestamp?: string;
  benchmark_mode?: string;
  sampled_row_count?: number;
  sampled_avg_tokens?: number;
  tradeoff_summary?: ModelBenchmarkTradeoffSummary;
  matrix?: ModelBenchmarkRow[];
}

interface ModelBenchmarkHistoryResponse {
  count?: number;
  runs?: ModelBenchmarkHistoryRun[];
}

interface ModelIntrospectionMemoryProfile {
  estimated_min_vram_gb?: number;
  estimated_ideal_vram_gb?: number;
}

interface ModelIntrospectionSummary {
  model_id?: string;
  resolved?: boolean;
  source?: string;
  model_type?: string | null;
  architecture?: string;
  architecture_hint?: string | null;
  context_length?: number | null;
  license?: string | null;
  params_estimate_b?: number | null;
  memory_profile?: ModelIntrospectionMemoryProfile | null;
  warnings?: string[];
}

interface ModelIntrospectionResponse {
  project_id: number;
  introspection?: ModelIntrospectionSummary;
}

interface CloudBurstProvider {
  provider_id: string;
  display_name?: string;
  description?: string;
  supports_spot?: boolean;
  supports_live_execution?: boolean;
  supports_managed_cancel?: boolean;
  supports_live_logs?: boolean;
  regions?: string[];
}

interface CloudBurstGpuSku {
  gpu_sku: string;
  display_name?: string;
  vram_gb?: number;
  hourly_usd?: Record<string, number>;
}

interface CloudBurstCatalogResponse {
  project_id: number;
  providers?: CloudBurstProvider[];
  gpu_skus?: CloudBurstGpuSku[];
  provider_count?: number;
  gpu_sku_count?: number;
}

interface CloudBurstQuoteResponse {
  project_id: number;
  provider_id?: string;
  provider_name?: string;
  gpu_sku?: string;
  duration_hours?: number;
  spot_effective?: boolean;
  effective_hourly_usd?: number;
  cost_breakdown_usd?: {
    compute?: number;
    storage?: number;
    egress?: number;
    total?: number;
  };
  warnings?: string[];
}

interface CloudBurstLaunchPlanResponse {
  project_id: number;
  launch_id?: string;
  provider_id?: string;
  gpu_sku?: string;
  quote?: CloudBurstQuoteResponse;
  credentials?: {
    ready?: boolean;
    missing_keys?: string[];
    present_keys?: string[];
  };
  request_template?: Record<string, unknown>;
  record_path?: string;
}

interface CloudBurstRunArtifacts {
  sync_enabled?: boolean;
  source_dir?: string | null;
  target_dir?: string | null;
  policy?: string;
  include_globs?: string[];
  exclude_globs?: string[];
  last_sync_at?: string | null;
  last_sync_summary?: {
    status?: string;
    copied_count?: number;
    unchanged_count?: number;
    would_copy_count?: number;
    deleted_count?: number;
    file_count?: number;
    candidate_count?: number;
    remaining_count?: number;
    limited?: boolean;
    cursor?: string | null;
    next_cursor?: string | null;
    manifest_path?: string;
    manifest_updated?: boolean;
    total_bytes?: number;
    reason?: string;
    sampled_files?: string[];
    sampled_unchanged_files?: string[];
    errors?: string[];
  } | null;
}

interface CloudBurstMetricPoint {
  step?: number;
  epoch?: number;
  train_loss?: number | null;
  eval_loss?: number | null;
  learning_rate?: number | null;
  throughput_tps?: number | null;
  at?: string | null;
}

interface CloudBurstLogBridgeSummary {
  last_ingested_at?: string;
  last_ingested_count?: number;
  seen_hash_count?: number;
}

interface CloudBurstRunStatusResponse {
  project_id: number;
  run_id: string;
  launch_id?: string;
  idempotency_key?: string | null;
  idempotent_replay?: boolean;
  provider_id?: string;
  provider_job_id?: string | null;
  provider_status_raw?: string;
  provider_uptime_seconds?: number | null;
  provider_last_status_at?: string | null;
  provider_poll_count?: number;
  gpu_sku?: string;
  experiment_id?: number | null;
  status?: string;
  status_reason?: string;
  execution_mode_requested?: string;
  execution_mode_effective?: string;
  execution_mode_fallback_reason?: string | null;
  can_cancel?: boolean;
  cancel_requested?: boolean;
  created_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  logs_tail?: string[];
  logs_tail_count?: number;
  metrics_tail?: CloudBurstMetricPoint[];
  metrics_tail_count?: number;
  log_bridge?: CloudBurstLogBridgeSummary | null;
  artifacts?: CloudBurstRunArtifacts | null;
  current_run_cost?: number;
  status_timeline?: Array<{
    status: string;
    reason: string;
    at: string;
  }>;
  record_path?: string;
}

interface CloudBurstRunListResponse {
  project_id: number;
  count?: number;
  limit?: number;
  runs?: CloudBurstRunStatusResponse[];
}

const METRIC_PREFIX = 'SLM_METRIC ';
const PLAN_PROFILE_STORAGE_PREFIX = 'slm-training-plan-profile';
const MODEL_WIZARD_TASK_PROFILES = [
  'auto',
  'instruction_sft',
  'chat_sft',
  'qa',
  'rag_qa',
  'tool_calling',
  'structured_extraction',
  'summarization',
  'seq2seq',
  'classification',
  'preference',
];

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

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const out: string[] = [];
  value.forEach((item) => {
    const token = String(item || '').trim();
    if (token && !out.includes(token)) {
      out.push(token);
    }
  });
  return out;
}

function parseBool(value: unknown): boolean | null {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    return value !== 0;
  }
  if (typeof value === 'string') {
    const token = value.trim().toLowerCase();
    if (token === 'true' || token === '1' || token === 'yes' || token === 'on') {
      return true;
    }
    if (token === 'false' || token === '0' || token === 'no' || token === 'off' || token === '') {
      return false;
    }
  }
  return null;
}

function parseNonNegativeInt(value: unknown): number {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return 0;
  }
  return Math.max(0, Math.trunc(num));
}

type ConfigFieldKey =
  | 'training_mode'
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
  | 'gradient_checkpointing'
  | 'multimodal_require_media'
  | 'alignment_auto_filter'
  | 'alignment_quality_threshold'
  | 'alignment_beta'
  | 'alignment_max_prompt_length'
  | 'alignment_max_length'
  | 'alignment_min_keep_ratio'
  | 'alignment_dataset_path'
  | 'alignment_include_playground_feedback'
  | 'alignment_playground_max_pairs'
  | 'observability_enabled'
  | 'observability_log_steps'
  | 'observability_max_layers'
  | 'observability_probe_attention'
  | 'observability_probe_top_k';

type TrainingWorkspaceView = 'overview' | 'setup' | 'runs';
type TrainingSetupTab = 'basics' | 'config' | 'power' | 'review';
type ModelSelectionApplySource = 'recommendation' | 'benchmark' | 'consensus';

export default function TrainingPanel({
  projectId,
  onNextStep,
  title = 'Training Experiments',
  hideStepFooter = false,
  hideCreateControls = false,
  hideExperimentList = false,
  forceCreateVisible = false,
  setupMode = 'advanced',
}: TrainingPanelProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [showCreate, setShowCreate] = useState(Boolean(forceCreateVisible && !hideCreateControls));
  const [activeExperiment, setActiveExperiment] = useState<Experiment | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [selectedForCompare, setSelectedForCompare] = useState<number[]>([]);
  const [showCompare, setShowCompare] = useState(false);
  const [taskState, setTaskState] = useState<string>('');
  const [trainingError, setTrainingError] = useState<string>('');

  const [name, setName] = useState('');
  const [baseModel, setBaseModel] = useState('microsoft/phi-2');
  const [trainingMode, setTrainingMode] = useState('sft');
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
  const [multimodalRequireMedia, setMultimodalRequireMedia] = useState(false);
  const [alignmentAutoFilter, setAlignmentAutoFilter] = useState(false);
  const [alignmentQualityThreshold, setAlignmentQualityThreshold] = useState('3.0');
  const [alignmentBeta, setAlignmentBeta] = useState('0.1');
  const [alignmentMaxPromptLength, setAlignmentMaxPromptLength] = useState('1024');
  const [alignmentMaxLength, setAlignmentMaxLength] = useState('2048');
  const [alignmentMinKeepRatio, setAlignmentMinKeepRatio] = useState('0.4');
  const [alignmentDatasetPath, setAlignmentDatasetPath] = useState('');
  const [alignmentIncludePlaygroundFeedback, setAlignmentIncludePlaygroundFeedback] = useState(true);
  const [alignmentPlaygroundMaxPairs, setAlignmentPlaygroundMaxPairs] = useState('5000');
  const [observabilityEnabled, setObservabilityEnabled] = useState(true);
  const [observabilityLogSteps, setObservabilityLogSteps] = useState(50);
  const [observabilityMaxLayers, setObservabilityMaxLayers] = useState(12);
  const [observabilityProbeAttention, setObservabilityProbeAttention] = useState(true);
  const [observabilityProbeTopK, setObservabilityProbeTopK] = useState(6);
  const [useProfileDefaults, setUseProfileDefaults] = useState(true);
  const [touchedConfig, setTouchedConfig] = useState<Record<ConfigFieldKey, boolean>>({
    training_mode: false,
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
    multimodal_require_media: false,
    alignment_auto_filter: false,
    alignment_quality_threshold: false,
    alignment_beta: false,
    alignment_max_prompt_length: false,
    alignment_max_length: false,
    alignment_min_keep_ratio: false,
    alignment_dataset_path: false,
    alignment_include_playground_feedback: false,
    alignment_playground_max_pairs: false,
    observability_enabled: false,
    observability_log_steps: false,
    observability_max_layers: false,
    observability_probe_attention: false,
    observability_probe_top_k: false,
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
  // P20 — pre-run confirm modal state. Holds the experiment whose Start
  // click is awaiting cost-estimate confirmation; cleared on confirm/cancel.
  const [pendingStartExperiment, setPendingStartExperiment] = useState<Experiment | null>(null);
  const [trainingWarnings, setTrainingWarnings] = useState<string[]>([]);
  const [runtimeCatalog, setRuntimeCatalog] = useState<TrainingRuntimeCatalogResponse | null>(null);
  const [runtimeCatalogError, setRuntimeCatalogError] = useState('');
  const [trainingRecipes, setTrainingRecipes] = useState<TrainingRecipe[]>([]);
  const [selectedRecipeId, setSelectedRecipeId] = useState('');
  const [recipeResolveLoading, setRecipeResolveLoading] = useState(false);
  const [recipeResolveError, setRecipeResolveError] = useState('');
  const [workspaceView, setWorkspaceView] = useState<TrainingWorkspaceView>('overview');
  const [setupTab, setSetupTab] = useState<TrainingSetupTab>('basics');
  const [wizardTargetDevice, setWizardTargetDevice] = useState('laptop');
  const [wizardPrimaryLanguage, setWizardPrimaryLanguage] = useState('english');
  const [wizardVramGb, setWizardVramGb] = useState('8');
  const [wizardTaskProfile, setWizardTaskProfile] = useState('auto');
  const [wizardLoading, setWizardLoading] = useState(false);
  const [wizardError, setWizardError] = useState('');
  const [wizardResult, setWizardResult] = useState<ModelWizardResponse | null>(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState('');
  const [benchmarkResult, setBenchmarkResult] = useState<ModelBenchmarkSweepResponse | null>(null);
  const [benchmarkHistory, setBenchmarkHistory] = useState<ModelBenchmarkHistoryRun[]>([]);
  const [reviewSelectionActionNote, setReviewSelectionActionNote] = useState('');
  const [baseModelIntrospection, setBaseModelIntrospection] = useState<ModelIntrospectionSummary | null>(null);
  const [baseModelIntrospectionLoading, setBaseModelIntrospectionLoading] = useState(false);
  const [baseModelIntrospectionError, setBaseModelIntrospectionError] = useState('');
  const [wizardAutoRan, setWizardAutoRan] = useState(false);
  const [cloudBurstCatalog, setCloudBurstCatalog] = useState<CloudBurstCatalogResponse | null>(null);
  const [cloudBurstProviderId, setCloudBurstProviderId] = useState('');
  const [cloudBurstGpuSku, setCloudBurstGpuSku] = useState('');
  const [cloudBurstDurationHours, setCloudBurstDurationHours] = useState('2');
  const [cloudBurstStorageGb, setCloudBurstStorageGb] = useState('50');
  const [cloudBurstEgressGb, setCloudBurstEgressGb] = useState('0');
  const [cloudBurstSpot, setCloudBurstSpot] = useState(true);
  const [cloudBurstRegion, setCloudBurstRegion] = useState('');
  const [cloudBurstImage, setCloudBurstImage] = useState('');
  const [cloudBurstStartupScript, setCloudBurstStartupScript] = useState('');
  const [cloudBurstExperimentId, setCloudBurstExperimentId] = useState('');
  const [cloudBurstExecutionMode, setCloudBurstExecutionMode] = useState('auto');
  const [cloudBurstAllowFallbackToSimulation, setCloudBurstAllowFallbackToSimulation] = useState(true);
  const [cloudBurstIdempotencyKey, setCloudBurstIdempotencyKey] = useState('');
  const [cloudBurstSyncCursor, setCloudBurstSyncCursor] = useState('');
  const [cloudBurstLoadingCatalog, setCloudBurstLoadingCatalog] = useState(false);
  const [cloudBurstLoadingQuote, setCloudBurstLoadingQuote] = useState(false);
  const [cloudBurstLoadingPlan, setCloudBurstLoadingPlan] = useState(false);
  const [cloudBurstError, setCloudBurstError] = useState('');
  const [cloudBurstInfo, setCloudBurstInfo] = useState('');
  const [cloudBurstQuote, setCloudBurstQuote] = useState<CloudBurstQuoteResponse | null>(null);
  const [cloudBurstPlan, setCloudBurstPlan] = useState<CloudBurstLaunchPlanResponse | null>(null);
  const [cloudBurstRuns, setCloudBurstRuns] = useState<CloudBurstRunStatusResponse[]>([]);
  const [cloudBurstActiveRunId, setCloudBurstActiveRunId] = useState('');
  const [cloudBurstActiveRun, setCloudBurstActiveRun] = useState<CloudBurstRunStatusResponse | null>(null);
  const [cloudBurstLoadingRuns, setCloudBurstLoadingRuns] = useState(false);
  const [cloudBurstSubmittingJob, setCloudBurstSubmittingJob] = useState(false);
  const [cloudBurstCancellingJob, setCloudBurstCancellingJob] = useState(false);
  const [cloudBurstSyncingArtifacts, setCloudBurstSyncingArtifacts] = useState(false);
  const [cloudBurstPrefillStage, setCloudBurstPrefillStage] = useState('');
  const [observabilitySummary, setObservabilitySummary] = useState<TrainingObservabilitySummary | null>(null);
  const [observabilityRecentCount, setObservabilityRecentCount] = useState(0);
  const [observabilityError, setObservabilityError] = useState('');
  const [observabilityLoading, setObservabilityLoading] = useState(false);
  const [vibeTimeline, setVibeTimeline] = useState<VibeCheckSnapshot[]>([]);
  const [vibeSelectedIndex, setVibeSelectedIndex] = useState(0);
  const [vibeConfig, setVibeConfig] = useState<VibeCheckTimelineResponse['config'] | null>(null);
  const [vibeError, setVibeError] = useState('');
  const [vibeLoading, setVibeLoading] = useState(false);
  const [showHardwareModal, setShowHardwareModal] = useState(false);

  const statusColor = (status: string) =>
    status === 'completed'
      ? 'badge-success'
      : status === 'running'
        ? 'badge-info'
        : status === 'failed'
          ? 'badge-error'
          : 'badge-warning';

  const activeExperimentKey = activeExperiment ? `${activeExperiment.id}:${activeExperiment.status}` : '';
  const createFormVisible = !hideCreateControls && (forceCreateVisible || showCreate);
  const canConfigureExperiments = !hideCreateControls;
  const canViewRuns = !hideExperimentList;
  const showWorkspaceTabs = canConfigureExperiments && canViewRuns;
  const isAlignmentMode = trainingMode === 'dpo' || trainingMode === 'orpo';
  const isSetupAdvancedMode = setupMode === 'advanced';
  const setupTabOrder: TrainingSetupTab[] = isSetupAdvancedMode
    ? ['basics', 'config', 'power', 'review']
    : ['basics', 'config', 'review'];
  const setupTabIndex = setupTabOrder.indexOf(setupTab);
  const showSetupBasics = setupTab === 'basics';
  const showSetupConfig = setupTab === 'config';
  const showSetupPower = isSetupAdvancedMode && setupTab === 'power';
  const showSetupReview = setupTab === 'review';
  const canSetupGoBack = setupTabIndex > 0;
  const canSetupGoNext = setupTabIndex >= 0 && setupTabIndex < setupTabOrder.length - 1;

  const experimentStats = useMemo(() => {
    const running = experiments.filter((item) => item.status === 'running').length;
    const completed = experiments.filter((item) => item.status === 'completed').length;
    const failed = experiments.filter((item) => item.status === 'failed').length;
    const pending = experiments.filter((item) => item.status === 'pending').length;
    return {
      total: experiments.length,
      running,
      completed,
      failed,
      pending,
    };
  }, [experiments]);

  const recommendedAction = useMemo(() => {
    if (canConfigureExperiments && experimentStats.total === 0) {
      return {
        title: 'Create your first experiment',
        detail: 'Open Setup and use recipe + preflight before launching.',
      };
    }
    if (canViewRuns && experimentStats.running > 0) {
      return {
        title: 'Monitor active runs',
        detail: `${experimentStats.running} experiment(s) are running. Open Runs and launch dashboard.`,
      };
    }
    if (canConfigureExperiments && canViewRuns) {
      return {
        title: 'Tune and iterate',
        detail: 'Adjust config in Setup, then create another run and compare results.',
      };
    }
    return {
      title: 'Review experiment status',
      detail: 'Open the available section and continue with the next training action.',
    };
  }, [canConfigureExperiments, canViewRuns, experimentStats]);

  const cloudProviders = Array.isArray(cloudBurstCatalog?.providers) ? cloudBurstCatalog.providers : [];
  const cloudGpuSkus = Array.isArray(cloudBurstCatalog?.gpu_skus) ? cloudBurstCatalog.gpu_skus : [];
  const selectedCloudProvider = cloudProviders.find((item) => item.provider_id === cloudBurstProviderId) || null;
  const runtimeOptions = Array.isArray(runtimeCatalog?.runtimes) ? runtimeCatalog.runtimes : [];
  const selectedRuntimeCatalogId =
    trainingRuntimeId === 'auto'
      ? String(runtimeCatalog?.default_runtime_id || '').trim().toLowerCase()
      : String(trainingRuntimeId || '').trim().toLowerCase();
  const selectedRuntimeSpec =
    runtimeOptions.find(
      (item) => String(item.runtime_id || '').trim().toLowerCase() === selectedRuntimeCatalogId,
    ) || null;
  const selectedRuntimeModalities = asStringList(selectedRuntimeSpec?.supported_modalities);
  const selectedRuntimeModalitiesDeclared = parseBool(
    selectedRuntimeSpec?.declares_supported_modalities,
  );
  const cloudBurstActiveStatus = String(cloudBurstActiveRun?.status || '').trim().toLowerCase();
  const cloudBurstActiveIsTerminal = ['completed', 'failed', 'cancelled'].includes(cloudBurstActiveStatus);
  const cloudBurstMetrics = useMemo(() => {
    const rows = Array.isArray(cloudBurstActiveRun?.metrics_tail)
      ? cloudBurstActiveRun.metrics_tail
      : [];
    const parsed = rows.map((row) => {
      const item = asRecord(row);
      const step = Number(item.step);
      const epoch = Number(item.epoch);
      const trainLoss = Number(item.train_loss);
      const evalLoss = Number(item.eval_loss);
      const learningRate = Number(item.learning_rate);
      const throughputTps = Number(item.throughput_tps);
      return {
        step: Number.isFinite(step) ? step : undefined,
        epoch: Number.isFinite(epoch) ? epoch : undefined,
        train_loss: Number.isFinite(trainLoss) ? trainLoss : null,
        eval_loss: Number.isFinite(evalLoss) ? evalLoss : null,
        learning_rate: Number.isFinite(learningRate) ? learningRate : null,
        throughput_tps: Number.isFinite(throughputTps) ? throughputTps : null,
        at: typeof item.at === 'string' ? item.at : null,
      } as CloudBurstMetricPoint;
    });
    return parsed;
  }, [cloudBurstActiveRun?.metrics_tail]);
  const cloudBurstLatestMetric = cloudBurstMetrics.length > 0
    ? cloudBurstMetrics[cloudBurstMetrics.length - 1]
    : null;
  const cloudBurstMetricSeries = useMemo(() => {
    const buildSeries = (field: 'train_loss' | 'eval_loss') => {
      const points = cloudBurstMetrics
        .map((item, idx) => ({ idx, value: Number(item[field]) }))
        .filter((item) => Number.isFinite(item.value));
      if (points.length < 2) {
        return '';
      }
      const values = points.map((item) => item.value);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min;
      return points
        .map((item, localIndex) => {
          const x = points.length <= 1 ? 0 : (localIndex / (points.length - 1)) * 100;
          const normalized = span <= 0 ? 0.5 : (item.value - min) / span;
          const y = 100 - (normalized * 100);
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(' ');
    };
    return {
      train: buildSeries('train_loss'),
      eval: buildSeries('eval_loss'),
    };
  }, [cloudBurstMetrics]);

  const preflightContractDetails = useMemo(() => {
    const capabilitySummary = asRecord(preflightPreview?.capability_summary);
    const capabilityContract = asRecord(capabilitySummary.capability_contract);
    const dataset = asRecord(capabilitySummary.dataset);
    const mediaContract = asRecord(dataset.media_contract);
    const adapterContext = asRecord(dataset.adapter_context);
    const runtimeSummary = asRecord(capabilitySummary.runtime);
    const modelSummary = asRecord(capabilitySummary.model);
    const modelModalityContract = asRecord(capabilitySummary.model_modality_contract);
    const modelCompatibilityGate = asRecord(modelSummary.compatibility_gate);
    const modelIntrospection = asRecord(modelSummary.introspection);

    const runtimeSupportedModalities = asStringList(
      capabilityContract.runtime_supported_modalities ?? runtimeSummary.supported_modalities,
    );
    const adapterDeclaredProfiles = asStringList(capabilityContract.adapter_declared_task_profiles);
    const adapterPreferredTasks = asStringList(capabilityContract.adapter_preferred_training_tasks);
    const modelGateErrors = asStringList(modelCompatibilityGate.errors);
    const modelGateHints = asStringList(modelCompatibilityGate.hints);
    const modelSupportedArchitectures = asStringList(modelCompatibilityGate.supported_architectures);
    const modelModalityErrors = asStringList(modelModalityContract.errors);
    const modelModalityWarnings = asStringList(modelModalityContract.warnings);
    const modelModalityHints = asStringList(modelModalityContract.hints);
    const modelModalitySupportedModalities = asStringList(modelModalityContract.supported_modalities);
    const modelModalityOk = parseBool(modelModalityContract.ok);
    const mediaContractErrors = asStringList(mediaContract.errors);
    const mediaContractWarnings = asStringList(mediaContract.warnings);
    const mediaContractHints = asStringList(mediaContract.hints);
    const mediaContractOk = parseBool(mediaContract.ok);

    let modelModalityStatus: 'pass' | 'blocked' | 'warning' | 'unknown' = 'unknown';
    if (modelModalityOk === false || modelModalityErrors.length > 0) {
      modelModalityStatus = 'blocked';
    } else if (modelModalityWarnings.length > 0) {
      modelModalityStatus = 'warning';
    } else if (modelModalityOk === true) {
      modelModalityStatus = 'pass';
    }

    let mediaContractStatus: 'pass' | 'blocked' | 'warning' | 'unknown' = 'unknown';
    if (mediaContractOk === false || mediaContractErrors.length > 0) {
      mediaContractStatus = 'blocked';
    } else if (mediaContractWarnings.length > 0) {
      mediaContractStatus = 'warning';
    } else if (mediaContractOk === true) {
      mediaContractStatus = 'pass';
    }

    return {
      taskType: String(capabilityContract.task_type || capabilitySummary.task_type || 'unknown'),
      trainingMode: String(capabilityContract.training_mode || capabilitySummary.training_mode || 'unknown'),
      trainerBackend: String(
        capabilityContract.trainer_backend_requested || capabilitySummary.trainer_backend_requested || 'unknown',
      ),
      runtimeId: String(
        capabilityContract.runtime_id ||
        runtimeSummary.resolved_runtime_id ||
        runtimeSummary.requested_runtime_id ||
        'unknown',
      ),
      runtimeBackend: String(capabilityContract.runtime_backend || capabilitySummary.runtime_backend || 'unknown'),
      runtimeKnown: parseBool(capabilityContract.runtime_known),
      runtimeSupportedModalities,
      runtimeModalitiesDeclared: parseBool(
        capabilityContract.runtime_modalities_declared ?? runtimeSummary.modalities_declared,
      ),
      adapterId: String(capabilityContract.adapter_id || adapterContext.adapter_id || 'unknown'),
      adapterSource: String(adapterContext.adapter_source || 'unknown'),
      adapterTaskProfile: String(
        capabilityContract.adapter_task_profile || adapterContext.task_profile || 'none',
      ),
      adapterTaskProfileSource: String(adapterContext.task_profile_source || 'unknown'),
      adapterModality: String(
        capabilityContract.adapter_modality || adapterContext.adapter_modality || 'unknown',
      ),
      adapterDeclaredProfiles,
      adapterPreferredTasks,
      modelId: String(modelSummary.id || 'unknown'),
      modelFamily: String(modelSummary.family || 'unknown'),
      modelArchitecture: String(modelSummary.architecture || 'unknown'),
      modelGateOk: parseBool(modelCompatibilityGate.ok),
      modelGateErrors,
      modelGateHints,
      modelSupportedArchitectures,
      modelIntrospectionSource: String(modelIntrospection.source || 'none'),
      modelModalityArchitecture: String(
        modelModalityContract.architecture || modelSummary.architecture || 'unknown',
      ),
      modelModalityAdapterModality: String(
        modelModalityContract.adapter_modality ||
        capabilityContract.adapter_modality ||
        adapterContext.adapter_modality ||
        'unknown',
      ),
      modelModalitySupportedModalities:
        modelModalitySupportedModalities.length > 0 ? modelModalitySupportedModalities : ['text'],
      modelModalityOk,
      modelModalityErrors,
      modelModalityWarnings,
      modelModalityHints,
      modelModalityStatus,
      mediaContractExpectedModality: String(
        mediaContract.expected_modality || adapterContext.adapter_modality || 'text',
      ),
      mediaContractSampledRows: parseNonNegativeInt(mediaContract.sampled_rows),
      mediaContractMediaRows: parseNonNegativeInt(mediaContract.media_rows),
      mediaContractImageRows: parseNonNegativeInt(mediaContract.image_rows),
      mediaContractAudioRows: parseNonNegativeInt(mediaContract.audio_rows),
      mediaContractMixedRows: parseNonNegativeInt(mediaContract.multimodal_rows),
      mediaContractMissingLocalImages: parseNonNegativeInt(mediaContract.missing_local_images),
      mediaContractMissingLocalAudios: parseNonNegativeInt(mediaContract.missing_local_audios),
      mediaContractRemoteImageRefs: parseNonNegativeInt(mediaContract.remote_image_refs),
      mediaContractRemoteAudioRefs: parseNonNegativeInt(mediaContract.remote_audio_refs),
      mediaContractRequireMedia: parseBool(mediaContract.require_media),
      mediaContractErrors,
      mediaContractWarnings,
      mediaContractHints,
      mediaContractStatus,
      rawCapabilitySummary: capabilitySummary,
    };
  }, [preflightPreview]);

  const essentialsModelGateSummary = useMemo(() => {
    const gateOk = preflightContractDetails.modelGateOk;
    const statusLabel = gateOk === true ? 'Pass' : gateOk === false ? 'Blocked' : 'Unknown';
    const statusClass = gateOk === true ? 'ok' : gateOk === false ? 'blocked' : 'unknown';
    const topIssue =
      preflightContractDetails.modelGateErrors[0] ||
      preflightContractDetails.modelGateHints[0] ||
      '';
    return {
      statusLabel,
      statusClass,
      modelId: preflightContractDetails.modelId,
      architecture: preflightContractDetails.modelArchitecture,
      source: preflightContractDetails.modelIntrospectionSource,
      topIssue,
      supportedArchitectures: preflightContractDetails.modelSupportedArchitectures.join(', '),
    };
  }, [preflightContractDetails]);

  const modelSelectionSummary = useMemo(() => {
    const recommendationRows = Array.isArray(wizardResult?.recommendations)
      ? wizardResult.recommendations
      : [];
    const recommendationWinner = recommendationRows[0] || null;
    const recommendationWinnerId = String(recommendationWinner?.model_id || '').trim();
    const recommendationWinnerScore = Number.isFinite(Number(recommendationWinner?.match_score))
      ? Number(recommendationWinner?.match_score)
      : null;
    const recommendationWinnerAdaptiveBias = Number.isFinite(Number(recommendationWinner?.adaptive_bias))
      ? Number(recommendationWinner?.adaptive_bias)
      : null;
    const recommendationWinnerReason = Array.isArray(recommendationWinner?.match_reasons)
      ? String(recommendationWinner?.match_reasons?.[0] || '').trim()
      : '';

    const currentBenchmarkRows = Array.isArray(benchmarkResult?.matrix)
      ? benchmarkResult.matrix
      : [];
    let benchmarkWinnerId = String(
      benchmarkResult?.tradeoff_summary?.best_balance_model_id
      || currentBenchmarkRows[0]?.model_id
      || '',
    ).trim();
    let benchmarkWinnerSource = benchmarkResult?.run_id
      ? `latest run (${benchmarkResult.run_id})`
      : '';
    let benchmarkWinnerMode = String(benchmarkResult?.benchmark_mode || '').trim();
    let benchmarkWinnerRow = currentBenchmarkRows
      .find((row) => String(row?.model_id || '').trim() === benchmarkWinnerId)
      || currentBenchmarkRows[0]
      || null;

    if (!benchmarkWinnerId) {
      const historyRun = benchmarkHistory[0] || null;
      const historyRows = Array.isArray(historyRun?.matrix) ? historyRun.matrix : [];
      benchmarkWinnerId = String(
        historyRun?.tradeoff_summary?.best_balance_model_id
        || historyRows[0]?.model_id
        || '',
      ).trim();
      benchmarkWinnerSource = historyRun?.run_id
        ? `history (${historyRun.run_id})`
        : historyRun
          ? 'history'
          : '';
      benchmarkWinnerMode = String(historyRun?.benchmark_mode || '').trim();
      benchmarkWinnerRow = historyRows
        .find((row) => String(row?.model_id || '').trim() === benchmarkWinnerId)
        || historyRows[0]
        || null;
    }

    const sameWinner = Boolean(
      recommendationWinnerId
      && benchmarkWinnerId
      && recommendationWinnerId === benchmarkWinnerId,
    );
    const activeModelId = String(baseModel || '').trim();
    const activeModelLabel = !activeModelId
      ? 'none'
      : activeModelId === recommendationWinnerId && activeModelId === benchmarkWinnerId
        ? 'matches recommendation + benchmark'
        : activeModelId === recommendationWinnerId
          ? 'matches recommendation'
          : activeModelId === benchmarkWinnerId
            ? 'matches benchmark'
            : 'custom/manual';

    return {
      hasAny: Boolean(recommendationWinnerId || benchmarkWinnerId),
      recommendationWinnerId,
      recommendationWinnerScore,
      recommendationWinnerAdaptiveBias,
      recommendationWinnerReason,
      benchmarkWinnerId,
      benchmarkWinnerSource,
      benchmarkWinnerMode,
      benchmarkWinnerAccuracy: Number.isFinite(Number(benchmarkWinnerRow?.estimated_accuracy_percent))
        ? Number(benchmarkWinnerRow?.estimated_accuracy_percent)
        : null,
      benchmarkWinnerLatencyMs: Number.isFinite(Number(benchmarkWinnerRow?.estimated_latency_ms))
        ? Number(benchmarkWinnerRow?.estimated_latency_ms)
        : null,
      benchmarkWinnerThroughputTps: Number.isFinite(Number(benchmarkWinnerRow?.estimated_throughput_tps))
        ? Number(benchmarkWinnerRow?.estimated_throughput_tps)
        : null,
      sameWinner,
      winnerAlignmentLabel: sameWinner
        ? 'Winners align: recommendation and benchmark agree.'
        : recommendationWinnerId && benchmarkWinnerId
          ? 'Recommendation and benchmark differ; verify trade-off before launch.'
          : 'Run recommendation + benchmark to compare winners.',
      activeModelId,
      activeModelLabel,
    };
  }, [wizardResult, benchmarkResult, benchmarkHistory, baseModel]);

  const buildTrainingConfigPayload = (): Record<string, unknown> => {
    const learningRate = Number.parseFloat(lr);
    const retryShrink = Number.parseFloat(oomRetrySeqShrink);
    const alignmentThreshold = Number.parseFloat(alignmentQualityThreshold);
    const alignmentBetaValue = Number.parseFloat(alignmentBeta);
    const alignmentPromptLengthValue = Number.parseInt(alignmentMaxPromptLength, 10);
    const alignmentMaxLengthValue = Number.parseInt(alignmentMaxLength, 10);
    const alignmentKeepRatio = Number.parseFloat(alignmentMinKeepRatio);
    const alignmentFeedbackMaxPairsValue = Number.parseInt(alignmentPlaygroundMaxPairs, 10);
    const parsedTargetModules = targetModules
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);

    const config: Record<string, unknown> = {
      base_model: baseModel,
    };
    const includeField = (key: ConfigFieldKey): boolean => !useProfileDefaults || touchedConfig[key];
    if (includeField('training_mode')) config.training_mode = trainingMode;
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
    if (includeField('multimodal_require_media')) config.multimodal_require_media = multimodalRequireMedia;
    if (includeField('alignment_auto_filter')) config.alignment_auto_filter = alignmentAutoFilter;
    if (includeField('alignment_quality_threshold') && Number.isFinite(alignmentThreshold)) {
      config.alignment_quality_threshold = alignmentThreshold;
    }
    if (includeField('alignment_beta') && Number.isFinite(alignmentBetaValue)) {
      config.alignment_beta = alignmentBetaValue;
    }
    if (includeField('alignment_max_prompt_length') && Number.isFinite(alignmentPromptLengthValue)) {
      config.alignment_max_prompt_length = alignmentPromptLengthValue;
    }
    if (includeField('alignment_max_length') && Number.isFinite(alignmentMaxLengthValue)) {
      config.alignment_max_length = alignmentMaxLengthValue;
    }
    if (includeField('alignment_min_keep_ratio') && Number.isFinite(alignmentKeepRatio)) {
      config.alignment_min_keep_ratio = alignmentKeepRatio;
    }
    if (includeField('alignment_dataset_path')) {
      config.alignment_dataset_path = alignmentDatasetPath.trim();
    }
    if (includeField('alignment_include_playground_feedback')) {
      config.alignment_include_playground_feedback = alignmentIncludePlaygroundFeedback;
    }
    if (includeField('alignment_playground_max_pairs') && Number.isFinite(alignmentFeedbackMaxPairsValue)) {
      config.alignment_playground_max_pairs = alignmentFeedbackMaxPairsValue;
    }
    if (includeField('observability_enabled')) config.observability_enabled = observabilityEnabled;
    if (includeField('observability_log_steps')) config.observability_log_steps = observabilityLogSteps;
    if (includeField('observability_max_layers')) config.observability_max_layers = observabilityMaxLayers;
    if (includeField('observability_probe_attention')) {
      config.observability_probe_attention = observabilityProbeAttention;
    }
    if (includeField('observability_probe_top_k')) config.observability_probe_top_k = observabilityProbeTopK;
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
    setTrainingMode(parseString(config.training_mode, trainingMode));
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
    setMultimodalRequireMedia(parseBoolean(config.multimodal_require_media, multimodalRequireMedia));
    setAlignmentAutoFilter(parseBoolean(config.alignment_auto_filter, alignmentAutoFilter));
    setAlignmentQualityThreshold(String(config.alignment_quality_threshold ?? alignmentQualityThreshold));
    setAlignmentBeta(String(config.alignment_beta ?? alignmentBeta));
    setAlignmentMaxPromptLength(String(config.alignment_max_prompt_length ?? alignmentMaxPromptLength));
    setAlignmentMaxLength(String(config.alignment_max_length ?? alignmentMaxLength));
    setAlignmentMinKeepRatio(String(config.alignment_min_keep_ratio ?? alignmentMinKeepRatio));
    setAlignmentDatasetPath(parseString(config.alignment_dataset_path, alignmentDatasetPath));
    setAlignmentIncludePlaygroundFeedback(
      parseBoolean(config.alignment_include_playground_feedback, alignmentIncludePlaygroundFeedback),
    );
    setAlignmentPlaygroundMaxPairs(String(config.alignment_playground_max_pairs ?? alignmentPlaygroundMaxPairs));
    setObservabilityEnabled(parseBoolean(config.observability_enabled, observabilityEnabled));
    setObservabilityLogSteps(Math.max(1, parseNumber(config.observability_log_steps, observabilityLogSteps)));
    setObservabilityMaxLayers(Math.max(1, parseNumber(config.observability_max_layers, observabilityMaxLayers)));
    setObservabilityProbeAttention(
      parseBoolean(config.observability_probe_attention, observabilityProbeAttention),
    );
    setObservabilityProbeTopK(Math.max(1, parseNumber(config.observability_probe_top_k, observabilityProbeTopK)));

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

  const loadObservabilitySummary = async (experimentId: number, options?: { silent?: boolean }) => {
    if (!Number.isFinite(experimentId) || experimentId <= 0) {
      setObservabilitySummary(null);
      setObservabilityRecentCount(0);
      return;
    }
    if (!options?.silent) {
      setObservabilityLoading(true);
    }
    try {
      const res = await api.get<TrainingObservabilityResponse>(
        `/projects/${projectId}/training/observability/telemetry`,
        {
          params: {
            experiment_id: experimentId,
            limit: 100,
          },
        },
      );
      setObservabilitySummary(res.data?.summary || null);
      setObservabilityRecentCount(Number(res.data?.recent?.count || 0));
      setObservabilityError('');
    } catch (err: any) {
      if (!options?.silent) {
        setObservabilityError(err?.response?.data?.detail || 'Failed to load observability telemetry');
      }
    } finally {
      if (!options?.silent) {
        setObservabilityLoading(false);
      }
    }
  };

  const loadVibeTimeline = async (experimentId: number, options?: { silent?: boolean }) => {
    if (!Number.isFinite(experimentId) || experimentId <= 0) {
      setVibeTimeline([]);
      setVibeConfig(null);
      setVibeSelectedIndex(0);
      return;
    }
    if (!options?.silent) {
      setVibeLoading(true);
    }
    try {
      const res = await api.get<VibeCheckTimelineResponse>(
        `/projects/${projectId}/training/experiments/${experimentId}/vibe-check/timeline`,
        { params: { limit: 120 } },
      );
      const rows = Array.isArray(res.data?.snapshots) ? res.data.snapshots : [];
      setVibeTimeline(rows);
      setVibeConfig(res.data?.config || null);
      setVibeSelectedIndex(rows.length > 0 ? rows.length - 1 : 0);
      setVibeError('');
    } catch (err: any) {
      if (!options?.silent) {
        setVibeError(err?.response?.data?.detail || 'Failed to load vibe-check timeline');
      }
    } finally {
      if (!options?.silent) {
        setVibeLoading(false);
      }
    }
  };

  const loadTrainingRecipes = async () => {
    try {
      const res = await api.get<TrainingRecipeCatalogResponse>(`/projects/${projectId}/training/recipes`);
      const items = Array.isArray(res.data?.recipes) ? res.data.recipes : [];
      setTrainingRecipes(items);
      if (!selectedRecipeId && items.length > 0) {
        const balanced = items.find((item) => item.recipe_id === 'recipe.sft.balanced');
        setSelectedRecipeId((balanced || items[0]).recipe_id);
      }
      setRecipeResolveError('');
    } catch (err: any) {
      setTrainingRecipes([]);
      setRecipeResolveError(err?.response?.data?.detail || 'Failed to load recipe catalog');
    }
  };

  const loadCloudBurstCatalog = async () => {
    setCloudBurstLoadingCatalog(true);
    try {
      const res = await api.get<CloudBurstCatalogResponse>(`/projects/${projectId}/training/cloud-burst/catalog`);
      const payload = res.data || null;
      setCloudBurstCatalog(payload);
      const providers = Array.isArray(payload?.providers) ? payload.providers : [];
      const gpuSkus = Array.isArray(payload?.gpu_skus) ? payload.gpu_skus : [];
      const providerSelected = Boolean(
        cloudBurstProviderId && providers.some((item) => item.provider_id === cloudBurstProviderId),
      );
      const gpuSelected = Boolean(
        cloudBurstGpuSku && gpuSkus.some((item) => item.gpu_sku === cloudBurstGpuSku),
      );
      if (!providerSelected && providers.length > 0) {
        setCloudBurstProviderId(providers[0].provider_id);
      }
      if (!gpuSelected && gpuSkus.length > 0) {
        setCloudBurstGpuSku(gpuSkus[0].gpu_sku);
      }
      setCloudBurstError('');
    } catch (err: any) {
      setCloudBurstCatalog(null);
      setCloudBurstError(err?.response?.data?.detail || 'Failed to load cloud burst catalog');
    } finally {
      setCloudBurstLoadingCatalog(false);
    }
  };

  const requestCloudBurstQuote = async () => {
    if (!cloudBurstProviderId || !cloudBurstGpuSku) {
      setCloudBurstError('Select provider and GPU SKU before requesting quote.');
      return;
    }
    setCloudBurstLoadingQuote(true);
    setCloudBurstError('');
    try {
      const durationValue = Number.parseFloat(cloudBurstDurationHours);
      const storageValue = Number.parseInt(cloudBurstStorageGb, 10);
      const egressValue = Number.parseFloat(cloudBurstEgressGb);
      const res = await api.post<CloudBurstQuoteResponse>(
        `/projects/${projectId}/training/cloud-burst/quote`,
        {
          provider_id: cloudBurstProviderId,
          gpu_sku: cloudBurstGpuSku,
          duration_hours: Number.isFinite(durationValue) ? durationValue : 2.0,
          storage_gb: Number.isFinite(storageValue) ? storageValue : 50,
          egress_gb: Number.isFinite(egressValue) ? egressValue : 0.0,
          spot: cloudBurstSpot,
        },
      );
      setCloudBurstQuote(res.data || null);
    } catch (err: any) {
      setCloudBurstQuote(null);
      setCloudBurstError(err?.response?.data?.detail || 'Failed to estimate cloud burst quote');
    } finally {
      setCloudBurstLoadingQuote(false);
    }
  };

  const requestCloudBurstPlan = async () => {
    if (!cloudBurstProviderId || !cloudBurstGpuSku) {
      setCloudBurstError('Select provider and GPU SKU before building launch plan.');
      return;
    }
    setCloudBurstLoadingPlan(true);
    setCloudBurstError('');
    try {
      const durationValue = Number.parseFloat(cloudBurstDurationHours);
      const parsedExperimentId = Number.parseInt(cloudBurstExperimentId, 10);
      const res = await api.post<CloudBurstLaunchPlanResponse>(
        `/projects/${projectId}/training/cloud-burst/launch-plan`,
        {
          provider_id: cloudBurstProviderId,
          gpu_sku: cloudBurstGpuSku,
          duration_hours: Number.isFinite(durationValue) ? durationValue : 2.0,
          experiment_id: Number.isFinite(parsedExperimentId) && parsedExperimentId > 0
            ? parsedExperimentId
            : undefined,
          region: cloudBurstRegion.trim() || undefined,
          image: cloudBurstImage.trim(),
          startup_script: cloudBurstStartupScript.trim(),
          spot: cloudBurstSpot,
        },
      );
      setCloudBurstPlan(res.data || null);
    } catch (err: any) {
      setCloudBurstPlan(null);
      setCloudBurstError(err?.response?.data?.detail || 'Failed to build cloud burst launch plan');
    } finally {
      setCloudBurstLoadingPlan(false);
    }
  };

  const loadCloudBurstJobs = async (options?: { silent?: boolean }) => {
    if (!options?.silent) {
      setCloudBurstLoadingRuns(true);
    }
    try {
      const res = await api.get<CloudBurstRunListResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs?limit=12`,
      );
      const rows = Array.isArray(res.data?.runs) ? res.data.runs : [];
      setCloudBurstRuns(rows);
      if (!cloudBurstActiveRunId && rows.length > 0) {
        const firstRunId = String(rows[0]?.run_id || '').trim();
        setCloudBurstActiveRunId(firstRunId);
      }
      setCloudBurstError('');
    } catch (err: any) {
      if (!options?.silent) {
        setCloudBurstError(err?.response?.data?.detail || 'Failed to load cloud burst jobs');
      }
    } finally {
      if (!options?.silent) {
        setCloudBurstLoadingRuns(false);
      }
    }
  };

  const loadCloudBurstJobStatus = async (
    runId: string,
    options?: { silent?: boolean; logsTail?: number },
  ) => {
    const trimmedRunId = String(runId || '').trim();
    if (!trimmedRunId) {
      setCloudBurstActiveRun(null);
      setCloudBurstSyncCursor('');
      return;
    }
    if (!options?.silent) {
      setCloudBurstLoadingRuns(true);
    }
    try {
      const logsTail = Math.max(20, Math.min(1000, Number(options?.logsTail || 200)));
      const res = await api.get<CloudBurstRunStatusResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs/${trimmedRunId}?logs_tail=${logsTail}`,
      );
      const run = res.data || null;
      setCloudBurstActiveRun(run);
      setCloudBurstActiveRunId(trimmedRunId);
      const nextCursor = String(run?.artifacts?.last_sync_summary?.next_cursor || '').trim();
      setCloudBurstSyncCursor(nextCursor);
      setCloudBurstError('');
    } catch (err: any) {
      if (!options?.silent) {
        setCloudBurstError(err?.response?.data?.detail || 'Failed to load cloud burst job status');
      }
    } finally {
      if (!options?.silent) {
        setCloudBurstLoadingRuns(false);
      }
    }
  };

  const submitCloudBurstManagedJob = async () => {
    if (!cloudBurstProviderId || !cloudBurstGpuSku) {
      setCloudBurstError('Select provider and GPU SKU before submitting a managed job.');
      return;
    }
    setCloudBurstSubmittingJob(true);
    setCloudBurstError('');
    setCloudBurstInfo('');
    try {
      const durationValue = Number.parseFloat(cloudBurstDurationHours);
      const parsedExperimentId = Number.parseInt(cloudBurstExperimentId, 10);
      const res = await api.post<CloudBurstRunStatusResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs/submit`,
        {
          provider_id: cloudBurstProviderId,
          gpu_sku: cloudBurstGpuSku,
          duration_hours: Number.isFinite(durationValue) ? durationValue : 2.0,
          experiment_id: Number.isFinite(parsedExperimentId) && parsedExperimentId > 0
            ? parsedExperimentId
            : undefined,
          region: cloudBurstRegion.trim() || undefined,
          image: cloudBurstImage.trim(),
          startup_script: cloudBurstStartupScript.trim(),
          spot: cloudBurstSpot,
          auto_artifact_sync: true,
          artifact_sync_policy: 'smart',
          execution_mode: cloudBurstExecutionMode,
          allow_fallback_to_simulation: cloudBurstAllowFallbackToSimulation,
          idempotency_key: cloudBurstIdempotencyKey.trim() || undefined,
        },
      );
      const run = res.data || null;
      setCloudBurstActiveRun(run);
      setCloudBurstActiveRunId(String(run?.run_id || '').trim());
      setCloudBurstSyncCursor('');
      if (run?.idempotent_replay) {
        setCloudBurstInfo('Idempotency replay returned an existing managed run.');
      } else if (
        String(run?.execution_mode_requested || '') !== String(run?.execution_mode_effective || '')
        && String(run?.execution_mode_fallback_reason || '').trim()
      ) {
        setCloudBurstInfo(String(run?.execution_mode_fallback_reason || '').trim());
      }
      await loadCloudBurstJobs({ silent: true });
    } catch (err: any) {
      setCloudBurstError(err?.response?.data?.detail || 'Failed to submit managed cloud burst job');
    } finally {
      setCloudBurstSubmittingJob(false);
    }
  };

  const cancelCloudBurstManagedJob = async (runId: string) => {
    const trimmedRunId = String(runId || '').trim();
    if (!trimmedRunId) {
      return;
    }
    setCloudBurstCancellingJob(true);
    setCloudBurstError('');
    setCloudBurstInfo('');
    try {
      const res = await api.post<CloudBurstRunStatusResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs/${trimmedRunId}/cancel`,
      );
      setCloudBurstActiveRun(res.data || null);
      await loadCloudBurstJobs({ silent: true });
    } catch (err: any) {
      setCloudBurstError(err?.response?.data?.detail || 'Failed to cancel cloud burst job');
    } finally {
      setCloudBurstCancellingJob(false);
    }
  };

  const syncCloudBurstManagedArtifacts = async (runId: string) => {
    const trimmedRunId = String(runId || '').trim();
    if (!trimmedRunId) {
      return;
    }
    setCloudBurstSyncingArtifacts(true);
    setCloudBurstError('');
    setCloudBurstInfo('');
    try {
      const res = await api.post<CloudBurstRunStatusResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs/${trimmedRunId}/sync-artifacts`,
        {
          policy: 'smart',
          dry_run: false,
          max_files: 2000,
          cursor: cloudBurstSyncCursor.trim() || undefined,
        },
      );
      setCloudBurstActiveRun(res.data || null);
      const syncSummary = asRecord(asRecord(res.data).sync);
      const nextCursor = String(syncSummary.next_cursor || '').trim();
      setCloudBurstSyncCursor(nextCursor);
      if (nextCursor) {
        setCloudBurstInfo(
          `Partial sync complete. Continue syncing with next cursor (${nextCursor.slice(0, 24)}...).`,
        );
      }
      await loadCloudBurstJobs({ silent: true });
    } catch (err: any) {
      setCloudBurstError(err?.response?.data?.detail || 'Failed to sync cloud burst artifacts');
    } finally {
      setCloudBurstSyncingArtifacts(false);
    }
  };

  const applySelectedRecipe = async () => {
    if (!selectedRecipeId) {
      setRecipeResolveError('Select a recipe first.');
      return;
    }
    setRecipeResolveLoading(true);
    setRecipeResolveError('');
    setTrainingWarnings([]);
    try {
      const baseConfig = buildTrainingConfigPayload();
      const res = await api.post<TrainingRecipeResolveResponse>(
        `/projects/${projectId}/training/recipes/resolve`,
        {
          recipe_id: selectedRecipeId,
          base_config: baseConfig,
          include_preflight: true,
        },
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
      const resolvedCfg =
        (res.data?.resolved_training_config && typeof res.data.resolved_training_config === 'object'
          ? res.data.resolved_training_config
          : res.data?.recipe_config) || {};
      applySuggestedConfig(resolvedCfg);

      const preflight = res.data?.preflight || null;
      setPreflightPreview(preflight);
      const warnings = Array.isArray(preflight?.warnings) ? preflight.warnings.filter(Boolean) : [];
      setTrainingWarnings(warnings);

      const missing = Array.isArray(res.data?.recipe_missing_required_fields)
        ? res.data.recipe_missing_required_fields.filter(Boolean)
        : [];
      if (missing.length > 0) {
        setRecipeResolveError(`Recipe applied, but missing required fields: ${missing.join(', ')}`);
      }
    } catch (err: any) {
      setRecipeResolveError(err?.response?.data?.detail || 'Failed to apply recipe');
    } finally {
      setRecipeResolveLoading(false);
    }
  };

  const introspectBaseModel = async (options?: { modelId?: string; silent?: boolean }) => {
    const modelId = String(options?.modelId || baseModel || '').trim();
    if (!modelId) {
      if (!options?.silent) {
        setBaseModelIntrospectionError('Enter a model id/path first.');
      }
      return;
    }

    setBaseModelIntrospectionLoading(true);
    if (!options?.silent) {
      setBaseModelIntrospectionError('');
    }
    try {
      const res = await api.post<ModelIntrospectionResponse>(
        `/projects/${projectId}/training/model-selection/introspect`,
        {
          model_id: modelId,
          allow_network: true,
        },
      );
      setBaseModelIntrospection(res.data?.introspection || null);
    } catch (err: any) {
      if (!options?.silent) {
        setBaseModelIntrospectionError(
          err?.response?.data?.detail || 'Failed to introspect base model metadata',
        );
      }
    } finally {
      setBaseModelIntrospectionLoading(false);
    }
  };

  const runModelWizard = async (options?: { silent?: boolean }) => {
    setWizardLoading(true);
    setWizardError('');
    try {
      const vramValue = Number.parseFloat(wizardVramGb);
      const payload = {
        target_device: wizardTargetDevice,
        primary_language: wizardPrimaryLanguage,
        available_vram_gb: Number.isFinite(vramValue) && vramValue > 0 ? vramValue : undefined,
        task_profile: wizardTaskProfile !== 'auto' ? wizardTaskProfile : undefined,
        top_k: 3,
      };
      const res = await api.post<ModelWizardResponse>(
        `/projects/${projectId}/training/model-selection/recommend`,
        payload,
      );
      setWizardResult(res.data || null);
      const rows = Array.isArray(res.data?.recommendations) ? res.data.recommendations : [];
      const recommendationModelIds = rows
        .map((item) => String(item?.model_id || '').trim())
        .filter(Boolean);
      void api
        .post(`/projects/${projectId}/training/model-selection/telemetry`, {
          action: 'recommend',
          source: 'training_setup_wizard',
          auto_run: Boolean(options?.silent),
          target_device: payload.target_device,
          primary_language: payload.primary_language,
          available_vram_gb: payload.available_vram_gb,
          task_profile: payload.task_profile,
          top_k: payload.top_k,
          recommendation_count: recommendationModelIds.length,
          recommendation_model_ids: recommendationModelIds,
        })
        .catch(() => { });
    } catch (err: any) {
      setWizardResult(null);
      if (!options?.silent) {
        setWizardError(err?.response?.data?.detail || 'Failed to load model recommendations');
      }
    } finally {
      setWizardLoading(false);
    }
  };

  const loadModelBenchmarkHistory = async (options?: { silent?: boolean }) => {
    try {
      const res = await api.get<ModelBenchmarkHistoryResponse>(
        `/projects/${projectId}/training/model-selection/benchmark-sweep/history?limit=6`,
      );
      const rows = Array.isArray(res.data?.runs) ? res.data.runs : [];
      setBenchmarkHistory(rows);
    } catch (err: any) {
      if (!options?.silent) {
        setBenchmarkError(err?.response?.data?.detail || 'Failed to load benchmark history');
      }
    }
  };

  const runModelBenchmarkSweep = async () => {
    setBenchmarkLoading(true);
    setBenchmarkError('');
    try {
      const vramValue = Number.parseFloat(wizardVramGb);
      const recommendedModelIds = (Array.isArray(wizardResult?.recommendations)
        ? wizardResult.recommendations
        : []
      )
        .map((item) => String(item?.model_id || '').trim())
        .filter(Boolean);
      const payload = {
        target_device: wizardTargetDevice,
        primary_language: wizardPrimaryLanguage,
        available_vram_gb: Number.isFinite(vramValue) && vramValue > 0 ? vramValue : undefined,
        task_profile: wizardTaskProfile !== 'auto' ? wizardTaskProfile : undefined,
        model_ids: recommendedModelIds,
        max_models: Math.max(1, Math.min(3, recommendedModelIds.length || 3)),
        sample_size: 96,
        persist_run: true,
      };
      const res = await api.post<ModelBenchmarkSweepResponse>(
        `/projects/${projectId}/training/model-selection/benchmark-sweep`,
        payload,
      );
      setBenchmarkResult(res.data || null);
      await loadModelBenchmarkHistory({ silent: true });
    } catch (err: any) {
      setBenchmarkResult(null);
      setBenchmarkError(err?.response?.data?.detail || 'Failed to run benchmark sweep');
    } finally {
      setBenchmarkLoading(false);
    }
  };

  const applyModelSelectionChoice = ({
    modelId,
    rankIndex,
    selectedScore,
    defaults,
    applySource,
  }: {
    modelId: string;
    rankIndex: number;
    selectedScore?: number;
    defaults?: ModelWizardRecommendation['suggested_defaults'];
    applySource?: ModelSelectionApplySource;
  }) => {
    const trimmedModelId = String(modelId || '').trim();
    if (!trimmedModelId) return;
    const nextConfig: Record<string, unknown> = {
      base_model: trimmedModelId,
    };
    const resolvedDefaults = defaults || {};
    if (typeof resolvedDefaults.task_type === 'string' && resolvedDefaults.task_type.trim()) {
      nextConfig.task_type = resolvedDefaults.task_type;
    }
    if (typeof resolvedDefaults.chat_template === 'string' && resolvedDefaults.chat_template.trim()) {
      nextConfig.chat_template = resolvedDefaults.chat_template;
    }
    if (typeof resolvedDefaults.use_lora === 'boolean') {
      nextConfig.use_lora = resolvedDefaults.use_lora;
    }
    if (typeof resolvedDefaults.batch_size === 'number' && Number.isFinite(resolvedDefaults.batch_size)) {
      nextConfig.batch_size = Math.max(1, resolvedDefaults.batch_size);
    }
    if (typeof resolvedDefaults.max_seq_length === 'number' && Number.isFinite(resolvedDefaults.max_seq_length)) {
      nextConfig.max_seq_length = Math.max(128, resolvedDefaults.max_seq_length);
    }
    applySuggestedConfig(nextConfig);
    const vramValue = Number.parseFloat(wizardVramGb);
    const rows = Array.isArray(wizardResult?.recommendations) ? wizardResult.recommendations : [];
    void api
      .post(`/projects/${projectId}/training/model-selection/telemetry`, {
        action: 'apply',
        source: 'training_setup_wizard',
        apply_source: applySource || 'recommendation',
        target_device: wizardTargetDevice,
        primary_language: wizardPrimaryLanguage,
        available_vram_gb: Number.isFinite(vramValue) && vramValue > 0 ? vramValue : undefined,
        task_profile: wizardTaskProfile !== 'auto' ? wizardTaskProfile : undefined,
        recommendation_count: rows.length,
        recommendation_model_ids: rows
          .map((row) => String(row?.model_id || '').trim())
          .filter(Boolean),
        selected_model_id: trimmedModelId,
        selected_rank: Math.max(1, rankIndex + 1),
        selected_score: Number.isFinite(Number(selectedScore))
          ? Number(selectedScore)
          : undefined,
      })
      .catch(() => { });
    void introspectBaseModel({ modelId: trimmedModelId, silent: true });
  };

  const applyModelWizardRecommendation = (item: ModelWizardRecommendation, rankIndex: number) => {
    applyModelSelectionChoice({
      modelId: item.model_id,
      rankIndex,
      selectedScore: Number(item.match_score),
      defaults: item.suggested_defaults,
      applySource: 'recommendation',
    });
  };

  const applyBenchmarkWinner = () => {
    const matrix = Array.isArray(benchmarkResult?.matrix) ? benchmarkResult.matrix : [];
    if (matrix.length === 0) {
      setBenchmarkError('No benchmark winner available yet. Run benchmark sweep first.');
      return;
    }
    const winnerId = String(
      benchmarkResult?.tradeoff_summary?.best_balance_model_id
      || matrix[0]?.model_id
      || '',
    ).trim();
    if (!winnerId) {
      setBenchmarkError('Benchmark winner is unavailable in the current benchmark result.');
      return;
    }
    const winnerIndex = matrix.findIndex((item) => String(item?.model_id || '').trim() === winnerId);
    const winnerRow = winnerIndex >= 0 ? matrix[winnerIndex] : matrix[0];
    const wizardRow = (Array.isArray(wizardResult?.recommendations) ? wizardResult.recommendations : [])
      .find((item) => String(item?.model_id || '').trim() === winnerId);

    applyModelSelectionChoice({
      modelId: winnerId,
      rankIndex: winnerIndex >= 0 ? winnerIndex : 0,
      selectedScore: Number(winnerRow?.estimated_quality_score),
      defaults: wizardRow?.suggested_defaults,
      applySource: 'benchmark',
    });
  };

  const applyReviewConsensusWinner = () => {
    const recommendationWinnerId = String(modelSelectionSummary.recommendationWinnerId || '').trim();
    const benchmarkWinnerId = String(modelSelectionSummary.benchmarkWinnerId || '').trim();

    let selectedModelId = '';
    let selectedWinnerLabel: 'consensus' | 'benchmark' | 'recommendation' = 'recommendation';

    if (recommendationWinnerId && benchmarkWinnerId && recommendationWinnerId === benchmarkWinnerId) {
      selectedModelId = recommendationWinnerId;
      selectedWinnerLabel = 'consensus';
    } else if (benchmarkWinnerId) {
      selectedModelId = benchmarkWinnerId;
      selectedWinnerLabel = 'benchmark';
    } else if (recommendationWinnerId) {
      selectedModelId = recommendationWinnerId;
      selectedWinnerLabel = 'recommendation';
    }

    if (!selectedModelId) {
      setReviewSelectionActionNote('No model winner available yet. Run recommendation or benchmark first.');
      return;
    }

    const recommendationRows = Array.isArray(wizardResult?.recommendations)
      ? wizardResult.recommendations
      : [];
    const currentBenchmarkRows = Array.isArray(benchmarkResult?.matrix) ? benchmarkResult.matrix : [];
    const historyBenchmarkRows = Array.isArray(benchmarkHistory[0]?.matrix) ? benchmarkHistory[0].matrix || [] : [];
    const recommendationIndex = recommendationRows
      .findIndex((item) => String(item?.model_id || '').trim() === selectedModelId);
    const currentBenchmarkIndex = currentBenchmarkRows
      .findIndex((item) => String(item?.model_id || '').trim() === selectedModelId);
    const historyBenchmarkIndex = historyBenchmarkRows
      .findIndex((item) => String(item?.model_id || '').trim() === selectedModelId);

    const recommendationRow = recommendationIndex >= 0 ? recommendationRows[recommendationIndex] : null;
    const benchmarkRow = currentBenchmarkIndex >= 0
      ? currentBenchmarkRows[currentBenchmarkIndex]
      : historyBenchmarkIndex >= 0
        ? historyBenchmarkRows[historyBenchmarkIndex]
        : currentBenchmarkRows[0] || historyBenchmarkRows[0] || null;
    const benchmarkRankIndex = currentBenchmarkIndex >= 0
      ? currentBenchmarkIndex
      : historyBenchmarkIndex >= 0
        ? historyBenchmarkIndex
        : 0;
    const selectedRankIndex = recommendationIndex >= 0 ? recommendationIndex : benchmarkRankIndex;
    const recommendationScore = Number(recommendationRow?.match_score);
    const benchmarkScore = Number(benchmarkRow?.estimated_quality_score);
    const selectedScore = Number.isFinite(recommendationScore)
      ? recommendationScore
      : Number.isFinite(benchmarkScore)
        ? benchmarkScore
        : undefined;

    applyModelSelectionChoice({
      modelId: selectedModelId,
      rankIndex: selectedRankIndex,
      selectedScore,
      defaults: recommendationRow?.suggested_defaults,
      applySource: selectedWinnerLabel,
    });

    setBenchmarkError('');
    setReviewSelectionActionNote(
      `Applied ${selectedWinnerLabel} winner (${selectedModelId}) to base model.`,
    );
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

  const goToNextSetupTab = () => {
    if (!canSetupGoNext) return;
    const nextTab = setupTabOrder[setupTabIndex + 1];
    if (nextTab) {
      setSetupTab(nextTab);
    }
  };

  const goToPreviousSetupTab = () => {
    if (!canSetupGoBack) return;
    const prevTab = setupTabOrder[setupTabIndex - 1];
    if (prevTab) {
      setSetupTab(prevTab);
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
    if (forceCreateVisible || !canViewRuns) {
      setWorkspaceView('setup');
    } else if (!canConfigureExperiments) {
      setWorkspaceView('runs');
    } else {
      setWorkspaceView('overview');
    }

    setExperiments([]);
    setActiveExperiment(null);
    setMetrics([]);
    setTrainingLogs([]);
    setSelectedForCompare([]);
    setShowCompare(false);
    setShowCreate(Boolean(forceCreateVisible && !hideCreateControls));
    setSetupTab('basics');
    setTaskState('');
    setTrainingError('');
    setTrainingMode('sft');
    setTrainingRuntimeId('auto');
    setTaskType('causal_lm');
    setTrainerBackend('auto');
    setAlignmentAutoFilter(false);
    setAlignmentQualityThreshold('3.0');
    setAlignmentBeta('0.1');
    setAlignmentMaxPromptLength('1024');
    setAlignmentMaxLength('2048');
    setAlignmentMinKeepRatio('0.4');
    setAlignmentDatasetPath('');
    setAlignmentIncludePlaygroundFeedback(true);
    setAlignmentPlaygroundMaxPairs('5000');
    setObservabilityEnabled(true);
    setObservabilityLogSteps(50);
    setObservabilityMaxLayers(12);
    setObservabilityProbeAttention(true);
    setObservabilityProbeTopK(6);
    setMultimodalRequireMedia(false);
    setUseProfileDefaults(true);
    setTouchedConfig({
      training_mode: false,
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
      multimodal_require_media: false,
      alignment_auto_filter: false,
      alignment_quality_threshold: false,
      alignment_beta: false,
      alignment_max_prompt_length: false,
      alignment_max_length: false,
      alignment_min_keep_ratio: false,
      alignment_dataset_path: false,
      alignment_include_playground_feedback: false,
      alignment_playground_max_pairs: false,
      observability_enabled: false,
      observability_log_steps: false,
      observability_max_layers: false,
      observability_probe_attention: false,
      observability_probe_top_k: false,
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
    setTrainingRecipes([]);
    setSelectedRecipeId('');
    setRecipeResolveLoading(false);
    setRecipeResolveError('');
    setWizardTargetDevice('laptop');
    setWizardPrimaryLanguage('english');
    setWizardVramGb('8');
    setWizardTaskProfile('auto');
    setWizardLoading(false);
    setWizardError('');
    setWizardResult(null);
    setBenchmarkLoading(false);
    setBenchmarkError('');
    setBenchmarkResult(null);
    setBenchmarkHistory([]);
    setReviewSelectionActionNote('');
    setBaseModelIntrospection(null);
    setBaseModelIntrospectionLoading(false);
    setBaseModelIntrospectionError('');
    setWizardAutoRan(false);
    setCloudBurstCatalog(null);
    setCloudBurstProviderId('');
    setCloudBurstGpuSku('');
    setCloudBurstDurationHours('2');
    setCloudBurstStorageGb('50');
    setCloudBurstEgressGb('0');
    setCloudBurstSpot(true);
    setCloudBurstRegion('');
    setCloudBurstImage('');
    setCloudBurstStartupScript('');
    setCloudBurstExperimentId('');
    setCloudBurstExecutionMode('auto');
    setCloudBurstAllowFallbackToSimulation(true);
    setCloudBurstIdempotencyKey('');
    setCloudBurstSyncCursor('');
    setCloudBurstLoadingCatalog(false);
    setCloudBurstLoadingQuote(false);
    setCloudBurstLoadingPlan(false);
    setCloudBurstError('');
    setCloudBurstInfo('');
    setCloudBurstQuote(null);
    setCloudBurstPlan(null);
    setCloudBurstRuns([]);
    setCloudBurstActiveRunId('');
    setCloudBurstActiveRun(null);
    setCloudBurstLoadingRuns(false);
    setCloudBurstSubmittingJob(false);
    setCloudBurstCancellingJob(false);
    setCloudBurstSyncingArtifacts(false);
    setCloudBurstPrefillStage('');
    setObservabilitySummary(null);
    setObservabilityRecentCount(0);
    setObservabilityError('');
    setObservabilityLoading(false);
    void loadPreferredPlanProfile();
    void loadTrainingRuntimes();
    void loadTrainingRecipes();
    void loadCloudBurstCatalog();
    void loadCloudBurstJobs({ silent: true });
    void loadModelBenchmarkHistory({ silent: true });
    refreshExperiments().catch((err) => console.error('Failed to load experiments', err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, forceCreateVisible, hideCreateControls, hideExperimentList]);

  useEffect(() => {
    let cancelled = false;
    const applyCloudBurstPrefill = async () => {
      const prefill = await loadWorkflowStagePrefill(projectId, ['cloud_burst']);
      if (cancelled || !prefill) {
        return;
      }
      const cfg = prefill.config || {};
      const providerToken = String(cfg.provider_id || '').trim();
      if (providerToken) setCloudBurstProviderId(providerToken);
      const skuToken = String(cfg.gpu_sku || '').trim();
      if (skuToken) setCloudBurstGpuSku(skuToken);

      const durationValue = Number(cfg.duration_hours);
      if (Number.isFinite(durationValue) && durationValue > 0) {
        setCloudBurstDurationHours(String(durationValue));
      }
      const storageValue = Number(cfg.storage_gb);
      if (Number.isFinite(storageValue) && storageValue > 0) {
        setCloudBurstStorageGb(String(Math.round(storageValue)));
      }
      const egressValue = Number(cfg.egress_gb);
      if (Number.isFinite(egressValue) && egressValue >= 0) {
        setCloudBurstEgressGb(String(egressValue));
      }
      if (typeof cfg.spot === 'boolean') {
        setCloudBurstSpot(cfg.spot);
      }
      const regionToken = String(cfg.region || '').trim();
      if (regionToken) setCloudBurstRegion(regionToken);
      const imageToken = String(cfg.image || '').trim();
      if (imageToken) setCloudBurstImage(imageToken);
      const startupToken = String(cfg.startup_script || '').trim();
      if (startupToken) setCloudBurstStartupScript(startupToken);
      const experimentValue = Number(cfg.experiment_id);
      if (Number.isFinite(experimentValue) && experimentValue > 0) {
        setCloudBurstExperimentId(String(Math.round(experimentValue)));
      }
      const executionModeToken = String(cfg.execution_mode || '').trim().toLowerCase();
      if (executionModeToken === 'auto' || executionModeToken === 'live' || executionModeToken === 'simulate') {
        setCloudBurstExecutionMode(executionModeToken);
      }
      if (typeof cfg.allow_fallback_to_simulation === 'boolean') {
        setCloudBurstAllowFallbackToSimulation(cfg.allow_fallback_to_simulation);
      }
      const idempotencyToken = String(cfg.idempotency_key || '').trim();
      if (idempotencyToken) {
        setCloudBurstIdempotencyKey(idempotencyToken);
      }
      setCloudBurstPrefillStage(prefill.stage);
    };
    void applyCloudBurstPrefill();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  useEffect(() => {
    const runId = String(cloudBurstActiveRunId || '').trim();
    if (!runId) {
      return;
    }
    void loadCloudBurstJobStatus(runId, { silent: true, logsTail: 240 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloudBurstActiveRunId, projectId]);

  useEffect(() => {
    const runId = String(cloudBurstActiveRunId || '').trim();
    if (!runId || cloudBurstActiveIsTerminal || !cloudBurstActiveStatus) {
      return;
    }
    const interval = window.setInterval(() => {
      void loadCloudBurstJobStatus(runId, { silent: true, logsTail: 240 });
      void loadCloudBurstJobs({ silent: true });
    }, 4000);
    return () => window.clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloudBurstActiveRunId, cloudBurstActiveStatus, cloudBurstActiveIsTerminal, projectId]);

  useEffect(() => {
    if (workspaceView !== 'setup') {
      setSetupTab('basics');
      return;
    }
    if (!setupTabOrder.includes(setupTab)) {
      setSetupTab('basics');
    }
  }, [workspaceView, setupTabOrder, setupTab]);

  useEffect(() => {
    const currentModel = String(baseModel || '').trim();
    const inspectedModel = String(baseModelIntrospection?.model_id || '').trim();
    if (!currentModel) {
      setBaseModelIntrospection(null);
      setBaseModelIntrospectionError('');
      return;
    }
    if (inspectedModel && inspectedModel !== currentModel) {
      setBaseModelIntrospection(null);
      setBaseModelIntrospectionError('');
    }
  }, [baseModel, baseModelIntrospection?.model_id]);

  useEffect(() => {
    setReviewSelectionActionNote('');
  }, [modelSelectionSummary.recommendationWinnerId, modelSelectionSummary.benchmarkWinnerId]);

  useEffect(() => {
    if (workspaceView !== 'setup' || !createFormVisible) {
      return;
    }
    if (!isSetupAdvancedMode || setupTab !== 'power') {
      return;
    }
    if (wizardAutoRan || wizardLoading) {
      return;
    }
    if (Array.isArray(wizardResult?.recommendations) && wizardResult.recommendations.length > 0) {
      return;
    }
    setWizardAutoRan(true);
    void runModelWizard({ silent: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceView, createFormVisible, wizardAutoRan, wizardLoading, projectId, isSetupAdvancedMode, setupTab]);

  useEffect(() => {
    if (forceCreateVisible || !canViewRuns) {
      if (workspaceView !== 'setup') {
        setWorkspaceView('setup');
      }
      return;
    }
    if (!canConfigureExperiments && workspaceView !== 'runs') {
      setWorkspaceView('runs');
      return;
    }
    if (workspaceView === 'setup' && !canConfigureExperiments) {
      setWorkspaceView('runs');
      return;
    }
    if (workspaceView === 'runs' && !canViewRuns) {
      setWorkspaceView('setup');
    }
  }, [forceCreateVisible, canConfigureExperiments, canViewRuns, workspaceView]);

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
        if (data.type === 'vibe_check' && data.snapshot) {
          const snapshot = data.snapshot as VibeCheckSnapshot;
          let nextIndex = 0;
          setVibeTimeline((prev) => {
            const current = prev.filter((item) => Number(item.step || 0) !== Number(snapshot.step || 0));
            const next = [...current, snapshot].sort((a, b) => Number(a.step || 0) - Number(b.step || 0));
            const capped = next.slice(-120);
            nextIndex = capped.length > 0 ? capped.length - 1 : 0;
            return capped;
          });
          setVibeSelectedIndex(nextIndex);
          setVibeError('');
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
    if (!(forceCreateVisible || showCreate)) {
      return;
    }
    void previewEffectiveConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showCreate, forceCreateVisible, projectId]);

  useEffect(() => {
    if (!activeExperiment) {
      setObservabilitySummary(null);
      setObservabilityRecentCount(0);
      setObservabilityError('');
      setObservabilityLoading(false);
      setVibeTimeline([]);
      setVibeConfig(null);
      setVibeSelectedIndex(0);
      setVibeError('');
      setVibeLoading(false);
      return;
    }
    const experimentId = activeExperiment.id;
    void loadObservabilitySummary(experimentId, { silent: false });
    void loadVibeTimeline(experimentId, { silent: false });
    const pollMs = activeExperiment.status === 'running' ? 7000 : 15000;
    const interval = window.setInterval(() => {
      void loadObservabilitySummary(experimentId, { silent: true });
      void loadVibeTimeline(experimentId, { silent: true });
    }, pollMs);
    return () => window.clearInterval(interval);
  }, [activeExperimentKey, projectId]);

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
      if (!forceCreateVisible) {
        setShowCreate(false);
      }
      setName('');
      setTrainingRuntimeId('auto');
      setPreflightPreview(null);
      setPreflightPreviewError('');
      setPreflightPlan(null);
      setPreflightPlanError('');
      setRecipeResolveError('');
      setTouchedConfig({
        training_mode: false,
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
        multimodal_require_media: false,
        alignment_auto_filter: false,
        alignment_quality_threshold: false,
        alignment_beta: false,
        alignment_max_prompt_length: false,
        alignment_max_length: false,
        alignment_min_keep_ratio: false,
        alignment_dataset_path: false,
        alignment_include_playground_feedback: false,
        alignment_playground_max_pairs: false,
        observability_enabled: false,
        observability_log_steps: false,
        observability_max_layers: false,
        observability_probe_attention: false,
        observability_probe_top_k: false,
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
        setObservabilitySummary(null);
        setObservabilityRecentCount(0);
        setObservabilityError('');
        void loadObservabilitySummary(experimentId, { silent: true });
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

  const handleApplyHardwareRecommendation = (rec: RecommendationResult) => {
    setBaseModel(rec.base_model);
    setUseLora(true);
    setLoraR(rec.lora_rank);
    setLoraAlpha(rec.lora_rank * 2);
    setBatchSize(rec.training_batch_size);
    setTouchedConfig((prev) => ({
      ...prev,
      use_lora: true,
      lora_r: true,
      lora_alpha: true,
      batch_size: true,
    }));
    setShowHardwareModal(false);
  };

  const viewDashboard = (exp: Experiment) => {
    setActiveExperiment(exp);
    setMetrics([]);
    setTrainingLogs([]);
    setTaskState('');
    setTrainingWarnings([]);
    setObservabilitySummary(null);
    setObservabilityRecentCount(0);
    setObservabilityError('');
    setVibeTimeline([]);
    setVibeConfig(null);
    setVibeSelectedIndex(0);
    setVibeError('');
    void loadObservabilitySummary(exp.id, { silent: true });
    void loadVibeTimeline(exp.id, { silent: true });
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
    const anomalyRate = Number(observabilitySummary?.gradient_anomaly_rate || 0);
    const hallucinationRate = Number(observabilitySummary?.hallucination_signal_rate || 0);
    const topLayers = Array.isArray(observabilitySummary?.top_layers)
      ? observabilitySummary?.top_layers || []
      : [];
    const topTokens = Array.isArray(observabilitySummary?.top_attention_tokens)
      ? observabilitySummary?.top_attention_tokens || []
      : [];
    const safeVibeIndex = Math.min(
      Math.max(vibeSelectedIndex, 0),
      Math.max(0, vibeTimeline.length - 1),
    );
    const selectedVibeSnapshot = vibeTimeline.length > 0 ? vibeTimeline[safeVibeIndex] : null;

    return (
      <div className="animate-fade-in training-panel-stack">
        <div className="card">
          <button
            className="btn btn-secondary btn-sm training-back-btn"
            onClick={() => {
              setActiveExperiment(null);
              setTaskState('');
              setTrainingWarnings([]);
              refreshExperiments().catch(() => undefined);
            }}
          >
            ← Back to Experiments
          </button>

          <div className="training-active-head">
            <div>
              <h3 className="training-active-title">{activeExperiment.name}</h3>
              <div className="training-active-meta">
                {activeExperiment.base_model} • {activeExperiment.training_mode}
              </div>
              {activeExperiment.domain_pack_applied && (
                <div className="training-active-submeta">
                  Pack: {activeExperiment.domain_pack_applied}
                  {activeExperiment.domain_pack_source ? ` (${activeExperiment.domain_pack_source})` : ''}
                </div>
              )}
              {activeExperiment.domain_profile_applied && (
                <div className="training-active-submeta">
                  Profile: {activeExperiment.domain_profile_applied}
                  {activeExperiment.domain_profile_source ? ` (${activeExperiment.domain_profile_source})` : ''}
                </div>
              )}
              {taskState && (
                <div className="training-active-submeta">
                  Worker task state: {taskState}
                </div>
              )}
            </div>
            <div className="training-inline-actions">
              {activeExperiment.status === 'running' && (
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => void handleCancel(activeExperiment.id)}
                >
                  Cancel
                </button>
              )}
              <span className={`badge ${statusColor(activeExperiment.status)} training-status-badge`}>
                {activeExperiment.status.toUpperCase()}
              </span>
            </div>
          </div>

          {trainingError && (
            <div className="training-alert training-alert--error">
              {trainingError}
            </div>
          )}
          {trainingWarnings.length > 0 && (
            <div className="training-alert training-alert--warning">
              Preflight warnings: {trainingWarnings.join(' | ')}
            </div>
          )}

          <div className="metrics-grid">
            <div className="metric-box box-blue">
              <span className="mb-label">Current Epoch</span>
              <span className="mb-value">{currentEpoch} / {totalEpochs}</span>
              <div className="metric-subtext">
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

          {/* P20 — Why-this-plan + checkpoints (Wave D backend exposure). */}
          <WhyThisPlanPanel projectId={projectId} experiment={activeExperiment} />
          <CheckpointsPanel
            projectId={projectId}
            experiment={activeExperiment}
            onLifecycleChange={() => {
              void refreshExperiments();
            }}
          />

          <div className="training-observability-panel">
            <div className="training-observability-panel__head">
              <h4>Observability Telemetry</h4>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => void loadObservabilitySummary(activeExperiment.id)}
                disabled={observabilityLoading}
              >
                {observabilityLoading ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
            {observabilityError && (
              <div className="training-alert training-alert--warning training-alert--tight">{observabilityError}</div>
            )}
            <div className="training-observability-panel__stats">
              <span className="badge badge-info">Events {observabilitySummary?.event_count ?? 0}</span>
              <span>Recent payloads: {observabilityRecentCount}</span>
              <span>Gradient anomaly rate: {(anomalyRate * 100).toFixed(1)}%</span>
              <span>Hallucination signal rate: {(hallucinationRate * 100).toFixed(1)}%</span>
              <span>Step range: {observabilitySummary?.step_min ?? '—'} - {observabilitySummary?.step_max ?? '—'}</span>
            </div>
            <div className="training-observability-panel__grid">
              <div>
                <div className="training-observability-panel__subtitle">Top Layers</div>
                <div className="training-observability-panel__list">
                  {topLayers.length === 0 ? (
                    <div className="training-observability-panel__muted">No gradient snapshots yet.</div>
                  ) : (
                    topLayers.slice(0, 6).map((layer, idx) => (
                      <div key={`obs-layer-${idx}`} className="training-observability-panel__row">
                        <span>{layer.layer || 'layer'}</span>
                        <strong>avg {Number(layer.avg_grad_norm || 0).toFixed(4)}</strong>
                        <span>max {Number(layer.max_grad_norm || 0).toFixed(4)}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
              <div>
                <div className="training-observability-panel__subtitle">Top Attention Tokens</div>
                <div className="training-observability-panel__list">
                  {topTokens.length === 0 ? (
                    <div className="training-observability-panel__muted">No attention probe samples yet.</div>
                  ) : (
                    topTokens.slice(0, 8).map((token, idx) => (
                      <div key={`obs-token-${idx}`} className="training-observability-panel__row">
                        <span>{token.token || 'token'}</span>
                        <strong>{token.count ?? 0}</strong>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="training-vibe-panel">
            <div className="training-vibe-panel__head">
              <h4>Vibe Check Timeline</h4>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => void loadVibeTimeline(activeExperiment.id)}
                disabled={vibeLoading}
              >
                {vibeLoading ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
            {vibeError && (
              <div className="training-alert training-alert--warning training-alert--tight">{vibeError}</div>
            )}
            <div className="training-vibe-panel__meta">
              <span>Snapshots: {vibeTimeline.length}</span>
              <span>Interval: {vibeConfig?.interval_steps ?? 50} steps</span>
              <span>Provider: {vibeConfig?.provider || 'mock'}</span>
              <span>Prompts: {Array.isArray(vibeConfig?.prompts) ? vibeConfig?.prompts.length : 0}</span>
            </div>
            {selectedVibeSnapshot ? (
              <div className="training-vibe-panel__body">
                <div className="training-vibe-panel__slider">
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, vibeTimeline.length - 1)}
                    value={safeVibeIndex}
                    onChange={(e) => setVibeSelectedIndex(Number(e.target.value || 0))}
                  />
                  <div className="training-vibe-panel__snapshot-meta">
                    <span>Step {selectedVibeSnapshot.step ?? '—'}</span>
                    <span>Epoch {selectedVibeSnapshot.epoch ?? '—'}</span>
                    <span>
                      Progress{' '}
                      {typeof selectedVibeSnapshot.progress === 'number'
                        ? `${(selectedVibeSnapshot.progress * 100).toFixed(1)}%`
                        : '—'}
                    </span>
                  </div>
                </div>
                <div className="training-vibe-grid">
                  {(Array.isArray(selectedVibeSnapshot.outputs) ? selectedVibeSnapshot.outputs : []).map((item, idx) => (
                    <article className="training-vibe-card" key={`vibe-${selectedVibeSnapshot.step || 0}-${idx}`}>
                      <div className="training-vibe-card__prompt">{item.prompt || 'Prompt'}</div>
                      <div className="training-vibe-card__reply">{item.reply || 'No reply generated.'}</div>
                    </article>
                  ))}
                </div>
              </div>
            ) : (
              <div className="training-vibe-panel__empty">
                No vibe snapshots yet. Snapshots appear every configured interval during training.
              </div>
            )}
          </div>

          <TerminalConsole logs={trainingLogs} height="320px" />
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in training-panel-stack">
      <div className="card">
        <div className="training-panel-head">
          <h3 className="training-panel-title">{title}</h3>
          {!hideCreateControls && !forceCreateVisible && (
            <button
              className="btn btn-primary"
              onClick={() => {
                setWorkspaceView('setup');
                setSetupTab('basics');
                setShowCreate(true);
                setPreflightPreview(null);
                setPreflightPreviewError('');
                setPreflightPlan(null);
                setPreflightPlanError('');
              }}
            >
              + New Experiment
            </button>
          )}
        </div>

        <div className="training-journey-strip">
          <div className="training-journey-card">
            <span className="training-journey-card__index">1</span>
            <div>
              <strong>Setup</strong>
              <p>Recipe, profile defaults, preflight, hyperparameters.</p>
            </div>
          </div>
          <div className="training-journey-card">
            <span className="training-journey-card__index">2</span>
            <div>
              <strong>Run</strong>
              <p>Create and start experiments, compare multiple runs.</p>
            </div>
          </div>
          <div className="training-journey-card">
            <span className="training-journey-card__index">3</span>
            <div>
              <strong>Monitor</strong>
              <p>Open dashboard for live epoch/loss and worker logs.</p>
            </div>
          </div>
        </div>

        {showWorkspaceTabs && (
          <div className="training-workspace-tabs">
            <button
              className={`training-workspace-tab ${workspaceView === 'overview' ? 'active' : ''}`}
              onClick={() => setWorkspaceView('overview')}
            >
              Overview
            </button>
            <button
              className={`training-workspace-tab ${workspaceView === 'setup' ? 'active' : ''}`}
              onClick={() => {
                setWorkspaceView('setup');
                setSetupTab('basics');
                if (!forceCreateVisible) {
                  setShowCreate(true);
                }
              }}
            >
              Setup
            </button>
            <button
              className={`training-workspace-tab ${workspaceView === 'runs' ? 'active' : ''}`}
              onClick={() => setWorkspaceView('runs')}
            >
              Runs
            </button>
          </div>
        )}

        {workspaceView === 'overview' && (
          <>
            <div className="training-overview-grid">
              <article className="training-overview-card">
                <span className="training-overview-card__label">Experiments</span>
                <strong>{experimentStats.total}</strong>
                <p>Total created</p>
              </article>
              <article className="training-overview-card">
                <span className="training-overview-card__label">Running</span>
                <strong>{experimentStats.running}</strong>
                <p>Active right now</p>
              </article>
              <article className="training-overview-card">
                <span className="training-overview-card__label">Completed</span>
                <strong>{experimentStats.completed}</strong>
                <p>Finished successfully</p>
              </article>
              <article className="training-overview-card">
                <span className="training-overview-card__label">Recommended Next</span>
                <strong>{recommendedAction.title}</strong>
                <p>{recommendedAction.detail}</p>
              </article>
            </div>
            <div className="training-overview-actions">
              {canConfigureExperiments && (
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setWorkspaceView('setup');
                    setSetupTab('basics');
                    if (!forceCreateVisible) setShowCreate(true);
                  }}
                >
                  Go to Setup
                </button>
              )}
              {canViewRuns && (
                <button className="btn btn-secondary" onClick={() => setWorkspaceView('runs')}>
                  View Runs
                </button>
              )}
            </div>
          </>
        )}

        {canConfigureExperiments && !showWorkspaceTabs && createFormVisible && (
          <div className="training-form-intro">
            <span>Configure your run settings below, then create the experiment.</span>
          </div>
        )}

        {canConfigureExperiments && workspaceView === 'setup' && !createFormVisible && !forceCreateVisible && (
          <div className="training-empty-helper">
            <p>Setup form is currently hidden.</p>
            <button className="btn btn-secondary" onClick={() => setShowCreate(true)}>
              Open Setup Form
            </button>
          </div>
        )}

        {canConfigureExperiments && workspaceView === 'setup' && createFormVisible && (
          <div className="training-create-shell">
            <div className="training-create-shell__head">
              <strong>Create Experiment</strong>
              <span className="training-create-shell__hint">
                {isSetupAdvancedMode
                  ? 'Use recipe + defaults for quick setup, then open advanced sections only if needed.'
                  : 'Essentials mode keeps only launch-critical controls visible. Switch to Advanced for full tuning.'}
              </span>
            </div>
            <ReadinessPanel projectId={projectId} />
            <div className="training-setup-tabs" role="tablist" aria-label="Training setup steps">
              {setupTabOrder.map((tab, idx) => {
                const label = tab === 'basics'
                  ? 'Basics'
                  : tab === 'config'
                    ? 'Config'
                    : tab === 'power'
                      ? 'Power Tools'
                      : 'Review';
                return (
                  <button
                    key={tab}
                    className={`training-setup-tab ${setupTab === tab ? 'active' : ''}`}
                    onClick={() => setSetupTab(tab)}
                    role="tab"
                    aria-selected={setupTab === tab}
                  >
                    <span className="setup-tab-index">{idx + 1}</span>
                    <span>{label}</span>
                  </button>
                );
              })}
            </div>

            {showSetupBasics && (
              <>
                <div className="form-group">
                  <label className="form-label">Experiment Name</label>
                  <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. llama3-sft-v1" />
                </div>
                <div className="form-group form-group--spaced">
                  <label className="form-label form-label-inline">
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
                <div className="form-group form-group--spaced">
                  <label className="form-label">Recipe Starter</label>
                  <div className="form-inline-actions">
                    <select
                      className="input training-recipe-select"
                      value={selectedRecipeId}
                      onChange={(e) => setSelectedRecipeId(e.target.value)}
                    >
                      <option value="">Select recipe</option>
                      {trainingRecipes.map((recipe) => (
                        <option key={recipe.recipe_id} value={recipe.recipe_id}>
                          {recipe.display_name}
                        </option>
                      ))}
                    </select>
                    <button
                      className="btn btn-secondary"
                      onClick={() => void applySelectedRecipe()}
                      disabled={recipeResolveLoading || !selectedRecipeId}
                    >
                      {recipeResolveLoading ? 'Applying...' : 'Apply Recipe'}
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={() => setShowHardwareModal(true)}
                      title="Optimize settings for target hardware"
                    >
                      ✨ Hardware Auto-Tuner
                    </button>
                  </div>
                  <div className="form-hint">
                    Recipe applies a domain-agnostic config patch, then runtime/profile defaults and preflight.
                  </div>
                  {recipeResolveError && (
                    <div className="training-alert training-alert--warning training-alert--tight">
                      {recipeResolveError}
                    </div>
                  )}
                </div>

                {!isSetupAdvancedMode && (
                  <div className="training-essentials-tools">
                    <button
                      className="btn btn-secondary"
                      onClick={() => void runPreflightPreview()}
                      disabled={preflightPreviewLoading}
                    >
                      {preflightPreviewLoading ? 'Checking...' : 'Run Quick Preflight'}
                    </button>
                    {preflightPreview && (
                      <span className={`training-essentials-preflight ${preflightPreview.ok ? 'ok' : 'blocked'}`}>
                        {preflightPreview.ok
                          ? 'Preflight passed'
                          : `${preflightPreview.errors.length} blocking issue(s)`}
                      </span>
                    )}
                    {preflightPreview && (
                      <div className={`training-essentials-model-gate training-essentials-model-gate--${essentialsModelGateSummary.statusClass}`}>
                        <strong>Model Gate: {essentialsModelGateSummary.statusLabel}</strong>
                        <span>
                          {essentialsModelGateSummary.modelId} • {essentialsModelGateSummary.architecture} • source {essentialsModelGateSummary.source}
                        </span>
                        {essentialsModelGateSummary.topIssue && (
                          <span>{essentialsModelGateSummary.topIssue}</span>
                        )}
                        {essentialsModelGateSummary.supportedArchitectures && (
                          <span>Supported: {essentialsModelGateSummary.supportedArchitectures}</span>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}

            {showSetupPower && (
            <details className="training-collapsible">
              <summary>
                <span>Validation & Planning</span>
                <small>Preflight, effective config, and suggested plans</small>
              </summary>
              <div className="training-collapsible__content">
                <div className="form-group form-group--spaced">
                  <div className="form-inline-actions">
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
                  <div className="training-alert training-alert--error">
                    {effectivePreviewError}
                  </div>
                )}
                {preflightPreviewError && (
                  <div className="training-alert training-alert--error">
                    {preflightPreviewError}
                  </div>
                )}
                {preflightPlanError && (
                  <div className="training-alert training-alert--error">
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
                    className={`training-preflight-panel ${preflightPreview.ok ? 'training-preflight-panel--ok' : 'training-preflight-panel--error'
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
                    {modelSelectionSummary.hasAny && (
                      <div className="training-preflight-panel__section">
                        <div className="training-preflight-panel__section-title">Model Selection Snapshot</div>
                        <div className="training-preflight-contract-grid">
                          <div className="training-preflight-contract-card">
                            <div className="training-preflight-contract-card__title">Recommendation Winner</div>
                            <div className="training-preflight-contract-card__row">
                              <span>Model</span>
                              <strong>{modelSelectionSummary.recommendationWinnerId || 'not available'}</strong>
                            </div>
                            <div className="training-preflight-contract-card__row">
                              <span>Match Score</span>
                              <strong>
                                {Number.isFinite(Number(modelSelectionSummary.recommendationWinnerScore))
                                  ? Number(modelSelectionSummary.recommendationWinnerScore).toFixed(2)
                                  : 'n/a'}
                              </strong>
                            </div>
                            <div className="training-preflight-contract-card__row">
                              <span>Adaptive Bias</span>
                              <strong>
                                {Number.isFinite(Number(modelSelectionSummary.recommendationWinnerAdaptiveBias))
                                  ? Number(modelSelectionSummary.recommendationWinnerAdaptiveBias).toFixed(2)
                                  : 'n/a'}
                              </strong>
                            </div>
                            <div className="training-preflight-contract-card__row">
                              <span>Top Reason</span>
                              <strong>{modelSelectionSummary.recommendationWinnerReason || 'n/a'}</strong>
                            </div>
                          </div>
                          <div className="training-preflight-contract-card">
                            <div className="training-preflight-contract-card__title">Benchmark Winner</div>
                            <div className="training-preflight-contract-card__row">
                              <span>Model</span>
                              <strong>{modelSelectionSummary.benchmarkWinnerId || 'not available'}</strong>
                            </div>
                            <div className="training-preflight-contract-card__row">
                              <span>Quality</span>
                              <strong>
                                {Number.isFinite(Number(modelSelectionSummary.benchmarkWinnerAccuracy))
                                  ? `${Number(modelSelectionSummary.benchmarkWinnerAccuracy).toFixed(1)}%`
                                  : 'n/a'}
                              </strong>
                            </div>
                            <div className="training-preflight-contract-card__row">
                              <span>Latency</span>
                              <strong>
                                {Number.isFinite(Number(modelSelectionSummary.benchmarkWinnerLatencyMs))
                                  ? `${Number(modelSelectionSummary.benchmarkWinnerLatencyMs).toFixed(1)} ms`
                                  : 'n/a'}
                              </strong>
                            </div>
                            <div className="training-preflight-contract-card__row">
                              <span>Throughput</span>
                              <strong>
                                {Number.isFinite(Number(modelSelectionSummary.benchmarkWinnerThroughputTps))
                                  ? `${Number(modelSelectionSummary.benchmarkWinnerThroughputTps).toFixed(1)} t/s`
                                  : 'n/a'}
                              </strong>
                            </div>
                            <div className="training-preflight-contract-card__row">
                              <span>Source</span>
                              <strong>
                                {modelSelectionSummary.benchmarkWinnerMode || 'real_sampled'}
                                {modelSelectionSummary.benchmarkWinnerSource
                                  ? ` • ${modelSelectionSummary.benchmarkWinnerSource}`
                                  : ''}
                              </strong>
                            </div>
                          </div>
                        </div>
                        <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--hint">
                          {modelSelectionSummary.winnerAlignmentLabel}
                          {modelSelectionSummary.activeModelId
                            ? ` Active model (${modelSelectionSummary.activeModelId}) ${modelSelectionSummary.activeModelLabel}.`
                            : ''}
                        </div>
                      </div>
                    )}
                    <div className="training-preflight-panel__section">
                      <div className="training-preflight-panel__section-title">Capability Contract Diagnostics</div>
                      <div className="training-preflight-contract-grid">
                        <div className="training-preflight-contract-card">
                          <div className="training-preflight-contract-card__title">Task + Trainer</div>
                          <div className="training-preflight-contract-card__row">
                            <span>Task Type</span>
                            <strong>{preflightContractDetails.taskType}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Training Mode</span>
                            <strong>{preflightContractDetails.trainingMode}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Trainer Backend</span>
                            <strong>{preflightContractDetails.trainerBackend}</strong>
                          </div>
                        </div>
                        <div className="training-preflight-contract-card">
                          <div className="training-preflight-contract-card__title">Runtime Contract</div>
                          <div className="training-preflight-contract-card__row">
                            <span>Runtime</span>
                            <strong>{preflightContractDetails.runtimeId}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Backend</span>
                            <strong>{preflightContractDetails.runtimeBackend}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Modalities</span>
                            <strong>
                              {preflightContractDetails.runtimeSupportedModalities.length > 0
                                ? preflightContractDetails.runtimeSupportedModalities.join(', ')
                                : 'text'}
                            </strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Metadata</span>
                            <strong>
                              {preflightContractDetails.runtimeModalitiesDeclared === true
                                ? 'Declared'
                                : preflightContractDetails.runtimeModalitiesDeclared === false
                                  ? 'Fallback assumption'
                                  : 'Unknown'}
                              {preflightContractDetails.runtimeKnown === false ? ' • runtime unresolved' : ''}
                            </strong>
                          </div>
                        </div>
                        <div className="training-preflight-contract-card">
                          <div className="training-preflight-contract-card__title">Adapter Contract</div>
                          <div className="training-preflight-contract-card__row">
                            <span>Adapter</span>
                            <strong>{preflightContractDetails.adapterId}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Adapter Source</span>
                            <strong>{preflightContractDetails.adapterSource}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Task Profile</span>
                            <strong>
                              {preflightContractDetails.adapterTaskProfile}
                              {preflightContractDetails.adapterTaskProfileSource
                                ? ` (${preflightContractDetails.adapterTaskProfileSource})`
                                : ''}
                            </strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Resolved Modality</span>
                            <strong>{preflightContractDetails.adapterModality}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Declared Profiles</span>
                            <strong>
                              {preflightContractDetails.adapterDeclaredProfiles.length > 0
                                ? preflightContractDetails.adapterDeclaredProfiles.join(', ')
                                : 'n/a'}
                            </strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Preferred Tasks</span>
                            <strong>
                              {preflightContractDetails.adapterPreferredTasks.length > 0
                                ? preflightContractDetails.adapterPreferredTasks.join(', ')
                                : 'n/a'}
                            </strong>
                          </div>
                        </div>
                        <div className="training-preflight-contract-card">
                          <div className="training-preflight-contract-card__title">Model Compatibility</div>
                          <div className="training-preflight-contract-card__row">
                            <span>Model</span>
                            <strong>{preflightContractDetails.modelId}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Family</span>
                            <strong>{preflightContractDetails.modelFamily}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Architecture</span>
                            <strong>{preflightContractDetails.modelArchitecture}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Gate Status</span>
                            <strong>
                              {preflightContractDetails.modelGateOk === true
                                ? 'Pass'
                                : preflightContractDetails.modelGateOk === false
                                  ? 'Blocked'
                                  : 'Unknown'}
                            </strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Metadata Source</span>
                            <strong>{preflightContractDetails.modelIntrospectionSource}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Supported Architectures</span>
                            <strong>
                              {preflightContractDetails.modelSupportedArchitectures.length > 0
                                ? preflightContractDetails.modelSupportedArchitectures.join(', ')
                                : 'n/a'}
                            </strong>
                          </div>
                          {preflightContractDetails.modelGateErrors.length > 0 && (
                            <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--error">
                              {preflightContractDetails.modelGateErrors.join(' | ')}
                            </div>
                          )}
                          {preflightContractDetails.modelGateHints.length > 0 && (
                            <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--hint">
                              {preflightContractDetails.modelGateHints.join(' | ')}
                            </div>
                          )}
                        </div>
                        <div className="training-preflight-contract-card">
                          <div className="training-preflight-contract-card__title-row">
                            <div className="training-preflight-contract-card__title">Model + Dataset Modality</div>
                            <span
                              className={`badge ${preflightContractDetails.modelModalityStatus === 'blocked'
                                ? 'badge-error'
                                : preflightContractDetails.modelModalityStatus === 'warning'
                                  ? 'badge-warning'
                                  : preflightContractDetails.modelModalityStatus === 'pass'
                                    ? 'badge-success'
                                    : 'badge-info'
                                }`}
                            >
                              {preflightContractDetails.modelModalityStatus === 'blocked'
                                ? 'BLOCKED'
                                : preflightContractDetails.modelModalityStatus === 'warning'
                                  ? 'WARNING'
                                  : preflightContractDetails.modelModalityStatus === 'pass'
                                    ? 'PASS'
                                    : 'UNKNOWN'}
                            </span>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Architecture</span>
                            <strong>{preflightContractDetails.modelModalityArchitecture}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Adapter Modality</span>
                            <strong>{preflightContractDetails.modelModalityAdapterModality}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Supported Modalities</span>
                            <strong>{preflightContractDetails.modelModalitySupportedModalities.join(', ')}</strong>
                          </div>
                          {preflightContractDetails.modelModalityErrors.length > 0 && (
                            <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--error">
                              {preflightContractDetails.modelModalityErrors.join(' | ')}
                            </div>
                          )}
                          {preflightContractDetails.modelModalityWarnings.length > 0 && (
                            <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--warning">
                              {preflightContractDetails.modelModalityWarnings.join(' | ')}
                            </div>
                          )}
                          {preflightContractDetails.modelModalityHints.length > 0 && (
                            <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--hint">
                              {preflightContractDetails.modelModalityHints.join(' | ')}
                            </div>
                          )}
                        </div>
                        <div className="training-preflight-contract-card">
                          <div className="training-preflight-contract-card__title-row">
                            <div className="training-preflight-contract-card__title">Media Asset Contract</div>
                            <span
                              className={`badge ${preflightContractDetails.mediaContractStatus === 'blocked'
                                ? 'badge-error'
                                : preflightContractDetails.mediaContractStatus === 'warning'
                                  ? 'badge-warning'
                                  : preflightContractDetails.mediaContractStatus === 'pass'
                                    ? 'badge-success'
                                    : 'badge-info'
                                }`}
                            >
                              {preflightContractDetails.mediaContractStatus === 'blocked'
                                ? 'BLOCKED'
                                : preflightContractDetails.mediaContractStatus === 'warning'
                                  ? 'WARNING'
                                  : preflightContractDetails.mediaContractStatus === 'pass'
                                    ? 'PASS'
                                    : 'UNKNOWN'}
                            </span>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Expected Modality</span>
                            <strong>{preflightContractDetails.mediaContractExpectedModality}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Strict Require Media</span>
                            <strong>
                              {preflightContractDetails.mediaContractRequireMedia === true
                                ? 'enabled'
                                : preflightContractDetails.mediaContractRequireMedia === false
                                  ? 'disabled'
                                  : 'unknown'}
                            </strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Media Rows</span>
                            <strong>
                              {preflightContractDetails.mediaContractMediaRows}
                              {' / '}
                              {preflightContractDetails.mediaContractSampledRows}
                            </strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Image / Audio Rows</span>
                            <strong>
                              {preflightContractDetails.mediaContractImageRows}
                              {' / '}
                              {preflightContractDetails.mediaContractAudioRows}
                            </strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Mixed Rows</span>
                            <strong>{preflightContractDetails.mediaContractMixedRows}</strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Missing Local Refs</span>
                            <strong>
                              {preflightContractDetails.mediaContractMissingLocalImages}
                              {' image, '}
                              {preflightContractDetails.mediaContractMissingLocalAudios}
                              {' audio'}
                            </strong>
                          </div>
                          <div className="training-preflight-contract-card__row">
                            <span>Remote URL Refs</span>
                            <strong>
                              {preflightContractDetails.mediaContractRemoteImageRefs}
                              {' image, '}
                              {preflightContractDetails.mediaContractRemoteAudioRefs}
                              {' audio'}
                            </strong>
                          </div>
                          {preflightContractDetails.mediaContractErrors.length > 0 && (
                            <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--error">
                              {preflightContractDetails.mediaContractErrors.join(' | ')}
                            </div>
                          )}
                          {preflightContractDetails.mediaContractWarnings.length > 0 && (
                            <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--warning">
                              {preflightContractDetails.mediaContractWarnings.join(' | ')}
                            </div>
                          )}
                          {preflightContractDetails.mediaContractHints.length > 0 && (
                            <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--hint">
                              {preflightContractDetails.mediaContractHints.join(' | ')}
                            </div>
                          )}
                        </div>
                      </div>
                      <details className="training-preflight-panel__details">
                        <summary>Raw capability summary JSON</summary>
                        <pre className="resolved-defaults-panel__json">
                          {JSON.stringify(preflightContractDetails.rawCapabilitySummary || {}, null, 2)}
                        </pre>
                      </details>
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
                            className={`training-plan-card ${suggestion.preflight?.ok ? 'training-plan-card--ok' : 'training-plan-card--error'
                              } ${isRecommended ? 'training-plan-card--recommended' : ''}`}
                          >
                            <div className="training-plan-card__head">
                              <strong>{suggestion.title || suggestion.profile}</strong>
                              <div className="training-plan-card__badges">
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
              </div>
            </details>
            )}

            {showSetupPower && (
            <details className="training-collapsible">
              <summary>
                <span>Cloud Burst Planning</span>
                <small>Estimate remote GPU lease cost and generate one-click launch plans</small>
              </summary>
              <div className="training-collapsible__content">
                {cloudBurstPrefillStage && (
                  <div className="form-hint">
                    Prefilled from workflow template stage: {cloudBurstPrefillStage}
                  </div>
                )}
                <div className="form-group form-group--spaced">
                  <div className="form-inline-actions">
                    <select
                      className="input training-recipe-select"
                      value={cloudBurstProviderId}
                      onChange={(e) => setCloudBurstProviderId(e.target.value)}
                    >
                      <option value="">Select provider</option>
                      {cloudProviders.map((provider) => (
                        <option key={provider.provider_id} value={provider.provider_id}>
                          {provider.display_name || provider.provider_id}
                        </option>
                      ))}
                    </select>
                    <select
                      className="input training-recipe-select"
                      value={cloudBurstGpuSku}
                      onChange={(e) => setCloudBurstGpuSku(e.target.value)}
                    >
                      <option value="">Select GPU SKU</option>
                      {cloudGpuSkus.map((sku) => (
                        <option key={sku.gpu_sku} value={sku.gpu_sku}>
                          {sku.display_name || sku.gpu_sku}
                        </option>
                      ))}
                    </select>
                    <input
                      className="input"
                      type="number"
                      min={0.25}
                      max={72}
                      step={0.25}
                      value={cloudBurstDurationHours}
                      onChange={(e) => setCloudBurstDurationHours(e.target.value)}
                      placeholder="hours"
                    />
                    <label className="form-label form-label-inline">
                      <input
                        type="checkbox"
                        checked={cloudBurstSpot}
                        onChange={(e) => setCloudBurstSpot(e.target.checked)}
                      />
                      Spot
                    </label>
                    <button
                      className="btn btn-secondary"
                      onClick={() => void loadCloudBurstCatalog()}
                      disabled={cloudBurstLoadingCatalog}
                    >
                      {cloudBurstLoadingCatalog ? 'Refreshing...' : 'Refresh Catalog'}
                    </button>
                  </div>
                </div>
                {selectedCloudProvider && (
                  <div className="form-hint">
                    Provider capabilities:
                    {' '}
                    live execution {selectedCloudProvider.supports_live_execution ? 'yes' : 'no'}
                    {' • '}
                    managed cancel {selectedCloudProvider.supports_managed_cancel ? 'yes' : 'no'}
                    {' • '}
                    live logs {selectedCloudProvider.supports_live_logs ? 'yes' : 'no'}
                  </div>
                )}
                <div className="training-grid-2">
                  <div className="form-group">
                    <label className="form-label">Execution Mode</label>
                    <select
                      className="input"
                      value={cloudBurstExecutionMode}
                      onChange={(e) => setCloudBurstExecutionMode(e.target.value)}
                    >
                      <option value="auto">Auto (prefer live when available)</option>
                      <option value="live">Live provider job</option>
                      <option value="simulate">Simulated managed run</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Idempotency Key (optional)</label>
                    <input
                      className="input"
                      value={cloudBurstIdempotencyKey}
                      onChange={(e) => setCloudBurstIdempotencyKey(e.target.value)}
                      placeholder="same key => same run response"
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label className="form-label form-label-inline">
                    <input
                      type="checkbox"
                      checked={cloudBurstAllowFallbackToSimulation}
                      onChange={(e) => setCloudBurstAllowFallbackToSimulation(e.target.checked)}
                    />
                    Allow fallback to simulation when live submit is unavailable
                  </label>
                </div>
                <div className="training-grid-2">
                  <div className="form-group">
                    <label className="form-label">Storage (GB)</label>
                    <input
                      className="input"
                      type="number"
                      min={10}
                      max={2000}
                      value={cloudBurstStorageGb}
                      onChange={(e) => setCloudBurstStorageGb(e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Egress (GB)</label>
                    <input
                      className="input"
                      type="number"
                      min={0}
                      max={5000}
                      step={0.5}
                      value={cloudBurstEgressGb}
                      onChange={(e) => setCloudBurstEgressGb(e.target.value)}
                    />
                  </div>
                </div>
                <div className="training-grid-2">
                  <div className="form-group">
                    <label className="form-label">Region (optional)</label>
                    <input
                      className="input"
                      value={cloudBurstRegion}
                      onChange={(e) => setCloudBurstRegion(e.target.value)}
                      placeholder={
                        Array.isArray(selectedCloudProvider?.regions) && selectedCloudProvider.regions.length > 0
                          ? selectedCloudProvider.regions.join(', ')
                          : 'provider default'
                      }
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Experiment ID (optional)</label>
                    <input
                      className="input"
                      type="number"
                      min={1}
                      value={cloudBurstExperimentId}
                      onChange={(e) => setCloudBurstExperimentId(e.target.value)}
                      placeholder="latest or manual id"
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label className="form-label">Image (optional override)</label>
                  <input
                    className="input"
                    value={cloudBurstImage}
                    onChange={(e) => setCloudBurstImage(e.target.value)}
                    placeholder="ghcr.io/slm/platform-trainer:latest"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Startup Script (optional override)</label>
                  <textarea
                    className="input"
                    value={cloudBurstStartupScript}
                    onChange={(e) => setCloudBurstStartupScript(e.target.value)}
                    rows={3}
                    placeholder="bash /workspace/entrypoint.sh"
                  />
                </div>
                <div className="form-inline-actions">
                  <button
                    className="btn btn-secondary"
                    onClick={() => void requestCloudBurstQuote()}
                    disabled={cloudBurstLoadingQuote}
                  >
                    {cloudBurstLoadingQuote ? 'Estimating...' : 'Estimate Quote'}
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => void requestCloudBurstPlan()}
                    disabled={cloudBurstLoadingPlan}
                  >
                    {cloudBurstLoadingPlan ? 'Planning...' : 'Build Launch Plan'}
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => void submitCloudBurstManagedJob()}
                    disabled={cloudBurstSubmittingJob}
                  >
                    {cloudBurstSubmittingJob ? 'Submitting...' : 'Submit Managed Job'}
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => void loadCloudBurstJobs()}
                    disabled={cloudBurstLoadingRuns}
                  >
                    {cloudBurstLoadingRuns ? 'Refreshing...' : 'Refresh Jobs'}
                  </button>
                </div>
                {cloudBurstError && (
                  <div className="training-alert training-alert--error">
                    {cloudBurstError}
                  </div>
                )}
                {cloudBurstInfo && (
                  <div className="training-alert training-alert--warning">
                    {cloudBurstInfo}
                  </div>
                )}
                {cloudBurstQuote && (
                  <div className="resolved-defaults-panel">
                    <div className="resolved-defaults-panel__title">Cloud Burst Quote</div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Provider</span>
                      <strong>{cloudBurstQuote.provider_name || cloudBurstQuote.provider_id || '-'}</strong>
                    </div>
                    <div className="resolved-defaults-panel__kv">
                      <span>GPU</span>
                      <strong>{cloudBurstQuote.gpu_sku || '-'}</strong>
                    </div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Total Cost (USD)</span>
                      <strong>
                        {Number.isFinite(Number(cloudBurstQuote.cost_breakdown_usd?.total))
                          ? `$${Number(cloudBurstQuote.cost_breakdown_usd?.total).toFixed(2)}`
                          : 'n/a'}
                      </strong>
                    </div>
                    {Array.isArray(cloudBurstQuote.warnings) && cloudBurstQuote.warnings.length > 0 && (
                      <div className="training-alert training-alert--warning training-alert--tight">
                        {cloudBurstQuote.warnings.join(' | ')}
                      </div>
                    )}
                  </div>
                )}
                {cloudBurstPlan && (
                  <div className="resolved-defaults-panel">
                    <div className="resolved-defaults-panel__title">Cloud Burst Launch Plan</div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Launch ID</span>
                      <strong>{cloudBurstPlan.launch_id || '-'}</strong>
                    </div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Credentials Ready</span>
                      <strong>{cloudBurstPlan.credentials?.ready ? 'yes' : 'no'}</strong>
                    </div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Missing Credentials</span>
                      <strong>
                        {Array.isArray(cloudBurstPlan.credentials?.missing_keys) && cloudBurstPlan.credentials?.missing_keys.length > 0
                          ? cloudBurstPlan.credentials?.missing_keys.join(', ')
                          : 'none'}
                      </strong>
                    </div>
                    <div className="resolved-defaults-panel__grid">
                      <div>
                        <div className="resolved-defaults-panel__subtitle">Request Template</div>
                        <pre className="resolved-defaults-panel__json">
                          {JSON.stringify(cloudBurstPlan.request_template || {}, null, 2)}
                        </pre>
                      </div>
                      <div>
                        <div className="resolved-defaults-panel__subtitle">Full Plan</div>
                        <pre className="resolved-defaults-panel__json">
                          {JSON.stringify(cloudBurstPlan || {}, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </div>
                )}
                {cloudBurstRuns.length > 0 && (
                  <div className="resolved-defaults-panel">
                    <div className="resolved-defaults-panel__title">Managed Cloud Burst Jobs</div>
                    {cloudBurstRuns.slice(0, 8).map((run, idx) => {
                      const runId = String(run.run_id || '').trim();
                      const runStatus = String(run.status || 'unknown').trim().toLowerCase();
                      const selected = runId && runId === cloudBurstActiveRunId;
                      return (
                        <div key={`cloud-burst-run-${runId || idx}`} className="resolved-defaults-panel__kv">
                          <span>
                            {runId || 'unknown'} • {String(run.provider_id || '-')} • {String(run.gpu_sku || '-')}
                            {run.execution_mode_effective ? ` • ${run.execution_mode_effective}` : ''}
                            {run.experiment_id ? ` • exp ${run.experiment_id}` : ''}
                          </span>
                          <strong>
                            {runStatus}
                            {' '}
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => void loadCloudBurstJobStatus(runId)}
                              disabled={!runId}
                            >
                              {selected ? 'Viewing' : 'Inspect'}
                            </button>
                          </strong>
                        </div>
                      );
                    })}
                  </div>
                )}
                {cloudBurstActiveRun && (
                  <div className="resolved-defaults-panel">
                    <div className="resolved-defaults-panel__title">Managed Job Status</div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Run ID</span>
                      <strong>{cloudBurstActiveRun.run_id || '-'}</strong>
                    </div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Execution</span>
                      <strong>
                        {String(cloudBurstActiveRun.execution_mode_effective || 'unknown')}
                        {cloudBurstActiveRun.execution_mode_requested
                          ? ` (requested ${cloudBurstActiveRun.execution_mode_requested})`
                          : ''}
                      </strong>
                    </div>
                    {cloudBurstActiveRun.execution_mode_fallback_reason && (
                      <div className="training-alert training-alert--warning training-alert--tight">
                        {cloudBurstActiveRun.execution_mode_fallback_reason}
                      </div>
                    )}
                    <div className="resolved-defaults-panel__kv">
                      <span>Status</span>
                      <strong>
                        {String(cloudBurstActiveRun.status || 'unknown')}
                        {cloudBurstActiveRun.cancel_requested ? ' (cancel requested)' : ''}
                      </strong>
                    </div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Status Reason</span>
                      <strong>{String(cloudBurstActiveRun.status_reason || '-')}</strong>
                    </div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Provider Job</span>
                      <strong>
                        {String(cloudBurstActiveRun.provider_id || '-')}
                        {cloudBurstActiveRun.provider_job_id ? ` • ${cloudBurstActiveRun.provider_job_id}` : ''}
                      </strong>
                    </div>
                    <div className="resolved-defaults-panel__kv">
                      <span>Provider Status</span>
                      <strong>{String(cloudBurstActiveRun.provider_status_raw || '-')}</strong>
                    </div>
                    {cloudBurstActiveRun.current_run_cost !== undefined && (
                      <div className="resolved-defaults-panel__kv">
                        <span>Current Run Cost</span>
                        <strong className="text-success">
                          ${Number(cloudBurstActiveRun.current_run_cost).toFixed(4)}
                        </strong>
                      </div>
                    )}

                    {cloudBurstActiveRun.status_timeline && cloudBurstActiveRun.status_timeline.length > 0 && (
                      <div className="cloud-burst-timeline">
                        <div className="cloud-burst-timeline__title">Run Timeline</div>
                        <div className="cloud-burst-timeline__track">
                          {cloudBurstActiveRun.status_timeline.map((event, idx) => (
                            <div key={`timeline-${idx}`} className="cloud-burst-timeline__event">
                              <div className="cloud-burst-timeline__dot" data-status={event.status}></div>
                              <div className="cloud-burst-timeline__content">
                                <div className="cloud-burst-timeline__status">{event.status}</div>
                                <div className="cloud-burst-timeline__time">
                                  {new Date(event.at).toLocaleTimeString()}
                                </div>
                                <div className="cloud-burst-timeline__reason">{event.reason}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {!!cloudBurstActiveRun.idempotent_replay && (
                      <div className="training-alert training-alert--warning training-alert--tight">
                        This run was returned via idempotency replay.
                      </div>
                    )}
                    <div className="form-inline-actions">
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => void loadCloudBurstJobStatus(String(cloudBurstActiveRun.run_id || ''))}
                        disabled={cloudBurstLoadingRuns}
                      >
                        {cloudBurstLoadingRuns ? 'Refreshing...' : 'Refresh Status'}
                      </button>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => void syncCloudBurstManagedArtifacts(String(cloudBurstActiveRun.run_id || ''))}
                        disabled={cloudBurstSyncingArtifacts}
                      >
                        {cloudBurstSyncingArtifacts
                          ? 'Syncing...'
                          : cloudBurstSyncCursor
                            ? 'Sync Next Batch'
                            : 'Sync Artifacts'}
                      </button>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => void cancelCloudBurstManagedJob(String(cloudBurstActiveRun.run_id || ''))}
                        disabled={cloudBurstCancellingJob || cloudBurstActiveRun.can_cancel === false}
                      >
                        {cloudBurstCancellingJob ? 'Cancelling...' : 'Cancel Job'}
                      </button>
                    </div>
                    {cloudBurstActiveRun.artifacts?.last_sync_summary && (
                      <div className="training-alert training-alert--warning training-alert--tight">
                        Last sync: {String(cloudBurstActiveRun.artifacts?.last_sync_summary?.status || 'unknown')}
                        {' • '}
                        {Number(cloudBurstActiveRun.artifacts?.last_sync_summary?.copied_count || 0)}
                        {' / '}
                        {Number(cloudBurstActiveRun.artifacts?.last_sync_summary?.file_count || 0)}
                        {' files'}
                        {' • unchanged '}
                        {Number(cloudBurstActiveRun.artifacts?.last_sync_summary?.unchanged_count || 0)}
                        {' • remaining '}
                        {Number(cloudBurstActiveRun.artifacts?.last_sync_summary?.remaining_count || 0)}
                        {Array.isArray(cloudBurstActiveRun.artifacts?.last_sync_summary?.errors)
                          && cloudBurstActiveRun.artifacts?.last_sync_summary?.errors?.length
                          ? ` • errors: ${cloudBurstActiveRun.artifacts?.last_sync_summary?.errors?.slice(0, 2).join(' | ')}`
                          : ''}
                      </div>
                    )}
                    {String(cloudBurstActiveRun.artifacts?.last_sync_summary?.next_cursor || '').trim() && (
                      <div className="form-hint">
                        Next sync cursor available. Continue sync to fetch remaining artifacts.
                      </div>
                    )}
                    {cloudBurstMetrics.length > 0 && (
                      <div className="cloud-burst-metrics">
                        <div className="resolved-defaults-panel__subtitle">Live Metrics</div>
                        <div className="cloud-burst-metrics__kv">
                          <span>
                            step {Number.isFinite(Number(cloudBurstLatestMetric?.step))
                              ? Number(cloudBurstLatestMetric?.step)
                              : '-'}
                          </span>
                          <span>
                            train {Number.isFinite(Number(cloudBurstLatestMetric?.train_loss))
                              ? Number(cloudBurstLatestMetric?.train_loss).toFixed(4)
                              : '-'}
                          </span>
                          <span>
                            eval {Number.isFinite(Number(cloudBurstLatestMetric?.eval_loss))
                              ? Number(cloudBurstLatestMetric?.eval_loss).toFixed(4)
                              : '-'}
                          </span>
                        </div>
                        <svg
                          className="cloud-burst-metrics__chart"
                          viewBox="0 0 100 100"
                          preserveAspectRatio="none"
                          role="img"
                          aria-label="Cloud burst loss metrics trend"
                        >
                          <rect x="0" y="0" width="100" height="100" className="cloud-burst-metrics__bg" />
                          {cloudBurstMetricSeries.train && (
                            <polyline
                              fill="none"
                              points={cloudBurstMetricSeries.train}
                              className="cloud-burst-metrics__line cloud-burst-metrics__line--train"
                            />
                          )}
                          {cloudBurstMetricSeries.eval && (
                            <polyline
                              fill="none"
                              points={cloudBurstMetricSeries.eval}
                              className="cloud-burst-metrics__line cloud-burst-metrics__line--eval"
                            />
                          )}
                        </svg>
                      </div>
                    )}
                    {Array.isArray(cloudBurstActiveRun.logs_tail) && cloudBurstActiveRun.logs_tail.length > 0 && (
                      <div>
                        <div className="resolved-defaults-panel__subtitle">Job Logs (tail)</div>
                        <pre className="resolved-defaults-panel__json">
                          {cloudBurstActiveRun.logs_tail.join('\n')}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </details>
            )}

            {(showSetupConfig || showSetupPower) && (
            <details className="training-collapsible" open>
              <summary>
                <span>{showSetupPower ? 'Power Tools' : 'Core Configuration'}</span>
                <small>
                  {showSetupPower
                    ? 'Advanced tuning, model recommendation, and PEFT controls'
                    : 'Only required controls for a reliable run'}
                </small>
              </summary>
              <div className="training-collapsible__content">
                <div className={showSetupPower ? 'training-config-grid' : 'training-config-grid training-config-grid--essentials'}>
                  <div>
                    <h4 className="training-config-section-title">
                      {showSetupPower ? 'Model & Recommender' : 'Essentials'}
                    </h4>
                    {showSetupPower && (
                    <div className="training-model-wizard">
                      <div className="training-model-wizard__head">
                        <strong>Model Selection Wizard</strong>
                        <span>Pick hardware + goal, then apply a recommended base model.</span>
                      </div>
                      <div className="training-model-wizard__controls">
                        <div className="form-group">
                          <label className="form-label">Target Device</label>
                          <select
                            className="input"
                            value={wizardTargetDevice}
                            onChange={(e) => setWizardTargetDevice(e.target.value)}
                          >
                            <option value="mobile">Mobile</option>
                            <option value="laptop">Laptop</option>
                            <option value="server">Server</option>
                          </select>
                        </div>
                        <div className="form-group">
                          <label className="form-label">Primary Goal</label>
                          <select
                            className="input"
                            value={wizardPrimaryLanguage}
                            onChange={(e) => setWizardPrimaryLanguage(e.target.value)}
                          >
                            <option value="english">English</option>
                            <option value="multilingual">Multilingual</option>
                            <option value="coding">Coding</option>
                          </select>
                        </div>
                        <div className="form-group">
                          <label className="form-label">Available VRAM (GB)</label>
                          <input
                            className="input"
                            type="number"
                            min={1}
                            step={1}
                            value={wizardVramGb}
                            onChange={(e) => setWizardVramGb(e.target.value)}
                            placeholder="Optional"
                          />
                        </div>
                        <div className="form-group">
                          <label className="form-label">Task Profile</label>
                          <select
                            className="input"
                            value={wizardTaskProfile}
                            onChange={(e) => setWizardTaskProfile(e.target.value)}
                          >
                            {MODEL_WIZARD_TASK_PROFILES.map((profile) => (
                              <option key={profile} value={profile}>
                                {profile}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                      <div className="training-model-wizard__actions">
                        <button
                          className="btn btn-secondary"
                          onClick={() => void runModelWizard()}
                          disabled={wizardLoading}
                        >
                          {wizardLoading ? 'Finding Models...' : 'Recommend Models'}
                        </button>
                        <button
                          className="btn btn-secondary"
                          onClick={() => void runModelBenchmarkSweep()}
                          disabled={wizardLoading || benchmarkLoading}
                        >
                          {benchmarkLoading ? 'Running Benchmark...' : 'Run Benchmark Sweep'}
                        </button>
                        <span className="training-model-wizard__hint">
                          Uses lightweight heuristics from model size, VRAM fit, and task goal.
                        </span>
                      </div>
                      {wizardError && (
                        <div className="training-alert training-alert--warning training-alert--tight">
                          {wizardError}
                        </div>
                      )}
                      {Array.isArray(wizardResult?.warnings) && wizardResult.warnings.length > 0 && (
                        <div className="training-model-wizard__warnings">
                          {wizardResult.warnings.join(' | ')}
                        </div>
                      )}
                      {wizardResult?.adaptive_ranking?.enabled && (
                        <div className="training-model-wizard__warnings">
                          Adaptive ranking active ({wizardResult.adaptive_ranking.context_label || 'global'}): boosted{' '}
                          {wizardResult.adaptive_ranking.boosted_model_count || 0} model(s) from prior applies.
                        </div>
                      )}
                      {benchmarkError && (
                        <div className="training-alert training-alert--warning training-alert--tight">
                          {benchmarkError}
                        </div>
                      )}
                      {Array.isArray(benchmarkResult?.warnings) && benchmarkResult.warnings.length > 0 && (
                        <div className="training-model-wizard__warnings">
                          {benchmarkResult.warnings.join(' | ')}
                        </div>
                      )}
                      {Array.isArray(benchmarkResult?.matrix) && benchmarkResult.matrix.length > 0 && (
                        <div className="training-model-benchmark">
                          <div className="training-model-benchmark__head">
                            <strong>Benchmark Sweep ({benchmarkResult.benchmark_mode || 'real_sampled'})</strong>
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={applyBenchmarkWinner}
                            >
                              Apply Benchmark Winner
                            </button>
                          </div>
                          <div className="training-model-benchmark__meta">
                            Run {benchmarkResult.run_id || 'n/a'} • Sampled {benchmarkResult.sampled_row_count || 0} rows • Avg{' '}
                            {Number.isFinite(Number(benchmarkResult.sampled_avg_tokens))
                              ? `${Number(benchmarkResult.sampled_avg_tokens).toFixed(1)} tokens`
                              : 'n/a'}
                          </div>
                          <div className="training-model-benchmark__summary">
                            <span>Best quality: {benchmarkResult.tradeoff_summary?.best_quality_model_id || 'n/a'}</span>
                            <span>Best speed: {benchmarkResult.tradeoff_summary?.best_speed_model_id || 'n/a'}</span>
                            <span>Best balance: {benchmarkResult.tradeoff_summary?.best_balance_model_id || 'n/a'}</span>
                          </div>
                          <div className="training-model-benchmark__rows">
                            {benchmarkResult.matrix.map((row, index) => (
                              <div className="training-model-benchmark__row" key={`${benchmarkResult.run_id || 'run'}-${row.model_id || index}`}>
                                <div className="training-model-benchmark__row-head">
                                  <strong>#{row.rank || index + 1} {row.model_id || 'unknown'}</strong>
                                  <span>{row.benchmark_mode || 'sampled_heuristic'}</span>
                                </div>
                                <div className="training-model-benchmark__row-metrics">
                                  <span>Quality {Number.isFinite(Number(row.estimated_accuracy_percent)) ? `${Number(row.estimated_accuracy_percent).toFixed(1)}%` : 'n/a'}</span>
                                  <span>Latency {Number.isFinite(Number(row.estimated_latency_ms)) ? `${Number(row.estimated_latency_ms).toFixed(1)} ms` : 'n/a'}</span>
                                  <span>Throughput {Number.isFinite(Number(row.estimated_throughput_tps)) ? `${Number(row.estimated_throughput_tps).toFixed(1)} t/s` : 'n/a'}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {benchmarkHistory.length > 0 && (
                        <div className="training-model-benchmark__history">
                          <strong>Recent Benchmark Runs</strong>
                          {benchmarkHistory.slice(0, 3).map((run, idx) => (
                            <div className="training-model-benchmark__history-row" key={`${run.run_id || 'history'}-${idx}`}>
                              <span>{run.run_id || 'run'}</span>
                              <span>{run.benchmark_mode || 'real_sampled'}</span>
                              <span>
                                Winner:{' '}
                                {run.tradeoff_summary?.best_balance_model_id
                                  || run.matrix?.[0]?.model_id
                                  || 'n/a'}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                      {Array.isArray(wizardResult?.recommendations) && wizardResult.recommendations.length > 0 && (
                        <div className="training-model-wizard__results">
                          {wizardResult.recommendations.map((item, idx) => (
                            <div className="training-model-wizard__card" key={item.model_id}>
                              <div className="training-model-wizard__card-head">
                                <strong>{item.model_id}</strong>
                                {Number.isFinite(Number(item.match_score)) && (
                                  <span className="badge badge-info">score {Number(item.match_score).toFixed(2)}</span>
                                )}
                              </div>
                              <div className="training-model-wizard__card-meta">
                                {item.params_b ? `${item.params_b}B params` : 'unknown size'} • min VRAM{' '}
                                {Number.isFinite(Number(item.estimated_min_vram_gb))
                                  ? `${Number(item.estimated_min_vram_gb)} GB`
                                  : 'n/a'}
                              </div>
                              <div className="training-model-wizard__card-meta">
                                {item.architecture || 'unknown'} • ctx{' '}
                                {Number.isFinite(Number(item.context_length))
                                  ? Number(item.context_length)
                                  : 'n/a'}
                                {item.license ? ` • ${item.license}` : ''}
                              </div>
                              {(Number.isFinite(Number(item.introspection_estimated_min_vram_gb))
                                || Number.isFinite(Number(item.introspection_estimated_ideal_vram_gb))) && (
                                  <div className="training-model-wizard__card-meta">
                                    Introspection VRAM:{' '}
                                    {Number.isFinite(Number(item.introspection_estimated_min_vram_gb))
                                      ? `${Number(item.introspection_estimated_min_vram_gb)} GB min`
                                      : 'n/a'}
                                    {' / '}
                                    {Number.isFinite(Number(item.introspection_estimated_ideal_vram_gb))
                                      ? `${Number(item.introspection_estimated_ideal_vram_gb)} GB ideal`
                                      : 'n/a'}
                                  </div>
                                )}
                              {item.metadata_source && (
                                <div className="training-model-wizard__meta-source">
                                  Metadata: {item.metadata_source}
                                </div>
                              )}
                              {Array.isArray(item.match_reasons) && item.match_reasons.length > 0 && (
                                <div className="training-model-wizard__reasons">
                                  {item.match_reasons.slice(0, 3).map((reason, idx) => (
                                    <div key={`${item.model_id}-reason-${idx}`}>{reason}</div>
                                  ))}
                                </div>
                              )}
                              <button
                                className="btn btn-secondary btn-sm"
                                onClick={() => applyModelWizardRecommendation(item, idx)}
                              >
                                Apply Model + Defaults
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    )}
                    {showSetupConfig && (
                      <>
                    <div className="form-group">
                      <label className="form-label">Base Model</label>
                      <input className="input" value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />
                      <div className="form-inline-actions">
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => void introspectBaseModel()}
                          disabled={baseModelIntrospectionLoading}
                        >
                          {baseModelIntrospectionLoading ? 'Inspecting...' : 'Introspect Model'}
                        </button>
                        <span className="form-hint">
                          Reads local/HF config metadata: architecture, context, license, memory hints.
                        </span>
                      </div>
                      {baseModelIntrospectionError && (
                        <div className="form-hint form-hint-warning">
                          {baseModelIntrospectionError}
                        </div>
                      )}
                      {baseModelIntrospection && (
                        <div className="training-model-introspection">
                          <div className="training-model-introspection__head">
                            <strong>Model Introspection</strong>
                            <span>{baseModelIntrospection.source || 'none'}</span>
                          </div>
                          <div className="training-model-introspection__grid">
                            <div className="training-model-introspection__row">
                              <span>Model ID</span>
                              <strong>{baseModelIntrospection.model_id || baseModel}</strong>
                            </div>
                            <div className="training-model-introspection__row">
                              <span>Architecture</span>
                              <strong>{baseModelIntrospection.architecture || 'unknown'}</strong>
                            </div>
                            <div className="training-model-introspection__row">
                              <span>Model Type</span>
                              <strong>{baseModelIntrospection.model_type || 'unknown'}</strong>
                            </div>
                            <div className="training-model-introspection__row">
                              <span>Context Length</span>
                              <strong>
                                {Number.isFinite(Number(baseModelIntrospection.context_length))
                                  ? Number(baseModelIntrospection.context_length)
                                  : 'n/a'}
                              </strong>
                            </div>
                            <div className="training-model-introspection__row">
                              <span>License</span>
                              <strong>{baseModelIntrospection.license || 'unknown'}</strong>
                            </div>
                            <div className="training-model-introspection__row">
                              <span>Params (est)</span>
                              <strong>
                                {Number.isFinite(Number(baseModelIntrospection.params_estimate_b))
                                  ? `${Number(baseModelIntrospection.params_estimate_b).toFixed(2)}B`
                                  : 'n/a'}
                              </strong>
                            </div>
                            <div className="training-model-introspection__row">
                              <span>VRAM (min/ideal)</span>
                              <strong>
                                {Number.isFinite(Number(baseModelIntrospection.memory_profile?.estimated_min_vram_gb))
                                  ? `${Number(baseModelIntrospection.memory_profile?.estimated_min_vram_gb)} GB`
                                  : 'n/a'}
                                {' / '}
                                {Number.isFinite(Number(baseModelIntrospection.memory_profile?.estimated_ideal_vram_gb))
                                  ? `${Number(baseModelIntrospection.memory_profile?.estimated_ideal_vram_gb)} GB`
                                  : 'n/a'}
                              </strong>
                            </div>
                          </div>
                          {Array.isArray(baseModelIntrospection.warnings) && baseModelIntrospection.warnings.length > 0 && (
                            <div className="training-model-introspection__warnings">
                              {baseModelIntrospection.warnings.join(' | ')}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    <div className="training-grid-2">
                      <div className="form-group">
                        <label className="form-label">Training Mode</label>
                        <select
                          className="input"
                          value={trainingMode}
                          onChange={(e) => {
                            const nextMode = e.target.value;
                            setTrainingMode(nextMode);
                            setTouchedConfig((prev) => ({ ...prev, training_mode: true }));
                            if (nextMode === 'dpo' || nextMode === 'orpo') {
                              setTaskType('causal_lm');
                              setTouchedConfig((prev) => ({ ...prev, task_type: true }));
                            }
                          }}
                        >
                          <option value="sft">SFT</option>
                          <option value="domain_pretrain">Domain Pretrain</option>
                          <option value="dpo">DPO</option>
                          <option value="orpo">ORPO</option>
                        </select>
                      </div>
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
                          <div className="form-hint form-hint-warning">
                            {runtimeCatalogError}
                          </div>
                        )}
                        {selectedRuntimeSpec && (
                          <div className="form-hint">
                            Modalities: {selectedRuntimeModalities.length > 0 ? selectedRuntimeModalities.join(', ') : 'text'}
                            {selectedRuntimeModalitiesDeclared === false ? ' (assumed default)' : ''}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="training-grid-2">
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
                    <div className="training-grid-2">
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
                      </>
                    )}
                    {showSetupPower && (
                      <>
                        <div className="training-grid-2">
                          <div className="form-group">
                            <label className="form-label">Task Type</label>
                            <select
                              className="input"
                              value={taskType}
                              onChange={(e) => {
                                setTaskType(e.target.value);
                                setTouchedConfig((prev) => ({ ...prev, task_type: true }));
                              }}
                              disabled={isAlignmentMode}
                            >
                              <option value="causal_lm">Causal LM</option>
                              <option value="seq2seq">Seq2Seq</option>
                              <option value="classification">Classification</option>
                            </select>
                            {isAlignmentMode && (
                              <div className="form-hint">
                                DPO/ORPO currently run on causal LM preference pairs.
                              </div>
                            )}
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
                        <div className="training-grid-2">
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
                        <div className="training-grid-2">
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
                      </>
                    )}
                  </div>

                  {showSetupPower && (
                  <div>
                    <h4 className="training-config-section-title">Advanced & PEFT</h4>
                    <div className="form-group training-toggle-row">
                      <input
                        type="checkbox"
                        checked={useLora}
                        onChange={(e) => {
                          setUseLora(e.target.checked);
                          setTouchedConfig((prev) => ({ ...prev, use_lora: true }));
                        }}
                      />
                      <label className="form-label form-label-inline-tight">Enable LoRA</label>
                    </div>
                    {useLora && (
                      <div className="training-lora-box">
                        <div className="training-grid-2">
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
                    <div className="form-group training-toggle-row">
                      <input
                        type="checkbox"
                        checked={gradientCheckpointing}
                        onChange={(e) => {
                          setGradientCheckpointing(e.target.checked);
                          setTouchedConfig((prev) => ({ ...prev, gradient_checkpointing: true }));
                        }}
                      />
                      <label className="form-label form-label-inline-tight">Use Gradient Checkpointing</label>
                    </div>
                    <div className="form-group training-toggle-row">
                      <input
                        type="checkbox"
                        checked={sequencePacking}
                        onChange={(e) => {
                          setSequencePacking(e.target.checked);
                          setTouchedConfig((prev) => ({ ...prev, sequence_packing: true }));
                        }}
                      />
                      <label className="form-label form-label-inline-tight">Enable Sequence Packing</label>
                    </div>
                    <div className="form-group training-toggle-row">
                      <input
                        type="checkbox"
                        aria-label="Require local media assets for multimodal batches"
                        checked={multimodalRequireMedia}
                        onChange={(e) => {
                          setMultimodalRequireMedia(e.target.checked);
                          setTouchedConfig((prev) => ({ ...prev, multimodal_require_media: true }));
                        }}
                      />
                      <label className="form-label form-label-inline-tight">Require Local Media Assets (Strict Multimodal)</label>
                    </div>
                    <div className="form-hint">
                      Blocks text-fallback for image/audio rows and fails on missing/remote media refs.
                      Preflight Plan can auto-relax this flag when strict mode would block launch.
                    </div>
                    <div className="form-group training-toggle-row">
                      <input
                        type="checkbox"
                        checked={flashAttention}
                        onChange={(e) => {
                          setFlashAttention(e.target.checked);
                          setTouchedConfig((prev) => ({ ...prev, flash_attention: true }));
                        }}
                      />
                      <label className="form-label form-label-inline-tight">Enable Flash Attention</label>
                    </div>
                    <div className="form-group training-toggle-row">
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
                      <label className="form-label form-label-inline-tight">Use BF16</label>
                    </div>
                    <div className="form-group training-toggle-row">
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
                      <label className="form-label form-label-inline-tight">Use FP16</label>
                    </div>
                    <div className="form-group training-toggle-row">
                      <input
                        type="checkbox"
                        checked={autoOomRetry}
                        onChange={(e) => {
                          setAutoOomRetry(e.target.checked);
                          setTouchedConfig((prev) => ({ ...prev, auto_oom_retry: true }));
                        }}
                      />
                      <label className="form-label form-label-inline-tight">Auto OOM Retry Planner</label>
                    </div>
                    <div className="training-grid-2">
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
                    {isAlignmentMode && (
                      <div className="training-lora-box">
                        <h5 className="training-config-section-title" style={{ marginTop: 0 }}>
                          Alignment Dataset Controls
                        </h5>
                        <div className="form-group training-toggle-row">
                          <input
                            type="checkbox"
                            checked={alignmentAutoFilter}
                            onChange={(e) => {
                              setAlignmentAutoFilter(e.target.checked);
                              setTouchedConfig((prev) => ({ ...prev, alignment_auto_filter: true }));
                            }}
                          />
                          <label className="form-label form-label-inline-tight">Auto filter preference pairs before run</label>
                        </div>
                        <div className="training-grid-2">
                          <div className="form-group">
                            <label className="form-label">Alignment Beta</label>
                            <input
                              className="input"
                              value={alignmentBeta}
                              onChange={(e) => {
                                setAlignmentBeta(e.target.value);
                                setTouchedConfig((prev) => ({ ...prev, alignment_beta: true }));
                              }}
                              placeholder="0.1"
                            />
                          </div>
                          <div className="form-group">
                            <label className="form-label">Alignment Quality Threshold</label>
                            <input
                              className="input"
                              value={alignmentQualityThreshold}
                              onChange={(e) => {
                                setAlignmentQualityThreshold(e.target.value);
                                setTouchedConfig((prev) => ({ ...prev, alignment_quality_threshold: true }));
                              }}
                              placeholder="3.0"
                            />
                          </div>
                        </div>
                        <div className="training-grid-2">
                          <div className="form-group">
                            <label className="form-label">Alignment Max Prompt Length</label>
                            <input
                              className="input"
                              value={alignmentMaxPromptLength}
                              onChange={(e) => {
                                setAlignmentMaxPromptLength(e.target.value);
                                setTouchedConfig((prev) => ({ ...prev, alignment_max_prompt_length: true }));
                              }}
                              placeholder="1024"
                            />
                          </div>
                          <div className="form-group">
                            <label className="form-label">Alignment Max Length</label>
                            <input
                              className="input"
                              value={alignmentMaxLength}
                              onChange={(e) => {
                                setAlignmentMaxLength(e.target.value);
                                setTouchedConfig((prev) => ({ ...prev, alignment_max_length: true }));
                              }}
                              placeholder="2048"
                            />
                          </div>
                        </div>
                        <div className="training-grid-2">
                          <div className="form-group">
                            <label className="form-label">Alignment Min Keep Ratio</label>
                            <input
                              className="input"
                              value={alignmentMinKeepRatio}
                              onChange={(e) => {
                                setAlignmentMinKeepRatio(e.target.value);
                                setTouchedConfig((prev) => ({ ...prev, alignment_min_keep_ratio: true }));
                              }}
                              placeholder="0.4"
                            />
                          </div>
                        </div>
                        <div className="form-group">
                          <label className="form-label">Alignment Dataset Path (Optional)</label>
                          <input
                            className="input"
                            value={alignmentDatasetPath}
                            onChange={(e) => {
                              setAlignmentDatasetPath(e.target.value);
                              setTouchedConfig((prev) => ({ ...prev, alignment_dataset_path: true }));
                            }}
                            placeholder="prepared/alignment/train.filtered.jsonl"
                          />
                          <div className="form-hint">
                            Leave empty to use prepared train split. Path is project-relative under data/projects/&lt;id&gt;.
                          </div>
                        </div>
                        <div className="form-group training-toggle-row">
                          <input
                            type="checkbox"
                            checked={alignmentIncludePlaygroundFeedback}
                            onChange={(e) => {
                              setAlignmentIncludePlaygroundFeedback(e.target.checked);
                              setTouchedConfig((prev) => ({
                                ...prev,
                                alignment_include_playground_feedback: true,
                              }));
                            }}
                          />
                          <label className="form-label form-label-inline-tight">
                            Merge playground downvote pairs into alignment train dataset
                          </label>
                        </div>
                        <div className="training-grid-2">
                          <div className="form-group">
                            <label className="form-label">Playground Feedback Max Pairs</label>
                            <input
                              className="input"
                              value={alignmentPlaygroundMaxPairs}
                              onChange={(e) => {
                                setAlignmentPlaygroundMaxPairs(e.target.value);
                                setTouchedConfig((prev) => ({
                                  ...prev,
                                  alignment_playground_max_pairs: true,
                                }));
                              }}
                              placeholder="5000"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                    <div className="training-lora-box">
                      <h5 className="training-config-section-title" style={{ marginTop: 0 }}>
                        Observability Telemetry
                      </h5>
                      <div className="form-group training-toggle-row">
                        <input
                          type="checkbox"
                          checked={observabilityEnabled}
                          onChange={(e) => {
                            setObservabilityEnabled(e.target.checked);
                            setTouchedConfig((prev) => ({ ...prev, observability_enabled: true }));
                          }}
                        />
                        <label className="form-label form-label-inline-tight">Enable gradient/attention telemetry emission</label>
                      </div>
                      <div className="form-group training-toggle-row">
                        <input
                          type="checkbox"
                          checked={observabilityProbeAttention}
                          onChange={(e) => {
                            setObservabilityProbeAttention(e.target.checked);
                            setTouchedConfig((prev) => ({ ...prev, observability_probe_attention: true }));
                          }}
                        />
                        <label className="form-label form-label-inline-tight">Run attention probe on logging steps</label>
                      </div>
                      <div className="training-grid-2">
                        <div className="form-group">
                          <label className="form-label">Observability Log Steps</label>
                          <input
                            className="input"
                            type="number"
                            min={1}
                            value={observabilityLogSteps}
                            onChange={(e) => {
                              setObservabilityLogSteps(Math.max(1, Number(e.target.value) || 1));
                              setTouchedConfig((prev) => ({ ...prev, observability_log_steps: true }));
                            }}
                          />
                        </div>
                        <div className="form-group">
                          <label className="form-label">Max Gradient Layers</label>
                          <input
                            className="input"
                            type="number"
                            min={1}
                            value={observabilityMaxLayers}
                            onChange={(e) => {
                              setObservabilityMaxLayers(Math.max(1, Number(e.target.value) || 1));
                              setTouchedConfig((prev) => ({ ...prev, observability_max_layers: true }));
                            }}
                          />
                        </div>
                      </div>
                      <div className="training-grid-2">
                        <div className="form-group">
                          <label className="form-label">Attention Top-K Tokens</label>
                          <input
                            className="input"
                            type="number"
                            min={1}
                            value={observabilityProbeTopK}
                            onChange={(e) => {
                              setObservabilityProbeTopK(Math.max(1, Number(e.target.value) || 1));
                              setTouchedConfig((prev) => ({ ...prev, observability_probe_top_k: true }));
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                  )}
                </div>
              </div>
            </details>
            )}

            {showSetupReview && (
              <div className="training-review-panel">
                <div className="training-review-grid">
                  <div className="training-review-item">
                    <span>Name</span>
                    <strong>{name || 'Untitled experiment'}</strong>
                  </div>
                  <div className="training-review-item">
                    <span>Base Model</span>
                    <strong>{baseModel || 'not set'}</strong>
                  </div>
                  <div className="training-review-item">
                    <span>Runtime</span>
                    <strong>{trainingRuntimeId}</strong>
                  </div>
                  <div className="training-review-item">
                    <span>Training</span>
                    <strong>{epochs} epochs · batch {batchSize} · lr {lr}</strong>
                  </div>
                </div>
                {modelSelectionSummary.hasAny && (
                  <div className="training-review-selection-panel">
                    <div className="training-review-selection-panel__head">
                      <strong>Preflight Model Selection Snapshot</strong>
                      <span>{modelSelectionSummary.winnerAlignmentLabel}</span>
                    </div>
                    <div className="training-review-selection-grid">
                      <div className="training-review-item">
                        <span>Recommendation Winner</span>
                        <strong>{modelSelectionSummary.recommendationWinnerId || 'not available'}</strong>
                      </div>
                      <div className="training-review-item">
                        <span>Benchmark Winner</span>
                        <strong>{modelSelectionSummary.benchmarkWinnerId || 'not available'}</strong>
                      </div>
                    </div>
                    <div className="training-review-selection-panel__actions">
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={applyReviewConsensusWinner}
                        disabled={!modelSelectionSummary.recommendationWinnerId && !modelSelectionSummary.benchmarkWinnerId}
                      >
                        Use Consensus Winner
                      </button>
                      {reviewSelectionActionNote && (
                        <span className="training-review-selection-panel__note">
                          {reviewSelectionActionNote}
                        </span>
                      )}
                    </div>
                  </div>
                )}
                {!isSetupAdvancedMode && (
                  <div className="training-essentials-tools">
                    <button
                      className="btn btn-secondary"
                      onClick={() => void runPreflightPreview()}
                      disabled={preflightPreviewLoading}
                    >
                      {preflightPreviewLoading ? 'Checking...' : 'Run Quick Preflight'}
                    </button>
                    {preflightPreview && (
                      <span className={`training-essentials-preflight ${preflightPreview.ok ? 'ok' : 'blocked'}`}>
                        {preflightPreview.ok
                          ? 'Preflight passed'
                          : `${preflightPreview.errors.length} blocking issue(s)`}
                      </span>
                    )}
                    {preflightPreview && (
                      <div className={`training-essentials-model-gate training-essentials-model-gate--${essentialsModelGateSummary.statusClass}`}>
                        <strong>Model Gate: {essentialsModelGateSummary.statusLabel}</strong>
                        <span>
                          {essentialsModelGateSummary.modelId} • {essentialsModelGateSummary.architecture} • source {essentialsModelGateSummary.source}
                        </span>
                        {essentialsModelGateSummary.topIssue && (
                          <span>{essentialsModelGateSummary.topIssue}</span>
                        )}
                        {essentialsModelGateSummary.supportedArchitectures && (
                          <span>Supported: {essentialsModelGateSummary.supportedArchitectures}</span>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {!showSetupReview && (
              <div className="training-create-shell__actions training-create-shell__actions--step">
                <button
                  className="btn btn-secondary"
                  onClick={goToPreviousSetupTab}
                  disabled={!canSetupGoBack}
                >
                  Back
                </button>
                <button
                  className="btn btn-primary"
                  onClick={goToNextSetupTab}
                  disabled={!canSetupGoNext}
                >
                  Continue
                </button>
                {!forceCreateVisible && (
                  <button
                    className="btn btn-secondary"
                    onClick={() => {
                      setShowCreate(false);
                      setSetupTab('basics');
                      setPreflightPreview(null);
                      setPreflightPreviewError('');
                      setPreflightPlan(null);
                      setPreflightPlanError('');
                    }}
                  >
                    Close
                  </button>
                )}
              </div>
            )}

            {showSetupReview && (
              <div className="training-create-shell__actions">
                <button className="btn btn-secondary" onClick={goToPreviousSetupTab}>
                  Back
                </button>
                <button className="btn btn-primary" onClick={handleCreate}>Create Experiment</button>
                {!forceCreateVisible && (
                  <button
                    className="btn btn-secondary"
                    onClick={() => {
                      setShowCreate(false);
                      setSetupTab('basics');
                      setPreflightPreview(null);
                      setPreflightPreviewError('');
                      setPreflightPlan(null);
                      setPreflightPlanError('');
                    }}
                  >
                    Cancel
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {trainingError && (
          <div className="training-alert training-alert--error">
            {trainingError}
          </div>
        )}
        {trainingWarnings.length > 0 && (
          <div className="training-alert training-alert--warning">
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

        {canViewRuns && (workspaceView === 'runs' || (!showWorkspaceTabs && workspaceView !== 'setup')) && (
          experiments.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🔬</div>
              <div className="empty-state-title">No experiments</div>
              <div className="empty-state-text">
                {hideCreateControls
                  ? 'No runs yet. Create one from the Training Config menu item.'
                  : 'Create a training experiment to start fine-tuning.'}
              </div>
            </div>
          ) : (
            <div className="training-experiment-list">
              {selectedForCompare.length > 1 && (
                <div className="training-compare-bar">
                  <button className="btn btn-primary" onClick={() => setShowCompare(true)}>
                    Compare Selected ({selectedForCompare.length})
                  </button>
                </div>
              )}
              {experiments.map((exp) => (
                <div
                  key={exp.id}
                  className="training-experiment-item"
                >
                  <div className="training-experiment-main">
                    <input
                      type="checkbox"
                      checked={selectedForCompare.includes(exp.id)}
                      onChange={() => toggleCompareSelection(exp.id)}
                      className="training-checkbox"
                    />
                    <div>
                      <div className="training-experiment-name">{exp.name}</div>
                      <div className="training-experiment-meta">
                        {exp.base_model} • {exp.training_mode}
                      </div>
                      {exp.domain_pack_applied && (
                        <div className="training-experiment-submeta">
                          Pack: {exp.domain_pack_applied}
                          {exp.domain_pack_source ? ` (${exp.domain_pack_source})` : ''}
                        </div>
                      )}
                      {exp.domain_profile_applied && (
                        <div className="training-experiment-submeta">
                          Profile: {exp.domain_profile_applied}
                          {exp.domain_profile_source ? ` (${exp.domain_profile_source})` : ''}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="training-experiment-actions">
                    <span className={`badge ${statusColor(exp.status)}`}>{exp.status}</span>
                    {exp.status === 'pending' && (
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={() => setPendingStartExperiment(exp)}
                      >
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
          )
        )}
      </div>

      {onNextStep && !hideStepFooter && (
        <StepFooter
          currentStep="Training"
          nextStep="Compression"
          nextStepIcon="🗜️"
          isComplete={experiments.some((e) => e.status === 'completed')}
          hint="Start and complete an experiment to proceed"
          onNext={onNextStep}
        />
      )}

      {showHardwareModal && (
        <HardwareRecommenderModal
          onClose={() => setShowHardwareModal(false)}
          onApply={handleApplyHardwareRecommendation}
        />
      )}

      {pendingStartExperiment && (
        <PreRunConfirmModal
          projectId={projectId}
          config={
            pendingStartExperiment.config && typeof pendingStartExperiment.config === 'object'
              ? (pendingStartExperiment.config as Record<string, unknown>)
              : {}
          }
          baseModel={pendingStartExperiment.base_model}
          targetProfileId={
            (pendingStartExperiment.config as Record<string, unknown> | null | undefined)?.[
              'target_profile_id'
            ] as string | undefined
          }
          onCancel={() => setPendingStartExperiment(null)}
          onConfirm={() => {
            const expId = pendingStartExperiment.id;
            setPendingStartExperiment(null);
            void handleStart(expId);
          }}
          confirmLabel="Launch"
        />
      )}
    </div>
  );
}
