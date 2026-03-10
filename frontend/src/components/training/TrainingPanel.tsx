import { useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { TerminalConsole } from '../shared/TerminalConsole';
import ExperimentCompare from './ExperimentCompare';
import { buildWsUrl } from '../../utils/ws';
import { loadWorkflowStagePrefill } from '../../utils/workflowGraphPrefill';
import './TrainingPanel.css';

interface TrainingPanelProps {
  projectId: number;
  onNextStep?: () => void;
  title?: string;
  hideStepFooter?: boolean;
  hideCreateControls?: boolean;
  hideExperimentList?: boolean;
  forceCreateVisible?: boolean;
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
  supported_languages?: string[];
  strengths?: string[];
  caveats?: string[];
  match_reasons?: string[];
  match_score?: number;
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

interface CloudBurstProvider {
  provider_id: string;
  display_name?: string;
  description?: string;
  supports_spot?: boolean;
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
  | 'alignment_auto_filter'
  | 'alignment_quality_threshold'
  | 'alignment_beta'
  | 'alignment_max_prompt_length'
  | 'alignment_max_length'
  | 'alignment_min_keep_ratio'
  | 'alignment_dataset_path';

type TrainingWorkspaceView = 'overview' | 'setup' | 'runs';

export default function TrainingPanel({
  projectId,
  onNextStep,
  title = 'Training Experiments',
  hideStepFooter = false,
  hideCreateControls = false,
  hideExperimentList = false,
  forceCreateVisible = false,
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
  const [alignmentAutoFilter, setAlignmentAutoFilter] = useState(false);
  const [alignmentQualityThreshold, setAlignmentQualityThreshold] = useState('3.0');
  const [alignmentBeta, setAlignmentBeta] = useState('0.1');
  const [alignmentMaxPromptLength, setAlignmentMaxPromptLength] = useState('1024');
  const [alignmentMaxLength, setAlignmentMaxLength] = useState('2048');
  const [alignmentMinKeepRatio, setAlignmentMinKeepRatio] = useState('0.4');
  const [alignmentDatasetPath, setAlignmentDatasetPath] = useState('');
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
    alignment_auto_filter: false,
    alignment_quality_threshold: false,
    alignment_beta: false,
    alignment_max_prompt_length: false,
    alignment_max_length: false,
    alignment_min_keep_ratio: false,
    alignment_dataset_path: false,
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
  const [trainingRecipes, setTrainingRecipes] = useState<TrainingRecipe[]>([]);
  const [selectedRecipeId, setSelectedRecipeId] = useState('');
  const [recipeResolveLoading, setRecipeResolveLoading] = useState(false);
  const [recipeResolveError, setRecipeResolveError] = useState('');
  const [workspaceView, setWorkspaceView] = useState<TrainingWorkspaceView>('overview');
  const [wizardTargetDevice, setWizardTargetDevice] = useState('laptop');
  const [wizardPrimaryLanguage, setWizardPrimaryLanguage] = useState('english');
  const [wizardVramGb, setWizardVramGb] = useState('8');
  const [wizardTaskProfile, setWizardTaskProfile] = useState('auto');
  const [wizardLoading, setWizardLoading] = useState(false);
  const [wizardError, setWizardError] = useState('');
  const [wizardResult, setWizardResult] = useState<ModelWizardResponse | null>(null);
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
  const [cloudBurstLoadingCatalog, setCloudBurstLoadingCatalog] = useState(false);
  const [cloudBurstLoadingQuote, setCloudBurstLoadingQuote] = useState(false);
  const [cloudBurstLoadingPlan, setCloudBurstLoadingPlan] = useState(false);
  const [cloudBurstError, setCloudBurstError] = useState('');
  const [cloudBurstQuote, setCloudBurstQuote] = useState<CloudBurstQuoteResponse | null>(null);
  const [cloudBurstPlan, setCloudBurstPlan] = useState<CloudBurstLaunchPlanResponse | null>(null);
  const [cloudBurstPrefillStage, setCloudBurstPrefillStage] = useState('');

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

  const buildTrainingConfigPayload = (): Record<string, unknown> => {
    const learningRate = Number.parseFloat(lr);
    const retryShrink = Number.parseFloat(oomRetrySeqShrink);
    const alignmentThreshold = Number.parseFloat(alignmentQualityThreshold);
    const alignmentBetaValue = Number.parseFloat(alignmentBeta);
    const alignmentPromptLengthValue = Number.parseInt(alignmentMaxPromptLength, 10);
    const alignmentMaxLengthValue = Number.parseInt(alignmentMaxLength, 10);
    const alignmentKeepRatio = Number.parseFloat(alignmentMinKeepRatio);
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
    setAlignmentAutoFilter(parseBoolean(config.alignment_auto_filter, alignmentAutoFilter));
    setAlignmentQualityThreshold(String(config.alignment_quality_threshold ?? alignmentQualityThreshold));
    setAlignmentBeta(String(config.alignment_beta ?? alignmentBeta));
    setAlignmentMaxPromptLength(String(config.alignment_max_prompt_length ?? alignmentMaxPromptLength));
    setAlignmentMaxLength(String(config.alignment_max_length ?? alignmentMaxLength));
    setAlignmentMinKeepRatio(String(config.alignment_min_keep_ratio ?? alignmentMinKeepRatio));
    setAlignmentDatasetPath(parseString(config.alignment_dataset_path, alignmentDatasetPath));

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
        .catch(() => {});
    } catch (err: any) {
      setWizardResult(null);
      if (!options?.silent) {
        setWizardError(err?.response?.data?.detail || 'Failed to load model recommendations');
      }
    } finally {
      setWizardLoading(false);
    }
  };

  const applyModelWizardRecommendation = (item: ModelWizardRecommendation, rankIndex: number) => {
    const defaults = item.suggested_defaults || {};
    const nextConfig: Record<string, unknown> = {
      base_model: item.model_id,
    };
    if (typeof defaults.task_type === 'string' && defaults.task_type.trim()) {
      nextConfig.task_type = defaults.task_type;
    }
    if (typeof defaults.chat_template === 'string' && defaults.chat_template.trim()) {
      nextConfig.chat_template = defaults.chat_template;
    }
    if (typeof defaults.use_lora === 'boolean') {
      nextConfig.use_lora = defaults.use_lora;
    }
    if (typeof defaults.batch_size === 'number' && Number.isFinite(defaults.batch_size)) {
      nextConfig.batch_size = Math.max(1, defaults.batch_size);
    }
    if (typeof defaults.max_seq_length === 'number' && Number.isFinite(defaults.max_seq_length)) {
      nextConfig.max_seq_length = Math.max(128, defaults.max_seq_length);
    }
    applySuggestedConfig(nextConfig);
    const vramValue = Number.parseFloat(wizardVramGb);
    const rows = Array.isArray(wizardResult?.recommendations) ? wizardResult.recommendations : [];
    void api
      .post(`/projects/${projectId}/training/model-selection/telemetry`, {
        action: 'apply',
        source: 'training_setup_wizard',
        target_device: wizardTargetDevice,
        primary_language: wizardPrimaryLanguage,
        available_vram_gb: Number.isFinite(vramValue) && vramValue > 0 ? vramValue : undefined,
        task_profile: wizardTaskProfile !== 'auto' ? wizardTaskProfile : undefined,
        recommendation_count: rows.length,
        recommendation_model_ids: rows
          .map((row) => String(row?.model_id || '').trim())
          .filter(Boolean),
        selected_model_id: item.model_id,
        selected_rank: Math.max(1, rankIndex + 1),
        selected_score: Number.isFinite(Number(item.match_score))
          ? Number(item.match_score)
          : undefined,
      })
      .catch(() => {});
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
      alignment_auto_filter: false,
      alignment_quality_threshold: false,
      alignment_beta: false,
      alignment_max_prompt_length: false,
      alignment_max_length: false,
      alignment_min_keep_ratio: false,
      alignment_dataset_path: false,
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
    setCloudBurstLoadingCatalog(false);
    setCloudBurstLoadingQuote(false);
    setCloudBurstLoadingPlan(false);
    setCloudBurstError('');
    setCloudBurstQuote(null);
    setCloudBurstPlan(null);
    setCloudBurstPrefillStage('');
    void loadPreferredPlanProfile();
    void loadTrainingRuntimes();
    void loadTrainingRecipes();
    void loadCloudBurstCatalog();
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
      setCloudBurstPrefillStage(prefill.stage);
    };
    void applyCloudBurstPrefill();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  useEffect(() => {
    if (workspaceView !== 'setup' || !createFormVisible) {
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
  }, [workspaceView, createFormVisible, wizardAutoRan, wizardLoading, projectId]);

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
        alignment_auto_filter: false,
        alignment_quality_threshold: false,
        alignment_beta: false,
        alignment_max_prompt_length: false,
        alignment_max_length: false,
        alignment_min_keep_ratio: false,
        alignment_dataset_path: false,
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
                Use recipe + defaults for quick setup, then open advanced sections only if needed.
              </span>
            </div>
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
                </div>
                {cloudBurstError && (
                  <div className="training-alert training-alert--error">
                    {cloudBurstError}
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
              </div>
            </details>

            <details className="training-collapsible" open>
              <summary>
                <span>Model & Hyperparameters</span>
                <small>Core settings plus advanced memory/PEFT controls</small>
              </summary>
              <div className="training-collapsible__content">
                <div className="training-config-grid">
                  <div>
                    <h4 className="training-config-section-title">Basic HParams</h4>
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
                    <div className="form-group">
                      <label className="form-label">Base Model</label>
                      <input className="input" value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />
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
                  </div>
                </div>
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
              </div>

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
                  </div>
                )}
                  </div>
                </div>
              </div>
            </details>

            <div className="training-create-shell__actions">
              <button className="btn btn-primary" onClick={handleCreate}>Create Experiment</button>
              {!forceCreateVisible && (
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
              )}
            </div>
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
    </div>
  );
}
