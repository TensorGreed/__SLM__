/**
 * Training orchestrator panel — config editor, run launcher, live metrics, checkpoints, and post-run review.
 */

import { useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type { ErrorEnvelope } from '../../api/errors';
import { parseErrorEnvelope } from '../../api/errors';
import type { Job as JobShape } from '../../api/jobs';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import ErrorPanel from '../shared/ErrorPanel';
import StepFooter from '../shared/StepFooter';
import { Term } from '../shared/Term';
import { ReadinessPanel } from '../shared/ReadinessPanel';
import ExperimentCompare from './ExperimentCompare';
import HardwareRecommenderModal from './HardwareRecommenderModal';
import type { RecommendationResult } from './HardwareRecommenderModal';
import CloudBurstPlanningSection from './CloudBurstPlanningSection';
import TrainingRunView from './TrainingRunView';
import TrainingRunsList from './TrainingRunsList';
import TrainingValidationSection from './TrainingValidationSection';
import TrainingAdvancedPeftSection from './TrainingAdvancedPeftSection';
import TrainingModelEssentialsSection from './TrainingModelEssentialsSection';
import TrainingMultiSeedSection from './TrainingMultiSeedSection';
import { useTrainingConfigForm } from './useTrainingConfigForm';
import PreRunConfirmModal from './PreRunConfirmModal';
import { buildWsUrl } from '../../utils/ws';
import {
    fetchOverrides,
    TRAINING_OVERRIDES_APPLIED_EVENT,
    type TrainingOverridesAppliedDetail,
} from '../../api/trainingConfigGaps';
import './TrainingPanel.css';
import './WaveDPanels.css';
import type {
  TrainingPanelProps,
  Experiment,
  TrainingMetric,
  TrainingEffectiveConfigResponse,
  TrainingPreflightReport,
  TrainingPreflightPreviewResponse,
  TrainingExperimentPreflightResponse,
  TrainingPreflightPlanSuggestion,
  TrainingPreflightPlanReport,
  TrainingPreflightPlanResponse,
  TrainingPreferencesResponse,
  TrainingRuntimeCatalogResponse,
  TrainingObservabilitySummary,
  TrainingObservabilityResponse,
  VibeCheckSnapshot,
  VibeCheckTimelineResponse,
  TrainingRecipe,
  TrainingRecipeCatalogResponse,
  TrainingRecipeResolveResponse,
  ModelWizardRecommendation,
  ModelWizardResponse,
  ModelBenchmarkSweepResponse,
  ModelBenchmarkHistoryRun,
  ModelBenchmarkHistoryResponse,
  ModelIntrospectionSummary,
  ModelIntrospectionResponse,
} from './trainingPanelTypes';
import {
  PLAN_PROFILE_STORAGE_PREFIX,
  parseMetricFromLogLine,
  describeWarmStartReason,
  asStringList,
  parseBool,
  type ConfigFieldKey,
  untouchedConfig,
  type TrainingWorkspaceView,
  type TrainingSetupTab,
  type ModelSelectionApplySource,
  buildModelSelectionSummary,
  buildPreflightContractDetails,
} from './trainingPanelUtils';


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
  // Subscribe to the jobs store so each in-flight experiment row
  // can render a live-loss sparkline + kill switch inline (mirrors
  // the bell). The store polls /api/jobs/active every 4s while any
  // training_start job is in-flight; metrics_recent flows through
  // from the experiment's trainer_state.json. Job-by-experiment-id
  // lookup is memoised so per-row render doesn't re-scan the list.
  const allJobs = useJobsStore((s) => s.jobs);
  const jobByExperimentId = useMemo(() => {
    const map = new Map<number, JobShape>();
    for (const j of allJobs) {
      if (j.kind !== 'training_start') continue;
      const expId = (j.params as Record<string, unknown> | undefined)
        ?.experiment_id;
      if (typeof expId === 'number') map.set(expId, j);
    }
    return map;
  }, [allJobs]);
  const [showCreate, setShowCreate] = useState(Boolean(forceCreateVisible && !hideCreateControls));
  const [activeExperiment, setActiveExperiment] = useState<Experiment | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [selectedForCompare, setSelectedForCompare] = useState<number[]>([]);
  const [showCompare, setShowCompare] = useState(false);
  const [taskState, setTaskState] = useState<string>('');
  // Diagnostics Intervention B — load-bearing render uses the shared
  // <ErrorPanel>. Some call sites build messages inline (preflight
  // failures with hint lists); those go through ``trainingErrorFromMessage``
  // to wrap as a synthetic envelope.
  const [trainingError, setTrainingError] = useState<ErrorEnvelope | null>(null);

  const [name, setName] = useState('');
  const configForm = useTrainingConfigForm();
  const {
    baseModel,
    setBaseModel,
    trainingMode,
    setTrainingMode,
    trainingRuntimeId,
    setTrainingRuntimeId,
    taskType,
    setTaskType,
    trainerBackend,
    setTrainerBackend,
    chatTemplate,
    setChatTemplate,
    lr,
    setLr,
    epochs,
    setEpochs,
    batchSize,
    setBatchSize,
    gradientAccumulationSteps,
    setGradientAccumulationSteps,
    maxSeqLength,
    setMaxSeqLength,
    optimizer,
    setOptimizer,
    saveSteps,
    setSaveSteps,
    evalSteps,
    setEvalSteps,
    sequencePacking,
    setSequencePacking,
    useLora,
    setUseLora,
    curriculum,
    setCurriculum,
    loraR,
    setLoraR,
    loraAlpha,
    setLoraAlpha,
    targetModules,
    setTargetModules,
    fp16,
    setFp16,
    bf16,
    setBf16,
    flashAttention,
    setFlashAttention,
    autoOomRetry,
    setAutoOomRetry,
    maxOomRetries,
    setMaxOomRetries,
    oomRetrySeqShrink,
    setOomRetrySeqShrink,
    gradientCheckpointing,
    setGradientCheckpointing,
    multimodalRequireMedia,
    setMultimodalRequireMedia,
    alignmentAutoFilter,
    setAlignmentAutoFilter,
    alignmentQualityThreshold,
    setAlignmentQualityThreshold,
    alignmentBeta,
    setAlignmentBeta,
    alignmentMaxPromptLength,
    setAlignmentMaxPromptLength,
    alignmentMaxLength,
    setAlignmentMaxLength,
    alignmentMinKeepRatio,
    setAlignmentMinKeepRatio,
    alignmentDatasetPath,
    setAlignmentDatasetPath,
    alignmentIncludePlaygroundFeedback,
    setAlignmentIncludePlaygroundFeedback,
    alignmentPlaygroundMaxPairs,
    setAlignmentPlaygroundMaxPairs,
    observabilityEnabled,
    setObservabilityEnabled,
    observabilityLogSteps,
    setObservabilityLogSteps,
    observabilityMaxLayers,
    setObservabilityMaxLayers,
    observabilityProbeAttention,
    setObservabilityProbeAttention,
    observabilityProbeTopK,
    setObservabilityProbeTopK,
    seed,
    setSeed,
    numSeeds,
    setNumSeeds,
    seedsExplicit,
    setSeedsExplicit,
    parallelSeeds,
    setParallelSeeds,
    setMultiSeedExpanded,
    useProfileDefaults,
    setUseProfileDefaults,
    touchedConfig,
    setTouchedConfig,
  } = configForm;
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
        detail: 'Open Setup and use a training preset + preflight before launching.',
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

  const preflightContractDetails = useMemo(() => buildPreflightContractDetails(preflightPreview), [preflightPreview]);

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

  const modelSelectionSummary = useMemo(() => buildModelSelectionSummary(wizardResult, benchmarkResult, benchmarkHistory, baseModel), [wizardResult, benchmarkResult, benchmarkHistory, baseModel]);

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
    // An epoch count the user actually sends is explicit; otherwise the
    // trainer scales epochs to the dataset size (backend auto_epochs).
    if (includeField('num_epochs')) {
      config.num_epochs = epochs;
      config.auto_epochs = false;
    }
    if (includeField('batch_size')) config.batch_size = batchSize;
    if (includeField('gradient_accumulation_steps')) config.gradient_accumulation_steps = gradientAccumulationSteps;
    if (includeField('max_seq_length')) config.max_seq_length = maxSeqLength;
    if (includeField('optimizer')) config.optimizer = optimizer;
    if (includeField('save_steps')) config.save_steps = saveSteps;
    if (includeField('eval_steps')) config.eval_steps = evalSteps;
    if (includeField('sequence_packing')) config.sequence_packing = sequencePacking;
    if (includeField('use_lora')) config.use_lora = useLora;
    // Phase 6d — only forward curriculum when the user has explicitly
    // touched it. Leaving it unset lets the backend's auto-default
    // heuristic fire (on for thin classification projects, off
    // otherwise); sending false explicitly would clobber that.
    if (includeField('curriculum')) config.curriculum = curriculum;
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
    // Quality-Lift phase 7 slice 3 — multi-seed variance reporting.
    // ``seed`` always rides (base PRNG seed; backend defaults to 42
    // matching the schema if we omit it, but explicit keeps the
    // payload deterministic for the user). ``num_seeds`` only when >1
    // — sending 1 every time would clutter the experiment config and
    // confuse drill-down. ``seeds`` (explicit comma list) wins over
    // num_seeds in the backend resolver; only forward when the user
    // typed something parseable. ``parallel_seeds`` only when multi-
    // seed is active — meaningless when num_seeds=1.
    const isMultiSeed = numSeeds > 1;
    const parsedExplicitSeeds = seedsExplicit
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n) && Number.isInteger(n));
    if (includeField('seed')) config.seed = seed;
    if (includeField('num_seeds') && isMultiSeed) config.num_seeds = numSeeds;
    if (includeField('seeds') && parsedExplicitSeeds.length > 1) {
      config.seeds = parsedExplicitSeeds;
    }
    if (includeField('parallel_seeds') && (isMultiSeed || parsedExplicitSeeds.length > 1)) {
      config.parallel_seeds = parallelSeeds;
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
    setCurriculum(parseBoolean(config.curriculum, curriculum));
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
    // Quality-Lift phase 7 slice 3 — multi-seed.
    setSeed(Math.max(0, Math.trunc(parseNumber(config.seed, seed))));
    const suggestedNumSeeds = Math.max(
      1, Math.min(8, Math.trunc(parseNumber(config.num_seeds, numSeeds))),
    );
    setNumSeeds(suggestedNumSeeds);
    if (Array.isArray(config.seeds)) {
      setSeedsExplicit(
        config.seeds
          .map((item) => String(item).trim())
          .filter(Boolean)
          .join(', '),
      );
    }
    setParallelSeeds(parseBoolean(config.parallel_seeds, parallelSeeds));
    // Auto-expand the multi-seed section so the suggested values are
    // visible — same affordance the coach deep-link uses.
    if (suggestedNumSeeds > 1) {
      setMultiSeedExpanded(true);
    }

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
        warm_start: res.data?.warm_start ?? null,
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
      setRecipeResolveError(err?.response?.data?.detail || 'Failed to load training presets');
    }
  };

  const applySelectedRecipe = async () => {
    if (!selectedRecipeId) {
      setRecipeResolveError('Select a training preset first.');
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
        warm_start: res.data?.warm_start ?? null,
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
        setRecipeResolveError(`Preset applied, but missing required fields: ${missing.join(', ')}`);
      }
    } catch (err: any) {
      setRecipeResolveError(err?.response?.data?.detail || 'Failed to apply preset');
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

  // Story 1.7 — experiment lifecycle recovery actions. All three
  // call into the new /experiments/{id}/{reset,delete} +
  // /experiments/bulk-archive-failed endpoints so the operator never
  // has to hand-craft SQL + mv commands to recover from a chain of
  // stale-checkpoint failures (see the 9/10/11 incident series).
  const handleResetExperiment = async (exp: Experiment) => {
    if (!window.confirm(
      `Reset experiment "${exp.name}" (#${exp.id})? Status flips back to PENDING, output directory gets archived to .bak.<timestamp>, and stale checkpoints are dropped. You can re-start it after.`,
    )) return;
    try {
      const res = await api.post(
        `/projects/${projectId}/training/experiments/${exp.id}/reset`,
      );
      const data = res.data || {};
      toast.success(
        `Reset experiment #${exp.id}. `
        + (data.archived_output_dir ? 'Output dir archived. ' : '')
        + (data.checkpoints_deleted ? `${data.checkpoints_deleted} stale checkpoint row(s) cleared.` : ''),
      );
      await refreshExperiments();
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Reset failed');
    }
  };

  const handleDeleteExperiment = async (exp: Experiment) => {
    const typed = window.prompt(
      `Delete experiment "${exp.name}" (#${exp.id}) — this removes the DB row + output directory PERMANENTLY. Type the experiment NAME to confirm:`,
    );
    if (typed !== exp.name) {
      if (typed !== null) toast.info('Name did not match — delete cancelled.');
      return;
    }
    try {
      const res = await api.delete(
        `/projects/${projectId}/training/experiments/${exp.id}`,
      );
      const data = res.data || {};
      toast.success(
        `Deleted experiment "${data.name || exp.name}" (#${exp.id}). `
        + (data.output_dir_removed ? 'Output dir removed. ' : 'No output dir to remove. ')
        + (data.checkpoints_deleted ? `${data.checkpoints_deleted} checkpoint row(s) cleared.` : ''),
      );
      await refreshExperiments();
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Delete failed');
    }
  };

  const handleBulkArchiveFailed = async () => {
    const failedCount = experiments.filter((e) => e.status === 'failed').length;
    if (failedCount === 0) return;
    if (!window.confirm(
      `Archive all ${failedCount} FAILED experiment(s)? Each one's output dir gets renamed to .bak.<timestamp> and its status flips back to PENDING. RUNNING experiments are skipped automatically.`,
    )) return;
    try {
      const res = await api.post(
        `/projects/${projectId}/training/experiments/bulk-archive-failed`,
      );
      const data = res.data || {};
      toast.success(
        `Archived ${data.reset_count || 0} failed experiment(s)`
        + (data.skipped_count ? `; skipped ${data.skipped_count} (race with RUNNING).` : '.'),
      );
      await refreshExperiments();
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Bulk archive failed');
    }
  };

  const refreshExperiments = async () => {
    const res = await api.get(`/projects/${projectId}/training/experiments`);
    const rows: Experiment[] = res.data || [];
    setExperiments(rows);
    // Functional update: refresh only the run that is open *now*. Reading
    // the closed-over ``activeExperiment`` re-opened a run the user had just
    // left via "Back to Experiments" (it still held the stale value).
    setActiveExperiment((prev) => (prev ? rows.find((e) => e.id === prev.id) ?? prev : prev));
  };

  // Hardening — Coach Mode's "Consider <model>" suggestion in the
  // trainability-forecast warning emits a window CustomEvent with
  // the recommended base model. We listen here and apply via
  // setBaseModel + success toast. We use a window event (not a URL
  // param) because the user is typically already on the training-
  // config page when they click, so react-router's same-path
  // navigate() doesn't re-mount us and a URL-param-on-mount read
  // would never re-fire.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ recommendedBaseModel?: string }>).detail;
      const recommended = detail?.recommendedBaseModel;
      if (typeof recommended === 'string' && recommended.trim()) {
        setBaseModel(recommended);
        toast.success(
          `Base model set to ${recommended} (Coach recommendation applied)`,
          5000,
        );
      }
    };
    window.addEventListener('brewslm:apply-recommended-base-model', handler);
    return () => {
      window.removeEventListener('brewslm:apply-recommended-base-model', handler);
    };
  }, []);

  // Quality-Lift phase 7 slice 3 — Coach Mode's variance nudge deep-
  // links here with ``?expand_multi_seed=1`` (URL fallback for the
  // first mount) AND a window CustomEvent (for the same-page click
  // when the user is already on training-config). The URL path also
  // covers a coach card consumed from a different page that
  // react-router navigates from. ``suggested_num_seeds`` (default 3)
  // sets the count + opens the section so the user lands ready to
  // launch.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    if (params.get('expand_multi_seed') === '1') {
      setMultiSeedExpanded(true);
      const sn = Number(params.get('suggested_num_seeds') || '');
      if (Number.isFinite(sn) && sn >= 2 && sn <= 8) {
        setNumSeeds(Math.trunc(sn));
        setTouchedConfig((prev) => ({ ...prev, num_seeds: true }));
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ suggestedNumSeeds?: number }>).detail;
      setMultiSeedExpanded(true);
      const sn = Number(detail?.suggestedNumSeeds);
      if (Number.isFinite(sn) && sn >= 2 && sn <= 8) {
        setNumSeeds(Math.trunc(sn));
        setTouchedConfig((prev) => ({ ...prev, num_seeds: true }));
      }
    };
    window.addEventListener('brewslm:expand-multi-seed', handler);
    return () => {
      window.removeEventListener('brewslm:expand-multi-seed', handler);
    };
  }, []);

  // Coach-stage-2 phase 2 — Training Config Gap patches persist as a
  // partial dict under project.runtime_config.training_config_overrides.
  // Read them on mount so the form shows the same values the gap
  // scanner reports as effective. Without this, a user who applied
  // `eval_steps=10` from the gap panel + then opened the form would
  // see eval_steps=100 in the field, train with 100, and blow past
  // the fix — credibility-killer.
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetchOverrides(projectId);
        if (cancelled) return;
        const overrides = res.overrides || {};
        if (Object.keys(overrides).length === 0) return;
        applySuggestedConfig(overrides);
      } catch {
        // Endpoint failure is non-fatal — the form just shows defaults.
        // The gap panel surface will still flag the gap, and the
        // existing toast/error stack already covers the rare case
        // where the endpoint itself is broken.
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // Same-page sync: when the gap panel's apply succeeds it dispatches
  // a DOM event carrying the new overrides dict. Pipe it straight into
  // applySuggestedConfig so the form updates without a re-mount.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<TrainingOverridesAppliedDetail>)
        .detail;
      if (!detail || detail.projectId !== projectId) return;
      if (!detail.overrides) return;
      applySuggestedConfig(detail.overrides);
    };
    window.addEventListener(TRAINING_OVERRIDES_APPLIED_EVENT, handler);
    return () => {
      window.removeEventListener(TRAINING_OVERRIDES_APPLIED_EVENT, handler);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

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
    setTrainingError(null);
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
    setTouchedConfig(untouchedConfig());
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
    setObservabilitySummary(null);
    setObservabilityRecentCount(0);
    setObservabilityError('');
    setObservabilityLoading(false);
    void loadPreferredPlanProfile();
    void loadTrainingRuntimes();
    void loadTrainingRecipes();
    void loadModelBenchmarkHistory({ silent: true });
    refreshExperiments().catch((err) => console.error('Failed to load experiments', err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, forceCreateVisible, hideCreateControls, hideExperimentList]);


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
    setTrainingError(null);
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
      setTouchedConfig(untouchedConfig());
    } catch (err) {
      setTrainingError(parseErrorEnvelope(err));
    }
  };

  const handleStart = async (experimentId: number) => {
    setTrainingError(null);
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
        // Build a synthetic envelope from the preflight result so the
        // shared <ErrorPanel> renders it the same way as backend
        // errors. The hint list lands in metadata so the user can
        // expand "Technical details" to see all of them.
        const msg = errors.length > 0
          ? `Preflight failed: ${errors.join(' | ')}${hintText}`
          : 'Preflight failed due to incompatible configuration.';
        setTrainingError(parseErrorEnvelope(msg));
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
    } catch (err) {
      setTrainingError(parseErrorEnvelope(err));
    }
  };

  const handleCancel = async (experimentId: number) => {
    setTrainingError(null);
    try {
      await api.post(`/projects/${projectId}/training/experiments/${experimentId}/cancel`);
      setActiveExperiment((prev) => (prev ? { ...prev, status: 'cancelled' } : prev));
      setTaskState('cancel_requested');
      refreshExperiments().catch(() => undefined);
    } catch (err) {
      setTrainingError(parseErrorEnvelope(err));
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
    return (
      <TrainingRunView
        projectId={projectId}
        activeExperiment={activeExperiment}
        metrics={metrics}
        taskState={taskState}
        trainingError={trainingError}
        trainingWarnings={trainingWarnings}
        trainingLogs={trainingLogs}
        observabilitySummary={observabilitySummary}
        observabilityRecentCount={observabilityRecentCount}
        observabilityError={observabilityError}
        observabilityLoading={observabilityLoading}
        vibeTimeline={vibeTimeline}
        vibeSelectedIndex={vibeSelectedIndex}
        vibeConfig={vibeConfig}
        vibeError={vibeError}
        vibeLoading={vibeLoading}
        onBack={() => {
          setActiveExperiment(null);
          setTaskState('');
          setTrainingWarnings([]);
          refreshExperiments().catch(() => undefined);
        }}
        onCancel={(experimentId) => void handleCancel(experimentId)}
        onDismissError={() => setTrainingError(null)}
        onRefreshExperiments={() => void refreshExperiments()}
        onRefreshObservability={(experimentId) => void loadObservabilitySummary(experimentId)}
        onRefreshVibe={(experimentId) => void loadVibeTimeline(experimentId)}
        onSelectVibe={setVibeSelectedIndex}
      />
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
              <p>
                <Term id="recipe" />, profile defaults, <Term id="preflight" />, hyperparameters.
              </p>
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
                  ? 'Use a training preset + defaults for quick setup, then open advanced sections only if needed.'
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
                  <label className="form-label">Training preset</label>
                  <div className="form-inline-actions">
                    <select
                      className="input training-recipe-select"
                      value={selectedRecipeId}
                      onChange={(e) => setSelectedRecipeId(e.target.value)}
                    >
                      <option value="">Select preset</option>
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
                      {recipeResolveLoading ? 'Applying...' : 'Apply preset'}
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
                    A training preset applies a domain-agnostic config patch, then runtime/profile defaults and preflight.
                  </div>
                  {recipeResolveError && (
                    <div className="training-alert training-alert--warning training-alert--tight">
                      {recipeResolveError}
                    </div>
                  )}
                  {effectivePreview?.warm_start && (
                    <div
                      className={`training-warm-start training-warm-start--${
                        effectivePreview.warm_start.source === 'checkpoint' ? 'warm' : 'cold'
                      }`}
                    >
                      <strong>Starting weights:</strong>{' '}
                      {effectivePreview.warm_start.source === 'checkpoint'
                        ? effectivePreview.warm_start.checkpoint_name ||
                          effectivePreview.warm_start.manifest?.display_name ||
                          'warm start'
                        : 'base model (cold start)'}
                      {' — '}
                      {describeWarmStartReason(effectivePreview.warm_start.reason)}
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
            <TrainingValidationSection
              projectId={projectId}
              effectivePreview={effectivePreview}
              effectivePreviewLoading={effectivePreviewLoading}
              effectivePreviewError={effectivePreviewError}
              preflightPreview={preflightPreview}
              preflightPreviewLoading={preflightPreviewLoading}
              preflightPreviewError={preflightPreviewError}
              preflightPlan={preflightPlan}
              preflightPlanLoading={preflightPlanLoading}
              preflightPlanError={preflightPlanError}
              preferredPlanProfile={preferredPlanProfile}
              preflightContractDetails={preflightContractDetails}
              modelSelectionSummary={modelSelectionSummary}
              onPreviewEffectiveConfig={() => void previewEffectiveConfig()}
              onRunPreflightPreview={() => void runPreflightPreview()}
              onRunPreflightPlan={() => void runPreflightPlan()}
              onApplyPlanSuggestion={applyPlanSuggestion}
            />
            )}

            {showSetupPower && (
            <CloudBurstPlanningSection key={projectId} projectId={projectId} />
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
                  <TrainingModelEssentialsSection
                    form={configForm}
                    applyBenchmarkWinner={applyBenchmarkWinner}
                    applyModelSelectionChoice={applyModelSelectionChoice}
                    applyModelWizardRecommendation={applyModelWizardRecommendation}
                    baseModelIntrospection={baseModelIntrospection}
                    baseModelIntrospectionError={baseModelIntrospectionError}
                    baseModelIntrospectionLoading={baseModelIntrospectionLoading}
                    benchmarkError={benchmarkError}
                    benchmarkHistory={benchmarkHistory}
                    benchmarkLoading={benchmarkLoading}
                    benchmarkResult={benchmarkResult}
                    buildTrainingConfigPayload={buildTrainingConfigPayload}
                    introspectBaseModel={introspectBaseModel}
                    isAlignmentMode={isAlignmentMode}
                    projectId={projectId}
                    runModelBenchmarkSweep={runModelBenchmarkSweep}
                    runModelWizard={runModelWizard}
                    runtimeCatalog={runtimeCatalog}
                    runtimeCatalogError={runtimeCatalogError}
                    selectedRuntimeModalities={selectedRuntimeModalities}
                    selectedRuntimeModalitiesDeclared={selectedRuntimeModalitiesDeclared}
                    selectedRuntimeSpec={selectedRuntimeSpec}
                    setWizardPrimaryLanguage={setWizardPrimaryLanguage}
                    setWizardTargetDevice={setWizardTargetDevice}
                    setWizardTaskProfile={setWizardTaskProfile}
                    setWizardVramGb={setWizardVramGb}
                    showSetupConfig={showSetupConfig}
                    showSetupPower={showSetupPower}
                    wizardError={wizardError}
                    wizardLoading={wizardLoading}
                    wizardPrimaryLanguage={wizardPrimaryLanguage}
                    wizardResult={wizardResult}
                    wizardTargetDevice={wizardTargetDevice}
                    wizardTaskProfile={wizardTaskProfile}
                    wizardVramGb={wizardVramGb}
                  />

                  {showSetupPower && (
                  <TrainingMultiSeedSection
                    form={configForm}
                  />
                  )}

                  {showSetupPower && (
                  <TrainingAdvancedPeftSection
                    form={configForm}
                    isAlignmentMode={isAlignmentMode}
                  />
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
          <ErrorPanel
            envelope={trainingError}
            onDismiss={() => setTrainingError(null)}
            testIdPrefix="training-error-bottom"
          />
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
          <TrainingRunsList
            hideCreateControls={hideCreateControls}
            experiments={experiments}
            jobByExperimentId={jobByExperimentId}
            selectedForCompare={selectedForCompare}
            handleBulkArchiveFailed={handleBulkArchiveFailed}
            handleDeleteExperiment={handleDeleteExperiment}
            handleResetExperiment={handleResetExperiment}
            setPendingStartExperiment={setPendingStartExperiment}
            setShowCompare={setShowCompare}
            toggleCompareSelection={toggleCompareSelection}
            viewDashboard={viewDashboard}
          />
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
