/**
 * Pure helpers + constants for TrainingPanel (log-line metric parsing,
 * config-field bookkeeping, value coercion).
 */

import type {
  TrainingMetric,
  TrainingPreflightReport,
  ModelWizardResponse,
  ModelBenchmarkSweepResponse,
  ModelBenchmarkHistoryRun,
} from './trainingPanelTypes';

export const METRIC_PREFIX = 'SLM_METRIC ';
export const PLAN_PROFILE_STORAGE_PREFIX = 'slm-training-plan-profile';
export const MODEL_WIZARD_TASK_PROFILES = [
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

export function parseNumericField(text: string, key: string): number | null {
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

export function parseMetricFromLogLine(text: string, experimentId: number): TrainingMetric | null {
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

export function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

/** Plain-language gloss of the warm-start resolution reason codes. */
export function describeWarmStartReason(reason: string | undefined): string {
  const raw = (reason || '').trim();
  if (!raw) return '';
  const sep = raw.indexOf(':');
  const code = sep === -1 ? raw : raw.slice(0, sep);
  const name = sep === -1 ? '' : raw.slice(sep + 1);
  switch (code) {
    case 'warm_start':
      return `Warm start from ${name || 'a pre-fine-tuned checkpoint'}`;
    case 'no_checkpoint_recommended':
      return 'Cold start — this recipe recommends no warm-start checkpoint';
    case 'checkpoint_planned':
      return `Cold start — warm start "${name}" is planned, not built yet`;
    case 'checkpoint_not_registered':
      return `Cold start — recommended warm start "${name}" is not registered`;
    case 'checkpoint_base_model_mismatch':
      return `Cold start — warm start "${name}" targets a different base model`;
    case 'checkpoint_artifact_missing':
      return `Cold start — warm start "${name}" weights not found locally`;
    default:
      return raw;
  }
}

export function asStringList(value: unknown): string[] {
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

export function parseBool(value: unknown): boolean | null {
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

export function parseNonNegativeInt(value: unknown): number {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return 0;
  }
  return Math.max(0, Math.trunc(num));
}

// Every config field the form can mark as user-touched. Quality-Lift phase 7
// slice 3 added the multi-seed keys (seed / num_seeds / seeds / parallel_seeds).
export const CONFIG_FIELD_KEYS = [
  'training_mode',
  'training_runtime_id',
  'task_type',
  'trainer_backend',
  'chat_template',
  'learning_rate',
  'num_epochs',
  'batch_size',
  'gradient_accumulation_steps',
  'max_seq_length',
  'optimizer',
  'save_steps',
  'eval_steps',
  'sequence_packing',
  'use_lora',
  'curriculum',
  'lora_r',
  'lora_alpha',
  'target_modules',
  'fp16',
  'bf16',
  'flash_attention',
  'auto_oom_retry',
  'max_oom_retries',
  'oom_retry_seq_shrink',
  'gradient_checkpointing',
  'multimodal_require_media',
  'alignment_auto_filter',
  'alignment_quality_threshold',
  'alignment_beta',
  'alignment_max_prompt_length',
  'alignment_max_length',
  'alignment_min_keep_ratio',
  'alignment_dataset_path',
  'alignment_include_playground_feedback',
  'alignment_playground_max_pairs',
  'observability_enabled',
  'observability_log_steps',
  'observability_max_layers',
  'observability_probe_attention',
  'observability_probe_top_k',
  'seed',
  'num_seeds',
  'seeds',
  'parallel_seeds',
] as const;

export type ConfigFieldKey = (typeof CONFIG_FIELD_KEYS)[number];

/** Fresh "nothing touched" map — one source so resets can't drift from the key list. */
export function untouchedConfig(): Record<ConfigFieldKey, boolean> {
  return Object.fromEntries(CONFIG_FIELD_KEYS.map((key) => [key, false])) as Record<ConfigFieldKey, boolean>;
}

export type TrainingWorkspaceView = 'overview' | 'setup' | 'runs';
export type TrainingSetupTab = 'basics' | 'config' | 'power' | 'review';
export type ModelSelectionApplySource = 'recommendation' | 'benchmark' | 'consensus';

/** Badge class for an experiment status. */
export function statusColor(status: string): string {
  return status === 'completed'
  ? 'badge-success'
  : status === 'running'
    ? 'badge-info'
    : status === 'failed'
      ? 'badge-error'
      : 'badge-warning';
}

/** Flattened capability/contract view of a preflight report (adapter, model, runtime, media). */
export function buildPreflightContractDetails(preflightPreview: TrainingPreflightReport | null) {
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
}

export type PreflightContractDetails = ReturnType<typeof buildPreflightContractDetails>;

/** Model-selection wizard vs benchmark winner roll-up for the Review tab. */
export function buildModelSelectionSummary(wizardResult: ModelWizardResponse | null, benchmarkResult: ModelBenchmarkSweepResponse | null, benchmarkHistory: ModelBenchmarkHistoryRun[], baseModel: string) {
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
}

export type ModelSelectionSummary = ReturnType<typeof buildModelSelectionSummary>;
