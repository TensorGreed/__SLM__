/**
 * API response + config shapes used by TrainingPanel and its sections.
 */

export interface TrainingPanelProps {
  projectId: number;
  onNextStep?: () => void;
  title?: string;
  hideStepFooter?: boolean;
  hideCreateControls?: boolean;
  hideExperimentList?: boolean;
  forceCreateVisible?: boolean;
  setupMode?: 'essentials' | 'advanced';
}

// The stored experiment config is the full TrainingConfig dump plus
// runtime blocks (``_runtime``, ``_warm_start``…); the fields read here
// are typed, everything else stays ``unknown``.
export interface ExperimentConfig extends Record<string, unknown> {
  num_epochs?: number;
  auto_epochs?: boolean;
  learning_rate?: number | string;
  batch_size?: number;
  use_lora?: boolean;
  lora_r?: number;
  target_modules?: string[] | string;
  task_type?: string;
}

export interface Experiment {
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

export interface TrainingMetric {
  experiment_id?: number;
  epoch?: number;
  step?: number;
  train_loss?: number | null;
  eval_loss?: number | null;
  [key: string]: unknown;
}

export interface WarmStartResolution {
  source?: string;
  effective_base_model?: string;
  checkpoint_name?: string | null;
  reason?: string;
  manifest?: { display_name?: string; status?: string } | null;
}

export interface TrainingEffectiveConfigResponse {
  domain_pack_applied?: string | null;
  domain_pack_source?: string | null;
  domain_profile_applied?: string | null;
  domain_profile_source?: string | null;
  profile_training_defaults?: Record<string, unknown> | null;
  resolved_training_config?: Record<string, unknown> | null;
  resolved_training_mode?: string;
  profile_defaults_applied?: string[];
  warm_start?: WarmStartResolution | null;
}

export interface TrainingPreflightReport {
  ok: boolean;
  errors: string[];
  warnings: string[];
  hints?: string[];
  capability_summary?: Record<string, unknown>;
}

export interface TrainingPreflightPreviewResponse extends TrainingEffectiveConfigResponse {
  preflight?: TrainingPreflightReport;
}

export interface TrainingExperimentPreflightResponse {
  experiment_id: number;
  status: string;
  resolved_training_config?: Record<string, unknown> | null;
  preflight?: TrainingPreflightReport;
}

export interface TrainingPlanChange {
  field: string;
  from?: unknown;
  to?: unknown;
  reason?: string;
}

export interface TrainingPreflightPlanSuggestion {
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

export interface TrainingPreflightPlanReport {
  base_preflight?: TrainingPreflightReport;
  suggestions: TrainingPreflightPlanSuggestion[];
  recommended_profile?: string;
  profile_order?: string[];
}

export interface TrainingPreflightPlanResponse extends TrainingEffectiveConfigResponse {
  plan?: TrainingPreflightPlanReport;
}

export interface TrainingPreferencesResponse {
  project_id: number;
  preferred_plan_profile: string;
  profile_options?: string[];
  source?: string;
}

export interface TrainingRuntimeSpec {
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

export interface TrainingRuntimeCatalogResponse {
  project_id: number;
  default_runtime_id?: string;
  runtime_count?: number;
  legacy_aliases?: Record<string, string>;
  runtimes?: TrainingRuntimeSpec[];
}

export interface ObservabilityLayerSummary {
  layer?: string;
  event_count?: number;
  avg_grad_norm?: number;
  max_grad_norm?: number;
  avg_update_ratio?: number;
}

export interface ObservabilityTokenSummary {
  token?: string;
  count?: number;
}

export interface TrainingObservabilitySummary {
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

export interface TrainingObservabilityResponse {
  summary?: TrainingObservabilitySummary;
  recent?: {
    count?: number;
    events?: Array<Record<string, unknown>>;
  };
}

export interface VibeCheckOutput {
  prompt_id?: string;
  prompt?: string;
  reply?: string;
  provider?: string;
  model_name?: string;
  latency_ms?: number | null;
  error?: string | null;
}

export interface VibeCheckSnapshot {
  step?: number;
  epoch?: number | null;
  progress?: number | null;
  train_loss?: number | null;
  eval_loss?: number | null;
  created_at?: string | null;
  outputs?: VibeCheckOutput[];
}

export interface VibeCheckTimelineResponse {
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

export interface TrainingRecipe {
  recipe_id: string;
  display_name: string;
  description?: string;
  category?: string;
  tags?: string[];
  required_fields?: string[];
  recommended_starting_checkpoint?: string;
}

export interface TrainingRecipeCatalogResponse {
  project_id: number;
  recipe_count?: number;
  recipes?: TrainingRecipe[];
}

export interface TrainingRecipeResolveResponse extends TrainingEffectiveConfigResponse {
  project_id: number;
  recipe?: TrainingRecipe & { config_patch?: Record<string, unknown> };
  recipe_missing_required_fields?: string[];
  recipe_config?: Record<string, unknown>;
  preflight?: TrainingPreflightReport;
}

export interface ModelWizardRecommendation {
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

export interface ModelWizardResponse {
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
  /**
   * Story 1.5 Gate 1 — when the project's prepared train.jsonl has
   * no target field, the API returns this flag + a message instead
   * of a model list. The UI surfaces a banner that blocks model pick
   * until the data shape is fixed.
   */
  blocked_by_data_shape?: boolean;
  data_shape_message?: string;
  adaptive_ranking?: {
    enabled?: boolean;
    context_label?: string;
    global_apply_events?: number;
    context_apply_events?: number;
    boosted_model_count?: number;
  };
}

export interface ModelBenchmarkRow {
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
  pareto_optimal?: boolean;
  dominated_by?: string[];
  suggested_defaults?: ModelWizardRecommendation['suggested_defaults'];
}

export interface ModelBenchmarkTradeoffSummary {
  best_quality_model_id?: string;
  best_speed_model_id?: string;
  best_balance_model_id?: string;
}

export interface ModelBenchmarkSweepResponse {
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
  pareto?: ModelBenchmarkParetoMeta;
  warnings?: string[];
}

export interface ModelBenchmarkParetoMeta {
  quality_key?: string;
  cost_key?: string;
  optimal_model_ids?: string[];
}

export interface ModelBenchmarkHistoryRun {
  run_id?: string;
  timestamp?: string;
  benchmark_mode?: string;
  sampled_row_count?: number;
  sampled_avg_tokens?: number;
  tradeoff_summary?: ModelBenchmarkTradeoffSummary;
  matrix?: ModelBenchmarkRow[];
}

export interface ModelBenchmarkHistoryResponse {
  count?: number;
  runs?: ModelBenchmarkHistoryRun[];
}

export interface ModelIntrospectionMemoryProfile {
  estimated_min_vram_gb?: number;
  estimated_ideal_vram_gb?: number;
}

export interface ModelIntrospectionSummary {
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

export interface ModelIntrospectionResponse {
  project_id: number;
  introspection?: ModelIntrospectionSummary;
}

export interface CloudBurstProvider {
  provider_id: string;
  display_name?: string;
  description?: string;
  supports_spot?: boolean;
  supports_live_execution?: boolean;
  supports_managed_cancel?: boolean;
  supports_live_logs?: boolean;
  regions?: string[];
}

export interface CloudBurstGpuSku {
  gpu_sku: string;
  display_name?: string;
  vram_gb?: number;
  hourly_usd?: Record<string, number>;
}

export interface CloudBurstCatalogResponse {
  project_id: number;
  providers?: CloudBurstProvider[];
  gpu_skus?: CloudBurstGpuSku[];
  provider_count?: number;
  gpu_sku_count?: number;
}

export interface CloudBurstQuoteResponse {
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

export interface CloudBurstLaunchPlanResponse {
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

export interface CloudBurstRunArtifacts {
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

export interface CloudBurstMetricPoint {
  step?: number;
  epoch?: number;
  train_loss?: number | null;
  eval_loss?: number | null;
  learning_rate?: number | null;
  throughput_tps?: number | null;
  at?: string | null;
}

export interface CloudBurstLogBridgeSummary {
  last_ingested_at?: string;
  last_ingested_count?: number;
  seen_hash_count?: number;
}

export interface CloudBurstRunStatusResponse {
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

export interface CloudBurstRunListResponse {
  project_id: number;
  count?: number;
  limit?: number;
  runs?: CloudBurstRunStatusResponse[];
}
