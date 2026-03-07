/* ── Types ─────────────────────────────────────────────────────────── */

export type PipelineStage =
    | 'ingestion'
    | 'cleaning'
    | 'gold_set'
    | 'synthetic'
    | 'dataset_prep'
    | 'data_adapter_preview'
    | 'tokenization'
    | 'training'
    | 'evaluation'
    | 'compression'
    | 'export'
    | 'completed';

export type ProjectStatus = 'draft' | 'active' | 'paused' | 'completed' | 'failed';
export type DatasetType = 'raw' | 'cleaned' | 'gold_dev' | 'gold_test' | 'synthetic' | 'train' | 'validation' | 'test';
export type DocumentStatus = 'pending' | 'processing' | 'accepted' | 'rejected' | 'error';
export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type TrainingMode = 'sft' | 'domain_pretrain' | 'dpo' | 'orpo';
export type DomainProfileStatus = 'draft' | 'active' | 'deprecated';
export type DomainPackStatus = 'draft' | 'active' | 'deprecated';

/* ── API Response Types ─────────────────────────────────────────────── */

export interface Project {
    id: number;
    name: string;
    description: string | null;
    status: ProjectStatus;
    pipeline_stage: PipelineStage;
    base_model_name: string | null;
    domain_pack_id: number | null;
    domain_profile_id: number | null;
    created_at: string;
    updated_at: string;
}

export interface ProjectListResponse {
    projects: Project[];
    total: number;
}

export interface ProjectStats {
    id: number;
    name: string;
    pipeline_stage: PipelineStage;
    status: ProjectStatus;
    dataset_count: number;
    experiment_count: number;
    total_documents: number;
}

export interface Dataset {
    id: number;
    project_id: number;
    name: string;
    dataset_type: DatasetType;
    description: string | null;
    record_count: number;
    file_path: string | null;
    is_locked: boolean;
    created_at: string;
    updated_at: string;
}

export interface RawDocument {
    id: number;
    dataset_id: number;
    filename: string;
    file_type: string;
    file_size_bytes: number;
    source: string | null;
    sensitivity: string | null;
    status: DocumentStatus;
    quality_score: number | null;
    chunk_count: number;
    ingested_at: string;
}

export interface Experiment {
    id: number;
    project_id: number;
    name: string;
    description: string | null;
    status: ExperimentStatus;
    training_mode: TrainingMode;
    base_model: string;
    config: Record<string, unknown> | null;
    final_train_loss: number | null;
    final_eval_loss: number | null;
    total_epochs: number | null;
    total_steps: number | null;
    output_dir: string | null;
    started_at: string | null;
    completed_at: string | null;
    created_at: string;
    domain_pack_applied?: string | null;
    domain_pack_source?: string | null;
    domain_profile_applied?: string | null;
    domain_profile_source?: string | null;
    profile_training_defaults?: Record<string, unknown> | null;
    resolved_training_config?: Record<string, unknown> | null;
    profile_defaults_applied?: string[];
}

export interface PipelineStageInfo {
    stage: PipelineStage;
    display_name: string;
    index: number;
    status: 'completed' | 'active' | 'pending';
}

export interface PipelineStatusResponse {
    project_id: number;
    current_stage: PipelineStage;
    progress_percent: number;
    stages: PipelineStageInfo[];
    auto_gate?: {
        experiment_id: number;
        pack_id: string | null;
        passed: boolean;
        failed_gate_ids: string[];
        missing_required_metrics: string[];
        captured_at: string | null;
    } | null;
}

export interface PipelineGraphNodePosition {
    x: number;
    y: number;
}

export interface StepRuntimeRequirements {
    execution_modes: string[];
    required_services: string[];
    required_env: string[];
    required_settings: string[];
    requires_gpu: boolean;
    min_vram_gb: number;
}

export interface PipelineGraphNode {
    id: string;
    stage: PipelineStage;
    display_name: string;
    index: number;
    kind: string;
    status: 'completed' | 'active' | 'pending';
    step_type: string;
    description: string;
    input_artifacts: string[];
    output_artifacts: string[];
    config_schema_ref: string;
    config?: Record<string, unknown>;
    runtime_requirements: StepRuntimeRequirements;
    position: PipelineGraphNodePosition;
}

export interface PipelineGraphEdge {
    id: string;
    source: string;
    target: string;
    kind: string;
}

export interface PipelineGraphResponse {
    project_id: number;
    graph_id: string;
    graph_label: string;
    graph_version: string;
    mode: string;
    current_stage: PipelineStage;
    nodes: PipelineGraphNode[];
    edges: PipelineGraphEdge[];
}

export interface PipelineGraphValidationResponse {
    project_id: number;
    current_stage: PipelineStage;
    valid: boolean;
    requested_source?: string;
    effective_source?: string;
    has_saved_override?: boolean;
    fallback_used: boolean;
    errors: string[];
    warnings: string[];
    graph: PipelineGraphResponse;
}

export interface PipelineGraphDryRunStep {
    id: string;
    stage: PipelineStage;
    status: 'completed' | 'active' | 'pending' | string;
    can_run_now: boolean;
    missing_inputs: string[];
    runtime_requirements: StepRuntimeRequirements;
    missing_runtime_requirements: string[];
    runtime_ready: boolean;
    input_artifacts: string[];
    output_artifacts: string[];
}

export interface PipelineGraphDryRunResponse {
    project_id: number;
    current_stage: PipelineStage;
    valid_graph: boolean;
    requested_source?: string;
    effective_source?: string;
    has_saved_override?: boolean;
    fallback_used: boolean;
    errors: string[];
    warnings: string[];
    available_artifacts: string[];
    active_step: PipelineGraphDryRunStep | null;
    plan: PipelineGraphDryRunStep[];
    graph: PipelineGraphResponse;
}

export interface PipelineGraphRunStepResponse {
    run_id: string;
    run_started_at: string;
    run_finished_at?: string;
    run_record_path?: string;
    project_id: number;
    requested_stage: PipelineStage;
    current_stage: PipelineStage;
    previous_stage?: PipelineStage;
    status: 'ready' | 'blocked' | 'completed' | 'invalid_graph' | string;
    valid_graph: boolean;
    requested_source?: string;
    effective_source?: string;
    has_saved_override?: boolean;
    fallback_used: boolean;
    errors: string[];
    warnings: string[];
    available_artifacts: string[];
    declared_inputs: string[];
    declared_outputs: string[];
    declared_runtime_requirements: StepRuntimeRequirements;
    missing_inputs: string[];
    missing_runtime_requirements: string[];
    can_execute: boolean;
    auto_advance: boolean;
    advanced: boolean;
    published_artifacts?: Array<Record<string, unknown>>;
    published_artifact_keys?: string[];
}

export interface PipelineGraphContractResponse {
    project_id: number;
    current_stage: PipelineStage;
    has_saved_override: boolean;
    requested_source: string;
    effective_source: string;
    graph: PipelineGraphResponse;
}

export interface PipelineGraphCompileChecks {
    active_stage_present: boolean;
    active_stage_node_id: string | null;
    active_stage_missing_inputs: string[];
    active_stage_runtime_requirements: StepRuntimeRequirements;
    active_stage_missing_runtime_requirements: string[];
    active_stage_runtime_ready: boolean;
    active_stage_ready_now: boolean;
}

export interface PipelineGraphCompileResponse {
    project_id: number;
    current_stage: PipelineStage;
    valid_graph: boolean;
    fallback_used: boolean;
    requested_source: string;
    effective_source: string;
    has_saved_override: boolean;
    errors: string[];
    warnings: string[];
    checks: PipelineGraphCompileChecks;
    available_artifacts: string[];
    graph: PipelineGraphResponse;
}

export interface PipelineGraphContractSaveResponse {
    project_id: number;
    saved: boolean;
    path: string;
    graph: PipelineGraphResponse;
}

export interface PipelineGraphContractResetResponse {
    project_id: number;
    reset: boolean;
}

export interface PipelineGraphStageTemplate {
    stage: PipelineStage;
    display_name: string;
    index: number;
    step_type: string;
    description: string;
    input_artifacts: string[];
    output_artifacts: string[];
    config_schema_ref: string;
    runtime_requirements: StepRuntimeRequirements;
}

export interface PipelineGraphStageCatalogResponse {
    project_id: number;
    stages: PipelineGraphStageTemplate[];
}

export interface PipelineGraphTemplate {
    template_id: string;
    display_name: string;
    description: string;
    graph: PipelineGraphResponse;
}

export interface PipelineGraphTemplateListResponse {
    project_id: number;
    templates: PipelineGraphTemplate[];
}

export interface WorkflowRunNode {
    id: number;
    run_id: number;
    node_id: string;
    stage: string;
    step_type: string;
    execution_backend: string;
    status: string;
    attempt_count: number;
    max_retries: number;
    dependencies: string[];
    input_artifacts: string[];
    output_artifacts: string[];
    runtime_requirements: StepRuntimeRequirements;
    missing_inputs: string[];
    missing_runtime_requirements: string[];
    published_artifact_keys: string[];
    error_message: string;
    node_log: Array<Record<string, unknown>>;
    started_at: string | null;
    finished_at: string | null;
    created_at: string | null;
    updated_at: string | null;
}

export interface WorkflowRunSummary {
    node_counts: Record<string, number>;
    total_nodes: number;
    available_artifacts_final?: string[];
    break_reason?: string;
    [key: string]: unknown;
}

export interface WorkflowRun {
    id: number;
    project_id: number;
    graph_id: string;
    graph_version: string;
    execution_backend: string;
    status: string;
    run_config: Record<string, unknown>;
    summary: WorkflowRunSummary;
    started_at: string | null;
    finished_at: string | null;
    created_at: string | null;
    updated_at: string | null;
    nodes: WorkflowRunNode[];
}

export interface WorkflowRunListResponse {
    project_id: number;
    limit: number;
    count: number;
    runs: WorkflowRun[];
}

export interface EvalResult {
    id: number;
    experiment_id: number;
    dataset_name: string;
    eval_type: string;
    metrics: Record<string, number>;
    pass_rate: number | null;
    risk_severity: string | null;
    created_at: string;
}

export interface DomainProfileSummary {
    id: number;
    profile_id: string;
    version: string;
    display_name: string;
    description: string;
    owner: string;
    status: DomainProfileStatus;
    schema_ref: string;
    is_system: boolean;
}

export interface DomainProfileResponse extends DomainProfileSummary {
    contract: Record<string, unknown>;
}

export interface DomainPackSummary {
    id: number;
    pack_id: string;
    version: string;
    display_name: string;
    description: string;
    owner: string;
    status: DomainPackStatus;
    schema_ref: string;
    default_profile_id: string | null;
    is_system: boolean;
}

export interface DomainPackResponse extends DomainPackSummary {
    contract: Record<string, unknown>;
}

/* ── Display Tabs ───────────────────────────────────────────────────── */
export const PIPELINE_TABS = [
    { key: 'data', label: 'Data', icon: '📂', stage: 'ingestion' },
    { key: 'cleaning', label: 'Cleaning', icon: '🧹', stage: 'cleaning' },
    { key: 'goldset', label: 'Gold Set', icon: '🏆', stage: 'gold_set' },
    { key: 'synthetic', label: 'Synthetic', icon: '🧪', stage: 'synthetic' },
    { key: 'dataprep', label: 'Dataset Prep', icon: '📋', stage: 'dataset_prep' },
    { key: 'tokenization', label: 'Tokenization', icon: '🔤', stage: 'tokenization' },
    { key: 'training', label: 'Training', icon: '🔬', stage: 'training' },
    { key: 'eval', label: 'Evaluation', icon: '📊', stage: 'evaluation' },
    { key: 'compression', label: 'Compression', icon: '📦', stage: 'compression' },
    { key: 'export', label: 'Export', icon: '🚀', stage: 'export' },
] as const;

export type TabKey = typeof PIPELINE_TABS[number]['key'];
