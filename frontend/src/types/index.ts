/* ── Types ─────────────────────────────────────────────────────────── */

export type PipelineStage =
    | 'ingestion'
    | 'cleaning'
    | 'gold_set'
    | 'synthetic'
    | 'dataset_prep'
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

/* ── API Response Types ─────────────────────────────────────────────── */

export interface Project {
    id: number;
    name: string;
    description: string | null;
    status: ProjectStatus;
    pipeline_stage: PipelineStage;
    base_model_name: string | null;
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
