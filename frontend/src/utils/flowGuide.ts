import type { PipelineStage, PipelineStatusResponse, Project } from '../types';
import { projectPath, stagePath, trainSectionPath } from './workspaceRoutes';

export interface RecommendedAction {
    path: string;
    title: string;
    description: string;
}

export const PIPELINE_STAGE_ORDER: PipelineStage[] = [
    'ingestion',
    'cleaning',
    'gold_set',
    'synthetic',
    'dataset_prep',
    'data_adapter_preview',
    'tokenization',
    'training',
    'evaluation',
    'compression',
    'export',
    'completed',
];

export const PIPELINE_STAGE_LABEL: Record<PipelineStage, string> = {
    ingestion: 'Data Ingestion',
    cleaning: 'Data Cleaning',
    gold_set: 'Answer Key',
    synthetic: 'Synthetic Data',
    dataset_prep: 'Split',
    data_adapter_preview: 'Adapter Preview',
    tokenization: 'Tokenization',
    training: 'Training',
    evaluation: 'Evaluation',
    compression: 'Compression',
    export: 'Export',
    completed: 'Completed',
};

export function getPipelineStageIndex(stage: string | null | undefined): number {
    if (!stage) {
        return 0;
    }
    const idx = PIPELINE_STAGE_ORDER.indexOf(stage as PipelineStage);
    return idx >= 0 ? idx : 0;
}

function getCurrentStage(project: Project, pipelineStatus: PipelineStatusResponse | null): PipelineStage {
    return (pipelineStatus?.current_stage || project.pipeline_stage || 'ingestion') as PipelineStage;
}

export function getRecommendedAction(
    projectId: number,
    project: Project,
    pipelineStatus: PipelineStatusResponse | null,
): RecommendedAction {
    const currentStage = getCurrentStage(project, pipelineStatus);

    switch (currentStage) {
        case 'ingestion':
            return {
                // ?import=auto opens the import wizard on arrival.
                path: stagePath(projectId, 'data', 'import=auto'),
                title: 'Import source data',
                description: 'Upload files or import from a remote dataset source.',
            };
        case 'cleaning':
            return {
                path: stagePath(projectId, 'cleaning'),
                title: 'Clean raw documents',
                description: 'Run cleaning rules and inspect rejected/error records.',
            };
        case 'gold_set':
            return {
                path: stagePath(projectId, 'goldset'),
                title: 'Build the answer key',
                description: 'Create trusted QA pairs to benchmark quality.',
            };
        case 'synthetic':
            return {
                path: stagePath(projectId, 'synthetic'),
                title: 'Generate synthetic examples',
                description: 'Expand coverage with teacher-generated Q&A if needed.',
            };
        case 'dataset_prep':
        case 'data_adapter_preview':
            return {
                path: stagePath(projectId, 'dataprep'),
                title: 'Prepare train/val/test splits',
                description: 'Normalize records into the training format your model expects.',
            };
        case 'tokenization':
            return {
                path: stagePath(projectId, 'tokenization'),
                title: 'Validate tokenization',
                description: 'Check sequence lengths and truncation before training.',
            };
        case 'training':
            if (!project.base_model_name) {
                return {
                    path: trainSectionPath(projectId, 'model'),
                    title: 'Configure model and hyperparameters',
                    description: 'Pick base model, runtime profile, and training preset.',
                };
            }
            return {
                path: stagePath(projectId, 'training'),
                title: 'Run and monitor training',
                description: 'Start a training run and track live metrics/logs.',
            };
        case 'evaluation':
            return {
                path: stagePath(projectId, 'eval'),
                title: 'Evaluate model quality',
                description: 'Run the eval and check which pass/fail rules passed.',
            };
        case 'compression':
            return {
                path: stagePath(projectId, 'compression'),
                title: 'Compress and quantize artifacts',
                description: 'Generate smaller deployment variants (LoRA merge/quantization).',
            };
        case 'export':
            return {
                path: stagePath(projectId, 'export'),
                title: 'Export deployment model',
                description: 'Produce serving-ready outputs (HF, GGUF, ONNX as available).',
            };
        case 'completed':
            return {
                path: trainSectionPath(projectId, 'pipeline-presets'),
                title: 'Automate repeated runs',
                description: 'Apply pipeline presets for repeatable experiments.',
            };
        default:
            return {
                path: projectPath(projectId),
                title: 'Open the overview',
                description: 'See where the project is and continue from the right step.',
            };
    }
}
