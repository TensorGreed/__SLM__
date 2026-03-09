import type { PipelineStage, PipelineStatusResponse, Project } from '../types';

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
    gold_set: 'Gold Set',
    synthetic: 'Synthetic Data',
    dataset_prep: 'Dataset Prep',
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

    if (!project.domain_pack_id && !project.domain_profile_id) {
        return {
            path: `/project/${projectId}/domain/packs`,
            title: 'Set domain context',
            description: 'Assign a domain pack/profile (or keep defaults) before deeper tuning.',
        };
    }

    switch (currentStage) {
        case 'ingestion':
            return {
                path: `/project/${projectId}/pipeline/data`,
                title: 'Import source data',
                description: 'Upload files or import from a remote dataset source.',
            };
        case 'cleaning':
            return {
                path: `/project/${projectId}/pipeline/cleaning`,
                title: 'Clean raw documents',
                description: 'Run cleaning rules and inspect rejected/error records.',
            };
        case 'gold_set':
            return {
                path: `/project/${projectId}/pipeline/goldset`,
                title: 'Build gold evaluation set',
                description: 'Create trusted QA pairs to benchmark quality.',
            };
        case 'synthetic':
            return {
                path: `/project/${projectId}/pipeline/synthetic`,
                title: 'Generate synthetic examples',
                description: 'Expand coverage with teacher-generated Q&A if needed.',
            };
        case 'dataset_prep':
        case 'data_adapter_preview':
            return {
                path: `/project/${projectId}/pipeline/dataprep`,
                title: 'Prepare train/val/test splits',
                description: 'Normalize records into the training format your model expects.',
            };
        case 'tokenization':
            return {
                path: `/project/${projectId}/pipeline/tokenization`,
                title: 'Validate tokenization',
                description: 'Check sequence lengths and truncation before training.',
            };
        case 'training':
            if (!project.base_model_name) {
                return {
                    path: `/project/${projectId}/training-config`,
                    title: 'Configure model and hyperparameters',
                    description: 'Pick base model, runtime profile, and training recipe.',
                };
            }
            return {
                path: `/project/${projectId}/pipeline/training`,
                title: 'Run and monitor training',
                description: 'Start a training run and track live metrics/logs.',
            };
        case 'evaluation':
            return {
                path: `/project/${projectId}/pipeline/eval`,
                title: 'Evaluate model quality',
                description: 'Run benchmark/eval pack and inspect gate results.',
            };
        case 'compression':
            return {
                path: `/project/${projectId}/pipeline/compression`,
                title: 'Compress and quantize artifacts',
                description: 'Generate smaller deployment variants (LoRA merge/quantization).',
            };
        case 'export':
            return {
                path: `/project/${projectId}/pipeline/export`,
                title: 'Export deployment model',
                description: 'Produce serving-ready outputs (HF, GGUF, ONNX as available).',
            };
        case 'completed':
            return {
                path: `/project/${projectId}/recipes`,
                title: 'Automate repeated runs',
                description: 'Apply recipe-driven workflows for repeatable experiments.',
            };
        default:
            return {
                path: `/project/${projectId}/guide`,
                title: 'Open project guide',
                description: 'Review the recommended flow and continue from the right step.',
            };
    }
}
