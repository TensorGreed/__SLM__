import api from '../api/client';
import type { PipelineGraphContractResponse } from '../types';

export interface WorkflowStagePrefill {
    stage: string;
    config: Record<string, unknown>;
}

function normalizeStageToken(value: unknown): string {
    return String(value || '').trim().toLowerCase();
}

function normalizeConfig(value: unknown): Record<string, unknown> {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return {};
    }
    return value as Record<string, unknown>;
}

export async function loadWorkflowStagePrefill(
    projectId: number,
    stageCandidates: string[],
): Promise<WorkflowStagePrefill | null> {
    const requestedStages = stageCandidates
        .map((item) => normalizeStageToken(item))
        .filter(Boolean);
    if (requestedStages.length === 0) {
        return null;
    }
    try {
        const res = await api.get<PipelineGraphContractResponse>(
            `/projects/${projectId}/pipeline/graph/contract`,
        );
        const nodes = Array.isArray(res.data?.graph?.nodes) ? res.data.graph.nodes : [];
        for (const stage of requestedStages) {
            const matchedNode = nodes.find((node) => normalizeStageToken(node.stage) === stage);
            if (!matchedNode) {
                continue;
            }
            const cfg = normalizeConfig(matchedNode.config);
            if (Object.keys(cfg).length === 0) {
                continue;
            }
            return {
                stage,
                config: cfg,
            };
        }
    } catch {
        return null;
    }
    return null;
}
