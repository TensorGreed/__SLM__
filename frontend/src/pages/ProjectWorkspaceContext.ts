import type { PipelineStatusResponse, Project } from '../types';

export interface ProjectWorkspaceContextValue {
    projectId: number;
    project: Project;
    pipelineStatus: PipelineStatusResponse | null;
    refreshPipelineStatus: () => Promise<void>;
}
