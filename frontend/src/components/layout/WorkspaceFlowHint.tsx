import { useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import type { PipelineStatusResponse, Project } from '../../types';
import { getRecommendedAction, PIPELINE_STAGE_LABEL } from '../../utils/flowGuide';
import './WorkspaceFlowHint.css';

interface WorkspaceFlowHintProps {
    projectId: number;
    project: Project;
    pipelineStatus: PipelineStatusResponse | null;
}

export default function WorkspaceFlowHint({ projectId, project, pipelineStatus }: WorkspaceFlowHintProps) {
    const location = useLocation();
    const navigate = useNavigate();

    const currentStage = pipelineStatus?.current_stage || project.pipeline_stage;
    const recommended = useMemo(
        () => getRecommendedAction(projectId, project, pipelineStatus),
        [projectId, project, pipelineStatus],
    );

    if (location.pathname.endsWith('/guide')) {
        return null;
    }

    const alreadyOnTarget = location.pathname.startsWith(recommended.path);

    return (
        <section className="card workspace-flow-hint">
            <div className="workspace-flow-hint-meta">
                <span className="workspace-flow-hint-label">Current Stage</span>
                <strong>{PIPELINE_STAGE_LABEL[currentStage]}</strong>
                <span className="workspace-flow-hint-progress">
                    {pipelineStatus?.progress_percent ?? 0}% complete
                </span>
            </div>
            <div className="workspace-flow-hint-next">
                <h4>Next: {recommended.title}</h4>
                <p>{recommended.description}</p>
            </div>
            {!alreadyOnTarget && (
                <button className="btn btn-secondary" onClick={() => navigate(recommended.path)}>
                    Continue
                </button>
            )}
        </section>
    );
}
