import { useOutletContext } from 'react-router-dom';

import PipelineProgress from '../components/dashboard/PipelineProgress';
import PipelineGraphEditor from '../components/pipeline/PipelineGraphEditor';
import PipelineGraphPreview from '../components/pipeline/PipelineGraphPreview';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectWorkflowPage.css';

export default function ProjectWorkflowPage() {
    const { projectId, project, pipelineStatus, refreshPipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const currentStage = pipelineStatus?.current_stage || project.pipeline_stage;

    return (
        <>
            {pipelineStatus && (
                <div className="card progress-card">
                    <PipelineProgress
                        stages={pipelineStatus.stages}
                        progressPercent={pipelineStatus.progress_percent}
                    />
                </div>
            )}
            <PipelineGraphPreview
                projectId={projectId}
                currentStage={currentStage}
                onPipelineUpdated={refreshPipelineStatus}
            />
            <PipelineGraphEditor
                projectId={projectId}
                currentStage={currentStage}
            />
        </>
    );
}
