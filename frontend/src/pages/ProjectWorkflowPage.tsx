import { useOutletContext } from 'react-router-dom';

import PipelineProgress from '../components/dashboard/PipelineProgress';
import PipelineGraphEditor from '../components/pipeline/PipelineGraphEditor';
import PipelineGraphPreview from '../components/pipeline/PipelineGraphPreview';
import WorkflowRunMonitor from '../components/pipeline/WorkflowRunMonitor';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectWorkflowPage.css';

export default function ProjectWorkflowPage() {
    const { projectId, project, pipelineStatus, refreshPipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const currentStage = pipelineStatus?.current_stage || project.pipeline_stage;

    return (
        <div className="workspace-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Workflow Builder</h2>
                    <p className="workspace-page-subtitle">
                        Visualize the pipeline graph, orchestrate transitions, and monitor workflow execution.
                    </p>
                </div>
            </section>
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
            <WorkflowRunMonitor
                projectId={projectId}
                currentStage={currentStage}
            />
            <PipelineGraphEditor
                projectId={projectId}
                currentStage={currentStage}
            />
        </div>
    );
}
