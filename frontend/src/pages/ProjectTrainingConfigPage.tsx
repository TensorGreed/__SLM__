import { useNavigate, useOutletContext } from 'react-router-dom';

import AlignmentScaffoldPanel from '../components/training/AlignmentScaffoldPanel';
import TrainingPanel from '../components/training/TrainingPanel';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectTrainingConfigPage.css';

export default function ProjectTrainingConfigPage() {
    const navigate = useNavigate();
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();

    return (
        <div className="workspace-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Training Configurations</h2>
                    <p className="workspace-page-subtitle">
                        Configure base model, runtime, recipes, preflight, and hyperparameters.
                    </p>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => navigate(`/project/${projectId}/pipeline/training`)}
                >
                    Open Training Stage
                </button>
            </section>

            <TrainingPanel
                projectId={projectId}
                title="Create and Configure Experiment"
                forceCreateVisible
                hideExperimentList
                hideStepFooter
            />

            <section className="card training-config-link-card">
                <div>
                    <h3>Playground moved to dedicated page</h3>
                    <p>
                        Use Playground for prompt presets, runtime adapters, and feedback logging.
                    </p>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => navigate(`/project/${projectId}/playground`)}
                >
                    Open Playground
                </button>
            </section>
            <AlignmentScaffoldPanel projectId={projectId} />
        </div>
    );
}
