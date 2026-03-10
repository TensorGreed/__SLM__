import { useNavigate, useOutletContext } from 'react-router-dom';

import AlignmentScaffoldPanel from '../components/training/AlignmentScaffoldPanel';
import TrainingPanel from '../components/training/TrainingPanel';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectTrainingConfigPage() {
    const navigate = useNavigate();
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
                <div>
                    <h3 style={{ margin: 0 }}>Training Config</h3>
                    <p style={{ marginTop: 6, color: 'var(--text-secondary)' }}>
                        Configure base model, runtime, recipes, preflight, and hyperparameters.
                    </p>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => navigate(`/project/${projectId}/pipeline/training`)}
                >
                    Open Training Stage
                </button>
            </div>

            <TrainingPanel
                projectId={projectId}
                title="Create and Configure Experiment"
                forceCreateVisible
                hideExperimentList
                hideStepFooter
            />

            <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
                <div>
                    <h3 style={{ margin: 0 }}>Playground moved to dedicated page</h3>
                    <p style={{ marginTop: 6, color: 'var(--text-secondary)' }}>
                        Use Playground for prompt presets, runtime adapters, and feedback logging.
                    </p>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => navigate(`/project/${projectId}/playground`)}
                >
                    Open Playground
                </button>
            </div>
            <AlignmentScaffoldPanel projectId={projectId} />
        </div>
    );
}
