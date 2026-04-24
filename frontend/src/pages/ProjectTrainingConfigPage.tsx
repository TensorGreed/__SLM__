import { useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';

import AlignmentScaffoldPanel from '../components/training/AlignmentScaffoldPanel';
import TrainingPanel from '../components/training/TrainingPanel';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectTrainingConfigPage.css';

export default function ProjectTrainingConfigPage() {
    const navigate = useNavigate();
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
    const [configMode, setConfigMode] = useState<'essentials' | 'advanced'>('essentials');

    return (
        <div className="workspace-page">
            <nav className="training-config-breadcrumb" aria-label="Breadcrumb">
                <button
                    type="button"
                    className="training-config-breadcrumb-link"
                    onClick={() => navigate(`/project/${projectId}/pipeline/training`)}
                >
                    ← Pipeline / Training
                </button>
                <span className="training-config-breadcrumb-sep">/</span>
                <span className="training-config-breadcrumb-current">Training Config</span>
            </nav>
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Training Configurations</h2>
                    <p className="workspace-page-subtitle">
                        Configure the experiment here, then return to the Pipeline Training tab
                        to launch runs and watch live metrics.
                    </p>
                </div>
                <div className="training-config-header-actions">
                    <div className="training-config-mode-switch" role="tablist" aria-label="Training configuration mode">
                        <button
                            type="button"
                            className={`training-config-mode-btn ${configMode === 'essentials' ? 'active' : ''}`}
                            onClick={() => setConfigMode('essentials')}
                            role="tab"
                            aria-selected={configMode === 'essentials' ? 'true' : 'false'}
                        >
                            Essentials
                        </button>
                        <button
                            type="button"
                            className={`training-config-mode-btn ${configMode === 'advanced' ? 'active' : ''}`}
                            onClick={() => setConfigMode('advanced')}
                            role="tab"
                            aria-selected={configMode === 'advanced' ? 'true' : 'false'}
                        >
                            Advanced
                        </button>
                    </div>
                </div>
            </section>

            <TrainingPanel
                projectId={projectId}
                title="Create and Configure Experiment"
                forceCreateVisible
                hideExperimentList
                hideStepFooter
                setupMode={configMode}
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
            {configMode === 'advanced' && <AlignmentScaffoldPanel projectId={projectId} />}
        </div>
    );
}
