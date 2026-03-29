import { useMemo, useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';

import api from '../api/client';
import { useProjectStore } from '../stores/projectStore';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import { getPipelineStageIndex, getRecommendedAction, PIPELINE_STAGE_LABEL } from '../utils/flowGuide';
import './ProjectGuidePage.css';

interface GuideStep {
    id: string;
    title: string;
    detail: string;
    path: string;
    complete: boolean;
}

export default function ProjectGuidePage() {
    const navigate = useNavigate();
    const { setActiveProject } = useProjectStore();
    const { projectId, project, pipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const [toggleLoading, setToggleLoading] = useState(false);

    const currentStage = pipelineStatus?.current_stage || project.pipeline_stage;
    const stageIndex = getPipelineStageIndex(currentStage);
    const recommended = useMemo(
        () => getRecommendedAction(projectId, project, pipelineStatus),
        [projectId, project, pipelineStatus],
    );

    const steps = useMemo<GuideStep[]>(() => {
        return [
            {
                id: 'domain',
                title: 'Set domain context',
                detail: 'Assign domain pack/profile or keep default generic behavior.',
                path: `/project/${projectId}/domain/packs`,
                complete: Boolean(project.domain_pack_id || project.domain_profile_id),
            },
            {
                id: 'ingest',
                title: 'Ingest source data',
                detail: 'Import files or remote datasets and process them into documents.',
                path: `/project/${projectId}/pipeline/data`,
                complete: stageIndex >= 1,
            },
            {
                id: 'prepare',
                title: 'Prepare training dataset',
                detail: 'Clean, label, generate synthetic data, split, and tokenize.',
                path: `/project/${projectId}/pipeline/cleaning`,
                complete: stageIndex >= 7,
            },
            {
                id: 'trainconfig',
                title: 'Configure training',
                detail: 'Choose model, runtime profile, hyperparameters, and recipe.',
                path: `/project/${projectId}/training-config`,
                complete: Boolean(project.base_model_name),
            },
            {
                id: 'train',
                title: 'Run training',
                detail: 'Launch experiment and monitor epochs/losses in real time.',
                path: `/project/${projectId}/pipeline/training`,
                complete: stageIndex >= 8,
            },
            {
                id: 'ship',
                title: 'Evaluate and ship',
                detail: 'Run evaluation gates, quantize/compress, and export artifacts.',
                path: `/project/${projectId}/pipeline/eval`,
                complete: stageIndex >= 11,
            },
        ];
    }, [projectId, project.domain_pack_id, project.domain_profile_id, project.base_model_name, stageIndex]);

    const firstIncompleteIndex = steps.findIndex((step) => !step.complete);

    const toggleBeginnerMode = async () => {
        setToggleLoading(true);
        try {
            const res = await api.put(`/projects/${projectId}`, {
                beginner_mode: !project.beginner_mode,
            });
            setActiveProject(res.data);
        } finally {
            setToggleLoading(false);
        }
    };

    return (
        <div className="project-guide-page workspace-page">
            <section className="card project-guide-hero">
                <div>
                    <h3>Start Here</h3>
                    <p>
                        Use this as your control center. It shows where your project is and what to do next.
                    </p>
                </div>
                <div className="project-guide-stage">
                    <span className="project-guide-stage-label">Current Stage</span>
                    <strong>{PIPELINE_STAGE_LABEL[currentStage]}</strong>
                    <span className="badge badge-info">{pipelineStatus?.progress_percent ?? 0}% complete</span>
                </div>
            </section>

            <section className="card project-guide-next">
                <div>
                    <h4>Recommended Next Action</h4>
                    <p>{recommended.description}</p>
                </div>
                <button className="btn btn-primary" onClick={() => navigate(recommended.path)}>
                    {recommended.title}
                </button>
            </section>

            <section className="card project-guide-beginner">
                <div>
                    <h4>Beginner Mode</h4>
                    <p>
                        {project.beginner_mode
                            ? 'Beginner Mode is enabled for this project. Use the guided wizard to refine your Domain Blueprint and launch safely.'
                            : 'Enable beginner-guided onboarding by creating a Domain Blueprint from plain-English intent and examples.'}
                    </p>
                    {project.active_domain_blueprint_version && (
                        <span className="badge badge-success">
                            Active Blueprint v{project.active_domain_blueprint_version}
                        </span>
                    )}
                </div>
                <button className="btn btn-secondary" onClick={() => navigate(`/project/${projectId}/wizard`)}>
                    Open Beginner Wizard
                </button>
                <button className="btn btn-ghost" onClick={toggleBeginnerMode} disabled={toggleLoading}>
                    {toggleLoading
                        ? 'Updating...'
                        : project.beginner_mode
                            ? 'Disable Beginner Mode'
                            : 'Enable Beginner Mode'}
                </button>
            </section>

            <section className="project-guide-steps">
                {steps.map((step, idx) => {
                    const isNow = !step.complete && idx === (firstIncompleteIndex >= 0 ? firstIncompleteIndex : steps.length - 1);
                    return (
                        <article
                            key={step.id}
                            className={`card project-guide-step ${step.complete ? 'done' : ''} ${isNow ? 'active' : ''}`}
                        >
                            <div className="project-guide-step-head">
                                <span className="project-guide-step-index">{idx + 1}</span>
                                <span className={`badge ${step.complete ? 'badge-success' : isNow ? 'badge-warning' : 'badge-info'}`}>
                                    {step.complete ? 'Done' : isNow ? 'Now' : 'Later'}
                                </span>
                            </div>
                            <h5>{step.title}</h5>
                            <p>{step.detail}</p>
                            <button className="btn btn-secondary" onClick={() => navigate(step.path)}>
                                Open
                            </button>
                        </article>
                    );
                })}
            </section>

            <section className="card project-guide-tools">
                <h4>Advanced Tools</h4>
                <div className="project-guide-tools-actions">
                    <button className="btn btn-ghost" onClick={() => navigate(`/project/${projectId}/workflow`)}>
                        Workflow Builder
                    </button>
                    <button className="btn btn-ghost" onClick={() => navigate(`/project/${projectId}/recipes`)}>
                        Pipeline Recipes
                    </button>
                    <button className="btn btn-ghost" onClick={() => navigate(`/project/${projectId}/domain/profiles`)}>
                        Domain Profiles
                    </button>
                </div>
            </section>
        </div>
    );
}
