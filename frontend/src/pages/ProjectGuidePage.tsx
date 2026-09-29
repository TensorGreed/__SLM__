/**
 * Progress guide page with achievement tracking and recommended next actions for pipeline advancement.
 */

import { useMemo, useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';

import api from '../api/client';
import { useProjectStore } from '../stores/projectStore';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import { getPipelineStageIndex, getRecommendedAction, PIPELINE_STAGE_LABEL } from '../utils/flowGuide';
import { useProgressionState } from '../components/gamification/useProgressionPoll';
import QuickstartCard from '../components/guide/QuickstartCard';
import './ProjectGuidePage.css';

// Map each guide step id to the achievement id that, when unlocked,
// earns the step its Lab Journal stamp. ``domain`` + ``trainconfig``
// don't have direct first-time achievements today (no RunEvent feeds
// them) — they just render unstamped.
const STAMP_ACHIEVEMENT_BY_STEP: Record<string, string | undefined> = {
    ingest: 'first_ingest',
    prepare: 'first_clean',
    train: 'first_train',
    ship: 'first_export',
};

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
    const { projectId, project, pipelineStatus, refreshPipelineStatus } =
        useOutletContext<ProjectWorkspaceContextValue>();
    const [toggleLoading, setToggleLoading] = useState(false);

    const currentStage = pipelineStatus?.current_stage || project.pipeline_stage;
    const stageIndex = getPipelineStageIndex(currentStage);
    const recommended = useMemo(
        () => getRecommendedAction(projectId, project, pipelineStatus),
        [projectId, project, pipelineStatus],
    );
    const progression = useProgressionState();
    const unlockedIds = useMemo(
        () => new Set(progression?.achievements_unlocked ?? []),
        [progression],
    );

    const steps = useMemo<GuideStep[]>(() => {
        const all: GuideStep[] = [
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
        // Domain packs/profiles are hidden in beginner mode — not a step.
        return project.beginner_mode ? all.filter((step) => step.id !== 'domain') : all;
    }, [projectId, project.domain_pack_id, project.domain_profile_id, project.base_model_name, project.beginner_mode, stageIndex]);

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
                    <h3>Coach · your plan</h3>
                    <p>
                        Where your project is, the one thing to do next, and every step to a model you can
                        chat with. The Coach bar at the top of each page follows you through them.
                    </p>
                </div>
                <div className="project-guide-stage">
                    <span className="project-guide-stage-label">Current Stage</span>
                    <strong>{PIPELINE_STAGE_LABEL[currentStage]}</strong>
                    <span className="badge badge-info">{pipelineStatus?.progress_percent ?? 0}% complete</span>
                </div>
            </section>

            <QuickstartCard
                projectId={projectId}
                hasBaseModel={Boolean(project.base_model_name)}
                initialDismissedNudges={
                    project.quickstart_tour_state?.dismissed_nudges ?? []
                }
                onRefresh={() => {
                    void refreshPipelineStatus();
                }}
            />

            <section className="card project-guide-next">
                <div>
                    <h4>Next step</h4>
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
                            ? 'Beginner Mode is on: advanced tools are hidden and the Coach walks you through each step. Autopilot can run the whole path for you.'
                            : 'Beginner Mode is off: every tool is visible. Turn it on to hide the advanced surfaces and get step-by-step coaching.'}
                    </p>
                    {project.active_domain_blueprint_version && (
                        <span className="badge badge-success">
                            Active Blueprint v{project.active_domain_blueprint_version}
                        </span>
                    )}
                </div>
                <button className="btn btn-secondary" onClick={() => navigate(`/project/${projectId}/wizard`)}>
                    Open Autopilot
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
                    const stampAchievement = STAMP_ACHIEVEMENT_BY_STEP[step.id];
                    const stamped = stampAchievement
                        ? unlockedIds.has(stampAchievement)
                        : false;
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
                                {stamped && (
                                    <span
                                        className="terminal-glow"
                                        title="Lab Journal: achievement unlocked"
                                        aria-label="Lab Journal stamp"
                                        data-testid={`guide-stamp-${step.id}`}
                                        style={{
                                            fontFamily: 'var(--font-mono)',
                                            fontSize: '0.85rem',
                                            marginLeft: 'auto',
                                            letterSpacing: '0.04em',
                                        }}
                                    >
                                        ▣
                                    </span>
                                )}
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
