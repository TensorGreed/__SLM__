import { useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import { PIPELINE_TABS } from '../../types';
import type { PipelineStatusResponse, TabKey } from '../../types';
import './ProjectSidebar.css';

interface ProjectSidebarProps {
    projectId: number;
    projectName: string;
    pipelineStatus: PipelineStatusResponse | null;
}

const STAGE_ORDER = ['ingestion', 'cleaning', 'gold_set', 'synthetic', 'dataset_prep', 'data_adapter_preview', 'tokenization', 'training', 'evaluation', 'compression', 'export', 'completed'];

const TAB_PREREQ_INDEX: Record<TabKey, number> = {
    data: 0,
    cleaning: 1,
    goldset: 1,
    synthetic: 2,
    dataprep: 2,
    tokenization: 2,
    training: 3,
    eval: 4,
    compression: 4,
    export: 4,
};

function getStageIndex(stage: string): number {
    const idx = STAGE_ORDER.indexOf(stage);
    return idx >= 0 ? idx : 0;
}

export default function ProjectSidebar({ projectId, projectName, pipelineStatus }: ProjectSidebarProps) {
    const location = useLocation();
    const navigate = useNavigate();

    const currentStageIndex = useMemo(
        () => (pipelineStatus ? getStageIndex(pipelineStatus.current_stage) : 0),
        [pipelineStatus],
    );

    const isGuideRoute = location.pathname === `/project/${projectId}/guide`;
    const isPipelineRoute =
        location.pathname === `/project/${projectId}/pipeline`
        || location.pathname.startsWith(`/project/${projectId}/pipeline/`);
    const isWorkflowRoute = location.pathname === `/project/${projectId}/workflow`;
    const isRecipesRoute = location.pathname === `/project/${projectId}/recipes`;
    const isTrainingConfigRoute = location.pathname === `/project/${projectId}/training-config`;
    const isPlaygroundRoute = location.pathname === `/project/${projectId}/playground`;
    const isDomainPacksRoute =
        location.pathname === `/project/${projectId}/domain/packs`
        || location.pathname === `/project/${projectId}/domain`;
    const isDomainProfilesRoute = location.pathname === `/project/${projectId}/domain/profiles`;

    const getStageStatus = (stageKey: string) => {
        if (!pipelineStatus) {
            return 'pending';
        }
        const found = pipelineStatus.stages.find((stage) => stage.stage === stageKey);
        return found?.status || 'pending';
    };

    const isTabUnlocked = (tabKey: TabKey): boolean => {
        const requiredIndex = TAB_PREREQ_INDEX[tabKey];
        return currentStageIndex >= requiredIndex;
    };

    return (
        <aside className="project-sidebar">
            <div
                className="project-sidebar-header"
                onClick={() => navigate('/')}
                role="button"
                tabIndex={0}
            >
                <div className="project-sidebar-logo">
                    <span className="logo-icon">◈</span>
                    <span className="logo-text">SLM Platform</span>
                </div>
            </div>

            <div className="project-sidebar-project">
                <span className="project-label">Project</span>
                <span className="project-name">{projectName}</span>
            </div>

            <nav className="project-sidebar-nav">
                <div className="nav-section-label">Guided Flow</div>
                <button
                    className={`workspace-nav-item ${isGuideRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/guide`)}
                >
                    <span className="nav-icon">1</span>
                    <span className="nav-copy">
                        <span className="nav-label">Start Here</span>
                        <span className="nav-caption">Recommended next step for your project.</span>
                    </span>
                </button>
                <button
                    className={`workspace-nav-item ${isPipelineRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/pipeline/data`)}
                >
                    <span className="nav-icon">2</span>
                    <span className="nav-copy">
                        <span className="nav-label">Data Pipeline</span>
                        <span className="nav-caption">Ingest, clean, prepare, tokenize, train.</span>
                    </span>
                </button>
                <button
                    className={`workspace-nav-item ${isTrainingConfigRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/training-config`)}
                >
                    <span className="nav-icon">3</span>
                    <span className="nav-copy">
                        <span className="nav-label">Training Setup</span>
                        <span className="nav-caption">Model, hyperparameters, runtime profile.</span>
                    </span>
                </button>
                <button
                    className={`workspace-nav-item ${isPlaygroundRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/playground`)}
                >
                    <span className="nav-icon">PG</span>
                    <span className="nav-copy">
                        <span className="nav-label">Playground</span>
                        <span className="nav-caption">Prompt presets, runtime adapters, and eval logs.</span>
                    </span>
                </button>

                <div className="nav-section-label">Automation</div>
                <button
                    className={`workspace-nav-item ${isWorkflowRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/workflow`)}
                >
                    <span className="nav-icon">⚙</span>
                    <span className="nav-copy">
                        <span className="nav-label">Workflow Builder</span>
                        <span className="nav-caption">Canvas DAG editor and workflow runs.</span>
                    </span>
                </button>
                <button
                    className={`workspace-nav-item ${isRecipesRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/recipes`)}
                >
                    <span className="nav-icon">🧪</span>
                    <span className="nav-copy">
                        <span className="nav-label">Pipeline Recipes</span>
                        <span className="nav-caption">Reusable end-to-end execution templates.</span>
                    </span>
                </button>

                <div className="nav-section-label">Domain</div>
                <button
                    className={`workspace-nav-item ${isDomainPacksRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/domain/packs`)}
                >
                    <span className="nav-icon">🧩</span>
                    <span className="nav-copy">
                        <span className="nav-label">Domain Packs</span>
                        <span className="nav-caption">Evaluation and policy defaults.</span>
                    </span>
                </button>
                <button
                    className={`workspace-nav-item ${isDomainProfilesRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/domain/profiles`)}
                >
                    <span className="nav-icon">📘</span>
                    <span className="nav-copy">
                        <span className="nav-label">Domain Profiles</span>
                        <span className="nav-caption">Task intent and quality bar definitions.</span>
                    </span>
                </button>

                {isPipelineRoute && (
                    <>
                        <div className="nav-section-label submenu-label">Pipeline Stages</div>
                        {PIPELINE_TABS.map((tab) => {
                            const status = getStageStatus(tab.stage);
                            const unlocked = isTabUnlocked(tab.key);
                            const active = location.pathname === `/project/${projectId}/pipeline/${tab.key}`;
                            return (
                                <button
                                    key={tab.key}
                                    className={`pipeline-subnav-item ${active ? 'active' : ''}`}
                                    onClick={() => {
                                        if (unlocked) {
                                            navigate(`/project/${projectId}/pipeline/${tab.key}`);
                                        }
                                    }}
                                    disabled={!unlocked}
                                    title={unlocked ? tab.label : 'Complete earlier steps first'}
                                >
                                    <span>{unlocked ? tab.icon : '🔒'}</span>
                                    <span className="nav-label">{tab.label}</span>
                                    <span className={`nav-status-dot ${status}`} />
                                </button>
                            );
                        })}
                    </>
                )}
            </nav>

            <div className="project-sidebar-footer">
                <div className="footer-version">v0.1.0</div>
            </div>
        </aside>
    );
}
