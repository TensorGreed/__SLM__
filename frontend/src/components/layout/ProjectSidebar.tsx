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

const STAGE_ORDER = ['ingestion', 'cleaning', 'gold_set', 'synthetic', 'dataset_prep', 'tokenization', 'training', 'evaluation', 'compression', 'export', 'completed'];

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

    const isPipelineRoute = location.pathname.startsWith(`/project/${projectId}/pipeline/`);
    const isWorkflowRoute = location.pathname === `/project/${projectId}/workflow`;
    const isRecipesRoute = location.pathname === `/project/${projectId}/recipes`;
    const isTrainingConfigRoute = location.pathname === `/project/${projectId}/training-config`;
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
                <div className="nav-section-label">Workspace</div>
                <button
                    className={`workspace-nav-item ${isPipelineRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/pipeline/data`)}
                >
                    <span className="nav-icon">🧭</span>
                    <span className="nav-label">Pipeline</span>
                </button>
                <button
                    className={`workspace-nav-item ${isTrainingConfigRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/training-config`)}
                >
                    <span className="nav-icon">🛠️</span>
                    <span className="nav-label">Training Config</span>
                </button>
                <button
                    className={`workspace-nav-item ${isWorkflowRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/workflow`)}
                >
                    <span className="nav-icon">🕸️</span>
                    <span className="nav-label">Workflow Graph</span>
                </button>
                <button
                    className={`workspace-nav-item ${isRecipesRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/recipes`)}
                >
                    <span className="nav-icon">🧪</span>
                    <span className="nav-label">Pipeline Recipes</span>
                </button>
                <button
                    className={`workspace-nav-item ${isDomainPacksRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/domain/packs`)}
                >
                    <span className="nav-icon">🧩</span>
                    <span className="nav-label">Domain Packs</span>
                </button>
                <button
                    className={`workspace-nav-item ${isDomainProfilesRoute ? 'active' : ''}`}
                    onClick={() => navigate(`/project/${projectId}/domain/profiles`)}
                >
                    <span className="nav-icon">📘</span>
                    <span className="nav-label">Domain Profiles</span>
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
