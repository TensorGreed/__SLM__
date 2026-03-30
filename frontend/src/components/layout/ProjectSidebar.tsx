import { useEffect, useMemo, useRef, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
    BookOpen,
    Bot,
    Boxes,
    Compass,
    FileCog,
    FolderTree,
    Home,
    Layers,
    Settings2,
    Sparkles,
    Workflow,
} from 'lucide-react';

import { PIPELINE_TABS } from '../../types';
import type { PipelineStatusResponse, TabKey } from '../../types';
import './ProjectSidebar.css';

interface ProjectSidebarProps {
    projectId: number;
    projectName: string;
    pipelineStatus: PipelineStatusResponse | null;
}

type RailKey = 'home' | 'pipeline' | 'training' | 'playground' | 'workflow' | 'domain';

const STAGE_ORDER = [
    'ingestion',
    'cleaning',
    'gold_set',
    'synthetic',
    'dataset_prep',
    'data_adapter_preview',
    'tokenization',
    'training',
    'evaluation',
    'compression',
    'export',
    'completed',
];

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

function isRailKey(value: unknown): value is RailKey {
    return value === 'home'
        || value === 'pipeline'
        || value === 'training'
        || value === 'playground'
        || value === 'workflow'
        || value === 'domain';
}

export default function ProjectSidebar({ projectId, projectName, pipelineStatus }: ProjectSidebarProps) {
    const location = useLocation();
    const navigate = useNavigate();
    const railHintRaw = (location.state as { sidebarRail?: unknown } | null)?.sidebarRail;
    const railHint = isRailKey(railHintRaw) ? railHintRaw : null;
    const lastNonWizardRailRef = useRef<RailKey>('home');
    const pipelineBasePath = `/project/${projectId}/pipeline`;
    const pipelineDataPath = `${pipelineBasePath}/data`;
    const pipelineTrainingPath = `${pipelineBasePath}/training`;

    const currentStageIndex = useMemo(
        () => (pipelineStatus ? getStageIndex(pipelineStatus.current_stage) : 0),
        [pipelineStatus],
    );

    const isGuideRoute = location.pathname === `/project/${projectId}/guide`;
    const isPipelineRoute =
        location.pathname === pipelineBasePath
        || location.pathname.startsWith(`${pipelineBasePath}/`);
    const isPipelineDataRoute = location.pathname === pipelineBasePath || location.pathname === pipelineDataPath;
    const isPipelineTrainingRoute = location.pathname === pipelineTrainingPath;
    const isWorkflowRoute = location.pathname === `/project/${projectId}/workflow`;
    const isRecipesRoute = location.pathname === `/project/${projectId}/recipes`;
    const isTrainingConfigRoute = location.pathname === `/project/${projectId}/training-config`;
    const isModelsRoute = location.pathname === `/project/${projectId}/models`;
    const isAdapterStudioRoute = location.pathname === `/project/${projectId}/adapter-studio`;
    const isPlaygroundRoute = location.pathname === `/project/${projectId}/playground`;
    const isDomainPacksRoute =
        location.pathname === `/project/${projectId}/domain/packs`
        || location.pathname === `/project/${projectId}/domain`;
    const isDomainProfilesRoute = location.pathname === `/project/${projectId}/domain/profiles`;
    const isWizardRoute = location.pathname === `/project/${projectId}/wizard`;

    const routeRailKey: RailKey = useMemo(() => {
        if (isTrainingConfigRoute || isPipelineTrainingRoute || isModelsRoute || isAdapterStudioRoute) return 'training';
        if (isPlaygroundRoute) return 'playground';
        if (isWorkflowRoute || isRecipesRoute) return 'workflow';
        if (isDomainPacksRoute || isDomainProfilesRoute) return 'domain';
        if (isPipelineRoute) return 'pipeline';
        return 'home';
    }, [
        isTrainingConfigRoute,
        isPipelineTrainingRoute,
        isModelsRoute,
        isAdapterStudioRoute,
        isPlaygroundRoute,
        isWorkflowRoute,
        isRecipesRoute,
        isDomainPacksRoute,
        isDomainProfilesRoute,
        isPipelineRoute,
    ]);

    useEffect(() => {
        if (!isWizardRoute) {
            lastNonWizardRailRef.current = routeRailKey;
        }
    }, [isWizardRoute, routeRailKey]);

    const selectedRailKey: RailKey = isWizardRoute
        ? (railHint ?? lastNonWizardRailRef.current)
        : routeRailKey;
    const isHomeWizardRoute = isWizardRoute && selectedRailKey === 'home';
    const isTrainingWizardRoute = isWizardRoute && selectedRailKey === 'training';

    const panelHeadingByRail: Record<RailKey, { kicker: string; title: string }> = {
        home: { kicker: 'Guided', title: 'Start and Discover' },
        pipeline: { kicker: 'Pipeline', title: 'Runs and Stages' },
        training: { kicker: 'Training', title: 'Model Configuration' },
        playground: { kicker: 'Playground', title: 'Prompt Testing' },
        workflow: { kicker: 'Automation', title: 'Recipes and Flows' },
        domain: { kicker: 'Domain', title: 'Packs and Profiles' },
    };

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

    const railItems: Array<{ key: RailKey; icon: ReactNode; title: string; onClick: () => void }> = [
        {
            key: 'home',
            icon: <Home size={16} />,
            title: 'Start',
            onClick: () => navigate(`/project/${projectId}/guide`),
        },
        {
            key: 'pipeline',
            icon: <FolderTree size={16} />,
            title: 'Pipeline',
            onClick: () => navigate(`/project/${projectId}/pipeline/data`),
        },
        {
            key: 'training',
            icon: <Settings2 size={16} />,
            title: 'Training',
            onClick: () => navigate(`/project/${projectId}/training-config`),
        },
        {
            key: 'playground',
            icon: <Bot size={16} />,
            title: 'Playground',
            onClick: () => navigate(`/project/${projectId}/playground`),
        },
        {
            key: 'workflow',
            icon: <Workflow size={16} />,
            title: 'Automation',
            onClick: () => navigate(`/project/${projectId}/workflow`),
        },
        {
            key: 'domain',
            icon: <Layers size={16} />,
            title: 'Domain',
            onClick: () => navigate(`/project/${projectId}/domain/packs`),
        },
    ];

    return (
        <aside className="project-sidebar">
            <div className="project-sidebar-rail">
                <button
                    className="rail-logo"
                    onClick={() => navigate('/')}
                    title="Back to BrewSLM projects"
                >
                    BS
                </button>
                <div className="project-sidebar-rail-nav">
                    {railItems.map((item) => (
                        <button
                            key={item.key}
                            className={`rail-item ${selectedRailKey === item.key ? 'active' : ''}`}
                            onClick={item.onClick}
                            title={item.title}
                        >
                            {item.icon}
                        </button>
                    ))}
                </div>
                <button
                    className="rail-item rail-item-bottom"
                    onClick={() => navigate(`/project/${projectId}/training-config`)}
                    title="Settings"
                >
                    <FileCog size={16} />
                </button>
            </div>

            <div className="project-sidebar-panel">
                <div className="project-sidebar-header">
                    <div className="project-sidebar-heading">
                        <span className="heading-kicker">{panelHeadingByRail[selectedRailKey].kicker}</span>
                        <span className="heading-title">{panelHeadingByRail[selectedRailKey].title}</span>
                    </div>
                    <span className="project-sidebar-collapse">«</span>
                </div>

                <div className="project-sidebar-project">
                    <span className="project-label">Project</span>
                    <span className="project-name" title={projectName}>{projectName}</span>
                </div>

                <nav className="project-sidebar-nav">
                    {selectedRailKey === 'home' && (
                        <>
                            <div className="nav-section-label">Data Source Discovery</div>
                            <button
                                className={`workspace-nav-item ${isGuideRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/guide`)}
                            >
                                <Compass size={15} />
                                <span className="nav-label">Start Here</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isHomeWizardRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/wizard`, { state: { sidebarRail: 'home' } })}
                            >
                                <Sparkles size={15} />
                                <span className="nav-label">Wizard Mode</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isPipelineDataRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/pipeline/data`)}
                            >
                                <FolderTree size={15} />
                                <span className="nav-label">Go to Data Pipeline</span>
                            </button>
                        </>
                    )}

                    {selectedRailKey === 'pipeline' && (
                        <>
                            <div className="nav-section-label">Data Pipeline</div>
                            <button
                                className={`workspace-nav-item ${isPipelineDataRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/pipeline/data`)}
                            >
                                <FolderTree size={15} />
                                <span className="nav-label">Runs</span>
                            </button>
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
                                        <span className="pipeline-subnav-leading">{unlocked ? tab.icon : '•'}</span>
                                        <span className="nav-label">{tab.label}</span>
                                        <span className={`nav-status-dot ${status}`} />
                                    </button>
                                );
                            })}
                        </>
                    )}

                    {selectedRailKey === 'training' && (
                        <>
                            <div className="nav-section-label">Classification and Entitlements</div>
                            <button
                                className={`workspace-nav-item ${isTrainingConfigRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/training-config`)}
                            >
                                <Settings2 size={15} />
                                <span className="nav-label">Configurations</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isModelsRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/models`)}
                            >
                                <Boxes size={15} />
                                <span className="nav-label">Base Model Registry</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isAdapterStudioRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/adapter-studio`)}
                            >
                                <Boxes size={15} />
                                <span className="nav-label">Adapter Studio</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isTrainingWizardRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/wizard`, { state: { sidebarRail: 'training' } })}
                            >
                                <Sparkles size={15} />
                                <span className="nav-label">Guided Setup</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isPipelineTrainingRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/pipeline/training`)}
                            >
                                <FolderTree size={15} />
                                <span className="nav-label">Training Stage</span>
                            </button>
                        </>
                    )}

                    {selectedRailKey === 'playground' && (
                        <>
                            <div className="nav-section-label">Validation and Vibe Check</div>
                            <button
                                className={`workspace-nav-item ${isPlaygroundRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/playground`)}
                            >
                                <Bot size={15} />
                                <span className="nav-label">Playground Runs</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isTrainingConfigRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/training-config`)}
                            >
                                <Settings2 size={15} />
                                <span className="nav-label">Model Configuration</span>
                            </button>
                        </>
                    )}

                    {selectedRailKey === 'workflow' && (
                        <>
                            <div className="nav-section-label">Automation</div>
                            <button
                                className={`workspace-nav-item ${isWorkflowRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/workflow`)}
                            >
                                <Workflow size={15} />
                                <span className="nav-label">Workflow Builder</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isRecipesRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/recipes`)}
                            >
                                <BookOpen size={15} />
                                <span className="nav-label">Recipes</span>
                            </button>
                        </>
                    )}

                    {selectedRailKey === 'domain' && (
                        <>
                            <div className="nav-section-label">Domain Controls</div>
                            <button
                                className={`workspace-nav-item ${isDomainPacksRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/domain/packs`)}
                            >
                                <Boxes size={15} />
                                <span className="nav-label">Domain Packs</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isDomainProfilesRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/domain/profiles`)}
                            >
                                <Layers size={15} />
                                <span className="nav-label">Domain Profiles</span>
                            </button>
                        </>
                    )}
                </nav>
            </div>
        </aside>
    );
}
