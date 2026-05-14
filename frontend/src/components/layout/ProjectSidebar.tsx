import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
    Activity,
    BookOpen,
    Bot,
    Boxes,
    ChevronsLeft,
    ChevronsRight,
    ClipboardList,
    FileCode,
    FolderTree,
    Layers,
    Lock,
    PenSquare,
    Puzzle,
    Rocket,
    Search,
    Settings2,
    Sparkles,
    Unlock,
    Workflow,
} from 'lucide-react';

import { PIPELINE_TABS } from '../../types';
import type { PipelineStatusResponse, TabKey } from '../../types';
import { useProjectStore } from '../../stores/projectStore';
import api from '../../api/client';
import { openCommandPalette } from './commandPaletteBridge';
import BrandMark from './BrandMark';
import './ProjectSidebar.css';

const SIDEBAR_COLLAPSED_KEY = 'brewslm_sidebar_collapsed';

function readCollapsedPref(): boolean {
    if (typeof window === 'undefined') return false;
    try {
        return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === '1';
    } catch {
        return false;
    }
}

function writeCollapsedPref(value: boolean): void {
    if (typeof window === 'undefined') return;
    try {
        window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, value ? '1' : '0');
    } catch {
        // ignore
    }
}

function applyBodyClass(collapsed: boolean): void {
    if (typeof document === 'undefined') return;
    document.body.classList.toggle('sidebar-collapsed', collapsed);
}

interface ProjectSidebarProps {
    projectId: number;
    projectName: string;
    pipelineStatus: PipelineStatusResponse | null;
    beginnerMode?: boolean;
}

type RailKey = 'pipeline' | 'training' | 'workflow' | 'domain';

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
    return value === 'pipeline'
        || value === 'training'
        || value === 'workflow'
        || value === 'domain';
}

export default function ProjectSidebar({ projectId, projectName, pipelineStatus, beginnerMode }: ProjectSidebarProps) {
    const location = useLocation();
    const navigate = useNavigate();
    const { activeProject, setActiveProject } = useProjectStore();
    const isBeginner = Boolean(
        beginnerMode !== undefined ? beginnerMode : activeProject?.beginner_mode,
    );
    const [togglingBeginner, setTogglingBeginner] = useState(false);
    const [toggleError, setToggleError] = useState<string | null>(null);
    const [isCollapsed, setIsCollapsed] = useState<boolean>(() => readCollapsedPref());

    // Sync the body class + localStorage every time the state flips.
    useEffect(() => {
        applyBodyClass(isCollapsed);
        writeCollapsedPref(isCollapsed);
    }, [isCollapsed]);

    // Cmd-B / Ctrl-B toggles the sidebar (VS Code muscle memory).
    useEffect(() => {
        const onKey = (event: KeyboardEvent) => {
            if (!(event.metaKey || event.ctrlKey)) return;
            if (event.key.toLowerCase() !== 'b') return;
            // Don't fire while the user is typing in an input/textarea.
            const target = event.target as HTMLElement | null;
            if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
                return;
            }
            event.preventDefault();
            setIsCollapsed((prev) => !prev);
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, []);

    const toggleCollapsed = () => setIsCollapsed((prev) => !prev);
    const railHintRaw = (location.state as { sidebarRail?: unknown } | null)?.sidebarRail;
    const railHint = isRailKey(railHintRaw) ? railHintRaw : null;
    const lastNonWizardRailRef = useRef<RailKey>('pipeline');

    const handleToggleBeginnerMode = async () => {
        const enabling = !isBeginner;
        const message = enabling
            ? 'Switch to beginner mode? Recipes, Domain Packs, Domain Profiles, Workflow Builder, and the Extension Studio will be hidden to keep the workspace focused. You can leave beginner mode at any time.'
            : 'Leave beginner mode? You will regain access to Recipes, Domain Packs, Domain Profiles, Workflow Builder, and the Extension Studio. You can turn beginner mode back on from Project Settings at any time.';
        if (!window.confirm(message)) {
            return;
        }
        setTogglingBeginner(true);
        setToggleError(null);
        try {
            const response = await api.put(`/projects/${projectId}`, { beginner_mode: enabling });
            setActiveProject(response.data);
        } catch (err) {
            const detail =
                (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
                ?? (err instanceof Error ? err.message : enabling ? 'Unable to enter beginner mode.' : 'Unable to leave beginner mode.');
            setToggleError(typeof detail === 'string' ? detail : enabling ? 'Unable to enter beginner mode.' : 'Unable to leave beginner mode.');
        } finally {
            setTogglingBeginner(false);
        }
    };
    const pipelineBasePath = `/project/${projectId}/pipeline`;
    const pipelineDataPath = `${pipelineBasePath}/data`;
    const pipelineTrainingPath = `${pipelineBasePath}/training`;

    const currentStageIndex = useMemo(
        () => (pipelineStatus ? getStageIndex(pipelineStatus.current_stage) : 0),
        [pipelineStatus],
    );

    const isPipelineDataRoute = location.pathname === pipelineBasePath || location.pathname === pipelineDataPath;
    const isPipelineTrainingRoute = location.pathname === pipelineTrainingPath;
    const isWorkflowRoute = location.pathname === `/project/${projectId}/workflow`;
    const isRecipesRoute = location.pathname === `/project/${projectId}/recipes`;
    const isTrainingConfigRoute = location.pathname === `/project/${projectId}/training-config`;
    const isModelsRoute = location.pathname === `/project/${projectId}/models`;
    const isAdapterStudioRoute = location.pathname === `/project/${projectId}/adapter-studio`;
    const isAutopilotRoute = location.pathname === `/project/${projectId}/autopilot`;
    const isManifestRoute = location.pathname === `/project/${projectId}/manifest`;
    const isDeploymentsRoute = location.pathname === `/project/${projectId}/deployments`;
    const isObservabilityRoute = location.pathname === `/project/${projectId}/observability`;
    const isExtensionStudioRoute = location.pathname === `/project/${projectId}/extensions`;
    const isPlaygroundRoute = location.pathname === `/project/${projectId}/playground`;
    const isAnnotateRoute = location.pathname.startsWith(`/project/${projectId}/annotate`);
    const isDomainPacksRoute =
        location.pathname === `/project/${projectId}/domain/packs`
        || location.pathname === `/project/${projectId}/domain`;
    const isDomainProfilesRoute = location.pathname === `/project/${projectId}/domain/profiles`;
    const isWizardRoute = location.pathname === `/project/${projectId}/wizard`;

    const routeRailKey: RailKey = useMemo(() => {
        if (
            isTrainingConfigRoute
            || isPipelineTrainingRoute
            || isModelsRoute
            || isAdapterStudioRoute
            || isAutopilotRoute
            || isPlaygroundRoute
            || isDeploymentsRoute
            || isObservabilityRoute
            || isExtensionStudioRoute
        ) return 'training';
        if (isWorkflowRoute || isRecipesRoute || isManifestRoute) return 'workflow';
        if (isDomainPacksRoute || isDomainProfilesRoute) return 'domain';
        return 'pipeline';
    }, [
        isTrainingConfigRoute,
        isPipelineTrainingRoute,
        isModelsRoute,
        isAdapterStudioRoute,
        isAutopilotRoute,
        isPlaygroundRoute,
        isDeploymentsRoute,
        isObservabilityRoute,
        isExtensionStudioRoute,
        isWorkflowRoute,
        isRecipesRoute,
        isManifestRoute,
        isDomainPacksRoute,
        isDomainProfilesRoute,
    ]);

    useEffect(() => {
        if (!isWizardRoute) {
            lastNonWizardRailRef.current = routeRailKey;
        }
    }, [isWizardRoute, routeRailKey]);

    const selectedRailKey: RailKey = isWizardRoute
        ? (railHint ?? lastNonWizardRailRef.current)
        : routeRailKey;
    const isTrainingWizardRoute = isWizardRoute && selectedRailKey === 'training';

    const panelHeadingByRail: Record<RailKey, { kicker: string; title: string }> = {
        pipeline: { kicker: 'Pipeline', title: 'Runs and Stages' },
        training: { kicker: 'Training', title: 'Model Configuration' },
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
        ...(isBeginner
            ? []
            : [
                  {
                      key: 'workflow' as RailKey,
                      icon: <Workflow size={16} />,
                      title: 'Automation',
                      onClick: () => navigate(`/project/${projectId}/workflow`),
                  },
                  {
                      key: 'domain' as RailKey,
                      icon: <Layers size={16} />,
                      title: 'Domain',
                      onClick: () => navigate(`/project/${projectId}/domain/packs`),
                  },
              ]),
    ];

    return (
        <aside className={`project-sidebar ${isCollapsed ? 'is-collapsed' : ''}`}>
            <div className="project-sidebar-panel">
                <div className="project-sidebar-brand">
                    <button
                        type="button"
                        className="brand-logo"
                        onClick={() => navigate('/')}
                        title="Back to BrewSLM projects"
                    >
                        <span className="brand-logo-mark">
                            <BrandMark size={22} />
                        </span>
                        <span className="brand-logo-name">BrewSLM</span>
                    </button>
                </div>

                <div className="project-sidebar-project">
                    <span className="project-label">Project</span>
                    <span className="project-name" title={projectName}>{projectName}</span>
                </div>

                <div
                    className="project-sidebar-rail-strip"
                    aria-label="Workspace sections"
                >
                    {railItems.map((item) => (
                        <button
                            key={item.key}
                            type="button"
                            aria-pressed={selectedRailKey === item.key}
                            className={`rail-pill ${selectedRailKey === item.key ? 'active' : ''}`}
                            onClick={item.onClick}
                            title={item.title}
                        >
                            <span className="rail-pill-icon">{item.icon}</span>
                            <span className="rail-pill-label">{item.title}</span>
                        </button>
                    ))}
                </div>

                <div className="project-sidebar-section-label">
                    {panelHeadingByRail[selectedRailKey].title}
                </div>

                <nav className="project-sidebar-nav">
                    {selectedRailKey === 'pipeline' && (
                        <>
                            <div className="nav-section-label">Data Pipeline</div>
                            <button
                                className={`workspace-nav-item ${isPipelineDataRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/pipeline/data`)}
                                title="Runs"
                            >
                                <FolderTree size={15} />
                                <span className="nav-label">Runs</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isAnnotateRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/annotate`)}
                                title="Annotation"
                            >
                                <PenSquare size={15} />
                                <span className="nav-label">Annotation</span>
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
                                title="Configurations"
                            >
                                <Settings2 size={15} />
                                <span className="nav-label">Configurations</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isModelsRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/models`)}
                                title="Base Model Registry"
                            >
                                <Boxes size={15} />
                                <span className="nav-label">Base Model Registry</span>
                            </button>
                            {!isBeginner && (
                                <button
                                    className={`workspace-nav-item ${isAdapterStudioRoute ? 'active' : ''}`}
                                    onClick={() => navigate(`/project/${projectId}/adapter-studio`)}
                                    title="Adapter Studio"
                                >
                                    <Boxes size={15} />
                                    <span className="nav-label">Adapter Studio</span>
                                </button>
                            )}
                            <button
                                className={`workspace-nav-item ${isAutopilotRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/autopilot`)}
                                title="Autopilot Planner"
                            >
                                <ClipboardList size={15} />
                                <span className="nav-label">Autopilot Planner</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isPlaygroundRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/playground`)}
                                title="Playground"
                            >
                                <Bot size={15} />
                                <span className="nav-label">Playground</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isDeploymentsRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/deployments`)}
                                title="Deployments"
                            >
                                <Rocket size={15} />
                                <span className="nav-label">Deployments</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isObservabilityRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/observability`)}
                                title="Observability"
                            >
                                <Activity size={15} />
                                <span className="nav-label">Observability</span>
                            </button>
                            {!isBeginner && (
                                <button
                                    className={`workspace-nav-item ${isExtensionStudioRoute ? 'active' : ''}`}
                                    onClick={() => navigate(`/project/${projectId}/extensions`)}
                                    title="Extension Studio"
                                >
                                    <Puzzle size={15} />
                                    <span className="nav-label">Extension Studio</span>
                                </button>
                            )}
                            <button
                                className={`workspace-nav-item ${isTrainingWizardRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/wizard`, { state: { sidebarRail: 'training' } })}
                                title="Guided Setup"
                            >
                                <Sparkles size={15} />
                                <span className="nav-label">Guided Setup</span>
                            </button>
                        </>
                    )}

                    {selectedRailKey === 'workflow' && !isBeginner && (
                        <>
                            <div className="nav-section-label">Automation</div>
                            <button
                                className={`workspace-nav-item ${isWorkflowRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/workflow`)}
                                title="Workflow Builder"
                            >
                                <Workflow size={15} />
                                <span className="nav-label">Workflow Builder</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isRecipesRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/recipes`)}
                                title="Recipes"
                            >
                                <BookOpen size={15} />
                                <span className="nav-label">Recipes</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isManifestRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/manifest`)}
                                title="Pipeline as Code"
                            >
                                <FileCode size={15} />
                                <span className="nav-label">Pipeline as Code</span>
                            </button>
                        </>
                    )}

                    {selectedRailKey === 'domain' && !isBeginner && (
                        <>
                            <div className="nav-section-label">Domain Controls</div>
                            <button
                                className={`workspace-nav-item ${isDomainPacksRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/domain/packs`)}
                                title="Domain Packs"
                            >
                                <Boxes size={15} />
                                <span className="nav-label">Domain Packs</span>
                            </button>
                            <button
                                className={`workspace-nav-item ${isDomainProfilesRoute ? 'active' : ''}`}
                                onClick={() => navigate(`/project/${projectId}/domain/profiles`)}
                                title="Domain Profiles"
                            >
                                <Layers size={15} />
                                <span className="nav-label">Domain Profiles</span>
                            </button>
                        </>
                    )}
                </nav>

                <button
                    type="button"
                    className="sidebar-cmdk-hint"
                    onClick={() => openCommandPalette()}
                    aria-label="Open command palette"
                    title="Open command palette (⌘K)"
                >
                    <Search size={13} className="sidebar-cmdk-hint-icon" />
                    <span className="sidebar-cmdk-hint-label">Quick search</span>
                    <span className="sidebar-cmdk-hint-key">
                        <kbd>⌘</kbd>
                        <kbd>K</kbd>
                    </span>
                </button>

                {isBeginner ? (
                    <div className="project-sidebar-beginner">
                        <div className="beginner-badge" role="note" aria-label="Beginner mode active">
                            Beginner mode
                        </div>
                        <p className="beginner-note">
                            Recipes, Domain Packs, Workflow, and advanced studios are hidden to keep the workspace focused.
                        </p>
                        <button
                            type="button"
                            className="workspace-nav-item leave-beginner-button"
                            onClick={handleToggleBeginnerMode}
                            disabled={togglingBeginner}
                            title="Leave beginner mode"
                        >
                            <Unlock size={15} />
                            <span className="nav-label">
                                {togglingBeginner ? 'Leaving beginner mode…' : 'Leave beginner mode'}
                            </span>
                        </button>
                        {toggleError && <div className="beginner-error" role="alert">{toggleError}</div>}
                    </div>
                ) : (
                    <div className="project-sidebar-beginner project-sidebar-beginner-off">
                        <button
                            type="button"
                            className="workspace-nav-item enter-beginner-button"
                            onClick={handleToggleBeginnerMode}
                            disabled={togglingBeginner}
                            title="Enter beginner mode"
                        >
                            <Lock size={15} />
                            <span className="nav-label">
                                {togglingBeginner ? 'Entering beginner mode…' : 'Enter beginner mode'}
                            </span>
                        </button>
                        {toggleError && <div className="beginner-error" role="alert">{toggleError}</div>}
                    </div>
                )}

                <button
                    type="button"
                    className="sidebar-collapse-toggle"
                    onClick={toggleCollapsed}
                    aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                    title={isCollapsed ? 'Expand sidebar (⌘B)' : 'Collapse sidebar (⌘B)'}
                >
                    {isCollapsed ? <ChevronsRight size={14} /> : <ChevronsLeft size={14} />}
                    <span className="sidebar-collapse-toggle-label">
                        {isCollapsed ? 'Expand' : 'Collapse'}
                    </span>
                </button>
            </div>
        </aside>
    );
}
