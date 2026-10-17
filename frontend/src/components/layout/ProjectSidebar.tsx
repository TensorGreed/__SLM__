/**
 * The workspace sidebar — six destinations (Epic U, phase U1):
 * Overview, Data, Train, Results, Ship, Settings. Plus the project name,
 * the command palette hint and a collapse toggle. Nothing is hidden by
 * beginner mode; every page is reachable through its destination or
 * Settings.
 */

import { useEffect, useState, type ReactNode } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
    ChevronsLeft,
    ChevronsRight,
    Database,
    FlaskConical,
    Home,
    Rocket,
    Search,
    Settings2,
    Sparkles,
} from 'lucide-react';

import { openCommandPalette } from './commandPaletteBridge';
import BrandMark from './BrandMark';
import { DESTINATION_LABEL, DESTINATION_ORDER, projectPath, type Destination } from '../../utils/workspaceRoutes';
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

const DESTINATION_ICON: Record<Destination, ReactNode> = {
    overview: <Home size={16} />,
    data: <Database size={16} />,
    train: <FlaskConical size={16} />,
    results: <Sparkles size={16} />,
    ship: <Rocket size={16} />,
    settings: <Settings2 size={16} />,
};

interface ProjectSidebarProps {
    projectId: number;
    projectName: string;
}

export default function ProjectSidebar({ projectId, projectName }: ProjectSidebarProps) {
    const navigate = useNavigate();
    const [isCollapsed, setIsCollapsed] = useState<boolean>(() => readCollapsedPref());

    useEffect(() => {
        applyBodyClass(isCollapsed);
        writeCollapsedPref(isCollapsed);
    }, [isCollapsed]);

    // Cmd-B / Ctrl-B toggles the sidebar (VS Code muscle memory).
    useEffect(() => {
        const onKey = (event: KeyboardEvent) => {
            if (!(event.metaKey || event.ctrlKey)) return;
            if (event.key.toLowerCase() !== 'b') return;
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

                <nav className="project-sidebar-nav" aria-label="Workspace">
                    {DESTINATION_ORDER.map((destination) => (
                        <NavLink
                            key={destination}
                            to={projectPath(projectId, destination)}
                            end={destination === 'overview'}
                            className={({ isActive }) => `workspace-nav-item ${isActive ? 'active' : ''}`}
                            title={DESTINATION_LABEL[destination]}
                            data-testid={`nav-${destination}`}
                        >
                            {DESTINATION_ICON[destination]}
                            <span className="nav-label">{DESTINATION_LABEL[destination]}</span>
                        </NavLink>
                    ))}
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

                <button
                    type="button"
                    className="sidebar-collapse-toggle"
                    onClick={() => setIsCollapsed((prev) => !prev)}
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
