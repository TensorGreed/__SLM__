import { useProjectStore } from '../../stores/projectStore';
import { PIPELINE_TABS } from '../../types';
import type { TabKey } from '../../types';
import './Sidebar.css';

interface SidebarProps {
    projectName?: string;
    onNavigateHome: () => void;
}

export default function Sidebar({ projectName, onNavigateHome }: SidebarProps) {
    const { activeTab, setActiveTab, pipelineStatus } = useProjectStore();

    const getStageStatus = (stageKey: string) => {
        if (!pipelineStatus) return 'pending';
        const found = pipelineStatus.stages.find((s) => s.stage === stageKey);
        return found?.status || 'pending';
    };

    return (
        <aside className="sidebar">
            <div className="sidebar-header" onClick={onNavigateHome} role="button" tabIndex={0}>
                <div className="sidebar-logo">
                    <span className="logo-icon">◈</span>
                    <span className="logo-text">SLM Platform</span>
                </div>
            </div>

            {projectName && (
                <div className="sidebar-project">
                    <span className="project-label">Project</span>
                    <span className="project-name">{projectName}</span>
                </div>
            )}

            <nav className="sidebar-nav">
                <div className="nav-section-label">Pipeline Stages</div>
                {PIPELINE_TABS.map((tab) => {
                    const status = getStageStatus(tab.stage);
                    return (
                        <button
                            key={tab.key}
                            className={`nav-item ${activeTab === tab.key ? 'active' : ''} stage-${status}`}
                            onClick={() => setActiveTab(tab.key as TabKey)}
                        >
                            <span className="nav-icon">{tab.icon}</span>
                            <span className="nav-label">{tab.label}</span>
                            <span className={`nav-status-dot ${status}`} />
                        </button>
                    );
                })}
            </nav>

            <div className="sidebar-footer">
                <div className="footer-version">v0.1.0</div>
            </div>
        </aside>
    );
}
