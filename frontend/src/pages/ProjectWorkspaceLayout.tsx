import { useEffect, useMemo } from 'react';
import { Navigate, Outlet, useParams } from 'react-router-dom';

import TopBar from '../components/layout/TopBar';
import ProjectSidebar from '../components/layout/ProjectSidebar';
import CommandPalette from '../components/layout/CommandPalette';
import WorkspaceFlowHint from '../components/layout/WorkspaceFlowHint';
import DecisionLogDrawer from '../components/autopilot/DecisionLogDrawer';
import ManifestExportButton from '../components/manifest/ManifestExportButton';
import ProgressChip from '../components/gamification/ProgressChip';
import CoachToggle from '../components/coach/CoachToggle';
import { useGamificationPoller } from '../components/gamification/useProgressionPoll';
import { useProjectStore } from '../stores/projectStore';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectWorkspaceLayout() {
    const { id } = useParams<{ id: string }>();
    const {
        activeProject,
        pipelineStatus,
        fetchProject,
        fetchPipelineStatus,
    } = useProjectStore();

    const projectId = Number.parseInt(id || '', 10);
    const projectIdValid = Number.isFinite(projectId);

    useEffect(() => {
        if (!projectIdValid) {
            return;
        }
        void fetchProject(projectId);
        void fetchPipelineStatus(projectId);
    }, [projectId, projectIdValid, fetchProject, fetchPipelineStatus]);

    // Lab Journal: drive the gamification poller while a project is
    // active. Hook is a no-op when projectId is 0 / invalid, so it's
    // safe to call unconditionally above the early-return guards.
    useGamificationPoller(projectIdValid ? projectId : 0);

    const refreshPipelineStatus = async () => {
        if (!projectIdValid) {
            return;
        }
        await fetchPipelineStatus(projectId);
    };

    const contextValue = useMemo<ProjectWorkspaceContextValue | null>(() => {
        if (!projectIdValid || !activeProject || activeProject.id !== projectId) {
            return null;
        }
        return {
            projectId,
            project: activeProject,
            pipelineStatus,
            refreshPipelineStatus,
        };
    }, [projectId, projectIdValid, activeProject, pipelineStatus, refreshPipelineStatus]);

    if (!projectIdValid) {
        return <Navigate to="/" replace />;
    }

    if (!activeProject || activeProject.id !== projectId || !contextValue) {
        return (
            <div className="app-layout">
                <div className="main-content" style={{ marginLeft: 0 }}>
                    <div className="page-container">
                        <div className="skeleton" style={{ height: 48, width: 300, marginBottom: 24 }} />
                        <div className="skeleton" style={{ height: 200 }} />
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="app-layout">
            <ProjectSidebar
                projectId={projectId}
                projectName={activeProject.name}
                pipelineStatus={pipelineStatus}
                beginnerMode={activeProject.beginner_mode}
            />
            <div className="main-content">
                <TopBar
                    title={activeProject.name}
                    subtitle={activeProject.description || undefined}
                    withSidebar
                    actions={
                        <>
                            <ProgressChip projectId={projectId} />
                            <CoachToggle projectId={projectId} />
                            <ManifestExportButton
                                projectId={projectId}
                                projectName={activeProject.name}
                                size="sm"
                            />
                            <span className={`badge ${activeProject.status === 'active' ? 'badge-success' : 'badge-info'}`}>
                                {activeProject.status}
                            </span>
                        </>
                    }
                />
                <div className="page-container">
                    <WorkspaceFlowHint
                        projectId={projectId}
                        project={activeProject}
                        pipelineStatus={pipelineStatus}
                    />
                    <Outlet context={contextValue} />
                </div>
            </div>
            <DecisionLogDrawer projectId={projectId} />
            <CommandPalette
                projectId={projectId}
                beginnerMode={activeProject.beginner_mode}
            />
        </div>
    );
}
