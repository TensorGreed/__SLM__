import { useEffect, useMemo } from 'react';
import { Navigate, Outlet, useParams } from 'react-router-dom';

import TopBar from '../components/layout/TopBar';
import ProjectSidebar from '../components/layout/ProjectSidebar';
import WorkspaceFlowHint from '../components/layout/WorkspaceFlowHint';
import DecisionLogDrawer from '../components/autopilot/DecisionLogDrawer';
import ManifestExportButton from '../components/manifest/ManifestExportButton';
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
        </div>
    );
}
