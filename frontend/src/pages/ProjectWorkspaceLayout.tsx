/**
 * Root layout for the project workspace: sidebar (six destinations), top
 * bar, the command palette. Guidance is inline on each destination
 * (CoachNotices) and on the Overview.
 */

import { useEffect, useMemo } from 'react';
import { Navigate, Outlet, useParams } from 'react-router-dom';

import TopBar from '../components/layout/TopBar';
import ProjectSidebar from '../components/layout/ProjectSidebar';
import CommandPalette from '../components/layout/CommandPalette';
import DecisionLogDrawer from '../components/autopilot/DecisionLogDrawer';
import HealthCheckButton from '../components/shared/HealthCheckButton';
import ParentProjectBackChip from '../components/layout/ParentProjectBackChip';
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
            <ProjectSidebar projectId={projectId} projectName={activeProject.name} />
            <div className="main-content">
                <TopBar
                    title={activeProject.name}
                    subtitle={activeProject.description || undefined}
                    withSidebar
                    actions={
                        <>
                            {activeProject.parent_project_id ? (
                                <ParentProjectBackChip
                                    parentProjectId={activeProject.parent_project_id}
                                />
                            ) : null}
                            <HealthCheckButton projectId={projectId} />
                        </>
                    }
                />
                <div className="page-container">
                    <Outlet context={contextValue} />
                </div>
            </div>
            <DecisionLogDrawer projectId={projectId} />
            <CommandPalette projectId={projectId} />
        </div>
    );
}
