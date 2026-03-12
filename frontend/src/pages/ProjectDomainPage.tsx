import DomainPackManager from '../components/domain/DomainPackManager';
import DomainProfileManager from '../components/domain/DomainProfileManager';
import { useProjectStore } from '../stores/projectStore';
import type { Project } from '../types';
import { useOutletContext } from 'react-router-dom';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectDomainPage() {
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();
    const { setActiveProject } = useProjectStore();

    return (
        <div className="workspace-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Domain Controls</h2>
                    <p className="workspace-page-subtitle">
                        Manage domain packs and profiles that shape preprocessing, validation, and defaults.
                    </p>
                </div>
            </section>
            <DomainPackManager
                projectId={projectId}
                activeDomainPackId={project.domain_pack_id}
                onAssigned={(updated: Project) => setActiveProject(updated)}
            />
            <DomainProfileManager
                projectId={projectId}
                activeDomainProfileId={project.domain_profile_id}
                onAssigned={(updated: Project) => setActiveProject(updated)}
            />
        </div>
    );
}
