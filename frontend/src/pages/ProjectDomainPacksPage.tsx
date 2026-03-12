import { useOutletContext } from 'react-router-dom';

import DomainPackManager from '../components/domain/DomainPackManager';
import { useProjectStore } from '../stores/projectStore';
import type { Project } from '../types';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectDomainPacksPage() {
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();
    const { setActiveProject } = useProjectStore();

    return (
        <div className="workspace-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Domain Packs</h2>
                    <p className="workspace-page-subtitle">
                        Set reusable policy bundles and hook defaults for this project.
                    </p>
                </div>
            </section>
            <DomainPackManager
                projectId={projectId}
                activeDomainPackId={project.domain_pack_id}
                onAssigned={(updated: Project) => setActiveProject(updated)}
            />
        </div>
    );
}
