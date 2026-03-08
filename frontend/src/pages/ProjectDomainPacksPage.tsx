import { useOutletContext } from 'react-router-dom';

import DomainPackManager from '../components/domain/DomainPackManager';
import { useProjectStore } from '../stores/projectStore';
import type { Project } from '../types';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectDomainPacksPage() {
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();
    const { setActiveProject } = useProjectStore();

    return (
        <DomainPackManager
            projectId={projectId}
            activeDomainPackId={project.domain_pack_id}
            onAssigned={(updated: Project) => setActiveProject(updated)}
        />
    );
}
