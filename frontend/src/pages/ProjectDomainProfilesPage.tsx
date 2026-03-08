import { useOutletContext } from 'react-router-dom';

import DomainProfileManager from '../components/domain/DomainProfileManager';
import { useProjectStore } from '../stores/projectStore';
import type { Project } from '../types';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectDomainProfilesPage() {
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();
    const { setActiveProject } = useProjectStore();

    return (
        <DomainProfileManager
            projectId={projectId}
            activeDomainProfileId={project.domain_profile_id}
            onAssigned={(updated: Project) => setActiveProject(updated)}
        />
    );
}
