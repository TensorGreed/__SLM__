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
        <>
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
        </>
    );
}
