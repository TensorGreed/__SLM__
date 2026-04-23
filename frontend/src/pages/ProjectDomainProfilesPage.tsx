import { useOutletContext } from 'react-router-dom';

import DomainProfileManager from '../components/domain/DomainProfileManager';
import Term from '../components/shared/Term';
import { useProjectStore } from '../stores/projectStore';
import type { Project } from '../types';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectDomainProfilesPage() {
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();
    const { setActiveProject } = useProjectStore();

    return (
        <div className="workspace-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">
                        <Term id="domain_profile" plural advanced />
                    </h2>
                    <p className="workspace-page-subtitle">
                        Configure task schemas, quality <Term id="gate" plural />, and deployment checks for the active domain.
                    </p>
                </div>
            </section>
            <DomainProfileManager
                projectId={projectId}
                activeDomainProfileId={project.domain_profile_id}
                onAssigned={(updated: Project) => setActiveProject(updated)}
            />
        </div>
    );
}
