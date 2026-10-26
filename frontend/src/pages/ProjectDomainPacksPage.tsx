/**
 * Page for assigning reusable domain pack policy bundles and hook defaults to projects.
 */

import { useState } from 'react';
import { useOutletContext } from 'react-router-dom';

import DomainPackContentList from '../components/domain/DomainPackContentList';
import DomainPackImportPanel from '../components/domain/DomainPackImportPanel';
import DomainPackManager from '../components/domain/DomainPackManager';
import Term from '../components/shared/Term';
import { useProjectStore } from '../stores/projectStore';
import type { Project } from '../types';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectDomainPacksPage({ embedded = false }: { embedded?: boolean } = {}) {
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();
    const { setActiveProject } = useProjectStore();
    const [refreshKey, setRefreshKey] = useState(0);

    return (
        <div className="workspace-page">
            {!embedded && (
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">
                        <Term id="domain_pack" plural advanced />
                    </h2>
                    <p className="workspace-page-subtitle">
                        Set reusable policy bundles and hook defaults for this project.
                    </p>
                </div>
            </section>
            )}
            <DomainPackImportPanel onImported={() => setRefreshKey((k) => k + 1)} />
            <DomainPackContentList projectId={projectId} refreshKey={refreshKey} />
            <DomainPackManager
                key={refreshKey}
                projectId={projectId}
                activeDomainPackId={project.domain_pack_id}
                onAssigned={(updated: Project) => setActiveProject(updated)}
            />
        </div>
    );
}
