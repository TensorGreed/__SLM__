/**
 * Project Settings — the workspace's long tail (Epic U, phases U1 + U6).
 *
 * Workspace preferences (inline tips, advanced sections, the manifest
 * export) and the tools a normal day never needs, each as a section that
 * loads only when opened: Autopilot Planner, Workflow Builder, Domain Packs,
 * Domain Profiles, Extension Studio. ``?open=<id>`` opens one.
 */

import { useState, type ReactNode } from 'react';
import { useNavigate, useOutletContext, useSearchParams } from 'react-router-dom';

import api from '../api/client';
import ManifestExportButton from '../components/manifest/ManifestExportButton';
import DetailsDrawer, { type DetailSection } from '../components/shared/DetailsDrawer';
import { useCoachMode } from '../components/coach/useCoachMode';
import { useProjectStore } from '../stores/projectStore';
import { isSettingsSection, SETTINGS_SECTIONS, type SettingsSectionId } from '../utils/workspaceRoutes';
import ProjectAutopilotPage from './ProjectAutopilotPage';
import ProjectDomainPacksPage from './ProjectDomainPacksPage';
import ProjectDomainProfilesPage from './ProjectDomainProfilesPage';
import ProjectExtensionStudioPage from './ProjectExtensionStudioPage';
import ProjectWorkflowPage from './ProjectWorkflowPage';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectSettingsPage.css';

const SECTION_CONTENT: Record<SettingsSectionId, () => ReactNode> = {
    autopilot: () => <ProjectAutopilotPage embedded />,
    workflow: () => <ProjectWorkflowPage embedded />,
    'domain-packs': () => <ProjectDomainPacksPage embedded />,
    'domain-profiles': () => <ProjectDomainProfilesPage embedded />,
    extensions: () => <ProjectExtensionStudioPage embedded />,
};

interface ToolLink {
    label: string;
    path: string;
    hint: string;
}

/** Tools that live under another destination; linked here for the long-tail reader. */
const ELSEWHERE: ToolLink[] = [
    { label: 'Pipeline presets', path: 'train?open=pipeline-presets', hint: 'End-to-end pipeline compositions (under Train).' },
    { label: 'Adapter Studio', path: 'train?open=adapters', hint: 'Data adapter contracts (under Train).' },
    { label: 'Pipeline as Code', path: 'ship?stage=package&open=manifest', hint: 'The project as a brewslm.yaml manifest (under Ship).' },
];

export default function ProjectSettingsPage() {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();
    const setActiveProject = useProjectStore((s) => s.setActiveProject);
    const { isOn: tipsOn, toggle: toggleTips } = useCoachMode(projectId);
    const [savingBeginner, setSavingBeginner] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const open = searchParams.get('open');
    const openSection = isSettingsSection(open) ? open : null;

    const toggleBeginner = async () => {
        setSavingBeginner(true);
        setError(null);
        try {
            const res = await api.put(`/projects/${projectId}`, { beginner_mode: !project.beginner_mode });
            setActiveProject(res.data);
        } catch (err) {
            const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setError(typeof detail === 'string' ? detail : 'Could not save the setting.');
        } finally {
            setSavingBeginner(false);
        }
    };

    const sections: DetailSection[] = SETTINGS_SECTIONS.map((section) => ({
        id: section.id,
        title: section.label,
        hint: section.hint,
        render: SECTION_CONTENT[section.id],
    }));

    return (
        <div className="workspace-page settings-page" data-testid="settings-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Settings</h2>
                    <p className="workspace-page-subtitle">Workspace preferences and the tools you rarely need.</p>
                </div>
            </section>

            <section className="card settings-section">
                <h3>Workspace</h3>
                <label className="settings-row">
                    <input
                        type="checkbox"
                        checked={tipsOn}
                        onChange={toggleTips}
                        data-testid="settings-inline-tips"
                    />
                    <span>
                        <strong>Show inline tips</strong>
                        <small>Suggestions on the Overview and inside each step. Off hides them for this project.</small>
                    </span>
                </label>
                <label className="settings-row">
                    <input
                        type="checkbox"
                        checked={Boolean(project.beginner_mode)}
                        onChange={() => { void toggleBeginner(); }}
                        disabled={savingBeginner}
                        data-testid="settings-beginner-mode"
                    />
                    <span>
                        <strong>Start advanced sections collapsed</strong>
                        <small>Power tools open on request. Every page stays reachable either way.</small>
                    </span>
                </label>
                {error && <div className="settings-error" role="alert">{error}</div>}
                <div className="settings-row settings-row--actions">
                    <span>
                        <strong>Export the project manifest</strong>
                        <small>A brewslm.yaml of this project's configuration.</small>
                    </span>
                    <ManifestExportButton projectId={projectId} projectName={project.name} size="sm" />
                </div>
            </section>

            <DetailsDrawer
                testId="settings-details"
                title="Tools"
                hint="automation, domain controls, extensions — each loads when opened"
                openSection={openSection}
                sections={sections}
            />

            <section className="card settings-section">
                <h3>Elsewhere</h3>
                <ul className="settings-links">
                    {ELSEWHERE.map((link) => (
                        <li key={link.path}>
                            <button
                                type="button"
                                className="settings-link"
                                onClick={() => navigate(`/project/${projectId}/${link.path}`)}
                            >
                                <strong>{link.label}</strong>
                                <small>{link.hint}</small>
                            </button>
                        </li>
                    ))}
                </ul>
            </section>
        </div>
    );
}
