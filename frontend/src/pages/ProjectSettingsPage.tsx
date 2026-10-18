/**
 * Project Settings — the workspace's long tail (Epic U, phase U1).
 *
 * Workspace preferences (inline tips, advanced sections) and links to the
 * tools a normal day never needs: automation, domain controls, studios,
 * the manifest. Later phases fold these pages into sections here.
 */

import { useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';

import api from '../api/client';
import ManifestExportButton from '../components/manifest/ManifestExportButton';
import { useCoachMode } from '../components/coach/useCoachMode';
import { useProjectStore } from '../stores/projectStore';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectSettingsPage.css';

interface ToolLink {
    label: string;
    path: string;
    hint: string;
}

const TOOL_GROUPS: Array<{ title: string; links: ToolLink[] }> = [
    {
        title: 'Automation',
        links: [
            { label: 'Autopilot Planner', path: 'autopilot', hint: 'Plan diffs, repair preview, rollback.' },
            { label: 'Workflow Builder', path: 'workflow', hint: 'Compose and run pipeline graphs.' },
            { label: 'Pipeline presets', path: 'train?open=pipeline-presets', hint: 'End-to-end pipeline compositions (under Train).' },
            { label: 'Pipeline as Code', path: 'ship?stage=package&open=manifest', hint: 'The project as a brewslm.yaml manifest (under Ship).' },
        ],
    },
    {
        title: 'Domain',
        links: [
            { label: 'Domain Packs', path: 'domain/packs', hint: 'Overlay a domain pack on the project.' },
            { label: 'Domain Profiles', path: 'domain/profiles', hint: 'Profiles that tune the pack.' },
        ],
    },
    {
        title: 'Studios',
        links: [
            { label: 'Adapter Studio', path: 'train?open=adapters', hint: 'Data adapter contracts (under Train).' },
            { label: 'Extension Studio', path: 'extensions', hint: 'Runtime and pack plugins.' },
        ],
    },
];

export default function ProjectSettingsPage() {
    const navigate = useNavigate();
    const { projectId, project } = useOutletContext<ProjectWorkspaceContextValue>();
    const setActiveProject = useProjectStore((s) => s.setActiveProject);
    const { isOn: tipsOn, toggle: toggleTips } = useCoachMode(projectId);
    const [savingBeginner, setSavingBeginner] = useState(false);
    const [error, setError] = useState<string | null>(null);

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
                        <small>Suggestions on the Overview and the stage pages. Off hides them for this project.</small>
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
                        <small>Advanced evaluation and power tools open on request. Every page stays reachable either way.</small>
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

            {TOOL_GROUPS.map((group) => (
                <section key={group.title} className="card settings-section">
                    <h3>{group.title}</h3>
                    <ul className="settings-links">
                        {group.links.map((link) => (
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
            ))}
        </div>
    );
}
