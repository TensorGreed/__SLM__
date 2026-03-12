import { useOutletContext } from 'react-router-dom';

import ChatPlaygroundPanel from '../components/training/ChatPlaygroundPanel';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectPlaygroundPage.css';

export default function ProjectPlaygroundPage() {
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();

    return (
        <div className="workspace-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Prompt Playground</h2>
                    <p className="workspace-page-subtitle">
                    Run prompts against local/runtime adapters, apply presets, and log response quality feedback.
                    </p>
                </div>
            </section>
            <ChatPlaygroundPanel projectId={projectId} />
        </div>
    );
}
