import { useOutletContext } from 'react-router-dom';

import ChatPlaygroundPanel from '../components/training/ChatPlaygroundPanel';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

export default function ProjectPlaygroundPage() {
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ margin: 0 }}>Playground</h3>
                <p style={{ marginTop: 6, color: 'var(--text-secondary)' }}>
                    Run prompts against local/runtime adapters, apply presets, and log response quality feedback.
                </p>
            </div>
            <ChatPlaygroundPanel projectId={projectId} />
        </div>
    );
}
