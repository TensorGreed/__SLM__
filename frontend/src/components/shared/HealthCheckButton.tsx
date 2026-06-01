/**
 * HealthCheckButton — Diagnostics Intervention C.
 *
 * Top-bar button on every project page that opens the smoke-test
 * modal. Renders a stable icon + label; the actual work happens
 * inside <HealthCheckModal> when the button is clicked. Keeps the
 * button itself featherweight so it doesn't slow down page mounts.
 */

import { useState } from 'react';

import HealthCheckModal from './HealthCheckModal';


interface HealthCheckButtonProps {
    projectId: number;
}


export default function HealthCheckButton({ projectId }: HealthCheckButtonProps) {
    const [open, setOpen] = useState(false);

    return (
        <>
            <button
                type="button"
                className="topbar-icon-btn"
                onClick={() => setOpen(true)}
                title="Run a parallel read-only smoke test across every project surface"
                aria-label="Run project health check"
                data-testid="health-check-button"
            >
                🩺 Health
            </button>
            {open && (
                <HealthCheckModal
                    projectId={projectId}
                    onClose={() => setOpen(false)}
                />
            )}
        </>
    );
}
