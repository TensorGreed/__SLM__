/**
 * TopBar toggle for Coach Mode (USER-SUCCESS Epic 4 Phase 1).
 *
 * Compact chip rendered next to ``ProgressChip``. Reflects the
 * effective on/off state for the active project and flips it on
 * click. Default state is XP-gated (level < 3 → on), so brand-new
 * users see the toggle already lit.
 */

import { useCoachMode } from './useCoachMode';

interface CoachToggleProps {
    projectId: number;
}

export default function CoachToggle({ projectId }: CoachToggleProps) {
    const { isOn, toggle, defaultOn, isReady } = useCoachMode(projectId);

    // While the gamification level fetch is in flight, render in the
    // default state (no flicker between off→on when an override has
    // already been applied — the store override is read synchronously
    // from localStorage so ``isOn`` is correct immediately; the only
    // unknown is the *default* underlying it).
    const label = isOn ? '🧭 Coach: on' : '🧭 Coach: off';
    const tooltip = isReady
        ? `Coach Mode is ${isOn ? 'on' : 'off'} for this project (default: ${defaultOn ? 'on' : 'off'}). Click to toggle.`
        : 'Loading Coach Mode default…';

    return (
        <button
            type="button"
            className={isOn ? 'badge badge-success' : 'badge badge-info'}
            onClick={toggle}
            title={tooltip}
            aria-pressed={isOn}
            aria-label={`Toggle Coach Mode (currently ${isOn ? 'on' : 'off'})`}
            data-testid="coach-toggle"
            style={{
                cursor: 'pointer',
                padding: '4px 10px',
                fontSize: '0.78rem',
                letterSpacing: '0.02em',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                height: 28,
                lineHeight: 1,
                background: isOn
                    ? 'rgba(34, 197, 94, 0.16)'
                    : 'rgba(148, 163, 184, 0.18)',
                border: `1px solid ${isOn ? 'rgba(34, 197, 94, 0.4)' : 'rgba(148, 163, 184, 0.4)'}`,
                color: isOn ? 'rgb(21, 128, 61)' : 'var(--text-secondary)',
            }}
        >
            {label}
        </button>
    );
}
