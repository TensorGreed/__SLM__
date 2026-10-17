import { COACH_DEFAULT_ON, useCoachModeStore } from '../../stores/coachModeStore';

interface UseCoachModeResult {
    isOn: boolean;
    toggle: () => void;
    defaultOn: boolean;
    /** Kept for callers that waited on the old asynchronous default; always true now. */
    isReady: boolean;
}

/**
 * The effective "inline tips" state for a project + a toggle. Tips are on
 * by default for every project (beginner or not); a manual per-project
 * override always wins. The switch lives on the project's Settings page.
 */
export function useCoachMode(projectId: number): UseCoachModeResult {
    const manualOverrides = useCoachModeStore((s) => s.manualOverrides);
    const toggleStore = useCoachModeStore((s) => s.toggle);

    const defaultOn = COACH_DEFAULT_ON;
    const override = manualOverrides[String(projectId)];
    const isOn = override === 'on' ? true : override === 'off' ? false : defaultOn;

    return {
        isOn,
        toggle: () => toggleStore(projectId, defaultOn),
        defaultOn,
        isReady: true,
    };
}
