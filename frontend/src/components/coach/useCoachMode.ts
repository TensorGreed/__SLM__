import { useEffect, useState } from 'react';

import { fetchProgression } from '../../api/gamification';
import {
    computeCoachDefaultOn,
    useCoachModeStore,
} from '../../stores/coachModeStore';

interface UseCoachModeResult {
    isOn: boolean;
    toggle: () => void;
    // Exposed so callers can render the underlying default state (e.g.
    // a "Coach is on by default for new projects" hint on the toggle).
    defaultOn: boolean;
    // ``true`` while the gamification level is still being fetched —
    // callers can avoid rendering a flicker between default-on and
    // override-off by waiting until ``isReady`` flips true.
    isReady: boolean;
}

/**
 * Returns the effective Coach Mode on/off state for a project + a
 * toggle action. The default state is XP-gated: projects on a user
 * who's still onboarding (gamification level < 3) default ON; power
 * users default OFF. Manual per-project overrides win.
 */
export function useCoachMode(projectId: number): UseCoachModeResult {
    const [level, setLevel] = useState<number | null>(null);
    const [isReady, setIsReady] = useState(false);
    const manualOverrides = useCoachModeStore((s) => s.manualOverrides);
    const toggleStore = useCoachModeStore((s) => s.toggle);

    useEffect(() => {
        let cancelled = false;
        setIsReady(false);
        const load = async () => {
            try {
                const progression = await fetchProgression(projectId);
                if (!cancelled) {
                    setLevel(progression.level);
                }
            } catch {
                // Treat gamification fetch failure as "no progression
                // data" — defaultOn falls back to true (the "newbie"
                // bias), matching computeCoachDefaultOn(null).
                if (!cancelled) {
                    setLevel(null);
                }
            } finally {
                if (!cancelled) setIsReady(true);
            }
        };
        void load();
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    const defaultOn = computeCoachDefaultOn(level);
    const override = manualOverrides[String(projectId)];
    const isOn =
        override === 'on' ? true : override === 'off' ? false : defaultOn;

    return {
        isOn,
        toggle: () => toggleStore(projectId, defaultOn),
        defaultOn,
        isReady,
    };
}
