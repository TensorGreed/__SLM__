/**
 * Single shared poller for the per-project gamification state.
 *
 * Drives:
 *   - The persistent ProgressChip in the TopBar (level + XP delta).
 *   - The LabJournalDrawer status section when open.
 *   - Achievement / level-up toasts (diffs against the last-seen
 *     ``recent_unlocks`` and pushes the new entries into the toast
 *     store).
 *
 * One poll = one HTTP call. We don't run separate polls per
 * component — the hook caches the latest state in the toast-store-
 * equivalent local module state and every consumer reads from the
 * same source. The frontend doesn't have a global state library
 * outside Zustand toasts, so we use a lightweight Zustand store
 * scoped to this module.
 */

import { useEffect } from 'react';
import { create } from 'zustand';

import {
    fetchProgression,
    type ProgressionState,
} from '../../api/gamification';
import { useToastStore } from '../../stores/toastStore';

interface ProgressionStoreState {
    /** Project the cache belongs to. Resets when the user navigates. */
    projectId: number | null;
    /** Latest snapshot from /api/projects/{id}/gamification. */
    state: ProgressionState | null;
    /** ISO timestamps of unlocks we've already toasted, to prevent
     *  duplicate toasts on every poll. */
    seenUnlockTimestamps: Set<string>;
    setProject: (projectId: number) => void;
    setState: (state: ProgressionState) => void;
}

const useProgressionStore = create<ProgressionStoreState>((set) => ({
    projectId: null,
    state: null,
    seenUnlockTimestamps: new Set(),
    setProject: (projectId) =>
        set((prev) =>
            prev.projectId === projectId
                ? prev
                : {
                      projectId,
                      state: null,
                      seenUnlockTimestamps: new Set(),
                  },
        ),
    setState: (state) => set({ state }),
}));

/** Read-only access to the cached progression state. Consumers like
 *  ProgressChip / LabJournalDrawer subscribe to this. */
export function useProgressionState(): ProgressionState | null {
    return useProgressionStore((s) => s.state);
}

const POLL_INTERVAL_MS = 10_000;

/**
 * Mount once near the top of the project workspace. Polls every
 * 10s while mounted; diffs recent_unlocks against what we've
 * already toasted and pushes new entries through the achievement-
 * toast variant.
 *
 * The poll is best-effort: a single failure (network blip, 404
 * during a race) is logged + skipped so a flaky read never disrupts
 * the workspace.
 */
export function useGamificationPoller(projectId: number): void {
    const setProject = useProgressionStore((s) => s.setProject);
    const setState = useProgressionStore((s) => s.setState);
    const addAchievementToast = useToastStore((s) => s.addAchievementToast);

    useEffect(() => {
        if (!projectId) return;
        setProject(projectId);

        let cancelled = false;
        let timer: ReturnType<typeof setTimeout> | null = null;

        const tick = async () => {
            try {
                const next = await fetchProgression(projectId);
                if (cancelled) return;

                // Diff: emit a toast for every new unlock the cache
                // hasn't seen yet. ``recent_unlocks`` is sorted
                // newest-first; iterate reverse so toasts appear in
                // chronological order.
                const seen = useProgressionStore.getState().seenUnlockTimestamps;
                const newOnes = (next.recent_unlocks || []).filter(
                    (u) => !seen.has(u.unlocked_at),
                );
                if (newOnes.length > 0) {
                    const reversed = [...newOnes].reverse();
                    for (const unlock of reversed) {
                        addAchievementToast({
                            kind: unlock.kind,
                            title:
                                unlock.title ??
                                (unlock.kind === 'level_up'
                                    ? `Level ${unlock.level_after}`
                                    : 'Achievement'),
                            description: unlock.description,
                            xp_awarded: unlock.xp_awarded,
                            level_after: unlock.level_after,
                            tier: unlock.tier,
                        });
                    }
                    useProgressionStore.setState((prev) => {
                        const merged = new Set(prev.seenUnlockTimestamps);
                        for (const u of newOnes) merged.add(u.unlocked_at);
                        return { seenUnlockTimestamps: merged };
                    });
                }

                setState(next);
            } catch (err) {
                // Best-effort; don't spam the console.
                if (!cancelled) {
                    console.debug('[gamification] poll failed', err);
                }
            } finally {
                if (!cancelled) {
                    timer = setTimeout(tick, POLL_INTERVAL_MS);
                }
            }
        };

        void tick();
        return () => {
            cancelled = true;
            if (timer) clearTimeout(timer);
        };
    }, [projectId, setProject, setState, addAchievementToast]);
}

/** Reset the cache — used in tests; also handy if the user creates a
 *  new project mid-session. */
export function resetProgressionCache(): void {
    useProgressionStore.setState({
        projectId: null,
        state: null,
        seenUnlockTimestamps: new Set(),
    });
}
