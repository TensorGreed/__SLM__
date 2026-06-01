/**
 * Global jobs store (Hardening Phase H1).
 *
 * Keeps the active-jobs snapshot the notification bell renders, and
 * runs a smart polling loop that's only active when there's
 * something to watch (in-flight jobs OR the bell dropdown is open).
 * Idle state stops the poll so we're not hitting the server every
 * 4 seconds for nothing.
 *
 * Transition notifications: when a job we were watching flips from
 * RUNNING → SUCCEEDED / FAILED, we fire a toast so the user can
 * react even when the bell is collapsed.
 */

import { create } from 'zustand';

import { listActiveJobs, type Job } from '../api/jobs';
import { computeLossTrend } from '../components/layout/TrainingLossSparkline';
import { toast } from './toastStore';


const POLL_INTERVAL_MS = 4000;

// Number of consecutive polls in the ``up`` trend state before the
// bell surfaces the kill-switch action. Polls happen every 4s, so 3
// = ~12s of sustained divergence. Tight enough that we don't make the
// user wait through half a run; loose enough that a single noisy
// window doesn't trigger a false alarm.
export const KILL_SWITCH_TREND_UP_THRESHOLD = 3;


interface JobsState {
    jobs: Job[];
    isLoading: boolean;
    /** True when the polling loop is running. */
    isPolling: boolean;
    /** True when the bell dropdown is open. Keeps polling alive
     *  even when no job is in-flight (so newly-started jobs show
     *  up promptly). */
    bellOpen: boolean;
    lastError: string | null;
    /** Consecutive-polls-with-trend=up counter per training job.
     *  Reset to 0 when trend changes to flat/down. Read by the
     *  bell's kill-switch detector; surfacing happens at
     *  ``KILL_SWITCH_TREND_UP_THRESHOLD``. */
    upTrendCountById: Record<number, number>;
    /** Internal: previous-tick status map so we can detect
     *  RUNNING→terminal transitions and fire a toast once. */
    _lastStatusById: Record<number, string>;
    /** Internal: setInterval handle. */
    _pollHandle: ReturnType<typeof setInterval> | null;

    fetchOnce: () => Promise<void>;
    startPolling: () => void;
    stopPolling: () => void;
    setBellOpen: (open: boolean) => void;
    refreshAfterLocalChange: () => Promise<void>;
}


function _nextUpTrendCounts(
    prev: Record<number, number>,
    jobs: Job[],
): Record<number, number> {
    const next: Record<number, number> = {};
    for (const j of jobs) {
        if (j.kind !== 'training_start') continue;
        if (j.status !== 'queued' && j.status !== 'running') continue;
        // No metrics_recent yet → carry the previous count forward
        // unchanged. A pre-checkpoint window shouldn't reset the
        // counter; otherwise the trend would re-arm every time the
        // first checkpoint loads after a poll.
        if (!Array.isArray(j.metrics_recent)) {
            if (prev[j.id]) next[j.id] = prev[j.id];
            continue;
        }
        const trend = computeLossTrend(j.metrics_recent);
        if (trend === 'up') {
            next[j.id] = (prev[j.id] || 0) + 1;
        } else {
            // Reset on flat/down — a single healthy poll forgives the
            // counter. Surfaces the kill switch only on sustained
            // divergence, not on transient noise.
            next[j.id] = 0;
        }
    }
    return next;
}


function _shouldKeepPolling(jobs: Job[], bellOpen: boolean): boolean {
    if (bellOpen) return true;
    return jobs.some((j) => j.status === 'queued' || j.status === 'running');
}


function _detectTransitionsAndToast(
    prev: Record<string, string>,
    next: Job[],
): Record<number, string> {
    const nextMap: Record<number, string> = {};
    for (const j of next) {
        nextMap[j.id] = j.status;
        const was = prev[String(j.id)];
        // Only toast when we actually observed a prior in-flight
        // status — first-sight terminal jobs (e.g. page-load
        // surfacing a recently-completed job) shouldn't fire.
        if (
            (was === 'queued' || was === 'running')
            && (j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled')
        ) {
            if (j.status === 'succeeded') {
                toast.success(`${j.title} — done`, 4000);
            } else if (j.status === 'failed') {
                toast.error(`${j.title} failed: ${j.error || 'see notification bell'}`);
            } else {
                toast.info(`${j.title} cancelled`, 3000);
            }
        }
    }
    return nextMap;
}


export const useJobsStore = create<JobsState>((set, get) => ({
    jobs: [],
    isLoading: false,
    isPolling: false,
    bellOpen: false,
    lastError: null,
    upTrendCountById: {},
    _lastStatusById: {},
    _pollHandle: null,

    fetchOnce: async () => {
        const state = get();
        if (state.isLoading) return;
        set({ isLoading: true, lastError: null });
        try {
            const data = await listActiveJobs({
                includeRecentlyCompleted: true,
                limit: 50,
            });
            const transitions = _detectTransitionsAndToast(
                state._lastStatusById,
                data.jobs,
            );
            const upTrendCountById = _nextUpTrendCounts(
                state.upTrendCountById,
                data.jobs,
            );
            set({
                jobs: data.jobs,
                isLoading: false,
                _lastStatusById: transitions,
                upTrendCountById,
            });
            // Self-stop the loop if nothing's in-flight and the
            // bell is closed — saves a poll/4s when idle.
            if (!_shouldKeepPolling(data.jobs, get().bellOpen)) {
                get().stopPolling();
            }
        } catch (err) {
            const msg =
                (err as { message?: string })?.message
                || 'jobs fetch failed';
            set({ isLoading: false, lastError: msg });
        }
    },

    startPolling: () => {
        const state = get();
        if (state.isPolling) return;
        set({ isPolling: true });
        // Fire immediately so the bell isn't stale for the first
        // poll tick.
        void state.fetchOnce();
        const handle = setInterval(() => {
            void get().fetchOnce();
        }, POLL_INTERVAL_MS);
        set({ _pollHandle: handle });
    },

    stopPolling: () => {
        const handle = get()._pollHandle;
        if (handle) clearInterval(handle);
        set({ _pollHandle: null, isPolling: false });
    },

    setBellOpen: (open: boolean) => {
        set({ bellOpen: open });
        if (open) {
            // Opening the bell always triggers a refresh + ensures
            // polling so newly-started jobs surface immediately.
            get().startPolling();
        } else {
            // Closing the bell: keep polling iff something's
            // in-flight; otherwise stop to save bandwidth.
            if (!_shouldKeepPolling(get().jobs, false)) {
                get().stopPolling();
            }
        }
    },

    refreshAfterLocalChange: async () => {
        // Called by call sites that just started a job (e.g. synth
        // run, reroute clone) — kick off the poll so the new job
        // shows up in the bell on the very next tick.
        get().startPolling();
        await get().fetchOnce();
    },
}));
