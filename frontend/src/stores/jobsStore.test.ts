/**
 * Tests for the bell's up-trend counter
 * (jobsStore._nextUpTrendCounts equivalent path, exercised via
 * fetchOnce → store state).
 *
 * The kill-switch surfaces when a training job's loss has been
 * trending up for ``KILL_SWITCH_TREND_UP_THRESHOLD`` consecutive
 * polls. This test file pins the counter's accumulation,
 * forgiveness, carry-forward, and isolation behaviour — the
 * kill-switch component logic depends on all four.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
    KILL_SWITCH_TREND_UP_THRESHOLD,
    useJobsStore,
} from './jobsStore';


const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));
vi.mock('../api/client', () => ({ default: apiMock }));


function _trainingJob(
    id: number,
    metrics: Array<{ step: number; train_loss?: number }>,
): unknown {
    return {
        id,
        kind: 'training_start',
        title: `train #${id}`,
        status: 'running',
        progress: 0.5,
        progress_message: null,
        project_id: 17,
        user_id: null,
        params: { experiment_id: id + 100 },
        result: null,
        error: null,
        queued_at: '2026-06-01T00:00:00Z',
        started_at: '2026-06-01T00:00:00Z',
        completed_at: null,
        dismissed_at: null,
        metrics_recent: metrics,
    };
}


function _stubFetchActiveJobs(jobs: unknown[]) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.startsWith('/jobs/active')) {
            return { data: { count: jobs.length, jobs } };
        }
        return { data: {} };
    });
}


async function _runFetch() {
    await useJobsStore.getState().fetchOnce();
}


function _resetStore() {
    useJobsStore.setState({
        jobs: [],
        isLoading: false,
        isPolling: false,
        bellOpen: false,
        lastError: null,
        upTrendCountById: {},
        _lastStatusById: {},
        _pollHandle: null,
    });
}


describe('jobsStore up-trend counter', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        _resetStore();
    });

    it('increments on each poll while trend is up', async () => {
        // 5 strictly-increasing losses → trend resolves to ``up``.
        // Three polls of the same shape → counter = 3, which is
        // exactly the kill-switch threshold.
        const upMetrics = [
            { step: 100, train_loss: 0.2 },
            { step: 200, train_loss: 0.3 },
            { step: 300, train_loss: 0.4 },
            { step: 400, train_loss: 0.5 },
            { step: 500, train_loss: 0.6 },
        ];
        _stubFetchActiveJobs([_trainingJob(1, upMetrics)]);
        await _runFetch();
        await _runFetch();
        await _runFetch();
        expect(useJobsStore.getState().upTrendCountById[1]).toBe(3);
        expect(KILL_SWITCH_TREND_UP_THRESHOLD).toBeLessThanOrEqual(
            useJobsStore.getState().upTrendCountById[1],
        );
    });

    it('resets to zero when trend turns down', async () => {
        // Two up polls, then one down → counter goes 1, 2, 0.
        const upMetrics = [
            { step: 100, train_loss: 0.2 },
            { step: 200, train_loss: 0.3 },
            { step: 300, train_loss: 0.4 },
            { step: 400, train_loss: 0.5 },
        ];
        const downMetrics = [
            { step: 100, train_loss: 0.5 },
            { step: 200, train_loss: 0.4 },
            { step: 300, train_loss: 0.3 },
            { step: 400, train_loss: 0.2 },
        ];
        _stubFetchActiveJobs([_trainingJob(1, upMetrics)]);
        await _runFetch();
        await _runFetch();
        _stubFetchActiveJobs([_trainingJob(1, downMetrics)]);
        await _runFetch();
        expect(useJobsStore.getState().upTrendCountById[1]).toBe(0);
    });

    it('resets to zero when trend turns flat', async () => {
        // Up then flat — single healthy poll forgives the counter.
        // Surfaces the kill switch only on sustained divergence.
        const upMetrics = [
            { step: 100, train_loss: 0.2 },
            { step: 200, train_loss: 0.3 },
            { step: 300, train_loss: 0.4 },
            { step: 400, train_loss: 0.5 },
        ];
        const flatMetrics = [
            { step: 100, train_loss: 0.500 },
            { step: 200, train_loss: 0.501 },
            { step: 300, train_loss: 0.499 },
            { step: 400, train_loss: 0.502 },
        ];
        _stubFetchActiveJobs([_trainingJob(1, upMetrics)]);
        await _runFetch();
        _stubFetchActiveJobs([_trainingJob(1, flatMetrics)]);
        await _runFetch();
        expect(useJobsStore.getState().upTrendCountById[1]).toBe(0);
    });

    it('carries the counter forward when metrics_recent is missing', async () => {
        // Pre-first-checkpoint or trainer_state read failure → no
        // metrics_recent in the payload. The bell should NOT reset
        // the counter; otherwise a transient read failure mid-run
        // would re-arm the trend just as the kill switch was about
        // to fire. Carry-forward is the load-bearing behaviour.
        const upMetrics = [
            { step: 100, train_loss: 0.2 },
            { step: 200, train_loss: 0.3 },
            { step: 300, train_loss: 0.4 },
            { step: 400, train_loss: 0.5 },
        ];
        _stubFetchActiveJobs([_trainingJob(1, upMetrics)]);
        await _runFetch();
        await _runFetch();
        // Now serve a payload without metrics_recent (simulate
        // missing field). The job's value will be undefined for
        // metrics_recent — our stub crafted them with the field;
        // mimic absence by overriding the shape directly.
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.startsWith('/jobs/active')) {
                const noMetrics = _trainingJob(1, []) as Record<string, unknown>;
                delete noMetrics.metrics_recent;
                return { data: { count: 1, jobs: [noMetrics] } };
            }
            return { data: {} };
        });
        await _runFetch();
        // Counter survived the metrics-less poll.
        expect(useJobsStore.getState().upTrendCountById[1]).toBe(2);
    });

    it('tracks counters per job independently', async () => {
        // One job diverging, one job healthy — the up-trend
        // counter must not bleed between jobs.
        const upMetrics = [
            { step: 100, train_loss: 0.2 },
            { step: 200, train_loss: 0.3 },
            { step: 300, train_loss: 0.4 },
            { step: 400, train_loss: 0.5 },
        ];
        const downMetrics = [
            { step: 100, train_loss: 0.5 },
            { step: 200, train_loss: 0.4 },
            { step: 300, train_loss: 0.3 },
            { step: 400, train_loss: 0.2 },
        ];
        _stubFetchActiveJobs([
            _trainingJob(1, upMetrics),
            _trainingJob(2, downMetrics),
        ]);
        await _runFetch();
        await _runFetch();
        expect(useJobsStore.getState().upTrendCountById[1]).toBe(2);
        expect(useJobsStore.getState().upTrendCountById[2]).toBe(0);
    });

    it('does not track non-training jobs', async () => {
        // Synth jobs, RAG-clone jobs, etc. have no
        // metrics_recent path and shouldn't appear in the
        // trend-counter map at all.
        _stubFetchActiveJobs([
            {
                ...(_trainingJob(1, []) as Record<string, unknown>),
                kind: 'synth_playbook',
            },
        ]);
        await _runFetch();
        expect(useJobsStore.getState().upTrendCountById[1]).toBeUndefined();
    });

    it('does not track terminal-status jobs', async () => {
        // A completed/failed/cancelled training job shouldn't
        // accumulate counter ticks — they're not at risk of
        // diverging further.
        const upMetrics = [
            { step: 100, train_loss: 0.2 },
            { step: 200, train_loss: 0.3 },
            { step: 300, train_loss: 0.4 },
            { step: 400, train_loss: 0.5 },
        ];
        _stubFetchActiveJobs([
            {
                ...(_trainingJob(1, upMetrics) as Record<string, unknown>),
                status: 'succeeded',
            },
        ]);
        await _runFetch();
        expect(useJobsStore.getState().upTrendCountById[1]).toBeUndefined();
    });
});
