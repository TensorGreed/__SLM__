/**
 * Phase 7e follow-up: every API call that starts a Job wakes the jobs store's poll (api/jobWake.ts, hooked into the
 * real axios client), so a Job started from any page is watched until it ends; the poll still stops itself when
 * idle. Measured bug: Train > Launch never woke it, so a 40-minute training Job ran unwatched and Results never
 * learned the run finished (no training-end refresh, no Score form on the new run).
 *
 * These tests use the REAL client (no ``vi.mock('../api/client')``) with a fake axios adapter, so the response hook
 * itself is exercised.
 */

import { StrictMode } from 'react';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import type { AxiosAdapter, InternalAxiosRequestConfig } from 'axios';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../components/training/PreRunConfirmModal', () => ({
    default: ({ onConfirm }: { onConfirm: () => void }) => (
        <div data-testid="pre-run-modal"><button type="button" onClick={onConfirm}>Launch</button></div>
    ),
}));
vi.mock('../components/training/CloudBurstPlanningSection', () => ({ default: () => <div data-testid="cloud-burst" /> }));
vi.mock('../components/training/WhyThisPlanPanel', () => ({ default: () => <div data-testid="why-this-plan" /> }));
vi.mock('../components/shared/ReadinessPanel', () => ({ ReadinessPanel: () => <div data-testid="readiness-panel" /> }));
vi.mock('../components/training/SqlExecutionConfigCard', () => ({ default: () => <div data-testid="sql-execution-card" /> }));
vi.mock('../components/training/AutoLiftSettingCard', () => ({ default: () => <div data-testid="auto-lift-card" /> }));
vi.mock('../components/training/ToolScoringConfigCard', () => ({ default: () => <div data-testid="tool-scoring-card" /> }));

import api from '../api/client';
import { responseStartsJob } from '../api/jobWake';
import EvalPanel from '../components/evaluation/EvalPanel';
import TrainingPanel from '../components/training/TrainingPanel';
import { useJobsStore } from './jobsStore';
import { useProjectStore } from './projectStore';

type Handler = (config: InternalAxiosRequestConfig) => { status?: number; data: unknown } | undefined;

const calls: Array<{ method: string; url: string }> = [];
let handler: Handler = () => undefined;
const originalAdapter = api.defaults.adapter;

const fakeAdapter: AxiosAdapter = async (config) => {
    const method = String(config.method || 'get').toLowerCase();
    const url = String(config.url || '');
    calls.push({ method, url });
    const res = handler(config) ?? { data: {} };
    return { data: res.data, status: res.status ?? 200, statusText: 'OK', headers: {}, config };
};

function jobsActiveCalls(): number {
    return calls.filter((c) => c.method === 'get' && c.url.startsWith('/jobs/active')).length;
}

function job(id: number, kind: string, status: string, projectId = 4) {
    return {
        id, kind, title: `${kind} #${id}`, status, progress: null, progress_message: null, project_id: projectId,
        user_id: null, params: {}, result: null, error: null, queued_at: null, started_at: null, completed_at: null,
        dismissed_at: null,
    };
}

function resetStore() {
    useJobsStore.getState().stopPolling();
    useJobsStore.setState({
        jobs: [], isLoading: false, isPolling: false, bellOpen: false, lastError: null, upTrendCountById: {},
        _lastStatusById: {}, _pollHandle: null, _refetchQueued: false,
    });
}

async function tick(ms: number) {
    await act(async () => { await vi.advanceTimersByTimeAsync(ms); });
}

beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    calls.length = 0;
    handler = () => undefined;
    api.defaults.adapter = fakeAdapter;
    resetStore();
    useProjectStore.setState({ activeProject: null });
    window.history.replaceState({}, '', '/');
});

afterEach(() => {
    // Unmount before the adapter is restored, so a late request of this test's panels never reaches the network.
    cleanup();
    resetStore();
    api.defaults.adapter = originalAdapter;
    vi.useRealTimers();
});

describe('responseStartsJob', () => {
    const res = (method: string, url: string, status: number, data: unknown) => ({ status, data, config: { method, url } });

    it('counts a 202 write, a Job body, a body carrying a Job, and Train > Launch', () => {
        expect(responseStartsJob(res('post', '/projects/4/evaluation/run-heldout?async_job=true', 202, { id: 1 }))).toBe(true);
        expect(responseStartsJob(res('post', '/projects/4/synthetic/run-playbook', 200, job(7, 'synth_playbook', 'queued')))).toBe(true);
        expect(responseStartsJob(res('post', '/projects/4/x', 201, { job: job(8, 'x', 'queued') }))).toBe(true);
        expect(responseStartsJob(res('post', '/projects/4/training/experiments/21/start', 200, { task_id: 't1' }))).toBe(true);
    });

    it('does not count reads or ordinary writes', () => {
        expect(responseStartsJob(res('get', '/jobs/12', 200, job(12, 'x', 'running')))).toBe(false);
        expect(responseStartsJob(res('get', '/projects/4/evaluation/run-heldout', 202, {}))).toBe(false);
        expect(responseStartsJob(res('post', '/projects/4/training/experiments', 201, { id: 21, status: 'pending' }))).toBe(false);
        expect(responseStartsJob(res('put', '/projects/4/settings/auto-lift-eval', 200, { enabled: true }))).toBe(false);
        expect(responseStartsJob(res('post', '/projects/4/training/experiments/21/cancel', 200, { status: 'cancelled' }))).toBe(false);
    });
});

describe('jobs store wake-up', () => {
    it('Train > Launch wakes the poll; it runs until the training Job ends, then Results refreshes and opens the Score form; then it stops', async () => {
        let runStatus = 'pending';
        let jobStatus: string | null = null;
        handler = (config) => {
            const method = String(config.method || 'get').toLowerCase();
            const url = String(config.url || '');
            if (url.startsWith('/jobs/active')) {
                const jobs = jobStatus ? [job(90, 'training_start', jobStatus)] : [];
                return { data: { count: jobs.length, jobs } };
            }
            if (method === 'post') {
                if (url.endsWith('/training/experiments/preflight')) {
                    return { data: { preflight: { ok: true, errors: [], warnings: [] }, resolved_training_config: {}, profile_defaults_applied: [] } };
                }
                if (url.endsWith('/training/experiments/21/start')) {
                    runStatus = 'running';
                    jobStatus = 'running';
                    return { data: { task_id: 't1' } };
                }
                return { data: {} };
            }
            if (url.includes('/training/runtimes')) return { data: { project_id: 4, default_runtime_id: 'auto', runtimes: [] } };
            if (url.endsWith('/training/experiments')) {
                return { data: [{ id: 21, name: 'run-21', status: runStatus, started_at: null, base_model: 'Org/Base-0.5B', config: { num_epochs: 3 }, created_at: '2026-11-20T12:00:00Z' }] };
            }
            if (url.includes('/preflight')) return { data: { preflight: { ok: true, errors: [], warnings: [] } } };
            if (url.includes('/eval-sets')) return { data: { eval_sets: [] } };
            if (url.match(/\/evaluation\/results\/\d+$/)) return { data: [] };
            if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
            if (url.includes('/evaluation/pack-preference')) {
                return { data: { preferred_pack_id: null, active_pack_id: null, active_pack_source: 'default', active_pack: { display_name: 'Auto' } } };
            }
            if (url.includes('/evaluation/safety-scorecard/')) return { data: { overall_risk: 'low', red_flags: [] } };
            if (url.includes('/evaluation/gates/')) {
                return { data: { captured_at: '2026-03-26T00:00:00Z', passed: true, failed_gate_ids: [], missing_required_metrics: [], checks: [] } };
            }
            if (url.includes('remediation-plans')) return { data: { project_id: 4, count: 0, plans: [] } };
            if (url.includes('/gold/review')) return { data: { total: 0, rows: [] } };
            return { data: {} };
        };

        const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
        // Train page: the store is idle (nothing in flight, bell closed).
        const train = render(<StrictMode><TrainingPanel projectId={4} /></StrictMode>);
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
        await screen.findByTestId('train-form-waiting-run');
        expect(useJobsStore.getState().isPolling).toBe(false);
        await user.click(screen.getByTestId('train-form-train'));
        await user.click(await screen.findByRole('button', { name: 'Launch' }));
        await waitFor(() => expect(calls.some((c) => c.method === 'post' && c.url.endsWith('/training/experiments/21/start'))).toBe(true));

        // Launch woke the poll: the training Job is seen at once and the loop keeps running while it is in flight.
        await waitFor(() => expect(useJobsStore.getState().jobs.map((j) => j.status)).toEqual(['running']));
        expect(useJobsStore.getState().isPolling).toBe(true);
        train.unmount();

        // Results page (StrictMode, as main.tsx renders it) while the run trains.
        render(<StrictMode><MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter></StrictMode>);
        expect(await screen.findByRole('button', { name: /run-21/ })).toBeInTheDocument();
        const before = jobsActiveCalls();
        await tick(4000);
        await tick(4000);
        expect(jobsActiveCalls()).toBeGreaterThanOrEqual(before + 2);
        expect(useJobsStore.getState().isPolling).toBe(true);
        expect(screen.queryByTestId('eval-run-form')).toBeNull();

        // Training finishes server-side; the next poll sees it, Results reloads the run list and opens the form.
        runStatus = 'completed';
        jobStatus = 'succeeded';
        await tick(4000);
        expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();

        // Nothing in flight and the bell is closed: the poll stopped itself.
        await waitFor(() => expect(useJobsStore.getState().isPolling).toBe(false));
        const idle = jobsActiveCalls();
        await tick(12000);
        expect(jobsActiveCalls()).toBe(idle);
    });

    // The other Job starters (synth, evals, exports of reports, RAG comparison, project clone, calibration, crawls,
    // benchmarks, pipelines…) answer 202 or a Job body; each wakes the poll through the same hook.
    for (const [label, url, status, body] of [
        ['a held-out eval (202)', '/projects/4/evaluation/run-heldout?async_job=true', 202, job(51, 'heldout_eval', 'queued')],
        ['a synth playbook run (202)', '/projects/4/synthetic/run-playbook?async_job=true', 202, job(52, 'synth_playbook', 'queued')],
        ['a reroute-to-RAG clone (202)', '/projects/4/reroute-to-rag?async_job=true', 202, job(53, 'reroute_to_rag', 'queued')],
        ['a calibration reference labelling (202)', '/projects/4/synthetic/calibration/reference-labelling', 202, job(54, 'judge_calibration_reference', 'queued')],
        ['an auto-RAG comparison (Job body, 200)', '/projects/4/auto-rag/comparison/run', 200, job(55, 'auto_rag_comparison', 'queued')],
        ['a lift check (202, no Job body)', '/projects/4/evaluation/summary/lift-check', 202, { started: true, job_id: 56 }],
    ] as const) {
        it(`${label} wakes the poll and it runs until that Job ends`, async () => {
            let jobStatus: string | null = null;
            const jobId = 'id' in body ? body.id : body.job_id;
            handler = (config) => {
                const u = String(config.url || '');
                if (u.startsWith('/jobs/active')) {
                    const jobs = jobStatus ? [job(jobId, 'x', jobStatus)] : [];
                    return { data: { count: jobs.length, jobs } };
                }
                if (u === url) {
                    jobStatus = 'running';
                    return { status, data: body };
                }
                return { data: {} };
            };
            expect(useJobsStore.getState().isPolling).toBe(false);
            await act(async () => { await api.post(url, {}); });
            await waitFor(() => expect(useJobsStore.getState().jobs.map((j) => j.id)).toEqual([jobId]));
            expect(useJobsStore.getState().isPolling).toBe(true);
            const n = jobsActiveCalls();
            await tick(4000);
            expect(jobsActiveCalls()).toBe(n + 1);
            jobStatus = 'succeeded';
            await tick(4000);
            await waitFor(() => expect(useJobsStore.getState().isPolling).toBe(false));
            const idle = jobsActiveCalls();
            await tick(8000);
            expect(jobsActiveCalls()).toBe(idle);
        });
    }

    it('an ordinary write does not start the poll; an idle read stops it (bell closed)', async () => {
        handler = (config) => (String(config.url).startsWith('/jobs/active') ? { data: { count: 0, jobs: [] } } : { data: { ok: true } });
        await act(async () => { await api.post('/projects/4/settings/whatever', {}); });
        await tick(8000);
        expect(useJobsStore.getState().isPolling).toBe(false);
        expect(jobsActiveCalls()).toBe(0);
        // A wake with nothing in flight reads once and stops.
        await act(async () => { await useJobsStore.getState().refreshAfterLocalChange(); });
        expect(useJobsStore.getState().isPolling).toBe(false);
        await tick(8000);
        expect(jobsActiveCalls()).toBe(1);
    });

    it('a page load mid-job (the bell mount read) starts the poll and stops it when the Job ends', async () => {
        let jobStatus = 'running';
        handler = (config) => (String(config.url).startsWith('/jobs/active')
            ? { data: { count: 1, jobs: [job(60, 'training_start', jobStatus)] } }
            : { data: {} });
        await act(async () => { await useJobsStore.getState().fetchOnce(); });
        expect(useJobsStore.getState().isPolling).toBe(true);
        await tick(4000);
        expect(jobsActiveCalls()).toBe(2);
        jobStatus = 'succeeded';
        await tick(4000);
        expect(useJobsStore.getState().isPolling).toBe(false);
        await tick(8000);
        expect(jobsActiveCalls()).toBe(3);
    });

    it('a read already in flight when a Job starts cannot stop the loop with its stale answer', async () => {
        let release: (() => void) | null = null;
        let jobStarted = false;
        handler = () => undefined;
        api.defaults.adapter = async (config) => {
            const url = String(config.url || '');
            calls.push({ method: String(config.method), url });
            if (url.startsWith('/jobs/active')) {
                const stale = !jobStarted;
                if (stale) await new Promise<void>((r) => { release = r; });
                const jobs = stale ? [] : [job(70, 'synth_playbook', 'running')];
                return { data: { count: jobs.length, jobs }, status: 200, statusText: 'OK', headers: {}, config };
            }
            jobStarted = true;
            return { data: job(70, 'synth_playbook', 'queued'), status: 202, statusText: 'OK', headers: {}, config };
        };
        // A read (the bell's mount read) is in flight and will answer "nothing running" from before the Job existed.
        const mountRead = useJobsStore.getState().fetchOnce();
        await act(async () => { await api.post('/projects/4/synthetic/run-playbook?async_job=true', {}); });
        await act(async () => { release!(); await mountRead; });
        await waitFor(() => expect(useJobsStore.getState().jobs.map((j) => j.id)).toEqual([70]));
        expect(useJobsStore.getState().isPolling).toBe(true);
    });
});
