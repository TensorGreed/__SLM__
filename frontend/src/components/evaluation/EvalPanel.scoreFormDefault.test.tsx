/**
 * UI simplification phase 7e (gap G3): on Results › Verdict, a trained run with no evals opens the "Score this run"
 * form by default (the card already says "evaluate it now"); a run with any eval keeps it closed, as before. The
 * button still toggles and names what it does.
 */

import { act, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../training/SqlExecutionConfigCard', () => ({ default: () => <div data-testid="sql-execution-card" /> }));
vi.mock('../training/AutoLiftSettingCard', () => ({ default: () => <div data-testid="auto-lift-card" /> }));
vi.mock('../training/ToolScoringConfigCard', () => ({ default: () => <div data-testid="tool-scoring-card" /> }));

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';
import { useJobsStore } from '../../stores/jobsStore';
import type { Job } from '../../api/jobs';

const RESULT = { id: 1002, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 1, metrics: { exact_match: 1 }, details: {} };

type Runs = Array<Record<string, unknown>>;

/** ``runs`` may be a function: the run list as it is now (S1: it changes when training finishes). ``summaryByRun``:
 *  the summary card's answer per run (default ``{}``, a run that can be evaluated). */
function mockApi(runs: Runs | (() => Runs), resultsByRun: Record<number, unknown[]>, summaryByRun: Record<number, Record<string, unknown>> = {}) {
    apiMock.get.mockImplementation(async (url: string, config?: { params?: { experiment_id?: number } }) => {
        if (url.includes('/eval-sets')) return { data: { eval_sets: [] } };
        if (url.includes('/seed-groups/')) return { data: { leader_id: 21, children: [], recommended_experiment_id: null } };
        if (url.includes('/training/experiments')) return { data: typeof runs === 'function' ? runs() : runs };
        if (url.endsWith('/evaluation/summary')) {
            const id = config?.params?.experiment_id;
            return { data: (id != null ? summaryByRun[id] : undefined) ?? {} };
        }
        const results = url.match(/\/evaluation\/results\/(\d+)$/);
        if (results) return { data: resultsByRun[Number(results[1])] ?? [] };
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
    });
}

function resultsFetched(runId: number): boolean {
    return apiMock.get.mock.calls.some(([url]) => String(url).endsWith(`/evaluation/results/${runId}`));
}

function renderPanel() {
    const user = userEvent.setup();
    render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
    return user;
}

describe('EvalPanel: the Score this run form on a run with no evals (phase 7e, G3)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        useProjectStore.setState({ activeProject: null });
        useJobsStore.setState({ jobs: [] });
    });

    it('a trained run with no evals opens the form by default; the button closes it', async () => {
        mockApi([{ id: 21, name: 'exp-21', status: 'completed', base_model: 'Org/Base-0.5B' }], { 21: [] });
        const user = renderPanel();
        expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();
        expect(screen.getByTestId('eval-score-run-open')).toHaveTextContent('Close Score this run');
        await user.click(screen.getByTestId('eval-score-run-open'));
        expect(screen.queryByTestId('eval-run-form')).toBeNull();
        expect(screen.getByTestId('eval-score-run-open')).toHaveTextContent('+ Score this run');
    });

    it('a run with any eval keeps the form closed by default', async () => {
        mockApi([{ id: 21, name: 'exp-21', status: 'completed', base_model: 'Org/Base-0.5B' }], { 21: [RESULT] });
        const user = renderPanel();
        await waitFor(() => expect(resultsFetched(21)).toBe(true));
        expect(await screen.findByTestId('eval-score-run-open')).toHaveTextContent('+ Score this run');
        await new Promise((r) => setTimeout(r, 50));
        expect(screen.queryByTestId('eval-run-form')).toBeNull();
        await user.click(screen.getByTestId('eval-score-run-open'));
        expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();
    });

    it('a run that has not finished training keeps the form closed', async () => {
        mockApi([{ id: 21, name: 'exp-21', status: 'running', base_model: 'Org/Base-0.5B' }], { 21: [] });
        renderPanel();
        await waitFor(() => expect(resultsFetched(21)).toBe(true));
        await screen.findByTestId('eval-score-run-open');
        await new Promise((r) => setTimeout(r, 50));
        expect(screen.queryByTestId('eval-run-form')).toBeNull();
    });

    it('switching from a run with no evals (form opened by default) to one with evals closes it', async () => {
        mockApi([
            { id: 22, name: 'exp-22', status: 'completed', base_model: 'Org/Base-0.5B' },
            { id: 21, name: 'exp-21', status: 'completed', base_model: 'Org/Base-0.5B' },
        ], { 22: [], 21: [RESULT] });
        const user = renderPanel();
        expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        await waitFor(() => expect(screen.queryByTestId('eval-run-form')).toBeNull());
    });
    // Review of 7e, M1: the runs the summary card says cannot be evaluated (the same source: ``cannotBeEvaluated``),
    // and a multi-seed leader (its seeds hold the weights), never open the form by default.
    for (const [label, summary] of [
        ['a simulated run', { experiment_id: 21, verdict: 'not_evaluated', auto_eval_skip_reason: 'simulated_run', failures: [], headline: null }],
        ['a run that saved no model weights', { experiment_id: 21, verdict: 'not_evaluated', auto_eval_skip_reason: 'no_model_weights', failures: [], headline: null }],
        ['a multi-seed leader', { experiment_id: 21, verdict: 'not_evaluated', seed_group_leader: true, failures: [], headline: null }],
    ] as const) {
        it(`${label} with no evals keeps the form closed (review M1)`, async () => {
            mockApi([{ id: 21, name: 'exp-21', status: 'completed', base_model: 'Org/Base-0.5B' }], { 21: [] }, { 21: summary });
            const user = renderPanel();
            await waitFor(() => expect(resultsFetched(21)).toBe(true));
            await waitFor(() => expect(apiMock.get.mock.calls.some(([url, cfg]) => String(url).endsWith('/evaluation/summary')
                && (cfg as { params?: { experiment_id?: number } } | undefined)?.params?.experiment_id === 21)).toBe(true));
            await new Promise((r) => setTimeout(r, 50));
            expect(screen.queryByTestId('eval-run-form')).toBeNull();
            expect(screen.getByTestId('eval-score-run-open')).toHaveTextContent('+ Score this run');
            // The button still opens it (a user pick).
            await user.click(screen.getByTestId('eval-score-run-open'));
            expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();
        });
    }

    const trainingJob = (id: number, status: Job['status'], projectId = 4): Job => ({
        id, kind: 'training_start', title: 'Training', status, progress: null, progress_message: null, project_id: projectId,
        user_id: null, params: {}, result: null, error: null, queued_at: null, started_at: null, completed_at: null, dismissed_at: null,
    });

    it('a run that finishes training while Results is open: the run list reloads and the form opens once (review S1)', async () => {
        let status = 'running';
        mockApi(() => [{ id: 21, name: 'exp-21', status, base_model: 'Org/Base-0.5B' }], { 21: [] });
        act(() => { useJobsStore.setState({ jobs: [trainingJob(90, 'running')] }); });
        const user = renderPanel();
        await waitFor(() => expect(resultsFetched(21)).toBe(true));
        await new Promise((r) => setTimeout(r, 50));
        expect(screen.queryByTestId('eval-run-form')).toBeNull();
        const listCalls = () => apiMock.get.mock.calls.filter(([url]) => String(url).endsWith('/training/experiments')).length;
        expect(listCalls()).toBe(1);
        // Another project's training Job finishing does not reload this list.
        act(() => { useJobsStore.setState({ jobs: [trainingJob(90, 'running'), trainingJob(91, 'succeeded', 5)] }); });
        await new Promise((r) => setTimeout(r, 30));
        expect(listCalls()).toBe(1);
        status = 'completed';
        act(() => { useJobsStore.setState({ jobs: [trainingJob(90, 'succeeded'), trainingJob(91, 'succeeded', 5)] }); });
        expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();
        expect(listCalls()).toBe(2);
        // Once: closing it sticks while the jobs store keeps polling.
        await user.click(screen.getByTestId('eval-score-run-open'));
        act(() => { useJobsStore.setState({ jobs: [trainingJob(90, 'succeeded'), trainingJob(91, 'succeeded', 5)] }); });
        await new Promise((r) => setTimeout(r, 50));
        expect(screen.queryByTestId('eval-run-form')).toBeNull();
    });

    it('a new run that finishes while Results shows an evaluated one becomes the shown run and opens the form (review S1)', async () => {
        let runs: Runs = [{ id: 21, name: 'exp-21', status: 'completed', base_model: 'Org/Base-0.5B' }];
        mockApi(() => runs, { 21: [RESULT], 22: [] });
        renderPanel();
        await waitFor(() => expect(resultsFetched(21)).toBe(true));
        await new Promise((r) => setTimeout(r, 50));
        expect(screen.queryByTestId('eval-run-form')).toBeNull();
        runs = [{ id: 22, name: 'exp-22', status: 'completed', base_model: 'Org/Base-0.5B' }, ...runs];
        act(() => { useJobsStore.setState({ jobs: [trainingJob(92, 'succeeded')] }); });
        await waitFor(() => expect(resultsFetched(22)).toBe(true));
        expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();
    });
});
