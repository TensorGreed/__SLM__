/**
 * UI simplification phase 7e (gap G3): on Results › Verdict, a trained run with no evals opens the "Score this run"
 * form by default (the card already says "evaluate it now"); a run with any eval keeps it closed, as before. The
 * button still toggles and names what it does.
 */

import { render, screen, waitFor } from '@testing-library/react';
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

const RESULT = { id: 1002, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 1, metrics: { exact_match: 1 }, details: {} };

function mockApi(runs: Array<Record<string, unknown>>, resultsByRun: Record<number, unknown[]>) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/eval-sets')) return { data: { eval_sets: [] } };
        if (url.includes('/training/experiments')) return { data: runs };
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
});
