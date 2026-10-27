/**
 * Multi-turn eval surfaces: the per-turn judge toggle, its summary, and the
 * preference-pairs-from-eval action.
 */

import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';

const DJ = {
    rows: 4,
    mean: 0.75,
    worst_turn_mean: 0.5,
    by_turn_index: [
        { turn: 1, mean: 0.9, n: 4 },
        { turn: 2, mean: 0.6, n: 4 },
    ],
};

function mockApi(metrics: Record<string, unknown>) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/experiments')) return { data: [{ id: 21, name: 'exp-21' }] };
        if (url.includes('/evaluation/results/21')) {
            return {
                data: [{
                    id: 1001,
                    dataset_name: 'gold_dev',
                    eval_type: 'f1',
                    pass_rate: 0.5,
                    metrics: { f1: 0.4, exact_match: 0.2, ...metrics },
                    details: {},
                }],
            };
        }
        if (url.includes('/evaluation/safety-scorecard/21')) return { data: { overall_risk: 'low', red_flags: [] } };
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) {
            return { data: { preferred_pack_id: null, active_pack_id: null, active_pack_source: 'default', active_pack: { display_name: 'Auto' } } };
        }
        if (url.includes('/evaluation/gates/21')) {
            return { data: { captured_at: '2026-03-26T00:00:00Z', passed: true, failed_gate_ids: [], missing_required_metrics: [], checks: [] } };
        }
        if (url.includes('/evaluation/scorecard/21')) {
            return { data: { experiment_id: 21, is_ship: true, decision: 'SHIP', reasons: [], failed_gates: [], missing_metrics: [],
                gate_report: { passed: true, checks: [], missing_required_metrics: [], failed_gate_ids: [] } } };
        }
        if (url.includes('failure-clusters')) {
            return { data: { eval_result_id: 1001, experiment_id: 21, dataset_name: 'gold_dev', eval_type: 'llm_judge',
                total_failures_analyzed: 0, reason_code_totals: {}, dominant_reason_code: null, clusters: [], remediation_plans: [] } };
        }
        if (url.includes('remediation-plans')) return { data: { project_id: 4, count: 0, plans: [] } };
        if (url.includes('/gold/review')) return { data: { total: 0, rows: [] } };
        return { data: {} };
    });
}

describe('EvalPanel multi-turn surfaces', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        useProjectStore.setState({ activeProject: null });
    });

    it('sends dialogue_judge: true when the toggle is on', async () => {
        mockApi({});
        apiMock.post.mockResolvedValue({ data: { id: 7 } });
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        await user.click(await screen.findByRole('button', { name: /Evaluate on test examples/ }));
        await user.click(await screen.findByTestId('eval-dialogue-judge-toggle'));
        await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        const [url, body] = apiMock.post.mock.calls[0];
        expect(url).toContain('/evaluation/run-heldout');
        expect(body.dialogue_judge).toBe(true);
    });

    it('omits dialogue_judge by default', async () => {
        mockApi({});
        apiMock.post.mockResolvedValue({ data: { id: 7 } });
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        await user.click(await screen.findByRole('button', { name: /Evaluate on test examples/ }));
        await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect(apiMock.post.mock.calls[0][1]).not.toHaveProperty('dialogue_judge');
    });

    it('renders the dialogue_judge summary with run and result provenance', async () => {
        mockApi({ dialogue_judge: DJ });
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('dialogue-judge-panel')).toHaveTextContent('Run #21 · gold_dev · result 1001');
        expect(screen.getByTestId('dj-mean')).toHaveTextContent('75.0%');
        expect(screen.getByTestId('dj-worst')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('dj-turn-2')).toHaveTextContent('60.0%');
    });

    it('says when the replay cap left multi-turn rows out', async () => {
        mockApi({ dialogue_judge: { ...DJ, rows_not_replayed: 3, replay_cap_hit: true } });
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('dj-cap-hit')).toHaveTextContent('3 multi-turn rows were not replayed');
    });

    it('hides the dialogue_judge summary when absent, but offers pairs once the answer judge scored rows', async () => {
        mockApi({ judge: { judged: 12 } });
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        await screen.findByTestId('pairs-from-eval-1001');
        expect(screen.queryByTestId('dialogue-judge-panel')).toBeNull();
    });

    it('offers pairs when only the per-turn judge scored rows (exact-match eval type)', async () => {
        mockApi({ dialogue_judge: DJ });
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('pairs-from-eval-1001')).toBeInTheDocument();
    });

    it('offers no pairs when no judge scored a row', async () => {
        mockApi({ judge: { judged: 0, skipped: 'no_judge_available' }, dialogue_judge: { skipped: 'no_judge_available' } });
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('dj-skipped')).toHaveTextContent('Per-turn judge skipped: no_judge_available');
        expect(screen.queryByTestId('pairs-from-eval-1001')).toBeNull();
    });
});
