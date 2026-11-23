/**
 * UI simplification phase 7g (gap G2): after a successful Run the Score this run form collapses, a one-line
 * "Queued N evals: …" stands in its place, the toggle keeps one label ("Score this run"), the form is remounted per
 * submit, and the 7e default never re-opens it by itself after the Run (it did when the run's completion landed later).
 */

import { StrictMode } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, formMounts } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    formMounts: { count: 0 },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../training/SqlExecutionConfigCard', () => ({ default: () => <div data-testid="sql-execution-card" /> }));
vi.mock('../training/AutoLiftSettingCard', () => ({ default: () => <div data-testid="auto-lift-card" /> }));
vi.mock('../training/ToolScoringConfigCard', () => ({ default: () => <div data-testid="tool-scoring-card" /> }));
vi.mock('./EvalRunForm', async (importOriginal) => {
    const actual = await importOriginal<typeof import('./EvalRunForm')>();
    const { useEffect } = await import('react');
    function StubForm({ onQueued }: { onQueued?: (text: string) => void }) {
        useEffect(() => { formMounts.count += 1; }, []);
        return (
            <div data-testid="eval-run-form">
                <button
                    type="button"
                    data-testid="stub-run"
                    onClick={() => onQueued?.('Queued 2 evals: fine-tuned, base model (jobs #7, #8). The bell says when they are ready.')}
                >
                    Run
                </button>
            </div>
        );
    }
    return { ...actual, default: StubForm };
});

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';
import { useJobsStore } from '../../stores/jobsStore';
import type { Job } from '../../api/jobs';

function mockApi(runs: () => Array<Record<string, unknown>>) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/eval-sets')) return { data: { eval_sets: [] } };
        if (url.includes('/training/experiments')) return { data: runs() };
        if (url.endsWith('/evaluation/summary')) return { data: {} };
        if (/\/evaluation\/results\/\d+$/.test(url)) return { data: [] };
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) {
            return { data: { preferred_pack_id: null, active_pack_id: null, active_pack_source: 'default', active_pack: { display_name: 'Auto' } } };
        }
        if (url.includes('/evaluation/safety-scorecard/')) return { data: { overall_risk: 'low', red_flags: [] } };
        if (url.includes('/evaluation/gates/')) {
            return { data: { captured_at: '2026-03-26T00:00:00Z', passed: true, failed_gate_ids: [], missing_required_metrics: [], checks: [] } };
        }
        if (url.includes('remediation-plans')) return { data: { project_id: 4, count: 0, plans: [] } };
        return { data: {} };
    });
}

const trainingJob = (id: number, status: Job['status']): Job => ({
    id, kind: 'training_start', title: 'Training', status, progress: null, progress_message: null, project_id: 4,
    user_id: null, params: {}, result: null, error: null, queued_at: null, started_at: null, completed_at: null, dismissed_at: null,
});

const listCalls = () => apiMock.get.mock.calls.filter(([u]) => String(u).endsWith('/training/experiments')).length;

describe('EvalPanel: Score this run after Run (phase 7g, G2)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        formMounts.count = 0;
        useProjectStore.setState({ activeProject: null });
        useJobsStore.setState({ jobs: [] });
    });

    for (const strict of [false, true]) {
        it(`collapses to a queued line with a stable label, and stays closed when training ends later${strict ? ' (StrictMode)' : ''}`, async () => {
            let status = 'running';
            mockApi(() => [{ id: 21, name: 'exp-21', status, base_model: 'Org/Base-0.5B' }]);
            act(() => { useJobsStore.setState({ jobs: [trainingJob(90, 'running')] }); });
            const user = userEvent.setup();
            const panel = <MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>;
            render(strict ? <StrictMode>{panel}</StrictMode> : panel);
            const toggle = await screen.findByTestId('eval-score-run-open');
            expect(toggle).toHaveTextContent(/^Score this run$/);
            // The user opens it before the run list knows training ended (the 7f order), and runs.
            await user.click(toggle);
            await user.click(await screen.findByTestId('stub-run'));
            expect(screen.queryByTestId('eval-run-form')).toBeNull();
            expect(screen.getByTestId('eval-score-run-queued')).toHaveTextContent('Queued 2 evals: fine-tuned, base model (jobs #7, #8)');
            expect(screen.getByTestId('eval-score-run-open')).toHaveTextContent(/^Score this run$/);
            expect(screen.getByTestId('eval-score-run-open')).toHaveAttribute('data-open', 'false');
            // Training ends now: the run list reloads, the run has no evals yet, and the 7e default must not re-open it.
            const before = listCalls();
            status = 'completed';
            act(() => { useJobsStore.setState({ jobs: [trainingJob(90, 'succeeded')] }); });
            await waitFor(() => expect(listCalls()).toBeGreaterThan(before));
            await new Promise((r) => setTimeout(r, 50));
            expect(screen.queryByTestId('eval-run-form')).toBeNull();
            expect(screen.getByTestId('eval-score-run-queued')).toBeInTheDocument();
            // Re-opening drops the line and mounts a fresh form (state reset per submit).
            const mountsBefore = formMounts.count;
            await user.click(screen.getByTestId('eval-score-run-open'));
            expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();
            expect(formMounts.count).toBeGreaterThan(mountsBefore);
            expect(screen.queryByTestId('eval-score-run-queued')).toBeNull();
            expect(screen.getByTestId('eval-score-run-open')).toHaveTextContent(/^Score this run$/);
        });
    }
});
