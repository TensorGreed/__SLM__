import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, Outlet } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, navigateMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        patch: vi.fn(),
        delete: vi.fn(),
    },
    navigateMock: vi.fn(),
    toastMock: {
        success: vi.fn(),
        error: vi.fn(),
        info: vi.fn(),
        warning: vi.fn(),
    },
}));

vi.mock('../api/client', () => ({ default: apiMock }));
vi.mock('../stores/toastStore', () => ({ toast: toastMock }));
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return { ...actual, useNavigate: () => navigateMock };
});

import ProjectEvalComparePage from './ProjectEvalComparePage';
import type { CompareResponse } from '../api/experimentCompare';


function ContextProvider() {
    return (
        <Outlet
            context={{
                projectId: 5,
                project: {} as any,
                pipelineStatus: null,
                refreshPipelineStatus: async () => {},
            }}
        />
    );
}


function renderPage(search: string) {
    return render(
        <MemoryRouter initialEntries={[`/project/5/eval/compare${search}`]}>
            <Routes>
                <Route path="/project/:id" element={<ContextProvider />}>
                    <Route path="eval/compare" element={<ProjectEvalComparePage />} />
                </Route>
            </Routes>
        </MemoryRouter>,
    );
}


function makeCompareResponse(overrides: Partial<CompareResponse> = {}): CompareResponse {
    return {
        project_id: 5,
        a: {
            experiment_id: 11, name: 'Run A', base_model: 'smollm-135m',
            training_mode: 'sft', status: 'completed',
            started_at: '2026-05-25T10:00:00Z',
            completed_at: '2026-05-25T11:00:00Z',
            eval_result_id: 91, eval_pass_rate: 0.70,
            eval_type: 'f1', dataset_name: 'gold_test',
            metrics: { f1: 0.70, exact_match: 0.65 },
        },
        b: {
            experiment_id: 12, name: 'Run B', base_model: 'qwen-0.5b',
            training_mode: 'sft', status: 'completed',
            started_at: '2026-05-26T10:00:00Z',
            completed_at: '2026-05-26T11:00:00Z',
            eval_result_id: 92, eval_pass_rate: 0.50,
            eval_type: 'f1', dataset_name: 'gold_test',
            metrics: { f1: 0.50, exact_match: 0.45 },
        },
        metric_deltas: [
            { metric_id: 'f1', a_value: 0.70, b_value: 0.50, delta: -0.20, direction: 'regressed', higher_is_better: true },
            { metric_id: 'exact_match', a_value: 0.65, b_value: 0.45, delta: -0.20, direction: 'regressed', higher_is_better: true },
        ],
        cluster_diff: {
            a_total: 3,
            b_total: 5,
            only_in_a: [],
            only_in_b: [{ reason_code: 'hallucination', output_pattern: 'len-med', failure_count: 2 }],
            shared: [{ reason_code: 'coverage_gap', output_pattern: 'len-short', a_count: 3, b_count: 3, delta: 0 }],
        },
        config_diff: [
            { field: 'base_model', a_value: 'smollm-135m', b_value: 'qwen-0.5b', changed: true, primary: true },
            { field: 'learning_rate', a_value: 2e-4, b_value: 1e-4, changed: true, primary: true },
            { field: 'training_mode', a_value: 'sft', b_value: 'sft', changed: false, primary: true },
        ],
        winner: 'a',
        regressed: true,
        ...overrides,
    };
}


describe('ProjectEvalComparePage', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        navigateMock.mockReset();
        toastMock.success.mockReset();
        toastMock.error.mockReset();
    });

    it('renders the comparison + Fix-the-gap CTA when B regressed', async () => {
        apiMock.get.mockResolvedValue({ data: makeCompareResponse() });
        renderPage('?a=11&b=12');

        await waitFor(() => {
            expect(screen.getByTestId('eval-compare')).toBeInTheDocument();
        });
        // Endpoint called with the right params.
        expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/5/evaluation/compare',
            { params: { a: 11, b: 12 } },
        );
        // Both side cards render + A is marked as the winner.
        expect(screen.getByTestId('eval-compare-side-a')).toHaveTextContent('Run A');
        expect(screen.getByTestId('eval-compare-side-b')).toHaveTextContent('Run B');
        const winnerCard = screen.getByTestId('eval-compare-side-a');
        expect(winnerCard.className).toContain('--winner');
        // Fix-the-gap CTA visible because regressed=true.
        expect(screen.getByTestId('eval-compare-fix-gap')).toBeInTheDocument();
    });

    it('hides the Fix-the-gap CTA when B is the winner', async () => {
        apiMock.get.mockResolvedValue({
            data: makeCompareResponse({ winner: 'b', regressed: false }),
        });
        renderPage('?a=11&b=12');
        await waitFor(() => {
            expect(screen.getByTestId('eval-compare')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('eval-compare-fix-gap')).not.toBeInTheDocument();
    });

    it('renders metric deltas with regression direction on regressed rows', async () => {
        apiMock.get.mockResolvedValue({ data: makeCompareResponse() });
        renderPage('?a=11&b=12');
        await waitFor(() => {
            expect(screen.getByTestId('eval-compare-metric-f1')).toBeInTheDocument();
        });
        const f1Row = screen.getByTestId('eval-compare-metric-f1');
        expect(f1Row).toHaveAttribute('data-direction', 'regressed');
        // Delta column renders the signed numeric.
        expect(f1Row.textContent).toMatch(/-0\.2/);
    });

    it('lists "new in B" clusters under the regressed cluster block', async () => {
        apiMock.get.mockResolvedValue({ data: makeCompareResponse() });
        renderPage('?a=11&b=12');
        const newInB = await screen.findByTestId('eval-compare-only-b');
        expect(newInB.textContent).toMatch(/New in B/);
        expect(newInB.textContent).toMatch(/hallucination/);
        expect(newInB.textContent).toMatch(/2/);
    });

    it('clicking Fix-the-gap calls rerun-from-manifest and navigates on success', async () => {
        apiMock.get.mockResolvedValue({ data: makeCompareResponse() });
        apiMock.post.mockResolvedValue({
            data: { id: 99, name: 'Rerun', base_model: 'smollm-135m', status: 'pending' },
        });

        renderPage('?a=11&b=12');
        const rerun = await screen.findByTestId('eval-compare-rerun');
        await userEvent.click(rerun);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalled();
        });
        // Posts to the rerun-from-manifest path for the winning exp (A = 11).
        expect(apiMock.post.mock.calls[0][0]).toBe(
            '/projects/5/training/runs/11/rerun-from-manifest',
        );
        // Toast announces the new experiment id; nav lands on training-config.
        expect(toastMock.success).toHaveBeenCalledWith(
            expect.stringContaining('99'),
        );
        expect(navigateMock).toHaveBeenCalledWith('/project/5/training-config');
    });

    it('shows a manifest-missing toast when rerun returns 404 manifest_not_captured', async () => {
        apiMock.get.mockResolvedValue({ data: makeCompareResponse() });
        apiMock.post.mockRejectedValue({
            response: { status: 404, data: { detail: 'manifest_not_captured' } },
        });

        renderPage('?a=11&b=12');
        const rerun = await screen.findByTestId('eval-compare-rerun');
        await userEvent.click(rerun);
        await waitFor(() => {
            expect(toastMock.error).toHaveBeenCalled();
        });
        expect(toastMock.error.mock.calls[0][0]).toMatch(/no training manifest/i);
        // Navigation does NOT happen on failure.
        expect(navigateMock).not.toHaveBeenCalled();
    });

    it('surfaces an error when the URL lacks a / b params', async () => {
        renderPage('');
        await waitFor(() => {
            expect(screen.getByTestId('eval-compare-error')).toBeInTheDocument();
        });
        expect(apiMock.get).not.toHaveBeenCalled();
    });
});
