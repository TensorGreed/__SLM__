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
        // Toast announces the new experiment id; nav lands on Train → Plan and gaps.
        expect(toastMock.success).toHaveBeenCalledWith(
            expect.stringContaining('99'),
        );
        expect(navigateMock).toHaveBeenCalledWith('/project/5/train?open=plan');
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

    it('an older response without row_evidence renders no evidence block', async () => {
        apiMock.get.mockResolvedValue({ data: makeCompareResponse() });
        renderPage('?a=11&b=12');
        await screen.findByTestId('eval-compare');
        expect(screen.queryByTestId('eval-compare-evidence')).not.toBeInTheDocument();
    });

    it('puts the row-paired evidence block at the top', async () => {
        apiMock.get.mockResolvedValue({
            data: makeCompareResponse({
                row_evidence: {
                    verdict: 'within_noise', reason: null, message: null,
                    metric_id: 'exact_match', score_key: 'correct',
                    n: 78, better: 2, worse: 7, same: 69,
                    mean_diff: -5 / 78, ci_low: -0.13937, ci_high: 0.01117,
                    only_in_a: 0, only_in_b: 0, unscored: 0, metrics: [],
                    comparability: { comparable: true, differences: [], notes: [] },
                    rescore: null,
                },
            }),
        });
        renderPage('?a=11&b=12');
        const block = await screen.findByTestId('eval-compare-evidence');
        expect(block.textContent).toContain(
            'B is within noise of A: 2 better, 7 worse, 69 same of 78 rows; change −6.4 pts (95% CI −14 to +1)',
        );
        // Above the side cards.
        const summary = screen.getByTestId('eval-compare-summary');
        expect(block.compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });

    it('surfaces an error when the URL lacks a / b params', async () => {
        renderPage('');
        await waitFor(() => {
            expect(screen.getByTestId('eval-compare-error')).toBeInTheDocument();
        });
        expect(apiMock.get).not.toHaveBeenCalled();
    });

    it('shows the prompt format per side and the backend note when formats differ', async () => {
        const prov = (id: number, v?: number) => ({
            experiment_id: id, experiment_name: `e${id}`, eval_result_id: id, evaluated_at: null,
            answer_key: null, judge: null, has_row_scores: true,
            scoring: { eval_type: 'f1', scoring_mode: null, fingerprints: {}, ...(v ? { prompt_format_version: v, system_rows: 3 } : {}) },
        });
        const base = makeCompareResponse();
        apiMock.get.mockResolvedValue({ data: makeCompareResponse({
            a: { ...base.a, provenance: prov(11) },
            b: { ...base.b, provenance: prov(12, 2) },
            row_evidence: {
                a_eval_result_id: 91, b_eval_result_id: 92, headline_metric_id: 'f1', n: 0, better: 0, worse: 0, same: 0,
                mean_diff: null, ci_low: null, ci_high: null, only_in_a: 0, only_in_b: 0, unscored: 0, metrics: [],
                verdict: 'not_comparable', reason: null, metric_id: null, rescore: null,
                comparability: { comparable: false, differences: [{ kind: 'scoring', message: 'The two results used different prompt formats (v1 vs v2).', a: 1, b: 2 }], notes: [] },
            } as any,
        }) });
        renderPage('?a=11&b=12');
        await waitFor(() => expect(screen.getByTestId('eval-compare-format-a')).toHaveTextContent('prompt rendering v1'));
        expect(screen.getByTestId('eval-compare-format-b')).toHaveTextContent('prompt rendering v2');
        expect(screen.getByText(/different prompt formats/)).toBeInTheDocument();
    });

    it('offers the evaluated eval sets and compares on the picked one', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/eval-sets')) {
                return { data: { eval_sets: [{ slug: 'fresh', last_evals: { 'eval_set:fresh@v1+native': { eval_result_id: 3 } } }] } };
            }
            return { data: makeCompareResponse() };
        });
        const user = userEvent.setup();
        renderPage('?a=11&b=12');
        const select = await screen.findByTestId('eval-compare-set-select');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/5/evaluation/compare', { params: { a: 11, b: 12 } });
        expect(select).toHaveTextContent('eval set fresh v1 (native)');
        await user.selectOptions(select, 'eval_set:fresh@v1+native');
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/5/evaluation/compare', { params: { a: 11, b: 12, dataset_name: 'eval_set:fresh@v1+native' } },
        ));
    });

    it('base vs fine-tuned reads as a lift: roles on the cards, no "B regressed", no rerun offer', async () => {
        apiMock.get.mockResolvedValue({
            data: makeCompareResponse({
                winner: 'a', regressed: false,
                lift: { base_side: 'a', swapped: true, label: 'fine-tuned (B) vs base model (A)' },
            }),
        });
        renderPage('?a=12&b=11');
        await waitFor(() => expect(screen.getByTestId('eval-compare')).toBeInTheDocument());
        expect(screen.getByTestId('eval-compare-side-a')).toHaveTextContent('A · base model');
        expect(screen.getByTestId('eval-compare-side-b')).toHaveTextContent('B · fine-tuned');
        expect(screen.getByTestId('eval-compare-verdict')).toHaveTextContent('Fine-tuned below base');
        expect(screen.getByTestId('eval-compare-lift')).toHaveTextContent('the base model is shown on A');
        expect(screen.queryByText('B regressed')).toBeNull();
        expect(screen.queryByTestId('eval-compare-fix-gap')).toBeNull();
    });

    it('offers the base model on another prompt format, off by default', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/eval-sets')) {
                return { data: { eval_sets: [{ slug: 'fresh', last_evals: {
                    'eval_set:fresh@v1+native': { eval_result_id: 3 },
                    'eval_set:fresh@v1+instructed': { eval_result_id: 4, baseline: true },
                } }] } };
            }
            return { data: makeCompareResponse({ regressed: false, lift: { base_side: 'a', swapped: false, label: 'x' } }) };
        });
        const user = userEvent.setup();
        renderPage('?a=11&b=12&set=eval_set:fresh@v1%2Bnative');
        const base = await screen.findByTestId('eval-compare-base-variant-select');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/5/evaluation/compare',
            { params: { a: 11, b: 12, dataset_name: 'eval_set:fresh@v1+native' } });
        expect((base as HTMLSelectElement).value).toBe('eval_set:fresh@v1+native');
        await user.selectOptions(base, 'eval_set:fresh@v1+instructed');
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith('/projects/5/evaluation/compare', {
            params: { a: 11, b: 12, dataset_name: 'eval_set:fresh@v1+native', base_dataset_name: 'eval_set:fresh@v1+instructed' },
        }));
    });

    it('offers no base-model picker when neither side is a baseline run', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/eval-sets')) {
                return { data: { eval_sets: [{ slug: 'fresh', last_evals: {
                    'eval_set:fresh@v1+native': { eval_result_id: 3 },
                    'eval_set:fresh@v1+instructed': { eval_result_id: 4, baseline: true },
                } }] } };
            }
            return { data: makeCompareResponse() };
        });
        renderPage('?a=11&b=12&set=eval_set:fresh@v1%2Bnative');
        await screen.findByTestId('eval-compare-set-select');
        expect(screen.queryByTestId('eval-compare-base-variant')).toBeNull();
    });

    it('shows no eval-set picker when the project has no eval-set results', async () => {
        apiMock.get.mockResolvedValue({ data: makeCompareResponse() });
        renderPage('?a=11&b=12');
        await waitFor(() => expect(screen.getByTestId('eval-compare')).toBeInTheDocument());
        expect(screen.queryByTestId('eval-compare-set-picker')).toBeNull();
    });

    it('O1: on a base vs fine-tuned pair the cluster headings read as the fine-tune\'s effect', async () => {
        apiMock.get.mockResolvedValue({
            data: makeCompareResponse({ regressed: false, lift: { base_side: 'a', swapped: false, label: 'fine-tuned (B) vs base model (A)' } }),
        });
        renderPage('?a=11&b=12');
        expect(await screen.findByTestId('eval-compare-only-b')).toHaveTextContent('Fails only with the fine-tune');
        expect(screen.getByTestId('eval-compare-only-a')).toHaveTextContent('Fixed by the fine-tune');
        expect(screen.queryByText('New in B (regressions)')).toBeNull();
    });

    it('O2: with a task scoring mode the exact_match row sits under "Other stored metrics (string match)"', async () => {
        const scoring = { eval_type: 'exact_match', scoring_mode: 'tool_call', fingerprints: {} };
        const prov = (id: number) => ({ experiment_id: id, experiment_name: 'x', eval_result_id: id, evaluated_at: null, answer_key: null, judge: null, scoring, has_row_scores: true });
        const base = makeCompareResponse();
        apiMock.get.mockResolvedValue({
            data: makeCompareResponse({
                a: { ...base.a, provenance: prov(91) },
                b: { ...base.b, provenance: prov(92) },
                metric_deltas: [
                    { metric_id: 'tool_call_pass_rate', a_value: 0.65, b_value: 0.45, delta: -0.2, direction: 'regressed', higher_is_better: true },
                    { metric_id: 'exact_match', a_value: 0.65, b_value: 0.45, delta: -0.2, direction: 'regressed', higher_is_better: true },
                ],
            }),
        });
        renderPage('?a=11&b=12');
        const aside = await screen.findByTestId('eval-compare-string-match');
        expect(aside).toHaveTextContent('Other stored metrics (string match) (1)');
        expect(aside).toContainElement(screen.getByTestId('eval-compare-metric-exact_match'));
        expect(screen.getByTestId('eval-compare-metrics')).toContainElement(screen.getByTestId('eval-compare-metric-tool_call_pass_rate'));
        expect(aside).not.toContainElement(screen.getByTestId('eval-compare-metric-tool_call_pass_rate'));
    });

    it('O2: without a task scoring mode exact_match stays in the main table', async () => {
        apiMock.get.mockResolvedValue({ data: makeCompareResponse() });
        renderPage('?a=11&b=12');
        await screen.findByTestId('eval-compare-metric-exact_match');
        expect(screen.queryByTestId('eval-compare-string-match')).toBeNull();
    });
});
