/**
 * AutoRagComparisonPanel — Hardening tests for the "Run comparison"
 * UI affordance.
 *
 * Three scenarios cover the new flow:
 *   1. 404 empty-state renders the primary "Run comparison" CTA
 *      AND POSTs to the run endpoint on click + fires an info toast.
 *   2. Cached payload renders a "Re-run comparison" button in the
 *      header that hits the same endpoint.
 *   3. 409 idempotency response surfaces a warning toast referencing
 *      the existing job (the user's expected next step is to wait
 *      for the existing job, not retry).
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: {
        success: vi.fn(),
        error: vi.fn(),
        info: vi.fn(),
        warning: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));
// jobsStore.refreshAfterLocalChange is fired after a successful run —
// stub it so the test doesn't spin up the real polling loop.
const refreshSpy = vi.fn();
// The panel reads this project's comparison Jobs from the store (hook form)
// and calls getState().refreshAfterLocalChange after starting one.
const jobsState: { jobs: Array<Record<string, unknown>> } = { jobs: [] };
vi.mock('../../stores/jobsStore', () => ({
    useJobsStore: Object.assign(
        (selector: (state: { jobs: unknown[] }) => unknown) => selector(jobsState),
        { getState: () => ({ refreshAfterLocalChange: refreshSpy, jobs: jobsState.jobs }) },
    ),
}));

import AutoRagComparisonPanel from './AutoRagComparisonPanel';


const HAPPY_CACHED_PAYLOAD = {
    project_id: 4,
    recipe_id: 'qa-sft',
    cached_at: '2026-05-26T12:00:00Z',
    summary: {
        off_mean_f1: 0.10,
        on_mean_f1: 0.30,
        absolute_lift: 0.20,
        relative_lift_pct: 200.0,
        n_val_rows: 28,
        rag_k: 3,
        phase_9c_reference_lift_pct: 146.49,
    },
    rows: [
        {
            question: 'Q1?',
            reference: 'A1',
            without_rag: { generated: 'wrong', f1: 0.1 },
            with_rag: { generated: 'A1', f1: 1.0, retrieved_row_count: 3 },
        },
    ],
};


describe('AutoRagComparisonPanel — run-comparison button', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        toastMock.success.mockReset();
        toastMock.error.mockReset();
        toastMock.info.mockReset();
        toastMock.warning.mockReset();
        refreshSpy.mockReset();
    });

    it('renders the primary "Run comparison" CTA on 404 empty-state', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 404, data: { detail: 'No comparison cached' } },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                id: 33,
                kind: 'auto_rag_comparison',
                title: 'Auto-RAG comparison · project #4',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 4,
                user_id: null,
                params: {},
                result: null,
                error: null,
                queued_at: '2026-05-26T12:00:00Z',
                started_at: null,
                completed_at: null,
                dismissed_at: null,
            },
        });
        render(<AutoRagComparisonPanel projectId={4} />);
        const btn = await screen.findByTestId('auto-rag-comparison-run-btn');
        expect(btn).toHaveTextContent(/run comparison/i);
        // CLI fallback still available behind a disclosure.
        expect(screen.getByTestId('auto-rag-comparison-empty-cmd')).toBeInTheDocument();

        await userEvent.click(btn);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/4/auto-rag/comparison/run',
            );
        });
        // Info toast references the job id from the response.
        expect(toastMock.info).toHaveBeenCalledWith(
            expect.stringContaining('#33'),
            4000,
        );
        // Jobs store gets kicked so the bell shows the new job on
        // the next tick.
        expect(refreshSpy).toHaveBeenCalled();
    });

    it('renders a "Re-run comparison" affordance on the cached payload header', async () => {
        apiMock.get.mockResolvedValueOnce({ data: HAPPY_CACHED_PAYLOAD });
        apiMock.post.mockResolvedValueOnce({
            data: {
                id: 34,
                kind: 'auto_rag_comparison',
                title: 'Auto-RAG comparison · project #4',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 4,
                user_id: null,
                params: {},
                result: null,
                error: null,
                queued_at: '2026-05-26T12:00:00Z',
                started_at: null,
                completed_at: null,
                dismissed_at: null,
            },
        });
        render(<AutoRagComparisonPanel projectId={4} />);
        const btn = await screen.findByTestId('auto-rag-comparison-rerun-btn');
        expect(btn).toHaveTextContent(/re-run comparison/i);
        await userEvent.click(btn);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/4/auto-rag/comparison/run',
            );
        });
    });

    it('surfaces a warning toast on 409 (comparison already running)', async () => {
        apiMock.get.mockRejectedValueOnce({ response: { status: 404 } });
        apiMock.post.mockRejectedValueOnce({
            response: {
                status: 409,
                data: {
                    detail: {
                        error_code: 'AUTO_RAG_COMPARISON_ALREADY_RUNNING',
                        message:
                            'An auto-RAG comparison Job for project 4 is already in flight.',
                        metadata: {
                            existing_job_id: 7,
                        },
                    },
                },
            },
        });
        render(<AutoRagComparisonPanel projectId={4} />);
        const btn = await screen.findByTestId('auto-rag-comparison-run-btn');
        await userEvent.click(btn);
        await waitFor(() => {
            expect(toastMock.warning).toHaveBeenCalledWith(
                expect.stringContaining('already in flight'),
                4000,
            );
        });
        // No info toast / no refresh: this isn't a successful start.
        expect(toastMock.info).not.toHaveBeenCalled();
        expect(refreshSpy).not.toHaveBeenCalled();
    });

    it('renders the shared "pick a recipe first" CTA on RECIPE_REQUIRED 400', async () => {
        // Legacy NULL-recipe project: backend now returns a
        // structured 400 with error_code=RECIPE_REQUIRED. Panel must
        // mount the shared NoRecipeEmptyState (instead of silently
        // hiding like it does for "recipe set but not RAG-eligible").
        apiMock.get.mockRejectedValueOnce({
            response: {
                status: 400,
                data: {
                    detail: {
                        error_code: 'RECIPE_REQUIRED',
                        message:
                            'Project has no selected recipe — auto-RAG '
                            + 'comparison needs the recipe.',
                    },
                },
            },
        });
        render(<AutoRagComparisonPanel projectId={4} />);
        const cta = await screen.findByTestId(
            'auto-rag-comparison-recipe-required',
        );
        expect(cta.textContent).toMatch(/Choose a task type first/);
        const link = cta.querySelector('a') as HTMLAnchorElement;
        const href = link.getAttribute('href') || '';
        expect(href.startsWith('/project/4/recipe-picker?')).toBe(true);
        expect(href).toMatch(/return_to=/);
    });

    it('still silently hides on non-RECIPE_REQUIRED 400 (recipe ineligible)', async () => {
        // Recipe IS set but isn't RAG-eligible (e.g. classification).
        // Backend returns a plain-string detail; panel keeps the
        // legacy silent-hide behavior — the user has explicit signal
        // elsewhere (the project's selected recipe isn't QA).
        apiMock.get.mockRejectedValueOnce({
            response: {
                status: 400,
                data: { detail: "Recipe 'classification' has no auto-RAG corpus shape yet." },
            },
        });
        const { container } = render(<AutoRagComparisonPanel projectId={4} />);
        await waitFor(() => {
            // Loading state has cleared.
            expect(
                screen.queryByTestId('auto-rag-comparison-loading'),
            ).not.toBeInTheDocument();
        });
        // No CTA, no panel, no error banner.
        expect(
            screen.queryByTestId('auto-rag-comparison-recipe-required'),
        ).not.toBeInTheDocument();
        expect(
            screen.queryByTestId('auto-rag-comparison-error'),
        ).not.toBeInTheDocument();
        expect(container.firstChild).toBeNull();
    });
});


describe('AutoRagComparisonPanel — base model next to the fine-tuned run', () => {
    const BASE = {
        cached_at: '2026-10-01T12:00:00Z',
        experiment_id: null,
        base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        summary: {
            off_mean_f1: 0.1056,
            on_mean_f1: 0.1363,
            absolute_lift: 0.0307,
            relative_lift_pct: 29.1,
            n_val_rows: 21,
            rag_k: 3,
        },
        rows: [
            {
                question: 'Base-only question?',
                reference: 'ref',
                without_rag: { generated: 'no idea', f1: 0.0 },
                with_rag: { generated: 'grounded', f1: 0.5, retrieved_row_count: 3 },
            },
        ],
    };
    const FINE_TUNED = {
        ...HAPPY_CACHED_PAYLOAD,
        experiment_id: 25,
        base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        summary: { ...HAPPY_CACHED_PAYLOAD.summary, off_mean_f1: 0.179, on_mean_f1: 0.1462, relative_lift_pct: -18.3 },
    };

    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        toastMock.info.mockReset();
        refreshSpy.mockReset();
    });

    it('shows both models with their own numbers and provenance', async () => {
        apiMock.get.mockResolvedValueOnce({ status: 200, data: { ...FINE_TUNED, base: BASE } });
        render(<AutoRagComparisonPanel projectId={18} />);

        expect(await screen.findByTestId('auto-rag-comparison-off-f1')).toHaveTextContent('0.1790');
        expect(screen.getByTestId('auto-rag-comparison-lift')).toHaveTextContent('-18.3%');
        expect(screen.getByTestId('auto-rag-comparison-base-off-f1')).toHaveTextContent('0.1056');
        expect(screen.getByTestId('auto-rag-comparison-base-on-f1')).toHaveTextContent('0.1363');
        expect(screen.getByTestId('auto-rag-comparison-base-lift')).toHaveTextContent('+29.1%');
        // Provenance: which run and which base model each side measured.
        expect(screen.getByTestId('auto-rag-comparison-card')).toHaveTextContent('run #25');
        expect(screen.getByTestId('auto-rag-comparison-base-card')).toHaveTextContent('SmolLM2-135M-Instruct');
        // The verdict names the highest of the four numbers — here the
        // fine-tuned run WITHOUT retrieval, not a flattering one.
        expect(screen.getByTestId('auto-rag-comparison-verdict')).toHaveTextContent(
            /run #25 without retrieval \(F1 0\.179\)/,
        );
    });

    it('switches the per-row list between the two models', async () => {
        apiMock.get.mockResolvedValueOnce({ status: 200, data: { ...FINE_TUNED, base: BASE } });
        const user = userEvent.setup();
        render(<AutoRagComparisonPanel projectId={18} />);

        await screen.findByTestId('auto-rag-comparison-rows-base');
        expect(screen.queryByText('Base-only question?')).not.toBeInTheDocument();
        await user.click(screen.getByTestId('auto-rag-comparison-rows-base'));
        expect(screen.getByText('Base-only question?')).toBeInTheDocument();
        expect(screen.getByTestId('auto-rag-comparison-rows-caption')).toHaveTextContent(/base model/);
    });

    it('offers to run the base-model comparison when only the fine-tuned one exists', async () => {
        apiMock.get.mockResolvedValueOnce({ status: 200, data: { ...FINE_TUNED, base: null } });
        apiMock.post.mockResolvedValueOnce({ status: 202, data: { id: 77 } });
        const user = userEvent.setup();
        render(<AutoRagComparisonPanel projectId={18} />);

        await user.click(await screen.findByTestId('auto-rag-comparison-base-run-btn'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/18/auto-rag/comparison/run',
                null,
                { params: { model: 'base' } },
            );
        });
        expect(toastMock.info).toHaveBeenCalledWith(expect.stringMatching(/Base-model/), 4000);
        // No verdict while one side is missing.
        expect(screen.queryByTestId('auto-rag-comparison-verdict')).not.toBeInTheDocument();
    });

    it('renders a base-only comparison (no trained run yet)', async () => {
        apiMock.get.mockResolvedValueOnce({
            status: 200,
            data: { project_id: 18, recipe_id: 'qa-sft', cached_at: null, summary: null, rows: [], base_model: BASE.base_model, base: BASE },
        });
        render(<AutoRagComparisonPanel projectId={18} />);

        expect(await screen.findByTestId('auto-rag-comparison-base-lift')).toHaveTextContent('+29.1%');
        expect(screen.getByTestId('auto-rag-comparison-finetuned-run-btn')).toBeInTheDocument();
        expect(screen.getByText('Base-only question?')).toBeInTheDocument();
    });
});


describe('AutoRagComparisonPanel — live refresh + stale run', () => {
    const PAYLOAD = {
        ...HAPPY_CACHED_PAYLOAD,
        project_id: 18,
        experiment_id: 25,
        latest_experiment_id: 25,
        stale: false,
        base: null,
    };
    const job = (status: string, extra: Record<string, unknown> = {}) => ({
        id: 91,
        kind: 'auto_rag_comparison',
        project_id: 18,
        status,
        params: { model: 'fine_tuned' },
        progress_message: 'scoring row 3/21 (with-RAG)',
        completed_at: status === 'succeeded' ? '2026-10-01T18:00:00Z' : null,
        ...extra,
    });

    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        jobsState.jobs = [];
    });

    it('re-reads the comparison when its Job finishes, without a reload', async () => {
        apiMock.get
            .mockResolvedValueOnce({ status: 200, data: PAYLOAD })
            .mockResolvedValueOnce({
                status: 200,
                data: { ...PAYLOAD, summary: { ...PAYLOAD.summary, off_mean_f1: 0.4321 } },
            });
        jobsState.jobs = [job('running')];
        const { rerender } = render(<AutoRagComparisonPanel projectId={18} />);

        expect(await screen.findByTestId('auto-rag-comparison-off-f1')).toHaveTextContent('0.1000');
        // While the Job runs: progress on that side, and no second run allowed.
        expect(screen.getByTestId('auto-rag-comparison-running')).toHaveTextContent('scoring row 3/21');
        expect(screen.getByTestId('auto-rag-comparison-rerun-btn')).toBeDisabled();
        expect(screen.getByTestId('auto-rag-comparison-base-run-btn')).toBeDisabled();
        expect(apiMock.get).toHaveBeenCalledTimes(1);

        jobsState.jobs = [job('succeeded')];
        rerender(<AutoRagComparisonPanel projectId={18} />);

        await waitFor(() => {
            expect(screen.getByTestId('auto-rag-comparison-off-f1')).toHaveTextContent('0.4321');
        });
        expect(apiMock.get).toHaveBeenCalledTimes(2);
        expect(screen.queryByTestId('auto-rag-comparison-running')).not.toBeInTheDocument();
        expect(screen.getByTestId('auto-rag-comparison-rerun-btn')).not.toBeDisabled();
    });

    it('ignores comparison Jobs of other projects and other Job kinds', async () => {
        apiMock.get.mockResolvedValueOnce({ status: 200, data: PAYLOAD });
        jobsState.jobs = [job('running', { project_id: 4 }), job('running', { id: 92, kind: 'training_start' })];
        const { rerender } = render(<AutoRagComparisonPanel projectId={18} />);
        await screen.findByTestId('auto-rag-comparison-off-f1');
        expect(screen.queryByTestId('auto-rag-comparison-running')).not.toBeInTheDocument();

        jobsState.jobs = [job('succeeded', { project_id: 4 })];
        rerender(<AutoRagComparisonPanel projectId={18} />);
        await screen.findByTestId('auto-rag-comparison-off-f1');
        expect(apiMock.get).toHaveBeenCalledTimes(1);
    });

    it('fills the empty state in when the first comparison finishes', async () => {
        apiMock.get
            .mockRejectedValueOnce({ response: { status: 404, data: { detail: 'No auto-RAG comparison cached yet' } } })
            .mockResolvedValueOnce({ status: 200, data: PAYLOAD });
        jobsState.jobs = [job('running', { params: { model: 'base' } })];
        const { rerender } = render(<AutoRagComparisonPanel projectId={18} />);

        expect(await screen.findByTestId('auto-rag-comparison-running')).toHaveTextContent(/Base-model comparison running/);
        expect(screen.getByTestId('auto-rag-comparison-run-btn')).toBeDisabled();

        jobsState.jobs = [job('succeeded', { params: { model: 'base' } })];
        rerender(<AutoRagComparisonPanel projectId={18} />);
        expect(await screen.findByTestId('auto-rag-comparison-off-f1')).toBeInTheDocument();
    });

    it('flags a fine-tuned comparison measured on an older run', async () => {
        apiMock.get.mockResolvedValueOnce({
            status: 200,
            data: { ...PAYLOAD, latest_experiment_id: 26, stale: true },
        });
        render(<AutoRagComparisonPanel projectId={18} />);

        const note = await screen.findByTestId('auto-rag-comparison-stale');
        expect(note).toHaveTextContent('These numbers are for run #25');
        expect(note).toHaveTextContent('Your latest run is #26');
    });

    it('shows no stale flag when the comparison is for the latest run', async () => {
        apiMock.get.mockResolvedValueOnce({ status: 200, data: PAYLOAD });
        render(<AutoRagComparisonPanel projectId={18} />);
        await screen.findByTestId('auto-rag-comparison-off-f1');
        expect(screen.queryByTestId('auto-rag-comparison-stale')).not.toBeInTheDocument();
    });
});


describe('AutoRagComparisonPanel — row counts + "within noise"', () => {
    const summary = { off_mean_f1: 0.2165, on_mean_f1: 0.2645, absolute_lift: 0.048, relative_lift_pct: 22.2, n_val_rows: 21, rag_k: 3 };
    const base = {
        cached_at: '2026-10-01T12:00:00Z',
        experiment_id: null,
        base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        summary: { off_mean_f1: 0.1056, on_mean_f1: 0.1363, absolute_lift: 0.0307, relative_lift_pct: 29.1, n_val_rows: 21, rag_k: 3 },
        rows: [],
        evidence: { n: 21, better: 14, worse: 6, same: 1, mean_diff: 0.0307, ci_low: -0.0054, ci_high: 0.0668, verdict: 'within_noise' },
    };
    const payload = (evidence: Record<string, unknown>) => ({
        project_id: 18, recipe_id: 'qa-sft', cached_at: '2026-10-01T19:00:00Z', experiment_id: 26,
        latest_experiment_id: 26, stale: false, base_model: base.base_model,
        summary, rows: [], evidence, base,
    });

    beforeEach(() => {
        apiMock.get.mockReset();
        jobsState.jobs = [];
    });

    it('marks a positive lift whose interval includes zero as within noise', async () => {
        apiMock.get.mockResolvedValueOnce({
            status: 200,
            data: payload({ n: 21, better: 14, worse: 7, same: 0, mean_diff: 0.048, ci_low: 0.0033, ci_high: 0.0928, verdict: 'better' }),
        });
        render(<AutoRagComparisonPanel projectId={18} />);

        const baseEvidence = await screen.findByTestId('auto-rag-comparison-base-evidence');
        expect(baseEvidence).toHaveAttribute('data-verdict', 'within_noise');
        expect(baseEvidence).toHaveTextContent('Within noise');
        expect(screen.getByTestId('auto-rag-comparison-base-row-counts')).toHaveTextContent(
            'Retrieval helped 14 rows, hurt 6 rows, no change on 1.',
        );
        expect(baseEvidence).toHaveTextContent('95% range −0.005 to +0.067');
        expect(baseEvidence).toHaveTextContent(/could be chance/);
        // +29.1% is shown, but not painted as a win.
        const baseLift = screen.getByTestId('auto-rag-comparison-base-lift');
        expect(baseLift).toHaveTextContent('+29.1%');
        expect(baseLift.className).not.toMatch(/is-positive/);
    });

    it('a gain beyond row noise on a fine-tuned run still carries the one-run caveat', async () => {
        apiMock.get.mockResolvedValueOnce({
            status: 200,
            data: payload({ n: 21, better: 14, worse: 7, same: 0, mean_diff: 0.048, ci_low: 0.0033, ci_high: 0.0928, verdict: 'better' }),
        });
        render(<AutoRagComparisonPanel projectId={18} />);

        const evidence = await screen.findByTestId('auto-rag-comparison-evidence');
        expect(evidence).toHaveAttribute('data-verdict', 'better');
        expect(screen.getByTestId('auto-rag-comparison-row-counts')).toHaveTextContent('Retrieval helped 14 rows, hurt 7 rows.');
        expect(evidence).toHaveTextContent(/one training run: another seed can move it/);
        expect(screen.getByTestId('auto-rag-comparison-lift').className).toMatch(/is-positive/);
        // The base model has no training seed — no such caveat there.
        expect(screen.getByTestId('auto-rag-comparison-base-evidence')).not.toHaveTextContent(/training run/);
    });

    it('qualifies "highest of the four" when the winner\'s retrieval edge is within noise', async () => {
        apiMock.get.mockResolvedValueOnce({
            status: 200,
            data: payload({ n: 21, better: 11, worse: 9, same: 1, mean_diff: 0.048, ci_low: -0.02, ci_high: 0.116, verdict: 'within_noise' }),
        });
        render(<AutoRagComparisonPanel projectId={18} />);

        const verdict = await screen.findByTestId('auto-rag-comparison-verdict');
        expect(verdict).toHaveTextContent(/run #26 with retrieval \(F1 0\.265\)/);
        expect(verdict).toHaveTextContent(/edge over the same model without retrieval is within noise/);
        expect(screen.getByTestId('auto-rag-comparison-lift').className).not.toMatch(/is-positive/);
    });

    it('says when there are too few rows to tell', async () => {
        apiMock.get.mockResolvedValueOnce({
            status: 200,
            data: payload({ n: 2, better: 2, worse: 0, same: 0, mean_diff: 0.3, ci_low: null, ci_high: null, verdict: 'too_few_rows' }),
        });
        render(<AutoRagComparisonPanel projectId={18} />);

        const evidence = await screen.findByTestId('auto-rag-comparison-evidence');
        expect(evidence).toHaveTextContent('Too few rows to tell');
        expect(evidence).toHaveTextContent(/Only 2 scored rows/);
    });
});
