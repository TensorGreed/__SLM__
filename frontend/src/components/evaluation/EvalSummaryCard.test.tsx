import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalSummaryCard from './EvalSummaryCard';
import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';
import type { Project } from '../../types';

const BETTER = {
    project_id: 7,
    experiment_id: 21,
    verdict: 'better',
    message: null,
    headline: {
        metric_id: 'exact_match',
        baseline_value: 0.2,
        trained_value: 0.55,
        absolute_delta: 0.35,
        relative_delta_pct: 175,
        direction: 'improved',
    },
    baseline: { experiment_id: 20, base_model: 'SmolLM2-135M' },
    trained: { experiment_id: 21, experiment_name: 'exp-21' },
    eval_type: 'exact_match',
    evaluated_samples: 20,
    failures: [
        { prompt: 'How do I reset my password?', reference: 'Settings > Reset', prediction: 'Call support' },
    ],
    failed_count: 9,
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

afterEach(() => {
    useProjectStore.setState({ activeProject: null });
});

describe('EvalSummaryCard', () => {
    it('answers better-than-base, by how much, and shows failures', async () => {
        apiMock.get.mockResolvedValue({ data: BETTER });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);

        const card = await screen.findByTestId('eval-summary');
        expect(card).toHaveAttribute('data-verdict', 'better');
        expect(screen.getByText('Better than the base model')).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-headline')).toHaveTextContent('0.200 (base)');
        expect(screen.getByTestId('eval-summary-headline')).toHaveTextContent('+0.350 (+175%)');
        expect(screen.getByText(/Run #21 · exp-21 vs base SmolLM2-135M/)).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-failures')).toHaveTextContent('9 of 20 test examples');
        expect(screen.getByText('Call support')).toBeInTheDocument();
        expect(apiMock.get).toHaveBeenCalledWith('/projects/7/evaluation/summary', { params: { experiment_id: 21 } });
    });

    it('offers to re-run the lift check on an evaluated run', async () => {
        apiMock.get.mockResolvedValue({ data: BETTER });
        apiMock.post.mockResolvedValue({ data: { started: true, job_id: 5 } });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);
        await userEvent.setup().click(await screen.findByTestId('eval-summary-rerun'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/7/evaluation/summary/lift-check', { experiment_id: 21 }));
    });

    it('summarises a RAG-first project from its judged passages check', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 21, experiment_id: null, kind: 'rag_passages', verdict: 'better', message: null,
                headline: { metric_id: 'judge_correct', baseline_value: 0.03, trained_value: 0.62, absolute_delta: 0.59, relative_delta_pct: 1966.7, direction: 'improved' },
                evidence: { n: 19, better: 12, worse: 0, same: 7, mean_diff: 0.59, ci_low: 0.36, ci_high: 0.82, verdict: 'better', metric_id: 'judge_correct' },
                baseline: { experiment_id: null, base_model: 'Qwen/Qwen2.5-1.5B-Instruct' }, trained: null,
                eval_type: 'rag_passages', evaluated_samples: 19, split: 'test',
                judge: { judge: 'ollama:gemma4:12b', score: 0.62, judged: 19, unjudged: 0, correct: 10, partial: 3, wrong: 6 },
                failures: [{ prompt: 'Q?', reference: 'R', prediction: 'P', row_judge_verdict: 'wrong', row_judge_reason: 'wrong passage' }],
                failed_count: 9,
                retrieval: { k: 5, reranker: 'cross-encoder/ms-marco-MiniLM-L-6-v2' },
                retrieval_sweep: [
                    { label: 'top-3', judge_score: 0.61, on_mean_f1: 0.4, chosen: false },
                    { label: 'top-5', judge_score: 0.58, on_mean_f1: 0.41, chosen: false },
                    { label: 'top-3 + reranker ms-marco-MiniLM-L-6-v2', judge_score: 0.66, on_mean_f1: 0.42, chosen: false },
                    { label: 'top-5 + reranker ms-marco-MiniLM-L-6-v2', judge_score: 0.71, on_mean_f1: 0.44, chosen: true },
                ],
            },
        });
        apiMock.post.mockResolvedValue({ data: { id: 70 } });
        render(<EvalSummaryCard projectId={21} experimentId={null} />);
        const card = await screen.findByTestId('eval-summary');
        expect(card).toHaveAttribute('data-verdict', 'better');
        expect(screen.getByTestId('eval-summary-passages-run')).toHaveTextContent('Base model + your document passages vs the base model alone (Qwen/Qwen2.5-1.5B-Instruct) · 19 test examples');
        expect(screen.getByTestId('eval-summary-headline')).toHaveTextContent('0.030 (base) → 0.620 (base + your passages)');
        expect(screen.getByTestId('eval-summary-row-counts')).toHaveTextContent('Retrieval helped 12 rows, hurt 0 rows, no change on 7.');
        expect(screen.getByTestId('eval-summary-retrieval')).toHaveTextContent('Retrieval served: top-5 + reranker ms-marco-MiniLM-L-6-v2 — chosen by the judge over top-3 (0.61), top-5 (0.58), top-3 + reranker ms-marco-MiniLM-L-6-v2 (0.66).');
        expect(screen.queryByTestId('eval-summary-check-seeds')).not.toBeInTheDocument();
        await userEvent.setup().click(screen.getByTestId('eval-summary-rerun-passages'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/21/auto-rag/comparison/run', null, { params: { model: 'base', corpus: 'documents', split: 'test' } }));
    });

    it('names the LLM-judge headline and shows each failure\'s verdict', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                ...BETTER,
                headline: { ...BETTER.headline, metric_id: 'judge_correct', baseline_value: 0.1, trained_value: 0.45, absolute_delta: 0.35 },
                judge: { judge: 'ollama:gemma4:12b', score: 0.45, judged: 19, unjudged: 1, correct: 6, partial: 5, wrong: 8 },
                failures: [
                    { prompt: 'What does section 8 allow?', reference: 'Disclosure with consent…', prediction: 'Anything the head decides.', row_judge_verdict: 'wrong', row_judge_reason: 'Contradicts the consent requirement.' },
                ],
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);
        await screen.findByTestId('eval-summary');
        expect(screen.getByTestId('eval-summary-headline')).toHaveTextContent('judge: answer correct: 0.100 (base) → 0.450');
        expect(screen.getByTestId('eval-summary-judge')).toHaveTextContent('Judged by ollama:gemma4:12b: 6 correct, 5 partial, 8 wrong of 19 answers (1 unjudged)');
        expect(screen.getByTestId('eval-summary-failures')).toHaveTextContent('Judge: wrong — Contradicts the consent requirement.');
    });

    it('does not announce "better" when the lift is within noise', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                ...BETTER,
                evidence: {
                    n: 20, better: 9, worse: 4, same: 7, mean_diff: 0.25, ci_low: -0.04, ci_high: 0.54,
                    verdict: 'within_noise', metric_id: 'exact_match',
                },
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);

        const card = await screen.findByTestId('eval-summary');
        expect(screen.getByText('Ahead of the base model — but within noise')).toBeInTheDocument();
        expect(screen.queryByText('Better than the base model')).not.toBeInTheDocument();
        expect(card.className).toMatch(/eval-summary--neutral/);
        const evidence = screen.getByTestId('eval-summary-evidence');
        expect(evidence).toHaveAttribute('data-verdict', 'within_noise');
        expect(screen.getByTestId('eval-summary-row-counts')).toHaveTextContent(
            'Fine-tuning helped 9 rows, hurt 4 rows, no change on 7.',
        );
        expect(evidence).toHaveTextContent('average change +0.250 exact match per row, 95% range −0.040 to +0.540');
        expect(evidence).toHaveTextContent(/could be chance/);
        // The headline numbers are still shown as measured.
        expect(screen.getByTestId('eval-summary-headline')).toHaveTextContent('+0.350 (+175%)');
    });

    it('keeps "better" for a gain beyond row noise, with the one-run caveat', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                ...BETTER,
                evidence: {
                    n: 20, better: 9, worse: 2, same: 9, mean_diff: 0.35, ci_low: 0.09, ci_high: 0.61,
                    verdict: 'better', metric_id: 'exact_match',
                },
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);

        await screen.findByTestId('eval-summary');
        expect(screen.getByText('Better than the base model')).toBeInTheDocument();
        const evidence = screen.getByTestId('eval-summary-evidence');
        expect(evidence).toHaveTextContent('Gain on these rows');
        expect(evidence).toHaveTextContent(/one training run: another seed can move it/);
    });

    it('says when there are too few rows, and shows nothing extra for older results', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...BETTER,
                evidence: { n: 2, better: 2, worse: 0, same: 0, mean_diff: 0.5, ci_low: null, ci_high: null, verdict: 'too_few_rows', metric_id: 'exact_match' },
            },
        });
        const { unmount } = render(<EvalSummaryCard projectId={7} experimentId={21} />);
        expect(await screen.findByText('Better than the base model — too few rows to be sure')).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-evidence')).toHaveTextContent(/Only 2 scored rows/);
        unmount();

        apiMock.get.mockResolvedValueOnce({ data: { ...BETTER, evidence: null } });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);
        expect(await screen.findByText('Better than the base model')).toBeInTheDocument();
        expect(screen.queryByTestId('eval-summary-evidence')).not.toBeInTheDocument();
    });

    it('summarises a multi-seed run: mean ± std, per-seed runs, and whether the seeds agree', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                ...BETTER,
                experiment_id: 40,
                trained: { experiment_id: 40, experiment_name: 'exp-21 · 3 seeds' },
                n_seeds: 3,
                representative_experiment_id: 43,
                headline: { ...BETTER.headline, baseline_value: 0.074, trained_value: 0.187, trained_std: 0.015, absolute_delta: 0.113, relative_delta_pct: 153, n_seeds: 3 },
                seeds: [
                    { experiment_id: 41, seed_value: 1, headline: { ...BETTER.headline, trained_value: 0.172 } },
                    { experiment_id: 42, seed_value: 2, headline: { ...BETTER.headline, trained_value: 0.201 } },
                    { experiment_id: 43, seed_value: 3, headline: { ...BETTER.headline, trained_value: 0.189 } },
                ],
                seed_evidence: {
                    kind: 'seeds', n: 3, verdict: 'better', baseline_value: 0.074, values: [0.172, 0.201, 0.189],
                    mean: 0.1873, std: 0.0146, min: 0.172, max: 0.201, ci_low: 0.151, ci_high: 0.224,
                    all_better: true, all_worse: false, metric_id: 'exact_match',
                },
                evidence: { n: 20, better: 9, worse: 2, same: 9, mean_diff: 0.35, ci_low: 0.09, ci_high: 0.61, verdict: 'better', metric_id: 'exact_match' },
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={40} />);

        await screen.findByTestId('eval-summary');
        expect(screen.getByText('Better than the base model — across 3 seeds')).toBeInTheDocument();
        expect(screen.getByText(/Run #40 · exp-21 · 3 seeds vs base/)).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-headline')).toHaveTextContent('0.074 (base) → 0.187 ± 0.015 (fine-tuned, mean of 3 seeds)');
        const seedNote = screen.getByTestId('eval-summary-seed-evidence');
        expect(seedNote).toHaveAttribute('data-verdict', 'better');
        expect(seedNote).toHaveTextContent('Holds across seeds');
        expect(seedNote).toHaveTextContent('exact match 0.172, 0.201, 0.189 (mean 0.187 ± 0.015, base 0.074) — every seed beat the base model.');
        expect(screen.getByTestId('eval-summary-seed-runs')).toHaveTextContent('#41 (seed 1) 0.172, #42 (seed 2) 0.201, #43 (seed 3) 0.189');
        expect(screen.getByTestId('eval-summary-seed-runs')).toHaveTextContent('from run #43 (the median seed)');
        // The row-level note stays, without the one-run caveat (seeds cover that).
        expect(screen.getByTestId('eval-summary-evidence')).not.toHaveTextContent(/one training run/);
        // A seed group has no "check across seeds" button.
        expect(screen.queryByTestId('eval-summary-check-seeds')).not.toBeInTheDocument();
    });

    it('does not announce better when the seeds disagree', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                ...BETTER,
                n_seeds: 3,
                seeds: [],
                seed_evidence: {
                    kind: 'seeds', n: 3, verdict: 'within_noise', baseline_value: 0.2165, values: [0.2645, 0.2257, 0.2105],
                    mean: 0.2336, std: 0.0278, min: 0.2105, max: 0.2645, ci_low: -0.052, ci_high: 0.086,
                    all_better: false, all_worse: false, metric_id: 'f1',
                },
                evidence: null,
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);

        const card = await screen.findByTestId('eval-summary');
        expect(screen.getByText('Ahead of the base model on average — but the seeds disagree')).toBeInTheDocument();
        expect(card.className).toMatch(/eval-summary--neutral/);
        const note = screen.getByTestId('eval-summary-seed-evidence');
        expect(note).toHaveTextContent('Seeds disagree');
        expect(note).toHaveTextContent('F1 0.265, 0.226, 0.210');
        expect(note).toHaveTextContent(/another seed could land on either side/);
    });

    it('offers to check a single evaluated run across 3 seeds', async () => {
        apiMock.get.mockResolvedValue({ data: BETTER });
        apiMock.post.mockResolvedValue({ data: { status: 'training_started', experiment_id: 50, experiment_name: 'exp-21 · 3 seeds', source_experiment_id: 21, num_seeds: 3 } });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);

        const btn = await screen.findByTestId('eval-summary-check-seeds');
        expect(btn).toHaveTextContent('Check across 3 seeds');
        await userEvent.setup().click(btn);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/evaluation/summary/check-seeds',
                { experiment_id: 21, num_seeds: 3 },
            );
        });
    });

    it('reports the server-resolved run and offers one-click eval when not evaluated', async () => {
        apiMock.get.mockResolvedValue({
            data: { project_id: 7, experiment_id: 33, verdict: 'not_evaluated', headline: null, failures: [] },
        });
        apiMock.post.mockResolvedValue({ data: { id: 1, status: 'queued' } });
        const onResolved = vi.fn();
        render(<EvalSummaryCard projectId={7} experimentId={null} onResolved={onResolved} />);

        const btn = await screen.findByTestId('eval-summary-run');
        expect(onResolved).toHaveBeenCalledWith(33);
        await userEvent.setup().click(btn);
        await waitFor(() =>
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/evaluation/summary/lift-check',
                { experiment_id: 33 },
            ),
        );
    });
});

describe('EvalPanel advanced disclosure', () => {
    function mockPanelApi() {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/evaluation/summary')) return { data: BETTER };
            if (url.includes('/training/experiments')) return { data: [{ id: 21, name: 'exp-21' }] };
            if (url.includes('/evaluation/results/')) return { data: [] };
            throw new Error(`unmocked GET ${url}`);
        });
    }

    it('beginner projects see the summary card with Advanced collapsed', async () => {
        useProjectStore.setState({ activeProject: { id: 7, name: 'p', beginner_mode: true } as Project });
        mockPanelApi();
        render(<MemoryRouter><EvalPanel projectId={7} /></MemoryRouter>);

        expect(await screen.findByTestId('eval-summary')).toBeInTheDocument();
        // The server-resolved latest run is auto-selected for the user.
        await waitFor(() => expect(screen.getByRole('button', { name: 'exp-21' })).toHaveClass('btn-primary'));
        const toggle = screen.getByTestId('eval-advanced-toggle');
        expect(toggle).toHaveTextContent('Advanced evaluation');
        expect(screen.queryByRole('tablist', { name: 'Evaluation sections' })).not.toBeInTheDocument();

        await userEvent.setup().click(toggle);
        expect(screen.getByRole('tablist', { name: 'Evaluation sections' })).toBeInTheDocument();
        expect(toggle).toHaveTextContent('Hide advanced evaluation');
    });

    it('non-beginner projects open with Advanced expanded', async () => {
        useProjectStore.setState({ activeProject: { id: 7, name: 'p', beginner_mode: false } as Project });
        mockPanelApi();
        render(<MemoryRouter><EvalPanel projectId={7} /></MemoryRouter>);

        expect(await screen.findByTestId('eval-summary')).toBeInTheDocument();
        expect(screen.getByRole('tablist', { name: 'Evaluation sections' })).toBeInTheDocument();
    });
});
