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

    it('shows a token-classification run on its own: answer key first, then test examples', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                kind: 'span_eval', project_id: 7, experiment_id: 59, verdict: 'evaluated', message: null, headline: null,
                trained: { experiment_id: 59, experiment_name: 'deberta tc' },
                answer_key: {
                    eval_result_id: 2, dataset_name: 'gold_test', precision: 0.83, recall: 0.85, f1: 0.84, rows: 44,
                    written_by: 'cloud:anthropic:claude-haiku-4-5',
                    weakest_types: [{ type: 'bank_account', f1: 0.21, support: 9 }, { type: 'person_name', f1: 0.65, support: 46 }],
                },
                test: { eval_result_id: 1, dataset_name: 'test', precision: 0.87, recall: 0.92, f1: 0.89, rows: 44, weakest_types: [] },
                dataset_name: 'gold_test', evaluated_samples: 44, failed_count: 1,
                failures: [{ prompt: 'Pay to account 12345678.', reference: '{"entities": []}', prediction: '{"entities": []}' }],
            },
        });
        apiMock.post.mockResolvedValue({ data: { started: true } });
        render(<EvalSummaryCard projectId={7} experimentId={59} />);
        expect(await screen.findByText('Span F1 0.84 against the answer key')).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-span-review-none')).toHaveTextContent('Nobody has checked the answer key yet');
        expect(screen.getByTestId('eval-summary-span-key')).toHaveTextContent(
            'Answer key (written by cloud:anthropic:claude-haiku-4-5): precision 0.83 · recall 0.85 · F1 0.84 on 44 texts',
        );
        expect(screen.getByTestId('eval-summary-span-test')).toHaveTextContent('F1 0.89 on 44 texts');
        expect(screen.getByTestId('eval-summary-span-weakest')).toHaveTextContent('bank_account 0.21 (9 spans), person_name 0.65 (46 spans)');
        expect(screen.getByTestId('redaction-try')).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-failures')).toHaveTextContent('1 of 44 answer-key texts');
        expect(screen.queryByTestId('eval-summary-check-seeds')).toBeNull();
        await userEvent.setup().click(screen.getByTestId('eval-summary-rerun-span'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/7/evaluation/summary/lift-check', { experiment_id: 59 }));
    });

    it('tunes per-type thresholds for a span run with recall floors and says what changed', async () => {
        const spanSummary = {
            kind: 'span_eval', project_id: 7, experiment_id: 59, verdict: 'evaluated', message: null, headline: null,
            trained: { experiment_name: 'tc' }, answer_key: null,
            test: { eval_result_id: 1, dataset_name: 'test', precision: 0.8, recall: 0.8, f1: 0.8, rows: 44, weakest_types: [] },
            dataset_name: 'test', evaluated_samples: 44, failed_count: 0, failures: [],
        };
        apiMock.get.mockImplementation((url: string) => Promise.resolve({
            data: url.endsWith('/span-recall-floors')
                ? { floors: { iban: 0.95 }, default: null, types: ['bank_account', 'iban'], suggestions: {} }
                : spanSummary,
        }));
        const row = (floor: number | null, status: string | null, rb: number, ra: number, pb: number, pa: number, th: number) => ({
            threshold: th, f1_before: 0, f1_after: 0, precision_before: pb, precision_after: pa,
            recall_before: rb, recall_after: ra, support: 10, predicted: 10, floor, floor_status: status,
        });
        apiMock.post.mockResolvedValue({
            data: {
                experiment_id: 59, split: 'val', rows: 136, micro_f1_before: 0.931, micro_f1_after: 0.937, raised: { bank_account: 0.7 },
                report: { bank_account: row(null, null, 0.9, 0.88, 0.8, 0.95, 0.7), iban: row(0.95, 'not_reachable', 0.9, 0.9, 0.97, 0.97, 0) },
                floors_met: [], floors_not_reachable: ['iban'], recall_bias: 0,
                recall_bias_suggestion: {
                    recall_bias: 0.2, grid: [0.05, 0.1, 0.2], micro_precision_cost: 0.04,
                    per_type: { iban: { floor: 0.95, bias: 0.2, recall_before: 0.9, precision_before: 0.97, recall_at_bias: 0.96, precision_at_bias: 0.9, precision_cost: 0.07 } },
                },
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={59} />);
        const user = userEvent.setup();
        await user.click(await screen.findByTestId('eval-summary-tune-thresholds'));
        expect(await screen.findByTestId('span-tuner-floor-iban')).toHaveValue(0.95);
        await user.click(screen.getByTestId('eval-summary-tune-run'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/7/span-thresholds/tune', {
            experiment_id: 59, floors: { iban: 0.95 }, apply_recall_bias: false,
        }));
        expect(await screen.findByTestId('eval-summary-tuned')).toHaveTextContent(
            'Thresholds tuned on 136 validation rows: F1 0.93 → 0.94 · raised bank_account ≥ 0.7 · recall floors met: 0, not reachable: 1',
        );
        expect(screen.getByTestId('span-tuner-row-iban')).toHaveTextContent('iban0.950.90 → 0.900.97 → 0.970not reachable at threshold 0');
        expect(screen.queryByTestId('span-tuner-row-bank_account')).toBeNull();
        expect(screen.getByTestId('span-tuner-bias-suggestion')).toHaveTextContent(
            'iban: recall bias 0.2 meets the 0.95 floor (recall 0.90 → 0.96, precision 0.97 → 0.90).',
        );
        expect(screen.getByTestId('span-tuner-bias-suggestion')).toHaveTextContent('Suggested recall bias: 0.2; overall precision −0.04');
    });

    it('shows an encoder classifier on its own, with external benchmark scores beside it', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                kind: 'head_eval', project_id: 32, experiment_id: 84, verdict: 'evaluated', headline: null,
                message: 'Scored on its own: an encoder base model can\'t answer zero-shot, so there is no base-model lift.',
                trained: { experiment_name: 'deberta sentiment' },
                head: { accuracy: 0.67, macro_f1: 0.63, rows: 100 },
                dataset_name: 'test', evaluated_samples: 100, failed_count: 0, failures: [],
                benchmarks: [{
                    id: 5, name: 'Yelp reviews', slug: 'yelp', task_profile: 'classification',
                    source: { locator: 'hf:Yelp/yelp_review_full:test', source_id: 'hf', dataset_id: 'Yelp/yelp_review_full', split: 'test', revision: null },
                    licence: 'other', local_only: false, rows: 241, mapping: {}, scored_labels: ['negative', 'positive'], created_at: 'x',
                    result: { experiment_id: 84, evaluated_at: 'x', rows: 241, headline: 'accuracy', score: 0.7386 }, latest_result: null,
                }],
            },
        });
        apiMock.post.mockResolvedValue({ data: { started: true } });
        render(<EvalSummaryCard projectId={32} experimentId={84} />);
        expect(await screen.findByText('Accuracy 0.67 on the test examples · macro-F1 0.63')).toBeInTheDocument();
        expect(screen.getByText(/classifier head on an encoder, scored on its own/)).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-benchmark-yelp')).toHaveTextContent(
            'Yelp reviews: accuracy 0.74 on 241 rows (Yelp/yelp_review_full:test · licence other)',
        );
        expect(screen.queryByTestId('eval-summary-check-seeds')).toBeNull();
        expect(screen.getByTestId('eval-summary-rerun-span')).toBeInTheDocument();
    });

    it('shows per-regime span scores with recall first when the project tags its types', async () => {
        const regime = (r: string, label: string, recall: number, fn: number, support: number, unscored: string[] = []) => ({
            regime: r, label, custom: false, types: ['a'], unscored_types: unscored, tp: support - fn, fp: 1, fn, support,
            precision: 0.9, recall, f1: 0.8,
        });
        apiMock.get.mockResolvedValue({
            data: {
                kind: 'span_eval', project_id: 7, experiment_id: 59, verdict: 'evaluated', message: null, headline: null,
                trained: { experiment_id: 59, experiment_name: 'deberta tc' },
                answer_key: null,
                test: {
                    eval_result_id: 1, dataset_name: 'test', precision: 0.87, recall: 0.92, f1: 0.89, rows: 44, weakest_types: [],
                    by_regime: [regime('hipaa', 'HIPAA', 0.71, 4, 14, ['diagnosis']), regime('pci_dss', 'PCI-DSS', 0.95, 1, 20)],
                },
                dataset_name: 'test', evaluated_samples: 44, failed_count: 0, failures: [],
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={59} />);
        const table = await screen.findByTestId('eval-summary-span-regimes');
        expect(table).toHaveTextContent('By regime (test examples)');
        expect(screen.getByTestId('eval-summary-regime-hipaa')).toHaveTextContent('HIPAA · 1 type not in this eval0.710.900.804 of 14');
        expect(screen.getByTestId('eval-summary-regime-pci_dss')).toHaveTextContent('PCI-DSS0.95');
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

    it('explains a noise-checked sweep: kept or switched, with each arm vs the current retrieval', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 22, experiment_id: null, kind: 'rag_passages', verdict: 'better', message: null,
                headline: { metric_id: 'judge_correct', baseline_value: 0.08, trained_value: 0.55, absolute_delta: 0.47, relative_delta_pct: 600, direction: 'improved' },
                evidence: null, baseline: { experiment_id: null, base_model: 'Qwen' }, trained: null, eval_type: 'rag_passages', evaluated_samples: 19, split: 'test',
                failures: [], failed_count: 0,
                retrieval: { k: 3, reranker: null },
                retrieval_sweep: [
                    { label: 'top-3', judge_score: 0.55, on_mean_f1: 0.4, chosen: true, current: true, vs_current: null },
                    { label: 'top-5', judge_score: 0.39, on_mean_f1: 0.3, chosen: false, current: false, vs_current: { verdict: 'worse', n: 19, better: 2, worse: 8, same: 9 } },
                    { label: 'top-3 + reranker ms-marco-MiniLM-L-6-v2', judge_score: 0.53, on_mean_f1: 0.4, chosen: false, current: false, vs_current: { verdict: 'within_noise', n: 19, better: 4, worse: 5, same: 10 } },
                ],
                retrieval_decision: { current: { k: 3, reranker: null }, chosen: { k: 3, reranker: null }, switched: false, reason: 'kept top-3: no arm beat it beyond row noise' },
            },
        });
        render(<EvalSummaryCard projectId={22} experimentId={null} />);
        await screen.findByTestId('eval-summary');
        expect(screen.getByTestId('eval-summary-retrieval')).toHaveTextContent('Retrieval served: top-3 (BM25 only) — kept top-3: no arm beat it beyond row noise.');
        expect(screen.getByTestId('eval-summary-sweep-arms')).toHaveTextContent('top-3 0.55 (current) · top-5 0.39 (loses to current: 2 better, 8 worse) · top-3 + reranker ms-marco-MiniLM-L-6-v2 0.53 (within noise: 4 better, 5 worse).');
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

    it('splits a passages run\'s wrong answers into retrieval missed vs the model misread', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 7, experiment_id: null, kind: 'rag_passages', verdict: 'better', message: null,
                headline: { metric_id: 'judge_correct', baseline_value: 0.04, trained_value: 0.53, absolute_delta: 0.49, relative_delta_pct: 1125, direction: 'improved' },
                evidence: { n: 46, better: 27, worse: 2, same: 17, mean_diff: 0.49, ci_low: 0.36, ci_high: 0.62, verdict: 'better', metric_id: 'judge_correct' },
                baseline: { experiment_id: null, base_model: 'Qwen/Qwen2.5-1.5B-Instruct' }, trained: null,
                eval_type: 'rag_passages', evaluated_samples: 46, split: 'test',
                retrieval: { k: 5, reranker: 'cross-encoder/ms-marco-MiniLM-L-6-v2' },
                judge: { judge: 'ollama:gemma4:12b', score: 0.53, judged: 46, unjudged: 0, correct: 20, partial: 9, wrong: 17, retrieval: { judged: 46, found: 29, missed: 17, found_rate: 0.63, wrong_missed: 9, wrong_misread: 8 } },
                failures: [
                    { prompt: 'Who retains the record?', reference: 'The head of the institution.', prediction: 'The Minister.', row_judge_verdict: 'wrong', row_judge_reason: 'Wrong actor.', row_retrieval_verdict: 'found', row_retrieval_passage: 2, row_retrieval_reason: 'Passage 2 is s. 9.' },
                    { prompt: 'What is the fee?', reference: 'Five dollars.', prediction: 'No fee.', row_judge_verdict: 'wrong', row_judge_reason: 'Contradicts.', row_retrieval_verdict: 'missed', row_retrieval_passage: null, row_retrieval_reason: 'Only s. 12 retrieved.' },
                ],
                failed_count: 17,
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={null} />);
        expect(await screen.findByTestId('eval-summary-retrieval-split')).toHaveTextContent(
            'Retrieval found the answer in 29 of 46 rows. Of the 17 wrong answers, retrieval missed it in 9 and found it in 8 — the model misread those. Retrieval is the bigger lever here.',
        );
        const rows = screen.getAllByTestId('eval-summary-failure-retrieval');
        expect(rows[0]).toHaveTextContent('Retrieval: retrieval found it (passage 2) — the model misread it — Passage 2 is s. 9.');
        expect(rows[1]).toHaveTextContent('Retrieval: retrieval missed it — Only s. 12 retrieved.');
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

describe('EvalPanel Verdict workflow (one layout, no Advanced toggle)', () => {
    beforeEach(() => {
        useProjectStore.setState({ activeProject: { id: 7, name: 'p', beginner_mode: true } as Project });
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/evaluation/summary')) return { data: BETTER };
            if (url.includes('/training/experiments')) return { data: [{ id: 21, name: 'exp-21' }] };
            if (url.includes('/evaluation/results/')) return { data: [] };
            // Every other panel self-hides on a failed fetch.
            throw new Error(`unmocked GET ${url}`);
        });
    });

    it('shows the summary card and selects the summarised run, without any disclosure toggle', async () => {
        render(<MemoryRouter><EvalPanel projectId={7} workflow="verdict" /></MemoryRouter>);
        expect(await screen.findByTestId('eval-summary')).toBeInTheDocument();
        await waitFor(() => expect(screen.getByRole('button', { name: 'exp-21' })).toHaveClass('btn-primary'));
        expect(screen.queryByTestId('eval-advanced-toggle')).toBeNull();
        expect(screen.queryByRole('tablist', { name: 'Evaluation sections' })).not.toBeInTheDocument();
    });
});

describe('EvalSummaryCard — answer-key review', () => {
    it('scores the checked rows apart from the unchecked ones and flags a stale eval', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                kind: 'span_eval', project_id: 7, experiment_id: 59, verdict: 'evaluated', message: null, headline: null,
                trained: { experiment_name: 'tc' },
                answer_key: {
                    eval_result_id: 2, dataset_name: 'gold_test', precision: 0.83, recall: 0.85, f1: 0.84, rows: 44, weakest_types: [],
                    written_by: 'cloud:anthropic:claude-sonnet-5-5',
                    review: {
                        total: 44, reviewed: 12, accepted: 8, corrected: 3, rejected: 1, unreviewed: 32,
                        reviewed_scores: { precision: 0.9, recall: 0.92, f1: 0.91, rows: 11 },
                        unreviewed_scores: { precision: 0.8, recall: 0.82, f1: 0.81, rows: 32 },
                        stale: true, changed_since_eval: 2,
                    },
                },
                test: null, dataset_name: 'gold_test', evaluated_samples: 44, failed_count: 0, failures: [],
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={59} />);
        const line = await screen.findByTestId('eval-summary-span-review');
        expect(line).toHaveTextContent('12 of 44 rows checked by a person (3 corrected, 1 rejected) · F1 0.91 on the 11 checked rows, 0.81 on the 32 unchecked · 2 rows changed since this eval — re-evaluate to score the corrected key.');
        expect(screen.queryByTestId('eval-summary-span-review-none')).not.toBeInTheDocument();
    });
});

describe('EvalSummaryCard — answer-key sizing', () => {
    it('says which types can show a 0.05 difference, with a per-type table and a refine link', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                kind: 'span_eval', project_id: 7, experiment_id: 59, verdict: 'evaluated', message: null, headline: null,
                trained: { experiment_name: 'tc' },
                answer_key: {
                    eval_result_id: 2, dataset_name: 'gold_test', precision: 0.83, recall: 0.85, f1: 0.84, rows: 44, weakest_types: [],
                    sizing: {
                        gap: 0.05, short: ['api_key', 'medication'], enough: ['email'], shortfall: 1202,
                        types: [
                            { type: 'api_key', support: 2, f1: 0.6667, needed: 712, shortfall: 710, detectable_gap: 0.9428, enough: false },
                            { type: 'medication', support: 20, f1: 0.8, needed: 512, shortfall: 492, detectable_gap: 0.253, enough: false },
                            { type: 'email', support: 600, f1: 1.0, needed: 152, shortfall: 0, detectable_gap: 0.0252, enough: true },
                        ],
                    },
                },
                test: null, dataset_name: 'gold_test', evaluated_samples: 44, failed_count: 0, failures: [],
            },
        });
        render(<EvalSummaryCard projectId={7} experimentId={59} />);
        const block = await screen.findByTestId('eval-summary-span-sizing');
        expect(block).toHaveTextContent('Can the answer key tell runs apart? 1 of 3 types have enough reference spans to show a 0.05 F1 difference; the other 2 can only show 0.94 or more — 1202 more spans short in all.');
        expect(screen.getByTestId('eval-summary-sizing-refine')).toHaveAttribute('href', '/project/7/data?step=synthetic');
        expect(screen.queryByTestId('eval-summary-sizing-table')).not.toBeInTheDocument();
        await userEvent.click(screen.getByTestId('eval-summary-sizing-toggle'));
        expect(screen.getByTestId('eval-summary-sizing-api_key')).toHaveTextContent('api_key0.672712 (710 short)≥ 0.94');
        expect(screen.getByTestId('eval-summary-sizing-email')).toHaveTextContent('email1.00600152≥ 0.03');
    });
});
