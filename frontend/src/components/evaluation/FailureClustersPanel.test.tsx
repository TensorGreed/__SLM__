import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import FailureClustersPanel from './FailureClustersPanel';

const EVAL_RESULTS = [
    { id: 501, dataset_name: 'gold_test', eval_type: 'llm_judge', pass_rate: 0.62 },
    { id: 502, dataset_name: 'gold_dev', eval_type: 'f1', pass_rate: 0.41 },
];

const CLUSTER_RESPONSE = {
    eval_result_id: 501,
    experiment_id: 21,
    dataset_name: 'gold_test',
    eval_type: 'llm_judge',
    total_failures_analyzed: 9,
    reason_code_totals: { hallucination: 5, coverage_gap: 3, safety_failure: 1 },
    dominant_reason_code: 'hallucination',
    clusters: [
        {
            cluster_id: 'cluster-1',
            reason_code: 'hallucination',
            output_pattern: 'len-medium:lead-prose:digits-y',
            failure_count: 5,
            share_of_total: 0.5555,
            classifier_confidence: 0.78,
            classifier_reason: 'Low reference overlap with verbose output suggests fabricated details.',
            exemplars: [
                {
                    prompt: 'Who was Prime Minister in 1980?',
                    reference: 'Pierre Trudeau',
                    prediction: 'It was Brian Mulroney in 1980.',
                    judge_score: 1,
                    judge_rationale: 'factual error',
                },
            ],
        },
        {
            cluster_id: 'cluster-2',
            reason_code: 'coverage_gap',
            output_pattern: 'len-short:lead-refusal:digits-n',
            failure_count: 3,
            share_of_total: 0.3333,
            classifier_confidence: 0.86,
            classifier_reason: 'Model response suggests missing domain coverage for this slice.',
            exemplars: [],
        },
    ],
    remediation_plans: [
        {
            plan_id: 'plan-abc',
            artifact_id: 9001,
            created_at: '2026-04-24T12:00:00Z',
            root_causes: ['hallucination'],
            summary: {
                total_failures_analyzed: 9,
                cluster_count: 2,
                recommendation_count: 4,
                dominant_root_cause: 'hallucination',
            },
        },
    ],
};

describe('FailureClustersPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('fetches clusters for the first eval result and renders counts + remediation plan', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CLUSTER_RESPONSE });

        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);

        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/1/evaluation/501/failure-clusters',
            );
        });

        // Summary shows total + cluster count — the text is split across
        // multiple <strong> nodes so match on the container text.
        await waitFor(() => {
            const summaryText = document.querySelector('.failure-clusters-summary')?.textContent ?? '';
            expect(summaryText).toMatch(/9 failures · 2 clusters/);
            expect(summaryText).toMatch(/Dominant:\s*hallucination/);
        });

        // Both cluster rows present — each reason_code shows up in the
        // summary chip AND the cluster head chip, so assert on count ≥ 1.
        expect(screen.getAllByText('hallucination').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('coverage_gap').length).toBeGreaterThanOrEqual(1);

        // Output-pattern chips visible.
        expect(screen.getByText('len-medium:lead-prose:digits-y')).toBeInTheDocument();
        expect(screen.getByText('len-short:lead-refusal:digits-n')).toBeInTheDocument();

        // Remediation plan surfaces.
        expect(screen.getByText('plan-abc')).toBeInTheDocument();
    });

    it('expands a cluster on click and reveals its exemplar prompt/prediction', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CLUSTER_RESPONSE });

        const user = userEvent.setup();
        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        const headButtons = await screen.findAllByRole('button', { expanded: false });
        const targetHead = headButtons.find((btn) =>
            btn.textContent?.includes('hallucination'),
        );
        expect(targetHead).toBeDefined();
        await user.click(targetHead!);

        expect(
            screen.getByText(/low reference overlap with verbose output/i),
        ).toBeInTheDocument();
        expect(
            screen.getByText('Who was Prime Minister in 1980?'),
        ).toBeInTheDocument();
        expect(
            screen.getByText('It was Brian Mulroney in 1980.'),
        ).toBeInTheDocument();
    });

    it('refetches when the user switches eval result in the dropdown', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/501/failure-clusters')) {
                return { data: CLUSTER_RESPONSE };
            }
            if (url.endsWith('/502/failure-clusters')) {
                return {
                    data: {
                        ...CLUSTER_RESPONSE,
                        eval_result_id: 502,
                        total_failures_analyzed: 2,
                        clusters: [],
                        remediation_plans: [],
                        reason_code_totals: {},
                        dominant_reason_code: null,
                    },
                };
            }
            return { data: {} };
        });

        const user = userEvent.setup();
        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);
        await waitFor(() =>
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/1/evaluation/501/failure-clusters',
            ),
        );

        await user.selectOptions(
            screen.getByRole('combobox', { name: /eval result to cluster/i }),
            '502',
        );

        await waitFor(() =>
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/1/evaluation/502/failure-clusters',
            ),
        );

        // Empty state for the 0-cluster second result.
        expect(
            await screen.findByText(/no failures to cluster/i),
        ).toBeInTheDocument();
    });

    it('shows a fallback when the experiment has no eval results yet', () => {
        render(<FailureClustersPanel projectId={1} evalResults={[]} />);
        expect(apiMock.get).not.toHaveBeenCalled();
        expect(
            screen.getByText(/run at least one evaluation to cluster its failures/i),
        ).toBeInTheDocument();
    });

    it('shows an error card when the server returns a 400', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { data: { detail: 'eval_result_not_found' } },
        });
        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);
        expect(
            await screen.findByText(/eval_result_not_found/i),
        ).toBeInTheDocument();
    });
});
