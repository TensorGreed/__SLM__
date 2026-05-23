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
        apiMock.post.mockReset();
        apiMock.patch.mockReset();
        apiMock.put.mockReset();
        apiMock.delete.mockReset();
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
        // Theme 8 Epic 3: expand also fires the explain POST.
        apiMock.post.mockResolvedValueOnce({
            data: {
                cluster_id: 'cluster-1',
                explanation: '',
                status: 'judge_unavailable',
                cached: false,
                generated_at: null,
                model: null,
                exemplar_count: 0,
                note: 'No judge configured.',
            },
        });

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

    // ── Theme 8 Epic 3 — per-cluster failure explanations ────────

    it('fires explain POST on first cluster expand + renders the explanation', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CLUSTER_RESPONSE });
        apiMock.post.mockResolvedValueOnce({
            data: {
                cluster_id: 'cluster-1',
                explanation: 'Model is dropping the negation in "not urgent".',
                status: 'ok',
                cached: false,
                generated_at: '2026-05-21T12:00:00Z',
                model: 'gpt-4o-mini',
                exemplar_count: 3,
            },
        });

        const user = userEvent.setup();
        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        const headButtons = await screen.findAllByRole('button', { expanded: false });
        const targetHead = headButtons.find((btn) =>
            btn.textContent?.includes('hallucination'),
        );
        await user.click(targetHead!);

        // The mocked POST resolves on the next microtask, so we
        // skip asserting the transient loading state (race-prone)
        // and assert the final POST contract + OK render directly.
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/evaluation/501/clusters/cluster-1/explain',
                null,
                expect.any(Object),
            );
        });

        const ok = await screen.findByTestId('cluster-explanation-ok');
        expect(ok).toHaveTextContent(/Model is dropping the negation/);
        expect(ok).toHaveTextContent(/gpt-4o-mini/);
    });

    it('does not re-fire the explain POST when re-expanding the same cluster', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CLUSTER_RESPONSE });
        apiMock.post.mockResolvedValueOnce({
            data: {
                cluster_id: 'cluster-1',
                explanation: 'First explanation text.',
                status: 'ok',
                cached: false,
                generated_at: null,
                model: 'gpt-4o-mini',
                exemplar_count: 3,
            },
        });

        const user = userEvent.setup();
        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        const findHead = async () => {
            const buttons = await screen.findAllByRole('button');
            return buttons.find((btn) => btn.textContent?.includes('hallucination'))!;
        };
        await user.click(await findHead());
        await waitFor(() =>
            expect(screen.getByTestId('cluster-explanation-ok')).toBeInTheDocument(),
        );

        // Collapse + re-expand → no second POST should fire.
        await user.click(await findHead());
        await user.click(await findHead());
        await screen.findByTestId('cluster-explanation-ok');
        expect(apiMock.post).toHaveBeenCalledTimes(1);
    });

    it('shows the soft-fallback chip when the backend reports judge_unavailable', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CLUSTER_RESPONSE });
        apiMock.post.mockResolvedValueOnce({
            data: {
                cluster_id: 'cluster-1',
                explanation: '',
                status: 'judge_unavailable',
                cached: false,
                generated_at: null,
                model: null,
                exemplar_count: 5,
                note: 'No judge model is configured.',
            },
        });

        const user = userEvent.setup();
        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        const headButtons = await screen.findAllByRole('button', { expanded: false });
        const targetHead = headButtons.find((btn) =>
            btn.textContent?.includes('hallucination'),
        );
        await user.click(targetHead!);

        const soft = await screen.findByTestId('cluster-explanation-judge-unavailable');
        expect(soft).toHaveTextContent(/No judge model/);
        // Exemplars + classifier reason still render — explanation is
        // additive, not a replacement.
        expect(
            screen.getByText('Who was Prime Minister in 1980?'),
        ).toBeInTheDocument();
    });

    it('renders the cached badge when the backend reports cached=true', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CLUSTER_RESPONSE });
        apiMock.post.mockResolvedValueOnce({
            data: {
                cluster_id: 'cluster-1',
                explanation: 'Cached explanation.',
                status: 'ok',
                cached: true,
                generated_at: '2026-05-21T12:00:00Z',
                model: 'gpt-4o-mini',
                exemplar_count: 3,
            },
        });

        const user = userEvent.setup();
        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        const headButtons = await screen.findAllByRole('button', { expanded: false });
        const targetHead = headButtons.find((btn) =>
            btn.textContent?.includes('hallucination'),
        );
        await user.click(targetHead!);

        await screen.findByTestId('cluster-explanation-cached');
    });

    it('shows error + retry button when explain POST fails, retry re-fires', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CLUSTER_RESPONSE });
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'judge endpoint exploded' } },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                cluster_id: 'cluster-1',
                explanation: 'Retry succeeded.',
                status: 'ok',
                cached: false,
                generated_at: null,
                model: 'gpt-4o-mini',
                exemplar_count: 3,
            },
        });

        const user = userEvent.setup();
        render(<FailureClustersPanel projectId={1} evalResults={EVAL_RESULTS} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        const headButtons = await screen.findAllByRole('button', { expanded: false });
        const targetHead = headButtons.find((btn) =>
            btn.textContent?.includes('hallucination'),
        );
        await user.click(targetHead!);

        const err = await screen.findByTestId('cluster-explanation-error');
        expect(err).toHaveTextContent(/judge endpoint exploded/);

        await user.click(screen.getByTestId('cluster-explanation-retry'));
        const ok = await screen.findByTestId('cluster-explanation-ok');
        expect(ok).toHaveTextContent('Retry succeeded.');
        expect(apiMock.post).toHaveBeenCalledTimes(2);
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

    // ── USER-SUCCESS Epic 2b: cluster-targeted augment button ─────────

    it('renders the per-cluster augment control inside an expanded cluster card', async () => {
        apiMock.get.mockResolvedValue({ data: CLUSTER_RESPONSE });
        render(<FailureClustersPanel projectId={5} evalResults={EVAL_RESULTS} />);

        const clusterButton = await screen.findByRole('button', {
            name: /hallucination/i,
        });
        await userEvent.click(clusterButton);

        expect(screen.getByTestId('failure-cluster-augment-cluster-1')).toBeInTheDocument();
        expect(screen.getByTestId('failure-cluster-augment-run-cluster-1')).toBeInTheDocument();
    });

    it('augment button POSTs to the cluster-augment endpoint and surfaces the result', async () => {
        apiMock.get.mockResolvedValue({ data: CLUSTER_RESPONSE });
        apiMock.post.mockResolvedValue({
            data: {
                rows: [
                    {
                        payload: { question: 'Q', answer: 'A' },
                        synth_confidence: 0.9,
                        synth_source: 'playbook:qa-sft:cluster_targeted:cluster=cluster-1',
                    },
                ],
                backend_used: 'ollama:llama3.1:8b',
                elapsed_sec: 1.42,
                prompt_snippet: 'You are generating…',
            },
        });

        render(<FailureClustersPanel projectId={5} evalResults={EVAL_RESULTS} />);
        const clusterButton = await screen.findByRole('button', {
            name: /hallucination/i,
        });
        await userEvent.click(clusterButton);
        await userEvent.click(screen.getByTestId('failure-cluster-augment-run-cluster-1'));

        await waitFor(() => {
            expect(screen.getByTestId('failure-cluster-augment-ok-cluster-1')).toBeInTheDocument();
        });
        const okMessage = screen.getByTestId('failure-cluster-augment-ok-cluster-1');
        expect(okMessage.textContent).toMatch(/Generated 1 rows/);
        expect(okMessage.textContent).toMatch(/ollama:llama3.1:8b/);

        // Confirm the endpoint shape: POST to /clusters/cluster-1/augment.
        const postCall = apiMock.post.mock.calls.find((call) =>
            String(call[0]).includes('/clusters/cluster-1/augment'),
        );
        expect(postCall).toBeDefined();
        // Third arg is the axios config; target_count is in `params`.
        const config = postCall?.[2] as { params: Record<string, unknown> } | undefined;
        expect(config?.params?.target_count).toBe(20);
    });

    it('augment button surfaces error detail when the request fails', async () => {
        apiMock.get.mockResolvedValue({ data: CLUSTER_RESPONSE });
        apiMock.post.mockRejectedValue({
            response: { status: 503, data: { detail: 'No synth backend reachable.' } },
        });

        render(<FailureClustersPanel projectId={5} evalResults={EVAL_RESULTS} />);
        const clusterButton = await screen.findByRole('button', {
            name: /hallucination/i,
        });
        await userEvent.click(clusterButton);
        await userEvent.click(screen.getByTestId('failure-cluster-augment-run-cluster-1'));

        await waitFor(() => {
            expect(screen.getByTestId('failure-cluster-augment-error-cluster-1')).toBeInTheDocument();
        });
        expect(screen.getByTestId('failure-cluster-augment-error-cluster-1').textContent).toMatch(/No synth backend/);
    });
});
