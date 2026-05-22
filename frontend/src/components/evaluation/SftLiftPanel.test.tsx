import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import SftLiftPanel from './SftLiftPanel';

const HAPPY_SUMMARY = {
    status: 'ok',
    project_id: 1,
    message: null,
    baseline: {
        experiment_id: 10,
        experiment_name: 'Baseline · SmolLM2-135M',
        base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        training_mode: 'sft',
        completed_at: '2026-05-21T12:00:00Z',
        eval_result_id: 100,
        dataset_name: 'test',
        eval_type: 'exact_match',
        metrics: { exact_match: 0.18, f1: 0.32 },
        pass_rate: 0.18,
    },
    trained: {
        experiment_id: 11,
        experiment_name: 'qa-sft-experiment-3',
        base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        training_mode: 'sft',
        completed_at: '2026-05-22T12:00:00Z',
        eval_result_id: 101,
        dataset_name: 'test',
        eval_type: 'exact_match',
        metrics: { exact_match: 0.5, f1: 0.65 },
        pass_rate: 0.5,
    },
    metric_lifts: [
        {
            metric_id: 'f1',
            baseline_value: 0.32,
            trained_value: 0.65,
            absolute_delta: 0.33,
            relative_delta_pct: 103.1,
            direction: 'improved',
            is_headline: true,
        },
        {
            metric_id: 'exact_match',
            baseline_value: 0.18,
            trained_value: 0.5,
            absolute_delta: 0.32,
            relative_delta_pct: 177.8,
            direction: 'improved',
            is_headline: true,
        },
    ],
    gate_status: [
        {
            gate_id: 'min_exact_match',
            metric_id: 'exact_match',
            threshold: 0.4,
            operator: 'gte',
            required: true,
            baseline_value: 0.18,
            trained_value: 0.5,
            baseline_passes: false,
            trained_passes: true,
            delta_to_threshold: 0.1,
            status: 'cleared',
        },
        {
            gate_id: 'min_f1',
            metric_id: 'f1',
            threshold: 0.5,
            operator: 'gte',
            required: true,
            baseline_value: 0.32,
            trained_value: 0.65,
            baseline_passes: false,
            trained_passes: true,
            delta_to_threshold: 0.15,
            status: 'cleared',
        },
    ],
    eval_pack_id: 'evalpack.general.default',
    task_profile_used: 'instruction_sft',
};

describe('SftLiftPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders the baseline → trained headline with the picked headline metric', async () => {
        apiMock.get.mockResolvedValueOnce({ data: HAPPY_SUMMARY });
        render(<SftLiftPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('sft-lift-panel')).toBeInTheDocument();
        });
        expect(screen.getByTestId('sft-lift-baseline-name')).toHaveTextContent(
            /Baseline · SmolLM2-135M/,
        );
        expect(screen.getByTestId('sft-lift-trained-name')).toHaveTextContent(
            'qa-sft-experiment-3',
        );
        // Headline metric is f1; the delta shows absolute + relative.
        expect(screen.getByTestId('sft-lift-headline-delta')).toHaveTextContent(
            '+0.33',
        );
        expect(screen.getByTestId('sft-lift-headline-delta')).toHaveTextContent(
            '+103%',
        );
    });

    it('renders per-metric rows with baseline/trained/delta', async () => {
        apiMock.get.mockResolvedValueOnce({ data: HAPPY_SUMMARY });
        render(<SftLiftPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('sft-lift-row-f1')).toBeInTheDocument();
        });
        expect(screen.getByTestId('sft-lift-baseline-f1')).toHaveTextContent('0.32');
        expect(screen.getByTestId('sft-lift-trained-f1')).toHaveTextContent('0.65');
        expect(screen.getByTestId('sft-lift-delta-f1')).toHaveTextContent('+0.33');
        expect(screen.getByTestId('sft-lift-row-exact_match')).toBeInTheDocument();
    });

    it('shows gate-cleared badges + summary counts when training cleared gates', async () => {
        apiMock.get.mockResolvedValueOnce({ data: HAPPY_SUMMARY });
        render(<SftLiftPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('sft-lift-gate-rows')).toBeInTheDocument();
        });
        const summary = screen.getByTestId('sft-lift-gate-summary');
        expect(summary).toHaveTextContent('2 cleared');
        expect(summary).toHaveTextContent('0 still failing');
        expect(screen.getAllByTestId('sft-lift-gate-cleared').length).toBe(2);
    });

    it('shows "still failing" badge + delta for an unmet gate', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...HAPPY_SUMMARY,
                gate_status: [
                    {
                        ...HAPPY_SUMMARY.gate_status[0],
                        gate_id: 'min_accuracy',
                        metric_id: 'accuracy',
                        threshold: 0.5,
                        baseline_value: 0.2,
                        trained_value: 0.42,
                        baseline_passes: false,
                        trained_passes: false,
                        delta_to_threshold: -0.08,
                        status: 'still_failing',
                    },
                ],
            },
        });
        render(<SftLiftPanel projectId={1} />);

        await waitFor(() =>
            expect(
                screen.getByTestId('sft-lift-gate-still_failing'),
            ).toBeInTheDocument(),
        );
        const row = screen.getByTestId('sft-lift-gate-row-min_accuracy');
        expect(row).toHaveTextContent('still below threshold');
        expect(row).toHaveTextContent('accuracy');
        expect(row).toHaveTextContent('-0.08');
    });

    it('renders the no-baseline fallback when prereqs missing', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                status: 'no_baseline',
                project_id: 1,
                message: 'Run the Quickstart Baseline tile first.',
                baseline: null,
                trained: HAPPY_SUMMARY.trained,
                metric_lifts: [],
                gate_status: [],
            },
        });
        render(<SftLiftPanel projectId={1} />);

        await waitFor(() => {
            expect(
                screen.getByTestId('sft-lift-panel-no-baseline'),
            ).toBeInTheDocument();
        });
        expect(
            screen.getByText(/Run the Quickstart Baseline tile first/),
        ).toBeInTheDocument();
    });

    it('renders the no-trained fallback when training hasn\'t happened yet', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                status: 'no_trained',
                project_id: 1,
                message: 'No trained eval results yet.',
                baseline: HAPPY_SUMMARY.baseline,
                trained: null,
                metric_lifts: [],
                gate_status: [],
            },
        });
        render(<SftLiftPanel projectId={1} />);

        await waitFor(() =>
            expect(
                screen.getByTestId('sft-lift-panel-no-trained'),
            ).toBeInTheDocument(),
        );
    });

    it('renders the no-overlap fallback when metric keys are disjoint', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                status: 'no_overlap',
                project_id: 1,
                message: 'Eval results share no comparable metrics.',
                baseline: HAPPY_SUMMARY.baseline,
                trained: HAPPY_SUMMARY.trained,
                metric_lifts: [],
                gate_status: [],
            },
        });
        render(<SftLiftPanel projectId={1} />);

        await waitFor(() =>
            expect(
                screen.getByTestId('sft-lift-panel-no-overlap'),
            ).toBeInTheDocument(),
        );
    });

    it('renders the loading skeleton then the panel when fetch resolves', async () => {
        let resolve: ((value: { data: typeof HAPPY_SUMMARY }) => void) | undefined;
        apiMock.get.mockReturnValueOnce(
            new Promise((r) => {
                resolve = r;
            }),
        );

        render(<SftLiftPanel projectId={1} />);
        expect(screen.getByTestId('sft-lift-panel-loading')).toBeInTheDocument();

        resolve?.({ data: HAPPY_SUMMARY });
        await waitFor(() =>
            expect(screen.getByTestId('sft-lift-panel')).toBeInTheDocument(),
        );
    });

    it('shows error inline when fetch fails', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { data: { detail: 'sft lift exploded' } },
        });
        render(<SftLiftPanel projectId={1} />);

        await waitFor(() =>
            expect(screen.getByTestId('sft-lift-panel-error')).toBeInTheDocument(),
        );
        expect(screen.getByTestId('sft-lift-panel-error')).toHaveTextContent(
            'sft lift exploded',
        );
    });
});
