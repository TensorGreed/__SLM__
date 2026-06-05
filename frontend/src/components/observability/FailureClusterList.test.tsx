import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { FailureCluster } from '../../types/observability';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    toastMock: {
        success: vi.fn(),
        error: vi.fn(),
        info: vi.fn(),
        warning: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import FailureClusterList from './FailureClusterList';

function makeCluster(overrides: Partial<FailureCluster>): FailureCluster {
    return {
        id: 1,
        project_id: 7,
        stage: 'training',
        reason_code: 'training_runtime_error',
        signature: 'abcdef123456',
        failure_count: 3,
        first_seen_at: '2026-05-07T10:00:00Z',
        last_seen_at: '2026-05-07T11:00:00Z',
        exemplar_event_ids: [101, 102, 103],
        exemplar_summaries: [
            'CUDA OOM at step 1200',
            'CUDA OOM at step 4500',
            'CUDA OOM at step 9876',
        ],
        exemplar_run_ids: ['exp-1', 'exp-1', 'exp-2'],
        last_computed_at: '2026-05-07T12:00:00Z',
        ...overrides,
    };
}

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    toastMock.success.mockReset();
    toastMock.error.mockReset();
});

describe('FailureClusterList', () => {
    it('renders cluster rows ordered by failure_count', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 7,
                limit: 100,
                clusters: [
                    makeCluster({ failure_count: 9, signature: 'aaa' }),
                    makeCluster({
                        failure_count: 2,
                        signature: 'bbb',
                        reason_code: 'training_dispatch_error',
                    }),
                ],
            },
        });
        render(
            <FailureClusterList projectId={7} onSelectRun={vi.fn()} />,
        );
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/7/failure-clusters',
            );
        });
        expect(await screen.findByText('9×')).toBeInTheDocument();
        expect(screen.getByText('2×')).toBeInTheDocument();
        expect(
            screen.getByText('training_runtime_error'),
        ).toBeInTheDocument();
        expect(
            screen.getByText('training_dispatch_error'),
        ).toBeInTheDocument();
    });

    it('renders empty state when no clusters', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 7, limit: 100, clusters: [] },
        });
        render(
            <FailureClusterList projectId={7} onSelectRun={vi.fn()} />,
        );
        expect(
            await screen.findByText(/No failure clusters yet/i),
        ).toBeInTheDocument();
    });

    it('expanding a cluster shows exemplars; clicking an exemplar fires onSelectRun with run_id', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 7, limit: 100, clusters: [makeCluster({})] },
        });
        const onSelect = vi.fn();
        render(
            <FailureClusterList projectId={7} onSelectRun={onSelect} />,
        );
        const user = userEvent.setup();
        await user.click(
            await screen.findByRole('button', {
                name: /Expand exemplars/i,
            }),
        );
        expect(screen.getByText(/CUDA OOM at step 1200/)).toBeInTheDocument();

        await user.click(
            screen.getByRole('button', {
                name: /Open events for cluster exemplar 101/i,
            }),
        );
        expect(onSelect).toHaveBeenCalledWith('exp-1');
    });

    it('Recompute POSTs and renders the summary line', async () => {
        apiMock.get.mockResolvedValue({
            data: { project_id: 7, limit: 100, clusters: [] },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                project_id: 7,
                window_start: null,
                window_end: null,
                events_considered: 12,
                events_skipped_no_reason_code: 0,
                clusters_total: 3,
                clusters_created: 2,
                clusters_updated: 1,
                computed_at: '2026-05-07T12:00:00Z',
            },
        });

        render(
            <FailureClusterList projectId={7} onSelectRun={vi.fn()} />,
        );
        await screen.findByText(/No failure clusters yet/i);

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /^Recompute$/i }),
        );
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/failure-clusters/recompute',
                {},
            );
        });
        expect(
            await screen.findByText(/scanned 12 event/i),
        ).toBeInTheDocument();
    });

    it('surfaces stable error code on fetch failure', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 404, data: { detail: 'project_not_found' } },
        });
        render(
            <FailureClusterList projectId={9} onSelectRun={vi.fn()} />,
        );
        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent('project_not_found');
    });

    // ─────────────────────────────────────────────────────────────────
    // Gap-#5 slice 3: "Adopt as gate" CTA on cluster rows
    // ─────────────────────────────────────────────────────────────────

    it('clicking "Adopt as gate" POSTs adopt-gate-from-cluster with the cluster id', async () => {
        const cluster = makeCluster({ id: 42, reason_code: 'safety_failure' });
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 7, limit: 100, clusters: [cluster] },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                project_id: 7,
                preferred_pack_id: 'evalpack.project.scaffolded',
                scaffolded_pack: { pack_id: 'evalpack.project.scaffolded', task_specs: [{ task_profile: 'qa', gates: [] }] },
                new_gate: {
                    gate_id: 'min_safety_pass_rate_from_cluster',
                    metric_id: 'safety_pass_rate',
                    operator: 'gte',
                    threshold: 0.95,
                    required: false,
                },
                cluster_reason_code: 'safety_failure',
            },
        });

        render(<FailureClusterList projectId={7} onSelectRun={vi.fn()} />);
        const button = await screen.findByTestId('failure-cluster-42-adopt-gate');
        await userEvent.click(button);

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/evaluation/adopt-gate-from-cluster',
                { cluster_id: 42 },
            );
        });
        // Success toast names the new gate so the user can find it
        // in the editor.
        await waitFor(() => {
            expect(toastMock.success).toHaveBeenCalled();
        });
        expect(toastMock.success.mock.calls[0][0]).toMatch(/min_safety_pass_rate_from_cluster/);
    });

    it('surfaces the backend error code when adopt-gate is rejected', async () => {
        const cluster = makeCluster({ id: 99, reason_code: 'totally_unknown' });
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 7, limit: 100, clusters: [cluster] },
        });
        apiMock.post.mockRejectedValueOnce({
            response: {
                status: 400,
                data: { detail: 'no_gate_suggestion_for_reason_code:totally_unknown' },
            },
        });

        render(<FailureClusterList projectId={7} onSelectRun={vi.fn()} />);
        const button = await screen.findByTestId('failure-cluster-99-adopt-gate');
        await userEvent.click(button);

        await waitFor(() => {
            expect(toastMock.error).toHaveBeenCalled();
        });
        expect(toastMock.error.mock.calls[0][0])
            .toMatch(/no_gate_suggestion_for_reason_code:totally_unknown/);
    });
});
