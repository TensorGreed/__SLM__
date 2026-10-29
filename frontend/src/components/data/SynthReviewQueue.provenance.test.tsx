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

import SynthReviewQueue from './SynthReviewQueue';

const provenancePayload = {
    project_id: 7, dataset_id: 1, total_rows: 3, total_pending: 3, total_accepted: 0,
    facets: {
        jobs: [{ synth_job_id: 11, count: 2 }, { synth_job_id: 12, count: 1 }],
        backends: [{ synth_backend: 'cloud:anthropic:m', count: 2 }, { synth_backend: 'command:x', count: 1 }],
    },
    groups: [
        {
            synth_source: 'job:11', label: 'job:11 · cloud:anthropic:m · alpha · batch', count: 2,
            rows: [
                { id: 1, synth_confidence: 0.9, synth_job_id: 11, synth_backend: 'cloud:anthropic:m', synth_batch: true, preview: '{"t": 1}', payload: {} },
                { id: 2, synth_confidence: 0.8, synth_job_id: 11, synth_backend: 'cloud:anthropic:m', synth_batch: true, preview: '{"t": 2}', payload: {} },
            ],
        },
        {
            synth_source: 'job:12', label: 'job:12 · command:x · alpha · direct', count: 1,
            rows: [{ id: 3, synth_confidence: 0.7, synth_job_id: 12, synth_backend: 'command:x', synth_batch: false, preview: '{"t": 3}', payload: {} }],
        },
    ],
    accepted_groups: [],
};

describe('SynthReviewQueue provenance', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.get.mockResolvedValue({ data: provenancePayload });
    });

    it('labels job groups with the job id, backend, mode and delivery, and stamps each row', async () => {
        render(<SynthReviewQueue projectId={7} />);
        const labels = await screen.findAllByTestId('synth-review-queue-group-label');
        expect(labels.map((l) => l.textContent)).toEqual([
            'job:11 · cloud:anthropic:m · alpha · batch',
            'job:12 · command:x · alpha · direct',
        ]);
        expect(screen.getByTestId('synth-review-queue-provenance-1').textContent).toBe('job #11 · cloud:anthropic:m · batch');
        expect(screen.getByTestId('synth-review-queue-provenance-3').textContent).toBe('job #12 · command:x');
    });

    it('groups by job / backend through the group_by param', async () => {
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={7} />);
        await user.click(await screen.findByTestId('synth-review-queue-groupby-job'));
        await waitFor(() => expect(apiMock.get).toHaveBeenLastCalledWith(
            '/projects/7/synthetic/review-queue', { params: { group_by: 'job' } },
        ));
        await user.click(screen.getByTestId('synth-review-queue-groupby-backend'));
        await waitFor(() => expect(apiMock.get).toHaveBeenLastCalledWith(
            '/projects/7/synthetic/review-queue', { params: { group_by: 'backend' } },
        ));
    });

    it('filters by one job and one backend', async () => {
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={7} />);
        await user.selectOptions(await screen.findByTestId('synth-review-queue-filter-job'), '11');
        await waitFor(() => expect(apiMock.get).toHaveBeenLastCalledWith(
            '/projects/7/synthetic/review-queue', { params: { synth_job_id: 11 } },
        ));
        await user.selectOptions(screen.getByTestId('synth-review-queue-filter-backend'), 'command:x');
        await waitFor(() => expect(apiMock.get).toHaveBeenLastCalledWith(
            '/projects/7/synthetic/review-queue', { params: { synth_job_id: 11, synth_backend: 'command:x' } },
        ));
    });
});
