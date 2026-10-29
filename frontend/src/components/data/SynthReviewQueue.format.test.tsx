import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), patch: vi.fn(), put: vi.fn(), delete: vi.fn() },
    toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import SynthReviewQueue from './SynthReviewQueue';

const payload = {
    project_id: 5, dataset_id: 1, total_rows: 1, total_pending: 1, total_accepted: 0,
    facets: { jobs: [], backends: [], verdicts: [], reason_codes: [] },
    groups: [{
        synth_source: 'playbook:code-remediation:code_examples', count: 1,
        rows: [{ id: 1, synth_confidence: 0.8, preview: '{"c": 1}', payload: {} }],
    }],
    accepted_groups: [],
};

const dry = {
    dry_run: true, checked: 4, mismatches: 2, by_language: { xml: 2 }, rejected: 0,
    examples: [{ id: 1, language: 'xml', detail: 'not well-formed XML: syntax error', task: 'Read the rules' }],
};

describe('SynthReviewQueue code format re-check', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.get.mockImplementation((url: string) => {
            if (url.endsWith('/synthetic/backends/cloud/models')) return Promise.resolve({ data: { providers: [] } });
            if (url.endsWith('/synthetic/backends')) return Promise.resolve({ data: { backends: [] } });
            return Promise.resolve({ data: payload });
        });
    });

    it('shows the dry-run count, then rejects only after the confirm', async () => {
        apiMock.post.mockResolvedValueOnce({ data: dry }).mockResolvedValueOnce({ data: { ...dry, dry_run: false, rejected: 2 } });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-format-check'));
        const confirm = await screen.findByTestId('synth-review-queue-format-confirm');
        expect(confirm.textContent).toContain('2 of 4 pending code examples');
        expect(confirm.textContent).toContain('xml: 2');
        expect(apiMock.post).toHaveBeenCalledTimes(1);
        expect(apiMock.post).toHaveBeenCalledWith('/projects/5/synthetic/review-queue/recheck-code-format', { dry_run: true });
        await user.click(screen.getByTestId('synth-review-queue-format-apply'));
        await waitFor(() => expect(apiMock.post).toHaveBeenLastCalledWith(
            '/projects/5/synthetic/review-queue/recheck-code-format', { dry_run: false },
        ));
        await waitFor(() => expect(screen.queryByTestId('synth-review-queue-format-confirm')).toBeNull());
    });

    it('cancel leaves everything alone and a clean queue offers no reject', async () => {
        apiMock.post.mockResolvedValueOnce({ data: { ...dry, mismatches: 0, by_language: {}, examples: [] } });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-format-check'));
        const confirm = await screen.findByTestId('synth-review-queue-format-confirm');
        expect(confirm.textContent).toContain('All 4 pending code examples parse');
        expect(screen.queryByTestId('synth-review-queue-format-apply')).toBeNull();
        await user.click(screen.getByTestId('synth-review-queue-format-cancel'));
        expect(screen.queryByTestId('synth-review-queue-format-confirm')).toBeNull();
        expect(apiMock.post).toHaveBeenCalledTimes(1);
    });
});
