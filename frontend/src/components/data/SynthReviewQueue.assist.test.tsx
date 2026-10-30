import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import SynthReviewQueue from './SynthReviewQueue';

const failAssist = { verdict: 'fail', reason_code: 'unsafe_pattern', note: 'disables TLS verification', backend: 'cloud:anthropic:m', job_id: 21 };

const payload = {
    project_id: 5, dataset_id: 1, total_rows: 3, total_pending: 3, total_accepted: 0,
    facets: {
        jobs: [], backends: [],
        verdicts: [{ verdict: 'fail', count: 2 }, { verdict: 'pass', count: 1 }],
        reason_codes: [{ reason_code: 'ok', count: 1 }, { reason_code: 'unsafe_pattern', count: 2 }],
    },
    groups: [
        {
            synth_source: 'fail:unsafe_pattern', count: 2,
            rows: [
                { id: 1, synth_confidence: 0.9, review_assist: failAssist, preview: '{"c": 1}', payload: {} },
                { id: 2, synth_confidence: 0.8, review_assist: failAssist, preview: '{"c": 2}', payload: {} },
            ],
        },
        {
            synth_source: 'pass:ok', count: 1,
            rows: [{ id: 3, synth_confidence: 0.7, review_assist: { verdict: 'pass', reason_code: 'ok', note: 'fine' }, preview: '{"c": 3}', payload: {} }],
        },
    ],
    accepted_groups: [],
};

function mockGets() {
    apiMock.get.mockImplementation((url: string) => {
        if (url.endsWith('/synthetic/backends/cloud/models')) {
            return Promise.resolve({ data: { providers: [{
                provider: 'anthropic', key_saved: true,
                models: [{ id: 'claude-haiku-4-5', label: 'Haiku' }],
            }] } });
        }
        if (url.endsWith('/synthetic/backends')) return Promise.resolve({ data: { backends: [] } });
        return Promise.resolve({ data: payload });
    });
}

describe('SynthReviewQueue pre-screen', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        Object.values(toastMock).forEach((fn) => fn.mockReset());
        mockGets();
    });

    it('shows each row verdict, reason and note', async () => {
        render(<SynthReviewQueue projectId={5} />);
        const badge = await screen.findByTestId('synth-review-queue-assist-1');
        expect(badge.textContent).toBe('fail · unsafe_pattern — disables TLS verification');
        expect(screen.getByTestId('synth-review-queue-assist-3').textContent).toBe('pass — fine');
    });

    it('filters by verdict and reason through the query params', async () => {
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.selectOptions(await screen.findByTestId('synth-review-queue-filter-verdict'), 'fail');
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-queue', { params: { verdict: 'fail' } },
        ));
        await user.selectOptions(screen.getByTestId('synth-review-queue-filter-reason'), 'unsafe_pattern');
        await waitFor(() => expect(apiMock.get).toHaveBeenLastCalledWith(
            '/projects/5/synthetic/review-queue', { params: { verdict: 'fail', reason_code: 'unsafe_pattern' } },
        ));
    });

    it('rejects a whole reason group with assist:<reason_code> and accepts a passing group', async () => {
        apiMock.post.mockResolvedValue({ data: { accepted: 1, rejected: 2, not_found: 0, not_pending: 0, total_remaining_pending: 0 } });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-groupby-assist_reason'));
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-queue', { params: { group_by: 'assist_reason' } },
        ));
        await user.click(await screen.findByTestId('synth-review-queue-group-reject-fail:unsafe_pattern'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-queue/bulk-update',
            { row_ids: [1, 2], action: 'reject', reject_reason: 'assist:unsafe_pattern', decided_via: 'group:assist_reason:fail:unsafe_pattern' },
        ));
        await user.click(screen.getByTestId('synth-review-queue-group-accept-pass:ok'));
        await waitFor(() => expect(apiMock.post).toHaveBeenLastCalledWith(
            '/projects/5/synthetic/review-queue/bulk-update',
            { row_ids: [3], action: 'accept', reject_reason: null, decided_via: 'group:assist_reason:pass:ok' },
        ));
    });

    it('re-fetches the group just before a bulk call and sends only its current ids', async () => {
        apiMock.post.mockResolvedValue({ data: { accepted: 0, rejected: 1, not_found: 0, not_pending: 0, total_remaining_pending: 0 } });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-groupby-assist_reason'));
        const button = await screen.findByTestId('synth-review-queue-group-reject-fail:unsafe_pattern');
        // Meanwhile row 2 was re-judged as a pass and row 1 stays a fail.
        const changed = {
            ...payload,
            groups: [
                { ...payload.groups[0], count: 1, rows: [payload.groups[0].rows[0]] },
                { ...payload.groups[1], rows: [
                    payload.groups[1].rows[0],
                    { ...payload.groups[0].rows[1], review_assist: { verdict: 'pass', reason_code: 'ok', note: '' } },
                ] },
            ],
        };
        apiMock.get.mockImplementation(() => Promise.resolve({ data: changed }));
        const before = apiMock.get.mock.calls.length;
        await user.click(button);
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-queue/bulk-update',
            { row_ids: [1], action: 'reject', reject_reason: 'assist:unsafe_pattern', decided_via: 'group:assist_reason:fail:unsafe_pattern' },
        ));
        expect(apiMock.get.mock.calls.length).toBeGreaterThan(before);
    });

    it('does not bulk-act on a group that emptied since it was rendered', async () => {
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-groupby-assist_reason'));
        const button = await screen.findByTestId('synth-review-queue-group-reject-fail:unsafe_pattern');
        apiMock.get.mockImplementation(() => Promise.resolve({ data: { ...payload, groups: [payload.groups[1]] } }));
        await user.click(button);
        await waitFor(() => expect(screen.getByText(/no pending rows left to reject/)).toBeTruthy());
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('offers reject only (never accept) on fail, unsure and not-pre-screened groups', async () => {
        const groups = [
            { synth_source: 'unsure:judge_unparseable', count: 1,
              rows: [{ id: 4, synth_confidence: 0.5, review_assist: { verdict: 'unsure', reason_code: 'judge_unparseable', note: '' }, preview: '{}', payload: {} }] },
            { synth_source: '(not pre-screened)', count: 1, rows: [{ id: 5, synth_confidence: 0.5, preview: '{}', payload: {} }] },
            payload.groups[0], payload.groups[1],
        ];
        apiMock.get.mockImplementation((url: string) => {
            if (url.endsWith('/synthetic/backends/cloud/models')) return Promise.resolve({ data: { providers: [] } });
            if (url.endsWith('/synthetic/backends')) return Promise.resolve({ data: { backends: [] } });
            return Promise.resolve({ data: { ...payload, groups } });
        });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-groupby-assist_reason'));
        await screen.findByTestId('synth-review-queue-group-reject-unsure:judge_unparseable');
        expect(screen.getByTestId('synth-review-queue-group-reject-(not pre-screened)')).toBeTruthy();
        expect(screen.queryByTestId('synth-review-queue-group-accept-unsure:judge_unparseable')).toBeNull();
        expect(screen.queryByTestId('synth-review-queue-group-accept-(not pre-screened)')).toBeNull();
        expect(screen.getByTestId('synth-review-queue-group-accept-pass:ok')).toBeTruthy();
        expect(screen.queryByTestId('synth-review-queue-group-accept-fail:unsafe_pattern')).toBeNull();
        expect(screen.getByTestId('synth-review-queue-group-reject-fail:unsafe_pattern')).toBeTruthy();
    });

    it('starts a pre-screen with the picked backend, rubric text and rows per call', async () => {
        apiMock.post.mockResolvedValue({ data: { id: 77, kind: 'synth_review_assist', status: 'queued' } });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-assist-toggle'));
        await user.selectOptions(await screen.findByTestId('synth-review-queue-assist-provider'), 'anthropic');
        await user.selectOptions(await screen.findByTestId('synth-review-queue-assist-cloud-model'), 'claude-haiku-4-5');
        await user.type(screen.getByTestId('synth-review-queue-assist-rubric'), 'Only read operations.');
        const perCall = screen.getByTestId('synth-review-queue-assist-rows-per-call');
        await user.clear(perCall);
        await user.type(perCall, '40');
        await user.click(await screen.findByTestId('synth-review-queue-assist-batch'));
        await user.click(screen.getByTestId('synth-review-queue-assist-start'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/review-assist',
            {
                backend: 'cloud:anthropic:claude-haiku-4-5', use_batch: true,
                rubric_text: 'Only read operations.', rows_per_call: 25,
            },
            { params: { async_job: true } },
        ));
        expect(toastMock.info).toHaveBeenCalledWith(expect.stringContaining('job #77'), 6000);
    });

    it('shows the budget message when the start is refused with 402', async () => {
        apiMock.post.mockRejectedValue({ response: { status: 402, data: { error_code: 'llm_budget_exceeded', message: 'API budget reached (project)' } } });
        const user = userEvent.setup();
        render(<SynthReviewQueue projectId={5} />);
        await user.click(await screen.findByTestId('synth-review-queue-assist-toggle'));
        await user.click(await screen.findByTestId('synth-review-queue-assist-start'));
        await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith('API budget reached (project)', 8000));
    });
});
