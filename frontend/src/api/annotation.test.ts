import { beforeEach, describe, expect, it, vi } from 'vitest';

import { fetchNextRow } from './annotation';


// Hoisted so the mock factory below can close over the same spies the
// test reads. ``vi.mock`` is hoisted above imports — without this
// pattern the assertions would see a different ``post`` than the call.
const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));
vi.mock('./client', () => ({ default: apiMock }));


function mockNextRowResponse(strategy: string) {
    apiMock.post.mockResolvedValueOnce({
        data: {
            row: { id: 7, job_id: 1, raw_payload: { text: 'x' } },
            queue_empty: false,
            strategy,
        },
    });
}


describe('fetchNextRow strategy wiring (Epic F Phase 1)', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
    });

    it('defaults the strategy param to ``fifo`` when the caller omits it', async () => {
        // Pins the historical behaviour: existing callers that pass
        // only (projectId, jobId, userId) get FIFO. The active
        // strategy must be opt-in to protect a labeler from being
        // silently switched to a slower, model-dependent path.
        mockNextRowResponse('fifo');
        await fetchNextRow(2, 3, 1);
        expect(apiMock.post).toHaveBeenCalledTimes(1);
        const [, body] = apiMock.post.mock.calls[0];
        expect(body).toEqual({ user_id: 1, strategy: 'fifo' });
    });

    it('forwards an explicit ``active`` strategy in the request body', async () => {
        // The strategy travels in the JSON body (not the URL) so the
        // backend's pydantic schema sees it on every reviewer-side
        // call. Asserting the body shape keeps a future "the field
        // moved to a query param" refactor from silently downgrading
        // every labeler back to FIFO.
        mockNextRowResponse('active');
        await fetchNextRow(2, 3, 99, 'active');
        const [url, body] = apiMock.post.mock.calls[0];
        expect(url).toBe('/projects/2/label-jobs/3/next-row');
        expect(body).toEqual({ user_id: 99, strategy: 'active' });
    });

    it('returns the backend\'s strategy echo so the UI can flag fallbacks', async () => {
        // The backend silently degrades active→fifo when no scoreable
        // experiment exists. Echoing the actual strategy back lets a
        // future UI surface the fallback (e.g., a "no model yet,
        // showing FIFO" hint) rather than misleading the user.
        mockNextRowResponse('fifo');
        const result = await fetchNextRow(2, 3, null, 'active');
        expect(result.strategy).toBe('fifo');
        expect(result.row?.id).toBe(7);
    });
});
