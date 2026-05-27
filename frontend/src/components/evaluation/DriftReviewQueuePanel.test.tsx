import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        patch: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: {
        success: vi.fn(),
        error: vi.fn(),
        info: vi.fn(),
        warning: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import DriftReviewQueuePanel from './DriftReviewQueuePanel';


function makeRow(overrides: Record<string, unknown> = {}) {
    return {
        id: 1,
        project_id: 7,
        source_drift_check_id: null,
        cluster_reason_code: 'hallucination',
        cluster_signature: 'sig-001',
        payload: {
            question: 'What about edge case X?',
            answer: "I don't have reliable info on this.",
            is_hallucination_trap: true,
        },
        status: 'pending',
        source_confidence: 'rough',
        triage_note: null,
        created_at: '2026-05-27T20:00:00Z',
        triaged_at: null,
        ...overrides,
    };
}


/** Route-aware GET mock. Drives:
 *   - /projects/{id}                → recipe lookup (qa-sft default)
 *   - /drift/settings               → opt-in flag (default OFF)
 *   - /drift/review-queue           → list of pending rows
 */
function installGetRouter(opts: {
    recipeId?: string | null;
    settings?: { enabled: boolean; count: number };
    rows?: ReturnType<typeof makeRow>[];
} = {}) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/drift/settings')) {
            return {
                data: {
                    project_id: 7,
                    enabled: opts.settings?.enabled ?? false,
                    count: opts.settings?.count ?? 5,
                },
            };
        }
        if (url.includes('/drift/review-queue')) {
            return { data: { project_id: 7, rows: opts.rows ?? [] } };
        }
        if (/\/projects\/\d+(\?|$)/.test(url)) {
            return {
                data: {
                    selected_recipe: opts.recipeId
                        ? { recipe_id: opts.recipeId }
                        : null,
                },
            };
        }
        return { data: {} };
    });
}


describe('DriftReviewQueuePanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.put.mockReset();
        toastMock.success.mockReset();
        toastMock.error.mockReset();
    });

    it('shows the opt-in banner when auto-refresh is disabled', async () => {
        installGetRouter({ settings: { enabled: false, count: 5 } });
        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-optin-banner')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('drift-review-auto-on')).not.toBeInTheDocument();
    });

    it('shows the auto-on chip when auto-refresh is enabled', async () => {
        installGetRouter({ settings: { enabled: true, count: 8 } });
        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-auto-on')).toBeInTheDocument();
        });
        expect(screen.getByTestId('drift-review-auto-on').textContent).toMatch(/8/);
        expect(screen.queryByTestId('drift-review-optin-banner')).not.toBeInTheDocument();
    });

    it('enables auto-refresh on click and updates the chip', async () => {
        installGetRouter({ settings: { enabled: false, count: 5 } });
        // PUT returns the new state — panel re-renders with the chip.
        apiMock.put.mockResolvedValue({
            data: { project_id: 7, enabled: true, count: 5 },
        });
        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-enable-auto')).toBeInTheDocument();
        });

        await userEvent.click(screen.getByTestId('drift-review-enable-auto'));
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-auto-on')).toBeInTheDocument();
        });
        // PUT body sent the flip.
        expect(apiMock.put).toHaveBeenCalledWith(
            '/projects/7/drift/settings',
            { enabled: true },
        );
        // Success toast fired.
        expect(toastMock.success).toHaveBeenCalled();
    });

    it('renders pending rows with cluster context + accept/reject buttons', async () => {
        installGetRouter({
            recipeId: 'qa-sft',
            rows: [makeRow({ id: 42 }), makeRow({ id: 43, cluster_reason_code: 'coverage_gap' })],
        });
        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-row-42')).toBeInTheDocument();
        });
        // Two pending rows.
        expect(screen.getByTestId('drift-review-row-42')).toHaveAttribute('data-status', 'pending');
        expect(screen.getByTestId('drift-review-row-43')).toHaveAttribute('data-status', 'pending');
        // Cluster pill on each row.
        expect(screen.getByTestId('drift-review-row-42').textContent).toMatch(/hallucination/);
        expect(screen.getByTestId('drift-review-row-43').textContent).toMatch(/coverage_gap/);
        // Accept/reject buttons present.
        expect(screen.getByTestId('drift-review-row-42-accept')).toBeInTheDocument();
        expect(screen.getByTestId('drift-review-row-42-reject')).toBeInTheDocument();
    });

    it('"Generate now" hits the refresh endpoint and reloads the queue', async () => {
        installGetRouter({ recipeId: 'qa-sft', rows: [] });
        apiMock.post.mockImplementation(async (url: string) => {
            if (url.endsWith('/drift/refresh-traps')) {
                return {
                    data: {
                        project_id: 7,
                        generated: 3,
                        clusters_targeted: ['hallucination'],
                        simulated: false,
                        row_ids: [11, 12, 13],
                    },
                };
            }
            return { data: {} };
        });

        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review')).toBeInTheDocument();
        });

        // Click Generate now.
        await userEvent.click(screen.getByTestId('drift-review-refresh'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalled();
        });
        // POST hit refresh-traps with count + simulate from settings.
        const [url, body, opts] = apiMock.post.mock.calls[0];
        expect(url).toBe('/projects/7/drift/refresh-traps');
        // Body sent as null (params live in the query string); the
        // axios client passes the count + simulate in opts.params.
        expect(body).toBeNull();
        expect(opts.params).toEqual({ count: 5, simulate: false });
        // Last-summary line surfaces the generation outcome.
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-last-summary')).toBeInTheDocument();
        });
        expect(screen.getByTestId('drift-review-last-summary').textContent).toMatch(/3 traps/);
        expect(screen.getByTestId('drift-review-last-summary').textContent).toMatch(/hallucination/);
    });

    it('Accept triages a row → POST + queue reloaded without the row', async () => {
        installGetRouter({
            recipeId: 'qa-sft',
            rows: [makeRow({ id: 99 })],
        });
        apiMock.post.mockImplementation(async (url: string) => {
            if (url.endsWith('/triage')) {
                return { data: { id: 99, status: 'accepted', triaged_at: '2026-05-27T20:01:00Z', triage_note: null } };
            }
            return { data: {} };
        });

        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-row-99')).toBeInTheDocument();
        });

        // Now flip the queue mock to empty so the reload-after-accept
        // proves the row is gone from the pending view.
        installGetRouter({ recipeId: 'qa-sft', rows: [] });
        await userEvent.click(screen.getByTestId('drift-review-row-99-accept'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/drift/review-queue/99/triage',
                { accept: true },
            );
        });
        await waitFor(() => {
            expect(screen.queryByTestId('drift-review-row-99')).not.toBeInTheDocument();
        });
        expect(toastMock.success).toHaveBeenCalled();
    });

    it('shows an empty-state message when no pending rows exist', async () => {
        installGetRouter({ rows: [] });
        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-empty')).toBeInTheDocument();
        });
        expect(screen.getByTestId('drift-review-empty').textContent)
            .toMatch(/Generate now/);
    });

    it('changing the status filter re-fetches with the new filter', async () => {
        installGetRouter({ rows: [] });
        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-filter')).toBeInTheDocument();
        });

        // Switch to "accepted" — the GET fires with status=accepted.
        await userEvent.selectOptions(
            screen.getByTestId('drift-review-filter'),
            'accepted',
        );
        await waitFor(() => {
            const calls = apiMock.get.mock.calls.filter(
                (c) => typeof c[0] === 'string' && c[0].includes('/drift/review-queue'),
            );
            const hasAccepted = calls.some(
                (c) => c[1]?.params?.status === 'accepted',
            );
            expect(hasAccepted).toBe(true);
        });
    });

    it('surfaces an error toast when triage fails', async () => {
        installGetRouter({ rows: [makeRow({ id: 1 })] });
        apiMock.post.mockRejectedValue({
            response: { status: 409, data: { detail: 'queue_row_already_triaged' } },
        });
        render(<DriftReviewQueuePanel projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('drift-review-row-1')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('drift-review-row-1-reject'));
        await waitFor(() => {
            expect(toastMock.error).toHaveBeenCalled();
        });
        expect(toastMock.error.mock.calls[0][0]).toMatch(/already_triaged/);
    });
});
