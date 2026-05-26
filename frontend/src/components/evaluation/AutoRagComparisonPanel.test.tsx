/**
 * AutoRagComparisonPanel — Hardening tests for the "Run comparison"
 * UI affordance.
 *
 * Three scenarios cover the new flow:
 *   1. 404 empty-state renders the primary "Run comparison" CTA
 *      AND POSTs to the run endpoint on click + fires an info toast.
 *   2. Cached payload renders a "Re-run comparison" button in the
 *      header that hits the same endpoint.
 *   3. 409 idempotency response surfaces a warning toast referencing
 *      the existing job (the user's expected next step is to wait
 *      for the existing job, not retry).
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
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
// jobsStore.refreshAfterLocalChange is fired after a successful run —
// stub it so the test doesn't spin up the real polling loop.
const refreshSpy = vi.fn();
vi.mock('../../stores/jobsStore', () => ({
    useJobsStore: {
        getState: () => ({ refreshAfterLocalChange: refreshSpy }),
    },
}));

import AutoRagComparisonPanel from './AutoRagComparisonPanel';


const HAPPY_CACHED_PAYLOAD = {
    project_id: 4,
    recipe_id: 'qa-sft',
    cached_at: '2026-05-26T12:00:00Z',
    summary: {
        off_mean_f1: 0.10,
        on_mean_f1: 0.30,
        absolute_lift: 0.20,
        relative_lift_pct: 200.0,
        n_val_rows: 28,
        rag_k: 3,
        phase_9c_reference_lift_pct: 146.49,
    },
    rows: [
        {
            question: 'Q1?',
            reference: 'A1',
            without_rag: { generated: 'wrong', f1: 0.1 },
            with_rag: { generated: 'A1', f1: 1.0, retrieved_row_count: 3 },
        },
    ],
};


describe('AutoRagComparisonPanel — run-comparison button', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        toastMock.success.mockReset();
        toastMock.error.mockReset();
        toastMock.info.mockReset();
        toastMock.warning.mockReset();
        refreshSpy.mockReset();
    });

    it('renders the primary "Run comparison" CTA on 404 empty-state', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 404, data: { detail: 'No comparison cached' } },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                id: 33,
                kind: 'auto_rag_comparison',
                title: 'Auto-RAG comparison · project #4',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 4,
                user_id: null,
                params: {},
                result: null,
                error: null,
                queued_at: '2026-05-26T12:00:00Z',
                started_at: null,
                completed_at: null,
                dismissed_at: null,
            },
        });
        render(<AutoRagComparisonPanel projectId={4} />);
        const btn = await screen.findByTestId('auto-rag-comparison-run-btn');
        expect(btn).toHaveTextContent(/run comparison/i);
        // CLI fallback still available behind a disclosure.
        expect(screen.getByTestId('auto-rag-comparison-empty-cmd')).toBeInTheDocument();

        await userEvent.click(btn);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/4/auto-rag/comparison/run',
            );
        });
        // Info toast references the job id from the response.
        expect(toastMock.info).toHaveBeenCalledWith(
            expect.stringContaining('#33'),
            4000,
        );
        // Jobs store gets kicked so the bell shows the new job on
        // the next tick.
        expect(refreshSpy).toHaveBeenCalled();
    });

    it('renders a "Re-run comparison" affordance on the cached payload header', async () => {
        apiMock.get.mockResolvedValueOnce({ data: HAPPY_CACHED_PAYLOAD });
        apiMock.post.mockResolvedValueOnce({
            data: {
                id: 34,
                kind: 'auto_rag_comparison',
                title: 'Auto-RAG comparison · project #4',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 4,
                user_id: null,
                params: {},
                result: null,
                error: null,
                queued_at: '2026-05-26T12:00:00Z',
                started_at: null,
                completed_at: null,
                dismissed_at: null,
            },
        });
        render(<AutoRagComparisonPanel projectId={4} />);
        const btn = await screen.findByTestId('auto-rag-comparison-rerun-btn');
        expect(btn).toHaveTextContent(/re-run comparison/i);
        await userEvent.click(btn);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/4/auto-rag/comparison/run',
            );
        });
    });

    it('surfaces a warning toast on 409 (comparison already running)', async () => {
        apiMock.get.mockRejectedValueOnce({ response: { status: 404 } });
        apiMock.post.mockRejectedValueOnce({
            response: {
                status: 409,
                data: {
                    detail: {
                        error_code: 'AUTO_RAG_COMPARISON_ALREADY_RUNNING',
                        message:
                            'An auto-RAG comparison Job for project 4 is already in flight.',
                        metadata: {
                            existing_job_id: 7,
                        },
                    },
                },
            },
        });
        render(<AutoRagComparisonPanel projectId={4} />);
        const btn = await screen.findByTestId('auto-rag-comparison-run-btn');
        await userEvent.click(btn);
        await waitFor(() => {
            expect(toastMock.warning).toHaveBeenCalledWith(
                expect.stringContaining('already in flight'),
                4000,
            );
        });
        // No info toast / no refresh: this isn't a successful start.
        expect(toastMock.info).not.toHaveBeenCalled();
        expect(refreshSpy).not.toHaveBeenCalled();
    });

    it('renders the shared "pick a recipe first" CTA on RECIPE_REQUIRED 400', async () => {
        // Legacy NULL-recipe project: backend now returns a
        // structured 400 with error_code=RECIPE_REQUIRED. Panel must
        // mount the shared NoRecipeEmptyState (instead of silently
        // hiding like it does for "recipe set but not RAG-eligible").
        apiMock.get.mockRejectedValueOnce({
            response: {
                status: 400,
                data: {
                    detail: {
                        error_code: 'RECIPE_REQUIRED',
                        message:
                            'Project has no selected recipe — auto-RAG '
                            + 'comparison needs the recipe.',
                    },
                },
            },
        });
        render(<AutoRagComparisonPanel projectId={4} />);
        const cta = await screen.findByTestId(
            'auto-rag-comparison-recipe-required',
        );
        expect(cta.textContent).toMatch(/Pick a recipe first/);
        const link = cta.querySelector('a') as HTMLAnchorElement;
        expect(link.getAttribute('href')).toBe('/project/4/recipes');
    });

    it('still silently hides on non-RECIPE_REQUIRED 400 (recipe ineligible)', async () => {
        // Recipe IS set but isn't RAG-eligible (e.g. classification).
        // Backend returns a plain-string detail; panel keeps the
        // legacy silent-hide behavior — the user has explicit signal
        // elsewhere (the project's selected recipe isn't QA).
        apiMock.get.mockRejectedValueOnce({
            response: {
                status: 400,
                data: { detail: "Recipe 'classification' has no auto-RAG corpus shape yet." },
            },
        });
        const { container } = render(<AutoRagComparisonPanel projectId={4} />);
        await waitFor(() => {
            // Loading state has cleared.
            expect(
                screen.queryByTestId('auto-rag-comparison-loading'),
            ).not.toBeInTheDocument();
        });
        // No CTA, no panel, no error banner.
        expect(
            screen.queryByTestId('auto-rag-comparison-recipe-required'),
        ).not.toBeInTheDocument();
        expect(
            screen.queryByTestId('auto-rag-comparison-error'),
        ).not.toBeInTheDocument();
        expect(container.firstChild).toBeNull();
    });
});
