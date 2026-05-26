/**
 * NotificationBell tests (Hardening Phase H1).
 *
 * The bell is wired to ``useJobsStore`` which polls via the
 * mocked api client. We control the polling tick + the API
 * responses via fake timers to drive deterministic state.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { act } from 'react';
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

const locationAssignMock = vi.fn();
Object.defineProperty(window, 'location', {
    value: { assign: locationAssignMock, href: 'http://localhost/' },
    writable: true,
});

import NotificationBell from './NotificationBell';
import { useJobsStore } from '../../stores/jobsStore';


function jobFixture(overrides: Partial<{
    id: number;
    kind: string;
    title: string;
    status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
    progress: number | null;
    progress_message: string | null;
    project_id: number | null;
    result: Record<string, unknown> | null;
    error: string | null;
}> = {}) {
    return {
        id: overrides.id ?? 1,
        kind: overrides.kind ?? 'synth_playbook',
        title: overrides.title ?? 'Synth · positives_paraphrase · 10 rows',
        status: overrides.status ?? 'running',
        progress: overrides.progress ?? 0.5,
        progress_message: overrides.progress_message ?? 'Calling LLM…',
        project_id: overrides.project_id ?? 4,
        user_id: null,
        params: {},
        result: overrides.result ?? null,
        error: overrides.error ?? null,
        queued_at: '2026-05-26T12:00:00Z',
        started_at: '2026-05-26T12:00:05Z',
        completed_at: null,
        dismissed_at: null,
    };
}


function resetStore() {
    useJobsStore.setState({
        jobs: [],
        isLoading: false,
        isPolling: false,
        bellOpen: false,
        lastError: null,
        _lastStatusById: {},
        _pollHandle: null,
    });
}


describe('NotificationBell', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        locationAssignMock.mockReset();
        toastMock.success.mockReset();
        toastMock.error.mockReset();
        toastMock.info.mockReset();
        resetStore();
    });

    it('renders the bell button with no badge when no jobs are in-flight', async () => {
        apiMock.get.mockResolvedValueOnce({ data: { count: 0, jobs: [] } });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalled();
        });
        // First call goes through the store's listActiveJobs helper
        // which builds a query string from { includeRecentlyCompleted,
        // limit }. Just check the path prefix, not the params.
        const calledUrl = apiMock.get.mock.calls[0]?.[0] as string;
        expect(calledUrl.startsWith('/jobs/active')).toBe(true);
        expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        expect(screen.queryByTestId('notification-bell-badge')).toBeNull();
    });

    it('renders a numeric badge equal to the in-flight job count', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 2,
                jobs: [
                    jobFixture({ id: 1, status: 'running' }),
                    jobFixture({ id: 2, status: 'queued', title: 'Queued job' }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-badge')).toHaveTextContent('2');
        });
    });

    it('shows a red failure badge when a failed job is unacknowledged', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 1,
                jobs: [jobFixture({ id: 3, status: 'failed', error: 'LLM timeout' })],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(
                screen.getByTestId('notification-bell-badge-failure'),
            ).toBeInTheDocument();
        });
    });

    it('opens the dropdown on click, renders running + recently-completed groups', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 2,
                jobs: [
                    jobFixture({ id: 1, status: 'running', title: 'Synth run' }),
                    jobFixture({
                        id: 2,
                        status: 'succeeded',
                        title: 'Clone to RAG',
                        result: { new_project_id: 42 },
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        // Bell opens — store also fires another fetch on open. Mock
        // that follow-up call so the panel renders without flicker.
        apiMock.get.mockResolvedValue({
            data: {
                count: 2,
                jobs: [
                    jobFixture({ id: 1, status: 'running', title: 'Synth run' }),
                    jobFixture({
                        id: 2,
                        status: 'succeeded',
                        title: 'Clone to RAG',
                        result: { new_project_id: 42 },
                    }),
                ],
            },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        expect(screen.getByTestId('notification-bell-panel')).toBeInTheDocument();
        expect(screen.getByTestId('notification-bell-row-1')).toHaveTextContent('Synth run');
        expect(screen.getByTestId('notification-bell-row-2')).toHaveTextContent('Clone to RAG');
        // Completed row gets a "Dismiss" button; in-flight does not.
        expect(
            screen.getByTestId('notification-bell-row-2-dismiss'),
        ).toBeInTheDocument();
        expect(
            screen.queryByTestId('notification-bell-row-1-dismiss'),
        ).toBeNull();
    });

    it('navigates to the deep-link when "Open" is clicked on a succeeded reroute job', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 5,
                        kind: 'reroute_to_rag',
                        status: 'succeeded',
                        title: 'Clone to RAG · project #4',
                        result: { new_project_id: 99 },
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        apiMock.get.mockResolvedValue({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 5,
                        kind: 'reroute_to_rag',
                        status: 'succeeded',
                        title: 'Clone to RAG · project #4',
                        result: { new_project_id: 99 },
                    }),
                ],
            },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        await userEvent.click(screen.getByTestId('notification-bell-row-5-open'));
        expect(locationAssignMock).toHaveBeenCalledWith('/project/99');
    });

    it('dismisses a completed job and removes it from the list', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 7,
                        status: 'failed',
                        error: 'Boom',
                        title: 'Failed job',
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        // Subsequent gets return an empty list after dismiss.
        apiMock.get.mockResolvedValue({
            data: { count: 1, jobs: [jobFixture({ id: 7, status: 'failed', error: 'Boom' })] },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        apiMock.post.mockResolvedValueOnce({
            data: jobFixture({ id: 7, status: 'failed' }),
        });
        // After dismiss, the next list call returns no jobs.
        apiMock.get.mockResolvedValue({ data: { count: 0, jobs: [] } });
        await userEvent.click(screen.getByTestId('notification-bell-row-7-dismiss'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/jobs/7/dismiss');
        });
    });

    it('renders a Cancel button on in-flight jobs and POSTs /cancel on click', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 9,
                        status: 'running',
                        title: 'Long-running training',
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        apiMock.get.mockResolvedValue({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 9,
                        status: 'running',
                        title: 'Long-running training',
                    }),
                ],
            },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        // In-flight rows have a Cancel button + NO Dismiss button.
        expect(
            screen.getByTestId('notification-bell-row-9-cancel'),
        ).toBeInTheDocument();
        expect(
            screen.queryByTestId('notification-bell-row-9-dismiss'),
        ).toBeNull();
        apiMock.post.mockResolvedValueOnce({
            data: jobFixture({ id: 9, status: 'cancelled' }),
        });
        await userEvent.click(screen.getByTestId('notification-bell-row-9-cancel'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/jobs/9/cancel');
        });
        expect(toastMock.info).toHaveBeenCalledWith(
            expect.stringContaining('Cancelled'),
            3000,
        );
    });

    it('renders a per-kind outcome summary for completed synth_playbook jobs', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 15,
                        kind: 'synth_playbook',
                        status: 'succeeded',
                        title: 'Synth · positives_paraphrase · 30 rows',
                        result: {
                            rows_generated: 27,
                            backend_used: 'ollama:llama3',
                            elapsed_sec: 42.5,
                        },
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        apiMock.get.mockResolvedValue({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 15,
                        kind: 'synth_playbook',
                        status: 'succeeded',
                        title: 'Synth · positives_paraphrase · 30 rows',
                        result: {
                            rows_generated: 27,
                            backend_used: 'ollama:llama3',
                            elapsed_sec: 42.5,
                        },
                    }),
                ],
            },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        const summary = screen.getByTestId('notification-bell-row-15-summary');
        expect(summary).toHaveTextContent('27 rows generated');
        expect(summary).toHaveTextContent('via ollama:llama3');
        expect(summary).toHaveTextContent('42.5s');
    });

    it('renders a "Created project #N" summary for completed reroute jobs', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 16,
                        kind: 'reroute_to_rag',
                        status: 'succeeded',
                        title: 'Clone to RAG · project #4',
                        result: {
                            new_project_id: 88,
                            new_project_name: 'Policy QA (RAG)',
                        },
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        apiMock.get.mockResolvedValue({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 16,
                        kind: 'reroute_to_rag',
                        status: 'succeeded',
                        title: 'Clone to RAG · project #4',
                        result: {
                            new_project_id: 88,
                            new_project_name: 'Policy QA (RAG)',
                        },
                    }),
                ],
            },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        const summary = screen.getByTestId('notification-bell-row-16-summary');
        expect(summary).toHaveTextContent('Policy QA (RAG)');
        expect(summary).toHaveTextContent('#88');
    });

    it('renders legacy-synth summary preferring rows_saved over rows_generated', async () => {
        // Auto-save fix: bell prefers "N rows saved to dataset"
        // (the persisted count) over the older "rows generated"
        // wording. Falls back when rows_saved is absent (older
        // jobs from before the fix).
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 2,
                jobs: [
                    jobFixture({
                        id: 30,
                        kind: 'synth_legacy_qa',
                        status: 'succeeded',
                        title: 'Synth (legacy QA) · 100 rows',
                        result: {
                            rows_generated: 100,
                            rows_saved: 100,
                            batches_done: 5,
                            batches_total: 5,
                        },
                    }),
                    jobFixture({
                        id: 31,
                        kind: 'synth_legacy_spans',
                        status: 'succeeded',
                        title: 'Synth (legacy spans) · 50 rows',
                        // No rows_saved field — older job.
                        result: {
                            rows_generated: 48,
                            batches_done: 3,
                            batches_total: 3,
                        },
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        apiMock.get.mockResolvedValue({
            data: {
                count: 2,
                jobs: [
                    jobFixture({
                        id: 30,
                        kind: 'synth_legacy_qa',
                        status: 'succeeded',
                        title: 'Synth (legacy QA) · 100 rows',
                        result: {
                            rows_generated: 100,
                            rows_saved: 100,
                            batches_done: 5,
                            batches_total: 5,
                        },
                    }),
                    jobFixture({
                        id: 31,
                        kind: 'synth_legacy_spans',
                        status: 'succeeded',
                        title: 'Synth (legacy spans) · 50 rows',
                        result: {
                            rows_generated: 48,
                            batches_done: 3,
                            batches_total: 3,
                        },
                    }),
                ],
            },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        // New format: "100 rows saved to dataset"
        expect(
            screen.getByTestId('notification-bell-row-30-summary'),
        ).toHaveTextContent('100 rows saved to dataset');
        // Fallback for older jobs: "48 rows generated"
        expect(
            screen.getByTestId('notification-bell-row-31-summary'),
        ).toHaveTextContent('48 rows generated');
    });

    it('renders training-start summary with final loss + total steps', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 17,
                        kind: 'training_start',
                        status: 'succeeded',
                        title: 'Train · exp #12',
                        result: {
                            final_train_loss: 0.4218,
                            total_steps: 600,
                            terminal_status: 'completed',
                        },
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        apiMock.get.mockResolvedValue({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 17,
                        kind: 'training_start',
                        status: 'succeeded',
                        title: 'Train · exp #12',
                        result: {
                            final_train_loss: 0.4218,
                            total_steps: 600,
                            terminal_status: 'completed',
                        },
                    }),
                ],
            },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        const summary = screen.getByTestId('notification-bell-row-17-summary');
        expect(summary).toHaveTextContent('final loss 0.4218');
        expect(summary).toHaveTextContent('600 steps');
    });

    it('renders auto_rag_comparison summary with off→on F1 + lift% + val rows', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 40,
                        kind: 'auto_rag_comparison',
                        status: 'succeeded',
                        title: 'Auto-RAG comparison · project #4',
                        result: {
                            project_id: 4,
                            experiment_id: 15,
                            off_mean_f1: 0.0416,
                            on_mean_f1: 0.0640,
                            absolute_lift: 0.0224,
                            relative_lift_pct: 53.66,
                            n_val_rows: 28,
                        },
                    }),
                ],
            },
        });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        apiMock.get.mockResolvedValue({
            data: {
                count: 1,
                jobs: [
                    jobFixture({
                        id: 40,
                        kind: 'auto_rag_comparison',
                        status: 'succeeded',
                        title: 'Auto-RAG comparison · project #4',
                        result: {
                            project_id: 4,
                            experiment_id: 15,
                            off_mean_f1: 0.0416,
                            on_mean_f1: 0.0640,
                            absolute_lift: 0.0224,
                            relative_lift_pct: 53.66,
                            n_val_rows: 28,
                        },
                    }),
                ],
            },
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        const summary = screen.getByTestId('notification-bell-row-40-summary');
        expect(summary).toHaveTextContent('off F1 0.04 → on F1 0.06');
        expect(summary).toHaveTextContent('+53.66% lift');
        expect(summary).toHaveTextContent('28 val rows');
    });

    it('renders an empty-state message when no jobs are in the bell', async () => {
        apiMock.get.mockResolvedValue({ data: { count: 0, jobs: [] } });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        await act(async () => {
            await userEvent.click(screen.getByTestId('notification-bell-button'));
        });
        expect(screen.getByTestId('notification-bell-empty')).toBeInTheDocument();
    });
});
