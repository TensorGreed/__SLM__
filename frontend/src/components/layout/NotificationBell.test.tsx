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

import NotificationBell, { cloudSpendText, isPausedJob, jobOutcomeSummary } from './NotificationBell';
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

    it('flags a job waiting for its usage window as paused', () => {
        expect(isPausedJob(jobFixture({ progress_message: 'waiting for usage window — resumes ~14:05 UTC' }))).toBe(true);
        expect(isPausedJob(jobFixture({ progress_message: 'chunk 2/4 · 50 rows so far' }))).toBe(false);
        expect(isPausedJob(jobFixture({ status: 'failed', progress_message: 'waiting for usage window' }))).toBe(false);
    });

    it('renders a paused chip on the row', async () => {
        const data = { count: 1, jobs: [jobFixture({ id: 5, progress_message: 'waiting for usage window — resumes ~14:05 UTC' })] };
        apiMock.get.mockResolvedValue({ data });
        render(<NotificationBell />);
        await waitFor(() => expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument());
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        expect(await screen.findByTestId('notification-bell-row-5-paused')).toHaveTextContent('paused');
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

    it.each([
        ['synth_playbook', '/project/3/data?open=review-queue&step=synthetic#synth-review-queue'],
        ['synth_review_assist', '/project/3/data?open=review-queue&step=synthetic#synth-review-queue'],
        ['judge_calibration_reference', '/project/3/data?open=review-queue&step=synthetic#synth-review-queue'],
        ['training_start', '/project/3/train?open=plan'],
        ['post_training_lift_eval', '/project/3/results'],
        ['documents_qa_flow', '/project/3/train#documents-qa-flow'],
        ['seed_documents', '/project/3/data#seed-documents'],
        ['benchmark_eval', '/project/3/results?tab=compare#external-benchmarks'],
    ])('deep-links a %s job to the new workspace URL', async (kind, expected) => {
        const payload = {
            data: {
                count: 1,
                jobs: [jobFixture({ id: 8, kind, status: 'succeeded', project_id: 3, title: kind })],
            },
        };
        apiMock.get.mockResolvedValue(payload);
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        await userEvent.click(screen.getByTestId('notification-bell-row-8-open'));
        expect(locationAssignMock).toHaveBeenCalledWith(expected);
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

    it('summarises a pre-screen job as verdict counts plus spend', () => {
        const job = jobFixture({
            kind: 'synth_review_assist', status: 'succeeded',
            result: {
                rows_judged: 40, rows_unparseable: 2, verdicts: { pass: 30, fail: 8, unsure: 2 },
                tokens: 12000, cost_usd: 0.04,
            },
        });
        expect(jobOutcomeSummary(job as never)).toBe(
            'pass 30 \u00b7 fail 8 \u00b7 unsure 2 \u00b7 2 unreadable \u00b7 12k tokens (~$0.04 at list price)',
        );
        const batched = jobFixture({
            kind: 'synth_review_assist', status: 'succeeded',
            result: { verdicts: { pass: 3, fail: 0, unsure: 0 }, batch: { fell_back: 1 } },
        });
        expect(jobOutcomeSummary(batched as never)).toBe('pass 3 \u00b7 fail 0 \u00b7 unsure 0 \u00b7 via batch (half price), 1 request redone directly');
    });

    it('summarises a reference-labelling job as labels written and rows skipped', () => {
        const job = jobFixture({
            kind: 'judge_calibration_reference', status: 'succeeded',
            result: {
                rows_sampled: 20, labelled: 17, labels: { pass: 12, fail: 5 },
                skipped_unsure: 2, skipped_unparseable: 1, skipped_too_long: 0, labeller: 'ollama:big',
            },
        });
        expect(jobOutcomeSummary(job as never)).toBe(
            'labelled 17 of 20 \u00b7 pass 12 \u00b7 fail 5 \u00b7 3 skipped (unsure or unreadable) \u00b7 by ollama:big',
        );
    });

    it('adds agree / disagree counts to a pre-screen summary once two judges have run', () => {
        const job = jobFixture({
            kind: 'synth_review_assist', status: 'succeeded',
            result: {
                verdicts: { pass: 3, fail: 1, unsure: 0 },
                consensus: { none: 0, single: 0, agree_pass: 2, agree_fail: 1, disagree: 1 },
            },
        });
        expect(jobOutcomeSummary(job as never)).toBe('pass 3 \u00b7 fail 1 \u00b7 unsure 0 \u00b7 agree 3 \u00b7 disagree 1');
        const single = jobFixture({
            kind: 'synth_review_assist', status: 'succeeded',
            result: { verdicts: { pass: 4, fail: 0, unsure: 0 }, consensus: { none: 0, single: 4, agree_pass: 0, agree_fail: 0, disagree: 0 } },
        });
        expect(jobOutcomeSummary(single as never)).toBe('pass 4 \u00b7 fail 0 \u00b7 unsure 0');
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

    it('says "via batch" for a span relabel that ran as a Message Batch', async () => {
        const job = jobFixture({
            id: 18,
            kind: 'span_relabel',
            status: 'succeeded',
            title: 'Training labels by cloud:anthropic:haiku · project #4 · batch',
            result: {
                split: 'synthetic', labelled_rows: 40, entities: 120, tokens: 52000, cost_usd: 0.04,
                via_batch: true, batch: { requests: 7, succeeded: 6, fell_back: 1 },
            },
        });
        apiMock.get.mockResolvedValue({ data: { count: 1, jobs: [job] } });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        const summary = await screen.findByTestId('notification-bell-row-18-summary');
        expect(summary).toHaveTextContent('training labels: 40 rows relabelled');
        expect(summary).toHaveTextContent('via batch (half price), 1 request redone directly');
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
        expect(summary).toHaveTextContent(/^run #15 · off F1/);
    });

    it('auto_rag_comparison summary names the scored run and its split', async () => {
        const jobs = [
            jobFixture({
                id: 63, kind: 'auto_rag_comparison', status: 'succeeded', title: 'Auto-RAG comparison · fine-tuned · run #160 · project #38',
                result: { project_id: 38, model: 'fine_tuned', experiment_id: 160, split: 'test', off_mean_f1: 0.15, on_mean_f1: 0.34, n_val_rows: 157 },
            }),
        ];
        apiMock.get.mockResolvedValue({ data: { count: jobs.length, jobs } });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));
        expect(screen.getByTestId('notification-bell-row-63-summary')).toHaveTextContent(
            'run #160 · off F1 0.15 → on F1 0.34 · 157 test rows',
        );
    });

    it('auto_rag_comparison summary qualifies the lift with its row evidence', async () => {
        const result = { project_id: 18, off_mean_f1: 0.1056, on_mean_f1: 0.1363, relative_lift_pct: 29.08, n_val_rows: 21 };
        const jobs = [
            jobFixture({
                id: 61, kind: 'auto_rag_comparison', status: 'succeeded', title: 'Auto-RAG comparison · base model · project #18',
                result: { ...result, model: 'base', evidence: { verdict: 'within_noise', n: 21, better: 14, worse: 6, same: 1 } },
            }),
            jobFixture({
                id: 62, kind: 'auto_rag_comparison', status: 'succeeded', title: 'Auto-RAG comparison · project #18',
                result: { ...result, model: 'fine_tuned', evidence: { verdict: 'better', n: 21, better: 14, worse: 7, same: 0 } },
            }),
        ];
        apiMock.get.mockResolvedValue({ data: { count: jobs.length, jobs } });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));

        expect(screen.getByTestId('notification-bell-row-61-summary')).toHaveTextContent(
            'base model · off F1 0.11 → on F1 0.14 · +29.08% lift (within noise · 14 rows better, 6 worse) · 21 val rows',
        );
        expect(screen.getByTestId('notification-bell-row-62-summary')).toHaveTextContent(
            '+29.08% lift (beyond row noise · 14 rows better, 7 worse)',
        );
    });

    it('lift-check summary for a seed group says whether the seeds agree', async () => {
        const headline = { metric_id: 'f1', baseline_value: 0.074, trained_value: 0.187, direction: 'improved', n_seeds: 3 };
        const jobs = [
            jobFixture({
                id: 71, kind: 'post_training_lift_eval', status: 'succeeded', title: 'Did fine-tuning help? · run #40 (3 seeds)',
                result: { headline, lift_status: 'ok', n_seeds: 3, seed_evidence: { kind: 'seeds', verdict: 'better', n: 3, std: 0.0146 } },
            }),
            jobFixture({
                id: 72, kind: 'post_training_lift_eval', status: 'succeeded', title: 'Did fine-tuning help? · run #44 (3 seeds)',
                result: { headline, lift_status: 'ok', n_seeds: 3, seed_evidence: { kind: 'seeds', verdict: 'within_noise', n: 3, std: 0.0278 } },
            }),
        ];
        apiMock.get.mockResolvedValue({ data: { count: jobs.length, jobs } });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));

        expect(screen.getByTestId('notification-bell-row-71-summary')).toHaveTextContent(
            'f1: 0.074 → 0.187 ± 0.015 across 3 seeds (holds across seeds)',
        );
        expect(screen.getByTestId('notification-bell-row-72-summary')).toHaveTextContent(
            'f1: 0.074 → 0.187 ± 0.028 across 3 seeds (seeds disagree)',
        );
    });

    it('lift-check summary does not claim "better than base" when the change is within noise', async () => {
        const headline = { metric_id: 'f1', baseline_value: 0.106, trained_value: 0.136, direction: 'improved' };
        const jobs = [
            jobFixture({
                id: 51, kind: 'post_training_lift_eval', status: 'succeeded',
                title: 'Did fine-tuning help? · run #26',
                result: { headline, lift_status: 'ok', evidence: { verdict: 'within_noise', n: 21, better: 14, worse: 6, same: 1 } },
            }),
            jobFixture({
                id: 52, kind: 'post_training_lift_eval', status: 'succeeded',
                title: 'Did fine-tuning help? · run #27',
                result: { headline, lift_status: 'ok', evidence: { verdict: 'better', n: 21, better: 17, worse: 4, same: 0 } },
            }),
            jobFixture({
                id: 53, kind: 'post_training_lift_eval', status: 'succeeded',
                title: 'Did fine-tuning help? · run #28',
                result: { headline, lift_status: 'ok', evidence: { verdict: 'too_few_rows', n: 2, better: 2, worse: 0, same: 0 } },
            }),
            // Recorded before evidence existed: falls back to the bare direction.
            jobFixture({
                id: 54, kind: 'post_training_lift_eval', status: 'succeeded',
                title: 'Did fine-tuning help? · run #20',
                result: { headline, lift_status: 'ok' },
            }),
        ];
        apiMock.get.mockResolvedValue({ data: { count: jobs.length, jobs } });
        render(<NotificationBell />);
        await waitFor(() => {
            expect(screen.getByTestId('notification-bell-button')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('notification-bell-button'));

        const noise = screen.getByTestId('notification-bell-row-51-summary');
        expect(noise).toHaveTextContent('f1: 0.106 → 0.136 (within noise · 14 rows better, 6 worse)');
        expect(noise).not.toHaveTextContent('better than base');
        expect(screen.getByTestId('notification-bell-row-52-summary')).toHaveTextContent(
            '(better than base · 17 rows better, 4 worse)',
        );
        expect(screen.getByTestId('notification-bell-row-53-summary')).toHaveTextContent(
            '(too few rows to tell · 2 rows better, 0 worse)',
        );
        expect(screen.getByTestId('notification-bell-row-54-summary')).toHaveTextContent(
            'f1: 0.106 → 0.136 (better than base)',
        );
    });

    it('a span evaluation reports its own scores, answer key first', async () => {
        const jobs = [
            jobFixture({
                id: 61, kind: 'post_training_lift_eval', status: 'succeeded', title: 'Did fine-tuning help? · run #59',
                result: { kind: 'span_eval', experiment_id: 59, answer_key: { f1: 0.8396 }, test: { f1: 0.8931 } },
            }),
        ];
        apiMock.get.mockResolvedValue({ data: { count: jobs.length, jobs } });
        render(<NotificationBell />);
        await userEvent.click(await screen.findByTestId('notification-bell-button'));
        expect(screen.getByTestId('notification-bell-row-61-summary')).toHaveTextContent(
            'span F1 0.84 on the answer key · span F1 0.89 on the test examples',
        );
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


describe('NotificationBell seed pool line', () => {
    it('shows how the seed passages were drawn, by source and language', () => {
        const job = jobFixture({
            kind: 'synth_playbook', status: 'succeeded',
            result: {
                rows_generated: 6,
                seed_pool: {
                    chunks_eligible: 17798, chunks_used: 2000, sampled: true,
                    by_source: { '3': 1200, '5': 800 }, by_language: { java: 1200, yaml: 800 },
                },
            },
        });
        const text = jobOutcomeSummary(job as never) ?? '';
        expect(text).toContain('seeds: 2000 of 17798 chunks (sampled) \u00b7 sources #3 1200, #5 800 \u00b7 java 1200, yaml 800');
        const small = jobFixture({
            kind: 'synth_playbook', status: 'succeeded',
            result: { rows_generated: 1, seed_pool: { chunks_eligible: 5, chunks_used: 5, sampled: false, by_source: {}, by_language: {} } },
        });
        expect(jobOutcomeSummary(small as never)).toContain('seeds: 5 of 5 chunks');
        expect(jobOutcomeSummary(small as never)).not.toContain('(sampled)');
        const filtered = jobFixture({
            kind: 'synth_playbook', status: 'succeeded',
            result: {
                rows_generated: 1,
                seed_pool: {
                    chunks_eligible: 7, chunks_used: 7, sampled: false, by_source: { '3': 7 }, by_language: {},
                    sources: {
                        '3': { total: 20, kept: 7, dropped_glob: 9, dropped_relevance: 4, exempt: 0 },
                        '5': { total: 3, kept: 0, dropped_glob: 3, dropped_relevance: 0, exempt: 0 },
                    },
                },
            },
        });
        expect(jobOutcomeSummary(filtered as never)).toContain('filters dropped 12 by path, 4 by relevance');
    });
});

describe('NotificationBell cloud spend line', () => {
    it('appends tokens and the list-price estimate to any kind', () => {
        const seed = jobFixture({
            kind: 'seed_documents', status: 'succeeded',
            result: { written: 4, tokens: 12_400, cost_usd: 0.031 },
        });
        expect(jobOutcomeSummary(seed as never)).toContain('12k tokens (~$0.03 at list price)');
        const unknownKind = jobFixture({ kind: 'some_new_kind', status: 'succeeded', result: { tokens: 800 } });
        expect(jobOutcomeSummary(unknownKind as never)).toBe('800 tokens');
    });

    it('a batched playbook run says it went through a Message Batch and how its cost was priced', () => {
        const job = jobFixture({
            kind: 'synth_playbook', status: 'succeeded',
            result: {
                rows_generated: 20, backend_used: 'cloud:anthropic:m', elapsed_sec: 600,
                tokens: 5000, cost_usd: 0.02, batch: { requests: 2, succeeded: 1, fell_back: 1 },
            },
        });
        const text = jobOutcomeSummary(job as never) ?? '';
        expect(text).toContain('via batch (half price), 1 request redone directly');
        expect(text).toContain('5k tokens (~$0.02 with Message Batch pricing)');
        expect(text).not.toContain('at list price');
    });

    it('says what a failed job spent and stays quiet without usage', () => {
        const failed = jobFixture({ kind: 'documents_qa_flow', status: 'failed', result: { tokens: 3000, cost_usd: 0.01 } });
        expect(jobOutcomeSummary(failed as never)).toBe('spent 3k tokens (~$0.01 at list price)');
    });

    it('document-source jobs count every rejection (rejection_counts), not the capped list', () => {
        const job = {
            id: 1, kind: 'document_source_git_repo', status: 'succeeded', project_id: 3,
            result: { documents_created: 4, rejected: [{ path: 'a', reason: 'binary' }], rejection_counts: { binary: 700, too_large: 12 } },
        };
        expect(jobOutcomeSummary(job as never)).toBe('4 documents imported, 712 rejected');
        const noCounts = { ...job, result: { documents_created: 1, rejected: [{ path: 'a', reason: 'binary' }] } };
        expect(jobOutcomeSummary(noCounts as never)).toBe('1 documents imported, 1 rejected');
        expect(cloudSpendText({ rows: 3 })).toBeNull();
        expect(cloudSpendText({ tokens: 0 })).toBeNull();
    });
});

describe('NotificationBell passage-trained lift check', () => {
    it('names both with-passages arms and the row evidence', () => {
        const job = jobFixture({
            kind: 'post_training_lift_eval', status: 'succeeded',
            result: {
                kind: 'passage_reader_eval',
                base_with_passages: { score: 0.585 }, finetune_with_passages: { score: 0.7 },
                evidence: { verdict: 'within_noise', better: 6, worse: 3 },
            },
        });
        expect(jobOutcomeSummary(job as never)).toBe('judge score with passages: base 0.585 → fine-tuned 0.700 (within noise · 6 better, 3 worse)');
    });
});


describe('NotificationBell synth rejects', () => {
    it('mentions the rejected count only when there is one', () => {
        const withRejects = jobFixture({
            kind: 'synth_playbook', status: 'succeeded',
            result: { rows_generated: 12, backend_used: 'b', rejected_count: 5 },
        });
        expect(jobOutcomeSummary(withRejects as never)).toContain('12 rows generated');
        expect(jobOutcomeSummary(withRejects as never)).toContain('5 rejected');
        const none = jobFixture({
            kind: 'synth_playbook', status: 'succeeded',
            result: { rows_generated: 12, rejected_count: 0 },
        });
        expect(jobOutcomeSummary(none as never)).not.toContain('rejected');
    });
});

describe('NotificationBell tool catalog import', () => {
    it('summarises the imported tool count with the catalog version and source', () => {
        const job = jobFixture({ id: 90, kind: 'tool_catalog_import', status: 'succeeded', project_id: 4, result: { catalog: { tools: [{ name: 'a' }, { name: 'b' }], version: 'a1b2c3d4e5f6', source: { kind: 'mcp_http' } } } });
        expect(jobOutcomeSummary(job as never)).toBe('2 tools imported from mcp_http (catalog version a1b2c3d4e5f6)');
        const bare = jobFixture({ id: 91, kind: 'tool_catalog_import', status: 'succeeded', project_id: 4, result: {} });
        expect(jobOutcomeSummary(bare as never)).toBeNull();
    });
});

describe('NotificationBell pipeline_eval', () => {
    const result = { run_id: 'r-1', pipeline_id: 'pl-ab12cd34', rows: 20, share_fully_correct: 0.65, attribution: { first_wrong_stage: { extract: 2, rewrite: 5 }, correct: 13 } };

    it('summarises rows, share fully correct and the top first-wrong stage', () => {
        const job = jobFixture({ id: 70, kind: 'pipeline_eval', status: 'succeeded', project_id: null, result });
        expect(jobOutcomeSummary(job as never)).toBe('20 rows · 65% fully correct · first-wrong: rewrite');
    });

    it('deep-links a project-less run to the pipeline results', async () => {
        apiMock.get.mockReset();
        locationAssignMock.mockReset();
        resetStore();
        const job = { ...jobFixture({ id: 71, kind: 'pipeline_eval', status: 'succeeded', project_id: null, result }), params: { pipeline_id: 'pl-ab12cd34', run_id: 'r-1' } };
        apiMock.get.mockResolvedValue({ data: { count: 1, jobs: [job] } });
        render(<NotificationBell />);
        await userEvent.click(await screen.findByTestId('notification-bell-button'));
        await userEvent.click(await screen.findByTestId('notification-bell-row-71-open'));
        expect(locationAssignMock).toHaveBeenCalledWith('/pipelines/pl-ab12cd34?run=r-1');
    });
});
