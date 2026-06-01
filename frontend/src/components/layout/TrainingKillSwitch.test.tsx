import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import TrainingKillSwitch from './TrainingKillSwitch';
import type { Job } from '../../api/jobs';
import { useJobsStore } from '../../stores/jobsStore';
import { useToastStore } from '../../stores/toastStore';


const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));


function _trainingJob(overrides: Partial<Job> = {}): Job {
    return {
        id: 42,
        kind: 'training_start',
        title: 'train #42',
        status: 'running',
        progress: 0.5,
        progress_message: null,
        project_id: 17,
        user_id: null,
        params: { experiment_id: 100 },
        result: null,
        error: null,
        queued_at: '2026-06-01T00:00:00Z',
        started_at: '2026-06-01T00:00:00Z',
        completed_at: null,
        dismissed_at: null,
        metrics_recent: [],
        ...overrides,
    };
}


function _seedCounter(jobId: number, count: number) {
    useJobsStore.setState({
        upTrendCountById: { [jobId]: count },
        // Stub other fields the store needs to be valid.
        jobs: [],
        isLoading: false,
        isPolling: false,
        bellOpen: false,
        lastError: null,
        _lastStatusById: {},
        _pollHandle: null,
    });
}


describe('TrainingKillSwitch', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
        apiMock.get.mockReset();
    });

    it('renders nothing when up-trend count is below threshold', () => {
        // The load-bearing guard: the kill switch must not surface
        // prematurely. 2 polls of divergence is below the 3-poll
        // threshold; the bell row stays clean.
        _seedCounter(42, 2);
        const { container } = render(
            <TrainingKillSwitch job={_trainingJob()} />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders the kill button when up-trend count meets threshold', () => {
        _seedCounter(42, 3);
        render(<TrainingKillSwitch job={_trainingJob()} />);
        const btn = screen.getByTestId(
            'notification-bell-row-42-kill-switch',
        );
        expect(btn).toHaveTextContent('kill');
        // Title carries the rationale + elapsed seconds so the
        // user can decide without context-switching.
        expect(btn.getAttribute('title') || '').toContain(
            'trending up',
        );
    });

    it('opens the confirm dialog on click', async () => {
        const user = userEvent.setup();
        _seedCounter(42, 3);
        render(<TrainingKillSwitch job={_trainingJob()} />);
        await user.click(
            screen.getByTestId('notification-bell-row-42-kill-switch'),
        );
        // Three distinct actions: cancel+clone (primary danger),
        // cancel-only, keep-going (dismiss).
        expect(
            screen.getByTestId(
                'notification-bell-row-42-kill-switch-cancel-clone',
            ),
        ).toBeInTheDocument();
        expect(
            screen.getByTestId(
                'notification-bell-row-42-kill-switch-cancel-only',
            ),
        ).toBeInTheDocument();
        expect(
            screen.getByTestId(
                'notification-bell-row-42-kill-switch-dismiss',
            ),
        ).toBeInTheDocument();
    });

    it('cancel + clone fires both API calls in order', async () => {
        const user = userEvent.setup();
        _seedCounter(42, 3);
        // Stub the two endpoints the action hits.
        apiMock.post.mockImplementation(async (url: string) => {
            if (url === '/jobs/42/cancel') {
                return { data: { id: 42, status: 'cancelled' } };
            }
            if (url.endsWith('/training/experiments/100/clone')) {
                return {
                    data: { id: 999, name: 'train #42 (retry)' },
                };
            }
            return { data: {} };
        });
        // /jobs/active gets called by refreshAfterLocalChange — stub
        // empty so the action doesn't error.
        apiMock.get.mockResolvedValue({ data: { count: 0, jobs: [] } });

        render(<TrainingKillSwitch job={_trainingJob()} />);
        await user.click(
            screen.getByTestId('notification-bell-row-42-kill-switch'),
        );
        await user.click(
            screen.getByTestId(
                'notification-bell-row-42-kill-switch-cancel-clone',
            ),
        );

        // Both POSTs fired. Order matters: cancel first (always
        // stop the running job), clone second (only if user
        // opted in).
        const postUrls = apiMock.post.mock.calls.map((c) => c[0]);
        expect(postUrls).toContain('/jobs/42/cancel');
        expect(postUrls).toContain(
            '/projects/17/training/experiments/100/clone',
        );
        const cancelIdx = postUrls.indexOf('/jobs/42/cancel');
        const cloneIdx = postUrls.indexOf(
            '/projects/17/training/experiments/100/clone',
        );
        expect(cancelIdx).toBeLessThan(cloneIdx);
    });

    it('cancel-only path fires cancel but NOT clone', async () => {
        const user = userEvent.setup();
        _seedCounter(42, 3);
        apiMock.post.mockResolvedValue({ data: { id: 42, status: 'cancelled' } });
        apiMock.get.mockResolvedValue({ data: { count: 0, jobs: [] } });

        render(<TrainingKillSwitch job={_trainingJob()} />);
        await user.click(
            screen.getByTestId('notification-bell-row-42-kill-switch'),
        );
        await user.click(
            screen.getByTestId(
                'notification-bell-row-42-kill-switch-cancel-only',
            ),
        );

        const postUrls = apiMock.post.mock.calls.map((c) => c[0]);
        expect(postUrls).toContain('/jobs/42/cancel');
        expect(postUrls).not.toContain(
            '/projects/17/training/experiments/100/clone',
        );
    });

    it('keep-going dismisses the confirm without API calls', async () => {
        const user = userEvent.setup();
        _seedCounter(42, 3);
        render(<TrainingKillSwitch job={_trainingJob()} />);
        await user.click(
            screen.getByTestId('notification-bell-row-42-kill-switch'),
        );
        await user.click(
            screen.getByTestId(
                'notification-bell-row-42-kill-switch-dismiss',
            ),
        );
        // Back to the bare kill button — no APIs hit.
        expect(apiMock.post).not.toHaveBeenCalled();
        expect(
            screen.queryByTestId(
                'notification-bell-row-42-kill-switch-confirm',
            ),
        ).toBeNull();
    });

    it('renders nothing for non-training jobs', () => {
        // Even if a synth or eval job somehow ends up with an
        // up-trend counter (it shouldn't — the store filters),
        // the component must refuse to surface for it.
        _seedCounter(42, 5);
        const { container } = render(
            <TrainingKillSwitch
                job={_trainingJob({ kind: 'synth_playbook' })}
            />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders nothing for non-in-flight jobs', () => {
        // A succeeded job's counter could be stale; component
        // refuses to render so the user can't kill a terminal job.
        _seedCounter(42, 5);
        const { container } = render(
            <TrainingKillSwitch
                job={_trainingJob({ status: 'succeeded' })}
            />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('attaches a "Start retry now" action to the success toast after cancel+clone', async () => {
        // Closes the kill→clone→relaunch loop in-toast. The
        // success toast that lands post-cancel+clone must carry a
        // ToastAction whose label is "Start retry now" — clicking
        // it fires POST /start on the cloned experiment id
        // (verified in the next test).
        const user = userEvent.setup();
        _seedCounter(42, 3);
        useToastStore.setState({ toasts: [] });
        apiMock.post.mockImplementation(async (url: string) => {
            if (url === '/jobs/42/cancel') {
                return { data: { id: 42, status: 'cancelled' } };
            }
            if (url.endsWith('/training/experiments/100/clone')) {
                return {
                    data: { id: 777, name: 'train #42 (retry)' },
                };
            }
            return { data: {} };
        });
        apiMock.get.mockResolvedValue({ data: { count: 0, jobs: [] } });

        render(<TrainingKillSwitch job={_trainingJob()} />);
        await user.click(
            screen.getByTestId('notification-bell-row-42-kill-switch'),
        );
        await user.click(
            screen.getByTestId(
                'notification-bell-row-42-kill-switch-cancel-clone',
            ),
        );

        // Pull the latest toast directly from the store — easier
        // than waiting for the toast renderer to mount in
        // isolation.
        const toasts = useToastStore.getState().toasts;
        const success = toasts.find((t) => t.type === 'success');
        expect(success).toBeDefined();
        expect(success!.action).toBeDefined();
        expect(success!.action!.label).toBe('Start retry now');
        // The message names the cloned experiment so the user
        // knows what they're about to start.
        expect(success!.message).toContain('train #42 (retry)');
    });

    it('clicking the toast action POSTs to /start on the cloned experiment', async () => {
        // The other half of the action contract: clicking the
        // toast's "Start retry now" button hits the start
        // endpoint with the cloned experiment id (not the
        // original).
        const user = userEvent.setup();
        _seedCounter(42, 3);
        useToastStore.setState({ toasts: [] });
        apiMock.post.mockImplementation(async (url: string) => {
            if (url === '/jobs/42/cancel') {
                return { data: { id: 42, status: 'cancelled' } };
            }
            if (url.endsWith('/training/experiments/100/clone')) {
                return {
                    data: { id: 777, name: 'train #42 (retry)' },
                };
            }
            if (url.endsWith('/training/experiments/777/start')) {
                return { data: { status: 'running' } };
            }
            return { data: {} };
        });
        apiMock.get.mockResolvedValue({ data: { count: 0, jobs: [] } });

        render(<TrainingKillSwitch job={_trainingJob()} />);
        await user.click(
            screen.getByTestId('notification-bell-row-42-kill-switch'),
        );
        await user.click(
            screen.getByTestId(
                'notification-bell-row-42-kill-switch-cancel-clone',
            ),
        );

        // Invoke the action directly from the store — the
        // component-level test for the toast renderer covers
        // the button-click → onClick wiring; here we verify the
        // action's effect.
        const action = useToastStore
            .getState()
            .toasts.find((t) => t.type === 'success')?.action;
        expect(action).toBeDefined();
        // Reset post mock so we can isolate the start call.
        apiMock.post.mockClear();
        apiMock.post.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/experiments/777/start')) {
                return { data: { status: 'running' } };
            }
            return { data: {} };
        });
        await action!.onClick();
        const startCalls = apiMock.post.mock.calls.filter(
            (c) => typeof c[0] === 'string' && c[0].endsWith('/training/experiments/777/start'),
        );
        expect(startCalls).toHaveLength(1);
        // user not needed here; silence the lint if it complained.
        void user;
    });
});
