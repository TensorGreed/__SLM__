import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import ExperimentLiveSignals from './ExperimentLiveSignals';
import type { Job } from '../../api/jobs';
import { useJobsStore } from '../../stores/jobsStore';


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
        id: 100,
        kind: 'training_start',
        title: 'train #100',
        status: 'running',
        progress: 0.5,
        progress_message: null,
        project_id: 17,
        user_id: null,
        params: { experiment_id: 20 },
        result: null,
        error: null,
        queued_at: '2026-06-02T00:00:00Z',
        started_at: '2026-06-02T00:00:00Z',
        completed_at: null,
        dismissed_at: null,
        metrics_recent: [
            { step: 100, train_loss: 0.5 },
            { step: 200, train_loss: 0.4 },
            { step: 300, train_loss: 0.3 },
            { step: 400, train_loss: 0.2 },
        ],
        ...overrides,
    };
}


function _resetStore() {
    useJobsStore.setState({
        jobs: [],
        isLoading: false,
        isPolling: false,
        bellOpen: false,
        lastError: null,
        upTrendCountById: {},
        _lastStatusById: {},
        _pollHandle: null,
    });
}


describe('ExperimentLiveSignals', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
        apiMock.get.mockReset();
        _resetStore();
    });

    it('renders nothing when no job is passed', () => {
        // Experiment row exists but jobs poll hasn't surfaced a
        // tracking job yet (newly-created PENDING experiments
        // that haven't been started, or completed runs no
        // longer in the recently-completed window). Row stays
        // clean.
        const { container } = render(
            <ExperimentLiveSignals job={undefined} />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders nothing for a non-training_start job', () => {
        // Defensive: caller's job-by-experiment-id lookup
        // shouldn't surface non-training jobs (it filters), but
        // if a future refactor relaxes that filter, the component
        // refuses to render for foreign job kinds.
        const { container } = render(
            <ExperimentLiveSignals job={_trainingJob({ kind: 'synth_playbook' })} />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders nothing for a succeeded job', () => {
        // The experiment's status badge already carries the
        // terminal state. Live signals on a finished run would
        // mislead — show nothing.
        const { container } = render(
            <ExperimentLiveSignals job={_trainingJob({ status: 'succeeded' })} />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders nothing for a failed job', () => {
        const { container } = render(
            <ExperimentLiveSignals job={_trainingJob({ status: 'failed' })} />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders the sparkline for an in-flight job with metrics', () => {
        render(<ExperimentLiveSignals job={_trainingJob()} />);
        // The sparkline component carries its own test-id; we
        // just verify it landed inside our wrapper.
        const sparkline = screen.getByTestId('training-loss-sparkline');
        expect(sparkline).toBeInTheDocument();
        // Trend should resolve to ``down`` for the fixture
        // (0.5 → 0.4 → 0.3 → 0.2). Verifies the points were
        // actually passed through, not dropped.
        expect(sparkline).toHaveAttribute('data-trend', 'down');
    });

    it('still renders for a queued job (no metrics yet)', () => {
        // Queued state = job picked up by the runner but the
        // first checkpoint hasn't been written. The sparkline
        // shows the empty/dashed placeholder.
        render(
            <ExperimentLiveSignals
                job={_trainingJob({ status: 'queued', metrics_recent: [] })}
            />,
        );
        const sparkline = screen.getByTestId('training-loss-sparkline');
        expect(sparkline).toHaveAttribute('data-trend', 'empty');
    });

    it('omits the sparkline when metrics_recent is undefined', () => {
        // Pre-first-checkpoint OR backend returned no
        // metrics_recent (legacy server). The kill switch still
        // mounts (it's keyed on the trend counter, separate from
        // the metrics array), but the sparkline doesn't.
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { metrics_recent, ...rest } = _trainingJob();
        render(
            <ExperimentLiveSignals job={rest as Job} />,
        );
        expect(screen.queryByTestId('training-loss-sparkline')).toBeNull();
    });

    it('surfaces the kill switch when sustained divergence is reached', () => {
        // Seed the trend counter above threshold so the kill
        // switch surfaces. The threshold is 3 — bell tests pin
        // this contract; here we just verify the dashboard row
        // inherits it.
        useJobsStore.setState({
            jobs: [],
            isLoading: false,
            isPolling: false,
            bellOpen: false,
            lastError: null,
            upTrendCountById: { 100: 3 },
            _lastStatusById: {},
            _pollHandle: null,
        });
        render(<ExperimentLiveSignals job={_trainingJob({ id: 100 })} />);
        expect(
            screen.getByTestId('notification-bell-row-100-kill-switch'),
        ).toBeInTheDocument();
    });

    it('does not surface the kill switch below the threshold', () => {
        // Same fixture but trend counter = 2 (below threshold).
        useJobsStore.setState({
            jobs: [],
            isLoading: false,
            isPolling: false,
            bellOpen: false,
            lastError: null,
            upTrendCountById: { 100: 2 },
            _lastStatusById: {},
            _pollHandle: null,
        });
        render(<ExperimentLiveSignals job={_trainingJob({ id: 100 })} />);
        expect(
            screen.queryByTestId('notification-bell-row-100-kill-switch'),
        ).toBeNull();
    });

    it('tags the wrapper with the experiment id for selector targeting', () => {
        // Dashboard sometimes wants to query the live signals for
        // a specific experiment (e.g., scroll-into-view after a
        // clone-and-retry). The wrapper carries
        // ``experiment-live-signals-<expId>`` test-id so callers
        // can find the right row.
        render(
            <ExperimentLiveSignals
                job={_trainingJob({ params: { experiment_id: 42 } })}
            />,
        );
        expect(
            screen.getByTestId('experiment-live-signals-42'),
        ).toBeInTheDocument();
    });
});
