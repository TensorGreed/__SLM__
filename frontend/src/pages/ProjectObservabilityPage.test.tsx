import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../api/client', () => ({ default: apiMock }));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>(
        'react-router-dom',
    );
    return {
        ...actual,
        useOutletContext: () => ({
            projectId: 7,
            project: { id: 7, name: 'Demo Project' },
            pipelineStatus: null,
            refreshPipelineStatus: vi.fn(),
        }),
        useParams: () => ({ id: '7' }),
    };
});

import ProjectObservabilityPage from './ProjectObservabilityPage';

const TIMELINE_RESPONSE = {
    project_id: 7,
    window_start: null,
    window_end: null,
    total_events: 4,
    total_runs: 2,
    orphaned_count: 0,
    truncated: false,
    tree: [
        {
            run_id: 'exp-1',
            parent_run_id: null,
            is_orphan: false,
            stage: 'training',
            stages_present: ['training'],
            summary: 'Training started',
            actor: 'system',
            first_ts: '2026-05-07T10:00:00Z',
            last_ts: '2026-05-07T10:00:30Z',
            duration_seconds: 30,
            event_count: 2,
            severity_counts: { info: 2 },
            highest_severity: 'info',
            latest_reason_code: null,
            children: [],
        },
    ],
};

const CLUSTERS_RESPONSE = {
    project_id: 7,
    limit: 100,
    clusters: [
        {
            id: 1,
            project_id: 7,
            stage: 'training',
            reason_code: 'training_runtime_error',
            signature: 'abc123',
            failure_count: 4,
            first_seen_at: '2026-05-07T09:00:00Z',
            last_seen_at: '2026-05-07T11:00:00Z',
            exemplar_event_ids: [101],
            exemplar_summaries: ['cuda oom at step 1'],
            exemplar_run_ids: ['exp-99'],
            last_computed_at: '2026-05-07T12:00:00Z',
        },
    ],
};

const BUNDLES_RESPONSE = {
    project_id: 7,
    limit: 50,
    bundles: [],
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

function installDefaultHandlers() {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url === '/projects/7/timeline') {
            return { data: TIMELINE_RESPONSE };
        }
        if (url === '/projects/7/failure-clusters') {
            return { data: CLUSTERS_RESPONSE };
        }
        if (url === '/projects/7/support-bundles') {
            return { data: BUNDLES_RESPONSE };
        }
        if (url === '/run-events/run/exp-1') {
            return {
                data: {
                    run_id: 'exp-1',
                    limit: 200,
                    events: [
                        {
                            id: 1,
                            run_id: 'exp-1',
                            parent_run_id: null,
                            stage: 'training',
                            severity: 'info',
                            reason_code: null,
                            actor: 'system',
                            summary: 'Training started',
                            payload: {},
                            ts: '2026-05-07T10:00:00Z',
                            created_at: '2026-05-07T10:00:00Z',
                        },
                    ],
                },
            };
        }
        throw new Error(`Unexpected GET ${url}`);
    });
}

describe('ProjectObservabilityPage', () => {
    it('renders summary badges + timeline tree on first load', async () => {
        installDefaultHandlers();
        render(<ProjectObservabilityPage />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/7/timeline',
                expect.any(Object),
            );
        });
        expect(await screen.findByText(/2 run\(s\)/)).toBeInTheDocument();
        expect(screen.getByText(/4 event\(s\)/)).toBeInTheDocument();
        expect(screen.getByText('exp-1')).toBeInTheDocument();
    });

    it('clicking a timeline run opens the drilldown drawer', async () => {
        installDefaultHandlers();
        render(<ProjectObservabilityPage />);
        const user = userEvent.setup();
        await user.click(
            await screen.findByRole('button', {
                name: /Open events for exp-1/i,
            }),
        );
        // Drawer renders the run id heading.
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/run-events/run/exp-1');
        });
        expect(
            await screen.findByRole('dialog', { name: /Events for run exp-1/i }),
        ).toBeInTheDocument();
    });

    it('changing a filter triggers a re-fetch with the right params', async () => {
        installDefaultHandlers();
        render(<ProjectObservabilityPage />);
        await screen.findByText(/2 run\(s\)/);
        const callsBefore = apiMock.get.mock.calls.filter(
            (c: unknown[]) => c[0] === '/projects/7/timeline',
        ).length;

        const user = userEvent.setup();
        await user.selectOptions(
            screen.getByLabelText(/Filter by severity/i),
            'error',
        );
        await waitFor(() => {
            const callsAfter = apiMock.get.mock.calls.filter(
                (c: unknown[]) => c[0] === '/projects/7/timeline',
            );
            expect(callsAfter.length).toBeGreaterThan(callsBefore);
            const last = callsAfter[callsAfter.length - 1];
            expect((last[1] as { params: { severity?: string } }).params.severity).toBe(
                'error',
            );
        });
    });

    it('surfaces a top-level alert when timeline fetch fails', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url === '/projects/7/timeline') {
                return Promise.reject({
                    response: { status: 500, data: { detail: 'server_error' } },
                });
            }
            if (url === '/projects/7/failure-clusters') {
                return { data: CLUSTERS_RESPONSE };
            }
            if (url === '/projects/7/support-bundles') {
                return { data: BUNDLES_RESPONSE };
            }
            throw new Error(`Unexpected GET ${url}`);
        });
        render(<ProjectObservabilityPage />);
        const alerts = await screen.findAllByRole('alert');
        const hasError = alerts.some((node) =>
            node.textContent?.includes('server_error'),
        );
        expect(hasError).toBe(true);
    });
});
