import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type {
    DeploymentVersion,
    DeploymentVersionListResponse,
} from '../types/deployment';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../api/client', () => ({ default: apiMock }));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
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

import ProjectDeploymentsPage from './ProjectDeploymentsPage';

function makeVersion(overrides: Partial<DeploymentVersion>): DeploymentVersion {
    return {
        id: 1,
        project_id: 7,
        export_id: 11,
        registry_entry_id: null,
        version: 1,
        target_id: 'sdk.apple_coreml_stub',
        target_kind: 'sdk',
        endpoint_name: null,
        endpoint_handle: null,
        region: null,
        instance_type: null,
        status: 'pending',
        plan_payload: {},
        promoted_reason: null,
        rejected_reason: null,
        rolled_back_reason: null,
        rolled_back_to_id: null,
        actor: 'system',
        created_at: '2026-05-04T00:00:00Z',
        promoted_at: null,
        rejected_at: null,
        rolled_back_at: null,
        superseded_at: null,
        ...overrides,
    };
}

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

describe('ProjectDeploymentsPage', () => {
    it('renders summary counts and the versions table on load', async () => {
        const list: DeploymentVersionListResponse = {
            project_id: 7,
            deployment_versions: [
                makeVersion({ id: 1, version: 1, status: 'promoted' }),
                makeVersion({ id: 2, version: 2, status: 'pending' }),
                makeVersion({ id: 3, version: 3, status: 'rejected' }),
            ],
        };

        apiMock.get.mockImplementation(async (url: string) => {
            if (url === '/projects/7/deployments') return { data: list };
            if (url.startsWith('/deployments/1/score'))
                return Promise.reject({ response: { status: 404, data: { detail: 'score_not_found' } } });
            if (url.startsWith('/deployments/1/telemetry')) {
                return {
                    data: {
                        deployment_version_id: 1,
                        window_start: '2026-05-04T00:00:00Z',
                        window_end: '2026-05-04T01:00:00Z',
                        window_seconds: 3600,
                        sample_count: 0,
                        request_volume: { total: 0, per_second: 0, per_minute: 0 },
                        latency_ms: { p50: 0, p95: 0, p99: 0, min: 0, max: 0, mean: 0 },
                        errors: { count: 0, rate: 0 },
                        tokens: {
                            input_total: 0,
                            output_total: 0,
                            input_per_second: 0,
                            output_per_second: 0,
                            total_per_second: 0,
                        },
                    },
                };
            }
            if (url.startsWith('/deployments/1/drift/checks')) {
                return {
                    data: {
                        deployment_version_id: 1,
                        limit: 50,
                        drift_checks: [],
                    },
                };
            }
            throw new Error(`Unexpected GET ${url}`);
        });

        render(<ProjectDeploymentsPage />);

        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/projects/7/deployments');
        });

        expect(await screen.findByText(/total: 3/)).toBeInTheDocument();
        expect(screen.getByText(/promoted: 1/)).toBeInTheDocument();
        expect(screen.getByText(/pending: 1/)).toBeInTheDocument();
        // Versions table renders all three.
        expect(screen.getAllByText(/^v[1-3]/).length).toBe(3);
        // The page auto-selects the promoted version (v1) and renders the
        // score / telemetry / drift cards for it.
        expect(await screen.findByText(/Deployability score/i)).toBeInTheDocument();
        expect(await screen.findByText(/Live telemetry/i)).toBeInTheDocument();
        expect(await screen.findByText(/^Drift$/i)).toBeInTheDocument();
    });

    it('renders the empty state when no deployment versions exist', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 7, deployment_versions: [] },
        });
        render(<ProjectDeploymentsPage />);

        expect(
            await screen.findByText(/No deployment versions yet/i),
        ).toBeInTheDocument();
        // Downstream card headings (h3) must not mount when nothing is
        // selected. The page subtitle paragraph mentions "deployability
        // score" so we scope to role=heading to avoid a false positive.
        expect(
            screen.queryByRole('heading', { name: /Deployability score/i }),
        ).not.toBeInTheDocument();
        expect(
            screen.queryByRole('heading', { name: /Live telemetry/i }),
        ).not.toBeInTheDocument();
    });

    it('surfaces list-fetch errors as a top-level alert', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 500, data: { detail: 'server_error' } },
        });
        render(<ProjectDeploymentsPage />);
        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent('server_error');
    });

    it('bumping refreshKey re-fetches the downstream cards after an action', async () => {
        // Track every score endpoint hit so we can assert that the card
        // re-fetches when refreshKey changes (i.e. after a row action).
        const scoreCalls: string[] = [];
        const list: DeploymentVersionListResponse = {
            project_id: 7,
            deployment_versions: [
                makeVersion({ id: 1, version: 1, status: 'pending' }),
            ],
        };

        apiMock.get.mockImplementation(async (url: string) => {
            if (url === '/projects/7/deployments') return { data: list };
            if (url.startsWith('/deployments/1/score')) {
                scoreCalls.push(url);
                return Promise.reject({
                    response: { status: 404, data: { detail: 'score_not_found' } },
                });
            }
            if (url.startsWith('/deployments/1/telemetry')) {
                return {
                    data: {
                        deployment_version_id: 1,
                        window_start: '2026-05-04T00:00:00Z',
                        window_end: '2026-05-04T01:00:00Z',
                        window_seconds: 3600,
                        sample_count: 0,
                        request_volume: { total: 0, per_second: 0, per_minute: 0 },
                        latency_ms: { p50: 0, p95: 0, p99: 0, min: 0, max: 0, mean: 0 },
                        errors: { count: 0, rate: 0 },
                        tokens: {
                            input_total: 0,
                            output_total: 0,
                            input_per_second: 0,
                            output_per_second: 0,
                            total_per_second: 0,
                        },
                    },
                };
            }
            if (url.startsWith('/deployments/1/drift/checks')) {
                return {
                    data: { deployment_version_id: 1, limit: 50, drift_checks: [] },
                };
            }
            throw new Error(`Unexpected GET ${url}`);
        });

        render(<ProjectDeploymentsPage />);
        // First-load score fetch.
        await waitFor(() => {
            expect(scoreCalls.length).toBeGreaterThanOrEqual(1);
        });
        const initialCount = scoreCalls.length;

        // Click the page-level Refresh button to re-run fetchVersions,
        // which bumps refreshKey and should trigger a second score
        // fetch even though the selected dv id hasn't changed.
        const user = (await import('@testing-library/user-event')).default.setup();
        await user.click(screen.getByRole('button', { name: /^Refresh$/i }));

        await waitFor(() => {
            expect(scoreCalls.length).toBeGreaterThan(initialCount);
        });
    });
});
