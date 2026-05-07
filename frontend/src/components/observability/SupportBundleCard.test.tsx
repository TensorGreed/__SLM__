import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type {
    SupportBundleListItem,
    SupportBundleMetadata,
} from '../../types/observability';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import SupportBundleCard from './SupportBundleCard';

const SAMPLE_METADATA: SupportBundleMetadata = {
    bundle_uid: 'abc1234567890def',
    project_id: 7,
    size_bytes: 4096,
    sha256: 'fakesha',
    section_counts: {
        project: 1,
        run_events: 12,
        deployment_versions: 0,
    },
    redactions_applied: {
        run_events: { total: 5, by_reason: { hf_token: 5 } },
        project: { total: 0, by_reason: {} },
    },
    expires_at: '2026-05-08T12:00:00Z',
    created_at: '2026-05-07T12:00:00Z',
    download_url: '/api/support-bundles/abc1234567890def/download?token=tok',
    download_token: 'tok',
    actor: 'system',
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

describe('SupportBundleCard', () => {
    it('shows empty state when no bundles exist', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 7, limit: 50, bundles: [] },
        });
        render(<SupportBundleCard projectId={7} />);
        expect(
            await screen.findByText(/No support bundles generated yet/i),
        ).toBeInTheDocument();
    });

    it('Generate bundle POSTs and renders the preview with redactions', async () => {
        // Initial list fetch returns empty.
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 7, limit: 50, bundles: [] },
        });
        // Create returns metadata.
        apiMock.post.mockResolvedValueOnce({ data: SAMPLE_METADATA });
        // Re-fetch after create returns the new bundle.
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 7,
                limit: 50,
                bundles: [
                    {
                        bundle_uid: SAMPLE_METADATA.bundle_uid,
                        size_bytes: SAMPLE_METADATA.size_bytes,
                        sha256: SAMPLE_METADATA.sha256,
                        section_counts: SAMPLE_METADATA.section_counts,
                        redactions_applied: SAMPLE_METADATA.redactions_applied,
                        actor: 'system',
                        created_at: SAMPLE_METADATA.created_at,
                        expires_at: SAMPLE_METADATA.expires_at,
                    } as SupportBundleListItem,
                ],
            },
        });

        render(<SupportBundleCard projectId={7} />);
        await screen.findByText(/No support bundles/i);

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Generate bundle/i }),
        );

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/support-bundle',
                {},
            );
        });
        expect(
            await screen.findByText('abc1234567890def'),
        ).toBeInTheDocument();
        // Section counts + redaction summary both reference run_events;
        // assert by structural query instead of bare text to avoid the
        // duplicate-match error.
        const runEventsHits = screen.getAllByText('run_events');
        expect(runEventsHits.length).toBeGreaterThanOrEqual(2);
        // Redaction summary surfaces the per-section totals.
        expect(screen.getByText(/Redactions \(5 total\)/)).toBeInTheDocument();
        expect(screen.getByText(/5 scrubbed/i)).toBeInTheDocument();
        // Download link points at the API URL with the token query.
        const link = screen.getByRole('link', { name: /Download zip/i });
        expect(link).toHaveAttribute(
            'href',
            '/api/support-bundles/abc1234567890def/download?token=tok',
        );
    });

    it('lists existing bundles in a table', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                project_id: 7,
                limit: 50,
                bundles: [
                    {
                        bundle_uid: 'aaa111bbb222ccc333',
                        size_bytes: 2048,
                        sha256: 'sha',
                        section_counts: { run_events: 3 },
                        redactions_applied: {
                            run_events: { total: 1, by_reason: { hf_token: 1 } },
                        },
                        actor: 'ops',
                        created_at: '2026-05-07T10:00:00Z',
                        expires_at: '2026-05-08T10:00:00Z',
                    },
                ],
            },
        });
        render(<SupportBundleCard projectId={7} />);
        expect(await screen.findByText(/aaa111bbb222/)).toBeInTheDocument();
        expect(screen.getByText('ops')).toBeInTheDocument();
    });

    it('surfaces error when create fails', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { project_id: 7, limit: 50, bundles: [] },
        });
        apiMock.post.mockRejectedValueOnce({
            response: { status: 500, data: { detail: 'server_error' } },
        });
        render(<SupportBundleCard projectId={7} />);
        await screen.findByText(/No support bundles/i);

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Generate bundle/i }),
        );
        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent('server_error');
    });
});
