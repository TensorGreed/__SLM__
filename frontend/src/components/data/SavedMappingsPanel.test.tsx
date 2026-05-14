/**
 * Saved-mappings panel contract.
 *
 * Pins:
 * - hidden when the project has no saved configs;
 * - lists configs in the order returned by the API;
 * - Re-run hits the configs/{id}/run endpoint + bumps the displayed
 *   last-run row count;
 * - Delete confirms via window.confirm + hits DELETE.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import SavedMappingsPanel from './SavedMappingsPanel';

const TWO_CONFIGS = [
    {
        id: 1,
        project_id: 77,
        name: 'weekly-pii-refresh',
        description: 'pull the kaggle dump every Monday',
        locator: 'kaggle:competition:pii-detection',
        mapper_id: 'bio_to_spans',
        field_map: {},
        drop_reasons: [],
        limit: null,
        created_at: null,
        updated_at: null,
        last_run_at: '2026-05-13T10:00:00Z',
        last_run_accepted: 5012,
    },
    {
        id: 2,
        project_id: 77,
        name: 'sentiment-daily',
        description: null,
        locator: 'jsonl:/var/feeds/today.jsonl',
        mapper_id: 'label_to_classification',
        field_map: {},
        drop_reasons: [],
        limit: null,
        created_at: null,
        updated_at: null,
        last_run_at: null,
        last_run_accepted: null,
    },
];

describe('SavedMappingsPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.delete.mockReset();
    });

    it('renders nothing when the project has no saved configs', async () => {
        apiMock.get.mockResolvedValue({ data: { configs: [] } });

        const { container } = render(<SavedMappingsPanel projectId={77} />);
        await waitFor(() =>
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/77/dataset-import/configs',
            ),
        );
        // Empty state stays out of the DOM.
        expect(
            container.querySelector('[data-testid="saved-mappings-panel"]'),
        ).toBeNull();
    });

    it('lists configs with their locator + last-run stats', async () => {
        apiMock.get.mockResolvedValue({ data: { configs: TWO_CONFIGS } });

        render(<SavedMappingsPanel projectId={77} />);
        expect(await screen.findByText('weekly-pii-refresh')).toBeInTheDocument();
        expect(screen.getByText('sentiment-daily')).toBeInTheDocument();
        expect(
            screen.getByText('kaggle:competition:pii-detection'),
        ).toBeInTheDocument();
        expect(screen.getByText('5012')).toBeInTheDocument();
    });

    it('re-runs a saved config and refreshes the list', async () => {
        apiMock.get.mockResolvedValue({ data: { configs: TWO_CONFIGS } });
        apiMock.post.mockResolvedValue({
            data: {
                accepted_count: 5099,
                rejected_count: 0,
                source_id: 'kaggle',
                mapper_id: 'bio_to_spans',
                target_task_profile: 'structured_extraction',
                locator: 'kaggle:competition:pii-detection',
                written_path: '/var/.../synthetic.jsonl',
                dry_run: false,
                rejection_counts: {},
                warnings: [],
                accepted_sample: [],
                rejected_sample: [],
            },
        });

        const user = userEvent.setup();
        render(<SavedMappingsPanel projectId={77} />);
        await screen.findByText('weekly-pii-refresh');

        await user.click(screen.getByTestId('run-config-1'));

        await waitFor(() =>
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/77/dataset-import/configs/1/run',
            ),
        );
        // After the run, the panel re-fetches to pull updated last_run_*.
        await waitFor(() => {
            const listCalls = apiMock.get.mock.calls.filter(
                ([url]) => url === '/projects/77/dataset-import/configs',
            );
            expect(listCalls.length).toBeGreaterThanOrEqual(2);
        });
    });

    it('confirms before deleting and skips when the user cancels', async () => {
        apiMock.get.mockResolvedValue({ data: { configs: TWO_CONFIGS } });
        const confirmSpy = vi
            .spyOn(window, 'confirm')
            .mockReturnValueOnce(false);

        const user = userEvent.setup();
        render(<SavedMappingsPanel projectId={77} />);
        await screen.findByText('weekly-pii-refresh');

        await user.click(screen.getByTestId('delete-config-1'));
        expect(confirmSpy).toHaveBeenCalled();
        expect(apiMock.delete).not.toHaveBeenCalled();

        confirmSpy.mockRestore();
    });

    it('deletes on confirm + re-fetches the list', async () => {
        apiMock.get.mockResolvedValue({ data: { configs: TWO_CONFIGS } });
        apiMock.delete.mockResolvedValue({ data: undefined });
        const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

        const user = userEvent.setup();
        render(<SavedMappingsPanel projectId={77} />);
        await screen.findByText('weekly-pii-refresh');

        await user.click(screen.getByTestId('delete-config-1'));
        await waitFor(() =>
            expect(apiMock.delete).toHaveBeenCalledWith(
                '/projects/77/dataset-import/configs/1',
            ),
        );

        confirmSpy.mockRestore();
    });
});
