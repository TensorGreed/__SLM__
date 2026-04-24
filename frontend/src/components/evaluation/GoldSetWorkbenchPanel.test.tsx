import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import GoldSetWorkbenchPanel from './GoldSetWorkbenchPanel';

const DATASET_RESPONSE = {
    datasets: [
        { id: 5, name: 'Train Set', dataset_type: 'train', record_count: 452, file_path: '/x' },
        { id: 3, name: 'Gold Dev Set', dataset_type: 'gold_dev', record_count: 5, file_path: '/g' },
    ],
};

function makeRow(id: number, status = 'pending') {
    return {
        id,
        gold_set_id: 3,
        version_id: 1,
        source_row_key: `key-${id}`,
        source_dataset_id: 5,
        input: { question: `Q${id}` },
        expected: {},
        rationale: '',
        labels: {},
        reviewer_id: null,
        status,
        created_at: null,
        updated_at: null,
        reviewed_at: null,
    };
}

describe('GoldSetWorkbenchPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.patch.mockReset();
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/datasets')) {
                return { data: DATASET_RESPONSE };
            }
            if (url.includes('/queue')) {
                return { data: { gold_set_id: 3, count: 0, limit: 50, offset: 0, items: [] } };
            }
            return { data: {} };
        });
    });

    it('auto-picks a gold set + source, samples rows, and opens labeller', async () => {
        const user = userEvent.setup();
        apiMock.post.mockResolvedValueOnce({
            data: {
                gold_set_id: 3,
                version_id: 1,
                version: 1,
                requested: 2,
                created: 2,
                skipped_duplicates: 0,
                strategy: 'random',
                stratify_by: null,
                rows: [makeRow(101), makeRow(102)],
            },
        });

        render(<GoldSetWorkbenchPanel projectId={1} />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/projects/1/datasets');
        });

        const sampleBtn = await screen.findByRole('button', { name: /sample rows into gold set/i });
        await user.click(sampleBtn);

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/api/gold-sets/3/rows/sample',
                expect.objectContaining({
                    source_dataset_id: 5,
                    target_count: 20,
                    strategy: 'random',
                    seed: 42,
                }),
            );
        });

        await screen.findByText(/row #101/i);
        expect(screen.getByText('1 / 2')).toBeInTheDocument();
    });

    it('PATCHes the active row when approving via Enter on the Approve button', async () => {
        const user = userEvent.setup();
        apiMock.post.mockResolvedValueOnce({
            data: {
                gold_set_id: 3,
                version_id: 1,
                version: 1,
                requested: 1,
                created: 1,
                skipped_duplicates: 0,
                strategy: 'random',
                stratify_by: null,
                rows: [makeRow(202)],
            },
        });
        apiMock.patch.mockResolvedValueOnce({ data: makeRow(202, 'approved') });

        render(<GoldSetWorkbenchPanel projectId={1} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        await user.click(await screen.findByRole('button', { name: /sample rows into gold set/i }));
        await screen.findByText(/row #202/i);

        await user.click(screen.getByRole('button', { name: /approve \(a\)/i }));

        await waitFor(() => {
            expect(apiMock.patch).toHaveBeenCalledWith(
                '/api/gold-sets/3/rows/202',
                expect.objectContaining({ status: 'approved' }),
            );
        });
    });

    it('triggers approve via keyboard shortcut "A" when labeller section has focus', async () => {
        const user = userEvent.setup();
        apiMock.post.mockResolvedValueOnce({
            data: {
                gold_set_id: 3,
                version_id: 1,
                version: 1,
                requested: 1,
                created: 1,
                skipped_duplicates: 0,
                strategy: 'random',
                stratify_by: null,
                rows: [makeRow(303)],
            },
        });
        apiMock.patch.mockResolvedValueOnce({ data: makeRow(303, 'approved') });

        render(<GoldSetWorkbenchPanel projectId={1} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        await user.click(await screen.findByRole('button', { name: /sample rows into gold set/i }));
        await screen.findByText(/row #303/i);

        const labeller = screen.getByLabelText(/single-row labeller with keyboard shortcuts/i);
        labeller.focus();
        await user.keyboard('a');

        await waitFor(() => {
            expect(apiMock.patch).toHaveBeenCalledWith(
                '/api/gold-sets/3/rows/303',
                expect.objectContaining({ status: 'approved' }),
            );
        });
    });

    it('surfaces a backend error message when sampling fails', async () => {
        const user = userEvent.setup();
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'stratify_by_required' } },
        });

        render(<GoldSetWorkbenchPanel projectId={1} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());

        // flip to stratified without providing stratify_by to trip the client-side guard
        const strategySelect = screen
            .getAllByRole('combobox')
            .find((el) => (el as HTMLSelectElement).value === 'random') as HTMLSelectElement;
        await user.selectOptions(strategySelect, 'stratified');
        await user.click(await screen.findByRole('button', { name: /sample rows into gold set/i }));

        expect(
            await screen.findByText(/stratified sampling needs a stratify-by field/i),
        ).toBeInTheDocument();
        expect(apiMock.post).not.toHaveBeenCalled();
    });
});
