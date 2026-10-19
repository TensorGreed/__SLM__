/**
 * Data → Prepare: the "Remove repeated page headers and footers" option is
 * on by default, is sent with the cleaning request, and the result says
 * how many furniture lines a document lost.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./LabelNoiseReviewPanel', () => ({ default: () => <div data-testid="label-noise" /> }));

import CleaningPanel from './CleaningPanel';

const RESULT = {
    document_id: 7,
    quality_score: 0.8,
    chunk_count: 12,
    original_chars: 20000,
    cleaned_chars: 18000,
    furniture_lines_removed: 301,
    page_furniture: {
        enabled: true,
        pages: 142,
        furniture_lines_removed: 301,
        furniture_segments_trimmed: 209,
        examples: ['Page 9 of 142 13:47 - 20-Jan-2026'],
    },
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/ingestion/documents')) {
            return { data: [{ id: 7, filename: 'p17.pdf', status: 'accepted', quality_score: null, chunk_count: 0 }] };
        }
        if (url.includes('/cleaning/tasks/')) {
            return { data: { task_id: 't1', project_id: 1, status: 'completed', total: 1, completed: 1, current_document_id: null, results: [RESULT], errors: [], error: null } };
        }
        return { data: {} };
    });
    apiMock.post.mockResolvedValue({ data: { task_id: 't1', status: 'pending' } });
});

describe('CleaningPanel — page furniture option', () => {
    it('is on by default and sent with the request', async () => {
        const user = userEvent.setup();
        render(<MemoryRouter><CleaningPanel projectId={1} /></MemoryRouter>);
        const box = screen.getByTestId('cleaning-remove-page-furniture') as HTMLInputElement;
        expect(box.checked).toBe(true);
        expect(screen.getByText(/Remove repeated page headers and footers/)).toBeInTheDocument();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Clean All Documents' })).toBeEnabled());
        await user.click(screen.getByRole('button', { name: 'Clean All Documents' }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect(apiMock.post.mock.calls[0][1]).toMatchObject({ document_ids: [7], remove_page_furniture: true });

        await waitFor(() => expect(screen.getByText(/Cleaned 1\/1/)).toBeInTheDocument(), { timeout: 4000 });
        await user.click(screen.getByTestId('data-details-toggle'));
        await user.click(screen.getByTestId('data-details-section-results'));
        expect(screen.getByTestId('cleaning-furniture-7')).toHaveTextContent(
            '142 pages · 301 header/footer lines removed · 209 trimmed off body lines',
        );
        expect(screen.getByText('Page 9 of 142 13:47 - 20-Jan-2026')).toBeInTheDocument();
    });

    it('can be turned off to keep page layout', async () => {
        const user = userEvent.setup();
        render(<MemoryRouter><CleaningPanel projectId={1} /></MemoryRouter>);
        await user.click(screen.getByTestId('cleaning-remove-page-furniture'));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Clean All Documents' })).toBeEnabled());
        await user.click(screen.getByRole('button', { name: 'Clean All Documents' }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect(apiMock.post.mock.calls[0][1]).toMatchObject({ remove_page_furniture: false });
    });
});
