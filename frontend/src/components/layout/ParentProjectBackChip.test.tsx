import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import ParentProjectBackChip from './ParentProjectBackChip';


describe('ParentProjectBackChip', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders the parent name after the fetch resolves', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { id: 7, name: 'Policy QA' },
        });
        render(<ParentProjectBackChip parentProjectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('parent-project-chip')).toHaveTextContent(
                '← cloned from Policy QA',
            );
        });
        // Chip is an anchor pointing at the parent project page.
        expect(screen.getByTestId('parent-project-chip')).toHaveAttribute(
            'href',
            '/project/7',
        );
    });

    it('falls back to a numeric label when the parent fetch fails', async () => {
        apiMock.get.mockRejectedValueOnce({ response: { status: 404 } });
        render(<ParentProjectBackChip parentProjectId={42} />);
        // After the failed fetch settles, the chip still renders with
        // the numeric fallback.
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalled();
        });
        expect(screen.getByTestId('parent-project-chip')).toHaveTextContent(
            '← cloned from project #42',
        );
    });

    it('renders the numeric label initially while the fetch is pending', async () => {
        // Mock returns a never-resolving promise so the initial state stays put.
        apiMock.get.mockImplementationOnce(
            () => new Promise(() => { /* never resolves */ }),
        );
        render(<ParentProjectBackChip parentProjectId={9} />);
        expect(screen.getByTestId('parent-project-chip')).toHaveTextContent(
            '← cloned from project #9',
        );
    });
});
