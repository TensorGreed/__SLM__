import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import SeedSourcesSelect from './SeedSourcesSelect';

const source = (id: number, title: string, document_count: number, usage = 'any') => ({
    id, title, source_kind: 'git_repo', locator: '', usage, document_count,
});

describe('SeedSourcesSelect', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.get.mockResolvedValue({
            data: { sources: [source(3, 'samples-repo', 120), source(5, 'helm-charts', 1, 'rag')], totals: {} },
        });
    });

    it('lists sources by title + id and emits the chosen ids (null when none)', async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        render(<SeedSourcesSelect projectId={7} onChange={onChange} />);
        await screen.findByText(/samples-repo \(source #3, 120 docs, Any use\)/);
        expect(screen.getByText(/helm-charts \(source #5, 1 doc, Retrieval only\)/)).toBeInTheDocument();
        expect(apiMock.get).toHaveBeenCalledWith('/projects/7/provenance/sources');
        expect(onChange).toHaveBeenLastCalledWith(null);
        await user.click(screen.getByTestId('seed-source-3'));
        expect(onChange).toHaveBeenLastCalledWith([3]);
        await user.click(screen.getByTestId('seed-source-5'));
        expect(onChange).toHaveBeenLastCalledWith([3, 5]);
        await user.click(screen.getByTestId('seed-source-3'));
        await waitFor(() => expect(onChange).toHaveBeenLastCalledWith([5]));
    });

    it('renders nothing when the project has no sources', async () => {
        apiMock.get.mockResolvedValue({ data: { sources: [], totals: {} } });
        render(<SeedSourcesSelect projectId={7} onChange={vi.fn()} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(screen.queryByTestId('seed-sources')).not.toBeInTheDocument();
    });
});
