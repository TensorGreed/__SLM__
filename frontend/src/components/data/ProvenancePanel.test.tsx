import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn(), patch: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import ProvenancePanel from './ProvenancePanel';

const src = (over: Record<string, unknown>) => ({
    id: 1, source_kind: 'web_crawl', title: 'Docs crawl', locator: 'https://docs.example.com/', source_ref: null,
    licence: 'MIT', licence_verified: false, licence_verified_by: null, licence_verified_at: null, retrieved_at: null,
    usage: 'any', document_count: 3, row_count: 30, licence_flags: [], in_prepared: null, usage_locked: false, ...over,
});

const listing = {
    sources: [
        src({}),
        src({ id: 2, source_kind: 'git_repo', title: 'Repo', locator: 'https://github.com/org/repo', source_ref: 'abcdef1234567890',
            licence: null, licence_flags: ['unknown'] }),
        src({ id: 3, source_kind: 'hf', title: 'toy/dataset', locator: 'toy/dataset', licence: 'CC-BY-NC-4.0',
            licence_flags: ['non_commercial'], licence_verified: true, licence_verified_by: 'admin',
            in_prepared: { train: 8, val: 1, test: 1, excluded: 0 } }),
        src({ id: 4, source_kind: 'predefined_split', title: 'Test split', locator: 'test.jsonl', usage_locked: true }),
    ],
    totals: { sources: 4, documents: 12, unknown_licence: 1, non_commercial: 1 },
};

describe('ProvenancePanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset(); apiMock.patch.mockReset();
        apiMock.get.mockResolvedValue({ data: listing });
        apiMock.patch.mockResolvedValue({ data: { sources: listing.sources } });
    });

    it('renders sources with ref label, warning badges and verified mark', async () => {
        render(<ProvenancePanel projectId={5} />);
        await screen.findByTestId('prov-row-1');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/5/provenance/sources');
        expect(screen.getByTestId('prov-locator-2').textContent).toBe('https://github.com/org/repo @ abcdef123456');
        expect(screen.getByTestId('prov-flag-2-unknown')).toBeTruthy();
        expect(screen.getByTestId('prov-flag-3-non_commercial')).toBeTruthy();
        expect(screen.getByTestId('prov-verified-3').textContent).toContain('Verified by admin');
        expect(screen.getByTestId('prov-splits-3').textContent).toBe('8 / 1 / 1');
    });

    it('shows 0 / 0 / 0 for a source a prepared split holds nothing of, and Not prepared for none', async () => {
        apiMock.get.mockResolvedValue({ data: {
            ...listing,
            sources: [
                src({ id: 7, title: 'late', in_prepared: { train: 0, val: 0, test: 0, excluded: 0 } }),
                src({ id: 8, title: 'never', in_prepared: null }),
            ],
        } });
        render(<ProvenancePanel projectId={5} />);
        await screen.findByTestId('prov-row-7');
        expect(screen.getByTestId('prov-splits-7').textContent).toBe('0 / 0 / 0');
        expect(screen.getByTestId('prov-splits-8').textContent).toBe('Not prepared');
    });

    it('bulk-sets usage and refreshes; locked rows disable the usage control', async () => {
        const user = userEvent.setup();
        render(<ProvenancePanel projectId={5} />);
        await screen.findByTestId('prov-row-1');
        await user.click(screen.getByTestId('prov-pick-1'));
        await user.click(screen.getByTestId('prov-pick-2'));
        await user.selectOptions(screen.getByTestId('prov-usage-select'), 'eval');
        await user.click(screen.getByTestId('prov-set-usage'));
        await waitFor(() => expect(apiMock.patch).toHaveBeenCalledWith(
            '/projects/5/provenance/sources', { source_ids: [1, 2], usage: 'eval' }));
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(2));
        await user.click(screen.getByTestId('prov-pick-4'));
        expect((screen.getByTestId('prov-usage-select') as HTMLSelectElement).disabled).toBe(true);
        expect((screen.getByTestId('prov-set-usage') as HTMLButtonElement).disabled).toBe(true);
    });

    it('sets licence, clears to unknown and marks verified for all selected', async () => {
        const user = userEvent.setup();
        render(<ProvenancePanel projectId={5} />);
        await screen.findByTestId('prov-row-1');
        await user.click(screen.getByTestId('prov-select-all'));
        await user.type(screen.getByTestId('prov-licence-input'), 'Apache-2.0');
        await user.click(screen.getByTestId('prov-set-licence'));
        await waitFor(() => expect(apiMock.patch).toHaveBeenLastCalledWith(
            '/projects/5/provenance/sources', { source_ids: [1, 2, 3, 4], licence: 'Apache-2.0' }));
        await user.click(screen.getByTestId('prov-clear-licence'));
        await waitFor(() => expect(apiMock.patch).toHaveBeenLastCalledWith(
            '/projects/5/provenance/sources', { source_ids: [1, 2, 3, 4], licence: null }));
        await user.click(screen.getByTestId('prov-verify'));
        await waitFor(() => expect(apiMock.patch).toHaveBeenLastCalledWith(
            '/projects/5/provenance/sources', { source_ids: [1, 2, 3, 4], licence_verified: true }));
    });
});
