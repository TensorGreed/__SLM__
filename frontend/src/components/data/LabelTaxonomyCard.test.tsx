import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import { useToastStore } from '../../stores/toastStore';
import LabelTaxonomyCard from './LabelTaxonomyCard';

const STATE = {
    taxonomy: {
        name: 'toy',
        source_version: '2',
        entries: [
            { id: 'LEAK-1', title: 'Data leak', parent: null, aliases: ['exposure'], attrs: { severity: 'high' }, crosswalk: { stdA: ['A.1'], stdB: ['B-7', 'B-8'] } },
            { id: 'LEAK-1.1', title: 'Log leak', parent: 'LEAK-1', aliases: [], attrs: {}, crosswalk: {} },
            { id: 'KEY-2', title: 'Hard-coded key', parent: null, aliases: ['secret'], attrs: {}, crosswalk: {} },
        ],
    },
    version: 'aaaaaaaaaaaa',
    entry_count: 3,
    namespaces: ['stdA', 'stdB'],
    history: [
        { version: 'aaaaaaaaaaaa', entry_count: 3, saved_at: '2026-10-26', current: true, restorable: true },
        { version: 'bbbbbbbbbbbb', entry_count: 2, saved_at: '2026-10-25', current: false, restorable: true },
        { version: 'dddddddddddd', entry_count: 1, saved_at: '2026-10-01', current: false, restorable: false },
    ],
};

describe('LabelTaxonomyCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset(); apiMock.post.mockReset(); apiMock.put.mockReset(); apiMock.delete.mockReset();
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/resolve')) return { data: { raw: 'leak 1', id: 'LEAK-1', matched_by: 'id', title: 'Data leak', candidates: [] } };
            return { data: STATE };
        });
    });

    it('shows version and count in the collapsed header, then the tree with crosswalk and attrs', async () => {
        const user = userEvent.setup();
        render(<LabelTaxonomyCard projectId={5} />);
        await waitFor(() => expect(screen.getByTestId('taxonomy-version')).toHaveTextContent('aaaaaaaaaaaa'));
        expect(screen.getByTestId('taxonomy-count')).toHaveTextContent('3 entries');
        expect(screen.queryByTestId('taxonomy-tree')).toBeNull();
        await user.click(screen.getByTestId('taxonomy-toggle'));
        expect(screen.getByTestId('taxonomy-xw-LEAK-1-stdB')).toHaveTextContent('stdB: B-7, B-8');
        expect(screen.getByTestId('taxonomy-attrs-LEAK-1')).toHaveTextContent('severity: high');
        expect(screen.queryByTestId('taxonomy-node-LEAK-1.1')).toBeNull();
        await user.click(screen.getByTestId('taxonomy-toggle-LEAK-1'));
        expect(screen.getByTestId('taxonomy-node-LEAK-1.1')).toBeInTheDocument();
        await user.type(screen.getByTestId('taxonomy-search'), 'secret');
        expect(screen.queryByTestId('taxonomy-node-LEAK-1')).toBeNull();
        expect(screen.getByTestId('taxonomy-node-KEY-2')).toBeInTheDocument();
    });

    it('validates with grouped errors, and shows 422 errors on save', async () => {
        const user = userEvent.setup();
        apiMock.post.mockResolvedValue({ data: { valid: false, errors: [{ code: 'duplicate_id', id: 'KEY-2', message: 'twice' }, { code: 'duplicate_id', id: 'LEAK-1', message: 'twice' }], warnings: [], entry_count: 0, version: null } });
        apiMock.put.mockRejectedValue({ response: { data: { error_code: 'INVALID_TAXONOMY', detail: 'bad', metadata: { errors: [{ code: 'unknown_parent', id: 'KEY-2', message: 'no parent X' }] } } } });
        render(<LabelTaxonomyCard projectId={5} />);
        await user.click(screen.getByTestId('taxonomy-toggle'));
        await screen.findByTestId('taxonomy-version');
        await user.type(screen.getByTestId('taxonomy-crosswalk-from'), 'refsA, refsB');
        await user.click(screen.getByTestId('taxonomy-text'));
        await user.paste('- id: KEY-2');
        await user.click(screen.getByTestId('taxonomy-validate'));
        const errs = await screen.findByTestId('taxonomy-errors');
        expect(within(errs).getByText('duplicate_id')).toBeInTheDocument();
        expect(errs).toHaveTextContent('KEY-2');
        expect(apiMock.post).toHaveBeenCalledWith('/projects/5/label-taxonomy/validate', { content: '- id: KEY-2', format: 'auto', crosswalk_from: ['refsA', 'refsB'] });
        await user.click(screen.getByTestId('taxonomy-save'));
        await waitFor(() => expect(screen.getByTestId('taxonomy-errors')).toHaveTextContent('unknown_parent'));
    });

    it('saves valid text and updates the version', async () => {
        const user = userEvent.setup();
        apiMock.put.mockResolvedValue({ data: { ...STATE, version: 'cccccccccccc' } });
        render(<LabelTaxonomyCard projectId={5} />);
        await user.click(screen.getByTestId('taxonomy-toggle'));
        await screen.findByTestId('taxonomy-version');
        await user.click(screen.getByTestId('taxonomy-text'));
        await user.paste('- id: KEY-2');
        await user.click(screen.getByTestId('taxonomy-save'));
        await waitFor(() => expect(screen.getByTestId('taxonomy-version')).toHaveTextContent('cccccccccccc'));
        expect(apiMock.put).toHaveBeenCalledWith('/projects/5/label-taxonomy', { content: '- id: KEY-2', format: 'auto' });
    });

    it('restores an earlier version and resolves a label', async () => {
        const user = userEvent.setup();
        apiMock.post.mockResolvedValue({ data: { ...STATE, version: 'bbbbbbbbbbbb', entry_count: 2 } });
        render(<LabelTaxonomyCard projectId={5} />);
        await user.click(screen.getByTestId('taxonomy-toggle'));
        await screen.findByTestId('taxonomy-version');
        await user.click(screen.getByTestId('taxonomy-restore-bbbbbbbbbbbb'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/5/label-taxonomy/restore', { version: 'bbbbbbbbbbbb' }));
        await waitFor(() => expect(screen.getByTestId('taxonomy-count')).toHaveTextContent('2 entries'));
        await user.type(screen.getByTestId('taxonomy-try-input'), 'leak 1');
        await user.click(screen.getByTestId('taxonomy-try-btn'));
        expect(await screen.findByTestId('taxonomy-try-result')).toHaveTextContent('resolves to LEAK-1');
        expect(screen.getByTestId('taxonomy-try-result')).toHaveTextContent('matched by id');
    });

    it('disables restore for versions whose snapshot is no longer kept', async () => {
        const user = userEvent.setup();
        render(<LabelTaxonomyCard projectId={5} />);
        await screen.findByTestId('taxonomy-version');
        await user.click(screen.getByTestId('taxonomy-toggle'));
        expect(screen.getByTestId('taxonomy-stub-dddddddddddd')).toHaveTextContent('no longer kept');
        expect(screen.queryByTestId('taxonomy-restore-dddddddddddd')).toBeNull();
        expect(screen.getByTestId('taxonomy-restore-bbbbbbbbbbbb')).toBeEnabled();
    });

    it('toasts the envelope message when restore, save-without-issues or delete fail', async () => {
        const user = userEvent.setup();
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        const detail = 'Taxonomy version can no longer be restored: its snapshot is no longer kept.';
        apiMock.post.mockRejectedValue({ message: 'Request failed with status code 409', response: { data: { detail } } });
        apiMock.delete.mockRejectedValue({ message: 'Request failed with status code 500', response: { data: { detail: 'Could not clear it.' } } });
        render(<LabelTaxonomyCard projectId={5} />);
        await screen.findByTestId('taxonomy-version');
        await user.click(screen.getByTestId('taxonomy-toggle'));
        await user.click(screen.getByTestId('taxonomy-restore-bbbbbbbbbbbb'));
        await waitFor(() => expect(useToastStore.getState().toasts.some((t) => t.message === detail)).toBe(true));
        await user.click(screen.getByTestId('taxonomy-delete'));
        await waitFor(() => expect(useToastStore.getState().toasts.some((t) => t.message === 'Could not clear it.')).toBe(true));
        expect(useToastStore.getState().toasts.some((t) => t.message.includes('status code'))).toBe(false);
    });

    it('explains an ambiguous label with its candidates', async () => {
        const user = userEvent.setup();
        apiMock.get.mockImplementation(async (url: string) => (url.endsWith('/resolve')
            ? { data: { raw: 'leak 1.1', id: null, matched_by: 'ambiguous', title: null, candidates: ['LEAK-11', 'LEAK-1.1'] } }
            : { data: STATE }));
        render(<LabelTaxonomyCard projectId={5} />);
        await screen.findByTestId('taxonomy-version');
        await user.click(screen.getByTestId('taxonomy-toggle'));
        await user.type(screen.getByTestId('taxonomy-try-input'), 'leak 1.1');
        await user.click(screen.getByTestId('taxonomy-try-btn'));
        expect(await screen.findByTestId('taxonomy-try-result')).toHaveTextContent('is ambiguous');
        expect(screen.getByTestId('taxonomy-try-result')).toHaveTextContent('LEAK-11');
    });
});
