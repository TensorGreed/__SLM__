import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import DomainPackContentList from './DomainPackContentList';
import DomainPackImportPanel from './DomainPackImportPanel';

const SUMMARY = {
    taxonomy_entries: 3, probe_packs: 1, probes: 2, critical_probes: 1, rubrics: 0, schemas: 0,
    tool_catalog: false, documents: 1, gold_dev_rows: 4, gold_test_rows: 2, project_template: true,
};
const base = {
    valid: true, errors: [], warnings: [], summary: SUMMARY, manifest: { pack_id: 'toy-pack', version: '1.0.0' },
    content_hash: 'h', existing: null, proposed_version: '1.0.0', unchanged: false, diff: null,
};
const zip = () => new File(['x'], 'toy.zip', { type: 'application/zip' });

async function pick() {
    render(<DomainPackImportPanel />);
    fireEvent.change(screen.getByTestId('dp-import-file'), { target: { files: [zip()] } });
}

describe('DomainPackImportPanel', () => {
    beforeEach(() => {
        Object.values(apiMock).forEach((f) => f.mockReset());
    });

    it('groups errors by code and disables Import while invalid', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                ...base, valid: false,
                errors: [
                    { code: 'missing_file', path: 'a.yaml', message: 'gone' },
                    { code: 'missing_file', path: 'b.yaml', message: 'gone' },
                    { code: 'bad_version', path: 'pack.yaml', message: 'nope' },
                ],
                warnings: [{ code: 'unreferenced_file', path: 'x.txt', message: 'unused' }],
            },
        });
        await pick();
        expect(await screen.findByTestId('dp-import-errors-missing_file')).toHaveTextContent('2');
        expect(screen.getByTestId('dp-import-errors-bad_version')).toBeInTheDocument();
        expect(screen.getByTestId('dp-import-warnings-unreferenced_file')).toBeInTheDocument();
        expect(screen.getByTestId('dp-import-commit')).toBeDisabled();
    });

    it('renders the diff for a re-import and imports', async () => {
        apiMock.post
            .mockResolvedValueOnce({
                data: {
                    ...base, existing: { version: '1.0.0', content_hash: 'o' }, proposed_version: '1.0.1',
                    diff: {
                        version: { from: '1.0.0', to: '1.0.1' },
                        sections: {
                            probes: { added: ['toy-pack/p2'], removed: [], changed: ['toy-pack/p1'] },
                            gold: { dev: { from: 3, to: 4 } },
                        },
                    },
                },
            })
            .mockResolvedValueOnce({
                data: { pack: { pack_id: 'toy-pack' }, version: '1.0.1', unchanged: false, diff: null },
            });
        const onImported = vi.fn();
        render(<DomainPackImportPanel onImported={onImported} />);
        fireEvent.change(screen.getByTestId('dp-import-file'), { target: { files: [zip()] } });
        expect(await screen.findByTestId('dp-diff-probes')).toHaveTextContent('1 added, 0 removed, 1 changed');
        expect(screen.getByTestId('dp-diff-gold-dev')).toHaveTextContent('3 to 4');
        expect(screen.getByTestId('dp-import-ident')).toHaveTextContent('toy-pack@1.0.1');
        fireEvent.click(screen.getByTestId('dp-import-commit'));
        expect(await screen.findByTestId('dp-import-done')).toHaveTextContent('Imported toy-pack@1.0.1');
        expect(onImported).toHaveBeenCalled();
    });

    it('shows a gold change when only the content changed', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                ...base, existing: { version: '1.0.0', content_hash: 'o' }, proposed_version: '1.0.1',
                diff: {
                    version: { from: '1.0.0', to: '1.0.1' },
                    sections: { gold: { dev: { from: 4, to: 4, changed: true }, test: { from: 2, to: 2, changed: false } } },
                },
            },
        });
        await pick();
        expect(await screen.findByTestId('dp-diff-gold-dev')).toHaveTextContent('content changed');
        expect(screen.queryByTestId('dp-diff-gold-test')).toBeNull();
    });

    it('shows import warnings after a successful import', async () => {
        apiMock.post
            .mockResolvedValueOnce({ data: base })
            .mockResolvedValueOnce({
                data: {
                    pack: { pack_id: 'toy-pack' }, version: '1.0.0', unchanged: false, diff: null,
                    warnings: [{ code: 'unreferenced_file', path: 'x.txt', message: 'unused' }],
                },
            });
        await pick();
        fireEvent.click(await screen.findByTestId('dp-import-commit'));
        expect(await screen.findByTestId('dp-import-done-warnings-unreferenced_file')).toBeInTheDocument();
    });

    it('shows the unchanged state and disables Import', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: { ...base, unchanged: true, existing: { version: '1.0.0', content_hash: 'h' } },
        });
        await pick();
        expect(await screen.findByTestId('dp-import-unchanged')).toBeInTheDocument();
        expect(screen.getByTestId('dp-import-commit')).toBeDisabled();
    });

    it('surfaces structured 422 errors from the import call', async () => {
        apiMock.post
            .mockResolvedValueOnce({ data: base })
            .mockRejectedValueOnce({
                response: { data: { detail: 'invalid', metadata: { errors: [{ code: 'bad_gold_row', message: 'line 2' }] } } },
            });
        await pick();
        await screen.findByTestId('dp-import-summary');
        fireEvent.click(screen.getByTestId('dp-import-commit'));
        expect(await screen.findByTestId('dp-import-errors-bad_gold_row')).toBeInTheDocument();
    });
});

describe('DomainPackContentList', () => {
    beforeEach(() => {
        Object.values(apiMock).forEach((f) => f.mockReset());
    });

    const content = {
        pack_id: 'toy-pack', version: '1.0.1',
        versions: [{ version: '1.0.0' }, { version: '1.0.1' }],
        manifest: {}, summary: SUMMARY, files: [],
    };

    function mockList() {
        apiMock.get.mockImplementation((url: string) => {
            if (url === '/domain-packs') {
                return Promise.resolve({ data: { packs: [
                    { pack_id: 'sys', is_system: true }, { pack_id: 'toy-pack', is_system: false },
                    { pack_id: 'plain', is_system: false },
                ] } });
            }
            if (url === '/domain-packs/toy-pack/content') return Promise.resolve({ data: content });
            return Promise.reject({ response: { status: 404 } });
        });
    }

    it('lists only packs with content and instantiates then navigates', async () => {
        mockList();
        const assign = vi.fn();
        Object.defineProperty(window, 'location', { value: { ...window.location, assign }, writable: true });
        apiMock.post.mockResolvedValueOnce({ data: { project: { id: 99 }, summary: { warnings: [] } } });
        render(<DomainPackContentList projectId={5} />);
        expect(await screen.findByTestId('dp-content-toy-pack')).toHaveTextContent('toy-pack@1.0.1');
        expect(screen.queryByTestId('dp-content-plain')).toBeNull();
        fireEvent.change(screen.getByLabelText('Project name for toy-pack'), { target: { value: 'My proj' } });
        fireEvent.change(screen.getByLabelText('Version of toy-pack'), { target: { value: '1.0.0' } });
        fireEvent.click(screen.getByTestId('dp-create-toy-pack'));
        await waitFor(() => expect(assign).toHaveBeenCalledWith('/project/99'));
        expect(apiMock.post).toHaveBeenCalledWith('/domain-packs/toy-pack/instantiate', {
            project_name: 'My proj', version: '1.0.0',
        });
    });

    it('exports the current project with the chosen document scope', async () => {
        mockList();
        const blob = new Blob(['z']);
        (URL as unknown as Record<string, unknown>).createObjectURL = vi.fn(() => 'blob:x');
        (URL as unknown as Record<string, unknown>).revokeObjectURL = vi.fn();
        render(<DomainPackContentList projectId={5} />);
        await screen.findByTestId('dp-content-toy-pack');
        apiMock.get.mockResolvedValueOnce({
            data: blob, headers: { 'content-disposition': 'attachment; filename="toy-pack-1.0.2.zip"' },
        });
        fireEvent.change(screen.getByTestId('dp-export-docs'), { target: { value: 'all' } });
        fireEvent.click(screen.getByTestId('dp-export'));
        await waitFor(() =>
            expect(apiMock.get).toHaveBeenCalledWith('/projects/5/domain-pack/export', {
                params: { include_documents: 'all' }, responseType: 'blob',
            }),
        );
    });
});
