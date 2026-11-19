import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import GoldFileManager from './GoldFileManager';

const LABEL = { id: 9, name: 'Gold Dev Set', dataset_type: 'gold_dev' as const, record_count: 3, is_locked: false };

const REJECTED = {
    response: {
        status: 422,
        data: {
            error_code: 'GOLD_IMPORT_REJECTED_ROWS',
            metadata: {
                rejected: {
                    count: 3,
                    by_reason: {
                        invalid_json: { label: 'Not valid JSON', count: 2, lines: [2, 5], rows: [{ line: 2, preview: '{bad' }, { line: 5, preview: 'oops' }] },
                        empty_row: { label: 'No content (every field is empty)', count: 1, lines: [3], rows: [{ line: 3, preview: '{}' }] },
                    },
                },
                row_counts: { usable: 4, rejected: 3 },
            },
        },
    },
};

function pickFile(name = 'gold.jsonl') {
    const input = screen.getByTestId('gold-file-input') as HTMLInputElement;
    const file = new File(['{"text":"a","label":"x"}\n'], name, { type: 'text/plain' });
    return userEvent.upload(input, file);
}

describe('GoldFileManager', () => {
    beforeEach(() => {
        apiMock.get.mockReset(); apiMock.post.mockReset(); apiMock.delete.mockReset();
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    it('labels the set with dataset, type and row count', () => {
        render(<GoldFileManager projectId={5} datasetType="gold_dev" label={LABEL} onChanged={vi.fn()} />);
        expect(screen.getByTestId('gold-set-label')).toHaveTextContent('Practice answer key (gold_dev)');
        expect(screen.getByTestId('gold-set-label')).toHaveTextContent('Gold Dev Set');
        expect(screen.getByTestId('gold-set-label')).toHaveTextContent('3 rows');
    });

    it('imports a file (append) and refreshes', async () => {
        apiMock.post.mockResolvedValue({ data: { imported: 2, total: 5, mode: 'append', source_filename: 'gold.jsonl', dropped_rows: 0, dropped_lines: [], dataset: LABEL } });
        const onChanged = vi.fn();
        const user = userEvent.setup();
        render(<GoldFileManager projectId={5} datasetType="gold_test" label={{ ...LABEL, dataset_type: 'gold_test' }} onChanged={onChanged} />);
        await pickFile();
        await user.click(screen.getByTestId('gold-file-import'));
        await waitFor(() => expect(onChanged).toHaveBeenCalled());
        const [url, form] = apiMock.post.mock.calls[0];
        expect(url).toBe('/projects/5/gold/import-file');
        expect((form as FormData).get('dataset_type')).toBe('gold_test');
        expect((form as FormData).get('mode')).toBe('append');
        expect((form as FormData).get('drop_lines')).toBeNull();
    });

    it('replace asks first, and sends mode=replace', async () => {
        apiMock.post.mockResolvedValue({ data: { imported: 1, total: 1, mode: 'replace', source_filename: 'gold.jsonl', dropped_rows: 0, dropped_lines: [], dataset: LABEL } });
        const user = userEvent.setup();
        render(<GoldFileManager projectId={5} datasetType="gold_dev" label={LABEL} onChanged={vi.fn()} />);
        await pickFile();
        await user.selectOptions(screen.getByTestId('gold-file-mode'), 'replace');
        await user.click(screen.getByTestId('gold-file-import'));
        expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('Replace all 3 rows'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect((apiMock.post.mock.calls[0][1] as FormData).get('mode')).toBe('replace');
    });

    it('groups rejected rows by reason, lets the user select and bulk-drop, then imports', async () => {
        apiMock.post.mockRejectedValueOnce(REJECTED);
        const user = userEvent.setup();
        const onChanged = vi.fn();
        render(<GoldFileManager projectId={5} datasetType="gold_dev" label={LABEL} onChanged={onChanged} />);
        await pickFile();
        await user.click(screen.getByTestId('gold-file-import'));

        expect(await screen.findByTestId('gold-reject-group-invalid_json')).toHaveTextContent('Not valid JSON (2)');
        expect(screen.getByTestId('gold-reject-group-empty_row')).toHaveTextContent('line 3');
        expect(screen.getByTestId('gold-file-rejects')).toHaveTextContent('3 rows in gold.jsonl cannot be used');
        expect(screen.getByTestId('gold-reject-drop')).toBeDisabled();

        // one row, then a whole group: the server is told exactly those lines
        apiMock.post.mockRejectedValueOnce({
            response: { status: 422, data: { error_code: 'GOLD_IMPORT_REJECTED_ROWS', metadata: { rejected: { count: 1, by_reason: { empty_row: REJECTED.response.data.metadata.rejected.by_reason.empty_row } }, row_counts: { usable: 4 } } } },
        });
        await user.click(screen.getByTestId('gold-reject-group-select-invalid_json'));
        await user.click(screen.getByTestId('gold-reject-drop'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
        expect(JSON.parse((apiMock.post.mock.calls[1][1] as FormData).get('drop_lines') as string).sort()).toEqual([2, 5]);
        expect(await screen.findByTestId('gold-file-rejects')).toHaveTextContent('1 row in gold.jsonl cannot be used');
        expect(screen.queryByTestId('gold-reject-group-invalid_json')).toBeNull();

        apiMock.post.mockResolvedValueOnce({ data: { imported: 4, total: 7, mode: 'append', source_filename: 'gold.jsonl', dropped_rows: 3, dropped_lines: [2, 3, 5], dataset: LABEL } });
        await user.click(screen.getByTestId('gold-reject-row-3'));
        await user.click(screen.getByTestId('gold-reject-drop'));
        await waitFor(() => expect(onChanged).toHaveBeenCalled());
        expect(JSON.parse((apiMock.post.mock.calls[2][1] as FormData).get('drop_lines') as string).sort()).toEqual([2, 3, 5]);
        expect(screen.queryByTestId('gold-file-rejects')).toBeNull();
    });

    it('clears the set after confirming', async () => {
        apiMock.delete.mockResolvedValue({ data: { deleted: 3, remaining: 0, dataset: LABEL } });
        const onChanged = vi.fn();
        const user = userEvent.setup();
        render(<GoldFileManager projectId={5} datasetType="gold_dev" label={LABEL} onChanged={onChanged} />);
        await user.click(screen.getByTestId('gold-clear-set'));
        await waitFor(() => expect(apiMock.delete).toHaveBeenCalledWith('/projects/5/gold/gold_dev'));
        expect(onChanged).toHaveBeenCalled();
    });

    it('shows a locked set as read-only', () => {
        render(<GoldFileManager projectId={5} datasetType="gold_test" label={{ ...LABEL, is_locked: true }} onChanged={vi.fn()} />);
        expect(screen.getByTestId('gold-files-locked')).toBeInTheDocument();
        expect(screen.queryByTestId('gold-file-input')).toBeNull();
    });

    it('surfaces a server error', async () => {
        apiMock.post.mockRejectedValue({ response: { status: 409, data: { error_code: 'GOLD_LOCKED', message: 'Gold dataset is locked.' } } });
        const user = userEvent.setup();
        render(<GoldFileManager projectId={5} datasetType="gold_dev" label={LABEL} onChanged={vi.fn()} />);
        await pickFile();
        await user.click(screen.getByTestId('gold-file-import'));
        expect(await screen.findByTestId('gold-file-error')).toHaveTextContent('Gold dataset is locked.');
    });
});
