import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import EvalSetsCard from './EvalSetsCard';
import { evalDatasetLabel, evalResultDatasetLabel, evalSetDatasetName, type EvalSet } from '../../api/evalSets';

const SET: EvalSet = {
    id: 1,
    slug: 'fresh',
    name: 'Fresh',
    current_version: 2,
    rows: 3,
    is_locked: false,
    versions: [{ version: 1, rows: 4 }, { version: 2, rows: 3 }],
    variants: [{ name: 'native', spec: { system: 'strip' }, label: 'system message stripped' }],
    dataset_name: 'eval_set:fresh@v2',
    last_evals: { 'eval_set:fresh@v2+native': { eval_result_id: 31, experiment_id: 9, pass_rate: 0.5 } },
};

describe('evalSets helpers', () => {
    it('labels eval-set results with set, version, variant, rows, content hash and prompt format', () => {
        expect(evalSetDatasetName('fresh', 2, 'native')).toBe('eval_set:fresh@v2+native');
        expect(evalDatasetLabel('eval_set:fresh@v2+native')).toBe('eval set fresh v2 (native)');
        expect(evalDatasetLabel('gold_dev')).toBe('gold_dev');
        expect(evalResultDatasetLabel({
            dataset_name: 'eval_set:fresh@v2+native',
            metrics: { prompt_format_version: 2 },
            details: { dataset: { content_sha256: 'abcdef1234567890', eval_set: { name: 'Fresh', rows: 3 } } },
        })).toBe('eval set fresh v2 (native) "Fresh" · 3 rows · content abcdef12 · prompt format v2');
        expect(evalResultDatasetLabel({ dataset_name: 'test', metrics: {}, details: {} })).toBe('test');
    });
});

describe('EvalSetsCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.delete.mockReset();
        toastMock.success.mockReset();
    });

    it('lists sets with version, rows, variants and the evaluated labels', () => {
        render(<EvalSetsCard projectId={4} sets={[SET]} onChanged={vi.fn()} />);
        expect(screen.getByTestId('eval-set-fresh')).toHaveTextContent('fresh · v2 · 3 rows · 2 versions');
        expect(screen.getByTestId('eval-set-variants-fresh')).toHaveTextContent('native (system message stripped)');
        expect(screen.getByTestId('eval-set-evals-fresh')).toHaveTextContent('eval set fresh v2 (native) · last result #31 (run #9) · pass 50.0%');
    });

    it('groups rejected rows by reason and imports after the selected rows are dropped', async () => {
        const onChanged = vi.fn();
        apiMock.post
            .mockRejectedValueOnce({
                response: {
                    data: {
                        error_code: 'EVAL_SET_IMPORT_REJECTED_ROWS',
                        metadata: {
                            rejected: {
                                count: 2,
                                by_reason: {
                                    invalid_json: { label: 'Not valid JSON', count: 1, lines: [2], rows: [{ line: 2, preview: 'oops' }] },
                                    empty_row: { label: 'No content', count: 1, lines: [5], rows: [{ line: 5, preview: '{}' }] },
                                },
                            },
                            row_counts: { usable: 3 },
                        },
                    },
                },
            })
            .mockResolvedValueOnce({ data: { ...SET, slug: 'new', name: 'new', current_version: 1, rows: 3 } });
        const user = userEvent.setup();
        render(<EvalSetsCard projectId={4} sets={[]} onChanged={onChanged} />);
        await user.type(screen.getByTestId('eval-set-name'), 'new');
        await user.upload(screen.getByTestId('eval-set-file'), new File(['x'], 'rows.jsonl'));
        await user.click(screen.getByTestId('eval-set-upload'));
        expect(await screen.findByTestId('eval-set-rejects')).toHaveTextContent('2 rows in rows.jsonl cannot be used');
        expect(screen.getByTestId('eval-set-reject-group-invalid_json')).toHaveTextContent('Not valid JSON (1)');
        await user.click(screen.getByTestId('eval-set-reject-row-2'));
        expect(screen.getByTestId('eval-set-reject-drop')).toHaveTextContent('Drop 1 selected row');
        await user.click(screen.getByTestId('eval-set-reject-group-select-empty_row'));
        await user.click(screen.getByTestId('eval-set-reject-drop'));
        await waitFor(() => expect(onChanged).toHaveBeenCalled());
        const [url, form] = apiMock.post.mock.calls[1];
        expect(url).toBe('/projects/4/eval-sets');
        expect(JSON.parse((form as FormData).get('drop_lines') as string).sort()).toEqual([2, 5]);
        expect((form as FormData).get('name')).toBe('new');
    });

    it('defines a variant, deletes selected rows as a new version and copies a built-in set', async () => {
        const onChanged = vi.fn();
        apiMock.post.mockResolvedValue({ data: { eval_set: SET, deleted: 1, ...SET } });
        apiMock.get.mockResolvedValue({
            data: { slug: 'fresh', version: 2, total: 3, offset: 0, limit: 50,
                rows: [0, 1, 2].map((i) => ({ position: i, row: { question: `q${i}` } })) },
        });
        const user = userEvent.setup();
        render(<EvalSetsCard projectId={4} sets={[SET]} onChanged={onChanged} />);

        await user.click(screen.getByTestId('eval-set-add-variant-fresh'));
        await user.type(screen.getByTestId('eval-set-variant-name'), 'instructed');
        await user.selectOptions(screen.getByTestId('eval-set-variant-mode'), 'replace');
        expect(screen.getByTestId('eval-set-variant-save')).toBeDisabled();
        await user.type(screen.getByTestId('eval-set-variant-text'), 'Answer exactly.');
        await user.click(screen.getByTestId('eval-set-variant-save'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/4/eval-sets/fresh/variants', { name: 'instructed', system: 'replace', text: 'Answer exactly.' },
        ));

        await user.click(screen.getByTestId('eval-set-rows-fresh'));
        await user.click(await screen.findByTestId('eval-set-row-1'));
        await user.click(screen.getByTestId('eval-set-rows-delete'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/4/eval-sets/fresh/rows/delete', { positions: [1], version: 2 },
        ));

        await user.selectOptions(screen.getByTestId('eval-set-copy-split'), 'gold_dev');
        await user.click(screen.getByTestId('eval-set-copy-button'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/4/eval-sets/from-source', { name: 'Practice answer key copy', source: { kind: 'split', split: 'gold_dev' } },
        ));
        expect(onChanged).toHaveBeenCalledTimes(3);
    });

    it('opens the base-vs-fine-tuned view on exactly one set + version + variant', async () => {
        apiMock.get.mockResolvedValue({
            data: { status: 'not_comparable', project_id: 4, dataset_name: 'eval_set:fresh@v2+native',
                message: 'Base and fine-tuned evals cannot be paired: prompt-format variant was redefined.',
                baseline: null, trained: null, metric_lifts: [], gate_status: [] },
        });
        const user = userEvent.setup();
        render(<EvalSetsCard projectId={4} sets={[SET]} onChanged={vi.fn()} />);
        await user.click(screen.getByTestId('eval-set-lift-eval_set:fresh@v2+native'));
        expect(await screen.findByTestId('sft-lift-panel-not-comparable')).toHaveTextContent('redefined');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/4/evaluation/sft-lift-summary',
            { params: { dataset_name: 'eval_set:fresh@v2+native' } });
    });

    it('hides edits on a locked set', () => {
        render(<EvalSetsCard projectId={4} sets={[{ ...SET, is_locked: true }]} onChanged={vi.fn()} />);
        expect(screen.getByTestId('eval-set-fresh')).toHaveTextContent('locked');
        expect(screen.queryByTestId('eval-set-delete-fresh')).toBeNull();
        expect(screen.queryByTestId('eval-set-version-file-fresh')).toBeNull();
        expect(screen.getByTestId('eval-set-add-variant-fresh')).toBeInTheDocument();
    });
});
