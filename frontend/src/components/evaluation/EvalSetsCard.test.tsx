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
import { displayPassRate, evalDatasetLabel, evalResultDatasetLabel, evalSetDatasetName, lastEvalText, type EvalSet } from '../../api/evalSets';

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
    it('shows a real 0.0 pass rate as 0, keeps the stored value otherwise', () => {
        expect(displayPassRate({ pass_rate: 0.4, metrics: { exact_match: 0, f1: 0.4 } })).toBe(0.4);
        expect(displayPassRate({ pass_rate: null, metrics: { exact_match: 0 } })).toBe(0);
        expect(displayPassRate({ pass_rate: null, metrics: { rouge_l: 0.3 } })).toBeNull();
    });

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
        expect(evalResultDatasetLabel({
            dataset_name: 'eval_set:tv2@v1',
            metrics: { prompt_format_version: 2 },
            details: { dataset: { eval_set: { rows: 500, kept_despite_leak: { count: 2, row_ids: ['q-7', '#12'] } } } },
        })).toBe('eval set tv2 v1 · 500 rows · prompt format v2 · 2 rows kept despite a training-data leak flag (q-7, #12)');
    });

    it('names the model that answered and shows both sides of base vs fine-tuned on a set (P5 gap 8)', () => {
        expect(evalResultDatasetLabel({
            dataset_name: 'eval_set:fresh@v1+native',
            metrics: { prompt_format_version: 2, evaluated_model: { model_path: 'org/base-1b', kind: 'base' } },
            details: {},
        })).toBe('eval set fresh v1 (native) · prompt format v2 · answered by the base model org/base-1b');
        expect(evalResultDatasetLabel({
            dataset_name: 'test', metrics: { evaluated_model: { model_path: '/m/x', kind: 'custom' } }, details: {},
        })).toBe('test · answered by model path /m/x');
        expect(evalResultDatasetLabel({
            dataset_name: 'test',
            metrics: { evaluated_model: { model_path: 'org/base-1b', kind: 'base', filed_under_baseline: 3,
                generation: { max_new_tokens: 128, temperature: 0, decoding: 'neutral' } } },
            details: {},
        })).toBe('test · answered by the base model org/base-1b (max 128 new tokens, temperature 0, neutral decoding)');
        expect(lastEvalText({ eval_result_id: 265, experiment_id: 214, pass_rate: 0.877,
            base: { eval_result_id: 263, experiment_id: 215, pass_rate: 0.785, base_model: 'org/base-1b' } }))
            .toBe('last result #265 (run #214) · pass 87.7% · base model org/base-1b: pass 78.5% (#263)');
        expect(lastEvalText({ eval_result_id: 263, experiment_id: 215, pass_rate: 0.785, baseline: true,
            base: { eval_result_id: 263, experiment_id: 215, pass_rate: 0.785 } }))
            .toBe('last result #263 (base model, run #215) · pass 78.5%');
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

    it('keeps selected leak-flagged rows on purpose (never other rejects) and shows it on the set', async () => {
        const onChanged = vi.fn();
        const rejected = {
            response: {
                data: {
                    error_code: 'EVAL_SET_IMPORT_REJECTED_ROWS',
                    metadata: {
                        rejected: {
                            count: 3,
                            by_reason: {
                                leaks_training_data: { label: 'Leaks training data', count: 2, lines: [1, 3],
                                    rows: [{ line: 1, preview: 'a' }, { line: 3, preview: 'b' }] },
                                invalid_json: { label: 'Not valid JSON', count: 1, lines: [4], rows: [{ line: 4, preview: 'oops' }] },
                            },
                        },
                        row_counts: { usable: 2 },
                    },
                },
            },
        };
        const kept = { count: 2, row_indexes: [0, 2], row_ids: ['#1', '#3'], lines: [1, 3] };
        apiMock.post
            .mockRejectedValueOnce(rejected)
            .mockRejectedValueOnce({ response: { data: { ...rejected.response.data, metadata: {
                rejected: { count: 1, by_reason: { invalid_json: rejected.response.data.metadata.rejected.by_reason.invalid_json } },
                row_counts: { usable: 4 } } } } })
            .mockResolvedValueOnce({ data: { ...SET, slug: 'tv2', name: 'tv2', current_version: 1, rows: 4,
                versions: [{ version: 1, rows: 4, kept_despite_leak: kept }] } });
        const user = userEvent.setup();
        render(<EvalSetsCard projectId={4} sets={[]} onChanged={onChanged} />);
        await user.type(screen.getByTestId('eval-set-name'), 'tv2');
        await user.upload(screen.getByTestId('eval-set-file'), new File(['x'], 'rows.jsonl'));
        await user.click(screen.getByTestId('eval-set-upload'));
        await screen.findByTestId('eval-set-rejects');
        // a non-leak reject cannot be kept
        await user.click(screen.getByTestId('eval-set-reject-row-4'));
        expect(screen.getByTestId('eval-set-reject-keep')).toBeDisabled();
        await user.click(screen.getByTestId('eval-set-reject-row-4'));
        await user.click(screen.getByTestId('eval-set-reject-group-select-leaks_training_data'));
        expect(screen.getByTestId('eval-set-reject-keep')).toHaveTextContent('Keep 2 selected leak-flagged rows anyway');
        await user.click(screen.getByTestId('eval-set-reject-keep'));
        expect(await screen.findByTestId('eval-set-rejects')).toHaveTextContent('2 kept despite a leak flag');
        expect(screen.queryByTestId('eval-set-reject-keep')).toBeNull();  // no leak group left
        await user.click(screen.getByTestId('eval-set-reject-row-4'));
        await user.click(screen.getByTestId('eval-set-reject-drop'));
        await waitFor(() => expect(onChanged).toHaveBeenCalled());
        const keepForm = apiMock.post.mock.calls[1][1] as FormData;
        expect(JSON.parse(keepForm.get('keep_leak_lines') as string)).toEqual([1, 3]);
        expect(keepForm.get('drop_lines')).toBeNull();
        const finalForm = apiMock.post.mock.calls[2][1] as FormData;
        expect(JSON.parse(finalForm.get('keep_leak_lines') as string)).toEqual([1, 3]);
        expect(JSON.parse(finalForm.get('drop_lines') as string)).toEqual([4]);
        expect(toastMock.success.mock.calls[0][0]).toContain('2 kept despite a leak flag');
    });

    it('shows a version kept despite leak flags in the list', () => {
        const set = { ...SET, current_version: 1, versions: [{ version: 1, rows: 3, kept_despite_leak: { count: 1, row_ids: ['q-boil'] } }] };
        render(<EvalSetsCard projectId={4} sets={[set]} onChanged={vi.fn()} />);
        expect(screen.getByTestId('eval-set-fresh')).toHaveTextContent('1 row kept despite a training-data leak flag (q-boil)');
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

    it('shows a row\'s own context in the row view', async () => {
        apiMock.get.mockResolvedValue({
            data: { slug: 'fresh', version: 2, total: 2, offset: 0, limit: 50, rows: [
                { position: 0, row: { question: 'q0', row_context: { system: 'Follow the house style.' } } },
                { position: 1, row: { question: 'q1' } },
            ] },
        });
        const user = userEvent.setup();
        render(<EvalSetsCard projectId={4} sets={[SET]} onChanged={vi.fn()} />);
        await user.click(screen.getByTestId('eval-set-rows-fresh'));
        expect(await screen.findByTestId('eval-set-row-context-0')).toHaveTextContent(/system message/i);
        expect(screen.queryByTestId('eval-set-row-context-1')).toBeNull();
    });

    it('labels an output schema as used only where the backend says the row renders it', async () => {
        const schema = { type: 'object', properties: { name: { type: 'string' } } };
        const page = (reads: string[], mode: string | null, profile: string) => ({
            data: {
                slug: 'fresh', version: 2, total: 2, offset: 0, limit: 50,
                effective_task_profile: profile, row_context_reads: reads, row_context_schema_mode: mode,
                rows: [
                    { position: 0, row: { question: 'Who is 30?', answer: '{"name": "Ann"}', row_context: { output_schema: schema } } },
                    { position: 1, row: { messages: [{ role: 'user', content: 'Hi' }, { role: 'assistant', content: '{}' }],
                        row_context: JSON.stringify({ output_schema: schema }) } },
                ],
            },
        });
        const user = userEvent.setup();
        // qa-handler task ("lead"): a prompt/completion row renders it, a chat row does not.
        apiMock.get.mockResolvedValue(page(['system', 'output_schema'], 'lead', 'qa'));
        const { unmount } = render(<EvalSetsCard projectId={4} sets={[SET]} onChanged={vi.fn()} />);
        await user.click(screen.getByTestId('eval-set-rows-fresh'));
        expect(await screen.findByTestId('eval-set-row-context-0')).toHaveTextContent('Own output schema (1 field)');
        expect(screen.getByTestId('eval-set-row-context-1')).toHaveTextContent('not used by this task: output_schema');
        expect(screen.getByTestId('eval-set-row-context-1')).not.toHaveTextContent('Own output schema');
        unmount();
        // a task that does not read it at all (e.g. classification): neither row claims it.
        apiMock.get.mockResolvedValue(page(['system'], null, 'classification'));
        render(<EvalSetsCard projectId={4} sets={[SET]} onChanged={vi.fn()} />);
        await user.click(screen.getByTestId('eval-set-rows-fresh'));
        expect(await screen.findByTestId('eval-set-row-context-0')).toHaveTextContent('not used by this task: output_schema');
        expect(screen.getByTestId('eval-set-row-context-0')).not.toHaveTextContent('Own output schema');
    });

    it('copies a built-in set with a prompt-format variant in one call', async () => {
        const onChanged = vi.fn();
        apiMock.post.mockResolvedValue({ data: { ...SET, slug: 'test-copy', name: 'Test examples copy' } });
        const user = userEvent.setup();
        render(<EvalSetsCard projectId={4} sets={[]} onChanged={onChanged} />);
        await user.click(screen.getByTestId('eval-set-copy-with-variant'));
        expect(screen.getByTestId('eval-set-copy-button')).toBeDisabled();
        await user.type(screen.getByTestId('eval-set-copy-variant-name'), 'native');
        await user.click(screen.getByTestId('eval-set-copy-button'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/4/eval-sets/from-source', {
            name: 'Test examples copy', source: { kind: 'split', split: 'test' }, variant: { name: 'native', system: 'strip' },
        }));
        expect(onChanged).toHaveBeenCalled();
    });

    it('lists rows that leak training data as their own selectable reject group', async () => {
        apiMock.post.mockRejectedValueOnce({
            response: { data: { error_code: 'EVAL_SET_IMPORT_REJECTED_ROWS', metadata: {
                rejected: { count: 1, by_reason: { leaks_training_data: {
                    label: 'Leaks training data', count: 1, lines: [3],
                    rows: [{ line: 3, preview: '{"question": "q"} | matches a training row (exact): q' }] } } },
                row_counts: { usable: 2 } } } },
        });
        const user = userEvent.setup();
        render(<EvalSetsCard projectId={4} sets={[]} onChanged={vi.fn()} />);
        await user.type(screen.getByTestId('eval-set-name'), 'new');
        await user.upload(screen.getByTestId('eval-set-file'), new File(['x'], 'rows.jsonl'));
        await user.click(screen.getByTestId('eval-set-upload'));
        expect(await screen.findByTestId('eval-set-reject-group-leaks_training_data')).toHaveTextContent('Leaks training data (1)');
        expect(screen.getByTestId('eval-set-reject-row-3')).not.toBeChecked();
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

    it('offers the base model on another variant of the same version as a labelled cross-variant pair', async () => {
        const label = 'Cross-variant pair on eval set fresh v2: the base model answered the \'instructed\' variant, '
            + 'the fine-tuned run the \'native\' variant. Both the model and the prompt differ, so the change is not '
            + 'the model\'s alone; only rows both evaluated are paired.';
        apiMock.get.mockResolvedValue({
            data: { status: 'ok', project_id: 4, dataset_name: 'eval_set:fresh@v2+native',
                base_dataset_name: 'eval_set:fresh@v2+instructed',
                cross_variant: { slug: 'fresh', version: 2, variant_a: 'instructed', variant_b: 'native', label },
                baseline: { experiment_id: 3, experiment_name: 'Baseline', base_model: 'b', training_mode: null,
                    completed_at: null, eval_result_id: 40, dataset_name: 'eval_set:fresh@v2+instructed',
                    eval_type: 'exact_match', metrics: { exact_match: 0.5 }, pass_rate: 0.5 },
                trained: { experiment_id: 9, experiment_name: 'Run', base_model: 'b', training_mode: 'lora',
                    completed_at: null, eval_result_id: 31, dataset_name: 'eval_set:fresh@v2+native',
                    eval_type: 'exact_match', metrics: { exact_match: 1 }, pass_rate: 1 },
                metric_lifts: [], gate_status: [] },
        });
        const sets = [{ ...SET, last_evals: {
            'eval_set:fresh@v2+native': { eval_result_id: 31, experiment_id: 9, pass_rate: 1 },
            'eval_set:fresh@v2+instructed': { eval_result_id: 40, experiment_id: 3, pass_rate: 0.5, baseline: true },
            'eval_set:fresh@v1+plain': { eval_result_id: 20, experiment_id: 3, pass_rate: 0.5, baseline: true },
        } }];
        const user = userEvent.setup();
        render(<EvalSetsCard projectId={4} sets={sets as EvalSet[]} onChanged={vi.fn()} />);
        await user.click(screen.getByTestId('eval-set-lift-eval_set:fresh@v2+native'));
        const picker = await screen.findByLabelText("Base model's prompt format");
        // only the other variant of the SAME version is offered (v1 is another version)
        expect(Array.from((picker as HTMLSelectElement).options).map((o) => o.value))
            .toEqual(['eval_set:fresh@v2+native', 'eval_set:fresh@v2+instructed']);
        await user.selectOptions(picker, 'eval_set:fresh@v2+instructed');
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith('/projects/4/evaluation/sft-lift-summary',
            { params: { dataset_name: 'eval_set:fresh@v2+native', base_dataset_name: 'eval_set:fresh@v2+instructed' } }));
        expect(await screen.findByTestId('sft-lift-cross-variant')).toHaveTextContent('Both the model and the prompt differ');
    });

    it('hides edits on a locked set', () => {
        render(<EvalSetsCard projectId={4} sets={[{ ...SET, is_locked: true }]} onChanged={vi.fn()} />);
        expect(screen.getByTestId('eval-set-fresh')).toHaveTextContent('locked');
        expect(screen.queryByTestId('eval-set-delete-fresh')).toBeNull();
        expect(screen.queryByTestId('eval-set-version-file-fresh')).toBeNull();
        expect(screen.getByTestId('eval-set-add-variant-fresh')).toBeInTheDocument();
    });
});
