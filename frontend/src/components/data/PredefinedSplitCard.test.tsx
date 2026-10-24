import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import PredefinedSplitCard from './PredefinedSplitCard';

const file = (name: string) => new File(['{"question":"q","answer":"a"}\n'], name, { type: 'application/json' });

const OK = {
    split_source: 'predefined',
    total_entries: 9,
    splits: { train: 6, val: 1, test: 2 },
    file_hashes: {},
    dataset_versions: { train: 3, val: 3, test: 3 },
    prepared_version: 3,
    source_files: {
        train: { filename: 'my_train.jsonl' },
        test: { filename: 'my_test.jsonl' },
    },
    duplicate_report: { train: 2, val: 0, test: 0 },
    leak_report: { on_leak: 'drop_from_train', pairs: [], dropped_from_train: 2, dropped_sample_rows: [4, 7] },
};

describe('PredefinedSplitCard', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
    });

    it('needs a train file, posts multipart with the leak choice, and shows the provenance result', async () => {
        const user = userEvent.setup();
        render(<PredefinedSplitCard projectId={5} />);
        expect(screen.getByTestId('pdsplit-submit')).toBeDisabled();
        apiMock.post.mockResolvedValue({ data: OK });
        await user.upload(screen.getByTestId('pdsplit-train'), file('my_train.jsonl'));
        await user.upload(screen.getByTestId('pdsplit-test'), file('my_test.jsonl'));
        await user.selectOptions(screen.getByTestId('pdsplit-onleak'), 'drop_from_train');
        await user.click(screen.getByTestId('pdsplit-submit'));

        await waitFor(() => expect(screen.getByTestId('pdsplit-result')).toBeInTheDocument());
        const [url, body] = apiMock.post.mock.calls[0];
        expect(url).toBe('/projects/5/dataset/predefined-split');
        expect((body as FormData).get('on_leak')).toBe('drop_from_train');
        expect(((body as FormData).get('train') as File).name).toBe('my_train.jsonl');
        expect((body as FormData).get('val')).toBeNull();
        const result = screen.getByTestId('pdsplit-result');
        expect(result).toHaveTextContent('dataset version 3');
        expect(result).toHaveTextContent('my_test.jsonl');
        expect(result).toHaveTextContent('not provided');
        expect(screen.getByTestId('pdsplit-leak')).toHaveTextContent('Dropped 2 train row(s)');
        expect(screen.getByTestId('pdsplit-leak')).toHaveTextContent('lines 4, 7');
    });

    it('shows grouped reasons with row numbers on a 409 leak', async () => {
        const user = userEvent.setup();
        apiMock.post.mockRejectedValue({
            response: {
                status: 409,
                data: {
                    error_code: 'PREDEFINED_SPLIT_LEAKAGE',
                    stage: 'dataset',
                    message: 'Rows in the train file also appear in val/test; nothing was imported.',
                    actionable_fix: 'Remove the overlap yourself.',
                    troubleshooting_id: 'err_x',
                    metadata: {
                        pairs: [{ pair: 'train_test', counts: { train: 2, test: 2 }, sample_rows: { train: [3, 9], test: [1, 2] } }],
                    },
                },
            },
        });
        render(<PredefinedSplitCard projectId={5} />);
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-submit'));
        const problem = await screen.findByTestId('pdsplit-problem');
        expect(problem).toHaveTextContent('nothing was imported');
        expect(problem).toHaveTextContent('Train file rows that also appear in Test: 2 row(s) (lines 3, 9)');
    });

    it('shows rejected rows grouped by file and reason on a 422', async () => {
        const user = userEvent.setup();
        apiMock.post.mockRejectedValue({
            response: {
                status: 422,
                data: {
                    error_code: 'PREDEFINED_SPLIT_ROWS_REJECTED',
                    message: '2 row(s) could not be mapped',
                    actionable_fix: 'Fix them.',
                    troubleshooting_id: 'err_y',
                    metadata: {
                        rejected: { test: { count: 2, by_reason: { adapter_unmappable_or_empty: { count: 2, sample_rows: [5, 8] } } } },
                    },
                },
            },
        });
        render(<PredefinedSplitCard projectId={5} />);
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-submit'));
        const problem = await screen.findByTestId('pdsplit-problem');
        expect(problem).toHaveTextContent('Test file, adapter unmappable or empty: 2 row(s) (lines 5, 8)');
    });

    it('shows the unparseable-line and review-status reasons by file line', async () => {
        const user = userEvent.setup();
        apiMock.post.mockRejectedValue({
            response: {
                status: 422,
                data: {
                    error_code: 'PREDEFINED_SPLIT_ROWS_REJECTED',
                    message: '3 row(s) could not be used',
                    actionable_fix: 'Fix them.',
                    troubleshooting_id: 'err_z',
                    metadata: {
                        rejected: {
                            train: {
                                count: 3,
                                by_reason: {
                                    unparseable_line: { count: 2, sample_rows: [3, 5] },
                                    review_status_pending: { count: 1, sample_rows: [9] },
                                },
                            },
                        },
                    },
                },
            },
        });
        render(<PredefinedSplitCard projectId={5} />);
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-submit'));
        const problem = await screen.findByTestId('pdsplit-problem');
        expect(problem).toHaveTextContent('Train file, unparseable line: 2 row(s) (lines 3, 5)');
        expect(problem).toHaveTextContent('Train file, review status pending: 1 row(s) (lines 9)');
    });

    it('offers the split form chat template and posts the choice', async () => {
        const user = userEvent.setup();
        const onChange = vi.fn();
        apiMock.post.mockResolvedValue({ data: OK });
        render(
            <PredefinedSplitCard
                projectId={5}
                chatTemplates={[{ value: 'llama3', label: 'Llama 3' }, { value: 'chatml', label: 'ChatML' }]}
                chatTemplate="chatml"
                onChatTemplateChange={onChange}
            />,
        );
        expect(screen.getByTestId('pdsplit-template')).toHaveValue('chatml');
        await user.selectOptions(screen.getByTestId('pdsplit-template'), 'llama3');
        expect(onChange).toHaveBeenCalledWith('llama3');
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect((apiMock.post.mock.calls[0][1] as FormData).get('chat_template')).toBe('chatml');
    });

    it('sends no chat template while the profile default applies', async () => {
        const user = userEvent.setup();
        apiMock.post.mockResolvedValue({ data: OK });
        render(
            <PredefinedSplitCard
                projectId={5}
                chatTemplates={[{ value: 'llama3', label: 'Llama 3' }]}
                chatTemplate="llama3"
                sendChatTemplate={false}
                onChatTemplateChange={() => undefined}
            />,
        );
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect((apiMock.post.mock.calls[0][1] as FormData).get('chat_template')).toBeNull();
    });

    it('asks for a field mapping on an unrecognised input column and resubmits with it', async () => {
        const user = userEvent.setup();
        apiMock.post.mockRejectedValueOnce({
            response: {
                status: 422,
                data: {
                    error_code: 'PREDEFINED_SPLIT_ROWS_REJECTED',
                    message: 'Could not tell which column is the input (columns: utterance, reply)',
                    actionable_fix: 'Set a field mapping.',
                    troubleshooting_id: 'err_m',
                    metadata: {
                        reason: 'input_column_unrecognised',
                        columns: ['utterance', 'reply'],
                        rejected: { train: { count: 4, by_reason: { input_column_unrecognised: { count: 4, sample_rows: [1, 2, 3, 4] } } } },
                    },
                },
            },
        });
        render(<PredefinedSplitCard projectId={5} />);
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-submit'));
        const mapper = await screen.findByTestId('pdsplit-mapper');
        expect(mapper).toHaveTextContent('utterance, reply');
        expect(screen.getByTestId('pdsplit-map-input')).toHaveValue('utterance');
        expect(screen.getByTestId('pdsplit-map-answer')).toHaveValue('reply');
        apiMock.post.mockResolvedValueOnce({ data: OK });
        await user.click(screen.getByTestId('pdsplit-map-apply'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
        const body = apiMock.post.mock.calls[1][1] as FormData;
        expect(JSON.parse(String(body.get('field_mapping')))).toEqual({ question: 'utterance', answer: 'reply' });
    });

    it('offers to drop the current test file when none is chosen and sends the confirmation', async () => {
        const user = userEvent.setup();
        apiMock.post.mockResolvedValue({ data: { ...OK, dropped_existing_test_document_ids: [7] } });
        render(<PredefinedSplitCard projectId={5} ownTestRows={44} />);
        const box = screen.getByTestId('pdsplit-drop-test');
        expect(screen.getByText(/Drop my current test file \(44 rows\)/)).toBeInTheDocument();
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(box);
        await user.click(screen.getByTestId('pdsplit-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect((apiMock.post.mock.calls[0][1] as FormData).get('drop_existing_test')).toBe('true');
        expect(await screen.findByTestId('pdsplit-dropped-test')).toBeInTheDocument();
        // choosing a test file hides the option and never sends it
        await user.upload(screen.getByTestId('pdsplit-test'), file('e.jsonl'));
        expect(screen.queryByTestId('pdsplit-drop-test')).toBeNull();
        await user.click(screen.getByTestId('pdsplit-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
        expect((apiMock.post.mock.calls[1][1] as FormData).get('drop_existing_test')).toBeNull();
    });

    it('shows no drop option when the project has no own test file', () => {
        render(<PredefinedSplitCard projectId={5} />);
        expect(screen.queryByTestId('pdsplit-drop-test')).toBeNull();
    });
});

describe('PredefinedSplitCard column mapper', () => {
    const unrecognised = (columns: string[]) => ({
        response: {
            status: 422,
            data: {
                error_code: 'PREDEFINED_SPLIT_ROWS_REJECTED',
                message: 'Could not tell which column is the input',
                actionable_fix: 'Set a field mapping',
                troubleshooting_id: 'err_1',
                metadata: { reason: 'input_column_unrecognised', columns },
            },
        },
    });

    async function openMapper(columns: string[], adapter?: Record<string, unknown>) {
        const user = userEvent.setup();
        apiMock.post.mockReset();
        apiMock.post.mockRejectedValueOnce(unrecognised(columns));
        render(<PredefinedSplitCard projectId={5} adapter={adapter as never} />);
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-submit'));
        await screen.findByTestId('pdsplit-mapper');
        return user;
    }

    const select = (id: string) => screen.getByTestId(id) as HTMLSelectElement;

    it('never preselects an id or target-named column as the input, and the answer differs from it', async () => {
        await openMapper(['id', 'utterance', 'intent']);
        expect(select('pdsplit-map-input').value).toBe('utterance');
        expect(select('pdsplit-map-answer').value).toBe('intent');
    });

    it('does not preselect label / idx columns as the input', async () => {
        await openMapper(['idx', 'label', 'sentence']);
        expect(select('pdsplit-map-input').value).toBe('sentence');
        expect(select('pdsplit-map-answer').value).toBe('label');
    });

    it('clears the answer when the user picks it as the input, and never sends question == answer', async () => {
        const user = await openMapper(['id', 'utterance', 'intent']);
        await user.selectOptions(select('pdsplit-map-input'), 'intent');
        expect(select('pdsplit-map-answer').value).toBe('');
        apiMock.post.mockResolvedValueOnce({ data: OK });
        await user.click(screen.getByTestId('pdsplit-map-apply'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
        const mapping = JSON.parse(String((apiMock.post.mock.calls[1][1] as FormData).get('field_mapping')));
        expect(mapping).toEqual({ question: 'intent' });
    });

    it('sends the project adapter wiring with the mapping, so the adapter is kept', async () => {
        const user = await openMapper(['idx', 'sentence', 'label'], {
            adapterId: 'classification-label',
            taskProfile: 'classification',
            adapterConfig: { text_fields: ['sentence'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: OK });
        await user.click(screen.getByTestId('pdsplit-map-apply'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
        const form = apiMock.post.mock.calls[1][1] as FormData;
        expect(JSON.parse(String(form.get('field_mapping')))).toEqual({ text: 'sentence', answer: 'label' });
        expect(form.get('adapter_id')).toBe('classification-label');
        expect(form.get('task_profile')).toBe('classification');
        expect(JSON.parse(String(form.get('adapter_config')))).toEqual({ text_fields: ['sentence'] });
    });

    it('sends no adapter fields without a field mapping, and skips an unpinned (auto) task profile', async () => {
        const user = userEvent.setup();
        apiMock.post.mockReset();
        apiMock.post.mockResolvedValue({ data: OK });
        render(<PredefinedSplitCard projectId={5} adapter={{ adapterId: 'qa-pair', taskProfile: 'auto' }} />);
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(1));
        const form = apiMock.post.mock.calls[0][1] as FormData;
        expect(form.get('adapter_id')).toBeNull();
        expect(form.get('field_mapping')).toBeNull();
    });

    it('names the columns the overlap check used, and resets the drop-test checkbox after an import', async () => {
        const user = userEvent.setup();
        apiMock.post.mockReset();
        apiMock.post.mockResolvedValue({ data: { ...OK, overlap_input_columns: ['question', 'context'] } });
        render(<PredefinedSplitCard projectId={5} ownTestRows={7} />);
        await user.upload(screen.getByTestId('pdsplit-train'), file('t.jsonl'));
        await user.click(screen.getByTestId('pdsplit-drop-test'));
        expect(screen.getByTestId('pdsplit-drop-test')).toBeChecked();
        await user.click(screen.getByTestId('pdsplit-submit'));
        await screen.findByTestId('pdsplit-result');
        expect((apiMock.post.mock.calls[0][1] as FormData).get('drop_existing_test')).toBe('true');
        expect(screen.getByTestId('pdsplit-overlap-columns')).toHaveTextContent('Overlap checked on: question + context');
        expect(screen.getByTestId('pdsplit-drop-test')).not.toBeChecked();
    });

    it('sends the input as text for a structured-extraction adapter, and shows the overlap columns on a 409', async () => {
        const user = await openMapper(['narrative', 'structured_output'], { adapterId: 'structured-extraction' });
        apiMock.post.mockRejectedValueOnce({
            response: { status: 409, data: {
                error_code: 'PREDEFINED_SPLIT_LEAKAGE', message: 'Rows overlap', actionable_fix: 'Remove them.',
                troubleshooting_id: 'e1', metadata: { pairs: [], overlap_input_columns: ['text'] },
            } },
        });
        await user.click(screen.getByTestId('pdsplit-map-apply'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
        const mapping = JSON.parse(String((apiMock.post.mock.calls[1][1] as FormData).get('field_mapping')));
        expect(mapping).toEqual({ text: 'narrative', answer: 'structured_output' });
        expect(await screen.findByTestId('pdsplit-problem-overlap-columns')).toHaveTextContent('Overlap checked on: text');
    });
});
