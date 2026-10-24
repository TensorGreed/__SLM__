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
});
