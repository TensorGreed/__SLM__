import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));
const refreshSpy = vi.fn();
const jobsState: { jobs: Array<Record<string, unknown>> } = { jobs: [] };
vi.mock('../../stores/jobsStore', () => ({
    useJobsStore: Object.assign(
        (selector: (state: { jobs: unknown[] }) => unknown) => selector(jobsState),
        { getState: () => ({ refreshAfterLocalChange: refreshSpy, jobs: jobsState.jobs }) },
    ),
}));

import DocumentsQaFlowCard from './DocumentsQaFlowCard';

const ELIGIBLE = {
    project_id: 1, eligible: true, blockers: [], passages: 560, backend: 'ollama', current_recipe_id: null, already_generated: 0,
    plan: { max_passages: 60, pairs_per_passage: 3, estimated_training_pairs: 180, estimated_answer_key_rows: 60, llm_calls: 60 },
};

describe('DocumentsQaFlowCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        toastMock.info.mockReset();
        jobsState.jobs = [];
    });

    it('shows the plan and starts the flow', async () => {
        apiMock.get.mockResolvedValue({ data: ELIGIBLE });
        apiMock.post.mockResolvedValue({ data: { id: 9, kind: 'documents_qa_flow', status: 'queued' } });
        render(<DocumentsQaFlowCard projectId={1} />);

        const plan = await screen.findByTestId('documents-qa-flow-plan');
        expect(plan).toHaveTextContent('About 180 training pairs from 60 passages (3 per passage), written by ollama.');
        expect(plan).toHaveTextContent('60 different questions kept as the answer key.');
        await userEvent.setup().click(screen.getByTestId('documents-qa-flow-start'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/flows/documents-to-qa', { max_passages: 60, pairs_per_passage: 3, train: true });
        });
        expect(toastMock.info).toHaveBeenCalledWith(expect.stringMatching(/job #9/), 5000);
    });

    it('"review first" stops after generating', async () => {
        apiMock.get.mockResolvedValue({ data: ELIGIBLE });
        apiMock.post.mockResolvedValue({ data: { id: 10 } });
        const user = userEvent.setup();
        render(<DocumentsQaFlowCard projectId={1} />);
        await user.click(await screen.findByTestId('documents-qa-flow-review-first'));
        expect(screen.getByTestId('documents-qa-flow-start')).toHaveTextContent('Generate the Q&A pairs');
        await user.click(screen.getByTestId('documents-qa-flow-start'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/flows/documents-to-qa', expect.objectContaining({ train: false }));
        });
    });

    it('shows blockers instead of a button when not eligible', async () => {
        apiMock.get.mockResolvedValue({ data: { ...ELIGIBLE, eligible: false, backend: null, blockers: ['No generation model is reachable. Start Ollama with a chat model.'] } });
        render(<DocumentsQaFlowCard projectId={1} />);
        expect(await screen.findByTestId('documents-qa-flow-blockers')).toHaveTextContent('No generation model is reachable');
        expect(screen.queryByTestId('documents-qa-flow-start')).not.toBeInTheDocument();
    });

    it('stays hidden for projects without document passages or that already generated rows', async () => {
        apiMock.get.mockResolvedValueOnce({ data: { ...ELIGIBLE, passages: 0, eligible: false, blockers: ['x'] } });
        const { unmount } = render(<DocumentsQaFlowCard projectId={1} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(screen.queryByTestId('documents-qa-flow')).not.toBeInTheDocument();
        unmount();
        apiMock.get.mockResolvedValueOnce({ data: { ...ELIGIBLE, already_generated: 120 } });
        render(<DocumentsQaFlowCard projectId={1} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(2));
        expect(screen.queryByTestId('documents-qa-flow')).not.toBeInTheDocument();
    });

    it('shows progress while the Job runs and the result when it finishes', async () => {
        apiMock.get.mockResolvedValue({ data: ELIGIBLE });
        jobsState.jobs = [{ id: 11, kind: 'documents_qa_flow', project_id: 1, status: 'running', progress_message: 'Writing questions for passage 12/60 (ollama:qwen2.5)' }];
        const { rerender } = render(<DocumentsQaFlowCard projectId={1} />);
        expect(await screen.findByTestId('documents-qa-flow-running')).toHaveTextContent('passage 12/60');
        expect(screen.queryByTestId('documents-qa-flow-start')).not.toBeInTheDocument();

        jobsState.jobs = [{
            id: 11, kind: 'documents_qa_flow', project_id: 1, status: 'succeeded',
            result: { training_pairs: 171, answer_key_rows: 58, passages_used: 58, backend: 'ollama:qwen2.5', split: { train: 136, val: 17, test: 18, dedup_dropped: 0 }, experiment_id: 31, stopped_reason: null, warnings: ['2 of 60 passages produced no usable questions.'] },
        }];
        rerender(<DocumentsQaFlowCard projectId={1} />);
        const result = await screen.findByTestId('documents-qa-flow-result');
        expect(result).toHaveTextContent('171 question→answer pairs from 58 passages (ollama:qwen2.5), 58 answer-key rows, split 136 / 17 / 18');
        expect(result).toHaveTextContent('Training run #31 started');
        expect(result).toHaveTextContent('2 of 60 passages produced no usable questions.');
    });
});
