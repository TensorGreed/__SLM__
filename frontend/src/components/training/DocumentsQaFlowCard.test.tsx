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

    it('stays hidden for projects without document passages', async () => {
        apiMock.get.mockResolvedValueOnce({ data: { ...ELIGIBLE, passages: 0, eligible: false, blockers: ['x'] } });
        render(<DocumentsQaFlowCard projectId={1} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(screen.queryByTestId('documents-qa-flow')).not.toBeInTheDocument();
    });

    it('offers to train the already generated pairs again on the current base model', async () => {
        apiMock.get.mockResolvedValue({ data: { ...ELIGIBLE, already_generated: 120 } });
        apiMock.post.mockResolvedValue({ data: { id: 12 } });
        render(<DocumentsQaFlowCard projectId={1} />);
        expect(await screen.findByTestId('documents-qa-flow-generated-before')).toHaveTextContent('120 question→answer pairs were generated');
        expect(screen.queryByTestId('documents-qa-flow-plan')).not.toBeInTheDocument();
        await userEvent.setup().click(screen.getByTestId('documents-qa-flow-retrain'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/flows/documents-to-qa', expect.objectContaining({ reuse_existing: true, train: true }));
        });
        expect(screen.getByTestId('documents-qa-flow-rerun')).toHaveTextContent('Run again (write new pairs)');
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

    it('says when retrieval was promising and training went ahead', async () => {
        apiMock.get.mockResolvedValue({ data: ELIGIBLE });
        jobsState.jobs = [{
            id: 13, kind: 'documents_qa_flow', project_id: 1, status: 'succeeded',
            result: {
                training_pairs: 180, answer_key_rows: 60, passages_used: 60, backend: 'ollama:gemma4:12b',
                split: { train: 142, val: 17, test: 19, dedup_dropped: 0 }, experiment_id: 41, stopped_reason: null, warnings: [],
                passages_gate: { status: 'retrieval_promising', score: 0.5526, correct: 8, partial: 5, wrong: 6, judged: 19, judge: 'ollama:gemma4:12b', split: 'test',
                    retrieval: { k: 3, reranker: null } },
            },
        }];
        render(<DocumentsQaFlowCard projectId={1} />);
        const gate = await screen.findByTestId('documents-qa-flow-gate');
        expect(gate).toHaveAttribute('data-status', 'retrieval_promising');
        expect(gate).toHaveTextContent("(top-3, the sweep's best) already got 8 of 19 fully right but 6 wrong (5 partly; judge score 0.55). Not an assistant on its own yet, so training went ahead");
        expect(screen.queryByTestId('documents-qa-flow-reroute')).not.toBeInTheDocument();
        expect(screen.getByTestId('documents-qa-flow-result')).toHaveTextContent('Training run #41 started');
    });

    it('shows the pre-training gate when retrieval already answers, with reroute and train-anyway', async () => {
        apiMock.get.mockResolvedValue({ data: ELIGIBLE });
        apiMock.post.mockResolvedValue({ data: { id: 61 } });
        jobsState.jobs = [{
            id: 12, kind: 'documents_qa_flow', project_id: 1, status: 'succeeded',
            result: {
                training_pairs: 180, answer_key_rows: 60, passages_used: 60, backend: 'ollama:gemma4:12b',
                split: { train: 142, val: 17, test: 19, dedup_dropped: 0 }, experiment_id: null,
                stopped_reason: 'Retrieval already answers: …', warnings: [],
                passages_gate: { status: 'retrieval_ready', score: 0.6053, correct: 8, partial: 7, wrong: 4, judged: 19, judge: 'ollama:gemma4:12b', split: 'test' },
            },
        }];
        const user = userEvent.setup();
        render(<DocumentsQaFlowCard projectId={1} />);
        const gate = await screen.findByTestId('documents-qa-flow-gate');
        expect(gate).toHaveAttribute('data-status', 'retrieval_ready');
        expect(gate).toHaveTextContent('already got 8 of 19 test examples fully right (7 partly, 4 wrong; judge score 0.61, by ollama:gemma4:12b)');
        expect(screen.getByTestId('documents-qa-flow-result')).not.toHaveTextContent('Stopped before training');
        await user.click(screen.getByTestId('documents-qa-flow-train-anyway'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/flows/documents-to-qa', expect.objectContaining({ reuse_existing: true, train_if_retrieval_ready: true, train: true }));
        });
        await user.click(screen.getByTestId('documents-qa-flow-reroute'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(expect.stringMatching(/reroute-to-rag\?async_job=true/), expect.anything());
        });
    });
});
