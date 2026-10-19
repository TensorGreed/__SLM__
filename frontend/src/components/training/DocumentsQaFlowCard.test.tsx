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
    plan: { max_passages: 60, pairs_per_passage: 3, estimated_training_pairs: 180, estimated_answer_key_rows: 60, llm_calls: 60, sized_for: { target_test_rows: 45, judge_gap: 0.15, passages: 150, passages_needed: 150, test_ratio: 0.1, expected_test_rows: 45, capped_by_available: false, reaches_target: true } },
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
        expect(screen.getByTestId('documents-qa-flow-sizing')).toHaveTextContent('sized so a 0.15 judge-score gap clears noise: 45 test examples from 150 passages.');
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

    it('trains the generated pairs again to answer from the passages', async () => {
        apiMock.get.mockResolvedValue({ data: { ...ELIGIBLE, already_generated: 120 } });
        apiMock.post.mockResolvedValue({ data: { id: 13 } });
        const user = userEvent.setup();
        render(<DocumentsQaFlowCard projectId={1} />);
        await user.click(await screen.findByTestId('documents-qa-flow-with-passages'));
        expect(screen.getByTestId('documents-qa-flow-retrain')).toHaveTextContent('reading your passages');
        await user.click(screen.getByTestId('documents-qa-flow-retrain'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/flows/documents-to-qa', expect.objectContaining({ reuse_existing: true, train_with_passages: true }));
        });
    });

    it('shows progress while the Job runs and the result when it finishes', async () => {
        apiMock.get.mockResolvedValue({ data: ELIGIBLE });
        jobsState.jobs = [{ id: 11, kind: 'documents_qa_flow', project_id: 1, status: 'running', progress_message: 'Writing questions for passage 12/60 (ollama:qwen2.5)' }];
        const { rerender } = render(<DocumentsQaFlowCard projectId={1} />);
        expect(await screen.findByTestId('documents-qa-flow-running')).toHaveTextContent('passage 12/60');
        expect(screen.queryByTestId('documents-qa-flow-start')).not.toBeInTheDocument();

        jobsState.jobs = [{
            id: 11, kind: 'documents_qa_flow', project_id: 1, status: 'succeeded',
            result: { training_pairs: 171, answer_key_rows: 58, passages_used: 58, backend: 'ollama:qwen2.5', split: { train: 136, val: 17, test: 18, dedup_dropped: 0 }, experiment_id: 31, stopped_reason: null, warnings: ['2 of 60 passages produced no usable questions.'], eval_rejected: { deictic: 1, unsupported: 2 } },
        }];
        rerender(<DocumentsQaFlowCard projectId={1} />);
        const result = await screen.findByTestId('documents-qa-flow-result');
        expect(result).toHaveTextContent('171 question→answer pairs from 58 passages (ollama:qwen2.5), 58 answer-key rows (3 dropped: a reader couldn\'t quote the answer from the passage), split 136 / 17 / 18');
        expect(screen.getByTestId('documents-qa-flow-eval-rejected')).toHaveTextContent("(3 dropped: a reader couldn't quote the answer from the passage)");
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
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/flows/documents-to-qa', expect.objectContaining({ reuse_existing: true, train_if_retrieval_ready: true, train: true, passages_check: false }));
        });
        await user.click(screen.getByTestId('documents-qa-flow-reroute'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(expect.stringMatching(/reroute-to-rag\?async_job=true/), expect.anything());
        });
    });
    it('runs the flow with a cloud model whose key is saved on the project', async () => {
        const CLOUD = { project_id: 1, providers: [
            { provider: 'anthropic', key_saved: true, models: [{ id: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5' }] },
            { provider: 'openai', key_saved: false, models: [{ id: 'gpt-4o', label: 'GPT-4o' }] },
        ] };
        apiMock.get.mockImplementation((url: string) =>
            Promise.resolve({ data: url.includes('/backends/cloud/models') ? CLOUD : ELIGIBLE }),
        );
        apiMock.post.mockResolvedValue({ data: { id: 11 } });
        const user = userEvent.setup();
        render(<DocumentsQaFlowCard projectId={1} />);
        await screen.findByTestId('documents-qa-flow-plan');
        await waitFor(() => expect(screen.getByTestId('documents-qa-flow-provider')).toHaveTextContent('anthropic (key saved)'));
        await user.selectOptions(screen.getByTestId('documents-qa-flow-provider'), 'openai');
        expect(screen.getByTestId('documents-qa-flow-no-key')).toHaveTextContent('No openai key saved');
        expect(screen.getByTestId('documents-qa-flow-cloud-model')).toBeDisabled();
        await user.selectOptions(screen.getByTestId('documents-qa-flow-provider'), 'anthropic');
        await user.selectOptions(screen.getByTestId('documents-qa-flow-cloud-model'), 'claude-sonnet-5-5');
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/1/flows/documents-to-qa/preview', { params: { backend: 'cloud:anthropic:claude-sonnet-5-5' } },
        ));
        await user.click(screen.getByTestId('documents-qa-flow-start'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/flows/documents-to-qa',
            { max_passages: 60, pairs_per_passage: 3, train: true, backend: 'cloud:anthropic:claude-sonnet-5-5' },
        ));
    });

    it('trains on a teacher model\'s answers to the same passages, keeping and growing the test examples', async () => {
        apiMock.get.mockResolvedValue({ data: { ...ELIGIBLE, already_generated: 120 } });
        apiMock.post.mockResolvedValue({ data: { id: 14 } });
        const user = userEvent.setup();
        render(<DocumentsQaFlowCard projectId={1} />);
        await user.click(await screen.findByTestId('documents-qa-flow-teacher'));
        expect(screen.getByTestId('documents-qa-flow-with-passages')).toBeChecked();
        expect(screen.getByTestId('documents-qa-flow-retrain')).toHaveTextContent("teacher's answers");
        await user.click(screen.getByTestId('documents-qa-flow-keep-test'));
        await user.selectOptions(screen.getByTestId('documents-qa-flow-gap'), '0.08');
        await user.click(screen.getByTestId('documents-qa-flow-retrain'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/flows/documents-to-qa', expect.objectContaining({
                reuse_existing: true, train_with_passages: true, answers_from_teacher: true, keep_test: true, decision_gap: 0.08,
            }));
        });
    });

    it('says how the teacher step and the val-chosen retrieval went', async () => {
        apiMock.get.mockResolvedValue({ data: { ...ELIGIBLE, already_generated: 120 } });
        jobsState.jobs = [{
            id: 20, kind: 'documents_qa_flow', project_id: 1, status: 'succeeded',
            result: {
                project_id: 1, passages_total: 10, passages_used: 0, passages_failed: 0, training_pairs: 120, answer_key_rows: 10,
                backend: 'reused', split: { train: 90, val: 20, test: 20, dedup_dropped: 0 }, experiment_id: 7, stopped_reason: null, warnings: [],
                train_with_passages: true,
                teacher: { teacher: 'ollama:gpt-oss:120b', usable: 100, filter: { applied: true, judge: 'ollama:gemma4:12b' },
                           counts: { train: { rows: 90, reused: 0, asked: 90, answered: 90, failed: 0, abstained: 6, filter_judged: 84, filter_dropped: 4 } } },
                passages_gate: { status: 'retrieval_promising', correct: 10, judged: 20, wrong: 6, partial: 4, score: 0.6, split: 'test',
                                 sweep_split: 'val', sweep: { split: 'val', rows: 20, chosen_score: 0.62 } },
            },
        }];
        render(<DocumentsQaFlowCard projectId={1} />);
        expect(await screen.findByTestId('documents-qa-flow-teacher-result')).toHaveTextContent('6 training answers say the passages don\'t cover it, 4 dropped as wrong');
        expect(screen.getByTestId('documents-qa-flow-gate-sweep')).toHaveTextContent('chosen on 20 validation rows, 0.62 there; scored once on the test examples');
    });
});
