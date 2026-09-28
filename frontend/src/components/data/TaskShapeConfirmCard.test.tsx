import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import TaskShapeConfirmCard from './TaskShapeConfirmCard';
import type { TaskShapeDetection } from '../../api/taskShape';

const QA = {
    task_profile: 'qa',
    label: 'Question answering',
    description: 'Each row is a question and the answer the model should give.',
    adapter_id: 'qa-pair',
    recipe_id: 'qa-sft',
    mapper_id: 'qa_pair_passthrough',
    field_map: { question_field: 'q', answer_field: 'a' },
    confidence: 0.92,
    map_rate: 1,
    rationale: ["detected QA pair: question='q', answer='a'", '100% of rows fit the question answering format'],
    source: 'columns',
};

const DETECTION: TaskShapeDetection = {
    top: QA,
    candidates: [QA],
    needs_confirmation: false,
    confirm_threshold: 0.8,
    rows_examined: 20,
};

describe('TaskShapeConfirmCard', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
        apiMock.post.mockImplementation(async (_url: string, body: { task_profile: string }) => ({
            data: {
                task_profile: body.task_profile,
                label: body.task_profile === 'qa' ? 'Question answering' : 'Summarization',
                adapter_id: 'qa-pair',
                recipe_id: 'qa-sft',
            },
        }));
    });

    it('explains the best guess and confirms it in one click', async () => {
        const onConfirmed = vi.fn();
        const user = userEvent.setup();
        render(<TaskShapeConfirmCard projectId={9} detection={DETECTION} onConfirmed={onConfirmed} />);

        expect(screen.getByTestId('task-shape-top').textContent).toContain('Question answering');
        expect(screen.getByTestId('task-shape-top').textContent).toContain('92%');
        expect(screen.getByText(/100% of rows fit/)).toBeInTheDocument();
        expect(screen.queryByTestId('task-shape-unsure')).toBeNull();

        await user.click(screen.getByTestId('task-shape-confirm'));
        await waitFor(() =>
            expect(apiMock.post).toHaveBeenCalledWith('/projects/9/task-shape/confirm', { task_profile: 'qa' }),
        );
        expect(await screen.findByTestId('task-shape-confirmed')).toHaveTextContent('Question answering');
        expect(onConfirmed).toHaveBeenCalledWith(expect.objectContaining({ task_profile: 'qa' }), QA);
    });

    it('flags an unsure guess and lets the user pick something else from the catalog', async () => {
        const user = userEvent.setup();
        render(
            <TaskShapeConfirmCard
                projectId={9}
                detection={{ ...DETECTION, needs_confirmation: true }}
                catalog={[
                    { task_profile: 'qa', label: 'Question answering', description: '' },
                    { task_profile: 'summarization', label: 'Summarization', description: '' },
                ]}
            />,
        );
        expect(screen.getByTestId('task-shape-unsure')).toBeInTheDocument();
        await user.click(screen.getByTestId('task-shape-choose'));
        await user.selectOptions(screen.getByTestId('task-shape-alternative'), 'summarization');
        await user.click(screen.getByTestId('task-shape-alternative-confirm'));
        await waitFor(() =>
            expect(apiMock.post).toHaveBeenCalledWith('/projects/9/task-shape/confirm', {
                task_profile: 'summarization',
            }),
        );
        expect(await screen.findByTestId('task-shape-confirmed')).toHaveTextContent('Summarization');
    });

    it('renders nothing without a detection', () => {
        const { container } = render(<TaskShapeConfirmCard projectId={9} detection={null} />);
        expect(container).toBeEmptyDOMElement();
    });
});
