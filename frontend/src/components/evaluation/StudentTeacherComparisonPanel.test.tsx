import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import StudentTeacherComparisonPanel from './StudentTeacherComparisonPanel';

const EXPERIMENTS = [
    { id: 11, name: 'student-kd' },
    { id: 10, name: 'teacher-run' },
];

const OK = {
    status: 'ok',
    project_id: 1,
    is_distillation_run: true,
    teacher_baseline_run_id: 10,
    student: {
        experiment_id: 11,
        experiment_name: 'student-kd',
        base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        eval_result_id: 101,
        dataset_name: 'test',
        eval_type: 'f1',
        metrics: { f1: 0.65 },
        pass_rate: 0.65,
    },
    teacher: {
        experiment_id: 10,
        experiment_name: 'teacher-run',
        base_model: 'teacher',
        eval_result_id: 100,
        dataset_name: 'test',
        eval_type: 'f1',
        metrics: { f1: 0.8 },
        pass_rate: 0.8,
    },
    metric_comparisons: [
        {
            metric_id: 'f1',
            student_value: 0.65,
            teacher_value: 0.8,
            quality_retained: 0.8125,
            direction: 'regressed',
            is_headline: true,
        },
    ],
    slice_comparisons: [
        {
            slice: 'short',
            metric_id: 'f1',
            student_value: 0.6,
            teacher_value: 0.8,
            quality_retained: 0.75,
            direction: 'regressed',
        },
    ],
    headline_quality_retained: 0.8125,
    message: null,
};

function noTeacher(overrides: Record<string, unknown> = {}) {
    return {
        status: 'no_teacher_baseline',
        project_id: 1,
        is_distillation_run: false,
        teacher_baseline_run_id: null,
        student: null,
        teacher: null,
        metric_comparisons: [],
        slice_comparisons: [],
        headline_quality_retained: null,
        message: 'No teacher baseline run set.',
        ...overrides,
    };
}

describe('StudentTeacherComparisonPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders headline retention + metric + slice rows on ok', async () => {
        apiMock.get.mockResolvedValueOnce({ data: OK });
        render(<StudentTeacherComparisonPanel projectId={1} experimentId={11} experiments={EXPERIMENTS} />);
        await waitFor(() => expect(screen.getByTestId('st-comparison-panel')).toBeInTheDocument());
        expect(screen.getByTestId('st-headline-retained')).toHaveTextContent('81% retained');
        expect(screen.getByTestId('st-retained-f1')).toHaveTextContent('81% retained');
        expect(screen.getByTestId('st-slice-short-f1')).toHaveTextContent('75%');
        expect(screen.getByTestId('st-teacher-name')).toHaveTextContent('teacher-run');
    });

    it('self-hides on non-distillation runs with no teacher set', async () => {
        apiMock.get.mockResolvedValueOnce({ data: noTeacher() });
        const { container } = render(
            <StudentTeacherComparisonPanel projectId={1} experimentId={11} experiments={[]} />,
        );
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(container).toBeEmptyDOMElement();
    });

    it('shows the teacher picker on a distillation run without a teacher set', async () => {
        apiMock.get.mockResolvedValueOnce({ data: noTeacher({ is_distillation_run: true }) });
        render(<StudentTeacherComparisonPanel projectId={1} experimentId={11} experiments={EXPERIMENTS} />);
        await waitFor(() =>
            expect(screen.getByTestId('st-comparison-no-teacher-baseline')).toBeInTheDocument(),
        );
        expect(screen.getByTestId('st-teacher-picker')).toBeInTheDocument();
        // The student itself isn't offered as a teacher option.
        expect(screen.queryByRole('option', { name: /student-kd/ })).not.toBeInTheDocument();
    });

    it('refetches with teacher_run_id when a teacher is picked', async () => {
        apiMock.get.mockResolvedValue({ data: noTeacher({ is_distillation_run: true }) });
        render(<StudentTeacherComparisonPanel projectId={1} experimentId={11} experiments={EXPERIMENTS} />);
        await waitFor(() => expect(screen.getByTestId('st-teacher-picker')).toBeInTheDocument());

        apiMock.get.mockResolvedValueOnce({ data: OK });
        fireEvent.change(screen.getByLabelText('Teacher baseline run'), { target: { value: '10' } });

        await waitFor(() => expect(screen.getByTestId('st-comparison-panel')).toBeInTheDocument());
        const lastCall = apiMock.get.mock.calls.at(-1);
        expect(lastCall?.[0]).toContain('/student-teacher-comparison/11');
        expect(lastCall?.[1]).toEqual({ params: { teacher_run_id: 10 } });
    });

    it('renders "exceeds" when teacher metric is zero', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...OK,
                metric_comparisons: [
                    {
                        metric_id: 'f1',
                        student_value: 0.5,
                        teacher_value: 0.0,
                        quality_retained: null,
                        direction: 'exceeds',
                        is_headline: true,
                    },
                ],
                slice_comparisons: [],
                headline_quality_retained: null,
            },
        });
        render(<StudentTeacherComparisonPanel projectId={1} experimentId={11} experiments={EXPERIMENTS} />);
        await waitFor(() => expect(screen.getByTestId('st-comparison-panel')).toBeInTheDocument());
        expect(screen.getByTestId('st-retained-f1')).toHaveTextContent('exceeds');
    });

    it('renders nothing until an experiment is selected', () => {
        const { container } = render(
            <StudentTeacherComparisonPanel projectId={1} experimentId={null} experiments={EXPERIMENTS} />,
        );
        expect(container).toBeEmptyDOMElement();
        expect(apiMock.get).not.toHaveBeenCalled();
    });
});
