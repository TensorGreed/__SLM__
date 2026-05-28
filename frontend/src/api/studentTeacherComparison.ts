/**
 * Typed wrapper for the Track 1 Epic A slice 3 student-vs-teacher comparison.
 *
 * Endpoint: GET /api/projects/{id}/evaluation/student-teacher-comparison/{experimentId}
 *           [?teacher_run_id=N]
 *
 * Returns quality_retained = student/teacher per metric + per slice, computed
 * from stored EvalResult rows (no model calls).
 */

import api from './client';

export interface StudentTeacherExperimentRef {
    experiment_id: number;
    experiment_name: string;
    base_model: string;
    eval_result_id: number;
    dataset_name: string;
    eval_type: string;
    metrics: Record<string, number>;
    pass_rate: number | null;
}

export type RetentionDirection = 'retained_or_better' | 'regressed' | 'exceeds';

export interface MetricComparisonRow {
    metric_id: string;
    student_value: number;
    teacher_value: number;
    /** student / teacher; null when teacher scored 0 (UI renders "exceeds"). */
    quality_retained: number | null;
    direction: RetentionDirection;
    is_headline: boolean;
}

export interface SliceComparisonRow {
    slice: string;
    metric_id: string;
    student_value: number;
    teacher_value: number;
    quality_retained: number | null;
    direction: RetentionDirection;
}

export type StudentTeacherStatus =
    | 'ok'
    | 'no_teacher_baseline'
    | 'no_student_eval'
    | 'no_teacher_eval'
    | 'no_overlap';

export interface StudentTeacherComparison {
    status: StudentTeacherStatus;
    project_id: number;
    /** True when the student experiment was trained with offline KD; the panel
     * self-hides on non-distillation runs. */
    is_distillation_run: boolean;
    teacher_baseline_run_id: number | null;
    student: StudentTeacherExperimentRef | null;
    teacher: StudentTeacherExperimentRef | null;
    metric_comparisons: MetricComparisonRow[];
    slice_comparisons: SliceComparisonRow[];
    headline_quality_retained: number | null;
    message?: string | null;
}

export async function fetchStudentTeacherComparison(
    projectId: number,
    experimentId: number,
    teacherRunId?: number | null,
): Promise<StudentTeacherComparison> {
    const res = await api.get<StudentTeacherComparison>(
        `/projects/${projectId}/evaluation/student-teacher-comparison/${experimentId}`,
        teacherRunId != null ? { params: { teacher_run_id: teacherRunId } } : undefined,
    );
    return res.data;
}
