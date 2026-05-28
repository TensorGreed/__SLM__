/**
 * Student-vs-teacher distillation comparison panel (Track 1, Epic A, slice 3).
 *
 * Shows how much of a teacher baseline run's quality the distilled student
 * kept: quality_retained = student/teacher, per metric + per slice. Pulls from
 * stored EvalResult rows (no model calls).
 *
 * UX rules so it isn't noise on every project:
 *  - Self-hides entirely unless the selected run was trained with offline KD
 *    (`is_distillation_run`) OR a teacher baseline is already resolved/picked.
 *  - When no teacher is set, offers an inline picker of the project's other
 *    experiments — no config/query-param editing required.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import {
    fetchStudentTeacherComparison,
    type MetricComparisonRow,
    type SliceComparisonRow,
    type StudentTeacherComparison,
} from '../../api/studentTeacherComparison';

interface ExperimentOption {
    id: number;
    name: string;
}

interface Props {
    projectId: number;
    experimentId: number | null;
    /** Project experiments — candidate teacher baselines for the picker. */
    experiments?: ExperimentOption[];
    /** Refetch trigger (e.g. eval just re-ran). */
    refreshToken?: unknown;
}

function errorDetail(err: unknown, fallback: string): string {
    const d = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    return typeof d === 'string' && d ? d : fallback;
}

function formatValue(v: number | null | undefined): string {
    if (v === null || v === undefined || !Number.isFinite(v)) return '—';
    return v.toFixed(2);
}

function formatRetained(v: number | null): string {
    if (v === null) return 'exceeds';
    return `${Math.round(v * 100)}%`;
}

function retentionColor(direction: string): string {
    return direction === 'regressed' ? 'var(--color-error)' : 'var(--color-success)';
}

function MetricRow({ row }: { row: MetricComparisonRow }) {
    return (
        <div
            data-testid={`st-metric-${row.metric_id}`}
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-sm)',
                padding: 'var(--space-xs) 0',
                fontSize: '0.85rem',
            }}
        >
            <code style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, minWidth: 110 }}>
                {row.metric_id}
            </code>
            <span style={{ color: 'var(--text-secondary)' }}>
                student <code>{formatValue(row.student_value)}</code> · teacher{' '}
                <code>{formatValue(row.teacher_value)}</code>
            </span>
            <span
                data-testid={`st-retained-${row.metric_id}`}
                style={{ marginLeft: 'auto', color: retentionColor(row.direction), fontWeight: 600 }}
            >
                {formatRetained(row.quality_retained)} retained
            </span>
        </div>
    );
}

function SliceRow({ row }: { row: SliceComparisonRow }) {
    return (
        <div
            data-testid={`st-slice-${row.slice}-${row.metric_id}`}
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-sm)',
                padding: '2px 0',
                fontSize: '0.8rem',
                color: 'var(--text-secondary)',
            }}
        >
            <code style={{ fontFamily: 'var(--font-mono)' }}>{row.slice}</code>
            <span>· {row.metric_id} {formatValue(row.student_value)}/{formatValue(row.teacher_value)}</span>
            <span style={{ marginLeft: 'auto', color: retentionColor(row.direction), fontWeight: 600 }}>
                {formatRetained(row.quality_retained)}
            </span>
        </div>
    );
}

export default function StudentTeacherComparisonPanel({
    projectId,
    experimentId,
    experiments = [],
    refreshToken,
}: Props) {
    const [data, setData] = useState<StudentTeacherComparison | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string>('');
    const [selectedTeacher, setSelectedTeacher] = useState<number | null>(null);

    const teacherOptions = useMemo(
        () => experiments.filter((e) => e.id !== experimentId),
        [experiments, experimentId],
    );

    const load = useCallback(async () => {
        if (experimentId == null) return;
        setLoading(true);
        setError('');
        try {
            setData(await fetchStudentTeacherComparison(projectId, experimentId, selectedTeacher));
        } catch (err) {
            setError(errorDetail(err, 'Failed to load distillation comparison.'));
        } finally {
            setLoading(false);
        }
    }, [projectId, experimentId, selectedTeacher]);

    useEffect(() => {
        load();
    }, [load, refreshToken]);

    // A new student selected → forget the previously picked teacher.
    useEffect(() => {
        setSelectedTeacher(null);
    }, [experimentId]);

    if (experimentId == null) return null;
    if (loading && !data) {
        // Stay quiet on first load — only known-distillation runs get a spinner
        // (we don't yet know is_distillation_run, so render nothing until data).
        return null;
    }

    if (error) {
        return (
            <section
                className="card"
                role="alert"
                data-testid="st-comparison-error"
                style={{ padding: 'var(--space-md)', background: 'var(--color-error-bg)', color: 'var(--color-error)' }}
            >
                {error}
            </section>
        );
    }

    if (!data || !data.status || !Array.isArray(data.metric_comparisons)) return null;

    const teacherResolved = data.status !== 'no_teacher_baseline';
    const relevant = data.is_distillation_run || teacherResolved || selectedTeacher != null;
    // Don't nag projects that aren't distilling and have no teacher set.
    if (!relevant) return null;

    const TeacherPicker = teacherOptions.length > 0 && (
        <label
            data-testid="st-teacher-picker"
            style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', fontSize: '0.85rem' }}
        >
            <span style={{ color: 'var(--text-secondary)' }}>Teacher run:</span>
            <select
                aria-label="Teacher baseline run"
                value={selectedTeacher ?? ''}
                onChange={(e) => setSelectedTeacher(e.target.value ? Number(e.target.value) : null)}
            >
                <option value="">{teacherResolved ? '(auto)' : '— pick a run —'}</option>
                {teacherOptions.map((e) => (
                    <option key={e.id} value={e.id}>
                        #{e.id} · {e.name}
                    </option>
                ))}
            </select>
        </label>
    );

    // Not yet comparable — show a compact, actionable card (picker if possible).
    if (data.status !== 'ok') {
        const titles: Record<string, string> = {
            no_teacher_baseline: 'Compare against a teacher baseline',
            no_student_eval: 'Student vs teacher — evaluate this run first',
            no_teacher_eval: 'Student vs teacher — evaluate the teacher run',
            no_overlap: "Student vs teacher — metrics didn't overlap",
        };
        const hints: Record<string, string> = {
            no_teacher_baseline:
                'This run was distilled. Pick the teacher run it learned from to see how much quality it kept.',
            no_student_eval: 'Run evaluation on this experiment, then come back.',
            no_teacher_eval: 'Evaluate the selected teacher run on the same eval set, then compare.',
            no_overlap: 'Re-run eval on both with the same task profile / eval set.',
        };
        return (
            <section
                className="card"
                data-testid={`st-comparison-${data.status.replace(/_/g, '-')}`}
                style={{ padding: 'var(--space-md)', display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}
            >
                <h4 style={{ margin: 0 }}>{titles[data.status]}</h4>
                <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.85rem' }}>
                    {hints[data.status]}
                </p>
                {TeacherPicker || (
                    <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.8rem' }}>
                        No other experiments yet to use as a teacher baseline.
                    </p>
                )}
            </section>
        );
    }

    const { student, teacher, metric_comparisons, slice_comparisons, headline_quality_retained } = data;

    return (
        <section
            className="card"
            data-testid="st-comparison-panel"
            style={{ padding: 'var(--space-md)', display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}
        >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
                <div>
                    <h4 style={{ margin: 0 }}>Distillation quality retained</h4>
                    <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                        student ·{' '}
                        <strong data-testid="st-student-name">{student?.experiment_name}</strong>{' '}
                        vs teacher ·{' '}
                        <strong data-testid="st-teacher-name">{teacher?.experiment_name}</strong>
                        {headline_quality_retained !== null && (
                            <>
                                {' '}—{' '}
                                <strong data-testid="st-headline-retained" style={{ color: 'var(--color-success)' }}>
                                    {formatRetained(headline_quality_retained)} retained
                                </strong>
                            </>
                        )}
                    </p>
                </div>
                {TeacherPicker}
            </div>

            <div data-testid="st-metric-rows">
                {metric_comparisons.map((row) => (
                    <MetricRow key={row.metric_id} row={row} />
                ))}
            </div>

            {slice_comparisons.length > 0 && (
                <div data-testid="st-slice-rows">
                    <h5 style={{ margin: '0 0 var(--space-xs)' }}>Per-slice retention</h5>
                    {slice_comparisons.map((row) => (
                        <SliceRow key={`${row.slice}-${row.metric_id}`} row={row} />
                    ))}
                </div>
            )}
        </section>
    );
}
