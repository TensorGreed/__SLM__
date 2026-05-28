/**
 * Student-vs-teacher distillation comparison panel (Track 1, Epic A, slice 3).
 *
 * Shows how much of a teacher baseline run's quality the distilled student
 * kept: quality_retained = student/teacher, per metric + per slice. Pulls from
 * stored EvalResult rows (no model calls). Soft-fallback cards for the
 * not-ready states so the panel never errors out the Eval tab.
 */

import { useCallback, useEffect, useState } from 'react';

import {
    fetchStudentTeacherComparison,
    type MetricComparisonRow,
    type SliceComparisonRow,
    type StudentTeacherComparison,
} from '../../api/studentTeacherComparison';

interface Props {
    projectId: number;
    experimentId: number | null;
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
    if (direction === 'regressed') return 'var(--color-error)';
    if (direction === 'exceeds') return 'var(--color-success)';
    return 'var(--color-success)';
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
                style={{
                    marginLeft: 'auto',
                    color: retentionColor(row.direction),
                    fontWeight: 600,
                }}
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

function FallbackCard({ testid, title, message }: { testid: string; title: string; message?: string | null }) {
    return (
        <section className="card" data-testid={testid} style={{ padding: 'var(--space-md)' }}>
            <h4 style={{ margin: 0 }}>{title}</h4>
            {message && (
                <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0' }}>{message}</p>
            )}
        </section>
    );
}

export default function StudentTeacherComparisonPanel({ projectId, experimentId, refreshToken }: Props) {
    const [data, setData] = useState<StudentTeacherComparison | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string>('');

    const load = useCallback(async () => {
        if (experimentId == null) return;
        setLoading(true);
        setError('');
        try {
            setData(await fetchStudentTeacherComparison(projectId, experimentId));
        } catch (err) {
            setError(errorDetail(err, 'Failed to load distillation comparison.'));
        } finally {
            setLoading(false);
        }
    }, [projectId, experimentId]);

    useEffect(() => {
        load();
    }, [load, refreshToken]);

    if (experimentId == null) return null;

    if (loading) {
        return (
            <section className="card" data-testid="st-comparison-loading" style={{ padding: 'var(--space-md)' }}>
                <div style={{ color: 'var(--text-secondary)' }}>Computing student-vs-teacher retention…</div>
            </section>
        );
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

    if (data.status === 'no_teacher_baseline') {
        return (
            <FallbackCard
                testid="st-comparison-no-teacher-baseline"
                title="Student vs teacher — no teacher baseline set"
                message={data.message}
            />
        );
    }
    if (data.status === 'no_student_eval') {
        return (
            <FallbackCard
                testid="st-comparison-no-student-eval"
                title="Student vs teacher — no student eval yet"
                message={data.message}
            />
        );
    }
    if (data.status === 'no_teacher_eval') {
        return (
            <FallbackCard
                testid="st-comparison-no-teacher-eval"
                title="Student vs teacher — teacher run not evaluated"
                message={data.message}
            />
        );
    }
    if (data.status === 'no_overlap') {
        return (
            <FallbackCard
                testid="st-comparison-no-overlap"
                title="Student vs teacher — metrics didn't overlap"
                message={data.message}
            />
        );
    }

    const { student, teacher, metric_comparisons, slice_comparisons, headline_quality_retained } = data;

    return (
        <section
            className="card"
            data-testid="st-comparison-panel"
            style={{ padding: 'var(--space-md)', display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}
        >
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
