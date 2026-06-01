/**
 * Header strip for the annotation labeler.
 *
 * Renders the job name + label_type, a "labeled / target" count, and
 * a progress bar that shows progress relative to the configured target
 * (or against total when no target is set).
 */

import type { JobStats } from '../../api/annotation';

interface AnnotationProgressProps {
    jobName: string;
    stats: JobStats;
}

function computeProgressFraction(stats: JobStats): number {
    const denom = stats.target_rows && stats.target_rows > 0
        ? stats.target_rows
        : stats.total;
    if (denom <= 0) return 0;
    return Math.min(1, stats.labeled / denom);
}

function agreementLabel(metric: string): string {
    if (metric === 'cohens_kappa') return 'κ';
    if (metric === 'span_f1') return 'span-F1';
    if (metric === 'preference_agreement') return 'agree';
    return metric;
}

export default function AnnotationProgress({
    jobName,
    stats,
}: AnnotationProgressProps) {
    const fraction = computeProgressFraction(stats);
    const denom = stats.target_rows && stats.target_rows > 0
        ? stats.target_rows
        : stats.total;
    const denomLabel = stats.target_rows && stats.target_rows > 0
        ? `${stats.target_rows} target`
        : `${stats.total} total`;

    return (
        <div className="annotation-progress" data-testid="annotation-progress">
            <div
                style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'baseline',
                    marginBottom: 'var(--space-xs)',
                }}
            >
                <div>
                    <strong style={{ fontSize: '0.95rem' }}>{jobName}</strong>
                    <span
                        style={{
                            marginLeft: 8,
                            color: 'var(--text-secondary)',
                            fontSize: '0.8rem',
                        }}
                    >
                        {stats.label_type}
                    </span>
                </div>
                <div
                    style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}
                    data-testid="annotation-progress-counts"
                >
                    <span data-testid="annotation-progress-labeled">
                        {stats.labeled}
                    </span>
                    {' / '}
                    <span data-testid="annotation-progress-denom">
                        {denom}
                    </span>
                    {' '}
                    <span style={{ opacity: 0.7 }}>({denomLabel})</span>
                    {stats.assigned > 0 && (
                        <span
                            style={{ marginLeft: 12, opacity: 0.7 }}
                            data-testid="annotation-progress-assigned"
                        >
                            {stats.assigned} in flight
                        </span>
                    )}
                    {stats.inter_annotator_agreement && (
                        <span
                            style={{
                                marginLeft: 12,
                                padding: '1px 6px',
                                border: '1px solid var(--border-color)',
                                borderRadius: 'var(--radius-sm)',
                                background: 'var(--bg-secondary)',
                            }}
                            data-testid="annotation-progress-agreement"
                            title={
                                'Inter-annotator agreement across '
                                + `${stats.inter_annotator_agreement.reviewer_count}`
                                + ' reviewers · '
                                + `${stats.inter_annotator_agreement.overlap_rows}`
                                + ' overlap row(s) · '
                                + `${stats.inter_annotator_agreement.pair_count}`
                                + ' reviewer pair(s)'
                            }
                        >
                            {agreementLabel(
                                stats.inter_annotator_agreement.metric,
                            )}
                            {' '}
                            {stats.inter_annotator_agreement.value.toFixed(2)}
                        </span>
                    )}
                </div>
            </div>
            <div
                role="progressbar"
                aria-valuenow={Math.round(fraction * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                style={{
                    height: 6,
                    background: 'var(--bg-secondary)',
                    borderRadius: 'var(--radius-sm)',
                    overflow: 'hidden',
                }}
            >
                <div
                    data-testid="annotation-progress-bar"
                    style={{
                        width: `${fraction * 100}%`,
                        height: '100%',
                        background: 'var(--text-primary)',
                        transition: 'width 200ms ease',
                    }}
                />
            </div>
        </div>
    );
}
