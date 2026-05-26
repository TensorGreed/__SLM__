/**
 * ArchetypeComparisonPanel — USER-SUCCESS Epic 8 Phase 8b.
 *
 * Mounted on the Training Config page above the trainability-
 * forecast panel. Shows the current project's structural features
 * against the per-recipe archetype's p25-p75 band, with a status
 * badge per feature and a one-click suggested-action button when
 * the backend computed one (matches Coach Mode's action contract
 * so we reuse the same handlers — runPlaybookAsync via the Jobs
 * framework for synth-row actions, window.location.assign for
 * navigate targets).
 *
 * Self-hide rules:
 *   * `summary === "healthy"` AND `n_user_projects < 1` →
 *     don't lecture the first user about their own archetype
 *     (cold-start UX).
 *   * Fetch error / 4xx → silently render nothing. The panel is
 *     advisory; failures should not push noise.
 */

import { useEffect, useState } from 'react';

import {
    fetchProjectArchetypeComparison,
    type FeatureComparison,
    type FeatureStatus,
    type ProjectArchetypeComparison,
} from '../../api/archetypeComparison';
import { runPlaybookAsync } from '../../api/synthPlaybook';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import './ArchetypeComparisonPanel.css';


interface Props {
    projectId: number;
}


/** Same hard-nav pattern as the other panels — keeps the component
 *  mountable from contexts without a Router (existing
 *  TrainingConfigPage tests etc.). The Coach suggestion handlers
 *  use react-router; we mirror their *behavior* via window.
 */
function navigateTo(url: string): void {
    window.location.assign(url);
}


const STATUS_LABEL: Record<FeatureStatus, string> = {
    ok: 'in band',
    below: 'below cohort',
    above: 'above cohort',
    missing: 'no data',
};


function formatValue(value: number | null, unit: string): string {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return '—';
    }
    if (unit === 'rows' || unit === 'chars') {
        return Math.round(value).toLocaleString();
    }
    // ratio + bits → two decimals.
    return value.toFixed(2);
}


function formatBand(
    p25: number | null,
    p50: number | null,
    p75: number | null,
    unit: string,
): string {
    if (p25 === null || p75 === null) return '—';
    if (p50 !== null && p25 === p50 && p50 === p75) {
        // Single-contribution cohort — show the singleton, not a
        // misleading "X – X" range.
        return `~${formatValue(p50, unit)}`;
    }
    return `${formatValue(p25, unit)} – ${formatValue(p75, unit)}`;
}


export default function ArchetypeComparisonPanel({ projectId }: Props) {
    const [data, setData] = useState<ProjectArchetypeComparison | null>(null);
    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        fetchProjectArchetypeComparison(projectId)
            .then((payload) => {
                if (!cancelled) setData(payload);
            })
            .catch(() => {
                // Advisory panel — 4xx / network errors silently hide.
                if (!cancelled) setData(null);
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    if (loading || !data) return null;

    // Cold-start self-hide: nothing for the first user to compare against
    // their own work yet, and nothing about their data is drifting.
    if (
        data.summary === 'healthy'
        && data.archetype.n_user_projects < 1
    ) {
        return null;
    }

    const handleAction = async (feature: FeatureComparison) => {
        if (!feature.suggested_action || submitting) return;
        const { kind, params } = feature.suggested_action;

        if (kind === 'navigate') {
            const target = String(params['target'] || '');
            if (target === 'data-studio-diversity') {
                navigateTo(`/project/${projectId}/data-studio#diversity`);
                return;
            }
            // Generic fallback — drop the user on the project page.
            navigateTo(`/project/${projectId}`);
            return;
        }

        if (kind === 'run_playbook') {
            const mode = String(params['mode'] || '') as Parameters<
                typeof runPlaybookAsync
            >[1]['mode'];
            const targetCount = Number(params['target_count']);
            const targetClass = params['target_class'] as string | undefined;
            if (!mode || !Number.isFinite(targetCount) || targetCount < 1) {
                toast.error('Archetype suggestion is missing action parameters.');
                return;
            }
            setSubmitting(true);
            try {
                const job = await runPlaybookAsync(projectId, {
                    mode,
                    targetCount,
                    targetClass: targetClass ?? null,
                    backend: null,
                });
                toast.info(
                    `Synth ${mode} queued — track in the bell (job #${job.id})`,
                    4000,
                );
                void useJobsStore.getState().refreshAfterLocalChange();
            } catch (err) {
                const detail =
                    (err as { response?: { data?: { detail?: string } } })?.response
                        ?.data?.detail;
                toast.error(
                    detail
                        ?? 'Archetype suggestion failed. Check the synth panel for details.',
                );
            } finally {
                setSubmitting(false);
            }
        }
    };

    const summaryCopy: Record<typeof data.summary, string> = {
        healthy: 'Your project shape matches successful cohorts.',
        below_cohort:
            'Several features are below the successful cohort range — addressing them improves your odds.',
        above_cohort:
            'Several features are above the successful cohort range — usually fine, but worth a sanity check.',
        mixed:
            'Mixed signals — some features are below the cohort, others above. Review each row.',
    };

    return (
        <section
            className={`archetype-cmp archetype-cmp--${data.summary}`}
            data-testid="archetype-comparison-panel"
        >
            <header className="archetype-cmp__head">
                <div>
                    <h3>How your project compares to successful ones</h3>
                    <p className="archetype-cmp__subtitle">
                        {summaryCopy[data.summary]} Based on{' '}
                        <strong>{data.archetype.n_passing_projects}</strong>{' '}
                        {data.archetype.n_passing_projects === 1 ? 'project' : 'projects'}
                        {data.archetype.n_template_seeds > 0 && (
                            <> (incl. {data.archetype.n_template_seeds} template seed
                            {data.archetype.n_template_seeds === 1 ? '' : 's'})</>
                        )}
                        .
                    </p>
                </div>
            </header>

            <table className="archetype-cmp__table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Your value</th>
                        <th>Cohort range (p25 – p75)</th>
                        <th>Status</th>
                        <th>Suggestion</th>
                    </tr>
                </thead>
                <tbody>
                    {data.features.map((f) => (
                        <tr
                            key={f.feature_id}
                            className={`archetype-cmp__row archetype-cmp__row--${f.status}`}
                            data-testid={`archetype-comparison-row-${f.feature_id}`}
                        >
                            <td>{f.label}</td>
                            <td className="archetype-cmp__val">
                                {formatValue(f.your_value, f.unit)} {f.unit !== 'ratio' ? '' : ''}
                            </td>
                            <td className="archetype-cmp__val">
                                {formatBand(f.archetype_p25, f.archetype_p50, f.archetype_p75, f.unit)}
                            </td>
                            <td>
                                <span
                                    className={`archetype-cmp__badge archetype-cmp__badge--${f.status}`}
                                    data-testid={`archetype-comparison-status-${f.feature_id}`}
                                >
                                    {STATUS_LABEL[f.status]}
                                </span>
                            </td>
                            <td>
                                {f.suggestion && (
                                    <div className="archetype-cmp__suggestion-cell">
                                        <span className="archetype-cmp__suggestion-text">
                                            {f.suggestion}
                                        </span>
                                        {f.suggested_action && (
                                            <button
                                                type="button"
                                                className="archetype-cmp__action-btn"
                                                onClick={() => handleAction(f)}
                                                disabled={submitting}
                                                data-testid={`archetype-comparison-action-${f.feature_id}`}
                                            >
                                                {f.suggested_action.kind === 'run_playbook'
                                                    ? 'Generate via playbook'
                                                    : 'Open'}
                                            </button>
                                        )}
                                    </div>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <details className="archetype-cmp__cohort">
                <summary>Cohort ({data.archetype.cohort_provenance.length})</summary>
                <ul className="archetype-cmp__cohort-list">
                    {data.archetype.cohort_provenance.map((c) => (
                        <li key={`${c.source}-${c.id}`} data-testid={`archetype-cohort-${c.id}`}>
                            <span className={`archetype-cmp__cohort-badge archetype-cmp__cohort-badge--${c.source}`}>
                                {c.source}
                            </span>
                            <span>{c.name}</span>
                            {c.pass_rate !== null && (
                                <span className="archetype-cmp__cohort-f1">
                                    f1 {c.pass_rate.toFixed(2)}
                                </span>
                            )}
                        </li>
                    ))}
                </ul>
            </details>
        </section>
    );
}
