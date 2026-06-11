/**
 * TrainingConfigGapsPanel — Coach-stage-2 phase 1.
 *
 * Parallel to DataHealthReportPanel but for the *training config* side:
 * given the project's recipe + base model + labelled-row count, what
 * gaps exist in the hyperparameters the trainer will actually use?
 *
 * Phase 1 is read-only — every signal's action is a ``navigate``
 * pointer to the relevant config surface. Phase 2 will add
 * ``apply_config_patch`` so signals like "max_seq_length truncates
 * 23% of rows" can be one-click bumped.
 *
 * Backed by GET /api/projects/{id}/training-config-gaps.
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import type { ErrorEnvelope } from '../../api/errors';
import { parseErrorEnvelope } from '../../api/errors';
import ErrorPanel from '../shared/ErrorPanel';
import {
    fetchTrainingConfigGaps,
    type GapSeverity,
    type GapSuggestedAction,
    type TrainingConfigGapReport,
} from '../../api/trainingConfigGaps';
import './TrainingConfigGapsPanel.css';

interface TrainingConfigGapsPanelProps {
    projectId: number;
}

const SEVERITY_META: Record<GapSeverity, { label: string; icon: string }> = {
    ok: { label: 'OK', icon: '✓' },
    warn: { label: 'Warning', icon: '!' },
    block: { label: 'Blocker', icon: '✕' },
};

const OVERALL_HEADLINE: Record<GapSeverity, string> = {
    ok: 'All clear — no training-config gaps detected.',
    warn: 'Warnings — training will run but the config could be tuned.',
    block: 'Blockers — training is likely to fail or produce unreliable results until these are fixed.',
};

// Map navigate target → URL. Mirrors the small registry in
// CoachSuggestion.tsx; kept local so the panel can render its action
// chips even when not invoked through the Coach action router.
function targetUrl(
    projectId: number,
    action: GapSuggestedAction | null,
): string | null {
    if (!action || !action.target) return null;
    const params = (action.params || {}) as Record<string, unknown>;
    switch (action.target) {
        case 'training-config': {
            const qs = new URLSearchParams();
            const evalSteps = params['recommended_eval_steps'];
            if (typeof evalSteps === 'number' && Number.isFinite(evalSteps)) {
                qs.set('recommended_eval_steps', String(Math.trunc(evalSteps)));
            }
            const epochs = params['recommended_num_epochs'];
            if (typeof epochs === 'number' && Number.isFinite(epochs)) {
                qs.set('recommended_num_epochs', String(Math.trunc(epochs)));
            }
            const warmup = params['recommended_warmup_ratio'];
            if (typeof warmup === 'number' && Number.isFinite(warmup)) {
                qs.set('recommended_warmup_ratio', String(warmup));
            }
            const tail = qs.toString() ? `?${qs.toString()}` : '';
            return `/project/${projectId}/training-config${tail}`;
        }
        case 'training-base-model-picker': {
            const recommended = params['recommended_base_model'];
            const tail =
                typeof recommended === 'string' && recommended.length > 0
                    ? `?recommended_base_model=${encodeURIComponent(recommended)}`
                    : '';
            return `/project/${projectId}/training-config${tail}#base-model`;
        }
        case 'recipe-picker':
            return `/project/${projectId}/recipe-picker`;
        default:
            return null;
    }
}

export default function TrainingConfigGapsPanel({
    projectId,
}: TrainingConfigGapsPanelProps) {
    const navigate = useNavigate();
    const [data, setData] = useState<TrainingConfigGapReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<ErrorEnvelope | null>(null);
    const [expanded, setExpanded] = useState<Set<string>>(new Set());

    const refresh = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetchTrainingConfigGaps(projectId);
            setData(res);
        } catch (err) {
            setError(parseErrorEnvelope(err));
            setData(null);
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    const toggleExpanded = (id: string) => {
        setExpanded((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    if (loading && !data) {
        return (
            <div
                className="tcg-panel tcg-panel--loading"
                data-testid="training-config-gaps"
            >
                Loading Training Config gaps…
            </div>
        );
    }

    if (error) {
        return (
            <div
                className="tcg-panel tcg-panel--error"
                data-testid="training-config-gaps"
            >
                <ErrorPanel
                    envelope={error}
                    onDismiss={() => setError(null)}
                    testIdPrefix="training-config-gaps-load-error"
                    actions={
                        <button
                            type="button"
                            className="btn btn-link"
                            onClick={() => void refresh()}
                            data-testid="training-config-gaps-retry"
                        >
                            Retry
                        </button>
                    }
                />
            </div>
        );
    }

    if (!data) return null;

    const populatedGroups = data.groups.filter((g) => g.signals.length > 0);

    return (
        <section
            id="training-config-gaps-panel"
            className="tcg-panel"
            data-testid="training-config-gaps"
            data-overall={data.overall}
        >
            <header className={`tcg-panel__head tcg-panel__head--${data.overall}`}>
                <div className="tcg-panel__head-line">
                    <span
                        className={`tcg-panel__overall-badge tcg-panel__overall-badge--${data.overall}`}
                        data-testid="training-config-gaps-overall-badge"
                    >
                        {SEVERITY_META[data.overall].icon} {SEVERITY_META[data.overall].label}
                    </span>
                    <h3 className="tcg-panel__title">Training Config Gaps</h3>
                </div>
                <p className="tcg-panel__headline">{OVERALL_HEADLINE[data.overall]}</p>
                <p className="tcg-panel__counts">
                    {data.severity_summary.block > 0 && (
                        <>
                            <strong>{data.severity_summary.block}</strong> blocker{data.severity_summary.block === 1 ? '' : 's'}
                        </>
                    )}
                    {data.severity_summary.block > 0 && data.severity_summary.warn > 0 && ' · '}
                    {data.severity_summary.warn > 0 && (
                        <>
                            <strong>{data.severity_summary.warn}</strong> warning{data.severity_summary.warn === 1 ? '' : 's'}
                        </>
                    )}
                    {(data.severity_summary.block > 0 || data.severity_summary.warn > 0) && data.severity_summary.ok > 0 && ' · '}
                    {data.severity_summary.ok > 0 && (
                        <>
                            <strong>{data.severity_summary.ok}</strong> ok
                        </>
                    )}
                </p>
            </header>

            {populatedGroups.length === 0 ? (
                <p className="tcg-panel__headline">No signals yet.</p>
            ) : (
                <div className="tcg-panel__groups">
                    {populatedGroups.map((group) => (
                        <div
                            key={group.id}
                            className="tcg-panel__group"
                            data-testid={`training-config-gaps-group-${group.id}`}
                        >
                            <header className="tcg-panel__group-head">
                                <h4 className="tcg-panel__group-title">{group.title}</h4>
                                <span className="tcg-panel__group-subtitle">
                                    {group.subtitle}
                                </span>
                            </header>
                            <ul className="tcg-panel__signal-list">
                                {group.signals.map((sig) => {
                                    const isExpanded = expanded.has(sig.id);
                                    const url = targetUrl(projectId, sig.suggested_action);
                                    return (
                                        <li
                                            key={sig.id}
                                            className={`tcg-panel__signal tcg-panel__signal--${sig.severity}`}
                                            data-testid={`training-config-gaps-signal-${sig.id}`}
                                            data-severity={sig.severity}
                                        >
                                            <div className="tcg-panel__signal-head">
                                                <span
                                                    className={`tcg-panel__sev-badge tcg-panel__sev-badge--${sig.severity}`}
                                                    aria-label={SEVERITY_META[sig.severity].label}
                                                >
                                                    {SEVERITY_META[sig.severity].icon}
                                                </span>
                                                <div className="tcg-panel__signal-body">
                                                    {sig.plain_english && (
                                                        <p className="tcg-panel__plain-english">
                                                            {sig.plain_english}
                                                        </p>
                                                    )}
                                                    <p className="tcg-panel__headline-line">
                                                        {sig.headline}
                                                    </p>
                                                    {sig.why_it_matters && (
                                                        <button
                                                            type="button"
                                                            className="tcg-panel__why-toggle"
                                                            onClick={() => toggleExpanded(sig.id)}
                                                            data-testid={`training-config-gaps-why-${sig.id}`}
                                                            aria-label={
                                                                isExpanded
                                                                    ? `Hide why this matters for ${sig.id}`
                                                                    : `Show why this matters for ${sig.id}`
                                                            }
                                                        >
                                                            {isExpanded ? '− Why this matters' : '+ Why this matters'}
                                                        </button>
                                                    )}
                                                    {isExpanded && sig.why_it_matters && (
                                                        <p
                                                            className="tcg-panel__why"
                                                            data-testid={`training-config-gaps-why-text-${sig.id}`}
                                                        >
                                                            {sig.why_it_matters}
                                                        </p>
                                                    )}
                                                </div>
                                                <div className="tcg-panel__actions">
                                                    {sig.suggested_action?.label && (
                                                        <button
                                                            type="button"
                                                            className="tcg-panel__action btn btn-sm"
                                                            onClick={() => {
                                                                if (url) navigate(url);
                                                            }}
                                                            disabled={!url}
                                                            data-testid={`training-config-gaps-action-${sig.id}`}
                                                            title={
                                                                url
                                                                    ? `Navigate to ${sig.suggested_action.target ?? ''}`
                                                                    : 'Action not yet wired up'
                                                            }
                                                        >
                                                            {sig.suggested_action.label}
                                                        </button>
                                                    )}
                                                </div>
                                            </div>
                                        </li>
                                    );
                                })}
                            </ul>
                        </div>
                    ))}
                </div>
            )}

            <footer className="tcg-panel__foot">
                Computed at {new Date(data.computed_at).toLocaleTimeString()}.{' '}
                <button
                    type="button"
                    className="btn btn-link"
                    onClick={() => void refresh()}
                    data-testid="training-config-gaps-refresh"
                >
                    Refresh
                </button>
            </footer>
        </section>
    );
}
