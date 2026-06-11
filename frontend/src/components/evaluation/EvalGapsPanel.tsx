/**
 * EvalGapsPanel — Coach-stage-2 phase 3.
 *
 * Eval-side parallel to TrainingConfigGapsPanel. Given the project's
 * gold set + completed runs + train/eval label distributions, surfaces
 * three gaps:
 *
 *   - archetype coverage (does the gold set look like prior-passing
 *     gold sets?)
 *   - regression baseline (is there a promoted checkpoint to compare
 *     new runs against?)
 *   - train/eval label-KL (does the eval set predict prod?)
 *
 * Read-only in phase 3 — every signal's action is a navigate pointer.
 *
 * Backed by GET /api/projects/{id}/eval-gaps.
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import type { ErrorEnvelope } from '../../api/errors';
import { parseErrorEnvelope } from '../../api/errors';
import ErrorPanel from '../shared/ErrorPanel';
import {
    fetchEvalGaps,
    type EvalGapPatchResult,
    type EvalGapReport,
    type EvalGapSeverity,
} from '../../api/evalGaps';
import type { GapSuggestedAction } from '../../api/trainingConfigGaps';
import EvalGapPatchPreviewModal from './EvalGapPatchPreviewModal';
import '../training/TrainingConfigGapsPanel.css';

interface EvalGapsPanelProps {
    projectId: number;
}

const SEVERITY_META: Record<EvalGapSeverity, { label: string; icon: string }> = {
    ok: { label: 'OK', icon: '✓' },
    warn: { label: 'Warning', icon: '!' },
    block: { label: 'Blocker', icon: '✕' },
};

const OVERALL_HEADLINE: Record<EvalGapSeverity, string> = {
    ok: 'All clear — your eval setup looks honest.',
    warn: 'Warnings — your eval may not predict prod performance.',
    block: 'Blockers — eval is missing prerequisites and the numbers will be misleading.',
};

function targetUrl(
    projectId: number,
    action: GapSuggestedAction | null,
): string | null {
    if (!action || !action.target) return null;
    switch (action.target) {
        case 'archetype-comparison-panel':
            return `/project/${projectId}/training-config#archetype-comparison-panel`;
        case 'checkpoints-panel':
            return `/project/${projectId}/training-config#checkpoints-panel`;
        case 'data-studio-splits':
            return `/project/${projectId}/data-studio#splits`;
        case 'recipe-picker':
            return `/project/${projectId}/recipe-picker`;
        default:
            return null;
    }
}

export default function EvalGapsPanel({ projectId }: EvalGapsPanelProps) {
    const navigate = useNavigate();
    const [data, setData] = useState<EvalGapReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<ErrorEnvelope | null>(null);
    const [expanded, setExpanded] = useState<Set<string>>(new Set());
    // Phase 5 — patch modal state. signalId = open the patch preview;
    // null = closed. After apply we re-fetch the panel + surface a
    // small toast confirming what landed.
    const [patchSignalId, setPatchSignalId] = useState<string | null>(null);
    const [lastApplied, setLastApplied] = useState<EvalGapPatchResult | null>(
        null,
    );

    const refresh = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetchEvalGaps(projectId);
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

    const handlePatchApplied = useCallback(
        async (result: EvalGapPatchResult) => {
            setLastApplied(result);
            await refresh();
        },
        [refresh],
    );

    if (loading && !data) {
        return (
            <div
                className="tcg-panel tcg-panel--loading"
                data-testid="eval-gaps"
            >
                Loading Eval gaps…
            </div>
        );
    }

    if (error) {
        return (
            <div className="tcg-panel tcg-panel--error" data-testid="eval-gaps">
                <ErrorPanel
                    envelope={error}
                    onDismiss={() => setError(null)}
                    testIdPrefix="eval-gaps-load-error"
                    actions={
                        <button
                            type="button"
                            className="btn btn-link"
                            onClick={() => void refresh()}
                            data-testid="eval-gaps-retry"
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
            id="eval-gaps-panel"
            className="tcg-panel"
            data-testid="eval-gaps"
            data-overall={data.overall}
        >
            <header className={`tcg-panel__head tcg-panel__head--${data.overall}`}>
                <div className="tcg-panel__head-line">
                    <span
                        className={`tcg-panel__overall-badge tcg-panel__overall-badge--${data.overall}`}
                        data-testid="eval-gaps-overall-badge"
                    >
                        {SEVERITY_META[data.overall].icon} {SEVERITY_META[data.overall].label}
                    </span>
                    <h3 className="tcg-panel__title">Eval Gaps</h3>
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
                            data-testid={`eval-gaps-group-${group.id}`}
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
                                            data-testid={`eval-gaps-signal-${sig.id}`}
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
                                                            data-testid={`eval-gaps-why-${sig.id}`}
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
                                                            data-testid={`eval-gaps-why-text-${sig.id}`}
                                                        >
                                                            {sig.why_it_matters}
                                                        </p>
                                                    )}
                                                </div>
                                                <div className="tcg-panel__actions">
                                                    {sig.apply_patch_kind && (
                                                        <button
                                                            type="button"
                                                            className="tcg-panel__action btn btn-sm"
                                                            onClick={() => {
                                                                setLastApplied(null);
                                                                setPatchSignalId(sig.id);
                                                            }}
                                                            data-testid={`eval-gaps-apply-${sig.id}`}
                                                            title="Preview the patch before applying"
                                                        >
                                                            Apply fix
                                                        </button>
                                                    )}
                                                    {sig.suggested_action?.label && (
                                                        <button
                                                            type="button"
                                                            className="tcg-panel__action btn btn-sm"
                                                            onClick={() => {
                                                                if (url) navigate(url);
                                                            }}
                                                            disabled={!url}
                                                            data-testid={`eval-gaps-action-${sig.id}`}
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

            {lastApplied && (
                <div
                    className="tcg-panel__apply-toast"
                    data-testid="eval-gaps-apply-toast"
                    role="status"
                >
                    <strong>Applied:</strong> {lastApplied.patch_label}.
                    <button
                        type="button"
                        className="tcg-panel__apply-toast-dismiss"
                        onClick={() => setLastApplied(null)}
                        aria-label="Dismiss apply summary"
                    >
                        ×
                    </button>
                </div>
            )}

            <footer className="tcg-panel__foot">
                Computed at {new Date(data.computed_at).toLocaleTimeString()}.{' '}
                <button
                    type="button"
                    className="btn btn-link"
                    onClick={() => void refresh()}
                    data-testid="eval-gaps-refresh"
                >
                    Refresh
                </button>
            </footer>

            {patchSignalId && (
                <EvalGapPatchPreviewModal
                    projectId={projectId}
                    signalId={patchSignalId}
                    onClose={() => setPatchSignalId(null)}
                    onApplied={(r) => void handlePatchApplied(r)}
                />
            )}
        </section>
    );
}
