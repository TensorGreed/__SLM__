/**
 * Active-learning recommender panel (Theme 8 Epic 2).
 *
 * Shows on the Eval tab below FailureClustersPanel when the
 * selected experiment has a completed eval with failed rows.
 * Surfaces up to N candidate rows (model got wrong) and lets the
 * user one-click promote the gold answers into the project's
 * SYNTHETIC dataset for the next training run.
 *
 * Honest framing: this is **failed-row promotion**, not full
 * active learning. Eval handlers today don't emit per-row
 * confidence/logprob, so we can't distinguish overconfident-wrong
 * from knows-it-doesn't-know. We rank by row score severity (when
 * available) instead. Confidence-aware ranking deferred until
 * handlers expose those signals.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import {
    fetchActiveLearningProposal,
    promoteActiveLearningRows,
    type ActiveLearningCandidate,
    type ActiveLearningProposal,
} from '../../api/activeLearning';
import { useToastStore } from '../../stores/toastStore';

interface ActiveLearningPanelProps {
    projectId: number;
    experimentId: number;
    /** Triggers a refetch when this changes (e.g. eval just re-ran). */
    refreshToken?: unknown;
}

function extractErrorMessage(err: unknown): string {
    if (typeof err === 'object' && err !== null) {
        const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) return detail;
        const message = (err as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) return message;
    }
    return 'Unknown error';
}

function truncate(s: string, n: number): string {
    if (!s) return '';
    return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

export default function ActiveLearningPanel({
    projectId,
    experimentId,
    refreshToken,
}: ActiveLearningPanelProps) {
    const [proposal, setProposal] = useState<ActiveLearningProposal | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string>('');
    const [selectedIndexes, setSelectedIndexes] = useState<Set<number>>(new Set());
    const [promoting, setPromoting] = useState(false);
    const [showAll, setShowAll] = useState(false);
    const { addToast } = useToastStore();

    const loadProposal = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const res = await fetchActiveLearningProposal(projectId, experimentId, 50);
            setProposal(res);
            // Default selection = every actionable candidate (the
            // user can untick if they want a subset before promoting).
            const defaultSel = new Set(
                res.candidates
                    .filter((c) => !c.already_promoted)
                    .map((c) => c.row_index),
            );
            setSelectedIndexes(defaultSel);
        } catch (err) {
            setError(extractErrorMessage(err));
        } finally {
            setLoading(false);
        }
    }, [projectId, experimentId]);

    useEffect(() => {
        loadProposal();
    }, [loadProposal, refreshToken]);

    const toggleRow = (rowIndex: number) => {
        setSelectedIndexes((prev) => {
            const next = new Set(prev);
            if (next.has(rowIndex)) {
                next.delete(rowIndex);
            } else {
                next.add(rowIndex);
            }
            return next;
        });
    };

    const handlePromote = async () => {
        if (selectedIndexes.size === 0) return;
        setPromoting(true);
        try {
            const res = await promoteActiveLearningRows(
                projectId,
                experimentId,
                Array.from(selectedIndexes),
            );
            const skippedNote =
                res.skipped_already_promoted > 0
                    ? ` (${res.skipped_already_promoted} already in dataset)`
                    : '';
            addToast(
                `Added ${res.promoted_count} example${res.promoted_count === 1 ? '' : 's'} to synthetic dataset${skippedNote}`,
                'success',
                4500,
            );
            await loadProposal();
            setSelectedIndexes(new Set());
        } catch (err) {
            const message = extractErrorMessage(err);
            addToast(`Promote failed: ${message}`, 'error', 5000);
        } finally {
            setPromoting(false);
        }
    };

    const visibleCandidates = useMemo<ActiveLearningCandidate[]>(() => {
        if (!proposal || !Array.isArray(proposal.candidates)) return [];
        return showAll ? proposal.candidates : proposal.candidates.slice(0, 5);
    }, [proposal, showAll]);

    if (loading) {
        return (
            <section
                className="card"
                data-testid="active-learning-panel-loading"
                style={{ padding: 'var(--space-md)' }}
            >
                <div style={{ color: 'var(--text-secondary)' }}>
                    Looking for failed eval rows you can use to improve the model…
                </div>
            </section>
        );
    }

    if (error) {
        return (
            <section
                className="card"
                role="alert"
                data-testid="active-learning-panel-error"
                style={{
                    padding: 'var(--space-md)',
                    background: 'var(--color-error-bg)',
                    color: 'var(--color-error)',
                }}
            >
                Couldn't load active-learning candidates: {error}
            </section>
        );
    }

    // Defensive: handle empty / malformed responses (e.g. tests that
    // mock unrelated endpoints + leave this one returning `{}`).
    // Treat anything without a candidates array as "nothing to show".
    if (
        !proposal
        || !Array.isArray(proposal.candidates)
        || !proposal.total_predictions
    ) {
        return null;
    }

    if (proposal.total_failures === 0) {
        return (
            <section
                className="card"
                data-testid="active-learning-panel-empty"
                style={{ padding: 'var(--space-md)' }}
            >
                <h4 style={{ margin: 0 }}>🎉 No failed eval rows to learn from</h4>
                <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0' }}>
                    Every row in this eval result passed. Nothing to promote into
                    the synthetic dataset right now.
                </p>
            </section>
        );
    }

    return (
        <section
            className="card"
            data-testid="active-learning-panel"
            style={{
                padding: 'var(--space-md)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-md)',
            }}
        >
            <div
                style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    justifyContent: 'space-between',
                    gap: 'var(--space-md)',
                }}
            >
                <div>
                    <h4 style={{ margin: 0 }}>
                        Add {proposal.total_failures} failing example
                        {proposal.total_failures === 1 ? '' : 's'} to next training
                    </h4>
                    <p
                        style={{
                            margin: '4px 0 0',
                            color: 'var(--text-secondary)',
                            fontSize: '0.9rem',
                        }}
                    >
                        The model got these wrong. Promoting their gold answers into
                        the synthetic dataset lets the next training run learn from
                        them.
                        {proposal.promoted_count > 0 && (
                            <>
                                {' '}
                                <strong>{proposal.promoted_count} already promoted</strong>{' '}
                                from earlier sessions.
                            </>
                        )}
                    </p>
                </div>
                <div
                    style={{
                        display: 'flex',
                        gap: 'var(--space-sm)',
                        alignItems: 'center',
                    }}
                >
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={handlePromote}
                        disabled={promoting || selectedIndexes.size === 0}
                        data-testid="active-learning-promote"
                    >
                        {promoting
                            ? 'Adding…'
                            : `Add ${selectedIndexes.size} to training`}
                    </button>
                </div>
            </div>

            <div
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 'var(--space-sm)',
                }}
            >
                {visibleCandidates.map((c) => {
                    const checked = selectedIndexes.has(c.row_index);
                    return (
                        <label
                            key={c.row_index}
                            data-testid={`active-learning-row-${c.row_index}`}
                            style={{
                                display: 'grid',
                                gridTemplateColumns: 'auto 1fr',
                                gap: 'var(--space-sm)',
                                padding: 'var(--space-sm)',
                                borderRadius: 'var(--radius-sm)',
                                border: '1px solid var(--border-color)',
                                background: c.already_promoted
                                    ? 'var(--bg-subtle)'
                                    : 'var(--bg-card)',
                                cursor: c.already_promoted ? 'not-allowed' : 'pointer',
                                opacity: c.already_promoted ? 0.65 : 1,
                            }}
                        >
                            <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => toggleRow(c.row_index)}
                                disabled={c.already_promoted || promoting}
                                aria-label={`Row ${c.row_index} — ${c.failure_reason}`}
                                style={{ marginTop: 4 }}
                            />
                            <div style={{ fontSize: '0.85rem' }}>
                                <div style={{ fontWeight: 600 }}>
                                    Row #{c.row_index}{' '}
                                    <span
                                        className="badge badge-warning"
                                        style={{ marginLeft: 4 }}
                                    >
                                        {c.failure_reason}
                                    </span>
                                    {c.already_promoted && (
                                        <span
                                            className="badge badge-success"
                                            style={{ marginLeft: 4 }}
                                            data-testid={`active-learning-row-${c.row_index}-promoted`}
                                        >
                                            ✓ already promoted
                                        </span>
                                    )}
                                </div>
                                <div
                                    style={{
                                        marginTop: 4,
                                        color: 'var(--text-secondary)',
                                    }}
                                >
                                    <strong>Q:</strong> {truncate(c.prompt, 120)}
                                </div>
                                <div
                                    style={{
                                        marginTop: 2,
                                        color: 'var(--color-error)',
                                    }}
                                >
                                    <strong>Model said:</strong> {truncate(c.prediction, 120)}
                                </div>
                                <div
                                    style={{
                                        marginTop: 2,
                                        color: 'var(--color-success)',
                                    }}
                                >
                                    <strong>Should say:</strong> {truncate(c.reference, 120)}
                                </div>
                            </div>
                        </label>
                    );
                })}
            </div>

            {proposal.candidates.length > 5 && (
                <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => setShowAll((v) => !v)}
                    data-testid="active-learning-toggle-all"
                    style={{ alignSelf: 'flex-start', fontSize: '0.85rem' }}
                >
                    {showAll
                        ? 'Show top 5'
                        : `Show all ${proposal.candidates.length} candidates`}
                </button>
            )}
        </section>
    );
}
