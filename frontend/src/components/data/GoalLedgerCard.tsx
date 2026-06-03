/**
 * Arc H — Goal progress ledger card.
 *
 * Single-card surface that answers the question every user has at
 * every moment: "where am I in my project, and what's blocking
 * shipping?". The backend goal_service computes the 4-component
 * ledger (data_ready / gold_set / predicted_pass / eval_pass_rate);
 * this card just renders it with:
 *
 *   - One bar at the top: overall_progress with status colour-coding.
 *   - One row per component: label + value + status chip + a
 *     ``<Term>`` linking to the matching Academy lesson (Arc G).
 *   - A blockers strip at the bottom so the user always knows the
 *     next-action.
 *   - When ``has_explicit_goal`` is false, the card nudges the user
 *     to state their own goal instead of riding the f1 ≥ 0.70 default.
 */

import { useEffect, useState } from 'react';

import {
    getProjectGoalProgress,
    setProjectGoal,
} from '../../api/goal';
import type {
    GoalComponentStatus,
    GoalProgressComponent,
    GoalProgressResponse,
} from '../../api/goal';
import Term from '../shared/Term';
import './GoalLedgerCard.css';


interface GoalLedgerCardProps {
    projectId: number;
}


function _statusLabel(status: GoalComponentStatus): string {
    if (status === 'met') return 'Met';
    if (status === 'attention') return 'Needs work';
    return 'Pending';
}


function _ledgerStatusCopy(
    status: GoalProgressResponse['status'],
): { label: string; tone: 'green' | 'amber' | 'red' } {
    if (status === 'ready_to_ship') {
        return { label: 'Ready to ship', tone: 'green' };
    }
    if (status === 'in_progress') {
        return { label: 'In progress', tone: 'amber' };
    }
    return { label: 'Blocked', tone: 'red' };
}


function ComponentRow({ component }: { component: GoalProgressComponent }) {
    const pct = component.value === null
        ? 0
        : Math.round(component.value * 100);
    return (
        <li
            className={`goal-ledger__component goal-ledger__component--${component.status}`}
            data-testid={`goal-ledger-component-${component.id}`}
        >
            <div className="goal-ledger__component-head">
                <span className="goal-ledger__component-label">
                    <Term id={component.concept_id} label={component.label} />
                </span>
                <span
                    className={`goal-ledger__component-status goal-ledger__component-status--${component.status}`}
                >
                    {_statusLabel(component.status)}
                </span>
            </div>
            <div className="goal-ledger__component-bar" aria-hidden="true">
                <span
                    className={`goal-ledger__component-bar-fill goal-ledger__component-bar-fill--${component.status}`}
                    style={{ width: component.value === null ? '0%' : `${pct}%` }}
                />
            </div>
            <p className="goal-ledger__component-detail">{component.detail}</p>
        </li>
    );
}


export default function GoalLedgerCard({ projectId }: GoalLedgerCardProps) {
    const [progress, setProgress] = useState<GoalProgressResponse | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [editing, setEditing] = useState<boolean>(false);
    const [draftThreshold, setDraftThreshold] = useState<string>('0.85');
    const [draftTitle, setDraftTitle] = useState<string>('');
    const [saving, setSaving] = useState<boolean>(false);

    const load = async () => {
        setLoading(true);
        try {
            const payload = await getProjectGoalProgress(projectId);
            setProgress(payload);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load goal progress.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void load();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const handleSaveGoal = async () => {
        const threshold = Number(draftThreshold);
        if (!Number.isFinite(threshold)) {
            setError('Threshold must be a number between 0 and 1.');
            return;
        }
        setSaving(true);
        try {
            await setProjectGoal(projectId, {
                targetMetric: 'f1',
                targetThreshold: threshold,
                title: draftTitle.trim() || null,
            });
            setEditing(false);
            await load();
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to set goal.');
        } finally {
            setSaving(false);
        }
    };

    if (loading && !progress) {
        return (
            <section className="goal-ledger goal-ledger--loading" data-testid="goal-ledger-loading">
                Loading goal progress…
            </section>
        );
    }

    if (error && !progress) {
        return (
            <section className="goal-ledger goal-ledger--error" data-testid="goal-ledger-error">
                <p>{error}</p>
                <button type="button" className="btn btn-secondary" onClick={() => void load()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!progress) return null;

    const ledgerStatus = _ledgerStatusCopy(progress.status);
    const overallPct = Math.round(progress.overall_progress * 100);

    return (
        <section
            className={`goal-ledger goal-ledger--${ledgerStatus.tone}`}
            data-testid="goal-ledger"
        >
            <header className="goal-ledger__head">
                <div>
                    <p className="goal-ledger__eyebrow">Goal</p>
                    <h3 className="goal-ledger__title">
                        {progress.goal.title || (
                            progress.goal.target_metric === 'f1'
                                ? `Ship with ${progress.goal.target_metric.toUpperCase()} ≥ ${(progress.goal.target_threshold * 100).toFixed(0)}%`
                                : `Ship with ${progress.goal.target_metric} ≥ ${(progress.goal.target_threshold * 100).toFixed(0)}%`
                        )}
                    </h3>
                    {!progress.has_explicit_goal && !editing && (
                        <p className="goal-ledger__default-note" data-testid="goal-ledger-default-note">
                            No goal set yet — showing default ({progress.goal.target_metric} ≥ {(progress.goal.target_threshold * 100).toFixed(0)}%).
                            <button
                                type="button"
                                className="btn-link goal-ledger__default-cta"
                                onClick={() => setEditing(true)}
                            >
                                Set your own →
                            </button>
                        </p>
                    )}
                </div>
                <span
                    className={`goal-ledger__status goal-ledger__status--${ledgerStatus.tone}`}
                    data-testid="goal-ledger-status"
                >
                    {ledgerStatus.label}
                </span>
            </header>

            <div className="goal-ledger__overall" aria-label={`${overallPct}% toward goal`}>
                <div className="goal-ledger__overall-head">
                    <span className="goal-ledger__overall-pct" data-testid="goal-ledger-overall-pct">
                        {overallPct}%
                    </span>
                    <span className="goal-ledger__overall-label">toward goal</span>
                </div>
                <div className="goal-ledger__overall-bar" aria-hidden="true">
                    <span
                        className={`goal-ledger__overall-bar-fill goal-ledger__overall-bar-fill--${ledgerStatus.tone}`}
                        style={{ width: `${overallPct}%` }}
                    />
                </div>
            </div>

            <ul className="goal-ledger__components" data-testid="goal-ledger-components">
                {progress.components.map((component) => (
                    <ComponentRow key={component.id} component={component} />
                ))}
            </ul>

            {progress.blockers.length > 0 && (
                <div className="goal-ledger__blockers" data-testid="goal-ledger-blockers">
                    <p className="goal-ledger__blockers-title">What's blocking shipping</p>
                    <ul>
                        {progress.blockers.map((blocker, idx) => (
                            <li key={`${idx}-${blocker.slice(0, 20)}`}>{blocker}</li>
                        ))}
                    </ul>
                </div>
            )}

            {editing && (
                <div className="goal-ledger__edit" data-testid="goal-ledger-edit">
                    <label>
                        Title (optional)
                        <input
                            type="text"
                            value={draftTitle}
                            onChange={(e) => setDraftTitle(e.target.value)}
                            placeholder="e.g. Ship refund classifier"
                        />
                    </label>
                    <label>
                        F1 target (0-1)
                        <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.05"
                            value={draftThreshold}
                            onChange={(e) => setDraftThreshold(e.target.value)}
                        />
                    </label>
                    <div className="goal-ledger__edit-actions">
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={() => void handleSaveGoal()}
                            disabled={saving}
                        >
                            {saving ? 'Saving…' : 'Save goal'}
                        </button>
                        <button
                            type="button"
                            className="btn btn-ghost"
                            onClick={() => setEditing(false)}
                            disabled={saving}
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}
        </section>
    );
}
