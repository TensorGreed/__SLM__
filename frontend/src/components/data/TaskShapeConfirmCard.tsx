/**
 * "What kind of task is this?" — shows the unified detector's best guess
 * in plain language with its reasons, and confirms it in one click. The
 * confirmed shape is what dataset prep, training and evaluation use.
 */

import { useMemo, useState } from 'react';
import {
    confirmTaskShape,
    type ConfirmTaskShapeResult,
    type TaskShapeCandidate,
    type TaskShapeCatalogEntry,
    type TaskShapeDetection,
} from '../../api/taskShape';

interface TaskShapeConfirmCardProps {
    projectId: number;
    detection: TaskShapeDetection | null | undefined;
    /** Full list of shapes for "something else"; defaults to the detected candidates. */
    catalog?: TaskShapeCatalogEntry[];
    /** Already-confirmed shape (shown instead of the question). */
    confirmedLabel?: string | null;
    onConfirmed?: (result: ConfirmTaskShapeResult, candidate: TaskShapeCandidate | null) => void;
}

function percent(value: number): string {
    return `${Math.round(value * 100)}%`;
}

function errorText(err: unknown): string {
    const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    return err instanceof Error ? err.message : 'Could not save the task type.';
}

export default function TaskShapeConfirmCard({
    projectId,
    detection,
    catalog,
    confirmedLabel,
    onConfirmed,
}: TaskShapeConfirmCardProps) {
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');
    const [confirmed, setConfirmed] = useState<string | null>(confirmedLabel ?? null);
    const [choosing, setChoosing] = useState(false);
    const [alternative, setAlternative] = useState('');

    const options = useMemo(() => {
        const seen = new Set<string>();
        const rows: { task_profile: string; label: string }[] = [];
        for (const c of detection?.candidates ?? []) {
            if (!seen.has(c.task_profile)) {
                seen.add(c.task_profile);
                rows.push({ task_profile: c.task_profile, label: `${c.label} (${percent(c.confidence)})` });
            }
        }
        for (const entry of catalog ?? []) {
            if (!seen.has(entry.task_profile)) {
                seen.add(entry.task_profile);
                rows.push({ task_profile: entry.task_profile, label: entry.label });
            }
        }
        return rows;
    }, [detection, catalog]);

    if (!detection?.top) return null;
    const top = detection.top;

    const confirm = async (taskProfile: string) => {
        setSaving(true);
        setError('');
        try {
            const result = await confirmTaskShape(projectId, taskProfile);
            setConfirmed(result.label);
            setChoosing(false);
            const candidate = detection.candidates.find((c) => c.task_profile === taskProfile) ?? null;
            onConfirmed?.(result, candidate);
        } catch (err) {
            setError(errorText(err));
        } finally {
            setSaving(false);
        }
    };

    return (
        <section className="card task-shape-card" data-testid="task-shape-card">
            <h4 className="task-shape-card__title">What kind of task is this?</h4>
            {confirmed ? (
                <p data-testid="task-shape-confirmed">
                    Confirmed: <strong>{confirmed}</strong>. Data prep, training and evaluation will use this.{' '}
                    <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setConfirmed(null); setChoosing(true); }}>
                        Change
                    </button>
                </p>
            ) : (
                <>
                    <p data-testid="task-shape-top">
                        Looks like <strong>{top.label}</strong>{' '}
                        <span className="task-shape-card__confidence">({percent(top.confidence)} confident)</span>
                        {' — '}
                        {top.description}
                    </p>
                    {top.rationale.length > 0 && (
                        <ul className="task-shape-card__reasons">
                            {top.rationale.map((reason) => (
                                <li key={reason}>{reason}</li>
                            ))}
                        </ul>
                    )}
                    {detection.needs_confirmation && (
                        <p className="form-hint-warning" data-testid="task-shape-unsure">
                            We're not sure about this one — check it, or pick what your data is for.
                        </p>
                    )}
                    <div className="task-shape-card__actions">
                        <button
                            type="button"
                            className="btn btn-primary"
                            disabled={saving}
                            onClick={() => void confirm(top.task_profile)}
                            data-testid="task-shape-confirm"
                        >
                            {saving ? 'Saving…' : `Yes, it's ${top.label.toLowerCase()}`}
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            disabled={saving}
                            onClick={() => setChoosing((v) => !v)}
                            data-testid="task-shape-choose"
                        >
                            Something else…
                        </button>
                    </div>
                </>
            )}
            {choosing && (
                <div className="task-shape-card__choose">
                    <label className="form-label" htmlFor="task-shape-alternative">This data is for</label>
                    <select
                        id="task-shape-alternative"
                        className="input"
                        value={alternative}
                        onChange={(e) => setAlternative(e.target.value)}
                        data-testid="task-shape-alternative"
                    >
                        <option value="">Choose a task type…</option>
                        {options.map((o) => (
                            <option key={o.task_profile} value={o.task_profile}>{o.label}</option>
                        ))}
                    </select>
                    <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        disabled={!alternative || saving}
                        onClick={() => void confirm(alternative)}
                        data-testid="task-shape-alternative-confirm"
                    >
                        Use this
                    </button>
                </div>
            )}
            {error && <div className="error-banner" role="alert">{error}</div>}
        </section>
    );
}
