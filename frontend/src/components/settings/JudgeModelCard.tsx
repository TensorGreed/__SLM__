/**
 * JudgeModelCard — Settings → "Judge model". Picks which model scores the
 * project's open-ended evals (lift checks, answer keys, built-in checks) and
 * says, for the pick in force, whether it is local, metered API credits or a
 * subscription CLI. An EVAL_JUDGE_BACKEND env var overrides the pick.
 */

import { useCallback, useEffect, useState } from 'react';

import {
    BILLING_LABELS,
    fetchEvalJudge,
    saveEvalJudge,
    type EvalJudgeState,
    type JudgeResilienceSettings,
} from '../../api/evalJudge';
import { extractApiErrorMessage } from '../../utils/apiError';
import EvalJudgeTrustBadge from '../evaluation/EvalJudgeTrustBadge';
import EvalJudgeCalibrationCard from './EvalJudgeCalibrationCard';

export default function JudgeModelCard({ projectId }: { projectId: number }) {
    const [state, setState] = useState<EvalJudgeState | null>(null);
    const [failed, setFailed] = useState(false);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        setFailed(false);
        fetchEvalJudge(projectId)
            .then((data) => { if (!cancelled) setState(data); })
            .catch(() => { if (!cancelled) setFailed(true); });
        return () => { cancelled = true; };
    }, [projectId]);

    const save = useCallback(async (enabled: boolean, backend: string | null) => {
        setSaving(true);
        setError(null);
        try {
            setState(await saveEvalJudge(projectId, { enabled, backend }));
        } catch (err) {
            setError(extractApiErrorMessage(err, 'Could not save the judge model.').message);
        } finally {
            setSaving(false);
        }
    }, [projectId]);

    const saveResilience = useCallback(async (next: JudgeResilienceSettings) => {
        if (!state) return;
        setSaving(true);
        setError(null);
        try {
            setState(await saveEvalJudge(projectId, { enabled: state.enabled, backend: state.backend, resilience: next }));
        } catch (err) {
            setError(extractApiErrorMessage(err, 'Could not save the judge settings.').message);
        } finally {
            setSaving(false);
        }
    }, [projectId, state]);

    if (failed) return null;
    if (!state) return <section className="card settings-section" data-testid="judge-model-card"><h3>Judge model</h3><p>Loading…</p></section>;

    const resolved = state.resolved;
    return (
        <section className="card settings-section" data-testid="judge-model-card">
            <h3>Judge model</h3>
            <p className="settings-hint">The model that scores open-ended answers in evals. Prompts sent to a cloud or CLI judge leave this machine.</p>
            {state.env_override && (
                <p className="settings-hint" role="status" data-testid="judge-env-override">
                    EVAL_JUDGE_BACKEND is set to <code>{state.env_override}</code> and overrides this choice.
                </p>
            )}
            <label className="settings-row">
                <input
                    type="checkbox"
                    checked={state.enabled}
                    disabled={saving}
                    onChange={(e) => { void save(e.target.checked, state.backend); }}
                    data-testid="judge-enabled"
                />
                <span>
                    <strong>Use a judge model</strong>
                    <small>Off falls back to the built-in checks only.</small>
                </span>
            </label>
            <label className="settings-row">
                <span>
                    <strong>Judge</strong>
                    <select
                        value={state.backend ?? ''}
                        disabled={saving}
                        onChange={(e) => { void save(state.enabled, e.target.value || null); }}
                        aria-label="Judge model"
                        data-testid="judge-select"
                    >
                        {state.options.map((o) => (
                            <option key={o.value} value={o.value} disabled={!o.available}>
                                {o.label}{o.available ? '' : ' (not available)'} · {BILLING_LABELS[o.billing] ?? o.billing}
                            </option>
                        ))}
                    </select>
                </span>
            </label>
            <p className="settings-hint" data-testid="judge-resolved">
                {resolved ? (
                    <>
                        In use: <code>{resolved.describe}</code>{' '}
                        <span className={`settings-chip settings-chip--${resolved.billing}`} data-testid="judge-billing">
                            {BILLING_LABELS[resolved.billing] ?? resolved.billing}
                        </span>
                    </>
                ) : 'No judge is reachable right now.'}
            </p>
            {resolved && state.trust?.verdict && (
                <p className="settings-hint" data-testid="judge-trust-line">
                    <EvalJudgeTrustBadge trust={state.trust.verdict} testId="judge-trust" />
                </p>
            )}
            {resolved && state.trust?.rubric && (
                <p className="settings-hint" data-testid="judge-trust-rubric-line">
                    Plan rubric: <EvalJudgeTrustBadge trust={state.trust.rubric} what="plans" testId="judge-trust-rubric" />
                </p>
            )}
            {state.resilience && (
                <JudgeOutageSettings
                    resilience={state.resilience}
                    saving={saving}
                    onSave={(next) => { void saveResilience(next); }}
                />
            )}
            {error && <div className="settings-error" role="alert" data-testid="judge-error">{error}</div>}
            <EvalJudgeCalibrationCard projectId={projectId} />
        </section>
    );
}

/** Max-wait choices for "wait for the judge" (seconds). */
const MAX_WAIT_CHOICES: Array<[number, string]> = [
    [900, '15 minutes'],
    [3600, '1 hour'],
    [7200, '2 hours'],
    [14400, '4 hours'],
    [21600, '6 hours'],
];

function waitLabel(seconds: number): string {
    const hit = MAX_WAIT_CHOICES.find(([s]) => s === seconds);
    if (hit) return hit[1];
    return seconds >= 3600 ? `${Math.round((seconds / 3600) * 10) / 10} hours` : `${Math.round(seconds / 60)} minutes`;
}

/** "When the judge is unreachable": breaker on / off, stop or wait (up to N hours). Collapsed by default. */
function JudgeOutageSettings({
    resilience,
    saving,
    onSave,
}: {
    resilience: NonNullable<EvalJudgeState['resilience']>;
    saving: boolean;
    onSave: (next: JudgeResilienceSettings) => void;
}) {
    const stored = resilience.stored || {};
    const update = (patch: JudgeResilienceSettings) => onSave({ ...stored, ...patch });
    // Never offer more than the server accepts (half the task queue's visibility timeout).
    const cap = typeof resilience.max_wait_cap_seconds === 'number' ? resilience.max_wait_cap_seconds : Infinity;
    const allowed = MAX_WAIT_CHOICES.filter(([s]) => s <= cap);
    const waitChoices = allowed.some(([s]) => s === resilience.max_wait_seconds)
        ? allowed
        : [...allowed, [resilience.max_wait_seconds, waitLabel(resilience.max_wait_seconds)] as [number, string]];
    return (
        <details className="settings-details" data-testid="judge-outage">
            <summary>When the judge is unreachable</summary>
            <p className="settings-hint">
                A judge call that fails because of the network or the provider is retried. After {resilience.consecutive} rows
                in a row whose call still failed, this eval stops asking (each new eval starts asking again): every answer
                is still saved, and the rows the judge did not score are left out of the judged score, the pass rate, the
                failure list and the base-vs-fine-tuned pairing, with the reason.
            </p>
            <label className="settings-row">
                <input
                    type="checkbox"
                    checked={resilience.breaker}
                    disabled={saving}
                    onChange={(e) => update({ breaker: e.target.checked })}
                    data-testid="judge-outage-breaker"
                />
                <span>
                    <strong>Stop calling an unreachable judge</strong>
                    <small>Off asks the judge about every row once, without retries, however long each failure takes (as before this setting existed).</small>
                </span>
            </label>
            <label className="settings-row">
                <span>
                    <strong>Then</strong>
                    <select
                        value={resilience.mode}
                        disabled={saving || !resilience.breaker}
                        onChange={(e) => update({ mode: e.target.value === 'wait' ? 'wait' : 'stop' })}
                        aria-label="When the judge is unreachable"
                        data-testid="judge-outage-mode"
                    >
                        <option value="stop">Leave the remaining rows unjudged</option>
                        <option value="wait">Wait for the judge, checking every few minutes</option>
                    </select>
                </span>
            </label>
            {resilience.mode === 'wait' && resilience.breaker && (
                <label className="settings-row">
                    <span>
                        <strong>Wait at most</strong>
                        <select
                            value={String(resilience.max_wait_seconds)}
                            disabled={saving}
                            onChange={(e) => update({ max_wait_seconds: Number(e.target.value) })}
                            aria-label="Longest wait for the judge"
                            data-testid="judge-outage-max-wait"
                        >
                            {waitChoices.map(([s, label]) => (
                                <option key={s} value={String(s)}>{label}</option>
                            ))}
                        </select>
                        <small>In total for the eval, probe calls included; only an eval running as a background job waits (others stop). Then the remaining rows are left unjudged.</small>
                    </span>
                </label>
            )}
        </details>
    );
}
