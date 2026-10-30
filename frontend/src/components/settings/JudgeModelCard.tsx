/**
 * JudgeModelCard — Settings → "Judge model". Picks which model scores the
 * project's open-ended evals (lift checks, answer keys, built-in checks) and
 * says, for the pick in force, whether it is local, metered API credits or a
 * subscription CLI. An EVAL_JUDGE_BACKEND env var overrides the pick.
 */

import { useCallback, useEffect, useState } from 'react';

import { BILLING_LABELS, fetchEvalJudge, saveEvalJudge, type EvalJudgeState } from '../../api/evalJudge';
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
            {error && <div className="settings-error" role="alert" data-testid="judge-error">{error}</div>}
            <EvalJudgeCalibrationCard projectId={projectId} />
        </section>
    );
}
