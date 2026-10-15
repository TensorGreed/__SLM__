/**
 * Threshold tuning for a span-tagging run, with per-type recall floors.
 *
 * For redaction a missed span is the leak, so a type can carry a floor —
 * the minimum validation recall its threshold must keep. The editor loads
 * the project's floors (GET /span-recall-floors), offers the compliance
 * bundles' suggestions (only when the user clicks), saves them (PUT), and
 * tunes the run on the validation split with the floors as edited. The
 * result table shows per type: floor, recall / precision before → after,
 * threshold, and whether the floor was met or is out of reach (then the
 * recall bias that would meet it, and its precision cost).
 */

import { useEffect, useState } from 'react';

import {
    beforeAfter,
    fetchRecallFloors,
    floorStatusText,
    recallBiasLines,
    saveRecallFloors,
    tuneSpanThresholds,
    type RecallFloorsPayload,
    type TuneThresholdsResult,
} from '../../api/spanThresholds';
import './SpanThresholdTuner.css';

interface SpanThresholdTunerProps {
    projectId: number;
    experimentId: number;
    onTuned?: (result: TuneThresholdsResult) => void;
}

function errorText(err: unknown, fallback: string): string {
    const data = (err as { response?: { data?: { detail?: unknown; message?: unknown } } })?.response?.data;
    const detail = data?.message ?? data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

/** Inputs hold strings; blank = no floor. */
function parseFloors(values: Record<string, string>): { floors: Record<string, number>; bad: string[] } {
    const floors: Record<string, number> = {};
    const bad: string[] = [];
    for (const [type, raw] of Object.entries(values)) {
        if (!raw.trim()) continue;
        const v = Number(raw);
        if (Number.isFinite(v) && v >= 0 && v <= 1) floors[type] = v;
        else bad.push(type);
    }
    return { floors, bad };
}

export default function SpanThresholdTuner({ projectId, experimentId, onTuned }: SpanThresholdTunerProps) {
    const [payload, setPayload] = useState<RecallFloorsPayload | null>(null);
    const [values, setValues] = useState<Record<string, string>>({});
    const [defaultFloor, setDefaultFloor] = useState('');
    const [applyBias, setApplyBias] = useState(false);
    const [busy, setBusy] = useState<'save' | 'tune' | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [saved, setSaved] = useState(false);
    const [result, setResult] = useState<TuneThresholdsResult | null>(null);

    const adopt = (p: RecallFloorsPayload) => {
        setPayload(p);
        const next: Record<string, string> = {};
        for (const t of p.types) next[t] = p.floors[t] != null ? String(p.floors[t]) : '';
        for (const [t, v] of Object.entries(p.floors)) next[t] = String(v);
        setValues(next);
        setDefaultFloor(p.default != null ? String(p.default) : '');
    };

    useEffect(() => {
        let cancelled = false;
        fetchRecallFloors(projectId)
            .then((p) => { if (!cancelled) adopt(p); })
            .catch((err) => { if (!cancelled) setError(errorText(err, 'Could not load the recall floors.')); });
        return () => { cancelled = true; };
    }, [projectId]);

    const edited = () => {
        const { floors, bad } = parseFloors(values);
        const def = defaultFloor.trim() ? Number(defaultFloor) : null;
        if (def != null && !(Number.isFinite(def) && def >= 0 && def <= 1)) bad.push('default');
        return { floors, def, bad };
    };

    const save = async () => {
        const { floors, def, bad } = edited();
        if (bad.length) { setError(`Floors must be between 0 and 1 (${bad.join(', ')}).`); return; }
        setBusy('save');
        setError(null);
        try {
            adopt(await saveRecallFloors(projectId, { floors, default: def }));
            setSaved(true);
        } catch (err) {
            setError(errorText(err, 'Could not save the recall floors.'));
        } finally {
            setBusy(null);
        }
    };

    const tune = async () => {
        const { floors, def, bad } = edited();
        if (bad.length) { setError(`Floors must be between 0 and 1 (${bad.join(', ')}).`); return; }
        setBusy('tune');
        setError(null);
        try {
            const res = await tuneSpanThresholds(projectId, {
                experiment_id: experimentId,
                floors: def != null ? { ...floors, _default: def } : floors,
                apply_recall_bias: applyBias,
            });
            setResult(res);
            onTuned?.(res);
        } catch (err) {
            setError(errorText(err, 'Could not tune thresholds.'));
        } finally {
            setBusy(null);
        }
    };

    const suggestions = Object.entries(payload?.suggestions ?? {});
    const useSuggestions = () => {
        setValues((prev) => {
            const next = { ...prev };
            for (const [t, s] of suggestions) next[t] = String(s.floor);
            return next;
        });
        setSaved(false);
    };
    const types = Object.keys(values).sort();
    const floored = result ? Object.entries(result.report).filter(([, r]) => r.floor != null) : [];
    const others = result ? Object.entries(result.report).filter(([, r]) => r.floor == null) : [];

    return (
        <div className="span-tuner" data-testid="span-tuner">
            <h4 className="span-tuner__title">Recall floors</h4>
            <p className="span-tuner__hint">
                For redaction a missed span is worse than an extra one. Give a type a floor (0–1) and tuning keeps at
                least that recall on the validation split, taking the best precision that still meets it. Types
                without a floor are tuned for F1. Never tuned on the test examples or benchmarks.
            </p>
            {payload && types.length === 0 && (
                <p className="span-tuner__hint">The project has no entity types yet — only a default floor can be set.</p>
            )}
            <div className="span-tuner__floors">
                {types.map((t) => (
                    <label key={t} className="span-tuner__floor">
                        <code>{t}</code>
                        <input
                            type="number"
                            min={0}
                            max={1}
                            step={0.01}
                            placeholder="none"
                            value={values[t] ?? ''}
                            data-testid={`span-tuner-floor-${t}`}
                            onChange={(e) => { setValues({ ...values, [t]: e.target.value }); setSaved(false); }}
                        />
                        {payload?.suggestions[t] && (
                            <span className="span-tuner__suggested">
                                suggested {payload.suggestions[t].floor} ({payload.suggestions[t].regimes.join(', ')})
                            </span>
                        )}
                    </label>
                ))}
                <label className="span-tuner__floor">
                    <span>Default for other types</span>
                    <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.01}
                        placeholder="none"
                        value={defaultFloor}
                        data-testid="span-tuner-floor-default"
                        onChange={(e) => { setDefaultFloor(e.target.value); setSaved(false); }}
                    />
                </label>
            </div>
            <div className="span-tuner__actions">
                {suggestions.length > 0 && (
                    <button type="button" className="btn btn-secondary" onClick={useSuggestions} data-testid="span-tuner-use-suggested">
                        Use compliance suggestions
                    </button>
                )}
                <button type="button" className="btn btn-secondary" onClick={() => void save()} disabled={busy !== null} data-testid="span-tuner-save">
                    {busy === 'save' ? 'Saving…' : 'Save floors'}
                </button>
                <label className="span-tuner__bias">
                    <input type="checkbox" checked={applyBias} onChange={(e) => setApplyBias(e.target.checked)} data-testid="span-tuner-apply-bias" />
                    If a floor is out of reach, apply the suggested recall bias (more spans of every type)
                </label>
                <button type="button" className="btn btn-primary" onClick={() => void tune()} disabled={busy !== null} data-testid="eval-summary-tune-run">
                    {busy === 'tune' ? 'Tuning…' : 'Tune on the validation split'}
                </button>
                {saved && <span className="span-tuner__saved" data-testid="span-tuner-saved">✓ saved on project</span>}
            </div>
            {error && <p className="span-tuner__error" data-testid="span-tuner-error">{error}</p>}
            {result && floored.length > 0 && (
                <table className="span-tuner__table" data-testid="span-tuner-result">
                    <thead>
                        <tr><th>Type</th><th>Floor</th><th>Recall</th><th>Precision</th><th>Threshold</th><th>Floor met?</th></tr>
                    </thead>
                    <tbody>
                        {floored.map(([t, r]) => (
                            <tr key={t} data-testid={`span-tuner-row-${t}`}>
                                <td><code>{t}</code></td>
                                <td>{r.floor?.toFixed(2)}</td>
                                <td>{beforeAfter(r.recall_before, r.recall_after)}</td>
                                <td>{beforeAfter(r.precision_before, r.precision_after)}</td>
                                <td>{r.threshold}</td>
                                <td className={`span-tuner__status span-tuner__status--${r.floor_status ?? 'none'}`}>{floorStatusText(r.floor_status)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
            {result && others.length > 0 && floored.length > 0 && (
                <p className="span-tuner__hint">
                    Tuned for F1 (no floor): {others.map(([t, r]) => `${t} ≥ ${r.threshold}`).join(', ')}.
                </p>
            )}
            {result && (result.recall_bias ?? 0) > 0 && (
                <p className="span-tuner__hint" data-testid="span-tuner-bias-applied">
                    Recall bias {result.recall_bias} applied (written to thresholds.json; every type gets more spans).
                </p>
            )}
            {result && recallBiasLines(result.recall_bias_suggestion).length > 0 && (result.recall_bias ?? 0) === 0 && (
                <ul className="span-tuner__bias-lines" data-testid="span-tuner-bias-suggestion">
                    {recallBiasLines(result.recall_bias_suggestion).map((line) => <li key={line}>{line}</li>)}
                </ul>
            )}
            {result?.floors_unknown && result.floors_unknown.length > 0 && (
                <p className="span-tuner__hint">This run has no {result.floors_unknown.join(', ')} label — those floors were not used.</p>
            )}
        </div>
    );
}
