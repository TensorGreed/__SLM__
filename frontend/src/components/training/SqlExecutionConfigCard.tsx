/**
 * SqlExecutionConfigCard — opt-in SQL execution match for projects whose
 * answers are SQL queries: the SQLite files (inside the project's data
 * directory) that each predicted and answer-key query run against, the time
 * limit, row cap, float tolerance, row-order rule and the refusal marker.
 *
 * Off by default; evals are unchanged until it is turned on with at least one
 * database.
 */

import { useCallback, useEffect, useState } from 'react';

import {
    getSqlExecutionConfig,
    putSqlExecutionConfig,
    sqlExecutionErrors,
    type SqlExecutionConfigResponse,
    type SqlOrderMode,
} from '../../api/sqlExecution';
import { toast } from '../../stores/toastStore';
import './ToolScoringConfigCard.css';

const ORDER_OPTIONS: Array<{ id: SqlOrderMode; label: string }> = [
    { id: 'auto', label: 'Check row order only when the answer-key query sorts its outer result' },
    { id: 'always', label: 'Always check row order' },
    { id: 'never', label: 'Never check row order' },
];

export default function SqlExecutionConfigCard({ projectId }: { projectId: number }) {
    const [state, setState] = useState<SqlExecutionConfigResponse | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [enabled, setEnabled] = useState(false);
    const [pathsText, setPathsText] = useState('');
    const [timeoutMs, setTimeoutMs] = useState('2000');
    const [rowCap, setRowCap] = useState('10000');
    const [floatTol, setFloatTol] = useState('0.000001');
    const [order, setOrder] = useState<SqlOrderMode>('auto');
    const [marker, setMarker] = useState('NO_QUERY');
    const [saving, setSaving] = useState(false);
    const [errors, setErrors] = useState<string[]>([]);

    const apply = useCallback((res: SqlExecutionConfigResponse) => {
        setState(res);
        const c = res.effective ?? {};
        setEnabled(c.enabled ?? false);
        setPathsText((c.db_paths ?? []).join('\n'));
        setTimeoutMs(String(c.timeout_ms ?? 2000));
        setRowCap(String(c.row_cap ?? 10000));
        setFloatTol(String(c.float_rel_tol ?? 0.000001));
        setOrder(c.order ?? 'auto');
        setMarker(c.refusal_marker ?? 'NO_QUERY');
    }, []);

    useEffect(() => {
        let cancelled = false;
        getSqlExecutionConfig(projectId)
            .then((res) => { if (!cancelled) apply(res); })
            .catch(() => { if (!cancelled) setLoadError('Could not load the SQL execution settings.'); });
        return () => { cancelled = true; };
    }, [projectId, apply]);

    const save = async () => {
        setSaving(true);
        setErrors([]);
        try {
            const res = await putSqlExecutionConfig(projectId, {
                enabled,
                db_paths: pathsText.split('\n').map((p) => p.trim()).filter(Boolean),
                timeout_ms: Number(timeoutMs),
                row_cap: Number(rowCap),
                float_rel_tol: Number(floatTol),
                order,
                refusal_marker: marker.trim(),
            });
            apply(res);
            toast.success('SQL execution settings saved.', 3000);
        } catch (err) {
            setErrors(sqlExecutionErrors(err));
        } finally {
            setSaving(false);
        }
    };

    if (loadError) return <p className="u-hint" data-testid="sx-error">{loadError}</p>;
    if (!state) return <p className="u-hint">Loading SQL execution settings…</p>;

    return (
        <div className="ts-config" data-testid="sx-config">
            <p className="u-hint u-mt0" data-testid="sx-status-line">
                {state.effective?.active
                    ? `On: evals run each predicted and answer-key query on ${state.databases.length} database${state.databases.length === 1 ? '' : 's'} and compare the results.`
                    : 'Off: evals score answers as before.'}{' '}
                Database files must be inside the project data directory ({state.data_dir}); they are opened read-only.
            </p>

            <label className="ts-config__row">
                <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} data-testid="sx-enabled" />
                Score SQL answers by running them
            </label>

            <fieldset className="ts-config__group">
                <legend>Databases</legend>
                <div className="ts-config__row">
                    <label htmlFor="sx-paths">SQLite files, one per line (relative to the project data directory)</label>
                    <textarea id="sx-paths" rows={3} value={pathsText} onChange={(e) => setPathsText(e.target.value)} data-testid="sx-paths" />
                </div>
                {state.databases.length > 0 && (
                    <ul className="ts-config__sub" data-testid="sx-databases">
                        {state.databases.map((d) => (
                            <li key={d.path}>{d.path}: {d.ok ? `found (${d.size_bytes ?? 0} bytes)` : d.error}</li>
                        ))}
                    </ul>
                )}
            </fieldset>

            <fieldset className="ts-config__group">
                <legend>Limits and comparison</legend>
                <div className="ts-config__row">
                    <label htmlFor="sx-timeout">Time limit per query (ms)</label>
                    <input id="sx-timeout" type="number" value={timeoutMs} onChange={(e) => setTimeoutMs(e.target.value)} data-testid="sx-timeout" />
                    <label htmlFor="sx-row-cap">Row cap</label>
                    <input id="sx-row-cap" type="number" value={rowCap} onChange={(e) => setRowCap(e.target.value)} data-testid="sx-row-cap" />
                    <label htmlFor="sx-tol">Float tolerance (relative)</label>
                    <input id="sx-tol" type="text" value={floatTol} onChange={(e) => setFloatTol(e.target.value)} data-testid="sx-tol" />
                </div>
                <div className="ts-config__row">
                    <label htmlFor="sx-order">Row order</label>
                    <select id="sx-order" value={order} onChange={(e) => setOrder(e.target.value as SqlOrderMode)} data-testid="sx-order">
                        {ORDER_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
                    </select>
                </div>
                <div className="ts-config__row">
                    <label htmlFor="sx-marker">Refusal marker</label>
                    <input id="sx-marker" type="text" value={marker} onChange={(e) => setMarker(e.target.value)} data-testid="sx-marker" />
                    <span className="ts-config__sub">An answer-key row equal to this passes only when the model answers exactly this.</span>
                </div>
            </fieldset>

            {errors.length > 0 && (
                <ul className="ts-config__errors" data-testid="sx-save-errors">
                    {errors.map((e, i) => <li key={i}>{e}</li>)}
                </ul>
            )}
            <div className="ts-config__row">
                <button type="button" className="btn btn-primary" onClick={save} disabled={saving} data-testid="sx-save">
                    {saving ? 'Saving…' : 'Save settings'}
                </button>
            </div>
        </div>
    );
}
