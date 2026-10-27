/**
 * ToolScoringConfigCard — how tool calls are judged against the project's tool
 * catalog: which checks run (valid call, arguments match the answer key, plan
 * order), where the calls sit in a structured plan, which argument keys must
 * match, and the optional read-only dry run.
 *
 * Shown on the Train page for `structured-plan` projects and projects whose
 * task shape is tool calling. Without a catalog the settings are saved but have
 * no effect until tools are imported on the Data page.
 */

import { useCallback, useEffect, useState } from 'react';

import {
    TOOL_SCORING_MODES,
    getToolScoringConfig,
    putToolScoringConfig,
    toolScoringErrors,
    type ToolScoringConfigResponse,
    type ToolScoringMode,
} from '../../api/toolScoring';
import { toast } from '../../stores/toastStore';
import './ToolScoringConfigCard.css';

interface Props {
    projectId: number;
    /** Plans carry calls under a pointer; a single tool call does not. */
    isPlan: boolean;
}

export default function ToolScoringConfigCard({ projectId, isPlan }: Props) {
    const [state, setState] = useState<ToolScoringConfigResponse | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [enabled, setEnabled] = useState(true);
    const [modes, setModes] = useState<ToolScoringMode[]>(TOOL_SCORING_MODES.map((m) => m.id));
    const [stepsPointer, setStepsPointer] = useState('/steps');
    const [toolField, setToolField] = useState('tool');
    const [argsField, setArgsField] = useState('args');
    const [idField, setIdField] = useState('step');
    const [dependsField, setDependsField] = useState('depends_on');
    const [argKeysText, setArgKeysText] = useState('');
    const [dryRun, setDryRun] = useState(false);
    const [saving, setSaving] = useState(false);
    const [errors, setErrors] = useState<string[]>([]);

    const apply = useCallback((res: ToolScoringConfigResponse) => {
        setState(res);
        const c = { ...(res.effective ?? {}), ...res.config };
        setEnabled(c.enabled ?? true);
        setModes(c.modes ?? TOOL_SCORING_MODES.map((m) => m.id));
        setStepsPointer(c.steps_pointer ?? '/steps');
        setToolField(c.tool_field ?? 'tool');
        setArgsField(c.args_field ?? 'args');
        setIdField(c.id_field ?? 'step');
        setDependsField(c.depends_field ?? 'depends_on');
        setArgKeysText((c.arg_keys ?? []).join(', '));
        setDryRun(c.dry_run ?? false);
    }, []);

    useEffect(() => {
        let cancelled = false;
        getToolScoringConfig(projectId)
            .then((res) => { if (!cancelled) apply(res); })
            .catch(() => { if (!cancelled) setLoadError('Could not load the tool scoring settings.'); });
        return () => { cancelled = true; };
    }, [projectId, apply]);

    const toggleMode = (id: ToolScoringMode, on: boolean) =>
        setModes((prev) => (on ? [...prev.filter((m) => m !== id), id] : prev.filter((m) => m !== id)));

    const save = async () => {
        setSaving(true);
        setErrors([]);
        const argKeys = argKeysText.split(',').map((k) => k.trim()).filter(Boolean);
        try {
            const res = await putToolScoringConfig(projectId, {
                enabled,
                modes,
                ...(isPlan
                    ? { steps_pointer: stepsPointer.trim(), tool_field: toolField.trim(), args_field: argsField.trim(), id_field: idField.trim(), depends_field: dependsField.trim() }
                    : {}),
                arg_keys: argKeys,
                dry_run: dryRun,
            });
            apply(res);
            toast.success('Tool scoring settings saved.', 3000);
        } catch (err) {
            setErrors(toolScoringErrors(err));
        } finally {
            setSaving(false);
        }
    };

    if (loadError) return <p className="u-hint" data-testid="ts-error">{loadError}</p>;
    if (!state) return <p className="u-hint">Loading tool scoring settings…</p>;

    const catalog = state.catalog;
    const visibleModes = TOOL_SCORING_MODES.filter((m) => isPlan || m.id !== 'plan_order_valid');
    return (
        <div className="ts-config" data-testid="ts-config">
            <p className="u-hint u-mt0" data-testid="ts-catalog-line">
                {catalog
                    ? `Scored against catalog version ${catalog.version} (${catalog.tool_count} tool${catalog.tool_count === 1 ? '' : 's'}, source ${catalog.source_kind}).`
                    : 'No tool catalog yet: import one on the Data page (Labels step). These settings take effect once there is a catalog.'}
            </p>

            <label className="ts-config__row">
                <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} data-testid="ts-enabled" />
                Score tool calls against the catalog
            </label>

            <fieldset className="ts-config__group">
                <legend>Checks</legend>
                {visibleModes.map((m) => (
                    <label className="ts-config__row" key={m.id}>
                        <input type="checkbox" checked={modes.includes(m.id)} onChange={(e) => toggleMode(m.id, e.target.checked)} data-testid={`ts-mode-${m.id}`} />
                        <span>{m.label}<span className="ts-config__sub">{m.hint}</span></span>
                    </label>
                ))}
            </fieldset>

            {isPlan && (
                <fieldset className="ts-config__group">
                    <legend>Where the calls are in a plan</legend>
                    <div className="ts-config__row">
                        <label htmlFor="ts-steps">Steps pointer</label>
                        <input id="ts-steps" type="text" value={stepsPointer} onChange={(e) => setStepsPointer(e.target.value)} data-testid="ts-steps-pointer" />
                    </div>
                    <div className="ts-config__row">
                        <label htmlFor="ts-tool-field">Tool field</label>
                        <input id="ts-tool-field" type="text" value={toolField} onChange={(e) => setToolField(e.target.value)} data-testid="ts-tool-field" />
                        <label htmlFor="ts-args-field">Arguments field</label>
                        <input id="ts-args-field" type="text" value={argsField} onChange={(e) => setArgsField(e.target.value)} data-testid="ts-args-field" />
                    </div>
                    <div className="ts-config__row">
                        <label htmlFor="ts-id-field">Step id field</label>
                        <input id="ts-id-field" type="text" value={idField} onChange={(e) => setIdField(e.target.value)} data-testid="ts-id-field" />
                        <label htmlFor="ts-depends-field">Depends-on field</label>
                        <input id="ts-depends-field" type="text" value={dependsField} onChange={(e) => setDependsField(e.target.value)} data-testid="ts-depends-field" />
                    </div>
                </fieldset>
            )}

            <fieldset className="ts-config__group">
                <legend>Argument match</legend>
                <div className="ts-config__row">
                    <label htmlFor="ts-arg-keys">Argument keys to compare</label>
                    <input id="ts-arg-keys" type="text" value={argKeysText} placeholder="blank compares every key in the answer key"
                        onChange={(e) => setArgKeysText(e.target.value)} data-testid="ts-arg-keys" />
                </div>
            </fieldset>

            <fieldset className="ts-config__group">
                <legend>Dry run</legend>
                <label className="ts-config__row">
                    <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} data-testid="ts-dry-run" />
                    <span>
                        Run valid calls against the tool server
                        <span className="ts-config__sub" data-testid="ts-dry-run-note">
                            Read-only tools only; never calls mutating tools. Informational: it does not change the pass rate.
                            The server operator must enable dry runs.
                        </span>
                    </span>
                </label>
            </fieldset>

            {errors.length > 0 && (
                <ul className="ts-config__errors" data-testid="ts-save-errors">
                    {errors.map((e, i) => <li key={i}>{e}</li>)}
                </ul>
            )}
            <div className="ts-config__row">
                <button type="button" className="btn btn-primary" onClick={save} disabled={saving} data-testid="ts-save">
                    {saving ? 'Saving…' : 'Save settings'}
                </button>
            </div>
        </div>
    );
}
