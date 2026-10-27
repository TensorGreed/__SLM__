/**
 * PipelineEditor — the stage list of a pipeline: add / remove / reorder
 * stages and set each stage's project, model, prompt, inputs, fan-out,
 * retrieval, max tokens and output. Save surfaces the backend's per-field
 * 400 errors beside the stage they belong to.
 */

import { useEffect, useState } from 'react';

import { getGatewaySettings, type ServableRun } from '../../api/gateway';
import {
    parsePipelineError,
    type FieldError, type InputFrom, type Pipeline, type PipelineDefinition, type PipelineStage,
} from '../../api/pipelines';
import { runLabel } from '../gateway/GatewayCard';
import './pipelines.css';

export interface ProjectOption { id: number; name: string }

interface InputRow { field: string; from: InputFrom; pointer: string }

export interface StageDraft {
    name: string;
    project_id: number | '';
    model: string;
    mode: 'recipe' | 'template';
    template: string;
    system: string;
    inputs: InputRow[];
    fanOn: boolean;
    fanPointer: string;
    itemKey: string;
    retrieval: 'default' | 'on' | 'off';
    max_tokens: string;
    output: 'json' | 'text';
}

export interface PipelineDraft { name: string; description: string; stages: StageDraft[] }

export function emptyStage(index: number): StageDraft {
    return {
        name: `stage_${index + 1}`, project_id: '', model: '', mode: 'template', template: '', system: '',
        inputs: [{ field: '', from: index === 0 ? 'row' : 'previous', pointer: '' }],
        fanOn: false, fanPointer: '', itemKey: '', retrieval: 'default', max_tokens: '512', output: 'json',
    };
}

export function emptyDraft(): PipelineDraft {
    return { name: '', description: '', stages: [emptyStage(0)] };
}

export function draftFromPipeline(p: Pipeline): PipelineDraft {
    return {
        name: p.name, description: p.description ?? '',
        stages: p.stages.map((s) => ({
            name: s.name, project_id: s.project_id, model: s.model ?? '',
            mode: s.prompt?.mode ?? 'template', template: s.prompt?.template ?? '', system: s.prompt?.system ?? '',
            inputs: Object.entries(s.inputs ?? {}).map(([field, v]) => ({ field, from: v.from, pointer: v.pointer })),
            fanOn: Boolean(s.fan_out), fanPointer: s.fan_out?.pointer ?? '', itemKey: (s.fan_out?.item_key ?? []).join('\n'),
            retrieval: s.retrieval === true ? 'on' : s.retrieval === false ? 'off' : 'default',
            max_tokens: String(s.max_tokens ?? 512), output: s.output ?? 'json',
        })),
    };
}

/** The max-tokens field as sent: empty = the server default (512); anything else is sent as typed
 * (0 or 9000 stay out of range, so the server's per-field error shows instead of a silent 512). */
export function maxTokensValue(raw: string): number | undefined {
    if (!raw.trim()) return undefined;
    const n = Number(raw);
    return Number.isFinite(n) ? n : 0;
}

export function maxTokensProblem(raw: string): string | null {
    const n = maxTokensValue(raw);
    if (n === undefined) return null;
    return Number.isInteger(n) && n >= 1 && n <= 4096 ? null : 'Max tokens must be a whole number from 1 to 4096.';
}

export function draftToDefinition(d: PipelineDraft): PipelineDefinition {
    return {
        name: d.name.trim(), description: d.description.trim(),
        stages: d.stages.map((s): PipelineStage => {
            const inputs: PipelineStage['inputs'] = {};
            for (const r of s.inputs) if (r.field.trim()) inputs[r.field.trim()] = { from: r.from, pointer: r.pointer.trim() };
            const stage: PipelineStage = {
                name: s.name.trim(), project_id: Number(s.project_id), model: s.model,
                prompt: s.mode === 'recipe' ? { mode: 'recipe' } : { mode: 'template', template: s.template, ...(s.system ? { system: s.system } : {}) },
                inputs, retrieval: s.retrieval === 'default' ? null : s.retrieval === 'on',
                max_tokens: maxTokensValue(s.max_tokens), output: s.output,
            };
            if (s.fanOn) {
                stage.fan_out = { pointer: s.fanPointer.trim(), item_key: s.itemKey.split(/[\n,]/).map((k) => k.trim()).filter(Boolean) };
            }
            return stage;
        }),
    };
}

/** Errors that belong to stage `i` (field like `stages[0].name`), and the rest. */
export function stageErrors(errors: FieldError[], i: number): FieldError[] {
    const re = new RegExp(`^stages(\\[${i}\\]|\\.${i}\\b|/${i}\\b)`);
    return errors.filter((e) => re.test(e.field));
}

function ErrorList({ errors }: { errors: FieldError[] }) {
    if (!errors.length) return null;
    return (
        <ul className="pipelines__errors" role="alert">
            {errors.map((e, i) => <li key={i} data-testid={`pipeline-error-${e.field}`}>{e.field}: {e.message}</li>)}
        </ul>
    );
}

interface StageProps {
    index: number; total: number; stage: StageDraft; projects: ProjectOption[]; errors: FieldError[];
    onChange: (next: StageDraft) => void; onMove: (delta: number) => void; onRemove: () => void;
}

function StageCard({ index, total, stage, projects, errors, onChange, onMove, onRemove }: StageProps) {
    const [runs, setRuns] = useState<ServableRun[]>([]);
    const set = (patch: Partial<StageDraft>) => onChange({ ...stage, ...patch });
    const first = index === 0;

    useEffect(() => {
        if (stage.project_id === '') { setRuns([]); return; }
        let live = true;
        getGatewaySettings(Number(stage.project_id))
            .then((r) => { if (live) setRuns(r.servable_runs ?? []); })
            .catch(() => { if (live) setRuns([]); });
        return () => { live = false; };
    }, [stage.project_id]);

    const setInput = (i: number, patch: Partial<InputRow>) =>
        set({ inputs: stage.inputs.map((r, j) => (j === i ? { ...r, ...patch } : r)) });
    const fromOptions: InputFrom[] = first ? ['row'] : stage.fanOn ? ['row', 'previous', 'item'] : ['row', 'previous'];

    return (
        <fieldset className="pipelines__stage" data-testid={`pipeline-stage-${index}`}>
            <legend>Stage {index + 1}</legend>
            <div className="pipelines__row">
                <label className="pipelines__field">Name
                    <input className="input" value={stage.name} data-testid={`stage-${index}-name`}
                        onChange={(e) => set({ name: e.target.value })} />
                </label>
                <label className="pipelines__field">Project
                    <select className="input" value={String(stage.project_id)} data-testid={`stage-${index}-project`}
                        onChange={(e) => set({ project_id: e.target.value ? Number(e.target.value) : '', model: '' })}>
                        <option value="">Choose a project</option>
                        {projects.map((p) => <option key={p.id} value={p.id}>#{p.id} · {p.name}</option>)}
                    </select>
                </label>
                <label className="pipelines__field">Model
                    <select className="input" value={stage.model} data-testid={`stage-${index}-model`}
                        onChange={(e) => set({ model: e.target.value })}>
                        <option value="">Project default</option>
                        <option value=":base">Base model</option>
                        {runs.map((r) => <option key={r.id} value={`:run-${r.id}`}>{runLabel(r)}</option>)}
                    </select>
                </label>
                <div className="pipelines__row">
                    <button type="button" className="btn btn-sm btn-ghost" disabled={first} data-testid={`stage-${index}-up`}
                        onClick={() => onMove(-1)}>Move up</button>
                    <button type="button" className="btn btn-sm btn-ghost" disabled={index === total - 1} data-testid={`stage-${index}-down`}
                        onClick={() => onMove(1)}>Move down</button>
                    <button type="button" className="btn btn-sm btn-ghost" disabled={total === 1} data-testid={`stage-${index}-remove`}
                        onClick={onRemove}>Remove</button>
                </div>
            </div>

            <div className="pipelines__row">
                <label className="pipelines__field">Prompt mode
                    <select className="input" value={stage.mode} data-testid={`stage-${index}-mode`}
                        onChange={(e) => set({ mode: e.target.value as StageDraft['mode'] })}>
                        <option value="template">Template</option>
                        <option value="recipe">Recipe (the project's own prompt)</option>
                    </select>
                </label>
                <label className="pipelines__field">Output
                    <select className="input" value={stage.output} onChange={(e) => set({ output: e.target.value as StageDraft['output'] })}>
                        <option value="json">JSON</option>
                        <option value="text">Text</option>
                    </select>
                    <span className="pipelines__hint" data-testid={`stage-${index}-output-hint`}>
                        JSON parses the reply (a code fence is fine). Use text for bare-label or free-text outputs.
                    </span>
                </label>
                <label className="pipelines__field">Retrieve passages
                    <select className="input" value={stage.retrieval} onChange={(e) => set({ retrieval: e.target.value as StageDraft['retrieval'] })}>
                        <option value="default">Project default</option>
                        <option value="on">On</option>
                        <option value="off">Off</option>
                    </select>
                </label>
                <label className="pipelines__field">Max tokens
                    <input className="input" type="number" min={1} max={4096} value={stage.max_tokens}
                        data-testid={`stage-${index}-max-tokens`} onChange={(e) => set({ max_tokens: e.target.value })} />
                    {maxTokensProblem(stage.max_tokens) && (
                        <span className="pipelines__error" data-testid={`stage-${index}-max-tokens-problem`}>{maxTokensProblem(stage.max_tokens)}</span>
                    )}
                </label>
            </div>

            {stage.mode === 'template' && (
                <>
                    <label className="pipelines__field">Prompt template ({'{field}'} placeholders)
                        <textarea className="input" rows={3} value={stage.template}
                            onChange={(e) => set({ template: e.target.value })} />
                    </label>
                    <label className="pipelines__field">System prompt (optional)
                        <textarea className="input" rows={2} value={stage.system}
                            onChange={(e) => set({ system: e.target.value })} />
                    </label>
                </>
            )}

            <div className="pipelines__inputs">
                <strong>Inputs</strong>
                {stage.inputs.map((r, i) => (
                    <div className="pipelines__row" key={i} data-testid={`stage-${index}-input-${i}`}>
                        <label className="pipelines__field">Field
                            <input className="input" value={r.field} onChange={(e) => setInput(i, { field: e.target.value })} />
                        </label>
                        <label className="pipelines__field">From
                            <select className="input" value={r.from} onChange={(e) => setInput(i, { from: e.target.value as InputFrom })}>
                                {fromOptions.map((o) => <option key={o} value={o}>{o}</option>)}
                            </select>
                        </label>
                        <label className="pipelines__field">Pointer
                            <input className="input" value={r.pointer} placeholder="/path/in/json"
                                onChange={(e) => setInput(i, { pointer: e.target.value })} />
                        </label>
                        <button type="button" className="btn btn-sm btn-ghost"
                            onClick={() => set({ inputs: stage.inputs.filter((_, j) => j !== i) })}>Remove input</button>
                    </div>
                ))}
                <button type="button" className="btn btn-sm btn-secondary"
                    onClick={() => set({ inputs: [...stage.inputs, { field: '', from: first ? 'row' : 'previous', pointer: '' }] })}>Add input</button>
            </div>

            {!first && (
                <div className="pipelines__inputs">
                    <label className="pipelines__check">
                        <input type="checkbox" checked={stage.fanOn} data-testid={`stage-${index}-fanout`}
                            onChange={(e) => set({ fanOn: e.target.checked })} />
                        {' '}Fan out over the previous stage's items (one call per item)
                    </label>
                    {stage.fanOn && (
                        <div className="pipelines__row">
                            <label className="pipelines__field">Items pointer
                                <input className="input" value={stage.fanPointer} placeholder="/items/*"
                                    onChange={(e) => set({ fanPointer: e.target.value })} />
                            </label>
                            <label className="pipelines__field">Item key pointers (one per line)
                                <textarea className="input" rows={2} value={stage.itemKey} placeholder="/label"
                                    onChange={(e) => set({ itemKey: e.target.value })} />
                            </label>
                        </div>
                    )}
                </div>
            )}
            <ErrorList errors={errors} />
        </fieldset>
    );
}

interface Props {
    draft: PipelineDraft;
    projects: ProjectOption[];
    errors: FieldError[];
    message: string | null;
    saving: boolean;
    onChange: (next: PipelineDraft) => void;
    onSave: () => void;
}

export default function PipelineEditor({ draft, projects, errors, message, saving, onChange, onSave }: Props) {
    const setStage = (i: number, next: StageDraft) =>
        onChange({ ...draft, stages: draft.stages.map((s, j) => (j === i ? next : s)) });
    const move = (i: number, delta: number) => {
        const stages = [...draft.stages];
        const [s] = stages.splice(i, 1);
        stages.splice(i + delta, 0, s);
        onChange({ ...draft, stages });
    };
    const general = errors.filter((e) => !/^stages/.test(e.field));

    return (
        <section className="pipelines__section" data-testid="pipeline-editor">
            <h3>Stages</h3>
            <div className="pipelines__row">
                <label className="pipelines__field">Pipeline name
                    <input className="input" value={draft.name} data-testid="pipeline-name"
                        onChange={(e) => onChange({ ...draft, name: e.target.value })} />
                </label>
                <label className="pipelines__field">Description
                    <input className="input" value={draft.description}
                        onChange={(e) => onChange({ ...draft, description: e.target.value })} />
                </label>
            </div>
            <ErrorList errors={general} />
            {draft.stages.map((s, i) => (
                <StageCard key={i} index={i} total={draft.stages.length} stage={s} projects={projects}
                    errors={stageErrors(errors, i)} onChange={(n) => setStage(i, n)}
                    onMove={(d) => move(i, d)}
                    onRemove={() => onChange({ ...draft, stages: draft.stages.filter((_, j) => j !== i) })} />
            ))}
            <div className="pipelines__row">
                <button type="button" className="btn btn-secondary" disabled={draft.stages.length >= 8} data-testid="pipeline-add-stage"
                    onClick={() => onChange({ ...draft, stages: [...draft.stages, emptyStage(draft.stages.length)] })}>Add stage</button>
                <button type="button" className="btn btn-primary" disabled={saving} data-testid="pipeline-save" onClick={onSave}>
                    {saving ? 'Saving...' : 'Save pipeline'}
                </button>
            </div>
            {message && <p className="pipelines__error" role="alert" data-testid="pipeline-message">{message}</p>}
        </section>
    );
}

export { parsePipelineError };
