/**
 * RowContextChip - compact view of a row's own task context (`row_context`).
 *
 * Collapsed: small chips ("Own system message", "Own tool list (7)" or
 * "Project catalog", "Own output schema (3 fields)", "not used by this task: labels").
 * Expanded: the system text, the tool signatures and the output schema (JSON). Renders nothing when the row carries no context.
 * Accepts the eval-view shape (`system_source`, `tools[].signature`, `digest`)
 * and a raw data row's `row_context` (counts computed from the raw tools list).
 */
import { useState } from 'react';
import type { AnyRowContext, EvalRowContext, RawRowContext } from '../../types/rowContext';
import './RowContextChip.css';

interface Props {
    /** An object, or the JSON text prepared splits store (`row_context_service.encode`). */
    context?: AnyRowContext | string | null;
    /** "project" when the row has no own tool list and the project catalog applies. */
    toolListSource?: 'row' | 'project' | null;
    /** Raw rows only: false when this project does not use per-row tools. */
    toolsUsed?: boolean;
    /** Raw rows only: false when this project's task does not read a row's own output schema. */
    schemaUsed?: boolean;
    testId?: string;
}

interface ToolView { name: string; detail: string; allowed: boolean }

const RAW_UNUSED = ['labels', 'schema_text'];

function schemaText(value: unknown): string {
    if (typeof value === 'string') return value;
    try {
        return JSON.stringify(value, null, 2) ?? '';
    } catch {
        return '';
    }
}

function fieldCount(value: unknown): number {
    const props = value && typeof value === 'object' ? (value as { properties?: unknown }).properties : undefined;
    return props && typeof props === 'object' && !Array.isArray(props) ? Object.keys(props).length : 0;
}

function isEvalShape(c: AnyRowContext): c is EvalRowContext {
    const e = c as EvalRowContext;
    return e.digest !== undefined || e.system_source !== undefined || e.tools_count !== undefined
        || e.unused_kinds !== undefined || (e.tools ?? []).some((t) => t && 'signature' in t);
}

function normalise(c: AnyRowContext, toolsUsed: boolean, schemaUsed: boolean) {
    let tools: ToolView[] = [];
    let count = 0;
    let unused: string[] = [];
    let schema = '';
    let schemaFields = 0;
    if (isEvalShape(c)) {
        tools = (c.tools ?? []).map((t) => ({ name: t.name ?? '', detail: t.signature ?? '', allowed: t.allowed !== false }));
        count = c.tools_count ?? tools.length;
        unused = c.unused_kinds ?? [];
        if (typeof c.output_schema === 'string' && c.output_schema) {
            schema = c.output_schema + (c.output_schema_truncated ? ' ...(truncated)' : '');
            schemaFields = c.output_schema_fields ?? 0;
        }
    } else {
        const raw = c as RawRowContext;
        const list = Array.isArray(raw.tools) ? raw.tools : [];
        tools = list.map((t) => ({ name: t?.name ?? '', detail: t?.description ?? '', allowed: true }));
        count = tools.length;
        unused = RAW_UNUSED.filter((k) => {
            const v = (raw as Record<string, unknown>)[k];
            return v != null && !(Array.isArray(v) && v.length === 0) && v !== '';
        });
        if (!toolsUsed && count > 0) unused = ['tools', ...unused];
        const rawSchema = (raw as Record<string, unknown>).output_schema;
        if (rawSchema != null && rawSchema !== '') {
            if (schemaUsed) {
                schema = schemaText(rawSchema);
                schemaFields = fieldCount(rawSchema);
            } else {
                unused = [...unused, 'output_schema'];
            }
        }
    }
    const system = typeof c.system === 'string' && c.system.trim() ? c.system : '';
    return {
        tools, count, unused, system, schema, schemaFields,
        truncated: isEvalShape(c) && Boolean(c.system_truncated), evalShape: isEvalShape(c),
    };
}

function parseContext(context: AnyRowContext | string | null | undefined): AnyRowContext | null {
    if (typeof context === 'string') {
        try {
            const parsed: unknown = JSON.parse(context);
            return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as AnyRowContext) : null;
        } catch {
            return null;
        }
    }
    return context && typeof context === 'object' ? context : null;
}

export default function RowContextChip({
    context, toolListSource = null, toolsUsed = true, schemaUsed = true, testId = 'row-context-chip',
}: Props) {
    const [open, setOpen] = useState(false);
    const ctx = parseContext(context);
    if (!ctx) return null;
    const v = normalise(ctx, toolsUsed, schemaUsed);
    const ownTools = v.count > 0 && !v.unused.includes('tools');
    const projectCatalog = !ownTools && toolListSource === 'project';
    if (!v.system && !ownTools && !projectCatalog && !v.schema && v.unused.length === 0) return null;
    const expandable = Boolean(v.system) || (ownTools && v.tools.length > 0) || Boolean(v.schema);
    return (
        <div className="row-context" data-testid={testId}>
            <ul className="row-context__chips">
                {v.system && <li className="row-context__chip">Own system message</li>}
                {ownTools && <li className="row-context__chip">Own tool list ({v.count})</li>}
                {projectCatalog && <li className="row-context__chip">Project catalog</li>}
                {v.schema && (
                    <li className="row-context__chip">
                        Own output schema{v.schemaFields > 0 ? ` (${v.schemaFields} field${v.schemaFields === 1 ? '' : 's'})` : ''}
                    </li>
                )}
                {v.unused.length > 0 && (
                    <li className="row-context__chip row-context__chip--muted">not used by this task: {v.unused.join(', ')}</li>
                )}
            </ul>
            {expandable && (
                <button type="button" className="row-context__toggle" onClick={() => setOpen((o) => !o)} data-testid={`${testId}-toggle`}>
                    {open ? 'Hide context' : 'Show context'}
                </button>
            )}
            {open && (
                <div className="row-context__body" data-testid={`${testId}-body`}>
                    {v.system && (
                        <pre className="row-context__system" data-testid={`${testId}-system`}>
                            {v.system}{v.truncated ? ' ...(truncated)' : ''}
                        </pre>
                    )}
                    {ownTools && v.tools.length > 0 && (
                        <ul className="row-context__tools" data-testid={`${testId}-tools`}>
                            {v.tools.map((t, i) => (
                                <li key={`${t.name}-${i}`}>
                                    {v.evalShape
                                        ? <code>{t.detail || t.name}</code>
                                        : <><code>{t.name}</code>{t.detail && <span className="row-context__desc"> {t.detail}</span>}</>}
                                    {!t.allowed && <span className="row-context__flag"> not allowed by project</span>}
                                </li>
                            ))}
                        </ul>
                    )}
                    {v.schema && (
                        <pre className="row-context__system" data-testid={`${testId}-schema`}>{v.schema}</pre>
                    )}
                </div>
            )}
        </div>
    );
}
