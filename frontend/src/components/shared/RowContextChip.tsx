/**
 * RowContextChip - compact view of a row's own task context (`row_context`).
 *
 * Collapsed: small chips ("Own system message", "Own tool list (7)" or
 * "Project catalog", "not used by this task: labels"). Expanded: the system text
 * and the tool signatures. Renders nothing when the row carries no context.
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
    testId?: string;
}

interface ToolView { name: string; detail: string; allowed: boolean }

const RAW_UNUSED = ['output_schema', 'labels', 'schema_text'];

function isEvalShape(c: AnyRowContext): c is EvalRowContext {
    const e = c as EvalRowContext;
    return e.digest !== undefined || e.system_source !== undefined || e.tools_count !== undefined
        || e.unused_kinds !== undefined || (e.tools ?? []).some((t) => t && 'signature' in t);
}

function normalise(c: AnyRowContext, toolsUsed: boolean) {
    let tools: ToolView[] = [];
    let count = 0;
    let unused: string[] = [];
    if (isEvalShape(c)) {
        tools = (c.tools ?? []).map((t) => ({ name: t.name ?? '', detail: t.signature ?? '', allowed: t.allowed !== false }));
        count = c.tools_count ?? tools.length;
        unused = c.unused_kinds ?? [];
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
    }
    const system = typeof c.system === 'string' && c.system.trim() ? c.system : '';
    return { tools, count, unused, system, truncated: isEvalShape(c) && Boolean(c.system_truncated), evalShape: isEvalShape(c) };
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

export default function RowContextChip({ context, toolListSource = null, toolsUsed = true, testId = 'row-context-chip' }: Props) {
    const [open, setOpen] = useState(false);
    const ctx = parseContext(context);
    if (!ctx) return null;
    const v = normalise(ctx, toolsUsed);
    const ownTools = v.count > 0 && !v.unused.includes('tools');
    const projectCatalog = !ownTools && toolListSource === 'project';
    if (!v.system && !ownTools && !projectCatalog && v.unused.length === 0) return null;
    const expandable = Boolean(v.system) || (ownTools && v.tools.length > 0);
    return (
        <div className="row-context" data-testid={testId}>
            <ul className="row-context__chips">
                {v.system && <li className="row-context__chip">Own system message</li>}
                {ownTools && <li className="row-context__chip">Own tool list ({v.count})</li>}
                {projectCatalog && <li className="row-context__chip">Project catalog</li>}
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
                </div>
            )}
        </div>
    );
}
