/** Per-row task context (`row_context`) as stored on a data row. */
export interface RowContextTool {
    name?: string;
    description?: string;
    input_schema?: Record<string, unknown>;
    annotations?: Record<string, unknown>;
    read_only?: boolean;
    destructive?: boolean;
}

export interface RawRowContext {
    system?: string;
    tools?: RowContextTool[];
    output_schema?: Record<string, unknown>;
    labels?: string[];
    schema_text?: string;
}

/** What an eval row / predictions record says about the context it was prompted with. */
export interface EvalRowContext {
    system?: string;
    system_source?: 'row_context.system' | 'system_prompt';
    system_truncated?: boolean;
    tools?: Array<{ name?: string; signature?: string; allowed?: boolean }>;
    tools_count?: number;
    tools_source?: 'row';
    unused_kinds?: string[];
    digest?: string;
}

export type AnyRowContext = RawRowContext | EvalRowContext;

/** `metrics.row_context` on an eval result (only when rows carried context). */
export interface RowContextMetrics {
    rows?: number;
    rows_with_context?: number;
    kinds?: Record<string, Record<string, number>>;
    digest?: string;
    unused_kinds?: string[];
}

/** `metrics.tool_flags.per_row` when rows carried their own tool lists. */
export interface ToolFlagsPerRow {
    rows?: number;
    rows_total?: number;
    distinct_lists?: number;
    tools_seen?: number;
    project_flags_applied?: boolean;
}

/** Task profiles that read a row's own tool list (backend `row_context_service.CONSUMERS["tools"]`). */
export const TOOL_LIST_CONSUMERS: readonly string[] = ['tool_calling'];

/** Whether a project's task reads per-row tool lists: its task profile (the dataset adapter preset's,
 *  else the selected recipe's) is a tools consumer. The same rule the eval metrics use. */
export function projectReadsRowTools(project: unknown): boolean {
    const p = (project ?? {}) as {
        dataset_adapter_preset?: { task_profile?: string | null } | null;
        selected_recipe?: { task_profile?: string | null } | null;
    };
    const profile = p.dataset_adapter_preset?.task_profile ?? p.selected_recipe?.task_profile ?? '';
    return TOOL_LIST_CONSUMERS.includes(String(profile));
}
