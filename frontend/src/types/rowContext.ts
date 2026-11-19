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

/** `GET /projects/{id}/gold/entries` carries the backend's own answer to "which row_context kinds does this
 *  project's task read": the evaluator's effective task profile and the kinds it consumes
 *  (`row_context_service.kinds_read_by`). The UI labels unused context from this, never from its own rule. */
export interface RowContextReads {
    effective_task_profile?: string | null;
    row_context_reads?: string[];
}

/** Whether the task reads a row_context kind (`system`, `tools`, ...), per the backend response. */
export function taskReadsRowContext(resp: RowContextReads | null | undefined, kind: string): boolean {
    return Array.isArray(resp?.row_context_reads) && resp.row_context_reads.includes(kind);
}
