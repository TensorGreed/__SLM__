/**
 * ToolCallMetricsPanel — results of tool-call scoring against the project's
 * tool catalog: how many calls named a known, allowed tool with arguments that
 * fit its schema, how many matched the answer key, and (for plans) whether the
 * step order is valid.
 *
 * `ToolCallMetricsSection` is the shared body (also rendered inside the
 * structured-plan panel when `metrics.tools` exists). `ToolCallMetricsPanel` is
 * the standalone card for the `tool_call` scoring mode (the tool_calling task
 * shape). `ToolCallRowView` shows each call of one row with its badges.
 *
 * Rates that do not apply say so instead of showing a number. The catalog
 * version, source and settings version are shown so a number can be traced to
 * the catalog it was scored with.
 */

import './ToolCallMetricsPanel.css';

export interface ToolPerToolStat {
    calls?: number;
    valid?: number;
    invalid_args?: number;
    not_allowed?: number;
}

export interface ToolDryRunSummary {
    enabled?: boolean;
    calls?: number;
    ok?: number;
    errors?: number;
    skipped?: number;
}

export interface ToolsBlock {
    catalog_version?: string | null;
    flags_version?: string | null;
    source_kind?: string | null;
    modes?: string[];
    tool_call_valid_rate?: number | null;
    tool_args_match_rate?: number | null;
    plan_order_valid_rate?: number | null;
    pass_rate?: number | null;
    per_tool?: Record<string, ToolPerToolStat>;
    unknown_tools?: Record<string, number>;
    errors_top?: Array<{ message?: string; count?: number }>;
    dry_run?: ToolDryRunSummary | null;
}

export interface ToolCallMetrics {
    scoring_mode?: string;
    tool_call_pass_rate?: number | null;
    tool_selection_accuracy?: number | null;
    tools?: ToolsBlock | null;
    total?: number;
    correct?: number;
}

export interface ToolCallResultEntry {
    tool?: string;
    allowed?: boolean | null;
    known?: boolean | null;
    args_valid?: boolean | null;
    args_errors?: string[];
    args_match?: boolean | null;
    dry_run?: { ok?: boolean; is_error?: boolean; error?: string | null; snippet?: string; skipped?: string } | null;
}

export interface ToolRowResult {
    calls?: ToolCallResultEntry[];
    tool_call_valid?: boolean | null;
    tool_args_match?: boolean | null;
    plan_order_valid?: boolean | null;
    pass?: boolean | null;
    errors?: string[];
}

export interface ToolCallRow {
    scoring_mode?: string | null;
    row_tool_results?: ToolRowResult | null;
    row_tools_pass?: boolean | null;
    row_tool_selection_correct?: boolean | null;
    row_tool_args_match?: boolean | null;
    row_tool_call_pass?: boolean | null;
}

export function isToolCallMetrics(metrics: { scoring_mode?: string | null } | null | undefined): boolean {
    return metrics?.scoring_mode === 'tool_call';
}

export function hasToolsBlock(metrics: { tools?: unknown } | null | undefined): boolean {
    return metrics?.tools != null && typeof metrics.tools === 'object';
}

/** Rows scored by the tool-call mode (or carrying per-call results). */
export function isToolCallRow(row: ToolCallRow | null | undefined): boolean {
    return row?.scoring_mode === 'tool_call' || (row?.row_tool_results != null && typeof row.row_tool_results === 'object');
}

function pct(value: number | null | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'not applicable';
}


export function ToolCallMetricsSection({ tools }: { tools: ToolsBlock }) {
    const perTool = Object.entries(tools.per_tool ?? {});
    const unknown = Object.entries(tools.unknown_tools ?? {});
    const errors = tools.errors_top ?? [];
    const dry = tools.dry_run ?? null;
    return (
        <div className="tc-metrics" data-testid="tool-call-metrics-section">
            <p className="u-hint u-mt0" data-testid="tcm-provenance">
                Scored against tool catalog version {tools.catalog_version ?? 'unknown'}
                {tools.source_kind ? ` · source ${tools.source_kind}` : ''}
                {tools.flags_version ? ` · flags ${tools.flags_version}` : ''}
                {tools.modes && tools.modes.length > 0 ? ` · checks ${tools.modes.join(', ')}` : ''}
            </p>
            <dl className="tc-metrics__grid">
                <div>
                    <dt>Valid tool call</dt>
                    <dd data-testid="tcm-valid">{pct(tools.tool_call_valid_rate)}</dd>
                </div>
                <div>
                    <dt>Arguments match</dt>
                    <dd data-testid="tcm-args">{pct(tools.tool_args_match_rate)}</dd>
                </div>
                <div>
                    <dt>Plan order valid</dt>
                    <dd data-testid="tcm-order">{pct(tools.plan_order_valid_rate)}</dd>
                </div>
                <div>
                    <dt>Tool checks pass</dt>
                    <dd data-testid="tcm-pass">{pct(tools.pass_rate)}</dd>
                </div>
            </dl>

            {perTool.length > 0 && (
                <div data-testid="tcm-per-tool">
                    <h4 className="tc-metrics__h4">By tool</h4>
                    <table className="tc-metrics__table">
                        <thead><tr><th>Tool</th><th>Calls</th><th>Valid</th><th>Invalid arguments</th><th>Not allowed</th></tr></thead>
                        <tbody>
                            {perTool.map(([name, s]) => (
                                <tr key={name} data-testid={`tcm-tool-${name}`}>
                                    <td><code>{name}</code></td>
                                    <td>{s.calls ?? 0}</td>
                                    <td>{s.valid ?? 0}</td>
                                    <td>{s.invalid_args ?? 0}</td>
                                    <td>{s.not_allowed ?? 0}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {unknown.length > 0 && (
                <div data-testid="tcm-unknown">
                    <h4 className="tc-metrics__h4">Tools not in the catalog</h4>
                    <ul className="tc-metrics__list">
                        {unknown.map(([name, count]) => <li key={name} data-testid={`tcm-unknown-${name}`}><code>{name}</code> called {count} time{count === 1 ? '' : 's'}</li>)}
                    </ul>
                </div>
            )}

            {errors.length > 0 && (
                <div data-testid="tcm-errors">
                    <h4 className="tc-metrics__h4">Most common problems</h4>
                    <ul className="tc-metrics__list">
                        {errors.map((e, i) => <li key={i}>{e.message} <span className="tc-metrics__sub">{e.count ?? ''} rows</span></li>)}
                    </ul>
                </div>
            )}

            {dry && (
                <p className="u-hint" data-testid="tcm-dry-run">
                    {dry.enabled
                        ? `Dry run: ${dry.calls ?? 0} read-only call${dry.calls === 1 ? '' : 's'} executed, ${dry.ok ?? 0} succeeded, ${dry.errors ?? 0} failed${dry.skipped ? `, ${dry.skipped} skipped (mutating, never executed)` : ''}. Mutating tools were never called. Informational only.`
                        : 'Dry run was off.'}
                </p>
            )}
        </div>
    );
}

interface PanelProps {
    metrics: ToolCallMetrics;
    /** Where the numbers come from, e.g. "Run #21 · gold_test · result 1001". */
    provenance: string;
}

export default function ToolCallMetricsPanel({ metrics, provenance }: PanelProps) {
    const tools = metrics.tools ?? null;
    return (
        <section className="card tc-metrics" data-testid="tool-call-metrics">
            <h3 className="tc-metrics__title">Tool calls</h3>
            <p className="u-hint u-mt0">
                {provenance}. A row passes when the model picks the right tool with matching arguments and, when a catalog
                is set, the call is valid for that catalog.
            </p>
            <dl className="tc-metrics__grid">
                <div>
                    <dt>Tool-call pass rate</dt>
                    <dd data-testid="tcm-headline">
                        {pct(metrics.tool_call_pass_rate)}
                        <span className="tc-metrics__sub">{metrics.correct ?? 0} of {metrics.total ?? 0} rows</span>
                    </dd>
                </div>
                <div>
                    <dt>Right tool picked</dt>
                    <dd data-testid="tcm-selection">{pct(metrics.tool_selection_accuracy)}</dd>
                </div>
            </dl>
            {tools
                ? <ToolCallMetricsSection tools={tools} />
                : <p className="u-hint" data-testid="tcm-no-catalog">No tool catalog was set: only the tool pick and its arguments were checked.</p>}
        </section>
    );
}

function Chip({ ok, testId, children }: { ok: boolean | null | undefined; testId: string; children: string }) {
    const cls = ok === true ? 'is-pass' : ok === false ? 'is-fail' : '';
    return <li className={`tcr__chip ${cls}`} data-testid={testId}>{children}</li>;
}

export function ToolCallRowView({ row }: { row: ToolCallRow }) {
    const result = row.row_tool_results ?? null;
    if (!result) {
        // A tool_calling row scored without a catalog: only the pick and its arguments were checked.
        if (typeof row.row_tool_selection_correct !== 'boolean' && typeof row.row_tool_call_pass !== 'boolean') return null;
        return (
            <div className="tcr" data-testid="tool-call-row">
                <ul className="tcr__chips">
                    <Chip ok={row.row_tool_call_pass} testId="tcr-chip-call-pass">{row.row_tool_call_pass ? 'call passes' : 'call fails'}</Chip>
                    <Chip ok={row.row_tool_selection_correct} testId="tcr-chip-selection">{row.row_tool_selection_correct ? 'right tool picked' : 'wrong tool picked'}</Chip>
                    <Chip ok={row.row_tool_args_match} testId="tcr-chip-args">{row.row_tool_args_match ? 'arguments match' : 'arguments differ'}</Chip>
                </ul>
            </div>
        );
    }
    const calls = result.calls ?? [];
    const rowErrors = result.errors ?? [];
    const pass = result.pass ?? row.row_tools_pass;
    return (
        <div className="tcr" data-testid="tool-call-row">
            <ul className="tcr__chips">
                <Chip ok={pass} testId="tcr-chip-pass">{pass ? 'tool checks pass' : 'tool checks fail'}</Chip>
                <Chip ok={result.tool_call_valid} testId="tcr-chip-valid">
                    {result.tool_call_valid == null ? 'validity not checked' : result.tool_call_valid ? 'call valid' : 'call invalid'}
                </Chip>
                {result.tool_args_match != null && (
                    <Chip ok={result.tool_args_match} testId="tcr-chip-match">{result.tool_args_match ? 'arguments match' : 'arguments differ'}</Chip>
                )}
                {result.plan_order_valid != null && (
                    <Chip ok={result.plan_order_valid} testId="tcr-chip-order">{result.plan_order_valid ? 'order valid' : 'order invalid'}</Chip>
                )}
            </ul>
            {calls.length > 0 && (
                <table className="tcr__table" data-testid="tcr-calls">
                    <thead><tr><th>#</th><th>Tool</th><th>Known</th><th>Allowed</th><th>Arguments valid</th><th>Arguments match</th><th>Dry run</th></tr></thead>
                    <tbody>
                        {calls.map((c, i) => (
                            <tr key={i} data-testid={`tcr-call-${i}`}>
                                <td>{i + 1}</td>
                                <td>{c.tool}</td>
                                <td className={c.known === false ? 'is-fail' : ''}>{c.known == null ? '' : c.known ? 'yes' : 'no'}</td>
                                <td className={c.allowed === false ? 'is-fail' : ''}>{c.allowed == null ? '' : c.allowed ? 'yes' : 'no'}</td>
                                <td className={c.args_valid === false ? 'is-fail' : ''}>
                                    {c.args_valid == null ? '' : c.args_valid ? 'yes' : 'no'}
                                    {(c.args_errors ?? []).map((e, n) => <span key={n} className="tcr__err" data-testid={`tcr-call-${i}-err-${n}`}>{e}</span>)}
                                </td>
                                <td className={c.args_match === false ? 'is-fail' : ''}>{c.args_match == null ? '' : c.args_match ? 'yes' : 'no'}</td>
                                <td data-testid={`tcr-call-${i}-dry`}>
                                    {c.dry_run == null ? '' : c.dry_run.skipped ? `skipped: ${c.dry_run.skipped}` : c.dry_run.ok ? 'ok' : `failed${c.dry_run.error ? `: ${c.dry_run.error}` : ''}`}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
            {rowErrors.length > 0 && (
                <ul className="tcr__list" data-testid="tcr-errors">
                    {rowErrors.map((e, i) => <li key={i}>{e}</li>)}
                </ul>
            )}
        </div>
    );
}
