/**
 * SqlExecutionMetricsPanel — results of the opt-in SQL execution match: each
 * row's predicted query and answer-key query were run read-only against the
 * project's SQLite files and their results compared. A row passes only when
 * the results match on every database.
 *
 * Rows whose answer-key query failed are left out of the rate (and counted);
 * blocked queries (writes, ATTACH, PRAGMA, more than one statement) fail and
 * are counted. The databases and settings are listed so a number can be traced
 * to what it was scored against. `SqlExecutionRowView` shows one row's verdict.
 */

import './SqlExecutionMetricsPanel.css';

export interface SqlExecutionBlock {
    status?: string;
    error?: string;
    exec_match_rate?: number | null;
    n_rows?: number;
    n_scored?: number;
    n_passed?: number;
    n_unscoreable?: number;
    n_blocked?: number;
    n_timeout?: number;
    n_over_cap?: number;
    n_pred_error?: number;
    n_mismatch?: number;
    n_refusal_rows?: number;
    refusal_correct_rate?: number | null;
    databases?: string[];
    order?: string;
    timeout_ms?: number;
    row_cap?: number;
    float_rel_tol?: number;
    refusal_marker?: string;
    errors_top?: Array<{ message?: string; count?: number }>;
}

export interface SqlExecutionMetrics {
    sql_execution?: SqlExecutionBlock | null;
    exec_match_rate?: number | null;
    sql_execution_fingerprint?: string | null;
}

export interface SqlExecRowResult {
    pass?: boolean | null;
    status?: string;
    ordered?: boolean | null;
    refusal?: boolean;
    blocked?: boolean;
    error?: string | null;
    per_db?: Array<{ db?: string; pass?: boolean | null; gold_rows?: number | null; pred_rows?: number | null; pred_error?: string; gold_error?: string }>;
}

export interface SqlExecRow {
    row_sql_exec?: SqlExecRowResult | null;
    row_sql_exec_pass?: boolean | null;
}

export function hasSqlExecution(metrics: { sql_execution?: unknown } | null | undefined): boolean {
    return metrics?.sql_execution != null && typeof metrics.sql_execution === 'object';
}

export function isSqlExecRow(row: SqlExecRow | null | undefined): boolean {
    return row?.row_sql_exec != null && typeof row.row_sql_exec === 'object';
}

function pct(value: number | null | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'not applicable';
}

const ORDER_TEXT: Record<string, string> = {
    auto: 'row order checked when the answer-key query sorts its outer result',
    always: 'row order always checked',
    never: 'row order never checked',
};

interface PanelProps {
    metrics: SqlExecutionMetrics;
    /** Where the numbers come from, e.g. "Run #21 · gold_test · result 1001". */
    provenance: string;
}

export default function SqlExecutionMetricsPanel({ metrics, provenance }: PanelProps) {
    const block = metrics.sql_execution ?? null;
    if (!block) return null;
    if (block.status !== 'scored') {
        return (
            <section className="card sx-metrics" data-testid="sql-exec-metrics">
                <h3 className="sx-metrics__title">SQL execution match</h3>
                <p className="u-hint u-mt0" data-testid="sxm-config-error">
                    {provenance}. Not scored: {block.error ?? 'the settings could not be used'}. Fix the database list in the
                    SQL execution settings under Results › Verdict › How it's scored.
                </p>
            </section>
        );
    }
    const dbs = block.databases ?? [];
    const errors = block.errors_top ?? [];
    return (
        <section className="card sx-metrics" data-testid="sql-exec-metrics">
            <h3 className="sx-metrics__title">SQL execution match</h3>
            <p className="u-hint u-mt0" data-testid="sxm-provenance">
                {provenance}. Each predicted query and its answer-key query ran read-only on{' '}
                {dbs.length === 1 ? 'database' : `${dbs.length} databases`} {dbs.join(', ')}; a row passes only when the results
                match on every one. {ORDER_TEXT[block.order ?? 'auto'] ?? ''}
                {metrics.sql_execution_fingerprint ? ` · scoring ${metrics.sql_execution_fingerprint.slice(-8)}` : ''}
            </p>
            <dl className="sx-metrics__grid">
                <div>
                    <dt>Execution match</dt>
                    <dd data-testid="sxm-headline">
                        {pct(block.exec_match_rate)}
                        <span className="sx-metrics__sub">{block.n_passed ?? 0} of {block.n_scored ?? 0} scored rows</span>
                    </dd>
                </div>
                <div>
                    <dt>Answer key failed (left out)</dt>
                    <dd data-testid="sxm-unscoreable">{block.n_unscoreable ?? 0}</dd>
                </div>
                <div>
                    <dt>Blocked queries</dt>
                    <dd data-testid="sxm-blocked">{block.n_blocked ?? 0}</dd>
                </div>
                <div>
                    <dt>Timed out / too many rows</dt>
                    <dd data-testid="sxm-limits">{block.n_timeout ?? 0} / {block.n_over_cap ?? 0}</dd>
                </div>
                <div>
                    <dt>Refusal rows correct</dt>
                    <dd data-testid="sxm-refusal">
                        {pct(block.refusal_correct_rate)}
                        <span className="sx-metrics__sub">{block.n_refusal_rows ?? 0} rows expect {block.refusal_marker ?? 'the marker'}</span>
                    </dd>
                </div>
            </dl>
            {errors.length > 0 && (
                <div data-testid="sxm-errors">
                    <h4 className="sx-metrics__h4">Most common problems</h4>
                    <ul className="sx-metrics__list">
                        {errors.map((e, i) => <li key={i}>{e.message} <span className="sx-metrics__count">{e.count ?? ''} rows</span></li>)}
                    </ul>
                </div>
            )}
        </section>
    );
}

const STATUS_TEXT: Record<string, string> = {
    match: 'results match',
    mismatch: 'results differ',
    pred_error: 'query failed',
    blocked: 'query blocked',
    timeout: 'timed out',
    over_cap: 'too many rows',
    gold_error: 'answer key failed: left out',
    refusal_correct: 'refused as expected',
    refusal_wrong: 'should have refused',
    unexpected_refusal: 'refused, but a query was expected',
};

export function SqlExecutionRowView({ row }: { row: SqlExecRow }) {
    const res = row.row_sql_exec ?? null;
    if (!res) return null;
    const pass = row.row_sql_exec_pass ?? res.pass;
    const cls = pass === true ? 'is-pass' : pass === false ? 'is-fail' : '';
    return (
        <div className="sxr" data-testid="sql-exec-row">
            <ul className="sxr__chips">
                <li className={`sxr__chip ${cls}`} data-testid="sxr-status">{STATUS_TEXT[res.status ?? ''] ?? res.status ?? 'scored'}</li>
                {res.ordered != null && <li className="sxr__chip">{res.ordered ? 'order checked' : 'order ignored'}</li>}
            </ul>
            {(res.per_db ?? []).length > 0 && (
                <ul className="sxr__list" data-testid="sxr-per-db">
                    {(res.per_db ?? []).map((d, i) => (
                        <li key={i}>
                            {d.db}: {d.pass === true ? 'match' : d.pass === false ? 'differs' : 'not scored'}
                            {d.gold_rows != null ? ` · answer key ${d.gold_rows} rows` : ''}
                            {d.pred_rows != null ? ` · prediction ${d.pred_rows} rows` : ''}
                        </li>
                    ))}
                </ul>
            )}
            {res.error && <p className="sxr__err" data-testid="sxr-error">{res.error}</p>}
        </div>
    );
}
