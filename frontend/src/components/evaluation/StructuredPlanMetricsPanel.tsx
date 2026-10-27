/**
 * StructuredPlanMetricsPanel — results of a structured-plan eval (the
 * `structured_plan` scoring mode of the structured-extraction handler: a model
 * writes a JSON plan for a scenario and the plan is validated against a
 * JSON-Schema bundle, compared field by field at JSON pointers and, optionally,
 * scored per criterion by a rubric judge).
 *
 * The headline is the pass rate: valid JSON, schema-valid, every pointer
 * matches the answer key and no critical rubric criterion fails. Parts that did
 * not run (no schema, no judge) say so instead of showing a number.
 *
 * Also exports StructuredPlanRowView: per-row schema errors, pointer results
 * (predicted vs gold) and rubric scores.
 */

import './StructuredPlanMetricsPanel.css';

export interface StructuredPlanSchemaError {
    pointer?: string;
    message?: string;
    validator?: string;
    count?: number;
}

export interface StructuredPlanPointerStat {
    match_rate?: number | null;
    support?: number;
    missing_rate?: number | null;
    /** Rows where the pointer was absent from both the output and the answer key (not counted as missing). */
    both_missing?: number;
}

export interface StructuredPlanRubricCriterionStat {
    description?: string;
    weight?: number;
    critical?: boolean;
    mean?: number | null;
    n?: number;
    low_count?: number;
}

export interface StructuredPlanRubricBlock {
    /** "no_judge" when a rubric is configured but no judge could run. */
    status?: string;
    id?: string;
    title?: string;
    sha256?: string;
    judge?: string | null;
    judged?: number;
    unjudged?: number;
    mean_weighted?: number | null;
    pass_rate?: number | null;
    critical_fail_count?: number;
    per_criterion?: Record<string, StructuredPlanRubricCriterionStat>;
    criteria_order?: string[];
}

export interface StructuredPlanMetrics {
    scoring_mode?: string;
    structured_plan_pass_rate?: number | null;
    schema_valid_rate?: number | null;
    json_validity_rate?: number | null;
    field_match_rate?: number | null;
    schema_missing?: boolean;
    /** The live schema differs from the one this split was prepared with. */
    schema_changed_since_prep?: boolean;
    prep_schema_sha256?: string;
    schema?: { root?: string; sha256?: string } | null;
    schema_errors_top?: StructuredPlanSchemaError[];
    per_pointer?: Record<string, StructuredPlanPointerStat>;
    pointers?: string[];
    rubric?: StructuredPlanRubricBlock | null;
    rubric_pass_rate?: number | null;
    rubric_mean_score?: number | null;
    total?: number;
    correct?: number;
}

export interface StructuredPlanPointerResult {
    match?: boolean;
    pred?: unknown[];
    gold?: unknown[];
    source?: string;
}

export interface StructuredPlanRubricScore {
    score?: number;
    reason?: string;
}

export interface StructuredPlanRow {
    scoring_mode?: string | null;
    is_valid_json?: boolean | null;
    row_schema_valid?: boolean | null;
    schema_errors?: Array<StructuredPlanSchemaError | string> | null;
    row_pointer_results?: Record<string, StructuredPlanPointerResult> | null;
    row_structured_plan_pass?: boolean | null;
    row_rubric_scores?: Record<string, StructuredPlanRubricScore> | null;
    row_rubric_weighted?: number | null;
    row_rubric_critical_failed?: string[] | null;
    row_rubric_passed?: boolean | null;
    row_rubric_status?: string | null;
}

export function isStructuredPlanMetrics(metrics: { scoring_mode?: string | null } | null | undefined): boolean {
    return metrics?.scoring_mode === 'structured_plan';
}

/** Rows scored by the structured-plan mode (or carrying its flags). */
export function isStructuredPlanRow(row: { scoring_mode?: string | null; row_structured_plan_pass?: unknown } | null | undefined): boolean {
    return row?.scoring_mode === 'structured_plan'
        || (row != null && row.row_structured_plan_pass !== undefined && row.row_structured_plan_pass !== null);
}

function pct(value: number | null | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

function score(value: number | null | undefined): string {
    return typeof value === 'number' ? value.toFixed(2) : '—';
}

function shortSha(sha: string | null | undefined): string {
    return sha ? sha.slice(0, 8) : '';
}

function show(values: unknown[] | undefined): string {
    if (!values || values.length === 0) return '(missing)';
    return values.map((v) => (typeof v === 'string' ? v : JSON.stringify(v))).join(', ');
}

interface PanelProps {
    metrics: StructuredPlanMetrics;
    /** Where the numbers come from, e.g. "Run #21 · gold_test · result 1001". */
    provenance: string;
}

export default function StructuredPlanMetricsPanel({ metrics, provenance }: PanelProps) {
    const errors = metrics.schema_errors_top ?? [];
    const pointers = Object.entries(metrics.per_pointer ?? {});
    const rubric = metrics.rubric ?? null;
    const rubricRan = rubric != null && rubric.status !== 'no_judge' && (rubric.judged ?? 0) + (rubric.unjudged ?? 0) > 0;
    const order = rubric?.criteria_order ?? Object.keys(rubric?.per_criterion ?? {});
    const schemaLine = metrics.schema?.root
        ? `schema ${metrics.schema.root} (sha ${shortSha(metrics.schema.sha256)})`
        : 'no schema';
    const rubricLine = rubric?.id ? `rubric ${rubric.id} (sha ${shortSha(rubric.sha256)})` : 'no rubric';
    const judgeLine = rubric?.judge ? `judge ${rubric.judge}` : 'no judge';
    return (
        <section className="card sp-metrics" data-testid="structured-plan-metrics">
            <h3 className="sp-metrics__title">Structured plan</h3>
            <p className="u-hint u-mt0">
                {provenance}. A row passes when its output is valid JSON, validates against the schema, every pointer
                field matches the answer key and no critical rubric criterion fails.
            </p>
            <p className="u-hint u-mt0" data-testid="sp-provenance">
                Scored with {schemaLine} · {rubricLine} · {judgeLine}
            </p>

            {metrics.schema_changed_since_prep && (
                <p className="sp-metrics__bad" data-testid="sp-schema-changed">
                    The schema was changed after this split was prepared (prepared with sha {shortSha(metrics.prep_schema_sha256)}).
                    The prompts and the schema check used the current schema; re-prepare the data to train on it.
                </p>
            )}

            <dl className="sp-metrics__grid">
                <div>
                    <dt>Pass rate</dt>
                    <dd data-testid="sp-pass">
                        {pct(metrics.structured_plan_pass_rate)}
                        <span className="sp-metrics__sub">{metrics.correct ?? 0} of {metrics.total ?? 0} rows</span>
                    </dd>
                </div>
                <div>
                    <dt>Valid JSON</dt>
                    <dd data-testid="sp-json">{pct(metrics.json_validity_rate)}</dd>
                </div>
                <div>
                    <dt>Schema valid</dt>
                    <dd data-testid="sp-schema-valid">
                        {metrics.schema_missing ? 'not evaluated' : pct(metrics.schema_valid_rate)}
                        {metrics.schema_missing && <span className="sp-metrics__sub">no schema saved: only checked for a JSON object</span>}
                    </dd>
                </div>
                <div>
                    <dt>All pointers match</dt>
                    <dd data-testid="sp-field-match">
                        {pointers.length > 0 ? pct(metrics.field_match_rate) : 'not evaluated'}
                        {pointers.length === 0 && <span className="sp-metrics__sub">no pointer checks configured</span>}
                    </dd>
                </div>
                <div>
                    <dt>Rubric</dt>
                    <dd data-testid="sp-rubric-pass">
                        {rubricRan ? pct(rubric?.pass_rate ?? metrics.rubric_pass_rate) : rubric?.status === 'no_judge' ? 'no judge' : 'not run'}
                        {rubricRan && (
                            <span className="sp-metrics__sub">mean score {score(rubric?.mean_weighted ?? metrics.rubric_mean_score)} of 5</span>
                        )}
                    </dd>
                </div>
            </dl>

            {rubric?.status === 'no_judge' && (
                <p className="sp-metrics__bad" data-testid="sp-no-judge">
                    A rubric is configured but no judge model was available, so the rubric did not count toward the pass.
                </p>
            )}

            {errors.length > 0 && (
                <div data-testid="sp-errors">
                    <h4 className="sp-metrics__h4">Most common schema errors</h4>
                    <table className="sp-metrics__table">
                        <thead><tr><th>Where</th><th>Rule</th><th>Message</th><th>Rows</th></tr></thead>
                        <tbody>
                            {errors.map((e, i) => (
                                <tr key={i} data-testid={`sp-error-${i}`}>
                                    <td><code>{e.pointer || '/'}</code></td>
                                    <td>{e.validator ?? ''}</td>
                                    <td>{e.message}</td>
                                    <td>{e.count ?? ''}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {pointers.length > 0 && (
                <div data-testid="sp-pointers">
                    <h4 className="sp-metrics__h4">Field match by pointer</h4>
                    <table className="sp-metrics__table">
                        <thead><tr><th>Pointer</th><th>Match rate</th><th>Rows</th><th>Missing (answer key has it)</th></tr></thead>
                        <tbody>
                            {pointers.map(([ptr, s]) => (
                                <tr key={ptr} data-testid={`sp-pointer-${ptr}`}>
                                    <td><code>{ptr}</code></td>
                                    <td>{pct(s.match_rate)}</td>
                                    <td>{s.support ?? ''}</td>
                                    <td>{pct(s.missing_rate)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {rubricRan && rubric && (
                <div data-testid="sp-rubric">
                    <h4 className="sp-metrics__h4">Rubric by criterion</h4>
                    <p className="u-hint u-mt0" data-testid="sp-rubric-line">
                        {rubric.title || rubric.id} · {judgeLine} · {rubric.judged ?? 0} judged, {rubric.unjudged ?? 0} unjudged
                        {(rubric.unjudged ?? 0) > 0 ? ' (unjudged rows are not counted as failures)' : ''}
                        {(rubric.critical_fail_count ?? 0) > 0 ? ` · ${rubric.critical_fail_count} with a critical criterion failed` : ''}
                    </p>
                    <table className="sp-metrics__table">
                        <thead>
                            <tr><th>Criterion</th><th>Weight</th><th>Critical</th><th>Mean score</th><th>Scores of 1-2</th><th>Rows</th></tr>
                        </thead>
                        <tbody>
                            {order.map((id) => {
                                const c = rubric.per_criterion?.[id];
                                if (!c) return null;
                                return (
                                    <tr key={id} data-testid={`sp-criterion-${id}`}>
                                        <td><code>{id}</code>{c.description ? <span className="sp-metrics__sub">{c.description}</span> : null}</td>
                                        <td>{c.weight ?? ''}</td>
                                        <td>{c.critical ? <span className="sp-metrics__badge">critical</span> : ''}</td>
                                        <td>{score(c.mean)}</td>
                                        <td>{c.low_count ?? 0}</td>
                                        <td>{c.n ?? ''}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            )}
        </section>
    );
}

function Chip({ ok, testId, children }: { ok: boolean | null | undefined; testId: string; children: string }) {
    const cls = ok === true ? 'is-pass' : ok === false ? 'is-fail' : '';
    return <li className={`sp-row__chip ${cls}`} data-testid={testId}>{children}</li>;
}

export function StructuredPlanRowView({ row }: { row: StructuredPlanRow }) {
    const errors = row.schema_errors ?? [];
    const pointers = Object.entries(row.row_pointer_results ?? {});
    const scores = Object.entries(row.row_rubric_scores ?? {});
    const critical = row.row_rubric_critical_failed ?? [];
    const judged = row.row_rubric_status === 'judged';
    return (
        <div className="sp-row" data-testid="structured-plan-row">
            <ul className="sp-row__chips">
                <Chip ok={row.is_valid_json} testId="sp-chip-json">{row.is_valid_json ? 'valid JSON' : 'not valid JSON'}</Chip>
                <Chip ok={row.row_schema_valid} testId="sp-chip-schema">
                    {row.row_schema_valid == null ? 'schema not checked' : row.row_schema_valid ? 'schema valid' : 'schema invalid'}
                </Chip>
                <Chip ok={row.row_structured_plan_pass} testId="sp-chip-pass">
                    {row.row_structured_plan_pass ? 'pass' : 'fail'}
                </Chip>
                {row.row_rubric_status && (
                    <Chip ok={judged ? row.row_rubric_passed : null} testId="sp-chip-rubric">
                        {judged
                            ? `rubric ${row.row_rubric_passed ? 'passed' : 'failed'} (${score(row.row_rubric_weighted)} of 5)`
                            : 'rubric unjudged'}
                    </Chip>
                )}
            </ul>

            {errors.length > 0 && (
                <ul className="sp-row__list" data-testid="sp-row-errors">
                    {errors.map((e, i) => (
                        <li key={i}>
                            {typeof e === 'string' ? e : <><code>{e.pointer || '/'}</code> {e.message}</>}
                        </li>
                    ))}
                </ul>
            )}

            {pointers.length > 0 && (
                <table className="sp-row__table" data-testid="sp-row-pointers">
                    <thead><tr><th>Pointer</th><th>Predicted</th><th>Expected</th><th>Match</th></tr></thead>
                    <tbody>
                        {pointers.map(([ptr, r]) => (
                            <tr key={ptr} data-testid={`sp-row-pointer-${ptr}`}>
                                <td>{ptr}</td>
                                <td>{show(r.pred)}</td>
                                <td>{show(r.gold)}{r.source === 'acceptable' ? ' (any of)' : ''}</td>
                                <td className={r.match ? '' : 'is-fail'}>{r.match ? 'yes' : 'no'}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}

            {critical.length > 0 && (
                <p className="sp-metrics__bad" data-testid="sp-row-critical">Critical criteria failed: {critical.join(', ')}</p>
            )}
            {scores.length > 0 && (
                <table className="sp-row__table" data-testid="sp-row-rubric">
                    <thead><tr><th>Criterion</th><th>Score</th><th>Reason</th></tr></thead>
                    <tbody>
                        {scores.map(([id, s]) => (
                            <tr key={id} data-testid={`sp-row-criterion-${id}`}>
                                <td>{id}</td>
                                <td className={critical.includes(id) ? 'is-fail' : ''}>{s.score ?? ''}</td>
                                <td>{s.reason ?? ''}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}
