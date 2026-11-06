/**
 * MultiOptionMetricsPanel — results of a `multi_option` eval (the opt-in scoring
 * mode of the structured-extraction handler: the model answers with several
 * alternative options, each built on a strategy and delivered in several
 * formats, and an operator-registered option verifier judges every option).
 *
 * The headline is the row rule's pass rate (top option / any option / all
 * options pass). Rows the verifier could not run on (infrastructure) are
 * excluded from every rate and counted separately; an option the verifier
 * timed out / crashed on counts as failing, with its reason. A rate with no
 * scored row reads "not scored", never 0%.
 *
 * Also exports MultiOptionRowView: one card per option in the model's order —
 * strategy chip (muted when outside the project's taxonomy), title, the
 * rationale / tradeoffs / rollback, one tab per format with the fix text, a
 * format × check grid and the cross-format agreement badge.
 */

import { useState } from 'react';
import './MultiOptionMetricsPanel.css';

export interface MultiOptionFormatVerdict {
    checks?: Record<string, boolean | null> | null;
    passes?: boolean | null;
    messages?: string[] | null;
}

export interface MultiOptionAgreement {
    agree?: boolean | null;
    differences?: string[] | null;
}

export interface MultiOptionCard {
    rank?: number;
    strategy?: string | null;
    in_taxonomy?: boolean | null;
    malformed?: boolean;
    title?: string;
    rationale?: string;
    tradeoffs?: string;
    rollback?: string;
    fixes?: Record<string, string | null> | null;
    formats?: Record<string, MultiOptionFormatVerdict> | null;
    agreement?: MultiOptionAgreement | null;
    passes?: boolean | null;
    error?: string | null;
    failure_reason?: string | null;
    note?: string | null;
    extra_fields?: string[];
    fixes_truncated?: string[];
}

export interface MultiOptionSettings {
    formats?: string[];
    min_options?: number;
    max_options?: number;
    row_pass?: string;
    require_agreement?: boolean;
    strategy_taxonomy?: string[];
    verifier?: { id?: string; revision?: number } | null;
}

export interface MultiOptionMetrics {
    scoring_mode?: string;
    multi_option?: MultiOptionSettings | null;
    multi_option_pass_rate?: number | null;
    option_pass_rate?: number | null;
    rows_with_passing_option_rate?: number | null;
    top_option_pass_rate?: number | null;
    all_options_pass_rate?: number | null;
    distinct_strategies_mean?: number | null;
    rows_with_two_distinct_passing_rate?: number | null;
    out_of_taxonomy_rate?: number | null;
    cross_format_agreement_rate?: number | null;
    option_count_in_range_rate?: number | null;
    json_validity_rate?: number | null;
    malformed_rate?: number | null;
    format_pass_rate?: Record<string, number | null> | null;
    check_pass_rate?: Record<string, number | null> | null;
    verifier_error_rows?: number;
    rows_missing_verifier_payload?: number;
    option_verifier_failures?: number;
    verifier_error?: string;
    verifier_unavailable?: string;
    multi_option_fingerprint?: string;
    options_scored?: number;
    total?: number;
    correct?: number;
}

export interface MultiOptionRow {
    scoring_mode?: string | null;
    row_option_count?: number | null;
    row_option_count_in_range?: boolean | null;
    row_options?: MultiOptionCard[] | null;
    row_options_truncated?: boolean | null;
    row_options_passing?: number | null;
    row_top_option_pass?: boolean | null;
    row_any_option_pass?: boolean | null;
    row_distinct_strategies?: number | null;
    row_distinct_strategies_all?: number | null;
    row_verifier_error?: boolean | null;
    row_option_verifier_failures?: number | null;
    row_verifier_note?: string | null;
    row_multi_option_pass?: boolean | null;
    row_explanation?: string | null;
    row_malformed?: boolean | null;
}

export function isMultiOptionMetrics(metrics: { scoring_mode?: string | null } | null | undefined): boolean {
    return metrics?.scoring_mode === 'multi_option';
}

/** Rows scored by the multi-option mode. */
export function isMultiOptionRow(row: { scoring_mode?: string | null; row_options?: unknown } | null | undefined): boolean {
    return row?.scoring_mode === 'multi_option' && Array.isArray(row.row_options);
}

const ROW_RULE_LABEL: Record<string, string> = {
    top_option: 'the top option passes',
    any_option: 'at least one option passes',
    all_options: 'every option passes',
};

function pct(value: number | null | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

/** A headline rate: null means no row could be scored (never shown as 0%). */
function scoredPct(value: number | null | undefined): string {
    return typeof value === 'number' ? pct(value) : 'not scored';
}

const FAILURE_REASON_LABEL: Record<string, string> = {
    verifier_timeout: 'the verifier timed out on this option',
    verifier_crash: 'the verifier crashed on this option',
    verifier_bad_output: 'the verifier printed no usable verdict for this option',
    verifier_output_too_large: "the verifier's output for this option was over the size cap",
    verifier_undecided: 'the verifier did not decide this option',
};

export function failureReasonLabel(reason: string): string {
    return FAILURE_REASON_LABEL[reason] ?? reason.replace(/_/g, ' ');
}

/** Width step (nearest 10%) for a rate bar: no inline styles. */
export function widthStep(value: number | null | undefined): number {
    if (typeof value !== 'number' || Number.isNaN(value)) return 0;
    return Math.max(0, Math.min(100, Math.round(value * 10) * 10));
}

function RateBars({ rates, testId }: { rates: Record<string, number | null>; testId: string }) {
    const entries = Object.entries(rates);
    if (entries.length === 0) return null;
    return (
        <ul className="mo-bars" data-testid={testId}>
            {entries.map(([name, value]) => (
                <li key={name} className="mo-bars__row" data-testid={`${testId}-${name}`}>
                    <span className="mo-bars__name"><code>{name}</code></span>
                    <span className="mo-bars__track" aria-label={`${pct(value)} pass`}>
                        <span className={`mo-bars__fill mo-bars__fill--w-${widthStep(value)}`} />
                    </span>
                    <span className="mo-bars__value">{pct(value)}</span>
                </li>
            ))}
        </ul>
    );
}

interface PanelProps {
    metrics: MultiOptionMetrics;
    /** Where the numbers come from, e.g. "Run #21 · test · result 1001". */
    provenance: string;
}

export default function MultiOptionMetricsPanel({ metrics, provenance }: PanelProps) {
    const cfg = metrics.multi_option ?? {};
    const verifier = cfg.verifier;
    const verifierLine = verifier?.id ? `verifier ${verifier.id} (revision ${verifier.revision ?? 1})` : 'no verifier';
    const formats = cfg.formats ?? [];
    const rule = ROW_RULE_LABEL[cfg.row_pass ?? 'top_option'] ?? cfg.row_pass ?? '';
    const errors = metrics.verifier_error_rows ?? 0;
    const missingPayload = metrics.rows_missing_verifier_payload ?? 0;
    const optionFailures = metrics.option_verifier_failures ?? 0;
    const formatRates = metrics.format_pass_rate ?? {};
    const checkRates = metrics.check_pass_rate ?? {};
    return (
        <section className="card mo-metrics" data-testid="multi-option-metrics">
            <h3 className="mo-metrics__title">Multiple options</h3>
            <p className="u-hint u-mt0">
                {provenance}. A row passes when {rule}. An option passes when every format passes the verifier's
                checks{cfg.require_agreement ? ' and the formats end in the same state' : ''}.
            </p>
            <p className="u-hint u-mt0" data-testid="mo-provenance">
                Scored with {verifierLine} · formats {formats.length > 0 ? formats.join(', ') : 'none set'} ·
                {' '}{cfg.min_options ?? 1}-{cfg.max_options ?? 1} options
                {metrics.multi_option_fingerprint ? ` · scoring ${metrics.multi_option_fingerprint.slice(0, 8)}` : ''}
            </p>

            {metrics.verifier_unavailable && (
                <p className="mo-metrics__bad" data-testid="mo-verifier-unavailable">
                    The option verifier could not run: {metrics.verifier_unavailable}
                </p>
            )}
            {errors > 0 && (
                <p className="mo-metrics__warn" data-testid="mo-verifier-errors">
                    {errors} row{errors === 1 ? '' : 's'} could not be verified and {errors === 1 ? 'is' : 'are'} left out of every rate below.
                </p>
            )}
            {missingPayload > 0 && (
                <p className="mo-metrics__warn" data-testid="mo-missing-payload">
                    {missingPayload} row{missingPayload === 1 ? ' has' : 's have'} no verifier_payload at the top level of the
                    test file, so the verifier was not run on {missingPayload === 1 ? 'it' : 'them'}.
                </p>
            )}
            {optionFailures > 0 && (
                <p className="mo-metrics__warn" data-testid="mo-option-failures">
                    {optionFailures} option{optionFailures === 1 ? '' : 's'} timed out, crashed or broke the verifier's output and
                    {optionFailures === 1 ? ' counts' : ' count'} as failing.
                </p>
            )}

            <dl className="mo-metrics__grid">
                <div>
                    <dt>Pass rate</dt>
                    <dd data-testid="mo-pass">
                        {scoredPct(metrics.multi_option_pass_rate)}
                        <span className="mo-metrics__sub">{metrics.correct ?? 0} of {metrics.total ?? 0} verified rows</span>
                    </dd>
                </div>
                <div>
                    <dt>Options passing every check</dt>
                    <dd data-testid="mo-option-pass">
                        {pct(metrics.option_pass_rate)}
                        <span className="mo-metrics__sub">{metrics.options_scored ?? 0} options verified</span>
                    </dd>
                </div>
                <div>
                    <dt>Rows with a passing option</dt>
                    <dd data-testid="mo-any">{pct(metrics.rows_with_passing_option_rate)}</dd>
                </div>
                <div>
                    <dt>Top option passes</dt>
                    <dd data-testid="mo-top">{pct(metrics.top_option_pass_rate)}</dd>
                </div>
                <div>
                    <dt>Distinct passing strategies</dt>
                    <dd data-testid="mo-distinct">
                        {typeof metrics.distinct_strategies_mean === 'number' ? metrics.distinct_strategies_mean.toFixed(2) : '—'}
                        <span className="mo-metrics__sub">
                            {pct(metrics.rows_with_two_distinct_passing_rate)} of rows have two or more
                        </span>
                    </dd>
                </div>
                <div>
                    <dt>Strategy outside the taxonomy</dt>
                    <dd data-testid="mo-taxonomy">
                        {metrics.out_of_taxonomy_rate === undefined ? 'no taxonomy' : pct(metrics.out_of_taxonomy_rate)}
                    </dd>
                </div>
                <div>
                    <dt>Formats agree</dt>
                    <dd data-testid="mo-agreement">
                        {metrics.cross_format_agreement_rate === undefined ? 'not reported' : pct(metrics.cross_format_agreement_rate)}
                    </dd>
                </div>
                <div>
                    <dt>Option count in range</dt>
                    <dd data-testid="mo-count">{pct(metrics.option_count_in_range_rate)}</dd>
                </div>
                <div>
                    <dt>Malformed output</dt>
                    <dd data-testid="mo-malformed">{pct(metrics.malformed_rate)}</dd>
                </div>
            </dl>

            {Object.keys(formatRates).length > 0 && (
                <div>
                    <h4 className="mo-metrics__h4">Pass rate by format</h4>
                    <RateBars rates={formatRates} testId="mo-format" />
                </div>
            )}
            {Object.keys(checkRates).length > 0 && (
                <div>
                    <h4 className="mo-metrics__h4">Pass rate by verifier check</h4>
                    <RateBars rates={checkRates} testId="mo-check" />
                </div>
            )}
        </section>
    );
}

function Chip({ ok, testId, children, muted }: { ok?: boolean | null; testId: string; children: string; muted?: boolean }) {
    const cls = ok === true ? 'is-pass' : ok === false ? 'is-fail' : '';
    return <li className={`mo-chip ${cls} ${muted ? 'is-muted' : ''}`} data-testid={testId}>{children}</li>;
}

function verdictText(value: boolean | null | undefined): string {
    return value === true ? 'pass' : value === false ? 'fail' : 'n/a';
}

function verdictClass(value: boolean | null | undefined): string {
    return value === true ? 'is-pass' : value === false ? 'is-fail' : 'is-na';
}

export function OptionCard({ option, testId }: { option: MultiOptionCard; testId: string }) {
    const verdicts = option.formats && Object.keys(option.formats).length > 0 ? option.formats : null;
    const formatNames = Object.keys(verdicts ?? option.fixes ?? {});
    const [active, setActive] = useState<string | null>(formatNames[0] ?? null);
    const checks = Array.from(new Set(formatNames.flatMap((f) => Object.keys(option.formats?.[f]?.checks ?? {}))));
    const rank = option.rank ?? 0;
    const outside = option.in_taxonomy === false;
    const status = option.error ? 'unverified' : option.passes === true ? 'passes' : 'fails';
    const truncatedFormats = new Set(option.fixes_truncated ?? []);
    const agreement = option.agreement;
    const failedMessages = formatNames.flatMap((f) => {
        const v = option.formats?.[f];
        return v?.passes === false ? (v.messages ?? []).map((m) => `${f}: ${m}`) : [];
    });
    return (
        <article className={`mo-card mo-card--${status}`} data-testid={testId}>
            <header className="mo-card__head">
                <span className="mo-card__rank">Option {rank + 1}{rank === 0 ? ' (top)' : ''}</span>
                {option.strategy ? (
                    <span className={`mo-card__strategy ${outside ? 'is-muted' : ''}`} data-testid={`${testId}-strategy`}>
                        {option.strategy}{outside ? ' (not in taxonomy)' : ''}
                    </span>
                ) : (
                    <span className="mo-card__strategy is-muted">no strategy</span>
                )}
                <span className={`mo-card__status mo-card__status--${status}`} data-testid={`${testId}-status`}>{status}</span>
            </header>
            {option.title && <p className="mo-card__title">{option.title}</p>}
            {option.malformed && <p className="mo-metrics__bad">{option.note || 'not a JSON object'}</p>}
            {option.error && <p className="mo-metrics__warn" data-testid={`${testId}-error`}>Not verified: {option.error}</p>}
            {option.failure_reason && (
                <p className="mo-metrics__bad" data-testid={`${testId}-failure`}>
                    Fails: {failureReasonLabel(option.failure_reason)}{option.note ? ` (${option.note})` : ''}
                </p>
            )}
            {(option.rationale || option.tradeoffs || option.rollback) && (
                <details className="mo-card__details">
                    <summary>Rationale, tradeoffs, rollback</summary>
                    {option.rationale && <p><strong>Rationale.</strong> {option.rationale}</p>}
                    {option.tradeoffs && <p><strong>Tradeoffs.</strong> {option.tradeoffs}</p>}
                    {option.rollback && <p><strong>Rollback.</strong> {option.rollback}</p>}
                </details>
            )}
            {formatNames.length > 0 && (
                <div className="mo-card__formats">
                    <div className="mo-card__tabs">
                        {formatNames.map((f) => (
                            <button
                                key={f}
                                type="button"
                                className={`mo-card__tab ${f === active ? 'is-active' : ''} ${verdictClass(option.formats?.[f]?.passes)}`}
                                data-testid={`${testId}-tab-${f}`}
                                onClick={() => setActive(f)}
                            >
                                {f} · {verdictText(option.formats?.[f]?.passes)}
                            </button>
                        ))}
                    </div>
                    {active && (
                        <pre className="mo-card__fix" data-testid={`${testId}-fix`}>
                            {option.fixes?.[active] || '(no fix in this format)'}
                            {truncatedFormats.has(active) ? '\n(truncated)' : ''}
                        </pre>
                    )}
                </div>
            )}
            {checks.length > 0 && (
                <table className="mo-card__grid" data-testid={`${testId}-checks`}>
                    <thead>
                        <tr>
                            <th>Check</th>
                            {formatNames.map((f) => <th key={f}>{f}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {checks.map((c) => (
                            <tr key={c}>
                                <td><code>{c}</code></td>
                                {formatNames.map((f) => {
                                    const value = option.formats?.[f]?.checks?.[c];
                                    return (
                                        <td key={f} className={verdictClass(value)} data-testid={`${testId}-check-${f}-${c}`}>
                                            {verdictText(value)}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
            {agreement && typeof agreement.agree === 'boolean' && (
                <div className="mo-card__agreement" data-testid={`${testId}-agreement`}>
                    <span className={`mo-chip ${agreement.agree ? 'is-pass' : 'is-fail'}`}>
                        {agreement.agree ? 'formats agree' : 'formats disagree'}
                    </span>
                    {(agreement.differences ?? []).length > 0 && (
                        <ul className="mo-card__list">
                            {(agreement.differences ?? []).map((d, i) => <li key={i}>{d}</li>)}
                        </ul>
                    )}
                </div>
            )}
            {failedMessages.length > 0 && (
                <ul className="mo-card__list" data-testid={`${testId}-messages`}>
                    {failedMessages.map((m, i) => <li key={i}>{m}</li>)}
                </ul>
            )}
        </article>
    );
}

export function MultiOptionRowView({ row }: { row: MultiOptionRow }) {
    const options = row.row_options ?? [];
    const unverified = row.row_verifier_error === true;
    return (
        <div className="mo-row" data-testid="multi-option-row">
            <ul className="mo-chips">
                <Chip ok={unverified ? null : row.row_multi_option_pass} testId="mo-chip-pass">
                    {unverified ? 'not verified' : row.row_multi_option_pass ? 'pass' : 'fail'}
                </Chip>
                <Chip ok={row.row_option_count_in_range} testId="mo-chip-count">
                    {`${row.row_option_count ?? 0} option${row.row_option_count === 1 ? '' : 's'}`}
                </Chip>
                <Chip testId="mo-chip-passing">{`${row.row_options_passing ?? 0} passing`}</Chip>
                <Chip testId="mo-chip-distinct">{`${row.row_distinct_strategies ?? 0} distinct passing strategies`}</Chip>
            </ul>
            {row.row_malformed && <p className="mo-metrics__bad" data-testid="mo-row-malformed">The output is not a JSON object with an options list.</p>}
            {row.row_verifier_note && <p className="mo-metrics__warn" data-testid="mo-row-note">{row.row_verifier_note}</p>}
            {row.row_explanation && <p className="mo-row__explanation">{row.row_explanation}</p>}
            <div className="mo-row__cards">
                {options.map((o, i) => <OptionCard key={i} option={o} testId={`mo-option-${i}`} />)}
            </div>
            {row.row_options_truncated && (
                <p className="u-hint" data-testid="mo-row-options-truncated">
                    (truncated) Only the first {options.length} options are shown; every option was scored.
                </p>
            )}
        </div>
    );
}
