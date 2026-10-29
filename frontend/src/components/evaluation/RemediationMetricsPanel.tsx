/**
 * RemediationMetricsPanel — results of a code-remediation eval (the
 * `remediation` scoring mode of the structured-extraction handler: a model
 * writes a unified-diff patch for a reported finding, and the patch is applied
 * to the code and checked).
 *
 * The headline is the pass rate: the patch applies, the patched code still
 * parses (when a check could run), the finding is gone (when a detector ran)
 * and any cited passage id exists. Parts that could not be evaluated say so
 * instead of showing a number.
 *
 * Also exports RemediationRowView, the per-row patch (as a diff), status chips,
 * check messages and cited passages with their provenance.
 *
 * When the project opted in to scoring held-out task -> code rows
 * (`score_code_examples`), the `code_example_*` metrics get their own
 * "Task -> code rows" block and those rows their own row view
 * (CodeExampleRowView). Without that setting nothing here changes.
 */

import './RemediationMetricsPanel.css';

export interface RemediationMetrics {
    scoring_mode?: string;
    remediation_pass_rate?: number | null;
    patch_applies_rate?: number | null;
    code_parses_rate?: number | null;
    code_checkable_count?: number;
    /** Share of ALL scored rows whose finding is resolved (rows whose patch did not apply count as not resolved). */
    issue_resolved_overall_rate?: number | null;
    /** Over rows whose patch applied only (informational: the denominator depends on the model). */
    issue_resolved_rate?: number | null;
    new_findings_rate?: number | null;
    detector_evaluated_count?: number;
    /** Rows where the detector did not report the finding on the ORIGINAL code, so they are left out. */
    detector_blind_count?: number;
    detector_error?: string | null;
    citations_valid_rate?: number | null;
    citation_rate?: number | null;
    retrieval_found_rate?: number | null;
    json_validity_rate?: number | null;
    malformed_rate?: number | null;
    patch_reasons?: Record<string, number>;
    skipped_non_remediation_rows?: number;
    total?: number;
    correct?: number;
    /** Judge score (correct 1 / partial 0.5 / wrong 0 per row), when a judge ran. */
    judge_correct?: number | null;
    judge?: { judge_correct?: number | null; score?: number | null; n?: number; model?: string } | null;
    /** Extra declared output fields: field -> accuracy block (`metrics.extra_fields` from the backend). */
    extra_fields?: Record<string, { correct?: number; total?: number; accuracy?: number | null }>;
    extra_field_accuracy_mean?: number | null;
    /** Opt-in (`score_code_examples`): held-out task -> code rows. Absent otherwise. */
    code_example_count?: number;
    code_example_decided_count?: number;
    code_example_correct?: number;
    code_example_pass_rate?: number | null;
    code_example_output_parses_rate?: number | null;
    code_example_checkable_count?: number;
    code_example_not_checkable_count?: number;
    code_example_code_checks_pass_rate?: number | null;
    code_example_citations_valid_rate?: number | null;
    code_example_rows_with_passages?: number;
    code_example_judged_count?: number;
    code_example_judge_pass_rate?: number | null;
    code_example_judge_unscored_count?: number;
    code_example_judge_unscored_reasons?: Record<string, number>;
    code_example_judge_label?: string | null;
    code_example_by_language?: Record<string, { scored?: number; decided?: number; passed?: number; not_checkable?: number; pass_rate?: number | null }>;
}

export interface RemediationPassage {
    id: string;
    text?: string;
    source?: string | null;
    path?: string | null;
    start_line?: number | null;
    end_line?: number | null;
    symbol?: string | null;
    passage_id?: string | null;
}

export interface RemediationFinding {
    label?: string;
    start_line?: number;
    end_line?: number;
    evidence?: string | null;
}

/** `row_passage_headers` entry: where a passage id points (no text). */
export interface RemediationPassageHeader {
    id: string;
    where: string;
}

export interface RemediationRow {
    scoring_mode?: string | null;
    row_patch_applies?: boolean | null;
    row_patch_reason?: string | null;
    row_patch_offsets?: number[] | null;
    row_code_checkable?: boolean | null;
    row_code_parses?: boolean | null;
    row_check_messages?: string[] | null;
    row_issue_resolved?: boolean | null;
    row_new_findings?: string[] | null;
    row_detector?: string | null;
    row_citations_valid?: boolean | null;
    row_cited?: boolean | null;
    row_remediation_pass?: boolean | null;
    row_malformed?: boolean | null;
    row_finding?: RemediationFinding | null;
    row_passage_headers?: RemediationPassageHeader[] | null;
    row_passage_ids?: string[] | null;
    row_cited_ids?: string[] | null;
    row_detector_note?: string | null;
    parsed_prediction?: Record<string, unknown> | null;
    /** Opt-in code-example rows (`score_code_examples`). */
    row_kind?: string | null;
    row_code_example_output_parses?: boolean | null;
    row_code_example_checkable?: boolean | null;
    row_code_example_checks_pass?: boolean | null;
    row_code_example_check_messages?: string[] | null;
    row_code_example_citations_valid?: boolean | null;
    row_code_example_judge_pass?: boolean | null;
    row_code_example_judge_verdict?: string | null;
    row_code_example_judge_reason?: string | null;
    row_code_example_judge_unscored?: string | null;
    row_code_example_pass?: boolean | null;
    path?: string | null;
    language?: string | null;
    start_line?: number | null;
}

export function isRemediationMetrics(metrics: { scoring_mode?: string | null } | null | undefined): boolean {
    return metrics?.scoring_mode === 'remediation';
}

/** Rows scored by the remediation mode (or carrying its flags). */
export function isRemediationRow(row: { scoring_mode?: string | null; row_patch_applies?: unknown } | null | undefined): boolean {
    return row?.scoring_mode === 'remediation' || (row != null && row.row_patch_applies !== undefined && row.row_patch_applies !== null);
}

function pct(value: number | null | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

function range(f: { start_line?: number; end_line?: number }): string {
    if (typeof f.start_line !== 'number') return '';
    const end = f.end_line ?? f.start_line;
    return f.start_line === end ? `line ${f.start_line}` : `lines ${f.start_line}–${end}`;
}

/** Human wording for the applier's failure reasons. */
const REASON_TEXT: Record<string, string> = {
    empty_patch: 'empty patch',
    not_a_diff: 'not a diff',
    other_file: 'names a different file',
    multiple_files: 'touches several files',
    context_mismatch: 'context does not match the code',
    overlapping_hunks: 'hunks overlap',
    out_of_range: 'outside the shown code',
    no_op: 'changes nothing',
};

export function reasonText(reason: string): string {
    return REASON_TEXT[reason] ?? reason.replace(/_/g, ' ');
}

/** Why the judge left a code-example row unscored. */
const UNSCORED_TEXT: Record<string, string> = {
    judge_disabled: 'judge turned off for this eval',
    disabled_for_project: 'judge turned off in the remediation settings',
    no_judge_available: 'no judge model available',
    judge_no_verdict: 'judge gave no readable verdict',
    judge_error: 'judge failed',
    judge_not_run: 'judge did not run',
};

export function unscoredText(reason: string): string {
    return UNSCORED_TEXT[reason] ?? reason.replace(/_/g, ' ');
}

/** The "Task -> code rows" block (opt-in `score_code_examples`). */
export function CodeExampleMetricsSection({ metrics }: { metrics: RemediationMetrics }) {
    const scored = metrics.code_example_count ?? 0;
    if (scored <= 0) return null;
    const decided = metrics.code_example_decided_count ?? 0;
    const checkable = metrics.code_example_checkable_count ?? 0;
    const notCheckable = metrics.code_example_not_checkable_count ?? 0;
    const unscored = metrics.code_example_judge_unscored_count ?? 0;
    const reasons = Object.entries(metrics.code_example_judge_unscored_reasons ?? {}).sort((a, b) => b[1] - a[1]);
    const languages = Object.entries(metrics.code_example_by_language ?? {});
    const hasCitations = typeof metrics.code_example_citations_valid_rate === 'number';
    return (
        <div data-testid="ce-metrics">
            <h4 className="remediation__h4">Task → code rows</h4>
            <p className="u-hint u-mt0">
                A row passes when its output parses, the code passes the checks that pass on the reference code (where
                one applies), every citation is a given passage, and the judge
                {metrics.code_example_judge_label ? ` (${metrics.code_example_judge_label})` : ''} calls it correct.
                Rows the judge could not score are left out of the pass rate.
            </p>
            <dl className="remediation__grid">
                <div>
                    <dt>Pass rate</dt>
                    <dd data-testid="ce-pass">
                        {decided > 0 ? pct(metrics.code_example_pass_rate) : 'not evaluated'}
                        <span className="remediation__sub">{metrics.code_example_correct ?? 0} of {decided} scored rows ({scored} total)</span>
                    </dd>
                </div>
                <div>
                    <dt>Output parses</dt>
                    <dd data-testid="ce-parses">{pct(metrics.code_example_output_parses_rate)}</dd>
                </div>
                <div>
                    <dt>Code checks pass</dt>
                    <dd data-testid="ce-checks">
                        {checkable > 0 ? pct(metrics.code_example_code_checks_pass_rate) : 'not evaluated'}
                        <span className="remediation__sub">
                            {checkable} checkable, {notCheckable} not checkable
                        </span>
                    </dd>
                </div>
                <div>
                    <dt>Citations valid</dt>
                    <dd data-testid="ce-citations">
                        {hasCitations ? pct(metrics.code_example_citations_valid_rate) : 'not evaluated'}
                        <span className="remediation__sub">
                            {hasCitations ? `over ${metrics.code_example_rows_with_passages ?? 0} rows with passages` : 'no passages on these rows'}
                        </span>
                    </dd>
                </div>
                <div>
                    <dt>Judge pass</dt>
                    <dd data-testid="ce-judge">
                        {(metrics.code_example_judged_count ?? 0) > 0 ? pct(metrics.code_example_judge_pass_rate) : 'not run'}
                        <span className="remediation__sub">{metrics.code_example_judged_count ?? 0} judged, {unscored} unscored</span>
                    </dd>
                </div>
            </dl>
            {reasons.length > 0 && (
                <ul className="remediation-row__messages" data-testid="ce-unscored">
                    {reasons.map(([reason, n]) => (
                        <li key={reason} data-testid={`ce-unscored-${reason}`}>{n} unscored: {unscoredText(reason)}</li>
                    ))}
                </ul>
            )}
            {languages.length > 0 && (
                <table className="remediation__table" data-testid="ce-languages">
                    <thead><tr><th>Language</th><th>Rows</th><th>Scored</th><th>Passed</th><th>Not checkable</th><th>Pass rate</th></tr></thead>
                    <tbody>
                        {languages.map(([lang, b]) => (
                            <tr key={lang} data-testid={`ce-lang-${lang}`}>
                                <td>{lang}</td><td>{b.scored ?? 0}</td><td>{b.decided ?? 0}</td><td>{b.passed ?? 0}</td>
                                <td>{b.not_checkable ?? 0}</td><td>{pct(b.pass_rate)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}

function judgeScore(m: RemediationMetrics): number | null {
    const v = m.judge_correct ?? m.judge?.judge_correct ?? m.judge?.score;
    return typeof v === 'number' ? v : null;
}

interface PanelProps {
    metrics: RemediationMetrics;
    /** Where the numbers come from, e.g. "Run #21 · gold_test · result 1001". */
    provenance: string;
}

export default function RemediationMetricsPanel({ metrics, provenance }: PanelProps) {
    const checkable = metrics.code_checkable_count ?? 0;
    const detectorRows = metrics.detector_evaluated_count ?? 0;
    const blind = metrics.detector_blind_count ?? 0;
    const resolvedRate = metrics.issue_resolved_overall_rate ?? metrics.issue_resolved_rate;
    const detectorRan = detectorRows > 0 || typeof resolvedRate === 'number';
    const hasCitationRows = typeof metrics.citations_valid_rate === 'number' || typeof metrics.citation_rate === 'number';
    const reasons = Object.entries(metrics.patch_reasons ?? {}).sort((a, b) => b[1] - a[1]);
    const fields = Object.entries(metrics.extra_fields ?? {});
    const judge = judgeScore(metrics);
    const skipped = metrics.skipped_non_remediation_rows ?? 0;
    // Opt-in code-example scoring on a split without remediation rows: no remediation numbers to show.
    const onlyCodeExamples = (metrics.code_example_count ?? 0) > 0 && typeof metrics.remediation_pass_rate !== 'number';
    return (
        <section className="card remediation" data-testid="remediation-metrics">
            <h3 className="remediation__title">Remediation</h3>
            <p className="u-hint u-mt0">
                {provenance}. A row passes when its patch applies to the code, the patched code still parses (where a
                check could run), the finding is gone (where a detector ran) and every cited passage exists.
            </p>

            {!onlyCodeExamples && <dl className="remediation__grid">
                <div>
                    <dt>Pass rate</dt>
                    <dd data-testid="rem-pass">
                        {pct(metrics.remediation_pass_rate)}
                        <span className="remediation__sub">{metrics.correct ?? 0} of {metrics.total ?? 0} rows</span>
                    </dd>
                </div>
                <div>
                    <dt>Patch applies</dt>
                    <dd data-testid="rem-applies">{pct(metrics.patch_applies_rate)}</dd>
                </div>
                <div>
                    <dt>Patched code parses</dt>
                    <dd data-testid="rem-parses">
                        {checkable > 0 ? pct(metrics.code_parses_rate) : 'not evaluated'}
                        <span className="remediation__sub">
                            {checkable > 0
                                ? `over ${checkable} row${checkable === 1 ? '' : 's'} a code check could run on`
                                : 'no code check applies to these rows'}
                        </span>
                    </dd>
                </div>
                <div>
                    <dt>Finding resolved</dt>
                    <dd data-testid="rem-resolved">
                        {detectorRan ? pct(resolvedRate) : 'not evaluated'}
                        <span className="remediation__sub">
                            {detectorRan
                                ? `${detectorRows} row${detectorRows === 1 ? '' : 's'} checked by the detector${
                                    blind > 0 ? `, ${blind} left out (detector did not see the finding in the original)` : ''}`
                                : 'no detector configured'}
                        </span>
                    </dd>
                </div>
                {detectorRan && (
                    <div>
                        <dt>New findings introduced</dt>
                        <dd data-testid="rem-new-findings">{pct(metrics.new_findings_rate)}</dd>
                    </div>
                )}
                <div>
                    <dt>Citations valid</dt>
                    <dd data-testid="rem-citations">
                        {hasCitationRows ? pct(metrics.citations_valid_rate) : 'not evaluated'}
                        <span className="remediation__sub">
                            {hasCitationRows ? `${pct(metrics.citation_rate)} of rows cite a passage` : 'no passages on these rows'}
                        </span>
                    </dd>
                </div>
                {typeof metrics.retrieval_found_rate === 'number' && (
                    <div>
                        <dt>Passages contained the fix</dt>
                        <dd data-testid="rem-retrieval-found">{pct(metrics.retrieval_found_rate)}</dd>
                    </div>
                )}
                <div>
                    <dt>Malformed outputs</dt>
                    <dd data-testid="rem-malformed" className={(metrics.malformed_rate ?? 0) > 0 ? 'is-bad' : ''}>
                        {pct(metrics.malformed_rate)}
                        <span className="remediation__sub">JSON valid on {pct(metrics.json_validity_rate)}</span>
                    </dd>
                </div>
                <div>
                    <dt>Judge</dt>
                    <dd data-testid="rem-judge">
                        {judge != null ? judge.toFixed(2) : 'not run'}
                        {judge != null && <span className="remediation__sub">correct 1 / partial 0.5 / wrong 0 per row</span>}
                    </dd>
                </div>
            </dl>}

            <CodeExampleMetricsSection metrics={metrics} />

            {metrics.detector_error && (
                <p className="remediation-row__bad" data-testid="rem-detector-error">
                    Detector problem: {metrics.detector_error}. Rows it could not check are left out of the resolved rate.
                </p>
            )}
            {skipped > 0 && (
                <p className="u-hint" data-testid="rem-skipped">
                    {skipped} row{skipped === 1 ? '' : 's'} without a finding were left out (training-only rows).
                </p>
            )}

            {reasons.length > 0 && (
                <div data-testid="rem-reasons">
                    <h4 className="remediation__h4">Why patches did not apply</h4>
                    <table className="remediation__table">
                        <thead><tr><th>Reason</th><th>Rows</th></tr></thead>
                        <tbody>
                            {reasons.map(([reason, n]) => (
                                <tr key={reason} data-testid={`rem-reason-${reason}`}>
                                    <td>{reasonText(reason)}</td>
                                    <td>{n}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {fields.length > 0 && (
                <div data-testid="rem-fields">
                    <h4 className="remediation__h4">Extra output fields</h4>
                    <table className="remediation__table">
                        <thead><tr><th>Field</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr></thead>
                        <tbody>
                            {fields.map(([name, a]) => (
                                <tr key={name}><td><code>{name}</code></td><td>{a.correct ?? '—'}</td><td>{a.total ?? '—'}</td><td>{pct(a.accuracy)}</td></tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </section>
    );
}

function diffLineClass(line: string): string {
    if (line.startsWith('@@')) return 'is-hunk';
    if (line.startsWith('+++') || line.startsWith('---')) return 'is-hunk';
    if (line.startsWith('+')) return 'is-added';
    if (line.startsWith('-')) return 'is-removed';
    return '';
}

/** A unified diff with added / removed lines styled by class. */
export function DiffView({ patch }: { patch: string }) {
    const lines = patch.replace(/\r\n/g, '\n').split('\n');
    return (
        <pre className="remediation-row__diff" data-testid="rem-diff">
            {lines.map((line, i) => (
                <span key={i} className={`remediation-row__diff-line ${diffLineClass(line)}`}>{line || ' '}</span>
            ))}
        </pre>
    );
}

export function passageHeader(p: RemediationPassage): string {
    if (p.path) {
        const lines = typeof p.start_line === 'number'
            ? `:${p.start_line}${typeof p.end_line === 'number' && p.end_line !== p.start_line ? `–${p.end_line}` : ''}`
            : '';
        return `${p.path}${lines}${p.symbol ? ` (${p.symbol})` : ''}`;
    }
    return p.source || 'passage';
}

function Chip({ ok, testId, children }: { ok: boolean | null | undefined; testId: string; children: string }) {
    const cls = ok === true ? 'is-pass' : ok === false ? 'is-fail' : '';
    return <li className={`remediation-row__chip ${cls}`} data-testid={testId}>{children}</li>;
}

/** One scored task -> code row (opt-in `score_code_examples`). */
export function CodeExampleRowView({ row }: { row: RemediationRow }) {
    const pred = row.parsed_prediction ?? null;
    const code = typeof pred?.code === 'string' ? pred.code : '';
    const rationale = typeof pred?.rationale === 'string' ? pred.rationale : '';
    const cited = row.row_cited_ids ?? [];
    const messages = row.row_code_example_check_messages ?? [];
    const checkable = row.row_code_example_checkable === true;
    const verdict = row.row_code_example_judge_verdict;
    const unscored = row.row_code_example_judge_unscored;
    return (
        <div className="remediation-row" data-testid="code-example-row">
            {row.language && <p className="remediation-row__where" data-testid="code-example-row-where">task → code · {row.language}</p>}
            <ul className="remediation-row__chips">
                <Chip ok={row.row_code_example_pass} testId="ce-chip-pass">
                    {row.row_code_example_pass == null ? 'unscored' : row.row_code_example_pass ? 'passes' : 'fails'}
                </Chip>
                <Chip ok={row.row_code_example_output_parses} testId="ce-chip-parses">
                    {row.row_code_example_output_parses ? 'output parses' : 'output does not parse'}
                </Chip>
                <Chip ok={checkable ? row.row_code_example_checks_pass : null} testId="ce-chip-checks">
                    {!checkable ? 'not checkable' : row.row_code_example_checks_pass ? 'code checks pass' : 'code checks fail'}
                </Chip>
                <Chip ok={row.row_code_example_citations_valid} testId="ce-chip-citations">
                    {row.row_code_example_citations_valid == null
                        ? 'no passages'
                        : row.row_code_example_citations_valid ? 'citations valid' : 'no valid citation'}
                </Chip>
                <Chip ok={verdict ? row.row_code_example_judge_pass : null} testId="ce-chip-judge">
                    {verdict ? `judge: ${verdict}` : `judge unscored${unscored ? `: ${unscoredText(unscored)}` : ''}`}
                </Chip>
            </ul>
            {row.row_code_example_judge_reason && (
                <p className="u-hint u-mt0" data-testid="ce-judge-reason">{row.row_code_example_judge_reason}</p>
            )}
            {code && <pre className="remediation-row__diff" data-testid="ce-code">{code}</pre>}
            {rationale && <p className="u-hint" data-testid="ce-rationale">{rationale}</p>}
            {messages.length > 0 && (
                <ul className="remediation-row__messages" data-testid="ce-messages">
                    {messages.map((m, i) => <li key={i}>{m}</li>)}
                </ul>
            )}
            {cited.length > 0 && <p className="u-hint u-mt0" data-testid="ce-cited">Cited: {cited.join(', ')}</p>}
        </div>
    );
}

export function RemediationRowView({ row }: { row: RemediationRow }) {
    if (row.row_kind === 'code_example' && row.row_code_example_pass !== undefined) {
        return <CodeExampleRowView row={row} />;
    }
    const pred = row.parsed_prediction ?? null;
    const patch = typeof pred?.patch === 'string' ? pred.patch : '';
    const rationale = typeof pred?.rationale === 'string' ? pred.rationale : '';
    const cited = Array.isArray(pred?.citations)
        ? (pred?.citations as unknown[]).map(String)
        : (row.row_cited_ids ?? []);
    const passages = row.row_passage_headers ?? [];
    const messages = row.row_check_messages ?? [];
    const where = [
        row.path,
        row.language,
        typeof row.start_line === 'number' ? `starts at line ${row.start_line}` : null,
    ].filter(Boolean).join(' · ');
    const f = row.row_finding;
    const checkable = row.row_code_checkable === true;
    const resolvedEvaluated = row.row_issue_resolved !== null && row.row_issue_resolved !== undefined;
    const newFindings = row.row_new_findings ?? [];
    return (
        <div className="remediation-row" data-testid="remediation-row">
            {where && <p className="remediation-row__where" data-testid="remediation-row-where">{where}</p>}
            {f && (
                <p className="remediation-row__where" data-testid="remediation-row-finding">
                    Finding: <code>{f.label}</code> · {range(f)}{f.evidence ? ` · ${f.evidence}` : ''}
                </p>
            )}
            {row.row_malformed && (
                <p className="remediation-row__bad" data-testid="remediation-row-malformed">
                    Malformed output — no patch could be read.
                </p>
            )}

            <ul className="remediation-row__chips">
                <Chip ok={row.row_patch_applies} testId="rem-chip-applies">
                    {row.row_patch_applies ? 'patch applies' : `patch does not apply${row.row_patch_reason ? `: ${reasonText(row.row_patch_reason)}` : ''}`}
                </Chip>
                <Chip ok={checkable ? row.row_code_parses : null} testId="rem-chip-parses">
                    {!checkable ? 'not checkable' : row.row_code_parses ? 'patched code parses' : 'patched code does not parse'}
                </Chip>
                <Chip ok={resolvedEvaluated ? row.row_issue_resolved : null} testId="rem-chip-resolved">
                    {!resolvedEvaluated
                        ? 'resolution not evaluated'
                        : row.row_issue_resolved ? 'finding resolved' : 'finding still present'}
                </Chip>
                <Chip ok={row.row_citations_valid} testId="rem-chip-citations">
                    {row.row_citations_valid == null
                        ? 'no passages'
                        : row.row_citations_valid ? 'citations valid' : 'citation not in the passages'}
                </Chip>
            </ul>
            {row.row_detector && (
                <p className="u-hint u-mt0" data-testid="rem-detector">Detector: {row.row_detector}</p>
            )}
            {newFindings.length > 0 && (
                <p className="remediation-row__bad" data-testid="rem-new-findings-row">
                    New findings in the patched code: {newFindings.join(', ')}
                </p>
            )}

            {patch ? <DiffView patch={patch} /> : null}
            {rationale && <p className="u-hint" data-testid="rem-rationale">{rationale}</p>}

            {messages.length > 0 && (
                <ul className="remediation-row__messages" data-testid="rem-messages">
                    {messages.map((m, i) => <li key={i}>{m}</li>)}
                </ul>
            )}

            {passages.length > 0 && (
                <ul className="remediation-row__passages" data-testid="rem-passages">
                    {passages.map((p) => (
                        <li key={p.id} className={cited.includes(p.id) ? 'is-cited' : ''} data-testid={`rem-passage-${p.id}`}>
                            [{p.id}] {p.where}{cited.includes(p.id) ? ' (cited)' : ''}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
