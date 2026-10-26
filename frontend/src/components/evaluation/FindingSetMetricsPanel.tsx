/**
 * FindingSetMetricsPanel — results of a findings eval (the finding_set
 * scoring mode of the structured-extraction handler: a model lists labelled
 * line ranges, scored against the answer key's ranges).
 *
 * A finding counts as found only when its label is the same AND its line
 * range overlaps the answer key's range by at least the IoU threshold
 * (strict). Clean chunks (the answer key lists nothing) get their own
 * false-positive rate, and output that could not be parsed is counted as
 * malformed — separately from findings the model missed.
 *
 * Also exports FindingSetRowView, the per-row matched / missed / spurious
 * list with the row's provenance (path, language, start line).
 */

import './FindingSetMetricsPanel.css';

export interface FindingSetPerLabel {
    precision: number;
    recall: number;
    f1: number;
    support: number;
    tp?: number;
    fp?: number;
    fn?: number;
}

export interface FindingSetMetrics {
    scoring_mode?: string;
    precision?: number;
    recall?: number;
    f1?: number;
    precision_macro?: number;
    recall_macro?: number;
    f1_macro?: number;
    per_class?: Record<string, FindingSetPerLabel>;
    tp_total?: number;
    fp_total?: number;
    fn_total?: number;
    total?: number;
    clean_chunk_count?: number;
    clean_chunk_fp_count?: number;
    clean_chunk_fp_rate?: number | null;
    malformed_count?: number;
    malformed_rate?: number | null;
    invalid_finding_count?: number;
    json_validity_rate?: number | null;
    exact_match?: number | null;
    attr_accuracy?: Record<string, { correct: number; total: number; accuracy: number | null }>;
    attr_accuracy_mean?: number | null;
    iou_threshold?: number;
    hierarchical?: {
        precision?: number;
        recall?: number;
        f1?: number;
        partial_matches?: number;
        parent_credit?: number;
    };
}

export interface FindingRange {
    label: string;
    start_line: number;
    end_line: number;
    pred_start_line?: number;
    pred_end_line?: number;
    iou?: number;
}

export interface FindingSetRow {
    scoring_mode?: string | null;
    row_matched_findings?: FindingRange[] | null;
    row_missed_findings?: FindingRange[] | null;
    row_spurious_findings?: FindingRange[] | null;
    row_malformed?: boolean | null;
    row_is_clean_reference?: boolean | null;
    row_precision?: number | null;
    row_recall?: number | null;
    path?: string | null;
    language?: string | null;
    start_line?: number | null;
}

export function isFindingSetMetrics(metrics: { scoring_mode?: string | null } | null | undefined): boolean {
    return metrics?.scoring_mode === 'finding_set';
}

function pct(value: number | null | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

function range(f: { start_line: number; end_line: number }): string {
    return f.start_line === f.end_line ? `line ${f.start_line}` : `lines ${f.start_line}–${f.end_line}`;
}

interface PanelProps {
    metrics: FindingSetMetrics;
    /** Where the numbers come from, e.g. "Run #21 · gold_test · result 1001". */
    provenance: string;
}

export default function FindingSetMetricsPanel({ metrics, provenance }: PanelProps) {
    const perLabel = Object.entries(metrics.per_class ?? {}).sort((a, b) => a[1].f1 - b[1].f1 || b[1].support - a[1].support);
    const attrs = Object.entries(metrics.attr_accuracy ?? {});
    const h = metrics.hierarchical;
    const malformed = metrics.malformed_count ?? 0;
    const invalid = metrics.invalid_finding_count ?? 0;
    return (
        <section className="card finding-set" data-testid="finding-set-metrics">
            <h3 className="finding-set__title">Findings</h3>
            <p className="u-hint u-mt0">
                {provenance}. A finding is found only when its label is the same and its lines overlap the answer key&apos;s
                lines by at least {metrics.iou_threshold ?? 0.5} (strict).
            </p>

            <dl className="finding-set__grid">
                <div><dt>Precision</dt><dd data-testid="fs-precision">{pct(metrics.precision)}</dd></div>
                <div><dt>Recall</dt><dd data-testid="fs-recall">{pct(metrics.recall)}</dd></div>
                <div><dt>F1</dt><dd data-testid="fs-f1">{pct(metrics.f1)}</dd></div>
                <div><dt>Macro F1</dt><dd data-testid="fs-f1-macro">{pct(metrics.f1_macro)}</dd></div>
                <div>
                    <dt>Clean chunks flagged</dt>
                    <dd data-testid="fs-clean-fp">
                        {pct(metrics.clean_chunk_fp_rate)}
                        <span className="finding-set__sub">
                            {metrics.clean_chunk_fp_count ?? 0} of {metrics.clean_chunk_count ?? 0} clean chunks
                        </span>
                    </dd>
                </div>
                <div>
                    <dt>Malformed outputs</dt>
                    <dd data-testid="fs-malformed" className={malformed > 0 ? 'is-bad' : ''}>
                        {malformed}
                        <span className="finding-set__sub">{pct(metrics.malformed_rate)} of rows (counted apart from missed findings)</span>
                    </dd>
                </div>
                <div>
                    <dt>Invalid findings dropped</dt>
                    <dd data-testid="fs-invalid" className={invalid > 0 ? 'is-bad' : ''}>{invalid}</dd>
                </div>
                <div>
                    <dt>Found / spurious / missed</dt>
                    <dd data-testid="fs-counts">
                        {metrics.tp_total ?? 0} / {metrics.fp_total ?? 0} / {metrics.fn_total ?? 0}
                    </dd>
                </div>
            </dl>

            {attrs.length > 0 && (
                <div data-testid="fs-attrs">
                    <h4 className="finding-set__h4">Attribute accuracy</h4>
                    <p className="u-hint u-mt0">Over findings that were found; mean {pct(metrics.attr_accuracy_mean)}.</p>
                    <table className="finding-set__table">
                        <thead><tr><th>Attribute</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr></thead>
                        <tbody>
                            {attrs.map(([name, a]) => (
                                <tr key={name}><td><code>{name}</code></td><td>{a.correct}</td><td>{a.total}</td><td>{pct(a.accuracy)}</td></tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {h && (
                <div data-testid="fs-hierarchical">
                    <h4 className="finding-set__h4">Parent-label credit</h4>
                    <p className="u-hint u-mt0">
                        A prediction naming a parent of the right label on the right lines earns {h.parent_credit ?? 0.5} credit.
                        Strict numbers above are unchanged.
                    </p>
                    <p className="finding-set__line">
                        P {pct(h.precision)} · R {pct(h.recall)} · F1 {pct(h.f1)} · {h.partial_matches ?? 0} partial match{h.partial_matches === 1 ? '' : 'es'}
                    </p>
                </div>
            )}

            {perLabel.length > 0 && (
                <div data-testid="fs-per-label">
                    <h4 className="finding-set__h4">Per label (weakest first)</h4>
                    <table className="finding-set__table">
                        <thead>
                            <tr><th>Label</th><th>Support</th><th>Found</th><th>Spurious</th><th>Missed</th><th>P</th><th>R</th><th>F1</th></tr>
                        </thead>
                        <tbody>
                            {perLabel.map(([label, m]) => (
                                <tr key={label} data-testid={`fs-label-${label}`}>
                                    <td><code>{label}</code></td>
                                    <td>{m.support}</td>
                                    <td>{m.tp ?? '—'}</td>
                                    <td>{m.fp ?? '—'}</td>
                                    <td>{m.fn ?? '—'}</td>
                                    <td>{pct(m.precision)}</td>
                                    <td>{pct(m.recall)}</td>
                                    <td>{pct(m.f1)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </section>
    );
}

export function FindingSetRowView({ row }: { row: FindingSetRow }) {
    const matched = row.row_matched_findings ?? [];
    const missed = row.row_missed_findings ?? [];
    const spurious = row.row_spurious_findings ?? [];
    const where = [
        row.path,
        row.language,
        typeof row.start_line === 'number' ? `starts at line ${row.start_line}` : null,
    ].filter(Boolean).join(' · ');
    return (
        <div className="finding-set-row" data-testid="finding-set-row">
            {where && <p className="finding-set-row__where" data-testid="finding-set-row-where">{where}</p>}
            {row.row_malformed && (
                <p className="finding-set-row__bad" data-testid="finding-set-row-malformed">
                    Malformed output — no findings could be read, so every answer-key finding counts as missed.
                </p>
            )}
            {row.row_is_clean_reference && spurious.length === 0 && !row.row_malformed && (
                <p className="u-hint">Clean chunk: the answer key lists no findings, and none were reported.</p>
            )}
            {row.row_is_clean_reference && spurious.length > 0 && (
                <p className="finding-set-row__bad">Clean chunk flagged: the answer key lists no findings.</p>
            )}
            <ul className="finding-set-row__list">
                {matched.map((f, i) => (
                    <li key={`m-${i}`} className="is-pass" data-testid="finding-matched">
                        <span className="finding-set-row__tag">matched</span>
                        <code>{f.label}</code> · {range(f)}
                        {typeof f.pred_start_line === 'number' && (
                            <span className="finding-set-row__sub">
                                {' '}predicted {range({ start_line: f.pred_start_line, end_line: f.pred_end_line ?? f.pred_start_line })}
                                {typeof f.iou === 'number' && ` · IoU ${f.iou.toFixed(2)}`}
                            </span>
                        )}
                    </li>
                ))}
                {missed.map((f, i) => (
                    <li key={`x-${i}`} className="is-fail" data-testid="finding-missed">
                        <span className="finding-set-row__tag">missed</span>
                        <code>{f.label}</code> · {range(f)}
                    </li>
                ))}
                {spurious.map((f, i) => (
                    <li key={`s-${i}`} className="is-fail" data-testid="finding-spurious">
                        <span className="finding-set-row__tag">spurious</span>
                        <code>{f.label}</code> · {range(f)}
                    </li>
                ))}
            </ul>
        </div>
    );
}
