/**
 * FieldMatchMetricsPanel — per-field results of a typed JSON-record eval
 * (the field_match scoring mode of the structured-extraction handler).
 *
 * Shows, per field: exact match, null precision / recall (the positive
 * class is "predicted null"), how many values were made up for a field the
 * answer key leaves null (hallucinated), and — when the project declared an
 * output schema — the out-of-vocabulary and type-violation rates. A dash
 * means the rate is undefined (no examples), not zero.
 *
 * Provenance: the header names the run and dataset the numbers come from.
 */

import './FieldMatchMetricsPanel.css';

export interface FieldMatchFieldMetrics {
    em?: number;
    f1?: number;
    support?: number;
    null_precision?: number | null;
    null_recall?: number | null;
    pred_null?: number;
    gold_null?: number;
    hallucinated?: number;
    abstained_wrongly?: number;
    out_of_vocab_rate?: number | null;
    type_violation_rate?: number | null;
}

export interface FieldMatchMetrics {
    scoring_mode?: string;
    per_field?: Record<string, FieldMatchFieldMetrics>;
    null_precision?: number | null;
    null_recall?: number | null;
    hallucinated?: number;
    abstained_wrongly?: number;
    schema_valid_rate?: number | null;
    json_validity_rate?: number | null;
    overall_em?: number | null;
}

interface FieldMatchMetricsPanelProps {
    metrics: FieldMatchMetrics;
    /** Where the numbers come from, e.g. "Run #21 · gold_dev (result 1001)". */
    provenance: string;
}

/** True when a result carries the typed field_match breakdown. */
export function hasFieldMatchBreakdown(metrics: FieldMatchMetrics | undefined | null): boolean {
    if (!metrics || !metrics.per_field || Object.keys(metrics.per_field).length === 0) return false;
    if (metrics.scoring_mode && metrics.scoring_mode !== 'field_match') return false;
    return 'null_precision' in metrics || Object.values(metrics.per_field).some((f) => 'null_precision' in f);
}

function pct(value: number | null | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

function count(value: number | undefined): string {
    return typeof value === 'number' ? String(value) : '—';
}

export default function FieldMatchMetricsPanel({ metrics, provenance }: FieldMatchMetricsPanelProps) {
    const fields = Object.entries(metrics.per_field ?? {});
    if (fields.length === 0) return null;

    return (
        <section className="card fm-metrics" data-testid="field-match-metrics">
            <header className="fm-metrics__head">
                <h4 className="fm-metrics__title">Field-by-field results</h4>
                <span className="u-meta" data-testid="field-match-provenance">{provenance}</span>
            </header>
            <dl className="fm-metrics__summary">
                <div>
                    <dt>Schema-valid rows</dt>
                    <dd data-testid="field-match-schema-valid">{pct(metrics.schema_valid_rate)}</dd>
                </div>
                <div>
                    <dt>Null precision</dt>
                    <dd data-testid="field-match-null-precision">{pct(metrics.null_precision)}</dd>
                </div>
                <div>
                    <dt>Null recall</dt>
                    <dd data-testid="field-match-null-recall">{pct(metrics.null_recall)}</dd>
                </div>
                <div>
                    <dt>Made-up values</dt>
                    <dd data-testid="field-match-hallucinated">{count(metrics.hallucinated)}</dd>
                </div>
                <div>
                    <dt>Wrongly left null</dt>
                    <dd data-testid="field-match-abstained">{count(metrics.abstained_wrongly)}</dd>
                </div>
            </dl>
            <div className="fm-metrics__scroll">
                <table className="fm-metrics__table">
                    <thead>
                        <tr>
                            <th scope="col">Field</th>
                            <th scope="col">Exact match</th>
                            <th scope="col">Null precision</th>
                            <th scope="col">Null recall</th>
                            <th scope="col">Made up</th>
                            <th scope="col">Out of vocabulary</th>
                            <th scope="col">Type violations</th>
                        </tr>
                    </thead>
                    <tbody>
                        {fields.map(([name, f]) => (
                            <tr key={name} data-testid={`field-match-row-${name}`}>
                                <th scope="row"><code>{name}</code></th>
                                <td>{pct(f.em)}</td>
                                <td data-testid={`field-match-${name}-null-precision`}>{pct(f.null_precision)}</td>
                                <td data-testid={`field-match-${name}-null-recall`}>{pct(f.null_recall)}</td>
                                <td data-testid={`field-match-${name}-hallucinated`}>{count(f.hallucinated)}</td>
                                <td data-testid={`field-match-${name}-oov`}>{pct(f.out_of_vocab_rate)}</td>
                                <td data-testid={`field-match-${name}-type`}>{pct(f.type_violation_rate)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <p className="u-meta u-m0">
                Null precision: of the fields the model left null, how many the answer key also leaves
                null. Null recall: of the fields the answer key leaves null, how many the model left
                null. A dash means no examples, not zero. Rows with invalid JSON are not counted here.
            </p>
        </section>
    );
}
