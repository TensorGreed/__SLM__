/**
 * One metric vocabulary for the UI (UI simplify phase 7k): names, scales and score-of-record keys, read
 * from `src/generated/metricVocabulary.json` (exported from `backend/app/services/metric_vocabulary.py` by
 * `python3 scripts/export_metric_vocabulary.py`; CI fails when it is stale).
 *
 * Scale rule: a `share` (share of rows that pass) shows as "85.7%" and differs in points ("+4.8 pts"); a
 * `mean01` (token F1 and the F1 variants, the judge score) shows as "0.184" and differs as "+0.048";
 * `unbounded` (perplexity, loss) is a plain number, lower is better.
 */

import vocabulary from '../../generated/metricVocabulary.json';

export type MetricScale = 'share' | 'mean01' | 'unbounded';

interface MetricEntry {
    label: string;
    short: string;
    scale: string;
    higher_is_better: boolean;
    shape_labels?: Record<string, string>;
    shape_scales?: Record<string, string>;
}

const METRICS = vocabulary.metrics as Record<string, MetricEntry>;

export const VOCABULARY_HEADLINE_METRICS: readonly string[] = vocabulary.preferred_headline_metrics;
export const VOCABULARY_TASK_SCORING_MODES: readonly string[] = vocabulary.task_scoring_modes;
export const VOCABULARY_SCORE_KEYS: Record<string, readonly string[]> = vocabulary.score_keys as Record<string, string[]>;
export const SCORED_AS_MODES: Record<string, string> = vocabulary.scored_as.modes as Record<string, string>;
export const SCORED_AS_SHAPES: Record<string, string> = vocabulary.scored_as.shapes as Record<string, string>;
export const SCORED_AS_EVAL_TYPES: Record<string, string> = vocabulary.scored_as.eval_types as Record<string, string>;

/** `share` / `mean01` / `unbounded` for a metric key on a result of `shape`. */
export function metricScale(key: string, shape?: string | null): MetricScale {
    const entry = METRICS[key];
    if (entry) return ((entry.shape_scales ?? {})[shape ?? ''] ?? entry.scale) as MetricScale;
    const token = key.toLowerCase();
    if (token.includes('loss') || token.includes('perplexity')) return 'unbounded';
    if (token.endsWith('_pass_rate')) return 'share';
    return 'mean01';
}

export function metricHigherIsBetter(key: string): boolean {
    const entry = METRICS[key];
    if (entry) return entry.higher_is_better;
    return metricScale(key) !== 'unbounded';
}

/** "Token F1", "Span F1 (micro)", "Plan pass rate"… for a key on a result of `shape`. */
export function metricLabel(key: string, shape?: string | null): string {
    const entry = METRICS[key];
    if (!entry) return key.replace(/_/g, ' ');
    return (entry.shape_labels ?? {})[shape ?? ''] ?? entry.label;
}

function lowerFirst(label: string): string {
    if (label.length > 1 && label[1] !== label[1].toLowerCase() && label[1] === label[1].toUpperCase()) return label;
    return label.charAt(0).toLowerCase() + label.slice(1);
}

/** The name inside a sentence. With a known shape: its label, first letter lower-cased ("span F1 (micro)",
 *  "token F1" on free text); with no shape: the short name ("F1", "plan pass rate", "judge score"). */
export function metricName(key: string, shape?: string | null): string {
    const entry = METRICS[key];
    if (!entry) return key.replace(/_/g, ' ');
    if (!shape) return entry.short || entry.label;
    return lowerFirst(metricLabel(key, shape));
}

function finite(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

/** "85.7%" (share), "0.184" (0–1 mean), "12.345" (unbounded); "—" when missing. */
export function formatMetricValue(key: string, value: number | null | undefined, shape?: string | null): string {
    if (!finite(value)) return '—';
    return metricScale(key, shape) === 'share' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
}

/** A difference in the score's own unit: "+4.8 pts" (share) or "+0.048". */
export function formatMetricDelta(key: string, delta: number | null | undefined, shape?: string | null): string {
    if (!finite(delta)) return '—';
    const sign = delta > 0 ? '+' : delta < 0 ? '-' : '+';
    if (metricScale(key, shape) === 'share') return `${sign}${Math.abs(delta * 100).toFixed(1)} pts`;
    return `${sign}${Math.abs(delta).toFixed(3)}`;
}
