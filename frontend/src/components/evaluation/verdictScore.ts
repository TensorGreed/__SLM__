/**
 * Score of record for one eval result (Results › Verdict, UI simplification pass phase 1).
 *
 * Read-only helpers over what an eval result already stores: which number is the result's score
 * for its scoring mode, how many rows it rests on, a Wilson 95% interval when the score is a
 * per-row pass / fail share, and whether only some of the evaluated file's rows were scored.
 * Nothing here changes how a result is scored; it only picks and labels what is stored.
 */

import { displayPassRate, isEvalSetName } from '../../api/evalSets';
import { scoredNullAsAbsent } from '../../api/nullScoring';

/** Task scoring modes: a result scored by one of these has a task-specific score of record. */
export const TASK_SCORING_MODES: readonly string[] = [
    'tool_call',
    'field_match',
    'span_set',
    'finding_set',
    'remediation',
    'structured_plan',
    'multi_option',
];

const MODE_LABEL: Record<string, string> = {
    tool_call: 'tool call',
    field_match: 'JSON fields',
    span_set: 'spans',
    finding_set: 'findings',
    remediation: 'remediation',
    structured_plan: 'structured plan',
    multi_option: 'multiple options',
};

const EVAL_TYPE_LABEL: Record<string, string> = {
    exact_match: 'exact match (string)',
    f1: 'token F1',
    llm_judge: 'judge',
    perplexity: 'perplexity',
};

export interface VerdictResultLike {
    id: number;
    dataset_name: string;
    eval_type: string;
    pass_rate: number | null;
    metrics?: Record<string, unknown> | null;
    details?: Record<string, unknown> | null;
}

export interface ScoreOfRecord {
    /** "tool call", "JSON fields", "exact match (string)". */
    scoredAs: string;
    /** The score's name, e.g. "Strict tool-call pass rate". */
    label: string;
    value: number | null;
    /** Passing rows and scored rows when the score is a per-row pass / fail share. */
    pass: number | null;
    scored: number | null;
    /** Wilson 95% interval; null when the score is not a per-row share. */
    ci: [number, number] | null;
    /** Another stored rate shown beside the score of record (the default tool-call pass rate next to strict). */
    other: { label: string; value: number } | null;
}

function num(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Wilson score interval (95%) for ``pass`` of ``n``; null when n is 0. */
export function wilsonInterval(pass: number, n: number, z = 1.96): [number, number] | null {
    if (!(n > 0)) return null;
    const p = pass / n;
    const z2 = z * z;
    const denom = 1 + z2 / n;
    const center = (p + z2 / (2 * n)) / denom;
    const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
    return [Math.max(0, center - half), Math.min(1, center + half)];
}

export function scoringModeOf(result: VerdictResultLike): string | null {
    const mode = result.metrics?.scoring_mode;
    return typeof mode === 'string' && mode ? mode : null;
}

export function hasTaskScoringMode(result: VerdictResultLike): boolean {
    const mode = scoringModeOf(result);
    return !!mode && TASK_SCORING_MODES.includes(mode);
}

/** "tool call" for a task scoring mode, else the eval type in plain words. */
export function scoredAsText(result: VerdictResultLike): string {
    const mode = scoringModeOf(result);
    if (mode) return MODE_LABEL[mode] ?? mode.replace(/_/g, ' ');
    return EVAL_TYPE_LABEL[result.eval_type] ?? result.eval_type.replace(/_/g, ' ');
}

const HEADLINE_NAMES: Array<[string, string]> = [
    ['exact_match', 'Exact match'],
    ['f1', 'Token F1'],
    ['judge_correct', 'Judge score (correct 1, partial 0.5)'],
    ['accuracy', 'Accuracy'],
    ['macro_f1', 'Macro F1'],
];

/** Names the stored pass rate by the metric it equals (a QA result's headline is token F1), else "Pass rate". */
export function headlineName(metrics: Record<string, unknown>, value: number | null): string {
    if (value === null) return 'Pass rate';
    for (const [key, name] of HEADLINE_NAMES) {
        const v = num(metrics[key]);
        if (v !== null && Math.abs(v - value) < 0.00005) return name;
    }
    return 'Pass rate';
}

/** The score a result stands on for its scoring mode, its row counts and interval. */
export function scoreOfRecord(result: VerdictResultLike): ScoreOfRecord {
    const metrics = result.metrics ?? {};
    const scoredAs = scoredAsText(result);
    const total = num(metrics.total);
    const mode = scoringModeOf(result);
    const overall = (metrics.slice_breakdown as { overall?: { scored?: number; pass?: number; pass_ci?: [number, number] | null; strict?: { scored?: number; pass?: number; ci?: [number, number] | null } | null } | null } | undefined)?.overall ?? null;

    if (mode === 'tool_call') {
        const strict = num(metrics.tool_call_strict_pass_rate);
        const plain = num(metrics.tool_call_pass_rate);
        if (strict !== null) {
            // The strict count is not stored on its own; the rate is n / total rounded to 4 places.
            const pass = total !== null ? Math.round(strict * total) : null;
            const ci = overall?.strict?.ci ?? (pass !== null && total !== null ? wilsonInterval(pass, total) : null);
            return {
                scoredAs: `${scoredAs} (strict)`,
                label: 'Strict tool-call pass rate (extra arguments count as wrong)',
                value: strict,
                pass,
                scored: total,
                ci,
                other: plain !== null ? { label: 'default pass rate (ignores arguments the answer key does not have)', value: plain } : null,
            };
        }
        const pass = num(metrics.correct);
        return {
            scoredAs,
            label: 'Tool-call pass rate',
            value: plain ?? displayPassRate(result),
            pass,
            scored: total,
            ci: overall?.pass_ci ?? (pass !== null && total !== null ? wilsonInterval(pass, total) : null),
            other: null,
        };
    }

    const value = displayPassRate(result);
    const correct = num(metrics.correct);
    // An interval only when the score IS the share of passing rows (not an F1 or a mean judge score).
    const perRow = value !== null && correct !== null && total !== null && total > 0 && Math.abs(correct / total - value) < 0.0006;
    return {
        scoredAs,
        label: headlineName(metrics, value),
        value,
        pass: perRow ? correct : null,
        scored: total,
        ci: perRow ? (overall?.pass_ci ?? wilsonInterval(correct as number, total as number)) : null,
        other: null,
    };
}

/** Rows in the evaluated file: the eval set's rows, else the split's ``rows_available``. */
export function rowsInSet(result: VerdictResultLike): number | null {
    const dataset = (result.details?.dataset ?? null) as Record<string, unknown> | null;
    const block = (dataset?.eval_set ?? null) as Record<string, unknown> | null;
    return num(block?.rows) ?? num(dataset?.rows_available);
}

/** "Partial: 100 of 600 rows scored" when the eval was capped, else null. */
export function partialRowsText(result: VerdictResultLike): string | null {
    const scored = num(result.metrics?.total);
    const rows = rowsInSet(result);
    if (scored === null || rows === null || rows <= scored) return null;
    return `Partial: ${scored.toLocaleString('en-US')} of ${rows.toLocaleString('en-US')} rows scored (row limit). This score covers those rows only.`;
}

/** Rows a judge left without a verdict, which the pass rate leaves out (0 / absent → null). */
export function unscoredRowsText(result: VerdictResultLike): string | null {
    const excluded = num(result.metrics?.pass_rate_unjudged_excluded);
    if (!excluded) return null;
    return `${excluded} row${excluded === 1 ? '' : 's'} left without a score (unjudged) ${excluded === 1 ? 'is' : 'are'} not in this rate.`;
}

export function pct(value: number | null | undefined): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
}

/** "87.0% (95% CI 81.6%–91.0%)" or "87.0%". */
export function scoreText(score: ScoreOfRecord): string {
    const interval = score.ci ? ` (95% CI ${pct(score.ci[0])}–${pct(score.ci[1])})` : '';
    return `${pct(score.value)}${interval}`;
}

/** "How it's scored" line: mode, strict arguments, null as absent, prompt rendering, decoding. */
export function howScoredText(result: VerdictResultLike): string {
    const metrics = result.metrics ?? {};
    const parts = [`scored as ${scoredAsText(result)}`];
    if (scoringModeOf(result) === 'tool_call') {
        parts.push(num(metrics.tool_call_strict_pass_rate) !== null ? 'strict arguments on' : 'strict arguments off');
    }
    if (scoredNullAsAbsent(metrics as Parameters<typeof scoredNullAsAbsent>[0])) parts.push('null counts as absent');
    parts.push(`prompt rendering v${num(metrics.prompt_format_version) ?? 1}`);
    const inference = (result.details?.inference ?? null) as Record<string, unknown> | null;
    const tokens = num(inference?.max_new_tokens);
    const temperature = num(inference?.temperature);
    if (tokens !== null) parts.push(`max ${tokens} new tokens${temperature !== null ? `, temperature ${temperature}` : ''}`);
    return parts.join(' · ');
}

/** Newest result per dataset label (results arrive newest first). */
export function newestPerDataset<T extends { dataset_name: string }>(results: T[]): T[] {
    const seen = new Set<string>();
    const out: T[] = [];
    for (const r of results) {
        if (seen.has(r.dataset_name)) continue;
        seen.add(r.dataset_name);
        out.push(r);
    }
    return out;
}

/** True for a result scored by the base model through a model path (filed under the run). */
export function isBaseModelResult(result: VerdictResultLike): boolean {
    const record = result.metrics?.evaluated_model as { kind?: string } | null | undefined;
    return record?.kind === 'base';
}

/** Lift keys for a run's results: '' for the built-in sets (one call), each eval-set label otherwise. */
export function liftKeys(results: VerdictResultLike[]): string[] {
    const keys: string[] = [];
    for (const r of newestPerDataset(results.filter((x) => !isBaseModelResult(x)))) {
        const key = isEvalSetName(r.dataset_name) ? r.dataset_name : '';
        if (!keys.includes(key)) keys.push(key);
    }
    return keys;
}

/**
 * O2: a plain string-match result shown next to a task-scored one. When the list holds any result
 * with a task scoring mode, results without a scoring mode scored as exact match are "other stored
 * metrics (string match)": kept and one click away, not next to the score of record.
 */
export function isStringMatchAside(result: VerdictResultLike, all: VerdictResultLike[]): boolean {
    if (scoringModeOf(result) || result.eval_type !== 'exact_match') return false;
    return all.some((r) => hasTaskScoringMode(r));
}
