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
import {
    metricLabel,
    metricScale,
    SCORED_AS_EVAL_TYPES,
    SCORED_AS_MODES,
    SCORED_AS_SHAPES,
    VOCABULARY_HEADLINE_METRICS,
    VOCABULARY_SCORE_KEYS,
    VOCABULARY_TASK_SCORING_MODES,
    type MetricScale,
} from './metricVocabulary';

/** Task scoring modes: a result scored by one of these has a task-specific score of record. */
export const TASK_SCORING_MODES: readonly string[] = VOCABULARY_TASK_SCORING_MODES;

/** Modes whose ``exact_match`` is the mode's own pass rate under its legacy name (a duplicate row). */
export const EXACT_MATCH_ALIAS_MODES: readonly string[] = ['tool_call', 'structured_plan', 'multi_option', 'remediation'];

const MODE_LABEL: Record<string, string> = SCORED_AS_MODES;

/** "tool call", "JSON fields", "structured plan"… for a scoring mode (its id in words when unknown). */
export function scoringModeLabel(mode: string): string {
    return MODE_LABEL[mode] ?? mode.replace(/_/g, ' ');
}

const EVAL_TYPE_LABEL: Record<string, string> = SCORED_AS_EVAL_TYPES;

export interface VerdictResultLike {
    id: number;
    dataset_name: string;
    eval_type: string;
    pass_rate: number | null;
    metrics?: Record<string, unknown> | null;
    details?: Record<string, unknown> | null;
}

export interface ScoreOfRecord {
    /** The stored metric the score is (``tool_call_strict_pass_rate``, ``f1``…); null when none is stored. */
    key: string | null;
    /** ``resultShape``: the scoring mode, or classification / safety / judge / perplexity / text. */
    shape: string;
    /** "tool call", "JSON fields", "free text vs answer key (exact match + token F1)". */
    scoredAs: string;
    /** The score's name, e.g. "Strict tool-call pass rate". */
    label: string;
    value: number | null;
    /** False for a value that is not a share (token F1, the judge score, perplexity): shown as 0–1 / a plain number. */
    percent: boolean;
    /** The metric's scale (phase 7k): share (%), mean01 (0–1), unbounded (perplexity, loss); null with no key. */
    scale: MetricScale | null;
    /** The score is the share of rows that pass (``isShareOfRowsKey``). */
    share: boolean;
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

/** A stored metric's value: the number, or the mean of a seed-group aggregate's ``{mean, std, min, max, n}``
 *  block (phase 7k: a seed-group leader's score is the mean over its seeds). */
export function metricValue(metrics: Record<string, unknown> | null | undefined, key: string): number | null {
    const raw = (metrics ?? {})[key];
    const direct = num(raw);
    if (direct !== null) return direct;
    if (raw && typeof raw === 'object' && !Array.isArray(raw)) return num((raw as { mean?: unknown }).mean);
    return null;
}

/** A seed-group leader's seed run ids (phase 7k): ``config._seed_group.child_experiment_ids`` once the group
 *  finished, else ``[]`` for a leader still training — ``_seed_group.role === 'leader'`` / ``planned_seeds`` (written
 *  at the fan-out), an explicit ``seeds: [..]`` list of more than one, or ``num_seeds`` > 1; null for any other run
 *  (a seed child has ``role: 'child'``). */
export function seedGroupRuns(run: { config?: Record<string, unknown> | null } | null | undefined): number[] | null {
    const config = run?.config ?? null;
    if (!config) return null;
    const group = config._seed_group as { child_experiment_ids?: unknown; role?: unknown; planned_seeds?: unknown } | null | undefined;
    if (group && typeof group === 'object') {
        if (group.role === 'child') return null;
        if (Array.isArray(group.child_experiment_ids)) {
            return group.child_experiment_ids.map((id) => num(id)).filter((id): id is number => id !== null);
        }
        if (group.role === 'leader' || (Array.isArray(group.planned_seeds) && group.planned_seeds.length > 1)) return [];
    }
    if (Array.isArray(config.seeds) && config.seeds.length > 1) return [];
    const n = num(config.num_seeds);
    return n !== null && n > 1 ? [] : null;
}

/** "Mean of 3 seeds (#5, #6, #7)" for a seed-group aggregate result (``details.per_seed``), else null. */
export function seedMeanText(result: VerdictResultLike): string | null {
    const perSeed = (result.details as { per_seed?: unknown } | null | undefined)?.per_seed;
    if (!Array.isArray(perSeed) || perSeed.length === 0) return null;
    const ids = perSeed
        .map((s) => num((s as { experiment_id?: unknown } | null)?.experiment_id))
        .filter((id): id is number => id !== null);
    const n = perSeed.length;
    return `Mean of ${n} seed${n === 1 ? '' : 's'}${ids.length ? ` (${ids.map((id) => `#${id}`).join(', ')})` : ''}`;
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

const SHAPE_LABEL: Record<string, string> = SCORED_AS_SHAPES;

/** "tool call" for a task scoring mode, "classification (label match)" / "safety checks" for those
 *  handlers, else the eval type in plain words. */
export function scoredAsText(result: VerdictResultLike): string {
    const mode = scoringModeOf(result);
    if (mode) return scoringModeLabel(mode);
    const shape = resultShape(result);
    if (SHAPE_LABEL[shape]) return SHAPE_LABEL[shape];
    return EVAL_TYPE_LABEL[result.eval_type] ?? result.eval_type.replace(/_/g, ' ');
}

/**
 * Mirrors the backend's ``_PREFERRED_HEADLINE_METRICS`` (``sft_lift_summary_service.py``): the lift headline
 * is the first of these that both results carry. The score of record follows the same order, so the
 * Verdict card's score and its base vs fine-tuned line name the same metric.
 */
export const PREFERRED_HEADLINE_METRICS: readonly string[] = VOCABULARY_HEADLINE_METRICS;

/**
 * Score-of-record keys per result shape, in the backend's headline order (a subsequence of
 * ``PREFERRED_HEADLINE_METRICS``). One exception: a tool-call result with strict argument matching on stands
 * on ``tool_call_strict_pass_rate`` (decision: strict is the score of record when stored); its lift line uses
 * the strict lift row. The first key the result stores wins; a shape's keys never depend on the value, so
 * the base model and the fine-tuned run on one set show the same metric.
 */
export const SCORE_KEYS: Record<string, readonly string[]> = VOCABULARY_SCORE_KEYS;

/**
 * What kind of result this is, for picking and naming its score: the stored scoring mode, else
 * ``classification`` (accuracy + macro F1 stored; ``exact_match`` / ``f1`` are their aliases), ``safety``
 * (per-test-type pass rates; ``exact_match`` / ``f1`` alias the pass rate), ``judge`` (an LLM-judge eval),
 * ``perplexity``, or ``text`` (QA and other free text: exact match / token F1).
 */
export function resultShape(result: VerdictResultLike): string {
    const mode = scoringModeOf(result);
    if (mode) return mode;
    const metrics = result.metrics ?? {};
    if (result.eval_type === 'perplexity') return 'perplexity';
    if (result.eval_type === 'llm_judge') return 'judge';
    // metricValue reads a seed-group aggregate's {mean, std, …} block too (review fix: an aggregate classification
    // result fell through to 'text' and read "Token F1").
    if (metricValue(metrics, 'accuracy') !== null && metricValue(metrics, 'macro_f1') !== null) return 'classification';
    if (metrics.per_test_type && typeof metrics.per_test_type === 'object') return 'safety';
    return 'text';
}

/** The name of a stored metric for a result of this shape (``f1`` / ``exact_match`` mean different things per
 *  mode); from the shared metric vocabulary (phase 7k). */
export function scoreKeyLabel(key: string, shape: string): string {
    return metricLabel(key, shape);
}

/** True when the metric is the share of rows that pass (so a Wilson interval applies); F1, judge means and
 *  perplexity are not. */
export function isShareOfRowsKey(key: string, shape: string): boolean {
    return metricScale(key, shape) === 'share';
}

/** The key a result's score of record stands on (null when it stores none of its shape's keys). */
export function scoreKeyOf(result: VerdictResultLike): string | null {
    const metrics = result.metrics ?? {};
    const shape = resultShape(result);
    const keys = SCORE_KEYS[shape] ?? SCORE_KEYS.text;
    for (const key of [...keys, ...PREFERRED_HEADLINE_METRICS]) {
        if (metricValue(metrics, key) !== null) return key;
    }
    return null;
}

function close(a: number, b: number): boolean {
    return Math.abs(a - b) < 0.0006;
}

/** Passing rows and the rows the rate is over, when they can be told for this key. */
function shareCounts(key: string, metrics: Record<string, unknown>, value: number): { pass: number; scored: number } | null {
    if (key === 'exec_match_rate') {
        const block = (metrics.sql_execution ?? null) as Record<string, unknown> | null;
        const passed = num(block?.n_passed);
        const scored = num(block?.n_scored);
        return passed !== null && scored !== null && scored > 0 ? { pass: passed, scored } : null;
    }
    const total = num(metrics.total) ?? num(metrics.total_evaluated) ?? num(metrics.total_tests);
    if (total === null || total <= 0) return null;
    const correct = num(metrics.correct) ?? num(metrics.passed);
    if (correct !== null && close(correct / total, value)) return { pass: correct, scored: total };
    // Tool-call rates are n / total rounded to 4 places (the strict count is not stored on its own).
    if (key === 'tool_call_strict_pass_rate' || key === 'tool_call_pass_rate') {
        return { pass: Math.round(value * total), scored: total };
    }
    return null;
}

/** A result whose score of record is the judge score: free text graded by a judge model, not exact match + token F1. */
function judgeScoredAs(shape: string, scoredAsBase: string): string {
    return shape === 'text' ? 'free text vs answer key, graded by a judge model' : `${scoredAsBase}, graded by a judge model`;
}

/** The score a result stands on for its scoring mode, its row counts and interval. */
export function scoreOfRecord(result: VerdictResultLike): ScoreOfRecord {
    const metrics = result.metrics ?? {};
    const shape = resultShape(result);
    const key = scoreKeyOf(result);
    const scoredAsBase = scoredAsText(result);
    const total = num(metrics.total) ?? num(metrics.total_evaluated) ?? num(metrics.total_tests);
    if (key === null) {
        return {
            key: null, shape, scoredAs: scoredAsBase, label: 'Pass rate', value: displayPassRate(result),
            percent: true, scale: null, share: false, pass: null, scored: total, ci: null, other: null,
        };
    }
    const value = metricValue(metrics, key) as number;
    const share = isShareOfRowsKey(key, shape);
    const counts = share ? shareCounts(key, metrics, value) : null;
    const overall = (metrics.slice_breakdown as { overall?: { scored?: number; pass?: number; pass_ci?: [number, number] | null; strict?: { scored?: number; pass?: number; ci?: [number, number] | null } | null } | null } | undefined)?.overall ?? null;
    let ci: [number, number] | null = null;
    if (counts) {
        // The slice breakdown's own interval wins when it is over the same rows and passes.
        const block = key === 'tool_call_strict_pass_rate'
            ? (overall?.strict ? { pass: overall.strict.pass, scored: overall.strict.scored, ci: overall.strict.ci } : null)
            : (overall ? { pass: overall.pass, scored: overall.scored, ci: overall.pass_ci } : null);
        const sameRows = block?.ci && typeof block.pass === 'number' && typeof block.scored === 'number' && block.scored > 0
            && close(block.pass / block.scored, value);
        ci = sameRows ? (block!.ci as [number, number]) : wilsonInterval(counts.pass, counts.scored);
    }
    const plain = metricValue(metrics, 'tool_call_pass_rate');
    return {
        key,
        shape,
        scoredAs: key === 'tool_call_strict_pass_rate' ? `${scoredAsBase} (strict)` : key === 'judge_correct' ? judgeScoredAs(shape, scoredAsBase) : scoredAsBase,
        label: scoreKeyLabel(key, shape),
        value,
        // Only a share of passing rows is a percentage; token F1 / F1 variants and the judge score (0–1 means) show
        // as 0.184, perplexity as a plain number (phase 7k scale rule).
        percent: share,
        scale: metricScale(key, shape),
        share,
        pass: counts?.pass ?? null,
        scored: counts?.scored ?? total,
        ci,
        other: key === 'tool_call_strict_pass_rate' && plain !== null
            ? { label: 'default pass rate (ignores arguments the answer key does not have)', value: plain }
            : null,
    };
}

/** Rows in the evaluated file: the eval set's rows, else the split's ``rows_available``. */
export function rowsInSet(result: VerdictResultLike): number | null {
    const dataset = (result.details?.dataset ?? null) as Record<string, unknown> | null;
    const block = (dataset?.eval_set ?? null) as Record<string, unknown> | null;
    return num(block?.rows) ?? num(dataset?.rows_available);
}

/** Rows the model answered: ``details.inference.samples`` (``metrics.evaluated_samples`` on older results). */
export function rowsAnswered(result: VerdictResultLike): number | null {
    const inference = (result.details?.inference ?? null) as Record<string, unknown> | null;
    return num(inference?.samples) ?? num(result.metrics?.evaluated_samples);
}

/**
 * "Partial: 100 of 600 rows scored (row limit 100)" when the eval answered fewer rows than the evaluated file
 * holds (a row limit), else null. Read from the rows answered, not the scorer's count, so rows a scorer
 * leaves out (``excludedRowsText``) never read as a row limit.
 */
export function partialRowsText(result: VerdictResultLike): string | null {
    const answered = rowsAnswered(result);
    const rows = rowsInSet(result);
    if (answered === null || rows === null || rows <= answered) return null;
    const dataset = (result.details?.dataset ?? null) as Record<string, unknown> | null;
    const cap = num(dataset?.max_samples);
    const why = cap !== null && answered >= cap ? ` (row limit ${cap.toLocaleString('en-US')})` : '';
    return `Partial: ${answered.toLocaleString('en-US')} of ${rows.toLocaleString('en-US')} rows scored${why}. This score covers those rows only.`;
}

const EXCLUDED_REASON: Record<string, string> = {
    multi_option: 'the option verifier could not check them, or they are not multiple-option rows',
    remediation: 'no finding to fix, or the detector could not check them',
    structured_plan: 'the scorer could not score them',
    exec_match_rate: 'the answer-key query could not be run',
};

/**
 * Rows the model answered that the scorer left out of the rate (verifier errors, rows the mode does not
 * score, answer-key queries that fail), e.g. "3 of 40 answered rows are not in this rate: the option
 * verifier could not check them". Only for a share-of-rows score whose row count is known; null otherwise.
 * Unjudged rows have their own line (``unscoredRowsText``).
 */
export function excludedRowsText(result: VerdictResultLike, score: ScoreOfRecord = scoreOfRecord(result)): string | null {
    const answered = rowsAnswered(result);
    if (answered === null || score.pass === null || score.scored === null || score.scored >= answered) return null;
    const left = answered - score.scored;
    const reason = (score.key === 'exec_match_rate' ? EXCLUDED_REASON.exec_match_rate : EXCLUDED_REASON[score.shape])
        ?? 'the scorer left them out';
    return `${left.toLocaleString('en-US')} of ${answered.toLocaleString('en-US')} answered rows ${left === 1 ? 'is' : 'are'} not in this rate: ${reason}. The rate is over ${score.scored.toLocaleString('en-US')} rows.`;
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

/** "87.0% (95% CI 81.6%–91.0%)", "87.0%", or a plain number for a score that is not a share (perplexity). */
export function scoreText(score: ScoreOfRecord): string {
    if (!score.percent) return typeof score.value === 'number' ? score.value.toFixed(3) : '—';
    const interval = score.ci ? ` (95% CI ${pct(score.ci[0])}–${pct(score.ci[1])})` : '';
    return `${pct(score.value)}${interval}`;
}

/** The score's value alone in its own scale: "85.7%" (share), "0.184" (0–1 mean), "12.300" (perplexity). */
export function scoreValueText(score: ScoreOfRecord): string {
    if (typeof score.value !== 'number') return '—';
    return score.percent ? pct(score.value) : score.value.toFixed(3);
}

/** "Token F1 0.184", "Plan pass rate 85.7%": a result's score of record named and formatted (phase 7k). */
export function scoreSummaryText(result: VerdictResultLike): string {
    const score = scoreOfRecord(result);
    return `${score.label} ${scoreValueText(score)}`;
}

/** Colour class for a score: only a share of passing rows is coloured by threshold (above 80% success, above 50%
 *  warning, else error); a 0–1 mean or perplexity has no meaningful pass threshold, so none. */
export function scoreToneClass(score: ScoreOfRecord): string {
    if (!score.percent || typeof score.value !== 'number') return '';
    return score.value > 0.8 ? 'u-text-success' : score.value > 0.5 ? 'u-text-warning' : 'u-text-error';
}

/** Strict tool-call argument matching as the result records it: on (the strict rate is stored), off (the
 *  result carries the tool-call fingerprint, which names strict only when it is on), or not recorded. */
export function strictArgsState(result: VerdictResultLike): 'on' | 'off' | 'not_recorded' {
    const metrics = result.metrics ?? {};
    if (num(metrics.tool_call_strict_pass_rate) !== null) return 'on';
    const fp = metrics.tool_call_fingerprint;
    if (typeof fp === 'string' && fp) return fp.includes('+strict_args') ? 'on' : 'off';
    return 'not_recorded';
}

/** "How it's scored" line: mode, strict arguments, null as absent, prompt rendering, decoding. */
export function howScoredText(result: VerdictResultLike): string {
    const metrics = result.metrics ?? {};
    // A judged result reads "graded by a judge model", as on the Verdict card (phase 7k); the strict tool-call
    // state has its own part below.
    const score = scoreOfRecord(result);
    const parts = [`scored as ${score.key === 'judge_correct' ? score.scoredAs : scoredAsText(result)}`];
    if (scoringModeOf(result) === 'tool_call') {
        const strict = strictArgsState(result);
        parts.push(strict === 'on' ? 'strict arguments on' : strict === 'off' ? 'strict arguments off' : 'strict arguments not recorded');
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

/** True when another model answered (``metrics.evaluated_model``: an explicit model path, base or custom),
 *  not the run's own weights, although the result is filed under the run. */
export function isOtherModelResult(result: VerdictResultLike): boolean {
    const record = result.metrics?.evaluated_model as { model_path?: string } | null | undefined;
    return !!record?.model_path;
}

/**
 * One result per evaluated set for the Verdict card: the newest the run's own weights answered; a set with
 * only results another model answered (an explicit model path) shows its newest, which the card labels
 * "answered by model path X". Results arrive newest first.
 */
export function verdictResultsPerSet<T extends VerdictResultLike>(results: T[]): T[] {
    const own = newestPerDataset(results.filter((r) => !isOtherModelResult(r)));
    const ownSets = new Set(own.map((r) => r.dataset_name));
    const other = newestPerDataset(results.filter((r) => isOtherModelResult(r) && !ownSets.has(r.dataset_name)));
    return results.filter((r) => own.includes(r) || other.includes(r));
}

/** Lift keys for a run's results: '' for the built-in sets (one call), each eval-set label otherwise. Only
 *  results the run's own weights answered ask for a lift. */
export function liftKeys(results: VerdictResultLike[]): string[] {
    const keys: string[] = [];
    for (const r of newestPerDataset(results.filter((x) => !isOtherModelResult(x)))) {
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
    if (resultShape(result) !== 'text' || result.eval_type !== 'exact_match') return false;
    return all.some((r) => hasTaskScoringMode(r));
}
