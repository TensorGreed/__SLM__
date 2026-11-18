/**
 * Shared wording for "is this lift more than noise?" (backend
 * ``paired_comparison_stats``): row counts + a 95% interval for the mean
 * per-row change. Used by the Auto-RAG comparison cards and the Eval tab's
 * lift check card so a lift on a few rows reads the same everywhere.
 */

/** How big a difference these rows can reliably show (backend
 *  ``eval_sizing_service``, Epic W2): 95% confidence / 80% power. ``basis``
 *  ``paired`` uses the spread of per-row differences (lift, A-vs-B);
 *  ``single`` the conservative unpaired bound for comparing two runs on
 *  rows like this eval's (√2 × its own one-sample precision). ``line`` is the plain sentence, with
 *  the row count and metric in it. */
export interface EvalSizing {
    basis: 'paired' | 'single';
    metric_id: string | null;
    n: number;
    score_kind: 'binary' | 'continuous';
    sd: number | null;
    detectable_gap: number | null;
    target_gap: number;
    rows_needed: number;
    enough: boolean;
    degenerate: 'too_few_rows' | 'all_pass' | 'all_fail' | 'zero_variance' | 'identical_scores' | 'constant_shift' | null;
    confidence: number;
    power: number;
    line: string;
    /** Single basis only: this eval's own precision against a fixed value (not a comparison; API only). */
    one_sample_gap?: number | null;
    one_sample_rows_needed?: number;
}

/** The one sizing figure to show. With paired evidence: the paired figure,
 *  only when the evidence can't call the change (within noise / too few rows)
 *  — never next to a significant verdict. Without paired evidence: the
 *  eval's own conservative (unpaired) comparison bound. */
export function sizingToShow(
    evidence: { verdict: string; sizing?: EvalSizing | null } | null | undefined,
    single: EvalSizing | null | undefined,
): EvalSizing | null {
    if (evidence) {
        const unproven = evidence.verdict === 'within_noise' || evidence.verdict === 'too_few_rows';
        return unproven && evidence.sizing ? evidence.sizing : null;
    }
    return single ?? null;
}

export interface LiftEvidence {
    n: number;
    better: number;
    worse: number;
    same: number;
    mean_diff: number | null;
    ci_low: number | null;
    ci_high: number | null;
    verdict: 'better' | 'worse' | 'within_noise' | 'too_few_rows';
    /** Exact two-sided McNemar p on the rows that changed; set only when every paired
     *  score is 0 / 1 (absent on older servers). Better / worse then also need p < 0.05. */
    mcnemar_p?: number | null;
    metric_id?: string | null;
    /** Epic W2: detectable difference on these paired rows (absent on older servers). */
    sizing?: EvalSizing | null;
    /** Rows both evals ran but one left without a score (unjudged, a verifier error): not in ``n``. */
    excluded_unscored?: number | null;
}

export interface JudgedSubsetSide {
    judged: number;
    unjudged: number;
    /** The judge was unreachable for some of them (the judged rows are then the first ones, not a sample). */
    outage?: boolean;
}

/** A judge-based lift metric whose value rests on the judged rows only, per side (backend
 *  ``sft_lift_summary_service._attach_judged_subset``). */
export type JudgedSubset = { baseline?: JudgedSubsetSide; trained?: JudgedSubsetSide };

/** "on judged rows only — fine-tuned 40 of 688 judged (judge unreachable), base 688 of 688", or null. */
export function judgedSubsetText(subset: JudgedSubset | null | undefined): string | null {
    if (!subset) return null;
    const sides: Array<[string, JudgedSubsetSide | undefined]> = [['fine-tuned', subset.trained], ['base', subset.baseline]];
    const parts = sides
        .filter((entry): entry is [string, JudgedSubsetSide] => !!entry[1] && entry[1].unjudged > 0)
        .map(([name, side]) => `${name} ${side.judged} of ${side.judged + side.unjudged} judged${side.outage ? ' (judge unreachable)' : ''}`);
    if (!parts.length) return null;
    const outage = sides.some(([, side]) => side?.outage);
    return `on judged rows only — ${parts.join(', ')}${outage ? '. Re-run the eval when the judge is back before relying on this lift' : ''}`;
}

/** " · 10 excluded (no score on one side)" for an evidence block that left rows out, else "". */
export function excludedRowsText(ev: Pick<LiftEvidence, 'excluded_unscored'> | null | undefined): string {
    const k = ev?.excluded_unscored;
    return typeof k === 'number' && k > 0 ? ` · ${k} excluded (no score on one side, e.g. unjudged)` : '';
}

/** Run-to-run evidence: the same config trained with N seeds, each scored
 *  against the base model (backend ``seed_spread_evidence``). */
export interface SeedEvidence {
    kind: 'seeds';
    n: number;
    verdict: 'better' | 'worse' | 'within_noise' | 'too_few_seeds';
    baseline_value: number;
    values: number[];
    mean: number | null;
    std: number | null;
    min: number | null;
    max: number | null;
    ci_low: number | null;
    ci_high: number | null;
    all_better: boolean;
    all_worse: boolean;
    metric_id?: string | null;
}

/** The sentence under a multi-seed headline: does the lift hold across
 *  seeds? This is the variance the row-level interval cannot see. */
export function seedEvidenceNote(ev: SeedEvidence, unit: string): { label: string; text: string } {
    const values = ev.values.map((v) => v.toFixed(3)).join(', ');
    const spread = ev.mean !== null && ev.std !== null
        ? `${ev.mean.toFixed(3)} ± ${ev.std.toFixed(3)}`
        : ev.mean !== null ? ev.mean.toFixed(3) : '—';
    const base = `base ${ev.baseline_value.toFixed(3)}`;
    if (ev.verdict === 'too_few_seeds') {
        return {
            label: 'One seed only',
            text: `${unit} ${values} (${base}). One training run cannot show run-to-run variance — check across seeds before relying on it.`,
        };
    }
    if (ev.verdict === 'within_noise') {
        return {
            label: 'Seeds disagree',
            text: `Across ${ev.n} seeds: ${unit} ${values} (mean ${spread}, ${base}). The seeds disagree too much to call this a gain or a loss — another seed could land on either side.`,
        };
    }
    const every = ev.verdict === 'better'
        ? (ev.all_better ? 'every seed beat the base model' : 'the seeds agree it is a gain')
        : (ev.all_worse ? 'every seed fell below the base model' : 'the seeds agree it is a drop');
    return {
        label: ev.verdict === 'better' ? 'Holds across seeds' : 'Drop across seeds',
        text: `Across ${ev.n} seeds: ${unit} ${values} (mean ${spread}, ${base}) — ${every}.`,
    };
}

/** "p = 0.125" (exact McNemar), or null when the scores aren't 0 / 1. */
export function mcnemarText(p: number | null | undefined): string | null {
    if (typeof p !== 'number' || !Number.isFinite(p)) return null;
    return p < 0.001 ? 'p < 0.001' : `p = ${p.toFixed(3)}`;
}

/** True when the interval alone excludes zero but the exact McNemar test (0 / 1 scores)
 *  isn't below 0.05: the backend then reports within noise. */
export function heldBackByMcNemar(ev: { verdict: string; ci_low: number | null; ci_high: number | null; mcnemar_p?: number | null }): boolean {
    const p = ev.mcnemar_p;
    if (ev.verdict !== 'within_noise' || typeof p !== 'number' || p < 0.05) return false;
    return (ev.ci_low !== null && ev.ci_low > 0) || (ev.ci_high !== null && ev.ci_high < 0);
}

export function signedScore(value: number): string {
    return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;
}

/** "Retrieval helped 14 rows, hurt 6 rows, no change on 1." */
export function rowCountsText(ev: LiftEvidence, subject: string): string {
    const rows = (count: number) => `${count} row${count === 1 ? '' : 's'}`;
    const parts = [`helped ${rows(ev.better)}`, `hurt ${rows(ev.worse)}`];
    if (ev.same > 0) parts.push(`no change on ${ev.same}`);
    const excluded = typeof ev.excluded_unscored === 'number' && ev.excluded_unscored > 0
        ? ` ${rows(ev.excluded_unscored)} left out: no score on one side (e.g. unjudged).`
        : '';
    return `${subject} ${parts.join(', ')}.${excluded}`;
}

/**
 * The sentence that keeps a lift on a few rows from being read as a result.
 * ``trained`` adds the run-to-run caveat: the interval covers which rows
 * were sampled, not what another training seed would give. ``unit`` names
 * the per-row score ("F1", "exact match").
 */
export function evidenceNote(
    ev: LiftEvidence,
    options: { trained: boolean; unit: string },
): { label: string; text: string } {
    const range = ev.mean_diff !== null && ev.ci_low !== null && ev.ci_high !== null
        ? `average change ${signedScore(ev.mean_diff)} ${options.unit} per row, 95% range ${signedScore(ev.ci_low)} to ${signedScore(ev.ci_high)}`
        : null;
    if (ev.verdict === 'too_few_rows') {
        return {
            label: 'Too few rows to tell',
            text: `Only ${ev.n} scored row${ev.n === 1 ? '' : 's'} — not enough to tell a gain from noise.`,
        };
    }
    const p = mcnemarText(ev.mcnemar_p);
    if (ev.verdict === 'within_noise') {
        if (heldBackByMcNemar(ev)) {
            const changed = ev.better + ev.worse;
            return {
                label: 'Within noise',
                text: `On ${ev.n} rows the ${range}, but only ${changed} row${changed === 1 ? '' : 's'} changed (${ev.better} better, ${ev.worse} worse): exact McNemar ${p}, so this could be chance. A gain or a drop needs p below 0.05: at least 6 changed rows (6 better, 0 worse is the smallest that counts), and more when they split both ways.`,
            };
        }
        return {
            label: 'Within noise',
            text: `On ${ev.n} rows the ${range} — it includes zero, so this lift could be chance. Don't read it as a gain or a loss.`,
        };
    }
    const direction = ev.verdict === 'better' ? 'gain' : 'drop';
    const seedCaveat = options.trained
        ? ' This is one training run: another seed can move it, so re-train and re-run before relying on it.'
        : '';
    return {
        label: ev.verdict === 'better' ? 'Gain on these rows' : 'Drop on these rows',
        text: `On ${ev.n} rows the ${range}${p ? ` (exact McNemar ${p})` : ''} — a ${direction} beyond row-to-row noise.${seedCaveat}`,
    };
}

export function metricUnit(metricId: string | null | undefined): string {
    const id = String(metricId || '').toLowerCase();
    if (id === 'f1') return 'F1';
    if (id === 'judge_correct') return 'judge score';
    if (id === 'exact_match') return 'exact match';
    if (id === 'accuracy') return 'accuracy';
    if (id === 'remediation_pass_rate') return 'remediation pass rate';
    if (id === 'code_example_pass_rate') return 'code-example pass rate';
    if (id === 'structured_plan_pass_rate') return 'plan pass rate';
    if (id === 'tool_call_pass_rate') return 'tool-call pass rate';
    if (id === 'tool_call_strict_pass_rate') return 'strict tool-call pass rate';
    if (id === 'tool_call_valid_rate') return 'valid tool-call rate';
    if (id === 'exec_match_rate') return 'execution match rate';
    return id || 'score';
}

/** Display name for a metric id in headlines ("judge_correct" is the
 *  LLM-judge correctness score: correct 1 / partial 0.5 / wrong 0 per row). */
export function metricDisplayName(metricId: string | null | undefined): string {
    const id = String(metricId || '');
    if (id === 'judge_correct') return 'judge: answer correct';
    if (id === 'remediation_pass_rate') return 'remediation pass rate';
    if (id === 'code_example_pass_rate') return 'code-example pass rate';
    if (id === 'structured_plan_pass_rate') return 'plan pass rate';
    if (id === 'tool_call_pass_rate') return 'tool-call pass rate';
    if (id === 'tool_call_strict_pass_rate') return 'strict tool-call pass rate';
    if (id === 'tool_call_valid_rate') return 'valid tool-call rate';
    if (id === 'exec_match_rate') return 'execution match rate';
    return id;
}

/** One compact line for dense places (a metric row, the bell):
 *  "within noise · 14 rows better, 6 worse". */
export function shortEvidenceText(ev: Pick<LiftEvidence, 'verdict' | 'better' | 'worse' | 'excluded_unscored' | 'mcnemar_p'>): string {
    const label = ev.verdict === 'within_noise'
        ? 'within noise'
        : ev.verdict === 'too_few_rows'
            ? 'too few rows to tell'
            : 'beyond row noise';
    const p = ev.verdict === 'too_few_rows' ? null : mcnemarText(ev.mcnemar_p);
    return `${label} · ${ev.better} row${ev.better === 1 ? '' : 's'} better, ${ev.worse} worse${p ? ` (McNemar ${p})` : ''}${excludedRowsText(ev)}`;
}

/** Stored evidence (a Job result, a sweep arm, a passages record) written before the exact McNemar
 *  check existed has no ``mcnemar_p`` key at all (newer results carry it, as null for continuous
 *  scores). Its better / worse was decided by the interval alone, so it is labelled rather than
 *  silently trusted. Returns "" for live or newer evidence and for within-noise verdicts. */
export function preGuardNote(ev: { verdict?: string; mcnemar_p?: number | null } | null | undefined): string {
    if (!ev || (ev.verdict !== 'better' && ev.verdict !== 'worse')) return '';
    return 'mcnemar_p' in ev ? '' : ' · recorded before the McNemar check';
}

/** A retrieval-sweep arm against the current settings (backend ``auto_rag_ab`` ``vs_current``).
 *  ``mcnemar_p`` / ``ci_*`` are absent on results recorded before they were stored. */
export interface VsCurrentEvidence {
    verdict: string;
    n: number;
    better: number;
    worse: number;
    same: number;
    mean_diff?: number | null;
    ci_low?: number | null;
    ci_high?: number | null;
    mcnemar_p?: number | null;
}

/** "within noise · 5 better, 0 worse (McNemar p = 0.062: too few changed rows to count)". */
export function vsCurrentText(vs: VsCurrentEvidence): string {
    const label = vs.verdict === 'better'
        ? 'beats current'
        : vs.verdict === 'worse'
            ? 'loses to current'
            : vs.verdict === 'too_few_rows'
                ? 'too few rows'
                : 'within noise';
    const p = vs.verdict === 'too_few_rows' ? null : mcnemarText(vs.mcnemar_p);
    const heldBack = heldBackByMcNemar({ verdict: vs.verdict, ci_low: vs.ci_low ?? null, ci_high: vs.ci_high ?? null, mcnemar_p: vs.mcnemar_p });
    const pNote = p ? ` (McNemar ${p}${heldBack ? ': too few changed rows to count' : ''})` : '';
    return `${label} · ${vs.better} better, ${vs.worse} worse${pNote}${preGuardNote(vs)}`;
}

/** True when a better / worse reading isn't backed by the rows. */
export function isUnproven(ev: Pick<LiftEvidence, 'verdict'> | null | undefined): boolean {
    return !!ev && (ev.verdict === 'within_noise' || ev.verdict === 'too_few_rows');
}

/** A multi-seed leader trains nothing itself: when it has no eval of its own,
 *  its pass/fail rules are read from one seed run (backend
 *  ``evaluate_experiment_auto_gates`` → ``evaluated_on``) — the median seed the
 *  lift card uses, or the first evaluated seed when the seeds can't be ranked. */
export interface GateEvaluatedOn {
    experiment_id: number;
    seed_value: number | null;
    n_seeds: number;
    basis: 'median_seed' | 'first_evaluated_seed';
}

/** "Measured on seed run #166 (seed 2), the median of 3 seeds." */
export function evaluatedOnText(on: GateEvaluatedOn): string {
    const seed = on.seed_value != null ? ` (seed ${on.seed_value})` : '';
    const which = on.basis === 'median_seed'
        ? `the median of ${on.n_seeds} seeds, as on the lift card`
        : `the first evaluated of ${on.n_seeds} seeds`;
    return `Measured on seed run #${on.experiment_id}${seed}, ${which}.`;
}
