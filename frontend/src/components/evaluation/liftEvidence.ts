/**
 * Shared wording for "is this lift more than noise?" (backend
 * ``paired_comparison_stats``): row counts + a 95% interval for the mean
 * per-row change. Used by the Auto-RAG comparison cards and the Eval tab's
 * lift check card so a lift on a few rows reads the same everywhere.
 */

export interface LiftEvidence {
    n: number;
    better: number;
    worse: number;
    same: number;
    mean_diff: number | null;
    ci_low: number | null;
    ci_high: number | null;
    verdict: 'better' | 'worse' | 'within_noise' | 'too_few_rows';
    metric_id?: string | null;
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

export function signedScore(value: number): string {
    return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;
}

/** "Retrieval helped 14 rows, hurt 6 rows, no change on 1." */
export function rowCountsText(ev: LiftEvidence, subject: string): string {
    const rows = (count: number) => `${count} row${count === 1 ? '' : 's'}`;
    const parts = [`helped ${rows(ev.better)}`, `hurt ${rows(ev.worse)}`];
    if (ev.same > 0) parts.push(`no change on ${ev.same}`);
    return `${subject} ${parts.join(', ')}.`;
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
    if (ev.verdict === 'within_noise') {
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
        text: `On ${ev.n} rows the ${range} — a ${direction} beyond row-to-row noise.${seedCaveat}`,
    };
}

export function metricUnit(metricId: string | null | undefined): string {
    const id = String(metricId || '').toLowerCase();
    if (id === 'f1') return 'F1';
    if (id === 'judge_correct') return 'judge score';
    if (id === 'exact_match') return 'exact match';
    if (id === 'accuracy') return 'accuracy';
    return id || 'score';
}

/** Display name for a metric id in headlines ("judge_correct" is the
 *  LLM-judge correctness score: correct 1 / partial 0.5 / wrong 0 per row). */
export function metricDisplayName(metricId: string | null | undefined): string {
    const id = String(metricId || '');
    if (id === 'judge_correct') return 'judge: answer correct';
    return id;
}

/** One compact line for dense places (a metric row, the bell):
 *  "within noise · 14 rows better, 6 worse". */
export function shortEvidenceText(ev: Pick<LiftEvidence, 'verdict' | 'better' | 'worse'>): string {
    const label = ev.verdict === 'within_noise'
        ? 'within noise'
        : ev.verdict === 'too_few_rows'
            ? 'too few rows to tell'
            : 'beyond row noise';
    return `${label} · ${ev.better} row${ev.better === 1 ? '' : 's'} better, ${ev.worse} worse`;
}

/** True when a better / worse reading isn't backed by the rows. */
export function isUnproven(ev: Pick<LiftEvidence, 'verdict'> | null | undefined): boolean {
    return !!ev && (ev.verdict === 'within_noise' || ev.verdict === 'too_few_rows');
}
