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
    if (id === 'exact_match') return 'exact match';
    if (id === 'accuracy') return 'accuracy';
    return id || 'score';
}
