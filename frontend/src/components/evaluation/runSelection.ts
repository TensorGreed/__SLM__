/**
 * Which run the Results tabs show. One rule, shared with the server's
 * ``eval_summary_service._latest_trained_experiment_id``: the newest
 * COMPLETED run that is neither a baseline (the auto lift check's
 * ``config.is_baseline`` run) nor one seed of a multi-seed group. A
 * baseline is never the default — it stays pickable, labelled as such.
 */

export interface RunSummary {
    id: number;
    name: string;
    status?: string | null;
    config?: Record<string, unknown> | null;
    seed_group_id?: string | number | null;
    seed_value?: number | null;
}

export function isBaselineRun(run: RunSummary): boolean {
    return run.config?.is_baseline === true;
}

function isSeedChild(run: RunSummary): boolean {
    return run.seed_group_id != null && run.seed_group_id !== '' && run.seed_value != null;
}

/** The run a Results tab opens on, or null when only baselines exist. */
export function defaultRunId(runs: RunSummary[]): number | null {
    const candidates = [...runs]
        .filter((run) => !isBaselineRun(run) && !isSeedChild(run))
        .sort((a, b) => Number(b.id) - Number(a.id));
    const completed = candidates.find((run) => String(run.status ?? '').toLowerCase() === 'completed');
    return (completed ?? candidates[0])?.id ?? null;
}

/** "Run #12 · my-run" (+ " · baseline") — the provenance line every card carries. */
export function runLabel(runs: RunSummary[], id: number | null): string {
    if (id == null) return '';
    const run = runs.find((r) => r.id === id);
    if (!run) return `Run #${id}`;
    return `Run #${run.id} · ${run.name}${isBaselineRun(run) ? ' · baseline' : ''}`;
}
