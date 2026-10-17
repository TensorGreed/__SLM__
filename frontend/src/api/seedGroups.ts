import api from './client';

/** One seed of a multi-seed run with its VALIDATION metric (never a test /
 *  answer-key / benchmark score — those are what the pick is reported on). */
export interface SeedPickChild {
    experiment_id: number;
    seed_value: number | null;
    value: number | null;
    picked: boolean;
    skipped_reason: string | null;
}

export interface SeedSelection {
    metric_id: string;
    metric_label: string;
    higher_is_better: boolean;
    rule: string;
    picked_experiment_id: number | null;
    picked_seed_value: number | null;
    n_seeds: number;
    n_scored: number;
    children: SeedPickChild[];
    export_id?: number;
    export_format?: string;
    picked_at?: string;
    text?: string | null;
}

export interface SeedPick extends SeedSelection {
    leader_experiment_id: number;
    task_type: string | null;
    text: string | null;
    /** The selection recorded by the last "Ship this seed", if any. */
    recorded: SeedSelection | null;
}

export interface ShipSeedResponse {
    leader_experiment_id: number;
    picked_experiment_id: number;
    picked_seed_value: number | null;
    selection: SeedSelection;
    /** The span lift check the ship ran first because the picked seed had no evaluation on record; null otherwise. */
    evaluated_before_ship: { ran: boolean; kind?: string } | null;
    export: {
        id: number;
        experiment_id: number;
        format: string;
        status: string;
        output_path: string | null;
        run_id: string | null;
        error: string | null;
    };
}

export async function fetchSeedPick(projectId: number, leaderId: number): Promise<SeedPick> {
    const res = await api.get<SeedPick>(`/projects/${projectId}/seed-groups/${leaderId}/pick`);
    return res.data;
}

export async function shipSeed(
    projectId: number,
    leaderId: number,
    exportFormat = 'huggingface',
): Promise<ShipSeedResponse> {
    const res = await api.post<ShipSeedResponse>(`/projects/${projectId}/seed-groups/${leaderId}/ship`, {
        export_format: exportFormat,
    });
    return res.data;
}
