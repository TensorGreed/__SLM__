/**
 * Probe Pack API client — Coach-stage-2 phase 8.
 *
 * The platform-authored, recipe-keyed adversarial probe pack: the
 * held-out ruler the user did NOT author, so the gate can grade against
 * something independent of the (self-authored, possibly-easy) gold set.
 *
 * This slice is read-only — the pack is assembled + inspectable
 * (`status: "ready_not_run"`); running it against the trained model and
 * folding an independent pass-rate into the gate is the next slice.
 */

import api from './client';

export type ProbeKind =
    | 'robustness'
    | 'safety_refusal'
    | 'format_robustness'
    | 'degenerate_input';

export type ProbeProperty =
    | 'prediction_stable_vs_base'
    | 'refuses_or_declines'
    | 'no_fabrication_when_unsupported'
    | 'handles_degenerate_gracefully';

export interface Probe {
    id: string;
    probe_kind: ProbeKind;
    property: ProbeProperty;
    input: string;
    rationale: string;
    /** Set only for stability probes — the model's output on this is
     * compared against its output on `input`. */
    base_input?: string;
}

export interface ProbeResult {
    id: string;
    probe_kind: ProbeKind;
    property: ProbeProperty;
    passed: boolean;
    output: string;
    base_output: string | null;
    reason: string;
}

export interface ProbePropertyScore {
    passed: number;
    total: number;
    pass_rate: number;
}

export interface ProbeRun {
    status: 'graded';
    probe_pass_rate: number;
    passed: number;
    total: number;
    per_property: Record<string, ProbePropertyScore>;
    results: ProbeResult[];
    run_at: string | null;
    eval_result_id?: number;
    experiment_id?: number;
}

export interface ProbePack {
    project_id?: number;
    task_profile: string | null;
    version: string;
    applicable: boolean;
    probe_count: number;
    kind_summary: Record<string, number>;
    probes: Probe[];
    /** "graded" once the pack has been run against a trained checkpoint. */
    status: 'ready_not_run' | 'no_pack_for_profile' | 'graded';
    note: string;
    /** Present once the pack has been run — the independent result. */
    run?: ProbeRun;
}

export async function fetchProbePack(projectId: number): Promise<ProbePack> {
    const res = await api.get<ProbePack>(`/projects/${projectId}/probe-pack`);
    return res.data;
}
