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
    | 'handles_degenerate_gracefully'
    | 'does_not_over_refuse'
    | 'returns_empty_findings'
    | 'findings_stable_vs_base'
    | 'must_match'
    | 'must_not_match'
    | 'json_pointer_not_in';

export type ProbeSeverity = 'low' | 'medium' | 'high';
export type ProbeSource = 'platform' | 'user';

/** Probe input of the code-findings pack: a chunk rendered with the shared prompt builder. */
export interface CodeProbeInput {
    code: string;
    language?: string | null;
    path?: string | null;
    start_line?: number | null;
}

export type ProbeInput = string | CodeProbeInput;

export interface Probe {
    id: string;
    probe_kind: ProbeKind;
    property: ProbeProperty;
    input: ProbeInput;
    rationale: string;
    /** Set only for stability probes — the model's output on this is
     * compared against its output on `input`. */
    base_input?: ProbeInput;
    /** User-authored checks only. */
    source?: ProbeSource;
    pack_id?: string;
    severity?: ProbeSeverity;
    critical?: boolean;
}

export interface ProbeResult {
    id: string;
    probe_kind: ProbeKind;
    property: ProbeProperty;
    passed: boolean;
    output: string;
    base_output: string | null;
    reason: string;
    /** e.g. 'base_had_no_findings' (findings_stable_vs_base scored on an empty clean answer). */
    note?: string | null;
    /** Phase 21 — per-probe weight (by kind, or per-probe override). */
    weight?: number;
    source?: ProbeSource;
    pack_id?: string;
    severity?: ProbeSeverity;
    critical?: boolean;
}

export interface ProbeSourceScore {
    passed: number;
    total: number;
    pass_rate: number;
}

export interface ProbePropertyScore {
    passed: number;
    total: number;
    pass_rate: number;
}

export interface ProbeKindScore {
    weight: number;
    passed: number;
    total: number;
    pass_rate: number;
}

export interface ProbeRun {
    status: 'graded';
    /** Weighted (by probe kind) — the headline + gate score. Built-in checks
     * only; null when only your own checks ran. */
    probe_pass_rate: number | null;
    /** Raw pass fraction of the built-in checks, next to the weighted score. */
    unweighted_pass_rate?: number | null;
    /** Severity-weighted pass rate of your own checks (absent when none ran). */
    user_probe_pass_rate?: number;
    user_per_property?: Record<string, ProbePropertyScore>;
    passed: number;
    total: number;
    per_property: Record<string, ProbePropertyScore>;
    /** Phase 21 — per-kind weighted breakdown. */
    weighted_by_kind?: Record<string, ProbeKindScore>;
    results: ProbeResult[];
    run_at: string | null;
    eval_result_id?: number;
    experiment_id?: number;
    /** Phase 18 — LLM-judge cost accounting for this run. */
    judge_calls?: number;
    judge_cached?: number;
    /** Critical user checks: any failure blocks the gate, even with the optional gate off. */
    critical_total?: number;
    critical_failed?: number;
    critical_failures?: { id: string; reason: string }[];
    by_source?: Partial<Record<ProbeSource, ProbeSourceScore>>;
}

export interface ProbeGateConfig {
    enabled: boolean;
    min_pass_rate: number;
    required: boolean;
}

export interface DivergencePoint {
    run_at: string | null;
    experiment_id?: number;
    eval_result_id?: number;
    gold_pass_rate: number;
    probe_pass_rate: number;
    divergence: number;
    /** Phase 23 — weight regime active at this eval; a change between
     * consecutive points means the score weighting was re-configured. */
    weight_regime?: string | null;
    /** Probe-arc finale — this point's per-probe outcomes re-scored under the
     * project's *current* weights, so the trend stays comparable across a
     * weight-regime change. Absent when the stored snapshot kept no per-probe
     * results to re-score. */
    probe_pass_rate_reweighted?: number;
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
    /** Phase 13 — the optional probe gate config (off by default). */
    gate_config?: ProbeGateConfig;
    /** Phase 16 — gold vs probe pass-rate over the last few runs. */
    divergence_history?: DivergencePoint[];
    /** Phase 22 — effective per-kind weights (defaults + project overrides). */
    kind_weights?: Record<string, number>;
    /** Phase 24 — cumulative LLM-judge spend across recent runs. */
    judge_spend?: JudgeSpend;
    /** User-authored check packs applying to this project. */
    user_packs?: UserPackSummary[];
    user_probes?: Probe[];
}

export interface UserPackSummary {
    id: string;
    name?: string | null;
    version?: string | null;
    probe_count: number;
    critical_count: number;
    source?: { kind: 'domain_pack' | 'manual'; pack_id?: string; version?: string } | null;
}

export interface UserProbePackIssue {
    code: string;
    path?: string | null;
    message: string;
}

export interface UserProbePacks {
    packs: Record<string, unknown>[];
    version: string | null;
    source: { kind: 'domain_pack' | 'manual'; pack_id?: string; version?: string } | null;
    history: UserProbeHistoryEntry[];
}

export interface UserProbeHistoryEntry {
    version: string;
    updated_at?: string;
    source?: { kind: 'domain_pack' | 'manual'; pack_id?: string; version?: string } | null;
    pack_count: number;
    probe_count: number;
    /** This is the set in use now. */
    current: boolean;
    /** Its full content is still stored, so it can be restored. */
    restorable: boolean;
}

export interface UserProbeValidation {
    valid: boolean;
    errors: UserProbePackIssue[];
    pack_count: number;
    probe_count: number;
    critical_count: number;
    /** Normalized packs (empty when invalid). */
    packs: Record<string, unknown>[];
}

export interface JudgeSpend {
    total_calls: number;
    total_cached: number;
    runs_with_judge: number;
    /** Real provider token total (phase 25), or estimate if `tokens_estimated`. */
    total_tokens: number;
    tokens_estimated: boolean;
}

export async function fetchProbePack(projectId: number): Promise<ProbePack> {
    const res = await api.get<ProbePack>(`/projects/${projectId}/probe-pack`);
    return res.data;
}

export async function setProbeGate(
    projectId: number,
    config: ProbeGateConfig,
): Promise<ProbeGateConfig> {
    const res = await api.put<ProbeGateConfig>(
        `/projects/${projectId}/probe-pack/gate`,
        config,
    );
    return res.data;
}

export async function setProbeKindWeights(
    projectId: number,
    weights: Record<string, number>,
): Promise<Record<string, number>> {
    const res = await api.put<Record<string, number>>(
        `/projects/${projectId}/probe-pack/kind-weights`,
        { weights },
    );
    return res.data;
}

export async function fetchUserProbePacks(projectId: number): Promise<UserProbePacks> {
    const res = await api.get<UserProbePacks>(`/projects/${projectId}/probe-pack/user`);
    return res.data;
}

export async function validateUserProbePack(
    projectId: number,
    content: string,
): Promise<UserProbeValidation> {
    const res = await api.post<UserProbeValidation>(
        `/projects/${projectId}/probe-pack/user/validate`,
        { content },
    );
    return res.data;
}

export async function saveUserProbePack(
    projectId: number,
    content: string,
): Promise<UserProbePacks> {
    const res = await api.put<UserProbePacks>(`/projects/${projectId}/probe-pack/user`, {
        content,
        mode: 'merge',
    });
    return res.data;
}

export async function restoreUserProbePacks(
    projectId: number,
    version: string,
): Promise<UserProbePacks> {
    const res = await api.post<UserProbePacks>(`/projects/${projectId}/probe-pack/user/restore`, {
        version,
    });
    return res.data;
}

export async function deleteUserProbePacks(projectId: number): Promise<void> {
    await api.delete(`/projects/${projectId}/probe-pack/user`);
}
