/**
 * Behavioral tests CRUD client (Quality-Lift phase 7 slice 2).
 *
 * Pairs with the phase 5 slice 1 schema validator on the backend:
 *   * BEHAVIORAL_TEST_KINDS (INV / DIR / MFT)
 *   * PERTURBATION_KINDS (typo / insert_token / case_change / whitespace_jitter)
 *   * DIR_EXPECTATION_KINDS (must_change / must_change_to / must_change_to_one_of)
 *
 * Each closed tuple is mirrored as a frontend constant so the editor's
 * dropdowns are exhaustive without a backend round-trip. If the backend
 * tuple ever grows, the frontend mirror must update in lockstep — the
 * validator will reject the unknown value otherwise.
 */

import api from './client';


// Mirrors backend/app/services/behavioral_test_schema.BEHAVIORAL_TEST_KINDS.
export const BEHAVIORAL_TEST_KINDS = ['INV', 'DIR', 'MFT'] as const;
export type BehavioralTestKind = typeof BEHAVIORAL_TEST_KINDS[number];

// Mirrors backend/app/services/behavioral_test_schema.PERTURBATION_KINDS.
export const PERTURBATION_KINDS = [
    'typo',
    'insert_token',
    'case_change',
    'whitespace_jitter',
] as const;
export type PerturbationKind = typeof PERTURBATION_KINDS[number];

// Mirrors backend/app/services/behavioral_test_schema.DIR_EXPECTATION_KINDS.
export const DIR_EXPECTATION_KINDS = [
    'must_change',
    'must_change_to',
    'must_change_to_one_of',
] as const;
export type DirExpectationKind = typeof DIR_EXPECTATION_KINDS[number];

export const CASE_CHANGE_OPTIONS = ['lower', 'upper', 'title'] as const;
export type CaseChangeOption = typeof CASE_CHANGE_OPTIONS[number];


export interface SeedExample {
    input: string;
    given_label?: string | null;
}

export interface MftExample {
    input: string;
    expected_label: string;
}

export interface Perturbation {
    kind: PerturbationKind;
    name?: string;
    intensity?: number;
    params?: Record<string, unknown>;
    seed?: unknown;
}

export interface Expectation {
    kind: 'same_label' | DirExpectationKind;
    target_label?: string;
    target_labels?: string[];
}

export interface BehavioralTest {
    test_id: string;
    kind: BehavioralTestKind;
    description?: string;
    pass_rate_floor?: number;
    // INV/DIR only:
    seed_examples?: SeedExample[];
    perturbations?: Perturbation[];
    expectation?: Expectation;
    n_perturbations_per_seed?: number;
    // MFT only:
    examples?: MftExample[];
}

export interface BehavioralTestsResponse {
    project_id: number;
    task_profile: string;
    behavioral_tests: BehavioralTest[];
}


export async function fetchBehavioralTests(projectId: number): Promise<BehavioralTestsResponse> {
    const resp = await api.get<BehavioralTestsResponse>(
        `/projects/${projectId}/behavioral-tests`,
    );
    return resp.data;
}

export async function saveBehavioralTests(
    projectId: number,
    tests: BehavioralTest[],
): Promise<BehavioralTestsResponse> {
    const resp = await api.put<BehavioralTestsResponse>(
        `/projects/${projectId}/behavioral-tests`,
        { behavioral_tests: tests },
    );
    return resp.data;
}
