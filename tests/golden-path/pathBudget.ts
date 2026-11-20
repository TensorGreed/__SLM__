/**
 * Main-path budgets for the E2E gates (UI simplification pass).
 *
 * RATCHET: a phase may only LOWER `MAIN_PATH_BUDGET`, in the same commit as
 * the UI change that earns it. Raising it needs a stated reason in the commit
 * message (a new step added to the path on purpose); never raise it to make a
 * run pass.
 */

/**
 * ENFORCED by `main-path.spec.ts`: the measured count of the real path
 * (New Project -> Data own files -> Train -> Results Evaluate + Failures ->
 * Ship export, all through the UI). Counting rule: `pathCounter.ts`.
 * The numbers are filled in from the first real run (see MAIN_PATH_MEASURED).
 */
export const MAIN_PATH_BUDGET = {
    /** Counted user actions. */
    maxActions: 22,
    /** Distinct screens after login. */
    maxScreens: 8,
} as const;

/**
 * NOT enforced: the 2026-11-19 UI-only reference run (23 actions / 6 screens),
 * measured by hand. `main-path.spec.ts` differs from it in the ways listed in
 * agent-notes/testing.md ("Main-path counter"); the plan's target is 12 / 5.
 */
export const REFERENCE_PATH = { actions: 23, screens: 6 } as const;

/**
 * ENFORCED by `golden-path.spec.ts`: it only drives the sample launch through
 * the UI (training goes through the API), so this guards that launch and
 * nothing else; it does NOT measure the main path.
 */
export const SAMPLE_LAUNCH_BUDGET = {
    maxActions: 1,
    maxScreens: 2,
} as const;
