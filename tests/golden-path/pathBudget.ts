/**
 * Main-path budget for the golden-path E2E (UI simplification pass).
 *
 * Today's measured numbers for the path this spec drives through the UI
 * (launch the sample project from the dashboard; training goes through the
 * API and is not a user action). Each simplification phase LOWERS these
 * constants in the same commit as the UI change that earns it; never raise
 * them to make a run pass. The Model 6 v2 reference path (23 actions, 6
 * screens before the pass; target 12 / 5) is measured by hand, see
 * agent-notes/testing.md.
 */
export const MAIN_PATH_BUDGET = {
    /** Counted user actions (see pathCounter.ts for the rule). */
    maxActions: 1,
    /** Distinct screens after login. */
    maxScreens: 2,
} as const;
