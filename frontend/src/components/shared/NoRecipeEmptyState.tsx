/**
 * NoRecipeEmptyState — directive CTA shown when a panel can't render
 * because the project has no `selected_recipe` set yet.
 *
 * Brief-driven + magic-create projects auto-apply a recipe at create
 * time (commit 1449877), so only legacy projects (pre-fix) hit this.
 * Several panels — synth playbooks, auto-RAG comparison, archetype
 * comparison — used to silently vanish on the NULL-recipe branch,
 * leaving legacy users with no signal about why an expected feature
 * was missing. This component standardizes the "pick a recipe first"
 * surface so the user sees the same prompt no matter which tab they
 * land on.
 *
 * Usage:
 *
 *     <NoRecipeEmptyState
 *         projectId={projectId}
 *         surface="auto-RAG comparison"
 *     />
 *
 * Renders a centered card with a uniform headline, a one-line hint
 * scoped to the calling surface, and a primary "Pick a recipe" button
 * that navigates to the recipe-picker page (the same target Coach
 * Mode's `recipe-picker` action uses).
 */

import EmptyState from './EmptyState';


interface Props {
    projectId: number;
    /** Short noun describing the surface this CTA is appearing in
     *  (e.g. ``"auto-RAG comparison"``, ``"archetype comparison"``,
     *  ``"synthetic playbooks"``). Slots into the description so the
     *  user knows which feature is gated on the recipe. */
    surface: string;
    /** Optional in-app path the recipe-picker page should redirect to
     *  after a successful apply. Defaults to the current pathname so
     *  the user lands back where they were, which makes the CTA → pick
     *  → CTA-disappears loop a single click round-trip. */
    returnTo?: string;
    /** Optional test id passthrough. Defaults to a stable id so the
     *  panel can scope its own assertions without colliding with
     *  other instances on the same page. */
    testId?: string;
}


export default function NoRecipeEmptyState({
    projectId,
    surface,
    returnTo,
    testId = 'no-recipe-empty-state',
}: Props) {
    // Default ``returnTo`` to wherever the CTA is being mounted —
    // resolved at render time so the legacy user comes back to the
    // exact page (and tab) they triggered the CTA from. Tests can
    // override via the explicit prop.
    const resolvedReturnTo =
        returnTo
        ?? (typeof window !== 'undefined'
            ? window.location.pathname + window.location.search
            : `/project/${projectId}/pipeline/data`);
    const href =
        `/project/${projectId}/recipe-picker`
        + `?return_to=${encodeURIComponent(resolvedReturnTo)}`;

    return (
        <div data-testid={testId}>
            <EmptyState
                icon="🧭"
                title="Pick a recipe first"
                description={
                    `${surface} is recipe-scoped — each task shape `
                    + `(Q&A, classification, span extraction, …) ships `
                    + `its own behavior. Pick a recipe to unlock this `
                    + `surface; existing projects can change it later.`
                }
                primary={{
                    label: 'Pick a recipe',
                    href,
                }}
            />
        </div>
    );
}
