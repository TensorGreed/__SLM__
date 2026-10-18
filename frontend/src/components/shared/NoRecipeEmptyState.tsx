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
 * that opens the Train page's task section (the same target Coach
 * Mode's `recipe-picker` action uses).
 */

import EmptyState from './EmptyState';
import { trainSectionPath } from '../../utils/workspaceRoutes';


interface Props {
    projectId: number;
    /** Short noun describing the surface this CTA is appearing in
     *  (e.g. ``"auto-RAG comparison"``, ``"archetype comparison"``,
     *  ``"synthetic playbooks"``). Slots into the description so the
     *  user knows which feature is gated on the recipe. */
    surface: string;
    /** Retained for callers; ignored now that the task section lives on
     *  the Train page (no separate page to return from). */
    returnTo?: string;
    /** Optional test id passthrough. Defaults to a stable id so the
     *  panel can scope its own assertions without colliding with
     *  other instances on the same page. */
    testId?: string;
}


export default function NoRecipeEmptyState({
    projectId,
    surface,
    testId = 'no-recipe-empty-state',
}: Props) {
    // The task section lives on the Train page now; there is no
    // separate page to return from, so ``returnTo`` is no longer used.
    const href = trainSectionPath(projectId, 'task');

    return (
        <div data-testid={testId}>
            <EmptyState
                icon="🧭"
                title="Choose a task type first"
                description={
                    `${surface} is recipe-scoped — each task shape `
                    + `(Q&A, classification, span extraction, …) ships `
                    + `its own behavior. Choose a task type to unlock this `
                    + `surface; existing projects can change it later.`
                }
                primary={{
                    label: 'Choose a task type',
                    href,
                }}
            />
        </div>
    );
}
