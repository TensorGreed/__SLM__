/**
 * RecipeTileFrame — shared visual container for the two recipe-tile
 * surfaces:
 *
 *   * the wizard's per-suggestion card (`RecipePicker.tsx`), with
 *     confidence badges, a task-profile/scoring grid, and a
 *     "Why this recipe?" callout
 *   * the standalone picker page's catalog tile
 *     (`ProjectRecipePickerPage.tsx`), with a "Currently applied"
 *     disabled state
 *
 * Only the outer frame + the icon/name header row are genuinely shared
 * — both layouts diverge after that. Extracting just the frame
 * captures the truly common visual contract (border / padding /
 * radius / background, and the accent-border treatment for the
 * "highlighted" tile) without forcing the two content layouts to
 * converge. Each consumer fills the body via `children`.
 *
 * "Highlighted" border maps to "top suggestion" in the wizard and
 * "currently applied" on the standalone page — same visual, two
 * different semantics, which is fine because the frame doesn't care
 * which it is.
 */

import type { ReactNode } from 'react';


interface Props {
    /** Test id passthrough for the outer container. Each call site
     *  scopes its own assertions to its own prefix (e.g.
     *  ``recipe-card-${recipe.id}`` vs.
     *  ``project-recipe-picker-card-${recipe.id}``). */
    testId: string;
    /** True when this tile should render the accent border treatment
     *  — "top suggestion" in the wizard, "currently applied" on the
     *  standalone page. */
    highlighted?: boolean;
    children: ReactNode;
}


export default function RecipeTileFrame({ testId, highlighted, children }: Props) {
    return (
        <div
            data-testid={testId}
            style={{
                padding: 'var(--space-md)',
                borderRadius: 'var(--radius-md)',
                border: highlighted
                    ? '2px solid var(--accent-primary)'
                    : '1px solid var(--border-color)',
                background: 'var(--bg-card)',
            }}
        >
            {children}
        </div>
    );
}
