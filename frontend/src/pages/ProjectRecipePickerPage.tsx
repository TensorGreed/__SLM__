/**
 * ProjectRecipePickerPage — standalone task-shape recipe picker.
 *
 * Distinct from `ProjectRecipesPage` (which manages **pipeline-DAG**
 * recipes — `recipe.pipeline.sft_default` and friends — stored under
 * `pipeline_recipes/state.json`). This page manages the **task-shape**
 * recipe (`qa-sft`, `classification`, `span-extraction`, …) stored on
 * `Project.selected_recipe`. The two recipe concepts coexist on every
 * project but are independent — picking a pipeline recipe doesn't
 * populate `selected_recipe`, and vice-versa.
 *
 * Before this page existed, the only place a user could pick a
 * task-shape recipe was inside `DatasetImportWizard`'s recipe step
 * (gated behind the source-introspection step). Coach Mode's
 * ``recipe-picker`` action + the shared ``NoRecipeEmptyState`` CTA
 * had no real landing page; they pointed at `/recipes`
 * (ProjectRecipesPage) which is the wrong concept. This page closes
 * that loop.
 *
 * Behavior:
 *   * Lists every catalog recipe as a tile with name + headline +
 *     icon + suggested base model.
 *   * If the project already has a recipe applied, marks it as
 *     "Currently applied" and lets the user switch.
 *   * On apply, PUTs to `/api/projects/{id}/recipe` via
 *     ``applyRecipeToProject``, surfaces a toast, and navigates
 *     back to the page the user came from. Honors an optional
 *     ``?return_to=<encoded path>`` query string (set by the
 *     CTA-emitting panels); falls back to ``/project/{id}/pipeline/data``.
 */

import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useOutletContext, useSearchParams } from 'react-router-dom';

import { applyRecipeToProject, listRecipes, type Recipe } from '../api/recipes';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import { toast } from '../stores/toastStore';


function extractErrorMessage(err: unknown, fallback: string): string {
    const detail =
        (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
    if (typeof detail === 'string' && detail.trim()) return detail;
    if (detail && typeof detail === 'object') {
        const message = (detail as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) return message;
    }
    if (err instanceof Error && err.message) return err.message;
    return fallback;
}


export default function ProjectRecipePickerPage() {
    const { projectId, project } =
        useOutletContext<ProjectWorkspaceContextValue>();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();

    const [catalog, setCatalog] = useState<Recipe[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [applyingId, setApplyingId] = useState<string | null>(null);

    // ``?return_to=<encoded path>`` lets the CTA-emitting panels send
    // the user back to where they came from after picking. We only
    // honor in-app relative paths (must start with ``/``) so the param
    // can't be used as an open-redirect vector.
    const returnTo = useMemo(() => {
        const raw = searchParams.get('return_to');
        if (raw && raw.startsWith('/')) return raw;
        return `/project/${projectId}/pipeline/data`;
    }, [searchParams, projectId]);

    const currentRecipeId = useMemo(() => {
        const snapshot = project?.selected_recipe;
        if (snapshot && typeof snapshot === 'object') {
            const id = (snapshot as { recipe_id?: unknown }).recipe_id;
            if (typeof id === 'string' && id) return id;
        }
        return null;
    }, [project]);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        setLoadError(null);
        listRecipes()
            .then((res) => {
                if (cancelled) return;
                setCatalog(res.recipes || []);
            })
            .catch((err) => {
                if (cancelled) return;
                setLoadError(
                    extractErrorMessage(err, 'Could not load the recipe catalog.'),
                );
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, []);

    const handlePick = async (recipe: Recipe) => {
        if (applyingId) return;
        setApplyingId(recipe.id);
        try {
            await applyRecipeToProject(projectId, recipe.id);
            toast.success(
                `Recipe set: ${recipe.name}. Base model defaulted to ${recipe.suggested_base_model}.`,
                4000,
            );
            // Hard-navigate so the destination page re-fetches the
            // project (its selected_recipe just changed). react-router
            // navigate() preserves component state, which would leave
            // the recipe-required CTAs still mounted on the panels
            // that triggered this flow.
            window.location.assign(returnTo);
        } catch (err) {
            toast.error(
                extractErrorMessage(err, `Could not apply recipe ${recipe.id}.`),
            );
            setApplyingId(null);
        }
    };

    return (
        <section
            data-testid="project-recipe-picker-page"
            style={{
                maxWidth: '960px',
                margin: '0 auto',
                padding: 'var(--space-lg) var(--space-md)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-lg)',
            }}
        >
            <header>
                <h1 style={{ margin: 0 }}>Pick a recipe for this project</h1>
                <p
                    style={{
                        margin: 'var(--space-xs) 0 0',
                        color: 'var(--text-secondary)',
                        fontSize: '0.95rem',
                    }}
                >
                    A recipe bundles the task shape, adapter, scoring mode, and
                    suggested base model — it shapes synthetic-data playbooks,
                    eval gates, and Coach Mode signals. You can change the
                    recipe later, but doing so may invalidate generated synth
                    rows.
                </p>
                {currentRecipeId && (
                    <p
                        data-testid="project-recipe-picker-current"
                        style={{
                            margin: 'var(--space-xs) 0 0',
                            color: 'var(--text-tertiary)',
                            fontSize: '0.85rem',
                        }}
                    >
                        Currently applied: <code>{currentRecipeId}</code>
                    </p>
                )}
            </header>

            {loading && (
                <p
                    role="status"
                    data-testid="project-recipe-picker-loading"
                    style={{ color: 'var(--text-secondary)' }}
                >
                    Loading recipe catalog…
                </p>
            )}

            {loadError && (
                <p
                    role="alert"
                    data-testid="project-recipe-picker-error"
                    style={{
                        padding: 'var(--space-md)',
                        background: 'var(--color-error-bg)',
                        color: 'var(--color-error)',
                        borderRadius: 'var(--radius-md)',
                    }}
                >
                    {loadError}
                </p>
            )}

            {!loading && !loadError && catalog.length > 0 && (
                <div
                    style={{
                        display: 'grid',
                        gridTemplateColumns:
                            'repeat(auto-fit, minmax(280px, 1fr))',
                        gap: 'var(--space-md)',
                    }}
                >
                    {catalog.map((recipe) => {
                        const isCurrent = recipe.id === currentRecipeId;
                        const isApplying = applyingId === recipe.id;
                        return (
                            <article
                                key={recipe.id}
                                data-testid={`project-recipe-picker-card-${recipe.id}`}
                                style={{
                                    padding: 'var(--space-md)',
                                    borderRadius: 'var(--radius-md)',
                                    border: isCurrent
                                        ? '2px solid var(--accent-primary)'
                                        : '1px solid var(--border-color)',
                                    background: 'var(--bg-card)',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: 'var(--space-sm)',
                                }}
                            >
                                <header
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 'var(--space-sm)',
                                    }}
                                >
                                    <span
                                        aria-hidden="true"
                                        style={{ fontSize: '1.5rem' }}
                                    >
                                        {recipe.icon}
                                    </span>
                                    <h3 style={{ margin: 0, fontSize: '1rem' }}>
                                        {recipe.name}
                                    </h3>
                                </header>
                                <p
                                    style={{
                                        margin: 0,
                                        fontSize: '0.9rem',
                                        color: 'var(--text-secondary)',
                                    }}
                                >
                                    {recipe.headline}
                                </p>
                                <p
                                    style={{
                                        margin: 0,
                                        fontSize: '0.8rem',
                                        color: 'var(--text-tertiary)',
                                    }}
                                >
                                    Suggested base model:{' '}
                                    <code>{recipe.suggested_base_model}</code>
                                </p>
                                <button
                                    type="button"
                                    className={
                                        isCurrent
                                            ? 'btn btn-secondary'
                                            : 'btn btn-primary'
                                    }
                                    disabled={isApplying || isCurrent}
                                    onClick={() => void handlePick(recipe)}
                                    data-testid={`project-recipe-picker-apply-${recipe.id}`}
                                >
                                    {isCurrent
                                        ? 'Currently applied'
                                        : isApplying
                                            ? 'Applying…'
                                            : 'Use this recipe'}
                                </button>
                            </article>
                        );
                    })}
                </div>
            )}
        </section>
    );
}
