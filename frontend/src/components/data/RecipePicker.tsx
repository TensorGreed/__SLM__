/**
 * Recipe Picker — Theme 2 Epic 3 + 4.
 *
 * Shown after the user uploads a file and the wizard has the column
 * headers in hand. Calls `POST /api/recipes/sniff`, ranks recipes by
 * how well their shape signatures match the headers, and renders the
 * top 3 as cards with:
 *
 *   - confidence badge
 *   - "Why this recipe?" callout listing which columns matched which roles
 *   - suggested base model + scoring mode summary
 *   - "Use this recipe" primary action
 *
 * "Override" link in the footer skips the picker entirely and falls
 * through to the dense mapper-config surface that already exists in
 * the import wizard — power users keep their muscle memory.
 */

import { useEffect, useMemo, useState } from 'react';

import {
    sniffRecipeFromHeaders,
    type Recipe,
    type RecipeSuggestion,
    type SniffResponse,
} from '../../api/recipes';
import { listRecipes } from '../../api/recipes';

interface RecipePickerProps {
    /** Column headers from the introspected file. */
    headers: string[];
    /** Called when the user accepts a recipe. */
    onSelect: (recipe: Recipe, suggestion: RecipeSuggestion | null) => void;
    /** Called when the user clicks the "Override" link to skip the picker. */
    onOverride: () => void;
    /** Called when the user clicks "Back" to return to the previous step. */
    onBack?: () => void;
    /** Inject a sniff function for tests; defaults to the real API client. */
    sniff?: (headers: string[]) => Promise<SniffResponse>;
    /** Inject a list function for tests. */
    listAll?: () => Promise<Recipe[]>;
}

function confidenceTone(confidence: number): 'success' | 'warning' | 'info' {
    if (confidence >= 0.85) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'info';
}

function confidenceLabel(confidence: number, fallback?: boolean): string {
    if (fallback) return 'fallback';
    if (confidence >= 0.85) return 'strong match';
    if (confidence >= 0.6) return 'partial match';
    return 'weak match';
}

function extractErrorMessage(err: unknown): string {
    if (typeof err === 'object' && err !== null) {
        const data = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof data === 'string' && data.trim()) return data;
        const message = (err as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) return message;
    }
    return 'Could not load recipe suggestions.';
}

export default function RecipePicker({
    headers,
    onSelect,
    onOverride,
    onBack,
    sniff = sniffRecipeFromHeaders,
    listAll,
}: RecipePickerProps) {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string>('');
    const [suggestions, setSuggestions] = useState<RecipeSuggestion[]>([]);
    const [catalog, setCatalog] = useState<Recipe[]>([]);
    const [showAll, setShowAll] = useState(false);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        setError('');

        const fetchCatalog = listAll
            ? listAll()
            : listRecipes().then((r) => r.recipes);

        Promise.all([sniff(headers), fetchCatalog])
            .then(([sniffRes, recipes]) => {
                if (cancelled) return;
                setSuggestions(sniffRes.suggestions ?? []);
                setCatalog(recipes ?? []);
            })
            .catch((err) => {
                if (cancelled) return;
                setError(extractErrorMessage(err));
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });

        return () => {
            cancelled = true;
        };
    }, [headers, sniff, listAll]);

    const recipesById = useMemo(() => {
        const map = new Map<string, Recipe>();
        for (const r of catalog) map.set(r.id, r);
        return map;
    }, [catalog]);

    const topThree = suggestions.slice(0, 3);
    const remainder = suggestions.slice(3);

    const handleSelect = (suggestion: RecipeSuggestion) => {
        const recipe = recipesById.get(suggestion.recipe_id);
        if (!recipe) return;
        onSelect(recipe, suggestion);
    };

    return (
        <div data-testid="recipe-picker">
            <div style={{ marginBottom: 'var(--space-md)' }}>
                <h3 style={{ margin: 0, fontSize: '1.1rem' }}>
                    Pick a recipe for this dataset
                </h3>
                <p
                    style={{
                        margin: 'var(--space-xs) 0 0',
                        color: 'var(--text-secondary)',
                        fontSize: '0.9rem',
                    }}
                >
                    A recipe bundles the task shape, adapter, scoring mode, and
                    suggested base model — so you don't have to wire these
                    together by hand. We ranked the recipes against the
                    columns in your file: <code>{headers.join(', ') || '—'}</code>.
                </p>
            </div>

            {loading && (
                <div
                    role="status"
                    style={{ color: 'var(--text-secondary)', padding: 'var(--space-md) 0' }}
                >
                    Loading recipe suggestions…
                </div>
            )}

            {error && (
                <div
                    role="alert"
                    style={{
                        padding: 'var(--space-md)',
                        background: 'var(--color-error-bg)',
                        color: 'var(--color-error)',
                        borderRadius: 'var(--radius-md)',
                        marginBottom: 'var(--space-md)',
                    }}
                >
                    {error}
                </div>
            )}

            {!loading && !error && suggestions.length === 0 && (
                <div
                    role="status"
                    style={{
                        padding: 'var(--space-md)',
                        background: 'var(--color-warning-bg)',
                        color: 'var(--color-warning)',
                        borderRadius: 'var(--radius-md)',
                    }}
                >
                    No recipe suggestions could be produced for these headers.
                    Use <em>Override</em> below to configure the import manually.
                </div>
            )}

            {!loading && !error && topThree.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                    {topThree.map((suggestion, idx) => (
                        <RecipeCard
                            key={suggestion.recipe_id}
                            suggestion={suggestion}
                            recipe={recipesById.get(suggestion.recipe_id)}
                            isTop={idx === 0}
                            onSelect={() => handleSelect(suggestion)}
                        />
                    ))}
                </div>
            )}

            {!loading && remainder.length > 0 && (
                <div style={{ marginTop: 'var(--space-md)' }}>
                    <button
                        type="button"
                        className="btn btn-ghost"
                        style={{ fontSize: '0.85rem' }}
                        onClick={() => setShowAll((v) => !v)}
                        data-testid="recipe-picker-toggle-all"
                    >
                        {showAll
                            ? 'Hide other recipes'
                            : `Show ${remainder.length} more recipe${remainder.length === 1 ? '' : 's'}`}
                    </button>

                    {showAll && (
                        <div
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 'var(--space-sm)',
                                marginTop: 'var(--space-sm)',
                            }}
                        >
                            {remainder.map((suggestion) => (
                                <RecipeCard
                                    key={suggestion.recipe_id}
                                    suggestion={suggestion}
                                    recipe={recipesById.get(suggestion.recipe_id)}
                                    isTop={false}
                                    onSelect={() => handleSelect(suggestion)}
                                />
                            ))}
                        </div>
                    )}
                </div>
            )}

            <div
                style={{
                    marginTop: 'var(--space-lg)',
                    paddingTop: 'var(--space-md)',
                    borderTop: '1px solid var(--border-color)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: 'var(--space-md)',
                }}
            >
                {onBack ? (
                    <button type="button" className="btn btn-ghost" onClick={onBack}>
                        ← Back
                    </button>
                ) : (
                    <span />
                )}
                <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={onOverride}
                    style={{ fontSize: '0.85rem' }}
                    data-testid="recipe-picker-override"
                    title="Skip recipe selection and configure mapping by hand."
                >
                    Override — configure manually →
                </button>
            </div>
        </div>
    );
}

interface RecipeCardProps {
    suggestion: RecipeSuggestion;
    recipe: Recipe | undefined;
    isTop: boolean;
    onSelect: () => void;
}

function RecipeCard({ suggestion, recipe, isTop, onSelect }: RecipeCardProps) {
    const tone = confidenceTone(suggestion.confidence);
    const matchedEntries = Object.entries(suggestion.matched_columns).filter(
        ([, header]) => header !== null && header !== undefined,
    ) as Array<[string, string]>;

    return (
        <div
            data-testid={`recipe-card-${suggestion.recipe_id}`}
            style={{
                padding: 'var(--space-md)',
                borderRadius: 'var(--radius-md)',
                border: isTop
                    ? '2px solid var(--accent-primary)'
                    : '1px solid var(--border-color)',
                background: 'var(--bg-card)',
            }}
        >
            <div
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 'var(--space-md)',
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                    <span style={{ fontSize: '1.4rem' }} aria-hidden="true">
                        {suggestion.icon || recipe?.icon || '🧪'}
                    </span>
                    <div>
                        <div style={{ fontWeight: 600 }}>{suggestion.recipe_name}</div>
                        {recipe?.headline && (
                            <div
                                style={{
                                    fontSize: '0.85rem',
                                    color: 'var(--text-secondary)',
                                    marginTop: 2,
                                }}
                            >
                                {recipe.headline}
                            </div>
                        )}
                    </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                    <span
                        className={`badge badge-${tone}`}
                        data-testid={`recipe-confidence-${suggestion.recipe_id}`}
                    >
                        {Math.round(suggestion.confidence * 100)}% · {confidenceLabel(suggestion.confidence, suggestion.fallback)}
                    </span>
                    {isTop && (
                        <span className="badge badge-accent" data-testid="recipe-recommended-badge">
                            Recommended
                        </span>
                    )}
                </div>
            </div>

            {recipe && (
                <div
                    style={{
                        marginTop: 'var(--space-sm)',
                        display: 'grid',
                        gridTemplateColumns: 'auto 1fr',
                        rowGap: 4,
                        columnGap: 'var(--space-md)',
                        fontSize: '0.85rem',
                        color: 'var(--text-secondary)',
                    }}
                >
                    <span>Task profile</span>
                    <code>{recipe.task_profile}</code>
                    <span>Scoring</span>
                    <code>{recipe.scoring_mode}</code>
                    <span>Suggested base</span>
                    <code>{recipe.suggested_base_model}</code>
                </div>
            )}

            {(matchedEntries.length > 0 || suggestion.fallback) && (
                <div
                    style={{
                        marginTop: 'var(--space-sm)',
                        padding: 'var(--space-sm)',
                        background: 'var(--bg-subtle)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.85rem',
                    }}
                    data-testid={`recipe-why-${suggestion.recipe_id}`}
                >
                    <strong>Why this recipe?</strong>{' '}
                    {suggestion.fallback ? (
                        <span>
                            None of the more specific recipes matched your columns,
                            so we're offering the generic instruction-tuning recipe
                            as a safe default. You can pick any other recipe with
                            the toggle above.
                        </span>
                    ) : (
                        <span>
                            {matchedEntries.map(([role, header], i) => (
                                <span key={role}>
                                    <code>{header}</code> looks like a{' '}
                                    <strong>{role}</strong> column
                                    {i < matchedEntries.length - 1 ? '; ' : '.'}
                                </span>
                            ))}
                        </span>
                    )}
                </div>
            )}

            <div style={{ marginTop: 'var(--space-md)', textAlign: 'right' }}>
                <button
                    type="button"
                    className={isTop ? 'btn btn-primary' : 'btn btn-secondary'}
                    onClick={onSelect}
                    data-testid={`recipe-select-${suggestion.recipe_id}`}
                >
                    {isTop ? 'Use this recipe →' : 'Use this instead'}
                </button>
            </div>
        </div>
    );
}
