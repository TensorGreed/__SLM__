/**
 * Panel for adding and managing trusted evaluation examples in gold dev and test datasets.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import api from '../../api/client';
import EmptyState from '../shared/EmptyState';
import StepFooter from '../shared/StepFooter';
import CoachStrip from '../coach/CoachStrip';
import LlmGoldGeneratePanel from './LlmGoldGeneratePanel';
import GoldEntryRowBody from './GoldEntryRowBody';
import type { GoldRowRecipe } from './GoldEntryRowBody';
import GoldEntryAddForm from './GoldEntryAddForm';
import type { GoldAddPayload } from './GoldEntryAddForm';
import './GoldSetPanel.css';


/** Recipes the LLM-assisted gold-gen panel handles. Mirrors
 *  ``SUPPORTED_RECIPES`` in ``backend/app/services/gold_llm_service.py``
 *  — keep in sync when extending. */
const SUPPORTED_LLM_GOLD_RECIPES = new Set([
    'qa-sft',
    'classification',
    'span-extraction',
    'summarization',
]);


/** Difficulty / trap filter for the entries list. Only meaningful
 *  for qa-sft; non-qa-sft recipes don't use these fields. ``all`` is
 *  the no-op default. */
type EntryFilter = 'all' | 'easy' | 'medium' | 'hard' | 'traps';


/** Normalize the difficulty field on an entry. Older entries (from
 *  before the LLM-gen path started tagging rows) may have ``undefined``
 *  or ``""`` — treat those as ``medium`` so the mix summary doesn't
 *  silently drop them. */
function normalizeDifficulty(raw: unknown): 'easy' | 'medium' | 'hard' {
    const token = String(raw || '').trim().toLowerCase();
    if (token === 'easy' || token === 'hard') return token;
    return 'medium';
}


/** Project templates (ticket-router, contract-clause-extractor,
 *  security-alert-summarizer, etc.) seed the gold JSONL through a
 *  shared materialization path that flattens EVERY recipe shape into
 *  the legacy ``{question, answer}`` Q+A keys. Specifically:
 *    * classification   → ``answer`` carries the label string verbatim
 *    * span-extraction  → ``answer`` is JSON.stringify({"entities": [...]})
 *    * summarization    → ``answer`` is JSON.stringify({"summary": "..."})
 *  Newer rows (LLM-gen + manual add via the per-recipe form) already
 *  carry recipe-shaped keys directly. This helper normalizes legacy
 *  rows into the recipe-shaped keys the panel's row renderer expects
 *  WITHOUT touching the underlying storage — keeping the on-disk JSONL
 *  consistent with what eval handlers (which read question/answer
 *  aliases) already accept.
 *
 *  Recipe-shaped keys on the input win over the legacy keys, so
 *  a half-migrated row (with both shapes present) renders the new
 *  shape. */
function normalizeEntryForRecipe(
    recipe: GoldRowRecipe,
    entry: Record<string, unknown>,
): Record<string, unknown> {
    if (recipe === 'qa-sft') {
        return entry;
    }
    const out: Record<string, unknown> = { ...entry };

    // Try to parse ``answer`` as JSON; many legacy rows have a
    // JSON-encoded dict there ({entities: [...]} or {summary: "..."}).
    const answer = entry.answer;
    let parsedAnswer: Record<string, unknown> | null = null;
    if (typeof answer === 'string' && answer.trim().startsWith('{')) {
        try {
            const candidate = JSON.parse(answer);
            if (
                candidate
                && typeof candidate === 'object'
                && !Array.isArray(candidate)
            ) {
                parsedAnswer = candidate as Record<string, unknown>;
            }
        } catch {
            // Not JSON — fall through. Classification's ``answer`` is
            // a plain string like "billing".
        }
    }

    if (recipe === 'classification') {
        if (out.text === undefined && typeof entry.question === 'string') {
            out.text = entry.question;
        }
        if (out.label === undefined) {
            // Legacy classification rows have the label flat in ``answer``.
            // Nested-under-``expected``.label is also possible from the
            // gold_set_workbench path.
            if (typeof answer === 'string' && !parsedAnswer) {
                out.label = answer;
            } else if (
                parsedAnswer
                && typeof parsedAnswer.label === 'string'
            ) {
                out.label = parsedAnswer.label;
            } else if (
                entry.expected
                && typeof entry.expected === 'object'
                && typeof (entry.expected as { label?: unknown }).label === 'string'
            ) {
                out.label = (entry.expected as { label: string }).label;
            }
        }
        return out;
    }

    if (recipe === 'span-extraction') {
        if (out.text === undefined && typeof entry.question === 'string') {
            out.text = entry.question;
        }
        if (!Array.isArray(out.entities)) {
            // Legacy: ``answer`` is JSON.stringify({"entities": [...]}).
            if (
                parsedAnswer
                && Array.isArray(parsedAnswer.entities)
            ) {
                out.entities = parsedAnswer.entities;
            } else if (
                entry.expected
                && typeof entry.expected === 'object'
                && Array.isArray((entry.expected as { entities?: unknown }).entities)
            ) {
                out.entities = (entry.expected as { entities: unknown[] }).entities;
            } else {
                // Negative example (no entities) — keep an empty array
                // so the row renders with the "negative example" hint
                // rather than blank.
                out.entities = [];
            }
        }
        return out;
    }

    if (recipe === 'summarization') {
        if (out.document === undefined && typeof entry.question === 'string') {
            out.document = entry.question;
        }
        if (out.summary === undefined) {
            // Legacy summarization rows have summary nested in the
            // JSON-encoded ``answer`` dict, OR plain in ``answer``.
            if (
                parsedAnswer
                && typeof parsedAnswer.summary === 'string'
            ) {
                out.summary = parsedAnswer.summary;
            } else if (typeof answer === 'string' && !parsedAnswer) {
                out.summary = answer;
            } else if (
                entry.expected
                && typeof entry.expected === 'object'
                && typeof (entry.expected as { summary?: unknown }).summary === 'string'
            ) {
                out.summary = (entry.expected as { summary: string }).summary;
            }
        }
        return out;
    }

    return out;
}

interface GoldSetPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

export default function GoldSetPanel({ projectId, onNextStep }: GoldSetPanelProps) {
    const [entries, setEntries] = useState<any[]>([]);
    const [datasetType, setDatasetType] = useState('gold_dev');
    // Recipe id flows down to the LLM-generate panel so it can build
    // the right prompt shape + render per-recipe row previews. The
    // panel handles its own self-hiding for unsupported recipes.
    const [recipeId, setRecipeId] = useState<string | null>(null);
    // Filter for the entries list. Only meaningful for qa-sft (other
    // recipes don't carry difficulty / hallucination-trap data); the
    // filter dropdown is hidden + the filter resets to ``all`` when
    // the recipe changes.
    const [entryFilter, setEntryFilter] = useState<EntryFilter>('all');

    const fetchEntries = useCallback(async () => {
        const res = await api.get(`/projects/${projectId}/gold/entries?dataset_type=${datasetType}`);
        setEntries(res.data.entries || []);
    }, [projectId, datasetType]);

    useEffect(() => { fetchEntries(); }, [fetchEntries]);

    // Filter resets when the recipe changes (e.g. user navigated to
    // a different project mid-session) so a stale qa-sft "hard only"
    // filter doesn't accidentally apply to a classification project.
    useEffect(() => {
        setEntryFilter('all');
    }, [recipeId]);

    /** Known span-extraction entity types — pulled from the current
     *  entries (after normalization) so the add-form's helper Type
     *  input lets the user reuse the existing type vocabulary
     *  instead of typing each one from scratch. Empty list when the
     *  recipe isn't span-extraction (the form branch doesn't render
     *  the helper at all in that case). */
    const knownSpanTypes = useMemo(() => {
        if (recipeId !== 'span-extraction') return [];
        const seen = new Set<string>();
        const out: string[] = [];
        for (const e of entries) {
            // Run each entry through the normalizer first so legacy
            // template rows (entities JSON-encoded in ``answer``)
            // contribute their types alongside the recipe-shaped
            // ones.
            const normalized = normalizeEntryForRecipe('span-extraction', e);
            const ents = normalized.entities;
            if (!Array.isArray(ents)) continue;
            for (const ent of ents) {
                if (!ent || typeof ent !== 'object') continue;
                const t = (ent as { type?: unknown }).type;
                if (typeof t !== 'string') continue;
                const trimmed = t.trim();
                if (!trimmed) continue;
                const key = trimmed.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                out.push(trimmed);
            }
        }
        return out.sort();
    }, [entries, recipeId]);

    /** Known classification labels — pulled from the current entries
     *  so the add-form's combobox lets the user reuse the existing
     *  label vocabulary instead of inventing new tokens per row.
     *  Empty list when the recipe isn't classification (the form
     *  branch doesn't render a label input in that case). */
    const knownClassificationLabels = useMemo(() => {
        if (recipeId !== 'classification') return [];
        const seen = new Set<string>();
        const out: string[] = [];
        for (const e of entries) {
            const candidates: unknown[] = [
                e.label,
                // Some import paths write the label nested under
                // ``expected.label`` (template-instantiated rows).
                (e.expected && typeof e.expected === 'object')
                    ? (e.expected as { label?: unknown }).label
                    : undefined,
            ];
            for (const c of candidates) {
                if (typeof c !== 'string') continue;
                const trimmed = c.trim();
                if (!trimmed) continue;
                const key = trimmed.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                out.push(trimmed);
            }
        }
        return out.sort();
    }, [entries, recipeId]);

    /** Mix summary + filtered entries. qa-sft only — for other recipes
     *  the summary is null + the filter is a no-op (filter dropdown
     *  is hidden, so this branch is unreachable from the UI). */
    const { mixSummary, filteredEntries } = useMemo(() => {
        if (recipeId !== 'qa-sft') {
            return {
                mixSummary: null as {
                    total: number; easy: number; medium: number; hard: number; traps: number;
                } | null,
                filteredEntries: entries,
            };
        }
        const summary = {
            total: entries.length,
            easy: 0,
            medium: 0,
            hard: 0,
            traps: 0,
        };
        for (const e of entries) {
            const d = normalizeDifficulty(e.difficulty);
            summary[d] += 1;
            if (e.is_hallucination_trap) summary.traps += 1;
        }
        const filtered = entries.filter((e) => {
            if (entryFilter === 'all') return true;
            if (entryFilter === 'traps') return !!e.is_hallucination_trap;
            return normalizeDifficulty(e.difficulty) === entryFilter;
        });
        return { mixSummary: summary, filteredEntries: filtered };
    }, [entries, entryFilter, recipeId]);

    // One-shot fetch of the project's selected_recipe so we can
    // gate the LLM-generate panel without changing this component's
    // prop contract.
    useEffect(() => {
        let cancelled = false;
        api.get(`/projects/${projectId}`)
            .then((res) => {
                if (cancelled) return;
                const sr = (res.data as { selected_recipe?: { recipe_id?: string } })
                    ?.selected_recipe;
                setRecipeId((sr && sr.recipe_id) || null);
            })
            .catch(() => {
                if (!cancelled) setRecipeId(null);
            });
        return () => { cancelled = true; };
    }, [projectId]);

    /** Submit handler passed down to ``GoldEntryAddForm``. The form
     *  builds the recipe-shaped payload; we attach ``dataset_type``
     *  and re-fetch on success. Errors surface inside the form. */
    const handleAddRow = async (payload: GoldAddPayload) => {
        await api.post(`/projects/${projectId}/gold/add`, {
            ...payload,
            dataset_type: datasetType,
        });
        await fetchEntries();
    };

    const handleLock = async () => {
        if (!confirm('Lock this dataset? No more entries can be added.')) return;
        await api.post(`/projects/${projectId}/gold/lock?dataset_type=${datasetType}`);
    };

    return (
        <div className="gold-panel animate-fade-in">
            <div className="card">
                <div className="gold-header">
                    <h3>Gold Evaluation Dataset</h3>
                    <div className="gold-controls">
                        <select className="input" value={datasetType} onChange={e => setDatasetType(e.target.value)} style={{ width: 'auto' }}>
                            <option value="gold_dev">Dev Set</option>
                            <option value="gold_test">Test Set</option>
                        </select>
                        <button className="btn btn-secondary" onClick={handleLock}>🔒 Lock</button>
                    </div>
                </div>

                <CoachStrip projectId={projectId} stage="gold_set" />

                {recipeId && SUPPORTED_LLM_GOLD_RECIPES.has(recipeId) && (
                    <LlmGoldGeneratePanel
                        projectId={projectId}
                        datasetType={datasetType}
                        recipeId={recipeId as GoldRowRecipe}
                        onRowsSaved={fetchEntries}
                    />
                )}

                {/* Per-recipe add form. Hidden entirely when no
                    recipe is set — the row shape is undecided so a
                    Q/A form would mislead the user. Recipe narrowing
                    matches GoldEntryRowBody's: anything unsupported
                    or null hides the form. */}
                {recipeId && SUPPORTED_LLM_GOLD_RECIPES.has(recipeId) ? (
                    <GoldEntryAddForm
                        recipeId={recipeId as GoldRowRecipe}
                        knownLabels={knownClassificationLabels}
                        knownSpanTypes={knownSpanTypes}
                        onAdd={handleAddRow}
                    />
                ) : (
                    <div
                        data-testid="gold-add-form-hidden-hint"
                        style={{
                            padding: 'var(--space-md)',
                            color: 'var(--text-tertiary)',
                            fontSize: '0.9rem',
                        }}
                    >
                        Pick a recipe (Project Settings → Recipe) to
                        unlock manual gold-row entry. The form's
                        fields depend on the recipe shape.
                    </div>
                )}
            </div>

            <div className="card">
                <div
                    className="gold-header"
                    data-testid="gold-entries-header"
                    style={{ flexWrap: 'wrap' }}
                >
                    <h3>
                        Entries <span className="badge badge-accent">{entries.length}</span>
                    </h3>
                    {recipeId === 'qa-sft' && entries.length > 0 && (
                        <div
                            className="gold-controls"
                            data-testid="gold-entries-filter-row"
                        >
                            <label
                                className="form-label"
                                htmlFor="gold-entries-filter"
                                style={{ margin: 0, fontWeight: 400 }}
                            >
                                Filter:
                            </label>
                            <select
                                id="gold-entries-filter"
                                className="input"
                                value={entryFilter}
                                onChange={(e) => setEntryFilter(e.target.value as EntryFilter)}
                                data-testid="gold-entries-filter"
                                style={{ width: 'auto' }}
                            >
                                <option value="all">All</option>
                                <option value="easy">Easy only</option>
                                <option value="medium">Medium only</option>
                                <option value="hard">Hard only</option>
                                <option value="traps">Hallucination traps only</option>
                            </select>
                        </div>
                    )}
                </div>

                {/* Mix summary — qa-sft only. Helps the user see the
                    current difficulty distribution at a glance so they
                    can ask the LLM-gen panel to fill the gaps in the
                    next Generate click. */}
                {recipeId === 'qa-sft' && mixSummary && mixSummary.total > 0 && (
                    <div
                        data-testid="gold-entries-mix-summary"
                        style={{
                            margin: '0 0 var(--space-sm)',
                            padding: 'var(--space-sm) var(--space-md)',
                            background: 'var(--bg-subtle)',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: '0.9rem',
                            color: 'var(--text-secondary)',
                        }}
                    >
                        <strong>{mixSummary.total}</strong>{' '}
                        entr{mixSummary.total === 1 ? 'y' : 'ies'}:{' '}
                        <span data-testid="gold-entries-mix-easy">
                            {mixSummary.easy} easy
                        </span>
                        {' / '}
                        <span data-testid="gold-entries-mix-medium">
                            {mixSummary.medium} medium
                        </span>
                        {' / '}
                        <span data-testid="gold-entries-mix-hard">
                            {mixSummary.hard} hard
                        </span>
                        {' · '}
                        <span data-testid="gold-entries-mix-traps">
                            {mixSummary.traps} hallucination trap
                            {mixSummary.traps === 1 ? '' : 's'}
                        </span>
                    </div>
                )}

                {/* Active-filter banner: when filtered, surface how
                    many of the total are showing so the user isn't
                    confused by the truncated list. */}
                {entryFilter !== 'all' && filteredEntries.length !== entries.length && (
                    <div
                        data-testid="gold-entries-filter-banner"
                        style={{
                            margin: '0 0 var(--space-sm)',
                            fontSize: '0.85rem',
                            color: 'var(--text-tertiary)',
                        }}
                    >
                        Showing <strong>{filteredEntries.length}</strong> of{' '}
                        {entries.length} (filter: <em>{entryFilter}</em>).
                        {' '}
                        <button
                            type="button"
                            className="btn btn-link"
                            onClick={() => setEntryFilter('all')}
                            data-testid="gold-entries-filter-clear"
                            style={{ padding: 0 }}
                        >
                            Clear filter
                        </button>
                    </div>
                )}

                <div className="entries-list">
                    {filteredEntries.map((e, i) => {
                        // Recipe narrowing: anything unsupported or
                        // null falls back to qa-sft for backward
                        // compat with rows saved before recipe-aware
                        // import landed.
                        const renderRecipe: GoldRowRecipe = (
                            recipeId && SUPPORTED_LLM_GOLD_RECIPES.has(recipeId)
                                ? recipeId
                                : 'qa-sft'
                        ) as GoldRowRecipe;
                        // Templates pre-date the per-recipe panel and
                        // shove every shape into ``{question, answer}``
                        // on disk. Normalize so non-qa-sft rows render
                        // their recipe-shaped fields instead of blank
                        // divs. qa-sft rows pass through unchanged.
                        const normalized = normalizeEntryForRecipe(renderRecipe, e);
                        return (
                            <div
                                key={i}
                                className="entry-item"
                                data-testid={`gold-entry-row-${i}`}
                            >
                                <GoldEntryRowBody
                                    recipeId={renderRecipe}
                                    row={{
                                        ...normalized,
                                        // Normalize so the qa-sft
                                        // badge renders "medium"
                                        // rather than "" / undefined
                                        // for legacy rows.
                                        difficulty: normalizeDifficulty(
                                            normalized.difficulty,
                                        ),
                                    }}
                                    testidPrefix={`gold-entry-row-${i}`}
                                />
                            </div>
                        );
                    })}
                    {entries.length === 0 && (
                        <EmptyState
                            title="No gold-set entries yet"
                            description="The gold set is the labelled ground-truth eval set. Add Q&A pairs above — 50–100 carefully labelled rows is enough to start scoring training runs."
                            docsHref="http://localhost:3001/docs/workflows/evaluation-and-remediation"
                        />
                    )}
                    {entries.length > 0 && filteredEntries.length === 0 && (
                        <div
                            data-testid="gold-entries-filtered-empty"
                            style={{
                                padding: 'var(--space-lg)',
                                textAlign: 'center',
                                color: 'var(--text-tertiary)',
                            }}
                        >
                            No entries match the current filter.
                        </div>
                    )}
                </div>
            </div>

            {onNextStep && (
                <StepFooter
                    currentStep="Gold Dataset"
                    nextStep="Synthetic Generation"
                    nextStepIcon="🧪"
                    isComplete={entries.length >= 5}
                    hint="Add at least 5 Q&A pairs for evaluation"
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
