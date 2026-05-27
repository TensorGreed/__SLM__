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

interface GoldSetPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

export default function GoldSetPanel({ projectId, onNextStep }: GoldSetPanelProps) {
    const [entries, setEntries] = useState<any[]>([]);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [difficulty, setDifficulty] = useState('medium');
    const [isHallucTrap, setIsHallucTrap] = useState(false);
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

    const handleAdd = async () => {
        if (!question.trim() || !answer.trim()) return;
        await api.post(`/projects/${projectId}/gold/add`, {
            question, answer, dataset_type: datasetType, difficulty, is_hallucination_trap: isHallucTrap,
        });
        setQuestion(''); setAnswer('');
        fetchEntries();
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
                        recipeId={recipeId}
                        onRowsSaved={fetchEntries}
                    />
                )}

                <div className="qa-form">
                    <div className="form-group">
                        <label className="form-label">Question</label>
                        <input className="input" placeholder="Enter a question..." value={question} onChange={e => setQuestion(e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Expected Answer</label>
                        <textarea className="input gold-textarea" placeholder="Expected answer..." value={answer} onChange={e => setAnswer(e.target.value)} />
                    </div>
                    <div className="form-row">
                        <select className="input" value={difficulty} onChange={e => setDifficulty(e.target.value)} style={{ width: 'auto' }}>
                            <option value="easy">Easy</option>
                            <option value="medium">Medium</option>
                            <option value="hard">Hard</option>
                        </select>
                        <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <input type="checkbox" checked={isHallucTrap} onChange={e => setIsHallucTrap(e.target.checked)} />
                            Hallucination Trap
                        </label>
                        <button className="btn btn-primary" onClick={handleAdd}>+ Add Pair</button>
                    </div>
                </div>
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
                    {filteredEntries.map((e, i) => (
                        <div
                            key={i}
                            className="entry-item"
                            data-testid={`gold-entry-row-${i}`}
                        >
                            {/* Render per-recipe via the shared body
                                component so the LLM-gen preview and
                                this list don't drift. qa-sft falls
                                through as the default — earlier
                                projects had a recipe but new ones
                                still go through this path with a
                                non-null recipeId. ``unknown`` projects
                                (recipeId === null) render as qa-sft
                                for backward compat with rows saved
                                before recipe-aware import landed. */}
                            <GoldEntryRowBody
                                recipeId={
                                    (recipeId && SUPPORTED_LLM_GOLD_RECIPES.has(recipeId)
                                        ? recipeId
                                        : 'qa-sft') as GoldRowRecipe
                                }
                                row={{
                                    ...e,
                                    // Normalize so the qa-sft badge
                                    // renders "medium" rather than
                                    // "" / undefined for legacy rows.
                                    difficulty: normalizeDifficulty(e.difficulty),
                                }}
                                testidPrefix={`gold-entry-row-${i}`}
                            />
                        </div>
                    ))}
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
