/**
 * GoldEntryAddForm — per-recipe inline form for manually adding a
 * gold-set row. Mirrors the per-recipe row-rendering pattern from
 * [GoldEntryRowBody](GoldEntryRowBody.tsx) so the add UX matches
 * what the user sees in the entries list and the LLM-gen preview.
 *
 * Field shapes per recipe:
 *   * qa-sft         → Question + Answer + difficulty + trap toggle
 *   * classification → Text + Label (combobox of known labels)
 *   * span-extraction→ Text + Entities JSON editor (live validation)
 *   * summarization  → Document (long) + Summary
 *
 * The component manages its own field state. On submit it builds a
 * recipe-shaped payload and calls ``onAdd`` — the parent handles the
 * actual POST + post-success cleanup (refetch, etc.). Local form
 * state is reset on successful submit.
 *
 * The form deliberately does NOT live inside `GoldSetPanel.tsx`
 * because the manual-add concerns (per-field validation, JSON editor,
 * label combobox) duplicate enough logic that inlining would make
 * the panel hard to read.
 */

import { useEffect, useMemo, useState } from 'react';

import type { GoldRowRecipe, GoldRowSpan } from './GoldEntryRowBody';


/** Payload built by the form on submit. Maps directly onto the
 *  ``GoldRowCreate`` wire shape that ``POST /gold/add`` accepts. */
export interface GoldAddPayload {
    [key: string]: unknown;
    // qa-sft
    question?: string;
    answer?: string;
    difficulty?: string;
    is_hallucination_trap?: boolean;
    // classification
    text?: string;
    label?: string;
    // span-extraction
    entities?: GoldRowSpan[];
    // summarization
    document?: string;
    summary?: string;
}


interface Props {
    recipeId: GoldRowRecipe;
    /** Existing classification labels (extracted from the current
     *  gold rows) — populates the combobox so users tend to reuse
     *  the existing label vocabulary instead of inventing new ones
     *  per row. */
    knownLabels?: string[];
    /** Called when the user clicks the add button with a fully
     *  validated payload. Returns a promise so the form can disable
     *  inputs / reset on success. Throws / rejects on error so the
     *  form can surface it without resetting fields the user typed. */
    onAdd: (payload: GoldAddPayload) => Promise<void>;
}


/** Span-extraction JSON validation result. */
interface SpanJsonValidation {
    spans: GoldRowSpan[];
    error: string | null;
}


function validateSpansJson(jsonText: string, text: string): SpanJsonValidation {
    const trimmed = jsonText.trim();
    if (!trimmed) {
        return { spans: [], error: null }; // empty == negative example
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(trimmed);
    } catch (err) {
        return {
            spans: [],
            error: `Invalid JSON: ${err instanceof Error ? err.message : 'parse error'}`,
        };
    }
    if (!Array.isArray(parsed)) {
        return {
            spans: [],
            error: 'Expected a JSON array of {type, start, end, text} objects.',
        };
    }
    const out: GoldRowSpan[] = [];
    for (let i = 0; i < parsed.length; i += 1) {
        const ent = parsed[i] as Record<string, unknown>;
        if (!ent || typeof ent !== 'object') {
            return { spans: [], error: `Entity ${i}: expected an object.` };
        }
        const type = String(ent.type || '').trim();
        const start = Number(ent.start);
        const end = Number(ent.end);
        const claimedText = String(ent.text || '');
        if (!type) {
            return { spans: [], error: `Entity ${i}: missing or empty "type".` };
        }
        if (!Number.isInteger(start) || start < 0) {
            return { spans: [], error: `Entity ${i}: "start" must be a non-negative integer.` };
        }
        if (!Number.isInteger(end) || end <= start) {
            return { spans: [], error: `Entity ${i}: "end" must be an integer greater than "start".` };
        }
        if (end > text.length) {
            return {
                spans: [],
                error: `Entity ${i}: "end" (${end}) exceeds text length (${text.length}).`,
            };
        }
        const sliced = text.slice(start, end);
        // Whitespace-tolerant match — same as the LLM-gen parser so
        // hand-typed offsets behave consistently with model output.
        if (claimedText && sliced.trim() !== claimedText.trim()) {
            return {
                spans: [],
                error: (
                    `Entity ${i}: offset mismatch — text[${start}:${end}] = `
                    + `"${sliced}" but you claimed "${claimedText}".`
                ),
            };
        }
        out.push({ type, start, end, text: sliced });
    }
    return { spans: out, error: null };
}


export default function GoldEntryAddForm({
    recipeId,
    knownLabels = [],
    onAdd,
}: Props) {
    // qa-sft state
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [difficulty, setDifficulty] = useState('medium');
    const [isTrap, setIsTrap] = useState(false);

    // classification state
    const [text, setText] = useState('');
    const [label, setLabel] = useState('');

    // span-extraction state
    const [spanText, setSpanText] = useState('');
    const [entitiesJson, setEntitiesJson] = useState('');
    // Highlight-to-select helper state. The user highlights a range
    // in the Text textarea, types an entity type, clicks "Add
    // highlighted span" — we capture the textarea's selection range
    // at click time and append the span to the entities JSON without
    // the user having to count characters by hand.
    const [selectionStart, setSelectionStart] = useState<number | null>(null);
    const [selectionEnd, setSelectionEnd] = useState<number | null>(null);
    const [selectionType, setSelectionType] = useState('');

    // summarization state
    const [doc, setDoc] = useState('');
    const [summary, setSummary] = useState('');

    const [submitting, setSubmitting] = useState(false);
    const [submitError, setSubmitError] = useState<string | null>(null);

    // Reset field state when the recipe changes so a stale
    // classification-label doesn't leak into the qa-sft form after
    // a recipe switch.
    useEffect(() => {
        setQuestion(''); setAnswer(''); setDifficulty('medium'); setIsTrap(false);
        setText(''); setLabel('');
        setSpanText(''); setEntitiesJson('');
        setSelectionStart(null); setSelectionEnd(null); setSelectionType('');
        setDoc(''); setSummary('');
        setSubmitError(null);
    }, [recipeId]);

    // Editing the source text invalidates any captured offsets —
    // a selection of [8:24] meant "jane@example.com" in the old
    // text but means something else (or is out of range) now.
    // Clearing on edit is friendlier than silently producing bad
    // spans when the user clicks "Add highlighted span".
    useEffect(() => {
        setSelectionStart(null);
        setSelectionEnd(null);
    }, [spanText]);

    /** Live span validation — runs as the user types so they see the
     *  offset error before they click submit. */
    const spanValidation = useMemo(
        () => (recipeId === 'span-extraction'
            ? validateSpansJson(entitiesJson, spanText)
            : { spans: [], error: null }),
        [recipeId, entitiesJson, spanText],
    );

    /** Current selection in the Text textarea. Only meaningful when
     *  start < end AND both are within ``spanText``'s length. */
    const selection = useMemo(() => {
        if (
            selectionStart === null
            || selectionEnd === null
            || selectionStart >= selectionEnd
            || selectionEnd > spanText.length
        ) {
            return null;
        }
        return {
            start: selectionStart,
            end: selectionEnd,
            text: spanText.slice(selectionStart, selectionEnd),
        };
    }, [selectionStart, selectionEnd, spanText]);

    /** Whether the entities JSON textarea is in a parseable state. We
     *  refuse to "Add highlighted span" while it's broken — appending
     *  would either drop the user's invalid hand edits or produce
     *  more broken JSON. They have to fix it first. */
    const entitiesJsonParseable = useMemo(() => {
        const trimmed = entitiesJson.trim();
        if (!trimmed) return true;  // empty == empty array, fine
        try {
            const parsed = JSON.parse(trimmed);
            return Array.isArray(parsed);
        } catch {
            return false;
        }
    }, [entitiesJson]);

    const canAddHighlightedSpan = (
        !!selection
        && selectionType.trim().length > 0
        && entitiesJsonParseable
    );

    /** Append the currently-highlighted span to the entities JSON.
     *  Preserves any hand-edited entries — we parse the current
     *  array, append, and pretty-print. Refuses to run when the
     *  JSON is broken (caller validates via ``canAddHighlightedSpan``). */
    const handleAddHighlightedSpan = () => {
        if (!canAddHighlightedSpan || !selection) return;
        const trimmed = entitiesJson.trim();
        let existing: unknown[] = [];
        if (trimmed) {
            try {
                const parsed = JSON.parse(trimmed);
                if (Array.isArray(parsed)) {
                    existing = parsed;
                }
            } catch {
                // Guarded above — defensive no-op.
                return;
            }
        }
        const newSpan = {
            type: selectionType.trim(),
            start: selection.start,
            end: selection.end,
            text: selection.text,
        };
        const next = [...existing, newSpan];
        setEntitiesJson(JSON.stringify(next, null, 2));
        // Clear the type input so the user is primed to add another
        // span of a different type next; keep the selection in case
        // they want to label the same range with a second tag.
        setSelectionType('');
    };

    const canSubmit = useMemo(() => {
        if (submitting) return false;
        switch (recipeId) {
            case 'qa-sft':
                return question.trim().length > 0 && answer.trim().length > 0;
            case 'classification':
                return text.trim().length > 0 && label.trim().length > 0;
            case 'span-extraction':
                return spanText.trim().length > 0 && spanValidation.error === null;
            case 'summarization':
                return doc.trim().length > 0 && summary.trim().length > 0;
            default:
                return false;
        }
    }, [
        recipeId, submitting, question, answer, text, label,
        spanText, spanValidation.error, doc, summary,
    ]);

    const handleSubmit = async () => {
        if (!canSubmit) return;
        setSubmitting(true);
        setSubmitError(null);
        let payload: GoldAddPayload;
        switch (recipeId) {
            case 'qa-sft':
                payload = {
                    question: question.trim(),
                    answer: answer.trim(),
                    difficulty,
                    is_hallucination_trap: isTrap,
                };
                break;
            case 'classification':
                payload = { text: text.trim(), label: label.trim() };
                break;
            case 'span-extraction':
                payload = {
                    text: spanText,  // preserve whitespace — offsets indexed against it
                    entities: spanValidation.spans,
                };
                break;
            case 'summarization':
                payload = { document: doc.trim(), summary: summary.trim() };
                break;
        }
        try {
            await onAdd(payload);
            // Reset on success.
            setQuestion(''); setAnswer(''); setIsTrap(false);
            setText(''); setLabel('');
            setSpanText(''); setEntitiesJson('');
            setDoc(''); setSummary('');
        } catch (err) {
            const msg = err instanceof Error ? err.message : 'Failed to add row';
            setSubmitError(msg);
        } finally {
            setSubmitting(false);
        }
    };

    // Render — fields differ by recipe; the submit button is shared.
    const submitLabel = recipeId === 'qa-sft'
        ? '+ Add Pair'
        : '+ Add Row';

    return (
        <div className="qa-form" data-testid="gold-add-form">
            {recipeId === 'qa-sft' && (
                <>
                    <div className="form-group">
                        <label className="form-label">Question</label>
                        <input
                            className="input"
                            placeholder="Enter a question..."
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            data-testid="gold-add-question"
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Expected Answer</label>
                        <textarea
                            className="input gold-textarea"
                            placeholder="Expected answer..."
                            value={answer}
                            onChange={(e) => setAnswer(e.target.value)}
                            data-testid="gold-add-answer"
                        />
                    </div>
                    <div className="form-row">
                        <select
                            aria-label="Difficulty"
                            className="input"
                            value={difficulty}
                            onChange={(e) => setDifficulty(e.target.value)}
                            style={{ width: 'auto' }}
                            data-testid="gold-add-difficulty"
                        >
                            <option value="easy">Easy</option>
                            <option value="medium">Medium</option>
                            <option value="hard">Hard</option>
                        </select>
                        <label
                            className="form-label"
                            style={{ display: 'flex', alignItems: 'center', gap: 4 }}
                        >
                            <input
                                type="checkbox"
                                checked={isTrap}
                                onChange={(e) => setIsTrap(e.target.checked)}
                                data-testid="gold-add-trap"
                            />
                            Hallucination Trap
                        </label>
                    </div>
                </>
            )}

            {recipeId === 'classification' && (
                <>
                    <div className="form-group">
                        <label className="form-label" htmlFor="gold-add-text-input">
                            Text
                        </label>
                        <textarea
                            id="gold-add-text-input"
                            className="input gold-textarea"
                            placeholder="The text to classify..."
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            data-testid="gold-add-text"
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="gold-add-label-input">
                            Label
                            {knownLabels.length > 0 && (
                                <span
                                    style={{
                                        color: 'var(--text-tertiary)',
                                        fontWeight: 400,
                                        marginLeft: 4,
                                    }}
                                    data-testid="gold-add-label-hint"
                                >
                                    (existing: {knownLabels.join(', ')})
                                </span>
                            )}
                        </label>
                        <input
                            id="gold-add-label-input"
                            className="input"
                            placeholder="e.g. positive, billing, spam"
                            value={label}
                            onChange={(e) => setLabel(e.target.value)}
                            list={knownLabels.length > 0 ? 'gold-add-known-labels' : undefined}
                            data-testid="gold-add-label"
                            style={{ fontFamily: 'monospace' }}
                        />
                        {knownLabels.length > 0 && (
                            <datalist id="gold-add-known-labels">
                                {knownLabels.map((l) => (
                                    <option key={l} value={l} />
                                ))}
                            </datalist>
                        )}
                    </div>
                </>
            )}

            {recipeId === 'span-extraction' && (
                <>
                    <div className="form-group">
                        <label className="form-label" htmlFor="gold-add-span-text-input">
                            Text
                        </label>
                        <textarea
                            id="gold-add-span-text-input"
                            className="input gold-textarea"
                            placeholder="The source text spans are indexed against..."
                            value={spanText}
                            onChange={(e) => setSpanText(e.target.value)}
                            onSelect={(e) => {
                                // Capture the current selection range
                                // so the "Add highlighted span" button
                                // can append it without the user
                                // counting characters by hand. Fires
                                // on every selection change including
                                // simple cursor moves — start === end
                                // is handled as "no selection" downstream.
                                const ta = e.currentTarget;
                                setSelectionStart(ta.selectionStart);
                                setSelectionEnd(ta.selectionEnd);
                            }}
                            data-testid="gold-add-span-text"
                        />
                    </div>

                    {/* Highlight-to-select helper. Sits between the
                        Text textarea (where the user highlights) and
                        the Entities JSON editor (where the appended
                        span lands). The button is disabled until
                        BOTH a non-empty selection AND a non-empty
                        type are present — common UX failure mode is
                        clicking with one but not the other. */}
                    <div
                        className="form-group"
                        data-testid="gold-add-span-helper"
                        style={{
                            display: 'flex',
                            alignItems: 'flex-end',
                            gap: 'var(--space-sm)',
                            flexWrap: 'wrap',
                        }}
                    >
                        <div style={{ flex: '0 0 auto' }}>
                            <label
                                className="form-label"
                                htmlFor="gold-add-span-helper-type"
                            >
                                Type
                            </label>
                            <input
                                id="gold-add-span-helper-type"
                                className="input"
                                placeholder="e.g. email, phone, person"
                                value={selectionType}
                                onChange={(e) => setSelectionType(e.target.value)}
                                data-testid="gold-add-span-helper-type"
                                style={{ fontFamily: 'monospace', width: '12em' }}
                            />
                        </div>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={handleAddHighlightedSpan}
                            disabled={!canAddHighlightedSpan}
                            data-testid="gold-add-span-helper-add"
                            title={
                                !entitiesJsonParseable
                                    ? 'Fix the JSON below before adding more spans.'
                                    : !selection
                                        ? 'Highlight a range in the Text textarea first.'
                                        : !selectionType.trim()
                                            ? 'Enter an entity type first.'
                                            : undefined
                            }
                        >
                            + Add highlighted span
                        </button>
                        {/* Selection preview — surfaces what the
                            button will append so the user verifies
                            BEFORE clicking. No selection → a hint
                            telling them how to make one. */}
                        <div
                            data-testid="gold-add-span-helper-preview"
                            style={{
                                flex: 1,
                                minWidth: 0,
                                fontSize: '0.85rem',
                                color: 'var(--text-secondary)',
                            }}
                        >
                            {selection ? (
                                <>
                                    Selected{' '}
                                    <span style={{ fontFamily: 'monospace' }}>
                                        "{selection.text}"
                                    </span>{' '}
                                    <span style={{ color: 'var(--text-tertiary)' }}>
                                        [{selection.start}:{selection.end}]
                                    </span>
                                </>
                            ) : (
                                <span style={{ color: 'var(--text-tertiary)' }}>
                                    Highlight a range in the Text
                                    above to capture its offsets.
                                </span>
                            )}
                        </div>
                    </div>

                    <div className="form-group">
                        <label className="form-label" htmlFor="gold-add-entities-input">
                            Entities JSON
                            <span
                                style={{
                                    color: 'var(--text-tertiary)',
                                    fontWeight: 400,
                                    marginLeft: 4,
                                }}
                            >
                                — array of {'{type, start, end, text}'} (empty = negative example)
                            </span>
                        </label>
                        <textarea
                            id="gold-add-entities-input"
                            className="input gold-textarea"
                            placeholder='[{"type":"email","start":8,"end":24,"text":"jane@example.com"}]'
                            value={entitiesJson}
                            onChange={(e) => setEntitiesJson(e.target.value)}
                            data-testid="gold-add-entities"
                            style={{
                                fontFamily: 'monospace',
                                minHeight: '6em',
                            }}
                        />
                        {spanValidation.error && (
                            <div
                                role="alert"
                                data-testid="gold-add-entities-error"
                                style={{
                                    marginTop: 4,
                                    fontSize: '0.85rem',
                                    color: 'var(--color-error)',
                                }}
                            >
                                {spanValidation.error}
                            </div>
                        )}
                        {!spanValidation.error && entitiesJson.trim() && (
                            <div
                                style={{
                                    marginTop: 4,
                                    fontSize: '0.85rem',
                                    color: 'var(--text-secondary)',
                                }}
                                data-testid="gold-add-entities-valid"
                            >
                                ✓ {spanValidation.spans.length}{' '}
                                entit{spanValidation.spans.length === 1 ? 'y' : 'ies'} parsed
                                {' + offsets verified against the text.'}
                            </div>
                        )}
                    </div>
                </>
            )}

            {recipeId === 'summarization' && (
                <>
                    <div className="form-group">
                        <label className="form-label" htmlFor="gold-add-document-input">
                            Document
                        </label>
                        <textarea
                            id="gold-add-document-input"
                            className="input gold-textarea"
                            placeholder="The long-form source text..."
                            value={doc}
                            onChange={(e) => setDoc(e.target.value)}
                            rows={6}
                            data-testid="gold-add-document"
                            style={{ minHeight: '8em' }}
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="gold-add-summary-input">
                            Summary
                        </label>
                        <textarea
                            id="gold-add-summary-input"
                            className="input gold-textarea"
                            placeholder="The reference summary (1-5 sentences)..."
                            value={summary}
                            onChange={(e) => setSummary(e.target.value)}
                            rows={3}
                            data-testid="gold-add-summary"
                        />
                    </div>
                </>
            )}

            {submitError && (
                <div
                    role="alert"
                    data-testid="gold-add-error"
                    style={{
                        padding: 'var(--space-sm)',
                        background: 'var(--color-error-bg)',
                        color: 'var(--color-error)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.9rem',
                    }}
                >
                    {submitError}
                </div>
            )}

            <div className="form-row">
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleSubmit}
                    disabled={!canSubmit}
                    data-testid="gold-add-submit"
                >
                    {submitting ? '⏳ Adding…' : submitLabel}
                </button>
            </div>
        </div>
    );
}
