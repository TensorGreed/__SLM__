/**
 * GoldEntryRowBody — shared per-recipe body renderer for a single
 * gold-set row. Used in two contexts so they don't drift:
 *   * LlmGoldGeneratePanel preview rows (testidPrefix
 *     ``llm-gold-preview-row-<idx>``)
 *   * GoldSetPanel entries list (testidPrefix ``gold-entry-row-<idx>``)
 *
 * The recipe drives the field framing:
 *   * qa-sft         → difficulty + trap badges + Q + A
 *   * classification → text + label-as-badge
 *   * span-extraction→ text + entity list with [start:end] offsets
 *   * summarization  → document (collapsed) + summary
 *
 * Caller controls testid namespacing via ``testidPrefix`` so both
 * consumers keep their existing testids. Each suffix this component
 * emits is documented next to the testid attribute.
 */

import type { ReactNode } from 'react';


export type GoldRowRecipe =
    | 'qa-sft'
    | 'classification'
    | 'span-extraction'
    | 'summarization';


export interface GoldRowSpan {
    type: string;
    start: number;
    end: number;
    text: string;
}


/** Shape covers both LLM-gen preview rows (rich, has rationale +
 *  source_excerpt + difficulty/trap labels) AND on-disk gold entries
 *  (saved subset). Every field is optional; the renderer picks the
 *  ones relevant to ``recipeId``. */
export interface GoldRowLike {
    // qa-sft
    question?: string;
    answer?: string;
    // classification
    text?: string;
    label?: string;
    // span-extraction
    entities?: GoldRowSpan[];
    // summarization
    document?: string;
    summary?: string;
    // qa-sft labels
    difficulty?: string;
    is_hallucination_trap?: boolean;
}


interface Props {
    recipeId: GoldRowRecipe;
    row: GoldRowLike;
    /** Per-row testid prefix. The component appends ``-text`` /
     *  ``-label`` / ``-entities`` / ``-document`` / ``-summary`` /
     *  ``-difficulty`` / ``-trap`` to identify sub-elements. */
    testidPrefix: string;
}


export default function GoldEntryRowBody({
    recipeId,
    row,
    testidPrefix,
}: Props): ReactNode {
    if (recipeId === 'classification') {
        return (
            <>
                <div
                    style={{ fontWeight: 600 }}
                    data-testid={`${testidPrefix}-text`}
                >
                    {row.text}
                </div>
                <div
                    style={{ marginTop: 4 }}
                    data-testid={`${testidPrefix}-label`}
                >
                    <span
                        className="badge badge-info"
                        style={{ fontFamily: 'monospace' }}
                    >
                        {row.label}
                    </span>
                </div>
            </>
        );
    }
    if (recipeId === 'span-extraction') {
        const entities = row.entities || [];
        return (
            <>
                <div
                    style={{ fontFamily: 'monospace', fontSize: '0.9rem' }}
                    data-testid={`${testidPrefix}-text`}
                >
                    {row.text}
                </div>
                <div
                    style={{ marginTop: 4, fontSize: '0.85rem' }}
                    data-testid={`${testidPrefix}-entities`}
                >
                    {entities.length === 0 ? (
                        <em style={{ color: 'var(--text-tertiary)' }}>
                            No entities (negative example)
                        </em>
                    ) : (
                        <ul
                            style={{
                                margin: 0,
                                paddingLeft: 'var(--space-md)',
                            }}
                        >
                            {entities.map((ent, ei) => (
                                <li key={ei}>
                                    <span
                                        className="badge badge-accent"
                                        style={{ marginRight: 4 }}
                                    >
                                        {ent.type}
                                    </span>
                                    <span style={{ fontFamily: 'monospace' }}>
                                        "{ent.text}"
                                    </span>
                                    <span
                                        style={{
                                            color: 'var(--text-tertiary)',
                                            marginLeft: 6,
                                        }}
                                    >
                                        [{ent.start}:{ent.end}]
                                    </span>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            </>
        );
    }
    if (recipeId === 'summarization') {
        // Document is potentially long — keep it readable but
        // collapsed-by-default so the row stays scannable.
        return (
            <>
                <details data-testid={`${testidPrefix}-document`}>
                    <summary style={{ fontWeight: 600, cursor: 'pointer' }}>
                        Document ({(row.document || '').length} chars) — click to expand
                    </summary>
                    <div
                        style={{
                            marginTop: 4,
                            whiteSpace: 'pre-wrap',
                            fontSize: '0.9rem',
                            color: 'var(--text-secondary)',
                        }}
                    >
                        {row.document}
                    </div>
                </details>
                <div
                    style={{ marginTop: 4 }}
                    data-testid={`${testidPrefix}-summary`}
                >
                    <strong>Summary:</strong> {row.summary}
                </div>
            </>
        );
    }
    // qa-sft: difficulty + hallucination-trap badges surface the
    // per-row labels at a glance so the user can spot-check the mix
    // (preview UX) or scan their gold set's distribution (entries UX).
    const difficulty = (row.difficulty || '').toString().trim();
    const isTrap = !!row.is_hallucination_trap;
    return (
        <>
            <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
                {difficulty && (
                    <span
                        className="badge badge-info"
                        data-testid={`${testidPrefix}-difficulty`}
                    >
                        {difficulty}
                    </span>
                )}
                {isTrap && (
                    <span
                        className="badge badge-warning"
                        data-testid={`${testidPrefix}-trap`}
                        title="Hallucination trap — answer should be 'I don't know'"
                    >
                        ⚠ trap
                    </span>
                )}
            </div>
            <div style={{ fontWeight: 600 }}>
                Q: {row.question}
            </div>
            <div style={{ marginTop: 4 }}>
                A: {row.answer}
            </div>
        </>
    );
}
