/**
 * Keyboard + drag-to-select span (NER) labeler.
 *
 * UX contract (Story 1.2):
 * - Renders the row's text with existing spans painted inline.
 * - Letter keys 'a' / 'b' / 'c' / ... select the active span type
 *   (indexed into spanTypes[]).
 * - Drag-to-select inside the text region → on mouseup, the
 *   selection becomes a span of the active type. No active type
 *   selected → selection is ignored.
 * - Click an existing span → removes it.
 * - 'j' or "Save & next" → onSubmit(spans).
 * - 'esc' or "Skip" → onSkip().
 *
 * Internal state holds the working span set; spans only escape via
 * onSubmit so the parent isn't re-rendered on every interaction.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export interface SpanAnnotation {
    start: number;
    end: number;
    type: string;
}

interface SpanLabelerProps {
    text: string;
    spanTypes: string[];
    onSubmit: (spans: SpanAnnotation[]) => void;
    onSkip: () => void;
    disabled?: boolean;
    /** Optional initial spans (test-friendly seed). */
    initialSpans?: SpanAnnotation[];
}

const TYPE_PALETTE = [
    '#fde68a',
    '#bfdbfe',
    '#bbf7d0',
    '#fecaca',
    '#ddd6fe',
    '#fed7aa',
    '#a7f3d0',
    '#fbcfe8',
    '#e9d5ff',
];

function typeColor(spanTypes: string[], type: string): string {
    const idx = spanTypes.indexOf(type);
    if (idx < 0) return TYPE_PALETTE[TYPE_PALETTE.length - 1];
    return TYPE_PALETTE[idx % TYPE_PALETTE.length];
}

function isEditableTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return true;
    return target.isContentEditable;
}

/**
 * Walk the container's text-node descendants in document order to
 * convert a (node, offset) pair into a flat character index relative
 * to the container's text content. Returns -1 when the node isn't
 * inside the container.
 */
function getCharOffset(
    container: HTMLElement,
    node: Node | null,
    offset: number,
): number {
    if (!node) return -1;
    if (!container.contains(node)) return -1;
    let total = 0;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let cur = walker.nextNode();
    while (cur) {
        if (cur === node) {
            return total + offset;
        }
        total += (cur.textContent || '').length;
        cur = walker.nextNode();
    }
    // Node is the container itself or an element node: offset is the
    // char count up to that point.
    return total;
}

interface RenderedSegment {
    text: string;
    spanIndex: number | null;
}

/** Break ``text`` into flat segments — runs of plain text alternating
 * with run of one span each. Overlapping spans are clipped at the
 * boundary of the first one (we don't support nested spans). */
function renderSegments(
    text: string,
    spans: SpanAnnotation[],
): RenderedSegment[] {
    if (!spans.length) {
        return [{ text, spanIndex: null }];
    }
    const sorted = spans
        .map((span, idx) => ({ span, idx }))
        .filter(({ span }) => span.end > span.start)
        .sort((a, b) => a.span.start - b.span.start);

    const segments: RenderedSegment[] = [];
    let cursor = 0;
    for (const { span, idx } of sorted) {
        const start = Math.max(span.start, cursor);
        const end = Math.min(span.end, text.length);
        if (start > cursor) {
            segments.push({
                text: text.slice(cursor, start),
                spanIndex: null,
            });
        }
        if (end > start) {
            segments.push({
                text: text.slice(start, end),
                spanIndex: idx,
            });
            cursor = end;
        }
    }
    if (cursor < text.length) {
        segments.push({
            text: text.slice(cursor),
            spanIndex: null,
        });
    }
    return segments;
}

export default function SpanLabeler({
    text,
    spanTypes,
    onSubmit,
    onSkip,
    disabled = false,
    initialSpans = [],
}: SpanLabelerProps) {
    const [spans, setSpans] = useState<SpanAnnotation[]>(initialSpans);
    const [activeType, setActiveType] = useState<string | null>(
        spanTypes[0] || null,
    );
    const textRef = useRef<HTMLDivElement | null>(null);

    const addSpanFromSelection = useCallback(() => {
        if (disabled) return;
        if (!activeType) return;
        const container = textRef.current;
        if (!container) return;
        const selection = window.getSelection();
        if (!selection || selection.rangeCount === 0) return;
        const anchorOffset = getCharOffset(
            container,
            selection.anchorNode,
            selection.anchorOffset,
        );
        const focusOffset = getCharOffset(
            container,
            selection.focusNode,
            selection.focusOffset,
        );
        if (anchorOffset < 0 || focusOffset < 0) return;
        const start = Math.min(anchorOffset, focusOffset);
        const end = Math.max(anchorOffset, focusOffset);
        if (end <= start) return;
        setSpans((prev) => [...prev, { start, end, type: activeType }]);
        selection.removeAllRanges();
    }, [activeType, disabled]);

    const removeSpan = useCallback(
        (idx: number) => {
            if (disabled) return;
            setSpans((prev) => prev.filter((_, i) => i !== idx));
        },
        [disabled],
    );

    useEffect(() => {
        if (disabled) return undefined;

        const handler = (event: KeyboardEvent) => {
            if (isEditableTarget(event.target)) return;
            if (event.metaKey || event.ctrlKey || event.altKey) return;
            const key = event.key;
            if (key === 'Escape') {
                event.preventDefault();
                onSkip();
                return;
            }
            if (key === 'j' || key === 'Enter') {
                event.preventDefault();
                onSubmit(spans);
                return;
            }
            // a / b / c / … → set active type
            if (key.length === 1 && key >= 'a' && key <= 'z') {
                const idx = key.charCodeAt(0) - 'a'.charCodeAt(0);
                if (idx < spanTypes.length) {
                    event.preventDefault();
                    setActiveType(spanTypes[idx]);
                }
            }
        };

        document.addEventListener('keydown', handler);
        return () => document.removeEventListener('keydown', handler);
    }, [spans, spanTypes, onSubmit, onSkip, disabled]);

    const segments = useMemo(() => renderSegments(text, spans), [text, spans]);

    return (
        <div className="span-labeler" data-testid="span-labeler">
            <div
                style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 'var(--space-sm)',
                    marginBottom: 'var(--space-md)',
                }}
            >
                {spanTypes.map((type, idx) => (
                    <button
                        key={type}
                        type="button"
                        className={`btn ${
                            activeType === type ? 'btn-primary' : 'btn-secondary'
                        }`}
                        onClick={() => setActiveType(type)}
                        disabled={disabled}
                        data-testid={`span-type-${type}`}
                        title={
                            idx < 26
                                ? `Activate ${type} (press ${String.fromCharCode(
                                      97 + idx,
                                  )})`
                                : `Activate ${type}`
                        }
                    >
                        {idx < 26 && (
                            <span
                                style={{
                                    display: 'inline-block',
                                    marginRight: 6,
                                    minWidth: 16,
                                    textAlign: 'center',
                                    border: '1px solid var(--border-color)',
                                    borderRadius: 4,
                                    padding: '0 4px',
                                    fontSize: '0.75rem',
                                    fontFamily: 'var(--font-mono, monospace)',
                                }}
                            >
                                {String.fromCharCode(97 + idx)}
                            </span>
                        )}
                        <span
                            style={{
                                display: 'inline-block',
                                width: 10,
                                height: 10,
                                marginRight: 6,
                                background: typeColor(spanTypes, type),
                                borderRadius: 2,
                                verticalAlign: 'middle',
                            }}
                        />
                        {type}
                    </button>
                ))}
            </div>

            <div
                ref={textRef}
                onMouseUp={addSpanFromSelection}
                data-testid="span-labeler-text"
                style={{
                    padding: 'var(--space-md)',
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-sm)',
                    fontFamily: 'var(--font-mono, monospace)',
                    fontSize: '0.95rem',
                    whiteSpace: 'pre-wrap',
                    lineHeight: 1.6,
                    userSelect: 'text',
                    cursor: activeType && !disabled ? 'text' : 'default',
                }}
            >
                {segments.map((segment, segIdx) =>
                    segment.spanIndex === null ? (
                        <span key={segIdx}>{segment.text}</span>
                    ) : (
                        <mark
                            key={segIdx}
                            data-testid={`span-mark-${segment.spanIndex}`}
                            onClick={() => removeSpan(segment.spanIndex as number)}
                            title={`${
                                spans[segment.spanIndex as number]?.type
                            } — click to remove`}
                            style={{
                                background: typeColor(
                                    spanTypes,
                                    spans[segment.spanIndex as number]?.type ??
                                        '',
                                ),
                                padding: '0 2px',
                                borderRadius: 2,
                                cursor: 'pointer',
                            }}
                        >
                            {segment.text}
                        </mark>
                    ),
                )}
            </div>

            <div
                style={{
                    display: 'flex',
                    gap: 'var(--space-sm)',
                    marginTop: 'var(--space-md)',
                    alignItems: 'center',
                }}
            >
                <div
                    style={{
                        color: 'var(--text-secondary)',
                        fontSize: '0.85rem',
                    }}
                    data-testid="span-labeler-status"
                >
                    {spans.length} span(s){' '}
                    {activeType ? (
                        <>
                            · active type: <strong>{activeType}</strong>
                        </>
                    ) : (
                        '· no type selected'
                    )}
                </div>
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => onSubmit(spans)}
                    disabled={disabled}
                    data-testid="span-submit"
                    style={{ marginLeft: 'auto' }}
                    title="Save spans and advance (press j)"
                >
                    Save &amp; next
                </button>
                <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={onSkip}
                    disabled={disabled}
                    data-testid="span-skip"
                    title="Skip this row (esc)"
                >
                    Skip
                </button>
            </div>
        </div>
    );
}
