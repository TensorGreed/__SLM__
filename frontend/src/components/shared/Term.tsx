import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

import { getTermDefinition, TERM_DEFINITIONS } from './glossary';
import { lookupGlossaryEntry, useGlossaryStore } from '../../stores/glossaryStore';
import './Term.css';

interface TermProps {
    id: string;
    /** Override the rendered label (falls back to beginner/advanced label from the definition). */
    label?: ReactNode;
    /** Pluralize the beginner label (appends `s`). */
    plural?: boolean;
    /** Force the advanced / jargon label instead of the beginner label. */
    advanced?: boolean;
    /** Hide the popover completely (only renders the label). */
    silent?: boolean;
    /** Extra class applied to the root span. */
    className?: string;
}

export function Term({ id, label, plural = false, advanced = false, silent = false, className }: TermProps) {
    const definition = getTermDefinition(id);
    const [hovered, setHovered] = useState(false);
    const [pinned, setPinned] = useState(false);
    const open = hovered || pinned;
    const popoverId = useRef(`term-popover-${id}-${Math.random().toString(36).slice(2, 9)}`);
    const loaded = useGlossaryStore((state) => state.loaded);
    const loading = useGlossaryStore((state) => state.loading);
    const fetchGlossary = useGlossaryStore((state) => state.fetchGlossary);

    useEffect(() => {
        if (!loaded && !loading) {
            void fetchGlossary();
        }
    }, [loaded, loading, fetchGlossary]);

    const glossaryEntry = useMemo(() => {
        if (!definition) return null;
        return lookupGlossaryEntry(definition.glossaryKey);
        // reactivity is achieved via `loaded` changing
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [definition, loaded]);

    if (!definition) {
        const raw = typeof label === 'string' ? label : id;
        return <span className={className}>{raw}</span>;
    }

    const baseLabel = label
        ?? (advanced ? definition.advancedLabel : definition.beginnerLabel);
    const displayLabel = plural && typeof baseLabel === 'string'
        ? `${baseLabel}s`
        : baseLabel;

    if (silent) {
        return <span className={className}>{displayLabel}</span>;
    }

    const plainLanguage = glossaryEntry?.plain_language || definition.fallback;
    const example = glossaryEntry?.example || null;
    const aka = advanced ? definition.beginnerLabel : definition.advancedLabel;

    return (
        <span
            className={`term ${open ? 'term-open' : ''} ${className || ''}`.trim()}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            onFocus={() => setHovered(true)}
            onBlur={() => setHovered(false)}
        >
            <button
                type="button"
                className="term-trigger"
                aria-describedby={open ? popoverId.current : undefined}
                aria-expanded={open ? 'true' : 'false'}
                onClick={() => setPinned((prev) => !prev)}
            >
                {displayLabel}
            </button>
            {open && (
                <span
                    className="term-popover"
                    role="tooltip"
                    id={popoverId.current}
                >
                    <span className="term-popover-header">
                        <span className="term-popover-title">{definition.beginnerLabel}</span>
                        <span className="term-popover-category">{definition.category}</span>
                    </span>
                    <span className="term-popover-aka">Also known as: {aka}</span>
                    <span className="term-popover-body">{plainLanguage}</span>
                    {example && <span className="term-popover-example">Example: {example}</span>}
                </span>
            )}
        </span>
    );
}

export { TERM_DEFINITIONS };

export default Term;
