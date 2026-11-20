/**
 * "Advanced" — the collapsed disclosure a page uses for controls most users
 * never need (UI simplification pass, phase 0). Collapsed by default; the
 * open / closed choice is remembered per page in localStorage (every access
 * is wrapped, the section works without storage). Children mount only while
 * open, so heavy panels fetch nothing until asked for.
 *
 * ⌘K contract: anything placed inside an AdvancedSection must stay reachable
 * from the palette. Register a command whose path carries ``?open=<id>`` (see
 * ``utils/workspaceCommands.ts``) and pass the page's ``?open=`` value as
 * ``openSection`` here: when it equals ``id`` the section opens (without
 * changing the remembered state). ``openNonce`` re-opens it for a same-page
 * action after the user closed it.
 */

import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import './AdvancedSection.css';

const STORAGE_PREFIX = 'brewslm.advanced.';

function readRemembered(pageKey: string): boolean {
    try {
        return window.localStorage.getItem(STORAGE_PREFIX + pageKey) === 'open';
    } catch {
        return false;
    }
}

function remember(pageKey: string, open: boolean): void {
    try {
        window.localStorage.setItem(STORAGE_PREFIX + pageKey, open ? 'open' : 'closed');
    } catch {
        // Storage blocked or full: the section still toggles for this visit.
    }
}

interface AdvancedSectionProps {
    /** Storage key: one per page (or per page + section when a page has several). */
    pageKey: string;
    /** The ``?open=`` id that targets this section. */
    id: string;
    /** Number of controls inside, shown as "Advanced (N)". */
    count?: number;
    title?: string;
    hint?: string;
    /** The page's current ``?open=`` value. */
    openSection?: string | null;
    openNonce?: number;
    testId?: string;
    children: ReactNode;
}

export default function AdvancedSection({
    pageKey,
    id,
    count,
    title = 'Advanced',
    hint,
    openSection = null,
    openNonce = 0,
    testId = 'advanced',
    children,
}: AdvancedSectionProps) {
    const requested = openSection === id;
    const [open, setOpen] = useState<boolean>(() => requested || readRemembered(pageKey));
    // A new open request opens the section — derived during render.
    const request = `${requested ? id : ''}#${openNonce}`;
    const [seenRequest, setSeenRequest] = useState(request);
    if (request !== seenRequest) {
        setSeenRequest(request);
        if (requested) setOpen(true);
    }

    const toggle = () => {
        const next = !open;
        setOpen(next);
        remember(pageKey, next);
    };

    const label = count !== undefined ? `${title} (${count})` : title;

    return (
        <section className="advanced-section" id={`${testId}-${id}`} data-testid={testId}>
            <button
                type="button"
                className="advanced-section__toggle"
                onClick={toggle}
                aria-expanded={open ? 'true' : 'false'}
                aria-controls={`${testId}-${id}-body`}
                data-testid={`${testId}-toggle`}
            >
                {open ? <ChevronDown className="u-icon" aria-hidden="true" /> : <ChevronRight className="u-icon" aria-hidden="true" />}
                <span>{label}</span>
                {hint && <small>{hint}</small>}
            </button>
            {open && (
                <div className="advanced-section__body" id={`${testId}-${id}-body`} data-testid={`${testId}-body`}>
                    {children}
                </div>
            )}
        </section>
    );
}
