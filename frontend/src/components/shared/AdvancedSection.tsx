/**
 * "Advanced" — the collapsed disclosure a page uses for controls most users
 * never need (UI simplification pass, phase 0). Collapsed by default; the
 * open / closed choice is remembered per page in localStorage (every access
 * is wrapped, the section works without storage).
 *
 * ⌘K contract: anything placed inside an AdvancedSection must stay reachable
 * from the palette. Register the section's id in ``utils/advancedSections.ts``,
 * add a command in ``utils/workspaceCommands.ts`` whose path carries
 * ``?open=<id>``, and pass the page's ``?open=`` value as ``openSection``
 * here: when it equals ``id`` the section opens (without changing the
 * remembered state) and scrolls into view. ``openNonce`` re-opens it for a
 * same-page action after the user closed it.
 *
 * Hidden fields keep sending today's values: by default the children are NOT
 * mounted while collapsed, so a form whose Advanced fields feed a request must
 * either hold that state in the parent (controlled inputs), or pass
 * ``keepMounted`` so the children render inside a ``hidden`` wrapper and keep
 * their own state and defaults. Never rely on an unmounted child to default.
 */

import { useEffect, useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import './AdvancedSection.css';

const STORAGE_PREFIX = 'brewslm.advanced.';

function storageKey(pageKey: string, id: string): string {
    return `${STORAGE_PREFIX}${pageKey}.${id}`;
}

function readRemembered(pageKey: string, id: string): boolean {
    try {
        return window.localStorage.getItem(storageKey(pageKey, id)) === 'open';
    } catch {
        return false;
    }
}

function remember(pageKey: string, id: string, open: boolean): void {
    try {
        window.localStorage.setItem(storageKey(pageKey, id), open ? 'open' : 'closed');
    } catch {
        // Storage blocked or full: the section still toggles for this visit.
    }
}

interface AdvancedSectionProps {
    /** Storage namespace, one per page. Read once at mount: a later change does not re-read the remembered state. */
    pageKey: string;
    /** The ``?open=`` id that targets this section; with ``pageKey`` it is the storage key. */
    id: string;
    /** Number of controls inside, shown as "Advanced (N)". */
    count?: number;
    title?: string;
    hint?: string;
    /** The page's current ``?open=`` value. */
    openSection?: string | null;
    openNonce?: number;
    /** Render children inside a ``hidden`` wrapper while collapsed (they keep their state). */
    keepMounted?: boolean;
    /** Defaults to ``advanced-<id>``. */
    testId?: string;
    /**
     * Scroll the section into view on an open request (default true). False when
     * the request targets something inside it that scrolls itself (Settings'
     * ``?open=<section>``), so the page ends on that, not on this header.
     */
    scrollOnRequest?: boolean;
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
    keepMounted = false,
    testId: testIdProp,
    scrollOnRequest = true,
    children,
}: AdvancedSectionProps) {
    const testId = testIdProp ?? `advanced-${id}`;
    const requested = openSection === id;
    const [open, setOpen] = useState<boolean>(() => requested || readRemembered(pageKey, id));
    // A new open request opens the section — derived during render.
    const request = `${requested ? id : ''}#${openNonce}`;
    const [seenRequest, setSeenRequest] = useState(request);
    if (request !== seenRequest) {
        setSeenRequest(request);
        if (requested) setOpen(true);
    }

    useEffect(() => {
        if (!requested || !scrollOnRequest) return;
        const timer = window.setTimeout(() => {
            document.getElementById(testId)?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
        }, 50);
        return () => window.clearTimeout(timer);
    }, [requested, openNonce, testId, scrollOnRequest]);

    const toggle = () => {
        const next = !open;
        setOpen(next);
        remember(pageKey, id, next);
    };

    const label = count !== undefined ? `${title} (${count})` : title;

    return (
        <section className="advanced-section" id={testId} data-testid={testId}>
            <button
                type="button"
                className="advanced-section__toggle"
                onClick={toggle}
                aria-expanded={open ? 'true' : 'false'}
                aria-controls={`${testId}-body`}
                data-testid={`${testId}-toggle`}
            >
                {open ? <ChevronDown className="u-icon" aria-hidden="true" /> : <ChevronRight className="u-icon" aria-hidden="true" />}
                <span>{label}</span>
                {hint && <small>{hint}</small>}
            </button>
            {(open || keepMounted) && (
                <div className="advanced-section__body" id={`${testId}-body`} data-testid={`${testId}-body`} hidden={!open}>
                    {children}
                </div>
            )}
        </section>
    );
}
