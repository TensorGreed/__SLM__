/**
 * "Details" — the collapsed drawer under a step's short form (Epic U phase
 * U4). Sections render only while open, so the long tail (profiles,
 * versions, studio panels) fetches nothing until asked for. ``openSection``
 * (+ ``openNonce`` to re-open) comes from ``?open=`` or a same-page action.
 */

import { useEffect, useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import './DetailsDrawer.css';

export interface DetailSection {
    id: string;
    title: string;
    hint?: string;
    /** Rendered only while the section is open. */
    render: () => ReactNode;
}

interface DetailsDrawerProps {
    sections: DetailSection[];
    openSection?: string | null;
    openNonce?: number;
    title?: string;
    hint?: string;
    /** Test id prefix: ``<prefix>-toggle``, ``<prefix>-section-<id>``. */
    testId?: string;
}

export default function DetailsDrawer({
    sections,
    openSection = null,
    openNonce = 0,
    title = 'Details',
    hint,
    testId = 'details',
}: DetailsDrawerProps) {
    const [open, setOpen] = useState<Set<string>>(() => new Set(openSection ? [openSection] : []));
    const [drawerOpen, setDrawerOpen] = useState<boolean>(Boolean(openSection));
    // A new open request opens that section — derived during render.
    const [seenRequest, setSeenRequest] = useState(`${openSection ?? ''}#${openNonce}`);
    const request = `${openSection ?? ''}#${openNonce}`;
    if (request !== seenRequest) {
        setSeenRequest(request);
        if (openSection) {
            setOpen((prev) => (prev.has(openSection) ? prev : new Set([...prev, openSection])));
            setDrawerOpen(true);
        }
    }

    useEffect(() => {
        if (!openSection) return;
        const timer = window.setTimeout(() => {
            document.getElementById(`${testId}-section-${openSection}`)?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
        }, 50);
        return () => window.clearTimeout(timer);
    }, [openSection, openNonce, testId]);

    const toggle = (id: string) => {
        setOpen((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    if (sections.length === 0) return null;

    return (
        <section className="card details-drawer" data-testid={testId}>
            <button
                type="button"
                className="details-drawer__toggle"
                onClick={() => setDrawerOpen((v) => !v)}
                aria-expanded={drawerOpen ? 'true' : 'false'}
                data-testid={`${testId}-toggle`}
            >
                {drawerOpen ? <ChevronDown className="u-icon" aria-hidden="true" /> : <ChevronRight className="u-icon" aria-hidden="true" />} {title}
                {hint && <small>{hint}</small>}
            </button>
            {drawerOpen && (
                <div className="details-drawer__sections">
                    {sections.map((section) => {
                        const isOpen = open.has(section.id);
                        return (
                            <div key={section.id} className={`details-drawer__section ${isOpen ? 'is-open' : ''}`} id={`${testId}-section-${section.id}`}>
                                <button
                                    type="button"
                                    className="details-drawer__section-toggle"
                                    onClick={() => toggle(section.id)}
                                    aria-expanded={isOpen ? 'true' : 'false'}
                                    data-testid={`${testId}-section-${section.id}`}
                                >
                                    <span>{section.title}</span>
                                    {section.hint && <small>{section.hint}</small>}
                                </button>
                                {isOpen && <div className="details-drawer__body">{section.render()}</div>}
                            </div>
                        );
                    })}
                </div>
            )}
        </section>
    );
}
