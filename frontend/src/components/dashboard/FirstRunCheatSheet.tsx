/**
 * FirstRunCheatSheet — one-time orientation card on the project list
 * (newbie UX Phase 3 / Phase 4.3).
 *
 * Surfaces the four things a brand-new user needs to know on day one:
 * the demo tiles below, the Cmd-K command palette, the docs, and how
 * to dismiss the card. Hidden permanently once dismissed (localStorage
 * key ``brewslm.cheatsheet.dismissed``).
 */

import { useEffect, useState } from 'react';
import { X, Keyboard, BookOpen, Sparkles, Rocket } from 'lucide-react';

import './FirstRunCheatSheet.css';

export const CHEATSHEET_DISMISSED_KEY = 'brewslm.cheatsheet.dismissed';

function readDismissed(): boolean {
    try {
        return window.localStorage.getItem(CHEATSHEET_DISMISSED_KEY) === '1';
    } catch {
        return false;
    }
}

function writeDismissed(): void {
    try {
        window.localStorage.setItem(CHEATSHEET_DISMISSED_KEY, '1');
    } catch {
        // ignore storage failures (private mode, quota)
    }
}

export default function FirstRunCheatSheet() {
    const [dismissed, setDismissed] = useState<boolean | null>(null);

    useEffect(() => {
        setDismissed(readDismissed());
    }, []);

    if (dismissed !== false) {
        return null;
    }

    const handleDismiss = () => {
        writeDismissed();
        setDismissed(true);
    };

    return (
        <section
            className="first-run-cheatsheet"
            role="region"
            aria-labelledby="first-run-cheatsheet-title"
        >
            <div className="first-run-cheatsheet-head">
                <Sparkles size={14} aria-hidden="true" />
                <h2 id="first-run-cheatsheet-title">First time here? Start here.</h2>
                <button
                    type="button"
                    className="first-run-cheatsheet-dismiss"
                    onClick={handleDismiss}
                    aria-label="Dismiss the first-run cheat sheet"
                >
                    <X size={14} aria-hidden="true" />
                </button>
            </div>
            <ul className="first-run-cheatsheet-list">
                <li>
                    <Rocket size={14} aria-hidden="true" />
                    <span>
                        <strong>Click a demo tile below.</strong> Each one seeds a real project with
                        sample data, a gold set, and a pre-filled Autopilot brief — one click to a
                        working pipeline.
                    </span>
                </li>
                <li>
                    <Keyboard size={14} aria-hidden="true" />
                    <span>
                        <strong>Press ⌘K (or Ctrl-K)</strong> anywhere to open the command palette.
                        Every page, panel, and action is reachable from there without hunting through
                        menus.
                    </span>
                </li>
                <li>
                    <BookOpen size={14} aria-hidden="true" />
                    <span>
                        <strong>Confused by a term?</strong> Hover any underlined word for a plain-language
                        definition, or open the{' '}
                        <a
                            href="http://localhost:3001/docs/concepts/glossary"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Glossary
                        </a>{' '}
                        and{' '}
                        <a
                            href="http://localhost:3001/docs/getting-started/quickstart"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Quickstart
                        </a>{' '}
                        for the full tour.
                    </span>
                </li>
            </ul>
        </section>
    );
}
