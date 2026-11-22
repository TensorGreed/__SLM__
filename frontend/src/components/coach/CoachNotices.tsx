/**
 * Inline Coach notices (Epic U, phase U6). Replaces the Coach bar: a
 * destination mounts one of these for the stage it shows, and each nudge
 * renders as a notice right there — title, one action, a "why?" trace —
 * dismissible per rule. Hidden when inline tips are off (Settings).
 *
 * UI simplification phase 6 (decision 4): at most ONE notice per page, the
 * most severe critical / warning one; every other nudge (more warnings and
 * all info-level tips, which are never shown open by default) waits under
 * "N more tips". The backend ``coach_service`` is unchanged.
 */

import { useEffect, useState } from 'react';

import { fetchCoachSuggestions, type CoachStage, type CoachSuggestion } from '../../api/coach';
import CoachSuggestionCard from './CoachSuggestion';
import { useCoachMode } from './useCoachMode';
import './CoachNotices.css';

interface CoachNoticesProps {
    projectId: number;
    stage: CoachStage;
    /** Critical / warning notices shown open before "N more tips" (default 1: one per page). */
    maxVisible?: number;
    /** A kicker above the notices (the Overview's "Worth a look"). */
    heading?: string;
    /** Defaults to ``coach-notices-<stage>``. */
    testId?: string;
    onActionCompleted?: () => void;
}

const SEVERITY_RANK: Record<string, number> = { critical: 0, warning: 1, info: 2 };

/** Most severe first; the backend order breaks ties. */
export function sortBySeverity(list: CoachSuggestion[]): CoachSuggestion[] {
    return list
        .map((suggestion, index) => ({ suggestion, index }))
        .sort((a, b) => ((SEVERITY_RANK[a.suggestion.severity] ?? 9) - (SEVERITY_RANK[b.suggestion.severity] ?? 9)) || a.index - b.index)
        .map(({ suggestion }) => suggestion);
}

/**
 * Which notices show open and which wait under "N more tips": up to
 * ``maxOpen`` of the most severe critical / warning ones open; info-level
 * tips are never open by default.
 */
export function splitNotices(sorted: CoachSuggestion[], maxOpen = 1): { open: CoachSuggestion[]; folded: CoachSuggestion[] } {
    const open: CoachSuggestion[] = [];
    const folded: CoachSuggestion[] = [];
    for (const suggestion of sorted) {
        if (open.length < maxOpen && suggestion.severity !== 'info') open.push(suggestion);
        else folded.push(suggestion);
    }
    return { open, folded };
}

function dismissKey(projectId: number, stage: string): string {
    return `brewslm.coach.dismissed.${projectId}.${stage}`;
}

export function readDismissed(projectId: number, stage: string): Set<string> {
    try {
        const raw = window.localStorage.getItem(dismissKey(projectId, stage));
        const parsed = raw ? JSON.parse(raw) : [];
        return new Set(Array.isArray(parsed) ? parsed.map(String) : []);
    } catch {
        return new Set();
    }
}

function writeDismissed(projectId: number, stage: string, ids: Set<string>): void {
    try {
        window.localStorage.setItem(dismissKey(projectId, stage), JSON.stringify([...ids]));
    } catch {
        // private mode / quota — the notice just shows again next time
    }
}

export default function CoachNotices({ projectId, stage, maxVisible = 1, heading, testId, onActionCompleted }: CoachNoticesProps) {
    const { isOn } = useCoachMode(projectId);
    const [suggestions, setSuggestions] = useState<CoachSuggestion[]>([]);
    const [dismissed, setDismissed] = useState<Set<string>>(() => readDismissed(projectId, stage));
    const [showAll, setShowAll] = useState(false);
    const [refreshKey, setRefreshKey] = useState(0);

    useEffect(() => {
        if (!isOn) return;
        let cancelled = false;
        fetchCoachSuggestions(projectId, stage)
            .then((res) => {
                if (cancelled) return;
                setSuggestions(sortBySeverity(Array.isArray(res?.suggestions) ? res.suggestions : []));
            })
            .catch(() => { if (!cancelled) setSuggestions([]); });
        return () => { cancelled = true; };
    }, [projectId, stage, isOn, refreshKey]);

    if (!isOn) return null;
    const visible = suggestions.filter((s) => !dismissed.has(s.rule_id ?? s.id));
    if (visible.length === 0) return null;
    const { open, folded } = splitNotices(visible, maxVisible);
    const shown = showAll ? visible : open;
    const hidden = showAll ? 0 : folded.length;

    const dismiss = (suggestion: CoachSuggestion) => {
        const next = new Set(dismissed);
        next.add(suggestion.rule_id ?? suggestion.id);
        setDismissed(next);
        writeDismissed(projectId, stage, next);
    };

    return (
        <section className="coach-notices" data-testid={testId ?? `coach-notices-${stage}`} aria-label="Suggestions for this step">
            {heading && <span className="coach-notices__heading">{heading}</span>}
            {shown.map((suggestion) => (
                <div key={suggestion.id} className="coach-notice">
                    <CoachSuggestionCard
                        projectId={projectId}
                        suggestion={suggestion}
                        onActionCompleted={() => {
                            setRefreshKey((v) => v + 1);
                            onActionCompleted?.();
                        }}
                    />
                    <button
                        type="button"
                        className="coach-notice__dismiss"
                        onClick={() => dismiss(suggestion)}
                        aria-label={`Hide: ${suggestion.title}`}
                        title="Hide this suggestion"
                        data-testid={`coach-notice-dismiss-${suggestion.id}`}
                    >
                        ×
                    </button>
                </div>
            ))}
            {hidden > 0 && (
                <button
                    type="button"
                    className="btn btn-ghost btn-sm coach-notices__more"
                    onClick={() => setShowAll(true)}
                    data-testid="coach-notices-more"
                >
                    {shown.length > 0 ? `${hidden} more ${hidden === 1 ? 'tip' : 'tips'}` : `${hidden} ${hidden === 1 ? 'tip' : 'tips'} for this step`}
                </button>
            )}
            {showAll && folded.length > 0 && (
                <button
                    type="button"
                    className="btn btn-ghost btn-sm coach-notices__more"
                    onClick={() => setShowAll(false)}
                    data-testid="coach-notices-fewer"
                >
                    Fewer tips
                </button>
            )}
        </section>
    );
}
