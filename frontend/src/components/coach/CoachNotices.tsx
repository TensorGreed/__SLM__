/**
 * Inline Coach notices (Epic U, phase U6). Replaces the Coach bar: a
 * destination mounts one of these for the stage it shows, and each nudge
 * renders as a notice right there — title, one action, a "why?" trace —
 * dismissible per rule. Hidden when inline tips are off (Settings).
 */

import { useEffect, useState } from 'react';

import { fetchCoachSuggestions, type CoachStage, type CoachSuggestion } from '../../api/coach';
import CoachSuggestionCard from './CoachSuggestion';
import { useCoachMode } from './useCoachMode';
import './CoachNotices.css';

interface CoachNoticesProps {
    projectId: number;
    stage: CoachStage;
    /** Notices shown before "Show N more". */
    maxVisible?: number;
    onActionCompleted?: () => void;
}

const SEVERITY_RANK: Record<string, number> = { critical: 0, warning: 1, info: 2 };

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

export default function CoachNotices({ projectId, stage, maxVisible = 2, onActionCompleted }: CoachNoticesProps) {
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
                const list = Array.isArray(res?.suggestions) ? [...res.suggestions] : [];
                list.sort((a, b) => (SEVERITY_RANK[a.severity] ?? 9) - (SEVERITY_RANK[b.severity] ?? 9));
                setSuggestions(list);
            })
            .catch(() => { if (!cancelled) setSuggestions([]); });
        return () => { cancelled = true; };
    }, [projectId, stage, isOn, refreshKey]);

    if (!isOn) return null;
    const visible = suggestions.filter((s) => !dismissed.has(s.rule_id ?? s.id));
    if (visible.length === 0) return null;
    const shown = showAll ? visible : visible.slice(0, maxVisible);
    const hidden = visible.length - shown.length;

    const dismiss = (suggestion: CoachSuggestion) => {
        const next = new Set(dismissed);
        next.add(suggestion.rule_id ?? suggestion.id);
        setDismissed(next);
        writeDismissed(projectId, stage, next);
    };

    return (
        <section className="coach-notices" data-testid={`coach-notices-${stage}`} aria-label="Suggestions for this step">
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
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setShowAll(true)} data-testid="coach-notices-more">
                    Show {hidden} more
                </button>
            )}
        </section>
    );
}
