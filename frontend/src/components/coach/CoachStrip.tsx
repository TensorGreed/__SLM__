/**
 * Per-panel Coach Mode strip (USER-SUCCESS Epic 4 Phase 1).
 *
 * Mounted inside a workflow panel (e.g. IngestionPanel). Renders the
 * top suggestions returned by ``GET /coach/{stage}`` as compact cards
 * with click-to-execute actions. Silent when:
 *   - Coach Mode is off for this project (per ``useCoachMode``).
 *   - The backend reports no suggestions (the panel is "healthy").
 */

import { useEffect, useState } from 'react';

import { fetchCoachSuggestions, type CoachStage, type CoachSuggestion } from '../../api/coach';
import CoachSuggestionCard from './CoachSuggestion';
import { useCoachMode } from './useCoachMode';

interface CoachStripProps {
    projectId: number;
    stage: CoachStage;
    beginnerMode?: boolean;
    /** Show only the top N suggestions, with a "Show N more" toggle. */
    maxVisible?: number;
}

// Surfaces beginner mode hides — never send a newcomer to a page they
// can't see.
const ADVANCED_ONLY_TARGETS = new Set(['domain-pack-manager']);

function visibleFor(suggestion: CoachSuggestion, beginnerMode: boolean): boolean {
    if (!beginnerMode) return true;
    const target = (suggestion.action?.params as { target?: unknown } | undefined)?.target;
    return !(typeof target === 'string' && ADVANCED_ONLY_TARGETS.has(target));
}

export default function CoachStrip({ projectId, stage, beginnerMode = false, maxVisible }: CoachStripProps) {
    const { isOn, isReady } = useCoachMode(projectId, beginnerMode);
    const [suggestions, setSuggestions] = useState<CoachSuggestion[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [refreshKey, setRefreshKey] = useState(0);
    const [showAll, setShowAll] = useState(false);

    useEffect(() => {
        if (!isOn) return;
        let cancelled = false;
        setIsLoading(true);
        setError(null);
        const load = async () => {
            try {
                const res = await fetchCoachSuggestions(projectId, stage);
                if (cancelled) return;
                // Defensive: a test mock or a malformed server response
                // can leave ``res`` without ``.suggestions``. Default to
                // an empty array so the render path treats it as the
                // healthy/no-suggestions case rather than crashing on
                // ``suggestions.length``.
                setSuggestions(
                    Array.isArray(res?.suggestions) ? res.suggestions : [],
                );
            } catch (err) {
                if (cancelled) return;
                const detail =
                    (err as { response?: { data?: { detail?: string } } })?.response
                        ?.data?.detail;
                setError(detail ?? 'Failed to load Coach Mode suggestions.');
            } finally {
                if (!cancelled) setIsLoading(false);
            }
        };
        void load();
        return () => {
            cancelled = true;
        };
    }, [projectId, stage, isOn, refreshKey]);

    // Coach is off — silent. We render nothing rather than an empty
    // placeholder so existing panel layouts don't shift when Coach
    // Mode is toggled off.
    if (!isOn) return null;

    if (!isReady || isLoading) {
        return (
            <div
                data-testid={`coach-strip-${stage}`}
                style={{
                    padding: 'var(--space-sm) var(--space-md)',
                    fontSize: 'var(--font-size-xs)',
                    color: 'var(--text-tertiary)',
                    fontStyle: 'italic',
                }}
            >
                Coach Mode · loading suggestions…
            </div>
        );
    }

    if (error) {
        return (
            <div
                data-testid={`coach-strip-${stage}`}
                style={{
                    padding: 'var(--space-sm) var(--space-md)',
                    fontSize: 'var(--font-size-xs)',
                    color: 'var(--text-tertiary)',
                }}
            >
                Coach Mode unavailable: {error}
            </div>
        );
    }

    const shown = suggestions.filter((s) => visibleFor(s, beginnerMode));

    return (
        <div
            data-testid={`coach-strip-${stage}`}
            style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-sm)',
                marginBottom: 'var(--space-md)',
            }}
        >
            {shown.length === 0 ? (
                <div
                    data-testid={`coach-strip-${stage}-healthy`}
                    style={{
                        padding: 'var(--space-xs) var(--space-md)',
                        fontSize: 'var(--font-size-xs)',
                        color: 'var(--text-tertiary)',
                        fontStyle: 'italic',
                    }}
                >
                    Coach Mode · looks healthy on this surface.
                </div>
            ) : (
                <>
                    {(showAll || maxVisible === undefined ? shown : shown.slice(0, maxVisible)).map((s) => (
                        <CoachSuggestionCard
                            key={s.id}
                            projectId={projectId}
                            suggestion={s}
                            onActionCompleted={() => setRefreshKey((k) => k + 1)}
                        />
                    ))}
                    {maxVisible !== undefined && shown.length > maxVisible && (
                        <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => setShowAll((v) => !v)}
                            data-testid={`coach-strip-${stage}-more`}
                        >
                            {showAll ? 'Show fewer' : `Show ${shown.length - maxVisible} more suggestion${shown.length - maxVisible === 1 ? '' : 's'}`}
                        </button>
                    )}
                </>
            )}
        </div>
    );
}
