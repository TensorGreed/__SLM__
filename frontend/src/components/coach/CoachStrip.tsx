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
}

export default function CoachStrip({ projectId, stage }: CoachStripProps) {
    const { isOn, isReady } = useCoachMode(projectId);
    const [suggestions, setSuggestions] = useState<CoachSuggestion[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [refreshKey, setRefreshKey] = useState(0);

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
            {suggestions.length === 0 ? (
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
                suggestions.map((s) => (
                    <CoachSuggestionCard
                        key={s.id}
                        projectId={projectId}
                        suggestion={s}
                        onActionCompleted={() => setRefreshKey((k) => k + 1)}
                    />
                ))
            )}
        </div>
    );
}
