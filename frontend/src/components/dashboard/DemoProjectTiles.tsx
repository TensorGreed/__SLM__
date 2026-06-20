/**
 * DemoProjectTiles — entry-point tiles on the project list that seed a
 * pre-loaded showcase project in one click (newbie UX Phase 3).
 *
 * Each tile maps to a backend archetype under
 * ``backend/data/demo_samples/``. Clicking POSTs to
 * ``/api/demo-projects/{slug}`` and navigates straight into the new
 * project's workspace.
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Rocket } from 'lucide-react';

import api from '../../api/client';
import type {
    DemoArchetype,
    DemoCatalogResponse,
    DemoSeedResponse,
} from '../../types/demoProjects';

import './DemoProjectTiles.css';

interface ApiErrorShape {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
}

function extractErrorMessage(err: unknown, fallback = 'Request failed.'): string {
    const e = err as ApiErrorShape;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    return e?.message || fallback;
}

export default function DemoProjectTiles() {
    const navigate = useNavigate();
    const [archetypes, setArchetypes] = useState<DemoArchetype[]>([]);
    const [loading, setLoading] = useState(true);
    const [seedingSlug, setSeedingSlug] = useState<string | null>(null);
    const [resettingSlug, setResettingSlug] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const busy = seedingSlug ?? resettingSlug;

    useEffect(() => {
        let cancelled = false;
        const fetchCatalog = async () => {
            try {
                const response = await api.get<DemoCatalogResponse>('/demo-projects');
                if (!cancelled) {
                    setArchetypes(response.data.archetypes || []);
                }
            } catch (err) {
                if (!cancelled) {
                    setError(extractErrorMessage(err, 'Could not load demo catalog.'));
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        };
        void fetchCatalog();
        return () => {
            cancelled = true;
        };
    }, []);

    const seedAndOpen = useCallback(
        async (slug: string) => {
            setSeedingSlug(slug);
            setError(null);
            try {
                const response = await api.post<DemoSeedResponse>(
                    `/demo-projects/${slug}`,
                    {},
                );
                navigate(`/project/${response.data.project.id}`);
            } catch (err) {
                setError(extractErrorMessage(err, 'Could not seed the demo project.'));
                setSeedingSlug(null);
            }
        },
        [navigate],
    );

    // Phase G2 — reset lifecycle: drop the (possibly broken) sample and
    // re-seed a clean copy, then open it. Destructive → confirm first.
    const resetAndOpen = useCallback(
        async (slug: string, name: string) => {
            const ok = window.confirm(
                `Reset “${name}” to a fresh sample? This deletes your changes to it.`,
            );
            if (!ok) return;
            setResettingSlug(slug);
            setError(null);
            try {
                const response = await api.post<DemoSeedResponse>(
                    `/demo-projects/${slug}/reset`,
                    {},
                );
                navigate(`/project/${response.data.project.id}`);
            } catch (err) {
                setError(extractErrorMessage(err, 'Could not reset the demo project.'));
                setResettingSlug(null);
            }
        },
        [navigate],
    );

    if (loading || (archetypes.length === 0 && !error)) {
        return null;
    }

    return (
        <section className="demo-project-tiles" aria-labelledby="demo-project-tiles-heading">
            <div className="demo-project-tiles-header">
                <Rocket size={14} aria-hidden="true" />
                <h2 id="demo-project-tiles-heading">Try a demo project</h2>
                <span className="demo-project-tiles-hint">
                    Pre-loaded with sample data, a gold set, and a ready-to-run autopilot plan.
                </span>
            </div>
            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}
            <div className="demo-project-tiles-grid">
                {archetypes.map((archetype) => {
                    const isSeeding = seedingSlug === archetype.slug;
                    const isResetting = resettingSlug === archetype.slug;
                    const otherBusy = Boolean(busy) && busy !== archetype.slug;
                    return (
                        <div key={archetype.slug} className="demo-project-tile-wrap">
                            <button
                                type="button"
                                className="demo-project-tile"
                                disabled={otherBusy || isResetting}
                                onClick={() => void seedAndOpen(archetype.slug)}
                                aria-label={`Open the ${archetype.name} demo project`}
                            >
                                <div className="demo-project-tile-header">
                                    <span className="demo-project-tile-name">{archetype.name}</span>
                                    <span className="demo-project-tile-badge">
                                        {archetype.task_profile}
                                    </span>
                                </div>
                                <p className="demo-project-tile-headline">{archetype.headline}</p>
                                <p className="demo-project-tile-description">
                                    {archetype.description}
                                </p>
                                <div className="demo-project-tile-footer">
                                    <span className="dim">target: {archetype.target_profile}</span>
                                    <span className="demo-project-tile-cta">
                                        {isSeeding ? 'Seeding…' : 'Open demo →'}
                                    </span>
                                </div>
                            </button>
                            <button
                                type="button"
                                className="demo-project-tile-reset"
                                disabled={otherBusy || isSeeding}
                                onClick={() => void resetAndOpen(archetype.slug, archetype.name)}
                                aria-label={`Reset the ${archetype.name} demo project`}
                            >
                                {isResetting ? 'Resetting…' : '↺ Reset to fresh'}
                            </button>
                        </div>
                    );
                })}
            </div>
        </section>
    );
}
