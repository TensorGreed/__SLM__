/**
 * DriftReviewQueuePanel — UI for E4's drift-triggered hallucination
 * trap refresh.
 *
 * Three surfaces in one card:
 *   1. Opt-in banner: shows whether auto-refresh is enabled +
 *      one-click toggle. Hidden once enabled; replaced with a quiet
 *      "auto-refresh on" status chip.
 *   2. "Generate now" button: manually triggers a refresh against
 *      the recent cluster patterns. Surfaces the generation summary
 *      (count + which clusters were targeted) on success.
 *   3. Pending-row list: each row renders the cluster context +
 *      recipe-shaped payload + Accept / Reject buttons. Rejected
 *      rows fall out of the pending view; the user can flip to
 *      "show all" to see the audit trail.
 *
 * Recipe-shaped row body reuses ``GoldEntryRowBody`` from the gold
 * workbench so the trap preview matches what the user sees when
 * inspecting gold rows elsewhere.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import type {
    DriftQueueRow,
    DriftQueueStatus,
    DriftSettings,
} from '../../api/drift';
import {
    fetchDriftSettings,
    listDriftReviewQueue,
    refreshDriftTraps,
    triageDriftRow,
    updateDriftSettings,
} from '../../api/drift';
import GoldEntryRowBody from '../data/GoldEntryRowBody';
import type { GoldRowLike, GoldRowRecipe } from '../data/GoldEntryRowBody';
import api from '../../api/client';
import { toast } from '../../stores/toastStore';
import './DriftReviewQueuePanel.css';


interface Props {
    projectId: number;
    /** The project's selected recipe id. Drives which renderer the
     *  per-row preview uses. Optional — falls back to qa-sft when
     *  absent (matches GoldEntryRowBody's permissive defaults). */
    recipeId?: string | null;
}


const RECIPE_SUPPORTED: ReadonlySet<string> = new Set([
    'qa-sft', 'classification', 'span-extraction', 'summarization',
]);


function resolveRowRecipe(recipeId: string | null | undefined): GoldRowRecipe {
    const token = (recipeId || '').trim().toLowerCase();
    if (RECIPE_SUPPORTED.has(token)) return token as GoldRowRecipe;
    return 'qa-sft';
}


export default function DriftReviewQueuePanel({ projectId, recipeId }: Props) {
    const [settings, setSettings] = useState<DriftSettings | null>(null);
    const [rows, setRows] = useState<DriftQueueRow[]>([]);
    const [statusFilter, setStatusFilter] = useState<DriftQueueStatus | 'all'>('pending');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState<'idle' | 'refreshing' | 'toggling' | 'triaging'>('idle');
    const [lastSummary, setLastSummary] = useState<{
        generated: number; simulated: boolean; clusters: string[];
    } | null>(null);
    // Self-fetch the recipe when the parent doesn't pass one — the
    // row renderer needs it to pick the right per-recipe layout
    // (qa-pair vs text+label vs spans vs doc+summary). Falls back to
    // qa-sft if the fetch fails or the project has no recipe set.
    const [resolvedRecipeId, setResolvedRecipeId] = useState<string | null>(
        recipeId ?? null,
    );
    useEffect(() => {
        if (recipeId) return;  // Parent gave us one; skip the fetch.
        let cancelled = false;
        api.get(`/projects/${projectId}`)
            .then((res) => {
                if (cancelled) return;
                const sr = (res.data as { selected_recipe?: { recipe_id?: string } })
                    ?.selected_recipe;
                setResolvedRecipeId(sr?.recipe_id ?? null);
            })
            .catch(() => {
                if (!cancelled) setResolvedRecipeId(null);
            });
        return () => { cancelled = true; };
    }, [projectId, recipeId]);

    const rowRecipe = useMemo(
        () => resolveRowRecipe(resolvedRecipeId),
        [resolvedRecipeId],
    );

    const loadSettings = useCallback(async () => {
        try {
            const s = await fetchDriftSettings(projectId);
            // Defensive: a test mock or older backend may return an
            // empty body. We still render the panel — the opt-in
            // banner uses defaults (enabled=false, count=5) so the
            // user can recover via the manual Generate now path.
            if (s && typeof s.enabled === 'boolean') {
                setSettings(s);
            } else {
                setSettings({ project_id: projectId, enabled: false, count: 5 });
            }
        } catch (err: any) {
            // 404 means the project doesn't exist (or auth) — surface
            // as a quiet error rather than a noisy toast.
            setError(
                err?.response?.data?.detail
                    || err?.message
                    || 'Failed to load drift settings',
            );
        }
    }, [projectId]);

    const loadQueue = useCallback(async () => {
        setError(null);
        try {
            const resp = await listDriftReviewQueue(projectId, {
                status: statusFilter === 'all' ? undefined : statusFilter,
                limit: 50,
            });
            setRows(Array.isArray(resp?.rows) ? resp.rows : []);
        } catch (err: any) {
            setError(
                err?.response?.data?.detail
                    || err?.message
                    || 'Failed to load drift queue',
            );
        }
    }, [projectId, statusFilter]);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        Promise.all([loadSettings(), loadQueue()]).finally(() => {
            if (!cancelled) setLoading(false);
        });
        return () => {
            cancelled = true;
        };
    }, [loadSettings, loadQueue]);

    const handleToggleEnabled = useCallback(async (next: boolean) => {
        setBusy('toggling');
        try {
            const updated = await updateDriftSettings(projectId, { enabled: next });
            setSettings(updated);
            toast.success(
                next
                    ? 'Auto-refresh enabled — drift checks will populate the queue.'
                    : 'Auto-refresh disabled. Manual refresh still works.',
            );
        } catch (err: any) {
            toast.error(
                err?.response?.data?.detail
                    || err?.message
                    || 'Failed to update settings',
            );
        } finally {
            setBusy('idle');
        }
    }, [projectId]);

    const handleRefresh = useCallback(async () => {
        setBusy('refreshing');
        setLastSummary(null);
        try {
            // simulate=false → use the LLM-backed generator (falls back
            // to placeholder rows server-side if no key is stored).
            const result = await refreshDriftTraps(projectId, {
                count: settings?.count,
                simulate: false,
            });
            setLastSummary({
                generated: result.generated,
                simulated: result.simulated,
                clusters: result.clusters_targeted,
            });
            await loadQueue();
            toast.success(
                `Generated ${result.generated} trap${result.generated === 1 ? '' : 's'}. Review in the queue.`,
            );
        } catch (err: any) {
            const detail = err?.response?.data?.detail;
            if (detail === 'recipe_required') {
                toast.error(
                    'Pick a recipe before generating drift traps.',
                );
            } else {
                toast.error(detail || err?.message || 'Refresh failed');
            }
        } finally {
            setBusy('idle');
        }
    }, [loadQueue, projectId, settings?.count]);

    const handleTriage = useCallback(
        async (row: DriftQueueRow, accept: boolean) => {
            setBusy('triaging');
            try {
                await triageDriftRow(projectId, row.id, { accept });
                toast.success(
                    accept
                        ? `Row #${row.id} accepted — appended to gold_test.`
                        : `Row #${row.id} rejected.`,
                );
                await loadQueue();
            } catch (err: any) {
                toast.error(
                    err?.response?.data?.detail
                        || err?.message
                        || 'Triage failed',
                );
            } finally {
                setBusy('idle');
            }
        },
        [loadQueue, projectId],
    );

    if (loading && !settings) {
        return (
            <section className="card drift-review" data-testid="drift-review-loading">
                <p>Loading drift queue…</p>
            </section>
        );
    }

    if (error) {
        return (
            <section className="card drift-review drift-review--error" data-testid="drift-review-error">
                <p>{error}</p>
            </section>
        );
    }

    return (
        <section className="card drift-review" data-testid="drift-review">
            <header className="drift-review__head">
                <div>
                    <h3 className="drift-review__title">Drift-trap review queue</h3>
                    <p className="drift-review__subtitle">
                        Fresh hallucination traps generated against the project's recent
                        failure-cluster patterns. Accept rows you want in
                        <code>gold_test</code>; reject the ones that miss.
                    </p>
                </div>
                <div className="drift-review__controls">
                    <select
                        value={statusFilter}
                        onChange={(e) => setStatusFilter(e.target.value as DriftQueueStatus | 'all')}
                        className="input"
                        data-testid="drift-review-filter"
                        aria-label="Filter drift queue by status"
                    >
                        <option value="pending">Pending</option>
                        <option value="accepted">Accepted</option>
                        <option value="rejected">Rejected</option>
                        <option value="all">All (audit)</option>
                    </select>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={handleRefresh}
                        disabled={busy !== 'idle'}
                        data-testid="drift-review-refresh"
                    >
                        {busy === 'refreshing' ? 'Generating…' : 'Generate now'}
                    </button>
                </div>
            </header>

            {settings && !settings.enabled && (
                <div
                    className="drift-review__optin-banner"
                    data-testid="drift-review-optin-banner"
                >
                    <div>
                        <strong>Automatic refresh is off.</strong> Drift checks won't
                        populate this queue on their own. Manual refresh via
                        <strong> Generate now</strong> still works.
                    </div>
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void handleToggleEnabled(true)}
                        disabled={busy !== 'idle'}
                        data-testid="drift-review-enable-auto"
                    >
                        Enable auto-refresh
                    </button>
                </div>
            )}

            {settings && settings.enabled && (
                <div
                    className="drift-review__status-chip"
                    data-testid="drift-review-auto-on"
                >
                    Auto-refresh on · <strong>{settings.count}</strong> traps per drift check
                    {' · '}
                    <button
                        type="button"
                        className="btn btn-link drift-review__link-flush"
                        onClick={() => void handleToggleEnabled(false)}
                        disabled={busy !== 'idle'}
                        data-testid="drift-review-disable-auto"
                    >
                        Disable
                    </button>
                </div>
            )}

            {lastSummary && (
                <div
                    className="drift-review__last-summary"
                    data-testid="drift-review-last-summary"
                >
                    Last refresh generated <strong>{lastSummary.generated}</strong> trap
                    {lastSummary.generated === 1 ? '' : 's'}
                    {lastSummary.simulated ? ' (simulated — no LLM key configured)' : ''}
                    {lastSummary.clusters.length > 0
                        ? <>
                            {' '}targeting {lastSummary.clusters.map((c, i) => (
                                <code key={`${c}-${i}`}>{c}</code>
                            )).reduce<React.ReactNode[]>((acc, el, i) => {
                                if (i === 0) return [el];
                                return [...acc, ', ', el];
                            }, [])}
                          </>
                        : ' (no recent clusters — generic traps).'}
                </div>
            )}

            {rows.length === 0 ? (
                <p
                    className="drift-review__empty"
                    data-testid="drift-review-empty"
                >
                    {statusFilter === 'pending'
                        ? 'No pending traps. Click Generate now to spin some, or wait for the next drift check.'
                        : `No ${statusFilter === 'all' ? '' : statusFilter} rows.`}
                </p>
            ) : (
                <ul className="drift-review__list" data-testid="drift-review-list">
                    {rows.map((row) => (
                        <DriftQueueRowCard
                            key={row.id}
                            row={row}
                            recipeId={rowRecipe}
                            busy={busy === 'triaging'}
                            onTriage={(accept) => void handleTriage(row, accept)}
                        />
                    ))}
                </ul>
            )}
        </section>
    );
}


interface DriftQueueRowCardProps {
    row: DriftQueueRow;
    recipeId: GoldRowRecipe;
    busy: boolean;
    onTriage: (accept: boolean) => void;
}


function DriftQueueRowCard({ row, recipeId, busy, onTriage }: DriftQueueRowCardProps) {
    const payload = row.payload as GoldRowLike;
    const isPending = row.status === 'pending';
    return (
        <li
            className={`drift-review__row drift-review__row--${row.status}`}
            data-testid={`drift-review-row-${row.id}`}
            data-status={row.status}
        >
            <div className="drift-review__row-meta">
                <span className={`drift-review__pill drift-review__pill--${row.status}`}>
                    {row.status}
                </span>
                {row.cluster_reason_code ? (
                    <span className="drift-review__cluster">
                        cluster: <code>{row.cluster_reason_code}</code>
                    </span>
                ) : (
                    <span className="drift-review__cluster drift-review__cluster--none">
                        generic trap
                    </span>
                )}
                <span className="drift-review__row-time">
                    {new Date(row.created_at).toLocaleString()}
                </span>
            </div>
            <div className="drift-review__row-body">
                <GoldEntryRowBody
                    recipeId={recipeId}
                    row={payload}
                    testidPrefix={`drift-review-row-${row.id}`}
                />
            </div>
            {isPending && (
                <div className="drift-review__row-actions">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => onTriage(false)}
                        disabled={busy}
                        data-testid={`drift-review-row-${row.id}-reject`}
                    >
                        Reject
                    </button>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => onTriage(true)}
                        disabled={busy}
                        data-testid={`drift-review-row-${row.id}-accept`}
                    >
                        Accept → gold_test
                    </button>
                </div>
            )}
            {!isPending && row.triage_note && (
                <p className="drift-review__triage-note">Note: {row.triage_note}</p>
            )}
        </li>
    );
}
