/**
 * Quality-Lift phase 8 slice 1 — AggregateRunBadge.
 *
 * Mounted in the EvalPanel result header when the EvalResult is
 * ``is_aggregate=true`` (a seed-group rollup written by phase 1's
 * experiment_aggregation_service). Surfaces the "your headline is a
 * rollup of N seeds" reality directly above the scalar metrics so
 * the user doesn't have to drill into a failing gate to discover
 * variance.
 *
 * Picked-data-provenance rule — every claim the badge makes is
 * verifiable: click "View per-seed runs" and the table lists every
 * child experiment that contributed, with seed value + scalar
 * metrics + experiment status. Failed children appear too (status =
 * failed, empty metrics) so a 2-of-3 mean is honest about the missing
 * seed rather than silently dropping it.
 *
 * What the badge does NOT do (deferred to slice 2):
 *   * Per-slice metric breakdown.
 *   * Aggregate behavioral-test pass-rate display.
 * Slice 1 is strictly the seed-group surfacing.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import {
    fetchSeedGroupDrillDown,
    type SeedGroupChildEvalResult,
} from '../../api/seedGroupDrillDown';
import './AggregateRunBadge.css';


export interface VarianceBlock {
    mean: number;
    std: number;
    min: number;
    max: number;
    n: number;
}


// Top-level metric keys we know are the "headline" ones a user wants
// to see at a glance. Order matters — we render in this order so the
// badge layout is stable across runs. Anything not in the list still
// flows through as the long-tail row when its leaf is a variance dict
// (per-slice / per-class drill-down is slice 2).
const HEADLINE_METRICS: ReadonlyArray<string> = [
    'pass_rate',
    'accuracy',
    'f1',
    'precision',
    'recall',
    'exact_match',
];


export function isVarianceBlock(value: unknown): value is VarianceBlock {
    if (typeof value !== 'object' || value === null) return false;
    const v = value as Record<string, unknown>;
    return (
        typeof v.mean === 'number'
        && typeof v.std === 'number'
        && typeof v.n === 'number'
    );
}


/** Walk a metrics dict and emit one (path, block) per top-level
 *  variance-bearing leaf — does NOT descend into per-slice/per-class
 *  nesting (that's slice 2's job). The badge stays scoped to the
 *  primary headline metrics so the layout is short. */
export function extractTopLevelVarianceMetrics(
    metrics: Record<string, unknown>,
): Array<{ name: string; block: VarianceBlock }> {
    const rows: Array<{ name: string; block: VarianceBlock }> = [];
    for (const name of HEADLINE_METRICS) {
        const value = metrics[name];
        if (isVarianceBlock(value)) {
            rows.push({ name, block: value });
        }
    }
    // Long-tail headline-shaped variance blocks (top-level keys not in
    // the curated list, e.g. ``macro_f1``). Render after the curated
    // rows so the curated layout is stable.
    for (const [name, value] of Object.entries(metrics)) {
        if (HEADLINE_METRICS.includes(name)) continue;
        if (isVarianceBlock(value)) {
            rows.push({ name, block: value });
        }
    }
    return rows;
}


function formatVariance(block: VarianceBlock): string {
    // 3-decimal precision is enough for headline metrics — anything
    // finer is noise relative to typical eval-set sizes. Drop trailing
    // zeros so 0.83 ± 0.04 doesn't render as 0.830 ± 0.040.
    const fmt = (v: number): string => {
        const s = v.toFixed(3);
        return s.replace(/\.?0+$/, '') || '0';
    };
    return `${fmt(block.mean)} ± ${fmt(block.std)}`;
}


export interface AggregateRunBadgeProps {
    projectId: number;
    seedGroupId: string;
    datasetName: string;
    evalType: string;
    metrics: Record<string, unknown>;
    /** Auto-open the drill-down on mount. Default false. Lets a
     *  caller (e.g. a coach-action that landed the user here) skip
     *  the click. */
    initiallyExpanded?: boolean;
}


export default function AggregateRunBadge({
    projectId,
    seedGroupId,
    datasetName,
    evalType,
    metrics,
    initiallyExpanded = false,
}: AggregateRunBadgeProps) {
    const headlineRows = useMemo(
        () => extractTopLevelVarianceMetrics(metrics),
        [metrics],
    );
    const [expanded, setExpanded] = useState(initiallyExpanded);
    const [children, setChildren] = useState<SeedGroupChildEvalResult[] | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const resp = await fetchSeedGroupDrillDown(projectId, seedGroupId, {
                datasetName, evalType,
            });
            setChildren(resp.children);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Failed to load per-seed runs.');
        } finally {
            setLoading(false);
        }
    }, [projectId, seedGroupId, datasetName, evalType]);

    useEffect(() => {
        if (expanded && children === null && !loading && !error) {
            void load();
        }
    }, [expanded, children, loading, error, load]);

    // The headline n+seed count — if no headline metric is a variance
    // block but the row IS aggregate (rare; might happen for a
    // string-only or pass-rate-only payload), fall back to a generic
    // label so the badge still renders.
    const headlineN: number | null = headlineRows[0]?.block.n ?? null;

    // Pick which metric columns to render in the per-seed drill-down
    // table — the same set the badge headlined, in the same order, so
    // the column-to-row mapping is obvious to the eye.
    const drillColumns: string[] = useMemo(
        () => headlineRows.map((r) => r.name),
        [headlineRows],
    );

    return (
        <div
            className="agg-run-badge"
            data-testid="aggregate-run-badge"
        >
            <div className="agg-run-badge__header">
                <button
                    type="button"
                    className="agg-run-badge__toggle"
                    onClick={() => setExpanded((v) => !v)}
                    aria-label={expanded ? 'Hide per-seed runs' : 'View per-seed runs'}
                    data-testid="agg-run-badge-toggle"
                >
                    {expanded ? (
                        <ChevronDown size={14} aria-hidden="true" />
                    ) : (
                        <ChevronRight size={14} aria-hidden="true" />
                    )}
                </button>
                <span className="agg-run-badge__title">
                    Multi-seed aggregate
                </span>
                {headlineN !== null && (
                    <span
                        className="agg-run-badge__count"
                        data-testid="agg-run-badge-count"
                    >
                        n={headlineN}
                    </span>
                )}
                <span className="agg-run-badge__group-id" title="seed_group_id">
                    {seedGroupId.slice(0, 8)}
                </span>
            </div>
            {headlineRows.length > 0 && (
                <ul
                    className="agg-run-badge__headline-list"
                    data-testid="agg-run-badge-headline-list"
                >
                    {headlineRows.map(({ name, block }) => (
                        <li
                            key={name}
                            className="agg-run-badge__headline-row"
                            data-testid={`agg-run-badge-headline-${name}`}
                        >
                            <span className="agg-run-badge__metric-name">{name}</span>
                            <span className="agg-run-badge__metric-value">
                                {formatVariance(block)}
                            </span>
                            <span
                                className="agg-run-badge__metric-range"
                                title={`min=${block.min}, max=${block.max}`}
                            >
                                [{block.min.toFixed(2)}, {block.max.toFixed(2)}]
                            </span>
                        </li>
                    ))}
                </ul>
            )}
            {expanded && (
                <div className="agg-run-badge__drill" data-testid="agg-run-badge-drill">
                    {loading && (
                        <div className="agg-run-badge__loading">Loading per-seed runs…</div>
                    )}
                    {error && (
                        <div className="agg-run-badge__error">
                            {error}{' '}
                            <button
                                type="button"
                                className="btn btn-ghost btn-sm"
                                onClick={() => void load()}
                            >
                                Retry
                            </button>
                        </div>
                    )}
                    {children !== null && !loading && !error && (
                        <table
                            className="agg-run-badge__table"
                            data-testid="agg-run-badge-table"
                        >
                            <thead>
                                <tr>
                                    <th>seed</th>
                                    <th>status</th>
                                    {drillColumns.map((col) => (
                                        <th key={col}>{col}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {children.map((row, idx) => (
                                    <tr
                                        key={`${row.experiment_id}-${idx}`}
                                        className={
                                            row.experiment_status !== 'completed'
                                                ? 'agg-run-badge__row--failed'
                                                : ''
                                        }
                                        data-testid={`agg-run-badge-row-${row.seed_value ?? 'na'}`}
                                    >
                                        <td>{row.seed_value ?? '—'}</td>
                                        <td>{row.experiment_status}</td>
                                        {drillColumns.map((col) => {
                                            const raw = row.metrics[col];
                                            const display = typeof raw === 'number'
                                                ? raw.toFixed(3).replace(/\.?0+$/, '') || '0'
                                                : '—';
                                            return <td key={col}>{display}</td>;
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            )}
        </div>
    );
}
