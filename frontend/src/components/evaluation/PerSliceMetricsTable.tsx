/**
 * Quality-Lift phase 8 slice 2 — PerSliceMetricsTable.
 *
 * Mounted in EvalPanel directly below the AggregateRunBadge and above
 * the ScorecardPanel. Surfaces the EvalResult's
 * ``metrics["per_slice"][<slice_id>][<metric_name>]`` payload (written
 * by phase 2 slice 2's slice evaluator) as a comparable table so a
 * user with N slices doesn't have to expand one failing gate at a
 * time to learn which slice is broken.
 *
 * Cell highlighting reads the live gate report — every GateCheck whose
 * ``metric_id`` matches ``per_slice.<slice_id>.<metric_name>`` is
 * cross-referenced; cells whose gate failed get the
 * ``--failed`` class so the user can scan the table and see "the
 * accuracy threshold of 0.80 is missed on long_input + abuse_report"
 * at a glance. Cells without a gate render plain (we don't invent
 * thresholds; honest gates only).
 *
 * Multi-seed aggregate rows are handled transparently — a per-slice
 * leaf can be a scalar or a ``{mean, std, …}`` variance block, and
 * the formatter falls through to ``mean ± std`` when it's the latter.
 *
 * Slices defined but not present in the metrics dict still render
 * (all-dash row) so the user sees "this slice matched zero rows" or
 * "the run predates this slice definition" rather than silently
 * dropping the row.
 *
 * Slices present in the metrics dict but NOT in slice_definitions
 * (orphans — definition deleted after the run) also surface, with a
 * "orphan" hint so the user can either re-add the definition or
 * accept the ambiguity.
 */

import { useEffect, useMemo, useState } from 'react';

import {
    fetchSliceDefinitions,
    type SliceDefinition,
} from '../../api/sliceDefinitions';
import './PerSliceMetricsTable.css';


export interface GateCheckLike {
    metric_id: string;
    threshold: number | null;
    operator: 'gte' | 'lte' | string;
    passed: boolean;
    required?: boolean;
}


export interface PerSliceMetricsTableProps {
    projectId: number;
    /** ``metrics`` blob from EvalResult; we read ``per_slice`` off it. */
    metrics: Record<string, unknown>;
    /** Live gate checks from the gate report — used to highlight
     *  failing per-slice cells. Cells without a matching gate render
     *  plain. Optional: when the gate report isn't loaded yet (or
     *  the eval pack has no per-slice gates) we still render the
     *  table; we just can't highlight. */
    gateChecks?: GateCheckLike[];
    /** When true, render an empty-state hint even with no per_slice
     *  data — useful for surfacing "Define some slices to see this
     *  table" instead of nothing. Default false (silent). */
    showEmptyStateWhenNoSlices?: boolean;
}


interface VarianceBlock {
    mean: number;
    std: number;
    min?: number;
    max?: number;
    n?: number;
}


function isVarianceBlock(value: unknown): value is VarianceBlock {
    if (typeof value !== 'object' || value === null) return false;
    const v = value as Record<string, unknown>;
    return typeof v.mean === 'number' && typeof v.std === 'number';
}


function formatCellValue(value: unknown): string {
    if (typeof value === 'number') {
        const s = value.toFixed(3);
        return s.replace(/\.?0+$/, '') || '0';
    }
    if (isVarianceBlock(value)) {
        const fmt = (v: number): string => {
            const s = v.toFixed(3);
            return s.replace(/\.?0+$/, '') || '0';
        };
        return `${fmt(value.mean)} ± ${fmt(value.std)}`;
    }
    return '—';
}


/** Pull the numeric value out of a per-slice cell — scalar passes
 *  through, variance block falls back to mean. Returns null when the
 *  cell isn't numeric, so the gate-comparator can short-circuit. */
function numericValueOf(value: unknown): number | null {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (isVarianceBlock(value) && Number.isFinite(value.mean)) return value.mean;
    return null;
}


/** Did this metric_id failure check fire for this (slice_id, metric_name)?
 *  We accept any GateCheck whose ``metric_id`` ends with
 *  ``per_slice.<slice_id>.<metric_name>`` so both the bare and
 *  eval-type-scoped (``classification.per_slice.<id>.<name>``) forms
 *  match. Returns the matching gate if one exists, else null. */
function findMatchingGate(
    gateChecks: ReadonlyArray<GateCheckLike>,
    sliceId: string,
    metricName: string,
): GateCheckLike | null {
    const suffix = `per_slice.${sliceId}.${metricName}`;
    for (const gate of gateChecks) {
        if (
            gate.metric_id === suffix
            || gate.metric_id.endsWith(`.${suffix}`)
        ) {
            return gate;
        }
    }
    return null;
}


export default function PerSliceMetricsTable({
    projectId,
    metrics,
    gateChecks = [],
    showEmptyStateWhenNoSlices = false,
}: PerSliceMetricsTableProps) {
    const [sliceDefs, setSliceDefs] = useState<SliceDefinition[] | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        void (async () => {
            try {
                const resp = await fetchSliceDefinitions(projectId);
                if (!cancelled) setSliceDefs(resp.slice_definitions?.slices ?? []);
            } catch (err: unknown) {
                if (!cancelled) {
                    setLoadError(
                        err instanceof Error ? err.message : 'Failed to load slice definitions.',
                    );
                    setSliceDefs([]);
                }
            }
        })();
        return () => { cancelled = true; };
    }, [projectId]);

    const perSlice = useMemo(() => {
        const raw = metrics?.['per_slice'];
        if (typeof raw !== 'object' || raw === null) return {};
        return raw as Record<string, Record<string, unknown>>;
    }, [metrics]);

    const hasPerSliceData = Object.keys(perSlice).length > 0;
    const hasSliceDefs = sliceDefs !== null && sliceDefs.length > 0;

    // Row order: defined slices first (in definition order), then
    // any orphans (in alphabetical order). Stable across renders.
    const rowOrder: Array<{ sliceId: string; displayName: string; orphan: boolean }> = useMemo(() => {
        const rows: Array<{ sliceId: string; displayName: string; orphan: boolean }> = [];
        const seen = new Set<string>();
        for (const def of sliceDefs ?? []) {
            rows.push({
                sliceId: def.slice_id,
                displayName: def.display_name || def.slice_id,
                orphan: false,
            });
            seen.add(def.slice_id);
        }
        const orphans = Object.keys(perSlice)
            .filter((id) => !seen.has(id))
            .sort();
        for (const id of orphans) {
            rows.push({ sliceId: id, displayName: id, orphan: true });
        }
        return rows;
    }, [sliceDefs, perSlice]);

    // Column order: support first (always shown when any slice has
    // a numeric support count), then the union of numeric metric
    // names across all slices, alphabetized. Non-numeric leaves
    // (e.g. nested per_class dicts within per_slice) are dropped
    // from the column set — slice 2 stays scoped to the one-cell
    // headline, not nested drill-downs.
    const columns: string[] = useMemo(() => {
        const names = new Set<string>();
        let supportSeen = false;
        for (const cell of Object.values(perSlice)) {
            for (const [name, value] of Object.entries(cell)) {
                if (name === 'support') {
                    supportSeen = true;
                    continue;
                }
                if (numericValueOf(value) !== null) {
                    names.add(name);
                }
            }
        }
        const sorted = Array.from(names).sort();
        return supportSeen ? ['support', ...sorted] : sorted;
    }, [perSlice]);

    if (sliceDefs === null) {
        // Still loading definitions — render nothing rather than
        // a flash of empty table. The EvalPanel will repaint when
        // the fetch resolves.
        return null;
    }

    if (!hasPerSliceData && !hasSliceDefs) {
        // No slices defined AND no per_slice metrics in the run. The
        // user hasn't engaged with phase 2 at all; rendering an empty
        // shell would be noise. Caller can flip the prop on if they
        // want to nudge.
        if (!showEmptyStateWhenNoSlices) return null;
        return (
            <div
                className="per-slice-table__empty"
                data-testid="per-slice-empty"
            >
                Define slices in the eval pack to see per-slice metric
                breakdowns here.
            </div>
        );
    }

    return (
        <section
            className="per-slice-table"
            data-testid="per-slice-metrics-table"
        >
            <header className="per-slice-table__header">
                <h4>Per-slice metrics</h4>
                <p className="per-slice-table__hint">
                    One row per slice, one column per metric.
                    Failed-gate cells highlighted in red — every
                    highlight cites a real gate threshold the user
                    set, never an invented bar.
                </p>
            </header>
            {loadError && (
                <div className="per-slice-table__warning" role="alert">
                    {loadError}
                </div>
            )}
            <div className="per-slice-table__scroll">
                <table data-testid="per-slice-metrics-table-table">
                    <thead>
                        <tr>
                            <th scope="col">slice</th>
                            {columns.map((col) => (
                                <th key={col} scope="col">{col}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {rowOrder.map(({ sliceId, displayName, orphan }) => {
                            const cell = perSlice[sliceId] ?? {};
                            const supportRaw = cell['support'];
                            const supportNum = typeof supportRaw === 'number' ? supportRaw : null;
                            const emptySlice = supportNum === 0;
                            return (
                                <tr
                                    key={sliceId}
                                    className={[
                                        emptySlice ? 'per-slice-table__row--empty' : '',
                                        orphan ? 'per-slice-table__row--orphan' : '',
                                    ].filter(Boolean).join(' ')}
                                    data-testid={`per-slice-row-${sliceId}`}
                                >
                                    <th scope="row">
                                        <span className="per-slice-table__slice-name">
                                            {displayName}
                                        </span>
                                        {orphan && (
                                            <span
                                                className="per-slice-table__orphan-tag"
                                                title="This slice has metrics from a run that predates its definition (or the definition was deleted)."
                                                data-testid={`per-slice-orphan-${sliceId}`}
                                            >
                                                orphan
                                            </span>
                                        )}
                                        {emptySlice && (
                                            <span
                                                className="per-slice-table__zero-tag"
                                                title="The slice predicate matched 0 rows on this eval set."
                                            >
                                                support=0
                                            </span>
                                        )}
                                    </th>
                                    {columns.map((col) => {
                                        const value = cell[col];
                                        const matchingGate = findMatchingGate(
                                            gateChecks, sliceId, col,
                                        );
                                        const failed = matchingGate !== null && !matchingGate.passed;
                                        const display = formatCellValue(value);
                                        return (
                                            <td
                                                key={col}
                                                className={
                                                    failed ? 'per-slice-table__cell--failed' : ''
                                                }
                                                data-testid={`per-slice-cell-${sliceId}-${col}`}
                                                title={matchingGate
                                                    ? (matchingGate.passed
                                                        ? `Passes gate ${matchingGate.metric_id} (${matchingGate.operator} ${matchingGate.threshold ?? '?'})`
                                                        : `Fails gate ${matchingGate.metric_id} (${matchingGate.operator} ${matchingGate.threshold ?? '?'})`)
                                                    : undefined}
                                            >
                                                {display}
                                            </td>
                                        );
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </section>
    );
}
