/**
 * Quality-Lift phase 8 slice 3 — BehavioralResultsTable.
 *
 * Closes the phase 5 visibility gap surfaced by the take-stock:
 * ScorecardPanel only renders behavioral pass-rates inside gate rows
 * (``isBehavioralGate(g)``), so a user who authored an INV/DIR/MFT
 * test but didn't gate it had no read surface for the result. This
 * table reads ``metrics["behavioral"][test_id]`` from the EvalResult
 * directly — independent of which tests are gated — so every
 * authored test gets a one-screen breakdown.
 *
 * Shape (matches behavioral_test_runner.run_behavioral_tests):
 *   metrics["behavioral"][test_id] = {
 *     kind: "INV" | "DIR" | "MFT",
 *     passed, total, pass_rate,           // scalars OR variance blocks
 *     failed_examples,                    // capped for JSON budget
 *     per_slice?: { [slice_id]: same shape, scoped to one slice },
 *     capped_at_budget?: number,          // present when sampled
 *   }
 *
 * No thresholds invented (honest-metrics-no-vanity rule). Cells
 * render the raw pass-rate; cells with ``total=0`` render with the
 * ``--empty`` class so "this test had no trials on this slice"
 * doesn't look like "perfect 0% pass-rate."
 *
 * Per-slice expander reuses the rendering primitives from the
 * top-level row — same column layout, indented under the parent
 * test. Per-slice rows ONLY render under the expander; they don't
 * pollute the top-level scan.
 */

import { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import './BehavioralResultsTable.css';


export interface BehavioralResultsTableProps {
    metrics: Record<string, unknown>;
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


function formatRate(value: unknown): string {
    // Pass-rate is 0..1 in both the scalar and variance-block forms;
    // we format as the same 3-dp strip the AggregateRunBadge uses
    // so the page-wide formatting reads consistently.
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


function formatCount(value: unknown): string {
    if (typeof value === 'number') return String(Math.round(value));
    if (isVarianceBlock(value)) {
        // Multi-seed aggregate counts — show the mean as a rounded
        // int (count semantics) plus the spread.
        const fmt = (v: number): string => v.toFixed(1).replace(/\.0$/, '');
        return `${fmt(value.mean)} ± ${fmt(value.std)}`;
    }
    return '—';
}


/** Did the test's total resolve to literally zero? Works for scalar
 *  and variance-block totals (mean=0 → empty). When ``total`` is
 *  missing we fall through as "not empty" because we can't tell. */
function isEmptyTotal(value: unknown): boolean {
    if (typeof value === 'number') return value === 0;
    if (isVarianceBlock(value)) return value.mean === 0;
    return false;
}


interface BehavioralTestBlock {
    kind?: string;
    passed?: unknown;
    total?: unknown;
    pass_rate?: unknown;
    per_slice?: Record<string, Record<string, unknown>>;
    capped_at_budget?: number;
}


export function extractBehavioralBlocks(
    metrics: Record<string, unknown>,
): Array<{ testId: string; block: BehavioralTestBlock }> {
    const raw = metrics?.['behavioral'];
    if (typeof raw !== 'object' || raw === null) return [];
    const out: Array<{ testId: string; block: BehavioralTestBlock }> = [];
    for (const [testId, value] of Object.entries(raw)) {
        if (typeof value === 'object' && value !== null) {
            out.push({ testId, block: value as BehavioralTestBlock });
        }
    }
    // Stable alphabetical order — same convention the pack editor uses.
    out.sort((a, b) => a.testId.localeCompare(b.testId));
    return out;
}


function PerSliceRow({
    sliceId,
    block,
}: {
    sliceId: string;
    block: Record<string, unknown>;
}) {
    const empty = isEmptyTotal(block.total);
    return (
        <tr
            className={`behavioral-results__row--per-slice ${empty ? 'behavioral-results__row--empty' : ''}`}
            data-testid={`behavioral-slice-row-${sliceId}`}
        >
            <th scope="row" className="behavioral-results__slice-label">
                <span className="behavioral-results__indent" aria-hidden="true">↳</span>
                {sliceId}
                {empty && (
                    <span className="behavioral-results__zero-tag">total=0</span>
                )}
            </th>
            <td>{typeof block.kind === 'string' ? block.kind : '—'}</td>
            <td>{formatRate(block.pass_rate)}</td>
            <td>{formatCount(block.passed)}</td>
            <td>{formatCount(block.total)}</td>
            <td>{/* capped column — unused on per-slice rows */}</td>
        </tr>
    );
}


function TestRow({
    testId,
    block,
}: {
    testId: string;
    block: BehavioralTestBlock;
}) {
    const perSlice = block.per_slice ?? {};
    const sliceIds = Object.keys(perSlice).sort();
    const hasSlices = sliceIds.length > 0;
    const [expanded, setExpanded] = useState(false);
    const empty = isEmptyTotal(block.total);
    return (
        <>
            <tr
                className={empty ? 'behavioral-results__row--empty' : ''}
                data-testid={`behavioral-test-row-${testId}`}
            >
                <th scope="row" className="behavioral-results__test-label">
                    {hasSlices ? (
                        <button
                            type="button"
                            className="behavioral-results__expand"
                            onClick={() => setExpanded((v) => !v)}
                            aria-label={
                                expanded
                                    ? `Collapse per-slice rows for ${testId}`
                                    : `Expand per-slice rows for ${testId}`
                            }
                            data-testid={`behavioral-test-expand-${testId}`}
                        >
                            {expanded ? (
                                <ChevronDown size={12} aria-hidden="true" />
                            ) : (
                                <ChevronRight size={12} aria-hidden="true" />
                            )}
                        </button>
                    ) : (
                        <span className="behavioral-results__expand-placeholder" />
                    )}
                    <span className="behavioral-results__test-id">{testId}</span>
                    {empty && (
                        <span className="behavioral-results__zero-tag">total=0</span>
                    )}
                </th>
                <td>{typeof block.kind === 'string' ? block.kind : '—'}</td>
                <td data-testid={`behavioral-test-rate-${testId}`}>
                    {formatRate(block.pass_rate)}
                </td>
                <td>{formatCount(block.passed)}</td>
                <td>{formatCount(block.total)}</td>
                <td>
                    {block.capped_at_budget !== undefined && (
                        <span
                            className="behavioral-results__capped-tag"
                            title={`Sampled to ${block.capped_at_budget} trials (full population was larger).`}
                            data-testid={`behavioral-test-capped-${testId}`}
                        >
                            capped @ {block.capped_at_budget}
                        </span>
                    )}
                </td>
            </tr>
            {hasSlices && expanded && sliceIds.map((sliceId) => (
                <PerSliceRow
                    key={`${testId}-${sliceId}`}
                    sliceId={sliceId}
                    block={perSlice[sliceId]}
                />
            ))}
        </>
    );
}


export default function BehavioralResultsTable({ metrics }: BehavioralResultsTableProps) {
    const tests = useMemo(() => extractBehavioralBlocks(metrics), [metrics]);
    if (tests.length === 0) return null;

    return (
        <section
            className="behavioral-results"
            data-testid="behavioral-results-table"
        >
            <header className="behavioral-results__header">
                <h4>Behavioral tests</h4>
                <p className="behavioral-results__hint">
                    Pass-rate per test from the behavioral runner —
                    one row per authored test, independent of whether
                    a gate references it. Click <code>▶</code> on a
                    row to see the per-slice breakdown.
                </p>
            </header>
            <div className="behavioral-results__scroll">
                <table data-testid="behavioral-results-table-table">
                    <thead>
                        <tr>
                            <th scope="col">test_id</th>
                            <th scope="col">kind</th>
                            <th scope="col">pass_rate</th>
                            <th scope="col">passed</th>
                            <th scope="col">total</th>
                            <th scope="col">notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tests.map(({ testId, block }) => (
                            <TestRow
                                key={testId}
                                testId={testId}
                                block={block}
                            />
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}
