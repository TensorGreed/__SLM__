/**
 * ClassificationChartsPanel — V1 of the ML-native visualisations arc.
 *
 * Renders two complementary views of a classification eval result:
 *
 *  1. **Per-class P/R/F1 bars** — sorted by F1 ascending, so the worst
 *     class lands at the top of the list (the one the user actually
 *     needs to look at). Support count rendered as a chip on each row
 *     so "this class is rare AND the model is bad at it" is one glance.
 *
 *  2. **Confusion matrix heatmap** — gold rows × predicted columns. Cell
 *     intensity scales with the share of that gold-class row's
 *     predictions; diagonal cells get a green tint when the share is
 *     high. An "__unparseable__" column surfaces when the model emitted
 *     a label outside the candidate set, distinct from "predicted
 *     wrong known class" — both are mistakes but the fix is different.
 *
 * The panel is data-driven: it renders whenever the eval result carries
 * `per_class` AND `confusion_matrix` (the classification handler always
 * emits both). For other task profiles it returns null silently.
 */

import { useMemo } from 'react';

export interface PerClassMetric {
    precision: number;
    recall: number;
    f1: number;
    support: number;
}

interface ClassificationChartsPanelProps {
    perClass: Record<string, PerClassMetric>;
    confusionMatrix: Record<string, Record<string, number>>;
    candidates?: string[];
    macroF1?: number | null;
    accuracy?: number | null;
}

const BAR_W = 120;
const BAR_H = 10;
const ROW_H = 28;
const LABEL_COL = 140;
const SUPPORT_COL = 60;

// Bar colours — kept consistent with the badge palette elsewhere
// (success/warning/error). The thresholds match the eval-results pass-rate
// chips so a class with F1=0.81 reads as "green" the same way the overall
// pass-rate does at 81%.
function barClass(value: number): string {
    if (value >= 0.8) return 'is-good';
    if (value >= 0.5) return 'is-mid';
    return 'is-bad';
}

// Heatmap intensity for the confusion matrix. Diagonal cells (correct
// predictions) get a green ramp; off-diagonal cells get a red ramp scaled
// by the count's share of the gold-row total. Returns an inline `fill`
// because each cell's colour is dynamic — there's no class-bucket that
// can encode "this exact 0.42 share" responsively.
function heatmapFill(share: number, isDiagonal: boolean, isUnparseable: boolean): string {
    const a = Math.max(0.04, Math.min(0.92, share));
    if (share === 0) return 'rgba(148, 163, 184, 0.04)';
    if (isUnparseable) return `rgba(234, 88, 12, ${a.toFixed(2)})`;
    if (isDiagonal) return `rgba(34, 197, 94, ${a.toFixed(2)})`;
    return `rgba(239, 68, 68, ${a.toFixed(2)})`;
}

export default function ClassificationChartsPanel({
    perClass,
    confusionMatrix,
    candidates,
    macroF1,
    accuracy,
}: ClassificationChartsPanelProps) {
    const sortedClasses = useMemo(() => {
        const entries = Object.entries(perClass || {});
        // Sort by F1 ascending so the worst class lands at the top — the
        // row the user needs to look at first. Ties broken by lower
        // support (rarer classes surface higher among same-F1 classes).
        return entries.sort((a, b) => {
            if (a[1].f1 !== b[1].f1) return a[1].f1 - b[1].f1;
            return a[1].support - b[1].support;
        });
    }, [perClass]);

    // Confusion matrix columns: prefer the candidate set order so it
    // matches what the gold labels look like; fall back to the union of
    // keys we actually see. Append "__unparseable__" at the end whenever
    // it appears in any row — distinct visual treatment.
    const matrixColumns = useMemo(() => {
        const seen = new Set<string>();
        const order: string[] = [];
        for (const label of candidates || []) {
            if (!seen.has(label)) {
                order.push(label);
                seen.add(label);
            }
        }
        for (const row of Object.values(confusionMatrix || {})) {
            for (const col of Object.keys(row || {})) {
                if (!seen.has(col)) {
                    order.push(col);
                    seen.add(col);
                }
            }
        }
        // Move "__unparseable__" to the end if present — it's not a real
        // candidate class, surfacing it next to real classes would
        // visually equate "wrong known class" with "emitted gibberish".
        const idx = order.indexOf('__unparseable__');
        if (idx !== -1) {
            order.splice(idx, 1);
            order.push('__unparseable__');
        }
        return order;
    }, [candidates, confusionMatrix]);

    const matrixRows = useMemo(() => {
        // Gold rows: candidate set order again (skip any candidate
        // missing from the matrix, e.g. classes with zero gold).
        const knownRows = Object.keys(confusionMatrix || {});
        const seen = new Set<string>();
        const order: string[] = [];
        for (const label of candidates || []) {
            if (knownRows.includes(label) && !seen.has(label)) {
                order.push(label);
                seen.add(label);
            }
        }
        for (const label of knownRows) {
            if (!seen.has(label)) {
                order.push(label);
                seen.add(label);
            }
        }
        return order;
    }, [candidates, confusionMatrix]);

    const hasPerClass = sortedClasses.length > 0;
    const hasMatrix = matrixRows.length > 0 && matrixColumns.length > 0;
    if (!hasPerClass && !hasMatrix) return null;

    const barsHeight = sortedClasses.length * ROW_H + 28; // header + rows
    const barsWidth = LABEL_COL + 3 * BAR_W + 60 + SUPPORT_COL;

    const cellSize = 28;
    const matrixPaddingLeft = 110; // row labels
    const matrixPaddingTop = 64;   // rotated column labels
    const matrixWidth = matrixPaddingLeft + matrixColumns.length * cellSize + 16;
    const matrixHeight = matrixPaddingTop + matrixRows.length * cellSize + 16;

    return (
        <section className="cls-charts" data-testid="cls-charts">
            <header className="cls-charts__head">
                <h3 className="cls-charts__title">Classification breakdown</h3>
                <span className="cls-charts__head-meta">
                    {accuracy != null && (
                        <>accuracy <strong>{(accuracy * 100).toFixed(1)}%</strong></>
                    )}
                    {macroF1 != null && (
                        <>{accuracy != null && ' · '}macro-F1 <strong>{(macroF1 * 100).toFixed(1)}%</strong></>
                    )}
                </span>
            </header>

            {hasPerClass && (
                <div className="cls-charts__bars" data-testid="cls-charts-bars">
                    <p className="cls-charts__hint">
                        Sorted by F1 ascending — the row at the top is the class the model is
                        worst at. Support is the count of that class in the eval set.
                    </p>
                    <svg
                        className="cls-charts__bars-svg"
                        role="img"
                        aria-label="Per-class precision, recall, and F1 bars"
                        viewBox={`0 0 ${barsWidth} ${barsHeight}`}
                    >
                        {/* Header row */}
                        <g className="cls-charts__bars-header">
                            <text x={LABEL_COL - 10} y={14} textAnchor="end">class</text>
                            <text x={LABEL_COL + BAR_W / 2} y={14} textAnchor="middle">precision</text>
                            <text x={LABEL_COL + BAR_W + 20 + BAR_W / 2} y={14} textAnchor="middle">recall</text>
                            <text x={LABEL_COL + 2 * BAR_W + 40 + BAR_W / 2} y={14} textAnchor="middle">F1</text>
                            <text x={LABEL_COL + 3 * BAR_W + 60 + SUPPORT_COL / 2} y={14} textAnchor="middle">support</text>
                        </g>
                        {sortedClasses.map(([label, m], i) => {
                            const y = 28 + i * ROW_H;
                            return (
                                <g
                                    key={label}
                                    className="cls-charts__bars-row"
                                    data-testid={`cls-bar-${label}`}
                                >
                                    <text x={LABEL_COL - 10} y={y + BAR_H + 6} textAnchor="end" className="cls-charts__bars-label">
                                        {label.length > 18 ? label.slice(0, 17) + '…' : label}
                                        <title>{label}</title>
                                    </text>
                                    {[
                                        { val: m.precision, x: LABEL_COL },
                                        { val: m.recall, x: LABEL_COL + BAR_W + 20 },
                                        { val: m.f1, x: LABEL_COL + 2 * BAR_W + 40 },
                                    ].map((b, idx) => (
                                        <g key={idx} className={`cls-charts__bar ${barClass(b.val)}`}>
                                            <rect x={b.x} y={y + 1} width={BAR_W} height={BAR_H} className="cls-charts__bar-bg" />
                                            <rect x={b.x} y={y + 1} width={Math.max(2, b.val * BAR_W)} height={BAR_H} className="cls-charts__bar-fill" />
                                            <text x={b.x + BAR_W + 6} y={y + BAR_H + 1} className="cls-charts__bar-value">
                                                {b.val.toFixed(2)}
                                            </text>
                                        </g>
                                    ))}
                                    <text
                                        x={LABEL_COL + 3 * BAR_W + 60 + SUPPORT_COL / 2}
                                        y={y + BAR_H + 6}
                                        textAnchor="middle"
                                        className="cls-charts__bars-support"
                                    >
                                        {m.support}
                                    </text>
                                </g>
                            );
                        })}
                    </svg>
                </div>
            )}

            {hasMatrix && (
                <div className="cls-charts__matrix" data-testid="cls-charts-matrix">
                    <p className="cls-charts__hint">
                        Confusion matrix — rows are the true label, columns are what the model
                        predicted. Diagonal green = correct; off-diagonal red = systematic
                        mistake. Cell intensity scales with the row's share, so each row
                        compares predictions independently.
                    </p>
                    <svg
                        className="cls-charts__matrix-svg"
                        role="img"
                        aria-label="Confusion matrix heatmap"
                        viewBox={`0 0 ${matrixWidth} ${matrixHeight}`}
                    >
                        {/* Column labels (rotated) */}
                        {matrixColumns.map((col, ci) => {
                            const cx = matrixPaddingLeft + ci * cellSize + cellSize / 2;
                            const cy = matrixPaddingTop - 8;
                            const label = col === '__unparseable__' ? 'unparsed' : col;
                            return (
                                <text
                                    key={col}
                                    x={cx}
                                    y={cy}
                                    textAnchor="start"
                                    className={`cls-charts__matrix-col-label ${col === '__unparseable__' ? 'is-unparseable' : ''}`}
                                    transform={`rotate(-45 ${cx} ${cy})`}
                                >
                                    {label.length > 12 ? label.slice(0, 11) + '…' : label}
                                    <title>{col}</title>
                                </text>
                            );
                        })}
                        {/* Row labels + cells */}
                        {matrixRows.map((rowLabel, ri) => {
                            const row = confusionMatrix[rowLabel] || {};
                            const rowTotal = Object.values(row).reduce((s, v) => s + (Number(v) || 0), 0);
                            return (
                                <g key={rowLabel} data-testid={`cls-matrix-row-${rowLabel}`}>
                                    <text
                                        x={matrixPaddingLeft - 8}
                                        y={matrixPaddingTop + ri * cellSize + cellSize / 2 + 4}
                                        textAnchor="end"
                                        className="cls-charts__matrix-row-label"
                                    >
                                        {rowLabel.length > 14 ? rowLabel.slice(0, 13) + '…' : rowLabel}
                                        <title>{rowLabel}</title>
                                    </text>
                                    {matrixColumns.map((colLabel, ci) => {
                                        const count = Number(row[colLabel] || 0);
                                        const share = rowTotal > 0 ? count / rowTotal : 0;
                                        const isDiagonal = colLabel === rowLabel;
                                        const isUnparseable = colLabel === '__unparseable__';
                                        return (
                                            <g
                                                key={colLabel}
                                                data-testid={`cls-matrix-cell-${rowLabel}-${colLabel}`}
                                            >
                                                <rect
                                                    x={matrixPaddingLeft + ci * cellSize}
                                                    y={matrixPaddingTop + ri * cellSize}
                                                    width={cellSize - 1}
                                                    height={cellSize - 1}
                                                    className="cls-charts__matrix-cell"
                                                    fill={heatmapFill(share, isDiagonal, isUnparseable)}
                                                >
                                                    <title>{`gold=${rowLabel} → predicted=${colLabel}: ${count} (${(share * 100).toFixed(0)}%)`}</title>
                                                </rect>
                                                {count > 0 && (
                                                    <text
                                                        x={matrixPaddingLeft + ci * cellSize + cellSize / 2 - 1}
                                                        y={matrixPaddingTop + ri * cellSize + cellSize / 2 + 4}
                                                        textAnchor="middle"
                                                        className={`cls-charts__matrix-count ${share > 0.5 ? 'is-light' : ''}`}
                                                    >
                                                        {count}
                                                    </text>
                                                )}
                                            </g>
                                        );
                                    })}
                                </g>
                            );
                        })}
                    </svg>
                </div>
            )}
        </section>
    );
}
