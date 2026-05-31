/**
 * GoldSetDiagnosticsPanel — V4 of the ML-native visualisations arc.
 *
 * Two views of the project's gold set side-by-side:
 *
 *  - **Class balance bars** — per-label count with a share-of-total
 *    bar, sorted descending. Class entropy reported in the header so
 *    "imbalanced" reads as a number, not a vibe. The 15%-of-total
 *    line (the same imbalance threshold the trainability forecast
 *    uses to flag under-represented classes) is rendered on the bars
 *    so a class drifting below it is visible without doing the math.
 *  - **Class similarity heatmap** — rows × cols = labels. Cells show
 *    mean pairwise Jaccard between rows of each class:
 *      * Diagonal = intra-class similarity. High = the rows of one
 *        class look the same as each other (low diversity, model
 *        learns nothing). Low = good variety in that class.
 *      * Off-diagonal = inter-class confusability. High = even a
 *        perfect classifier can't tell the classes apart from text
 *        alone. Low = the classes are separable.
 *    Cells with too few rows to score (a class with one example) are
 *    rendered as "n/a" tiles rather than fabricated zero-Jaccard
 *    values — surfacing what's actually unmeasured.
 *
 * Backed by GET /api/projects/{id}/gold/diagnostics. The endpoint
 * always returns 200; classification_eligible=false drives the
 * empty-state hint ("n/a for this recipe shape — class balance is a
 * classification concept").
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import './GoldSetDiagnosticsPanel.css';

interface ClassBalanceEntry {
    label: string;
    count: number;
    share: number;
}

interface ClassBalance {
    labels: ClassBalanceEntry[];
    total: number;
    entropy_nats: number;
}

interface Similarity {
    labels: string[];
    matrix: (number | null)[][];
    sample_per_class: number;
    insufficient_labels: string[];
}

interface DiagnosticsResponse {
    project_id: number;
    total_rows: number;
    classification_eligible: boolean;
    class_balance: ClassBalance;
    similarity: Similarity;
}

interface GoldSetDiagnosticsPanelProps {
    projectId: number;
}

// Same imbalance threshold the trainability forecast uses to flag
// under-represented classes (see `_signal_class_imbalance` in the
// backend). Keeping the line on the bars at the same value the
// backend gate uses means "the bar is below the line" reads the
// same as "this class will be flagged by Coach Mode" — no double-
// vocabulary for the user.
const IMBALANCE_THRESHOLD = 0.15;

// Heatmap palette — green at 0 (low similarity, separable) → yellow
// → red at 1 (high similarity, confusable / redundant). We map to
// inline `background` because each cell's colour is data-driven.
function heatmapFill(value: number | null): string {
    if (value === null) return 'rgba(148, 163, 184, 0.15)';
    // Lerp green → yellow at 0.5 → red at 1.0 with a slight gamma so
    // mid-range cells are still distinguishable from the ends.
    const v = Math.max(0, Math.min(1, value));
    if (v < 0.5) {
        // green to yellow
        const t = v / 0.5;
        const r = Math.round(34 + (234 - 34) * t);
        const g = Math.round(197 + (179 - 197) * t);
        const b = Math.round(94 + (8 - 94) * t);
        return `rgba(${r}, ${g}, ${b}, ${(0.18 + v * 0.5).toFixed(2)})`;
    }
    // yellow to red
    const t = (v - 0.5) / 0.5;
    const r = Math.round(234 + (239 - 234) * t);
    const g = Math.round(179 + (68 - 179) * t);
    const b = Math.round(8 + (68 - 8) * t);
    return `rgba(${r}, ${g}, ${b}, ${(0.18 + v * 0.5).toFixed(2)})`;
}

export default function GoldSetDiagnosticsPanel({ projectId }: GoldSetDiagnosticsPanelProps) {
    const [data, setData] = useState<DiagnosticsResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const fetch = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const res = await api.get<DiagnosticsResponse>(
                `/projects/${projectId}/gold/diagnostics`,
            );
            setData(res.data);
        } catch (err) {
            const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setError(typeof detail === 'string' ? detail : 'Failed to load gold-set diagnostics.');
            setData(null);
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetch();
    }, [fetch]);

    if (loading && !data) {
        return (
            <div className="gold-diag gold-diag--loading" data-testid="gold-diag">
                Loading gold-set diagnostics…
            </div>
        );
    }

    if (error) {
        return (
            <div className="gold-diag gold-diag--error" data-testid="gold-diag">
                {error}{' '}
                <button type="button" className="btn btn-link" onClick={() => void fetch()}>
                    Retry
                </button>
            </div>
        );
    }

    if (!data) return null;

    if (!data.classification_eligible || data.class_balance.labels.length === 0) {
        return (
            <section className="gold-diag" data-testid="gold-diag">
                <header className="gold-diag__head">
                    <h3 className="gold-diag__title">Gold-set diagnostics</h3>
                </header>
                <p className="gold-diag__empty" data-testid="gold-diag-empty">
                    {data.total_rows === 0 ? (
                        <>No gold rows yet — diagnostics activate once the gold set has at least a few labeled examples.</>
                    ) : (
                        <>No classification labels in this gold set. Class balance + class-similarity heatmap are classification-shape diagnostics; for span-extraction or summarization gold sets this panel stays quiet.</>
                    )}
                </p>
            </section>
        );
    }

    const balance = data.class_balance;
    const sim = data.similarity;

    return (
        <section className="gold-diag" data-testid="gold-diag">
            <header className="gold-diag__head">
                <h3 className="gold-diag__title">Gold-set diagnostics</h3>
                <span className="gold-diag__head-meta">
                    {balance.total} rows · {balance.labels.length} classes · entropy{' '}
                    <strong>{balance.entropy_nats.toFixed(2)}</strong> nats
                </span>
            </header>

            {/* Class-balance bars ------------------------------------ */}
            <div className="gold-diag__balance" data-testid="gold-diag-balance">
                <p className="gold-diag__hint">
                    Class balance — share of the gold set per label, sorted descending. The dashed
                    line marks the <strong>{(IMBALANCE_THRESHOLD * 100).toFixed(0)}%</strong>{' '}
                    floor Coach Mode uses to flag under-represented classes; a bar below the line
                    will fire the imbalance signal at training time.
                </p>
                <ul className="gold-diag__balance-list">
                    {balance.labels.map((entry) => {
                        const widthPct = Math.max(2, Math.round(entry.share * 100));
                        const belowFloor = entry.share < IMBALANCE_THRESHOLD;
                        return (
                            <li
                                key={entry.label}
                                className={`gold-diag__balance-row ${belowFloor ? 'is-below-floor' : ''}`}
                                data-testid={`gold-diag-balance-${entry.label}`}
                            >
                                <span className="gold-diag__balance-label" title={entry.label}>
                                    {entry.label}
                                </span>
                                <span className="gold-diag__balance-bar">
                                    <span
                                        className={`gold-diag__balance-fill gold-diag__balance-fill--w-${roundToTen(widthPct)}`}
                                    />
                                    <span
                                        className="gold-diag__balance-threshold"
                                        aria-hidden="true"
                                    />
                                </span>
                                <span className="gold-diag__balance-count">
                                    {entry.count} <span className="gold-diag__balance-share">({(entry.share * 100).toFixed(0)}%)</span>
                                </span>
                            </li>
                        );
                    })}
                </ul>
            </div>

            {/* Class similarity heatmap ---------------------------- */}
            {sim.labels.length >= 2 && (
                <div className="gold-diag__matrix" data-testid="gold-diag-matrix">
                    <p className="gold-diag__hint">
                        Class similarity — mean pairwise Jaccard between rows of each class
                        (sample of {sim.sample_per_class} per class). <strong>Diagonal cells</strong>
                        {' '}measure intra-class redundancy (high = rows look the same → low
                        diversity); <strong>off-diagonal cells</strong> measure inter-class
                        confusability (high = even a perfect classifier can't separate them
                        from text alone).
                    </p>
                    <table className="gold-diag__matrix-table">
                        <thead>
                            <tr>
                                <th></th>
                                {sim.labels.map((label) => (
                                    <th key={label} className="gold-diag__matrix-col">
                                        {label}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {sim.labels.map((rowLabel, ri) => (
                                <tr key={rowLabel}>
                                    <th className="gold-diag__matrix-row-label">{rowLabel}</th>
                                    {sim.labels.map((colLabel, ci) => {
                                        const value = sim.matrix[ri]?.[ci] ?? null;
                                        const isDiagonal = ri === ci;
                                        return (
                                            <td
                                                key={colLabel}
                                                className={`gold-diag__matrix-cell ${isDiagonal ? 'is-diagonal' : ''}`}
                                                data-testid={`gold-diag-cell-${rowLabel}-${colLabel}`}
                                                style={{ background: heatmapFill(value) }}
                                                title={
                                                    value === null
                                                        ? `${rowLabel} → ${colLabel}: not enough rows to score`
                                                        : `${rowLabel} → ${colLabel}: ${(value * 100).toFixed(0)}% mean Jaccard`
                                                }
                                            >
                                                {value === null ? (
                                                    <span className="gold-diag__matrix-na">n/a</span>
                                                ) : (
                                                    value.toFixed(2)
                                                )}
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    {sim.insufficient_labels.length > 0 && (
                        <p className="gold-diag__matrix-foot" data-testid="gold-diag-insufficient">
                            <strong>{sim.insufficient_labels.join(', ')}</strong>:
                            not enough rows to score similarity. Add more labeled examples to
                            measure this class's diversity.
                        </p>
                    )}
                </div>
            )}
        </section>
    );
}

// Snap a pct to the nearest 10 for the bar-width CSS class lookup —
// avoids inline styles per CLAUDE.md's CSS lint guidance (dynamic
// widths flagged as warnings; bucketed classes are clean).
function roundToTen(pct: number): number {
    return Math.max(0, Math.min(100, Math.round(pct / 10) * 10));
}
