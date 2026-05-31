/**
 * LossCurvePanel — V2 of the ML-native visualisations arc.
 *
 * Overlay of train-loss and eval-loss curves on the same axes. Reading
 * overfitting from a table of step/loss rows is muscle the user
 * shouldn't need: the divergence between training and validation
 * curves IS the diagnostic, and it lands in one glance when plotted.
 *
 * What the chart shows:
 *
 *  - **Train series** — solid line drawn from every metric row that
 *    carries a `train_loss`.
 *  - **Eval series** — dashed line with point markers, drawn from
 *    rows that carry an `eval_loss`. Eval is usually sparser than
 *    train (logged per-epoch vs per-step), so markers make each
 *    measurement legible without overloading the plot.
 *  - **Best-eval marker** — a vertical line at the step that
 *    achieved the minimum eval loss. That's the checkpoint a "promote
 *    the winner" action would target.
 *  - **Divergence shade** — when eval starts climbing back up after
 *    its minimum (i.e. some later eval point is meaningfully worse
 *    than the best), the region from best-eval-step → end is shaded.
 *    This is the overfitting region; further training is making the
 *    model worse on held-out data.
 *
 * The honest-beat hint under the chart names the divergence delta in
 * absolute terms ("eval climbed +0.18 from step 240 onward") rather
 * than just "overfitting detected" — so the reader can decide whether
 * the divergence is big enough to care about.
 */

import { useMemo } from 'react';

interface MetricPoint {
    step?: number | null;
    train_loss?: number | null;
    eval_loss?: number | null;
}

interface LossCurvePanelProps {
    metrics: MetricPoint[];
}

const W = 560;
const H = 260;
const PAD = { top: 18, right: 24, bottom: 40, left: 56 };

// Anything bigger than this much eval-loss climb from the minimum is
// surfaced as a divergence with a shaded region. 0.05 is empirically
// the smallest delta that's reliably above noise on a small SLM run.
// Below that, the eval climb is usually within step-to-step jitter and
// shading it would cry wolf.
const DIVERGENCE_MIN_DELTA = 0.05;

export default function LossCurvePanel({ metrics }: LossCurvePanelProps) {
    const model = useMemo(() => {
        // Filter out null/undefined BEFORE the Number coercion —
        // Number(null) returns 0, which would silently pass the isFinite
        // check and plot rows with no recorded loss at y=0.
        const trainPts = metrics
            .filter((m) => m.train_loss !== null && m.train_loss !== undefined)
            .map((m) => ({ step: Number(m.step), v: Number(m.train_loss) }))
            .filter((p) => Number.isFinite(p.step) && Number.isFinite(p.v))
            .sort((a, b) => a.step - b.step);
        const evalPts = metrics
            .filter((m) => m.eval_loss !== null && m.eval_loss !== undefined)
            .map((m) => ({ step: Number(m.step), v: Number(m.eval_loss) }))
            .filter((p) => Number.isFinite(p.step) && Number.isFinite(p.v))
            .sort((a, b) => a.step - b.step);

        if (trainPts.length === 0 && evalPts.length === 0) return null;

        const allValues = [...trainPts.map((p) => p.v), ...evalPts.map((p) => p.v)];
        const allSteps = [...trainPts.map((p) => p.step), ...evalPts.map((p) => p.step)];
        const yMin = Math.min(...allValues);
        const yMax = Math.max(...allValues);
        const ySpan = (yMax - yMin) || 1;
        // Pad y range 5% on each end so the lines don't touch the frame.
        const yLo = yMin - ySpan * 0.05;
        const yHi = yMax + ySpan * 0.05;
        const ySpanPadded = (yHi - yLo) || 1;
        const xMin = Math.min(...allSteps);
        const xMax = Math.max(...allSteps);
        const xSpan = (xMax - xMin) || 1;

        const x = (step: number) =>
            PAD.left + ((step - xMin) / xSpan) * (W - PAD.left - PAD.right);
        const y = (v: number) =>
            PAD.top + (1 - (v - yLo) / ySpanPadded) * (H - PAD.top - PAD.bottom);

        const path = (pts: { step: number; v: number }[]) =>
            pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.step).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ');

        // Best eval = the step with the minimum eval_loss. If no eval
        // points exist, no marker / no divergence detection — chart is
        // just train-only (early in a run, or eval is disabled).
        let bestEval: { step: number; v: number } | null = null;
        let divergenceDelta = 0;
        if (evalPts.length > 0) {
            bestEval = evalPts.reduce((acc, p) => (p.v < acc.v ? p : acc), evalPts[0]);
            // The largest later eval point's climb above the minimum is
            // the divergence we shade.
            const lateValues = evalPts.filter((p) => p.step > bestEval!.step).map((p) => p.v);
            if (lateValues.length > 0) {
                divergenceDelta = Math.max(...lateValues) - bestEval.v;
            }
        }
        const isDiverging = bestEval !== null && divergenceDelta >= DIVERGENCE_MIN_DELTA;

        return {
            trainPath: path(trainPts),
            evalPath: path(evalPts),
            evalMarkers: evalPts.map((p) => ({ cx: x(p.step), cy: y(p.v), step: p.step, v: p.v })),
            xMin,
            xMax,
            yLo,
            yHi,
            bestEval,
            bestEvalX: bestEval ? x(bestEval.step) : null,
            bestEvalY: bestEval ? y(bestEval.v) : null,
            divergenceDelta,
            isDiverging,
            divergeXStart: bestEval ? x(bestEval.step) : null,
            divergeXEnd: x(xMax),
            // 3-tick y-axis labels (top, mid, bottom).
            yTicks: [yLo, (yLo + yHi) / 2, yHi].map((v) => ({ v, y: y(v) })),
            xTicks: [xMin, Math.round((xMin + xMax) / 2), xMax].map((s) => ({ s, x: x(s) })),
            trainCount: trainPts.length,
            evalCount: evalPts.length,
        };
    }, [metrics]);

    if (!model) {
        return (
            <div className="loss-curve loss-curve--empty" data-testid="loss-curve">
                <p className="loss-curve__hint">
                    Loss curve will render here once the trainer emits its first metrics.
                </p>
            </div>
        );
    }

    return (
        <section className="loss-curve" data-testid="loss-curve">
            <header className="loss-curve__head">
                <h3 className="loss-curve__title">Loss curve</h3>
                <span className="loss-curve__head-meta">
                    {model.trainCount > 0 && (
                        <>train <strong>{model.trainCount}</strong> pts</>
                    )}
                    {model.evalCount > 0 && (
                        <>{model.trainCount > 0 && ' · '}eval <strong>{model.evalCount}</strong> pts</>
                    )}
                    {model.bestEval && (
                        <> · best eval @ step <strong>{model.bestEval.step}</strong>: <strong>{model.bestEval.v.toFixed(3)}</strong></>
                    )}
                </span>
            </header>

            <svg
                className="loss-curve__svg"
                role="img"
                aria-label="Training and evaluation loss curves"
                viewBox={`0 0 ${W} ${H}`}
            >
                {/* Divergence shading — drawn first so the lines paint on top. */}
                {model.isDiverging && model.divergeXStart !== null && (
                    <rect
                        x={model.divergeXStart}
                        y={PAD.top}
                        width={Math.max(0, model.divergeXEnd - model.divergeXStart)}
                        height={H - PAD.top - PAD.bottom}
                        className="loss-curve__diverge-shade"
                        data-testid="loss-curve-divergence"
                    />
                )}

                {/* Axes */}
                <line
                    x1={PAD.left}
                    y1={H - PAD.bottom}
                    x2={W - PAD.right}
                    y2={H - PAD.bottom}
                    className="loss-curve__axis"
                />
                <line
                    x1={PAD.left}
                    y1={PAD.top}
                    x2={PAD.left}
                    y2={H - PAD.bottom}
                    className="loss-curve__axis"
                />

                {/* Y ticks */}
                {model.yTicks.map((t, i) => (
                    <g key={`y-${i}`} className="loss-curve__tick">
                        <line
                            x1={PAD.left - 4}
                            y1={t.y}
                            x2={PAD.left}
                            y2={t.y}
                            className="loss-curve__axis"
                        />
                        <text
                            x={PAD.left - 8}
                            y={t.y + 4}
                            textAnchor="end"
                            className="loss-curve__tick-label"
                        >
                            {t.v.toFixed(2)}
                        </text>
                    </g>
                ))}

                {/* X ticks */}
                {model.xTicks.map((t, i) => (
                    <g key={`x-${i}`} className="loss-curve__tick">
                        <line
                            x1={t.x}
                            y1={H - PAD.bottom}
                            x2={t.x}
                            y2={H - PAD.bottom + 4}
                            className="loss-curve__axis"
                        />
                        <text
                            x={t.x}
                            y={H - PAD.bottom + 16}
                            textAnchor="middle"
                            className="loss-curve__tick-label"
                        >
                            {t.s}
                        </text>
                    </g>
                ))}

                {/* Axis labels */}
                <text
                    x={(PAD.left + W - PAD.right) / 2}
                    y={H - 8}
                    textAnchor="middle"
                    className="loss-curve__axis-label"
                >
                    Step
                </text>
                <text
                    x={14}
                    y={(PAD.top + H - PAD.bottom) / 2}
                    textAnchor="middle"
                    className="loss-curve__axis-label"
                    transform={`rotate(-90 14 ${(PAD.top + H - PAD.bottom) / 2})`}
                >
                    Loss — lower ↓
                </text>

                {/* Train + eval paths. Train is a solid line; eval is
                    dashed so the two series read as distinct without
                    needing different colours in colour-blind palettes. */}
                {model.trainPath && (
                    <path
                        d={model.trainPath}
                        className="loss-curve__path loss-curve__path--train"
                        fill="none"
                        data-testid="loss-curve-train-path"
                    />
                )}
                {model.evalPath && (
                    <path
                        d={model.evalPath}
                        className="loss-curve__path loss-curve__path--eval"
                        fill="none"
                        data-testid="loss-curve-eval-path"
                    />
                )}

                {/* Eval markers — usually sparser than train so each
                    measurement deserves a dot. */}
                {model.evalMarkers.map((m, i) => (
                    <circle
                        key={`eval-marker-${i}`}
                        cx={m.cx}
                        cy={m.cy}
                        r={3}
                        className="loss-curve__eval-marker"
                    >
                        <title>{`step ${m.step} · eval ${m.v.toFixed(3)}`}</title>
                    </circle>
                ))}

                {/* Best-eval vertical marker. */}
                {model.bestEvalX !== null && (
                    <g>
                        <line
                            x1={model.bestEvalX}
                            y1={PAD.top}
                            x2={model.bestEvalX}
                            y2={H - PAD.bottom}
                            className="loss-curve__best-marker"
                            data-testid="loss-curve-best-marker"
                        />
                        <circle
                            cx={model.bestEvalX}
                            cy={model.bestEvalY!}
                            r={5}
                            className="loss-curve__best-dot"
                        >
                            <title>{`Best eval @ step ${model.bestEval!.step}: ${model.bestEval!.v.toFixed(3)}`}</title>
                        </circle>
                    </g>
                )}
            </svg>

            <div className="loss-curve__legend">
                <span className="loss-curve__legend-item loss-curve__legend-item--train">train</span>
                <span className="loss-curve__legend-item loss-curve__legend-item--eval">eval</span>
                {model.bestEval && (
                    <span className="loss-curve__legend-item loss-curve__legend-item--best">best-eval</span>
                )}
                {model.isDiverging && (
                    <span className="loss-curve__legend-item loss-curve__legend-item--diverge">overfitting region</span>
                )}
            </div>

            {model.isDiverging && model.bestEval && (
                <p
                    className="loss-curve__diverge-note"
                    data-testid="loss-curve-divergence-note"
                >
                    Eval climbed <strong>+{model.divergenceDelta.toFixed(3)}</strong> from its minimum
                    at step <strong>{model.bestEval.step}</strong> — further training is making the
                    model worse on held-out data. The promote-the-winner pick is
                    step {model.bestEval.step}, not the final checkpoint.
                </p>
            )}
        </section>
    );
}
