/**
 * WarmStartDeltaChart — Track 1, Epic B/C.
 *
 * For a warm-started run, plots the live training-loss curve alongside a
 * "delta vs the pre-tuned baseline" series (loss(step) − the warm-start's
 * initial loss on the user's data). The baseline is the run's first recorded
 * train loss — i.e. what the warm-start checkpoint already achieved before any
 * of the user's gradient steps — so the delta series makes the *marginal*
 * contribution of the user's rows legible on top of the pre-tuned base.
 *
 * Renders only when the run actually warm-started (`_warm_start.source ===
 * 'checkpoint'`); a cold start has no pre-tuned baseline to subtract.
 */

import { useMemo } from 'react';

interface MetricPoint {
    step?: number | null;
    train_loss?: number | null;
}

interface WarmStartInfo {
    source?: string;
    checkpoint_name?: string | null;
    reason?: string;
}

interface WarmStartDeltaChartProps {
    metrics: MetricPoint[];
    warmStart?: WarmStartInfo | null;
}

const W = 520;
const H = 240;
const PAD = { top: 16, right: 18, bottom: 36, left: 52 };

export default function WarmStartDeltaChart({ metrics, warmStart }: WarmStartDeltaChartProps) {
    const points = useMemo(() => {
        return metrics
            .map((m) => ({ step: Number(m.step), loss: Number(m.train_loss) }))
            .filter((p) => Number.isFinite(p.step) && Number.isFinite(p.loss))
            .sort((a, b) => a.step - b.step);
    }, [metrics]);

    const model = useMemo(() => {
        if (points.length < 2) return null;
        const baseline = points[0].loss;
        const series = points.map((p) => ({ step: p.step, loss: p.loss, delta: p.loss - baseline }));
        const losses = series.map((s) => s.loss);
        const deltas = series.map((s) => s.delta);
        // Shared y-scale spanning both series + the delta's zero reference.
        const yMin = Math.min(...losses, ...deltas, 0);
        const yMax = Math.max(...losses, ...deltas, 0);
        const ySpan = yMax - yMin || 1;
        const xMin = series[0].step;
        const xMax = series[series.length - 1].step;
        const xSpan = xMax - xMin || 1;
        const x = (step: number) => PAD.left + ((step - xMin) / xSpan) * (W - PAD.left - PAD.right);
        const y = (v: number) => PAD.top + (1 - (v - yMin) / ySpan) * (H - PAD.top - PAD.bottom);
        const line = (key: 'loss' | 'delta') =>
            series.map((s, i) => `${i === 0 ? 'M' : 'L'}${x(s.step).toFixed(1)},${y(s[key]).toFixed(1)}`).join(' ');
        return {
            baseline,
            finalDelta: series[series.length - 1].delta,
            lossPath: line('loss'),
            deltaPath: line('delta'),
            zeroY: y(0),
        };
    }, [points]);

    if (!warmStart || warmStart.source !== 'checkpoint') return null;

    if (!model) {
        return (
            <div className="warmstart-delta warmstart-delta--empty" data-testid="warmstart-delta">
                Warm-started from <strong>{warmStart.checkpoint_name || 'a checkpoint'}</strong> — the
                delta-from-baseline curve appears once training metrics stream in.
            </div>
        );
    }

    const improved = model.finalDelta < 0;
    const magnitude = Math.abs(model.finalDelta).toFixed(3);

    return (
        <div className="warmstart-delta" data-testid="warmstart-delta">
            <div className="warmstart-delta__head">
                <strong>Your rows vs the pre-tuned baseline</strong>
                <span className="warmstart-delta__hint">
                    Warm-started from <code>{warmStart.checkpoint_name || 'checkpoint'}</code>; baseline ={' '}
                    {model.baseline.toFixed(3)} (loss before your rows)
                </span>
            </div>

            <svg className="warmstart-delta__chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training loss and delta-from-baseline over steps">
                <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} className="warmstart-delta__axis" />
                <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} className="warmstart-delta__axis" />
                {/* zero reference for the delta series */}
                <line x1={PAD.left} y1={model.zeroY} x2={W - PAD.right} y2={model.zeroY} className="warmstart-delta__zero" />
                <path d={model.lossPath} className="warmstart-delta__line warmstart-delta__line--loss" fill="none" data-testid="warmstart-delta-loss" />
                <path d={model.deltaPath} className="warmstart-delta__line warmstart-delta__line--delta" fill="none" data-testid="warmstart-delta-delta" />
                <text x={(PAD.left + W - PAD.right) / 2} y={H - 6} className="warmstart-delta__axis-label" textAnchor="middle">
                    step →
                </text>
            </svg>

            <div className="warmstart-delta__legend">
                <span className="warmstart-delta__legend-item"><span className="warmstart-delta__swatch is-loss" /> training loss</span>
                <span className="warmstart-delta__legend-item"><span className="warmstart-delta__swatch is-delta" /> Δ vs baseline (loss − start)</span>
            </div>

            <div className={`warmstart-delta__summary ${improved ? 'is-improved' : 'is-regressed'}`}>
                {improved
                    ? `Your rows reduced loss by ${magnitude} below the warm-start's starting point.`
                    : `Loss is ${magnitude} above the warm-start's starting point so far.`}
            </div>
        </div>
    );
}
