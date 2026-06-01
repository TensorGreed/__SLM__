/**
 * Inline live-loss sparkline for the NotificationBell training row.
 *
 * During a 10–15 min GB10 training run, the bell previously showed
 * only progress %. The user had no signal whether loss was trending
 * down healthily, flat-lining, or worse — they had to open the
 * dashboard to find out. This component closes that gap with a
 * tiny SVG sparkline + a colour-coded trend tint based on the
 * direction of the last vs first point in the window.
 *
 * Renders at ~80×16 px so it sits inline next to the job's progress
 * %. Pure presentational — takes the metrics_recent array the bell
 * already gets from /api/jobs/active and draws it. No data fetching
 * inside this component.
 */

import type { TrainingMetricsRecentPoint } from '../../api/jobs';


export interface TrainingLossSparklineProps {
    points: TrainingMetricsRecentPoint[];
    /** SVG width in pixels. */
    width?: number;
    /** SVG height in pixels. */
    height?: number;
    /** Padding inside the SVG so the polyline doesn't kiss the edges. */
    padding?: number;
}


// What counts as "trending down" / "flat" / "trending up". We
// compare the mean of the first 25% of points vs the last 25% so a
// single outlier at either end doesn't flip the tint. ~5% relative
// change is the dead-zone where we render flat / amber — small
// enough to ignore tokenizer noise, big enough that genuine
// flat-lining shows up before the run finishes.
export const TREND_FLAT_REL_THRESHOLD = 0.05;


export type LossTrend = 'down' | 'flat' | 'up';


/**
 * Pure trend classifier shared by ``TrainingLossSparkline`` (for the
 * tint) and the bell's kill-switch detector (for "diverging for N
 * consecutive polls" logic). Extracted so a future change to the
 * head/tail window or threshold can't drift the two consumers apart.
 */
export function computeLossTrend(points: TrainingMetricsRecentPoint[]): LossTrend {
    const losses = points
        .map((p) => p.train_loss)
        .filter((v): v is number => typeof v === 'number');
    if (losses.length < 4) return 'flat';
    const headEnd = Math.max(1, Math.floor(losses.length * 0.25));
    const tailStart = Math.min(
        losses.length - 1,
        Math.floor(losses.length * 0.75),
    );
    const head = losses.slice(0, headEnd);
    const tail = losses.slice(tailStart);
    const headMean = head.reduce((s, v) => s + v, 0) / head.length;
    const tailMean = tail.reduce((s, v) => s + v, 0) / tail.length;
    if (headMean === 0) return 'flat';
    const relChange = (tailMean - headMean) / Math.abs(headMean);
    if (relChange < -TREND_FLAT_REL_THRESHOLD) return 'down';
    if (relChange > TREND_FLAT_REL_THRESHOLD) return 'up';
    return 'flat';
}


function _trendDirection(losses: number[]): LossTrend {
    // Adapter that lets the existing render path keep its
    // ``losses: number[]`` shape while delegating to the exported
    // pure helper above (which takes the public Job shape).
    return computeLossTrend(losses.map((v) => ({ step: 0, train_loss: v })));
}


function _trendColour(direction: 'down' | 'flat' | 'up'): string {
    if (direction === 'down') return 'var(--color-success)';
    if (direction === 'up') return 'var(--color-error)';
    return 'var(--color-warning)';
}


function _trendLabel(direction: 'down' | 'flat' | 'up'): string {
    if (direction === 'down') return '↘';
    if (direction === 'up') return '↗';
    return '→';
}


export default function TrainingLossSparkline({
    points,
    width = 80,
    height = 16,
    padding = 1,
}: TrainingLossSparklineProps) {
    const losses = points
        .map((p) => p.train_loss)
        .filter((v): v is number => typeof v === 'number');

    // No data → render a flat-line placeholder so the row layout
    // doesn't shift when the first checkpoint lands.
    if (losses.length === 0) {
        return (
            <svg
                width={width}
                height={height}
                role="img"
                aria-label="Training loss sparkline (no data yet)"
                data-testid="training-loss-sparkline"
                data-trend="empty"
                style={{ verticalAlign: 'middle' }}
            >
                <line
                    x1={padding}
                    y1={height / 2}
                    x2={width - padding}
                    y2={height / 2}
                    stroke="var(--text-tertiary)"
                    strokeWidth={1}
                    strokeDasharray="2 2"
                />
            </svg>
        );
    }

    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const span = maxLoss - minLoss;
    const usableHeight = height - padding * 2;
    const usableWidth = width - padding * 2;

    // Build the polyline points. When all losses are equal the y
    // collapses to the midline rather than dividing by zero.
    const polylinePoints = losses
        .map((loss, idx) => {
            const x =
                losses.length === 1
                    ? width / 2
                    : padding + (idx / (losses.length - 1)) * usableWidth;
            const y =
                span === 0
                    ? height / 2
                    : padding + (1 - (loss - minLoss) / span) * usableHeight;
            return `${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(' ');

    const trend = _trendDirection(losses);
    const stroke = _trendColour(trend);
    const lastLoss = losses[losses.length - 1];

    return (
        <svg
            width={width}
            height={height}
            role="img"
            aria-label={
                `Training loss sparkline · ${losses.length} points · `
                + `last ${lastLoss.toFixed(4)} · `
                + `trend ${_trendLabel(trend)}`
            }
            data-testid="training-loss-sparkline"
            data-trend={trend}
            style={{ verticalAlign: 'middle' }}
        >
            <polyline
                fill="none"
                stroke={stroke}
                strokeWidth={1.25}
                strokeLinecap="round"
                strokeLinejoin="round"
                points={polylinePoints}
            />
            {/* Last-point dot — emphasises the latest value so the
             *  user's eye lands on the most recent step. */}
            <circle
                cx={
                    losses.length === 1
                        ? width / 2
                        : width - padding
                }
                cy={
                    span === 0
                        ? height / 2
                        : padding + (1 - (lastLoss - minLoss) / span) * usableHeight
                }
                r={1.5}
                fill={stroke}
                data-testid="training-loss-sparkline-last-dot"
            />
        </svg>
    );
}
