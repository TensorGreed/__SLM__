/**
 * TelemetryPanel — live request volume / latency / error / token-throughput
 * for a deployment version (priority.md P30, P26).
 *
 * Layout:
 * - Window picker (1m / 5m / 1h / 24h) drives `window_seconds` query param.
 * - KPI grid: request volume, error rate, p50/p95/p99 latency, token
 *   throughput.
 * - Inline SVG bar chart of the latency percentiles so the relative
 *   distribution lands at-a-glance — no chart library dependency.
 *
 * The panel polls the aggregate every 15s while the tab is open. The
 * chart re-keys on each fetch so transitions are obvious without
 * hand-rolling animation.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import api from '../../api/client';
import type { TelemetryAggregate } from '../../types/deployment';

interface Props {
    deploymentVersionId: number;
    /**
     * Bump from the parent to force an immediate re-fetch (in addition
     * to the 15s poll). Useful after a promote/reject/rollback so the
     * panel reflects the new live state without waiting a full tick.
     */
    refreshKey?: number;
}

interface ApiErrorShape {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
}

const WINDOWS: Array<{ key: string; label: string; seconds: number }> = [
    { key: '1m', label: '1 minute', seconds: 60 },
    { key: '5m', label: '5 minutes', seconds: 300 },
    { key: '1h', label: '1 hour', seconds: 3600 },
    { key: '24h', label: '24 hours', seconds: 24 * 3600 },
];

const POLL_INTERVAL_MS = 15_000;

function extractErrorMessage(err: unknown, fallback = 'Request failed.'): string {
    const e = err as ApiErrorShape;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    return e?.message || fallback;
}

function formatNumber(value: number, digits = 1): string {
    if (!Number.isFinite(value)) return '—';
    return value.toFixed(digits);
}

function formatPercent(rate: number): string {
    if (!Number.isFinite(rate)) return '—';
    return `${(rate * 100).toFixed(2)}%`;
}

interface PercentileBars {
    width: number;
    height: number;
    p50: number;
    p95: number;
    p99: number;
    max: number;
}

function PercentileBars({ width, height, p50, p95, p99, max }: PercentileBars) {
    const safeMax = Math.max(max, p99, p95, p50, 1);
    const barWidth = (width - 32) / 3 - 8;
    const barTop = 16;
    const usableHeight = height - barTop - 24;
    const heightFor = (value: number) =>
        Math.max(2, Math.round((value / safeMax) * usableHeight));
    const bars = [
        { label: 'p50', value: p50, x: 16, fill: 'var(--color-success, #2da44e)' },
        { label: 'p95', value: p95, x: 16 + barWidth + 12, fill: 'var(--color-warning, #d4a72c)' },
        { label: 'p99', value: p99, x: 16 + (barWidth + 12) * 2, fill: 'var(--color-danger, #cf222e)' },
    ];
    return (
        <svg
            role="img"
            aria-label="Latency percentile bars"
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
        >
            {bars.map((bar) => {
                const h = heightFor(bar.value);
                const y = barTop + (usableHeight - h);
                return (
                    <g key={bar.label}>
                        <rect
                            x={bar.x}
                            y={y}
                            width={barWidth}
                            height={h}
                            fill={bar.fill}
                            rx={4}
                        />
                        <text
                            x={bar.x + barWidth / 2}
                            y={y - 4}
                            fontSize={11}
                            textAnchor="middle"
                            fill="var(--text-secondary, #57606a)"
                        >
                            {bar.value.toFixed(0)}ms
                        </text>
                        <text
                            x={bar.x + barWidth / 2}
                            y={height - 8}
                            fontSize={11}
                            textAnchor="middle"
                            fill="var(--text-tertiary, #6e7781)"
                        >
                            {bar.label}
                        </text>
                    </g>
                );
            })}
        </svg>
    );
}

export default function TelemetryPanel({
    deploymentVersionId,
    refreshKey = 0,
}: Props) {
    const [windowSeconds, setWindowSeconds] = useState<number>(3600);
    const [aggregate, setAggregate] = useState<TelemetryAggregate | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const fetchAggregate = useCallback(async () => {
        setLoading(true);
        try {
            const response = await api.get<TelemetryAggregate>(
                `/deployments/${deploymentVersionId}/telemetry`,
                { params: { window_seconds: windowSeconds } },
            );
            setAggregate(response.data);
            setError(null);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to load telemetry.'));
        } finally {
            setLoading(false);
        }
    }, [deploymentVersionId, windowSeconds]);

    useEffect(() => {
        void fetchAggregate();
        if (timerRef.current) {
            clearInterval(timerRef.current);
        }
        timerRef.current = setInterval(() => {
            void fetchAggregate();
        }, POLL_INTERVAL_MS);
        return () => {
            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
        };
        // refreshKey is a deliberate dependency so the parent can force
        // an immediate re-fetch after a promote/reject/rollback.
    }, [fetchAggregate, refreshKey]);

    const kpis = useMemo(() => {
        if (!aggregate) return null;
        return {
            requestsTotal: aggregate.request_volume.total,
            requestsPerMinute: aggregate.request_volume.per_minute,
            errorRate: aggregate.errors.rate,
            errorCount: aggregate.errors.count,
            p50: aggregate.latency_ms.p50,
            p95: aggregate.latency_ms.p95,
            p99: aggregate.latency_ms.p99,
            tokensInPerSec: aggregate.tokens.input_per_second,
            tokensOutPerSec: aggregate.tokens.output_per_second,
            sampleCount: aggregate.sample_count,
            max: aggregate.latency_ms.max,
        };
    }, [aggregate]);

    return (
        <div className="card deployment-telemetry-card">
            <div className="deployment-section-header">
                <div>
                    <h3 style={{ marginTop: 0 }}>Live telemetry</h3>
                    <div className="dim">
                        Window: rolling {WINDOWS.find((w) => w.seconds === windowSeconds)?.label || `${windowSeconds}s`} ·
                        polled every 15s.
                    </div>
                </div>
                <div
                    className="deployment-window-picker"
                    role="group"
                    aria-label="Telemetry window"
                >
                    {WINDOWS.map((option) => (
                        <button
                            key={option.key}
                            type="button"
                            className={`btn btn-sm ${
                                windowSeconds === option.seconds
                                    ? 'btn-primary'
                                    : 'btn-secondary'
                            }`}
                            onClick={() => setWindowSeconds(option.seconds)}
                        >
                            {option.key}
                        </button>
                    ))}
                </div>
            </div>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            {!kpis ? (
                <div className="dim">{loading ? 'Loading telemetry…' : 'No telemetry yet.'}</div>
            ) : kpis.sampleCount === 0 ? (
                <div className="deployment-empty">
                    <div>No samples in this window.</div>
                    <div className="deployment-empty-detail">
                        The deployment plane is push-only — your inference
                        client (or a provider-side scrape sidecar) needs to
                        POST samples to{' '}
                        <code>
                            /api/deployments/{deploymentVersionId}/telemetry/ingest
                        </code>
                        . Until then this panel and the
                        <code> telemetry_health</code> score component will
                        report no signal.
                    </div>
                </div>
            ) : (
                <>
                    <div className="deployment-kpi-grid">
                        <div className="deployment-kpi">
                            <div className="kpi-label">Requests</div>
                            <div className="kpi-value">{kpis.requestsTotal}</div>
                            <div className="kpi-sub">{formatNumber(kpis.requestsPerMinute)} /min</div>
                        </div>
                        <div className="deployment-kpi">
                            <div className="kpi-label">Error rate</div>
                            <div className="kpi-value">{formatPercent(kpis.errorRate)}</div>
                            <div className="kpi-sub">{kpis.errorCount} error(s)</div>
                        </div>
                        <div className="deployment-kpi">
                            <div className="kpi-label">Latency (ms)</div>
                            <div className="kpi-value">{formatNumber(kpis.p95)}</div>
                            <div className="kpi-sub">
                                p50 {formatNumber(kpis.p50)} · p99 {formatNumber(kpis.p99)}
                            </div>
                        </div>
                        <div className="deployment-kpi">
                            <div className="kpi-label">Throughput</div>
                            <div className="kpi-value">
                                {formatNumber(kpis.tokensInPerSec + kpis.tokensOutPerSec)} tok/s
                            </div>
                            <div className="kpi-sub">
                                in {formatNumber(kpis.tokensInPerSec)} · out {formatNumber(kpis.tokensOutPerSec)}
                            </div>
                        </div>
                    </div>
                    <div className="deployment-chart-wrap">
                        <PercentileBars
                            width={360}
                            height={140}
                            p50={kpis.p50}
                            p95={kpis.p95}
                            p99={kpis.p99}
                            max={kpis.max}
                        />
                    </div>
                </>
            )}
        </div>
    );
}
