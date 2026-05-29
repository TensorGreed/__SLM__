/**
 * FrontierComparisonPanel — Track 1, Epic D.
 *
 * Honest in-product "is my SLM good enough vs gpt-4o-mini?" report on the Eval
 * tab: a headline ("X% as good as <frontier> at Y% cost, Z× latency"), a
 * per-metric quality table (SLM vs frontier, ratio), and cost/latency blocks
 * that label provenance (published frontier reference vs the project's
 * benchmark-sweep estimate) and degrade gracefully — never fabricating numbers.
 */

import { useCallback, useEffect, useState } from 'react';
import {
    fetchFrontierComparison,
    type FrontierComparison,
    type Provenance,
} from '../../api/frontierComparison';
import './FrontierComparisonPanel.css';

interface FrontierComparisonPanelProps {
    projectId: number;
    experimentId: number;
    refreshToken?: number;
}

function provenanceLabel(p: Provenance): string {
    return p === 'measured' ? 'measured' : p === 'estimated' ? 'estimated' : 'unavailable';
}

function fmtUsd(v: number | null): string {
    return v == null ? '—' : `$${v.toFixed(v < 1 ? 4 : 2)}`;
}

function fmtMs(v: number | null): string {
    return v == null ? '—' : `${v.toFixed(0)} ms`;
}

export default function FrontierComparisonPanel({
    projectId,
    experimentId,
    refreshToken,
}: FrontierComparisonPanelProps) {
    const [data, setData] = useState<FrontierComparison | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const load = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            setData(await fetchFrontierComparison(projectId, experimentId));
        } catch (err) {
            const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setError(typeof detail === 'string' && detail ? detail : 'Failed to load frontier comparison.');
            setData(null);
        } finally {
            setLoading(false);
        }
    }, [projectId, experimentId]);

    useEffect(() => {
        void load();
    }, [load, refreshToken]);

    if (error) return null; // 4xx/5xx → self-hide rather than nag the Eval tab

    const frontierName = data?.frontier_model.display_name || 'a frontier model';

    return (
        <div className="card frontier-cmp" data-testid="frontier-comparison">
            <div className="frontier-cmp__head">
                <h4 className="frontier-cmp__title">SLM vs {frontierName}</h4>
                <button type="button" className="btn btn-secondary btn-sm" onClick={() => void load()} disabled={loading}>
                    {loading ? 'Refreshing…' : 'Refresh'}
                </button>
            </div>

            {data && (
                <>
                    <div className="frontier-cmp__headline" data-testid="frontier-headline">
                        {data.headline}
                    </div>
                    <div className="frontier-cmp__ref">
                        Frontier figures: {data.frontier_model.source} (published reference, as of {data.frontier_model.as_of}).
                    </div>

                    {/* Quality */}
                    <section className="frontier-cmp__section">
                        <div className="frontier-cmp__section-title">Quality</div>
                        {data.quality.status === 'ok' ? (
                            <table className="frontier-cmp__table">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Your model</th>
                                        <th>{frontierName}</th>
                                        <th>% as good</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.quality.metric_comparisons.map((row) => (
                                        <tr key={row.metric_id} className={row.is_headline ? 'is-headline' : ''}>
                                            <td>{row.metric_id}</td>
                                            <td>{row.slm_value.toFixed(3)}</td>
                                            <td>{row.frontier_value.toFixed(3)}</td>
                                            <td>
                                                <span className={`frontier-cmp__pct frontier-cmp__pct--${row.direction}`}>
                                                    {row.quality_pct == null
                                                        ? 'exceeds'
                                                        : `${Math.round(row.quality_pct * 100)}%`}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        ) : (
                            <div className="frontier-cmp__fallback" data-testid="frontier-quality-fallback">
                                {data.quality.message}
                            </div>
                        )}
                    </section>

                    {/* Cost + latency */}
                    <div className="frontier-cmp__grid">
                        <section className="frontier-cmp__section">
                            <div className="frontier-cmp__section-title">
                                Cost ($/1M tokens)
                                <span className={`frontier-cmp__badge frontier-cmp__badge--${data.cost.provenance}`}>
                                    {provenanceLabel(data.cost.provenance)}
                                </span>
                            </div>
                            <ul className="frontier-cmp__kv">
                                <li><span>Your model</span><strong>{fmtUsd(data.cost.slm_usd_per_1m_tokens)}</strong></li>
                                <li><span>{frontierName}</span><strong>{fmtUsd(data.cost.frontier_usd_per_1m_tokens)}</strong></li>
                                {data.cost.cost_pct != null && (
                                    <li><span>Relative</span><strong>{data.cost.cost_pct}% of cost</strong></li>
                                )}
                            </ul>
                            {(data.cost.source || data.cost.message) && (
                                <div className="frontier-cmp__note">{data.cost.source || data.cost.message}</div>
                            )}
                        </section>

                        <section className="frontier-cmp__section">
                            <div className="frontier-cmp__section-title">
                                Latency
                                <span className={`frontier-cmp__badge frontier-cmp__badge--${data.latency.provenance}`}>
                                    {provenanceLabel(data.latency.provenance)}
                                </span>
                            </div>
                            <ul className="frontier-cmp__kv">
                                <li><span>Your model</span><strong>{fmtMs(data.latency.slm_latency_ms)}</strong></li>
                                <li><span>{frontierName}</span><strong>{fmtMs(data.latency.frontier_latency_ms)}</strong></li>
                                {data.latency.latency_ratio != null && (
                                    <li><span>Relative</span><strong>{data.latency.latency_ratio}× latency</strong></li>
                                )}
                            </ul>
                            {(data.latency.source || data.latency.message) && (
                                <div className="frontier-cmp__note">{data.latency.source || data.latency.message}</div>
                            )}
                        </section>
                    </div>
                </>
            )}
        </div>
    );
}
