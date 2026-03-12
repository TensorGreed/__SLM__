import { useEffect, useState } from 'react';
import api from '../../api/client';
import Skeleton from '../shared/Skeleton';
import './EDADashboard.css';

interface EDAStats {
    total_files: number;
    total_size_bytes: number;
    estimated_total_rows: number;
    sample_size: number;
    schema_keys_present: string[];
    token_distribution: {
        p50: number;
        p90: number;
        p99: number;
        max: number;
    };
    estimated_duplicate_ratio: number;
    toxicity?: {
        average_score: number;
        flagged_ratio: number;
        flagged_count: number;
    };
    topic_clusters?: Array<{
        cluster_id: number;
        size: number;
        share: number;
        label: string;
        sample_previews?: string[];
    }>;
    outlier_candidates?: Array<{
        sample_index: number;
        token_count: number;
        toxicity_score: number;
        reason: string;
        text_preview: string;
    }>;
}

export default function EDADashboard({ projectId }: { projectId: number }) {
    const [stats, setStats] = useState<EDAStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [removingOutliers, setRemovingOutliers] = useState(false);
    const [outlierStatus, setOutlierStatus] = useState('');

    const fetchStats = async () => {
        try {
            setLoading(true);
            const res = await api.get(`/projects/${projectId}/ingestion/eda`);
            setStats(res.data);
        } catch (e: any) {
            setError(e.response?.data?.detail || e.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void fetchStats();
    }, [projectId]);

    const handleRemoveOutliers = async () => {
        setOutlierStatus('');
        setRemovingOutliers(true);
        try {
            const res = await api.post(`/projects/${projectId}/ingestion/eda/remove-outliers`, {});
            setOutlierStatus(
                `Filtered rows: removed ${Number(res.data?.rows_removed || 0)} / ${Number(res.data?.rows_in || 0)}.`,
            );
            await fetchStats();
        } catch (e: any) {
            setOutlierStatus(`Outlier removal failed: ${e.response?.data?.detail || e.message}`);
        } finally {
            setRemovingOutliers(false);
        }
    };

    if (loading) return <Skeleton height={150} />;
    if (error) return <div className="eda-error">Failed to load EDA stats: {error}</div>;
    if (!stats || stats.total_files === 0) return null;

    return (
        <div className="eda-dashboard">
            <div className="eda-header">
                <h3>📊 Data Health Dashboard (Auto-EDA)</h3>
                <button className="btn btn-ghost btn-sm" onClick={fetchStats}>↻ Refresh</button>
            </div>

            <div className="eda-grid">
                <div className="eda-card">
                    <div className="eda-label">Total Files</div>
                    <div className="eda-value">{stats.total_files}</div>
                </div>
                <div className="eda-card">
                    <div className="eda-label">Estimated Rows</div>
                    <div className="eda-value">{stats.estimated_total_rows.toLocaleString()}</div>
                    <div className="eda-sub">from samples</div>
                </div>
                <div className="eda-card">
                    <div className="eda-label">Dup Ratio (Est)</div>
                    <div className={`eda-value ${stats.estimated_duplicate_ratio > 0.2 ? 'warn' : ''}`}>
                        {(stats.estimated_duplicate_ratio * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="eda-card">
                    <div className="eda-label">Token P50</div>
                    <div className="eda-value">{stats.token_distribution.p50}</div>
                    <div className="eda-sub">max {stats.token_distribution.max}</div>
                </div>
                <div className="eda-card">
                    <div className="eda-label">Toxicity Flagged</div>
                    <div className={`eda-value ${(stats.toxicity?.flagged_ratio || 0) > 0.15 ? 'warn' : ''}`}>
                        {((stats.toxicity?.flagged_ratio || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="eda-sub">avg score {(stats.toxicity?.average_score || 0).toFixed(2)}</div>
                </div>
            </div>

            {Array.isArray(stats.topic_clusters) && stats.topic_clusters.length > 0 && (
                <div className="eda-schema">
                    <div className="eda-label">Topical Clusters</div>
                    <div className="eda-keys">
                        {stats.topic_clusters.slice(0, 6).map((cluster) => (
                            <span key={cluster.cluster_id} className="eda-key-badge">
                                {cluster.label} ({Math.round(cluster.share * 100)}%)
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {Array.isArray(stats.outlier_candidates) && stats.outlier_candidates.length > 0 && (
                <div className="eda-outlier-row">
                    <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => void handleRemoveOutliers()}
                        disabled={removingOutliers}
                    >
                        {removingOutliers ? 'Removing...' : `Remove Outliers (${stats.outlier_candidates.length})`}
                    </button>
                    {outlierStatus && <span className="eda-sub">{outlierStatus}</span>}
                </div>
            )}

            {stats.schema_keys_present.length > 0 && (
                <div className="eda-schema">
                    <div className="eda-label">Detected Schema Keys (Top 50)</div>
                    <div className="eda-keys">
                        {stats.schema_keys_present.map(k => (
                            <span key={k} className="eda-key-badge">{k}</span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
