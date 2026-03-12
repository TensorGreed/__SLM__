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
}

export default function EDADashboard({ projectId }: { projectId: number }) {
    const [stats, setStats] = useState<EDAStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

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
        fetchStats();
    }, [projectId]);

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
            </div>

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
