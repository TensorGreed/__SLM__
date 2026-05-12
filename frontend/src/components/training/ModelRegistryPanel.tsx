import { useState, useEffect } from 'react';
import { Download, Star, Clock, HardDrive } from 'lucide-react';
import api from '../../api/client';
import EmptyState from '../shared/EmptyState';

interface Checkpoint {
    id: number;
    experiment_id: number;
    experiment_name: string;
    step: number;
    path: string;
    train_loss: number | null;
    eval_loss: number | null;
    created_at: string;
    is_best: boolean;
}

interface ModelRegistryPanelProps {
    projectId: number;
}

export default function ModelRegistryPanel({ projectId }: ModelRegistryPanelProps) {
    const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [filter, setFilter] = useState<'all' | 'best'>('all');

    useEffect(() => {
        const fetchCheckpoints = async () => {
            try {
                const res = await api.get(`/projects/${projectId}/training/checkpoints`);
                setCheckpoints(res.data.checkpoints || []);
            } catch (err) {
                console.error('Failed to fetch checkpoints:', err);
            } finally {
                setIsLoading(false);
            }
        };
        fetchCheckpoints();
    }, [projectId]);

    const filtered = filter === 'best' ? checkpoints.filter(c => c.is_best) : checkpoints;

    const formatDate = (iso: string) => {
        const d = new Date(iso);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    };

    if (isLoading) {
        return (
            <div className="card animate-fade-in" style={{ padding: 'var(--space-xl)', textAlign: 'center', color: 'var(--text-tertiary)' }}>
                Loading model registry...
            </div>
        );
    }

    return (
        <div className="card animate-fade-in">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-lg)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                    <HardDrive size={20} color="var(--color-primary)" />
                    <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 600, margin: 0 }}>
                        Model Registry
                    </h3>
                    <span className="badge" style={{ fontSize: 'var(--font-size-xs)' }}>{checkpoints.length} checkpoints</span>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                    <button
                        className={`btn btn-sm ${filter === 'all' ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => setFilter('all')}
                    >
                        All
                    </button>
                    <button
                        className={`btn btn-sm ${filter === 'best' ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => setFilter('best')}
                    >
                        <Star size={14} /> Best Only
                    </button>
                </div>
            </div>

            {filtered.length === 0 ? (
                checkpoints.length === 0 ? (
                    <EmptyState
                        title="No checkpoints yet"
                        description="Checkpoints land here as training runs write them. Each one is promotable + downloadable — once you have one, you can export it for a target."
                        docsHref="http://localhost:3001/docs/workflows/training"
                    />
                ) : (
                    <EmptyState
                        title="No best checkpoints found"
                        description="No checkpoint in the current view is marked as a 'best' candidate. Untoggle Best Only to see every captured checkpoint."
                    />
                )
            ) : (
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-primary)' }}>
                                <th style={thStyle}>Experiment</th>
                                <th style={thStyle}>Step</th>
                                <th style={thStyle}>Train Loss</th>
                                <th style={thStyle}>Eval Loss</th>
                                <th style={thStyle}>Created</th>
                                <th style={thStyle}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.map((cp) => (
                                <tr key={cp.id} style={{
                                    borderBottom: '1px solid var(--border-primary)',
                                    background: cp.is_best ? 'rgba(16, 185, 129, 0.05)' : 'transparent',
                                }}>
                                    <td style={tdStyle}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                                            {cp.is_best && <Star size={14} color="var(--color-warning)" fill="var(--color-warning)" />}
                                            <span style={{ fontWeight: cp.is_best ? 600 : 400 }}>{cp.experiment_name}</span>
                                        </div>
                                    </td>
                                    <td style={tdStyle}>
                                        <span className="badge" style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--font-size-xs)' }}>
                                            step-{cp.step}
                                        </span>
                                    </td>
                                    <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)', color: 'var(--color-warning)' }}>
                                        {cp.train_loss?.toFixed(4) || '—'}
                                    </td>
                                    <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)', color: 'var(--color-success)' }}>
                                        {cp.eval_loss?.toFixed(4) || '—'}
                                    </td>
                                    <td style={{ ...tdStyle, color: 'var(--text-tertiary)', fontSize: 'var(--font-size-sm)' }}>
                                        <Clock size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                                        {formatDate(cp.created_at)}
                                    </td>
                                    <td style={tdStyle}>
                                        <button className="btn btn-secondary btn-sm" title="Use for export">
                                            <Download size={14} /> Export
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

const thStyle: React.CSSProperties = {
    textAlign: 'left',
    padding: 'var(--space-sm) var(--space-md)',
    color: 'var(--text-tertiary)',
    fontSize: 'var(--font-size-xs)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    fontWeight: 500,
};

const tdStyle: React.CSSProperties = {
    padding: 'var(--space-sm) var(--space-md)',
    fontSize: 'var(--font-size-sm)',
};
