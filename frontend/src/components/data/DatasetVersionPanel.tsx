import { useState, useEffect } from 'react';
import { GitBranch, FileText, Clock, ArrowRight, CheckCircle } from 'lucide-react';
import api from '../../api/client';

interface DatasetVersion {
    id: number;
    name: string;
    dataset_type: string;
    record_count: number;
    is_locked: boolean;
    created_at: string;
    file_path: string | null;
}

interface DatasetVersionPanelProps {
    projectId: number;
}

export default function DatasetVersionPanel({ projectId }: DatasetVersionPanelProps) {
    const [datasets, setDatasets] = useState<DatasetVersion[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedId, setSelectedId] = useState<number | null>(null);

    useEffect(() => {
        const fetchDatasets = async () => {
            try {
                const res = await api.get(`/projects/${projectId}/datasets`);
                setDatasets(res.data.datasets || []);
            } catch (err) {
                console.error('Failed to fetch datasets:', err);
            } finally {
                setIsLoading(false);
            }
        };
        fetchDatasets();
    }, [projectId]);

    const formatDate = (iso: string) => {
        const d = new Date(iso);
        return d.toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
        });
    };

    const typeColors: Record<string, string> = {
        raw: '#3b82f6',
        cleaned: '#8b5cf6',
        gold_dev: '#f59e0b',
        gold_test: '#ef4444',
        synthetic: '#10b981',
        train: '#ec4899',
        val: '#6366f1',
        test: '#14b8a6',
    };

    if (isLoading) {
        return (
            <div className="card animate-fade-in" style={{ padding: 'var(--space-xl)', textAlign: 'center', color: 'var(--text-tertiary)' }}>
                Loading dataset versions...
            </div>
        );
    }

    return (
        <div className="card animate-fade-in">
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-lg)' }}>
                <GitBranch size={20} color="var(--color-primary)" />
                <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 600, margin: 0 }}>
                    Dataset Versions
                </h3>
                <span className="badge" style={{ fontSize: 'var(--font-size-xs)' }}>{datasets.length} datasets</span>
            </div>

            {datasets.length === 0 ? (
                <div style={{ textAlign: 'center', padding: 'var(--space-2xl)', color: 'var(--text-tertiary)' }}>
                    No datasets yet. Start by ingesting data.
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
                    {/* Timeline view */}
                    {datasets.map((ds, idx) => (
                        <div
                            key={ds.id}
                            onClick={() => setSelectedId(selectedId === ds.id ? null : ds.id)}
                            style={{
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: 'var(--space-md)',
                                padding: 'var(--space-md)',
                                borderRadius: 'var(--radius-md)',
                                background: selectedId === ds.id ? 'rgba(139, 92, 246, 0.08)' : 'transparent',
                                border: `1px solid ${selectedId === ds.id ? 'var(--color-primary)' : 'var(--border-primary)'}`,
                                cursor: 'pointer',
                                transition: 'all 0.15s ease',
                            }}
                        >
                            {/* Timeline dot + line */}
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 24, paddingTop: 4 }}>
                                <div style={{
                                    width: 12,
                                    height: 12,
                                    borderRadius: '50%',
                                    background: typeColors[ds.dataset_type] || '#666',
                                    boxShadow: `0 0 8px ${typeColors[ds.dataset_type] || '#666'}40`,
                                }} />
                                {idx < datasets.length - 1 && (
                                    <div style={{ width: 2, flex: 1, minHeight: 20, background: 'var(--border-primary)', marginTop: 4 }} />
                                )}
                            </div>

                            {/* Content */}
                            <div style={{ flex: 1 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-xs)' }}>
                                    <span style={{ fontWeight: 600, fontSize: 'var(--font-size-sm)' }}>{ds.name}</span>
                                    <span className="badge" style={{
                                        fontSize: 'var(--font-size-xs)',
                                        background: `${typeColors[ds.dataset_type] || '#666'}20`,
                                        color: typeColors[ds.dataset_type] || '#666',
                                        border: `1px solid ${typeColors[ds.dataset_type] || '#666'}40`,
                                    }}>
                                        {ds.dataset_type}
                                    </span>
                                    {ds.is_locked && (
                                        <span title="Locked (immutable)" style={{ display: 'inline-flex' }}>
                                            <CheckCircle size={14} color="var(--color-success)" />
                                        </span>
                                    )}
                                </div>
                                <div style={{ display: 'flex', gap: 'var(--space-lg)', color: 'var(--text-tertiary)', fontSize: 'var(--font-size-xs)' }}>
                                    <span><FileText size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />{ds.record_count.toLocaleString()} records</span>
                                    <span><Clock size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />{formatDate(ds.created_at)}</span>
                                </div>

                                {/* Expanded details */}
                                {selectedId === ds.id && (
                                    <div style={{
                                        marginTop: 'var(--space-sm)',
                                        padding: 'var(--space-sm)',
                                        background: 'rgba(0,0,0,0.2)',
                                        borderRadius: 'var(--radius-sm)',
                                        fontSize: 'var(--font-size-xs)',
                                        fontFamily: 'var(--font-mono)',
                                        color: 'var(--text-secondary)',
                                    }}>
                                        <div>ID: {ds.id}</div>
                                        <div>Type: {ds.dataset_type}</div>
                                        <div>Records: {ds.record_count}</div>
                                        <div>Locked: {ds.is_locked ? 'Yes ✓' : 'No'}</div>
                                        {ds.file_path && <div>Path: {ds.file_path}</div>}
                                    </div>
                                )}
                            </div>

                            {/* Arrow for next */}
                            {idx < datasets.length - 1 && (
                                <ArrowRight size={14} color="var(--text-tertiary)" style={{ marginTop: 6 }} />
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
