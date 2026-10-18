/**
 * Panel listing prepared dataset versions with record counts, types, and creation timestamps.
 */

import { useState, useEffect } from 'react';
import { GitBranch, FileText, Clock, ArrowRight, CheckCircle } from 'lucide-react';
import api from '../../api/client';
import EmptyState from '../shared/EmptyState';
import LoadingState from '../shared/LoadingState';
import './DatasetVersionPanel.css';

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

    const KNOWN_TYPES = ['raw', 'cleaned', 'gold_dev', 'gold_test', 'synthetic', 'train', 'val', 'test'];
    const typeClass = (t: string) => `dsv-type--${KNOWN_TYPES.includes(t) ? t : 'unknown'}`;

    if (isLoading) {
        return <LoadingState label="Loading dataset versions" lines={3} card />;
    }

    return (
        <div className="card animate-fade-in">
            <div className="u-row dsv-header">
                <GitBranch size={20} className="dsv-icon" aria-hidden="true" />
                <h3 className="dsv-title">Dataset Versions</h3>
                <span className="badge dsv-count">{datasets.length} datasets</span>
            </div>

            {datasets.length === 0 ? (
                <EmptyState
                    title="No datasets yet"
                    description="Datasets are versioned snapshots produced by ingestion, cleaning and prep. Start by ingesting a file or remote source above; new versions will land here as each stage runs."
                    docsHref="http://localhost:3001/docs/workflows/data-ingestion"
                />
            ) : (
                <div className="dsv-list">
                    {datasets.map((ds, idx) => (
                        <div
                            key={ds.id}
                            onClick={() => setSelectedId(selectedId === ds.id ? null : ds.id)}
                            className={`dsv-row ${typeClass(ds.dataset_type)}${selectedId === ds.id ? ' dsv-row--selected' : ''}`}
                        >
                            <div className="dsv-rail">
                                <div className="dsv-dot" />
                                {idx < datasets.length - 1 && <div className="dsv-line" />}
                            </div>

                            <div className="dsv-body">
                                <div className="dsv-name-row">
                                    <span className="dsv-name">{ds.name}</span>
                                    <span className="badge dsv-badge">{ds.dataset_type}</span>
                                    {ds.is_locked && (
                                        <span title="Locked (immutable)" className="dsv-lock">
                                            <CheckCircle size={14} className="dsv-lock-icon" aria-hidden="true" />
                                        </span>
                                    )}
                                </div>
                                <div className="dsv-facts">
                                    <span><FileText size={12} className="dsv-fact-icon" aria-hidden="true" />{ds.record_count.toLocaleString()} records</span>
                                    <span><Clock size={12} className="dsv-fact-icon" aria-hidden="true" />{formatDate(ds.created_at)}</span>
                                </div>

                                {selectedId === ds.id && (
                                    <div className="dsv-details">
                                        <div>ID: {ds.id}</div>
                                        <div>Type: {ds.dataset_type}</div>
                                        <div>Records: {ds.record_count}</div>
                                        <div>Locked: {ds.is_locked ? 'Yes' : 'No'}</div>
                                        {ds.file_path && <div>Path: {ds.file_path}</div>}
                                    </div>
                                )}
                            </div>

                            {idx < datasets.length - 1 && (
                                <ArrowRight size={14} className="dsv-next" aria-hidden="true" />
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
