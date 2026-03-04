import { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import api from '../../api/client';
import './ExperimentCompare.css';

interface ExperimentCompareProps {
    projectId: number;
    experimentIds: number[];
    onClose: () => void;
}

const COLORS = ['#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#3b82f6'];

export default function ExperimentCompare({ projectId, experimentIds, onClose }: ExperimentCompareProps) {
    const [data, setData] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        let isMounted = true;
        const fetchCompare = async () => {
            try {
                const res = await api.get(`/projects/${projectId}/training/compare?experiment_ids=${experimentIds.join(',')}`);
                if (isMounted) {
                    setData(res.data.experiments);
                    setIsLoading(false);
                }
            } catch (err) {
                console.error("Failed to fetch comparison", err);
                if (isMounted) setIsLoading(false);
            }
        };
        fetchCompare();
        return () => { isMounted = false; };
    }, [projectId, experimentIds]);

    const chartData = useMemo(() => {
        if (!data.length) return [];
        // Align histories by step
        const stepMap = new Map<number, any>();

        data.forEach((exp) => {
            const expKey = `exp_${exp.id}`;
            exp.history?.forEach((point: any) => {
                const step = point.step;
                if (!stepMap.has(step)) {
                    stepMap.set(step, { step });
                }
                const entry = stepMap.get(step);
                entry[`${expKey}_loss`] = point.eval_loss !== null ? point.eval_loss : point.train_loss; // Prefer eval_loss if available
            });
        });

        // Convert back to sorted array
        return Array.from(stepMap.values()).sort((a, b) => a.step - b.step);
    }, [data]);

    if (isLoading) {
        return (
            <div className="card animate-fade-in" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 400 }}>
                <div style={{ color: 'var(--text-tertiary)' }}>Loading comparison data...</div>
            </div>
        );
    }

    if (!data.length) {
        return (
            <div className="card animate-fade-in">
                <div style={{ color: 'var(--color-error)' }}>Failed to load valid comparison data.</div>
                <button className="btn btn-secondary" style={{ marginTop: 16 }} onClick={onClose}>Close Comparison</button>
            </div>
        );
    }

    const formatDuration = (seconds?: number) => {
        if (!seconds) return '—';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        return h > 0 ? `${h}h ${m}m` : `${m}m`;
    };

    return (
        <div className="card animate-fade-in experiment-compare-container">
            <div className="compare-header">
                <h3 className="compare-title">📊 Compare {data.length} Experiments</h3>
                <button className="btn btn-secondary btn-sm" onClick={onClose}>← Back to List</button>
            </div>

            <div className="compare-chart-container">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 30, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="step" stroke="#888" tick={{ fontSize: 12 }} />
                        <YAxis stroke="#888" tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1a1b23', border: '1px solid #333', borderRadius: 8 }}
                            labelStyle={{ color: '#888', marginBottom: 4 }}
                        />
                        <Legend />
                        {data.map((exp, idx) => (
                            <Line
                                key={exp.id}
                                type="monotone"
                                dataKey={`exp_${exp.id}_loss`}
                                name={`${exp.name}`}
                                stroke={COLORS[idx % COLORS.length]}
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 4 }}
                                isAnimationActive={false}
                                connectNulls
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="compare-table-wrapper">
                <table className="compare-table">
                    <thead>
                        <tr>
                            <th>Attribute</th>
                            {data.map((exp, idx) => (
                                <th key={exp.id}>
                                    <span className="compare-legend-color" style={{ background: COLORS[idx % COLORS.length] }}></span>
                                    {exp.name}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <th>Base Model</th>
                            {data.map(exp => <td key={exp.id}>{exp.base_model}</td>)}
                        </tr>
                        <tr>
                            <th>Training Mode</th>
                            {data.map(exp => <td key={exp.id} style={{ textTransform: 'uppercase' }}>{exp.training_mode}</td>)}
                        </tr>
                        <tr>
                            <th>Learning Rate</th>
                            {data.map(exp => <td key={exp.id}>{exp.config?.learning_rate || '—'}</td>)}
                        </tr>
                        <tr>
                            <th>Epochs / Batch</th>
                            {data.map(exp => <td key={exp.id}>{exp.config?.num_epochs || '—'}  / {exp.config?.batch_size || '—'}</td>)}
                        </tr>
                        <tr>
                            <th>LoRA Target</th>
                            {data.map(exp => <td key={exp.id}>
                                {exp.config?.use_lora ? `r=${exp.config?.lora_r} | ${exp.config?.target_modules?.join(', ')}` : 'Full Fine-tune'}
                            </td>)}
                        </tr>
                        <tr className="compare-metric-row">
                            <th>Final Train Loss</th>
                            {data.map(exp => <td key={exp.id} style={{ fontWeight: 600, color: 'var(--color-warning)' }}>
                                {exp.final_train_loss?.toFixed(4) || '—'}
                            </td>)}
                        </tr>
                        <tr className="compare-metric-row">
                            <th>Final Eval Loss</th>
                            {data.map(exp => <td key={exp.id} style={{ fontWeight: 600, color: 'var(--color-success)' }}>
                                {exp.final_eval_loss?.toFixed(4) || '—'}
                            </td>)}
                        </tr>
                        <tr>
                            <th>Training Time</th>
                            {data.map(exp => <td key={exp.id}>{formatDuration(exp.duration_seconds)}</td>)}
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
}
