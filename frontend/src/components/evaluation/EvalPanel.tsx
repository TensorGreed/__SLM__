import { useState } from 'react';
import api from '../../api/client';

interface EvalPanelProps { projectId: number; }

export default function EvalPanel({ projectId }: EvalPanelProps) {
    const [experiments, setExperiments] = useState<any[]>([]);
    const [selectedExp, setSelectedExp] = useState<number | null>(null);
    const [evalResults, setEvalResults] = useState<any[]>([]);
    const [scorecard, setScorecard] = useState<any>(null);
    const [loaded, setLoaded] = useState(false);

    if (!loaded) {
        api.get(`/projects/${projectId}/training/experiments`).then(r => { setExperiments(r.data); setLoaded(true); });
    }

    const loadResults = async (expId: number) => {
        setSelectedExp(expId);
        const res = await api.get(`/projects/${projectId}/evaluation/results/${expId}`);
        setEvalResults(res.data);
        const sc = await api.get(`/projects/${projectId}/evaluation/safety-scorecard/${expId}`);
        setScorecard(sc.data);
    };

    const riskColor = (risk: string) => risk === 'low' ? 'badge-success' : risk === 'medium' ? 'badge-warning' : risk === 'high' ? 'badge-error' : 'badge-info';

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Evaluation</h3>
                {experiments.length === 0 ? (
                    <div className="empty-state"><div className="empty-state-icon">📊</div><div className="empty-state-title">No experiments to evaluate</div></div>
                ) : (
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        {experiments.map(e => (
                            <button key={e.id} className={`btn ${selectedExp === e.id ? 'btn-primary' : 'btn-secondary'}`} onClick={() => loadResults(e.id)}>{e.name}</button>
                        ))}
                    </div>
                )}
            </div>

            {scorecard && (
                <div className="card">
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>🛡️ Safety Scorecard</h3>
                    <div style={{ display: 'flex', gap: 'var(--space-xl)', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                        <div><strong>Overall Risk:</strong> <span className={`badge ${riskColor(scorecard.overall_risk)}`}>{scorecard.overall_risk}</span></div>
                    </div>
                    {scorecard.red_flags?.length > 0 && (
                        <div style={{ background: 'var(--color-error-bg)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)', marginTop: 'var(--space-sm)' }}>
                            {scorecard.red_flags.map((f: string, i: number) => <div key={i} style={{ color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>⚠ {f}</div>)}
                        </div>
                    )}
                </div>
            )}

            {evalResults.length > 0 && (
                <div className="card">
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Results</h3>
                    <table className="docs-table">
                        <thead><tr><th>Dataset</th><th>Type</th><th>Metrics</th><th>Pass Rate</th></tr></thead>
                        <tbody>
                            {evalResults.map(r => (
                                <tr key={r.id}>
                                    <td>{r.dataset_name}</td>
                                    <td><span className="badge badge-accent">{r.eval_type}</span></td>
                                    <td style={{ fontSize: 'var(--font-size-xs)', fontFamily: 'monospace' }}>{JSON.stringify(r.metrics)}</td>
                                    <td>{r.pass_rate != null ? `${(r.pass_rate * 100).toFixed(1)}%` : '—'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
