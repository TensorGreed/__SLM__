import { useState, useEffect } from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import './EvalPanel.css';

interface EvalPanelProps { projectId: number; onNextStep?: () => void; }

export default function EvalPanel({ projectId, onNextStep }: EvalPanelProps) {
    const [experiments, setExperiments] = useState<any[]>([]);
    const [selectedExp, setSelectedExp] = useState<number | null>(null);
    const [evalResults, setEvalResults] = useState<any[]>([]);
    const [scorecard, setScorecard] = useState<any>(null);
    const [loaded, setLoaded] = useState(false);

    // LLM Judge Form
    const [showJudgeForm, setShowJudgeForm] = useState(false);
    type Provider = 'hf' | 'ollama' | 'openai';
    const [provider, setProvider] = useState<Provider>('hf');
    const [judgeModel, setJudgeModel] = useState('meta-llama/Meta-Llama-3-70B-Instruct');
    const [benchmarkName, setBenchmarkName] = useState('MMLU-Subset');
    const [isEvaluating, setIsEvaluating] = useState(false);

    const handleProviderChange = (p: Provider) => {
        setProvider(p);
        if (p === 'ollama') setJudgeModel('llama3');
        else if (p === 'openai') setJudgeModel('gpt-4o');
        else setJudgeModel('meta-llama/Meta-Llama-3-70B-Instruct');
    };

    useEffect(() => {
        if (!loaded) {
            api.get(`/projects/${projectId}/training/experiments`).then(r => { setExperiments(r.data); setLoaded(true); });
        }
    }, [loaded, projectId]);

    const loadResults = async (expId: number) => {
        setSelectedExp(expId);
        listResults(expId);
    };

    const listResults = async (expId: number) => {
        const res = await api.get(`/projects/${projectId}/evaluation/results/${expId}`);
        setEvalResults(res.data);
        const sc = await api.get(`/projects/${projectId}/evaluation/safety-scorecard/${expId}`);
        setScorecard(sc.data);
    }

    const runLLMJudge = async () => {
        if (!selectedExp) return;
        setIsEvaluating(true);

        // Create some dummy predictions to send to the simulation
        const dummyPredictions = Array.from({ length: 10 }).map((_, i) => ({
            prompt: `Question ${i + 1} from ${benchmarkName}`,
            reference: `The ground truth expected answer for Q${i + 1} is detailed here.`,
            prediction: `The model's actual generated answer for Q${i + 1}. It might be short or long varying by the run.`
        }));

        try {
            await api.post(`/projects/${projectId}/evaluation/llm-judge`, {
                experiment_id: selectedExp,
                dataset_name: benchmarkName,
                judge_model: judgeModel,
                predictions: dummyPredictions
            });
            await listResults(selectedExp);
            setShowJudgeForm(false);
        } catch (e) {
            console.error(e);
        } finally {
            setIsEvaluating(false);
        }
    };

    const riskColor = (risk: string) => risk === 'low' ? 'badge-success' : risk === 'medium' ? 'badge-warning' : risk === 'high' ? 'badge-error' : 'badge-info';

    const radarData = evalResults
        .filter(r => r.eval_type === 'llm_judge')
        .map(r => ({
            subject: r.dataset_name,
            A: r.pass_rate * 100,
            fullMark: 100,
        }));

    const llmJudgeResults = evalResults.filter(r => r.eval_type === 'llm_judge');

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Select Experiment / Model</h3>
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

            {selectedExp && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h2 style={{ fontSize: 'var(--font-size-lg)', margin: 0 }}>Evaluation Suite</h2>
                    <button className="btn btn-primary" onClick={() => setShowJudgeForm(!showJudgeForm)}>+ Run LLM Benchmark</button>
                </div>
            )}

            {showJudgeForm && (
                <div className="card" style={{ border: '1px solid var(--color-primary)' }}>
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Configure LLM Judge</h3>
                    <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-lg)' }}>
                        The Judge Model will be loaded into DGX VRAM to automatically score the SLM's generations against the specified benchmark.
                    </p>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1.5fr', gap: 'var(--space-lg)' }}>
                        <div className="form-group">
                            <label className="form-label">Judge Provider</label>
                            <select className="input" value={provider} onChange={e => handleProviderChange(e.target.value as Provider)}>
                                <option value="hf">HuggingFace (vLLM)</option>
                                <option value="ollama">Local (Ollama)</option>
                                <option value="openai">Cloud (OpenAI)</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">
                                {provider === 'hf' ? 'HF Hub ID / Local Path' : 'Model Name'}
                            </label>
                            <input className="input" value={judgeModel} onChange={e => setJudgeModel(e.target.value)} placeholder={provider === 'openai' ? 'gpt-4o' : provider === 'ollama' ? 'llama3' : 'meta-llama/...'} />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Benchmark Suite</label>
                            <select className="input" value={benchmarkName} onChange={e => setBenchmarkName(e.target.value)}>
                                <option value="MMLU-Subset">MMLU (Subset)</option>
                                <option value="HumanEval">HumanEval (Coding)</option>
                                <option value="GSM8K">GSM8K (Math)</option>
                                <option value="CustomQA">Custom Gold QA</option>
                            </select>
                        </div>
                    </div>
                    <div style={{ marginTop: 'var(--space-lg)' }}>
                        <button className="btn btn-primary" onClick={runLLMJudge} disabled={isEvaluating}>
                            {isEvaluating ? 'Evaluating on GPU...' : 'Run Benchmark'}
                        </button>
                    </div>
                </div>
            )}

            {selectedExp && evalResults.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: 'var(--space-xl)' }}>

                    {/* Radar Overview */}
                    <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Skill Radar (LLM Judge)</h3>
                        <div style={{ flex: 1, minHeight: 300, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
                            {radarData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                                        <PolarGrid stroke="#333" />
                                        <PolarAngleAxis dataKey="subject" tick={{ fill: '#888', fontSize: 12 }} />
                                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#666' }} />
                                        <Radar name="Pass Rate %" dataKey="A" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.5} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-tertiary)', textAlign: 'center', padding: 20 }}>
                                    No LLM-judge benchmarks run yet.<br />Click "Run LLM Benchmark" to populate.
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Standard Metrics */}
                    <div className="card">
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Metric History</h3>
                        <div className="table-container" style={{ maxHeight: 300, overflowY: 'auto' }}>
                            <table className="docs-table">
                                <thead><tr><th>Dataset</th><th>Type</th><th>Pass Rate</th></tr></thead>
                                <tbody>
                                    {evalResults.map(r => (
                                        <tr key={r.id}>
                                            <td>{r.dataset_name}</td>
                                            <td><span className="badge badge-accent" style={{ textTransform: 'uppercase' }}>{r.eval_type.replace('_', ' ')}</span></td>
                                            <td><strong style={{ color: r.pass_rate > 0.8 ? 'var(--color-success)' : r.pass_rate > 0.5 ? 'var(--color-warning)' : 'var(--color-error)' }}>
                                                {r.pass_rate != null ? `${(r.pass_rate * 100).toFixed(1)}%` : '—'}
                                            </strong></td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* Side-by-Side Comparison */}
            {llmJudgeResults.length > 0 && (
                <div className="card">
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>🔍 LLM Judge: Side-by-Side Comparison</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        {llmJudgeResults[0].metrics?.scored_predictions?.slice(0, 3).map((p: any, idx: number) => (
                            <div key={idx} style={{ background: 'var(--bg-tertiary)', borderRadius: 8, padding: 'var(--space-md)', display: 'grid', gridTemplateColumns: 'minmax(200px, 1fr) minmax(200px, 1fr) 250px', gap: 16, borderLeft: `4px solid ${p.judge_score >= 4 ? 'var(--color-success)' : 'var(--color-error)'}` }}>

                                <div>
                                    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: 4 }}>Prompt</div>
                                    <div style={{ fontSize: 13, background: 'var(--bg-primary)', padding: 8, borderRadius: 4, marginBottom: 8 }}>{p.prompt}</div>

                                    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: 4 }}>Ground Truth (Reference)</div>
                                    <div style={{ fontSize: 13, background: 'rgba(34, 197, 94, 0.1)', padding: 8, borderRadius: 4, color: 'var(--color-success)' }}>{p.reference}</div>
                                </div>

                                <div>
                                    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: 4 }}>Model Prediction</div>
                                    <div style={{ fontSize: 13, background: 'rgba(234, 179, 8, 0.1)', padding: 8, borderRadius: 4, border: '1px solid rgba(234,179,8,0.2)' }}>{p.prediction}</div>
                                </div>

                                <div style={{ borderLeft: '1px solid var(--border-color)', paddingLeft: 16 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                                        <span style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Judge Score</span>
                                        <span className={`badge ${p.judge_score >= 4 ? 'badge-success' : 'badge-error'}`} style={{ fontSize: 14 }}>{p.judge_score} / 5</span>
                                    </div>
                                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                                        <strong>Rationale:</strong> {p.judge_rationale}
                                    </div>
                                </div>

                            </div>
                        ))}
                    </div>
                    {llmJudgeResults[0].metrics?.scored_predictions?.length > 3 && (
                        <div style={{ textAlign: 'center', marginTop: 16 }}>
                            <button className="btn btn-secondary btn-sm">View All Predictions</button>
                        </div>
                    )}
                </div>
            )}

            {scorecard && (
                <div className="card">
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>🛡️ Safety Scorecard</h3>
                    <div style={{ display: 'flex', gap: 'var(--space-xl)', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                        <div><strong>Overall Risk:</strong> <span className={`badge ${riskColor(scorecard.overall_risk)}`}>{scorecard.overall_risk.toUpperCase()}</span></div>
                    </div>
                    {scorecard.red_flags?.length > 0 && (
                        <div style={{ background: 'var(--color-error-bg)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)', marginTop: 'var(--space-sm)' }}>
                            {scorecard.red_flags.map((f: string, i: number) => <div key={i} style={{ color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>⚠ {f}</div>)}
                        </div>
                    )}
                </div>
            )}

            {onNextStep && (
                <StepFooter
                    currentStep="Evaluation"
                    nextStep="Compression"
                    nextStepIcon="📦"
                    isComplete={evalResults.length > 0}
                    hint="Run at least one evaluation to continue"
                    onNext={onNextStep}
                />
            )}

        </div>
    );
}
