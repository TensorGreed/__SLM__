import { useEffect, useMemo, useState } from 'react';
import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer } from 'recharts';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import './EvalPanel.css';

interface EvalPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

type Provider = 'hf' | 'ollama' | 'openai';
type HeldoutEvalType = 'exact_match' | 'f1' | 'llm_judge';

interface ExperimentSummary {
    id: number;
    name: string;
}

interface ScoredPrediction {
    prompt: string;
    reference: string;
    prediction: string;
    judge_score: number;
    judge_rationale: string;
}

interface InferenceMetrics {
    engine?: string;
    device?: string;
    average_latency_ms?: number;
    token_throughput_tps?: number;
}

interface EvalMetrics {
    scored_predictions?: ScoredPrediction[];
    inference?: InferenceMetrics;
}

interface EvalResult {
    id: number;
    dataset_name: string;
    eval_type: string;
    pass_rate: number | null;
    metrics: EvalMetrics;
}

interface SafetyScorecard {
    overall_risk: string;
    red_flags?: string[];
}

interface HeldoutEvalRequest {
    experiment_id: number;
    dataset_name: string;
    eval_type: HeldoutEvalType;
    max_samples: number;
    max_new_tokens: number;
    temperature: number;
    model_path?: string;
    judge_model?: string;
}

interface EvaluationPackSummary {
    pack_id: string;
    display_name: string;
    description: string;
    gate_count: number;
}

interface EvaluationPackListResponse {
    packs: EvaluationPackSummary[];
}

interface EvaluationPackPreferenceResponse {
    preferred_pack_id: string | null;
    active_pack_id: string | null;
    active_pack_source: string;
    active_pack?: {
        display_name?: string;
    };
}

interface GateCheck {
    gate_id: string;
    metric_id: string;
    operator: 'gte' | 'lte' | string;
    threshold: number | null;
    required: boolean;
    actual: number | null;
    passed: boolean;
    reason: string;
}

interface GateReport {
    captured_at: string;
    passed: boolean;
    failed_gate_ids: string[];
    missing_required_metrics: string[];
    checks: GateCheck[];
    pack?: {
        pack_id?: string;
        display_name?: string;
    };
}

const AUTO_PACK_VALUE = '__auto__';

function getErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const maybeDetail = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail;
        if (typeof maybeDetail === 'string' && maybeDetail.trim()) {
            return maybeDetail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Failed to run evaluation';
}

export default function EvalPanel({ projectId, onNextStep }: EvalPanelProps) {
    const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
    const [selectedExp, setSelectedExp] = useState<number | null>(null);
    const [evalResults, setEvalResults] = useState<EvalResult[]>([]);
    const [scorecard, setScorecard] = useState<SafetyScorecard | null>(null);

    const [showRunForm, setShowRunForm] = useState(false);
    const [provider, setProvider] = useState<Provider>('hf');
    const [judgeModel, setJudgeModel] = useState('meta-llama/Meta-Llama-3-70B-Instruct');
    const [datasetName, setDatasetName] = useState('test');
    const [evalType, setEvalType] = useState<HeldoutEvalType>('exact_match');
    const [maxSamples, setMaxSamples] = useState(100);
    const [maxNewTokens, setMaxNewTokens] = useState(128);
    const [temperature, setTemperature] = useState(0);
    const [modelPath, setModelPath] = useState('');

    const [loadingExperiments, setLoadingExperiments] = useState(false);
    const [isEvaluating, setIsEvaluating] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [evaluationPacks, setEvaluationPacks] = useState<EvaluationPackSummary[]>([]);
    const [preferredPackId, setPreferredPackId] = useState<string | null>(null);
    const [activePackId, setActivePackId] = useState<string | null>(null);
    const [activePackLabel, setActivePackLabel] = useState('');
    const [activePackSource, setActivePackSource] = useState('default');
    const [loadingPackState, setLoadingPackState] = useState(false);
    const [isSavingPackPreference, setIsSavingPackPreference] = useState(false);
    const [gateReport, setGateReport] = useState<GateReport | null>(null);
    const [isLoadingGateReport, setIsLoadingGateReport] = useState(false);
    const [gateErrorMessage, setGateErrorMessage] = useState<string | null>(null);

    useEffect(() => {
        setSelectedExp(null);
        setEvalResults([]);
        setScorecard(null);
        setErrorMessage(null);
        setGateErrorMessage(null);
        setEvaluationPacks([]);
        setPreferredPackId(null);
        setActivePackId(null);
        setActivePackLabel('');
        setActivePackSource('default');
        setGateReport(null);
        setShowRunForm(false);
    }, [projectId]);

    useEffect(() => {
        let active = true;
        const loadExperiments = async () => {
            setLoadingExperiments(true);
            try {
                const res = await api.get<ExperimentSummary[]>(`/projects/${projectId}/training/experiments`);
                if (active) {
                    setExperiments(res.data);
                }
            } catch (error) {
                if (active) {
                    setErrorMessage(getErrorMessage(error));
                }
            } finally {
                if (active) {
                    setLoadingExperiments(false);
                }
            }
        };

        void loadExperiments();
        return () => {
            active = false;
        };
    }, [projectId]);

    const loadPackState = async () => {
        setLoadingPackState(true);
        try {
            const [packsRes, prefRes] = await Promise.all([
                api.get<EvaluationPackListResponse>(`/projects/${projectId}/evaluation/packs`),
                api.get<EvaluationPackPreferenceResponse>(`/projects/${projectId}/evaluation/pack-preference`),
            ]);
            setEvaluationPacks(Array.isArray(packsRes.data?.packs) ? packsRes.data.packs : []);
            setPreferredPackId(prefRes.data?.preferred_pack_id ?? null);
            setActivePackId(prefRes.data?.active_pack_id ?? null);
            setActivePackSource(prefRes.data?.active_pack_source || 'default');
            setActivePackLabel(prefRes.data?.active_pack?.display_name || '');
        } catch {
            // Keep evaluation flow functional even if pack endpoints are unavailable.
            setEvaluationPacks([]);
            setPreferredPackId(null);
            setActivePackId(null);
            setActivePackSource('default');
            setActivePackLabel('');
        } finally {
            setLoadingPackState(false);
        }
    };

    const loadGateReport = async (expId: number) => {
        setIsLoadingGateReport(true);
        setGateErrorMessage(null);
        try {
            const res = await api.get<GateReport>(`/projects/${projectId}/evaluation/gates/${expId}`);
            setGateReport(res.data);
        } catch (error) {
            setGateReport(null);
            setGateErrorMessage(getErrorMessage(error));
        } finally {
            setIsLoadingGateReport(false);
        }
    };

    useEffect(() => {
        void loadPackState();
    }, [projectId]);

    const savePackPreference = async (packId: string | null) => {
        setIsSavingPackPreference(true);
        setGateErrorMessage(null);
        try {
            const res = await api.put<EvaluationPackPreferenceResponse>(
                `/projects/${projectId}/evaluation/pack-preference`,
                { pack_id: packId },
            );
            setPreferredPackId(res.data?.preferred_pack_id ?? null);
            setActivePackId(res.data?.active_pack_id ?? null);
            setActivePackSource(res.data?.active_pack_source || 'default');
            setActivePackLabel(res.data?.active_pack?.display_name || '');
            if (selectedExp) {
                await loadGateReport(selectedExp);
            }
        } catch (error) {
            setErrorMessage(getErrorMessage(error));
        } finally {
            setIsSavingPackPreference(false);
        }
    };

    const handleProviderChange = (nextProvider: Provider) => {
        setProvider(nextProvider);
        if (nextProvider === 'ollama') {
            setJudgeModel('llama3');
        } else if (nextProvider === 'openai') {
            setJudgeModel('gpt-4o');
        } else {
            setJudgeModel('meta-llama/Meta-Llama-3-70B-Instruct');
        }
    };

    const listResults = async (expId: number) => {
        const [res, sc] = await Promise.all([
            api.get<EvalResult[]>(`/projects/${projectId}/evaluation/results/${expId}`),
            api.get<SafetyScorecard>(`/projects/${projectId}/evaluation/safety-scorecard/${expId}`),
        ]);
        setEvalResults(res.data);
        setScorecard(sc.data);
    };

    const loadResults = async (expId: number) => {
        setErrorMessage(null);
        setSelectedExp(expId);
        try {
            await Promise.all([listResults(expId), loadGateReport(expId)]);
        } catch (error) {
            setErrorMessage(getErrorMessage(error));
        }
    };

    const runHeldoutEval = async () => {
        if (!selectedExp) {
            return;
        }

        setIsEvaluating(true);
        setErrorMessage(null);

        const payload: HeldoutEvalRequest = {
            experiment_id: selectedExp,
            dataset_name: datasetName,
            eval_type: evalType,
            max_samples: Math.max(1, maxSamples),
            max_new_tokens: Math.max(1, maxNewTokens),
            temperature: Math.max(0, temperature),
        };

        if (modelPath.trim()) {
            payload.model_path = modelPath.trim();
        }
        if (evalType === 'llm_judge') {
            payload.judge_model = judgeModel;
        }

        try {
            await api.post(`/projects/${projectId}/evaluation/run-heldout`, payload);
            await Promise.all([listResults(selectedExp), loadGateReport(selectedExp)]);
            setShowRunForm(false);
        } catch (error) {
            setErrorMessage(getErrorMessage(error));
        } finally {
            setIsEvaluating(false);
        }
    };

    const riskColor = (risk: string) => {
        if (risk === 'low') return 'badge-success';
        if (risk === 'medium') return 'badge-warning';
        if (risk === 'high') return 'badge-error';
        return 'badge-info';
    };

    const radarData = useMemo(
        () =>
            evalResults
                .filter((result) => result.eval_type === 'llm_judge')
                .map((result) => ({
                    subject: result.dataset_name,
                    A: (result.pass_rate ?? 0) * 100,
                    fullMark: 100,
                })),
        [evalResults],
    );

    const llmJudgeResults = useMemo(
        () => evalResults.filter((result) => result.eval_type === 'llm_judge'),
        [evalResults],
    );

    const scoredPredictions = llmJudgeResults[0]?.metrics.scored_predictions ?? [];

    const latestInference = useMemo(() => {
        for (let i = evalResults.length - 1; i >= 0; i -= 1) {
            const inference = evalResults[i].metrics.inference;
            if (inference) {
                return inference;
            }
        }
        return null;
    }, [evalResults]);

    const activePackDisplayName = useMemo(() => {
        if (activePackLabel.trim()) {
            return activePackLabel;
        }
        if (!activePackId) {
            return 'Auto';
        }
        const found = evaluationPacks.find((pack) => pack.pack_id === activePackId);
        return found?.display_name || activePackId;
    }, [activePackId, activePackLabel, evaluationPacks]);

    const gateStatusBadge = useMemo(() => {
        if (!gateReport) {
            return { className: 'badge-info', label: 'NO REPORT' };
        }
        return gateReport.passed
            ? { className: 'badge-success', label: 'PASS' }
            : { className: 'badge-error', label: 'FAIL' };
    }, [gateReport]);

    const onPackPreferenceSelect = (value: string) => {
        const nextPackId = value === AUTO_PACK_VALUE ? null : value;
        void savePackPreference(nextPackId);
    };

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Select Experiment / Model</h3>
                {loadingExperiments ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">⏳</div>
                        <div className="empty-state-title">Loading experiments...</div>
                    </div>
                ) : experiments.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">📊</div>
                        <div className="empty-state-title">No experiments to evaluate</div>
                    </div>
                ) : (
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        {experiments.map((experiment) => (
                            <button
                                key={experiment.id}
                                className={`btn ${selectedExp === experiment.id ? 'btn-primary' : 'btn-secondary'}`}
                                onClick={() => void loadResults(experiment.id)}
                            >
                                {experiment.name}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {errorMessage && (
                <div className="card" style={{ border: '1px solid var(--color-error)', color: 'var(--color-error)' }}>
                    {errorMessage}
                </div>
            )}

            {selectedExp && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h2 style={{ fontSize: 'var(--font-size-lg)', margin: 0 }}>Evaluation Suite</h2>
                    <button className="btn btn-primary" onClick={() => setShowRunForm((open) => !open)}>
                        + Run Held-out Evaluation
                    </button>
                </div>
            )}

            {selectedExp && (
                <div className="card eval-pack-card">
                    <div className="eval-pack-header">
                        <div>
                            <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 4 }}>
                                Evaluation Pack & Auto Gates
                            </h3>
                            <p className="eval-pack-subtitle">
                                Choose a project-level gate profile and review pass/fail checks for the selected experiment.
                            </p>
                        </div>
                        <span className={`badge ${gateStatusBadge.className}`}>{gateStatusBadge.label}</span>
                    </div>

                    <div className="eval-pack-controls">
                        <div className="form-group">
                            <label className="form-label">Project Gate Pack</label>
                            <select
                                className="input"
                                value={preferredPackId ?? AUTO_PACK_VALUE}
                                onChange={(e) => onPackPreferenceSelect(e.target.value)}
                                disabled={loadingPackState || isSavingPackPreference}
                            >
                                <option value={AUTO_PACK_VALUE}>Auto (domain profile fallback)</option>
                                {evaluationPacks.map((pack, index) => (
                                    <option key={`${pack.pack_id}:${index}`} value={pack.pack_id}>
                                        {pack.display_name} ({pack.gate_count} gates)
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div className="eval-pack-meta">
                            <div>
                                <span>Active Pack</span>
                                <strong>{activePackDisplayName}</strong>
                            </div>
                            <div>
                                <span>Source</span>
                                <strong>{activePackSource.replace(/_/g, ' ')}</strong>
                            </div>
                        </div>
                    </div>

                    {(loadingPackState || isSavingPackPreference) && (
                        <div className="eval-pack-note">
                            {loadingPackState ? 'Loading evaluation pack settings...' : 'Saving pack preference...'}
                        </div>
                    )}

                    {isLoadingGateReport ? (
                        <div className="eval-pack-note">Running gate checks...</div>
                    ) : gateErrorMessage ? (
                        <div className="eval-pack-error">{gateErrorMessage}</div>
                    ) : gateReport ? (
                        <>
                            <div className="eval-gate-summary">
                                <span>
                                    Required gate failures: <strong>{gateReport.failed_gate_ids.length}</strong>
                                </span>
                                <span>
                                    Missing required metrics: <strong>{gateReport.missing_required_metrics.length}</strong>
                                </span>
                                <span>
                                    Checked at: <strong>{new Date(gateReport.captured_at).toLocaleString()}</strong>
                                </span>
                            </div>
                            <div className="table-container eval-gate-table">
                                <table className="docs-table">
                                    <thead>
                                        <tr>
                                            <th>Gate</th>
                                            <th>Metric</th>
                                            <th>Threshold</th>
                                            <th>Actual</th>
                                            <th>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {gateReport.checks.map((check) => (
                                            <tr key={check.gate_id}>
                                                <td>{check.gate_id}</td>
                                                <td>{check.metric_id}</td>
                                                <td>
                                                    {check.threshold == null
                                                        ? '—'
                                                        : `${check.operator === 'lte' ? '≤' : '≥'} ${check.threshold}`}
                                                </td>
                                                <td>{check.actual == null ? '—' : check.actual}</td>
                                                <td>
                                                    <span className={`badge ${check.passed ? 'badge-success' : 'badge-error'}`}>
                                                        {check.passed ? 'pass' : check.reason}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </>
                    ) : (
                        <div className="eval-pack-note">Select an experiment to compute gate checks.</div>
                    )}
                </div>
            )}

            {showRunForm && (
                <div className="card" style={{ border: '1px solid var(--color-primary)' }}>
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Configure Held-out Evaluation</h3>
                    <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-lg)' }}>
                        This runs inference on held-out dataset rows and stores exact-match, F1, or LLM-judge results.
                    </p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 'var(--space-lg)' }}>
                        <div className="form-group">
                            <label className="form-label">Dataset</label>
                            <select className="input" value={datasetName} onChange={(e) => setDatasetName(e.target.value)}>
                                <option value="test">Test</option>
                                <option value="validation">Validation</option>
                                <option value="gold_test">Gold Test</option>
                                <option value="gold_dev">Gold Dev</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Metric</label>
                            <select
                                className="input"
                                value={evalType}
                                onChange={(e) => setEvalType(e.target.value as HeldoutEvalType)}
                            >
                                <option value="exact_match">Exact Match</option>
                                <option value="f1">F1</option>
                                <option value="llm_judge">LLM Judge</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Max Samples</label>
                            <input
                                className="input"
                                type="number"
                                min={1}
                                max={5000}
                                value={maxSamples}
                                onChange={(e) => setMaxSamples(Number(e.target.value))}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Max New Tokens</label>
                            <input
                                className="input"
                                type="number"
                                min={1}
                                max={1024}
                                value={maxNewTokens}
                                onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Temperature</label>
                            <input
                                className="input"
                                type="number"
                                min={0}
                                max={2}
                                step={0.1}
                                value={temperature}
                                onChange={(e) => setTemperature(Number(e.target.value))}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Model Path (optional)</label>
                            <input
                                className="input"
                                value={modelPath}
                                onChange={(e) => setModelPath(e.target.value)}
                                placeholder="Leave blank to use experiment output/model"
                            />
                        </div>
                    </div>

                    {evalType === 'llm_judge' && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
                            <div className="form-group">
                                <label className="form-label">Judge Provider</label>
                                <select
                                    className="input"
                                    value={provider}
                                    onChange={(e) => handleProviderChange(e.target.value as Provider)}
                                >
                                    <option value="hf">HuggingFace (vLLM)</option>
                                    <option value="ollama">Local (Ollama)</option>
                                    <option value="openai">Cloud (OpenAI)</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Judge Model</label>
                                <input
                                    className="input"
                                    value={judgeModel}
                                    onChange={(e) => setJudgeModel(e.target.value)}
                                    placeholder={provider === 'openai' ? 'gpt-4o' : provider === 'ollama' ? 'llama3' : 'meta-llama/...'}
                                />
                            </div>
                        </div>
                    )}

                    <div style={{ marginTop: 'var(--space-lg)' }}>
                        <button className="btn btn-primary" onClick={() => void runHeldoutEval()} disabled={isEvaluating}>
                            {isEvaluating ? 'Running Held-out Evaluation...' : 'Run Evaluation'}
                        </button>
                    </div>
                </div>
            )}

            {latestInference && (
                <div className="card" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 'var(--space-md)' }}>
                    <div>
                        <div style={{ color: 'var(--text-tertiary)', fontSize: 12 }}>Runtime</div>
                        <strong>{latestInference.engine ?? 'unknown'}</strong>
                    </div>
                    <div>
                        <div style={{ color: 'var(--text-tertiary)', fontSize: 12 }}>Device</div>
                        <strong>{latestInference.device ?? 'unknown'}</strong>
                    </div>
                    <div>
                        <div style={{ color: 'var(--text-tertiary)', fontSize: 12 }}>Avg Latency</div>
                        <strong>{latestInference.average_latency_ms != null ? `${latestInference.average_latency_ms.toFixed(2)} ms` : '—'}</strong>
                    </div>
                    <div>
                        <div style={{ color: 'var(--text-tertiary)', fontSize: 12 }}>Token Throughput</div>
                        <strong>{latestInference.token_throughput_tps != null ? `${latestInference.token_throughput_tps.toFixed(2)} tok/s` : '—'}</strong>
                    </div>
                </div>
            )}

            {selectedExp && evalResults.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: 'var(--space-xl)' }}>
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
                                    No LLM-judge benchmarks run yet.<br />Run a held-out eval with metric set to LLM Judge.
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="card">
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Metric History</h3>
                        <div className="table-container" style={{ maxHeight: 300, overflowY: 'auto' }}>
                            <table className="docs-table">
                                <thead>
                                    <tr>
                                        <th>Dataset</th>
                                        <th>Type</th>
                                        <th>Pass Rate</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {evalResults.map((result) => (
                                        <tr key={result.id}>
                                            <td>{result.dataset_name}</td>
                                            <td>
                                                <span className="badge badge-accent" style={{ textTransform: 'uppercase' }}>
                                                    {result.eval_type.replace('_', ' ')}
                                                </span>
                                            </td>
                                            <td>
                                                <strong
                                                    style={{
                                                        color:
                                                            result.pass_rate != null && result.pass_rate > 0.8
                                                                ? 'var(--color-success)'
                                                                : result.pass_rate != null && result.pass_rate > 0.5
                                                                  ? 'var(--color-warning)'
                                                                  : 'var(--color-error)',
                                                    }}
                                                >
                                                    {result.pass_rate != null ? `${(result.pass_rate * 100).toFixed(1)}%` : '—'}
                                                </strong>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {scoredPredictions.length > 0 && (
                <div className="card">
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>🔍 LLM Judge: Side-by-Side Comparison</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        {scoredPredictions.slice(0, 3).map((prediction, idx) => (
                            <div
                                key={idx}
                                style={{
                                    background: 'var(--bg-tertiary)',
                                    borderRadius: 8,
                                    padding: 'var(--space-md)',
                                    display: 'grid',
                                    gridTemplateColumns: 'minmax(200px, 1fr) minmax(200px, 1fr) 250px',
                                    gap: 16,
                                    borderLeft: `4px solid ${prediction.judge_score >= 4 ? 'var(--color-success)' : 'var(--color-error)'}`,
                                }}
                            >
                                <div>
                                    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: 4 }}>Prompt</div>
                                    <div style={{ fontSize: 13, background: 'var(--bg-primary)', padding: 8, borderRadius: 4, marginBottom: 8 }}>{prediction.prompt}</div>

                                    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: 4 }}>Ground Truth (Reference)</div>
                                    <div style={{ fontSize: 13, background: 'rgba(34, 197, 94, 0.1)', padding: 8, borderRadius: 4, color: 'var(--color-success)' }}>{prediction.reference}</div>
                                </div>

                                <div>
                                    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: 4 }}>Model Prediction</div>
                                    <div style={{ fontSize: 13, background: 'rgba(234, 179, 8, 0.1)', padding: 8, borderRadius: 4, border: '1px solid rgba(234,179,8,0.2)' }}>{prediction.prediction}</div>
                                </div>

                                <div style={{ borderLeft: '1px solid var(--border-color)', paddingLeft: 16 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                                        <span style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Judge Score</span>
                                        <span className={`badge ${prediction.judge_score >= 4 ? 'badge-success' : 'badge-error'}`} style={{ fontSize: 14 }}>
                                            {prediction.judge_score} / 5
                                        </span>
                                    </div>
                                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                                        <strong>Rationale:</strong> {prediction.judge_rationale}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                    {scoredPredictions.length > 3 && (
                        <div style={{ textAlign: 'center', marginTop: 16 }}>
                            <button className="btn btn-secondary btn-sm" disabled>
                                Showing first 3 of {scoredPredictions.length}
                            </button>
                        </div>
                    )}
                </div>
            )}

            {scorecard && (
                <div className="card">
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>🛡️ Safety Scorecard</h3>
                    <div style={{ display: 'flex', gap: 'var(--space-xl)', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                        <div>
                            <strong>Overall Risk:</strong>{' '}
                            <span className={`badge ${riskColor(scorecard.overall_risk)}`}>{scorecard.overall_risk.toUpperCase()}</span>
                        </div>
                    </div>
                    {(scorecard.red_flags ?? []).length > 0 && (
                        <div style={{ background: 'var(--color-error-bg)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)', marginTop: 'var(--space-sm)' }}>
                            {(scorecard.red_flags ?? []).map((flag, idx) => (
                                <div key={idx} style={{ color: 'var(--color-error)', fontSize: 'var(--font-size-sm)' }}>
                                    ⚠ {flag}
                                </div>
                            ))}
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
