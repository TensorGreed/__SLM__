import { useEffect, useMemo, useState } from 'react';
import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer } from 'recharts';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import ScorecardPanel from './ScorecardPanel';
import GoldSetWorkbenchPanel from './GoldSetWorkbenchPanel';
import FailureClustersPanel from './FailureClustersPanel';
import './EvalPanel.css';

type EvalSubTab = 'runs' | 'workbench';

interface EvalPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

type Provider = 'hf' | 'ollama' | 'openai' | 'local_serve';
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

interface LocalJudgeServeRun {
    run_id: string;
    status?: string;
    source?: string;
    template_id?: string;
    template_name?: string;
    export_id?: number | null;
    model_id?: number | null;
    smoke_url?: string | null;
    first_token_url?: string | null;
    first_healthy_at?: string | null;
    startup_latency_ms?: number | null;
}

interface LocalJudgeServeRunListResponse {
    count?: number;
    runs?: LocalJudgeServeRun[];
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
    resolved_metric_key?: string | null;
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
    missing_required_schema_metrics?: string[];
    task_profile?: string;
    task_profile_source?: string;
    task_profile_selected?: string;
    task_profile_fallback_used?: boolean;
    task_spec?: {
        task_profile?: string;
        display_name?: string;
        required_metric_ids?: string[];
    };
    checks: GateCheck[];
    pack?: {
        pack_id?: string;
        display_name?: string;
    };
}

interface RemediationPlanGenerateRequest {
    experiment_id: number;
    evaluation_result_id?: number;
    max_failures: number;
}

interface RemediationPlanIndexEntry {
    plan_id: string;
    created_at?: string | null;
    experiment_id?: number | null;
    evaluation_result_id?: number | null;
    eval_type?: string | null;
    dataset_name?: string | null;
    root_causes?: string[];
    summary?: {
        total_failures_analyzed?: number | null;
        cluster_count?: number | null;
        recommendation_count?: number | null;
        dominant_root_cause?: string | null;
    };
}

interface RemediationPlanIndexResponse {
    project_id: number;
    count: number;
    plans: RemediationPlanIndexEntry[];
}

interface RemediationPlanCluster {
    cluster_id?: string;
    root_cause?: string;
    error_type?: string;
    slice?: string;
    failure_count?: number;
    confidence?: number;
}

interface RemediationPlanRecommendation {
    recommendation_id?: string;
    root_cause?: string;
    title?: string;
    confidence?: number;
    rationale?: string;
    expected_impact?: {
        metric?: string;
        estimated_delta?: number | null;
        confidence?: number | null;
        horizon?: string | null;
    };
    data_operations?: Array<string | { action?: string; description?: string }>;
    training_config_changes?: Array<string | { key?: string; from?: unknown; to?: unknown; reason?: string }>;
}

interface RemediationPlanDetail {
    plan_id: string;
    created_at?: string | null;
    source_evaluation?: {
        id?: number;
        eval_type?: string;
        dataset_name?: string;
        pass_rate?: number | null;
    };
    summary?: {
        total_failures_analyzed?: number | null;
        cluster_count?: number | null;
        recommendation_count?: number | null;
        dominant_root_cause?: string | null;
    };
    clusters?: RemediationPlanCluster[];
    recommendations?: RemediationPlanRecommendation[];
    analysis_evidence?: {
        failures_analyzed?: number;
    };
}

const AUTO_PACK_VALUE = '__auto__';

function getErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const maybeDetail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof maybeDetail === 'string' && maybeDetail.trim()) {
            return maybeDetail;
        }
        if (typeof maybeDetail === 'object' && maybeDetail !== null) {
            const detail = maybeDetail as {
                message?: unknown;
                actionable_fix?: unknown;
                error_code?: unknown;
            };
            const message = typeof detail.message === 'string' ? detail.message.trim() : '';
            const actionableFix = typeof detail.actionable_fix === 'string' ? detail.actionable_fix.trim() : '';
            if (message && actionableFix) {
                return `${message} ${actionableFix}`;
            }
            if (message) {
                return message;
            }
            const errorCode = typeof detail.error_code === 'string' ? detail.error_code.trim() : '';
            if (errorCode) {
                return errorCode;
            }
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
    const [localServeRuns, setLocalServeRuns] = useState<LocalJudgeServeRun[]>([]);
    const [loadingLocalServeRuns, setLoadingLocalServeRuns] = useState(false);
    const [selectedLocalServeRunId, setSelectedLocalServeRunId] = useState('auto');
    const [localServeAutoStart, setLocalServeAutoStart] = useState(false);
    const [localServeAutoStop, setLocalServeAutoStop] = useState(true);
    const [localServeSource, setLocalServeSource] = useState<'export' | 'registry'>('export');
    const [localServeExportId, setLocalServeExportId] = useState('');
    const [localServeModelId, setLocalServeModelId] = useState('');
    const [localServeTemplateId, setLocalServeTemplateId] = useState('runner.vllm');
    const [localServeHost, setLocalServeHost] = useState('127.0.0.1');
    const [localServePort, setLocalServePort] = useState('8000');
    const [datasetName, setDatasetName] = useState('test');
    const [evalType, setEvalType] = useState<HeldoutEvalType>('exact_match');
    const [maxSamples, setMaxSamples] = useState(100);
    const [maxNewTokens, setMaxNewTokens] = useState(128);
    const [temperature, setTemperature] = useState(0);
    const [modelPath, setModelPath] = useState('');
    const [subTab, setSubTab] = useState<EvalSubTab>('runs');

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
    const [remediationPlans, setRemediationPlans] = useState<RemediationPlanIndexEntry[]>([]);
    const [selectedRemediationPlanId, setSelectedRemediationPlanId] = useState('');
    const [selectedRemediationEvalResultId, setSelectedRemediationEvalResultId] = useState('');
    const [remediationPlan, setRemediationPlan] = useState<RemediationPlanDetail | null>(null);
    const [isLoadingRemediationPlans, setIsLoadingRemediationPlans] = useState(false);
    const [isLoadingRemediationPlan, setIsLoadingRemediationPlan] = useState(false);
    const [isGeneratingRemediationPlan, setIsGeneratingRemediationPlan] = useState(false);
    const [remediationErrorMessage, setRemediationErrorMessage] = useState<string | null>(null);
    const [remediationMaxFailures, setRemediationMaxFailures] = useState(200);

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
        setRemediationPlans([]);
        setSelectedRemediationPlanId('');
        setSelectedRemediationEvalResultId('');
        setRemediationPlan(null);
        setRemediationErrorMessage(null);
        setRemediationMaxFailures(200);
        setLocalServeRuns([]);
        setSelectedLocalServeRunId('auto');
        setLocalServeAutoStart(false);
        setLocalServeAutoStop(true);
        setLocalServeSource('export');
        setLocalServeExportId('');
        setLocalServeModelId('');
        setLocalServeTemplateId('runner.vllm');
        setLocalServeHost('127.0.0.1');
        setLocalServePort('8000');
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

    const loadLocalServeRuns = async () => {
        setLoadingLocalServeRuns(true);
        try {
            const res = await api.get<LocalJudgeServeRunListResponse>(
                `/projects/${projectId}/evaluation/local-judge/serve-runs`,
                { params: { limit: 50 } },
            );
            const runs = Array.isArray(res.data?.runs) ? res.data.runs : [];
            setLocalServeRuns(runs);
        } catch {
            setLocalServeRuns([]);
        } finally {
            setLoadingLocalServeRuns(false);
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

    const loadRemediationPlan = async (planId: string) => {
        const token = String(planId || '').trim();
        if (!token) {
            setRemediationPlan(null);
            setSelectedRemediationPlanId('');
            return;
        }
        setIsLoadingRemediationPlan(true);
        setRemediationErrorMessage(null);
        try {
            const res = await api.get<RemediationPlanDetail>(
                `/projects/${projectId}/evaluation/remediation-plans/${encodeURIComponent(token)}`,
            );
            setRemediationPlan(res.data || null);
            setSelectedRemediationPlanId(token);
        } catch (error) {
            setRemediationPlan(null);
            setRemediationErrorMessage(getErrorMessage(error));
        } finally {
            setIsLoadingRemediationPlan(false);
        }
    };

    const loadRemediationPlans = async (expId: number) => {
        setIsLoadingRemediationPlans(true);
        setRemediationErrorMessage(null);
        try {
            const res = await api.get<RemediationPlanIndexResponse>(
                `/projects/${projectId}/evaluation/remediation-plans`,
                {
                    params: { experiment_id: expId, limit: 30 },
                },
            );
            const plans = Array.isArray(res.data?.plans) ? res.data.plans : [];
            setRemediationPlans(plans);
            if (plans.length === 0) {
                setSelectedRemediationPlanId('');
                setRemediationPlan(null);
                return;
            }
            const preferredPlanId = String(selectedRemediationPlanId || '').trim();
            const nextPlan = plans.find((item) => item.plan_id === preferredPlanId) || plans[0];
            if (nextPlan?.plan_id) {
                await loadRemediationPlan(nextPlan.plan_id);
            }
        } catch (error) {
            setRemediationPlans([]);
            setRemediationPlan(null);
            setRemediationErrorMessage(getErrorMessage(error));
        } finally {
            setIsLoadingRemediationPlans(false);
        }
    };

    useEffect(() => {
        void loadPackState();
    }, [projectId]);

    useEffect(() => {
        if (evalType !== 'llm_judge') {
            return;
        }
        if (provider !== 'local_serve') {
            return;
        }
        void loadLocalServeRuns();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId, evalType, provider]);

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
        if (nextProvider === 'local_serve') {
            setJudgeModel('local-judge');
        } else if (nextProvider === 'ollama') {
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
            void loadRemediationPlans(expId);
        } catch (error) {
            setErrorMessage(getErrorMessage(error));
        }
    };

    useEffect(() => {
        if (!selectedExp) {
            return;
        }
        if (!evalResults.length) {
            setSelectedRemediationEvalResultId('');
            return;
        }
        const selectedToken = String(selectedRemediationEvalResultId || '').trim();
        const hasSelected = evalResults.some((row) => String(row.id) === selectedToken);
        if (hasSelected) {
            return;
        }
        setSelectedRemediationEvalResultId(String(evalResults[0].id));
    }, [evalResults, selectedExp, selectedRemediationEvalResultId]);

    const generateRemediationPlan = async () => {
        if (!selectedExp) {
            return;
        }
        setIsGeneratingRemediationPlan(true);
        setRemediationErrorMessage(null);
        try {
            const payload: RemediationPlanGenerateRequest = {
                experiment_id: selectedExp,
                max_failures: Math.max(1, Math.min(1000, Math.floor(remediationMaxFailures || 200))),
            };
            const evalResultId = Number.parseInt(String(selectedRemediationEvalResultId || '').trim(), 10);
            if (Number.isFinite(evalResultId) && evalResultId > 0) {
                payload.evaluation_result_id = evalResultId;
            }
            const res = await api.post<RemediationPlanDetail>(
                `/projects/${projectId}/evaluation/remediation-plans/generate`,
                payload,
            );
            const detail = res.data || null;
            setRemediationPlan(detail);
            setSelectedRemediationPlanId(String(detail?.plan_id || ''));
            await loadRemediationPlans(selectedExp);
        } catch (error) {
            setRemediationErrorMessage(getErrorMessage(error));
        } finally {
            setIsGeneratingRemediationPlan(false);
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
            if (provider === 'local_serve') {
                const runToken = selectedLocalServeRunId.trim() || 'auto';
                const params = new URLSearchParams();
                const judgeModelToken = judgeModel.trim();
                if (judgeModelToken) {
                    params.set('model', judgeModelToken);
                }
                if (localServeAutoStart) {
                    params.set('auto_start', '1');
                    params.set('source', localServeSource);
                    const exportId = Number.parseInt(localServeExportId, 10);
                    const modelId = Number.parseInt(localServeModelId, 10);
                    if (localServeSource === 'export' && Number.isFinite(exportId) && exportId > 0) {
                        params.set('export_id', String(exportId));
                    }
                    if (localServeSource === 'registry' && Number.isFinite(modelId) && modelId > 0) {
                        params.set('model_id', String(modelId));
                    }
                    if (localServeTemplateId.trim()) {
                        params.set('template_id', localServeTemplateId.trim());
                    }
                    if (localServeHost.trim()) {
                        params.set('host', localServeHost.trim());
                    }
                    const port = Number.parseInt(localServePort, 10);
                    if (Number.isFinite(port) && port > 0) {
                        params.set('port', String(port));
                    }
                    if (localServeAutoStop) {
                        params.set('auto_stop', '1');
                    }
                }
                const suffix = params.toString();
                payload.judge_model = `local_serve:${runToken}${suffix ? `?${suffix}` : ''}`;
            } else {
                payload.judge_model = judgeModel;
            }
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

    const remediationDataOps = (recommendation: RemediationPlanRecommendation): string[] => {
        const rows: string[] = [];
        for (const item of recommendation.data_operations || []) {
            if (typeof item === 'string' && item.trim()) {
                rows.push(item.trim());
                continue;
            }
            if (typeof item === 'object' && item !== null) {
                const action = String((item as { action?: unknown }).action || '').trim();
                const description = String((item as { description?: unknown }).description || '').trim();
                if (action && description) {
                    rows.push(`${action}: ${description}`);
                } else if (description) {
                    rows.push(description);
                } else if (action) {
                    rows.push(action);
                }
            }
        }
        return rows;
    };

    const remediationTrainingChanges = (recommendation: RemediationPlanRecommendation): string[] => {
        const rows: string[] = [];
        for (const item of recommendation.training_config_changes || []) {
            if (typeof item === 'string' && item.trim()) {
                rows.push(item.trim());
                continue;
            }
            if (typeof item === 'object' && item !== null) {
                const key = String((item as { key?: unknown }).key || '').trim();
                const from = (item as { from?: unknown }).from;
                const to = (item as { to?: unknown }).to;
                const reason = String((item as { reason?: unknown }).reason || '').trim();
                const base = key
                    ? `${key}: ${String(from ?? 'current')} -> ${String(to ?? 'recommended')}`
                    : reason || 'Adjust training configuration';
                rows.push(reason ? `${base} (${reason})` : base);
            }
        }
        return rows;
    };

    const onPackPreferenceSelect = (value: string) => {
        const nextPackId = value === AUTO_PACK_VALUE ? null : value;
        void savePackPreference(nextPackId);
    };

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="eval-subtab-switch" role="tablist" aria-label="Evaluation sections">
                <button
                    type="button"
                    role="tab"
                    aria-selected={subTab === 'runs' ? 'true' : 'false'}
                    className={`eval-subtab-btn ${subTab === 'runs' ? 'active' : ''}`}
                    onClick={() => setSubTab('runs')}
                >
                    Eval runs
                </button>
                <button
                    type="button"
                    role="tab"
                    aria-selected={subTab === 'workbench' ? 'true' : 'false'}
                    className={`eval-subtab-btn ${subTab === 'workbench' ? 'active' : ''}`}
                    onClick={() => setSubTab('workbench')}
                >
                    Workbench
                </button>
            </div>
            {subTab === 'workbench' && <GoldSetWorkbenchPanel projectId={projectId} />}
            {subTab === 'runs' && (<>
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
                                    Task profile: <strong>{gateReport.task_profile_selected || gateReport.task_profile || 'auto'}</strong>
                                </span>
                                <span>
                                    Checked at: <strong>{new Date(gateReport.captured_at).toLocaleString()}</strong>
                                </span>
                            </div>
                            {!!gateReport.task_profile_source && (
                                <div className="eval-pack-note">
                                    Resolved from <strong>{gateReport.task_profile_source}</strong>
                                    {gateReport.task_profile_fallback_used ? ' (fallback task spec used)' : ''}
                                </div>
                            )}
                            <div className="table-container eval-gate-table">
                                <table className="docs-table">
                                    <thead>
                                        <tr>
                                            <th>Gate</th>
                                            <th>Metric</th>
                                            <th>Resolved</th>
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
                                                <td>{check.resolved_metric_key || '—'}</td>
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

            {selectedExp && (
                <div className="card eval-pack-card">
                    <div className="eval-pack-header">
                        <div>
                            <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 4 }}>
                                Closed-Loop Remediation Planner
                            </h3>
                            <p className="eval-pack-subtitle">
                                Turn evaluation failures into concrete data and training fixes.
                            </p>
                        </div>
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                                if (selectedExp) {
                                    void loadRemediationPlans(selectedExp);
                                }
                            }}
                            disabled={isLoadingRemediationPlans || isLoadingRemediationPlan}
                        >
                            {isLoadingRemediationPlans ? 'Refreshing...' : 'Refresh Plans'}
                        </button>
                    </div>

                    <div className="eval-remediation-controls">
                        <div className="form-group">
                            <label className="form-label">Evaluation Result Scope</label>
                            <select
                                className="input"
                                value={selectedRemediationEvalResultId}
                                onChange={(e) => setSelectedRemediationEvalResultId(e.target.value)}
                            >
                                <option value="">Auto (latest available result)</option>
                                {evalResults.map((result) => (
                                    <option key={result.id} value={result.id}>
                                        #{result.id} • {result.dataset_name} • {result.eval_type}
                                        {' '}
                                        {result.pass_rate == null ? '' : `(${(result.pass_rate * 100).toFixed(1)}%)`}
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Failures to Analyze</label>
                            <input
                                className="input"
                                type="number"
                                min={1}
                                max={1000}
                                value={remediationMaxFailures}
                                onChange={(e) => setRemediationMaxFailures(Number(e.target.value))}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Generate Plan</label>
                            <button
                                className="btn btn-primary"
                                onClick={() => void generateRemediationPlan()}
                                disabled={isGeneratingRemediationPlan || !selectedExp}
                            >
                                {isGeneratingRemediationPlan ? 'Generating...' : 'Generate Remediation Plan'}
                            </button>
                        </div>
                    </div>

                    {remediationErrorMessage && (
                        <div className="eval-pack-error">{remediationErrorMessage}</div>
                    )}

                    {isLoadingRemediationPlans ? (
                        <div className="eval-pack-note">Loading remediation plans...</div>
                    ) : remediationPlans.length === 0 ? (
                        <div className="eval-pack-note">
                            No remediation plans generated yet for this experiment.
                        </div>
                    ) : (
                        <div className="eval-remediation-plan-list">
                            {remediationPlans.map((plan) => (
                                <button
                                    key={plan.plan_id}
                                    className={`btn btn-sm ${selectedRemediationPlanId === plan.plan_id ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => void loadRemediationPlan(plan.plan_id)}
                                >
                                    {plan.eval_type || 'evaluation'} • {plan.dataset_name || 'dataset'}
                                    {typeof plan.summary?.recommendation_count === 'number'
                                        ? ` • ${plan.summary.recommendation_count} actions`
                                        : ''}
                                </button>
                            ))}
                        </div>
                    )}

                    {isLoadingRemediationPlan && (
                        <div className="eval-pack-note">Loading plan details...</div>
                    )}

                    {!isLoadingRemediationPlan && remediationPlan && (
                        <div className="eval-remediation-details">
                            <div className="eval-remediation-summary">
                                <div>
                                    <span>Plan</span>
                                    <strong>{remediationPlan.plan_id}</strong>
                                </div>
                                <div>
                                    <span>Failures Analyzed</span>
                                    <strong>{Number(remediationPlan.summary?.total_failures_analyzed || remediationPlan.analysis_evidence?.failures_analyzed || 0)}</strong>
                                </div>
                                <div>
                                    <span>Dominant Root Cause</span>
                                    <strong>{String(remediationPlan.summary?.dominant_root_cause || 'unknown').replace(/_/g, ' ')}</strong>
                                </div>
                                <div>
                                    <span>Created</span>
                                    <strong>
                                        {remediationPlan.created_at
                                            ? new Date(remediationPlan.created_at).toLocaleString()
                                            : '—'}
                                    </strong>
                                </div>
                            </div>

                            {Array.isArray(remediationPlan.clusters) && remediationPlan.clusters.length > 0 && (
                                <div className="table-container eval-gate-table">
                                    <table className="docs-table">
                                        <thead>
                                            <tr>
                                                <th>Root Cause</th>
                                                <th>Slice</th>
                                                <th>Failures</th>
                                                <th>Confidence</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {remediationPlan.clusters.map((cluster, idx) => (
                                                <tr key={`${cluster.cluster_id || cluster.root_cause || 'cluster'}-${idx}`}>
                                                    <td>{String(cluster.root_cause || cluster.error_type || 'unknown').replace(/_/g, ' ')}</td>
                                                    <td>{cluster.slice || 'general'}</td>
                                                    <td>{Number(cluster.failure_count || 0)}</td>
                                                    <td>
                                                        {typeof cluster.confidence === 'number'
                                                            ? `${(cluster.confidence * 100).toFixed(0)}%`
                                                            : '—'}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}

                            {Array.isArray(remediationPlan.recommendations) && remediationPlan.recommendations.length > 0 && (
                                <div className="eval-remediation-recommendations">
                                    {remediationPlan.recommendations.map((recommendation, idx) => {
                                        const dataOps = remediationDataOps(recommendation);
                                        const trainingChanges = remediationTrainingChanges(recommendation);
                                        return (
                                            <div
                                                key={`${recommendation.recommendation_id || recommendation.title || 'recommendation'}-${idx}`}
                                                className="eval-remediation-recommendation-card"
                                            >
                                                <div className="eval-remediation-recommendation-header">
                                                    <strong>{recommendation.title || 'Remediation action'}</strong>
                                                    <span className="badge badge-info">
                                                        {String(recommendation.root_cause || 'coverage_gap').replace(/_/g, ' ')}
                                                    </span>
                                                </div>
                                                {typeof recommendation.confidence === 'number' && (
                                                    <div className="eval-pack-note">
                                                        Confidence {(recommendation.confidence * 100).toFixed(0)}%
                                                    </div>
                                                )}
                                                {recommendation.rationale && (
                                                    <div className="eval-remediation-rationale">{recommendation.rationale}</div>
                                                )}
                                                {dataOps.length > 0 && (
                                                    <div>
                                                        <div className="eval-remediation-list-title">Data Operations</div>
                                                        <ul className="eval-remediation-list">
                                                            {dataOps.map((item) => <li key={item}>{item}</li>)}
                                                        </ul>
                                                    </div>
                                                )}
                                                {trainingChanges.length > 0 && (
                                                    <div>
                                                        <div className="eval-remediation-list-title">Training Config Changes</div>
                                                        <ul className="eval-remediation-list">
                                                            {trainingChanges.map((item) => <li key={item}>{item}</li>)}
                                                        </ul>
                                                    </div>
                                                )}
                                                {recommendation.expected_impact && (
                                                    <div className="eval-pack-note">
                                                        Expected impact:
                                                        {' '}
                                                        {recommendation.expected_impact.metric || 'score'}
                                                        {typeof recommendation.expected_impact.estimated_delta === 'number'
                                                            ? ` +${recommendation.expected_impact.estimated_delta.toFixed(3)}`
                                                            : ''}
                                                        {recommendation.expected_impact.horizon
                                                            ? ` (${recommendation.expected_impact.horizon})`
                                                            : ''}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
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
                                    <option value="local_serve">Local Serve Runtime</option>
                                    <option value="hf">HuggingFace (vLLM)</option>
                                    <option value="ollama">Local (Ollama)</option>
                                    <option value="openai">Cloud (OpenAI)</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">
                                    {provider === 'local_serve' ? 'Judge Model Alias (served)' : 'Judge Model'}
                                </label>
                                <input
                                    className="input"
                                    value={judgeModel}
                                    onChange={(e) => setJudgeModel(e.target.value)}
                                    placeholder={
                                        provider === 'openai'
                                            ? 'gpt-4o'
                                            : provider === 'ollama'
                                              ? 'llama3'
                                              : provider === 'local_serve'
                                                ? 'local-judge'
                                                : 'meta-llama/...'
                                    }
                                />
                            </div>
                        </div>
                    )}

                    {evalType === 'llm_judge' && provider === 'local_serve' && (
                        <div className="eval-pack-card" style={{ marginTop: 'var(--space-md)' }}>
                            <div className="eval-pack-header">
                                <div>
                                    <h4 style={{ margin: 0, fontSize: 'var(--font-size-sm)' }}>Local Judge Serve Runtime</h4>
                                    <p className="eval-pack-subtitle">
                                        Select an existing serve run or auto-start one for judge evaluation.
                                    </p>
                                </div>
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => void loadLocalServeRuns()}
                                    disabled={loadingLocalServeRuns}
                                >
                                    {loadingLocalServeRuns ? 'Refreshing...' : 'Refresh Runs'}
                                </button>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
                                <div className="form-group">
                                    <label className="form-label">Serve Run</label>
                                    <select
                                        className="input"
                                        value={selectedLocalServeRunId}
                                        onChange={(e) => setSelectedLocalServeRunId(e.target.value)}
                                    >
                                        <option value="auto">Auto (latest compatible run)</option>
                                        {localServeRuns.map((run) => (
                                            <option key={run.run_id} value={run.run_id}>
                                                {run.run_id} • {run.template_id || run.template_name || 'template'} • {run.status || 'unknown'}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Auto Start Missing Run</label>
                                    <select
                                        className="input"
                                        value={localServeAutoStart ? 'yes' : 'no'}
                                        onChange={(e) => setLocalServeAutoStart(e.target.value === 'yes')}
                                    >
                                        <option value="no">No</option>
                                        <option value="yes">Yes</option>
                                    </select>
                                </div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
                                <div className="form-group">
                                    <label className="form-label">Auto Stop After Eval</label>
                                    <select
                                        className="input"
                                        value={localServeAutoStop ? 'yes' : 'no'}
                                        onChange={(e) => setLocalServeAutoStop(e.target.value === 'yes')}
                                        disabled={!localServeAutoStart}
                                    >
                                        <option value="yes">Yes</option>
                                        <option value="no">No</option>
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Auto Start Source</label>
                                    <select
                                        className="input"
                                        value={localServeSource}
                                        onChange={(e) => setLocalServeSource(e.target.value as 'export' | 'registry')}
                                        disabled={!localServeAutoStart}
                                    >
                                        <option value="export">Export</option>
                                        <option value="registry">Registry</option>
                                    </select>
                                </div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
                                {localServeSource === 'export' ? (
                                    <div className="form-group">
                                        <label className="form-label">Export ID (auto-start)</label>
                                        <input
                                            className="input"
                                            value={localServeExportId}
                                            onChange={(e) => setLocalServeExportId(e.target.value)}
                                            disabled={!localServeAutoStart}
                                            placeholder="e.g. 12"
                                        />
                                    </div>
                                ) : (
                                    <div className="form-group">
                                        <label className="form-label">Registry Model ID (auto-start)</label>
                                        <input
                                            className="input"
                                            value={localServeModelId}
                                            onChange={(e) => setLocalServeModelId(e.target.value)}
                                            disabled={!localServeAutoStart}
                                            placeholder="e.g. 7"
                                        />
                                    </div>
                                )}
                                <div className="form-group">
                                    <label className="form-label">Template ID (optional)</label>
                                    <input
                                        className="input"
                                        value={localServeTemplateId}
                                        onChange={(e) => setLocalServeTemplateId(e.target.value)}
                                        disabled={!localServeAutoStart}
                                        placeholder="runner.vllm"
                                    />
                                </div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
                                <div className="form-group">
                                    <label className="form-label">Host (auto-start)</label>
                                    <input
                                        className="input"
                                        value={localServeHost}
                                        onChange={(e) => setLocalServeHost(e.target.value)}
                                        disabled={!localServeAutoStart}
                                        placeholder="127.0.0.1"
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Port (auto-start)</label>
                                    <input
                                        className="input"
                                        value={localServePort}
                                        onChange={(e) => setLocalServePort(e.target.value)}
                                        disabled={!localServeAutoStart}
                                        placeholder="8000"
                                    />
                                </div>
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

            {selectedExp && (
                <ScorecardPanel projectId={projectId} experimentId={selectedExp} />
            )}

            {selectedExp && evalResults.length > 0 && (
                <FailureClustersPanel
                    projectId={projectId}
                    evalResults={evalResults.map((r) => ({
                        id: r.id,
                        dataset_name: r.dataset_name,
                        eval_type: r.eval_type,
                        pass_rate: r.pass_rate,
                    }))}
                />
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
            </>)}
        </div>
    );
}
