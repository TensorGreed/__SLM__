import React, { useEffect, useState } from 'react';
import api from '../../api/client';
import { TerminalConsole } from '../shared/TerminalConsole';
import './ExportPanel.css';

interface ExportPanelProps { projectId: number; }

interface ExperimentOption {
    id: number;
    name: string;
    status: string;
}

interface ExportRecord {
    id: number;
    export_format?: string;
    format?: string;
    quantization?: string | null;
    status: string;
    output_path?: string;
    file_size_bytes?: number | null;
    manifest?: Record<string, unknown>;
    created_at?: string;
    completed_at?: string | null;
}

interface DeploymentTargetSpec {
    target_id: string;
    kind: string;
    display_name: string;
    description: string;
    compatible: boolean;
    smoke_supported: boolean;
}

interface DeploymentTargetCatalogResponse {
    default_target_ids: string[];
    targets: DeploymentTargetSpec[];
}

interface ExportRunResponse {
    id: number;
    status: string;
    output_path?: string;
    file_size_bytes?: number | null;
    manifest?: Record<string, unknown>;
    deployment?: {
        summary?: {
            deployable_artifact?: boolean;
            local_smoke_passed?: boolean | null;
        };
    };
}

interface RegistryModel {
    id: number;
    experiment_id: number;
    name: string;
    version: string;
    stage: string;
    deployment_status: string;
    readiness?: {
        metrics?: {
            exact_match?: number | null;
            f1?: number | null;
            llm_judge_pass_rate?: number | null;
            safety_pass_rate?: number | null;
        };
    };
    deployment?: {
        endpoint_url?: string;
        environment?: string;
    };
}

interface RegistryListResponse {
    models: RegistryModel[];
}

interface ServeTemplate {
    template_id: string;
    display_name: string;
    description?: string;
    command?: string;
    setup_commands?: string[];
    healthcheck?: {
        url?: string;
        curl?: string;
    };
    smoke_test?: {
        prompt?: string;
        curl?: string;
    };
    first_token_probe?: {
        url?: string;
        curl?: string;
    };
    notes?: string[];
}

interface ServePlanResponse {
    source: string;
    export_id?: number;
    model_id?: number;
    model_name?: string;
    model_stage?: string;
    export_format?: string;
    host?: string;
    port?: number;
    run_dir?: string;
    templates?: ServeTemplate[];
}

interface ServeRunStatusResponse {
    run_id: string;
    source: string;
    export_id?: number | null;
    model_id?: number | null;
    template_id: string;
    template_name?: string;
    status: string;
    can_stop?: boolean;
    pid?: number | null;
    return_code?: number | null;
    command?: string;
    healthcheck_curl?: string | null;
    smoke_curl?: string | null;
    first_token_curl?: string | null;
    logs_tail?: string[];
    telemetry?: {
        path?: string | null;
        first_healthy_at?: string | null;
        startup_latency_ms?: number | null;
        smoke_passed?: boolean | null;
        first_token_at?: string | null;
        first_token_latency_ms?: number | null;
        throughput_tokens_per_sec?: number | null;
        throughput_token_count_estimate?: number | null;
        throughput_window_ms?: number | null;
        first_token_passed?: boolean | null;
        health_checks?: Array<{ ok?: boolean; status_code?: number | null; error?: string }>;
        smoke_checks?: Array<{ ok?: boolean; status_code?: number | null; error?: string }>;
        first_token_checks?: Array<{ ok?: boolean; status_code?: number | null; error?: string }>;
    };
}

interface DeployPlanResponse {
    target_id: string;
    target_kind: string;
    display_name?: string;
    summary?: string;
    endpoint_name?: string;
    curl_example?: string;
    steps?: string[];
    sdk_artifact?: {
        zip_path?: string;
        readme_path?: string;
        entrypoint_path?: string;
        runtime_path?: string;
        bundle_files?: string[];
        model_placement_paths?: string[];
        run_commands?: string[];
        smoke_commands?: string[];
        smoke_validation?: {
            smoke_passed?: boolean;
            errors?: string[];
            warnings?: string[];
            checks?: Array<{
                check_id?: string;
                status?: string;
                message?: string;
            }>;
        };
    };
    run_commands?: string[];
    smoke_commands?: string[];
}

interface DeployExecutionDetails {
    status?: string;
    message?: string;
    provider?: string;
    dry_run?: boolean;
    started_at?: string;
    finished_at?: string;
    http_status?: number;
    request?: Record<string, unknown> | null;
    response?: Record<string, unknown> | null;
}

interface DeployExecuteResponse extends DeployPlanResponse {
    execution?: DeployExecutionDetails;
}

interface OptimizationMetric {
    latency_ms: number;
    memory_gb: number;
    quality_score: number;
}

type OptimizationMetricKey = keyof OptimizationMetric;
type MetricSource = 'measured' | 'estimated' | 'mixed' | 'simulated';

interface OptimizationCandidate {
    id: string;
    name: string;
    quantization: string;
    runtime_template: string;
    metrics: OptimizationMetric;
    is_recommended: boolean;
    reasons: string[];
    metric_source?: string;
    metric_sources?: Partial<Record<OptimizationMetricKey, string>>;
    measurement?: {
        mode?: string;
        fallback_reason?: string;
        [key: string]: unknown;
    } | null;
}

interface OptimizationResponse {
    project_id: number;
    target_id: string;
    candidates: OptimizationCandidate[];
}

interface OptimizationMatrixCandidate {
    rank?: number;
    rank_score?: number[];
    id: string;
    name: string;
    quantization: string;
    runtime_template: string;
    artifact_identifier?: string;
    metrics: OptimizationMetric;
    metric_source?: string;
    metric_sources?: Partial<Record<OptimizationMetricKey, string>>;
    measurement?: {
        mode?: string;
        fallback_reason?: string;
        remediation_hint?: string;
        [key: string]: unknown;
    } | null;
    confidence?: {
        score?: number;
        level?: string;
        [key: string]: unknown;
    } | null;
    is_recommended?: boolean;
    reasons?: string[];
}

interface OptimizationMatrixTargetResult {
    target_id: string;
    target_device?: string;
    candidate_count?: number;
    measured_candidate_count?: number;
    mixed_candidate_count?: number;
    estimated_candidate_count?: number;
    recommended_candidate_id?: string | null;
    candidates?: OptimizationMatrixCandidate[];
}

interface OptimizationMatrixRunResponse {
    run_id: string;
    project_id: number;
    status: string;
    run_hash?: string | null;
    started_at?: string | null;
    completed_at?: string | null;
    target_ids: string[];
    prompt_set?: {
        prompt_set_id?: string;
        prompt_set_hash?: string;
        [key: string]: unknown;
    };
    runtime_fingerprint?: {
        cpu?: string;
        gpu?: string;
        toolchain?: string;
        python_version?: string;
        [key: string]: unknown;
    };
    summary?: {
        target_count?: number;
        candidate_evaluation_count?: number;
        measured_candidate_count?: number;
        mixed_candidate_count?: number;
        estimated_candidate_count?: number;
        max_probe_candidates_per_target?: number;
        [key: string]: unknown;
    };
    targets?: OptimizationMatrixTargetResult[];
    error?: string | null;
}

interface OptimizationMatrixRecommendationsResponse {
    project_id: number;
    run_id: string;
    status: string;
    target_id?: string | null;
    recommendations_by_target: Array<{
        target_id: string;
        target_device?: string;
        recommended_candidate_id?: string | null;
        recommendations: OptimizationMatrixCandidate[];
    }>;
}

function toErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const detail = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Operation failed';
}

const METRIC_SOURCE_BADGE_CLASS: Record<MetricSource, string> = {
    measured: 'badge-success',
    mixed: 'badge-info',
    estimated: 'badge-warning',
    simulated: 'badge-warning',
};

const METRIC_SOURCE_LABEL: Record<MetricSource, string> = {
    measured: 'Measured',
    mixed: 'Mixed Source',
    estimated: 'Estimated',
    simulated: 'Simulated',
};

const METRIC_SOURCE_SHORT_LABEL: Record<MetricSource, string> = {
    measured: 'Measured',
    mixed: 'Mixed',
    estimated: 'Estimated',
    simulated: 'Simulated',
};

function normalizeMetricSource(value: unknown): MetricSource | null {
    const token = String(value || '').trim().toLowerCase();
    if (!token) return null;
    if (token === 'measured') return 'measured';
    if (token === 'mixed') return 'mixed';
    if (token === 'simulated') return 'simulated';
    return 'estimated';
}

function candidateMetricSource(candidate: OptimizationCandidate | OptimizationMatrixCandidate): MetricSource {
    const explicit = normalizeMetricSource(candidate.metric_source);
    if (explicit) return explicit;
    const modeSource = normalizeMetricSource(candidate.measurement?.mode);
    if (modeSource) return modeSource;
    return 'estimated';
}

function metricSourceForKey(candidate: OptimizationCandidate | OptimizationMatrixCandidate, key: OptimizationMetricKey): MetricSource {
    const perMetric = normalizeMetricSource(candidate.metric_sources?.[key]);
    if (perMetric) return perMetric;
    const aggregate = candidateMetricSource(candidate);
    if (aggregate === 'mixed') return 'estimated';
    return aggregate;
}

function metricSourceHelpText(source: MetricSource): string {
    if (source === 'measured') {
        return 'Measured from real local benchmark probes on project artifacts.';
    }
    if (source === 'mixed') {
        return 'Combination of measured and estimated values.';
    }
    if (source === 'simulated') {
        return 'Simulated numbers for planning only; not measured on live artifacts.';
    }
    return 'Estimated from artifact profile/heuristics because probe data is unavailable.';
}

function candidateProvenanceText(candidate: OptimizationCandidate | OptimizationMatrixCandidate): string {
    const source = candidateMetricSource(candidate);
    const fallbackReason = String(candidate.measurement?.fallback_reason || '').trim();
    if (source === 'measured') {
        return 'All candidate metrics are measured from benchmark probe runs.';
    }
    if (source === 'mixed') {
        return fallbackReason
            ? `This card mixes measured and estimated values. ${fallbackReason}`
            : 'This card mixes measured and estimated values. Check per-metric tags for details.';
    }
    if (source === 'simulated') {
        return fallbackReason
            ? `Metrics are simulated for planning only. ${fallbackReason}`
            : 'Metrics are simulated for planning only and may differ from runtime behavior.';
    }
    return fallbackReason
        ? `Metrics are estimated from artifact metadata/heuristics. ${fallbackReason}`
        : 'Metrics are estimated from artifact metadata/heuristics and not directly benchmarked.';
}

function formatOptimizationMetricValue(
    key: OptimizationMetricKey,
    value: number | null | undefined,
    source: MetricSource,
): string {
    if (typeof value !== 'number' || Number.isNaN(value)) return '—';
    const prefix = source === 'measured' ? '' : '~';
    if (key === 'latency_ms') {
        return `${prefix}${value.toFixed(1)}ms`;
    }
    if (key === 'memory_gb') {
        return `${prefix}${value.toFixed(2)}GB`;
    }
    return `${prefix}${(value * 100).toFixed(1)}%`;
}

export default function ExportPanel({ projectId }: ExportPanelProps) {
    const [experiments, setExperiments] = useState<ExperimentOption[]>([]);
    const [exportsList, setExportsList] = useState<ExportRecord[]>([]);
    const [registryModels, setRegistryModels] = useState<RegistryModel[]>([]);

    const [selectedExp, setSelectedExp] = useState('');
    const [format, setFormat] = useState('gguf');
    const [quantization, setQuantization] = useState('4-bit');
    const [runSmokeTests, setRunSmokeTests] = useState(true);

    const [deploymentTargets, setDeploymentTargets] = useState<DeploymentTargetSpec[]>([]);
    const [defaultDeploymentTargets, setDefaultDeploymentTargets] = useState<string[]>([]);
    const [selectedDeploymentTargets, setSelectedDeploymentTargets] = useState<string[]>([]);
    const [isLoadingTargets, setIsLoadingTargets] = useState(false);
    const [targetLoadError, setTargetLoadError] = useState('');

    const [isLoading, setIsLoading] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const [expandedIds, setExpandedIds] = useState<number[]>([]);
    const [copyState, setCopyState] = useState<Record<string, boolean>>({});
    const [servePlan, setServePlan] = useState<ServePlanResponse | null>(null);
    const [isLoadingServePlan, setIsLoadingServePlan] = useState(false);
    const [activeServeRun, setActiveServeRun] = useState<ServeRunStatusResponse | null>(null);
    const [isStartingServeRun, setIsStartingServeRun] = useState(false);
    const [isStoppingServeRun, setIsStoppingServeRun] = useState(false);
    const [deployTargetId, setDeployTargetId] = useState('deployment.hf_inference_endpoint');
    const [deployPlan, setDeployPlan] = useState<DeployPlanResponse | null>(null);
    const [isLoadingDeployPlan, setIsLoadingDeployPlan] = useState(false);
    const [deployPlanExportId, setDeployPlanExportId] = useState<number | null>(null);
    const [isExecutingDeployPlan, setIsExecutingDeployPlan] = useState(false);
    const [deployExecution, setDeployExecution] = useState<DeployExecuteResponse | null>(null);
    const [deployDryRun, setDeployDryRun] = useState(true);
    const [deployHfToken, setDeployHfToken] = useState('');
    const [deployManagedApiUrl, setDeployManagedApiUrl] = useState('');
    const [deployManagedApiToken, setDeployManagedApiToken] = useState('');

    const [optimizationResults, setOptimizationResults] = useState<OptimizationResponse | null>(null);
    const [selectedOptimization, setSelectedOptimization] = useState<OptimizationCandidate | null>(null);
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [matrixRun, setMatrixRun] = useState<OptimizationMatrixRunResponse | null>(null);
    const [isRunningMatrix, setIsRunningMatrix] = useState(false);
    const [isRefreshingMatrix, setIsRefreshingMatrix] = useState(false);
    const [matrixTargetId, setMatrixTargetId] = useState('');
    const [matrixTopK, setMatrixTopK] = useState(3);
    const [matrixRecommendations, setMatrixRecommendations] = useState<OptimizationMatrixRecommendationsResponse | null>(null);
    const [isLoadingMatrixRecommendations, setIsLoadingMatrixRecommendations] = useState(false);

    const refreshAll = async () => {
        setIsLoading(true);
        try {
            const [expRes, exportRes, registryRes] = await Promise.all([
                api.get<ExperimentOption[]>(`/projects/${projectId}/training/experiments`),
                api.get<ExportRecord[]>(`/projects/${projectId}/export/list`),
                api.get<RegistryListResponse>(`/projects/${projectId}/registry/models`),
            ]);
            setExperiments(expRes.data || []);
            setExportsList(exportRes.data || []);
            setRegistryModels(registryRes.data.models || []);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        setSelectedExp('');
        setExpandedIds([]);
        setErrorMessage('');
        setStatusMessage('');
        setTargetLoadError('');
        setServePlan(null);
        setActiveServeRun(null);
        setDeployPlan(null);
        setDeployPlanExportId(null);
        setDeployExecution(null);
        setDeployDryRun(true);
        setDeployHfToken('');
        setDeployManagedApiUrl('');
        setDeployManagedApiToken('');
        setOptimizationResults(null);
        setSelectedOptimization(null);
        setMatrixRun(null);
        setMatrixRecommendations(null);
        setMatrixTargetId('');
        setMatrixTopK(3);
        void refreshAll();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    useEffect(() => {
        const runId = String(activeServeRun?.run_id || '').trim();
        const status = String(activeServeRun?.status || '').trim().toLowerCase();
        if (!runId || !['pending', 'running', 'stopping'].includes(status)) {
            return;
        }

        const interval = window.setInterval(() => {
            void (async () => {
                try {
                    const res = await api.get<ServeRunStatusResponse>(
                        `/projects/${projectId}/export/serve-runs/${runId}`,
                        { params: { logs_tail: 500 } },
                    );
                    setActiveServeRun(res.data || null);
                } catch {
                    // no-op, keep last known status
                }
            })();
        }, 1500);
        return () => window.clearInterval(interval);
    }, [activeServeRun?.run_id, activeServeRun?.status, projectId]);

    useEffect(() => {
        const loadDeploymentTargets = async () => {
            setIsLoadingTargets(true);
            setTargetLoadError('');
            try {
                const res = await api.get<DeploymentTargetCatalogResponse>(
                    `/projects/${projectId}/export/deployment-targets`,
                    {
                        params: { export_format: format },
                    },
                );
                const compatibleTargets = (res.data.targets || []).filter((target) => target.compatible);
                const defaultTargetIds = (res.data.default_target_ids || []).filter((targetId) =>
                    compatibleTargets.some((target) => target.target_id === targetId),
                );
                setDeploymentTargets(compatibleTargets);
                setDefaultDeploymentTargets(defaultTargetIds);
                setSelectedDeploymentTargets(defaultTargetIds);
            } catch (err) {
                setDeploymentTargets([]);
                setDefaultDeploymentTargets([]);
                setSelectedDeploymentTargets([]);
                setTargetLoadError(toErrorMessage(err));
            } finally {
                setIsLoadingTargets(false);
            }
        };

        void loadDeploymentTargets();
    }, [format, projectId]);

    const toggleDeploymentTarget = (targetId: string) => {
        setSelectedDeploymentTargets((prev) => (
            prev.includes(targetId)
                ? prev.filter((item) => item !== targetId)
                : [...prev, targetId]
        ));
    };

    const handleCreateExport = async () => {
        if (!selectedExp) return;
        setIsExporting(true);
        setErrorMessage('');
        setStatusMessage('Creating export...');
        try {
            const createRes = await api.post(`/projects/${projectId}/export/create`, {
                experiment_id: Number(selectedExp),
                export_format: format,
                quantization,
            });
            const runRes = await api.post<ExportRunResponse>(`/projects/${projectId}/export/${createRes.data.id}/run`, {
                deployment_targets: selectedDeploymentTargets,
                run_smoke_tests: runSmokeTests,
                benchmark_report: selectedOptimization ? {
                    target_id: optimizationResults?.target_id,
                    candidate_id: selectedOptimization.id,
                    metrics: selectedOptimization.metrics,
                    reasons: selectedOptimization.reasons,
                    metric_source: selectedOptimization.metric_source,
                    metric_sources: selectedOptimization.metric_sources,
                    measurement: selectedOptimization.measurement,
                } : null
            });
            const runPayload: ExportRecord = {
                id: runRes.data.id,
                status: runRes.data.status,
                output_path: runRes.data.output_path,
                file_size_bytes: runRes.data.file_size_bytes,
                manifest: runRes.data.manifest,
                format,
                quantization,
                created_at: new Date().toISOString(),
            };
            setExportsList((prev) => [runPayload, ...prev]);
            const deployable = runRes.data.deployment?.summary?.deployable_artifact;
            const smokePassed = runRes.data.deployment?.summary?.local_smoke_passed;
            const summaryParts: string[] = ['Export run completed.'];
            if (typeof deployable === 'boolean') {
                summaryParts.push(`Artifact deployable: ${deployable ? 'yes' : 'no'}.`);
            }
            if (runSmokeTests && smokePassed !== undefined && smokePassed !== null) {
                summaryParts.push(`Smoke tests: ${smokePassed ? 'passed' : 'failed'}.`);
            }
            setStatusMessage(summaryParts.join(' '));
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        } finally {
            setIsExporting(false);
        }
    };

    const handleRegisterModel = async () => {
        if (!selectedExp) return;
        setErrorMessage('');
        setStatusMessage('Registering model...');
        try {
            await api.post(`/projects/${projectId}/registry/models/register`, {
                experiment_id: Number(selectedExp),
            });
            await refreshAll();
            setStatusMessage('Model registered in candidate stage.');
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        }
    };

    const handlePromote = async (modelId: number, targetStage: 'staging' | 'production') => {
        setErrorMessage('');
        setStatusMessage(`Promoting model ${modelId} to ${targetStage}...`);
        try {
            await api.post(`/projects/${projectId}/registry/models/${modelId}/promote`, {
                target_stage: targetStage,
                force: false,
            });
            await refreshAll();
            setStatusMessage(`Model ${modelId} promoted to ${targetStage}.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        }
    };

    const handleDeploy = async (modelId: number, environment: 'staging' | 'production') => {
        setErrorMessage('');
        setStatusMessage(`Marking model ${modelId} as deployed (${environment})...`);
        try {
            await api.post(`/projects/${projectId}/registry/models/${modelId}/deploy`, {
                environment,
            });
            await refreshAll();
            setStatusMessage(`Model ${modelId} marked deployed (${environment}).`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        }
    };

    const handleExportServePlan = async (exportId: number) => {
        setIsLoadingServePlan(true);
        setErrorMessage('');
        setStatusMessage(`Generating serve plan for export ${exportId}...`);
        try {
            const res = await api.post<ServePlanResponse>(
                `/projects/${projectId}/export/${exportId}/serve-plan`,
                {
                    host: '127.0.0.1',
                    port: 8080,
                    smoke_test_prompt: 'Hello from BrewSLM',
                },
            );
            setServePlan(res.data || null);
            setStatusMessage(`Serve plan generated for export ${exportId}.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        } finally {
            setIsLoadingServePlan(false);
        }
    };

    const handleBuildDeployPlan = async (exportId: number) => {
        setIsLoadingDeployPlan(true);
        setErrorMessage('');
        setStatusMessage(`Building deploy plan (${deployTargetId})...`);
        try {
            const res = await api.post<DeployPlanResponse>(
                `/projects/${projectId}/export/${exportId}/deploy-as-api`,
                {
                    target_id: deployTargetId,
                },
            );
            setDeployPlan(res.data || null);
            setDeployPlanExportId(exportId);
            setDeployExecution(null);
            setStatusMessage(`Deploy plan generated for export ${exportId}.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        } finally {
            setIsLoadingDeployPlan(false);
        }
    };

    const handleExecuteDeployPlan = async () => {
        if (!deployPlan || !deployPlanExportId) return;
        setIsExecutingDeployPlan(true);
        setErrorMessage('');
        setStatusMessage(
            `${deployDryRun ? 'Running deploy dry-run' : 'Executing deploy action'} (${deployPlan.target_id})...`,
        );
        try {
            const payload: Record<string, unknown> = {
                target_id: deployPlan.target_id,
                dry_run: deployDryRun,
            };
            if (!deployDryRun && deployPlan.target_id === 'deployment.hf_inference_endpoint' && deployHfToken.trim()) {
                payload.hf_token = deployHfToken.trim();
            }
            if (!deployDryRun && deployPlan.target_id === 'deployment.vllm_managed') {
                if (deployManagedApiUrl.trim()) {
                    payload.managed_api_url = deployManagedApiUrl.trim();
                }
                if (deployManagedApiToken.trim()) {
                    payload.managed_api_token = deployManagedApiToken.trim();
                }
            }
            const res = await api.post<DeployExecuteResponse>(
                `/projects/${projectId}/export/${deployPlanExportId}/deploy-as-api/execute`,
                payload,
            );
            setDeployExecution(res.data || null);
            setStatusMessage(
                deployDryRun
                    ? `Dry-run completed for ${deployPlan.target_id}.`
                    : `Deployment request submitted for ${deployPlan.target_id}.`,
            );
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        } finally {
            setIsExecutingDeployPlan(false);
        }
    };

    const handleOptimize = async () => {
        if (selectedDeploymentTargets.length === 0) {
            setErrorMessage('Please select at least one deployment target to optimize for.');
            return;
        }
        setIsOptimizing(true);
        setErrorMessage('');
        setOptimizationResults(null);
        try {
            // Use the first selected target for optimization search
            const targetId = selectedDeploymentTargets[0];
            const res = await api.post<OptimizationResponse>(`/projects/${projectId}/export/optimize`, {
                target_id: targetId,
            });
            setOptimizationResults(res.data);
            setStatusMessage(`Optimization search completed for ${targetId}.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
        } finally {
            setIsOptimizing(false);
        }
    };

    const handleStartOptimizationMatrix = async () => {
        const targets = selectedDeploymentTargets.length > 0 ? selectedDeploymentTargets : deploymentTargets.map((item) => item.target_id);
        if (targets.length === 0) {
            setErrorMessage('Select at least one deployment target before running matrix benchmark.');
            return;
        }
        setIsRunningMatrix(true);
        setErrorMessage('');
        setMatrixRecommendations(null);
        try {
            const res = await api.post<OptimizationMatrixRunResponse>(
                `/projects/${projectId}/export/optimize/matrix/start`,
                {
                    target_ids: targets,
                    max_probe_candidates_per_target: 3,
                },
            );
            const payload = res.data || null;
            setMatrixRun(payload);
            const firstTarget = String(payload?.targets?.[0]?.target_id || targets[0] || '').trim();
            setMatrixTargetId(firstTarget);
            setStatusMessage(`Optimization matrix run ${payload?.run_id || ''} ${payload?.status || 'started'}.`);
            if (payload?.run_id) {
                await handleLoadMatrixRecommendations(payload.run_id, firstTarget);
            }
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
        } finally {
            setIsRunningMatrix(false);
        }
    };

    const handleRefreshOptimizationMatrix = async () => {
        const runId = String(matrixRun?.run_id || '').trim();
        if (!runId) return;
        setIsRefreshingMatrix(true);
        setErrorMessage('');
        try {
            const res = await api.get<OptimizationMatrixRunResponse>(
                `/projects/${projectId}/export/optimize/matrix/${encodeURIComponent(runId)}`,
            );
            const payload = res.data || null;
            setMatrixRun(payload);
            setStatusMessage(`Refreshed matrix run ${runId}.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
        } finally {
            setIsRefreshingMatrix(false);
        }
    };

    const handleLoadMatrixRecommendations = async (runIdOverride?: string, targetOverride?: string) => {
        const runId = String(runIdOverride || matrixRun?.run_id || '').trim();
        if (!runId) return;
        const selectedTarget = String(targetOverride || matrixTargetId || '').trim();
        setIsLoadingMatrixRecommendations(true);
        setErrorMessage('');
        try {
            const res = await api.get<OptimizationMatrixRecommendationsResponse>(
                `/projects/${projectId}/export/optimize/matrix/${encodeURIComponent(runId)}/recommendations`,
                {
                    params: {
                        target_id: selectedTarget || undefined,
                        top_k: Math.max(1, Math.min(5, Math.floor(matrixTopK || 3))),
                    },
                },
            );
            setMatrixRecommendations(res.data || null);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
        } finally {
            setIsLoadingMatrixRecommendations(false);
        }
    };

    const applyOptimization = (candidate: OptimizationCandidate | OptimizationMatrixCandidate) => {
        setFormat(candidate.runtime_template);
        setQuantization(candidate.quantization);
        setSelectedOptimization({
            id: candidate.id,
            name: candidate.name,
            quantization: candidate.quantization,
            runtime_template: candidate.runtime_template,
            metrics: candidate.metrics,
            is_recommended: Boolean(candidate.is_recommended),
            reasons: Array.isArray(candidate.reasons) ? candidate.reasons : [],
            metric_source: candidate.metric_source,
            metric_sources: candidate.metric_sources,
            measurement: candidate.measurement,
        });
        setStatusMessage(`Applied ${candidate.name} settings.`);
    };

    const handleRegistryServePlan = async (modelId: number) => {
        setIsLoadingServePlan(true);
        setErrorMessage('');
        setStatusMessage(`Generating serve plan for model ${modelId}...`);
        try {
            const res = await api.post<ServePlanResponse>(
                `/projects/${projectId}/registry/models/${modelId}/serve-plan`,
                {
                    host: '127.0.0.1',
                    port: 8080,
                    smoke_test_prompt: 'Hello from BrewSLM',
                },
            );
            setServePlan(res.data || null);
            setActiveServeRun(null);
            setStatusMessage(`Serve plan generated for model ${modelId}.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        } finally {
            setIsLoadingServePlan(false);
        }
    };

    const handleStartServeRun = async (templateId: string) => {
        if (!servePlan) return;
        setIsStartingServeRun(true);
        setErrorMessage('');
        setStatusMessage(`Starting serve run (${templateId})...`);
        try {
            const payload = {
                template_id: templateId,
                host: String(servePlan.host || '127.0.0.1'),
                port: Number(servePlan.port || 8080),
                smoke_test_prompt: 'Hello from BrewSLM',
            };
            let res;
            if (servePlan.source === 'registry' && servePlan.model_id) {
                res = await api.post<ServeRunStatusResponse>(
                    `/projects/${projectId}/registry/models/${servePlan.model_id}/serve-runs/start`,
                    payload,
                );
            } else if (servePlan.export_id) {
                res = await api.post<ServeRunStatusResponse>(
                    `/projects/${projectId}/export/${servePlan.export_id}/serve-runs/start`,
                    payload,
                );
            } else {
                throw new Error('Serve plan missing source identifier.');
            }
            setActiveServeRun(res.data || null);
            setStatusMessage(`Serve run ${res.data.run_id} started.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        } finally {
            setIsStartingServeRun(false);
        }
    };

    const handleStopServeRun = async () => {
        const runId = String(activeServeRun?.run_id || '').trim();
        if (!runId) return;
        setIsStoppingServeRun(true);
        setErrorMessage('');
        try {
            const res = await api.post<ServeRunStatusResponse>(
                `/projects/${projectId}/export/serve-runs/${runId}/stop`,
            );
            setActiveServeRun(res.data || null);
            setStatusMessage(`Stop requested for serve run ${runId}.`);
        } catch (err) {
            setErrorMessage(toErrorMessage(err));
            setStatusMessage('');
        } finally {
            setIsStoppingServeRun(false);
        }
    };

    const toggleExpand = (id: number) => {
        setExpandedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
    };

    const handleCopy = (key: string, text: string) => {
        navigator.clipboard.writeText(text);
        setCopyState((prev) => ({ ...prev, [key]: true }));
        setTimeout(() => setCopyState((prev) => ({ ...prev, [key]: false })), 1500);
    };

    const statusColor = (status: string) =>
        status === 'completed'
            ? 'badge-success'
            : status === 'in_progress' || status === 'running'
                ? 'badge-info'
                : status === 'failed'
                    ? 'badge-error'
                    : 'badge-warning';

    const stageColor = (stage: string) =>
        stage === 'production'
            ? 'badge-success'
            : stage === 'staging'
                ? 'badge-info'
                : stage === 'archived'
                    ? 'badge-warning'
                    : 'badge-accent';

    const runStatusColor = (status: string) =>
        status === 'completed'
            ? 'badge-success'
            : status === 'running'
                ? 'badge-info'
                : status === 'failed'
                    ? 'badge-error'
                    : status === 'cancelled'
                        ? 'badge-warning'
                        : 'badge-warning';

    const formatBytes = (bytes?: number | null) => {
        if (!bytes) return '—';
        if (bytes === 0) return '0 B';
        const k = 1024;
        const units = ['B', 'KB', 'MB', 'GB'];
        const idx = Math.floor(Math.log(bytes) / Math.log(k));
        return `${(bytes / Math.pow(k, idx)).toFixed(2)} ${units[idx]}`;
    };

    const fmtMetric = (value?: number | null) => (value == null ? '—' : `${(value * 100).toFixed(1)}%`);
    const fmtTimestamp = (value?: string | null) => {
        if (!value) return '—';
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) return value;
        return parsed.toLocaleTimeString();
    };

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-lg)' }}>Export and Registry</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 'var(--space-md)', marginBottom: 'var(--space-lg)' }}>
                    <div className="form-group">
                        <label className="form-label">Experiment</label>
                        <select className="input" value={selectedExp} onChange={(e) => setSelectedExp(e.target.value)}>
                            <option value="">Select...</option>
                            {experiments.map((exp) => (
                                <option key={exp.id} value={exp.id}>
                                    {exp.name} ({exp.status})
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Format</label>
                        <select className="input" value={format} onChange={(e) => setFormat(e.target.value)}>
                            <option value="gguf">GGUF (CPU)</option>
                            <option value="onnx">ONNX</option>
                            <option value="huggingface">HuggingFace</option>
                            <option value="tensorrt">TensorRT-LLM</option>
                            <option value="docker">Docker</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Quantization</label>
                        <select className="input" value={quantization} onChange={(e) => setQuantization(e.target.value)}>
                            <option value="none">None</option>
                            <option value="4-bit">4-bit</option>
                            <option value="8-bit">8-bit</option>
                        </select>
                    </div>
                </div>
                <div className="export-target-panel">
                    <div className="export-target-panel-header">
                        <div>
                            <div className="export-target-panel-title">Deployment Targets</div>
                            <div className="export-target-panel-subtitle">
                                {selectedDeploymentTargets.length} selected
                            </div>
                        </div>
                        <button
                            className="btn btn-secondary btn-sm"
                            type="button"
                            disabled={isLoadingTargets || defaultDeploymentTargets.length === 0}
                            onClick={() => setSelectedDeploymentTargets(defaultDeploymentTargets)}
                        >
                            Use Recommended
                        </button>
                    </div>
                    {isLoadingTargets && (
                        <div className="export-target-panel-note">Loading deployment target catalog...</div>
                    )}
                    {!isLoadingTargets && targetLoadError && (
                        <div className="export-target-panel-error">
                            Failed to load target catalog: {targetLoadError}. Export will use server defaults.
                        </div>
                    )}
                    {!isLoadingTargets && !targetLoadError && deploymentTargets.length === 0 && (
                        <div className="export-target-panel-note">
                            No compatible deployment targets available for this format.
                        </div>
                    )}
                    {!isLoadingTargets && !targetLoadError && deploymentTargets.length > 0 && (
                        <div className="export-target-grid">
                            {deploymentTargets.map((target) => {
                                const checked = selectedDeploymentTargets.includes(target.target_id);
                                return (
                                    <label
                                        key={target.target_id}
                                        className={`export-target-option ${checked ? 'selected' : ''}`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={checked}
                                            onChange={() => toggleDeploymentTarget(target.target_id)}
                                        />
                                        <div>
                                            <div className="export-target-name">
                                                {target.display_name}
                                                <span className="export-target-kind">{target.kind}</span>
                                            </div>
                                            <div className="export-target-description">{target.description}</div>
                                        </div>
                                    </label>
                                );
                            })}
                        </div>
                    )}
                    <label className="export-smoke-toggle">
                        <input
                            type="checkbox"
                            checked={runSmokeTests}
                            onChange={(e) => setRunSmokeTests(e.target.checked)}
                        />
                        <span>Run local smoke tests for selected runner targets</span>
                    </label>
                </div>

                {optimizationResults && (
                    <div className="optimization-results">
                        <div className="export-target-panel-header">
                            <div>
                                <div className="export-target-panel-title">Optimization Tradeoff Cards</div>
                                <div className="export-target-panel-subtitle">
                                    Results for {optimizationResults.target_id}
                                </div>
                            </div>
                            <button
                                className="btn btn-secondary btn-sm"
                                onClick={() => setOptimizationResults(null)}
                            >
                                Clear
                            </button>
                        </div>
                        <div className="optimization-candidates-grid">
                            {optimizationResults.candidates.map((c) => {
                                const aggregateSource = candidateMetricSource(c);
                                const aggregateHelp = candidateProvenanceText(c);
                                const latencySource = metricSourceForKey(c, 'latency_ms');
                                const memorySource = metricSourceForKey(c, 'memory_gb');
                                const qualitySource = metricSourceForKey(c, 'quality_score');
                                const sourceClassName = (source: MetricSource) => {
                                    if (source === 'measured') return 'is-measured';
                                    if (source === 'mixed') return 'is-mixed';
                                    if (source === 'simulated') return 'is-simulated';
                                    return 'is-estimated';
                                };
                                return (
                                    <div
                                        key={c.id}
                                        className={`optimization-card ${c.is_recommended ? 'recommended' : ''}`}
                                    >
                                        {c.is_recommended && (
                                            <div className="recommendation-badge">RECOMMENDED</div>
                                        )}
                                        <div className="optimization-card-header">
                                            <div className="serve-template-card__title">{c.name}</div>
                                            <span
                                                className={`badge ${METRIC_SOURCE_BADGE_CLASS[aggregateSource]} optimization-source-badge`}
                                                title={aggregateHelp}
                                                aria-label={`${METRIC_SOURCE_LABEL[aggregateSource]} metrics`}
                                            >
                                                {METRIC_SOURCE_LABEL[aggregateSource]}
                                            </span>
                                        </div>
                                        <div className="optimization-metrics">
                                            <div className="optimization-metric-item">
                                                <span className="optimization-metric-label">Latency</span>
                                                <span className="optimization-metric-value">
                                                    {formatOptimizationMetricValue('latency_ms', c.metrics.latency_ms, latencySource)}
                                                </span>
                                                <span
                                                    className={`optimization-metric-source ${sourceClassName(latencySource)}`}
                                                    title={metricSourceHelpText(latencySource)}
                                                >
                                                    {METRIC_SOURCE_SHORT_LABEL[latencySource]}
                                                </span>
                                            </div>
                                            <div className="optimization-metric-item">
                                                <span className="optimization-metric-label">Memory</span>
                                                <span className="optimization-metric-value">
                                                    {formatOptimizationMetricValue('memory_gb', c.metrics.memory_gb, memorySource)}
                                                </span>
                                                <span
                                                    className={`optimization-metric-source ${sourceClassName(memorySource)}`}
                                                    title={metricSourceHelpText(memorySource)}
                                                >
                                                    {METRIC_SOURCE_SHORT_LABEL[memorySource]}
                                                </span>
                                            </div>
                                            <div className="optimization-metric-item">
                                                <span className="optimization-metric-label">Quality</span>
                                                <span className="optimization-metric-value">
                                                    {formatOptimizationMetricValue('quality_score', c.metrics.quality_score, qualitySource)}
                                                </span>
                                                <span
                                                    className={`optimization-metric-source ${sourceClassName(qualitySource)}`}
                                                    title={metricSourceHelpText(qualitySource)}
                                                >
                                                    {METRIC_SOURCE_SHORT_LABEL[qualitySource]}
                                                </span>
                                            </div>
                                        </div>
                                        {aggregateSource !== 'measured' && (
                                            <div className="optimization-provenance-note" title={aggregateHelp}>
                                                {aggregateHelp}
                                            </div>
                                        )}
                                        {c.reasons.length > 0 && (
                                            <div className="optimization-reasons">
                                                <ul>
                                                    {c.reasons.map((r, i) => (
                                                        <li key={i}>{r}</li>
                                                    ))}
                                                </ul>
                                            </div>
                                        )}
                                        <button
                                            className="btn btn-secondary btn-sm"
                                            style={{ marginTop: 'auto' }}
                                            onClick={() => applyOptimization(c)}
                                        >
                                            Apply Settings
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}

                {matrixRun && (
                    <div className="optimization-results optimization-matrix-results">
                        <div className="export-target-panel-header">
                            <div>
                                <div className="export-target-panel-title">Benchmark Matrix Run</div>
                                <div className="export-target-panel-subtitle">
                                    Run {matrixRun.run_id} • {matrixRun.status}
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: 8 }}>
                                <span className={`badge ${runStatusColor(matrixRun.status)}`}>{matrixRun.status}</span>
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => void handleRefreshOptimizationMatrix()}
                                    disabled={isRefreshingMatrix}
                                >
                                    {isRefreshingMatrix ? 'Refreshing...' : 'Refresh'}
                                </button>
                            </div>
                        </div>

                        <div className="optimization-matrix-summary">
                            <div className="optimization-matrix-chip">
                                <span>Targets</span>
                                <strong>{Number(matrixRun.summary?.target_count || matrixRun.targets?.length || 0)}</strong>
                            </div>
                            <div className="optimization-matrix-chip">
                                <span>Candidates</span>
                                <strong>{Number(matrixRun.summary?.candidate_evaluation_count || 0)}</strong>
                            </div>
                            <div className="optimization-matrix-chip">
                                <span>Measured</span>
                                <strong>{Number(matrixRun.summary?.measured_candidate_count || 0)}</strong>
                            </div>
                            <div className="optimization-matrix-chip">
                                <span>Estimated</span>
                                <strong>{Number(matrixRun.summary?.estimated_candidate_count || 0)}</strong>
                            </div>
                            <div className="optimization-matrix-chip">
                                <span>Runtime Fingerprint</span>
                                <strong>
                                    {String(
                                        matrixRun.runtime_fingerprint?.gpu
                                        || matrixRun.runtime_fingerprint?.cpu
                                        || matrixRun.runtime_fingerprint?.toolchain
                                        || 'unknown',
                                    )}
                                </strong>
                            </div>
                        </div>

                        <div className="optimization-matrix-controls">
                            <div className="form-group">
                                <label className="form-label">Target</label>
                                <select
                                    className="input"
                                    value={matrixTargetId}
                                    onChange={(e) => setMatrixTargetId(e.target.value)}
                                >
                                    {(matrixRun.targets || []).map((target) => (
                                        <option key={target.target_id} value={target.target_id}>
                                            {target.target_id}
                                            {target.target_device ? ` (${target.target_device})` : ''}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Top K</label>
                                <input
                                    className="input"
                                    type="number"
                                    min={1}
                                    max={5}
                                    value={matrixTopK}
                                    onChange={(e) => setMatrixTopK(Number(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Recommendations</label>
                                <button
                                    className="btn btn-primary"
                                    onClick={() => void handleLoadMatrixRecommendations()}
                                    disabled={isLoadingMatrixRecommendations}
                                >
                                    {isLoadingMatrixRecommendations ? 'Loading...' : 'Load Recommendations'}
                                </button>
                            </div>
                        </div>

                        {(matrixRun.targets || []).length > 0 && (
                            <div className="table-container" style={{ maxHeight: 240 }}>
                                <table className="docs-table">
                                    <thead>
                                        <tr>
                                            <th>Target</th>
                                            <th>Candidates</th>
                                            <th>Measured</th>
                                            <th>Mixed</th>
                                            <th>Estimated</th>
                                            <th>Recommended</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {(matrixRun.targets || []).map((target) => (
                                            <tr key={target.target_id}>
                                                <td>{target.target_id}</td>
                                                <td>{Number(target.candidate_count || 0)}</td>
                                                <td>{Number(target.measured_candidate_count || 0)}</td>
                                                <td>{Number(target.mixed_candidate_count || 0)}</td>
                                                <td>{Number(target.estimated_candidate_count || 0)}</td>
                                                <td>{target.recommended_candidate_id || '—'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}

                        {matrixRecommendations && (
                            <div className="optimization-matrix-recommendations">
                                {matrixRecommendations.recommendations_by_target.map((row) => (
                                    <div key={row.target_id} className="optimization-matrix-recommendation-target">
                                        <div className="export-target-panel-header">
                                            <div>
                                                <div className="export-target-panel-title">
                                                    Recommendations • {row.target_id}
                                                </div>
                                                <div className="export-target-panel-subtitle">
                                                    {row.target_device || 'unknown device'}
                                                </div>
                                            </div>
                                        </div>
                                        <div className="optimization-candidates-grid">
                                            {(row.recommendations || []).map((candidate) => {
                                                const aggregateSource = candidateMetricSource(candidate);
                                                const aggregateHelp = candidateProvenanceText(candidate);
                                                const latencySource = metricSourceForKey(candidate, 'latency_ms');
                                                const memorySource = metricSourceForKey(candidate, 'memory_gb');
                                                const qualitySource = metricSourceForKey(candidate, 'quality_score');
                                                const confidenceScore = Number(candidate.confidence?.score);
                                                return (
                                                    <div
                                                        key={`${row.target_id}-${candidate.id}`}
                                                        className={`optimization-card ${
                                                            candidate.id === row.recommended_candidate_id ? 'recommended' : ''
                                                        }`}
                                                    >
                                                        {candidate.id === row.recommended_candidate_id && (
                                                            <div className="recommendation-badge">TOP RANK</div>
                                                        )}
                                                        <div className="optimization-card-header">
                                                            <div className="serve-template-card__title">
                                                                #{candidate.rank || '—'} {candidate.name}
                                                            </div>
                                                            <span
                                                                className={`badge ${METRIC_SOURCE_BADGE_CLASS[aggregateSource]} optimization-source-badge`}
                                                                title={aggregateHelp}
                                                            >
                                                                {METRIC_SOURCE_LABEL[aggregateSource]}
                                                            </span>
                                                        </div>
                                                        <div className="optimization-metrics">
                                                            <div className="optimization-metric-item">
                                                                <span className="optimization-metric-label">Latency</span>
                                                                <span className="optimization-metric-value">
                                                                    {formatOptimizationMetricValue('latency_ms', candidate.metrics.latency_ms, latencySource)}
                                                                </span>
                                                            </div>
                                                            <div className="optimization-metric-item">
                                                                <span className="optimization-metric-label">Memory</span>
                                                                <span className="optimization-metric-value">
                                                                    {formatOptimizationMetricValue('memory_gb', candidate.metrics.memory_gb, memorySource)}
                                                                </span>
                                                            </div>
                                                            <div className="optimization-metric-item">
                                                                <span className="optimization-metric-label">Quality</span>
                                                                <span className="optimization-metric-value">
                                                                    {formatOptimizationMetricValue('quality_score', candidate.metrics.quality_score, qualitySource)}
                                                                </span>
                                                            </div>
                                                        </div>
                                                        {Number.isFinite(confidenceScore) && (
                                                            <div className="optimization-provenance-note">
                                                                Confidence {(confidenceScore * 100).toFixed(0)}%
                                                            </div>
                                                        )}
                                                        {candidate.measurement?.fallback_reason && (
                                                            <div className="optimization-provenance-note">
                                                                {String(candidate.measurement.fallback_reason)}
                                                            </div>
                                                        )}
                                                        {candidate.measurement?.remediation_hint && (
                                                            <div className="optimization-provenance-note">
                                                                Remediation: {String(candidate.measurement.remediation_hint)}
                                                            </div>
                                                        )}
                                                        <button
                                                            className="btn btn-secondary btn-sm"
                                                            style={{ marginTop: 'auto' }}
                                                            onClick={() => applyOptimization(candidate)}
                                                        >
                                                            Apply Settings
                                                        </button>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                <div style={{ marginTop: '0.75rem', marginBottom: '0.75rem' }}>
                    <label className="form-label">Deploy / SDK Plan Target</label>
                    <select className="input" value={deployTargetId} onChange={(e) => setDeployTargetId(e.target.value)}>
                        <option value="deployment.hf_inference_endpoint">HuggingFace Inference Endpoint</option>
                        <option value="deployment.aws_sagemaker">AWS SageMaker Endpoint</option>
                        <option value="deployment.vllm_managed">Managed vLLM API</option>
                        <option value="sdk.apple_coreml_stub">Apple CoreML Stub App</option>
                        <option value="sdk.android_executorch_stub">Android ExecuTorch Stub App</option>
                    </select>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button
                        className="btn btn-primary"
                        onClick={() => void handleCreateExport()}
                        disabled={!selectedExp || isExporting || isLoading || isLoadingTargets}
                    >
                        {isExporting ? 'Exporting...' : 'Run Export'}
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={() => void handleOptimize()}
                        disabled={!selectedExp || isOptimizing || selectedDeploymentTargets.length === 0}
                    >
                        {isOptimizing ? 'Optimizing...' : 'Optimize for Target'}
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={() => void handleStartOptimizationMatrix()}
                        disabled={!selectedExp || isRunningMatrix || selectedDeploymentTargets.length === 0}
                    >
                        {isRunningMatrix ? 'Running Matrix...' : 'Run Benchmark Matrix'}
                    </button>
                    <button className="btn btn-secondary" onClick={() => void handleRegisterModel()} disabled={!selectedExp || isLoading}>
                        Register Selected Model
                    </button>
                </div>
                {(errorMessage || statusMessage) && (
                    <div style={{ marginTop: 'var(--space-md)', color: errorMessage ? 'var(--color-error)' : 'var(--color-success)' }}>
                        {errorMessage || statusMessage}
                    </div>
                )}
            </div>

            {registryModels.length > 0 && (
                <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                    <div style={{ padding: 'var(--space-lg)', borderBottom: '1px solid var(--border-color)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, margin: 0 }}>Model Registry</h3>
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                        <table className="export-history-table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Version</th>
                                    <th>Stage</th>
                                    <th>Exact Match</th>
                                    <th>F1</th>
                                    <th>Judge Pass</th>
                                    <th>Safety Pass</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {registryModels.map((model) => (
                                    <tr key={model.id}>
                                        <td>{model.name}</td>
                                        <td>{model.version}</td>
                                        <td><span className={`badge ${stageColor(model.stage)}`}>{model.stage}</span></td>
                                        <td>{fmtMetric(model.readiness?.metrics?.exact_match)}</td>
                                        <td>{fmtMetric(model.readiness?.metrics?.f1)}</td>
                                        <td>{fmtMetric(model.readiness?.metrics?.llm_judge_pass_rate)}</td>
                                        <td>{fmtMetric(model.readiness?.metrics?.safety_pass_rate)}</td>
                                        <td>
                                            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                                <button className="btn btn-secondary btn-sm" onClick={() => void handlePromote(model.id, 'staging')}>
                                                    Promote Staging
                                                </button>
                                                <button className="btn btn-secondary btn-sm" onClick={() => void handlePromote(model.id, 'production')}>
                                                    Promote Prod
                                                </button>
                                                <button className="btn btn-secondary btn-sm" onClick={() => void handleDeploy(model.id, model.stage === 'production' ? 'production' : 'staging')}>
                                                    Mark Deployed
                                                </button>
                                                <button
                                                    className="btn btn-secondary btn-sm"
                                                    onClick={() => void handleRegistryServePlan(model.id)}
                                                    disabled={isLoadingServePlan}
                                                >
                                                    {isLoadingServePlan ? 'Loading...' : 'Serve Plan'}
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {exportsList.length > 0 && (
                <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                    <div style={{ padding: 'var(--space-lg)', borderBottom: '1px solid var(--border-color)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, margin: 0 }}>Export History</h3>
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                        <table className="export-history-table">
                            <thead>
                                <tr>
                                    <th style={{ width: 40 }}></th>
                                    <th>Format</th>
                                    <th>Quantization</th>
                                    <th>Size</th>
                                    <th>Status</th>
                                    <th>Date</th>
                                </tr>
                            </thead>
                            <tbody>
                                {exportsList.map((exp) => {
                                    const isExpanded = expandedIds.includes(exp.id);
                                    const manifestJson = exp.manifest
                                        ? JSON.stringify(exp.manifest, null, 2)
                                        : '{\n  "status": "No manifest generated yet."\n}';

                                    return (
                                        <React.Fragment key={exp.id}>
                                            <tr className={`export-row ${isExpanded ? 'expanded' : ''}`} onClick={() => toggleExpand(exp.id)} style={{ cursor: 'pointer' }}>
                                                <td style={{ textAlign: 'center' }}><span className="expand-icon" style={{ display: 'inline-block', fontSize: 10 }}>▼</span></td>
                                                <td style={{ fontWeight: 600 }}>{String(exp.export_format || exp.format || 'unknown').toUpperCase()}</td>
                                                <td>{exp.quantization || 'None'}</td>
                                                <td>{formatBytes(exp.file_size_bytes)}</td>
                                                <td><span className={`badge ${statusColor(exp.status)}`}>{exp.status}</span></td>
                                                <td><span style={{ color: 'var(--text-tertiary)' }}>{exp.created_at ? new Date(exp.created_at).toLocaleString() : '—'}</span></td>
                                            </tr>
                                            {isExpanded && (
                                                <tr className="export-details-row">
                                                    <td></td>
                                                    <td colSpan={5}>
                                                        <div className="manifest-header">
                                                            <div className="manifest-title">Manifest and Output Path</div>
                                                            <div style={{ display: 'flex', gap: 8 }}>
                                                                <button
                                                                    className="btn btn-secondary btn-sm"
                                                                    onClick={(e) => { e.stopPropagation(); handleCopy(`path-${exp.id}`, exp.output_path || ''); }}
                                                                    disabled={!exp.output_path}
                                                                >
                                                                    {copyState[`path-${exp.id}`] ? 'Copied!' : 'Copy Path'}
                                                                </button>
                                                                <button
                                                                    className="btn btn-primary btn-sm"
                                                                    onClick={(e) => { e.stopPropagation(); handleCopy(`manifest-${exp.id}`, manifestJson); }}
                                                                >
                                                                    {copyState[`manifest-${exp.id}`] ? 'Copied!' : 'Copy Manifest'}
                                                                </button>
                                                                <button
                                                                    className="btn btn-secondary btn-sm"
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        void handleExportServePlan(exp.id);
                                                                    }}
                                                                    disabled={isLoadingServePlan}
                                                                >
                                                                    {isLoadingServePlan ? 'Loading...' : 'Serve Plan'}
                                                                </button>
                                                                <button
                                                                    className="btn btn-secondary btn-sm"
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        void handleBuildDeployPlan(exp.id);
                                                                    }}
                                                                    disabled={isLoadingDeployPlan}
                                                                >
                                                                    {isLoadingDeployPlan ? 'Planning...' : 'Deploy / SDK Plan'}
                                                                </button>
                                                            </div>
                                                        </div>

                                                        {exp.output_path && (
                                                            <div style={{ marginBottom: 16, fontSize: '0.8rem', color: 'var(--color-info)' }}>
                                                                <span style={{ color: 'var(--text-tertiary)' }}>Path:</span>{' '}
                                                                <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 4 }}>{exp.output_path}</code>
                                                            </div>
                                                        )}

                                                        <div className="manifest-viewer">
                                                            <pre><code>{manifestJson}</code></pre>
                                                        </div>
                                                    </td>
                                                </tr>
                                            )}
                                        </React.Fragment>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {servePlan && (
                <div className="card">
                    <div className="serve-plan-header">
                        <div>
                            <h3 style={{ margin: 0, fontSize: 'var(--font-size-md)' }}>One-Click Serve Plan</h3>
                            <p style={{ margin: '6px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                                Source: {servePlan.source}
                                {servePlan.export_format ? ` • Format: ${servePlan.export_format}` : ''}
                                {servePlan.model_name ? ` • Model: ${servePlan.model_name}` : ''}
                                {servePlan.model_stage ? ` • Stage: ${servePlan.model_stage}` : ''}
                            </p>
                        </div>
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => setServePlan(null)}
                            disabled={Boolean(activeServeRun?.can_stop)}
                        >
                            Close
                        </button>
                    </div>

                    {servePlan.run_dir && (
                        <div className="serve-plan-run-dir">
                            Run dir:{' '}
                            <code>{servePlan.run_dir}</code>
                        </div>
                    )}

                    <div className="serve-template-grid">
                        {(servePlan.templates || []).map((template) => (
                            <div key={template.template_id} className="serve-template-card">
                                <div className="serve-template-card__title">{template.display_name}</div>
                                {template.description && (
                                    <div className="serve-template-card__subtitle">{template.description}</div>
                                )}
                                <button
                                    className="btn btn-primary btn-sm"
                                    onClick={() => void handleStartServeRun(template.template_id)}
                                    disabled={isStartingServeRun || isStoppingServeRun || Boolean(activeServeRun?.can_stop)}
                                >
                                    {isStartingServeRun
                                        ? 'Starting...'
                                        : activeServeRun?.can_stop
                                            ? 'Running...'
                                            : 'Run Now'}
                                </button>

                                {template.command && (
                                    <div className="serve-template-block">
                                        <div className="serve-template-block__label">Launch Command</div>
                                        <pre className="serve-template-code">{template.command}</pre>
                                        <button
                                            className="btn btn-secondary btn-sm"
                                            onClick={() => handleCopy(`serve-cmd-${template.template_id}`, template.command || '')}
                                        >
                                            {copyState[`serve-cmd-${template.template_id}`] ? 'Copied!' : 'Copy Command'}
                                        </button>
                                    </div>
                                )}

                                {Array.isArray(template.setup_commands) && template.setup_commands.length > 0 && (
                                    <div className="serve-template-block">
                                        <div className="serve-template-block__label">Setup Commands</div>
                                        <pre className="serve-template-code">{template.setup_commands.join('\n')}</pre>
                                        <button
                                            className="btn btn-secondary btn-sm"
                                            onClick={() => handleCopy(`serve-setup-${template.template_id}`, (template.setup_commands || []).join('\n'))}
                                        >
                                            {copyState[`serve-setup-${template.template_id}`] ? 'Copied!' : 'Copy Setup'}
                                        </button>
                                    </div>
                                )}

                                {template.healthcheck?.curl && (
                                    <div className="serve-template-block">
                                        <div className="serve-template-block__label">Health Check Curl</div>
                                        <pre className="serve-template-code">{template.healthcheck.curl}</pre>
                                        <button
                                            className="btn btn-secondary btn-sm"
                                            onClick={() => handleCopy(`serve-health-${template.template_id}`, template.healthcheck?.curl || '')}
                                        >
                                            {copyState[`serve-health-${template.template_id}`] ? 'Copied!' : 'Copy Health Curl'}
                                        </button>
                                    </div>
                                )}

                                {template.smoke_test?.curl && (
                                    <div className="serve-template-block">
                                        <div className="serve-template-block__label">Smoke Test Curl</div>
                                        <pre className="serve-template-code">{template.smoke_test.curl}</pre>
                                        <button
                                            className="btn btn-secondary btn-sm"
                                            onClick={() => handleCopy(`serve-smoke-${template.template_id}`, template.smoke_test?.curl || '')}
                                        >
                                            {copyState[`serve-smoke-${template.template_id}`] ? 'Copied!' : 'Copy Smoke Curl'}
                                        </button>
                                    </div>
                                )}

                                {template.first_token_probe?.curl && (
                                    <div className="serve-template-block">
                                        <div className="serve-template-block__label">First-Token Probe Curl</div>
                                        <pre className="serve-template-code">{template.first_token_probe.curl}</pre>
                                        <button
                                            className="btn btn-secondary btn-sm"
                                            onClick={() => handleCopy(`serve-first-token-${template.template_id}`, template.first_token_probe?.curl || '')}
                                        >
                                            {copyState[`serve-first-token-${template.template_id}`] ? 'Copied!' : 'Copy First-Token Curl'}
                                        </button>
                                    </div>
                                )}

                                {Array.isArray(template.notes) && template.notes.length > 0 && (
                                    <div className="serve-template-notes">
                                        {template.notes.map((note, idx) => (
                                            <div key={`${template.template_id}-note-${idx}`}>• {note}</div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    {activeServeRun && (
                        <div className="serve-run-card">
                            <div className="serve-run-card__header">
                                <div>
                                    <div className="serve-template-card__title">Live Serve Status</div>
                                    <div className="serve-template-card__subtitle">
                                        Run ID: {activeServeRun.run_id} • Template: {activeServeRun.template_id}
                                    </div>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                    <span className={`badge ${runStatusColor(activeServeRun.status)}`}>{activeServeRun.status}</span>
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => void handleStopServeRun()}
                                        disabled={isStoppingServeRun || !activeServeRun.can_stop}
                                    >
                                        {isStoppingServeRun ? 'Stopping...' : 'Stop'}
                                    </button>
                                </div>
                            </div>

                            {activeServeRun.command && (
                                <div className="serve-template-block">
                                    <div className="serve-template-block__label">Active Command</div>
                                    <pre className="serve-template-code">{activeServeRun.command}</pre>
                                </div>
                            )}
                            {activeServeRun.telemetry && (
                                <div className="serve-run-telemetry">
                                    <div className="serve-run-telemetry__item">
                                        <span>First healthy</span>
                                        <strong>{fmtTimestamp(activeServeRun.telemetry.first_healthy_at)}</strong>
                                    </div>
                                    <div className="serve-run-telemetry__item">
                                        <span>Startup latency</span>
                                        <strong>
                                            {typeof activeServeRun.telemetry.startup_latency_ms === 'number'
                                                ? `${activeServeRun.telemetry.startup_latency_ms} ms`
                                                : '—'}
                                        </strong>
                                    </div>
                                    <div className="serve-run-telemetry__item">
                                        <span>Smoke checks</span>
                                        <strong>
                                            {Array.isArray(activeServeRun.telemetry.smoke_checks)
                                                ? activeServeRun.telemetry.smoke_checks.length
                                                : 0}
                                        </strong>
                                    </div>
                                    <div className="serve-run-telemetry__item">
                                        <span>Smoke passed</span>
                                        <strong>
                                            {activeServeRun.telemetry.smoke_passed === null || activeServeRun.telemetry.smoke_passed === undefined
                                                ? 'pending'
                                                : activeServeRun.telemetry.smoke_passed
                                                    ? 'yes'
                                                    : 'no'}
                                        </strong>
                                    </div>
                                    <div className="serve-run-telemetry__item">
                                        <span>First token</span>
                                        <strong>{fmtTimestamp(activeServeRun.telemetry.first_token_at)}</strong>
                                    </div>
                                    <div className="serve-run-telemetry__item">
                                        <span>First-token latency</span>
                                        <strong>
                                            {typeof activeServeRun.telemetry.first_token_latency_ms === 'number'
                                                ? `${activeServeRun.telemetry.first_token_latency_ms} ms`
                                                : '—'}
                                        </strong>
                                    </div>
                                    <div className="serve-run-telemetry__item">
                                        <span>Throughput</span>
                                        <strong>
                                            {typeof activeServeRun.telemetry.throughput_tokens_per_sec === 'number'
                                                ? `${activeServeRun.telemetry.throughput_tokens_per_sec.toFixed(2)} tok/s`
                                                : '—'}
                                        </strong>
                                    </div>
                                </div>
                            )}

                            <TerminalConsole logs={activeServeRun.logs_tail || []} height="260px" />

                            <div className="serve-run-card__actions">
                                {activeServeRun.healthcheck_curl && (
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => handleCopy(`run-health-${activeServeRun.run_id}`, activeServeRun.healthcheck_curl || '')}
                                    >
                                        {copyState[`run-health-${activeServeRun.run_id}`] ? 'Copied!' : 'Copy Health Curl'}
                                    </button>
                                )}
                                {activeServeRun.smoke_curl && (
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => handleCopy(`run-smoke-${activeServeRun.run_id}`, activeServeRun.smoke_curl || '')}
                                    >
                                        {copyState[`run-smoke-${activeServeRun.run_id}`] ? 'Copied!' : 'Copy Smoke Curl'}
                                    </button>
                                )}
                                {activeServeRun.first_token_curl && (
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => handleCopy(`run-first-token-${activeServeRun.run_id}`, activeServeRun.first_token_curl || '')}
                                    >
                                        {copyState[`run-first-token-${activeServeRun.run_id}`] ? 'Copied!' : 'Copy First-Token Curl'}
                                    </button>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {deployPlan && (
                <div className="card">
                    <div className="serve-plan-header">
                        <div>
                            <h3 style={{ margin: 0, fontSize: 'var(--font-size-md)' }}>Deploy / SDK Plan</h3>
                            <p style={{ margin: '6px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                                {deployPlan.display_name || deployPlan.target_id}
                            </p>
                        </div>
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                                setDeployPlan(null);
                                setDeployPlanExportId(null);
                                setDeployExecution(null);
                            }}
                        >
                            Close
                        </button>
                    </div>
                    {deployPlan.summary && (
                        <div className="serve-template-block">
                            <div className="serve-template-block__label">Summary</div>
                            <div>{deployPlan.summary}</div>
                        </div>
                    )}
                    {Array.isArray(deployPlan.steps) && deployPlan.steps.length > 0 && (
                        <div className="serve-template-block">
                            <div className="serve-template-block__label">Steps</div>
                            <ol style={{ margin: 0, paddingLeft: '1.2rem' }}>
                                {deployPlan.steps.map((step, idx) => (
                                    <li key={`deploy-step-${idx}`}>{step}</li>
                                ))}
                            </ol>
                        </div>
                    )}
                    <div className="serve-template-block">
                        <div className="serve-template-block__label">Execution</div>
                        <label className="export-smoke-toggle">
                            <input
                                type="checkbox"
                                checked={deployDryRun}
                                onChange={(e) => setDeployDryRun(e.target.checked)}
                            />
                            <span>Dry run (recommended before live deployment)</span>
                        </label>
                        {!deployDryRun && deployPlan.target_id === 'deployment.hf_inference_endpoint' && (
                            <input
                                className="input"
                                type="password"
                                placeholder="HuggingFace token (hf_...)"
                                value={deployHfToken}
                                onChange={(e) => setDeployHfToken(e.target.value)}
                            />
                        )}
                        {!deployDryRun && deployPlan.target_id === 'deployment.vllm_managed' && (
                            <>
                                <input
                                    className="input"
                                    placeholder="Managed API URL (https://...)"
                                    value={deployManagedApiUrl}
                                    onChange={(e) => setDeployManagedApiUrl(e.target.value)}
                                />
                                <input
                                    className="input"
                                    type="password"
                                    placeholder="Managed API token (optional)"
                                    value={deployManagedApiToken}
                                    onChange={(e) => setDeployManagedApiToken(e.target.value)}
                                />
                            </>
                        )}
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            <button
                                className="btn btn-primary btn-sm"
                                onClick={() => void handleExecuteDeployPlan()}
                                disabled={isExecutingDeployPlan || deployPlanExportId === null}
                            >
                                {isExecutingDeployPlan
                                    ? 'Executing...'
                                    : deployDryRun
                                        ? 'Run Dry-Run'
                                        : 'Execute Deployment'}
                            </button>
                            {deployPlanExportId !== null && (
                                <span style={{ color: 'var(--text-tertiary)', fontSize: 'var(--font-size-xs)' }}>
                                    Export ID: {deployPlanExportId}
                                </span>
                            )}
                        </div>
                    </div>
                    {deployPlan.curl_example && (
                        <div className="serve-template-block">
                            <div className="serve-template-block__label">Command Template</div>
                            <pre className="serve-template-code">{deployPlan.curl_example}</pre>
                            <button
                                className="btn btn-secondary btn-sm"
                                onClick={() => handleCopy(`deploy-curl-${deployPlan.target_id}`, deployPlan.curl_example || '')}
                            >
                                {copyState[`deploy-curl-${deployPlan.target_id}`] ? 'Copied!' : 'Copy Command'}
                            </button>
                        </div>
                    )}
                    {deployPlan.sdk_artifact?.zip_path && (
                        <div className="serve-template-block">
                            <div className="serve-template-block__label">SDK Artifact Zip</div>
                            <code>{deployPlan.sdk_artifact.zip_path}</code>
                            <div style={{ marginTop: 8 }}>
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => handleCopy(`deploy-sdk-${deployPlan.target_id}`, deployPlan.sdk_artifact?.zip_path || '')}
                                >
                                    {copyState[`deploy-sdk-${deployPlan.target_id}`] ? 'Copied!' : 'Copy Zip Path'}
                                </button>
                            </div>
                        </div>
                    )}
                    {deployPlan.sdk_artifact && (
                        <>
                            {deployPlan.sdk_artifact.readme_path && (
                                <div className="serve-template-block">
                                    <div className="serve-template-block__label">Bundle README</div>
                                    <code>{deployPlan.sdk_artifact.readme_path}</code>
                                </div>
                            )}
                            {deployPlan.sdk_artifact.entrypoint_path && (
                                <div className="serve-template-block">
                                    <div className="serve-template-block__label">Entrypoint</div>
                                    <code>{deployPlan.sdk_artifact.entrypoint_path}</code>
                                </div>
                            )}
                            {deployPlan.sdk_artifact.runtime_path && (
                                <div className="serve-template-block">
                                    <div className="serve-template-block__label">Runtime Config</div>
                                    <code>{deployPlan.sdk_artifact.runtime_path}</code>
                                </div>
                            )}
                            {Array.isArray(deployPlan.sdk_artifact.model_placement_paths)
                                && deployPlan.sdk_artifact.model_placement_paths.length > 0 && (
                                    <div className="serve-template-block">
                                        <div className="serve-template-block__label">Model Placement Paths</div>
                                        <ul className="export-inline-list">
                                            {deployPlan.sdk_artifact.model_placement_paths.map((path) => (
                                                <li key={path}><code>{path}</code></li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            {Array.isArray(deployPlan.sdk_artifact.run_commands)
                                && deployPlan.sdk_artifact.run_commands.length > 0 && (
                                    <div className="serve-template-block">
                                        <div className="serve-template-block__label">Run Commands</div>
                                        <pre className="serve-template-code">{deployPlan.sdk_artifact.run_commands.join('\n')}</pre>
                                    </div>
                                )}
                            {Array.isArray(deployPlan.sdk_artifact.smoke_commands)
                                && deployPlan.sdk_artifact.smoke_commands.length > 0 && (
                                    <div className="serve-template-block">
                                        <div className="serve-template-block__label">Smoke Commands</div>
                                        <pre className="serve-template-code">{deployPlan.sdk_artifact.smoke_commands.join('\n')}</pre>
                                    </div>
                                )}
                            {deployPlan.sdk_artifact.smoke_validation && (
                                <div className="serve-template-block">
                                    <div className="serve-template-block__label">Smoke Validation</div>
                                    <div>
                                        <span className={`badge ${deployPlan.sdk_artifact.smoke_validation.smoke_passed ? 'badge-success' : 'badge-error'}`}>
                                            {deployPlan.sdk_artifact.smoke_validation.smoke_passed ? 'PASS' : 'FAIL'}
                                        </span>
                                    </div>
                                    {Array.isArray(deployPlan.sdk_artifact.smoke_validation.errors)
                                        && deployPlan.sdk_artifact.smoke_validation.errors.length > 0 && (
                                            <ul className="export-inline-list export-inline-list-error">
                                                {deployPlan.sdk_artifact.smoke_validation.errors.map((error) => (
                                                    <li key={error}>{error}</li>
                                                ))}
                                            </ul>
                                        )}
                                    {Array.isArray(deployPlan.sdk_artifact.smoke_validation.warnings)
                                        && deployPlan.sdk_artifact.smoke_validation.warnings.length > 0 && (
                                            <ul className="export-inline-list">
                                                {deployPlan.sdk_artifact.smoke_validation.warnings.map((warning) => (
                                                    <li key={warning}>{warning}</li>
                                                ))}
                                            </ul>
                                        )}
                                    {Array.isArray(deployPlan.sdk_artifact.smoke_validation.checks)
                                        && deployPlan.sdk_artifact.smoke_validation.checks.length > 0 && (
                                            <div className="table-container" style={{ maxHeight: 220 }}>
                                                <table className="docs-table">
                                                    <thead>
                                                        <tr>
                                                            <th>Check</th>
                                                            <th>Status</th>
                                                            <th>Message</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {deployPlan.sdk_artifact.smoke_validation.checks.map((check, idx) => (
                                                            <tr key={`${check.check_id || 'check'}-${idx}`}>
                                                                <td>{check.check_id || 'check'}</td>
                                                                <td>{check.status || 'unknown'}</td>
                                                                <td>{check.message || '—'}</td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        )}
                                </div>
                            )}
                        </>
                    )}
                    {deployExecution?.execution && (
                        <div className="serve-template-block">
                            <div className="serve-template-block__label">Execution Result</div>
                            <div style={{ fontSize: 'var(--font-size-sm)' }}>
                                <strong>Status:</strong> {deployExecution.execution.status || 'unknown'}
                                {deployExecution.execution.dry_run ? ' (dry-run)' : ''}
                            </div>
                            {deployExecution.execution.message && (
                                <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                                    {deployExecution.execution.message}
                                </div>
                            )}
                            {deployExecution.execution.request && (
                                <pre className="serve-template-code">
                                    {JSON.stringify(deployExecution.execution.request, null, 2)}
                                </pre>
                            )}
                            {deployExecution.execution.response && (
                                <pre className="serve-template-code">
                                    {JSON.stringify(deployExecution.execution.response, null, 2)}
                                </pre>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
