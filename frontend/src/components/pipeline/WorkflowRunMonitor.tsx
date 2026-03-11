import { useCallback, useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type {
    PipelineAutopilotScorecardResponse,
    PipelineGraphCompileResponse,
    PipelineGraphContractSaveResponse,
    PipelineGraphResponse,
    PipelineGraphTemplate,
    PipelineGraphTemplateListResponse,
    PipelineStage,
    WorkflowRun,
    WorkflowRunListResponse,
} from '../../types';
import './WorkflowRunMonitor.css';

interface WorkflowRunMonitorProps {
    projectId: number;
    currentStage: PipelineStage;
}

interface WorkflowRunQueuedResponse {
    project_id: number;
    queued: boolean;
    run_id: number;
    run: WorkflowRun;
}

type AutopilotProfile = 'safe' | 'guided' | 'full';

interface AutopilotGateResult {
    profile: AutopilotProfile;
    checked_at: string;
    passed: boolean;
    valid_graph: boolean;
    active_stage_ready: boolean;
    missing_inputs: string[];
    missing_runtime: string[];
    errors: string[];
    warnings: string[];
    notes: string[];
}

const SOURCE_ARTIFACTS = new Set(['source.file', 'source.remote_dataset']);
const AUTOPILOT_PROFILE_RANK: Record<AutopilotProfile, number> = {
    safe: 0,
    guided: 1,
    full: 2,
};

function extractErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const detail = (error as { response?: { data?: { detail?: string | { message?: string } } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
        if (typeof detail === 'object' && detail && typeof detail.message === 'string') {
            return detail.message;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Request failed.';
}

function formatDateTime(value: string | null | undefined): string {
    if (!value) {
        return '—';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }
    return date.toLocaleString();
}

function formatRate(value: number | null | undefined): string {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return 'n/a';
    }
    return `${Math.round(value * 100)}%`;
}

function statusClass(status: string): string {
    const normalized = status.toLowerCase();
    if (normalized === 'completed') return 'ok';
    if (normalized === 'running') return 'active';
    if (normalized === 'failed' || normalized === 'blocked') return 'error';
    return 'neutral';
}

function normalizeAutopilotProfile(value: unknown): AutopilotProfile | null {
    const token = String(value || '').trim().toLowerCase();
    if (token === 'safe' || token === 'guided' || token === 'full') {
        return token;
    }
    return null;
}

function normalizeConfig(value: unknown): Record<string, unknown> {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return {};
    }
    return value as Record<string, unknown>;
}

function cloneGraph(graph: PipelineGraphResponse): PipelineGraphResponse {
    return {
        ...graph,
        nodes: (graph.nodes || []).map((node) => ({
            ...node,
            config: normalizeConfig(node.config),
        })),
        edges: [...(graph.edges || [])],
    };
}

function patchStageConfig(
    graph: PipelineGraphResponse,
    stageName: string,
    patch: Record<string, unknown>,
): PipelineGraphResponse {
    return {
        ...graph,
        nodes: graph.nodes.map((node) => {
            if (String(node.stage) !== stageName) {
                return node;
            }
            const currentConfig = normalizeConfig(node.config);
            return {
                ...node,
                config: {
                    ...currentConfig,
                    ...patch,
                },
            };
        }),
    };
}

function graphForAutopilotProfile(
    baseGraph: PipelineGraphResponse,
    profile: AutopilotProfile,
): PipelineGraphResponse {
    let next = cloneGraph(baseGraph);

    next = patchStageConfig(next, 'cloud_burst', {
        mode: 'plan',
    });

    if (profile === 'safe') {
        next = patchStageConfig(next, 'synthetic_conversation', { mode: 'noop' });
        next = patchStageConfig(next, 'semantic_curation', { mode: 'noop' });
        next = patchStageConfig(next, 'distillation', { mode: 'noop' });
        next = patchStageConfig(next, 'model_merge', { mode: 'noop' });
        return next;
    }

    if (profile === 'guided') {
        next = patchStageConfig(next, 'synthetic_conversation', {
            mode: 'generate_and_save',
        });
        next = patchStageConfig(next, 'semantic_curation', { mode: 'analyze' });
        next = patchStageConfig(next, 'distillation', { mode: 'noop' });
        next = patchStageConfig(next, 'model_merge', { mode: 'noop' });
        return next;
    }

    next = patchStageConfig(next, 'synthetic_conversation', {
        mode: 'generate_and_save',
    });
    next = patchStageConfig(next, 'semantic_curation', {
        mode: 'analyze',
    });
    next = patchStageConfig(next, 'distillation', {
        mode: 'create_and_start',
        name: 'autopilot-distillation',
    });
    next = patchStageConfig(next, 'model_merge', {
        mode: 'queue_merge',
    });
    return next;
}

export default function WorkflowRunMonitor({ projectId, currentStage }: WorkflowRunMonitorProps) {
    const [runs, setRuns] = useState<WorkflowRun[]>([]);
    const [templates, setTemplates] = useState<PipelineGraphTemplate[]>([]);
    const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
    const [selectedRun, setSelectedRun] = useState<WorkflowRun | null>(null);

    const [isLoading, setIsLoading] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [isApplyingTemplate, setIsApplyingTemplate] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const [executionBackend, setExecutionBackend] = useState('local');
    const [maxRetries, setMaxRetries] = useState(0);
    const [stopOnBlocked, setStopOnBlocked] = useState(true);
    const [stopOnFailure, setStopOnFailure] = useState(true);
    const [selectedTemplateId, setSelectedTemplateId] = useState('__saved__');
    const [autopilotProfile, setAutopilotProfile] = useState<AutopilotProfile>('safe');
    const [isAutopilotPreflightRunning, setIsAutopilotPreflightRunning] = useState(false);
    const [autopilotGate, setAutopilotGate] = useState<AutopilotGateResult | null>(null);
    const [autopilotScorecard, setAutopilotScorecard] = useState<PipelineAutopilotScorecardResponse | null>(null);
    const [isAutopilotScorecardLoading, setIsAutopilotScorecardLoading] = useState(false);
    const [autopilotProfileManual, setAutopilotProfileManual] = useState(false);

    const selectedTemplate = useMemo(
        () => templates.find((template) => template.template_id === selectedTemplateId) || null,
        [templates, selectedTemplateId],
    );
    const selectedRunIsActive = useMemo(
        () => !!selectedRun && (selectedRun.status === 'pending' || selectedRun.status === 'running'),
        [selectedRun],
    );
    const isAutopilotTemplate = useMemo(
        () => selectedTemplate?.template_id === 'template.autopilot_chat',
        [selectedTemplate],
    );
    const autopilotGraph = useMemo(() => {
        if (!selectedTemplate || !isAutopilotTemplate) {
            return null;
        }
        return graphForAutopilotProfile(selectedTemplate.graph, autopilotProfile);
    }, [autopilotProfile, isAutopilotTemplate, selectedTemplate]);
    const recommendedAutopilotProfile = useMemo(
        () => normalizeAutopilotProfile(autopilotScorecard?.recommended_profile),
        [autopilotScorecard],
    );
    const autopilotPromotionAvailable = useMemo(() => {
        if (!autopilotScorecard || !recommendedAutopilotProfile) {
            return false;
        }
        const currentRank = AUTOPILOT_PROFILE_RANK[autopilotProfile];
        const recommendedRank = AUTOPILOT_PROFILE_RANK[recommendedAutopilotProfile];
        return Boolean(autopilotScorecard.promotion_available) && recommendedRank > currentRank;
    }, [autopilotProfile, autopilotScorecard, recommendedAutopilotProfile]);

    const loadRuns = useCallback(async () => {
        setIsLoading(true);
        try {
            const [runsRes, templatesRes] = await Promise.all([
                api.get<WorkflowRunListResponse>(`/projects/${projectId}/pipeline/graph/workflow-runs`, {
                    params: { limit: 20 },
                }),
                api.get<PipelineGraphTemplateListResponse>(`/projects/${projectId}/pipeline/graph/templates`),
            ]);
            const runItems = runsRes.data.runs || [];
            setRuns(runItems);
            setTemplates(templatesRes.data.templates || []);

            if (selectedRunId !== null) {
                const found = runItems.find((item) => item.id === selectedRunId) || null;
                setSelectedRun(found);
            } else if (runItems.length > 0) {
                setSelectedRunId(runItems[0].id);
                setSelectedRun(runItems[0]);
            } else {
                setSelectedRun(null);
            }
            setErrorMessage('');
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsLoading(false);
        }
    }, [projectId, selectedRunId]);

    useEffect(() => {
        void loadRuns();
    }, [loadRuns, currentStage]);

    const fetchRunDetail = useCallback(async (runId: number, refreshList = false) => {
        try {
            const res = await api.get<WorkflowRun>(`/projects/${projectId}/pipeline/graph/workflow-runs/${runId}`);
            setSelectedRun(res.data);
            if (refreshList) {
                await loadRuns();
            }
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        }
    }, [projectId, loadRuns]);

    useEffect(() => {
        if (selectedRunId === null) {
            return;
        }
        void fetchRunDetail(selectedRunId);
    }, [selectedRunId, fetchRunDetail]);

    useEffect(() => {
        if (selectedRunId === null || !selectedRunIsActive) {
            return;
        }
        const intervalId = window.setInterval(() => {
            void fetchRunDetail(selectedRunId, true);
        }, 2000);
        return () => window.clearInterval(intervalId);
    }, [selectedRunId, selectedRunIsActive, fetchRunDetail]);

    const loadAutopilotScorecard = useCallback(async () => {
        if (!isAutopilotTemplate) {
            setAutopilotScorecard(null);
            return;
        }
        setIsAutopilotScorecardLoading(true);
        try {
            const res = await api.get<PipelineAutopilotScorecardResponse>(
                `/projects/${projectId}/pipeline/graph/autopilot/scorecard`,
                { params: { limit: 30 } },
            );
            setAutopilotScorecard(res.data);
        } catch {
            setAutopilotScorecard(null);
        } finally {
            setIsAutopilotScorecardLoading(false);
        }
    }, [isAutopilotTemplate, projectId]);

    useEffect(() => {
        if (!isAutopilotTemplate) {
            setAutopilotScorecard(null);
            setAutopilotProfileManual(false);
            return;
        }
        void loadAutopilotScorecard();
    }, [isAutopilotTemplate, loadAutopilotScorecard, selectedTemplateId]);

    useEffect(() => {
        setAutopilotGate(null);
    }, [autopilotProfile, selectedTemplateId]);

    useEffect(() => {
        if (!isAutopilotTemplate) {
            return;
        }
        if (autopilotProfileManual) {
            return;
        }
        if (!recommendedAutopilotProfile) {
            return;
        }
        setAutopilotProfile(recommendedAutopilotProfile);
    }, [autopilotProfileManual, isAutopilotTemplate, recommendedAutopilotProfile]);

    const handleRunWorkflow = async () => {
        setIsRunning(true);
        setStatusMessage('');
        try {
            const payload: Record<string, unknown> = {
                allow_fallback: true,
                use_saved_override: selectedTemplateId === '__saved__',
                execution_backend: executionBackend,
                max_retries: maxRetries,
                stop_on_blocked: stopOnBlocked,
                stop_on_failure: stopOnFailure,
                config: {},
            };
            if (selectedTemplate) {
                payload.graph = selectedTemplate.graph;
                payload.use_saved_override = false;
            }

            const res = await api.post<WorkflowRunQueuedResponse>(`/projects/${projectId}/pipeline/graph/run-async`, payload);
            setStatusMessage(`Workflow run #${res.data.run_id} queued.`);
            setSelectedRunId(res.data.run_id);
            setSelectedRun(res.data.run);
            setErrorMessage('');
            await loadRuns();
            if (isAutopilotTemplate) {
                await loadAutopilotScorecard();
            }
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsRunning(false);
        }
    };

    const runAutopilotPreflightGate = useCallback(async (
        graphOverride?: PipelineGraphResponse,
    ): Promise<AutopilotGateResult | null> => {
        const graph = graphOverride || autopilotGraph;
        if (!graph || !isAutopilotTemplate) {
            return null;
        }
        setIsAutopilotPreflightRunning(true);
        setStatusMessage('');
        try {
            const res = await api.post<PipelineGraphCompileResponse>(
                `/projects/${projectId}/pipeline/graph/compile`,
                {
                    graph,
                    allow_fallback: false,
                    use_saved_override: false,
                },
            );
            const checks = res.data?.checks || {
                active_stage_present: false,
                active_stage_ready_now: false,
                active_stage_missing_inputs: [],
                active_stage_missing_runtime_requirements: [],
            };
            const rawMissingInputs = Array.isArray(checks.active_stage_missing_inputs)
                ? checks.active_stage_missing_inputs
                : [];
            const missingInputs = rawMissingInputs.filter((item) => !SOURCE_ARTIFACTS.has(String(item)));
            const bootstrappedInputs = rawMissingInputs.filter((item) => SOURCE_ARTIFACTS.has(String(item)));
            const missingRuntime = Array.isArray(checks.active_stage_missing_runtime_requirements)
                ? checks.active_stage_missing_runtime_requirements
                : [];
            const errors = Array.isArray(res.data?.errors) ? res.data.errors : [];
            const warnings = Array.isArray(res.data?.warnings) ? res.data.warnings : [];
            const notes: string[] = [];
            if (bootstrappedInputs.length > 0) {
                notes.push(`Source artifacts will be auto-bootstrapped: ${bootstrappedInputs.join(', ')}`);
            }
            const activeStageReady = Boolean(checks.active_stage_present) && missingInputs.length === 0 && missingRuntime.length === 0;
            const passed = Boolean(res.data?.valid_graph) && errors.length === 0 && activeStageReady;
            const gate: AutopilotGateResult = {
                profile: autopilotProfile,
                checked_at: new Date().toISOString(),
                passed,
                valid_graph: Boolean(res.data?.valid_graph),
                active_stage_ready: activeStageReady,
                missing_inputs: missingInputs,
                missing_runtime: missingRuntime,
                errors,
                warnings,
                notes,
            };
            setAutopilotGate(gate);
            if (passed) {
                setStatusMessage('Autopilot preflight passed.');
                setErrorMessage('');
            } else {
                setStatusMessage('Autopilot preflight blocked. Review gate diagnostics.');
            }
            return gate;
        } catch (error) {
            setAutopilotGate(null);
            setErrorMessage(extractErrorMessage(error));
            return null;
        } finally {
            setIsAutopilotPreflightRunning(false);
        }
    }, [autopilotGraph, autopilotProfile, isAutopilotTemplate, projectId]);

    const handleApplyTemplateDefaults = useCallback(async () => {
        if (!selectedTemplate) {
            return;
        }
        setIsApplyingTemplate(true);
        setStatusMessage('');
        try {
            await api.put<PipelineGraphContractSaveResponse>(
                `/projects/${projectId}/pipeline/graph/contract`,
                { graph: autopilotGraph || selectedTemplate.graph },
            );
            setErrorMessage('');
            setStatusMessage(
                (
                    `Template '${selectedTemplate.display_name}' saved. `
                    + `Stage panels can now prefill from this graph contract`
                    + `${isAutopilotTemplate ? ` (profile: ${autopilotProfile})` : ''}.`
                ),
            );
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsApplyingTemplate(false);
        }
    }, [autopilotGraph, autopilotProfile, isAutopilotTemplate, projectId, selectedTemplate]);

    const handleRunAutopilot = useCallback(async () => {
        if (!selectedTemplate || !isAutopilotTemplate || !autopilotGraph) {
            return;
        }
        setIsRunning(true);
        setStatusMessage('');
        try {
            const gate = await runAutopilotPreflightGate(autopilotGraph);
            if (!gate || !gate.passed) {
                setStatusMessage('Autopilot run blocked by preflight gate.');
                return;
            }
            const profileMaxRetries = autopilotProfile === 'full' ? 2 : 1;
            const payload: Record<string, unknown> = {
                allow_fallback: true,
                use_saved_override: false,
                execution_backend: 'local',
                max_retries: profileMaxRetries,
                stop_on_blocked: true,
                stop_on_failure: true,
                graph: autopilotGraph,
                config: {
                    bootstrap_source_artifacts: true,
                    autopilot_template_id: selectedTemplate.template_id,
                    autopilot: {
                        profile: autopilotProfile,
                        preflight: {
                            checked_at: gate.checked_at,
                            passed: gate.passed,
                            valid_graph: gate.valid_graph,
                            active_stage_ready: gate.active_stage_ready,
                            missing_inputs: gate.missing_inputs,
                            missing_runtime: gate.missing_runtime,
                            errors: gate.errors,
                            warnings: gate.warnings,
                            notes: gate.notes,
                        },
                    },
                },
            };
            const res = await api.post<WorkflowRunQueuedResponse>(`/projects/${projectId}/pipeline/graph/run-async`, payload);
            setStatusMessage(
                (
                    `Autopilot run #${res.data.run_id} queued `
                    + `(profile=${autopilotProfile}, backend=local, retries=${profileMaxRetries}).`
                ),
            );
            setSelectedRunId(res.data.run_id);
            setSelectedRun(res.data.run);
            setErrorMessage('');
            await loadRuns();
            await loadAutopilotScorecard();
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsRunning(false);
        }
    }, [autopilotGraph, autopilotProfile, isAutopilotTemplate, loadAutopilotScorecard, loadRuns, projectId, runAutopilotPreflightGate, selectedTemplate]);

    const handleUseRecommendedProfile = useCallback(() => {
        if (!recommendedAutopilotProfile) {
            return;
        }
        setAutopilotProfileManual(true);
        setAutopilotProfile(recommendedAutopilotProfile);
        setErrorMessage('');
        setStatusMessage(`Autopilot profile set to recommended '${recommendedAutopilotProfile}'.`);
    }, [recommendedAutopilotProfile]);

    return (
        <div className="card workflow-run-card">
            <div className="workflow-run-header">
                <div>
                    <h3>Workflow Run Monitor</h3>
                    <p>Run full DAG templates and inspect per-node execution status with retries.</p>
                </div>
            </div>

            <div className="workflow-run-controls">
                <label>
                    Graph Source
                    <select
                        className="input"
                        value={selectedTemplateId}
                        onChange={(e) => setSelectedTemplateId(e.target.value)}
                        disabled={isRunning}
                    >
                        <option value="__saved__">Use Saved/Default Graph</option>
                        {templates.map((template) => (
                            <option key={template.template_id} value={template.template_id}>
                                {template.display_name}
                            </option>
                        ))}
                    </select>
                </label>
                <label>
                    Backend
                    <select
                        className="input"
                        value={executionBackend}
                        onChange={(e) => setExecutionBackend(e.target.value)}
                        disabled={isRunning}
                    >
                        <option value="local">local</option>
                        <option value="celery">celery</option>
                        <option value="external">external</option>
                    </select>
                </label>
                <label>
                    Max Retries
                    <input
                        className="input"
                        type="number"
                        min={0}
                        max={5}
                        value={maxRetries}
                        onChange={(e) => setMaxRetries(Math.max(0, Math.min(5, Number.parseInt(e.target.value || '0', 10) || 0)))}
                        disabled={isRunning}
                    />
                </label>
                <label>
                    Stop on Blocked
                    <select
                        className="input"
                        value={stopOnBlocked ? 'yes' : 'no'}
                        onChange={(e) => setStopOnBlocked(e.target.value === 'yes')}
                        disabled={isRunning}
                    >
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                    </select>
                </label>
                <label>
                    Stop on Failure
                    <select
                        className="input"
                        value={stopOnFailure ? 'yes' : 'no'}
                        onChange={(e) => setStopOnFailure(e.target.value === 'yes')}
                        disabled={isRunning}
                    >
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                    </select>
                </label>
                <button type="button" className="btn btn-primary" onClick={handleRunWorkflow} disabled={isRunning || isLoading}>
                    {isRunning ? 'Running...' : 'Run Workflow DAG'}
                </button>
                <button type="button" className="btn btn-secondary" onClick={() => void loadRuns()} disabled={isRunning || isLoading}>
                    Refresh Runs
                </button>
            </div>

            {isAutopilotTemplate && selectedTemplate && (
                <div className="workflow-run-autopilot">
                    <div className="workflow-run-autopilot__text">
                        <strong>Autopilot Chat Flow</strong>
                        <span>
                            Select execution profile, run preflight gate, save prefills, then launch one guided run.
                        </span>
                    </div>
                    <div className="workflow-run-autopilot__profile">
                        <label>
                            Execution Profile
                            <select
                                className="input"
                                value={autopilotProfile}
                                onChange={(e) => {
                                    setAutopilotProfileManual(true);
                                    setAutopilotProfile(e.target.value as AutopilotProfile);
                                }}
                                disabled={isApplyingTemplate || isRunning || isAutopilotPreflightRunning}
                            >
                                <option value="safe">Safe (noops on advanced heavy steps)</option>
                                <option value="guided">Guided (synthetic + semantic enabled)</option>
                                <option value="full">Full (enables distillation + model merge)</option>
                            </select>
                        </label>
                    </div>
                    <div className="workflow-run-autopilot__actions">
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => void runAutopilotPreflightGate()}
                            disabled={isApplyingTemplate || isRunning || isAutopilotPreflightRunning}
                        >
                            {isAutopilotPreflightRunning ? 'Checking...' : 'Run Preflight Gate'}
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => void handleApplyTemplateDefaults()}
                            disabled={isApplyingTemplate || isRunning || isAutopilotPreflightRunning}
                        >
                            {isApplyingTemplate ? 'Applying...' : 'Apply Prefills'}
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={() => void handleRunAutopilot()}
                            disabled={isApplyingTemplate || isRunning || isAutopilotPreflightRunning}
                        >
                            {isRunning ? 'Running...' : 'Run Autopilot Path'}
                        </button>
                    </div>
                    <div className="workflow-run-autopilot__scorecard">
                        <div className="workflow-run-autopilot__scorecard-head">
                            <strong>Scorecard</strong>
                            {isAutopilotScorecardLoading && <span>Refreshing...</span>}
                            {!isAutopilotScorecardLoading && recommendedAutopilotProfile && (
                                <span>
                                    recommended: {recommendedAutopilotProfile}
                                    {autopilotPromotionAvailable ? ' (promotion ready)' : ''}
                                </span>
                            )}
                        </div>
                        {autopilotScorecard?.reason && (
                            <div className="workflow-run-autopilot__scorecard-line">
                                {autopilotScorecard.reason}
                            </div>
                        )}
                        {autopilotScorecard && (
                            <div className="workflow-run-autopilot__scorecard-line">
                                safe {autopilotScorecard.by_profile.safe.runs} runs ({formatRate(autopilotScorecard.by_profile.safe.success_rate)} success) • guided {autopilotScorecard.by_profile.guided.runs} runs ({formatRate(autopilotScorecard.by_profile.guided.success_rate)} success) • full {autopilotScorecard.by_profile.full.runs} runs ({formatRate(autopilotScorecard.by_profile.full.success_rate)} success)
                            </div>
                        )}
                        {recommendedAutopilotProfile && recommendedAutopilotProfile !== autopilotProfile && (
                            <div className="workflow-run-autopilot__scorecard-actions">
                                <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={handleUseRecommendedProfile}
                                    disabled={isApplyingTemplate || isRunning || isAutopilotPreflightRunning}
                                >
                                    Use Recommended Profile
                                </button>
                            </div>
                        )}
                    </div>
                    {autopilotGate && (
                        <div className={`workflow-run-autopilot__gate ${autopilotGate.passed ? 'ok' : 'blocked'}`}>
                            <div className="workflow-run-autopilot__gate-head">
                                <strong>
                                    Preflight: {autopilotGate.passed ? 'PASS' : 'BLOCKED'}
                                </strong>
                                <span>
                                    profile={autopilotGate.profile} • {formatDateTime(autopilotGate.checked_at)}
                                </span>
                            </div>
                            {autopilotGate.notes.length > 0 && (
                                <div className="workflow-run-autopilot__gate-line">
                                    notes: {autopilotGate.notes.join(' | ')}
                                </div>
                            )}
                            {autopilotGate.missing_inputs.length > 0 && (
                                <div className="workflow-run-autopilot__gate-line">
                                    missing inputs: {autopilotGate.missing_inputs.join(', ')}
                                </div>
                            )}
                            {autopilotGate.missing_runtime.length > 0 && (
                                <div className="workflow-run-autopilot__gate-line">
                                    missing runtime: {autopilotGate.missing_runtime.join(', ')}
                                </div>
                            )}
                            {autopilotGate.errors.length > 0 && (
                                <div className="workflow-run-autopilot__gate-line">
                                    errors: {autopilotGate.errors.join(' | ')}
                                </div>
                            )}
                            {autopilotGate.warnings.length > 0 && (
                                <div className="workflow-run-autopilot__gate-line">
                                    warnings: {autopilotGate.warnings.join(' | ')}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {errorMessage && <div className="workflow-run-alert workflow-run-alert--error">{errorMessage}</div>}
            {statusMessage && <div className="workflow-run-alert workflow-run-alert--ok">{statusMessage}</div>}
            {isLoading && <div className="workflow-run-alert">Loading workflow runs...</div>}

            <div className="workflow-run-layout">
                <section className="workflow-run-list">
                    <h4>Recent Runs</h4>
                    {runs.length === 0 && <p className="workflow-run-empty">No workflow runs yet.</p>}
                    {runs.map((run) => (
                        <button
                            type="button"
                            key={run.id}
                            className={`workflow-run-item ${selectedRunId === run.id ? 'active' : ''}`}
                            onClick={() => {
                                setSelectedRunId(run.id);
                                setSelectedRun(run);
                            }}
                        >
                            <div className="workflow-run-item-row">
                                <strong>Run #{run.id}</strong>
                                <span className={`status-pill ${statusClass(run.status)}`}>{run.status}</span>
                            </div>
                            <div className="workflow-run-item-row">
                                <span>{run.execution_backend}</span>
                                <span>{formatDateTime(run.created_at)}</span>
                            </div>
                        </button>
                    ))}
                </section>

                <section className="workflow-run-detail">
                    <h4>Run Detail</h4>
                    {!selectedRun && <p className="workflow-run-empty">Select a run to inspect node details.</p>}
                    {selectedRun && (
                        <>
                            <div className="workflow-run-meta">
                                <span>Run #{selectedRun.id}</span>
                                <span>Status: {selectedRun.status}</span>
                                <span>Backend: {selectedRun.execution_backend}</span>
                                <span>Started: {formatDateTime(selectedRun.started_at)}</span>
                                <span>Finished: {formatDateTime(selectedRun.finished_at)}</span>
                            </div>
                            <div className="workflow-run-nodes">
                                {selectedRun.nodes.map((node) => (
                                    <article key={node.id} className="workflow-run-node">
                                        <div className="workflow-run-node-head">
                                            <strong>{node.stage}</strong>
                                            <span className={`status-pill ${statusClass(node.status)}`}>{node.status}</span>
                                        </div>
                                        <div className="workflow-run-node-meta">
                                            <span>node: {node.node_id}</span>
                                            <span>attempts: {node.attempt_count}/{Math.max(node.max_retries + 1, 1)}</span>
                                            <span>published: {node.published_artifact_keys.length}</span>
                                        </div>
                                        {node.missing_inputs.length > 0 && (
                                            <div className="workflow-run-node-warn">missing inputs: {node.missing_inputs.join(', ')}</div>
                                        )}
                                        {node.missing_runtime_requirements.length > 0 && (
                                            <div className="workflow-run-node-warn">
                                                missing runtime: {node.missing_runtime_requirements.join(', ')}
                                            </div>
                                        )}
                                        {node.error_message && (
                                            <div className="workflow-run-node-error">{node.error_message}</div>
                                        )}
                                    </article>
                                ))}
                            </div>
                        </>
                    )}
                </section>
            </div>
        </div>
    );
}
