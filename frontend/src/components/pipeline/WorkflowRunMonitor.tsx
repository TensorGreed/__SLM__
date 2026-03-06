import { useCallback, useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type {
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

function statusClass(status: string): string {
    const normalized = status.toLowerCase();
    if (normalized === 'completed') return 'ok';
    if (normalized === 'running') return 'active';
    if (normalized === 'failed' || normalized === 'blocked') return 'error';
    return 'neutral';
}

export default function WorkflowRunMonitor({ projectId, currentStage }: WorkflowRunMonitorProps) {
    const [runs, setRuns] = useState<WorkflowRun[]>([]);
    const [templates, setTemplates] = useState<PipelineGraphTemplate[]>([]);
    const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
    const [selectedRun, setSelectedRun] = useState<WorkflowRun | null>(null);

    const [isLoading, setIsLoading] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const [executionBackend, setExecutionBackend] = useState('local');
    const [maxRetries, setMaxRetries] = useState(0);
    const [stopOnBlocked, setStopOnBlocked] = useState(true);
    const [stopOnFailure, setStopOnFailure] = useState(true);
    const [selectedTemplateId, setSelectedTemplateId] = useState('__saved__');

    const selectedTemplate = useMemo(
        () => templates.find((template) => template.template_id === selectedTemplateId) || null,
        [templates, selectedTemplateId],
    );
    const selectedRunIsActive = useMemo(
        () => !!selectedRun && (selectedRun.status === 'pending' || selectedRun.status === 'running'),
        [selectedRun],
    );

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
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsRunning(false);
        }
    };

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
