import { Fragment, useCallback, useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type {
    PipelineGraphDryRunResponse,
    PipelineGraphResponse,
    PipelineGraphRunStepResponse,
    PipelineGraphValidationResponse,
    PipelineStage,
} from '../../types';
import './PipelineGraphPreview.css';

interface PipelineGraphPreviewProps {
    projectId: number;
    currentStage: PipelineStage;
    onPipelineUpdated?: () => Promise<void> | void;
}

function extractErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const detail = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Request failed.';
}

export default function PipelineGraphPreview({
    projectId,
    currentStage,
    onPipelineUpdated,
}: PipelineGraphPreviewProps) {
    const [graph, setGraph] = useState<PipelineGraphResponse | null>(null);
    const [isLoadingGraph, setIsLoadingGraph] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');

    const [actionLoading, setActionLoading] = useState<'validate' | 'dry-run' | 'run-step' | null>(null);
    const [validateResult, setValidateResult] = useState<PipelineGraphValidationResponse | null>(null);
    const [dryRunResult, setDryRunResult] = useState<PipelineGraphDryRunResponse | null>(null);
    const [runResult, setRunResult] = useState<PipelineGraphRunStepResponse | null>(null);

    const loadGraph = useCallback(async () => {
        setIsLoadingGraph(true);
        try {
            const res = await api.get<PipelineGraphResponse>(`/projects/${projectId}/pipeline/graph`);
            setGraph(res.data);
            setErrorMessage('');
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsLoadingGraph(false);
        }
    }, [projectId]);

    useEffect(() => {
        void loadGraph();
    }, [loadGraph, currentStage]);

    const orderedNodes = useMemo(() => {
        if (!graph) {
            return [];
        }
        return [...graph.nodes].sort((a, b) => a.index - b.index);
    }, [graph]);

    const handleValidate = async () => {
        setActionLoading('validate');
        setRunResult(null);
        try {
            const res = await api.post<PipelineGraphValidationResponse>(
                `/projects/${projectId}/pipeline/graph/validate`,
                {},
            );
            setValidateResult(res.data);
            setErrorMessage('');
            await loadGraph();
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setActionLoading(null);
        }
    };

    const handleDryRun = async () => {
        setActionLoading('dry-run');
        try {
            const res = await api.post<PipelineGraphDryRunResponse>(
                `/projects/${projectId}/pipeline/graph/dry-run`,
                {},
            );
            setDryRunResult(res.data);
            setErrorMessage('');
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setActionLoading(null);
        }
    };

    const handleRunStep = async () => {
        setActionLoading('run-step');
        try {
            const res = await api.post<PipelineGraphRunStepResponse>(
                `/projects/${projectId}/pipeline/graph/run-step`,
                {},
            );
            setRunResult(res.data);
            setErrorMessage('');
            if (res.data.advanced) {
                await onPipelineUpdated?.();
            }
            await loadGraph();
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setActionLoading(null);
        }
    };

    return (
        <div className="card pipeline-graph-card">
            <div className="pipeline-graph-header">
                <div>
                    <h3>Workflow Graph Preview</h3>
                    <p>Phase 2 contract runtime actions: validate, dry-run, and run active step.</p>
                </div>
                {graph && <span className="badge badge-info">{graph.mode}</span>}
            </div>

            <div className="pipeline-graph-actions">
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleValidate}
                    disabled={actionLoading !== null}
                >
                    {actionLoading === 'validate' ? 'Validating...' : 'Validate Graph'}
                </button>
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleDryRun}
                    disabled={actionLoading !== null}
                >
                    {actionLoading === 'dry-run' ? 'Running Dry-Run...' : 'Dry Run'}
                </button>
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleRunStep}
                    disabled={actionLoading !== null}
                >
                    {actionLoading === 'run-step' ? 'Running Step...' : 'Run Active Step'}
                </button>
            </div>

            {errorMessage && (
                <div className="pipeline-graph-alert pipeline-graph-alert--error">{errorMessage}</div>
            )}
            {isLoadingGraph && !graph && (
                <div className="pipeline-graph-alert">Loading workflow graph...</div>
            )}

            {validateResult && (
                <div className={`pipeline-graph-result ${validateResult.valid ? 'ok' : 'error'}`}>
                    <strong>Validate:</strong> {validateResult.valid ? 'valid graph' : 'invalid graph'} | fallback:{' '}
                    {validateResult.fallback_used ? 'yes' : 'no'} | errors: {validateResult.errors.length}
                </div>
            )}
            {dryRunResult && (
                <div className={`pipeline-graph-result ${dryRunResult.active_step?.can_run_now ? 'ok' : 'warn'}`}>
                    <strong>Dry Run:</strong> active stage `{dryRunResult.current_stage}` | ready:{' '}
                    {dryRunResult.active_step?.can_run_now ? 'yes' : 'no'} | missing:{' '}
                    {dryRunResult.active_step?.missing_inputs.length ?? 0} | runtime-missing:{' '}
                    {dryRunResult.active_step?.missing_runtime_requirements.length ?? 0}
                </div>
            )}
            {runResult && (
                <div className={`pipeline-graph-result ${runResult.status === 'completed' ? 'ok' : 'warn'}`}>
                    <strong>Run Step:</strong> status `{runResult.status}` | advanced:{' '}
                    {runResult.advanced ? 'yes' : 'no'} | current stage: `{runResult.current_stage}` | outputs:{' '}
                    {runResult.published_artifact_keys?.length || 0}
                </div>
            )}

            {graph && (
                <div className="pipeline-graph-scroll">
                    <div className="pipeline-graph-track">
                        {orderedNodes.map((node, index) => (
                            <Fragment key={node.id}>
                                <article className={`pipeline-graph-node pipeline-graph-node--${node.status}`}>
                                    <header className="pipeline-graph-node-head">
                                        <span className="pipeline-graph-node-index">{node.index + 1}</span>
                                        <span className={`pipeline-graph-node-status pipeline-graph-node-status--${node.status}`}>
                                            {node.status}
                                        </span>
                                    </header>

                                    <h4>{node.display_name}</h4>
                                    <p className="pipeline-graph-node-description">{node.description}</p>
                                    <div className="pipeline-graph-node-meta">
                                        <code>{node.step_type}</code>
                                        <span>{node.config_schema_ref}</span>
                                    </div>
                                    <div className="pipeline-graph-node-meta">
                                        <span>modes: {node.runtime_requirements?.execution_modes?.join(', ') || 'local'}</span>
                                        <span>gpu: {node.runtime_requirements?.requires_gpu ? 'yes' : 'no'}</span>
                                    </div>
                                    <div className="pipeline-graph-node-artifacts">
                                        <section>
                                            <span>In</span>
                                            <p>{node.input_artifacts.join(', ')}</p>
                                        </section>
                                        <section>
                                            <span>Out</span>
                                            <p>{node.output_artifacts.join(', ')}</p>
                                        </section>
                                    </div>
                                </article>
                                {index < orderedNodes.length - 1 && (
                                    <div className="pipeline-graph-connector" aria-hidden="true">
                                        <span />
                                    </div>
                                )}
                            </Fragment>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
