import { useCallback, useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type {
    PipelineGraphCompileResponse,
    PipelineGraphContractResetResponse,
    PipelineGraphContractResponse,
    PipelineGraphContractSaveResponse,
    PipelineGraphEdge,
    PipelineGraphNode,
    PipelineGraphResponse,
    PipelineGraphStageCatalogResponse,
    PipelineGraphStageTemplate,
    PipelineGraphValidationResponse,
    PipelineStage,
} from '../../types';
import './PipelineGraphEditor.css';

interface PipelineGraphEditorProps {
    projectId: number;
    currentStage: PipelineStage;
}

function parseArtifacts(value: string): string[] {
    return value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
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
    return 'Operation failed.';
}

function cloneGraph(graph: PipelineGraphResponse): PipelineGraphResponse {
    return {
        ...graph,
        nodes: [...graph.nodes].sort((a, b) => a.index - b.index).map((node, index) => ({
            ...node,
            index,
            position: {
                x: index * 280,
                y: 0,
            },
        })),
        edges: [...graph.edges],
    };
}

export default function PipelineGraphEditor({ projectId, currentStage }: PipelineGraphEditorProps) {
    const [isLoading, setIsLoading] = useState(false);
    const [isBusy, setIsBusy] = useState<null | 'validate' | 'compile' | 'save' | 'reset'>(null);
    const [errorMessage, setErrorMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const [catalog, setCatalog] = useState<PipelineGraphStageTemplate[]>([]);
    const [draftGraph, setDraftGraph] = useState<PipelineGraphResponse | null>(null);
    const [contractMeta, setContractMeta] = useState<{
        hasSavedOverride: boolean;
        requestedSource: string;
        effectiveSource: string;
    } | null>(null);

    const [validateResult, setValidateResult] = useState<PipelineGraphValidationResponse | null>(null);
    const [compileResult, setCompileResult] = useState<PipelineGraphCompileResponse | null>(null);

    const loadEditorState = useCallback(async () => {
        setIsLoading(true);
        try {
            const [contractRes, catalogRes] = await Promise.all([
                api.get<PipelineGraphContractResponse>(`/projects/${projectId}/pipeline/graph/contract`),
                api.get<PipelineGraphStageCatalogResponse>(`/projects/${projectId}/pipeline/graph/stage-catalog`),
            ]);
            setDraftGraph(cloneGraph(contractRes.data.graph));
            setCatalog(catalogRes.data.stages || []);
            setContractMeta({
                hasSavedOverride: contractRes.data.has_saved_override,
                requestedSource: contractRes.data.requested_source,
                effectiveSource: contractRes.data.effective_source,
            });
            setErrorMessage('');
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void loadEditorState();
    }, [loadEditorState, currentStage]);

    const stageToTemplate = useMemo(() => {
        const map = new Map<PipelineStage, PipelineGraphStageTemplate>();
        for (const template of catalog) {
            map.set(template.stage, template);
        }
        return map;
    }, [catalog]);

    const handleNodeChange = (nodeId: string, patch: Partial<PipelineGraphNode>) => {
        setDraftGraph((prev) => {
            if (!prev) {
                return prev;
            }
            return {
                ...prev,
                nodes: prev.nodes.map((node) => (node.id === nodeId ? { ...node, ...patch } : node)),
            };
        });
    };

    const handleMoveNode = (nodeId: string, direction: -1 | 1) => {
        setDraftGraph((prev) => {
            if (!prev) {
                return prev;
            }
            const ordered = [...prev.nodes].sort((a, b) => a.index - b.index);
            const idx = ordered.findIndex((node) => node.id === nodeId);
            const target = idx + direction;
            if (idx < 0 || target < 0 || target >= ordered.length) {
                return prev;
            }
            const temp = ordered[idx];
            ordered[idx] = ordered[target];
            ordered[target] = temp;
            const normalizedNodes = ordered.map((node, index) => ({
                ...node,
                index,
                position: { x: index * 280, y: 0 },
            }));
            return {
                ...prev,
                nodes: normalizedNodes,
            };
        });
    };

    const handleDeleteNode = (nodeId: string) => {
        setDraftGraph((prev) => {
            if (!prev) {
                return prev;
            }
            const remainingNodes = prev.nodes.filter((node) => node.id !== nodeId);
            const normalizedNodes = remainingNodes
                .sort((a, b) => a.index - b.index)
                .map((node, index) => ({ ...node, index, position: { x: index * 280, y: 0 } }));
            const remainingEdges = prev.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId);
            return {
                ...prev,
                nodes: normalizedNodes,
                edges: remainingEdges,
            };
        });
    };

    const handleAddNode = (stage: PipelineStage) => {
        const template = stageToTemplate.get(stage);
        if (!template) {
            return;
        }
        setDraftGraph((prev) => {
            if (!prev) {
                return prev;
            }
            if (prev.nodes.some((node) => node.stage === stage)) {
                setErrorMessage(`Stage '${stage}' already exists in graph.`);
                return prev;
            }
            const index = prev.nodes.length;
            const nodeId = `step:${stage}:${Date.now()}`;
            const newNode: PipelineGraphNode = {
                id: nodeId,
                stage: template.stage,
                display_name: template.display_name,
                index,
                kind: 'custom_step',
                status: 'pending',
                step_type: template.step_type,
                description: template.description,
                input_artifacts: [...template.input_artifacts],
                output_artifacts: [...template.output_artifacts],
                config_schema_ref: template.config_schema_ref,
                position: { x: index * 280, y: 0 },
            };
            setErrorMessage('');
            return {
                ...prev,
                nodes: [...prev.nodes, newNode],
            };
        });
    };

    const handleAddEdge = () => {
        setDraftGraph((prev) => {
            if (!prev) {
                return prev;
            }
            const ordered = [...prev.nodes].sort((a, b) => a.index - b.index);
            if (ordered.length < 2) {
                return prev;
            }
            const source = ordered[ordered.length - 2].id;
            const target = ordered[ordered.length - 1].id;
            if (prev.edges.some((edge) => edge.source === source && edge.target === target)) {
                return prev;
            }
            const edgeId = `edge:${source}->${target}:${Date.now()}`;
            const edge: PipelineGraphEdge = {
                id: edgeId,
                source,
                target,
                kind: 'contract_edge',
            };
            return {
                ...prev,
                edges: [...prev.edges, edge],
            };
        });
    };

    const handleEdgeChange = (edgeId: string, patch: Partial<PipelineGraphEdge>) => {
        setDraftGraph((prev) => {
            if (!prev) {
                return prev;
            }
            return {
                ...prev,
                edges: prev.edges.map((edge) => (edge.id === edgeId ? { ...edge, ...patch } : edge)),
            };
        });
    };

    const handleDeleteEdge = (edgeId: string) => {
        setDraftGraph((prev) => {
            if (!prev) {
                return prev;
            }
            return {
                ...prev,
                edges: prev.edges.filter((edge) => edge.id !== edgeId),
            };
        });
    };

    const handleValidate = async () => {
        if (!draftGraph) {
            return;
        }
        setIsBusy('validate');
        try {
            const res = await api.post<PipelineGraphValidationResponse>(
                `/projects/${projectId}/pipeline/graph/validate`,
                {
                    graph: draftGraph,
                    allow_fallback: false,
                },
            );
            setValidateResult(res.data);
            setStatusMessage('Graph validation completed.');
            setErrorMessage('');
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsBusy(null);
        }
    };

    const handleCompile = async () => {
        if (!draftGraph) {
            return;
        }
        setIsBusy('compile');
        try {
            const res = await api.post<PipelineGraphCompileResponse>(
                `/projects/${projectId}/pipeline/graph/compile`,
                {
                    graph: draftGraph,
                    allow_fallback: false,
                    use_saved_override: false,
                },
            );
            setCompileResult(res.data);
            setStatusMessage('Compile checks completed.');
            setErrorMessage('');
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsBusy(null);
        }
    };

    const handleSave = async () => {
        if (!draftGraph) {
            return;
        }
        setIsBusy('save');
        try {
            const res = await api.put<PipelineGraphContractSaveResponse>(
                `/projects/${projectId}/pipeline/graph/contract`,
                { graph: draftGraph },
            );
            setDraftGraph(cloneGraph(res.data.graph));
            setContractMeta({
                hasSavedOverride: true,
                requestedSource: 'saved_override',
                effectiveSource: 'saved_override',
            });
            setStatusMessage('Graph contract saved.');
            setErrorMessage('');
            await loadEditorState();
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsBusy(null);
        }
    };

    const handleReset = async () => {
        setIsBusy('reset');
        try {
            const res = await api.delete<PipelineGraphContractResetResponse>(
                `/projects/${projectId}/pipeline/graph/contract`,
            );
            if (res.data.reset) {
                setStatusMessage('Saved graph override reset to default.');
            } else {
                setStatusMessage('No saved override found; default graph already active.');
            }
            setErrorMessage('');
            await loadEditorState();
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
        } finally {
            setIsBusy(null);
        }
    };

    const orderedNodes = useMemo(
        () => (draftGraph ? [...draftGraph.nodes].sort((a, b) => a.index - b.index) : []),
        [draftGraph],
    );
    const nodeIds = orderedNodes.map((node) => node.id);

    return (
        <div className="card pipeline-editor-card">
            <div className="pipeline-editor-header">
                <div>
                    <h3>Visual Pipeline Editor</h3>
                    <p>Phase 3: edit node contracts, graph edges, compile, then save as project override.</p>
                </div>
                {contractMeta && (
                    <span className="badge badge-info">
                        source: {contractMeta.effectiveSource}
                    </span>
                )}
            </div>

            {errorMessage && <div className="pipeline-editor-alert pipeline-editor-alert--error">{errorMessage}</div>}
            {statusMessage && <div className="pipeline-editor-alert pipeline-editor-alert--ok">{statusMessage}</div>}
            {isLoading && <div className="pipeline-editor-alert">Loading graph editor...</div>}

            {!isLoading && draftGraph && (
                <>
                    <div className="pipeline-editor-toolbar">
                        <select
                            className="input pipeline-editor-select"
                            onChange={(event) => {
                                const stage = event.target.value as PipelineStage;
                                if (stage) {
                                    handleAddNode(stage);
                                    event.target.value = '';
                                }
                            }}
                            defaultValue=""
                            disabled={isBusy !== null}
                        >
                            <option value="">Add Stage Node...</option>
                            {catalog.map((template) => (
                                <option key={template.stage} value={template.stage}>
                                    {template.display_name}
                                </option>
                            ))}
                        </select>
                        <button type="button" className="btn btn-secondary" onClick={handleAddEdge} disabled={isBusy !== null}>
                            Add Edge
                        </button>
                        <button type="button" className="btn btn-secondary" onClick={handleValidate} disabled={isBusy !== null}>
                            {isBusy === 'validate' ? 'Validating...' : 'Validate'}
                        </button>
                        <button type="button" className="btn btn-secondary" onClick={handleCompile} disabled={isBusy !== null}>
                            {isBusy === 'compile' ? 'Compiling...' : 'Compile'}
                        </button>
                        <button type="button" className="btn btn-primary" onClick={handleSave} disabled={isBusy !== null}>
                            {isBusy === 'save' ? 'Saving...' : 'Save Contract'}
                        </button>
                        <button type="button" className="btn btn-ghost" onClick={handleReset} disabled={isBusy !== null}>
                            {isBusy === 'reset' ? 'Resetting...' : 'Reset to Default'}
                        </button>
                    </div>

                    <div className="pipeline-editor-metadata">
                        <span>Saved override: {contractMeta?.hasSavedOverride ? 'yes' : 'no'}</span>
                        <span>Requested source: {contractMeta?.requestedSource || 'default'}</span>
                        <span>Nodes: {orderedNodes.length}</span>
                        <span>Edges: {draftGraph.edges.length}</span>
                    </div>

                    <div className="pipeline-editor-section">
                        <h4>Nodes</h4>
                        <div className="pipeline-editor-nodes">
                            {orderedNodes.map((node) => (
                                <article key={node.id} className="pipeline-editor-node">
                                    <div className="pipeline-editor-node-header">
                                        <strong>{node.display_name}</strong>
                                        <div className="pipeline-editor-node-actions">
                                            <button type="button" className="btn btn-ghost" onClick={() => handleMoveNode(node.id, -1)}>
                                                ↑
                                            </button>
                                            <button type="button" className="btn btn-ghost" onClick={() => handleMoveNode(node.id, 1)}>
                                                ↓
                                            </button>
                                            <button type="button" className="btn btn-ghost" onClick={() => handleDeleteNode(node.id)}>
                                                Remove
                                            </button>
                                        </div>
                                    </div>
                                    <div className="pipeline-editor-grid">
                                        <label>
                                            Stage
                                            <input className="input" value={node.stage} readOnly />
                                        </label>
                                        <label>
                                            Step Type
                                            <input
                                                className="input"
                                                value={node.step_type}
                                                onChange={(e) => handleNodeChange(node.id, { step_type: e.target.value })}
                                            />
                                        </label>
                                        <label>
                                            Schema Ref
                                            <input
                                                className="input"
                                                value={node.config_schema_ref}
                                                onChange={(e) => handleNodeChange(node.id, { config_schema_ref: e.target.value })}
                                            />
                                        </label>
                                    </div>
                                    <label>
                                        Description
                                        <textarea
                                            className="input pipeline-editor-textarea"
                                            value={node.description}
                                            onChange={(e) => handleNodeChange(node.id, { description: e.target.value })}
                                        />
                                    </label>
                                    <div className="pipeline-editor-grid">
                                        <label>
                                            Input Artifacts (comma separated)
                                            <input
                                                className="input"
                                                value={node.input_artifacts.join(', ')}
                                                onChange={(e) => handleNodeChange(node.id, { input_artifacts: parseArtifacts(e.target.value) })}
                                            />
                                        </label>
                                        <label>
                                            Output Artifacts (comma separated)
                                            <input
                                                className="input"
                                                value={node.output_artifacts.join(', ')}
                                                onChange={(e) => handleNodeChange(node.id, { output_artifacts: parseArtifacts(e.target.value) })}
                                            />
                                        </label>
                                    </div>
                                </article>
                            ))}
                        </div>
                    </div>

                    <div className="pipeline-editor-section">
                        <h4>Edges</h4>
                        <div className="pipeline-editor-edges">
                            {draftGraph.edges.map((edge) => (
                                <div key={edge.id} className="pipeline-editor-edge-row">
                                    <select
                                        className="input"
                                        value={edge.source}
                                        onChange={(e) => handleEdgeChange(edge.id, { source: e.target.value })}
                                    >
                                        {nodeIds.map((nodeId) => (
                                            <option key={nodeId} value={nodeId}>
                                                {nodeId}
                                            </option>
                                        ))}
                                    </select>
                                    <span>→</span>
                                    <select
                                        className="input"
                                        value={edge.target}
                                        onChange={(e) => handleEdgeChange(edge.id, { target: e.target.value })}
                                    >
                                        {nodeIds.map((nodeId) => (
                                            <option key={nodeId} value={nodeId}>
                                                {nodeId}
                                            </option>
                                        ))}
                                    </select>
                                    <button type="button" className="btn btn-ghost" onClick={() => handleDeleteEdge(edge.id)}>
                                        Remove
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>

                    {validateResult && (
                        <div className={`pipeline-editor-result ${validateResult.valid ? 'ok' : 'error'}`}>
                            Validate: valid={validateResult.valid ? 'yes' : 'no'} | errors={validateResult.errors.length} | warnings={validateResult.warnings.length}
                        </div>
                    )}

                    {compileResult && (
                        <div className={`pipeline-editor-result ${compileResult.errors.length === 0 ? 'ok' : 'error'}`}>
                            Compile: active-stage-present={compileResult.checks.active_stage_present ? 'yes' : 'no'} | ready-now={compileResult.checks.active_stage_ready_now ? 'yes' : 'no'} | errors={compileResult.errors.length} | warnings={compileResult.warnings.length}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
