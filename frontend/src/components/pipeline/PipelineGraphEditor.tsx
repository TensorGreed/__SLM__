import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type DragEvent as ReactDragEvent,
    type MouseEvent as ReactMouseEvent,
} from 'react';

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
    PipelineGraphStage,
    PipelineGraphStageTemplate,
    PipelineGraphTemplate,
    PipelineGraphTemplateListResponse,
    PipelineGraphValidationResponse,
    PipelineStage,
    StepRuntimeRequirements,
} from '../../types';
import './PipelineGraphEditor.css';

interface PipelineGraphEditorProps {
    projectId: number;
    currentStage: PipelineStage;
}

interface CanvasEdgePath {
    id: string;
    sourceId: string;
    targetId: string;
    d: string;
    midX: number;
    midY: number;
}

interface NodeAddPosition {
    x: number;
    y: number;
}

interface NodeConfigPreset {
    id: string;
    label: string;
    description: string;
    config: Record<string, unknown>;
}

const NODE_WIDTH = 260;
const NODE_HEIGHT = 170;
const DEFAULT_NODE_SPACING_X = 320;
const DEFAULT_NODE_SPACING_Y = 240;

const DEFAULT_RUNTIME_REQUIREMENTS: StepRuntimeRequirements = {
    execution_modes: ['local'],
    required_services: [],
    required_env: [],
    required_settings: [],
    requires_gpu: false,
    min_vram_gb: 0,
};

function parseList(value: string): string[] {
    return value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
}

function normalizeRuntimeRequirements(value: StepRuntimeRequirements | null | undefined): StepRuntimeRequirements {
    return {
        ...DEFAULT_RUNTIME_REQUIREMENTS,
        ...(value || {}),
        execution_modes: Array.isArray(value?.execution_modes) && value.execution_modes.length > 0
            ? [...value.execution_modes]
            : [...DEFAULT_RUNTIME_REQUIREMENTS.execution_modes],
        required_services: Array.isArray(value?.required_services) ? [...value.required_services] : [],
        required_env: Array.isArray(value?.required_env) ? [...value.required_env] : [],
        required_settings: Array.isArray(value?.required_settings) ? [...value.required_settings] : [],
        requires_gpu: Boolean(value?.requires_gpu),
        min_vram_gb: typeof value?.min_vram_gb === 'number' ? value.min_vram_gb : 0,
    };
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

function stringifyJsonObject(value: unknown): string {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
        return JSON.stringify(value, null, 2);
    }
    return '{}';
}

function parseJsonObjectInput(raw: string): { value: Record<string, unknown>; error: string } {
    const text = raw.trim();
    if (!text) {
        return { value: {}, error: '' };
    }
    try {
        const parsed = JSON.parse(text);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            return { value: {}, error: 'Config JSON must be an object.' };
        }
        return { value: parsed as Record<string, unknown>, error: '' };
    } catch (error) {
        if (error instanceof Error) {
            return { value: {}, error: error.message };
        }
        return { value: {}, error: 'Invalid JSON.' };
    }
}

function getNodeConfigPresets(stepType: string): NodeConfigPreset[] {
    const normalized = (stepType || '').trim().toLowerCase();
    if (normalized === 'core.training') {
        return [
            {
                id: 'training.noop',
                label: 'Training No-op',
                description: 'Safe default: keep stage non-executing.',
                config: {
                    mode: 'noop',
                },
            },
            {
                id: 'training.create-start',
                label: 'Create + Start',
                description: 'Create a new experiment and dispatch training.',
                config: {
                    mode: 'create_and_start',
                    name: 'workflow-train',
                    base_model: 'microsoft/phi-3-mini-4k-instruct',
                    training_mode: 'sft',
                    config: {
                        num_epochs: 3,
                        batch_size: 2,
                        gradient_accumulation_steps: 8,
                        learning_rate: 0.0002,
                        max_seq_length: 2048,
                        use_lora: true,
                        lora_r: 16,
                    },
                    wait_for_terminal: false,
                },
            },
            {
                id: 'training.start-existing',
                label: 'Start Existing',
                description: 'Start an already-created experiment by id.',
                config: {
                    mode: 'start_existing',
                    experiment_id: 1,
                    wait_for_terminal: false,
                },
            },
        ];
    }
    if (normalized === 'core.evaluation') {
        return [
            {
                id: 'evaluation.noop',
                label: 'Evaluation No-op',
                description: 'Safe default: keep stage non-executing.',
                config: {
                    mode: 'noop',
                },
            },
            {
                id: 'evaluation.heldout',
                label: 'Heldout Eval',
                description: 'Run heldout evaluation on latest completed experiment.',
                config: {
                    mode: 'heldout',
                    dataset_name: 'test',
                    eval_type: 'exact_match',
                    max_samples: 100,
                    max_new_tokens: 128,
                    temperature: 0,
                    require_completed_experiment: true,
                },
            },
        ];
    }
    if (normalized === 'core.export') {
        return [
            {
                id: 'export.noop',
                label: 'Export No-op',
                description: 'Safe default: keep stage non-executing.',
                config: {
                    mode: 'noop',
                },
            },
            {
                id: 'export.create-run',
                label: 'Create + Run',
                description: 'Create an export and run packaging.',
                config: {
                    mode: 'create_and_run',
                    export_format: 'gguf',
                    quantization: '4bit',
                    require_completed_experiment: true,
                },
            },
            {
                id: 'export.run-existing',
                label: 'Run Existing',
                description: 'Run an existing export record by id.',
                config: {
                    mode: 'run_existing',
                    export_id: 1,
                },
            },
        ];
    }
    return [];
}

function computeNodePosition(index: number): { x: number; y: number } {
    const col = index % 4;
    const row = Math.floor(index / 4);
    return {
        x: 40 + col * DEFAULT_NODE_SPACING_X,
        y: 40 + row * DEFAULT_NODE_SPACING_Y,
    };
}

function cloneGraph(graph: PipelineGraphResponse): PipelineGraphResponse {
    const sorted = [...graph.nodes].sort((a, b) => a.index - b.index);
    return {
        ...graph,
        nodes: sorted.map((node, index) => ({
            ...node,
            index,
            runtime_requirements: normalizeRuntimeRequirements(node.runtime_requirements),
            position: node.position || computeNodePosition(index),
        })),
        edges: [...graph.edges],
    };
}

function normalizeGraphForSave(graph: PipelineGraphResponse): PipelineGraphResponse {
    const sorted = [...graph.nodes].sort((a, b) => {
        if (a.position.y === b.position.y) {
            return a.position.x - b.position.x;
        }
        return a.position.y - b.position.y;
    });
    const nodeIdOrder = new Map(sorted.map((node, idx) => [node.id, idx]));

    return {
        ...graph,
        nodes: sorted.map((node, index) => ({
            ...node,
            index,
            runtime_requirements: normalizeRuntimeRequirements(node.runtime_requirements),
        })),
        edges: graph.edges
            .filter((edge) => nodeIdOrder.has(edge.source) && nodeIdOrder.has(edge.target))
            .sort((a, b) => {
                const aSource = nodeIdOrder.get(a.source) ?? 0;
                const bSource = nodeIdOrder.get(b.source) ?? 0;
                if (aSource !== bSource) {
                    return aSource - bSource;
                }
                const aTarget = nodeIdOrder.get(a.target) ?? 0;
                const bTarget = nodeIdOrder.get(b.target) ?? 0;
                return aTarget - bTarget;
            }),
    };
}

export default function PipelineGraphEditor({ projectId, currentStage }: PipelineGraphEditorProps) {
    const [isLoading, setIsLoading] = useState(false);
    const [isBusy, setIsBusy] = useState<null | 'validate' | 'compile' | 'save' | 'reset'>(null);
    const [errorMessage, setErrorMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const [catalog, setCatalog] = useState<PipelineGraphStageTemplate[]>([]);
    const [templates, setTemplates] = useState<PipelineGraphTemplate[]>([]);
    const [selectedTemplateId, setSelectedTemplateId] = useState('__current__');
    const [draftGraph, setDraftGraph] = useState<PipelineGraphResponse | null>(null);
    const [contractMeta, setContractMeta] = useState<{
        hasSavedOverride: boolean;
        requestedSource: string;
        effectiveSource: string;
    } | null>(null);

    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
    const [connectSourceNodeId, setConnectSourceNodeId] = useState<string | null>(null);
    const [isInspectorOpen, setIsInspectorOpen] = useState(true);
    const [isCanvasDragOver, setIsCanvasDragOver] = useState(false);
    const [nodeConfigDraft, setNodeConfigDraft] = useState('{}');
    const [nodeConfigError, setNodeConfigError] = useState('');

    const [validateResult, setValidateResult] = useState<PipelineGraphValidationResponse | null>(null);
    const [compileResult, setCompileResult] = useState<PipelineGraphCompileResponse | null>(null);

    const canvasRef = useRef<HTMLDivElement | null>(null);
    const dragStateRef = useRef<{
        nodeId: string;
        offsetX: number;
        offsetY: number;
    } | null>(null);

    const markDirty = useCallback(() => {
        setValidateResult(null);
        setCompileResult(null);
    }, []);

    const updateGraph = useCallback((updater: (graph: PipelineGraphResponse) => PipelineGraphResponse) => {
        setDraftGraph((prev) => {
            if (!prev) {
                return prev;
            }
            return updater(prev);
        });
        markDirty();
    }, [markDirty]);

    const loadEditorState = useCallback(async () => {
        setIsLoading(true);
        try {
            const [contractRes, catalogRes, templateRes] = await Promise.all([
                api.get<PipelineGraphContractResponse>(`/projects/${projectId}/pipeline/graph/contract`),
                api.get<PipelineGraphStageCatalogResponse>(`/projects/${projectId}/pipeline/graph/stage-catalog`),
                api.get<PipelineGraphTemplateListResponse>(`/projects/${projectId}/pipeline/graph/templates`),
            ]);
            const nextGraph = cloneGraph(contractRes.data.graph);
            setDraftGraph(nextGraph);
            setCatalog(catalogRes.data.stages || []);
            setTemplates(templateRes.data.templates || []);
            setContractMeta({
                hasSavedOverride: contractRes.data.has_saved_override,
                requestedSource: contractRes.data.requested_source,
                effectiveSource: contractRes.data.effective_source,
            });
            setSelectedNodeId(nextGraph.nodes[0]?.id || null);
            setSelectedEdgeId(null);
            setConnectSourceNodeId(null);
            setSelectedTemplateId('__current__');
            setIsCanvasDragOver(false);
            setValidateResult(null);
            setCompileResult(null);
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
        const map = new Map<PipelineGraphStage, PipelineGraphStageTemplate>();
        for (const template of catalog) {
            map.set(template.stage, template);
        }
        return map;
    }, [catalog]);

    const nodeMap = useMemo(() => {
        const map = new Map<string, PipelineGraphNode>();
        for (const node of draftGraph?.nodes || []) {
            map.set(node.id, node);
        }
        return map;
    }, [draftGraph]);

    const selectedNode = useMemo(
        () => (selectedNodeId ? nodeMap.get(selectedNodeId) || null : null),
        [nodeMap, selectedNodeId],
    );
    const selectedNodePresets = useMemo(
        () => getNodeConfigPresets(selectedNode?.step_type || ''),
        [selectedNode?.step_type],
    );

    const selectedEdge = useMemo(
        () => draftGraph?.edges.find((edge) => edge.id === selectedEdgeId) || null,
        [draftGraph, selectedEdgeId],
    );

    useEffect(() => {
        if (!selectedNode) {
            setNodeConfigDraft('{}');
            setNodeConfigError('');
            return;
        }
        setNodeConfigDraft(stringifyJsonObject(selectedNode.config));
        setNodeConfigError('');
    }, [selectedNode]);

    const canvasEdges = useMemo<CanvasEdgePath[]>(() => {
        if (!draftGraph) {
            return [];
        }

        const result: CanvasEdgePath[] = [];
        for (const edge of draftGraph.edges) {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            if (!source || !target) {
                continue;
            }
            const startX = source.position.x + NODE_WIDTH;
            const startY = source.position.y + NODE_HEIGHT / 2;
            const endX = target.position.x;
            const endY = target.position.y + NODE_HEIGHT / 2;
            const delta = Math.max(80, Math.abs(endX - startX) * 0.4);
            const c1x = startX + delta;
            const c1y = startY;
            const c2x = endX - delta;
            const c2y = endY;
            const d = `M ${startX} ${startY} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${endX} ${endY}`;
            const midX = (startX + endX) / 2;
            const midY = (startY + endY) / 2;
            result.push({
                id: edge.id,
                sourceId: edge.source,
                targetId: edge.target,
                d,
                midX,
                midY,
            });
        }

        return result;
    }, [draftGraph, nodeMap]);

    const canvasBounds = useMemo(() => {
        const nodes = draftGraph?.nodes || [];
        if (nodes.length === 0) {
            return {
                width: 1200,
                height: 720,
            };
        }

        let maxX = 0;
        let maxY = 0;
        for (const node of nodes) {
            maxX = Math.max(maxX, node.position.x + NODE_WIDTH + 40);
            maxY = Math.max(maxY, node.position.y + NODE_HEIGHT + 40);
        }

        return {
            width: Math.max(1200, maxX),
            height: Math.max(720, maxY),
        };
    }, [draftGraph]);

    const updateNodePatch = useCallback((nodeId: string, patch: Partial<PipelineGraphNode>) => {
        updateGraph((prev) => ({
            ...prev,
            nodes: prev.nodes.map((node) => (node.id === nodeId ? { ...node, ...patch } : node)),
        }));
    }, [updateGraph]);

    const deleteNode = useCallback((nodeId: string) => {
        updateGraph((prev) => {
            const nextNodes = prev.nodes.filter((node) => node.id !== nodeId);
            const nextEdges = prev.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId);
            return {
                ...prev,
                nodes: nextNodes,
                edges: nextEdges,
            };
        });

        setSelectedNodeId((prev) => (prev === nodeId ? null : prev));
        setSelectedEdgeId(null);
        setConnectSourceNodeId((prev) => (prev === nodeId ? null : prev));
    }, [updateGraph]);

    const createEdge = useCallback((sourceId: string, targetId: string) => {
        if (!sourceId || !targetId || sourceId === targetId) {
            return;
        }
        updateGraph((prev) => {
            const edgeExists = prev.edges.some((edge) => edge.source === sourceId && edge.target === targetId);
            if (edgeExists) {
                return prev;
            }
            const edge: PipelineGraphEdge = {
                id: `edge:${sourceId}->${targetId}:${Date.now()}`,
                source: sourceId,
                target: targetId,
                kind: 'contract_edge',
            };
            return {
                ...prev,
                edges: [...prev.edges, edge],
            };
        });
    }, [updateGraph]);

    const deleteEdge = useCallback((edgeId: string) => {
        updateGraph((prev) => ({
            ...prev,
            edges: prev.edges.filter((edge) => edge.id !== edgeId),
        }));
        setSelectedEdgeId((prev) => (prev === edgeId ? null : prev));
    }, [updateGraph]);

    const handleCanvasClick = useCallback(() => {
        setSelectedNodeId(null);
        setSelectedEdgeId(null);
        setConnectSourceNodeId(null);
    }, []);

    const beginNodeDrag = useCallback((event: ReactMouseEvent<HTMLElement>, nodeId: string) => {
        if (isBusy !== null) {
            return;
        }
        const canvasRect = canvasRef.current?.getBoundingClientRect();
        const node = nodeMap.get(nodeId);
        if (!canvasRect || !node) {
            return;
        }
        event.preventDefault();
        dragStateRef.current = {
            nodeId,
            offsetX: event.clientX - canvasRect.left - node.position.x,
            offsetY: event.clientY - canvasRect.top - node.position.y,
        };
    }, [isBusy, nodeMap]);

    useEffect(() => {
        const handleMouseMove = (event: MouseEvent) => {
            const dragState = dragStateRef.current;
            if (!dragState || !canvasRef.current) {
                return;
            }
            const canvasRect = canvasRef.current.getBoundingClientRect();
            const rawX = event.clientX - canvasRect.left - dragState.offsetX;
            const rawY = event.clientY - canvasRect.top - dragState.offsetY;

            const nextX = Math.max(0, Math.min(rawX, canvasBounds.width - NODE_WIDTH));
            const nextY = Math.max(0, Math.min(rawY, canvasBounds.height - NODE_HEIGHT));

            setDraftGraph((prev) => {
                if (!prev) {
                    return prev;
                }
                return {
                    ...prev,
                    nodes: prev.nodes.map((node) => {
                        if (node.id !== dragState.nodeId) {
                            return node;
                        }
                        return {
                            ...node,
                            position: {
                                x: nextX,
                                y: nextY,
                            },
                        };
                    }),
                };
            });
            markDirty();
        };

        const handleMouseUp = () => {
            dragStateRef.current = null;
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [canvasBounds.height, canvasBounds.width, markDirty]);

    const autoLayout = useCallback(() => {
        updateGraph((prev) => ({
            ...prev,
            nodes: [...prev.nodes]
                .sort((a, b) => a.index - b.index)
                .map((node, index) => ({
                    ...node,
                    position: computeNodePosition(index),
                })),
        }));
    }, [updateGraph]);

    const handleAddNode = useCallback((stage: PipelineGraphStage, customPosition?: NodeAddPosition) => {
        const template = stageToTemplate.get(stage);
        if (!template) {
            return;
        }

        updateGraph((prev) => {
            if (prev.nodes.some((node) => node.stage === stage)) {
                setErrorMessage(`Stage '${stage}' already exists in graph.`);
                return prev;
            }
            const index = prev.nodes.length;
            const defaultPosition = computeNodePosition(index);
            const position = customPosition
                ? {
                    x: Math.max(0, Math.min(customPosition.x, canvasBounds.width - NODE_WIDTH)),
                    y: Math.max(0, Math.min(customPosition.y, canvasBounds.height - NODE_HEIGHT)),
                }
                : defaultPosition;
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
                config: {},
                runtime_requirements: normalizeRuntimeRequirements(template.runtime_requirements),
                position,
            };
            setStatusMessage(`Added stage node '${template.display_name}'.`);
            setErrorMessage('');
            setSelectedNodeId(nodeId);
            setSelectedEdgeId(null);
            return {
                ...prev,
                nodes: [...prev.nodes, newNode],
            };
        });
    }, [canvasBounds.height, canvasBounds.width, stageToTemplate, updateGraph]);

    const handleTemplateSourceChange = useCallback((nextTemplateId: string) => {
        setSelectedTemplateId(nextTemplateId);
        if (nextTemplateId === '__current__') {
            setStatusMessage('Loaded current saved/default graph.');
            setErrorMessage('');
            void loadEditorState();
            return;
        }

        const template = templates.find((item) => item.template_id === nextTemplateId);
        if (!template) {
            setErrorMessage(`Template '${nextTemplateId}' not found.`);
            return;
        }

        const nextGraph = cloneGraph(template.graph);
        setDraftGraph(nextGraph);
        setSelectedNodeId(nextGraph.nodes[0]?.id || null);
        setSelectedEdgeId(null);
        setConnectSourceNodeId(null);
        setValidateResult(null);
        setCompileResult(null);
        setStatusMessage(`Template '${template.display_name}' loaded as active graph source.`);
        setErrorMessage('');
    }, [loadEditorState, templates]);

    const handlePaletteDragStart = useCallback((event: ReactDragEvent<HTMLButtonElement>, stage: PipelineGraphStage) => {
        if (isBusy !== null) {
            return;
        }
        event.dataTransfer.setData('application/x-slm-stage', stage);
        event.dataTransfer.effectAllowed = 'copyMove';
    }, [isBusy]);

    const handleCanvasDragOver = useCallback((event: ReactDragEvent<HTMLDivElement>) => {
        event.preventDefault();
        if (isBusy !== null) {
            return;
        }
        event.dataTransfer.dropEffect = 'copy';
        setIsCanvasDragOver(true);
    }, [isBusy]);

    const handleCanvasDragLeave = useCallback(() => {
        setIsCanvasDragOver(false);
    }, []);

    const handleCanvasDrop = useCallback((event: ReactDragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsCanvasDragOver(false);
        if (isBusy !== null || !canvasRef.current) {
            return;
        }
        const stageRaw = event.dataTransfer.getData('application/x-slm-stage');
        if (!stageRaw) {
            return;
        }
        const stage = stageRaw as PipelineGraphStage;
        if (!stageToTemplate.has(stage)) {
            return;
        }

        const canvasRect = canvasRef.current.getBoundingClientRect();
        const dropX = event.clientX - canvasRect.left + canvasRef.current.scrollLeft - NODE_WIDTH / 2;
        const dropY = event.clientY - canvasRect.top + canvasRef.current.scrollTop - 26;
        handleAddNode(stage, { x: dropX, y: dropY });
    }, [handleAddNode, isBusy, stageToTemplate]);

    const handleValidate = useCallback(async () => {
        if (!draftGraph) {
            return;
        }
        setIsBusy('validate');
        try {
            const res = await api.post<PipelineGraphValidationResponse>(
                `/projects/${projectId}/pipeline/graph/validate`,
                {
                    graph: normalizeGraphForSave(draftGraph),
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
    }, [draftGraph, projectId]);

    const handleCompile = useCallback(async () => {
        if (!draftGraph) {
            return;
        }
        setIsBusy('compile');
        try {
            const res = await api.post<PipelineGraphCompileResponse>(
                `/projects/${projectId}/pipeline/graph/compile`,
                {
                    graph: normalizeGraphForSave(draftGraph),
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
    }, [draftGraph, projectId]);

    const handleSave = useCallback(async () => {
        if (!draftGraph) {
            return;
        }
        setIsBusy('save');
        try {
            const normalized = normalizeGraphForSave(draftGraph);
            const res = await api.put<PipelineGraphContractSaveResponse>(
                `/projects/${projectId}/pipeline/graph/contract`,
                { graph: normalized },
            );
            const nextGraph = cloneGraph(res.data.graph);
            setDraftGraph(nextGraph);
            setSelectedNodeId(nextGraph.nodes[0]?.id || null);
            setSelectedEdgeId(null);
            setConnectSourceNodeId(null);
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
    }, [draftGraph, loadEditorState, projectId]);

    const handleReset = useCallback(async () => {
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
    }, [loadEditorState, projectId]);

    const handleApplyNodeConfig = useCallback(() => {
        if (!selectedNode) {
            return;
        }
        const parsed = parseJsonObjectInput(nodeConfigDraft);
        if (parsed.error) {
            setNodeConfigError(parsed.error);
            return;
        }
        updateNodePatch(selectedNode.id, { config: parsed.value });
        setNodeConfigError('');
        setStatusMessage(`Updated node config for ${selectedNode.display_name}.`);
    }, [nodeConfigDraft, selectedNode, updateNodePatch]);

    const handleApplyNodePreset = useCallback((preset: NodeConfigPreset) => {
        if (!selectedNode) {
            return;
        }
        const nextText = JSON.stringify(preset.config, null, 2);
        setNodeConfigDraft(nextText);
        updateNodePatch(selectedNode.id, { config: preset.config });
        setNodeConfigError('');
        setStatusMessage(`Applied preset '${preset.label}' to ${selectedNode.display_name}.`);
    }, [selectedNode, updateNodePatch]);

    const graphStats = useMemo(() => {
        const nodeCount = draftGraph?.nodes.length || 0;
        const edgeCount = draftGraph?.edges.length || 0;
        return { nodeCount, edgeCount };
    }, [draftGraph]);

    return (
        <div className="card pipeline-editor-card">
            <div className="pipeline-editor-header">
                <div>
                    <h3>Visual Pipeline Editor</h3>
                    <p>Canvas mode: drag nodes, connect edges, inspect details, then validate/compile/save.</p>
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
                            value={selectedTemplateId}
                            onChange={(event) => handleTemplateSourceChange(event.target.value)}
                            disabled={isBusy !== null}
                        >
                            <option value="__current__">Use Saved/Default Graph</option>
                            {templates.map((template) => (
                                <option key={template.template_id} value={template.template_id}>
                                    {template.display_name}
                                </option>
                            ))}
                        </select>
                        <button type="button" className="btn btn-secondary" onClick={autoLayout} disabled={isBusy !== null}>
                            Auto Layout
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => setIsInspectorOpen((prev) => !prev)}
                            disabled={isBusy !== null}
                        >
                            {isInspectorOpen ? 'Hide Inspector' : 'Show Inspector'}
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
                        <span>Nodes: {graphStats.nodeCount}</span>
                        <span>Edges: {graphStats.edgeCount}</span>
                        <span>{connectSourceNodeId ? `Connect mode: ${connectSourceNodeId}` : 'Connect mode: off'}</span>
                    </div>

                    <div className={`pipeline-editor-layout ${isInspectorOpen ? '' : 'pipeline-editor-layout--inspector-closed'}`.trim()}>
                        <section className="pipeline-editor-canvas-wrap">
                            <header className="pipeline-editor-canvas-head">
                                <h4>Canvas</h4>
                                <p>Drag nodes by header. Drag stages from palette into canvas. Use Connect {'->'} target for edges.</p>
                            </header>
                            <div className="pipeline-editor-palette">
                                <h5>Stage Palette</h5>
                                <div className="pipeline-editor-palette-list">
                                    {catalog.map((template) => (
                                        <button
                                            key={template.stage}
                                            type="button"
                                            className="pipeline-editor-palette-item"
                                            draggable
                                            onDragStart={(event) => handlePaletteDragStart(event, template.stage)}
                                        >
                                            <strong>{template.display_name}</strong>
                                            <span>{template.stage}</span>
                                        </button>
                                    ))}
                                </div>
                            </div>
                            <div
                                ref={canvasRef}
                                className={`pipeline-editor-canvas ${isCanvasDragOver ? 'drag-over' : ''}`}
                                style={{
                                    width: canvasBounds.width,
                                    height: canvasBounds.height,
                                }}
                                onClick={handleCanvasClick}
                                onDragOver={handleCanvasDragOver}
                                onDragEnter={handleCanvasDragOver}
                                onDragLeave={handleCanvasDragLeave}
                                onDrop={handleCanvasDrop}
                            >
                                <svg className="pipeline-editor-svg" width={canvasBounds.width} height={canvasBounds.height}>
                                    <defs>
                                        <marker
                                            id="pipeline-arrow"
                                            viewBox="0 0 10 10"
                                            refX="8"
                                            refY="5"
                                            markerWidth="6"
                                            markerHeight="6"
                                            orient="auto-start-reverse"
                                        >
                                            <path d="M 0 0 L 10 5 L 0 10 z" />
                                        </marker>
                                    </defs>
                                    {canvasEdges.map((edge) => {
                                        const selected = edge.id === selectedEdgeId;
                                        const pendingConnectTarget = connectSourceNodeId && connectSourceNodeId !== edge.targetId && connectSourceNodeId !== edge.sourceId;
                                        const className = selected
                                            ? 'pipeline-editor-edge selected'
                                            : pendingConnectTarget
                                                ? 'pipeline-editor-edge muted'
                                                : 'pipeline-editor-edge';

                                        return (
                                            <g key={edge.id}>
                                                <path
                                                    className={className}
                                                    d={edge.d}
                                                    markerEnd="url(#pipeline-arrow)"
                                                    onClick={(event) => {
                                                        event.stopPropagation();
                                                        setSelectedEdgeId(edge.id);
                                                        setSelectedNodeId(null);
                                                        setConnectSourceNodeId(null);
                                                    }}
                                                />
                                                <circle
                                                    className={`pipeline-editor-edge-dot ${selected ? 'selected' : ''}`}
                                                    cx={edge.midX}
                                                    cy={edge.midY}
                                                    r={selected ? 6 : 4}
                                                />
                                            </g>
                                        );
                                    })}
                                </svg>

                                {draftGraph.nodes.map((node) => {
                                    const isSelected = selectedNodeId === node.id;
                                    const isConnectSource = connectSourceNodeId === node.id;
                                    return (
                                        <article
                                            key={node.id}
                                            className={`pipeline-editor-canvas-node ${isSelected ? 'selected' : ''} ${isConnectSource ? 'connect-source' : ''}`}
                                            style={{
                                                width: NODE_WIDTH,
                                                minHeight: NODE_HEIGHT,
                                                left: node.position.x,
                                                top: node.position.y,
                                            }}
                                            onClick={(event) => {
                                                event.stopPropagation();
                                                if (connectSourceNodeId && connectSourceNodeId !== node.id) {
                                                    createEdge(connectSourceNodeId, node.id);
                                                    setStatusMessage(`Connected ${connectSourceNodeId} -> ${node.id}.`);
                                                    setConnectSourceNodeId(null);
                                                    setSelectedEdgeId(null);
                                                    setSelectedNodeId(node.id);
                                                    return;
                                                }
                                                setSelectedNodeId(node.id);
                                                setSelectedEdgeId(null);
                                            }}
                                        >
                                            <header
                                                className="pipeline-editor-canvas-node-head"
                                                onMouseDown={(event) => beginNodeDrag(event, node.id)}
                                            >
                                                <strong>{node.display_name}</strong>
                                                <span>{node.stage}</span>
                                            </header>
                                            <div className="pipeline-editor-canvas-node-body">
                                                <code>{node.step_type}</code>
                                                <p>{node.description}</p>
                                            </div>
                                            <footer className="pipeline-editor-canvas-node-actions">
                                                <button
                                                    type="button"
                                                    className="btn btn-ghost"
                                                    onClick={(event) => {
                                                        event.stopPropagation();
                                                        setConnectSourceNodeId((prev) => (prev === node.id ? null : node.id));
                                                        setSelectedNodeId(node.id);
                                                        setSelectedEdgeId(null);
                                                    }}
                                                >
                                                    {isConnectSource ? 'Cancel Connect' : 'Connect'}
                                                </button>
                                                <button
                                                    type="button"
                                                    className="btn btn-ghost"
                                                    onClick={(event) => {
                                                        event.stopPropagation();
                                                        deleteNode(node.id);
                                                    }}
                                                >
                                                    Remove
                                                </button>
                                            </footer>
                                        </article>
                                    );
                                })}
                            </div>
                        </section>

                        {isInspectorOpen && (
                            <aside className="pipeline-editor-inspector">
                                <div className="pipeline-editor-inspector-head">
                                    <h4>Inspector</h4>
                                    <button
                                        type="button"
                                        className="btn btn-ghost"
                                        onClick={() => setIsInspectorOpen(false)}
                                    >
                                        Close
                                    </button>
                                </div>
                            {!selectedNode && !selectedEdge && (
                                <div className="pipeline-editor-inspector-empty">
                                    Select a node or edge to edit details.
                                </div>
                            )}

                            {selectedEdge && (
                                <div className="pipeline-editor-inspector-section">
                                    <h5>Selected Edge</h5>
                                    <label>
                                        Source Node
                                        <select
                                            className="input"
                                            value={selectedEdge.source}
                                            onChange={(event) => {
                                                updateGraph((prev) => ({
                                                    ...prev,
                                                    edges: prev.edges.map((edge) => (
                                                        edge.id === selectedEdge.id
                                                            ? { ...edge, source: event.target.value }
                                                            : edge
                                                    )),
                                                }));
                                            }}
                                        >
                                            {draftGraph.nodes.map((node) => (
                                                <option key={node.id} value={node.id}>
                                                    {node.display_name}
                                                </option>
                                            ))}
                                        </select>
                                    </label>
                                    <label>
                                        Target Node
                                        <select
                                            className="input"
                                            value={selectedEdge.target}
                                            onChange={(event) => {
                                                updateGraph((prev) => ({
                                                    ...prev,
                                                    edges: prev.edges.map((edge) => (
                                                        edge.id === selectedEdge.id
                                                            ? { ...edge, target: event.target.value }
                                                            : edge
                                                    )),
                                                }));
                                            }}
                                        >
                                            {draftGraph.nodes.map((node) => (
                                                <option key={node.id} value={node.id}>
                                                    {node.display_name}
                                                </option>
                                            ))}
                                        </select>
                                    </label>
                                    <button type="button" className="btn btn-ghost" onClick={() => deleteEdge(selectedEdge.id)}>
                                        Remove Edge
                                    </button>
                                </div>
                            )}

                            {selectedNode && (
                                <div className="pipeline-editor-inspector-section">
                                    <h5>{selectedNode.display_name}</h5>
                                    <div className="pipeline-editor-grid">
                                        <label>
                                            Stage
                                            <input className="input" value={selectedNode.stage} readOnly />
                                        </label>
                                        <label>
                                            Step Type
                                            <input
                                                className="input"
                                                value={selectedNode.step_type}
                                                onChange={(event) => updateNodePatch(selectedNode.id, { step_type: event.target.value })}
                                            />
                                        </label>
                                    </div>
                                    <label>
                                        Display Name
                                        <input
                                            className="input"
                                            value={selectedNode.display_name}
                                            onChange={(event) => updateNodePatch(selectedNode.id, { display_name: event.target.value })}
                                        />
                                    </label>
                                    <label>
                                        Description
                                        <textarea
                                            className="input pipeline-editor-textarea"
                                            value={selectedNode.description}
                                            onChange={(event) => updateNodePatch(selectedNode.id, { description: event.target.value })}
                                        />
                                    </label>
                                    <label>
                                        Config Schema Ref
                                        <input
                                            className="input"
                                            value={selectedNode.config_schema_ref}
                                            onChange={(event) => updateNodePatch(selectedNode.id, { config_schema_ref: event.target.value })}
                                        />
                                    </label>
                                    <label>
                                        Node Config (JSON object)
                                        <textarea
                                            className="input pipeline-editor-textarea"
                                            value={nodeConfigDraft}
                                            onChange={(event) => {
                                                setNodeConfigDraft(event.target.value);
                                                if (nodeConfigError) {
                                                    setNodeConfigError('');
                                                }
                                            }}
                                        />
                                    </label>
                                    {selectedNodePresets.length > 0 && (
                                        <div className="pipeline-editor-node-config-presets">
                                            {selectedNodePresets.map((preset) => (
                                                <button
                                                    key={preset.id}
                                                    type="button"
                                                    className="pipeline-editor-node-config-preset"
                                                    title={preset.description}
                                                    onClick={() => handleApplyNodePreset(preset)}
                                                >
                                                    {preset.label}
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                    <div className="pipeline-editor-inline-actions">
                                        <button
                                            type="button"
                                            className="btn btn-secondary"
                                            onClick={handleApplyNodeConfig}
                                        >
                                            Apply Node Config
                                        </button>
                                        {nodeConfigError && (
                                            <span className="pipeline-editor-inline-error">{nodeConfigError}</span>
                                        )}
                                    </div>
                                    <label>
                                        Input Artifacts (comma separated)
                                        <input
                                            className="input"
                                            value={selectedNode.input_artifacts.join(', ')}
                                            onChange={(event) => updateNodePatch(selectedNode.id, { input_artifacts: parseList(event.target.value) })}
                                        />
                                    </label>
                                    <label>
                                        Output Artifacts (comma separated)
                                        <input
                                            className="input"
                                            value={selectedNode.output_artifacts.join(', ')}
                                            onChange={(event) => updateNodePatch(selectedNode.id, { output_artifacts: parseList(event.target.value) })}
                                        />
                                    </label>

                                    <div className="pipeline-editor-grid">
                                        <label>
                                            Execution Modes
                                            <input
                                                className="input"
                                                value={selectedNode.runtime_requirements.execution_modes.join(', ')}
                                                onChange={(event) => updateNodePatch(selectedNode.id, {
                                                    runtime_requirements: {
                                                        ...selectedNode.runtime_requirements,
                                                        execution_modes: parseList(event.target.value),
                                                    },
                                                })}
                                            />
                                        </label>
                                        <label>
                                            Required Services
                                            <input
                                                className="input"
                                                value={selectedNode.runtime_requirements.required_services.join(', ')}
                                                onChange={(event) => updateNodePatch(selectedNode.id, {
                                                    runtime_requirements: {
                                                        ...selectedNode.runtime_requirements,
                                                        required_services: parseList(event.target.value),
                                                    },
                                                })}
                                            />
                                        </label>
                                    </div>

                                    <div className="pipeline-editor-grid">
                                        <label>
                                            Required Env Vars
                                            <input
                                                className="input"
                                                value={selectedNode.runtime_requirements.required_env.join(', ')}
                                                onChange={(event) => updateNodePatch(selectedNode.id, {
                                                    runtime_requirements: {
                                                        ...selectedNode.runtime_requirements,
                                                        required_env: parseList(event.target.value),
                                                    },
                                                })}
                                            />
                                        </label>
                                        <label>
                                            Required Settings
                                            <input
                                                className="input"
                                                value={selectedNode.runtime_requirements.required_settings.join(', ')}
                                                onChange={(event) => updateNodePatch(selectedNode.id, {
                                                    runtime_requirements: {
                                                        ...selectedNode.runtime_requirements,
                                                        required_settings: parseList(event.target.value),
                                                    },
                                                })}
                                            />
                                        </label>
                                    </div>

                                    <div className="pipeline-editor-grid">
                                        <label>
                                            Requires GPU
                                            <select
                                                className="input"
                                                value={selectedNode.runtime_requirements.requires_gpu ? 'yes' : 'no'}
                                                onChange={(event) => updateNodePatch(selectedNode.id, {
                                                    runtime_requirements: {
                                                        ...selectedNode.runtime_requirements,
                                                        requires_gpu: event.target.value === 'yes',
                                                    },
                                                })}
                                            >
                                                <option value="no">No</option>
                                                <option value="yes">Yes</option>
                                            </select>
                                        </label>
                                        <label>
                                            Min VRAM (GB)
                                            <input
                                                className="input"
                                                type="number"
                                                min={0}
                                                step={0.5}
                                                value={selectedNode.runtime_requirements.min_vram_gb}
                                                onChange={(event) => updateNodePatch(selectedNode.id, {
                                                    runtime_requirements: {
                                                        ...selectedNode.runtime_requirements,
                                                        min_vram_gb: Number.parseFloat(event.target.value || '0') || 0,
                                                    },
                                                })}
                                            />
                                        </label>
                                    </div>
                                </div>
                            )}
                            </aside>
                        )}
                    </div>

                    {validateResult && (
                        <div className={`pipeline-editor-result ${validateResult.valid ? 'ok' : 'error'}`}>
                            Validate: valid={validateResult.valid ? 'yes' : 'no'} | errors={validateResult.errors.length} | warnings={validateResult.warnings.length}
                        </div>
                    )}

                    {compileResult && (
                        <div className={`pipeline-editor-result ${compileResult.errors.length === 0 ? 'ok' : 'error'}`}>
                            Compile: active-stage-present={compileResult.checks.active_stage_present ? 'yes' : 'no'} | runtime-ready={compileResult.checks.active_stage_runtime_ready ? 'yes' : 'no'} | ready-now={compileResult.checks.active_stage_ready_now ? 'yes' : 'no'} | errors={compileResult.errors.length} | warnings={compileResult.warnings.length}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
