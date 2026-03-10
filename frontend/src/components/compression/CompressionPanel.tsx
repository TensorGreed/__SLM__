import { useState, useEffect } from 'react';
import { Terminal } from 'lucide-react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { TerminalConsole } from '../shared/TerminalConsole';
import { buildWsUrl } from '../../utils/ws';
import { loadWorkflowStagePrefill } from '../../utils/workflowGraphPrefill';

interface CompressionPanelProps { projectId: number; onNextStep?: () => void; }

interface CompressionResult {
    status?: string;
    report_path?: string;
    task_id?: string;
    [key: string]: unknown;
}

const BIT_OPTIONS_BY_FORMAT: Record<string, number[]> = {
    gguf: [2, 3, 4, 5, 6, 8, 16],
    onnx: [8],
};

const FORMAT_OPTIONS: Array<{ value: string; label: string }> = [
    { value: 'gguf', label: 'GGUF' },
    { value: 'onnx', label: 'ONNX (INT8)' },
];

export default function CompressionPanel({ projectId, onNextStep }: CompressionPanelProps) {
    const [modelPath, setModelPath] = useState('');
    const [bits, setBits] = useState(4);
    const [format, setFormat] = useState('gguf');
    const [loraPath, setLoraPath] = useState('');
    const [mergeMethod, setMergeMethod] = useState<'ties' | 'dex'>('ties');
    const [modelMergePathsText, setModelMergePathsText] = useState('');
    const [modelMergeWeightsText, setModelMergeWeightsText] = useState('');
    const [tiesDensity, setTiesDensity] = useState('0.2');
    const [mergePrefillStage, setMergePrefillStage] = useState('');
    const [result, setResult] = useState<CompressionResult | null>(null);
    const [isCompressing, setIsCompressing] = useState(false);
    const [compressionLogs, setCompressionLogs] = useState<string[]>([]);
    const [activeReportPath, setActiveReportPath] = useState<string | null>(null);
    const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
    const [taskState, setTaskState] = useState<string>('');
    const [compressionError, setCompressionError] = useState<string>('');

    const bitOptions = BIT_OPTIONS_BY_FORMAT[format] || [4];

    useEffect(() => {
        let cancelled = false;
        const applyMergePrefill = async () => {
            const prefill = await loadWorkflowStagePrefill(projectId, ['model_merge']);
            if (cancelled || !prefill) {
                return;
            }
            const cfg = prefill.config || {};
            const methodToken = String(cfg.merge_method || '').trim().toLowerCase();
            if (methodToken === 'ties' || methodToken === 'dex') {
                setMergeMethod(methodToken);
            }
            const modelPaths = Array.isArray(cfg.model_paths)
                ? cfg.model_paths.map((item) => String(item || '').trim()).filter(Boolean)
                : [];
            if (modelPaths.length >= 2) {
                setModelMergePathsText(modelPaths.join('\n'));
            }
            const weights = Array.isArray(cfg.weights)
                ? cfg.weights
                    .map((item) => Number(item))
                    .filter((item) => Number.isFinite(item) && item > 0)
                : [];
            if (weights.length > 0) {
                setModelMergeWeightsText(weights.map((item) => String(item)).join(','));
            }
            const densityValue = Number(cfg.ties_density);
            if (Number.isFinite(densityValue) && densityValue > 0) {
                setTiesDensity(String(densityValue));
            }
            setMergePrefillStage(prefill.stage);
        };
        void applyMergePrefill();
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    useEffect(() => {
        if (!bitOptions.includes(bits)) {
            setBits(bitOptions[0]);
        }
    }, [bits, bitOptions]);

    useEffect(() => {
        if (!isCompressing) return;

        const wsUrl = buildWsUrl(`/api/projects/${projectId}/compression/ws/logs`);
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "log") {
                    setCompressionLogs((prev) => [...prev, data.text]);
                }
            } catch (err) {
                console.error("WS Parse error", err);
            }
        };

        return () => {
            ws.close();
        };
    }, [isCompressing, projectId]);

    useEffect(() => {
        if (!activeReportPath) return;
        const interval = window.setInterval(async () => {
            try {
                const status = await api.get<CompressionResult>(`/projects/${projectId}/compression/jobs/status`, {
                    params: { report_path: activeReportPath },
                });
                if (status.data?.status === "completed" || status.data?.status === "failed") {
                    setResult(status.data);
                    setIsCompressing(false);
                    setActiveReportPath(null);
                    setActiveTaskId(null);
                    setTaskState('');
                } else if (activeTaskId) {
                    const task = await api.get(`/projects/${projectId}/compression/jobs/tasks/${activeTaskId}`);
                    setTaskState(String(task.data?.state || ''));
                }
            } catch (err) {
                console.error("Failed to poll compression status", err);
                setIsCompressing(false);
                setActiveReportPath(null);
                setActiveTaskId(null);
                setTaskState('');
            }
        }, 2000);
        return () => window.clearInterval(interval);
    }, [activeReportPath, activeTaskId, projectId]);

    const handleAction = async (actionFn: () => Promise<{ data: CompressionResult }>) => {
        setIsCompressing(true);
        setCompressionLogs([]);
        setResult(null);
        setActiveReportPath(null);
        setActiveTaskId(null);
        setTaskState('');
        setCompressionError('');
        try {
            const res = await actionFn();
            setResult(res.data);
            setActiveTaskId(typeof res.data?.task_id === 'string' ? res.data.task_id : null);
            if (res.data?.status === "queued" && res.data?.report_path) {
                setActiveReportPath(res.data.report_path);
            } else {
                setIsCompressing(false);
            }
        } catch (e) {
            console.error(e);
            setIsCompressing(false);
            setCompressionError('Compression request failed. Check configuration and logs.');
        }
    };

    const handleCancel = async () => {
        if (!activeTaskId) return;
        try {
            await api.post(`/projects/${projectId}/compression/jobs/tasks/${activeTaskId}/cancel`);
            setCompressionLogs((prev) => [...prev, '[ui] cancellation requested']);
            setIsCompressing(false);
            setActiveTaskId(null);
            setActiveReportPath(null);
            setTaskState('cancel_requested');
        } catch {
            setCompressionError('Failed to cancel compression task.');
        }
    };

    const handleQuantize = async () => {
        if (!modelPath) return;
        await handleAction(() => api.post<CompressionResult>(`/projects/${projectId}/compression/quantize`, { model_path: modelPath, bits, output_format: format }));
    };

    const handleMerge = async () => {
        if (!modelPath || !loraPath) return;
        await handleAction(() => api.post<CompressionResult>(`/projects/${projectId}/compression/merge-lora`, { base_model_path: modelPath, lora_adapter_path: loraPath }));
    };

    const handleBenchmark = async () => {
        if (!modelPath) return;
        await handleAction(() => api.post<CompressionResult>(`/projects/${projectId}/compression/benchmark`, { model_path: modelPath }));
    };

    const handleModelMerge = async () => {
        const modelPaths = modelMergePathsText
            .split(/\n|,/)
            .map((item) => item.trim())
            .filter(Boolean);
        if (modelPaths.length < 2) {
            setCompressionError('Model merge requires at least two model paths.');
            return;
        }
        const weightValues = modelMergeWeightsText
            .split(/,|\s+/)
            .map((item) => item.trim())
            .filter(Boolean)
            .map((item) => Number(item))
            .filter((item) => Number.isFinite(item) && item > 0);
        const parsedTiesDensity = Number.parseFloat(tiesDensity);
        await handleAction(() => api.post<CompressionResult>(`/projects/${projectId}/compression/merge-models`, {
            model_paths: modelPaths,
            merge_method: mergeMethod,
            weights: weightValues.length > 0 ? weightValues : undefined,
            ties_density: Number.isFinite(parsedTiesDensity) ? parsedTiesDensity : 0.2,
        }));
    };

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-lg)' }}>Compression Engine</h3>
                {mergePrefillStage && (
                    <div style={{ marginBottom: 'var(--space-md)', fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
                        Prefilled from workflow template stage: <strong>{mergePrefillStage}</strong>
                    </div>
                )}
                <div className="form-group"><label className="form-label">Model Path</label><input className="input" value={modelPath} onChange={e => setModelPath(e.target.value)} placeholder="Path to model directory" /></div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-md)', marginBottom: 'var(--space-lg)' }}>
                    <div className="form-group">
                        <label className="form-label">Quantization Bits</label>
                        <select className="input" value={bits} onChange={e => setBits(+e.target.value)}>
                            {bitOptions.map((option) => (
                                <option key={option} value={option}>
                                    {option}-bit
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Output Format</label>
                        <select className="input" value={format} onChange={e => setFormat(e.target.value)}>
                            {FORMAT_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
                <div style={{ marginBottom: 'var(--space-md)', color: 'var(--text-tertiary)', fontSize: 'var(--font-size-sm)' }}>
                    GGUF uses llama.cpp conversion/quantization; ONNX uses real export + dynamic INT8 quantization.
                </div>
                <div className="form-group"><label className="form-label">LoRA Adapter Path (for merge)</label><input className="input" value={loraPath} onChange={e => setLoraPath(e.target.value)} placeholder="Optional: path to LoRA adapter" /></div>
                <div className="form-group">
                    <label className="form-label">Model Soup Paths (for TIES/DEX merge)</label>
                    <textarea
                        className="input"
                        style={{ minHeight: 88, resize: 'vertical' }}
                        value={modelMergePathsText}
                        onChange={e => setModelMergePathsText(e.target.value)}
                        placeholder="/models/a\n/models/b\n/models/c"
                    />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 'var(--space-md)' }}>
                    <div className="form-group">
                        <label className="form-label">Merge Method</label>
                        <select className="input" value={mergeMethod} onChange={e => setMergeMethod(e.target.value as 'ties' | 'dex')}>
                            <option value="ties">TIES</option>
                            <option value="dex">DEX</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Weights (optional CSV)</label>
                        <input
                            className="input"
                            value={modelMergeWeightsText}
                            onChange={e => setModelMergeWeightsText(e.target.value)}
                            placeholder="0.5,0.3,0.2"
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label">TIES Density</label>
                        <input
                            className="input"
                            type="number"
                            min={0.01}
                            max={1}
                            step={0.01}
                            value={tiesDensity}
                            onChange={e => setTiesDensity(e.target.value)}
                        />
                    </div>
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 'var(--space-md)' }}>
                    <button className="btn btn-primary" onClick={handleQuantize} disabled={isCompressing}>📦 Quantize</button>
                    <button className="btn btn-secondary" onClick={handleMerge} disabled={isCompressing}>🔗 Merge LoRA</button>
                    <button className="btn btn-secondary" onClick={handleModelMerge} disabled={isCompressing}>🧬 Merge Models</button>
                    <button className="btn btn-secondary" onClick={handleBenchmark} disabled={isCompressing}>📐 Benchmark</button>
                    {isCompressing && activeTaskId && (
                        <button className="btn btn-secondary" onClick={() => void handleCancel()}>Cancel Job</button>
                    )}
                </div>
                {(taskState || compressionError) && (
                    <div style={{ marginTop: 'var(--space-sm)', fontSize: 'var(--font-size-sm)', color: compressionError ? 'var(--color-error)' : 'var(--text-tertiary)' }}>
                        {compressionError || `Worker task state: ${taskState}`}
                    </div>
                )}
            </div>

            {(isCompressing || compressionLogs.length > 0) && (
                <div className="card">
                    <h3 className="section-title text-sm mb-3 flex items-center gap-2">
                        <Terminal className="text-gray-500 w-4 h-4" /> Compression Logs
                    </h3>
                    <TerminalConsole logs={compressionLogs} height="300px" />
                </div>
            )}

            {result && (
                <div className="card">
                    <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-md)' }}>Result</h3>
                    <pre style={{ background: 'var(--bg-tertiary)', padding: 'var(--space-md)', borderRadius: 'var(--radius-md)', fontSize: 'var(--font-size-sm)', overflow: 'auto', color: 'var(--text-secondary)' }}>
                        {JSON.stringify(result, null, 2)}
                    </pre>
                </div>
            )}

            {onNextStep && (
                <StepFooter
                    currentStep="Compression"
                    nextStep="Export"
                    nextStepIcon="🚀"
                    isComplete={result != null}
                    hint="Quantize or merge your model to continue"
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
