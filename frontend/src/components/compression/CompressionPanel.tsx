import { useState, useEffect } from 'react';
import { Terminal } from 'lucide-react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { TerminalConsole } from '../shared/TerminalConsole';

interface CompressionPanelProps { projectId: number; onNextStep?: () => void; }

interface CompressionResult {
    status?: string;
    report_path?: string;
    [key: string]: unknown;
}

export default function CompressionPanel({ projectId, onNextStep }: CompressionPanelProps) {
    const [modelPath, setModelPath] = useState('');
    const [bits, setBits] = useState(4);
    const [format, setFormat] = useState('gguf');
    const [loraPath, setLoraPath] = useState('');
    const [result, setResult] = useState<CompressionResult | null>(null);
    const [isCompressing, setIsCompressing] = useState(false);
    const [compressionLogs, setCompressionLogs] = useState<string[]>([]);
    const [activeReportPath, setActiveReportPath] = useState<string | null>(null);

    useEffect(() => {
        if (!isCompressing) return;

        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${protocol}://${window.location.host}/api/projects/${projectId}/compression/ws/logs`;
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
                }
            } catch (err) {
                console.error("Failed to poll compression status", err);
                setIsCompressing(false);
                setActiveReportPath(null);
            }
        }, 2000);
        return () => window.clearInterval(interval);
    }, [activeReportPath, projectId]);

    const handleAction = async (actionFn: () => Promise<{ data: CompressionResult }>) => {
        setIsCompressing(true);
        setCompressionLogs([]);
        setResult(null);
        setActiveReportPath(null);
        try {
            const res = await actionFn();
            setResult(res.data);
            if (res.data?.status === "queued" && res.data?.report_path) {
                setActiveReportPath(res.data.report_path);
            } else {
                setIsCompressing(false);
            }
        } catch (e) {
            console.error(e);
            setIsCompressing(false);
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

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-lg)' }}>Compression Engine</h3>
                <div className="form-group"><label className="form-label">Model Path</label><input className="input" value={modelPath} onChange={e => setModelPath(e.target.value)} placeholder="Path to model directory" /></div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-md)', marginBottom: 'var(--space-lg)' }}>
                    <div className="form-group">
                        <label className="form-label">Quantization Bits</label>
                        <select className="input" value={bits} onChange={e => setBits(+e.target.value)}><option value={4}>4-bit</option><option value={8}>8-bit</option></select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Output Format</label>
                        <select className="input" value={format} onChange={e => setFormat(e.target.value)}><option value="gguf">GGUF</option><option value="onnx">ONNX</option><option value="huggingface">HuggingFace</option></select>
                    </div>
                </div>
                <div className="form-group"><label className="form-label">LoRA Adapter Path (for merge)</label><input className="input" value={loraPath} onChange={e => setLoraPath(e.target.value)} placeholder="Optional: path to LoRA adapter" /></div>
                <div style={{ display: 'flex', gap: 8, marginTop: 'var(--space-md)' }}>
                    <button className="btn btn-primary" onClick={handleQuantize} disabled={isCompressing}>📦 Quantize</button>
                    <button className="btn btn-secondary" onClick={handleMerge} disabled={isCompressing}>🔗 Merge LoRA</button>
                    <button className="btn btn-secondary" onClick={handleBenchmark} disabled={isCompressing}>📐 Benchmark</button>
                </div>
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
