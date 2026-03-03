import { useState } from 'react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';

interface CompressionPanelProps { projectId: number; onNextStep?: () => void; }

export default function CompressionPanel({ projectId, onNextStep }: CompressionPanelProps) {
    const [modelPath, setModelPath] = useState('');
    const [bits, setBits] = useState(4);
    const [format, setFormat] = useState('gguf');
    const [loraPath, setLoraPath] = useState('');
    const [result, setResult] = useState<any>(null);

    const handleQuantize = async () => {
        if (!modelPath) return;
        const res = await api.post(`/projects/${projectId}/compression/quantize`, { model_path: modelPath, bits, output_format: format });
        setResult(res.data);
    };

    const handleMerge = async () => {
        if (!modelPath || !loraPath) return;
        const res = await api.post(`/projects/${projectId}/compression/merge-lora`, { base_model_path: modelPath, lora_adapter_path: loraPath });
        setResult(res.data);
    };

    const handleBenchmark = async () => {
        if (!modelPath) return;
        const res = await api.post(`/projects/${projectId}/compression/benchmark`, { model_path: modelPath });
        setResult(res.data);
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
                    <button className="btn btn-primary" onClick={handleQuantize}>📦 Quantize</button>
                    <button className="btn btn-secondary" onClick={handleMerge}>🔗 Merge LoRA</button>
                    <button className="btn btn-secondary" onClick={handleBenchmark}>📐 Benchmark</button>
                </div>
            </div>

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
