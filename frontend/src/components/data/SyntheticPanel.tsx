import { useState } from 'react';
import api from '../../api/client';

interface SyntheticPanelProps { projectId: number; }

export default function SyntheticPanel({ projectId }: SyntheticPanelProps) {
    const [sourceText, setSourceText] = useState('');
    const [numPairs, setNumPairs] = useState(5);
    const [apiUrl, setApiUrl] = useState('');
    const [apiKey, setApiKey] = useState('');
    const [generatedPairs, setGeneratedPairs] = useState<any[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [saveResult, setSaveResult] = useState<any>(null);

    const handleGenerate = async () => {
        if (!sourceText.trim()) return;
        setIsGenerating(true);
        try {
            const res = await api.post(`/projects/${projectId}/synthetic/generate`, {
                source_text: sourceText, num_pairs: numPairs, api_url: apiUrl, api_key: apiKey,
            });
            setGeneratedPairs(res.data.pairs || []);
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Generation failed. Check teacher model settings.');
        } finally {
            setIsGenerating(false);
        }
    };

    const handleSave = async () => {
        const res = await api.post(`/projects/${projectId}/synthetic/save`, { pairs: generatedPairs, min_confidence: 0.4 });
        setSaveResult(res.data);
    };

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-lg)' }}>🧪 Synthetic Data Generation</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
                    <div className="form-group"><label className="form-label">Teacher Model API URL</label><input className="input" value={apiUrl} onChange={e => setApiUrl(e.target.value)} placeholder="https://api.openai.com/v1/chat/completions" /></div>
                    <div className="form-group"><label className="form-label">API Key</label><input className="input" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="sk-..." /></div>
                </div>
                <div className="form-group">
                    <label className="form-label">Source Text</label>
                    <textarea className="input" style={{ minHeight: 120, resize: 'vertical' }} value={sourceText} onChange={e => setSourceText(e.target.value)} placeholder="Paste domain text to generate Q&A pairs from..." />
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-md)', alignItems: 'center' }}>
                    <div className="form-group" style={{ marginBottom: 0 }}><label className="form-label">Pairs to Generate</label><input className="input" type="number" value={numPairs} onChange={e => setNumPairs(+e.target.value)} min={1} max={50} style={{ width: 80 }} /></div>
                    <button className="btn btn-primary" onClick={handleGenerate} disabled={isGenerating || !sourceText.trim()}>{isGenerating ? '⏳ Generating...' : '🧪 Generate'}</button>
                </div>
            </div>

            {generatedPairs.length > 0 && (
                <div className="card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600 }}>Generated Pairs <span className="badge badge-accent">{generatedPairs.length}</span></h3>
                        <button className="btn btn-primary" onClick={handleSave}>✅ Save Approved</button>
                    </div>
                    {saveResult && (
                        <div style={{ background: 'var(--color-success-bg)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)', marginBottom: 'var(--space-md)', color: 'var(--color-success)', fontSize: 'var(--font-size-sm)' }}>
                            Saved {saveResult.accepted} pairs ({saveResult.rejected} rejected). Total: {saveResult.total}
                        </div>
                    )}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {generatedPairs.map((p, i) => (
                            <div key={i} style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)' }}>
                                <div style={{ fontSize: 'var(--font-size-sm)', marginBottom: 4 }}><strong>Q:</strong> {p.question}</div>
                                <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 4 }}><strong>A:</strong> {p.answer}</div>
                                <div style={{ display: 'flex', gap: 8 }}>
                                    <span className={`badge ${p.confidence >= 0.7 ? 'badge-success' : p.confidence >= 0.4 ? 'badge-warning' : 'badge-error'}`}>
                                        Confidence: {(p.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
