import { useState } from 'react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { toast } from '../../stores/toastStore';

interface SyntheticPanelProps { projectId: number; onNextStep?: () => void; }

interface Chunk {
    chunk_id: number;
    text: string;
    source_doc: string;
    document_id: number;
    selected?: boolean;
}

export default function SyntheticPanel({ projectId, onNextStep }: SyntheticPanelProps) {
    type Provider = 'ollama' | 'openai' | 'custom';
    const [provider, setProvider] = useState<Provider>('ollama');

    const [sourceText, setSourceText] = useState('');
    const [numPairs, setNumPairs] = useState(5);
    const [apiUrl, setApiUrl] = useState('http://localhost:11434/v1/chat/completions');
    const [apiKey, setApiKey] = useState('');
    const [modelName, setModelName] = useState('llama3');
    const [generatedPairs, setGeneratedPairs] = useState<any[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [saveResult, setSaveResult] = useState<any>(null);

    // Auto-load chunks state
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [isLoadingChunks, setIsLoadingChunks] = useState(false);
    const [showChunkPicker, setShowChunkPicker] = useState(false);

    const handleProviderChange = (p: Provider) => {
        setProvider(p);
        if (p === 'ollama') {
            setApiUrl('http://localhost:11434/v1/chat/completions');
            setModelName('llama3');
            setApiKey('');
        } else if (p === 'openai') {
            setApiUrl('https://api.openai.com/v1/chat/completions');
            setModelName('gpt-4o');
            setApiKey('');
        } else {
            setApiUrl('');
            setModelName('');
            setApiKey('');
        }
    };

    const loadChunks = async () => {
        setIsLoadingChunks(true);
        try {
            const res = await api.get(`/projects/${projectId}/cleaning/chunks`);
            const loadedChunks = (res.data.chunks || []).map((c: any) => ({ ...c, selected: true }));
            setChunks(loadedChunks);
            setShowChunkPicker(true);
        } catch (err) {
            toast.error('No cleaned chunks found. Run Data Cleaning first.');
        } finally {
            setIsLoadingChunks(false);
        }
    };

    const toggleChunk = (idx: number) => {
        setChunks(prev => prev.map((c, i) => i === idx ? { ...c, selected: !c.selected } : c));
    };

    const applySelectedChunks = () => {
        const selected = chunks.filter(c => c.selected);
        const combined = selected.map(c => c.text).join('\n\n---\n\n');
        setSourceText(combined);
        setShowChunkPicker(false);
    };

    const handleGenerate = async () => {
        if (!sourceText.trim()) return;
        setIsGenerating(true);
        try {
            const res = await api.post(`/projects/${projectId}/synthetic/generate`, {
                source_text: sourceText, num_pairs: numPairs, api_url: apiUrl, api_key: apiKey, model_name: modelName
            });
            setGeneratedPairs(res.data.pairs || []);
        } catch (err: any) {
            toast.error(err.response?.data?.detail || 'Generation failed. Check teacher model settings.');
        } finally {
            setIsGenerating(false);
        }
    };

    const handleSave = async () => {
        const res = await api.post(`/projects/${projectId}/synthetic/save`, { pairs: generatedPairs, min_confidence: 0.4 });
        setSaveResult(res.data);
    };

    const selectedCount = chunks.filter(c => c.selected).length;
    const isDemoMode = generatedPairs.some(p => p.source === 'demo_heuristic');

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-lg)' }}>🧪 Synthetic Data Generation</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: provider === 'ollama' ? '1fr 1fr' : '1fr 1fr 1fr', gap: 'var(--space-md)' }}>
                        <div className="form-group">
                            <label className="form-label">Provider</label>
                            <select className="input" value={provider} onChange={e => handleProviderChange(e.target.value as Provider)}>
                                <option value="ollama">Local (Ollama)</option>
                                <option value="openai">Cloud (OpenAI)</option>
                                <option value="custom">Custom Endpoint</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Model Name</label>
                            <input className="input" value={modelName} onChange={e => setModelName(e.target.value)} placeholder={provider === 'ollama' ? 'llama3' : 'gpt-4o'} />
                        </div>
                        {provider !== 'ollama' && (
                            <div className="form-group">
                                <label className="form-label">API Key</label>
                                <input className="input" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="sk-..." />
                            </div>
                        )}
                    </div>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">API URL {provider === 'openai' && <span style={{ color: 'var(--text-tertiary)', fontSize: '0.8em' }}>(Locked)</span>}</label>
                        <input className="input" value={apiUrl} onChange={e => setApiUrl(e.target.value)} readOnly={provider === 'openai'} style={{ opacity: provider === 'openai' ? 0.7 : 1, fontFamily: 'monospace' }} />
                    </div>
                </div>

                {/* Source Text with Auto-Load */}
                <div className="form-group">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                        <label className="form-label" style={{ margin: 0 }}>Source Text</label>
                        <button
                            className="btn btn-secondary"
                            onClick={loadChunks}
                            disabled={isLoadingChunks}
                            style={{ fontSize: 'var(--font-size-xs)', padding: '4px 12px' }}
                        >
                            {isLoadingChunks ? '⏳ Loading...' : '📥 Load from Cleaned Data'}
                        </button>
                    </div>
                    <textarea className="input" style={{ minHeight: 120, resize: 'vertical' }} value={sourceText} onChange={e => setSourceText(e.target.value)} placeholder="Paste domain text here, or click 'Load from Cleaned Data' to auto-import chunks from the Cleaning step..." />
                    <small style={{ color: 'var(--text-secondary)', marginTop: '4px', display: 'block' }}>
                        {sourceText ? `${sourceText.length.toLocaleString()} characters loaded` : 'No text loaded yet. Use the button above to import cleaned chunks automatically.'}
                    </small>
                </div>

                <div style={{ display: 'flex', gap: 'var(--space-md)', alignItems: 'center' }}>
                    <div className="form-group" style={{ marginBottom: 0 }}><label className="form-label">Pairs to Generate</label><input className="input" type="number" value={numPairs} onChange={e => setNumPairs(+e.target.value)} min={1} max={50} style={{ width: 80 }} /></div>
                    <button className="btn btn-primary" onClick={handleGenerate} disabled={isGenerating || !sourceText.trim()}>{isGenerating ? '⏳ Generating...' : '🧪 Generate'}</button>
                </div>
            </div>

            {/* Chunk Picker Modal */}
            {showChunkPicker && (
                <div className="card" style={{ border: '2px solid var(--color-primary)', background: 'var(--bg-secondary)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, margin: 0 }}>
                            📥 Select Chunks to Import
                            <span className="badge badge-accent" style={{ marginLeft: 8 }}>{selectedCount} / {chunks.length} selected</span>
                        </h3>
                        <div style={{ display: 'flex', gap: 8 }}>
                            <button className="btn btn-secondary" style={{ fontSize: 'var(--font-size-xs)' }} onClick={() => setChunks(prev => prev.map(c => ({ ...c, selected: true })))}>Select All</button>
                            <button className="btn btn-secondary" style={{ fontSize: 'var(--font-size-xs)' }} onClick={() => setChunks(prev => prev.map(c => ({ ...c, selected: false })))}>Deselect All</button>
                        </div>
                    </div>
                    <div style={{ maxHeight: 300, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {chunks.map((chunk, idx) => (
                            <div
                                key={idx}
                                onClick={() => toggleChunk(idx)}
                                style={{
                                    padding: 'var(--space-sm) var(--space-md)',
                                    borderRadius: 'var(--radius-md)',
                                    background: chunk.selected ? 'rgba(139, 92, 246, 0.08)' : 'var(--bg-tertiary)',
                                    border: `1px solid ${chunk.selected ? 'var(--color-primary)' : 'transparent'}`,
                                    cursor: 'pointer', transition: 'all 0.2s ease',
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                    <input type="checkbox" checked={chunk.selected || false} onChange={() => { }} style={{ pointerEvents: 'none' }} />
                                    <span className="badge badge-info" style={{ fontSize: 10 }}>{chunk.source_doc}</span>
                                    <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)' }}>Chunk #{chunk.chunk_id}</span>
                                </div>
                                <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginTop: 4, maxHeight: 40, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {chunk.text.slice(0, 200)}...
                                </div>
                            </div>
                        ))}
                        {chunks.length === 0 && (
                            <div style={{ textAlign: 'center', padding: 'var(--space-lg)', color: 'var(--text-tertiary)' }}>
                                No chunks found. Run Data Cleaning on your documents first.
                            </div>
                        )}
                    </div>
                    <div style={{ display: 'flex', gap: 8, marginTop: 'var(--space-md)' }}>
                        <button className="btn btn-primary" onClick={applySelectedChunks} disabled={selectedCount === 0}>
                            ✅ Load {selectedCount} Chunk{selectedCount !== 1 ? 's' : ''} into Source Text
                        </button>
                        <button className="btn btn-secondary" onClick={() => setShowChunkPicker(false)}>Cancel</button>
                    </div>
                </div>
            )}

            {generatedPairs.length > 0 && (
                <div className="card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600 }}>Generated Pairs <span className="badge badge-accent">{generatedPairs.length}</span></h3>
                        <button className="btn btn-primary" onClick={handleSave}>✅ Save Approved</button>
                    </div>

                    {isDemoMode && (
                        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '.5rem', padding: '.75rem 1rem', background: 'rgba(99, 179, 237, .08)', border: '1px solid rgba(99, 179, 237, .2)', borderRadius: '8px', fontSize: '.85rem', color: 'rgba(255, 255, 255, .8)', marginBottom: '1rem' }}>
                            <span style={{ fontSize: '1.1rem', flexShrink: 0 }}>ℹ️</span>
                            <div>
                                <strong>Demo mode</strong> — pairs generated via heuristic extraction. Connect a teacher model (Ollama, OpenAI, etc.) for production-quality generation.
                            </div>
                        </div>
                    )}

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

            {onNextStep && (
                <StepFooter
                    currentStep="Synthetic Generation"
                    nextStep="Dataset Prep"
                    nextStepIcon="📋"
                    isComplete={generatedPairs.length > 0}
                    hint="Generate and save synthetic Q&A pairs to continue"
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
