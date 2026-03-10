import { useEffect, useState } from 'react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import { toast } from '../../stores/toastStore';
import { loadWorkflowStagePrefill } from '../../utils/workflowGraphPrefill';

interface SyntheticPanelProps { projectId: number; onNextStep?: () => void; }

interface Chunk {
    chunk_id: number;
    text: string;
    source_doc: string;
    document_id: number;
    selected?: boolean;
}

type GenerationMode = 'qa' | 'conversation';

function extractErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
        if (Array.isArray(detail)) {
            const messages = detail
                .map((item) => {
                    if (typeof item === 'string') {
                        return item;
                    }
                    if (typeof item === 'object' && item !== null) {
                        const msg = (item as { msg?: unknown }).msg;
                        const loc = (item as { loc?: unknown }).loc;
                        const locText = Array.isArray(loc) ? loc.join('.') : '';
                        if (typeof msg === 'string' && msg.trim()) {
                            return locText ? `${locText}: ${msg}` : msg;
                        }
                    }
                    return '';
                })
                .filter((item) => item);
            if (messages.length > 0) {
                return messages.join('; ');
            }
        }
        if (typeof detail === 'object' && detail !== null) {
            const message = (detail as { message?: unknown }).message;
            if (typeof message === 'string' && message.trim()) {
                return message;
            }
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Generation failed. Check teacher model settings.';
}

export default function SyntheticPanel({ projectId, onNextStep }: SyntheticPanelProps) {
    type Provider = 'ollama' | 'openai' | 'custom';
    const [provider, setProvider] = useState<Provider>('ollama');
    const [generationMode, setGenerationMode] = useState<GenerationMode>('qa');

    const [sourceText, setSourceText] = useState('');
    const [numPairs, setNumPairs] = useState(5);
    const [numDialogues, setNumDialogues] = useState(3);
    const [minTurns, setMinTurns] = useState(3);
    const [maxTurns, setMaxTurns] = useState(5);
    const [apiUrl, setApiUrl] = useState('http://localhost:11434/v1/chat/completions');
    const [apiKey, setApiKey] = useState('');
    const [modelName, setModelName] = useState('llama3');
    const [generatedPairs, setGeneratedPairs] = useState<any[]>([]);
    const [generatedConversations, setGeneratedConversations] = useState<any[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [saveResult, setSaveResult] = useState<any>(null);
    const [prefillSourceStage, setPrefillSourceStage] = useState('');

    // Auto-load chunks state
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [isLoadingChunks, setIsLoadingChunks] = useState(false);
    const [showChunkPicker, setShowChunkPicker] = useState(false);

    useEffect(() => {
        let cancelled = false;
        const applyPrefill = async () => {
            const prefill = await loadWorkflowStagePrefill(projectId, ['synthetic_conversation', 'synthetic']);
            if (cancelled || !prefill) {
                return;
            }
            const cfg = prefill.config || {};
            const modeToken = String(cfg.mode || '').trim().toLowerCase();
            if (prefill.stage === 'synthetic_conversation' || modeToken.includes('conversation')) {
                setGenerationMode('conversation');
            } else if (prefill.stage === 'synthetic') {
                setGenerationMode('qa');
            }

            const sourceTextPrefill = String(cfg.source_text || '').trim();
            if (sourceTextPrefill) {
                setSourceText(sourceTextPrefill);
            }
            const modelToken = String(cfg.model_name || '').trim();
            if (modelToken) {
                setModelName(modelToken);
            }
            const apiUrlToken = String(cfg.api_url || '').trim();
            if (apiUrlToken) {
                setApiUrl(apiUrlToken);
                const normalizedUrl = apiUrlToken.toLowerCase();
                if (normalizedUrl.includes('api.openai.com')) {
                    setProvider('openai');
                } else if (normalizedUrl.includes('localhost:11434') || normalizedUrl.includes('127.0.0.1:11434')) {
                    setProvider('ollama');
                } else {
                    setProvider('custom');
                }
            }

            const parsedPairs = Number(cfg.num_pairs);
            if (Number.isFinite(parsedPairs) && parsedPairs > 0) {
                setNumPairs(Math.max(1, Math.min(50, Math.round(parsedPairs))));
            }
            const parsedDialogues = Number(cfg.num_dialogues);
            if (Number.isFinite(parsedDialogues) && parsedDialogues > 0) {
                setNumDialogues(Math.max(1, Math.min(20, Math.round(parsedDialogues))));
            }
            const parsedMinTurns = Number(cfg.min_turns);
            const parsedMaxTurns = Number(cfg.max_turns);
            if (Number.isFinite(parsedMinTurns) && parsedMinTurns > 0) {
                setMinTurns(Math.max(1, Math.min(20, Math.round(parsedMinTurns))));
            }
            if (Number.isFinite(parsedMaxTurns) && parsedMaxTurns > 0) {
                const maxValue = Math.max(1, Math.min(20, Math.round(parsedMaxTurns)));
                setMaxTurns(maxValue);
            }
            setPrefillSourceStage(prefill.stage);
        };
        void applyPrefill();
        return () => {
            cancelled = true;
        };
    }, [projectId]);

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
            if (generationMode === 'qa') {
                const res = await api.post(`/projects/${projectId}/synthetic/generate`, {
                    source_text: sourceText,
                    num_pairs: numPairs,
                    api_url: apiUrl,
                    api_key: apiKey,
                    model_name: modelName,
                });
                setGeneratedPairs(res.data.pairs || []);
                setGeneratedConversations([]);
            } else {
                const res = await api.post(`/projects/${projectId}/synthetic/generate-conversations`, {
                    source_text: sourceText,
                    num_dialogues: numDialogues,
                    min_turns: minTurns,
                    max_turns: Math.max(minTurns, maxTurns),
                    api_url: apiUrl,
                    api_key: apiKey,
                    model_name: modelName,
                });
                setGeneratedConversations(res.data.conversations || []);
                setGeneratedPairs([]);
            }
            setSaveResult(null);
        } catch (err: any) {
            toast.error(extractErrorMessage(err));
        } finally {
            setIsGenerating(false);
        }
    };

    const handleSave = async () => {
        if (generationMode === 'qa') {
            const res = await api.post(`/projects/${projectId}/synthetic/save`, {
                pairs: generatedPairs,
                min_confidence: 0.4,
            });
            setSaveResult(res.data);
            return;
        }
        const res = await api.post(`/projects/${projectId}/synthetic/save-conversations`, {
            conversations: generatedConversations,
            min_confidence: 0.4,
        });
        setSaveResult(res.data);
    };

    const selectedCount = chunks.filter(c => c.selected).length;
    const activeGeneratedCount = generationMode === 'qa' ? generatedPairs.length : generatedConversations.length;
    const isDemoMode = generationMode === 'qa'
        ? generatedPairs.some(p => p.source === 'demo_heuristic')
        : generatedConversations.some(c => c.source === 'demo_heuristic');

    return (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xl)' }}>
            <div className="card">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-lg)' }}>🧪 Synthetic Data Generation</h3>
                {prefillSourceStage && (
                    <div style={{ marginBottom: 'var(--space-md)', fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
                        Prefilled from workflow template stage: <strong>{prefillSourceStage}</strong>
                    </div>
                )}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
                    <div className="form-group">
                        <label className="form-label">Generation Mode</label>
                        <select
                            className="input"
                            value={generationMode}
                            onChange={(e) => {
                                const nextMode = e.target.value as GenerationMode;
                                setGenerationMode(nextMode);
                                setGeneratedPairs([]);
                                setGeneratedConversations([]);
                                setSaveResult(null);
                            }}
                        >
                            <option value="qa">Single-turn Q&A</option>
                            <option value="conversation">Multi-turn Conversations</option>
                        </select>
                    </div>
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

                <div style={{ display: 'flex', gap: 'var(--space-md)', alignItems: 'center', flexWrap: 'wrap' }}>
                    {generationMode === 'qa' ? (
                        <div className="form-group" style={{ marginBottom: 0 }}>
                            <label className="form-label">Pairs to Generate</label>
                            <input
                                className="input"
                                type="number"
                                value={numPairs}
                                onChange={e => setNumPairs(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
                                min={1}
                                max={50}
                                style={{ width: 120 }}
                            />
                        </div>
                    ) : (
                        <>
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label className="form-label">Dialogues</label>
                                <input
                                    className="input"
                                    type="number"
                                    value={numDialogues}
                                    onChange={e => setNumDialogues(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
                                    min={1}
                                    max={20}
                                    style={{ width: 100 }}
                                />
                            </div>
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label className="form-label">Min Turns</label>
                                <input
                                    className="input"
                                    type="number"
                                    value={minTurns}
                                    onChange={e => {
                                        const nextMin = Math.max(1, Math.min(20, Number(e.target.value) || 1));
                                        setMinTurns(nextMin);
                                        setMaxTurns((prev) => Math.max(prev, nextMin));
                                    }}
                                    min={1}
                                    max={20}
                                    style={{ width: 100 }}
                                />
                            </div>
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label className="form-label">Max Turns</label>
                                <input
                                    className="input"
                                    type="number"
                                    value={maxTurns}
                                    onChange={e => setMaxTurns(Math.max(minTurns, Math.min(20, Number(e.target.value) || minTurns)))}
                                    min={minTurns}
                                    max={20}
                                    style={{ width: 100 }}
                                />
                            </div>
                        </>
                    )}
                    <button className="btn btn-primary" onClick={handleGenerate} disabled={isGenerating || !sourceText.trim()}>
                        {isGenerating ? '⏳ Generating...' : '🧪 Generate'}
                    </button>
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

            {activeGeneratedCount > 0 && (
                <div className="card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600 }}>
                            {generationMode === 'qa' ? 'Generated Pairs' : 'Generated Conversations'}
                            {' '}
                            <span className="badge badge-accent">{activeGeneratedCount}</span>
                        </h3>
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
                            Saved {saveResult.accepted} item(s) ({saveResult.rejected} rejected). Total: {saveResult.total}
                            {typeof saveResult.accepted_turns === 'number' ? ` • accepted turns: ${saveResult.accepted_turns}` : ''}
                        </div>
                    )}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {generationMode === 'qa' ? (
                            generatedPairs.map((p, i) => (
                                <div key={i} style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)' }}>
                                    <div style={{ fontSize: 'var(--font-size-sm)', marginBottom: 4 }}><strong>Q:</strong> {p.question}</div>
                                    <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 4 }}><strong>A:</strong> {p.answer}</div>
                                    <div style={{ display: 'flex', gap: 8 }}>
                                        <span className={`badge ${p.confidence >= 0.7 ? 'badge-success' : p.confidence >= 0.4 ? 'badge-warning' : 'badge-error'}`}>
                                            Confidence: {(p.confidence * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                </div>
                            ))
                        ) : (
                            generatedConversations.map((conversation, index) => (
                                <div key={conversation.conversation_id || index} style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: 'var(--space-md)' }}>
                                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                                        <strong style={{ fontSize: 'var(--font-size-sm)' }}>
                                            Conversation {index + 1}
                                        </strong>
                                        <span className="badge badge-info">
                                            {conversation.turn_count || 0} turns
                                        </span>
                                        {typeof conversation.confidence === 'number' && (
                                            <span className={`badge ${conversation.confidence >= 0.7 ? 'badge-success' : conversation.confidence >= 0.4 ? 'badge-warning' : 'badge-error'}`}>
                                                Confidence: {(conversation.confidence * 100).toFixed(0)}%
                                            </span>
                                        )}
                                    </div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                        {(conversation.messages || []).map((message: any, messageIndex: number) => (
                                            <div key={`${conversation.conversation_id || index}-${messageIndex}`} style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
                                                <strong>{String(message?.role || 'assistant')}:</strong> {String(message?.content || '')}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            )}

            {onNextStep && (
                <StepFooter
                    currentStep="Synthetic Generation"
                    nextStep="Dataset Prep"
                    nextStepIcon="📋"
                    isComplete={activeGeneratedCount > 0}
                    hint={generationMode === 'qa'
                        ? 'Generate and save synthetic Q&A pairs to continue'
                        : 'Generate and save synthetic multi-turn conversations to continue'}
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
