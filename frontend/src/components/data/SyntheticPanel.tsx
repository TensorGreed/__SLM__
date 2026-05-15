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

type GenerationMode = 'qa' | 'conversation' | 'span_extraction';

interface SpanEntity {
    type?: string;
    start?: number;
    end?: number;
    text?: string;
}

interface SpanRow {
    text?: string;
    entities?: SpanEntity[];
    confidence?: number;
    source?: string;
    model?: string;
}

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
    const [generatedSpans, setGeneratedSpans] = useState<SpanRow[]>([]);
    const [numSpans, setNumSpans] = useState(5);
    // When true, span_extraction generation runs server-side as a
    // batched async task: ceil(numSpans / 50) calls each fed a fresh
    // randomized sample of the project's cleaned chunks. The textarea
    // / chunk picker selection is ignored in this mode.
    const [useAllChunks, setUseAllChunks] = useState(false);
    // Live status of the in-flight async span job. ``null`` when no
    // task is running. The poller drives this every ~3s.
    const [spanTaskStatus, setSpanTaskStatus] = useState<{
        task_id: string;
        status: string;
        target_rows: number;
        rows_so_far: number;
        batches_done: number;
        batches_total: number;
        error?: string | null;
    } | null>(null);
    // Comma-separated list shown in the UI; parsed to a string[] when
    // calling the backend. Pre-filled from the project's prepared
    // manifest when task_profile is structured_extraction.
    const [entityTypesInput, setEntityTypesInput] = useState('');
    // Set to true when the prepared manifest declares span_set scoring
    // — drives the auto-switch to span_extraction mode and the warning
    // banner on the QA / conversation modes.
    const [projectIsSpanExtraction, setProjectIsSpanExtraction] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);
    const [saveResult, setSaveResult] = useState<any>(null);
    const [prefillSourceStage, setPrefillSourceStage] = useState('');

    // Auto-load chunks state
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [isLoadingChunks, setIsLoadingChunks] = useState(false);
    const [showChunkPicker, setShowChunkPicker] = useState(false);

    // Auto-detect span_extraction projects from the prepared manifest
    // so the panel doesn't ship Q&A pairs into a PII / NER dataset.
    useEffect(() => {
        let cancelled = false;
        const detectSpanExtraction = async () => {
            try {
                const res = await api.get(
                    `/projects/${projectId}/prepared-manifest`,
                );
                if (cancelled) return;
                const manifest = res.data || {};
                const taskProfile = String(manifest.task_profile || '').trim().toLowerCase();
                const outputSchema = manifest.output_schema || {};
                const scoringMode =
                    String(outputSchema.scoring_mode || '').trim().toLowerCase();
                const isSpanSet =
                    taskProfile === 'structured_extraction' &&
                    scoringMode === 'span_set';
                if (isSpanSet) {
                    setProjectIsSpanExtraction(true);
                    setGenerationMode('span_extraction');
                    const entityTypes = Array.isArray(manifest.entity_types)
                        ? manifest.entity_types.filter(
                              (e: unknown) => typeof e === 'string' && e.trim(),
                          )
                        : [];
                    if (entityTypes.length > 0) {
                        setEntityTypesInput(entityTypes.join(', '));
                    }
                }
            } catch {
                // Manifest endpoint missing or project not seeded yet —
                // not fatal, panel falls back to today's behavior.
            }
        };
        void detectSpanExtraction();
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    useEffect(() => {
        let cancelled = false;
        const applyPrefill = async () => {
            const prefill = await loadWorkflowStagePrefill(projectId, ['synthetic_conversation', 'synthetic']);
            if (cancelled || !prefill) {
                return;
            }
            const cfg = prefill.config || {};
            const modeToken = String(cfg.mode || '').trim().toLowerCase();
            // Don't let workflow prefill override the span_extraction
            // auto-pick — that one's keyed off the manifest's
            // declared scoring_mode and is the load-bearing default.
            if (projectIsSpanExtraction) {
                // Skip mode setting; keep span_extraction.
            } else if (
                prefill.stage === 'synthetic_conversation'
                || modeToken.includes('conversation')
            ) {
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

    // Default picker load size — a random sample of N chunks so the
    // browser doesn't have to render a 74k-row list. "Load more"
    // appends another N from the random pool when the user wants
    // breadth; "Use all N chunks" sets the server-side auto-sample
    // toggle to use the whole corpus without ever materializing it
    // in the DOM.
    const CHUNK_PICKER_PAGE_SIZE = 200;
    const [chunkPoolTotal, setChunkPoolTotal] = useState(0);
    const [isLoadingMoreChunks, setIsLoadingMoreChunks] = useState(false);

    const loadChunks = async () => {
        setIsLoadingChunks(true);
        try {
            const res = await api.get(
                `/projects/${projectId}/cleaning/chunks`,
                {
                    params: {
                        limit: CHUNK_PICKER_PAGE_SIZE,
                        random_sample: true,
                    },
                },
            );
            const loadedChunks = (res.data.chunks || []).map((c: any) => ({
                ...c,
                selected: false,
            }));
            setChunks(loadedChunks);
            setChunkPoolTotal(Number(res.data.total ?? loadedChunks.length));
            setShowChunkPicker(true);
        } catch (err) {
            toast.error('No cleaned chunks found. Run Data Cleaning first.');
        } finally {
            setIsLoadingChunks(false);
        }
    };

    const loadMoreChunks = async () => {
        if (isLoadingMoreChunks) return;
        setIsLoadingMoreChunks(true);
        try {
            const res = await api.get(
                `/projects/${projectId}/cleaning/chunks`,
                {
                    params: {
                        limit: CHUNK_PICKER_PAGE_SIZE,
                        random_sample: true,
                    },
                },
            );
            const fetched: any[] = res.data.chunks || [];
            // Dedupe by (document_id, chunk_id) — random sampling can
            // overlap; appending duplicates would confuse the picker.
            setChunks((prev) => {
                const seen = new Set(
                    prev.map((c) => `${c.document_id}:${c.chunk_id}`),
                );
                const fresh = fetched
                    .filter(
                        (c) => !seen.has(`${c.document_id}:${c.chunk_id}`),
                    )
                    .map((c) => ({ ...c, selected: false }));
                return [...prev, ...fresh];
            });
            setChunkPoolTotal(
                Number(res.data.total ?? chunkPoolTotal),
            );
        } catch (err) {
            toast.error('Failed to load more chunks.');
        } finally {
            setIsLoadingMoreChunks(false);
        }
    };

    const useAllChunksFromPicker = () => {
        setUseAllChunks(true);
        setShowChunkPicker(false);
        toast.success(
            `Using all ${chunkPoolTotal.toLocaleString()} cleaned chunks — each batch will sample 4–8k tokens server-side.`,
        );
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

    const parseEntityTypes = (raw: string): string[] => {
        return raw
            .split(',')
            .map((token) => token.trim())
            .filter((token) => token.length > 0);
    };

    // Generous per-request timeout for the generate endpoints — local
    // models on slower GPUs can take several minutes for large batches.
    // Default axios timeout is no-timeout but intermediaries (Vite proxy,
    // browser idle detection) can sever earlier; setting it explicitly
    // signals intent and keeps the request alive long enough.
    const LONG_REQUEST_TIMEOUT_MS = 10 * 60 * 1000;

    const handleGenerate = async () => {
        // useAllChunks span_extraction mode doesn't need source_text
        // — the server samples from the cleaned-chunk pool per batch.
        const skipSourceCheck =
            generationMode === 'span_extraction' && useAllChunks;
        if (!skipSourceCheck && !sourceText.trim()) return;
        setIsGenerating(true);
        try {
            if (generationMode === 'qa') {
                const res = await api.post(
                    `/projects/${projectId}/synthetic/generate`,
                    {
                        source_text: sourceText,
                        num_pairs: numPairs,
                        api_url: apiUrl,
                        api_key: apiKey,
                        model_name: modelName,
                    },
                    { timeout: LONG_REQUEST_TIMEOUT_MS },
                );
                setGeneratedPairs(res.data.pairs || []);
                setGeneratedConversations([]);
                setGeneratedSpans([]);
            } else if (generationMode === 'span_extraction') {
                // Two paths: a single-batch sync call (≤50 rows AND
                // user is feeding their own source_text), or a
                // batched async task (>50 rows OR sample-from-all-
                // cleaned-chunks mode). The async task is server-
                // looped at PER_BATCH_ROW_CAP=50 per call and uses a
                // fresh randomized chunk sample each batch.
                const wantsAsync = numSpans > 50 || useAllChunks;
                if (wantsAsync) {
                    setSpanTaskStatus(null);
                    setGeneratedSpans([]);
                    setGeneratedPairs([]);
                    setGeneratedConversations([]);
                    const startRes = await api.post(
                        `/projects/${projectId}/synthetic/generate-spans-async`,
                        {
                            target_rows: numSpans,
                            entity_types: parseEntityTypes(entityTypesInput),
                            api_url: apiUrl,
                            api_key: apiKey,
                            model_name: modelName,
                            use_all_chunks: useAllChunks,
                            source_text: useAllChunks ? '' : sourceText,
                        },
                    );
                    const taskId = startRes.data.task_id;
                    setSpanTaskStatus({
                        task_id: taskId,
                        status: startRes.data.status,
                        target_rows: startRes.data.target_rows,
                        rows_so_far: 0,
                        batches_done: 0,
                        batches_total: startRes.data.batches_total,
                    });
                    // Poll until the task reports completed/failed.
                    while (true) {
                        await new Promise((r) => setTimeout(r, 3000));
                        const statusRes = await api.get(
                            `/projects/${projectId}/synthetic/tasks/${taskId}`,
                        );
                        const data = statusRes.data;
                        setSpanTaskStatus({
                            task_id: taskId,
                            status: data.status,
                            target_rows: data.target_rows,
                            rows_so_far: data.rows_so_far,
                            batches_done: data.batches_done,
                            batches_total: data.batches_total,
                            error: data.error,
                        });
                        if (data.status === 'completed' || data.status === 'failed') {
                            setGeneratedSpans(data.rows || []);
                            if (data.status === 'failed') {
                                toast.error(
                                    data.error || 'Span generation failed.',
                                );
                            } else if (data.error) {
                                // Partial success — some batches errored
                                // but the task completed with the rows
                                // that did succeed.
                                toast.warning(
                                    `Generation finished with warnings: ${data.error}`,
                                );
                            }
                            break;
                        }
                    }
                } else {
                    const res = await api.post(
                        `/projects/${projectId}/synthetic/generate-spans`,
                        {
                            source_text: sourceText,
                            num_rows: numSpans,
                            entity_types: parseEntityTypes(entityTypesInput),
                            api_url: apiUrl,
                            api_key: apiKey,
                            model_name: modelName,
                        },
                        { timeout: LONG_REQUEST_TIMEOUT_MS },
                    );
                    setGeneratedSpans(res.data.rows || []);
                    setGeneratedPairs([]);
                    setGeneratedConversations([]);
                }
            } else {
                const res = await api.post(
                    `/projects/${projectId}/synthetic/generate-conversations`,
                    {
                        source_text: sourceText,
                        num_dialogues: numDialogues,
                        min_turns: minTurns,
                        max_turns: Math.max(minTurns, maxTurns),
                        api_url: apiUrl,
                        api_key: apiKey,
                        model_name: modelName,
                    },
                    { timeout: LONG_REQUEST_TIMEOUT_MS },
                );
                setGeneratedConversations(res.data.conversations || []);
                setGeneratedPairs([]);
                setGeneratedSpans([]);
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
        if (generationMode === 'span_extraction') {
            const res = await api.post(`/projects/${projectId}/synthetic/save-spans`, {
                rows: generatedSpans,
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
    const activeGeneratedCount =
        generationMode === 'qa'
            ? generatedPairs.length
            : generationMode === 'span_extraction'
                ? generatedSpans.length
                : generatedConversations.length;
    const isDemoMode =
        generationMode === 'qa'
            ? generatedPairs.some((p) => p.source === 'demo_heuristic')
            : generationMode === 'span_extraction'
                ? generatedSpans.some((r) => r.source === 'demo_heuristic')
                : generatedConversations.some((c) => c.source === 'demo_heuristic');

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
                                setGeneratedSpans([]);
                                setSaveResult(null);
                            }}
                        >
                            <option value="qa">Single-turn Q&A</option>
                            <option value="conversation">Multi-turn Conversations</option>
                            <option value="span_extraction">
                                PII / NER span extraction
                            </option>
                        </select>
                        {projectIsSpanExtraction && generationMode !== 'span_extraction' && (
                            <div
                                style={{
                                    marginTop: 'var(--space-sm)',
                                    padding: '8px 12px',
                                    background: 'rgba(239, 68, 68, 0.08)',
                                    border: '1px solid rgba(239, 68, 68, 0.25)',
                                    borderRadius: 'var(--radius-md)',
                                    fontSize: 'var(--font-size-sm)',
                                    color: 'rgb(153, 27, 27)',
                                }}
                            >
                                <strong>Heads up:</strong> this project is tagged{' '}
                                <code>task_profile: structured_extraction</code> with{' '}
                                <code>scoring_mode: span_set</code>. Q&A or conversation
                                output won't match the eval schema —{' '}
                                <strong>switch to "PII / NER span extraction"</strong> to
                                generate <code>{'{text, entities: […]}'}</code> rows
                                that the StructuredExtractionHandler can score.
                            </div>
                        )}
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
                    ) : generationMode === 'span_extraction' ? (
                        <>
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label className="form-label">Rows to Generate</label>
                                <input
                                    className="input"
                                    type="number"
                                    value={numSpans}
                                    onChange={(e) =>
                                        setNumSpans(
                                            Math.max(1, Math.min(5000, Number(e.target.value) || 1)),
                                        )
                                    }
                                    min={1}
                                    max={5000}
                                    style={{ width: 120 }}
                                    data-testid="span-num-rows"
                                />
                                <div
                                    style={{
                                        fontSize: '0.75rem',
                                        color: 'var(--text-tertiary)',
                                        marginTop: 4,
                                    }}
                                >
                                    {numSpans > 50
                                        ? `Will run in ${Math.ceil(numSpans / 50)} background batches.`
                                        : 'Single-shot generation.'}
                                </div>
                            </div>
                            <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: 280 }}>
                                <label className="form-label">
                                    Entity types{' '}
                                    <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>
                                        (comma-separated)
                                    </span>
                                </label>
                                <input
                                    className="input"
                                    value={entityTypesInput}
                                    onChange={(e) => setEntityTypesInput(e.target.value)}
                                    placeholder="email, phone, ssn, credit_card, person_name"
                                />
                            </div>
                            <div
                                className="form-group"
                                style={{
                                    marginBottom: 0,
                                    flexBasis: '100%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 8,
                                }}
                            >
                                <label
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 8,
                                        cursor: 'pointer',
                                        fontSize: '0.85rem',
                                    }}
                                >
                                    <input
                                        type="checkbox"
                                        checked={useAllChunks}
                                        onChange={(e) =>
                                            setUseAllChunks(e.target.checked)
                                        }
                                        data-testid="span-use-all-chunks"
                                    />
                                    Sample randomly from all cleaned chunks (per batch)
                                </label>
                                <span
                                    style={{
                                        fontSize: '0.75rem',
                                        color: 'var(--text-tertiary)',
                                    }}
                                >
                                    Ignores the source-text textarea; each
                                    batch gets a fresh 4–8k-token sample.
                                </span>
                            </div>
                            {spanTaskStatus && (
                                <div
                                    data-testid="span-task-progress"
                                    style={{
                                        flexBasis: '100%',
                                        padding: 'var(--space-sm)',
                                        background: 'var(--bg-secondary)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: 'var(--radius-sm)',
                                        fontSize: '0.85rem',
                                    }}
                                >
                                    <strong>{spanTaskStatus.status}</strong>
                                    {' — '}
                                    batch {spanTaskStatus.batches_done} /{' '}
                                    {spanTaskStatus.batches_total}
                                    {' · '}
                                    {spanTaskStatus.rows_so_far} /{' '}
                                    {spanTaskStatus.target_rows} rows
                                    {spanTaskStatus.error && (
                                        <div
                                            style={{
                                                color: 'var(--color-error, #c00)',
                                                marginTop: 4,
                                            }}
                                        >
                                            {spanTaskStatus.error}
                                        </div>
                                    )}
                                </div>
                            )}
                        </>
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
                {isGenerating && (
                    <div
                        style={{
                            marginTop: 'var(--space-md)',
                            padding: '8px 12px',
                            background: 'rgba(59, 130, 246, 0.08)',
                            border: '1px solid rgba(59, 130, 246, 0.25)',
                            borderRadius: 'var(--radius-md)',
                            fontSize: 'var(--font-size-sm)',
                            color: 'var(--text-secondary)',
                        }}
                    >
                        ⏳ Generation can take several minutes for large batches on
                        local models. The GPU is working even if this page looks
                        idle — please don't refresh.
                    </div>
                )}
            </div>

            {/* Chunk Picker Modal */}
            {showChunkPicker && (
                <div className="card" style={{ border: '2px solid var(--color-primary)', background: 'var(--bg-secondary)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                        <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, margin: 0 }}>
                            📥 Select Chunks to Import
                            <span className="badge badge-accent" style={{ marginLeft: 8 }}>{selectedCount} / {chunks.length} selected</span>
                            {chunkPoolTotal > chunks.length && (
                                <span
                                    style={{
                                        marginLeft: 8,
                                        fontSize: 'var(--font-size-xs)',
                                        color: 'var(--text-tertiary)',
                                        fontWeight: 400,
                                    }}
                                    data-testid="chunk-pool-total"
                                    title={`Random sample of ${chunks.length} chunks from ${chunkPoolTotal.toLocaleString()} total. Use "Reroll sample" to see a different random subset, or enable "Sample randomly from all cleaned chunks" on the generation form to use the whole pool server-side.`}
                                >
                                    (random sample of {chunkPoolTotal.toLocaleString()})
                                </span>
                            )}
                        </h3>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            {generationMode === 'span_extraction'
                                && chunkPoolTotal > chunks.length && (
                                <button
                                    className="btn btn-primary"
                                    style={{ fontSize: 'var(--font-size-xs)' }}
                                    onClick={useAllChunksFromPicker}
                                    title={`Use the full pool of ${chunkPoolTotal.toLocaleString()} chunks. Each generation batch will sample 4–8k tokens randomly server-side — no need to load them all in the browser.`}
                                    data-testid="span-use-all-from-picker"
                                >
                                    📦 Use all {chunkPoolTotal.toLocaleString()} (auto-sample per batch)
                                </button>
                            )}
                            <button
                                className="btn btn-secondary"
                                style={{ fontSize: 'var(--font-size-xs)' }}
                                onClick={loadChunks}
                                disabled={isLoadingChunks}
                                title="Fetch a different random sample from the pool"
                            >
                                {isLoadingChunks ? 'Loading…' : '🎲 Reroll sample'}
                            </button>
                            <button className="btn btn-secondary" style={{ fontSize: 'var(--font-size-xs)' }} onClick={() => setChunks(prev => prev.map(c => ({ ...c, selected: true })))}>Select all visible</button>
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
                        {chunks.length > 0 && chunkPoolTotal > chunks.length && (
                            <button
                                type="button"
                                className="btn btn-secondary"
                                onClick={loadMoreChunks}
                                disabled={isLoadingMoreChunks}
                                style={{
                                    fontSize: 'var(--font-size-xs)',
                                    alignSelf: 'center',
                                    marginTop: 'var(--space-sm)',
                                }}
                                data-testid="span-load-more-chunks"
                                title={`Append another random ${CHUNK_PICKER_PAGE_SIZE} chunks to the picker (deduplicated). Or click "Use all" to skip the picker entirely and sample the whole pool server-side.`}
                            >
                                {isLoadingMoreChunks
                                    ? 'Loading…'
                                    : `+ Load ${CHUNK_PICKER_PAGE_SIZE} more (${chunks.length} of ${chunkPoolTotal.toLocaleString()} loaded)`}
                            </button>
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
                            {generationMode === 'qa'
                                ? 'Generated Pairs'
                                : generationMode === 'span_extraction'
                                    ? 'Generated Spans'
                                    : 'Generated Conversations'}
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
                        ) : generationMode === 'span_extraction' ? (
                            generatedSpans.map((row, idx) => {
                                const entities = Array.isArray(row.entities) ? row.entities : [];
                                const confidence = typeof row.confidence === 'number' ? row.confidence : 0;
                                return (
                                    <div
                                        key={idx}
                                        style={{
                                            background: 'var(--bg-tertiary)',
                                            borderRadius: 'var(--radius-md)',
                                            padding: 'var(--space-md)',
                                            display: 'flex',
                                            flexDirection: 'column',
                                            gap: 8,
                                        }}
                                    >
                                        <div style={{ fontSize: 'var(--font-size-sm)', whiteSpace: 'pre-wrap' }}>
                                            <strong>Text:</strong> {row.text || '—'}
                                        </div>
                                        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                            {entities.length === 0 ? (
                                                <span
                                                    style={{
                                                        fontSize: 'var(--font-size-xs)',
                                                        color: 'var(--text-tertiary)',
                                                        fontStyle: 'italic',
                                                    }}
                                                >
                                                    no entities detected
                                                </span>
                                            ) : (
                                                entities.map((ent, ei) => (
                                                    <span
                                                        key={ei}
                                                        className="badge badge-accent"
                                                        title={`${ent.start}–${ent.end}`}
                                                        style={{ fontFamily: 'monospace' }}
                                                    >
                                                        {ent.type || '?'}: {ent.text || '—'}
                                                    </span>
                                                ))
                                            )}
                                        </div>
                                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                            <span
                                                className={`badge ${confidence >= 0.7
                                                    ? 'badge-success'
                                                    : confidence >= 0.4
                                                        ? 'badge-warning'
                                                        : 'badge-error'
                                                    }`}
                                            >
                                                Confidence: {(confidence * 100).toFixed(0)}%
                                            </span>
                                            <span
                                                style={{
                                                    fontSize: 'var(--font-size-xs)',
                                                    color: 'var(--text-tertiary)',
                                                }}
                                            >
                                                {entities.length} entit{entities.length === 1 ? 'y' : 'ies'}
                                            </span>
                                        </div>
                                    </div>
                                );
                            })
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
                    hint={
                        generationMode === 'qa'
                            ? 'Generate and save synthetic Q&A pairs to continue'
                            : generationMode === 'span_extraction'
                                ? 'Generate and save synthetic span-extraction rows to continue'
                                : 'Generate and save synthetic multi-turn conversations to continue'
                    }
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
