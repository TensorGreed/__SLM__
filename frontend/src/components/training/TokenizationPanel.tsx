import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import './TokenizationPanel.css';

interface TokenizationPanelProps {
    projectId: number;
    onNextStep: () => void | Promise<void>;
}

interface TokenStats {
    model_name: string;
    total_samples: number;
    total_tokens: number;
    avg_tokens: number;
    min_tokens: number;
    max_tokens: number;
    p50_tokens: number;
    p95_tokens: number;
    p99_tokens: number;
    exceeding_max: number;
    max_seq_length: number;
    histogram: { bucket: string; count: number }[];
}

const MODEL_PRESETS = [
    { id: 'meta-llama/Llama-3.2-1B', label: 'Llama 3.2 1B' },
    { id: 'meta-llama/Llama-3.2-3B', label: 'Llama 3.2 3B' },
    { id: 'microsoft/phi-3-mini-4k-instruct', label: 'Phi-3 Mini' },
    { id: 'Qwen/Qwen2.5-0.5B', label: 'Qwen 2.5 0.5B' },
    { id: 'Qwen/Qwen2.5-1.5B', label: 'Qwen 2.5 1.5B' },
    { id: 'google/gemma-2-2b', label: 'Gemma 2 2B' },
    { id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', label: 'TinyLlama 1.1B' },
];

const SPLITS = ['train', 'validation', 'test'] as const;

export default function TokenizationPanel({ projectId, onNextStep }: TokenizationPanelProps) {
    const [modelName, setModelName] = useState('meta-llama/Llama-3.2-1B');
    const [split, setSplit] = useState<string>('train');
    const [maxSeqLen, setMaxSeqLen] = useState(2048);
    const [loading, setLoading] = useState(false);
    const [stats, setStats] = useState<TokenStats | null>(null);

    // Vocab state
    const [vocab, setVocab] = useState<{ token: string; id: number }[]>([]);
    const [vocabLoading, setVocabLoading] = useState(false);

    const analyze = async () => {
        setLoading(true);
        setStats(null);
        try {
            const res = await api.post(`/projects/${projectId}/tokenization/analyze`, {
                model_name: modelName,
                split,
                max_seq_length: maxSeqLen,
            });
            setStats(res.data);
        } catch (err: any) {
            const msg = err?.response?.data?.detail || 'Analysis failed. Make sure you have split your dataset first.';
            alert(msg);
        } finally {
            setLoading(false);
        }
    };

    const loadVocab = async () => {
        setVocabLoading(true);
        try {
            const res = await api.get(`/projects/${projectId}/tokenization/vocab-sample`, {
                params: { model_name: modelName, sample_size: 200 },
            });
            setVocab(res.data.tokens || []);
        } catch {
            setVocab([]);
        } finally {
            setVocabLoading(false);
        }
    };

    const recommendedSeqLen = stats
        ? Math.min(32768, Math.pow(2, Math.ceil(Math.log2(stats.p95_tokens))))
        : null;

    const histogramColors = ['#6c5ce7', '#a855f7', '#c4a1f7'];

    return (
        <div className="tok-panel">
            {/* ── Model Selection ──────────────────────────────── */}
            <div className="tok-section">
                <h3><span className="icon">🔤</span> Tokenization Analysis</h3>
                <div className="tok-info">
                    <span className="info-icon">ℹ️</span>
                    Analyze how your dataset tokenizes under a specific model's tokenizer. This helps you set the optimal <code>max_seq_length</code> for training — balancing context coverage vs memory usage.
                </div>

                <p style={{ fontSize: '.78rem', color: 'rgba(255,255,255,.45)', margin: '0 0 .4rem' }}>Quick presets:</p>
                <div className="tok-presets">
                    {MODEL_PRESETS.map(m => (
                        <button
                            key={m.id}
                            className="tok-preset"
                            onClick={() => setModelName(m.id)}
                            style={modelName === m.id ? { background: 'rgba(168,85,247,.3)', borderColor: '#a855f7' } : undefined}
                        >
                            {m.label}
                        </button>
                    ))}
                </div>

                <div className="tok-model-row">
                    <div className="tok-field">
                        <label>Model Name (HuggingFace)</label>
                        <input value={modelName} onChange={e => setModelName(e.target.value)} placeholder="e.g. meta-llama/Llama-3.2-1B" />
                    </div>
                    <div className="tok-field">
                        <label>Split</label>
                        <select value={split} onChange={e => setSplit(e.target.value)}>
                            {SPLITS.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                    </div>
                    <div className="tok-field">
                        <label>Max Seq Length</label>
                        <input type="number" value={maxSeqLen} onChange={e => setMaxSeqLen(+e.target.value)}
                            min={128} max={32768} step={128} style={{ width: '110px' }}
                        />
                    </div>
                    <button className="tok-btn" onClick={analyze} disabled={loading || !modelName.trim()}>
                        {loading ? '⏳ Analyzing...' : '🔬 Analyze Tokens'}
                    </button>
                </div>
            </div>

            {/* ── Results ──────────────────────────────────────── */}
            {stats && (
                <div className="tok-section">
                    <h3><span className="icon">📈</span> Token Statistics — <span style={{ color: '#a855f7' }}>{stats.model_name}</span></h3>

                    <div className="tok-stats-grid">
                        <div className="tok-stat">
                            <div className="label">Samples</div>
                            <div className="value">{stats.total_samples.toLocaleString()}</div>
                        </div>
                        <div className="tok-stat">
                            <div className="label">Total Tokens</div>
                            <div className="value">{stats.total_tokens.toLocaleString()}</div>
                        </div>
                        <div className="tok-stat">
                            <div className="label">Avg Tokens</div>
                            <div className="value">{Math.round(stats.avg_tokens)}</div>
                        </div>
                        <div className="tok-stat">
                            <div className="label">P50</div>
                            <div className="value">{stats.p50_tokens}</div>
                        </div>
                        <div className="tok-stat">
                            <div className="label">P95</div>
                            <div className="value">{stats.p95_tokens}</div>
                        </div>
                        <div className="tok-stat">
                            <div className="label">P99</div>
                            <div className="value">{stats.p99_tokens}</div>
                        </div>
                        <div className="tok-stat">
                            <div className="label">Max</div>
                            <div className="value">{stats.max_tokens}</div>
                        </div>
                        <div className={`tok-stat ${stats.exceeding_max > 0 ? 'warning' : 'success'}`}>
                            <div className="label">Exceeding Max</div>
                            <div className="value">{stats.exceeding_max}</div>
                        </div>
                    </div>

                    {/* Token length histogram */}
                    {stats.histogram && stats.histogram.length > 0 && (
                        <div className="tok-chart">
                            <div className="tok-chart-label">Token Length Distribution</div>
                            <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={stats.histogram} margin={{ left: 10, right: 10 }}>
                                    <XAxis dataKey="bucket" tick={{ fill: 'rgba(255,255,255,.4)', fontSize: 11 }} />
                                    <YAxis tick={{ fill: 'rgba(255,255,255,.4)', fontSize: 11 }} />
                                    <Tooltip
                                        contentStyle={{ background: '#1e1e3e', border: '1px solid rgba(255,255,255,.1)', borderRadius: 8 }}
                                        labelStyle={{ color: '#a855f7' }}
                                    />
                                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                                        {stats.histogram.map((_, i) => (
                                            <Cell key={i} fill={histogramColors[i % histogramColors.length]} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Recommendation */}
                    {recommendedSeqLen && (
                        <div className="tok-recommendation">
                            <span className="rec-icon">💡</span>
                            <div>
                                Based on P95 = <code>{stats.p95_tokens}</code> tokens, the recommended
                                <code>max_seq_length</code> is <code>{recommendedSeqLen}</code>.
                                {recommendedSeqLen < maxSeqLen && (
                                    <span> You can safely reduce from {maxSeqLen} to save GPU memory.</span>
                                )}
                                {stats.exceeding_max > 0 && (
                                    <span style={{ color: '#ff6b6b' }}>
                                        &nbsp;⚠ {stats.exceeding_max} samples ({((stats.exceeding_max / stats.total_samples) * 100).toFixed(1)}%) exceed your current max — they will be truncated.
                                    </span>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* ── Vocabulary Sample ────────────────────────────── */}
            <div className="tok-section">
                <h3><span className="icon">📖</span> Vocabulary Sample</h3>
                <div style={{ display: 'flex', gap: '.75rem', alignItems: 'center', marginBottom: '1rem' }}>
                    <button className="tok-btn" onClick={loadVocab} disabled={vocabLoading || !modelName.trim()}>
                        {vocabLoading ? '⏳ Loading...' : '📖 Load Vocab Sample'}
                    </button>
                    {vocab.length > 0 && (
                        <span style={{ color: 'rgba(255,255,255,.4)', fontSize: '.82rem' }}>{vocab.length} tokens shown</span>
                    )}
                </div>
                {vocab.length > 0 && (
                    <div className="tok-vocab-grid">
                        {vocab.map((v, i) => (
                            <div className="tok-vocab-item" key={i}>
                                <span className="token">{v.token || `[${v.id}]`}</span>
                                <span className="tid">{v.id}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <StepFooter
                currentStep="Tokenization"
                nextStep="Training"
                nextStepIcon="🔬"
                isComplete={!!stats}
                hint="Analyze your dataset tokens before training"
                onNext={onNextStep}
            />
        </div>
    );
}
