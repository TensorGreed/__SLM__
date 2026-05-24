import { useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    KeyRound,
    Sparkles,
} from 'lucide-react';

import {
    runDataStudioAssist,
} from '../../api/dataStudio';
import type {
    DataStudioAssistFocus,
    DataStudioAssistProvider,
    DataStudioAssistResponse,
} from '../../api/dataStudio';
import './DataStudioAssistPanel.css';

interface DataStudioAssistPanelProps {
    projectId: number;
}

function formatPercent(value: number | undefined): string {
    const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
    return `${Math.round(normalized * 100)}%`;
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function statusLabel(status: DataStudioAssistResponse['status'] | null): string {
    if (status === 'ok') return 'Suggestions ready';
    if (status === 'invalid_response') return 'Needs retry';
    if (status === 'unavailable') return 'Unavailable';
    return 'Optional';
}

export default function DataStudioAssistPanel({
    projectId,
}: DataStudioAssistPanelProps) {
    const [focus, setFocus] = useState<DataStudioAssistFocus>('mapping');
    const [provider, setProvider] = useState<DataStudioAssistProvider>('ollama');
    const [modelName, setModelName] = useState('llama3');
    const [apiUrl, setApiUrl] = useState('');
    const [apiKey, setApiKey] = useState('');
    const [assist, setAssist] = useState<DataStudioAssistResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const runAssist = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await runDataStudioAssist(projectId, {
                focus,
                provider,
                model_name: modelName.trim() || 'llama3',
                api_url: apiUrl.trim() || undefined,
                api_key: apiKey.trim() || undefined,
            });
            setAssist(data);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to run LLM assist.');
        } finally {
            setLoading(false);
        }
    };

    const status = assist?.status ?? null;

    return (
        <section
            className={`data-studio-assist data-studio-assist--${status || 'idle'}`}
            data-testid="data-studio-assist"
        >
            <div className="data-studio-assist__header">
                <div>
                    <p className="data-studio-assist__eyebrow">Assist</p>
                    <h3>LLM assist</h3>
                    <p>Deterministic checks stay in control; suggestions require review.</p>
                </div>
                <span className={`data-studio-assist__status data-studio-assist__status--${status || 'idle'}`}>
                    {statusLabel(status)}
                </span>
            </div>

            <div className="data-studio-assist__controls">
                <div className="data-studio-assist__focus" aria-label="Assist focus">
                    <button
                        type="button"
                        className={focus === 'mapping' ? 'is-active' : ''}
                        onClick={() => setFocus('mapping')}
                    >
                        Mapping
                    </button>
                    <button
                        type="button"
                        className={focus === 'domain' ? 'is-active' : ''}
                        onClick={() => setFocus('domain')}
                    >
                        Domain
                    </button>
                </div>

                <label>
                    Provider
                    <select
                        value={provider}
                        onChange={(event) => setProvider(event.target.value as DataStudioAssistProvider)}
                    >
                        <option value="ollama">Ollama</option>
                        <option value="openai_compatible">OpenAI-compatible</option>
                    </select>
                </label>

                <label>
                    Model
                    <input
                        value={modelName}
                        onChange={(event) => setModelName(event.target.value)}
                        placeholder="llama3"
                    />
                </label>

                <label>
                    Endpoint
                    <input
                        value={apiUrl}
                        onChange={(event) => setApiUrl(event.target.value)}
                        placeholder={provider === 'ollama' ? 'http://localhost:11434/v1/chat/completions' : 'https://.../v1/chat/completions'}
                    />
                </label>

                <label>
                    API key
                    <span className="data-studio-assist__secret">
                        <KeyRound size={14} aria-hidden="true" />
                        <input
                            type="password"
                            value={apiKey}
                            onChange={(event) => setApiKey(event.target.value)}
                            placeholder="optional"
                        />
                    </span>
                </label>

                <button
                    type="button"
                    className="btn btn-primary data-studio-assist__run"
                    onClick={() => void runAssist()}
                    disabled={loading}
                >
                    <Sparkles size={16} aria-hidden="true" />
                    {loading ? 'Running...' : 'Run assist'}
                </button>
            </div>

            {error ? (
                <div className="data-studio-assist__message data-studio-assist__message--error">
                    <AlertTriangle size={16} aria-hidden="true" />
                    <span>{error}</span>
                </div>
            ) : null}

            {assist ? (
                <div className="data-studio-assist__result">
                    <div className="data-studio-assist__summary">
                        {assist.status === 'ok' ? (
                            <CheckCircle2 size={16} aria-hidden="true" />
                        ) : (
                            <AlertTriangle size={16} aria-hidden="true" />
                        )}
                        <span>{assist.summary}</span>
                    </div>

                    {assist.suggestions.length > 0 ? (
                        <div className="data-studio-assist__suggestions">
                            {assist.suggestions.map((suggestion) => (
                                <article className="data-studio-assist__suggestion" key={suggestion.id}>
                                    <div className="data-studio-assist__suggestion-head">
                                        <strong>{suggestion.title}</strong>
                                        <span>{formatPercent(suggestion.confidence)}</span>
                                    </div>
                                    {suggestion.rationale ? <p>{suggestion.rationale}</p> : null}
                                    {suggestion.evidence.length > 0 ? (
                                        <ul>
                                            {suggestion.evidence.slice(0, 4).map((item) => (
                                                <li key={item}>{item}</li>
                                            ))}
                                        </ul>
                                    ) : null}
                                    {suggestion.suggested_field_mapping ? (
                                        <pre>{compactJson({ field_mapping: suggestion.suggested_field_mapping })}</pre>
                                    ) : null}
                                    <small>
                                        {suggestion.requires_user_confirmation ? 'Review required' : 'Review recommended'}
                                        {' · '}
                                        {suggestion.target_tab}
                                    </small>
                                </article>
                            ))}
                        </div>
                    ) : null}

                    <details className="data-studio-assist__details">
                        <summary>Power details</summary>
                        <pre>{compactJson(assist)}</pre>
                    </details>
                </div>
            ) : null}
        </section>
    );
}
