import { useEffect, useState } from 'react';

import api from '../../api/client';
import './ChatPlaygroundPanel.css';

type PlaygroundProvider = 'openai_compatible' | 'llama_cpp' | 'mock';
type PlaygroundRole = 'system' | 'user' | 'assistant';

interface PlaygroundMessage {
  role: PlaygroundRole;
  content: string;
  createdAt: number;
}

interface PlaygroundModelOption {
  model_name: string;
  label: string;
  source: string;
  recommended_provider?: PlaygroundProvider | string | null;
  runtime_hint?: {
    artifact_kind?: string;
    runtime_model_ref?: string;
    path_exists?: boolean;
    recommended_provider?: string | null;
  };
}

interface PlaygroundModelsResponse {
  default_model_name?: string;
  models?: PlaygroundModelOption[];
}

interface PlaygroundProviderSpec {
  provider: string;
  label?: string;
  description?: string;
  default_api_url?: string | null;
  supports_stream?: boolean;
  local_first?: boolean;
}

interface PlaygroundProviderCatalogResponse {
  providers?: PlaygroundProviderSpec[];
  default_provider?: string;
}

interface PlaygroundSessionSummary {
  id: number;
  title: string;
  provider: string;
  model_name: string;
  message_count: number;
  last_message_preview?: string;
  updated_at?: string | null;
}

interface PlaygroundSessionListResponse {
  sessions?: PlaygroundSessionSummary[];
}

interface PlaygroundSessionDetailResponse {
  id: number;
  title: string;
  provider?: string;
  model_name?: string;
  api_url?: string | null;
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
  messages?: Array<{ role?: string; content?: string }>;
}

interface PlaygroundChatResponse {
  provider: string;
  model_name: string;
  requested_model_name?: string;
  resolved_model_name?: string;
  resolved_provider?: string;
  runtime_hint?: {
    artifact_kind?: string;
    runtime_model_ref?: string;
    path_exists?: boolean;
    recommended_provider?: string | null;
  };
  reply: string;
  latency_ms?: number;
  session_id?: number | null;
}

interface PlaygroundLogEvent {
  event_id: string;
  timestamp?: string;
  rating?: number | null;
  tags?: string[];
  quality_checks?: Array<{ code?: string; severity?: string; message?: string }>;
}

interface PlaygroundLogSummary {
  event_count?: number;
  positive_count?: number;
  negative_count?: number;
  top_tags?: Array<{ tag?: string; count?: number }>;
  top_quality_issues?: Array<{ code?: string; count?: number }>;
}

interface PlaygroundLogListResponse {
  summary?: PlaygroundLogSummary;
  events?: PlaygroundLogEvent[];
}

interface RagSnippet {
  snippet_id: string;
  source_doc: string;
  score: number;
  text: string;
}

interface RagCompareResponse {
  retrieved_snippets?: RagSnippet[];
  base?: {
    model_name?: string;
    reply?: string;
    latency_ms?: number;
  };
  tuned?: {
    model_name?: string;
    reply?: string;
    latency_ms?: number;
  };
}

interface PromptPreset {
  id: string;
  label: string;
  prompt: string;
  systemPrompt?: string;
  tags: string[];
}

interface ChatPlaygroundPanelProps {
  projectId: number;
}

const DEFAULT_API_URL = 'http://localhost:11434/v1/chat/completions';
const DEFAULT_LLAMA_CPP_API_URL = 'http://localhost:8080/v1/chat/completions';
const PROMPT_PRESETS: PromptPreset[] = [
  {
    id: 'preset.summarize_contract',
    label: 'Summarize Domain Contract',
    prompt: 'Summarize the domain contract assumptions and output a concise checklist.',
    systemPrompt: 'You are a precise ML platform assistant. Respond with concise checklists.',
    tags: ['summarization', 'contract'],
  },
  {
    id: 'preset.generate_eval_cases',
    label: 'Generate Eval Cases',
    prompt: 'Generate 5 edge-case evaluation prompts for this domain and expected answer criteria.',
    tags: ['evaluation', 'edge-case'],
  },
  {
    id: 'preset.structured_extract',
    label: 'Structured Extraction',
    prompt: 'Return JSON with fields: entity, confidence, rationale for this input: <paste text>',
    systemPrompt: 'Return strict JSON only. No markdown.',
    tags: ['json', 'extraction'],
  },
  {
    id: 'preset.rag_grounded_answer',
    label: 'RAG Grounded Answer',
    prompt: 'Given the context below, answer only with grounded facts and cite snippet ids.',
    tags: ['rag', 'grounding'],
  },
];

function coerceRole(value: string): PlaygroundRole {
  const token = value.trim().toLowerCase();
  if (token === 'assistant' || token === 'system') {
    return token;
  }
  return 'user';
}

function authHeaders(): HeadersInit {
  const token = window.localStorage.getItem('slm_token');
  if (token && token.trim()) {
    return {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    };
  }
  return {
    'Content-Type': 'application/json',
  };
}

export default function ChatPlaygroundPanel({ projectId }: ChatPlaygroundPanelProps) {
  const [provider, setProvider] = useState<PlaygroundProvider>('mock');
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [modelName, setModelName] = useState('microsoft/phi-2');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState('0.2');
  const [maxTokens, setMaxTokens] = useState(512);
  const [streamEnabled, setStreamEnabled] = useState(true);
  const [messages, setMessages] = useState<PlaygroundMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingReply, setStreamingReply] = useState('');
  const [error, setError] = useState('');
  const [lastMeta, setLastMeta] = useState<{ provider: string; modelName: string; latencyMs: number | null } | null>(
    null,
  );
  const [sessions, setSessions] = useState<PlaygroundSessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [modelOptions, setModelOptions] = useState<PlaygroundModelOption[]>([]);
  const [providerSpecs, setProviderSpecs] = useState<PlaygroundProviderSpec[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState('');
  const [feedbackRating, setFeedbackRating] = useState<number | null>(null);
  const [feedbackTagsText, setFeedbackTagsText] = useState('');
  const [feedbackNotes, setFeedbackNotes] = useState('');
  const [feedbackSaving, setFeedbackSaving] = useState(false);
  const [feedbackSummary, setFeedbackSummary] = useState<PlaygroundLogSummary | null>(null);
  const [feedbackEvents, setFeedbackEvents] = useState<PlaygroundLogEvent[]>([]);
  const [ragQuery, setRagQuery] = useState('');
  const [ragLoading, setRagLoading] = useState(false);
  const [ragError, setRagError] = useState('');
  const [ragResult, setRagResult] = useState<RagCompareResponse | null>(null);

  const loadSessions = async () => {
    setSessionsLoading(true);
    try {
      const res = await api.get<PlaygroundSessionListResponse>(`/projects/${projectId}/training/playground/sessions`);
      const rows = Array.isArray(res.data?.sessions) ? res.data.sessions : [];
      setSessions(rows);
    } catch (err: unknown) {
      const message =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(message || 'Failed to load playground sessions.');
    } finally {
      setSessionsLoading(false);
    }
  };

  const loadModelOptions = async () => {
    try {
      const res = await api.get<PlaygroundModelsResponse>(`/projects/${projectId}/training/playground/models`);
      const rows = Array.isArray(res.data?.models) ? res.data.models : [];
      setModelOptions(rows);
      const defaultModel = String(res.data?.default_model_name || '').trim();
      if (!modelName.trim() && defaultModel) {
        setModelName(defaultModel);
      }
    } catch {
      setModelOptions([]);
    }
  };

  const loadProviderCatalog = async () => {
    try {
      const res = await api.get<PlaygroundProviderCatalogResponse>(`/projects/${projectId}/training/playground/providers`);
      const rows = Array.isArray(res.data?.providers) ? res.data.providers : [];
      setProviderSpecs(rows);
    } catch {
      setProviderSpecs([]);
    }
  };

  const loadFeedbackLogs = async () => {
    try {
      const res = await api.get<PlaygroundLogListResponse>(`/projects/${projectId}/training/playground/logs`, {
        params: { limit: 20 },
      });
      setFeedbackSummary(res.data?.summary || null);
      setFeedbackEvents(Array.isArray(res.data?.events) ? res.data.events : []);
    } catch {
      setFeedbackSummary(null);
      setFeedbackEvents([]);
    }
  };

  useEffect(() => {
    setMessages([]);
    setInput('');
    setStreamingReply('');
    setError('');
    setLastMeta(null);
    setActiveSessionId(null);
    setSelectedPresetId('');
    setFeedbackRating(null);
    setFeedbackTagsText('');
    setFeedbackNotes('');
    setRagQuery('');
    setRagError('');
    setRagResult(null);
    void loadSessions();
    void loadModelOptions();
    void loadProviderCatalog();
    void loadFeedbackLogs();
  }, [projectId]);

  useEffect(() => {
    if (provider === 'llama_cpp') {
      const current = apiUrl.trim();
      if (!current || current === DEFAULT_API_URL) {
        setApiUrl(DEFAULT_LLAMA_CPP_API_URL);
      }
      return;
    }
    if (provider === 'openai_compatible') {
      const current = apiUrl.trim();
      if (!current || current === DEFAULT_LLAMA_CPP_API_URL) {
        setApiUrl(DEFAULT_API_URL);
      }
    }
  }, [provider, apiUrl]);

  const startNewChat = () => {
    setActiveSessionId(null);
    setMessages([]);
    setStreamingReply('');
    setInput('');
    setError('');
    setLastMeta(null);
    setFeedbackRating(null);
    setFeedbackTagsText('');
    setFeedbackNotes('');
  };

  const openSession = async (sessionId: number) => {
    setError('');
    try {
      const res = await api.get<PlaygroundSessionDetailResponse>(
        `/projects/${projectId}/training/playground/sessions/${sessionId}`,
      );
      const detail = res.data;
      const transcript = Array.isArray(detail?.messages) ? detail.messages : [];
      const restored: PlaygroundMessage[] = transcript
        .map((item, idx) => {
          const role = coerceRole(String(item?.role || 'user'));
          const content = String(item?.content || '').trim();
          if (!content) return null;
          return {
            role,
            content,
            createdAt: Date.now() + idx,
          };
        })
        .filter((item): item is PlaygroundMessage => item !== null);

      setActiveSessionId(Number(detail?.id || sessionId));
      setMessages(restored);
      const sessionProvider = String(detail?.provider || provider).trim().toLowerCase();
      setProvider(
        sessionProvider === 'mock'
          ? 'mock'
          : sessionProvider === 'llama_cpp'
            ? 'llama_cpp'
            : 'openai_compatible',
      );
      if (detail?.model_name && String(detail.model_name).trim()) {
        setModelName(String(detail.model_name));
      }
      if (detail?.api_url && String(detail.api_url).trim()) {
        setApiUrl(String(detail.api_url));
      }
      setSystemPrompt(String(detail?.system_prompt || ''));
      if (Number.isFinite(Number(detail?.temperature))) {
        setTemperature(String(detail?.temperature));
      }
      if (Number.isFinite(Number(detail?.max_tokens))) {
        setMaxTokens(Math.max(16, Math.min(4096, Number(detail?.max_tokens))));
      }
    } catch (err: unknown) {
      const message =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(message || 'Failed to load session transcript.');
    }
  };

  const selectedPreset = PROMPT_PRESETS.find((item) => item.id === selectedPresetId) || null;
  const selectedModelOption =
    modelOptions.find((item) => item.model_name.toLowerCase() === modelName.trim().toLowerCase()) || null;

  const applyPromptPreset = () => {
    if (!selectedPreset) {
      return;
    }
    setInput(selectedPreset.prompt);
    if (selectedPreset.systemPrompt) {
      setSystemPrompt(selectedPreset.systemPrompt);
    }
  };

  const applyModelRecommendedProvider = () => {
    const recommended = String(
      selectedModelOption?.recommended_provider
      || selectedModelOption?.runtime_hint?.recommended_provider
      || '',
    ).trim().toLowerCase();
    if (!recommended) {
      return;
    }
    if (recommended === 'llama_cpp') {
      setProvider('llama_cpp');
      if (!apiUrl.trim()) {
        setApiUrl(DEFAULT_LLAMA_CPP_API_URL);
      }
      return;
    }
    if (recommended === 'openai_compatible') {
      setProvider('openai_compatible');
      if (!apiUrl.trim()) {
        setApiUrl(DEFAULT_API_URL);
      }
    }
  };

  const saveFeedback = async (forcedRating?: number | null) => {
    const assistantMessages = messages
      .map((item, idx) => ({ ...item, idx }))
      .filter((item) => item.role === 'assistant');
    const lastAssistant = assistantMessages[assistantMessages.length - 1];
    if (!lastAssistant) {
      setError('No assistant response available for feedback.');
      return;
    }
    const promptText = [...messages]
      .reverse()
      .find((item) => item.role === 'user')?.content || '';
    const tags = feedbackTagsText
      .split(',')
      .map((item) => item.trim().toLowerCase())
      .filter(Boolean);

    setFeedbackSaving(true);
    try {
      await api.post(`/projects/${projectId}/training/playground/logs`, {
        session_id: activeSessionId || undefined,
        message_index: lastAssistant.idx,
        provider: lastMeta?.provider || provider,
        model_name: lastMeta?.modelName || modelName,
        preset_id: selectedPreset?.id || undefined,
        prompt: promptText || '(no user prompt found)',
        reply: lastAssistant.content,
        rating: forcedRating ?? feedbackRating,
        tags,
        notes: feedbackNotes.trim() || undefined,
      });
      setFeedbackRating(null);
      setFeedbackTagsText('');
      setFeedbackNotes('');
      await loadFeedbackLogs();
    } catch (err: unknown) {
      const message =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(message || 'Failed to save feedback log.');
    } finally {
      setFeedbackSaving(false);
    }
  };

  const swipeFeedback = async (direction: 'left' | 'right') => {
    const rating = direction === 'right' ? 1 : -1;
    setFeedbackRating(rating);
    await saveFeedback(rating);
  };

  const sendMessageNonStreaming = async (
    payload: Record<string, unknown>,
    fallbackModel: string,
    fallbackProvider: PlaygroundProvider,
  ) => {
    const res = await api.post<PlaygroundChatResponse>(`/projects/${projectId}/training/playground/chat`, payload);
    const reply = String(res.data?.reply || '').trim();
    if (!reply) {
      setError('Playground returned an empty assistant reply.');
      return;
    }
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: reply,
        createdAt: Date.now(),
      },
    ]);
    setLastMeta({
      provider: String(res.data?.resolved_provider || res.data?.provider || fallbackProvider),
      modelName: String(res.data?.requested_model_name || res.data?.model_name || fallbackModel),
      latencyMs: Number.isFinite(Number(res.data?.latency_ms)) ? Number(res.data?.latency_ms) : null,
    });
    const nextSessionId = Number(res.data?.session_id || 0);
    if (nextSessionId > 0) {
      setActiveSessionId(nextSessionId);
    }
    await loadSessions();
  };

  const sendMessageStreaming = async (
    payload: Record<string, unknown>,
    fallbackModel: string,
    fallbackProvider: PlaygroundProvider,
  ) => {
    const response = await fetch(`/api/projects/${projectId}/training/playground/chat/stream`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`Playground stream failed (${response.status})`);
    }
    if (!response.body) {
      throw new Error('Playground stream did not return a body.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let draft = '';

    while (true) {
      const read = await reader.read();
      if (read.done) break;
      buffer += decoder.decode(read.value, { stream: true });

      const events = buffer.split('\n\n');
      buffer = events.pop() || '';
      for (const eventBlock of events) {
        const lines = eventBlock
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean);
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;

          const parsed = JSON.parse(raw) as Record<string, unknown>;
          const eventType = String(parsed.type || '').trim().toLowerCase();
          if (eventType === 'error') {
            throw new Error(String(parsed.detail || 'Playground stream error'));
          }
          if (eventType === 'delta') {
            const piece = String(parsed.content || '');
            if (piece) {
              draft += piece;
              setStreamingReply(draft);
            }
            continue;
          }
          if (eventType === 'final') {
            const reply = String(parsed.reply || draft).trim();
            setStreamingReply('');
            if (reply) {
              setMessages((prev) => [
                ...prev,
                {
                  role: 'assistant',
                  content: reply,
                  createdAt: Date.now(),
                },
              ]);
            } else {
              setError('Playground returned an empty assistant reply.');
            }

            setLastMeta({
              provider: String(parsed.resolved_provider || parsed.provider || fallbackProvider),
              modelName: String(parsed.requested_model_name || parsed.model_name || fallbackModel),
              latencyMs: Number.isFinite(Number(parsed.latency_ms)) ? Number(parsed.latency_ms) : null,
            });

            const nextSessionId = Number(parsed.session_id || 0);
            if (nextSessionId > 0) {
              setActiveSessionId(nextSessionId);
            }
            await loadSessions();
            return;
          }
        }
      }
    }

    setStreamingReply('');
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) {
      return;
    }

    const outgoing: PlaygroundMessage = {
      role: 'user',
      content: text,
      createdAt: Date.now(),
    };
    const nextMessages = [...messages, outgoing];
    setMessages(nextMessages);
    setInput('');
    setLoading(true);
    setError('');
    setStreamingReply('');

    const tempNumber = Number.parseFloat(temperature);
    const payload: Record<string, unknown> = {
      provider,
      model_name: modelName || undefined,
      api_url: provider === 'mock' ? undefined : apiUrl,
      api_key: provider !== 'mock' && apiKey.trim() ? apiKey.trim() : undefined,
      system_prompt: systemPrompt.trim() || undefined,
      temperature: Number.isFinite(tempNumber) ? tempNumber : 0.2,
      max_tokens: maxTokens,
      auto_runtime_provider: true,
      session_id: activeSessionId || undefined,
      save_history: true,
      messages: nextMessages.map((item) => ({
        role: item.role,
        content: item.content,
      })),
    };

    try {
      if (streamEnabled) {
        await sendMessageStreaming(payload, modelName, provider);
      } else {
        await sendMessageNonStreaming(payload, modelName, provider);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to get playground response.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const runRagCompare = async () => {
    const query = ragQuery.trim();
    if (!query || ragLoading) {
      return;
    }
    setRagLoading(true);
    setRagError('');
    try {
      const res = await api.post<RagCompareResponse>(`/projects/${projectId}/training/playground/rag-compare`, {
        query,
        provider,
        tuned_model_name: modelName || undefined,
        api_url: provider === 'mock' ? undefined : apiUrl,
        api_key: provider !== 'mock' && apiKey.trim() ? apiKey.trim() : undefined,
        temperature: Number.isFinite(Number.parseFloat(temperature)) ? Number.parseFloat(temperature) : 0.2,
        max_tokens: maxTokens,
        top_k: 4,
      });
      setRagResult(res.data || null);
    } catch (err: unknown) {
      const message =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setRagResult(null);
      setRagError(message || 'Failed to run RAG compare.');
    } finally {
      setRagLoading(false);
    }
  };

  return (
    <div className="card playground-panel">
      <div className="playground-panel__head">
        <div>
          <h3>Chat Playground</h3>
          <p>Stream responses, switch models, and persist prompt sessions in project history.</p>
        </div>
        <div className="playground-panel__actions">
          <button className="btn btn-secondary" onClick={startNewChat} disabled={loading}>
            New Chat
          </button>
          <button className="btn btn-secondary" onClick={() => void loadSessions()} disabled={loading || sessionsLoading}>
            Refresh Sessions
          </button>
        </div>
      </div>

      <div className="playground-sessions">
        <div className="playground-sessions__title">Saved Sessions</div>
        <div className="playground-sessions__list">
          {sessions.length === 0 ? (
            <div className="playground-sessions__empty">{sessionsLoading ? 'Loading...' : 'No saved sessions yet.'}</div>
          ) : (
            sessions.map((session) => (
              <button
                key={session.id}
                className={`playground-session-item ${activeSessionId === session.id ? 'active' : ''}`}
                onClick={() => void openSession(session.id)}
                disabled={loading}
              >
                <span className="playground-session-item__title">{session.title || `Session ${session.id}`}</span>
                <span className="playground-session-item__meta">{session.model_name || session.provider}</span>
              </button>
            ))
          )}
        </div>
      </div>

      <div className="playground-settings">
        <div className="form-group">
          <label className="form-label">Prompt Preset</label>
          <select className="input" value={selectedPresetId} onChange={(e) => setSelectedPresetId(e.target.value)}>
            <option value="">Select a preset...</option>
            {PROMPT_PRESETS.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.label}
              </option>
            ))}
          </select>
        </div>
        <div className="form-group" style={{ alignSelf: 'end' }}>
          <button className="btn btn-secondary" onClick={applyPromptPreset} disabled={!selectedPreset}>
            Insert Preset Prompt
          </button>
        </div>
      </div>

      <div className="playground-settings">
        <div className="form-group">
          <label className="form-label">Provider</label>
          <select className="input" value={provider} onChange={(e) => setProvider(e.target.value as PlaygroundProvider)}>
            {providerSpecs.length > 0 ? (
              providerSpecs.map((spec) => (
                <option key={spec.provider} value={spec.provider}>
                  {spec.label || spec.provider}
                </option>
              ))
            ) : (
              <>
                <option value="mock">Mock (local, no model runtime)</option>
                <option value="openai_compatible">OpenAI-Compatible / Ollama</option>
                <option value="llama_cpp">llama.cpp Server</option>
              </>
            )}
          </select>
        </div>
        <div className="form-group">
          <label className="form-label">Model</label>
          <input className="input" value={modelName} onChange={(e) => setModelName(e.target.value)} list="playground-models" />
          <datalist id="playground-models">
            {modelOptions.map((item) => (
              <option key={`${item.source}:${item.model_name}`} value={item.model_name}>
                {item.label}
              </option>
            ))}
          </datalist>
          {selectedModelOption?.runtime_hint?.artifact_kind && (
            <div className="form-hint">
              Detected artifact: <code>{selectedModelOption.runtime_hint.artifact_kind}</code>
              {selectedModelOption.runtime_hint.runtime_model_ref ? (
                <>
                  {' '}
                  • Runtime ref: <code>{selectedModelOption.runtime_hint.runtime_model_ref}</code>
                </>
              ) : null}
            </div>
          )}
          {selectedModelOption?.recommended_provider && (
            <button className="btn btn-secondary btn-sm" type="button" onClick={applyModelRecommendedProvider}>
              Use Suggested Provider ({selectedModelOption.recommended_provider})
            </button>
          )}
        </div>
        <div className="form-group">
          <label className="form-label">Temperature</label>
          <input className="input" value={temperature} onChange={(e) => setTemperature(e.target.value)} />
        </div>
        <div className="form-group">
          <label className="form-label">Max Tokens</label>
          <input
            className="input"
            type="number"
            min={16}
            max={4096}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Math.max(16, Math.min(4096, Number(e.target.value) || 16)))}
          />
        </div>
      </div>

      <div className="playground-settings">
        <label className="playground-toggle">
          <input type="checkbox" checked={streamEnabled} onChange={(e) => setStreamEnabled(e.target.checked)} />
          Stream responses
        </label>
      </div>

      {provider === 'openai_compatible' && (
        <div className="playground-settings">
          <div className="form-group">
            <label className="form-label">API URL</label>
            <input className="input" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} />
          </div>
          <div className="form-group">
            <label className="form-label">API Key (Optional)</label>
            <input className="input" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
          </div>
        </div>
      )}
      {provider === 'llama_cpp' && (
        <div className="playground-settings">
          <div className="form-group">
            <label className="form-label">llama.cpp API URL</label>
            <input className="input" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} />
          </div>
          <div className="form-group">
            <label className="form-label">API Key (Optional)</label>
            <input className="input" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
          </div>
        </div>
      )}

      <div className="form-group">
        <label className="form-label">System Prompt (Optional)</label>
        <textarea
          className="input playground-system"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="You are a helpful assistant for domain-specific small language models."
        />
      </div>

      <div className="playground-settings">
        <div className="form-group" style={{ flex: 1 }}>
          <label className="form-label">RAG Compare Query</label>
          <input
            className="input"
            value={ragQuery}
            onChange={(e) => setRagQuery(e.target.value)}
            placeholder="Ask a question to compare base vs fine-tuned with retrieved snippets"
          />
        </div>
        <div className="form-group" style={{ alignSelf: 'end' }}>
          <button className="btn btn-secondary" type="button" onClick={() => void runRagCompare()} disabled={ragLoading || !ragQuery.trim()}>
            {ragLoading ? 'Comparing...' : 'Run RAG Compare'}
          </button>
        </div>
      </div>
      {ragError && <div className="playground-error">{ragError}</div>}
      {ragResult && (
        <div className="playground-settings" style={{ alignItems: 'stretch' }}>
          <div className="form-group" style={{ flex: 1 }}>
            <label className="form-label">
              Base Model
              {ragResult.base?.model_name ? ` (${ragResult.base.model_name})` : ''}
            </label>
            <textarea className="input playground-system" value={String(ragResult.base?.reply || '')} readOnly />
          </div>
          <div className="form-group" style={{ flex: 1 }}>
            <label className="form-label">
              Fine-Tuned Model
              {ragResult.tuned?.model_name ? ` (${ragResult.tuned.model_name})` : ''}
            </label>
            <textarea className="input playground-system" value={String(ragResult.tuned?.reply || '')} readOnly />
          </div>
        </div>
      )}
      {ragResult && Array.isArray(ragResult.retrieved_snippets) && ragResult.retrieved_snippets.length > 0 && (
        <div className="playground-meta">
          Context snippets: {ragResult.retrieved_snippets.map((item) => item.snippet_id).join(', ')}
        </div>
      )}

      {error && <div className="playground-error">{error}</div>}
      {lastMeta && (
        <div className="playground-meta">
          Provider: <strong>{lastMeta.provider}</strong> • Model: <strong>{lastMeta.modelName}</strong>
          {lastMeta.latencyMs !== null ? ` • Latency: ${lastMeta.latencyMs.toFixed(1)} ms` : ''}
        </div>
      )}

      <div className="playground-conversation">
        {messages.length === 0 && !streamingReply ? (
          <div className="playground-empty">No messages yet. Send a prompt to start.</div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={`${message.role}-${message.createdAt}`}
                className={`playground-message ${message.role === 'assistant' ? 'assistant' : message.role === 'system' ? 'system' : 'user'}`}
              >
                <div className="playground-message__role">{message.role}</div>
                <div className="playground-message__content">{message.content}</div>
              </div>
            ))}
            {streamingReply ? (
              <div className="playground-message assistant streaming">
                <div className="playground-message__role">assistant</div>
                <div className="playground-message__content">{streamingReply}</div>
              </div>
            ) : null}
          </>
        )}
      </div>

      <div className="playground-settings">
        <div className="form-group">
          <label className="form-label">Response Feedback</label>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className={`btn btn-secondary btn-sm ${feedbackRating === 1 ? 'active' : ''}`}
              type="button"
              onClick={() => setFeedbackRating(1)}
              disabled={feedbackSaving}
            >
              Mark Good
            </button>
            <button
              className={`btn btn-secondary btn-sm ${feedbackRating === -1 ? 'active' : ''}`}
              type="button"
              onClick={() => setFeedbackRating(-1)}
              disabled={feedbackSaving}
            >
              Mark Bad
            </button>
            <button className="btn btn-secondary btn-sm" type="button" onClick={() => void swipeFeedback('left')} disabled={feedbackSaving}>
              Swipe Left (Reject)
            </button>
            <button className="btn btn-secondary btn-sm" type="button" onClick={() => void swipeFeedback('right')} disabled={feedbackSaving}>
              Swipe Right (Accept)
            </button>
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Tags (comma-separated)</label>
          <input
            className="input"
            value={feedbackTagsText}
            onChange={(e) => setFeedbackTagsText(e.target.value)}
            placeholder="hallucination, concise, grounded"
          />
        </div>
        <div className="form-group">
          <label className="form-label">Notes</label>
          <input
            className="input"
            value={feedbackNotes}
            onChange={(e) => setFeedbackNotes(e.target.value)}
            placeholder="Optional annotation for future DPO/ORPO dataset curation."
          />
        </div>
        <div className="form-group" style={{ alignSelf: 'end' }}>
          <button className="btn btn-secondary" type="button" onClick={() => void saveFeedback()} disabled={feedbackSaving}>
            {feedbackSaving ? 'Saving...' : 'Save Feedback Log'}
          </button>
        </div>
      </div>

      {feedbackSummary && (
        <div className="playground-meta">
          Feedback logs: {feedbackSummary.event_count || 0} • positive: {feedbackSummary.positive_count || 0} • negative:{' '}
          {feedbackSummary.negative_count || 0}
          {Array.isArray(feedbackSummary.top_quality_issues) && feedbackSummary.top_quality_issues.length > 0 ? (
            <> • top issue: {feedbackSummary.top_quality_issues[0]?.code || '—'}</>
          ) : null}
        </div>
      )}
      {feedbackEvents.length > 0 && (
        <div className="playground-meta">
          Latest check:{' '}
          {Array.isArray(feedbackEvents[0]?.quality_checks) && feedbackEvents[0].quality_checks?.length
            ? String(feedbackEvents[0].quality_checks?.[0]?.code || 'ok')
            : 'ok'}
        </div>
      )}

      <div className="playground-composer">
        <textarea
          className="input playground-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Write your prompt..."
        />
        <button className="btn btn-primary" onClick={() => void sendMessage()} disabled={loading || !input.trim()}>
          {loading ? (streamEnabled ? 'Streaming...' : 'Sending...') : 'Send'}
        </button>
      </div>
    </div>
  );
}
