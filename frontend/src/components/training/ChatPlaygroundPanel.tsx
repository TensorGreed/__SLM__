import { useEffect, useState } from 'react';

import api from '../../api/client';
import './ChatPlaygroundPanel.css';

type PlaygroundProvider = 'openai_compatible' | 'mock';
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
}

interface PlaygroundModelsResponse {
  default_model_name?: string;
  models?: PlaygroundModelOption[];
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
  reply: string;
  latency_ms?: number;
  session_id?: number | null;
}

interface ChatPlaygroundPanelProps {
  projectId: number;
}

const DEFAULT_API_URL = 'http://localhost:11434/v1/chat/completions';

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

  useEffect(() => {
    setMessages([]);
    setInput('');
    setStreamingReply('');
    setError('');
    setLastMeta(null);
    setActiveSessionId(null);
    void loadSessions();
    void loadModelOptions();
  }, [projectId]);

  const startNewChat = () => {
    setActiveSessionId(null);
    setMessages([]);
    setStreamingReply('');
    setInput('');
    setError('');
    setLastMeta(null);
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
      setProvider((String(detail?.provider || provider).trim().toLowerCase() === 'mock' ? 'mock' : 'openai_compatible'));
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
      provider: String(res.data?.provider || fallbackProvider),
      modelName: String(res.data?.model_name || fallbackModel),
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
              provider: String(parsed.provider || fallbackProvider),
              modelName: String(parsed.model_name || fallbackModel),
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
      api_url: provider === 'openai_compatible' ? apiUrl : undefined,
      api_key: provider === 'openai_compatible' && apiKey.trim() ? apiKey.trim() : undefined,
      system_prompt: systemPrompt.trim() || undefined,
      temperature: Number.isFinite(tempNumber) ? tempNumber : 0.2,
      max_tokens: maxTokens,
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
          <label className="form-label">Provider</label>
          <select className="input" value={provider} onChange={(e) => setProvider(e.target.value as PlaygroundProvider)}>
            <option value="mock">Mock (local, no model runtime)</option>
            <option value="openai_compatible">OpenAI-Compatible / Ollama</option>
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

      <div className="form-group">
        <label className="form-label">System Prompt (Optional)</label>
        <textarea
          className="input playground-system"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="You are a helpful assistant for domain-specific small language models."
        />
      </div>

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
