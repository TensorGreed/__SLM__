import { useState } from 'react';

import api from '../../api/client';
import './ChatPlaygroundPanel.css';

type PlaygroundProvider = 'openai_compatible' | 'mock';

interface PlaygroundMessage {
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
}

interface PlaygroundChatResponse {
  project_id: number;
  message_count: number;
  provider: string;
  model_name: string;
  reply: string;
  endpoint?: string | null;
  latency_ms?: number;
  usage?: Record<string, unknown> | null;
}

interface ChatPlaygroundPanelProps {
  projectId: number;
}

const DEFAULT_API_URL = 'http://localhost:11434/v1/chat/completions';

export default function ChatPlaygroundPanel({ projectId }: ChatPlaygroundPanelProps) {
  const [provider, setProvider] = useState<PlaygroundProvider>('mock');
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [modelName, setModelName] = useState('microsoft/phi-2');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState('0.2');
  const [maxTokens, setMaxTokens] = useState(512);
  const [messages, setMessages] = useState<PlaygroundMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [lastMeta, setLastMeta] = useState<{ provider: string; modelName: string; latencyMs: number | null } | null>(null);

  const clearConversation = () => {
    setMessages([]);
    setInput('');
    setError('');
    setLastMeta(null);
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

    const tempNumber = Number.parseFloat(temperature);
    try {
      const res = await api.post<PlaygroundChatResponse>(
        `/projects/${projectId}/training/playground/chat`,
        {
          provider,
          model_name: modelName || undefined,
          api_url: provider === 'openai_compatible' ? apiUrl : undefined,
          api_key: provider === 'openai_compatible' && apiKey.trim() ? apiKey.trim() : undefined,
          system_prompt: systemPrompt.trim() || undefined,
          temperature: Number.isFinite(tempNumber) ? tempNumber : 0.2,
          max_tokens: maxTokens,
          messages: nextMessages.map((item) => ({
            role: item.role,
            content: item.content,
          })),
        },
      );
      const reply = String(res.data?.reply || '').trim();
      if (!reply) {
        setError('Playground returned an empty assistant reply.');
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: reply,
            createdAt: Date.now(),
          },
        ]);
      }
      setLastMeta({
        provider: String(res.data?.provider || provider),
        modelName: String(res.data?.model_name || modelName),
        latencyMs: Number.isFinite(Number(res.data?.latency_ms)) ? Number(res.data?.latency_ms) : null,
      });
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to get playground response.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card playground-panel">
      <div className="playground-panel__head">
        <div>
          <h3>Chat Playground (Phase 2 Scaffold)</h3>
          <p>Quickly test prompts using `mock` mode or any OpenAI-compatible endpoint (including Ollama).</p>
        </div>
        <div className="playground-panel__actions">
          <button className="btn btn-secondary" onClick={clearConversation} disabled={loading || messages.length === 0}>
            Clear
          </button>
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
          <input className="input" value={modelName} onChange={(e) => setModelName(e.target.value)} />
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
        {messages.length === 0 ? (
          <div className="playground-empty">No messages yet. Send a prompt to start.</div>
        ) : (
          messages.map((message) => (
            <div
              key={`${message.role}-${message.createdAt}`}
              className={`playground-message ${message.role === 'assistant' ? 'assistant' : 'user'}`}
            >
              <div className="playground-message__role">{message.role}</div>
              <div className="playground-message__content">{message.content}</div>
            </div>
          ))
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
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

