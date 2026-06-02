/**
 * Interactive prompt session manager supporting multiple inference backends and providers.
 */

import { useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import './ChatPlaygroundPanel.css';

type PlaygroundProvider = 'openai_compatible' | 'llama_cpp' | 'mock';
type PlaygroundRole = 'system' | 'user' | 'assistant';

/**
 * Arc 1 — per-turn provenance footer (adapter id + RAG hits + latency
 * vs. session average). The chat used to surface "what served this
 * reply" only at the bottom of the panel, and only for the most recent
 * turn. That made it impossible to compare turns or debug "why did the
 * model answer that?" without opening the network tab.
 *
 * The backend response already carries everything needed
 * (resolved_model_name, latency_ms, auto_rag.retrieved); this type
 * pins the subset the UI stashes on the message itself so the footer
 * survives across new turns + session restores.
 */
interface PlaygroundRagHit {
  rowId: string;
  score: number;
  preview: string;
}

interface PlaygroundMessageProvenance {
  adapterId: string | null;
  provider: string | null;
  latencyMs: number | null;
  ragApplied: boolean;
  ragHits: PlaygroundRagHit[];
  /** Backend may report a skip_reason when RAG was intended but no
   *  index was available — surface so the user understands the empty
   *  hit list isn't "no retrieval happened, model just answered". */
  ragSkipReason: string | null;
  /** First-class flag for reroute-to-RAG sibling projects (no LoRA
   *  loaded; serving the base model + retrieval). */
  ragFirstActive: boolean;
}

interface PlaygroundMessage {
  role: PlaygroundRole;
  content: string;
  createdAt: number;
  provenance?: PlaygroundMessageProvenance;
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

interface PlaygroundChatAutoRagBlock {
  applied?: boolean;
  k?: number;
  query?: string;
  retrieved?: Array<{
    row_id?: string | number;
    score?: number;
    payload?: Record<string, unknown>;
  }>;
  skip_reason?: string | null;
  preamble_inserted_at?: number;
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
  auto_rag?: PlaygroundChatAutoRagBlock;
  rag_first_active?: boolean;
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

/**
 * Arc 1 — extract a compact preview from a RAG-retrieved payload so
 * the footer can show context-snippet identity without dumping the
 * whole row. Walks the conventional QA-pair keys first (question,
 * answer, text) and falls back to JSON for shapes we don't recognise.
 * Capped to keep the footer readable.
 */
function _ragPayloadPreview(payload: Record<string, unknown> | null | undefined): string {
  if (!payload) return '';
  const cap = 140;
  for (const key of ['question', 'text', 'prompt', 'input', 'content']) {
    const value = (payload as Record<string, unknown>)[key];
    if (typeof value === 'string' && value.trim().length > 0) {
      const trimmed = value.trim().replace(/\s+/g, ' ');
      return trimmed.length > cap ? `${trimmed.slice(0, cap)}…` : trimmed;
    }
  }
  try {
    const text = JSON.stringify(payload);
    return text.length > cap ? `${text.slice(0, cap)}…` : text;
  } catch {
    return '';
  }
}

/**
 * Arc 1 — build the per-turn provenance bag from a chat response
 * (either the typed non-streaming body or a generic streaming SSE
 * `final` event). The two payload shapes diverge slightly so this
 * helper accepts both — keeping the streaming/non-streaming append
 * paths from drifting on what they stash.
 */
function _buildProvenance(
  source: PlaygroundChatResponse | Record<string, unknown>,
): PlaygroundMessageProvenance {
  const adapterId =
    (source as PlaygroundChatResponse).resolved_model_name
    || (source as Record<string, unknown>).resolved_model_name as string
    || (source as PlaygroundChatResponse).model_name
    || ((source as Record<string, unknown>).model_name as string)
    || null;
  const provider =
    (source as PlaygroundChatResponse).resolved_provider
    || ((source as Record<string, unknown>).resolved_provider as string)
    || (source as PlaygroundChatResponse).provider
    || ((source as Record<string, unknown>).provider as string)
    || null;
  const latencyRaw = (source as PlaygroundChatResponse).latency_ms
    ?? ((source as Record<string, unknown>).latency_ms as number | undefined);
  const latencyMs = Number.isFinite(Number(latencyRaw)) ? Number(latencyRaw) : null;
  const autoRag = ((source as PlaygroundChatResponse).auto_rag
    || ((source as Record<string, unknown>).auto_rag as PlaygroundChatAutoRagBlock | undefined)
    || {}) as PlaygroundChatAutoRagBlock;
  const retrieved = Array.isArray(autoRag.retrieved) ? autoRag.retrieved : [];
  const ragHits: PlaygroundRagHit[] = retrieved.map((chunk) => ({
    rowId: String(chunk?.row_id ?? ''),
    score: Number.isFinite(Number(chunk?.score)) ? Number(chunk?.score) : 0,
    preview: _ragPayloadPreview(chunk?.payload as Record<string, unknown>),
  }));
  const ragFirstActive = Boolean(
    (source as PlaygroundChatResponse).rag_first_active
    ?? ((source as Record<string, unknown>).rag_first_active as boolean | undefined),
  );
  return {
    adapterId: adapterId ? String(adapterId) : null,
    provider: provider ? String(provider) : null,
    latencyMs,
    ragApplied: Boolean(autoRag.applied),
    ragHits,
    ragSkipReason: autoRag.skip_reason ? String(autoRag.skip_reason) : null,
    ragFirstActive,
  };
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
  const [modelName, setModelName] = useState('HuggingFaceTB/SmolLM2-135M-Instruct');
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

  // Arc 1 — session-local average latency across all stored assistant
  // turns. Per-turn footer shows the delta vs. this average so the
  // user can see "this turn was 2x slower" without leaving the panel.
  const sessionAvgLatencyMs = useMemo(() => {
    const samples: number[] = [];
    for (const msg of messages) {
      const value = msg.provenance?.latencyMs;
      if (msg.role === 'assistant' && typeof value === 'number' && Number.isFinite(value)) {
        samples.push(value);
      }
    }
    if (samples.length === 0) return null;
    const total = samples.reduce((sum, n) => sum + n, 0);
    return total / samples.length;
  }, [messages]);

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
    const provenance = _buildProvenance(res.data || {});
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: reply,
        createdAt: Date.now(),
        provenance,
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
              const provenance = _buildProvenance(parsed);
              setMessages((prev) => [
                ...prev,
                {
                  role: 'assistant',
                  content: reply,
                  createdAt: Date.now(),
                  provenance,
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
        <div className="form-group playground-form-group--action">
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
        <div className="form-group playground-form-group--grow">
          <label className="form-label">RAG Compare Query</label>
          <input
            className="input"
            value={ragQuery}
            onChange={(e) => setRagQuery(e.target.value)}
            placeholder="Ask a question to compare base vs fine-tuned with retrieved snippets"
          />
        </div>
        <div className="form-group playground-form-group--action">
          <button className="btn btn-secondary" type="button" onClick={() => void runRagCompare()} disabled={ragLoading || !ragQuery.trim()}>
            {ragLoading ? 'Comparing...' : 'Run RAG Compare'}
          </button>
        </div>
      </div>
      {ragError && <div className="playground-error">{ragError}</div>}
      {ragResult && (
        <div className="playground-settings playground-settings--stretch">
          <div className="form-group playground-form-group--grow">
            <label className="form-label">
              Base Model
              {ragResult.base?.model_name ? ` (${ragResult.base.model_name})` : ''}
            </label>
            <textarea className="input playground-system" value={String(ragResult.base?.reply || '')} readOnly />
          </div>
          <div className="form-group playground-form-group--grow">
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
                {message.role === 'assistant' && message.provenance ? (
                  <MessageProvenanceFooter
                    provenance={message.provenance}
                    sessionAvgLatencyMs={sessionAvgLatencyMs}
                    messageKey={`${message.role}-${message.createdAt}`}
                  />
                ) : null}
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
          <div className="playground-feedback-actions">
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
        <div className="form-group playground-form-group--action">
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


/**
 * Arc 1 — collapsible per-turn provenance footer. Surfaces the three
 * "why did this turn answer that?" signals the user previously had to
 * open the network tab to see: which adapter served the reply, how
 * many RAG chunks were retrieved (and their top score), and how this
 * turn's latency compares to the session average.
 *
 * Click "context" to expand the retrieved chunks inline. When the
 * backend reports ``skip_reason`` (intended-RAG that fell back), the
 * footer flags it so the user understands the empty hit list isn't
 * "no retrieval ran".
 */
interface MessageProvenanceFooterProps {
  provenance: PlaygroundMessageProvenance;
  sessionAvgLatencyMs: number | null;
  /** Stable key so testids stay unique per message bubble. */
  messageKey: string;
}

function _truncateMid(value: string, cap: number = 40): string {
  if (value.length <= cap) return value;
  const half = Math.floor((cap - 1) / 2);
  return `${value.slice(0, half)}…${value.slice(value.length - half)}`;
}

function _formatLatencyDelta(
  current: number,
  avg: number | null,
): { label: string; tone: 'neutral' | 'faster' | 'slower' } | null {
  if (avg === null || avg <= 0) return null;
  const ratio = current / avg;
  // Treat ±10% as effectively-average — no point flagging noise.
  if (Math.abs(ratio - 1) < 0.10) {
    return { label: 'at session avg', tone: 'neutral' };
  }
  if (ratio < 1) {
    return { label: `${Math.round((1 - ratio) * 100)}% faster than avg`, tone: 'faster' };
  }
  return { label: `${Math.round((ratio - 1) * 100)}% slower than avg`, tone: 'slower' };
}

function MessageProvenanceFooter({
  provenance,
  sessionAvgLatencyMs,
  messageKey,
}: MessageProvenanceFooterProps) {
  const [expanded, setExpanded] = useState(false);
  const latencyDelta =
    typeof provenance.latencyMs === 'number'
      ? _formatLatencyDelta(provenance.latencyMs, sessionAvgLatencyMs)
      : null;
  const topRagScore =
    provenance.ragHits.length > 0
      ? Math.max(...provenance.ragHits.map((hit) => hit.score))
      : null;
  const hasRagBlock =
    provenance.ragApplied
    || provenance.ragHits.length > 0
    || provenance.ragSkipReason
    || provenance.ragFirstActive;

  return (
    <div
      className="playground-message__provenance"
      data-testid={`playground-provenance-${messageKey}`}
    >
      <div className="playground-message__provenance-row">
        {provenance.adapterId ? (
          <span
            className="playground-message__provenance-chip"
            title={`Served by ${provenance.adapterId}${provenance.provider ? ` (${provenance.provider})` : ''}`}
            data-testid={`playground-provenance-${messageKey}-adapter`}
          >
            via <code>{_truncateMid(provenance.adapterId)}</code>
          </span>
        ) : null}
        {provenance.latencyMs !== null ? (
          <span
            className={`playground-message__provenance-chip playground-message__provenance-chip--latency${latencyDelta ? ` playground-message__provenance-chip--${latencyDelta.tone}` : ''}`}
            data-testid={`playground-provenance-${messageKey}-latency`}
          >
            {provenance.latencyMs.toFixed(0)} ms
            {latencyDelta ? <small> ({latencyDelta.label})</small> : null}
          </span>
        ) : null}
        {hasRagBlock ? (
          <button
            type="button"
            className="playground-message__provenance-chip playground-message__provenance-chip--rag"
            onClick={() => setExpanded((value) => !value)}
            disabled={provenance.ragHits.length === 0}
            data-testid={`playground-provenance-${messageKey}-rag`}
            aria-label={
              provenance.ragHits.length === 0
                ? 'RAG retrieval block (no expandable hits)'
                : `${expanded ? 'Hide' : 'Show'} ${provenance.ragHits.length} retrieved RAG chunk${provenance.ragHits.length === 1 ? '' : 's'}`
            }
          >
            {provenance.ragHits.length > 0 ? (
              <>
                RAG: {provenance.ragHits.length} hit{provenance.ragHits.length === 1 ? '' : 's'}
                {topRagScore !== null ? (
                  <small> (top {topRagScore.toFixed(2)})</small>
                ) : null}
              </>
            ) : provenance.ragSkipReason ? (
              <>RAG skipped <small>({provenance.ragSkipReason})</small></>
            ) : (
              <>RAG-first base model</>
            )}
          </button>
        ) : null}
      </div>
      {expanded && provenance.ragHits.length > 0 ? (
        <ol
          className="playground-message__provenance-hits"
          data-testid={`playground-provenance-${messageKey}-hits`}
        >
          {provenance.ragHits.map((hit, idx) => (
            <li key={`${hit.rowId || idx}-${idx}`}>
              <code>{hit.rowId || `chunk #${idx + 1}`}</code>
              <small> score {hit.score.toFixed(3)}</small>
              {hit.preview ? <p>{hit.preview}</p> : null}
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
