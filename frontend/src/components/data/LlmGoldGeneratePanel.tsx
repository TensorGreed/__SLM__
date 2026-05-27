/**
 * LlmGoldGeneratePanel — generate gold-set Q&A pairs using a flagship
 * cloud LLM (OpenAI or Anthropic). Mounted at the top of GoldSetPanel
 * for qa-sft projects; self-hides on other recipes (LLM-assisted
 * generation for classification + span-extraction comes in a follow-up
 * phase).
 *
 * Flow:
 *   1. User picks provider + model + count + optional focus hint,
 *      pastes their API key (one-shot — for persistence they use the
 *      project secrets surface, same as the synth path).
 *   2. Click Generate → backend builds a project/recipe-aware prompt,
 *      calls the LLM, parses + returns the rows. Sync request — gold
 *      generation runs in ~5–30s for 10 pairs.
 *   3. Generated rows render inline with per-row Accept checkboxes.
 *      User selects which to keep, clicks "Save selected" → bulk
 *      import via the existing /gold/import endpoint. "Discard" wipes
 *      the preview without saving anything.
 *
 * Gold rows are the evaluation ground truth — never auto-save. The
 * preview step is load-bearing: lets the user catch LLM hallucinations
 * before they poison evals.
 */

import { useEffect, useState } from 'react';

import api from '../../api/client';
import { toast } from '../../stores/toastStore';


/**
 * UI-level provider tag. ``deepseek`` is a UX label only — at the
 * wire layer it maps to ``provider=openai`` + the Deepseek host
 * via ``api_url`` (their API is OpenAI-compatible). Keeping the
 * three-way distinction in the UI gives the user clear model
 * defaults + a recognizable "I'm using Deepseek" affordance.
 */
type Provider = 'openai' | 'anthropic' | 'deepseek';

/** Recipe ids the panel renders for. Mirrors backend ``SUPPORTED_RECIPES``.
 *  Each recipe gets its own row preview + save-payload mapping. */
type RecipeId = 'qa-sft' | 'classification' | 'span-extraction' | 'summarization';

/** Recipe-shaped generated row. Each recipe carries different keys:
 *   * qa-sft: question + answer
 *   * classification: text + label
 *   * span-extraction: text + entities[]
 *   * summarization: document + summary
 *  Shared: rationale + source_excerpt (always optional). */
interface GeneratedSpan {
    type: string;
    start: number;
    end: number;
    text: string;
}

interface GeneratedRow {
    // qa-sft
    question?: string;
    answer?: string;
    // classification
    text?: string;
    label?: string;
    // span-extraction
    entities?: GeneratedSpan[];
    // summarization
    document?: string;
    summary?: string;
    // shared
    rationale?: string;
    source_excerpt?: string;
}

interface GenerateResponse {
    rows: GeneratedRow[];
    /** Recipe the rows are shaped for. Drives the per-recipe preview
     *  render + the save-payload mapping. */
    recipe_id: RecipeId;
    provider: Provider;
    model: string;
    usage: { prompt_tokens: number; completion_tokens: number };
    prompt_preview: string;
    reference_chunk_count: number;
    estimated_cost_usd: number;
}

interface CostEstimateResponse {
    estimated_cost_usd: number;
    estimated_prompt_tokens: number;
    estimated_completion_tokens: number;
    reference_chunk_count: number;
    ground_in_source_requested: boolean;
    ground_in_source_effective: boolean;
}

interface SavedKeyResponse {
    has_stored_key: boolean;
    value_hint: string | null;
}

interface Props {
    projectId: number;
    datasetType: string;
    /** Recipe id from the parent. The panel branches its prompt
     *  + row preview + save payload on this. Defaults to qa-sft
     *  for backward compat with callers that haven't been updated. */
    recipeId?: RecipeId;
    /** Called after rows are saved into the gold set so the parent
     *  panel can re-fetch its entries list. */
    onRowsSaved: () => void;
    /** Optional override for the provider's default model — useful
     *  for tests + future "remember the last picked model" plumbing. */
    initialProvider?: Provider;
}

/** Per-recipe UX copy: panel headline + focus-hint placeholder.
 *  Centralized so adding a fifth recipe is a one-row edit. */
const RECIPE_UX: Record<RecipeId, { headline: string; focusPlaceholder: string }> = {
    'qa-sft': {
        headline: '✨ Generate Q&A with a flagship LLM',
        focusPlaceholder:
            'e.g. focus on edge cases around refunds; cover questions a first-time customer would ask',
    },
    classification: {
        headline: '✨ Generate classification examples with a flagship LLM',
        focusPlaceholder:
            'e.g. labels: positive, negative, neutral; cover sarcasm + mixed-sentiment edge cases',
    },
    'span-extraction': {
        headline: '✨ Generate span-extraction examples with a flagship LLM',
        focusPlaceholder:
            'e.g. span types: email, phone, ssn; cover cases with multiple PII per row',
    },
    summarization: {
        headline: '✨ Generate (document, summary) pairs with a flagship LLM',
        focusPlaceholder:
            'e.g. meeting transcripts → executive summaries; keep summaries to 2-3 sentences',
    },
};


const DEFAULT_MODELS: Record<Provider, { value: string; label: string }[]> = {
    openai: [
        { value: 'gpt-4o-mini', label: 'gpt-4o-mini (fast, cheap)' },
        { value: 'gpt-4o', label: 'gpt-4o (smarter, ~10× the cost)' },
    ],
    anthropic: [
        { value: 'claude-haiku-4-5-20251001', label: 'claude-haiku-4-5 (fast, cheap)' },
        { value: 'claude-sonnet-4-6', label: 'claude-sonnet-4-6 (smarter, ~5× the cost)' },
    ],
    deepseek: [
        { value: 'deepseek-chat', label: 'deepseek-chat (V3 family — cheap, general purpose)' },
        { value: 'deepseek-reasoner', label: 'deepseek-reasoner (R1 family — slower, stronger reasoning)' },
    ],
};


/** Deepseek's API is OpenAI-compatible; we send ``provider=openai``
 *  + this URL on the wire when the user picks Deepseek in the UI. */
const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions';


function extractErrorMessage(err: unknown): { code: string; message: string } {
    const detail = (
        err as { response?: { data?: { detail?: unknown } } }
    )?.response?.data?.detail;
    if (detail && typeof detail === 'object') {
        const d = detail as { error_code?: unknown; message?: unknown };
        return {
            code: String(d.error_code || 'UNKNOWN'),
            message: String(d.message || 'Generation failed'),
        };
    }
    if (typeof detail === 'string') {
        return { code: 'UPSTREAM_ERROR', message: detail };
    }
    // No HTTP response body — axios produces ``message: "Network Error"``
    // for connection-level failures (server unreachable, request
    // cancelled, proxy dropped the connection mid-response). When the
    // user has burned LLM tokens on a long reasoning model and gets
    // this back, the most likely cause is the request taking longer
    // than the frontend's axios timeout. Surface that explicitly so
    // they're not guessing.
    const rawMessage = (err as { message?: string })?.message || '';
    const axiosCode = (err as { code?: string })?.code || '';
    if (
        rawMessage === 'Network Error'
        || axiosCode === 'ECONNABORTED'
        || axiosCode === 'ERR_NETWORK'
    ) {
        return {
            code: 'NETWORK_ERROR',
            message:
                'The browser couldn\'t complete the request. If LLM tokens '
                + 'were billed, the call reached the provider but the response '
                + 'didn\'t come back in time. Common causes: (a) reasoning / '
                + '"Pro" models took longer than the 7-min frontend timeout, '
                + '(b) Vite dev proxy or VPN dropped the connection, '
                + '(c) backend logged an error mid-write — check '
                + '`tail -f /tmp/uvicorn.log | grep cloud_llm` for the LLM '
                + 'call\'s actual duration + outcome. Try again with a '
                + 'lighter model (gpt-4o-mini, deepseek-chat) to isolate.',
        };
    }
    return { code: 'UNKNOWN', message: rawMessage || 'Generation failed' };
}


/**
 * Per-recipe preview-row body. Renders the recipe's load-bearing
 * fields with shape-appropriate framing:
 *   * qa-sft → Q + A
 *   * classification → text + label-as-badge
 *   * span-extraction → text + entity-table
 *   * summarization → document (collapsed) + summary
 */
function PreviewRowBody({
    recipeId,
    row,
    idx,
}: {
    recipeId: RecipeId;
    row: GeneratedRow;
    idx: number;
}) {
    if (recipeId === 'classification') {
        return (
            <>
                <div style={{ fontWeight: 600 }} data-testid={`llm-gold-preview-row-${idx}-text`}>
                    {row.text}
                </div>
                <div style={{ marginTop: 4 }} data-testid={`llm-gold-preview-row-${idx}-label`}>
                    <span
                        className="badge badge-info"
                        style={{ fontFamily: 'monospace' }}
                    >
                        {row.label}
                    </span>
                </div>
            </>
        );
    }
    if (recipeId === 'span-extraction') {
        const entities = row.entities || [];
        return (
            <>
                <div
                    style={{ fontFamily: 'monospace', fontSize: '0.9rem' }}
                    data-testid={`llm-gold-preview-row-${idx}-text`}
                >
                    {row.text}
                </div>
                <div
                    style={{ marginTop: 4, fontSize: '0.85rem' }}
                    data-testid={`llm-gold-preview-row-${idx}-entities`}
                >
                    {entities.length === 0 ? (
                        <em style={{ color: 'var(--text-tertiary)' }}>
                            No entities (negative example)
                        </em>
                    ) : (
                        <ul style={{ margin: 0, paddingLeft: 'var(--space-md)' }}>
                            {entities.map((ent, ei) => (
                                <li key={ei}>
                                    <span className="badge badge-accent" style={{ marginRight: 4 }}>
                                        {ent.type}
                                    </span>
                                    <span style={{ fontFamily: 'monospace' }}>
                                        "{ent.text}"
                                    </span>
                                    <span style={{ color: 'var(--text-tertiary)', marginLeft: 6 }}>
                                        [{ent.start}:{ent.end}]
                                    </span>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            </>
        );
    }
    if (recipeId === 'summarization') {
        // Document is potentially long — keep it readable but
        // collapsed-by-default so the row preview stays scannable.
        return (
            <>
                <details data-testid={`llm-gold-preview-row-${idx}-document`}>
                    <summary style={{ fontWeight: 600, cursor: 'pointer' }}>
                        Document ({(row.document || '').length} chars) — click to expand
                    </summary>
                    <div
                        style={{
                            marginTop: 4,
                            whiteSpace: 'pre-wrap',
                            fontSize: '0.9rem',
                            color: 'var(--text-secondary)',
                        }}
                    >
                        {row.document}
                    </div>
                </details>
                <div
                    style={{ marginTop: 4 }}
                    data-testid={`llm-gold-preview-row-${idx}-summary`}
                >
                    <strong>Summary:</strong> {row.summary}
                </div>
            </>
        );
    }
    // Default: qa-sft.
    return (
        <>
            <div style={{ fontWeight: 600 }}>
                Q: {row.question}
            </div>
            <div style={{ marginTop: 4 }}>
                A: {row.answer}
            </div>
        </>
    );
}


export default function LlmGoldGeneratePanel({
    projectId,
    datasetType,
    recipeId = 'qa-sft',
    onRowsSaved,
    initialProvider,
}: Props) {
    const recipeUx = RECIPE_UX[recipeId] || RECIPE_UX['qa-sft'];
    const [provider, setProvider] = useState<Provider>(initialProvider || 'openai');
    const [model, setModel] = useState<string>(DEFAULT_MODELS.openai[0].value);
    // Override slot for unusual / unreleased / private model ids
    // (e.g. a Deepseek "DeepSeek-V4-Pro" the dropdown doesn't carry,
    // or a fresh OpenAI / Anthropic SKU). When non-empty, this is
    // what gets sent on the wire — the dropdown becomes informational.
    const [customModel, setCustomModel] = useState('');
    const [apiKey, setApiKey] = useState('');
    const [count, setCount] = useState(10);
    const [focusHint, setFocusHint] = useState('');
    // Grounding: when ON, backend pulls a strict-budget sample of the
    // project's cleaned chunks into the prompt + asks the LLM to anchor
    // each answer in them. Default ON because ungrounded rows look
    // plausible but aren't tied to what the model could actually learn
    // — the exact failure mode "gold standard" exists to prevent.
    const [groundInSource, setGroundInSource] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [genError, setGenError] = useState<{ code: string; message: string } | null>(null);
    const [preview, setPreview] = useState<GenerateResponse | null>(null);
    const [selectedIndexes, setSelectedIndexes] = useState<Set<number>>(new Set());
    const [saving, setSaving] = useState(false);
    // Pre-call cost estimate — refetched whenever the inputs that
    // affect cost change (provider, model, count, grounding). Cheap
    // call: no LLM involvement, just chunk-char counting + pricing math.
    const [costEstimate, setCostEstimate] = useState<CostEstimateResponse | null>(null);
    const [costEstimateLoading, setCostEstimateLoading] = useState(false);
    // Stored-key UX state. The panel asks the backend "do you already
    // have an API key for this provider on this project?" on mount +
    // whenever the provider changes. When yes, the input is replaced
    // with a hint row ("Using stored key (sk-…xyz) · Replace · Remove")
    // so the user doesn't paste a fresh key every Generate click.
    const [storedKey, setStoredKey] = useState<SavedKeyResponse | null>(null);
    // True when the user has clicked Replace and wants to override the
    // stored key for this one Generate (without removing it server-side).
    const [replacingStoredKey, setReplacingStoredKey] = useState(false);
    // Mirrors the "Save this key for future generations" checkbox.
    // Default OFF — saving an API key crosses a privacy line, the user
    // has to opt in explicitly.
    const [saveForFuture, setSaveForFuture] = useState(false);

    // Reset the model dropdown when provider changes so the user never
    // sees a stale model string from the other provider. Custom-model
    // override is also cleared — they're per-provider tokens.
    useEffect(() => {
        setModel(DEFAULT_MODELS[provider][0].value);
        setCustomModel('');
        // Clear the Replace + Save toggles when provider changes — they
        // were per-provider state. saveForFuture stays off by default;
        // the user has to re-opt-in for the new provider.
        setReplacingStoredKey(false);
        setSaveForFuture(false);
        setApiKey('');
    }, [provider]);

    // Look up the stored API key for this project + provider on mount
    // and whenever the provider changes. The endpoint never returns
    // the raw key — only a masked hint + a boolean — so this can fire
    // freely without leaking the secret to the client.
    useEffect(() => {
        let cancelled = false;
        setStoredKey(null);
        api.get<SavedKeyResponse>(
            `/projects/${projectId}/gold/generate-via-llm/saved-key`,
            { params: { provider } },
        )
            .then((res) => {
                if (!cancelled) setStoredKey(res.data);
            })
            .catch(() => {
                // Best-effort lookup — if the endpoint errors, fall
                // back to the "no stored key" UX so the user can still
                // paste one inline.
                if (!cancelled) {
                    setStoredKey({ has_stored_key: false, value_hint: null });
                }
            });
        return () => { cancelled = true; };
    }, [projectId, provider]);

    /**
     * Effective model — custom override wins when the user has typed
     * something (after trimming). Otherwise the dropdown value flows
     * to the wire + the cost-estimate badge.
     */
    const effectiveModel = customModel.trim() || model;

    /**
     * The backend's ``/generate-via-llm`` payload uses
     * provider=openai|anthropic. Deepseek (and any future
     * OpenAI-compatible host) maps to provider=openai +
     * api_url=<that host>. Kept in one place so the cost-estimate
     * fetch + the real generate POST agree on the wire shape.
     */
    const wirePayloadDefaults = () => {
        if (provider === 'deepseek') {
            return {
                provider: 'openai' as const,
                api_url: DEEPSEEK_API_URL,
            };
        }
        return {
            provider: provider as 'openai' | 'anthropic',
            api_url: undefined as string | undefined,
        };
    };

    // Refetch the cost estimate whenever inputs that affect price
    // change. Debounced via the dependency list — React will batch
    // back-to-back changes (e.g. typing in the count input) but we
    // accept a slightly chatty endpoint here because the backend
    // does no LLM work (just chunk-char counting + price math).
    useEffect(() => {
        let cancelled = false;
        setCostEstimateLoading(true);
        const wire = wirePayloadDefaults();
        api.post<CostEstimateResponse>(
            `/projects/${projectId}/gold/generate-via-llm/cost-estimate`,
            {
                provider: wire.provider,
                model: effectiveModel,
                count,
                ground_in_source: groundInSource,
            },
        )
            .then((res) => {
                if (!cancelled) setCostEstimate(res.data);
            })
            .catch(() => {
                // Estimate is best-effort — if the endpoint errors
                // (network blip, project not found mid-render), just
                // suppress the badge rather than block Generate.
                if (!cancelled) setCostEstimate(null);
            })
            .finally(() => {
                if (!cancelled) setCostEstimateLoading(false);
            });
        return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId, provider, model, customModel, count, groundInSource]);

    /** Refetch the stored-key hint after a PUT / DELETE so the panel
     *  reflects the new state without a full reload. */
    const refetchStoredKey = async () => {
        try {
            const res = await api.get<SavedKeyResponse>(
                `/projects/${projectId}/gold/generate-via-llm/saved-key`,
                { params: { provider } },
            );
            setStoredKey(res.data);
        } catch {
            setStoredKey({ has_stored_key: false, value_hint: null });
        }
    };

    const handleRemoveStoredKey = async () => {
        try {
            await api.delete(
                `/projects/${projectId}/gold/generate-via-llm/saved-key`,
                { params: { provider } },
            );
            toast.success('Stored API key removed.', 3000);
        } catch {
            toast.error('Failed to remove the stored key — try again.');
        } finally {
            // Always refetch so the UI matches server state even if the
            // DELETE raced with a concurrent change in another tab.
            await refetchStoredKey();
            setReplacingStoredKey(false);
            setApiKey('');
            setSaveForFuture(false);
        }
    };

    const handleGenerate = async () => {
        if (generating) return;
        setGenerating(true);
        setGenError(null);
        setPreview(null);
        setSelectedIndexes(new Set());
        try {
            const trimmedKey = apiKey.trim();
            // Opt-in save: fire PUT before the generate POST when the
            // user checked "Save this key for future generations" AND
            // they actually typed a key. PUT failure is non-fatal —
            // we still attempt the generation with the inline key so
            // the user isn't blocked by a secret-store hiccup.
            if (saveForFuture && trimmedKey) {
                try {
                    await api.put<SavedKeyResponse>(
                        `/projects/${projectId}/gold/generate-via-llm/saved-key`,
                        { provider, api_key: trimmedKey },
                    );
                    await refetchStoredKey();
                    setSaveForFuture(false);
                    setReplacingStoredKey(false);
                    toast.success(
                        'API key saved — future generations will reuse it.',
                        3000,
                    );
                } catch {
                    toast.warning(
                        'Could not save the API key for reuse, '
                        + 'but proceeding with this one-shot generation.',
                        4000,
                    );
                }
            }

            const wire = wirePayloadDefaults();
            const res = await api.post<GenerateResponse>(
                `/projects/${projectId}/gold/generate-via-llm`,
                {
                    provider: wire.provider,
                    api_url: wire.api_url,
                    model: effectiveModel,
                    count,
                    focus_hint: focusHint.trim() || undefined,
                    api_key: trimmedKey || undefined,
                    ground_in_source: groundInSource,
                },
                // Generation is sync. Frontend timeout (7 min) sits
                // ABOVE the backend's per-LLM-call timeout (5 min, see
                // ``_DEFAULT_TIMEOUT_SECONDS`` in cloud_llm_service.py).
                // Earlier both were 180s, so a slow reasoning model
                // could race the two timeouts and the frontend would
                // give up with "Network Error" before the backend
                // could return a structured 502. Vite proxy is 10 min
                // so it stays above this whole chain. See
                // ``extractErrorMessage`` for the Network-Error branch
                // that fires when this chain still gets exceeded.
                { timeout: 420_000 },
            );
            setPreview(res.data);
            // Default to all rows selected — the user opted into
            // generation; preview's job is to let them deselect bad
            // rows, not re-opt-in.
            setSelectedIndexes(new Set(res.data.rows.map((_, i) => i)));
        } catch (err) {
            setGenError(extractErrorMessage(err));
        } finally {
            setGenerating(false);
        }
    };

    const toggleRow = (idx: number) => {
        setSelectedIndexes((prev) => {
            const next = new Set(prev);
            if (next.has(idx)) {
                next.delete(idx);
            } else {
                next.add(idx);
            }
            return next;
        });
    };

    const handleSaveSelected = async () => {
        if (!preview || saving) return;
        const selectedRows = preview.rows.filter((_, i) => selectedIndexes.has(i));
        if (selectedRows.length === 0) {
            toast.warning('Select at least one row before saving.', 3000);
            return;
        }
        setSaving(true);
        try {
            // Build the save payload per recipe — only the
            // recipe-relevant fields are sent so the JSONL row stays
            // clean. ``import_qa_pairs`` spreads ``**pair`` so any
            // extra keys we send round-trip into the file unchanged.
            // Rationale is dropped (the user already used it for
            // review; no eval handler reads it today).
            const previewRecipe = preview.recipe_id;
            const pairs = selectedRows.map((r) => {
                switch (previewRecipe) {
                    case 'classification':
                        return { text: r.text, label: r.label };
                    case 'span-extraction':
                        return { text: r.text, entities: r.entities || [] };
                    case 'summarization':
                        return { document: r.document, summary: r.summary };
                    case 'qa-sft':
                    default:
                        return { question: r.question, answer: r.answer };
                }
            });
            await api.post(`/projects/${projectId}/gold/import`, {
                pairs,
                dataset_type: datasetType,
            });
            toast.success(
                `Saved ${selectedRows.length} rows to ${
                    datasetType === 'gold_dev' ? 'Dev' : 'Test'
                } set.`,
                4000,
            );
            setPreview(null);
            setSelectedIndexes(new Set());
            onRowsSaved();
        } catch (err) {
            const e = extractErrorMessage(err);
            toast.error(`Save failed: ${e.message}`);
        } finally {
            setSaving(false);
        }
    };

    const handleDiscard = () => {
        setPreview(null);
        setSelectedIndexes(new Set());
        setGenError(null);
    };

    return (
        <section
            className="card"
            data-testid="llm-gold-generate-panel"
            style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}
        >
            <header>
                <h3 style={{ margin: 0 }}>{recipeUx.headline}</h3>
                <p
                    style={{
                        margin: '4px 0 0',
                        color: 'var(--text-secondary)',
                        fontSize: '0.9rem',
                    }}
                >
                    BrewSLM builds a project-aware prompt, calls the LLM you
                    pick, and shows the generated Q&A pairs for review.
                    Nothing is saved to the gold set until you click
                    "Save selected" below.
                </p>
            </header>

            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                    gap: 'var(--space-md)',
                }}
            >
                <div className="form-group" style={{ margin: 0 }}>
                    <label className="form-label">Provider</label>
                    <select
                        className="input"
                        value={provider}
                        onChange={(e) => setProvider(e.target.value as Provider)}
                        data-testid="llm-gold-provider"
                    >
                        <option value="openai">OpenAI</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="deepseek">Deepseek</option>
                    </select>
                </div>
                <div className="form-group" style={{ margin: 0 }}>
                    <label className="form-label">Model</label>
                    <select
                        className="input"
                        value={model}
                        onChange={(e) => setModel(e.target.value)}
                        data-testid="llm-gold-model"
                        disabled={!!customModel.trim()}
                        title={
                            customModel.trim()
                                ? "Custom model override is active — clear it to use a dropdown choice."
                                : undefined
                        }
                    >
                        {DEFAULT_MODELS[provider].map((m) => (
                            <option key={m.value} value={m.value}>
                                {m.label}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="form-group" style={{ margin: 0 }}>
                    <label className="form-label"># of Q&A pairs</label>
                    <input
                        className="input"
                        type="number"
                        min={1}
                        max={50}
                        value={count}
                        onChange={(e) =>
                            setCount(Math.max(1, Math.min(50, Number(e.target.value) || 1)))
                        }
                        data-testid="llm-gold-count"
                    />
                </div>
            </div>

            <div
                className="form-group"
                style={{ margin: 0 }}
                data-testid="llm-gold-api-key-group"
            >
                {storedKey?.has_stored_key && !replacingStoredKey ? (
                    // Stored-key row: shows the masked hint inline so
                    // the user knows what's about to be charged BEFORE
                    // clicking Generate. Replace reveals the input
                    // (one-shot override); Remove clears it on the
                    // server.
                    <div data-testid="llm-gold-stored-key-row">
                        <label className="form-label">API key</label>
                        <div
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 'var(--space-sm)',
                                flexWrap: 'wrap',
                                padding: 'var(--space-sm)',
                                background: 'var(--bg-subtle)',
                                borderRadius: 'var(--radius-sm)',
                                fontSize: '0.9rem',
                            }}
                        >
                            <span data-testid="llm-gold-stored-key-hint">
                                Using stored key{' '}
                                <strong style={{ fontFamily: 'monospace' }}>
                                    {storedKey.value_hint || '••••••'}
                                </strong>
                            </span>
                            <button
                                type="button"
                                className="btn btn-link"
                                onClick={() => setReplacingStoredKey(true)}
                                data-testid="llm-gold-stored-key-replace"
                            >
                                Replace
                            </button>
                            <button
                                type="button"
                                className="btn btn-link"
                                onClick={handleRemoveStoredKey}
                                data-testid="llm-gold-stored-key-remove"
                            >
                                Remove
                            </button>
                        </div>
                    </div>
                ) : (
                    <>
                        <label className="form-label">
                            API key{' '}
                            <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>
                                {storedKey?.has_stored_key
                                    ? '(replacing stored key for this run)'
                                    : '(one-shot — or save below to reuse)'}
                            </span>
                        </label>
                        <input
                            className="input"
                            type="password"
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                            placeholder={
                                provider === 'openai' ? 'sk-...'
                                    : provider === 'anthropic' ? 'sk-ant-...'
                                        : 'sk-...'
                            }
                            data-testid="llm-gold-api-key"
                            style={{ fontFamily: 'monospace' }}
                        />
                        {!storedKey?.has_stored_key && (
                            <label
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 'var(--space-sm)',
                                    marginTop: 'var(--space-sm)',
                                    fontSize: '0.85rem',
                                    cursor: 'pointer',
                                }}
                                data-testid="llm-gold-save-key-label"
                            >
                                <input
                                    type="checkbox"
                                    checked={saveForFuture}
                                    onChange={(e) => setSaveForFuture(e.target.checked)}
                                    data-testid="llm-gold-save-key-toggle"
                                />
                                <span>
                                    Save this key for future generations
                                    {' '}<span style={{ color: 'var(--text-tertiary)' }}>
                                        (encrypted server-side; only a masked hint
                                        comes back to the UI)
                                    </span>
                                </span>
                            </label>
                        )}
                        {storedKey?.has_stored_key && replacingStoredKey && (
                            <button
                                type="button"
                                className="btn btn-link"
                                onClick={() => {
                                    setReplacingStoredKey(false);
                                    setApiKey('');
                                }}
                                data-testid="llm-gold-stored-key-cancel-replace"
                                style={{ marginTop: 'var(--space-sm)' }}
                            >
                                Cancel — keep using stored key
                            </button>
                        )}
                    </>
                )}
            </div>

            <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">
                    Custom model id{' '}
                    <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>
                        (optional — overrides the dropdown)
                    </span>
                </label>
                <input
                    className="input"
                    type="text"
                    value={customModel}
                    onChange={(e) => setCustomModel(e.target.value)}
                    placeholder={
                        provider === 'deepseek'
                            ? 'e.g. DeepSeek-V4-Pro, deepseek-coder, your private SKU'
                            : provider === 'openai'
                                ? 'e.g. gpt-5, o4-mini, your fine-tuned SKU'
                                : 'e.g. claude-opus-4-7, your private SKU'
                    }
                    data-testid="llm-gold-custom-model"
                    style={{ fontFamily: 'monospace' }}
                />
                <div
                    style={{
                        marginTop: 4,
                        fontSize: '0.8rem',
                        color: 'var(--text-tertiary)',
                    }}
                >
                    Drop in any model the provider accepts — useful for
                    SKUs the dropdown doesn't carry yet (unreleased, regional,
                    fine-tuned). Cost estimate falls back to the cheapest-tier
                    price when the model id is unrecognized.
                </div>
            </div>

            <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">
                    Focus hint{' '}
                    <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>
                        (optional)
                    </span>
                </label>
                <textarea
                    className="input"
                    rows={2}
                    value={focusHint}
                    onChange={(e) => setFocusHint(e.target.value)}
                    placeholder={recipeUx.focusPlaceholder}
                    data-testid="llm-gold-focus-hint"
                />
            </div>

            <label
                style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 'var(--space-sm)',
                    fontSize: '0.9rem',
                    cursor: 'pointer',
                }}
                data-testid="llm-gold-ground-toggle-label"
            >
                <input
                    type="checkbox"
                    checked={groundInSource}
                    onChange={(e) => setGroundInSource(e.target.checked)}
                    data-testid="llm-gold-ground-toggle"
                    style={{ marginTop: 3 }}
                />
                <span>
                    <strong>Ground in this project's source material</strong>
                    {' '}<span style={{ color: 'var(--text-tertiary)' }}>(recommended)</span>
                    <div
                        style={{
                            color: 'var(--text-secondary)',
                            fontSize: '0.85rem',
                            marginTop: 2,
                        }}
                    >
                        Pulls a small sample of your cleaned chunks into the
                        prompt so the LLM anchors each answer to actual
                        source text (not its own training data). Adds a few
                        cents to the cost — still capped per the badge below.
                    </div>
                </span>
            </label>

            <div
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-md)',
                    flexWrap: 'wrap',
                }}
            >
                <button
                    className="btn btn-primary"
                    onClick={handleGenerate}
                    disabled={generating}
                    data-testid="llm-gold-generate"
                >
                    {generating ? '⏳ Generating…' : `✨ Generate ${count} Q&A pair${count === 1 ? '' : 's'}`}
                </button>
                <span
                    data-testid="llm-gold-cost-estimate"
                    style={{
                        color: 'var(--text-secondary)',
                        fontSize: '0.85rem',
                    }}
                >
                    {costEstimateLoading
                        ? 'Estimating cost…'
                        : costEstimate
                            ? (
                                <>
                                    ≈ <strong data-testid="llm-gold-cost-amount">
                                        ${costEstimate.estimated_cost_usd.toFixed(4)}
                                    </strong>{' '}
                                    estimated
                                    {costEstimate.ground_in_source_requested
                                        && !costEstimate.ground_in_source_effective
                                        && ' · grounding off (no cleaned chunks)'}
                                    {costEstimate.ground_in_source_effective
                                        && ` · grounded in ${costEstimate.reference_chunk_count} chunks`}
                                </>
                            )
                            : 'cost estimate unavailable'}
                </span>
            </div>

            {genError && (
                <div
                    role="alert"
                    data-testid="llm-gold-error"
                    style={{
                        padding: 'var(--space-md)',
                        background: 'var(--color-error-bg)',
                        color: 'var(--color-error)',
                        borderRadius: 'var(--radius-md)',
                        fontSize: '0.9rem',
                    }}
                >
                    <strong>{genError.code}:</strong> {genError.message}
                </div>
            )}

            {preview && preview.rows.length > 0 && (
                <div
                    data-testid="llm-gold-preview"
                    style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-sm)',
                        marginTop: 'var(--space-sm)',
                        paddingTop: 'var(--space-md)',
                        borderTop: '1px solid var(--border-color)',
                    }}
                >
                    <div
                        style={{
                            display: 'flex',
                            alignItems: 'baseline',
                            justifyContent: 'space-between',
                            gap: 'var(--space-md)',
                        }}
                    >
                        <h4 style={{ margin: 0 }}>
                            Review {preview.rows.length} generated rows
                        </h4>
                        <span
                            style={{
                                color: 'var(--text-tertiary)',
                                fontSize: '0.85rem',
                            }}
                            data-testid="llm-gold-preview-meta"
                        >
                            {preview.recipe_id} · {preview.provider} · {preview.model} ·{' '}
                            {preview.usage.prompt_tokens + preview.usage.completion_tokens} tokens
                            {' · '}
                            <strong data-testid="llm-gold-preview-cost">
                                ${preview.estimated_cost_usd.toFixed(4)}
                            </strong> spent
                            {preview.reference_chunk_count > 0
                                && ` · grounded in ${preview.reference_chunk_count} chunks`}
                        </span>
                    </div>
                    <p
                        style={{
                            margin: 0,
                            color: 'var(--text-secondary)',
                            fontSize: '0.85rem',
                        }}
                    >
                        Uncheck any row you want to skip. Selected rows save to
                        the {datasetType === 'gold_dev' ? 'Dev' : 'Test'} set.
                    </p>
                    {preview.rows.map((row, idx) => {
                        const checked = selectedIndexes.has(idx);
                        return (
                            <label
                                key={idx}
                                data-testid={`llm-gold-preview-row-${idx}`}
                                style={{
                                    display: 'flex',
                                    gap: 'var(--space-sm)',
                                    padding: 'var(--space-sm)',
                                    border: '1px solid var(--border-color)',
                                    borderRadius: 'var(--radius-sm)',
                                    background: checked
                                        ? 'var(--bg-card)'
                                        : 'var(--bg-subtle)',
                                    opacity: checked ? 1 : 0.55,
                                    cursor: 'pointer',
                                }}
                            >
                                <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() => toggleRow(idx)}
                                    data-testid={`llm-gold-preview-row-${idx}-toggle`}
                                    style={{ marginTop: 4 }}
                                />
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <PreviewRowBody
                                        recipeId={preview.recipe_id}
                                        row={row}
                                        idx={idx}
                                    />
                                    {row.rationale && (
                                        <div
                                            style={{
                                                marginTop: 4,
                                                fontSize: '0.85rem',
                                                color: 'var(--text-tertiary)',
                                                fontStyle: 'italic',
                                            }}
                                        >
                                            Why: {row.rationale}
                                        </div>
                                    )}
                                    {row.source_excerpt && (
                                        <div
                                            data-testid={`llm-gold-preview-row-${idx}-source`}
                                            style={{
                                                marginTop: 4,
                                                padding: '4px 8px',
                                                fontSize: '0.8rem',
                                                color: 'var(--text-secondary)',
                                                background: 'var(--bg-subtle)',
                                                borderLeft: '3px solid var(--accent-primary)',
                                                borderRadius: 'var(--radius-sm)',
                                            }}
                                        >
                                            <strong>From source:</strong> "{row.source_excerpt}"
                                        </div>
                                    )}
                                </div>
                            </label>
                        );
                    })}
                    <div
                        style={{
                            display: 'flex',
                            gap: 'var(--space-sm)',
                            justifyContent: 'flex-end',
                            marginTop: 'var(--space-sm)',
                        }}
                    >
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={handleDiscard}
                            disabled={saving}
                            data-testid="llm-gold-discard"
                        >
                            Discard
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={handleSaveSelected}
                            disabled={saving || selectedIndexes.size === 0}
                            data-testid="llm-gold-save"
                        >
                            {saving
                                ? 'Saving…'
                                : `Save ${selectedIndexes.size} of ${preview.rows.length}`}
                        </button>
                    </div>
                </div>
            )}
        </section>
    );
}
