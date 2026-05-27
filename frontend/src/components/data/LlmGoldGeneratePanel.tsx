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

import { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';

import api from '../../api/client';
import { toast } from '../../stores/toastStore';
import { parseApiErrorDetail } from '../../utils/apiError';
import GoldEntryRowBody from './GoldEntryRowBody';


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
    // qa-sft only — populated by the LLM (defaults applied when
    // missing): difficulty is one of "easy" | "medium" | "hard",
    // is_hallucination_trap a boolean. Both round-trip into the
    // saved gold-row JSONL.
    difficulty?: 'easy' | 'medium' | 'hard';
    is_hallucination_trap?: boolean;
}

/** qa-sft explicit row-mix. When set, the four counts add up to the
 *  total row count and replace the single "Count" input. Fired on
 *  the wire as ``distribution`` on /generate-via-llm and on
 *  /preview-prompt. */
interface RowDistribution {
    easy: number;
    medium: number;
    hard: number;
    hallucination_traps: number;
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

interface PromptPreviewResponse {
    recipe_id: RecipeId;
    system_prompt: string;
    user_prompt: string;
    reference_chunk_count: number;
    known_labels: string[];
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
    // Structured backend detail → done. ``parseApiErrorDetail`` handles
    // both ``detail: {error_code, message}`` and plain-string ``detail``.
    const parsed = parseApiErrorDetail(err);
    if (parsed) {
        // Tweak the default fallback message to the LLM-gen context
        // when parseApiErrorDetail's structured branch had to backfill
        // its generic "Request failed" placeholder.
        if (parsed.code !== 'UPSTREAM_ERROR' && parsed.message === 'Request failed') {
            return { ...parsed, message: 'Generation failed' };
        }
        return parsed;
    }

    // No HTTP response body — axios produces ``message: "Network Error"``
    // for connection-level failures (server unreachable, request
    // cancelled, proxy dropped the connection mid-response). When the
    // user has burned LLM tokens on a long reasoning model and gets
    // this back, the most likely cause is the request taking longer
    // than the frontend's axios timeout. Surface that explicitly so
    // they're not guessing. This LLM-specific copy is why the helper
    // is local to this panel — the shared ``parseApiErrorDetail`` is
    // recipe-agnostic, but the network-error framing is LLM-flavored.
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
 * Inline "review & edit prompt before sending" section. Shown when
 * the user clicked Generate with the advanced toggle on. Renders:
 *   * Recipe header so the user knows what shape the LLM will return.
 *   * Reference-chunk count + known-label list (when applicable) so
 *     they understand the grounding/vocab context.
 *   * Editable system + user prompt textareas with ~token counts.
 *   * Send / Cancel buttons.
 *
 * Edits are NOT auto-persisted — Cancel discards them.
 */
function PromptReviewSection({
    review,
    userPrompt,
    systemPrompt,
    onUserPromptChange,
    onSystemPromptChange,
    onSend,
    onCancel,
    generating,
}: {
    review: PromptPreviewResponse;
    userPrompt: string;
    systemPrompt: string;
    onUserPromptChange: (v: string) => void;
    onSystemPromptChange: (v: string) => void;
    onSend: () => void;
    onCancel: () => void;
    generating: boolean;
}) {
    // 1 token ≈ 4 chars heuristic (matches the cost estimator's
    // backend formula). Worth surfacing because users editing prompts
    // can blow out the cost ceiling without realizing it.
    const approxUserTokens = Math.ceil(userPrompt.length / 4);
    const approxSystemTokens = Math.ceil(systemPrompt.length / 4);

    return (
        <section
            data-testid="llm-gold-prompt-review"
            style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-md)',
                marginTop: 'var(--space-sm)',
                padding: 'var(--space-md)',
                border: '2px solid var(--accent-primary)',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-subtle)',
            }}
        >
            <header>
                <h4 style={{ margin: 0 }} data-testid="llm-gold-prompt-review-header">
                    🔍 Review prompt before sending — recipe: {review.recipe_id}
                </h4>
                <p
                    style={{
                        margin: '4px 0 0',
                        color: 'var(--text-secondary)',
                        fontSize: '0.9rem',
                    }}
                >
                    This is what the LLM will see. Edit either prompt
                    below; click <strong>Send to LLM</strong> when ready.
                    Token counts are approximate (chars ÷ 4) — bigger
                    prompts cost more, even before the LLM responds.
                </p>
                {review.reference_chunk_count > 0 && (
                    <p
                        style={{
                            margin: '4px 0 0',
                            color: 'var(--text-tertiary)',
                            fontSize: '0.85rem',
                        }}
                        data-testid="llm-gold-prompt-review-refs"
                    >
                        Grounded in <strong>{review.reference_chunk_count}</strong>{' '}
                        reference chunk{review.reference_chunk_count === 1 ? '' : 's'}
                        — the REFERENCE MATERIAL section in the user prompt
                        below carries them inline; edit/remove there.
                    </p>
                )}
                {review.known_labels.length > 0 && (
                    <p
                        style={{
                            margin: '4px 0 0',
                            color: 'var(--text-tertiary)',
                            fontSize: '0.85rem',
                        }}
                        data-testid="llm-gold-prompt-review-labels"
                    >
                        Locked to <strong>{review.known_labels.length}</strong>{' '}
                        label{review.known_labels.length === 1 ? '' : 's'}{' '}
                        from your existing gold rows:{' '}
                        <span style={{ fontFamily: 'monospace' }}>
                            {review.known_labels.join(', ')}
                        </span>
                        . If you change the LABEL VOCABULARY section in the
                        user prompt, the vocab filter is suspended on
                        parse — your edit wins.
                    </p>
                )}
            </header>

            <div className="form-group" style={{ margin: 0 }}>
                <label
                    className="form-label"
                    style={{ display: 'flex', justifyContent: 'space-between' }}
                >
                    <span>System prompt</span>
                    <span
                        style={{
                            fontWeight: 400,
                            color: 'var(--text-tertiary)',
                        }}
                        data-testid="llm-gold-prompt-review-system-tokens"
                    >
                        ≈ {approxSystemTokens} tokens
                    </span>
                </label>
                <textarea
                    aria-label="System prompt"
                    className="input"
                    value={systemPrompt}
                    onChange={(e) => onSystemPromptChange(e.target.value)}
                    rows={4}
                    data-testid="llm-gold-prompt-review-system"
                    style={{
                        fontFamily: 'monospace',
                        fontSize: '0.85rem',
                        resize: 'vertical',
                    }}
                />
            </div>

            <div className="form-group" style={{ margin: 0 }}>
                <label
                    className="form-label"
                    style={{ display: 'flex', justifyContent: 'space-between' }}
                >
                    <span>User prompt</span>
                    <span
                        style={{
                            fontWeight: 400,
                            color: 'var(--text-tertiary)',
                        }}
                        data-testid="llm-gold-prompt-review-user-tokens"
                    >
                        ≈ {approxUserTokens} tokens
                    </span>
                </label>
                <textarea
                    aria-label="User prompt"
                    className="input"
                    value={userPrompt}
                    onChange={(e) => onUserPromptChange(e.target.value)}
                    rows={20}
                    data-testid="llm-gold-prompt-review-user"
                    style={{
                        fontFamily: 'monospace',
                        fontSize: '0.85rem',
                        resize: 'vertical',
                    }}
                />
            </div>

            <div
                style={{
                    display: 'flex',
                    gap: 'var(--space-sm)',
                    justifyContent: 'flex-end',
                }}
            >
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={onCancel}
                    disabled={generating}
                    data-testid="llm-gold-prompt-review-cancel"
                >
                    Cancel
                </button>
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={onSend}
                    disabled={generating || !userPrompt.trim()}
                    data-testid="llm-gold-prompt-review-send"
                >
                    {generating ? '⏳ Sending…' : '✨ Send to LLM'}
                </button>
            </div>
        </section>
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

    // E1 — failure-cluster "Fix in gold set" deep link. The cluster
    // card on FailureClustersPanel emits a URL with focus_cluster_id +
    // focus_hint + trap_count; we apply once on mount to prefill the
    // focus textarea + the hallucination-trap row count. Banner stays
    // visible (dismissible) so the user sees what triggered the
    // prefill before generating.
    const location = useLocation();
    const clusterFix = useMemo(() => {
        const params = new URLSearchParams(location.search);
        const clusterId = params.get('focus_cluster_id');
        const hint = params.get('focus_hint');
        const trapCountRaw = params.get('trap_count');
        const trapCount = trapCountRaw !== null ? Number(trapCountRaw) : null;
        return {
            clusterId: clusterId?.trim() || null,
            hint: hint?.trim() || null,
            trapCount: Number.isFinite(trapCount) && trapCount! > 0
                ? Math.max(1, Math.min(20, Math.round(trapCount as number)))
                : null,
        };
    }, [location.search]);
    const [clusterFixApplied, setClusterFixApplied] = useState(false);
    const [clusterFixDismissed, setClusterFixDismissed] = useState(false);
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
    // Advanced "review & edit prompt before sending" workflow.
    //   * ``reviewPromptBeforeSend`` — opt-in checkbox state.
    //   * ``promptReview`` — populated by /preview-prompt when the
    //     user clicks Generate with the toggle on. While non-null,
    //     the panel renders the review section instead of firing
    //     the LLM call. Edits land in ``editedUserPrompt`` /
    //     ``editedSystemPrompt`` so we can preserve them across
    //     fetches if the user toggles between sections.
    const [reviewPromptBeforeSend, setReviewPromptBeforeSend] = useState(false);
    const [promptReview, setPromptReview] = useState<PromptPreviewResponse | null>(null);
    const [editedUserPrompt, setEditedUserPrompt] = useState('');
    const [editedSystemPrompt, setEditedSystemPrompt] = useState('');
    const [promptPreviewLoading, setPromptPreviewLoading] = useState(false);
    // qa-sft "customize row mix" — when on, the single Count field is
    // replaced with 4 numeric inputs (easy/medium/hard/traps). Default
    // OFF for new users; advanced users opt in. Only shown for qa-sft
    // because other recipes' gold_template doesn't carry difficulty /
    // hallucination-trap fields.
    const [customizeMix, setCustomizeMix] = useState(false);
    const [mix, setMix] = useState<RowDistribution>({
        easy: 5,
        medium: 3,
        hard: 2,
        hallucination_traps: 0,
    });

    // E1 — apply cluster-fix prefill once. Runs after the mix +
    // focusHint state slots are declared so the initial setters exist.
    // qa-sft is the only recipe whose mix carries hallucination_traps;
    // other recipes get the focus_hint prefill but skip the mix edit.
    useEffect(() => {
        if (clusterFixApplied) return;
        if (!clusterFix.clusterId && !clusterFix.hint && clusterFix.trapCount === null) {
            return;
        }
        if (clusterFix.hint) {
            setFocusHint(clusterFix.hint);
        }
        if (clusterFix.trapCount !== null && recipeId === 'qa-sft') {
            setCustomizeMix(true);
            setMix((prev) => ({ ...prev, hallucination_traps: clusterFix.trapCount! }));
        }
        setClusterFixApplied(true);
    }, [clusterFix, clusterFixApplied, recipeId]);
    const mixTotal = mix.easy + mix.medium + mix.hard + mix.hallucination_traps;
    const mixActive = recipeId === 'qa-sft' && customizeMix;
    const effectiveCount = mixActive ? mixTotal : count;
    const mixOutOfRange = mixActive && (mixTotal < 1 || mixTotal > 50);

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
                count: effectiveCount,
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
    }, [projectId, provider, model, customModel, count, groundInSource,
        // Re-run when the row-mix changes so the cost badge tracks
        // the mix-total rather than the stale single-count.
        mixActive, mix.easy, mix.medium, mix.hard, mix.hallucination_traps]);

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

    /** Persist the API key when the user checked "Save this key" —
     *  factored out of the generate path so both the direct-fire
     *  flow and the review-then-fire flow share the same logic.
     *  Non-fatal: failure surfaces a warning toast but the caller
     *  continues with the inline key. */
    const maybeSaveApiKey = async (trimmedKey: string) => {
        if (!saveForFuture || !trimmedKey) return;
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
    };

    /** Fire the LLM call. Used by both the direct-Generate flow and
     *  the review-then-send flow — the only difference is whether
     *  prompt overrides are passed in. */
    const fireGenerate = async (overrides?: {
        user_prompt_override?: string;
        system_prompt_override?: string;
    }) => {
        const trimmedKey = apiKey.trim();
        await maybeSaveApiKey(trimmedKey);
        const wire = wirePayloadDefaults();
        const res = await api.post<GenerateResponse>(
            `/projects/${projectId}/gold/generate-via-llm`,
            {
                provider: wire.provider,
                api_url: wire.api_url,
                model: effectiveModel,
                count: effectiveCount,
                focus_hint: focusHint.trim() || undefined,
                api_key: trimmedKey || undefined,
                ground_in_source: groundInSource,
                user_prompt_override: overrides?.user_prompt_override,
                system_prompt_override: overrides?.system_prompt_override,
                // qa-sft only — backend silently ignores for other
                // recipes. When the user hasn't opted into the mix
                // UI, ``mixActive`` is false and we send undefined so
                // the backend uses its default single-difficulty path.
                distribution: mixActive ? mix : undefined,
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
    };

    const handleGenerate = async () => {
        if (generating || promptPreviewLoading) return;
        setGenError(null);
        setPreview(null);
        setSelectedIndexes(new Set());

        // Review-mode fork: fetch the would-be prompt + open the
        // review section instead of firing the LLM immediately. The
        // user reviews/edits and clicks "Send to LLM" to commit.
        if (reviewPromptBeforeSend) {
            setPromptPreviewLoading(true);
            try {
                const res = await api.post<PromptPreviewResponse>(
                    `/projects/${projectId}/gold/generate-via-llm/preview-prompt`,
                    {
                        count: effectiveCount,
                        focus_hint: focusHint.trim() || undefined,
                        ground_in_source: groundInSource,
                        distribution: mixActive ? mix : undefined,
                    },
                );
                setPromptReview(res.data);
                setEditedUserPrompt(res.data.user_prompt);
                setEditedSystemPrompt(res.data.system_prompt);
            } catch (err) {
                setGenError(extractErrorMessage(err));
            } finally {
                setPromptPreviewLoading(false);
            }
            return;
        }

        setGenerating(true);
        try {
            await fireGenerate();
        } catch (err) {
            setGenError(extractErrorMessage(err));
        } finally {
            setGenerating(false);
        }
    };

    /** Commit from the review section: fire the LLM with the edited
     *  prompts as overrides. Falls back to defaults for any field
     *  the user blanked out (better than letting the backend reject
     *  an empty string). */
    const handleSendFromReview = async () => {
        if (!promptReview || generating) return;
        const userOverride = editedUserPrompt.trim();
        if (!userOverride) {
            toast.warning('User prompt is empty — cannot send.', 3000);
            return;
        }
        const systemOverride = editedSystemPrompt.trim();
        // Only send a system override when the user actually changed it;
        // sending the unchanged default works too but it's clearer for
        // the backend if we let it use its default in that case.
        const systemChanged = systemOverride !== promptReview.system_prompt.trim();
        const userChanged = userOverride !== promptReview.user_prompt.trim();
        setGenerating(true);
        setGenError(null);
        try {
            await fireGenerate({
                user_prompt_override: userChanged ? userOverride : undefined,
                system_prompt_override: systemChanged ? systemOverride : undefined,
            });
            // Clear review state on success so the next Generate
            // click starts fresh (with the now-current form values).
            setPromptReview(null);
            setEditedUserPrompt('');
            setEditedSystemPrompt('');
        } catch (err) {
            setGenError(extractErrorMessage(err));
        } finally {
            setGenerating(false);
        }
    };

    const handleCancelReview = () => {
        setPromptReview(null);
        setEditedUserPrompt('');
        setEditedSystemPrompt('');
        setGenError(null);
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
                        // Forward difficulty + trap labels when present;
                        // gold_service.import_qa_pairs preserves them.
                        return {
                            question: r.question,
                            answer: r.answer,
                            difficulty: r.difficulty,
                            is_hallucination_trap: r.is_hallucination_trap,
                        };
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

            {/* E1 — cluster-fix prefill banner. Shown when the user
                landed here via a failure cluster's "Fix in gold set"
                button. Carries the cluster id + the focus hint that
                was applied so the user can verify what got prefilled
                before clicking Generate. Dismiss clears the banner
                only (URL params + applied prefill stay). */}
            {clusterFixApplied
                && !clusterFixDismissed
                && (clusterFix.clusterId || clusterFix.hint || clusterFix.trapCount !== null) && (
                <div
                    data-testid="llm-gold-cluster-fix-banner"
                    style={{
                        padding: 'var(--space-sm) var(--space-md)',
                        border: '1px solid var(--color-warning, #f0a020)',
                        background: 'var(--color-warning-bg, rgba(240, 160, 32, 0.10))',
                        borderRadius: 'var(--radius-sm)',
                        color: 'var(--text-primary)',
                        fontSize: '0.9rem',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '6px',
                    }}
                >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 'var(--space-sm)' }}>
                        <strong>
                            Generating traps for cluster
                            {clusterFix.clusterId
                                ? <> <code>{clusterFix.clusterId}</code></>
                                : null}
                        </strong>
                        <button
                            type="button"
                            className="btn btn-link"
                            onClick={() => setClusterFixDismissed(true)}
                            data-testid="llm-gold-cluster-fix-dismiss"
                            style={{ padding: 0 }}
                            aria-label="Dismiss cluster-fix banner"
                        >
                            Dismiss
                        </button>
                    </div>
                    {clusterFix.hint && (
                        <div data-testid="llm-gold-cluster-fix-hint">
                            Focus hint applied: <em>{clusterFix.hint}</em>
                        </div>
                    )}
                    {clusterFix.trapCount !== null && recipeId === 'qa-sft' && (
                        <div data-testid="llm-gold-cluster-fix-trap-count">
                            Row mix prefilled with <strong>{clusterFix.trapCount}</strong> hallucination
                            trap{clusterFix.trapCount === 1 ? '' : 's'}.
                        </div>
                    )}
                    {clusterFix.trapCount !== null && recipeId !== 'qa-sft' && (
                        <div
                            data-testid="llm-gold-cluster-fix-trap-skip"
                            style={{ color: 'var(--text-tertiary)' }}
                        >
                            Hallucination-trap distribution is qa-sft-only; the focus hint still flows
                            through to the LLM prompt.
                        </div>
                    )}
                </div>
            )}

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
                    <label className="form-label">
                        {mixActive ? '# of rows (from mix below)' : '# of rows'}
                    </label>
                    <input
                        className="input"
                        type="number"
                        min={1}
                        max={50}
                        value={mixActive ? mixTotal : count}
                        onChange={(e) =>
                            setCount(Math.max(1, Math.min(50, Number(e.target.value) || 1)))
                        }
                        disabled={mixActive}
                        title={mixActive ? 'Total is computed from the row mix below.' : undefined}
                        data-testid="llm-gold-count"
                    />
                </div>
            </div>

            {/* qa-sft only: opt-in "customize row mix" UX so advanced
                users can request "5 easy, 3 medium, 2 hard + 2 traps"
                instead of all-defaults. The four counts replace the
                single Count field; total flows to the wire as the
                ``distribution`` payload. */}
            {recipeId === 'qa-sft' && (
                <div
                    className="form-group"
                    style={{ margin: 0 }}
                    data-testid="llm-gold-mix-group"
                >
                    <label
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 'var(--space-sm)',
                            fontSize: '0.9rem',
                            cursor: 'pointer',
                        }}
                        data-testid="llm-gold-mix-toggle-label"
                    >
                        <input
                            type="checkbox"
                            checked={customizeMix}
                            onChange={(e) => setCustomizeMix(e.target.checked)}
                            data-testid="llm-gold-mix-toggle"
                        />
                        <span>
                            <strong>Customize row mix</strong>{' '}
                            <span style={{ color: 'var(--text-tertiary)' }}>
                                (advanced — request a specific
                                difficulty / hallucination-trap distribution)
                            </span>
                        </span>
                    </label>
                    {customizeMix && (
                        <div
                            data-testid="llm-gold-mix-inputs"
                            style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                                gap: 'var(--space-sm)',
                                marginTop: 'var(--space-sm)',
                            }}
                        >
                            {(['easy', 'medium', 'hard', 'hallucination_traps'] as const).map(
                                (bucket) => (
                                    <div key={bucket}>
                                        <label
                                            className="form-label"
                                            style={{ fontSize: '0.85rem' }}
                                        >
                                            {bucket === 'hallucination_traps'
                                                ? 'Hallucination traps'
                                                : bucket[0].toUpperCase() + bucket.slice(1)}
                                        </label>
                                        <input
                                            aria-label={`${bucket} count`}
                                            className="input"
                                            type="number"
                                            min={0}
                                            max={50}
                                            value={mix[bucket]}
                                            onChange={(e) => {
                                                const v = Math.max(
                                                    0,
                                                    Math.min(50, Number(e.target.value) || 0),
                                                );
                                                setMix((prev) => ({ ...prev, [bucket]: v }));
                                            }}
                                            data-testid={`llm-gold-mix-${bucket.replace(/_/g, '-')}`}
                                        />
                                    </div>
                                ),
                            )}
                            <div
                                style={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    justifyContent: 'flex-end',
                                }}
                                data-testid="llm-gold-mix-total"
                            >
                                <div
                                    style={{
                                        fontSize: '0.85rem',
                                        color: mixOutOfRange
                                            ? 'var(--color-error)'
                                            : 'var(--text-secondary)',
                                    }}
                                >
                                    Total: <strong>{mixTotal}</strong>
                                    {mixOutOfRange && (
                                        <div data-testid="llm-gold-mix-total-error">
                                            Total must be 1–50. Generate is disabled until you fix it.
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

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
                    disabled={
                        generating
                        || promptPreviewLoading
                        || promptReview !== null
                        || mixOutOfRange
                    }
                    data-testid="llm-gold-generate"
                >
                    {generating
                        ? '⏳ Generating…'
                        : promptPreviewLoading
                            ? '⏳ Loading prompt…'
                            : reviewPromptBeforeSend
                                ? `🔍 Review prompt for ${effectiveCount} row${effectiveCount === 1 ? '' : 's'}`
                                : `✨ Generate ${effectiveCount} row${effectiveCount === 1 ? '' : 's'}`}
                </button>
                <label
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 'var(--space-sm)',
                        fontSize: '0.85rem',
                        cursor: 'pointer',
                    }}
                    data-testid="llm-gold-review-toggle-label"
                >
                    <input
                        type="checkbox"
                        checked={reviewPromptBeforeSend}
                        onChange={(e) => setReviewPromptBeforeSend(e.target.checked)}
                        disabled={generating || promptReview !== null}
                        data-testid="llm-gold-review-toggle"
                    />
                    <span>
                        Review &amp; edit prompt before sending
                        {' '}<span style={{ color: 'var(--text-tertiary)' }}>(advanced)</span>
                    </span>
                </label>
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

            {promptReview && (
                <PromptReviewSection
                    review={promptReview}
                    userPrompt={editedUserPrompt}
                    systemPrompt={editedSystemPrompt}
                    onUserPromptChange={setEditedUserPrompt}
                    onSystemPromptChange={setEditedSystemPrompt}
                    onSend={handleSendFromReview}
                    onCancel={handleCancelReview}
                    generating={generating}
                />
            )}

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
                                    <GoldEntryRowBody
                                        recipeId={preview.recipe_id}
                                        row={row}
                                        testidPrefix={`llm-gold-preview-row-${idx}`}
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
