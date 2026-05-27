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


type Provider = 'openai' | 'anthropic';

interface GeneratedRow {
    question: string;
    answer: string;
    rationale: string;
    /** Present when grounding is on + the LLM included the field.
     *  The user uses this to spot-check that each answer is anchored
     *  to actual source material, not an LLM hallucination. */
    source_excerpt?: string;
}

interface GenerateResponse {
    rows: GeneratedRow[];
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

interface Props {
    projectId: number;
    datasetType: string;
    /** Called after rows are saved into the gold set so the parent
     *  panel can re-fetch its entries list. */
    onRowsSaved: () => void;
    /** Optional override for the provider's default model — useful
     *  for tests + future "remember the last picked model" plumbing. */
    initialProvider?: Provider;
}


const DEFAULT_MODELS: Record<Provider, { value: string; label: string }[]> = {
    openai: [
        { value: 'gpt-4o-mini', label: 'gpt-4o-mini (fast, cheap)' },
        { value: 'gpt-4o', label: 'gpt-4o (smarter, ~10× the cost)' },
    ],
    anthropic: [
        { value: 'claude-haiku-4-5-20251001', label: 'claude-haiku-4-5 (fast, cheap)' },
        { value: 'claude-sonnet-4-6', label: 'claude-sonnet-4-6 (smarter, ~5× the cost)' },
    ],
};


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
    const message = (err as { message?: string })?.message || 'Generation failed';
    return { code: 'UNKNOWN', message };
}


export default function LlmGoldGeneratePanel({
    projectId,
    datasetType,
    onRowsSaved,
    initialProvider,
}: Props) {
    const [provider, setProvider] = useState<Provider>(initialProvider || 'openai');
    const [model, setModel] = useState<string>(DEFAULT_MODELS.openai[0].value);
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

    // Reset the model dropdown when provider changes so the user never
    // sees a stale model string from the other provider.
    useEffect(() => {
        setModel(DEFAULT_MODELS[provider][0].value);
    }, [provider]);

    // Refetch the cost estimate whenever inputs that affect price
    // change. Debounced via the dependency list — React will batch
    // back-to-back changes (e.g. typing in the count input) but we
    // accept a slightly chatty endpoint here because the backend
    // does no LLM work (just chunk-char counting + price math).
    useEffect(() => {
        let cancelled = false;
        setCostEstimateLoading(true);
        api.post<CostEstimateResponse>(
            `/projects/${projectId}/gold/generate-via-llm/cost-estimate`,
            {
                provider,
                model,
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
    }, [projectId, provider, model, count, groundInSource]);

    const handleGenerate = async () => {
        if (generating) return;
        setGenerating(true);
        setGenError(null);
        setPreview(null);
        setSelectedIndexes(new Set());
        try {
            const res = await api.post<GenerateResponse>(
                `/projects/${projectId}/gold/generate-via-llm`,
                {
                    provider,
                    model,
                    count,
                    focus_hint: focusHint.trim() || undefined,
                    api_key: apiKey.trim() || undefined,
                    ground_in_source: groundInSource,
                },
                // Generation is sync; cap at 3 min so users on slower
                // models / larger counts aren't cut off by a default
                // axios timeout.
                { timeout: 180_000 },
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
            await api.post(`/projects/${projectId}/gold/import`, {
                pairs: selectedRows.map((r) => ({
                    question: r.question,
                    answer: r.answer,
                    // Existing /gold/import accepts difficulty / criticality
                    // / hallucination flags but lets them default. Rationale
                    // doesn't have a column today — drop it (the user has
                    // already used it for review).
                })),
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
                <h3 style={{ margin: 0 }}>✨ Generate Q&A with a flagship LLM</h3>
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
                    </select>
                </div>
                <div className="form-group" style={{ margin: 0 }}>
                    <label className="form-label">Model</label>
                    <select
                        className="input"
                        value={model}
                        onChange={(e) => setModel(e.target.value)}
                        data-testid="llm-gold-model"
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

            <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">
                    API key{' '}
                    <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>
                        (one-shot — or store under Project Secrets to reuse)
                    </span>
                </label>
                <input
                    className="input"
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={
                        provider === 'openai' ? 'sk-...' : 'sk-ant-...'
                    }
                    data-testid="llm-gold-api-key"
                    style={{ fontFamily: 'monospace' }}
                />
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
                    placeholder="e.g. focus on edge cases around refunds; cover questions a first-time customer would ask"
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
                            {preview.provider} · {preview.model} ·{' '}
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
                                    <div style={{ fontWeight: 600 }}>
                                        Q: {row.question}
                                    </div>
                                    <div style={{ marginTop: 4 }}>
                                        A: {row.answer}
                                    </div>
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
