/**
 * PlaybookPickerPanel — USER-SUCCESS Epic 2.
 *
 * Recipe-aware synthetic-data generation surface. Mounted at the top
 * of the Synthetic tab; the legacy "generate Q&A pairs / spans /
 * conversations" UI stays below as the manual escape hatch.
 *
 * v1 ships POSITIVES_PARAPHRASE only across the 6 recipes — the mode
 * selector renders only what's available for the project's recipe.
 * Hard-negatives + cluster-targeted come in Epic 2b.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';

import type {
    CloudProviderEntry,
    DryRunPlaybookResult,
    OllamaModelInfo,
    PlaybookCatalogEntry,
    PlaybookResult,
    SynthBackendInfo,
    SynthMode,
} from '../../api/synthPlaybook';
import {
    dryRunPlaybook,
    listCloudModels,
    listOllamaModels,
    listPlaybooks,
    listSynthBackends,
    runPlaybookAsync,
} from '../../api/synthPlaybook';
import NoRecipeEmptyState from '../shared/NoRecipeEmptyState';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import './PlaybookPickerPanel.css';

const VALID_SYNTH_MODES: ReadonlySet<SynthMode> = new Set([
    'positives_paraphrase',
    'hard_negatives',
    'class_balance_fill',
    'edge_cases',
    'refusals',
    'format_robustness',
    'cluster_targeted',
]);

interface Props {
    projectId: number;
}

const MODE_LABELS: Record<SynthMode, { label: string; hint: string }> = {
    positives_paraphrase: {
        label: 'Paraphrase positives',
        hint: 'Generate alternative phrasings of existing gold rows. Same labels / answers, varied wording.',
    },
    hard_negatives: {
        label: 'Hard negatives',
        hint: 'Generate examples that look like one class but should be labeled another.',
    },
    class_balance_fill: {
        label: 'Balance class distribution',
        hint: 'Generate more examples for under-represented classes.',
    },
    edge_cases: {
        label: 'Edge cases',
        hint: 'Generate examples that stress test boundary conditions.',
    },
    refusals: {
        label: 'Refusals',
        hint: 'Generate examples the model should decline.',
    },
    format_robustness: {
        label: 'Format robustness',
        hint: 'Generate inputs in varied formats to make the model resilient.',
    },
    cluster_targeted: {
        label: 'Target a failure cluster',
        hint: 'Generate examples mirroring a specific eval failure pattern.',
    },
};

export default function PlaybookPickerPanel({ projectId }: Props) {
    const [available, setAvailable] = useState<PlaybookCatalogEntry[]>([]);
    const [recipeId, setRecipeId] = useState<string | null>(null);
    const [recipeRequired, setRecipeRequired] = useState(false);
    const [selectedMode, setSelectedMode] = useState<SynthMode | null>(null);
    const [targetCount, setTargetCount] = useState(30);
    // Forecast-prefill banner: when the user clicks a forecast action,
    // the TrainabilityForecastPanel routes here with
    // ?prefill_mode=<SynthMode>&prefill_count=<N>. We honor the prefill
    // ONLY when the requested mode is actually in the recipe's
    // catalog — silently fall back to the catalog default otherwise so
    // the user isn't dropped on an empty form.
    const location = useLocation();
    const prefill = useMemo(() => {
        const params = new URLSearchParams(location.search);
        const modeToken = (params.get('prefill_mode') || '').trim();
        const countToken = (params.get('prefill_count') || '').trim();
        const mode = VALID_SYNTH_MODES.has(modeToken as SynthMode)
            ? (modeToken as SynthMode)
            : null;
        const count = Number(countToken);
        return {
            mode,
            count: Number.isFinite(count) && count > 0
                ? Math.max(1, Math.min(500, Math.round(count)))
                : null,
            raw: modeToken,
        };
    }, [location.search]);
    const [prefillApplied, setPrefillApplied] = useState(false);
    const [prefillDismissed, setPrefillDismissed] = useState(false);
    const [catalogLoading, setCatalogLoading] = useState(true);
    const [catalogError, setCatalogError] = useState<string | null>(null);
    const [running, setRunning] = useState(false);
    const [runError, setRunError] = useState<string | null>(null);
    const [result, setResult] = useState<PlaybookResult | null>(null);
    // ── Backend picker (Epic 5 Phase 5a) ──────────────────────────
    // ``null`` selectedBackend means "auto-pick on the server" (the
    // default for v1 + every existing call site). The dropdown only
    // appears when 2+ backends are available; single-backend installs
    // see no clutter.
    const [backends, setBackends] = useState<SynthBackendInfo[]>([]);
    const [selectedBackend, setSelectedBackend] = useState<string | null>(null);
    // P3 — Ollama model picker. When the user pins a specific Ollama
    // model, ``selectedBackend`` becomes ``ollama:<tag>``; auto-pick
    // leaves it null and the server picks per PREFERRED_MODEL_PATTERNS.
    const [ollamaModels, setOllamaModels] = useState<OllamaModelInfo[]>([]);
    const [ollamaAutoPick, setOllamaAutoPick] = useState<string | null>(null);
    const [selectedOllamaModel, setSelectedOllamaModel] = useState<string | null>(null);
    // Cloud picker — OpenAI / Anthropic / Deepseek. ``cloudProvider``
    // null means 'don't use cloud' (Ollama/auto wins). When set, the
    // model dropdown shows only that provider's curated models. The
    // effective backend pin becomes ``cloud:<provider>:<model>`` and
    // the API layer resolves the saved key before instantiating.
    const [cloudProviders, setCloudProviders] = useState<CloudProviderEntry[]>([]);
    const [cloudProvider, setCloudProvider] = useState<string | null>(null);
    const [cloudModel, setCloudModel] = useState<string | null>(null);
    // P1 — pre-flight + inline diagnostic state. ``preflight`` holds
    // the dry-run result (refusal_detected etc.) so the panel can
    // render an inline error + retry-with-Qwen affordance instead of
    // sending the user to the notification bell for diagnostics.
    const [preflighting, setPreflighting] = useState(false);
    const [preflight, setPreflight] = useState<DryRunPlaybookResult | null>(null);

    useEffect(() => {
        // Backend listing is best-effort — if the endpoint 5xx's,
        // we silently keep the picker hidden and fall back to
        // auto-pick. The catalog fetch below is the load-bearing one.
        let cancelled = false;
        listSynthBackends(projectId)
            .then((data) => {
                if (cancelled) return;
                setBackends(data.backends || []);
            })
            .catch(() => {
                if (cancelled) return;
                setBackends([]);
            });
        // Same idea for the Ollama model list — only relevant when
        // Ollama is actually one of the available backends, but we
        // fetch unconditionally because the per-backend check is
        // cheap enough and folding the result is conditional below.
        listOllamaModels(projectId)
            .then((data) => {
                if (cancelled) return;
                setOllamaModels(data.models || []);
                setOllamaAutoPick(data.default);
            })
            .catch(() => {
                if (cancelled) return;
                setOllamaModels([]);
                setOllamaAutoPick(null);
            });
        // Cloud catalog is install-static (OpenAI / Anthropic /
        // Deepseek with curated model lists) — only the ``key_saved``
        // flag is per-project. Fail-soft if the endpoint 5xx's so the
        // panel still renders with just the local options.
        listCloudModels(projectId)
            .then((data) => {
                if (cancelled) return;
                setCloudProviders(data.providers || []);
            })
            .catch(() => {
                if (cancelled) return;
                setCloudProviders([]);
            });
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    // Resolve the effective backend string the server should use,
    // given the user's three picks (cloud > ollama model > backend
    // kind). null means "let the server auto-pick". Cloud wins
    // because it's the most-explicit choice — the user typed in a
    // provider AND a model.
    const effectiveBackend = useMemo<string | null>(() => {
        if (cloudProvider && cloudModel) {
            return `cloud:${cloudProvider}:${cloudModel}`;
        }
        if (selectedOllamaModel) return `ollama:${selectedOllamaModel}`;
        return selectedBackend;
    }, [selectedBackend, selectedOllamaModel, cloudProvider, cloudModel]);

    // The active cloud provider's catalog row (models + key_saved).
    // Used by the model dropdown + the 'Save key first' affordance.
    const activeCloudProvider = useMemo(
        () => cloudProviders.find((p) => p.provider === cloudProvider) ?? null,
        [cloudProviders, cloudProvider],
    );

    useEffect(() => {
        let cancelled = false;
        setCatalogLoading(true);
        setCatalogError(null);
        listPlaybooks(projectId)
            .then((data) => {
                if (cancelled) return;
                setAvailable(data.playbooks);
                setRecipeId(data.recipe_id);
                setRecipeRequired(Boolean(data.recipe_required));
                // Resolve the prefill *now* that we know which modes
                // the recipe actually ships. Honors the requested mode
                // only when it's in the catalog; falls back to the
                // catalog default otherwise. The count is independent
                // — applied whether the requested mode resolves or not.
                const requestedModeAvailable = prefill.mode
                    ? data.playbooks.some((p) => p.mode === prefill.mode)
                    : false;
                const targetMode: SynthMode | null = requestedModeAvailable
                    ? prefill.mode
                    : data.playbooks[0]?.mode ?? null;
                if (targetMode && !selectedMode) {
                    setSelectedMode(targetMode);
                }
                if (prefill.count !== null) {
                    setTargetCount(prefill.count);
                }
                if (prefill.mode || prefill.count !== null) {
                    setPrefillApplied(true);
                }
            })
            .catch((err) => {
                if (cancelled) return;
                setCatalogError(err?.response?.data?.detail || err?.message || 'Failed to load playbooks');
            })
            .finally(() => {
                if (!cancelled) setCatalogLoading(false);
            });
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    // Kicks the actual async job. Separated from handleRun so the
    // retry-with-Qwen button can call it directly after the model
    // is changed — without re-running the dry-run that we just did.
    // ``backendOverride`` is the explicit backend to use; required
    // for the retry path where ``effectiveBackend`` would be stale
    // (the useMemo hasn't recomputed against the just-set
    // ``selectedOllamaModel`` because React batches state updates).
    const submitJob = useCallback(async (backendOverride?: string | null) => {
        if (!selectedMode) return;
        setRunning(true);
        setRunError(null);
        const backend = backendOverride !== undefined ? backendOverride : effectiveBackend;
        try {
            const job = await runPlaybookAsync(projectId, {
                mode: selectedMode,
                targetCount,
                backend,
            });
            toast.info(
                `Synth started — track in the notification bell (↑ top-right)`,
                4000,
            );
            void useJobsStore.getState().refreshAfterLocalChange();
            setResult({
                rows: [],
                backend_used: `job #${job.id} queued`,
                elapsed_sec: 0,
                prompt_snippet: '',
            } as PlaybookResult);
        } catch (err: any) {
            const status = err?.response?.status;
            const detail = err?.response?.data?.detail;
            if (status === 503) {
                setRunError(
                    detail || 'No synthetic-data backend is available. Install Ollama or set TEACHER_MODEL_API_URL.',
                );
            } else {
                setRunError(detail || err?.message || 'Run failed');
            }
        } finally {
            setRunning(false);
        }
    }, [projectId, selectedMode, targetCount, effectiveBackend]);

    // P2 — pre-flight dry-run. Generates 1 row against the chosen
    // model + playbook, reports back in ~3-10s. If it succeeds, we
    // kick the real async job for ``targetCount`` rows. If it fails
    // (refusal, empty output, backend down), we render the inline
    // error here without ever invoking the long-running job.
    const handleRun = useCallback(async () => {
        if (!selectedMode) return;
        setPreflighting(true);
        setRunError(null);
        setResult(null);
        setPreflight(null);
        try {
            const result = await dryRunPlaybook(projectId, {
                mode: selectedMode,
                targetCount: 1,
                backend: effectiveBackend,
            });
            setPreflight(result);
            if (!result.ok) {
                // Failure surfaced inline; user has the retry button
                // or can switch models. Do NOT kick the async job.
                return;
            }
            // Pre-flight passed — kick the real job.
            await submitJob();
        } catch (err: any) {
            const status = err?.response?.status;
            const detail = err?.response?.data?.detail;
            setRunError(detail || err?.message || `Pre-flight failed (${status ?? '?'})`);
        } finally {
            setPreflighting(false);
        }
    }, [projectId, selectedMode, effectiveBackend, submitJob]);

    // P1 — "Retry with Qwen 2.5". Pick the largest qwen2.5 we know
    // about, swap the model picker to it, re-run the dry-run. We
    // resolve to a concrete tag (e.g. ``qwen2.5:14b-instruct-q4_K_M``)
    // so the next call doesn't auto-pick whatever happens to be first
    // alphabetically.
    const qwenFallback = useMemo<string | null>(() => {
        const qwens = ollamaModels
            .filter((m) => m.name.startsWith("qwen2.5"))
            // Largest size first (parameter_size like "14.8B", "7.6B"
            // parses to a comparable float).
            .sort((a, b) => {
                const score = (s: string) => parseFloat(s.replace(/[^0-9.]/g, "")) || 0;
                return score(b.parameter_size) - score(a.parameter_size);
            });
        return qwens[0]?.name ?? null;
    }, [ollamaModels]);

    const retryWithQwen = useCallback(async () => {
        if (!qwenFallback || !selectedMode) return;
        setSelectedOllamaModel(qwenFallback);
        // ``selectedBackend`` may still be set to a non-ollama
        // backend; clear it so the new model wins.
        setSelectedBackend(null);
        setPreflighting(true);
        setRunError(null);
        setResult(null);
        try {
            const result = await dryRunPlaybook(projectId, {
                mode: selectedMode,
                targetCount: 1,
                backend: `ollama:${qwenFallback}`,
            });
            setPreflight(result);
            if (result.ok) {
                // Pass the qwen backend explicitly — effectiveBackend
                // is a useMemo and won't reflect the just-called
                // setSelectedOllamaModel until React re-renders.
                await submitJob(`ollama:${qwenFallback}`);
            }
        } catch (err: any) {
            const detail = err?.response?.data?.detail;
            setRunError(detail || err?.message || "Retry failed");
        } finally {
            setPreflighting(false);
        }
    }, [projectId, selectedMode, qwenFallback, submitJob]);

    // Build picker option list from available backends. Hidden when
    // fewer than 2 are available (single-backend installs see no UI
    // clutter — they get the same auto-pick behavior as before).
    const availableBackends = backends.filter((b) => b.available);
    const showBackendPicker = availableBackends.length >= 2;
    // Phase 5c — surface which picks actually honor the playbook's
    // response_schema. Auto-pick mirrors the registry order, so resolve
    // schema-awareness for the *first available* backend when in auto.
    const anySchemaAware = availableBackends.some((b) => b.schema_aware);
    const activeBackend = selectedBackend
        ? availableBackends.find((b) => b.describe === selectedBackend)
        : availableBackends[0];
    const activeIsSchemaAware = Boolean(activeBackend?.schema_aware);

    if (catalogLoading) {
        return (
            <section className="playbook-picker playbook-picker--loading" data-testid="playbook-picker-loading">
                <p>Loading playbook catalog…</p>
            </section>
        );
    }

    if (catalogError) {
        return (
            <section className="playbook-picker playbook-picker--error" data-testid="playbook-picker-error">
                <p>{catalogError}</p>
            </section>
        );
    }

    if (available.length === 0) {
        // Legacy projects (pre-dating the auto-apply-on-create fix)
        // can hit ``recipe_required=true`` with an empty list — the
        // server stopped dumping the full cross-task-shape catalog
        // because none of those playbooks would actually run. Render
        // the shared directive CTA so legacy users see the same
        // prompt on every recipe-gated surface.
        if (recipeRequired) {
            return (
                <NoRecipeEmptyState
                    projectId={projectId}
                    surface="Synthetic playbooks"
                    testId="playbook-picker-empty-recipe-required"
                />
            );
        }
        return (
            <section
                className="playbook-picker playbook-picker--empty"
                data-testid="playbook-picker-empty"
            >
                <p>
                    {recipeId
                        ? `No playbooks shipped for the '${recipeId}' recipe yet. Use the manual generators below.`
                        : 'Select a recipe on the Data tab to enable playbook-driven synthesis.'}
                </p>
            </section>
        );
    }

    const requestedModeAvailable = prefill.mode
        ? available.some((p) => p.mode === prefill.mode)
        : false;
    const showPrefillBanner =
        prefillApplied
        && !prefillDismissed
        && (prefill.mode !== null || prefill.count !== null);

    return (
        <section className="playbook-picker" data-testid="playbook-picker">
            <header className="playbook-picker__head">
                <h3 className="playbook-picker__title">Synthetic data playbooks</h3>
                <p className="playbook-picker__subtitle">
                    {recipeId
                        ? <>Recipe: <strong>{recipeId}</strong></>
                        : 'No recipe set'}
                    {' · '}
                    {available.length} mode{available.length === 1 ? '' : 's'} available
                </p>
            </header>

            {showPrefillBanner && (
                <div
                    className="playbook-picker__prefill-banner"
                    role="status"
                    data-testid="playbook-picker-prefill-banner"
                >
                    <span>
                        Prefilled from trainability forecast
                        {prefill.mode && requestedModeAvailable
                            ? <>: mode <strong>{MODE_LABELS[prefill.mode]?.label || prefill.mode}</strong></>
                            : prefill.mode && !requestedModeAvailable
                                ? <> — requested mode <code>{prefill.raw}</code> isn't in this recipe; using the catalog default</>
                                : null}
                        {prefill.count !== null
                            ? <>, target <strong>{prefill.count}</strong> rows</>
                            : null}
                        .
                    </span>
                    <button
                        type="button"
                        className="playbook-picker__prefill-dismiss"
                        onClick={() => setPrefillDismissed(true)}
                        data-testid="playbook-picker-prefill-dismiss"
                        aria-label="Dismiss forecast prefill banner"
                    >
                        ×
                    </button>
                </div>
            )}

            <fieldset className="playbook-picker__modes" data-testid="playbook-picker-modes">
                <legend>Mode</legend>
                {available.map((entry) => {
                    const meta = MODE_LABELS[entry.mode];
                    const active = selectedMode === entry.mode;
                    return (
                        <label
                            key={entry.mode}
                            className={`playbook-picker__mode ${active ? 'is-active' : ''}`}
                            data-testid={`playbook-picker-mode-${entry.mode}`}
                        >
                            <input
                                type="radio"
                                name="playbook-mode"
                                value={entry.mode}
                                checked={active}
                                onChange={() => setSelectedMode(entry.mode)}
                            />
                            <span className="playbook-picker__mode-label">
                                <strong>{meta?.label || entry.mode}</strong>
                                <span className="playbook-picker__mode-hint">{meta?.hint || ''}</span>
                            </span>
                        </label>
                    );
                })}
            </fieldset>

            <div className="playbook-picker__count">
                <label htmlFor="playbook-target-count">Target rows</label>
                <input
                    id="playbook-target-count"
                    type="number"
                    min={1}
                    max={500}
                    value={targetCount}
                    onChange={(e) => setTargetCount(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
                    data-testid="playbook-picker-count"
                />
            </div>

            {showBackendPicker && (
                <>
                    <div className="playbook-picker__count" data-testid="playbook-picker-backend-row">
                        <label htmlFor="playbook-backend-picker">Backend</label>
                        <select
                            id="playbook-backend-picker"
                            value={selectedBackend ?? ''}
                            onChange={(e) =>
                                setSelectedBackend(e.target.value || null)
                            }
                            data-testid="playbook-picker-backend"
                        >
                            <option value="">Auto (recommended)</option>
                            {availableBackends.map((b) => (
                                <option key={b.name} value={b.describe}>
                                    {b.describe}
                                    {b.schema_aware ? ' · schema-aware' : ''}
                                </option>
                            ))}
                        </select>
                        {activeIsSchemaAware && (
                            <span
                                className="playbook-picker__schema-badge"
                                title="This backend forwards the playbook's JSON Schema as response_format=json_schema and enforces it during decoding."
                                data-testid="playbook-picker-schema-badge"
                            >
                                ✓ schema-aware
                            </span>
                        )}
                    </div>
                    {anySchemaAware && (
                        <p
                            className="playbook-picker__schema-hint"
                            data-testid="playbook-picker-schema-hint"
                        >
                            Schema-aware backends (NeMo, vLLM) constrain the
                            decoder to the playbook's <code>response_schema</code>.
                            Others run free-form — the playbook's parser still
                            validates after the fact.
                        </p>
                    )}
                </>
            )}

            {cloudProviders.length > 0 && (
                <div
                    className="playbook-picker__count"
                    data-testid="playbook-picker-cloud-row"
                >
                    <label htmlFor="playbook-cloud-provider-picker">
                        Cloud provider
                    </label>
                    <select
                        id="playbook-cloud-provider-picker"
                        value={cloudProvider ?? ''}
                        onChange={(e) => {
                            const next = e.target.value || null;
                            setCloudProvider(next);
                            // Reset the model when the provider changes
                            // so we never carry an OpenAI model id over
                            // to Anthropic's dropdown.
                            setCloudModel(null);
                            // Picking a cloud provider also clears the
                            // local Ollama pin — the effective backend
                            // is one or the other, not both.
                            if (next) setSelectedOllamaModel(null);
                        }}
                        data-testid="playbook-picker-cloud-provider"
                    >
                        <option value="">None (use local)</option>
                        {cloudProviders.map((p) => (
                            <option key={p.provider} value={p.provider}>
                                {p.provider}{p.key_saved ? ' ✓ key saved' : ' · no key'}
                            </option>
                        ))}
                    </select>
                    {activeCloudProvider && (
                        <select
                            value={cloudModel ?? ''}
                            onChange={(e) => setCloudModel(e.target.value || null)}
                            disabled={!activeCloudProvider.key_saved}
                            data-testid="playbook-picker-cloud-model"
                            aria-label={`${activeCloudProvider.provider} model`}
                        >
                            <option value="">Pick a model…</option>
                            {activeCloudProvider.models.map((m) => (
                                <option key={m.id} value={m.id}>{m.label}</option>
                            ))}
                        </select>
                    )}
                    {activeCloudProvider && !activeCloudProvider.key_saved && (
                        <p
                            className="playbook-picker__cloud-no-key"
                            data-testid="playbook-picker-cloud-no-key"
                        >
                            No <strong>{activeCloudProvider.provider}</strong> API
                            key saved on this project. Save one in Project
                            Settings → Secrets (or via the gold generator's
                            "Save key for this project" toggle) and reload
                            this panel.
                        </p>
                    )}
                </div>
            )}

            {ollamaModels.length > 0 && (
                <div
                    className="playbook-picker__count"
                    data-testid="playbook-picker-ollama-model-row"
                >
                    <label htmlFor="playbook-ollama-model-picker">
                        Ollama model
                    </label>
                    <select
                        id="playbook-ollama-model-picker"
                        value={selectedOllamaModel ?? ''}
                        onChange={(e) => {
                            const next = e.target.value || null;
                            setSelectedOllamaModel(next);
                            // Picking an Ollama model also clears the
                            // cloud pick — the user can only have one
                            // backend active at a time. The mirror of
                            // the cloud-provider onChange clear.
                            if (next) {
                                setCloudProvider(null);
                                setCloudModel(null);
                            }
                        }}
                        data-testid="playbook-picker-ollama-model"
                    >
                        <option value="">
                            {ollamaAutoPick
                                ? `Auto (${ollamaAutoPick})`
                                : 'Auto (server picks)'}
                        </option>
                        {ollamaModels.map((m) => (
                            <option key={m.name} value={m.name}>
                                {m.name}
                                {m.parameter_size
                                    ? ` · ${m.parameter_size}`
                                    : ''}
                            </option>
                        ))}
                    </select>
                </div>
            )}

            {/* Active-backend disclosure — single source of truth for
                "what will run when you click Generate". Resolves the
                cloud > ollama > auto precedence visibly so the user
                never has to wonder. */}
            <div
                className="playbook-picker__active-backend"
                data-testid="playbook-picker-active-backend"
                role="status"
            >
                <span className="playbook-picker__active-backend-label">
                    Will run on:
                </span>
                <code className="playbook-picker__active-backend-value">
                    {effectiveBackend
                        ?? (ollamaAutoPick
                            ? `auto · ollama:${ollamaAutoPick}`
                            : 'auto (server picks)')}
                </code>
            </div>

            <div className="playbook-picker__actions">
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleRun}
                    disabled={!selectedMode || running || preflighting}
                    data-testid="playbook-picker-run"
                >
                    {preflighting
                        ? 'Checking model…'
                        : running
                            ? 'Running…'
                            : 'Generate'}
                </button>
                {preflighting && (
                    <span
                        className="playbook-picker__preflight-hint"
                        data-testid="playbook-picker-preflight-hint"
                    >
                        Pre-flighting with 1 row to catch refusals or empty
                        output before the full run.
                    </span>
                )}
            </div>

            {preflight && !preflight.ok && (
                <div
                    className="playbook-picker__preflight-failure"
                    role="alert"
                    data-testid="playbook-picker-preflight-failure"
                >
                    {preflight.refusal_detected ? (
                        <>
                            <p className="playbook-picker__preflight-headline">
                                <strong>{preflight.backend_used}</strong> refused
                                to generate this playbook's content on guardrail
                                grounds.
                            </p>
                            <p className="playbook-picker__preflight-detail">
                                The model returned:{' '}
                                <code data-testid="playbook-picker-preflight-snippet">
                                    {preflight.raw_llm_snippet.slice(0, 240)}
                                </code>
                            </p>
                            {qwenFallback
                                && selectedOllamaModel !== qwenFallback && (
                                <button
                                    type="button"
                                    className="btn btn-primary"
                                    onClick={() => void retryWithQwen()}
                                    disabled={preflighting || running}
                                    data-testid="playbook-picker-retry-qwen"
                                >
                                    Retry with {qwenFallback}
                                </button>
                            )}
                            {!qwenFallback && (
                                <p className="playbook-picker__preflight-detail">
                                    No Qwen 2.5 model is installed locally.
                                    Pull one with{' '}
                                    <code>ollama pull qwen2.5:14b-instruct</code>{' '}
                                    and reload, or pick a different model
                                    from the dropdown above.
                                </p>
                            )}
                        </>
                    ) : preflight.error ? (
                        <p className="playbook-picker__preflight-headline">
                            Backend unavailable:{' '}
                            <code>{preflight.error}</code>
                        </p>
                    ) : (
                        <>
                            <p className="playbook-picker__preflight-headline">
                                Pre-flight produced 0 accepted rows via{' '}
                                <strong>{preflight.backend_used}</strong> in{' '}
                                {(preflight.elapsed_sec ?? 0).toFixed(1)}s.
                            </p>
                            <p className="playbook-picker__preflight-detail">
                                The model returned:{' '}
                                <code data-testid="playbook-picker-preflight-snippet">
                                    {preflight.raw_llm_snippet.slice(0, 240) || '(empty response)'}
                                </code>
                            </p>
                            <p className="playbook-picker__preflight-detail">
                                Try a different model from the dropdown above
                                or pick a different playbook mode.
                            </p>
                        </>
                    )}
                </div>
            )}

            {preflight && preflight.ok && running && (
                <p
                    className="playbook-picker__preflight-ok"
                    data-testid="playbook-picker-preflight-ok"
                >
                    Pre-flight passed via{' '}
                    <code>{preflight.backend_used}</code> in{' '}
                    {(preflight.elapsed_sec ?? 0).toFixed(1)}s — kicking the full
                    run.
                </p>
            )}

            {runError && (
                <p className="playbook-picker__error" data-testid="playbook-picker-run-error">
                    {runError}
                </p>
            )}

            {result && (
                <div className="playbook-picker__result" data-testid="playbook-picker-result">
                    <p className="playbook-picker__result-headline">
                        Generated <strong>{result.rows.length}</strong> rows in {result.elapsed_sec.toFixed(2)}s via{' '}
                        <code>{result.backend_used}</code>
                    </p>
                    <ul className="playbook-picker__result-preview">
                        {result.rows.slice(0, 3).map((row, idx) => (
                            <li key={idx}>
                                <span className="playbook-picker__confidence">
                                    {Math.round(row.synth_confidence * 100)}%
                                </span>
                                <code>{JSON.stringify(row.payload).slice(0, 220)}</code>
                            </li>
                        ))}
                    </ul>
                    {result.rows.length > 3 && (
                        <p className="playbook-picker__result-footnote">
                            +{result.rows.length - 3} more rows landed in your synthetic dataset (pending review).
                        </p>
                    )}
                </div>
            )}
        </section>
    );
}
