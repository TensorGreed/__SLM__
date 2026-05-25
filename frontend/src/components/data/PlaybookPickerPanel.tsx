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

import { useCallback, useEffect, useState } from 'react';

import type {
    PlaybookCatalogEntry,
    PlaybookResult,
    SynthBackendInfo,
    SynthMode,
} from '../../api/synthPlaybook';
import {
    listPlaybooks,
    listSynthBackends,
    runPlaybook,
} from '../../api/synthPlaybook';
import './PlaybookPickerPanel.css';

interface Props {
    projectId: number;
    onRowsAccepted?: (result: PlaybookResult) => void;
}

const MODE_LABELS: Record<SynthMode, { label: string; hint: string }> = {
    positives_paraphrase: {
        label: 'Paraphrase positives',
        hint: 'Generate alternative phrasings of existing gold rows. Same labels / answers, varied wording.',
    },
    hard_negatives: {
        label: 'Hard negatives',
        hint: 'Generate examples that look like one class but should be labeled another. (Epic 2b)',
    },
    class_balance_fill: {
        label: 'Balance class distribution',
        hint: 'Generate more examples for under-represented classes. (Epic 2b)',
    },
    edge_cases: {
        label: 'Edge cases',
        hint: 'Generate examples that stress test boundary conditions. (Epic 2b)',
    },
    refusals: {
        label: 'Refusals',
        hint: 'Generate examples the model should decline. (Epic 2b)',
    },
    format_robustness: {
        label: 'Format robustness',
        hint: 'Generate inputs in varied formats to make the model resilient. (Epic 2b)',
    },
    cluster_targeted: {
        label: 'Target a failure cluster',
        hint: 'Generate examples mirroring a specific eval failure pattern. (Epic 2b)',
    },
};

export default function PlaybookPickerPanel({ projectId, onRowsAccepted }: Props) {
    const [available, setAvailable] = useState<PlaybookCatalogEntry[]>([]);
    const [recipeId, setRecipeId] = useState<string | null>(null);
    const [selectedMode, setSelectedMode] = useState<SynthMode | null>(null);
    const [targetCount, setTargetCount] = useState(30);
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
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    useEffect(() => {
        let cancelled = false;
        setCatalogLoading(true);
        setCatalogError(null);
        listPlaybooks(projectId)
            .then((data) => {
                if (cancelled) return;
                setAvailable(data.playbooks);
                setRecipeId(data.recipe_id);
                if (data.playbooks.length > 0 && !selectedMode) {
                    setSelectedMode(data.playbooks[0].mode);
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

    const handleRun = useCallback(async () => {
        if (!selectedMode) return;
        setRunning(true);
        setRunError(null);
        setResult(null);
        try {
            const r = await runPlaybook(projectId, {
                mode: selectedMode,
                targetCount,
                backend: selectedBackend,
            });
            setResult(r);
            onRowsAccepted?.(r);
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
    }, [projectId, selectedMode, targetCount, selectedBackend, onRowsAccepted]);

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
        return (
            <section className="playbook-picker playbook-picker--empty" data-testid="playbook-picker-empty">
                <p>
                    {recipeId
                        ? `No playbooks shipped for the '${recipeId}' recipe yet. Use the manual generators below.`
                        : 'Select a recipe on the Data tab to enable playbook-driven synthesis.'}
                </p>
            </section>
        );
    }

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

            <div className="playbook-picker__actions">
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleRun}
                    disabled={!selectedMode || running}
                    data-testid="playbook-picker-run"
                >
                    {running ? 'Running…' : 'Generate'}
                </button>
            </div>

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
