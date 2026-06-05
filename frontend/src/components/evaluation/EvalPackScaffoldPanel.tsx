/**
 * EvalPackScaffoldPanel — recipe-aware draft eval-pack with inline
 * editing + one-click "Use scaffold" save (E5).
 *
 * Mounted under the eval-pack picker when the project has no
 * ``preferred_pack_id`` set. Auto-loads the recipe-derived draft;
 * lets the user tweak each gate's threshold / required flag before
 * persisting. POSTs to ``/pack-scaffold`` which flips the project's
 * preference to ``evalpack.project.scaffolded``, so the next eval
 * run uses the saved pack.
 *
 * Advisory only — nothing is persisted until the user clicks "Use
 * scaffold". A "Discard" button clears edits without touching the
 * project (the GET endpoint stays the source of truth so the user
 * can re-pull the recipe defaults).
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import type {
    GateMetricOption,
    GateOptionsResponse,
    PerClassMetricOption,
    PerClassMetricOptionsResponse,
    ScaffoldDraftPack,
    ScaffoldGate,
    ScaffoldResponse,
} from '../../api/evalPackScaffold';
import {
    fetchGateOptions,
    fetchPackScaffold,
    fetchPerClassMetricOptions,
    savePackScaffold,
} from '../../api/evalPackScaffold';
import { toast } from '../../stores/toastStore';
import './EvalPackScaffoldPanel.css';


interface Props {
    projectId: number;
    onSaved?: (preferredPackId: string) => void;
}


function cloneGate(gate: ScaffoldGate): ScaffoldGate {
    return { ...gate };
}


function cloneDraft(draft: ScaffoldDraftPack): ScaffoldDraftPack {
    return {
        ...draft,
        tags: [...(draft.tags || [])],
        task_specs: (draft.task_specs || []).map((spec) => ({
            ...spec,
            required_metric_ids: [...(spec.required_metric_ids || [])],
            gates: (spec.gates || []).map(cloneGate),
        })),
        gates: (draft.gates || []).map(cloneGate),
    };
}


/** Build a unique gate_id slug from a metric_id, avoiding collisions
 *  with the existing gate_ids in the task_spec. Pattern matches the
 *  scaffolder's own convention: ``min_<metric_id>`` for gte gates,
 *  ``max_<metric_id>`` for lte. Returns ``min_<metric_id>_2``,
 *  ``..._3`` etc. when the natural slug is taken. */
function makeUniqueGateId(
    metricId: string,
    operator: string,
    existing: string[],
): string {
    const prefix = operator === 'lte' ? 'max' : 'min';
    const base = `${prefix}_${metricId}`;
    if (!existing.includes(base)) return base;
    for (let i = 2; i < 1000; i += 1) {
        const candidate = `${base}_${i}`;
        if (!existing.includes(candidate)) return candidate;
    }
    return `${base}_${Date.now()}`;
}


/** Sensible default threshold: middle of the metric's expected_range.
 *  Mirrors what a user would type by hand for a freshly-added gate.
 *  Falls back to 0.5 when the range is missing or malformed. */
function defaultThresholdFor(metric: GateMetricOption | undefined): number {
    const range = metric?.expected_range;
    if (Array.isArray(range) && range.length === 2) {
        const [lo, hi] = range;
        if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
            return Math.round((lo + hi) / 2 * 100) / 100;
        }
    }
    return 0.5;
}


export default function EvalPackScaffoldPanel({ projectId, onSaved }: Props) {
    const [response, setResponse] = useState<ScaffoldResponse | null>(null);
    const [draft, setDraft] = useState<ScaffoldDraftPack | null>(null);
    const [gateOptions, setGateOptions] = useState<GateOptionsResponse | null>(null);
    const [perClassOptions, setPerClassOptions] = useState<PerClassMetricOptionsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);
    /** Backend 400 code from the last save attempt, when it referenced
     *  a specific gate_id (e.g. ``invalid_gate_operator:eq``,
     *  ``threshold_out_of_range:min_f1``). Used to highlight the bad
     *  row inline so the user doesn't have to scan the toast. */
    const [inlineError, setInlineError] = useState<string | null>(null);

    const load = useCallback(async () => {
        setError(null);
        setLoading(true);
        try {
            const resp = await fetchPackScaffold(projectId);
            setResponse(resp);
            setDraft(cloneDraft(resp.draft_pack));
            // Gate options fetched separately + non-blocking: the
            // editor stays usable (with raw text inputs in the worst
            // case) even if the catalog endpoint is unreachable.
            try {
                const opts = await fetchGateOptions(projectId);
                // Backwards-compat for test mocks that resolve all GETs
                // with the same payload — only adopt the response when
                // it actually carries a metrics list.
                if (Array.isArray(opts?.metrics)) {
                    setGateOptions(opts);
                }
            } catch {
                // Swallow — the editor falls back to a free-text input
                // for metric/operator when the catalog is missing.
            }
            // Gap-#6 slice 2 — per-class metric IDs are discovered
            // from the project's latest classification eval result. The
            // payload is independent of the static catalog: empty when
            // the project hasn't run an eval yet, populated otherwise.
            // Fetch is non-blocking just like the catalog one above.
            try {
                const perClass = await fetchPerClassMetricOptions(projectId);
                if (Array.isArray(perClass?.classes)) {
                    setPerClassOptions(perClass);
                }
            } catch {
                // Swallow — the per-class optgroup is suppressed when
                // discovery fails (or hasn't been wired yet).
            }
        } catch (err: any) {
            const detail = err?.response?.data?.detail;
            // 400 with recipe_required is expected on bare projects —
            // surface it as a quiet inline state rather than a red
            // error block.
            if (detail === 'recipe_required') {
                setError('Pick a recipe before scaffolding an eval pack.');
                setDraft(null);
                setResponse(null);
            } else {
                setError(detail || err?.message || 'Failed to load scaffold');
            }
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void load();
    }, [load]);

    /** Dirty when the current draft differs from the baseline returned
     *  by the GET. Cheap JSON-equality is fine here — the draft is
     *  small and only re-checked on state changes. */
    const isDirty = useMemo(() => {
        if (!draft || !response) return false;
        return JSON.stringify(draft) !== JSON.stringify(response.draft_pack);
    }, [draft, response]);

    const handleGateChange = useCallback(
        (taskIdx: number, gateIdx: number, patch: Partial<ScaffoldGate>) => {
            // Editing a row clears the inline error — the user is
            // acting on the feedback, so the row no longer carries
            // the stale "bad" highlight.
            setInlineError(null);
            setDraft((prev) => {
                if (!prev) return prev;
                const next = cloneDraft(prev);
                const gate = next.task_specs[taskIdx].gates[gateIdx];
                Object.assign(gate, patch);
                // Keep top-level gates in sync with the default task spec.
                if (taskIdx === 0) {
                    next.gates = next.task_specs[0].gates.map(cloneGate);
                }
                return next;
            });
        },
        [],
    );

    const handleAddGate = useCallback(
        (taskIdx: number) => {
            setInlineError(null);
            setDraft((prev) => {
                if (!prev) return prev;
                const next = cloneDraft(prev);
                const spec = next.task_specs[taskIdx];
                const existingIds = spec.gates.map((g) => g.gate_id);

                // Prefer a recommended metric that isn't already used
                // in this task_spec; fall back to any metric; fall back
                // again to a free-text placeholder when no catalog.
                const usedMetricIds = new Set(spec.gates.map((g) => g.metric_id));
                const recommended = (gateOptions?.metrics || [])
                    .filter((m) => m.recommended && !usedMetricIds.has(m.metric_id));
                const fallback = (gateOptions?.metrics || [])
                    .filter((m) => !usedMetricIds.has(m.metric_id));
                const pick = recommended[0] || fallback[0];

                const metricId = pick?.metric_id || 'new_metric';
                const operator = pick?.default_operator || 'gte';
                const newGate: ScaffoldGate = {
                    gate_id: makeUniqueGateId(metricId, operator, existingIds),
                    metric_id: metricId,
                    operator,
                    threshold: defaultThresholdFor(pick),
                    required: false,
                };
                spec.gates.push(newGate);
                if (taskIdx === 0) {
                    next.gates = next.task_specs[0].gates.map(cloneGate);
                }
                return next;
            });
        },
        [gateOptions],
    );

    const handleRemoveGate = useCallback(
        (taskIdx: number, gateIdx: number) => {
            setInlineError(null);
            setDraft((prev) => {
                if (!prev) return prev;
                const next = cloneDraft(prev);
                next.task_specs[taskIdx].gates.splice(gateIdx, 1);
                if (taskIdx === 0) {
                    next.gates = next.task_specs[0].gates.map(cloneGate);
                }
                return next;
            });
        },
        [],
    );

    const handleSave = useCallback(async () => {
        if (!draft) return;
        setSaving(true);
        setInlineError(null);
        try {
            const result = await savePackScaffold(projectId, draft);
            toast.success(`Scaffold saved — active pack is now ${result.preferred_pack_id}.`);
            // Adopt the saved pack as the new baseline so isDirty
            // returns to false until the user edits again.
            setResponse((prev) => prev ? { ...prev, draft_pack: cloneDraft(result.scaffolded_pack) } : prev);
            setDraft(cloneDraft(result.scaffolded_pack));
            onSaved?.(result.preferred_pack_id);
        } catch (err: any) {
            const detail = err?.response?.data?.detail
                || err?.message
                || 'Failed to save scaffold';
            // Slice-1 validator error codes look like
            // ``threshold_out_of_range:<gate_id>`` — surface the gate
            // id inline so the user can find the bad row.
            if (typeof detail === 'string' && detail.includes(':')) {
                setInlineError(detail);
            }
            toast.error(detail);
        } finally {
            setSaving(false);
        }
    }, [draft, onSaved, projectId]);

    const handleDiscard = useCallback(() => {
        setInlineError(null);
        if (response) {
            setDraft(cloneDraft(response.draft_pack));
        }
    }, [response]);

    /** Pulled out so we can render the dropdown the same way for new
     *  and existing rows. Sorts recommended metrics to the top + adds
     *  a star marker so the user's eye lands there first. */
    const sortedMetricOptions = useMemo(() => {
        if (!gateOptions?.metrics) return [];
        const copy = [...gateOptions.metrics];
        copy.sort((a, b) => {
            if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
            return a.metric_id.localeCompare(b.metric_id);
        });
        return copy;
    }, [gateOptions]);

    /** Gap-#6 slice 2 — group per-class options by class name so the
     *  dropdown renders an <optgroup> per class. HTML <optgroup>s
     *  can't nest, so we use one group per class with the 3 metrics
     *  (precision / recall / f1) inside. Stable order: classes are
     *  already sorted by the backend; we preserve the per-class
     *  metric order returned (precision, recall, f1). */
    const perClassMetricsByClass = useMemo(() => {
        const grouped = new Map<string, PerClassMetricOption[]>();
        for (const metric of perClassOptions?.metrics || []) {
            const className = metric.class_name;
            if (!grouped.has(className)) grouped.set(className, []);
            grouped.get(className)!.push(metric);
        }
        return grouped;
    }, [perClassOptions]);

    const hasPerClassMetrics = perClassMetricsByClass.size > 0;

    const erroredGateId = useMemo(() => {
        if (!inlineError || !inlineError.includes(':')) return null;
        const tail = inlineError.split(':')[1] || '';
        return tail.trim() || null;
    }, [inlineError]);

    if (loading) {
        return (
            <section className="card eval-pack-scaffold" data-testid="eval-pack-scaffold-loading">
                <p>Loading scaffold…</p>
            </section>
        );
    }

    if (error) {
        return (
            <section className="card eval-pack-scaffold eval-pack-scaffold--note" data-testid="eval-pack-scaffold-empty">
                <p>{error}</p>
            </section>
        );
    }

    if (!draft || !response) return null;

    return (
        <section className="card eval-pack-scaffold" data-testid="eval-pack-scaffold">
            <header className="eval-pack-scaffold__head">
                <div>
                    <h3>Scaffolded eval pack</h3>
                    <p className="eval-pack-scaffold__subtitle">
                        Auto-generated from the <code>{response.recipe_id}</code> recipe
                        {response.gold_set_summary.row_count > 0
                            ? <> · gold set has <strong>{response.gold_set_summary.row_count}</strong> rows</>
                            : null}
                        . Edit any threshold below, then click <strong>Use scaffold</strong> to save it
                        as this project's gate pack.
                    </p>
                </div>
            </header>

            {inlineError && (
                <p
                    className="eval-pack-scaffold__error"
                    data-testid="eval-pack-scaffold-inline-error"
                >
                    ⚠️ Save rejected: <code>{inlineError}</code>
                </p>
            )}

            {draft.task_specs.map((spec, taskIdx) => (
                <div
                    key={spec.task_profile}
                    className="eval-pack-scaffold__task"
                    data-testid={`eval-pack-scaffold-task-${spec.task_profile}`}
                >
                    <header>
                        <strong>{spec.display_name}</strong>
                        <span className="eval-pack-scaffold__pill">
                            task_profile: {spec.task_profile}
                        </span>
                    </header>
                    {/* Gap-#6 slice 2 — empty-state hint when the project's
                        recipe expects per-class metrics (classification-shaped
                        task profile) but no classes have been discovered yet.
                        Once an eval runs, the per-class optgroups populate
                        the metric dropdown above. */}
                    {!hasPerClassMetrics
                        && /^(classification|structured_extraction)$/.test(spec.task_profile) && (
                        <p
                            className="eval-pack-scaffold__per-class-hint"
                            data-testid={`eval-pack-scaffold-task-${spec.task_profile}-per-class-hint`}
                        >
                            💡 Run a classification eval to discover per-class metrics
                            (precision/recall/f1 per class) and gate them individually.
                        </p>
                    )}
                    <table className="eval-pack-scaffold__gates">
                        <thead>
                            <tr>
                                <th>Gate</th>
                                <th>Metric</th>
                                <th>Operator</th>
                                <th>Threshold</th>
                                <th>Required</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {spec.gates.map((gate, gateIdx) => {
                                const rowErrored = erroredGateId === gate.gate_id;
                                return (
                                <tr
                                    key={`${gate.gate_id}-${gateIdx}`}
                                    data-testid={`eval-pack-scaffold-gate-${gate.gate_id}`}
                                    className={rowErrored ? 'eval-pack-scaffold__row--errored' : undefined}
                                >
                                    <td>
                                        <input
                                            type="text"
                                            value={gate.gate_id}
                                            onChange={(e) =>
                                                handleGateChange(taskIdx, gateIdx, {
                                                    gate_id: e.target.value,
                                                })
                                            }
                                            data-testid={`eval-pack-scaffold-gate-${gate.gate_id}-gate-id`}
                                            aria-label={`${gate.gate_id} gate id`}
                                            className="input eval-pack-scaffold__gate-id"
                                        />
                                    </td>
                                    <td>
                                        {sortedMetricOptions.length > 0 ? (
                                            <select
                                                value={gate.metric_id}
                                                onChange={(e) =>
                                                    handleGateChange(taskIdx, gateIdx, {
                                                        metric_id: e.target.value,
                                                    })
                                                }
                                                data-testid={`eval-pack-scaffold-gate-${gate.gate_id}-metric`}
                                                aria-label={`${gate.gate_id} metric`}
                                                className="input"
                                            >
                                                {/* Show the current metric_id even if it's not
                                                    in the catalog — keeps user-typed metrics
                                                    selectable rather than silently coercing.
                                                    Look in BOTH catalogs (standard + per-class)
                                                    before stamping "(custom)". */}
                                                {sortedMetricOptions.find((m) => m.metric_id === gate.metric_id)
                                                    || (perClassOptions?.metrics || []).find((m) => m.metric_id === gate.metric_id)
                                                    ? null
                                                    : <option value={gate.metric_id}>{gate.metric_id} (custom)</option>}
                                                <optgroup label="Standard metrics">
                                                    {sortedMetricOptions.map((m) => (
                                                        <option key={m.metric_id} value={m.metric_id}>
                                                            {m.recommended ? '★ ' : ''}{m.label} ({m.metric_id})
                                                        </option>
                                                    ))}
                                                </optgroup>
                                                {/* Gap-#6 slice 2 — one <optgroup> per discovered
                                                    class. HTML <optgroup>s can't nest, so this
                                                    is the cleanest way to surface "per-class
                                                    precision/recall/f1 for class X" without
                                                    drowning the user in a flat list of
                                                    precision_<class>/recall_<class>/f1_<class>
                                                    triples. */}
                                                {Array.from(perClassMetricsByClass.entries()).map(
                                                    ([className, metrics]) => (
                                                        <optgroup
                                                            key={`per-class-${className}`}
                                                            label={`Per-class · ${className}`}
                                                        >
                                                            {metrics.map((m) => (
                                                                <option key={m.metric_id} value={m.metric_id}>
                                                                    {m.label} ({m.metric_id})
                                                                </option>
                                                            ))}
                                                        </optgroup>
                                                    ),
                                                )}
                                            </select>
                                        ) : (
                                            <input
                                                type="text"
                                                value={gate.metric_id}
                                                onChange={(e) =>
                                                    handleGateChange(taskIdx, gateIdx, {
                                                        metric_id: e.target.value,
                                                    })
                                                }
                                                data-testid={`eval-pack-scaffold-gate-${gate.gate_id}-metric`}
                                                aria-label={`${gate.gate_id} metric`}
                                                className="input"
                                            />
                                        )}
                                    </td>
                                    <td>
                                        {gateOptions?.operators?.length ? (
                                            <select
                                                value={gate.operator}
                                                onChange={(e) =>
                                                    handleGateChange(taskIdx, gateIdx, {
                                                        operator: e.target.value,
                                                    })
                                                }
                                                data-testid={`eval-pack-scaffold-gate-${gate.gate_id}-operator`}
                                                aria-label={`${gate.gate_id} operator`}
                                                className="input"
                                            >
                                                {gateOptions.operators.map((op) => (
                                                    <option key={op.value} value={op.value}>
                                                        {op.label}
                                                    </option>
                                                ))}
                                            </select>
                                        ) : (
                                            <span>{gate.operator}</span>
                                        )}
                                    </td>
                                    <td>
                                        <input
                                            type="number"
                                            step="0.01"
                                            min={0}
                                            max={1}
                                            value={gate.threshold}
                                            onChange={(e) => {
                                                const raw = Number(e.target.value);
                                                if (Number.isFinite(raw)) {
                                                    handleGateChange(taskIdx, gateIdx, {
                                                        threshold: Math.max(0, Math.min(1, raw)),
                                                    });
                                                }
                                            }}
                                            data-testid={`eval-pack-scaffold-gate-${gate.gate_id}-threshold`}
                                            className="input eval-pack-scaffold__threshold"
                                        />
                                    </td>
                                    <td>
                                        <input
                                            type="checkbox"
                                            checked={gate.required}
                                            onChange={(e) =>
                                                handleGateChange(taskIdx, gateIdx, {
                                                    required: e.target.checked,
                                                })
                                            }
                                            data-testid={`eval-pack-scaffold-gate-${gate.gate_id}-required`}
                                            aria-label={`${gate.gate_id} required`}
                                        />
                                    </td>
                                    <td>
                                        <button
                                            type="button"
                                            className="btn btn-tertiary eval-pack-scaffold__remove"
                                            onClick={() => handleRemoveGate(taskIdx, gateIdx)}
                                            disabled={saving}
                                            data-testid={`eval-pack-scaffold-gate-${gate.gate_id}-remove`}
                                            aria-label={`Remove ${gate.gate_id}`}
                                        >
                                            Remove
                                        </button>
                                    </td>
                                </tr>
                                );
                            })}
                        </tbody>
                    </table>
                    <div className="eval-pack-scaffold__add-row">
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => handleAddGate(taskIdx)}
                            disabled={saving}
                            data-testid={`eval-pack-scaffold-task-${spec.task_profile}-add-gate`}
                        >
                            + Add gate
                        </button>
                    </div>
                </div>
            ))}

            <footer className="eval-pack-scaffold__actions">
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleDiscard}
                    disabled={saving || !isDirty}
                    data-testid="eval-pack-scaffold-discard"
                >
                    Discard edits
                </button>
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleSave}
                    disabled={saving || !isDirty}
                    data-testid="eval-pack-scaffold-save"
                >
                    {saving ? 'Saving…' : 'Use scaffold'}
                </button>
            </footer>
        </section>
    );
}
