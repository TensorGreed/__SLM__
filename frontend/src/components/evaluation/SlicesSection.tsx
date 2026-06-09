/**
 * Quality-Lift phase 7 slice 1 — Slices editor section.
 *
 * Concrete consumer of PackSectionEditor. Loads slice definitions from
 * the phase 2 slice 1 CRUD endpoint, lets the user edit slice_id /
 * display_name / where-clauses with closed-grammar dropdowns, and
 * persists via PUT. "Gate this slice" opens a small modal that
 * auto-composes the ``per_slice.<id>.<metric>`` metric_id the user
 * pastes into the gates section.
 *
 * Design notes:
 *   * The where-clause sub-editor is NOT generic — slice clauses have
 *     a specific shape (field / op / value) that doesn't compose with
 *     other sections (e.g. behavioral perturbations). Custom per-row
 *     here is cheaper than a generic nested-array primitive.
 *   * The "value" cell adapts to the op: ``exists`` hides the input,
 *     ``in`` / ``not_in`` take comma-separated values, numeric ops
 *     accept numbers only. The backend validator catches malformed
 *     types either way; this is just UX.
 *   * "Gate this slice" emits the canonical
 *     ``per_slice.<slice_id>.<metric>`` shape per the phase 2 slice 3
 *     gate evaluator's resolver. The user picks the metric (default
 *     ``f1``) and copies the result.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Target } from 'lucide-react';

import PackSectionEditor from './PackSectionEditor';
import {
    fetchSliceDefinitions,
    PLATFORM_FIELDS,
    saveSliceDefinitions,
    SLICE_OPERATORS,
} from '../../api/sliceDefinitions';
// Vite + esbuild serves the dev module with all named imports as
// runtime references — TypeScript-only types (``SliceDefinition`` /
// ``SliceClause`` are erased at compile time) must be brought in via
// ``import type`` or the browser blows up with "'SliceDefinition' is
// not exported". This bit dev-server loading after slice 1 — fix
// applied so the page renders.
import type { SliceDefinition, SliceClause } from '../../api/sliceDefinitions';
import './SlicesSection.css';

// Slice ID grammar mirrors the backend regex in
// ``slice_definitions_service`` (^[a-z][a-z0-9_]{0,63}$). UI-level
// catch — the validator rejects too, but a synchronous check keeps
// the Save button correctly disabled while typing.
const SLICE_ID_RE = /^[a-z][a-z0-9_]{0,63}$/;

// Numeric ops accept a number for the ``value`` cell. List ops take
// a comma-separated string that we split on save. ``exists`` has no
// value (presence check). Everything else accepts any scalar.
const NUMERIC_OPS = new Set(['gt', 'gte', 'lt', 'lte']);
const LIST_OPS = new Set(['in', 'not_in']);
const NO_VALUE_OPS = new Set(['exists']);

interface SlicesSectionProps {
    projectId: number;
    // Optional callback fires when the user clicks "Gate this slice".
    // Lets the parent (eventually) auto-insert a gate row in the gates
    // section instead of just copying the metric_id. Slice 1 keeps the
    // modal copy affordance as the default; phase 7 slice 2+ can wire
    // this for a one-click gate-add.
    onGateSlice?: (sliceId: string, metricId: string) => void;
}

interface GateModalState {
    sliceId: string;
    metric: string;
    copied: boolean;
}

function makeNewSlice(): SliceDefinition {
    return {
        slice_id: '',
        display_name: '',
        where: [{ field: 'input_length', op: 'gte', value: 100 }],
    };
}

function isSliceValid(slice: SliceDefinition): boolean {
    if (!SLICE_ID_RE.test(slice.slice_id)) return false;
    if (slice.where.length === 0) return false;
    for (const c of slice.where) {
        if (!c.field || !c.field.trim()) return false;
        if (!SLICE_OPERATORS.includes(c.op)) return false;
        if (NO_VALUE_OPS.has(c.op)) continue;
        if (NUMERIC_OPS.has(c.op)) {
            if (typeof c.value !== 'number' || Number.isNaN(c.value)) return false;
        } else if (LIST_OPS.has(c.op)) {
            if (!Array.isArray(c.value) || c.value.length === 0) return false;
        } else if (c.op === 'regex' || c.op === 'contains') {
            if (typeof c.value !== 'string' || !c.value) return false;
        }
    }
    return true;
}

function ClauseRow({
    clause,
    onChange,
    onRemove,
    canRemove,
    testIdPrefix,
}: {
    clause: SliceClause;
    onChange: (next: SliceClause) => void;
    onRemove: () => void;
    canRemove: boolean;
    testIdPrefix?: string;
}) {
    const setOp = (op: typeof SLICE_OPERATORS[number]) => {
        // Coerce the value to a sensible default for the new op so
        // the input doesn't render a number where a list belongs.
        let nextValue: unknown = clause.value;
        if (NO_VALUE_OPS.has(op)) {
            nextValue = true;
        } else if (NUMERIC_OPS.has(op)) {
            nextValue = typeof clause.value === 'number' ? clause.value : 0;
        } else if (LIST_OPS.has(op)) {
            nextValue = Array.isArray(clause.value) ? clause.value : [];
        } else if (op === 'regex' || op === 'contains') {
            nextValue = typeof clause.value === 'string' ? clause.value : '';
        } else {
            // eq / neq — keep whatever the user had if scalar.
            if (typeof clause.value === 'object') {
                nextValue = '';
            }
        }
        onChange({ ...clause, op, value: nextValue });
    };

    return (
        <div className="slices-section__clause" data-testid={testIdPrefix}>
            <input
                type="text"
                className="input slices-section__clause-field"
                list="slices-section__platform-fields"
                value={clause.field}
                onChange={(e) => onChange({ ...clause, field: e.target.value })}
                placeholder="field"
                aria-label="Field path"
                data-testid={testIdPrefix ? `${testIdPrefix}-field` : undefined}
            />
            <select
                className="input slices-section__clause-op"
                value={clause.op}
                onChange={(e) => setOp(e.target.value as typeof SLICE_OPERATORS[number])}
                aria-label="Operator"
                data-testid={testIdPrefix ? `${testIdPrefix}-op` : undefined}
            >
                {SLICE_OPERATORS.map((op) => (
                    <option key={op} value={op}>{op}</option>
                ))}
            </select>
            {NO_VALUE_OPS.has(clause.op) ? (
                <span className="slices-section__clause-value-stub">(presence check)</span>
            ) : NUMERIC_OPS.has(clause.op) ? (
                <input
                    type="number"
                    className="input slices-section__clause-value"
                    value={typeof clause.value === 'number' ? clause.value : 0}
                    onChange={(e) =>
                        onChange({ ...clause, value: Number(e.target.value) })
                    }
                    aria-label="Value"
                    data-testid={testIdPrefix ? `${testIdPrefix}-value` : undefined}
                />
            ) : LIST_OPS.has(clause.op) ? (
                <input
                    type="text"
                    className="input slices-section__clause-value"
                    value={Array.isArray(clause.value) ? clause.value.join(', ') : ''}
                    onChange={(e) =>
                        onChange({
                            ...clause,
                            value: e.target.value
                                .split(',')
                                .map((s) => s.trim())
                                .filter(Boolean),
                        })
                    }
                    placeholder="comma, separated, values"
                    aria-label="Values"
                    data-testid={testIdPrefix ? `${testIdPrefix}-value` : undefined}
                />
            ) : (
                <input
                    type="text"
                    className="input slices-section__clause-value"
                    value={typeof clause.value === 'string'
                        ? clause.value
                        : clause.value == null ? '' : String(clause.value)}
                    onChange={(e) => onChange({ ...clause, value: e.target.value })}
                    placeholder="value"
                    aria-label="Value"
                    data-testid={testIdPrefix ? `${testIdPrefix}-value` : undefined}
                />
            )}
            <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={onRemove}
                disabled={!canRemove}
                aria-label="Remove clause"
                data-testid={testIdPrefix ? `${testIdPrefix}-remove` : undefined}
            >
                ×
            </button>
        </div>
    );
}

function GateThisSliceModal({
    state,
    onClose,
    onCopied,
}: {
    state: GateModalState;
    onClose: () => void;
    onCopied: () => void;
}) {
    const metricId = `per_slice.${state.sliceId}.${state.metric}`;
    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(metricId);
            onCopied();
        } catch {
            // Some sandboxed contexts (file://, certain test envs)
            // block the clipboard API. Fall through silently — the
            // user can still select + Ctrl-C the visible text.
        }
    };
    return (
        <div className="slices-section__modal-backdrop" onClick={onClose}>
            <div
                className="slices-section__modal"
                onClick={(e) => e.stopPropagation()}
                role="dialog"
                aria-label="Gate this slice"
            >
                <h4>Gate this slice</h4>
                <p>
                    Add this metric_id to a gate in the Gates section to enforce
                    a per-slice threshold. The format mirrors phase 2 slice 3's
                    canonical gate path.
                </p>
                <div className="slices-section__modal-metric-row">
                    <code className="slices-section__modal-metric">{metricId}</code>
                    <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        onClick={() => void handleCopy()}
                    >
                        {state.copied ? 'Copied!' : 'Copy'}
                    </button>
                </div>
                <p className="slices-section__modal-help">
                    Use ``worst_slice_gte`` / ``worst_slice_lte`` as the operator
                    if you want a single gate to enforce the threshold across
                    every slice automatically (no need to add one per slice).
                </p>
                <div className="slices-section__modal-actions">
                    <button type="button" className="btn btn-ghost" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}

export default function SlicesSection({ projectId, onGateSlice }: SlicesSectionProps) {
    const [items, setItems] = useState<SliceDefinition[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [gateModal, setGateModal] = useState<GateModalState | null>(null);

    const reload = useCallback(async () => {
        setLoading(true);
        setLoadError(null);
        try {
            const resp = await fetchSliceDefinitions(projectId);
            setItems(resp.slice_definitions?.slices ?? []);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Failed to load slices.';
            setLoadError(message);
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void reload();
    }, [reload]);

    const handleSave = useCallback(
        async (next: SliceDefinition[]) => {
            // Normalize list-op values: the comma-split happens on
            // typing, so the values are already arrays. Numeric ops
            // already store numbers. Just send through.
            await saveSliceDefinitions(projectId, { slices: next });
        },
        [projectId],
    );

    const openGateModal = useCallback((sliceId: string) => {
        if (onGateSlice) {
            onGateSlice(sliceId, `per_slice.${sliceId}.f1`);
            return;
        }
        setGateModal({ sliceId, metric: 'f1', copied: false });
    }, [onGateSlice]);

    const renderItem = useMemo(() => {
        return (slice: SliceDefinition, index: number, mutate: (next: SliceDefinition) => void) => {
            const updateClause = (clauseIdx: number, next: SliceClause) => {
                const nextWhere = slice.where.map((c, i) => (i === clauseIdx ? next : c));
                mutate({ ...slice, where: nextWhere });
            };
            const removeClause = (clauseIdx: number) => {
                mutate({ ...slice, where: slice.where.filter((_, i) => i !== clauseIdx) });
            };
            const addClause = () => {
                mutate({
                    ...slice,
                    where: [...slice.where, { field: 'input_length', op: 'gte', value: 0 }],
                });
            };

            return (
                <div className="slices-section__form">
                    <div className="slices-section__form-row">
                        <label className="slices-section__label">
                            slice_id
                            <input
                                type="text"
                                className="input"
                                value={slice.slice_id}
                                onChange={(e) => mutate({ ...slice, slice_id: e.target.value.trim() })}
                                placeholder="e.g. long_input"
                                data-testid={`slices-item-${index}-id`}
                            />
                            {slice.slice_id && !SLICE_ID_RE.test(slice.slice_id) && (
                                <span className="slices-section__field-error">
                                    must match ^[a-z][a-z0-9_]{'{0,63}'}$
                                </span>
                            )}
                        </label>
                        <label className="slices-section__label">
                            display_name
                            <input
                                type="text"
                                className="input"
                                value={slice.display_name}
                                onChange={(e) => mutate({ ...slice, display_name: e.target.value })}
                                placeholder="Long inputs (>100 chars)"
                                data-testid={`slices-item-${index}-display-name`}
                            />
                        </label>
                    </div>
                    <div className="slices-section__where">
                        <div className="slices-section__where-header">
                            <strong>where (all clauses AND)</strong>
                            <button
                                type="button"
                                className="btn btn-ghost btn-sm"
                                onClick={addClause}
                                data-testid={`slices-item-${index}-clause-add`}
                            >
                                + clause
                            </button>
                        </div>
                        {slice.where.map((clause, ci) => (
                            <ClauseRow
                                key={`${index}-${ci}`}
                                clause={clause}
                                onChange={(next) => updateClause(ci, next)}
                                onRemove={() => removeClause(ci)}
                                canRemove={slice.where.length > 1}
                                testIdPrefix={`slices-item-${index}-clause-${ci}`}
                            />
                        ))}
                    </div>
                </div>
            );
        };
    }, []);

    if (loading) {
        return (
            <div className="slices-section slices-section--loading">
                Loading slices…
            </div>
        );
    }

    if (loadError) {
        return (
            <div className="slices-section slices-section--error">
                <span>{loadError}</span>
                <button type="button" className="btn btn-secondary" onClick={() => void reload()}>
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="slices-section">
            <datalist id="slices-section__platform-fields">
                {PLATFORM_FIELDS.map(({ name, description }) => (
                    <option key={name} value={name}>{description}</option>
                ))}
            </datalist>
            <PackSectionEditor<SliceDefinition>
                title="Slices"
                description="Named subsets of eval rows. Every eval emits per-slice metrics for any slice defined here; gates can target a specific slice via per_slice.<id>.<metric> or use worst_slice_gte / worst_slice_lte."
                initialItems={items}
                // Stable index-based key so the input doesn't unmount
                // every time the user edits ``slice_id`` (which would
                // happen if we keyed on ``slice.slice_id``). Reorder
                // isn't a feature in slice 1; if it lands later we can
                // stamp a per-item ``__local_id`` instead.
                itemKey={(_slice, index) => String(index)}
                newItem={makeNewSlice}
                renderItem={renderItem}
                isItemValid={isSliceValid}
                onSave={handleSave}
                addLabel="Add slice"
                itemLabel="slice"
                renderItemHeaderTrailing={(slice) =>
                    SLICE_ID_RE.test(slice.slice_id) ? (
                        <button
                            type="button"
                            className="btn btn-ghost btn-sm slices-section__gate-button"
                            onClick={() => openGateModal(slice.slice_id)}
                            title="Show a copy-pasteable metric_id for gating this slice"
                            data-testid={`slices-gate-${slice.slice_id}`}
                        >
                            <Target size={12} aria-hidden="true" /> Gate this slice
                        </button>
                    ) : null
                }
                testIdPrefix="slices"
            />

            {gateModal && (
                <GateThisSliceModal
                    state={gateModal}
                    onClose={() => setGateModal(null)}
                    onCopied={() => setGateModal({ ...gateModal, copied: true })}
                />
            )}
        </div>
    );
}
