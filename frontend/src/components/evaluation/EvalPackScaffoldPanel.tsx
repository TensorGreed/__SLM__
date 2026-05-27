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

import { useCallback, useEffect, useState } from 'react';

import type {
    ScaffoldDraftPack,
    ScaffoldGate,
    ScaffoldResponse,
} from '../../api/evalPackScaffold';
import { fetchPackScaffold, savePackScaffold } from '../../api/evalPackScaffold';
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


export default function EvalPackScaffoldPanel({ projectId, onSaved }: Props) {
    const [response, setResponse] = useState<ScaffoldResponse | null>(null);
    const [draft, setDraft] = useState<ScaffoldDraftPack | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);

    const load = useCallback(async () => {
        setError(null);
        setLoading(true);
        try {
            const resp = await fetchPackScaffold(projectId);
            setResponse(resp);
            setDraft(cloneDraft(resp.draft_pack));
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

    const handleGateChange = useCallback(
        (taskIdx: number, gateIdx: number, patch: Partial<ScaffoldGate>) => {
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

    const handleSave = useCallback(async () => {
        if (!draft) return;
        setSaving(true);
        try {
            const result = await savePackScaffold(projectId, draft);
            toast.success(`Scaffold saved — active pack is now ${result.preferred_pack_id}.`);
            onSaved?.(result.preferred_pack_id);
        } catch (err: any) {
            toast.error(
                err?.response?.data?.detail
                    || err?.message
                    || 'Failed to save scaffold',
            );
        } finally {
            setSaving(false);
        }
    }, [draft, onSaved, projectId]);

    const handleDiscard = useCallback(() => {
        if (response) {
            setDraft(cloneDraft(response.draft_pack));
        }
    }, [response]);

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
                    <table className="eval-pack-scaffold__gates">
                        <thead>
                            <tr>
                                <th>Gate</th>
                                <th>Metric</th>
                                <th>Operator</th>
                                <th>Threshold</th>
                                <th>Required</th>
                            </tr>
                        </thead>
                        <tbody>
                            {spec.gates.map((gate, gateIdx) => (
                                <tr
                                    key={gate.gate_id}
                                    data-testid={`eval-pack-scaffold-gate-${gate.gate_id}`}
                                >
                                    <td><code>{gate.gate_id}</code></td>
                                    <td>{gate.metric_id}</td>
                                    <td>{gate.operator}</td>
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
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ))}

            <footer className="eval-pack-scaffold__actions">
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleDiscard}
                    disabled={saving}
                    data-testid="eval-pack-scaffold-discard"
                >
                    Discard edits
                </button>
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleSave}
                    disabled={saving}
                    data-testid="eval-pack-scaffold-save"
                >
                    {saving ? 'Saving…' : 'Use scaffold'}
                </button>
            </footer>
        </section>
    );
}
