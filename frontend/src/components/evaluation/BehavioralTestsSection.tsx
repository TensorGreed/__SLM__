/**
 * Quality-Lift phase 7 slice 2 — Behavioral tests editor section.
 *
 * Second concrete consumer of PackSectionEditor. Loads behavioral
 * tests from the focused endpoint (``/api/projects/{id}/behavioral-
 * tests``), provides a kind-aware form (the test kind dropdown
 * INV/DIR/MFT drives which fields render), and persists via PUT.
 * "Gate this test" composes the top-level ``behavioral.<test_id>.
 * pass_rate`` metric_id; when the project has slice_definitions
 * configured, the modal also offers per-slice variants
 * (``behavioral.<test_id>.per_slice.<slice_id>.pass_rate``) so the
 * user can wire phase 6 slice 2 / 3 gates from the same surface.
 *
 * Closed grammar dropdowns mirror the backend validator tuples — see
 * behavioralTests.ts for the source of each constant.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Target } from 'lucide-react';

import PackSectionEditor from './PackSectionEditor';
import {
    BEHAVIORAL_TEST_KINDS,
    CASE_CHANGE_OPTIONS,
    DIR_EXPECTATION_KINDS,
    PERTURBATION_KINDS,
    fetchBehavioralTests,
    saveBehavioralTests,
} from '../../api/behavioralTests';
import type {
    BehavioralTest,
    BehavioralTestKind,
    Expectation,
    MftExample,
    Perturbation,
    PerturbationKind,
    SeedExample,
} from '../../api/behavioralTests';
import { fetchSliceDefinitions } from '../../api/sliceDefinitions';
import type { SliceDefinition } from '../../api/sliceDefinitions';
import './BehavioralTestsSection.css';


// test_id grammar mirrors the backend regex (see
// behavioral_test_schema._TEST_ID_RE). UI catch — the validator runs
// on save too, but a synchronous check keeps the Save button
// correctly disabled during typing.
const TEST_ID_RE = /^[a-z][a-z0-9_]{0,63}$/;


function defaultPerturbation(kind: PerturbationKind): Perturbation {
    switch (kind) {
        case 'typo':
            return { kind, intensity: 0.05 };
        case 'insert_token':
            return { kind, params: { token: 'not ', position: 0 } };
        case 'case_change':
            return { kind, params: { case: 'lower' } };
        case 'whitespace_jitter':
            return { kind, intensity: 0.2 };
    }
}


function makeNewTest(): BehavioralTest {
    return {
        test_id: '',
        kind: 'INV',
        description: '',
        seed_examples: [{ input: '', given_label: '' }],
        perturbations: [defaultPerturbation('typo')],
        expectation: { kind: 'same_label' },
        n_perturbations_per_seed: 1,
    };
}


function isTestValid(test: BehavioralTest): boolean {
    if (!TEST_ID_RE.test(test.test_id)) return false;
    if (!BEHAVIORAL_TEST_KINDS.includes(test.kind)) return false;
    if (test.kind === 'MFT') {
        const examples = test.examples ?? [];
        if (examples.length === 0) return false;
        return examples.every((ex) => ex.input.trim().length > 0 && ex.expected_label.trim().length > 0);
    }
    const seeds = test.seed_examples ?? [];
    const perts = test.perturbations ?? [];
    if (seeds.length === 0 || perts.length === 0) return false;
    if (!seeds.every((s) => s.input.trim().length > 0)) return false;
    // DIR-specific expectation requirement.
    if (test.kind === 'DIR') {
        const exp = test.expectation;
        if (!exp || !DIR_EXPECTATION_KINDS.includes(exp.kind as typeof DIR_EXPECTATION_KINDS[number])) return false;
        if (exp.kind === 'must_change_to' && (!exp.target_label || !exp.target_label.trim())) return false;
        if (exp.kind === 'must_change_to_one_of' && (!exp.target_labels || exp.target_labels.length === 0)) return false;
    }
    return true;
}


interface BehavioralTestsSectionProps {
    projectId: number;
    /** Optional callback fires when the user clicks "Gate this test".
     *  Lets a future slice wire a one-click gate-add into the gates
     *  section directly. Slice 2 default: open the copy-paste modal. */
    onGateTest?: (testId: string, metricId: string) => void;
}


function PerturbationRow({
    pert,
    onChange,
    onRemove,
    canRemove,
    testIdPrefix,
}: {
    pert: Perturbation;
    onChange: (next: Perturbation) => void;
    onRemove: () => void;
    canRemove: boolean;
    testIdPrefix: string;
}) {
    const setKind = (kind: PerturbationKind) => {
        onChange(defaultPerturbation(kind));
    };
    return (
        <div className="bt-section__pert" data-testid={testIdPrefix}>
            <select
                className="input bt-section__pert-kind"
                value={pert.kind}
                onChange={(e) => setKind(e.target.value as PerturbationKind)}
                aria-label="Perturbation kind"
                data-testid={`${testIdPrefix}-kind`}
            >
                {PERTURBATION_KINDS.map((k) => (
                    <option key={k} value={k}>{k}</option>
                ))}
            </select>
            {pert.kind === 'typo' && (
                <label className="bt-section__pert-param">
                    intensity
                    <input
                        type="number"
                        step={0.05}
                        min={0.01}
                        max={0.5}
                        className="input"
                        value={pert.intensity ?? 0.05}
                        onChange={(e) => onChange({ ...pert, intensity: Number(e.target.value) })}
                        aria-label="Typo intensity"
                        data-testid={`${testIdPrefix}-intensity`}
                    />
                </label>
            )}
            {pert.kind === 'insert_token' && (
                <>
                    <label className="bt-section__pert-param">
                        token
                        <input
                            type="text"
                            className="input"
                            value={(pert.params?.token as string) ?? ''}
                            onChange={(e) =>
                                onChange({
                                    ...pert,
                                    params: { ...(pert.params ?? {}), token: e.target.value },
                                })
                            }
                            aria-label="Insert token text"
                            data-testid={`${testIdPrefix}-token`}
                        />
                    </label>
                    <label className="bt-section__pert-param">
                        position
                        <input
                            type="number"
                            className="input"
                            value={(pert.params?.position as number) ?? 0}
                            onChange={(e) =>
                                onChange({
                                    ...pert,
                                    params: { ...(pert.params ?? {}), position: Number(e.target.value) },
                                })
                            }
                            aria-label="Insert token position"
                            data-testid={`${testIdPrefix}-position`}
                        />
                    </label>
                </>
            )}
            {pert.kind === 'case_change' && (
                <label className="bt-section__pert-param">
                    case
                    <select
                        className="input"
                        value={(pert.params?.case as string) ?? 'lower'}
                        onChange={(e) =>
                            onChange({
                                ...pert,
                                params: { ...(pert.params ?? {}), case: e.target.value },
                            })
                        }
                        aria-label="Case change variant"
                        data-testid={`${testIdPrefix}-case`}
                    >
                        {CASE_CHANGE_OPTIONS.map((c) => (
                            <option key={c} value={c}>{c}</option>
                        ))}
                    </select>
                </label>
            )}
            {pert.kind === 'whitespace_jitter' && (
                <label className="bt-section__pert-param">
                    intensity
                    <input
                        type="number"
                        step={0.05}
                        min={0.01}
                        max={1.0}
                        className="input"
                        value={pert.intensity ?? 0.2}
                        onChange={(e) => onChange({ ...pert, intensity: Number(e.target.value) })}
                        aria-label="Whitespace jitter intensity"
                        data-testid={`${testIdPrefix}-intensity`}
                    />
                </label>
            )}
            <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={onRemove}
                disabled={!canRemove}
                aria-label="Remove perturbation"
                data-testid={`${testIdPrefix}-remove`}
            >
                ×
            </button>
        </div>
    );
}


function GateThisTestModal({
    testId,
    sliceIds,
    onClose,
}: {
    testId: string;
    sliceIds: string[];
    onClose: () => void;
}) {
    const topLevelMetric = `behavioral.${testId}.pass_rate`;
    const [copied, setCopied] = useState<string | null>(null);
    const copy = async (value: string) => {
        try {
            await navigator.clipboard.writeText(value);
            setCopied(value);
        } catch {
            // Sandboxed contexts may block clipboard — fall through
            // silently; user can select-and-copy the visible text.
        }
    };
    return (
        <div className="bt-section__modal-backdrop" onClick={onClose}>
            <div
                className="bt-section__modal"
                onClick={(e) => e.stopPropagation()}
                role="dialog"
                aria-label="Gate this test"
            >
                <h4>Gate this test</h4>
                <p>
                    Add one of these metric_ids to a gate in the Gates section.
                    Phase 5 slice 2's flattener emits them so the existing gate
                    evaluator picks them up with no new code.
                </p>
                <div className="bt-section__modal-metric-row">
                    <code className="bt-section__modal-metric">{topLevelMetric}</code>
                    <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        onClick={() => void copy(topLevelMetric)}
                    >
                        {copied === topLevelMetric ? 'Copied!' : 'Copy'}
                    </button>
                </div>
                {sliceIds.length > 0 && (
                    <>
                        <p className="bt-section__modal-section-title">
                            Per-slice variants (Quality-Lift phase 6)
                        </p>
                        <p className="bt-section__modal-help">
                            Gate the same test on a specific slice. A single
                            INV test can pass overall but fail on a slice —
                            per-slice gates catch that regression.
                        </p>
                        <div className="bt-section__modal-slice-list">
                            {sliceIds.map((sid) => {
                                const metric = `behavioral.${testId}.per_slice.${sid}.pass_rate`;
                                return (
                                    <div key={sid} className="bt-section__modal-metric-row">
                                        <code className="bt-section__modal-metric">{metric}</code>
                                        <button
                                            type="button"
                                            className="btn btn-ghost btn-sm"
                                            onClick={() => void copy(metric)}
                                        >
                                            {copied === metric ? 'Copied!' : 'Copy'}
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    </>
                )}
                <div className="bt-section__modal-actions">
                    <button type="button" className="btn btn-ghost" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}


export default function BehavioralTestsSection({
    projectId,
    onGateTest,
}: BehavioralTestsSectionProps) {
    const [items, setItems] = useState<BehavioralTest[]>([]);
    const [slices, setSlices] = useState<SliceDefinition[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [gateModal, setGateModal] = useState<{ testId: string } | null>(null);

    const reload = useCallback(async () => {
        setLoading(true);
        setLoadError(null);
        try {
            // Parallel — slice list is independent of behavioral tests
            // and the "Gate this test" modal needs both.
            const [btResp, sliceResp] = await Promise.all([
                fetchBehavioralTests(projectId),
                fetchSliceDefinitions(projectId).catch(() => null),
            ]);
            setItems(btResp.behavioral_tests ?? []);
            setSlices(sliceResp?.slice_definitions?.slices ?? []);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Failed to load behavioral tests.';
            setLoadError(message);
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void reload();
    }, [reload]);

    const handleSave = useCallback(
        async (next: BehavioralTest[]) => {
            // Strip n_perturbations_per_seed when it's 1 to keep
            // payloads minimal; the backend default is 1 too. Drop
            // empty given_label values so the validator's None
            // semantics apply.
            const cleaned = next.map((t) => {
                const out: BehavioralTest = { ...t };
                if (out.kind === 'MFT') {
                    out.seed_examples = undefined;
                    out.perturbations = undefined;
                    out.expectation = undefined;
                    out.n_perturbations_per_seed = undefined;
                } else {
                    out.examples = undefined;
                    if (out.seed_examples) {
                        out.seed_examples = out.seed_examples.map((s) => ({
                            input: s.input,
                            ...(s.given_label && s.given_label.trim()
                                ? { given_label: s.given_label.trim() }
                                : {}),
                        }));
                    }
                    if (out.n_perturbations_per_seed === 1) {
                        out.n_perturbations_per_seed = undefined;
                    }
                    if (out.kind === 'INV') {
                        out.expectation = { kind: 'same_label' };
                    }
                }
                return out;
            });
            await saveBehavioralTests(projectId, cleaned);
        },
        [projectId],
    );

    const openGateModal = useCallback(
        (testId: string) => {
            if (onGateTest) {
                onGateTest(testId, `behavioral.${testId}.pass_rate`);
                return;
            }
            setGateModal({ testId });
        },
        [onGateTest],
    );

    const renderItem = useMemo(() => {
        return (test: BehavioralTest, index: number, mutate: (next: BehavioralTest) => void) => {
            const setKind = (kind: BehavioralTestKind) => {
                if (kind === test.kind) return;
                // Reset kind-specific fields to defaults so the form
                // doesn't carry over invalid state from the prior kind
                // (e.g. an INV's seed_examples shouldn't linger on an
                // MFT switch).
                const base = {
                    test_id: test.test_id,
                    description: test.description,
                    kind,
                };
                if (kind === 'MFT') {
                    mutate({
                        ...base,
                        examples: [{ input: '', expected_label: '' }],
                    });
                } else if (kind === 'DIR') {
                    mutate({
                        ...base,
                        seed_examples: [{ input: '', given_label: '' }],
                        perturbations: [defaultPerturbation('insert_token')],
                        expectation: { kind: 'must_change' },
                        n_perturbations_per_seed: 1,
                    });
                } else {
                    mutate({
                        ...base,
                        seed_examples: [{ input: '', given_label: '' }],
                        perturbations: [defaultPerturbation('typo')],
                        expectation: { kind: 'same_label' },
                        n_perturbations_per_seed: 1,
                    });
                }
            };

            // Helper mutators for nested arrays — keep these inline
            // so the closure captures the right mutate target.
            const updateSeed = (i: number, next: SeedExample) => {
                const seeds = test.seed_examples ?? [];
                mutate({ ...test, seed_examples: seeds.map((s, j) => (j === i ? next : s)) });
            };
            const removeSeed = (i: number) => {
                const seeds = test.seed_examples ?? [];
                mutate({ ...test, seed_examples: seeds.filter((_, j) => j !== i) });
            };
            const addSeed = () => {
                const seeds = test.seed_examples ?? [];
                mutate({ ...test, seed_examples: [...seeds, { input: '', given_label: '' }] });
            };
            const updatePert = (i: number, next: Perturbation) => {
                const perts = test.perturbations ?? [];
                mutate({ ...test, perturbations: perts.map((p, j) => (j === i ? next : p)) });
            };
            const removePert = (i: number) => {
                const perts = test.perturbations ?? [];
                mutate({ ...test, perturbations: perts.filter((_, j) => j !== i) });
            };
            const addPert = () => {
                const perts = test.perturbations ?? [];
                mutate({ ...test, perturbations: [...perts, defaultPerturbation('typo')] });
            };
            const updateMftExample = (i: number, next: MftExample) => {
                const ex = test.examples ?? [];
                mutate({ ...test, examples: ex.map((e, j) => (j === i ? next : e)) });
            };
            const removeMftExample = (i: number) => {
                const ex = test.examples ?? [];
                mutate({ ...test, examples: ex.filter((_, j) => j !== i) });
            };
            const addMftExample = () => {
                const ex = test.examples ?? [];
                mutate({ ...test, examples: [...ex, { input: '', expected_label: '' }] });
            };
            const updateExpectation = (next: Partial<Expectation>) => {
                mutate({
                    ...test,
                    expectation: { ...(test.expectation ?? { kind: 'same_label' }), ...next } as Expectation,
                });
            };

            return (
                <div className="bt-section__form">
                    <div className="bt-section__form-row">
                        <label className="bt-section__label">
                            test_id
                            <input
                                type="text"
                                className="input"
                                value={test.test_id}
                                onChange={(e) => mutate({ ...test, test_id: e.target.value.trim() })}
                                placeholder="e.g. typo_invariance"
                                data-testid={`bt-item-${index}-id`}
                            />
                            {test.test_id && !TEST_ID_RE.test(test.test_id) && (
                                <span className="bt-section__field-error">
                                    must match ^[a-z][a-z0-9_]{'{0,63}'}$
                                </span>
                            )}
                        </label>
                        <label className="bt-section__label">
                            kind
                            <select
                                className="input"
                                value={test.kind}
                                onChange={(e) => setKind(e.target.value as BehavioralTestKind)}
                                data-testid={`bt-item-${index}-kind`}
                            >
                                {BEHAVIORAL_TEST_KINDS.map((k) => (
                                    <option key={k} value={k}>{k}</option>
                                ))}
                            </select>
                        </label>
                    </div>
                    <label className="bt-section__label">
                        description
                        <input
                            type="text"
                            className="input"
                            value={test.description ?? ''}
                            onChange={(e) => mutate({ ...test, description: e.target.value })}
                            placeholder="What does this test catch?"
                            data-testid={`bt-item-${index}-description`}
                        />
                    </label>

                    {test.kind === 'MFT' ? (
                        <div className="bt-section__subgroup">
                            <div className="bt-section__subgroup-header">
                                <strong>examples</strong>
                                <button
                                    type="button"
                                    className="btn btn-ghost btn-sm"
                                    onClick={addMftExample}
                                    data-testid={`bt-item-${index}-add-example`}
                                >
                                    + example
                                </button>
                            </div>
                            {(test.examples ?? []).map((ex, i) => (
                                <div key={i} className="bt-section__mft-row" data-testid={`bt-item-${index}-example-${i}`}>
                                    <input
                                        type="text"
                                        className="input bt-section__example-input"
                                        value={ex.input}
                                        onChange={(e) => updateMftExample(i, { ...ex, input: e.target.value })}
                                        placeholder="input"
                                        aria-label="Example input"
                                    />
                                    <input
                                        type="text"
                                        className="input bt-section__example-label"
                                        value={ex.expected_label}
                                        onChange={(e) => updateMftExample(i, { ...ex, expected_label: e.target.value })}
                                        placeholder="expected_label"
                                        aria-label="Expected label"
                                    />
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-sm"
                                        onClick={() => removeMftExample(i)}
                                        disabled={(test.examples?.length ?? 0) <= 1}
                                        aria-label="Remove example"
                                    >
                                        ×
                                    </button>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <>
                            <div className="bt-section__subgroup">
                                <div className="bt-section__subgroup-header">
                                    <strong>seed_examples</strong>
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-sm"
                                        onClick={addSeed}
                                        data-testid={`bt-item-${index}-add-seed`}
                                    >
                                        + seed
                                    </button>
                                </div>
                                {(test.seed_examples ?? []).map((seed, i) => (
                                    <div key={i} className="bt-section__seed-row" data-testid={`bt-item-${index}-seed-${i}`}>
                                        <input
                                            type="text"
                                            className="input bt-section__example-input"
                                            value={seed.input}
                                            onChange={(e) => updateSeed(i, { ...seed, input: e.target.value })}
                                            placeholder="input"
                                            aria-label="Seed input"
                                        />
                                        <input
                                            type="text"
                                            className="input bt-section__example-label"
                                            value={seed.given_label ?? ''}
                                            onChange={(e) => updateSeed(i, { ...seed, given_label: e.target.value })}
                                            placeholder="given_label (optional)"
                                            aria-label="Given label"
                                        />
                                        <button
                                            type="button"
                                            className="btn btn-ghost btn-sm"
                                            onClick={() => removeSeed(i)}
                                            disabled={(test.seed_examples?.length ?? 0) <= 1}
                                            aria-label="Remove seed"
                                        >
                                            ×
                                        </button>
                                    </div>
                                ))}
                            </div>

                            <div className="bt-section__subgroup">
                                <div className="bt-section__subgroup-header">
                                    <strong>perturbations</strong>
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-sm"
                                        onClick={addPert}
                                        data-testid={`bt-item-${index}-add-pert`}
                                    >
                                        + perturbation
                                    </button>
                                </div>
                                {(test.perturbations ?? []).map((pert, i) => (
                                    <PerturbationRow
                                        key={i}
                                        pert={pert}
                                        onChange={(next) => updatePert(i, next)}
                                        onRemove={() => removePert(i)}
                                        canRemove={(test.perturbations?.length ?? 0) > 1}
                                        testIdPrefix={`bt-item-${index}-pert-${i}`}
                                    />
                                ))}
                            </div>

                            {test.kind === 'DIR' && (
                                <div className="bt-section__subgroup">
                                    <strong>expectation</strong>
                                    <div className="bt-section__dir-expectation">
                                        <select
                                            className="input"
                                            value={test.expectation?.kind ?? 'must_change'}
                                            onChange={(e) =>
                                                updateExpectation({
                                                    kind: e.target.value as typeof DIR_EXPECTATION_KINDS[number],
                                                    target_label: undefined,
                                                    target_labels: undefined,
                                                })
                                            }
                                            aria-label="DIR expectation kind"
                                            data-testid={`bt-item-${index}-dir-kind`}
                                        >
                                            {DIR_EXPECTATION_KINDS.map((k) => (
                                                <option key={k} value={k}>{k}</option>
                                            ))}
                                        </select>
                                        {test.expectation?.kind === 'must_change_to' && (
                                            <input
                                                type="text"
                                                className="input"
                                                value={test.expectation.target_label ?? ''}
                                                onChange={(e) => updateExpectation({ target_label: e.target.value })}
                                                placeholder="target_label"
                                                aria-label="DIR target label"
                                                data-testid={`bt-item-${index}-dir-target`}
                                            />
                                        )}
                                        {test.expectation?.kind === 'must_change_to_one_of' && (
                                            <input
                                                type="text"
                                                className="input"
                                                value={(test.expectation.target_labels ?? []).join(', ')}
                                                onChange={(e) =>
                                                    updateExpectation({
                                                        target_labels: e.target.value
                                                            .split(',')
                                                            .map((s) => s.trim())
                                                            .filter(Boolean),
                                                    })
                                                }
                                                placeholder="comma, separated, labels"
                                                aria-label="DIR target labels"
                                                data-testid={`bt-item-${index}-dir-targets`}
                                            />
                                        )}
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            );
        };
    }, []);

    if (loading) {
        return (
            <div className="bt-section bt-section--loading">
                Loading behavioral tests…
            </div>
        );
    }

    if (loadError) {
        return (
            <div className="bt-section bt-section--error">
                <span>{loadError}</span>
                <button type="button" className="btn btn-secondary" onClick={() => void reload()}>
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="bt-section">
            <PackSectionEditor<BehavioralTest>
                title="Behavioral tests"
                description="CheckList-style robustness probes (INV / DIR / MFT). Every eval runs these against the trained checkpoint; gates can fail ship on a regression via behavioral.<test_id>.pass_rate (or per-slice via behavioral.<test_id>.per_slice.<slice_id>.pass_rate)."
                initialItems={items}
                itemKey={(_test, index) => String(index)}
                newItem={makeNewTest}
                renderItem={renderItem}
                isItemValid={isTestValid}
                onSave={handleSave}
                addLabel="Add behavioral test"
                itemLabel="test"
                renderItemHeaderTrailing={(test) =>
                    TEST_ID_RE.test(test.test_id) ? (
                        <button
                            type="button"
                            className="btn btn-ghost btn-sm bt-section__gate-button"
                            onClick={() => openGateModal(test.test_id)}
                            title="Show copy-pasteable metric_ids for gating this test"
                            data-testid={`bt-gate-${test.test_id}`}
                        >
                            <Target size={12} aria-hidden="true" /> Gate this test
                        </button>
                    ) : null
                }
                testIdPrefix="bt"
            />

            {gateModal && (
                <GateThisTestModal
                    testId={gateModal.testId}
                    sliceIds={slices.map((s) => s.slice_id).filter((sid) => /^[a-z][a-z0-9_]{0,63}$/.test(sid))}
                    onClose={() => setGateModal(null)}
                />
            )}
        </div>
    );
}
