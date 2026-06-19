/**
 * ProbePackPanel — Coach-stage-2 phase 8.
 *
 * Surfaces the platform-authored, recipe-keyed adversarial probe pack:
 * the held-out ruler the user did NOT author. The point is honesty —
 * the user's gold set grades the model against examples the user wrote
 * (which a newbie's can be easy/biased). This pack grades *properties*
 * that must hold for any model on the task shape (robustness, refusal,
 * no-fabrication, degenerate-input), independent of the domain labels.
 *
 * This slice is read-only: the pack is assembled + inspectable
 * (`status: "ready_not_run"`). Running it against the trained model and
 * folding an independent pass-rate into the gate is the next slice — so
 * the panel is explicit that the grade isn't computed yet rather than
 * implying a score (feedback_honest_metrics_no_vanity).
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import { fetchProbePack, setProbeGate, setProbeKindWeights } from '../../api/probePack';
import type {
    DivergencePoint,
    Probe,
    ProbePack,
    ProbeResult,
} from '../../api/probePack';
import './ProbePackPanel.css';

interface ProbePackPanelProps {
    projectId: number;
    /** Phase 17 — open a run's scorecard when a sparkline point is clicked. */
    onOpenRun?: (experimentId: number) => void;
}

const KIND_LABEL: Record<string, string> = {
    robustness: 'Robustness',
    safety_refusal: 'Safety / refusal',
    format_robustness: 'Grounding / format',
    degenerate_input: 'Degenerate input',
};

const PROPERTY_LABEL: Record<string, string> = {
    prediction_stable_vs_base: 'Output must stay stable vs the clean version',
    refuses_or_declines: 'Must refuse or decline',
    no_fabrication_when_unsupported: 'Must not fabricate when unsupported',
    handles_degenerate_gracefully: 'Must handle gracefully (no crash / runaway)',
    does_not_over_refuse: 'Must answer a benign request (no over-refusal)',
};

function kindLabel(kind: string): string {
    return KIND_LABEL[kind] ?? kind;
}

export default function ProbePackPanel({ projectId, onOpenRun }: ProbePackPanelProps) {
    const [pack, setPack] = useState<ProbePack | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [expanded, setExpanded] = useState<Set<string>>(new Set());
    // Phase 17 — sparkline point the user is hovering (for the readout).
    const [hoveredRun, setHoveredRun] = useState<number | null>(null);
    // Phase 13 — optional probe-gate form state, seeded from the pack.
    const [gateEnabled, setGateEnabled] = useState(false);
    const [gatePct, setGatePct] = useState(70);
    const [gateSaving, setGateSaving] = useState(false);
    // Phase 22 — per-kind weight editor state, seeded from the pack.
    const [weights, setWeights] = useState<Record<string, number>>({});
    const [weightsSaving, setWeightsSaving] = useState(false);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            setPack(await fetchProbePack(projectId));
        } catch {
            setError('Could not load the probe pack.');
            setPack(null);
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void load();
    }, [load]);

    // Seed the gate form whenever a fresh pack arrives.
    useEffect(() => {
        const gc = pack?.gate_config;
        if (gc) {
            setGateEnabled(!!gc.enabled);
            setGatePct(Math.round((gc.min_pass_rate ?? 0.7) * 100));
        }
    }, [pack]);

    // Seed the weights editor when a fresh pack arrives.
    useEffect(() => {
        if (pack?.kind_weights) setWeights({ ...pack.kind_weights });
    }, [pack]);

    const saveWeights = useCallback(async () => {
        setWeightsSaving(true);
        try {
            await setProbeKindWeights(projectId, weights);
            await load();
        } catch {
            setError('Could not save the weights.');
        } finally {
            setWeightsSaving(false);
        }
    }, [projectId, weights, load]);

    const saveGate = useCallback(async () => {
        setGateSaving(true);
        try {
            await setProbeGate(projectId, {
                enabled: gateEnabled,
                min_pass_rate: Math.max(0, Math.min(1, gatePct / 100)),
                required: true,
            });
            await load();
        } catch {
            setError('Could not save the gate config.');
        } finally {
            setGateSaving(false);
        }
    }, [projectId, gateEnabled, gatePct, load]);

    const toggle = (id: string) =>
        setExpanded((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });

    // Defensive: never throw on a malformed payload — degrade to empty.
    const probes: Probe[] = useMemo(
        () => (Array.isArray(pack?.probes) ? pack!.probes : []),
        [pack],
    );

    // Gold-vs-probe history for the trend sparkline (defensive).
    const history: DivergencePoint[] = useMemo(
        () => (Array.isArray(pack?.divergence_history) ? pack!.divergence_history : []),
        [pack],
    );

    // Per-probe run result keyed by id, when the pack has been graded.
    const resultById = useMemo(() => {
        const map = new Map<string, ProbeResult>();
        const results = pack?.run?.results;
        if (Array.isArray(results)) {
            for (const r of results) {
                if (r && typeof r.id === 'string') map.set(r.id, r);
            }
        }
        return map;
    }, [pack]);

    if (loading && !pack) {
        return (
            <section className="probe-pack probe-pack--loading" data-testid="probe-pack">
                Loading independent probe pack…
            </section>
        );
    }
    if (error) {
        return (
            <section className="probe-pack probe-pack--error" data-testid="probe-pack">
                {error}{' '}
                <button type="button" className="btn btn-link" onClick={() => void load()}>
                    Retry
                </button>
            </section>
        );
    }
    if (!pack) return null;

    if (!pack.applicable) {
        return (
            <section id="probe-pack-panel" className="probe-pack probe-pack--inapplicable" data-testid="probe-pack" data-applicable="false">
                <header className="probe-pack__head">
                    <h3 className="probe-pack__title">Independent probe pack</h3>
                </header>
                <p className="probe-pack__note">{pack.note}</p>
            </section>
        );
    }

    const run = pack.run;
    const graded = !!run;

    return (
        <section
            id="probe-pack-panel"
            className="probe-pack"
            data-testid="probe-pack"
            data-applicable="true"
            data-graded={graded ? 'true' : 'false'}
        >
            <header className="probe-pack__head">
                <div className="probe-pack__head-line">
                    <span
                        className={`probe-pack__badge probe-pack__badge--${graded ? 'graded' : 'pending'}`}
                        data-testid="probe-pack-status"
                    >
                        {graded ? 'Graded · independent pass-rate' : 'Assembled · not yet graded'}
                    </span>
                    <h3 className="probe-pack__title">
                        Independent probe pack ({pack.probe_count})
                    </h3>
                </div>
                <p className="probe-pack__note">{pack.note}</p>
                {graded && run ? (
                    <div className="probe-pack__score" data-testid="probe-pack-score">
                        <span className="probe-pack__score-headline">
                            Weighted probe pass-rate:{' '}
                            <strong>{Math.round((run.probe_pass_rate ?? 0) * 100)}%</strong>{' '}
                            ({run.passed}/{run.total}
                            {typeof run.unweighted_pass_rate === 'number' &&
                                ` · raw ${Math.round(run.unweighted_pass_rate * 100)}%`}
                            )
                        </span>
                        {run.weighted_by_kind &&
                            Object.keys(run.weighted_by_kind).length > 0 && (
                                <ul
                                    className="probe-pack__weights"
                                    data-testid="probe-pack-weights"
                                >
                                    {Object.entries(run.weighted_by_kind).map(([kind, s]) => (
                                        <li key={kind} className="probe-pack__weight-chip">
                                            {kindLabel(kind)} ×{s.weight}:{' '}
                                            <strong>
                                                {s.passed}/{s.total}
                                            </strong>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        <ul className="probe-pack__props" data-testid="probe-pack-props">
                            {Object.entries(run.per_property || {}).map(([prop, score]) => (
                                <li key={prop} className="probe-pack__prop-chip">
                                    {PROPERTY_LABEL[prop] ?? prop}:{' '}
                                    <strong>
                                        {score.passed}/{score.total}
                                    </strong>
                                </li>
                            ))}
                        </ul>
                        <p className="probe-pack__score-note">
                            Graded against probes you didn't author — independent of
                            your gold-set pass-rate.
                        </p>
                        {(typeof run.judge_calls === 'number' ||
                            typeof run.judge_cached === 'number') && (
                            <p
                                className="probe-pack__judge-cost"
                                data-testid="probe-pack-judge-cost"
                            >
                                LLM judge: {run.judge_calls ?? 0} call
                                {(run.judge_calls ?? 0) === 1 ? '' : 's'}
                                {(run.judge_cached ?? 0) > 0 &&
                                    ` · ${run.judge_cached} reused from cache`}
                            </p>
                        )}
                    </div>
                ) : (
                    <ul className="probe-pack__kinds" data-testid="probe-pack-kinds">
                        {Object.entries(pack.kind_summary || {}).map(([kind, n]) => (
                            <li key={kind} className="probe-pack__kind-chip">
                                {kindLabel(kind)}: <strong>{n}</strong>
                            </li>
                        ))}
                    </ul>
                )}
            </header>

            {history.length >= 2 && (() => {
                const W = 220;
                const H = 44;
                const PAD = 6;
                const n = history.length;
                const x = (i: number) => PAD + (i / (n - 1)) * (W - 2 * PAD);
                const y = (r: number) =>
                    PAD + (1 - Math.max(0, Math.min(1, r))) * (H - 2 * PAD);
                const goldPts = history
                    .map((p, i) => `${x(i).toFixed(1)},${y(p.gold_pass_rate).toFixed(1)}`)
                    .join(' ');
                const probePts = history
                    .map((p, i) => `${x(i).toFixed(1)},${y(p.probe_pass_rate).toFixed(1)}`)
                    .join(' ');
                const latest = history[history.length - 1];
                return (
                    <div className="probe-pack__trend" data-testid="probe-pack-trend">
                        <div className="probe-pack__trend-head">
                            Two rulers over {n} run{n === 1 ? '' : 's'}
                            {latest && latest.divergence >= 0.15 && (
                                <span className="probe-pack__trend-gap">
                                    {' '}· latest gap {Math.round(latest.divergence * 100)}pts
                                </span>
                            )}
                            {history.some(
                                (p, i) =>
                                    i > 0 &&
                                    !!p.weight_regime &&
                                    p.weight_regime !== history[i - 1].weight_regime,
                            ) && (
                                <span
                                    className="probe-pack__trend-regime-note"
                                    data-testid="probe-pack-regime-note"
                                >
                                    {' '}· ▏ marks a score-weight change (trend not
                                    comparable across it)
                                </span>
                            )}
                        </div>
                        <svg
                            className="probe-pack__sparkline"
                            viewBox={`0 0 ${W} ${H}`}
                            width={W}
                            height={H}
                            role="img"
                            aria-label="Gold-set vs independent probe pass-rate over recent runs"
                        >
                            <polyline className="probe-pack__spark-gold" points={goldPts} fill="none" />
                            <polyline className="probe-pack__spark-probe" points={probePts} fill="none" />
                            {history.map((p, i) => {
                                if (i === 0) return null;
                                const prev = history[i - 1];
                                if (!p.weight_regime || p.weight_regime === prev.weight_regime) {
                                    return null;
                                }
                                return (
                                    <line
                                        key={`regime-${i}`}
                                        className="probe-pack__spark-regime"
                                        x1={x(i)}
                                        y1={PAD - 2}
                                        x2={x(i)}
                                        y2={H - PAD + 2}
                                        data-testid={`probe-spark-regime-${i}`}
                                    >
                                        <title>Score weights changed before this run</title>
                                    </line>
                                );
                            })}
                            {history.map((p, i) => {
                                const clickable = typeof p.experiment_id === 'number';
                                return (
                                    <circle
                                        key={p.eval_result_id ?? i}
                                        className={`probe-pack__spark-point${clickable ? ' probe-pack__spark-point--clickable' : ''}`}
                                        cx={x(i)}
                                        cy={y(p.probe_pass_rate)}
                                        r={hoveredRun === i ? 3.5 : 2.5}
                                        data-testid={`probe-spark-point-${i}`}
                                        onMouseEnter={() => setHoveredRun(i)}
                                        onMouseLeave={() => setHoveredRun(null)}
                                        onClick={
                                            clickable
                                                ? () => onOpenRun?.(p.experiment_id as number)
                                                : undefined
                                        }
                                    >
                                        <title>
                                            {`gold ${Math.round(p.gold_pass_rate * 100)}% · probe ${Math.round(p.probe_pass_rate * 100)}%`}
                                            {p.run_at ? ` · ${new Date(p.run_at).toLocaleDateString()}` : ''}
                                        </title>
                                    </circle>
                                );
                            })}
                        </svg>
                        <div className="probe-pack__trend-legend">
                            <span className="probe-pack__legend-gold">gold set</span>
                            <span className="probe-pack__legend-probe">independent probes</span>
                        </div>
                        {(() => {
                            const p = hoveredRun != null ? history[hoveredRun] : latest;
                            if (!p) return null;
                            return (
                                <div className="probe-pack__trend-readout" data-testid="probe-pack-trend-readout">
                                    {p.run_at ? `${new Date(p.run_at).toLocaleDateString()}: ` : ''}
                                    gold <strong>{Math.round(p.gold_pass_rate * 100)}%</strong>
                                    {' · '}probe <strong>{Math.round(p.probe_pass_rate * 100)}%</strong>
                                    {' · '}gap {Math.round(p.divergence * 100)}pts
                                    {typeof p.experiment_id === 'number' && onOpenRun && (
                                        <span className="probe-pack__trend-hint"> · click a point to open its scorecard</span>
                                    )}
                                </div>
                            );
                        })()}
                    </div>
                );
            })()}

            {pack.judge_spend && (
                <p className="probe-pack__judge-spend" data-testid="probe-pack-judge-spend">
                    Judge spend (recent): ~{pack.judge_spend.total_calls} call
                    {pack.judge_spend.total_calls === 1 ? '' : 's'} · ~
                    {pack.judge_spend.est_tokens >= 1000
                        ? `${Math.round(pack.judge_spend.est_tokens / 1000)}k`
                        : pack.judge_spend.est_tokens}{' '}
                    tokens across {pack.judge_spend.runs_with_judge} run
                    {pack.judge_spend.runs_with_judge === 1 ? '' : 's'}
                    {pack.judge_spend.total_cached > 0 &&
                        ` · ${pack.judge_spend.total_cached} reused from cache`}
                </p>
            )}

            <div className="probe-pack__gate" data-testid="probe-pack-gate">
                <label className="probe-pack__gate-toggle">
                    <input
                        type="checkbox"
                        checked={gateEnabled}
                        onChange={(e) => setGateEnabled(e.target.checked)}
                        data-testid="probe-gate-enabled"
                    />
                    Enforce as an eval gate
                </label>
                {gateEnabled && (
                    <span className="probe-pack__gate-threshold">
                        require probe pass-rate ≥
                        <input
                            type="number"
                            min={0}
                            max={100}
                            value={gatePct}
                            onChange={(e) => setGatePct(Number(e.target.value))}
                            data-testid="probe-gate-threshold"
                            className="probe-pack__gate-input"
                            aria-label="Minimum probe pass-rate percent"
                        />
                        %
                    </span>
                )}
                <button
                    type="button"
                    className="btn btn-sm probe-pack__gate-save"
                    onClick={() => void saveGate()}
                    disabled={gateSaving}
                    data-testid="probe-gate-save"
                >
                    {gateSaving ? 'Saving…' : 'Save gate'}
                </button>
                <p className="probe-pack__gate-note">
                    Off by default. When on, a low independent probe score
                    <strong> blocks</strong> the eval gate — not just nudges.
                </p>
            </div>

            {Object.keys(weights).length > 0 && (
                <div className="probe-pack__weights-editor" data-testid="probe-pack-weights-editor">
                    <span className="probe-pack__weights-editor-label">
                        Score weights by kind
                    </span>
                    {(['safety_refusal', 'format_robustness', 'degenerate_input', 'robustness'] as const)
                        .filter((kind) => kind in weights)
                        .map((kind) => (
                            <label key={kind} className="probe-pack__weight-field">
                                {kindLabel(kind)}
                                <input
                                    type="number"
                                    min={0}
                                    max={10}
                                    step={0.5}
                                    value={weights[kind]}
                                    onChange={(e) =>
                                        setWeights((w) => ({
                                            ...w,
                                            [kind]: Number(e.target.value),
                                        }))
                                    }
                                    data-testid={`probe-weight-${kind}`}
                                    className="probe-pack__weight-input"
                                    aria-label={`Weight for ${kindLabel(kind)}`}
                                />
                            </label>
                        ))}
                    <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => void saveWeights()}
                        disabled={weightsSaving}
                        data-testid="probe-weights-save"
                    >
                        {weightsSaving ? 'Saving…' : 'Save weights'}
                    </button>
                </div>
            )}

            <ul className="probe-pack__list">
                {probes.map((p) => {
                    const open = expanded.has(p.id);
                    const result = resultById.get(p.id);
                    return (
                        <li
                            key={p.id}
                            className="probe-pack__probe"
                            data-testid={`probe-${p.id}`}
                            data-passed={result ? String(result.passed) : undefined}
                        >
                            <button
                                type="button"
                                className="probe-pack__probe-head"
                                onClick={() => toggle(p.id)}
                                aria-label={open ? `Collapse ${p.id}` : `Expand ${p.id}`}
                            >
                                {result && (
                                    <span
                                        className={`probe-pack__verdict probe-pack__verdict--${result.passed ? 'pass' : 'fail'}`}
                                        data-testid={`probe-verdict-${p.id}`}
                                    >
                                        {result.passed ? '✓' : '✕'}
                                    </span>
                                )}
                                <span className={`probe-pack__kind probe-pack__kind--${p.probe_kind}`}>
                                    {kindLabel(p.probe_kind)}
                                </span>
                                <span className="probe-pack__probe-prop">
                                    {PROPERTY_LABEL[p.property] ?? p.property}
                                </span>
                                <span className="probe-pack__chevron">{open ? '−' : '+'}</span>
                            </button>
                            {open && (
                                <div className="probe-pack__probe-body" data-testid={`probe-body-${p.id}`}>
                                    {p.base_input && (
                                        <p className="probe-pack__io">
                                            <span className="probe-pack__io-label">Clean</span>
                                            <code>{p.base_input}</code>
                                        </p>
                                    )}
                                    <p className="probe-pack__io">
                                        <span className="probe-pack__io-label">
                                            {p.base_input ? 'Perturbed' : 'Input'}
                                        </span>
                                        <code>{p.input === '' ? '(empty string)' : p.input}</code>
                                    </p>
                                    {result && (
                                        <>
                                            {result.base_output != null && (
                                                <p className="probe-pack__io">
                                                    <span className="probe-pack__io-label">Out (clean)</span>
                                                    <code>{result.base_output || '(empty)'}</code>
                                                </p>
                                            )}
                                            <p className="probe-pack__io">
                                                <span className="probe-pack__io-label">Model out</span>
                                                <code>{result.output || '(empty)'}</code>
                                            </p>
                                            <p className="probe-pack__io">
                                                <span className="probe-pack__io-label">Verdict</span>
                                                <code>
                                                    {result.passed ? 'PASS' : 'FAIL'} — {result.reason}
                                                </code>
                                            </p>
                                        </>
                                    )}
                                    <p className="probe-pack__rationale">{p.rationale}</p>
                                </div>
                            )}
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}
