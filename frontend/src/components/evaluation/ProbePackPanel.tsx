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

import { fetchProbePack } from '../../api/probePack';
import type { Probe, ProbePack, ProbeResult } from '../../api/probePack';
import './ProbePackPanel.css';

interface ProbePackPanelProps {
    projectId: number;
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
};

function kindLabel(kind: string): string {
    return KIND_LABEL[kind] ?? kind;
}

export default function ProbePackPanel({ projectId }: ProbePackPanelProps) {
    const [pack, setPack] = useState<ProbePack | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [expanded, setExpanded] = useState<Set<string>>(new Set());

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
                            Probe pass-rate:{' '}
                            <strong>{Math.round((run.probe_pass_rate ?? 0) * 100)}%</strong>{' '}
                            ({run.passed}/{run.total})
                        </span>
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
