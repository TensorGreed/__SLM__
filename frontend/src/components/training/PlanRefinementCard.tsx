/**
 * PlanRefinementCard — Phase 1 of the post-data plan-refinement pass.
 *
 * A *plan-fit* lens distinct from the data-health / config-gap panels: does the
 * project's current plan (recipe / task shape / base model / target) suit the
 * MEASURED data? Deterministic — no cloud call in Phase 1. Surfaces the
 * aggregate profile the eventual cloud strategy pass would reason over, plus an
 * explicit transparency line: only those aggregates are ever eligible to leave
 * BrewSLM — never the user's ingested rows.
 */

import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, ShieldCheck, Sparkles, XCircle } from 'lucide-react';

import { getPlanRefinement, runCloudPlanRefinement } from '../../api/planRefinement';
import type { PlanRefinement, StrategyRefinement } from '../../api/planRefinement';
import './PlanRefinementCard.css';

const VERDICT = {
    ready: { label: 'Plan fits the data', cls: 'ready', Icon: CheckCircle2 },
    attention: { label: 'Plan needs attention', cls: 'attention', Icon: AlertTriangle },
    mismatch: { label: 'Plan mismatch', cls: 'mismatch', Icon: XCircle },
} as const;

interface Props {
    projectId: number;
}

export default function PlanRefinementCard({ projectId }: Props) {
    const [report, setReport] = useState<PlanRefinement | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [refinement, setRefinement] = useState<StrategyRefinement | null>(null);
    const [busy, setBusy] = useState(false);
    const [cloudNote, setCloudNote] = useState<string | null>(null);

    useEffect(() => {
        let active = true;
        getPlanRefinement(projectId)
            .then((r) => { if (active) { setReport(r); setRefinement(r.refinement); setError(null); } })
            .catch((e: any) => { if (active) setError(e?.response?.data?.detail || e?.message || 'Failed to load plan refinement.'); });
        return () => { active = false; };
    }, [projectId]);

    const handleGetStrategy = async () => {
        setBusy(true);
        setCloudNote(null);
        try {
            const res = await runCloudPlanRefinement(projectId);
            if (res.available && res.refinement) {
                setRefinement(res.refinement);
            } else {
                setCloudNote('No cloud model is configured (or the call failed). Add an API key under Project Settings → Secrets to enable AI strategy.');
            }
        } catch (e: any) {
            setCloudNote(e?.response?.data?.detail || e?.message || 'Strategy pass failed.');
        } finally {
            setBusy(false);
        }
    };

    const openSyntheticForGap = (suggested?: number) => {
        window.location.assign(
            `/project/${projectId}/pipeline/synthetic`
            + `?prefill_mode=class_balance_fill&prefill_count=${suggested ?? 30}`,
        );
    };

    if (error) {
        return (
            <section className="plan-refine plan-refine--error" data-testid="plan-refinement-card">
                <h3>Plan health</h3>
                <p>{error}</p>
            </section>
        );
    }
    if (!report) {
        return (
            <section className="plan-refine plan-refine--loading" data-testid="plan-refinement-card">
                <span>Assessing how your plan fits the data…</span>
            </section>
        );
    }

    const verdictKey = (report.plan_health.verdict in VERDICT
        ? report.plan_health.verdict
        : 'attention') as keyof typeof VERDICT;
    const v = VERDICT[verdictKey];
    const shape = report.cloud_safe_profile.label_distribution_shape;

    return (
        <section className={`plan-refine plan-refine--${v.cls}`} data-testid="plan-refinement-card">
            <header className="plan-refine__head">
                <div>
                    <p className="plan-refine__eyebrow">Plan health</p>
                    <h3 className="plan-refine__verdict">
                        <v.Icon size={18} aria-hidden="true" />
                        {v.label}
                    </h3>
                    <p className="plan-refine__sub">
                        Does your plan ({report.plan.task_profile || 'no task shape'}
                        {report.plan.base_model_name ? ` · ${report.plan.base_model_name}` : ''})
                        {' '}suit the {report.cloud_safe_profile.labelled_row_count} labelled rows you have?
                    </p>
                </div>
            </header>

            {report.plan_health.signals.length > 0 && (
                <ul className="plan-refine__signals">
                    {report.plan_health.signals.map((s) => (
                        <li key={s.id} className={`plan-refine__signal plan-refine__signal--${s.severity}`}>
                            <span>{s.headline}</span>
                        </li>
                    ))}
                </ul>
            )}

            <div className="plan-refine__profile" data-testid="plan-refinement-profile">
                <span><strong>{report.cloud_safe_profile.labelled_row_count}</strong> labelled rows</span>
                {shape && (
                    <>
                        <span><strong>{shape.num_classes}</strong> classes</span>
                        {shape.classes_below_floor > 0 && (
                            <span className="plan-refine__profile--warn">
                                <strong>{shape.classes_below_floor}</strong> below floor
                            </span>
                        )}
                    </>
                )}
                {report.cloud_safe_profile.forecast_verdict && (
                    <span>forecast: <strong>{report.cloud_safe_profile.forecast_verdict.replace(/_/g, ' ')}</strong></span>
                )}
            </div>

            {refinement ? (
                <div className="plan-refine__strategy" data-testid="plan-refinement-strategy">
                    <div className="plan-refine__strategy-head">
                        <Sparkles size={14} aria-hidden="true" />
                        <strong>AI strategy</strong>
                        {refinement.provenance?.model && (
                            <small>via {refinement.provenance.model}</small>
                        )}
                        {refinement.confidence != null && (
                            <small>· confidence {Math.round(refinement.confidence * 100)}%</small>
                        )}
                    </div>
                    {refinement.rationale && <p className="plan-refine__rationale">{refinement.rationale}</p>}
                    {Object.keys(refinement.plan_delta).length > 0 && (
                        <ul className="plan-refine__delta">
                            {refinement.plan_delta.recipe_id && <li>Switch recipe → <code>{refinement.plan_delta.recipe_id}</code></li>}
                            {refinement.plan_delta.task_profile && <li>Task shape → <code>{refinement.plan_delta.task_profile}</code></li>}
                            {refinement.plan_delta.base_model_size_class && <li>Base size → <code>{refinement.plan_delta.base_model_size_class}</code></li>}
                            {refinement.plan_delta.rag_first && <li>Enable RAG-first retrieval</li>}
                            {refinement.plan_delta.training_mode && <li>Training mode → <code>{refinement.plan_delta.training_mode}</code></li>}
                        </ul>
                    )}
                    {refinement.directional_config.length > 0 && (
                        <ul className="plan-refine__delta">
                            {refinement.directional_config.map((d) => (
                                <li key={d.kind}>{d.kind.replace(/_/g, ' ')}: {d.reason}</li>
                            ))}
                        </ul>
                    )}
                    {refinement.data_gaps.map((g) => (
                        <div className="plan-refine__gap" key={g.kind}>
                            <span>{g.detail}</span>
                            {g.kind === 'class_balance' && (
                                <button type="button" className="btn btn-secondary btn-sm"
                                    onClick={() => openSyntheticForGap(g.suggested_count)}>
                                    Generate ~{g.suggested_count ?? 30}
                                </button>
                            )}
                        </div>
                    ))}
                    <p className="plan-refine__note">
                        Strategy only — BrewSLM computes the actual hyperparameters from your data.
                        {refinement.from_cache && ' (cached for the current data)'}
                    </p>
                </div>
            ) : report.cloud_refinement.available ? (
                <button
                    type="button"
                    className="btn btn-primary btn-sm plan-refine__cta"
                    onClick={() => void handleGetStrategy()}
                    disabled={busy}
                    data-testid="plan-refinement-get-strategy"
                >
                    <Sparkles size={14} aria-hidden="true" />
                    {busy ? 'Consulting model…' : 'Get AI strategy recommendation'}
                </button>
            ) : null}
            {cloudNote && <p className="plan-refine__cloud-note">{cloudNote}</p>}

            <p className="plan-refine__privacy" data-testid="plan-refinement-privacy">
                <ShieldCheck size={13} aria-hidden="true" />
                {report.cloud_refinement.available
                    ? 'Cloud strategy enabled.'
                    : 'Deterministic only.'}{' '}
                {report.privacy.note}
            </p>
        </section>
    );
}
