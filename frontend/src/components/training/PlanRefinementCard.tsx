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
import { AlertTriangle, CheckCircle2, ShieldCheck, XCircle } from 'lucide-react';

import { getPlanRefinement } from '../../api/planRefinement';
import type { PlanRefinement } from '../../api/planRefinement';
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

    useEffect(() => {
        let active = true;
        getPlanRefinement(projectId)
            .then((r) => { if (active) { setReport(r); setError(null); } })
            .catch((e: any) => { if (active) setError(e?.response?.data?.detail || e?.message || 'Failed to load plan refinement.'); });
        return () => { active = false; };
    }, [projectId]);

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

            <p className="plan-refine__privacy" data-testid="plan-refinement-privacy">
                <ShieldCheck size={13} aria-hidden="true" />
                {report.cloud_refinement.available
                    ? 'Cloud refinement available.'
                    : 'Deterministic only.'}{' '}
                {report.privacy.note}
            </p>
        </section>
    );
}
