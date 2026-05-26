/**
 * RerouteRecommendationPanel — USER-SUCCESS Epic 7 Phase 7c.
 *
 * Mounted on the Eval tab above AutoRagComparisonPanel. Fetches the
 * Phase 7a `reroute-analysis` endpoint for the latest eval result
 * and renders a categorical recommendation:
 *
 *   * stay_the_course → panel self-hides (no noise on passing evals)
 *   * try_rag → amber-bordered card with the fired signals + a primary
 *     CTA "Switch to RAG (keeps your gold set)" that fires the
 *     Phase 7b clone endpoint after a confirmation modal.
 *   * try_prompt_engineering → blue card; CTA navigates to Playground.
 *   * expand_data → gray card; CTA navigates to Active Learning.
 *
 * 404 (no eval yet) silently renders nothing. 400 (mismatch) renders
 * nothing too — the panel is *advisory*, not load-bearing.
 */

import { useEffect, useState } from 'react';
import {
    fetchRerouteAnalysis,
    rerouteToRagAsync,
    type RerouteAnalysis,
    type RerouteSignal,
} from '../../api/rerouteAnalysis';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import './RerouteRecommendationPanel.css';


interface Props {
    projectId: number;
    evalResultId: number | null | undefined;
}


/**
 * Navigate via a hard page change rather than ``react-router``'s
 * ``useNavigate`` — keeps this panel mountable from contexts that
 * aren't wrapped in a Router (the existing EvalPanel test suites
 * render EvalPanel in isolation, and pulling ``useNavigate`` here
 * would force every one of them to wrap in MemoryRouter). The
 * post-clone flow already triggers a context switch to a different
 * project so a full reload is acceptable.
 */
function navigateTo(url: string): void {
    window.location.assign(url);
}


export default function RerouteRecommendationPanel({ projectId, evalResultId }: Props) {
    const [analysis, setAnalysis] = useState<RerouteAnalysis | null>(null);
    const [loading, setLoading] = useState(false);
    const [confirming, setConfirming] = useState(false);
    const [cloning, setCloning] = useState(false);

    useEffect(() => {
        if (!evalResultId) {
            setAnalysis(null);
            return;
        }
        let cancelled = false;
        setLoading(true);
        fetchRerouteAnalysis(projectId, evalResultId)
            .then((data) => {
                if (!cancelled) setAnalysis(data);
            })
            .catch(() => {
                // Advisory panel — failures (404/400/transient) silently
                // hide rather than render a noisy error state.
                if (!cancelled) setAnalysis(null);
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [projectId, evalResultId]);

    // Loading is intentionally invisible — analysis is a sub-second
    // operation and a flash of "loading…" above the Eval results
    // would be more noise than signal.
    if (loading || !analysis) {
        return null;
    }

    const kind = analysis.recommendation.kind;
    if (kind === 'stay_the_course') {
        // Passing eval → no reroute needed → no panel.
        return null;
    }

    const firedSignals = analysis.signals.filter((s) => s.fired);

    if (kind === 'try_rag') {
        return (
            <>
                <RerouteCard
                    flavor="try-rag"
                    title="This task looks more like a RAG fit"
                    subtitle={`Your eval is at ${formatPassRate(analysis.pass_rate)}. Switching approach would keep your gold set.`}
                    rationale={analysis.recommendation.rationale}
                    confidence={analysis.recommendation.confidence}
                    signals={firedSignals}
                    primaryLabel="Switch to RAG (keeps your gold set)"
                    onPrimaryClick={() => setConfirming(true)}
                    primaryDisabled={cloning}
                />
                {confirming && (
                    <SwitchToRagConfirmModal
                        projectId={projectId}
                        cloning={cloning}
                        onCancel={() => setConfirming(false)}
                        onConfirm={async () => {
                            // Hardening Phase H1 — fire the async-job
                            // variant. The clone (file copy + BM25
                            // index build) runs in the background; the
                            // user is freed up immediately. Notification
                            // bell surfaces progress + the "Open" link
                            // when the new project is ready.
                            setCloning(true);
                            try {
                                const job = await rerouteToRagAsync(projectId);
                                toast.info(
                                    `Cloning started — bell will notify when ready (job #${job.id})`,
                                    4000,
                                );
                                void useJobsStore
                                    .getState()
                                    .refreshAfterLocalChange();
                                setCloning(false);
                                setConfirming(false);
                            } catch (err) {
                                const detail =
                                    (err as { response?: { data?: { detail?: string } } })
                                        ?.response?.data?.detail || 'Reroute failed';
                                toast.error(`Could not switch to RAG: ${detail}`);
                                setCloning(false);
                                setConfirming(false);
                            }
                        }}
                    />
                )}
            </>
        );
    }

    if (kind === 'try_prompt_engineering') {
        return (
            <RerouteCard
                flavor="try-prompt"
                title="Iterate in the Playground first"
                subtitle={`Your eval is at ${formatPassRate(analysis.pass_rate)}. The output is a small slice of the input — careful prompting may suffice before another training run.`}
                rationale={analysis.recommendation.rationale}
                confidence={analysis.recommendation.confidence}
                signals={firedSignals}
                primaryLabel="Open Playground"
                onPrimaryClick={() => navigateTo(`/project/${projectId}/playground`)}
            />
        );
    }

    // expand_data (catch-all)
    return (
        <RerouteCard
            flavor="expand-data"
            title="Try more or higher-quality data"
            subtitle={`Your eval is at ${formatPassRate(analysis.pass_rate)}. No specific approach-mismatch signal fired — the model likely needs more training rows.`}
            rationale={analysis.recommendation.rationale}
            confidence={analysis.recommendation.confidence}
            signals={firedSignals}
            primaryLabel="Open Active Learning"
            onPrimaryClick={() => navigateTo(`/project/${projectId}/pipeline/data`)}
        />
    );
}


// ─────────────────────────────────────────────────────────────────────
// Reusable card subcomponent (different border/tint per flavor)
// ─────────────────────────────────────────────────────────────────────


interface RerouteCardProps {
    flavor: 'try-rag' | 'try-prompt' | 'expand-data';
    title: string;
    subtitle: string;
    rationale: string;
    confidence: number;
    signals: RerouteSignal[];
    primaryLabel: string;
    onPrimaryClick: () => void;
    primaryDisabled?: boolean;
}


function RerouteCard({
    flavor,
    title,
    subtitle,
    rationale,
    confidence,
    signals,
    primaryLabel,
    onPrimaryClick,
    primaryDisabled,
}: RerouteCardProps) {
    return (
        <section
            id="reroute-recommendation-panel"
            className={`reroute-card reroute-card--${flavor}`}
            data-testid={`reroute-card-${flavor}`}
        >
            <header className="reroute-card__head">
                <div>
                    <h3 className="reroute-card__title">{title}</h3>
                    <p className="reroute-card__subtitle">{subtitle}</p>
                </div>
                <span
                    className="reroute-card__confidence"
                    title={`Recommendation confidence (0.0 – 1.0)`}
                    data-testid="reroute-card-confidence"
                >
                    {(confidence * 100).toFixed(0)}% confident
                </span>
            </header>

            {signals.length > 0 && (
                <ul
                    className="reroute-card__signals"
                    data-testid="reroute-card-signals"
                >
                    {signals.map((s) => (
                        <li
                            key={s.id}
                            className="reroute-card__signal"
                            data-testid={`reroute-card-signal-${s.id}`}
                        >
                            <span className="reroute-card__signal-bullet" aria-hidden="true">•</span>
                            <span>{s.detail}</span>
                        </li>
                    ))}
                </ul>
            )}

            <details className="reroute-card__rationale">
                <summary>Why this recommendation?</summary>
                <p>{rationale}</p>
            </details>

            <div className="reroute-card__footer">
                <button
                    type="button"
                    className="btn btn-primary reroute-card__cta"
                    onClick={onPrimaryClick}
                    disabled={primaryDisabled}
                    data-testid="reroute-card-cta"
                >
                    {primaryLabel}
                </button>
            </div>
        </section>
    );
}


// ─────────────────────────────────────────────────────────────────────
// Switch-to-RAG confirmation modal
// ─────────────────────────────────────────────────────────────────────


interface SwitchToRagConfirmModalProps {
    projectId: number;
    cloning: boolean;
    onCancel: () => void;
    onConfirm: () => void;
}


function SwitchToRagConfirmModal({
    projectId,
    cloning,
    onCancel,
    onConfirm,
}: SwitchToRagConfirmModalProps) {
    return (
        <div
            className="reroute-modal__backdrop"
            role="dialog"
            aria-modal="true"
            aria-labelledby="reroute-modal-title"
            data-testid="reroute-modal"
        >
            <div className="reroute-modal">
                <div className="reroute-modal__head">
                    <h3 id="reroute-modal-title">Switch to RAG?</h3>
                </div>
                <div className="reroute-modal__body">
                    <p>
                        We&apos;ll create a sibling project from project{' '}
                        <strong>#{projectId}</strong>:
                    </p>
                    <ul>
                        <li>Copies your gold set + raw / prepared data forward</li>
                        <li>Uses the base model with retrieval (no training run)</li>
                        <li>Builds the BM25 retrieval index from your gold rows</li>
                        <li>
                            Links back to this project as a parent — you can keep
                            iterating on this SFT run independently
                        </li>
                    </ul>
                    <p className="reroute-modal__note">
                        You can delete the new project later if it doesn&apos;t help.
                        Creation is the moment to slow down.
                    </p>
                </div>
                <div className="reroute-modal__footer">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={onCancel}
                        disabled={cloning}
                        data-testid="reroute-modal-cancel"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={onConfirm}
                        disabled={cloning}
                        data-testid="reroute-modal-confirm"
                    >
                        {cloning ? 'Creating…' : 'Yes, switch to RAG'}
                    </button>
                </div>
            </div>
        </div>
    );
}


// ─────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────


function formatPassRate(rate: number | null): string {
    if (rate === null || rate === undefined || Number.isNaN(rate)) {
        return 'an indeterminate score';
    }
    return `F1 ${rate.toFixed(2)}`;
}
