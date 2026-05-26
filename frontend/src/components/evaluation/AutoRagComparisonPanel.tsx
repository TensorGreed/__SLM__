/**
 * AutoRagComparisonPanel — USER-SUCCESS Epic 9 Phase 9d.
 *
 * Renders the cached auto-RAG comparison the harness's --project
 * mode produces. Shows aggregate F1 (with-RAG vs without-RAG) at
 * the top + per-row expandable cards underneath showing both
 * generations + the retrieved (Q, A) pairs that fed the with-RAG
 * condition.
 *
 * Read-only: the comparison run is expensive (~2 min on GB10), so
 * the panel surfaces a "not yet run" empty state with the exact
 * harness command when the cache is missing. Wiring a "Run
 * comparison" button is deferred to Phase 9d.1 if/when users ask
 * for it.
 */

import { useEffect, useState } from 'react';
import api from '../../api/client';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import './AutoRagComparisonPanel.css';

interface AutoRagRow {
    question: string;
    reference: string;
    without_rag: { generated: string; f1: number };
    with_rag: { generated: string; f1: number; retrieved_row_count: number };
}

interface AutoRagComparisonResponse {
    project_id: number;
    recipe_id: string;
    cached_at: string;
    summary: {
        off_mean_f1: number;
        on_mean_f1: number;
        absolute_lift: number;
        relative_lift_pct: number | null;
        n_val_rows: number;
        rag_k: number;
        phase_9c_reference_lift_pct: number;
    };
    rows: AutoRagRow[];
}

interface Props {
    projectId: number;
}

export default function AutoRagComparisonPanel({ projectId }: Props) {
    const [data, setData] = useState<AutoRagComparisonResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [status, setStatus] = useState<number | null>(null);
    const [submitting, setSubmitting] = useState(false);

    const handleRunComparison = async () => {
        if (submitting) return;
        setSubmitting(true);
        try {
            const resp = await api.post(
                `/projects/${projectId}/auto-rag/comparison/run`,
            );
            const jobId = (resp.data as { id?: number })?.id;
            toast.info(
                jobId
                    ? `Auto-RAG comparison queued — track in the bell (job #${jobId})`
                    : 'Auto-RAG comparison queued — track in the bell',
                4000,
            );
            void useJobsStore.getState().refreshAfterLocalChange();
        } catch (err) {
            const respErr = err as {
                response?: { status?: number; data?: { detail?: unknown } };
                message?: string;
            };
            const httpStatus = respErr?.response?.status;
            const detail = respErr?.response?.data?.detail;
            if (httpStatus === 409) {
                // Idempotency — surface the existing-job hint that
                // the backend put in metadata.
                const message =
                    typeof detail === 'object' && detail !== null
                        ? (detail as { message?: string }).message
                            || 'A comparison run is already in flight for this project.'
                        : String(detail || 'A comparison is already running.');
                toast.warning(message, 4000);
            } else {
                const message =
                    typeof detail === 'string'
                        ? detail
                        : respErr?.message || 'Failed to start comparison run';
                toast.error(message, 4000);
            }
        } finally {
            setSubmitting(false);
        }
    };

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        setError(null);
        setStatus(null);
        api.get<AutoRagComparisonResponse>(
            `/projects/${projectId}/auto-rag/comparison`,
        ).then((resp) => {
            if (cancelled) return;
            setData(resp.data);
            setStatus(resp.status);
        }).catch((err) => {
            if (cancelled) return;
            setStatus(err?.response?.status ?? null);
            setError(
                err?.response?.data?.detail
                || err?.message
                || 'Failed to load auto-RAG comparison',
            );
        }).finally(() => {
            if (!cancelled) setLoading(false);
        });
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    if (loading) {
        return (
            <section
                className="auto-rag-comparison auto-rag-comparison--loading"
                data-testid="auto-rag-comparison-loading"
            >
                <p>Loading auto-RAG comparison…</p>
            </section>
        );
    }

    // 400 = recipe not RAG-eligible (classification, span, …). For
    // those projects the panel just doesn't render — keeps the Eval
    // tab clean for non-QA recipes. We surface a small message only
    // for QA projects that haven't run the comparison yet (404).
    if (status === 400) {
        return null;
    }
    if (status === 404) {
        return (
            <section
                className="auto-rag-comparison auto-rag-comparison--empty"
                data-testid="auto-rag-comparison-empty"
            >
                <h3>Auto-RAG comparison</h3>
                <p>
                    No comparison cached yet for this project. Auto-RAG
                    retrieves relevant Q&A pairs from your training corpus at
                    inference time; Phase 9c measured <strong>+146% F1 lift</strong>{' '}
                    on the policy-qa-style template vs the SFT-only baseline.
                </p>
                <div className="auto-rag-comparison__cta-row">
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={handleRunComparison}
                        disabled={submitting}
                        data-testid="auto-rag-comparison-run-btn"
                    >
                        {submitting ? 'Starting…' : 'Run comparison'}
                    </button>
                    <span className="auto-rag-comparison__hint">
                        Runs ~2 min on a GPU. Loads your latest trained model
                        and scores both with-RAG and without-RAG inference over
                        the val split. Track progress in the notification bell.
                    </span>
                </div>
                <details className="auto-rag-comparison__cli-fallback">
                    <summary>Or run from the CLI</summary>
                    <pre
                        className="auto-rag-comparison__cmd"
                        data-testid="auto-rag-comparison-empty-cmd"
                    >python -m backend.scripts.auto_rag_ab --project {projectId}</pre>
                </details>
            </section>
        );
    }
    if (error || !data) {
        return (
            <section
                className="auto-rag-comparison auto-rag-comparison--error"
                data-testid="auto-rag-comparison-error"
            >
                <p>{error || 'Auto-RAG comparison unavailable.'}</p>
            </section>
        );
    }

    const lift = data.summary.relative_lift_pct;
    const liftStr = lift !== null ? `${lift >= 0 ? '+' : ''}${lift.toFixed(1)}%` : '—';
    const liftIsPositive = lift !== null && lift > 0;

    return (
        <section
            className="auto-rag-comparison"
            data-testid="auto-rag-comparison"
        >
            <header className="auto-rag-comparison__head">
                <div>
                    <h3>Auto-RAG vs SFT-only</h3>
                    <p className="auto-rag-comparison__subtitle">
                        {data.summary.n_val_rows} val rows · top-{data.summary.rag_k} retrieval ·
                        cached {new Date(data.cached_at).toLocaleString()}
                    </p>
                </div>
                <button
                    type="button"
                    className="btn btn-secondary auto-rag-comparison__rerun"
                    onClick={handleRunComparison}
                    disabled={submitting}
                    data-testid="auto-rag-comparison-rerun-btn"
                    title="Re-runs inference twice (with + without RAG) on the val split. Watch progress in the notification bell."
                >
                    {submitting ? 'Starting…' : 'Re-run comparison'}
                </button>
            </header>
            <div className="auto-rag-comparison__totals">
                <div className="auto-rag-comparison__totals-cell">
                    <div className="auto-rag-comparison__totals-label">Without RAG</div>
                    <div
                        className="auto-rag-comparison__totals-value"
                        data-testid="auto-rag-comparison-off-f1"
                    >
                        {data.summary.off_mean_f1.toFixed(4)}
                    </div>
                    <div className="auto-rag-comparison__totals-sub">mean token-F1</div>
                </div>
                <div className="auto-rag-comparison__totals-cell">
                    <div className="auto-rag-comparison__totals-label">With auto-RAG</div>
                    <div
                        className={
                            'auto-rag-comparison__totals-value'
                            + (liftIsPositive ? ' is-positive' : '')
                        }
                        data-testid="auto-rag-comparison-on-f1"
                    >
                        {data.summary.on_mean_f1.toFixed(4)}
                    </div>
                    <div className="auto-rag-comparison__totals-sub">mean token-F1</div>
                </div>
                <div className="auto-rag-comparison__totals-cell">
                    <div className="auto-rag-comparison__totals-label">Lift</div>
                    <div
                        className={
                            'auto-rag-comparison__totals-value'
                            + (liftIsPositive ? ' is-positive' : '')
                        }
                        data-testid="auto-rag-comparison-lift"
                    >
                        {liftStr}
                    </div>
                    <div className="auto-rag-comparison__totals-sub">
                        Phase 9c ref: +{data.summary.phase_9c_reference_lift_pct.toFixed(1)}%
                    </div>
                </div>
            </div>
            <h4 className="auto-rag-comparison__section-title">Per-row comparison</h4>
            <ul className="auto-rag-comparison__rows">
                {data.rows.map((row, idx) => (
                    <AutoRagRowCard key={idx} idx={idx} row={row} />
                ))}
            </ul>
        </section>
    );
}


interface RowCardProps {
    idx: number;
    row: AutoRagRow;
}

function AutoRagRowCard({ idx, row }: RowCardProps) {
    const liftAbs = row.with_rag.f1 - row.without_rag.f1;
    const liftSign = liftAbs > 0 ? '+' : '';
    return (
        <li
            className="auto-rag-comparison__row"
            data-testid={`auto-rag-comparison-row-${idx}`}
        >
            <details>
                <summary>
                    <span className="auto-rag-comparison__row-q">
                        Q: <code>{row.question.slice(0, 140)}</code>
                    </span>
                    <span
                        className={
                            'auto-rag-comparison__row-lift'
                            + (liftAbs > 0 ? ' is-positive' : liftAbs < 0 ? ' is-negative' : '')
                        }
                    >
                        {liftSign}{(liftAbs * 100).toFixed(1)} F1 pts
                    </span>
                    <span className="auto-rag-comparison__row-f1pair">
                        off={row.without_rag.f1.toFixed(3)} · on={row.with_rag.f1.toFixed(3)}
                    </span>
                </summary>
                <div className="auto-rag-comparison__row-body">
                    <div className="auto-rag-comparison__row-block">
                        <strong>Reference:</strong>
                        <div className="auto-rag-comparison__row-text">{row.reference}</div>
                    </div>
                    <div className="auto-rag-comparison__row-block">
                        <strong>Without RAG (F1={row.without_rag.f1.toFixed(3)}):</strong>
                        <div className="auto-rag-comparison__row-text">{row.without_rag.generated}</div>
                    </div>
                    <div className="auto-rag-comparison__row-block">
                        <strong>
                            With auto-RAG (F1={row.with_rag.f1.toFixed(3)}, {row.with_rag.retrieved_row_count} chunks):
                        </strong>
                        <div className="auto-rag-comparison__row-text">{row.with_rag.generated}</div>
                    </div>
                </div>
            </details>
        </li>
    );
}
