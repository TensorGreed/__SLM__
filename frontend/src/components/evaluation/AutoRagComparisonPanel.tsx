/**
 * AutoRagComparisonPanel — USER-SUCCESS Epic 9 Phase 9d.
 *
 * "Does retrieval help?" for two models side by side:
 *   - the latest fine-tuned run (comparison.json), and
 *   - the untouched base model (comparison_base.json) — the RAG-first
 *     question. A model fine-tuned on plain question → answer pairs never
 *     saw retrieved context, so its with-RAG number alone understates
 *     what retrieval can do.
 * Each side shows mean token-F1 without / with retrieval on the validation
 * split, labelled with the run id / base model it was measured on, and can
 * be (re-)run as a background Job. Per-row cards below show both
 * generations for the selected model.
 */

import { useEffect, useState } from 'react';
import api from '../../api/client';
import NoRecipeEmptyState from '../shared/NoRecipeEmptyState';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import './AutoRagComparisonPanel.css';

interface AutoRagRow {
    question: string;
    reference: string;
    without_rag: { generated: string; f1: number };
    with_rag: { generated: string; f1: number; retrieved_row_count: number };
}

interface AutoRagSummary {
    off_mean_f1: number;
    on_mean_f1: number;
    absolute_lift?: number;
    relative_lift_pct: number | null;
    n_val_rows: number;
    rag_k: number;
}

interface AutoRagModelComparison {
    cached_at: string | null;
    summary: AutoRagSummary | null;
    rows: AutoRagRow[];
    experiment_id?: number | null;
    base_model?: string | null;
}

type ComparisonModel = 'fine_tuned' | 'base';

interface AutoRagComparisonResponse {
    project_id: number;
    recipe_id: string;
    // Top-level fields are the fine-tuned comparison (summary is null when
    // only the base-model one has been run).
    cached_at: string | null;
    summary: AutoRagSummary | null;
    rows: AutoRagRow[];
    experiment_id?: number | null;
    base_model?: string | null;
    base?: AutoRagModelComparison | null;
}

function hasSummary(summary: AutoRagSummary | null | undefined): summary is AutoRagSummary {
    return !!summary && typeof summary.off_mean_f1 === 'number' && typeof summary.on_mean_f1 === 'number';
}

function shortModelName(name: string | null | undefined): string {
    const value = String(name || '').trim();
    return value ? value.split('/').pop() || value : 'base model';
}

interface Props {
    projectId: number;
}

export default function AutoRagComparisonPanel({ projectId }: Props) {
    const [data, setData] = useState<AutoRagComparisonResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [status, setStatus] = useState<number | null>(null);
    // Structured error_code from the 400 path — used to render the
    // shared "pick a recipe first" CTA only when the project has no
    // recipe, vs. the silent-hide path when the recipe is set but
    // not RAG-eligible (e.g. classification).
    const [errorCode, setErrorCode] = useState<string | null>(null);
    const [submitting, setSubmitting] = useState(false);
    const [rowsModel, setRowsModel] = useState<ComparisonModel>('fine_tuned');

    const handleRunComparison = async (model: ComparisonModel = 'fine_tuned') => {
        if (submitting) return;
        setSubmitting(true);
        try {
            const url = `/projects/${projectId}/auto-rag/comparison/run`;
            const resp = model === 'base'
                ? await api.post(url, null, { params: { model: 'base' } })
                : await api.post(url);
            const jobId = (resp.data as { id?: number })?.id;
            const label = model === 'base' ? 'Base-model auto-RAG comparison' : 'Auto-RAG comparison';
            toast.info(
                jobId
                    ? `${label} queued — track in the bell (job #${jobId})`
                    : `${label} queued — track in the bell`,
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
        setErrorCode(null);
        api.get<AutoRagComparisonResponse>(
            `/projects/${projectId}/auto-rag/comparison`,
        ).then((resp) => {
            if (cancelled) return;
            setData(resp.data);
            setStatus(resp.status);
        }).catch((err) => {
            if (cancelled) return;
            setStatus(err?.response?.status ?? null);
            // Backend's structured 400 path returns
            // ``{error_code, message}`` for RECIPE_REQUIRED; legacy
            // 400s return a plain string. Normalize both into
            // (errorCode, message) so the render path can branch.
            const detail = err?.response?.data?.detail;
            if (detail && typeof detail === 'object') {
                setErrorCode(String(detail.error_code || '') || null);
                setError(String(detail.message || '') || 'Failed to load auto-RAG comparison');
            } else {
                setError(
                    (typeof detail === 'string' ? detail : '')
                    || err?.message
                    || 'Failed to load auto-RAG comparison',
                );
            }
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

    // 400 splits two ways now that the backend returns a structured
    // error_code: RECIPE_REQUIRED → render the shared "pick a recipe
    // first" CTA so legacy NULL-recipe projects get a signal instead
    // of a vanished panel; anything else (recipe set but not
    // RAG-eligible, e.g. classification) keeps the silent-hide
    // behavior since that's the correct signal for "doesn't apply
    // to your task shape."
    if (status === 400) {
        if (errorCode === 'RECIPE_REQUIRED') {
            return (
                <NoRecipeEmptyState
                    projectId={projectId}
                    surface="Auto-RAG comparison"
                    testId="auto-rag-comparison-recipe-required"
                />
            );
        }
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
                    No comparison yet for this project. Auto-RAG retrieves
                    relevant Q&A pairs from your training rows at inference
                    time. The comparison scores a model with and without
                    retrieval on the validation split, so you can see whether
                    retrieval actually helps here.
                </p>
                <div className="auto-rag-comparison__cta-row">
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => void handleRunComparison('fine_tuned')}
                        disabled={submitting}
                        data-testid="auto-rag-comparison-run-btn"
                    >
                        {submitting ? 'Starting…' : 'Run comparison'}
                    </button>
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void handleRunComparison('base')}
                        disabled={submitting}
                        data-testid="auto-rag-comparison-run-base-btn"
                    >
                        Run on the base model
                    </button>
                    <span className="auto-rag-comparison__hint">
                        Runs ~2 min on a GPU. "Run comparison" scores your
                        latest trained run; "Run on the base model" needs no
                        trained run. Track progress in the notification bell.
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

    const fineTuned: AutoRagModelComparison = {
        cached_at: data.cached_at,
        summary: data.summary,
        rows: data.rows || [],
        experiment_id: data.experiment_id,
        base_model: data.base_model,
    };
    const base: AutoRagModelComparison | null = data.base ?? null;
    const fineTunedReady = hasSummary(fineTuned.summary);
    const baseReady = hasSummary(base?.summary);
    // Defensive: tests / partial responses can hand back a `data` without
    // either summary; treat that as "comparison unavailable" rather than
    // crashing on the reads below.
    if (!fineTunedReady && !baseReady) {
        return null;
    }
    const baseModelName = shortModelName(base?.base_model || data.base_model);
    const fineTunedTitle = fineTuned.experiment_id != null
        ? `Fine-tuned · run #${fineTuned.experiment_id}`
        : 'Fine-tuned · latest run';

    // Which of the four numbers is highest — only when both models were
    // measured, so the sentence never compares against a missing side.
    let verdict: string | null = null;
    if (fineTunedReady && baseReady && fineTuned.summary && base?.summary) {
        const candidates = [
            { label: `${fineTunedTitle.toLowerCase()} without retrieval`, f1: fineTuned.summary.off_mean_f1 },
            { label: `${fineTunedTitle.toLowerCase()} with retrieval`, f1: fineTuned.summary.on_mean_f1 },
            { label: `base model without retrieval`, f1: base.summary.off_mean_f1 },
            { label: `base model with retrieval`, f1: base.summary.on_mean_f1 },
        ];
        const best = candidates.reduce((a, b) => (b.f1 > a.f1 ? b : a));
        verdict = `Highest of the four: ${best.label} (F1 ${best.f1.toFixed(3)}).`;
    }

    const activeRowsModel: ComparisonModel =
        rowsModel === 'base' ? (baseReady ? 'base' : 'fine_tuned') : (fineTunedReady ? 'fine_tuned' : 'base');
    const activeRows = activeRowsModel === 'base' ? (base?.rows || []) : fineTuned.rows;

    return (
        <section
            className="auto-rag-comparison"
            data-testid="auto-rag-comparison"
        >
            <header className="auto-rag-comparison__head">
                <div>
                    <h3>Auto-RAG comparison</h3>
                    <p className="auto-rag-comparison__subtitle">
                        Mean token-F1 on the validation split, without and with
                        retrieval, for your fine-tuned run and for the untouched
                        base model.
                    </p>
                </div>
            </header>
            {verdict && (
                <p className="auto-rag-comparison__verdict" data-testid="auto-rag-comparison-verdict">
                    {verdict}
                </p>
            )}
            <div className="auto-rag-comparison__models">
                <ModelComparisonCard
                    testIdPrefix="auto-rag-comparison"
                    runTestId="auto-rag-comparison-finetuned-run-btn"
                    title={fineTunedTitle}
                    provenance={shortModelName(data.base_model) + ' + your training'}
                    comparison={fineTunedReady ? fineTuned : null}
                    emptyText="Not run yet. Needs a completed training run."
                    submitting={submitting}
                    onRun={() => void handleRunComparison('fine_tuned')}
                />
                <ModelComparisonCard
                    testIdPrefix="auto-rag-comparison-base"
                    runTestId="auto-rag-comparison-base-run-btn"
                    title={`Base model · ${baseModelName}`}
                    provenance="No fine-tuning — what a RAG-first project serves"
                    comparison={baseReady ? base : null}
                    emptyText="Not run yet. Shows what retrieval alone does, without your fine-tune."
                    submitting={submitting}
                    onRun={() => void handleRunComparison('base')}
                />
            </div>
            <div className="auto-rag-comparison__rows-head">
                <h4 className="auto-rag-comparison__section-title">Per-row comparison</h4>
                {fineTunedReady && baseReady && (
                    <div className="auto-rag-comparison__rows-toggle" role="group" aria-label="Model for per-row comparison">
                        <button
                            type="button"
                            className={'btn btn-secondary' + (activeRowsModel === 'fine_tuned' ? ' is-active' : '')}
                            onClick={() => setRowsModel('fine_tuned')}
                            data-testid="auto-rag-comparison-rows-finetuned"
                        >
                            Fine-tuned
                        </button>
                        <button
                            type="button"
                            className={'btn btn-secondary' + (activeRowsModel === 'base' ? ' is-active' : '')}
                            onClick={() => setRowsModel('base')}
                            data-testid="auto-rag-comparison-rows-base"
                        >
                            Base model
                        </button>
                    </div>
                )}
            </div>
            <p className="auto-rag-comparison__rows-caption" data-testid="auto-rag-comparison-rows-caption">
                Showing {activeRows.length} row{activeRows.length === 1 ? '' : 's'} for{' '}
                {activeRowsModel === 'base' ? `the base model (${baseModelName})` : fineTunedTitle.toLowerCase()}.
            </p>
            <ul className="auto-rag-comparison__rows">
                {activeRows.map((row, idx) => (
                    <AutoRagRowCard key={`${activeRowsModel}-${idx}`} idx={idx} row={row} />
                ))}
            </ul>
        </section>
    );
}


interface ModelComparisonCardProps {
    /** Prefix for the off/on/lift/rerun test ids. */
    testIdPrefix: string;
    /** Test id of the Run button shown when this side has not been run. */
    runTestId: string;
    title: string;
    provenance: string;
    comparison: AutoRagModelComparison | null;
    emptyText: string;
    submitting: boolean;
    onRun: () => void;
}

function ModelComparisonCard({
    testIdPrefix,
    runTestId,
    title,
    provenance,
    comparison,
    emptyText,
    submitting,
    onRun,
}: ModelComparisonCardProps) {
    const summary = comparison?.summary ?? null;
    if (!comparison || !summary) {
        return (
            <div className="auto-rag-comparison__model auto-rag-comparison__model--empty" data-testid={`${testIdPrefix}-card`}>
                <div className="auto-rag-comparison__model-title">{title}</div>
                <div className="auto-rag-comparison__model-sub">{provenance}</div>
                <p className="auto-rag-comparison__model-empty">{emptyText}</p>
                <button
                    type="button"
                    className="btn btn-primary auto-rag-comparison__rerun"
                    onClick={onRun}
                    disabled={submitting}
                    data-testid={runTestId}
                >
                    {submitting ? 'Starting…' : 'Run'}
                </button>
            </div>
        );
    }
    const lift = summary.relative_lift_pct;
    const liftStr = lift !== null && lift !== undefined ? `${lift >= 0 ? '+' : ''}${lift.toFixed(1)}%` : '—';
    const tone = lift !== null && lift !== undefined && lift > 0
        ? ' is-positive'
        : lift !== null && lift !== undefined && lift < 0 ? ' is-negative' : '';
    return (
        <div className="auto-rag-comparison__model" data-testid={`${testIdPrefix}-card`}>
            <div className="auto-rag-comparison__model-head">
                <div>
                    <div className="auto-rag-comparison__model-title">{title}</div>
                    <div className="auto-rag-comparison__model-sub">
                        {provenance} · {summary.n_val_rows} val rows · top-{summary.rag_k} retrieval
                        {comparison.cached_at ? ` · measured ${new Date(comparison.cached_at).toLocaleString()}` : ''}
                    </div>
                </div>
                <button
                    type="button"
                    className="btn btn-secondary auto-rag-comparison__rerun"
                    onClick={onRun}
                    disabled={submitting}
                    data-testid={`${testIdPrefix}-rerun-btn`}
                    title="Re-runs inference twice (with + without RAG) on the val split. Watch progress in the notification bell."
                >
                    {submitting ? 'Starting…' : 'Re-run comparison'}
                </button>
            </div>
            <div className="auto-rag-comparison__totals">
                <div className="auto-rag-comparison__totals-cell">
                    <div className="auto-rag-comparison__totals-label">Without RAG</div>
                    <div className="auto-rag-comparison__totals-value" data-testid={`${testIdPrefix}-off-f1`}>
                        {summary.off_mean_f1.toFixed(4)}
                    </div>
                    <div className="auto-rag-comparison__totals-sub">mean token-F1</div>
                </div>
                <div className="auto-rag-comparison__totals-cell">
                    <div className="auto-rag-comparison__totals-label">With auto-RAG</div>
                    <div className={'auto-rag-comparison__totals-value' + tone} data-testid={`${testIdPrefix}-on-f1`}>
                        {summary.on_mean_f1.toFixed(4)}
                    </div>
                    <div className="auto-rag-comparison__totals-sub">mean token-F1</div>
                </div>
                <div className="auto-rag-comparison__totals-cell">
                    <div className="auto-rag-comparison__totals-label">Lift from retrieval</div>
                    <div className={'auto-rag-comparison__totals-value' + tone} data-testid={`${testIdPrefix}-lift`}>
                        {liftStr}
                    </div>
                    <div className="auto-rag-comparison__totals-sub">relative to without RAG</div>
                </div>
            </div>
        </div>
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
