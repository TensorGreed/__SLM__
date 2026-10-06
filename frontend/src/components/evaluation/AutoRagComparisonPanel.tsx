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

import { useEffect, useRef, useState } from 'react';
import api from '../../api/client';
import NoRecipeEmptyState from '../shared/NoRecipeEmptyState';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import { evidenceNote, rowCountsText, type LiftEvidence } from './liftEvidence';
import './AutoRagComparisonPanel.css';

/** The answer judge's verdict on one generated answer (long-answer tasks). */
interface AutoRagJudgeVerdict {
    score: number;
    verdict: 'correct' | 'partial' | 'wrong';
    reason: string;
}

interface AutoRagRow {
    question: string;
    reference: string;
    without_rag: { generated: string; f1: number; judge?: AutoRagJudgeVerdict | null };
    with_rag: {
        generated: string;
        f1: number;
        retrieved_row_count: number;
        judge?: AutoRagJudgeVerdict | null;
        retrieved_sources?: string[];
    };
}

interface AutoRagJudgeArm {
    score: number | null;
    counts?: { correct?: number; partial?: number; wrong?: number } | null;
}

/** Judge summary for one comparison: who judged + each arm's mean. */
interface AutoRagJudgeSummary {
    judge: string;
    without_rag: AutoRagJudgeArm;
    with_rag: AutoRagJudgeArm;
}

interface AutoRagSummary {
    off_mean_f1: number;
    on_mean_f1: number;
    absolute_lift?: number;
    relative_lift_pct: number | null;
    n_val_rows: number;
    rag_k: number;
    judge?: AutoRagJudgeSummary | null;
}

interface AutoRagModelComparison {
    cached_at: string | null;
    summary: AutoRagSummary | null;
    rows: AutoRagRow[];
    experiment_id?: number | null;
    base_model?: string | null;
    corpus?: 'qa' | 'documents' | null;
    evidence?: LiftEvidence | null;
    /** Paired row evidence on the judge scores (null when not judged). */
    judge_evidence?: LiftEvidence | null;
}

/** 'base_documents' = the base model retrieving the project's document
 *  passages (what a documents-only project serves). */
type ComparisonModel = 'fine_tuned' | 'base' | 'base_documents';

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
    evidence?: LiftEvidence | null;
    judge_evidence?: LiftEvidence | null;
    base?: AutoRagModelComparison | null;
    base_documents?: AutoRagModelComparison | null;
    /** The project's latest trained run, and whether the fine-tuned
     *  comparison was measured on an older one. */
    latest_experiment_id?: number | null;
    stale?: boolean;
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
    // Bumped to re-read the cached comparisons without the loading flash.
    const [reloadToken, setReloadToken] = useState(0);

    // This project's comparison Jobs, from the store the bell polls. One in
    // flight → show which side is running; one finishing → re-read the cache,
    // so the numbers appear without a page reload.
    const jobs = useJobsStore((state) => state.jobs);
    const comparisonJobs = (jobs || []).filter(
        (job) => job.kind === 'auto_rag_comparison' && job.project_id === projectId,
    );
    const inFlightJob = comparisonJobs.find(
        (job) => job.status === 'queued' || job.status === 'running',
    ) ?? null;
    const runningModel: ComparisonModel | null = inFlightJob
        ? (inFlightJob.params?.model === 'base'
            ? (inFlightJob.params?.corpus === 'documents' ? 'base_documents' : 'base')
            : 'fine_tuned')
        : null;
    const runningMessage = inFlightJob?.progress_message || null;
    const finishedKey = comparisonJobs
        .filter((job) => job.status === 'succeeded')
        .map((job) => `${job.id}:${job.completed_at ?? ''}`)
        .sort()
        .join(',');
    const lastFinishedKey = useRef<string | null>(null);
    useEffect(() => {
        if (lastFinishedKey.current !== null && lastFinishedKey.current !== finishedKey) {
            setReloadToken((token) => token + 1);
        }
        lastFinishedKey.current = finishedKey;
    }, [finishedKey]);

    const handleRunComparison = async (model: ComparisonModel = 'fine_tuned') => {
        if (submitting) return;
        setSubmitting(true);
        try {
            const url = `/projects/${projectId}/auto-rag/comparison/run`;
            const resp = model === 'base'
                ? await api.post(url, null, { params: { model: 'base' } })
                : model === 'base_documents'
                    ? await api.post(url, null, { params: { model: 'base', corpus: 'documents' } })
                    : await api.post(url);
            const jobId = (resp.data as { id?: number })?.id;
            const label = model === 'base'
                ? 'Base-model auto-RAG comparison'
                : model === 'base_documents'
                    ? 'Base model + document passages comparison'
                    : 'Auto-RAG comparison';
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

    const loadedProject = useRef<number | null>(null);
    useEffect(() => {
        let cancelled = false;
        // A refresh after a Job finishes keeps the current numbers on screen
        // until the new ones arrive; only a project switch shows "Loading…".
        const silent = loadedProject.current === projectId;
        loadedProject.current = projectId;
        if (!silent) {
            setLoading(true);
            setError(null);
            setStatus(null);
            setErrorCode(null);
        }
        api.get<AutoRagComparisonResponse>(
            `/projects/${projectId}/auto-rag/comparison`,
        ).then((resp) => {
            if (cancelled) return;
            setData(resp.data);
            setStatus(resp.status);
            setError(null);
            setErrorCode(null);
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
    }, [projectId, reloadToken]);

    const busy = submitting || inFlightJob !== null;

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
                        disabled={busy}
                        data-testid="auto-rag-comparison-run-btn"
                    >
                        {submitting ? 'Starting…' : 'Run comparison'}
                    </button>
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void handleRunComparison('base')}
                        disabled={busy}
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
                {inFlightJob && (
                    <p className="auto-rag-comparison__running" data-testid="auto-rag-comparison-running">
                        {runningModel === 'base' ? 'Base-model comparison' : 'Comparison'} running
                        {runningMessage ? ` — ${runningMessage}` : '…'} The results appear here when it finishes.
                    </p>
                )}
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
        evidence: data.evidence ?? null,
        judge_evidence: data.judge_evidence ?? null,
    };
    const base: AutoRagModelComparison | null = data.base ?? null;
    const baseDocuments: AutoRagModelComparison | null = data.base_documents ?? null;
    const fineTunedReady = hasSummary(fineTuned.summary);
    const baseReady = hasSummary(base?.summary);
    const baseDocumentsReady = hasSummary(baseDocuments?.summary);
    // Defensive: tests / partial responses can hand back a `data` without
    // any summary; treat that as "comparison unavailable" rather than
    // crashing on the reads below.
    if (!fineTunedReady && !baseReady && !baseDocumentsReady) {
        return null;
    }
    const baseModelName = shortModelName(base?.base_model || baseDocuments?.base_model || data.base_model);
    const fineTunedTitle = fineTuned.experiment_id != null
        ? `Fine-tuned · run #${fineTuned.experiment_id}`
        : 'Fine-tuned · latest run';

    // Which of the four numbers is highest — only when both models were
    // measured, so the sentence never compares against a missing side.
    let verdict: string | null = null;
    if (fineTunedReady && baseReady && fineTuned.summary && base?.summary) {
        const candidates: Array<{ label: string; f1: number; retrievalEvidence: LiftEvidence | null }> = [
            { label: `${fineTunedTitle.toLowerCase()} without retrieval`, f1: fineTuned.summary.off_mean_f1, retrievalEvidence: null },
            { label: `${fineTunedTitle.toLowerCase()} with retrieval`, f1: fineTuned.summary.on_mean_f1, retrievalEvidence: fineTuned.evidence ?? null },
            { label: `base model without retrieval`, f1: base.summary.off_mean_f1, retrievalEvidence: null },
            { label: `base model with retrieval`, f1: base.summary.on_mean_f1, retrievalEvidence: base.evidence ?? null },
        ];
        const best = candidates.reduce((a, b) => (b.f1 > a.f1 ? b : a));
        verdict = `Highest of the four: ${best.label} (F1 ${best.f1.toFixed(3)}).`;
        // The top number is a with-retrieval one whose edge over the same
        // model without retrieval isn't established — say so right here.
        const ev = best.retrievalEvidence;
        if (ev && (ev.verdict === 'within_noise' || ev.verdict === 'too_few_rows')) {
            verdict += ' Its edge over the same model without retrieval is within noise.';
        }
    }

    const readyModels: ComparisonModel[] = [
        ...(fineTunedReady ? ['fine_tuned' as const] : []),
        ...(baseReady ? ['base' as const] : []),
        ...(baseDocumentsReady ? ['base_documents' as const] : []),
    ];
    const activeRowsModel: ComparisonModel = readyModels.includes(rowsModel) ? rowsModel : readyModels[0];
    const activeRows = activeRowsModel === 'base'
        ? (base?.rows || [])
        : activeRowsModel === 'base_documents'
            ? (baseDocuments?.rows || [])
            : fineTuned.rows;
    const rowsModelLabel = (model: ComparisonModel): string =>
        model === 'base'
            ? `the base model (${baseModelName})`
            : model === 'base_documents'
                ? `the base model (${baseModelName}) with document passages`
                : fineTunedTitle.toLowerCase();

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
                    trained
                    provenance={shortModelName(data.base_model) + ' + your training'}
                    comparison={fineTunedReady ? fineTuned : null}
                    emptyText="Not run yet. Needs a completed training run."
                    submitting={submitting}
                    disabled={busy}
                    runningMessage={runningModel === 'fine_tuned' ? (runningMessage || 'starting…') : null}
                    staleNote={
                        fineTunedReady && data.stale && data.latest_experiment_id != null
                            ? `These numbers are for run #${fineTuned.experiment_id}. Your latest run is #${data.latest_experiment_id} — re-run to measure it.`
                            : null
                    }
                    onRun={() => void handleRunComparison('fine_tuned')}
                />
                <ModelComparisonCard
                    testIdPrefix="auto-rag-comparison-base"
                    runTestId="auto-rag-comparison-base-run-btn"
                    title={`Base model · ${baseModelName}`}
                    trained={false}
                    provenance="No fine-tuning — what a RAG-first project serves"
                    comparison={baseReady ? base : null}
                    emptyText="Not run yet. Shows what retrieval alone does, without your fine-tune."
                    submitting={submitting}
                    disabled={busy}
                    runningMessage={runningModel === 'base' ? (runningMessage || 'starting…') : null}
                    staleNote={null}
                    onRun={() => void handleRunComparison('base')}
                />
                <ModelComparisonCard
                    testIdPrefix="auto-rag-comparison-base-documents"
                    runTestId="auto-rag-comparison-base-documents-run-btn"
                    title={`Base model + document passages · ${baseModelName}`}
                    trained={false}
                    provenance="No fine-tuning — retrieves your cleaned document passages, cites them or says it doesn't know"
                    comparison={baseDocumentsReady ? baseDocuments : null}
                    emptyText="Not run yet. Shows what retrieval over your documents does, without your fine-tune and without the Q&A pairs."
                    submitting={submitting}
                    disabled={busy}
                    runningMessage={runningModel === 'base_documents' ? (runningMessage || 'starting…') : null}
                    staleNote={null}
                    onRun={() => void handleRunComparison('base_documents')}
                />
            </div>
            <div className="auto-rag-comparison__rows-head">
                <h4 className="auto-rag-comparison__section-title">Per-row comparison</h4>
                {readyModels.length > 1 && (
                    <div className="auto-rag-comparison__rows-toggle" role="group" aria-label="Model for per-row comparison">
                        {fineTunedReady && (
                            <button
                                type="button"
                                className={'btn btn-secondary' + (activeRowsModel === 'fine_tuned' ? ' is-active' : '')}
                                onClick={() => setRowsModel('fine_tuned')}
                                data-testid="auto-rag-comparison-rows-finetuned"
                            >
                                Fine-tuned
                            </button>
                        )}
                        {baseReady && (
                            <button
                                type="button"
                                className={'btn btn-secondary' + (activeRowsModel === 'base' ? ' is-active' : '')}
                                onClick={() => setRowsModel('base')}
                                data-testid="auto-rag-comparison-rows-base"
                            >
                                Base model
                            </button>
                        )}
                        {baseDocumentsReady && (
                            <button
                                type="button"
                                className={'btn btn-secondary' + (activeRowsModel === 'base_documents' ? ' is-active' : '')}
                                onClick={() => setRowsModel('base_documents')}
                                data-testid="auto-rag-comparison-rows-base-documents"
                            >
                                Base + documents
                            </button>
                        )}
                    </div>
                )}
            </div>
            <p className="auto-rag-comparison__rows-caption" data-testid="auto-rag-comparison-rows-caption">
                Showing {activeRows.length} row{activeRows.length === 1 ? '' : 's'} for {rowsModelLabel(activeRowsModel)}.
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
    /** A fine-tuned run (adds the "one training run" caveat to a verdict). */
    trained: boolean;
    provenance: string;
    comparison: AutoRagModelComparison | null;
    emptyText: string;
    submitting: boolean;
    /** A comparison Job is starting or in flight for this project (either
     *  model) — only one may run at a time. */
    disabled: boolean;
    /** Progress text when THIS side's Job is in flight. */
    runningMessage: string | null;
    /** Shown when the numbers are for an older run than the latest. */
    staleNote: string | null;
    onRun: () => void;
}

function ModelComparisonCard({
    testIdPrefix,
    runTestId,
    title,
    trained,
    provenance,
    comparison,
    emptyText,
    submitting,
    disabled,
    runningMessage,
    staleNote,
    onRun,
}: ModelComparisonCardProps) {
    const summary = comparison?.summary ?? null;
    const running = runningMessage !== null && (
        <p className="auto-rag-comparison__running" data-testid={`${testIdPrefix}-running`}>
            Running — {runningMessage}
        </p>
    );
    if (!comparison || !summary) {
        return (
            <div className="auto-rag-comparison__model auto-rag-comparison__model--empty" data-testid={`${testIdPrefix}-card`}>
                <div className="auto-rag-comparison__model-title">{title}</div>
                <div className="auto-rag-comparison__model-sub">{provenance}</div>
                <p className="auto-rag-comparison__model-empty">{emptyText}</p>
                {running}
                <button
                    type="button"
                    className="btn btn-primary auto-rag-comparison__rerun"
                    onClick={onRun}
                    disabled={disabled}
                    data-testid={runTestId}
                >
                    {submitting ? 'Starting…' : 'Run'}
                </button>
            </div>
        );
    }
    const lift = summary.relative_lift_pct;
    const liftStr = lift !== null && lift !== undefined ? `${lift >= 0 ? '+' : ''}${lift.toFixed(1)}%` : '—';
    const evidence = comparison.evidence ?? null;
    // Green / red only when the change is beyond row-to-row noise. Without
    // evidence (older payloads) fall back to the sign of the lift.
    const tone = evidence
        ? (evidence.verdict === 'better' ? ' is-positive' : evidence.verdict === 'worse' ? ' is-negative' : '')
        : lift !== null && lift !== undefined && lift > 0
            ? ' is-positive'
            : lift !== null && lift !== undefined && lift < 0 ? ' is-negative' : '';
    const note = evidence ? evidenceNote(evidence, { trained, unit: 'F1' }) : null;
    const judge = summary.judge ?? null;
    const judgeEvidence = comparison.judge_evidence ?? null;
    const judgeNote = judgeEvidence ? evidenceNote(judgeEvidence, { trained, unit: 'judge score' }) : null;
    const armText = (arm: AutoRagJudgeArm): string => {
        const c = arm.counts || {};
        return `${arm.score != null ? arm.score.toFixed(3) : '—'} (${c.correct ?? 0} correct, ${c.partial ?? 0} partial, ${c.wrong ?? 0} wrong)`;
    };
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
                    disabled={disabled}
                    data-testid={`${testIdPrefix}-rerun-btn`}
                    title="Re-runs inference twice (with + without RAG) on the val split. Watch progress in the notification bell."
                >
                    {submitting ? 'Starting…' : 'Re-run comparison'}
                </button>
            </div>
            {staleNote && (
                <p className="auto-rag-comparison__stale" data-testid={`${testIdPrefix}-stale`}>
                    {staleNote}
                </p>
            )}
            {running}
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
            {evidence && note && (
                <div
                    className={`auto-rag-comparison__evidence auto-rag-comparison__evidence--${evidence.verdict}`}
                    data-testid={`${testIdPrefix}-evidence`}
                    data-verdict={evidence.verdict}
                >
                    <span className="auto-rag-comparison__evidence-label">{note.label}</span>
                    <span data-testid={`${testIdPrefix}-row-counts`}>{rowCountsText(evidence, 'Retrieval')}</span>{' '}
                    <span>{note.text}</span>
                </div>
            )}
            {judge && (
                <div className="auto-rag-comparison__judge" data-testid={`${testIdPrefix}-judge`}>
                    <strong>Judge: answer correct</strong> (by <code>{judge.judge}</code>, facts not wording):{' '}
                    without retrieval {armText(judge.without_rag)} → with retrieval {armText(judge.with_rag)}.
                    {judgeEvidence && judgeNote && (
                        <span
                            className={`auto-rag-comparison__evidence auto-rag-comparison__evidence--${judgeEvidence.verdict}`}
                            data-testid={`${testIdPrefix}-judge-evidence`}
                            data-verdict={judgeEvidence.verdict}
                        >
                            {' '}{rowCountsText(judgeEvidence, 'Retrieval')} {judgeNote.text}
                        </span>
                    )}
                </div>
            )}
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
                        {row.without_rag.judge && (
                            <div className={`auto-rag-comparison__row-judge auto-rag-comparison__row-judge--${row.without_rag.judge.verdict}`}>
                                Judge: {row.without_rag.judge.verdict} — {row.without_rag.judge.reason}
                            </div>
                        )}
                    </div>
                    <div className="auto-rag-comparison__row-block">
                        <strong>
                            With auto-RAG (F1={row.with_rag.f1.toFixed(3)}, {row.with_rag.retrieved_row_count} chunks):
                        </strong>
                        <div className="auto-rag-comparison__row-text">{row.with_rag.generated}</div>
                        {row.with_rag.judge && (
                            <div className={`auto-rag-comparison__row-judge auto-rag-comparison__row-judge--${row.with_rag.judge.verdict}`}>
                                Judge: {row.with_rag.judge.verdict} — {row.with_rag.judge.reason}
                            </div>
                        )}
                        {row.with_rag.retrieved_sources && row.with_rag.retrieved_sources.length > 0 && (
                            <div className="auto-rag-comparison__row-sources">
                                Sources: {row.with_rag.retrieved_sources.join(' · ')}
                            </div>
                        )}
                    </div>
                </div>
            </details>
        </li>
    );
}
