/**
 * EvalSummaryCard — the Eval tab's default view (Wave 3b).
 *
 * Answers the three questions a newcomer actually has, for one run:
 *   1. Is the fine-tuned model better than the base model it started from?
 *   2. By how much? (headline metric, base → fine-tuned; perplexity is
 *      lower-is-better and the backend already says which way is good)
 *   3. Where does it still fail? (up to five real failing rows)
 * Everything else on the Eval tab lives under "Advanced evaluation".
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import { checkLiftAcrossSeeds, fetchEvalSummary, type EvalSummary } from '../../api/evalSummary';
import { evidenceNote, metricDisplayName, metricUnit, rowCountsText, seedEvidenceNote, type LiftEvidence, type SeedEvidence } from './liftEvidence';
import { toast } from '../../stores/toastStore';
import { useJobsStore } from '../../stores/jobsStore';
import './EvalSummaryCard.css';

interface EvalSummaryCardProps {
    projectId: number;
    experimentId: number | null;
    /** Bumps to refetch (e.g. new eval results arrived). */
    refreshToken?: unknown;
    /** Called with the run the server summarised (the latest trained run
     *  when ``experimentId`` is null) so the panel can select it. */
    onResolved?: (experimentId: number) => void;
}

const VERDICT_TEXT: Record<string, { title: string; tone: string }> = {
    better: { title: 'Better than the base model', tone: 'good' },
    worse: { title: 'Worse than the base model', tone: 'bad' },
    same: { title: 'No better than the base model', tone: 'neutral' },
    no_baseline: { title: 'Evaluated — no base-model comparison yet', tone: 'neutral' },
    no_comparison: { title: 'Evaluated — nothing comparable to the base model', tone: 'neutral' },
    not_evaluated: { title: 'Not evaluated yet', tone: 'neutral' },
    no_trained_run: { title: 'No trained model yet', tone: 'neutral' },
};

/** A "better" / "worse" headline whose row-level evidence can't back it up
 *  must not be announced as a result. */
function titleFor(
    verdict: string,
    evidence: LiftEvidence | null | undefined,
    seedEvidence: SeedEvidence | null | undefined,
): { title: string; tone: string } {
    const base = VERDICT_TEXT[verdict] ?? VERDICT_TEXT.no_comparison;
    if (verdict !== 'better' && verdict !== 'worse') return base;
    // Run-to-run evidence (seeds) outranks one run's row split.
    if (seedEvidence) {
        if (seedEvidence.verdict === 'within_noise') {
            return {
                title: verdict === 'better'
                    ? 'Ahead of the base model on average — but the seeds disagree'
                    : 'Behind the base model on average — but the seeds disagree',
                tone: 'neutral',
            };
        }
        if (seedEvidence.verdict === 'better' || seedEvidence.verdict === 'worse') {
            return {
                title: `${base.title} — across ${seedEvidence.n} seeds`,
                tone: base.tone,
            };
        }
    }
    if (!evidence) return base;
    if (evidence.verdict === 'within_noise') {
        return {
            title: verdict === 'better'
                ? 'Ahead of the base model — but within noise'
                : 'Behind the base model — but within noise',
            tone: 'neutral',
        };
    }
    if (evidence.verdict === 'too_few_rows') {
        return { title: `${base.title} — too few rows to be sure`, tone: 'neutral' };
    }
    return base;
}

function fmt(value: number): string {
    if (Math.abs(value) >= 10) return value.toFixed(1);
    return value.toFixed(3);
}

function errorText(err: unknown): string {
    const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
    return typeof detail === 'string' && detail ? detail : 'Could not start the evaluation.';
}

export default function EvalSummaryCard({ projectId, experimentId, refreshToken, onResolved }: EvalSummaryCardProps) {
    const [summary, setSummary] = useState<EvalSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [starting, setStarting] = useState(false);

    const load = useCallback(async () => {
        setLoading(true);
        try {
            const next = await fetchEvalSummary(projectId, experimentId);
            setSummary(next);
            if (next?.experiment_id && next.experiment_id !== experimentId) {
                onResolved?.(next.experiment_id);
            }
        } catch {
            setSummary(null);
        } finally {
            setLoading(false);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onResolved is a notify-only callback
    }, [projectId, experimentId]);

    useEffect(() => {
        void load();
    }, [load, refreshToken]);

    const rerunPassagesCheck = async () => {
        setStarting(true);
        try {
            await api.post(`/projects/${projectId}/auto-rag/comparison/run`, null, {
                params: { model: 'base', corpus: 'documents', split: 'test' },
            });
            toast.info('Passages check queued — the bell will tell you when it is ready.', 4000);
            void useJobsStore.getState().refreshAfterLocalChange();
        } catch (err) {
            toast.error(errorText(err));
        } finally {
            setStarting(false);
        }
    };

    const runEval = async (targetExperimentId: number) => {
        setStarting(true);
        try {
            // The lift check: base model + this run on the test examples,
            // paired row by row (what this card compares). Re-running picks
            // up a judge model configured since the first check.
            await api.post(`/projects/${projectId}/evaluation/summary/lift-check`, {
                experiment_id: targetExperimentId,
            });
            toast.info('Lift check queued — the bell will tell you when it is ready.', 4000);
            void useJobsStore.getState().refreshAfterLocalChange();
        } catch (err) {
            toast.error(errorText(err));
        } finally {
            setStarting(false);
        }
    };

    if (loading) {
        return <section className="card eval-summary" data-testid="eval-summary-loading">Checking your model…</section>;
    }
    if (!summary || !summary.verdict) {
        return null;
    }

    const evidence = summary.evidence ?? null;
    const seedEvidence = summary.seed_evidence ?? null;
    const verdict = titleFor(summary.verdict, evidence, seedEvidence);
    const unit = metricUnit(seedEvidence?.metric_id ?? evidence?.metric_id ?? summary.headline?.metric_id);
    const isPassages = summary.kind === 'rag_passages';
    const note = evidence ? evidenceNote(evidence, { trained: !seedEvidence && !isPassages, unit }) : null;
    const subject = isPassages ? 'Retrieval' : 'Fine-tuning';
    const seedNote = seedEvidence ? seedEvidenceNote(seedEvidence, unit) : null;
    const head = summary.headline;
    const runId = summary.experiment_id;
    const isSeedGroup = (summary.n_seeds ?? 1) > 1;
    // The multi-seed option: a single evaluated run can be re-trained with
    // 3 seeds so the verdict also covers run-to-run variance.
    const canCheckSeeds = runId != null && !isSeedGroup
        && (summary.verdict === 'better' || summary.verdict === 'worse' || summary.verdict === 'same');

    const checkSeeds = async (sourceExperimentId: number) => {
        setStarting(true);
        try {
            const out = await checkLiftAcrossSeeds(projectId, sourceExperimentId, 3);
            toast.info(
                `Training run #${sourceExperimentId}'s config with 3 seeds (run #${out.experiment_id}). The bell tells you when the lift check across seeds is in.`,
                6000,
            );
            void useJobsStore.getState().refreshAfterLocalChange();
        } catch (err) {
            toast.error(errorText(err));
        } finally {
            setStarting(false);
        }
    };

    return (
        <section
            className={`card eval-summary eval-summary--${verdict.tone}`}
            data-testid="eval-summary"
            data-verdict={summary.verdict}
        >
            <div className="eval-summary__top">
                <div>
                    <h3 className="eval-summary__title">{verdict.title}</h3>
                    {isPassages && (
                        <p className="eval-summary__run" data-testid="eval-summary-passages-run">
                            Base model + your document passages vs the base model alone
                            {summary.baseline?.base_model ? ` (${summary.baseline.base_model})` : ''}
                            {summary.evaluated_samples ? ` · ${summary.evaluated_samples} ${summary.split === 'test' ? 'test examples' : 'validation rows'}` : ''}
                            {' '}· no training run — retrieval does the work
                        </p>
                    )}
                    {runId != null && (
                        <p className="eval-summary__run">
                            Run #{runId}
                            {summary.trained?.experiment_name ? ` · ${summary.trained.experiment_name}` : ''}
                            {summary.baseline?.base_model ? ` vs base ${summary.baseline.base_model}` : ''}
                            {summary.evaluated_samples ? ` · ${summary.evaluated_samples} test examples` : ''}
                        </p>
                    )}
                </div>
                {runId != null && summary.verdict === 'not_evaluated' && (
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => void runEval(runId)}
                        disabled={starting}
                        data-testid="eval-summary-run"
                    >
                        {starting ? 'Starting…' : 'Evaluate this run'}
                    </button>
                )}
                {isPassages && (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void rerunPassagesCheck()}
                        disabled={starting}
                        data-testid="eval-summary-rerun-passages"
                        title="Scores the base model with and without passage retrieval again on the test examples, judged."
                    >
                        {starting ? 'Starting…' : 'Re-run passages check'}
                    </button>
                )}
                {runId != null && summary.verdict !== 'not_evaluated' && summary.headline && !isSeedGroup && (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void runEval(runId)}
                        disabled={starting}
                        data-testid="eval-summary-rerun"
                        title="Scores the base model and this run again on the current test examples. Picks up a judge model configured since the last check."
                    >
                        {starting ? 'Starting…' : 'Re-run lift check'}
                    </button>
                )}
                {canCheckSeeds && (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void checkSeeds(runId)}
                        disabled={starting}
                        data-testid="eval-summary-check-seeds"
                        title="Re-trains this run's exact config with 3 different seeds, then scores every seed against the base model. Shows whether the result holds run to run, not just on these rows."
                    >
                        {starting ? 'Starting…' : 'Check across 3 seeds'}
                    </button>
                )}
            </div>

            {head && (
                <p className="eval-summary__metric" data-testid="eval-summary-headline">
                    <code>{metricDisplayName(head.metric_id)}</code>: {fmt(head.baseline_value)} (base) → <strong>{fmt(head.trained_value)}</strong>
                    {head.trained_std != null ? ` ± ${fmt(head.trained_std)}` : ''}{' '}
                    ({isPassages ? 'base + your passages' : isSeedGroup ? `fine-tuned, mean of ${head.n_seeds ?? summary.n_seeds} seeds` : 'fine-tuned'})
                    <span className={`eval-summary__delta eval-summary__delta--${head.direction}`}>
                        {' '}{head.absolute_delta > 0 ? '+' : ''}{fmt(head.absolute_delta)}
                        {head.relative_delta_pct != null ? ` (${head.relative_delta_pct > 0 ? '+' : ''}${head.relative_delta_pct}%)` : ''}
                    </span>
                </p>
            )}
            {seedEvidence && seedNote && (
                <p
                    className={`eval-summary__evidence eval-summary__evidence--${seedEvidence.verdict}`}
                    data-testid="eval-summary-seed-evidence"
                    data-verdict={seedEvidence.verdict}
                >
                    <span className="eval-summary__evidence-label">{seedNote.label}</span>
                    <span>{seedNote.text}</span>
                    {summary.seeds && summary.seeds.length > 0 && (
                        <span className="eval-summary__seed-runs" data-testid="eval-summary-seed-runs">
                            {' '}Runs:{' '}
                            {summary.seeds.map((seed, index) => (
                                <span key={seed.experiment_id}>
                                    {index > 0 ? ', ' : ''}#{seed.experiment_id}
                                    {seed.seed_value != null ? ` (seed ${seed.seed_value})` : ''}
                                    {seed.headline ? ` ${fmt(seed.headline.trained_value)}` : ' not evaluated'}
                                </span>
                            ))}
                            {summary.representative_experiment_id != null
                                ? `. Failures and row counts below are from run #${summary.representative_experiment_id} (the median seed).`
                                : ''}
                        </span>
                    )}
                </p>
            )}
            {evidence && note && (
                <p
                    className={`eval-summary__evidence eval-summary__evidence--${evidence.verdict}`}
                    data-testid="eval-summary-evidence"
                    data-verdict={evidence.verdict}
                >
                    <span className="eval-summary__evidence-label">{note.label}</span>
                    <span data-testid="eval-summary-row-counts">{rowCountsText(evidence, subject)}</span>{' '}
                    <span>{note.text}</span>
                </p>
            )}
            {isPassages && summary.retrieval && (
                <p className="eval-summary__judge" data-testid="eval-summary-retrieval">
                    Retrieval served: top-{summary.retrieval.k}
                    {summary.retrieval.reranker ? ` + reranker ${summary.retrieval.reranker.split('/').pop()}` : ' (BM25 only)'}
                    {summary.retrieval_sweep && summary.retrieval_sweep.length > 1
                        ? ` — chosen by the judge over ${summary.retrieval_sweep
                            .filter((arm) => !arm.chosen)
                            .map((arm) => `${arm.label} (${arm.judge_score != null ? arm.judge_score.toFixed(2) : '—'})`)
                            .join(', ')}.`
                        : '.'}
                </p>
            )}
            {summary.judge && (
                <p className="eval-summary__judge" data-testid="eval-summary-judge">
                    Judged by <code>{summary.judge.judge || 'a judge model'}</code>: {summary.judge.correct} correct, {summary.judge.partial} partial,{' '}
                    {summary.judge.wrong} wrong of {summary.judge.judged} answers
                    {summary.judge.unjudged ? ` (${summary.judge.unjudged} unjudged)` : ''}. The judge reads the question, the
                    answer key and the model's answer and grades the facts, not the wording — token F1 is still shown below.
                </p>
            )}
            {summary.message && <p className="eval-summary__message">{summary.message}</p>}

            {summary.failures.length > 0 && (
                <div className="eval-summary__failures" data-testid="eval-summary-failures">
                    <h4>
                        Where it still fails
                        {summary.failed_count ? ` — ${summary.failed_count} of ${summary.evaluated_samples ?? '?'} test examples` : ''}
                    </h4>
                    <ol>
                        {summary.failures.map((failure, index) => (
                            <li key={`${index}-${failure.prompt.slice(0, 20)}`}>
                                <div className="eval-summary__prompt">{failure.prompt}</div>
                                <div className="eval-summary__pair">
                                    <span>Expected: <strong>{failure.reference || '—'}</strong></span>
                                    <span>Got: <strong>{failure.prediction || '(empty)'}</strong></span>
                                </div>
                                {failure.row_judge_verdict && (
                                    <div className={`eval-summary__judge-verdict eval-summary__judge-verdict--${failure.row_judge_verdict}`}>
                                        Judge: {failure.row_judge_verdict}{failure.row_judge_reason ? ` — ${failure.row_judge_reason}` : ''}
                                    </div>
                                )}
                            </li>
                        ))}
                    </ol>
                </div>
            )}
            {summary.eval_type === 'perplexity' && (
                <p className="eval-summary__message">
                    Measured by perplexity on test examples (lower is better) — continued pretraining on your documents has no
                    right/wrong answers per row.
                </p>
            )}
        </section>
    );
}
