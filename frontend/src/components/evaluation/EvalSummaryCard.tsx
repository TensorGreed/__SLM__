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
import { fetchEvalSummary, type EvalSummary } from '../../api/evalSummary';
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

    const runEval = async (targetExperimentId: number) => {
        setStarting(true);
        try {
            await api.post(`/projects/${projectId}/evaluation/run-heldout?async_job=true`, {
                experiment_id: targetExperimentId,
                dataset_name: 'test',
                eval_type: 'exact_match',
                max_samples: 100,
                max_new_tokens: 128,
                temperature: 0,
            });
            toast.info('Evaluation queued — the bell will tell you when it is ready.', 4000);
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

    const verdict = VERDICT_TEXT[summary.verdict] ?? VERDICT_TEXT.no_comparison;
    const head = summary.headline;
    const runId = summary.experiment_id;

    return (
        <section
            className={`card eval-summary eval-summary--${verdict.tone}`}
            data-testid="eval-summary"
            data-verdict={summary.verdict}
        >
            <div className="eval-summary__top">
                <div>
                    <h3 className="eval-summary__title">{verdict.title}</h3>
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
            </div>

            {head && (
                <p className="eval-summary__metric" data-testid="eval-summary-headline">
                    <code>{head.metric_id}</code>: {fmt(head.baseline_value)} (base) → <strong>{fmt(head.trained_value)}</strong>{' '}
                    (fine-tuned)
                    <span className={`eval-summary__delta eval-summary__delta--${head.direction}`}>
                        {' '}{head.absolute_delta > 0 ? '+' : ''}{fmt(head.absolute_delta)}
                        {head.relative_delta_pct != null ? ` (${head.relative_delta_pct > 0 ? '+' : ''}${head.relative_delta_pct}%)` : ''}
                    </span>
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
