/**
 * DocumentsQaFlowCard — the one-click path from a pile of documents to a
 * measured Q&A model: generated question→answer pairs, an answer key, a
 * split, a training run with the project defaults, and the automatic lift
 * check. Shown on the Training tab when the project has cleaned document
 * passages and no Q&A rows yet; hidden otherwise.
 *
 * Progress and the outcome come from the jobs store (the same Job the
 * bell tracks). After the flow starts a run, the usual training watcher +
 * lift check take over.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { fetchDocumentsQaFlowPreview, startDocumentsQaFlow, type DocumentsQaFlowPreview, type DocumentsQaFlowResult } from '../../api/flows';
import { rerouteToRagAsync } from '../../api/rerouteAnalysis';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import './DocumentsQaFlowCard.css';

interface Props {
    projectId: number;
    /** Called after the flow starts a training run, so the panel can reload. */
    onRunStarted?: () => void;
}

function errorText(err: unknown): string {
    const data = (err as { response?: { data?: { detail?: unknown; message?: string } } })?.response?.data;
    return data?.message ?? (typeof data?.detail === 'string' ? data.detail : 'Could not start the flow.');
}

export default function DocumentsQaFlowCard({ projectId, onRunStarted }: Props) {
    const [preview, setPreview] = useState<DocumentsQaFlowPreview | null>(null);
    const [hidden, setHidden] = useState(false);
    const [starting, setStarting] = useState(false);
    const [reviewFirst, setReviewFirst] = useState(false);

    const jobs = useJobsStore((state) => state.jobs);
    const flowJobs = (jobs || []).filter((job) => job.kind === 'documents_qa_flow' && job.project_id === projectId);
    const inFlight = flowJobs.find((job) => job.status === 'queued' || job.status === 'running') ?? null;
    const finished = flowJobs
        .filter((job) => job.status === 'succeeded' || job.status === 'failed')
        .sort((a, b) => b.id - a.id)[0] ?? null;
    const result = finished?.status === 'succeeded' ? ((finished.result || null) as DocumentsQaFlowResult | null) : null;

    const load = useCallback(async () => {
        try {
            const next = await fetchDocumentsQaFlowPreview(projectId);
            setPreview(next);
        } catch {
            setHidden(true);
        }
    }, [projectId]);

    useEffect(() => {
        void load();
    }, [load]);

    // Re-read the preview when a flow finishes (the project is a Q&A project now).
    const lastFinishedId = useRef<number | null>(null);
    useEffect(() => {
        const id = finished?.id ?? null;
        if (lastFinishedId.current !== null && lastFinishedId.current !== id) {
            void load();
            if (result?.experiment_id != null) onRunStarted?.();
        }
        lastFinishedId.current = id;
    }, [finished?.id, load, onRunStarted, result?.experiment_id]);

    const start = async (reuseExisting = false, trainIfRetrievalReady = false) => {
        if (!preview) return;
        setStarting(true);
        try {
            const job = await startDocumentsQaFlow(projectId, {
                maxPassages: preview.plan.max_passages,
                pairsPerPassage: preview.plan.pairs_per_passage,
                train: !reviewFirst,
                reuseExisting,
                trainIfRetrievalReady,
            });
            toast.info(
                reuseExisting
                    ? `Training the generated pairs again — track it in the bell (job #${job.id}).`
                    : `Building the Q&A assistant — track it in the bell (job #${job.id}).`,
                5000,
            );
            void useJobsStore.getState().refreshAfterLocalChange();
        } catch (err) {
            toast.error(errorText(err));
        } finally {
            setStarting(false);
        }
    };

    const reroute = async () => {
        setStarting(true);
        try {
            const job = await rerouteToRagAsync(projectId);
            toast.info(`Cloning into a RAG-first project — the bell will link to it when ready (job #${job.id}).`, 6000);
            void useJobsStore.getState().refreshAfterLocalChange();
        } catch (err) {
            toast.error(errorText(err));
        } finally {
            setStarting(false);
        }
    };

    // Hidden: not a documents-only project (no passages) or the preview
    // failed. A project that already holds flow-generated pairs keeps the
    // card so the same pairs can be trained again (another base model).
    if (hidden || !preview) return null;
    const notThisCase = preview.passages === 0 && !finished && !inFlight;
    if (notThisCase) return null;
    const generatedBefore = preview.already_generated > 0 && !finished && !inFlight && !result;

    const plan = preview.plan;
    return (
        <section className="card documents-qa-flow" id="documents-qa-flow" data-testid="documents-qa-flow">
            <h3 className="documents-qa-flow__title">Turn your documents into a Q&A assistant</h3>
            <p className="documents-qa-flow__lead">
                Your {preview.passages} document passages aren't training data yet. This flow writes question→answer pairs
                from them, keeps a separate answer key, splits the pairs, trains with your defaults and runs the lift check.
            </p>

            {inFlight && (
                <p className="documents-qa-flow__running" data-testid="documents-qa-flow-running">
                    Running — {inFlight.progress_message || 'starting…'}
                </p>
            )}

            {finished?.status === 'failed' && (
                <p className="documents-qa-flow__failed" data-testid="documents-qa-flow-failed">
                    The last attempt failed: {finished.error || 'see the bell for details'}.
                </p>
            )}

            {result && (
                <div className="documents-qa-flow__result" data-testid="documents-qa-flow-result">
                    <strong>Done.</strong>{' '}
                    {result.training_pairs} question→answer pairs from {result.passages_used} passages ({result.backend}),{' '}
                    {result.answer_key_rows} answer-key rows
                    {result.split
                        ? `, split ${result.split.train ?? '?'} / ${result.split.val ?? '?'} / ${result.split.test ?? '?'} (train / val / test)`
                        : ''}
                    {result.experiment_id != null
                        ? `. Training run #${result.experiment_id} started — the lift check follows on the Eval tab.`
                        : result.passages_gate?.status === 'retrieval_ready'
                            ? '.'
                            : result.stopped_reason
                                ? `. Stopped before training: ${result.stopped_reason}`
                                : '.'}
                    {result.warnings.length > 0 && (
                        <span className="documents-qa-flow__warnings"> {result.warnings.join(' ')}</span>
                    )}
                    <span className="documents-qa-flow__review">
                        {' '}The generated rows are in the Synthetic tab's review queue (source "documents_qa_flow") if you want to reject any.
                    </span>
                </div>
            )}

            {result?.passages_gate && result.passages_gate.status !== 'not_run' && (
                <div
                    className={`documents-qa-flow__gate documents-qa-flow__gate--${result.passages_gate.status}`}
                    data-testid="documents-qa-flow-gate"
                    data-status={result.passages_gate.status}
                >
                    <strong>Before training</strong>
                    {result.passages_gate.status === 'retrieval_ready' && (
                        <>
                            {' '}— the untouched base model answering from your passages already got{' '}
                            <strong>{result.passages_gate.correct} of {result.passages_gate.judged}</strong>{' '}
                            {result.passages_gate.split === 'test' ? 'test examples' : 'rows'} fully right
                            {' '}({result.passages_gate.partial} partly, {result.passages_gate.wrong} wrong; judge score{' '}
                            {result.passages_gate.score != null ? result.passages_gate.score.toFixed(2) : '—'}, by {result.passages_gate.judge || 'the judge model'}
                            {result.passages_gate.retrieval ? `; retrieval top-${result.passages_gate.retrieval.k}${result.passages_gate.retrieval.reranker ? ' + reranker' : ''}, the sweep's best` : ''}).
                            {' '}Training on generated pairs is unlikely to beat that, so the flow stopped here. Reroute to RAG to serve
                            the base model + your passages, or train anyway to compare on the same rows.
                            {result.experiment_id == null && (
                                <div className="documents-qa-flow__actions">
                                    <button type="button" className="btn btn-primary" onClick={() => void reroute()} disabled={starting} data-testid="documents-qa-flow-reroute">
                                        {starting ? 'Starting…' : 'Reroute to RAG (base model + your passages)'}
                                    </button>
                                    <button type="button" className="btn btn-secondary" onClick={() => void start(true, true)} disabled={starting} data-testid="documents-qa-flow-train-anyway">
                                        {starting ? 'Starting…' : 'Train anyway'}
                                    </button>
                                </div>
                            )}
                        </>
                    )}
                    {result.passages_gate.status === 'retrieval_promising' && (
                        <>
                            {' '}— the base model answering from your passages
                            {result.passages_gate.retrieval ? ` (top-${result.passages_gate.retrieval.k}${result.passages_gate.retrieval.reranker ? ' + reranker' : ''}, the sweep's best)` : ''}
                            {' '}already got <strong>{result.passages_gate.correct} of {result.passages_gate.judged}</strong> fully right but{' '}
                            {result.passages_gate.wrong} wrong ({result.passages_gate.partial} partly; judge score{' '}
                            {result.passages_gate.score != null ? result.passages_gate.score.toFixed(2) : '—'}). Not an assistant on its own yet, so
                            training went ahead — the lift check will judge the fine-tune against this retrieval on the same rows.
                        </>
                    )}
                    {result.passages_gate.status === 'retrieval_weak' && (
                        <>
                            {' '}— retrieval alone got {result.passages_gate.correct} of {result.passages_gate.judged} fully right
                            {' '}(judge score {result.passages_gate.score != null ? result.passages_gate.score.toFixed(2) : '—'}), not enough on its own — training went ahead.
                        </>
                    )}
                    {result.passages_gate.status === 'not_judged' && (
                        <> — the retrieval check ran without a judge model, so it could not call whether retrieval already answers; training went ahead.</>
                    )}
                    {result.passages_gate.status === 'error' && (
                        <> — the retrieval check could not run ({result.passages_gate.reason}); training went ahead.</>
                    )}
                </div>
            )}

            {!inFlight && !preview.eligible && (
                <p className="documents-qa-flow__blockers" data-testid="documents-qa-flow-blockers">{preview.blockers.join(' ')}</p>
            )}

            {generatedBefore && (
                <p className="documents-qa-flow__result" data-testid="documents-qa-flow-generated-before">
                    {preview.already_generated} question→answer pairs were generated from these documents earlier. Train them
                    again on the current base model to compare models on the same data, or run the flow again to write new ones.
                </p>
            )}

            {!inFlight && preview.eligible && !result && !generatedBefore && (
                <>
                    <ul className="documents-qa-flow__plan" data-testid="documents-qa-flow-plan">
                        <li>About {plan.estimated_training_pairs} training pairs from {plan.max_passages} passages ({plan.pairs_per_passage} per passage), written by {preview.backend}.</li>
                        <li>{plan.estimated_answer_key_rows} different questions kept as the answer key.</li>
                        <li>Split into train / validation / test examples.</li>
                        <li>Before training: the base model with and without your passages on the test examples, judged — if retrieval already answers well, the flow stops there and offers reroute to RAG instead.</li>
                        <li>Otherwise a training run with your current defaults.</li>
                        <li>The automatic lift check scores it against the base model.</li>
                    </ul>
                    <label className="documents-qa-flow__option">
                        <input type="checkbox" checked={reviewFirst} onChange={(e) => setReviewFirst(e.target.checked)} data-testid="documents-qa-flow-review-first" />
                        {' '}Stop after generating so I can review the rows before training
                    </label>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => void start()}
                        disabled={starting}
                        data-testid="documents-qa-flow-start"
                    >
                        {starting ? 'Starting…' : reviewFirst ? 'Generate the Q&A pairs' : 'Build a Q&A assistant from these documents'}
                    </button>
                    <p className="documents-qa-flow__hint">
                        {plan.llm_calls} calls to the generation model; a few minutes with a local model. Everything it writes is grounded in your passages, but it is still a model writing questions — spot-check them.
                    </p>
                </>
            )}
            {!inFlight && (result || generatedBefore) && (
                <div className="documents-qa-flow__actions">
                    {(result?.training_pairs ?? preview.already_generated) > 0 && (
                        <button type="button" className="btn btn-primary" onClick={() => void start(true)} disabled={starting} data-testid="documents-qa-flow-retrain">
                            {starting ? 'Starting…' : 'Train again on the current base model'}
                        </button>
                    )}
                    {preview.eligible && (
                        <button type="button" className="btn btn-secondary" onClick={() => void start()} disabled={starting} data-testid="documents-qa-flow-rerun">
                            {starting ? 'Starting…' : 'Run again (write new pairs)'}
                        </button>
                    )}
                </div>
            )}
        </section>
    );
}
