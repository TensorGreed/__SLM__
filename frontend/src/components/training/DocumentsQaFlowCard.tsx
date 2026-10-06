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

    const start = async () => {
        if (!preview) return;
        setStarting(true);
        try {
            const job = await startDocumentsQaFlow(projectId, {
                maxPassages: preview.plan.max_passages,
                pairsPerPassage: preview.plan.pairs_per_passage,
                train: !reviewFirst,
            });
            toast.info(`Building the Q&A assistant — track it in the bell (job #${job.id}).`, 5000);
            void useJobsStore.getState().refreshAfterLocalChange();
        } catch (err) {
            toast.error(errorText(err));
        } finally {
            setStarting(false);
        }
    };

    // Hidden: not a documents-only project (Q&A rows exist and no flow ran)
    // or the preview failed. A finished flow keeps the card so the result
    // stays visible.
    if (hidden || !preview) return null;
    const notThisCase = preview.passages === 0 && !finished && !inFlight;
    if (notThisCase) return null;
    if (preview.already_generated > 0 && !finished && !inFlight) return null;

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

            {!inFlight && !preview.eligible && (
                <p className="documents-qa-flow__blockers" data-testid="documents-qa-flow-blockers">{preview.blockers.join(' ')}</p>
            )}

            {!inFlight && preview.eligible && !result && (
                <>
                    <ul className="documents-qa-flow__plan" data-testid="documents-qa-flow-plan">
                        <li>About {plan.estimated_training_pairs} training pairs from {plan.max_passages} passages ({plan.pairs_per_passage} per passage), written by {preview.backend}.</li>
                        <li>{plan.estimated_answer_key_rows} different questions kept as the answer key.</li>
                        <li>Split into train / validation / test examples, then a training run with your current defaults.</li>
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
            {!inFlight && preview.eligible && result && (
                <button type="button" className="btn btn-secondary" onClick={() => void start()} disabled={starting} data-testid="documents-qa-flow-rerun">
                    {starting ? 'Starting…' : 'Run again'}
                </button>
            )}
        </section>
    );
}
