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
import { retrievalLabel } from '../evaluation/retrievalLabel';
import GenerationModelPicker from '../shared/GenerationModelPicker';

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
    /** Train on the served retrieval prompt (the passages in front of the question). */
    const [withPassages, setWithPassages] = useState(false);
    /** The generation model: null = the first reachable local one, or a
     *  `cloud:<provider>:<model>` pin using the project's saved key. */
    const [backend, setBackend] = useState<string | null>(null);
    /** Targets = a teacher model's answers to the same served passages. */
    const [teacherAnswers, setTeacherAnswers] = useState(false);
    const [teacherBackend, setTeacherBackend] = useState<string | null>(null);
    const [useBatch, setUseBatch] = useState(false);
    const [concurrency, setConcurrency] = useState(1);
    /** Keep the current test examples and grow them to resolve `decisionGap`. */
    const [keepTest, setKeepTest] = useState(false);
    const [decisionGap, setDecisionGap] = useState(0.15);

    const jobs = useJobsStore((state) => state.jobs);
    const flowJobs = (jobs || []).filter((job) => job.kind === 'documents_qa_flow' && job.project_id === projectId);
    const inFlight = flowJobs.find((job) => job.status === 'queued' || job.status === 'running') ?? null;
    const finished = flowJobs
        .filter((job) => job.status === 'succeeded' || job.status === 'failed')
        .sort((a, b) => b.id - a.id)[0] ?? null;
    const result = finished?.status === 'succeeded' ? ((finished.result || null) as DocumentsQaFlowResult | null) : null;

    const load = useCallback(async () => {
        try {
            const next = await fetchDocumentsQaFlowPreview(projectId, backend);
            setPreview(next);
        } catch {
            setHidden(true);
        }
    }, [projectId, backend]);

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
                trainWithPassages: withPassages || teacherAnswers,
                answersFromTeacher: teacherAnswers,
                teacherBackend: teacherAnswers ? (teacherBackend ?? backend) : null,
                useBatch,
                concurrency,
                keepTest,
                ...(keepTest ? { decisionGap } : {}),
                // "Train anyway" answers a gate that already ran on these rows;
                // re-running it repeats the whole retrieval sweep (~an hour).
                ...(trainIfRetrievalReady ? { passagesCheck: false } : {}),
                ...(reuseExisting ? {} : { backend }),
            });
            toast.info(
                reuseExisting
                    ? `Training the generated pairs again${withPassages ? ', reading your passages' : ''} — track it in the bell (job #${job.id}).`
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
    const options = (
        <div className="documents-qa-flow__options" data-testid="documents-qa-flow-options">
            <label className="documents-qa-flow__option" title="Each training pair's prompt becomes what serving sends: the passages your retrieval finds for the question, then the question. The run is then scored with the passages against the base model + the same passages.">
                <input type="checkbox" checked={withPassages || teacherAnswers} disabled={teacherAnswers} onChange={(e) => setWithPassages(e.target.checked)} data-testid="documents-qa-flow-with-passages" />
                {' '}Train the model to answer from your passages
            </label>
            <label className="documents-qa-flow__option" title="A teacher model answers every training question from exactly the passages your retrieval serves for it; its answer becomes the target, so training matches serving. When the teacher says the passages don't cover it, that answer is kept. Training answers the judge grades wrong against the pair's reference are dropped.">
                <input type="checkbox" checked={teacherAnswers} onChange={(e) => setTeacherAnswers(e.target.checked)} data-testid="documents-qa-flow-teacher" />
                {' '}Use a teacher model's answers to the same passages as the targets
            </label>
            {teacherAnswers && (
                <GenerationModelPicker
                    projectId={projectId}
                    value={teacherBackend ?? backend}
                    onChange={setTeacherBackend}
                    disabled={starting}
                    testIdPrefix="documents-qa-flow-teacher"
                />
            )}
            <label className="documents-qa-flow__option" title="Anthropic models only: every call goes as one Message Batch — half the price, can take up to a few hours.">
                <input type="checkbox" checked={useBatch} onChange={(e) => setUseBatch(e.target.checked)} data-testid="documents-qa-flow-batch" />
                {' '}Batch (half price, slower)
            </label>
            <label className="documents-qa-flow__option" title="Calls the model answers at the same time. A local Ollama model needs OLLAMA_NUM_PARALLEL at least this high; more slots use more GPU memory.">
                Calls at once{' '}
                <select value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))} aria-label="Calls at once" data-testid="documents-qa-flow-concurrency">
                    <option value={1}>1</option>
                    <option value={2}>2</option>
                    <option value={4}>4</option>
                    <option value={8}>8</option>
                </select>
            </label>
            <label className="documents-qa-flow__option" title="The current test examples stay exactly as they are (earlier runs stay comparable); new answer-key questions spread over your documents are added until the test set can resolve the gap. Training and validation questions that match a test question are dropped.">
                <input type="checkbox" checked={keepTest} onChange={(e) => setKeepTest(e.target.checked)} data-testid="documents-qa-flow-keep-test" />
                {' '}Keep the current test examples and grow them to resolve a gap of{' '}
                <select value={decisionGap} onChange={(e) => setDecisionGap(Number(e.target.value))} aria-label="Decision gap" data-testid="documents-qa-flow-gap">
                    <option value={0.15}>0.15 (45 rows)</option>
                    <option value={0.1}>0.10 (100 rows)</option>
                    <option value={0.08}>0.08 (157 rows)</option>
                </select>
            </label>
        </div>
    );
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
                    {result.eval_rejected && Object.values(result.eval_rejected).some((n) => n > 0) ? (
                        <span data-testid="documents-qa-flow-eval-rejected" title="Dropped before they became answer-key rows: a reader with the passage in front of it could not have been scored fairly on them.">
                            {' '}({Object.values(result.eval_rejected).reduce((a, b) => a + b, 0)} dropped: a reader couldn't quote the answer from the passage)
                        </span>
                    ) : null}
                    {result.split
                        ? `, split ${result.split.train ?? '?'} / ${result.split.val ?? '?'} / ${result.split.test ?? '?'} (train / val / test)`
                        : ''}
                    {result.experiment_id != null
                        ? result.train_with_passages
                            ? `. Training run #${result.experiment_id} started, trained to answer from your passages — its check (fine-tuned + passages vs base + the same passages) follows on Results → Verdict.`
                            : `. Training run #${result.experiment_id} started — the lift check follows on Results → Verdict.`
                        : result.passages_gate?.status === 'retrieval_ready'
                            ? '.'
                            : result.stopped_reason
                                ? `. Stopped before training: ${result.stopped_reason}`
                                : '.'}
                    {result.teacher && (
                        <span data-testid="documents-qa-flow-teacher-result">
                            {' '}Teacher ({result.teacher.teacher}): {result.teacher.usable} usable answers
                            {result.teacher.counts.train ? ` — ${result.teacher.counts.train.abstained} training answers say the passages don't cover it` : ''}
                            {result.teacher.filter.applied && result.teacher.counts.train ? `, ${result.teacher.counts.train.filter_dropped} dropped as wrong by ${result.teacher.filter.judge}` : ''}.
                        </span>
                    )}
                    {result.warnings.length > 0 && (
                        <span className="documents-qa-flow__warnings"> {result.warnings.join(' ')}</span>
                    )}
                    <span className="documents-qa-flow__review">
                        {' '}The generated rows are in the review queue on Data → Generate (source "documents_qa_flow") if you want to reject any.
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
                    {result.passages_gate.sweep && (
                        <span data-testid="documents-qa-flow-gate-sweep">
                            {' '}(retrieval chosen on {result.passages_gate.sweep.rows ?? '?'} validation rows
                            {result.passages_gate.sweep.chosen_score != null ? `, ${result.passages_gate.sweep.chosen_score.toFixed(2)} there` : ''}; scored once on the test examples)
                        </span>
                    )}
                    {result.passages_gate.status === 'retrieval_ready' && (
                        <>
                            {' '}— the untouched base model answering from your passages already got{' '}
                            <strong>{result.passages_gate.correct} of {result.passages_gate.judged}</strong>{' '}
                            {result.passages_gate.split === 'test' ? 'test examples' : 'rows'} fully right
                            {' '}({result.passages_gate.partial} partly, {result.passages_gate.wrong} wrong; judge score{' '}
                            {result.passages_gate.score != null ? result.passages_gate.score.toFixed(2) : '—'}, by {result.passages_gate.judge || 'the judge model'}
                            {result.passages_gate.retrieval ? `; retrieval ${retrievalLabel(result.passages_gate.retrieval, false)}, the sweep's best` : ''}).
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
                            {result.passages_gate.retrieval ? ` (${retrievalLabel(result.passages_gate.retrieval, false)}, the sweep's best)` : ''}
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

            {!inFlight && !result && (
                <GenerationModelPicker
                    projectId={projectId}
                    value={backend}
                    onChange={setBackend}
                    disabled={starting}
                    testIdPrefix="documents-qa-flow"
                />
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
                        <li>
                            Split into train / validation / test examples
                            {plan.sized_for ? (
                                <span data-testid="documents-qa-flow-sizing">
                                    {' '}— sized so a {plan.sized_for.judge_gap} judge-score gap clears noise: {plan.sized_for.target_test_rows} test examples
                                    {plan.sized_for.capped_by_available
                                        ? ` (your ${plan.sized_for.passages} passages can't reach ${plan.sized_for.passages_needed}, so ${Math.round(plan.sized_for.test_ratio * 100)}% of the pairs become test examples${plan.sized_for.reaches_target ? '' : ` — still only about ${plan.sized_for.expected_test_rows}; add documents for a firmer verdict`})`
                                        : ` from ${plan.sized_for.passages} passages`}.
                                </span>
                            ) : null}
                        </li>
                        <li>Before training: the base model with and without your passages on the test examples, judged — if retrieval already answers well, the flow stops there and offers reroute to RAG instead.</li>
                        <li>Otherwise a training run with your current defaults.</li>
                        <li>The automatic lift check scores it against the base model.</li>
                    </ul>
                    {options}
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
            {!inFlight && (result || generatedBefore) && (result?.training_pairs ?? preview.already_generated) > 0 && (
                options
            )}
            {!inFlight && (result || generatedBefore) && (
                <div className="documents-qa-flow__actions">
                    {(result?.training_pairs ?? preview.already_generated) > 0 && (
                        <button type="button" className="btn btn-primary" onClick={() => void start(true)} disabled={starting} data-testid="documents-qa-flow-retrain">
                            {starting ? 'Starting…' : teacherAnswers ? "Train again on the teacher's answers to your passages" : withPassages ? 'Train again on the current base model, reading your passages' : 'Train again on the current base model'}
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
