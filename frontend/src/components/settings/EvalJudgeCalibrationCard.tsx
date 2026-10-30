/**
 * "Calibrate eval judge" (Judge calibration V4): label judged eval answers blind, then see how far each
 * eval judge agrees with you, and whether it is too strict or too lenient.
 *
 * Lives under Settings → Judge model: calibration is about the judge (labels are pooled across every
 * eval it judged, keyed by the answer's content), not about one run. Same patterns as the review-queue
 * "Calibrate judges" card (V2): a blind labeller, a reference-model run, per-judge cards per tier, the
 * shared label-source setting. Nothing is gated by these numbers.
 */

import { useCallback, useEffect, useState } from 'react';

import { toast } from '../../stores/toastStore';
import { budgetExceededMessage, budgetWarningMessage } from '../../api/llmSpend';
import GenerationModelPicker from '../shared/GenerationModelPicker';
import {
    getCalibrationSettings,
    putCalibrationSettings,
    type CalibrationLabelSource,
    type CalibrationRate,
} from '../../api/judgeCalibration';
import {
    SCALE_LABELS,
    getEvalJudgeProgress,
    getEvalJudgeReport,
    getEvalJudgeSample,
    postEvalJudgeLabel,
    postEvalJudgeSkip,
    startEvalJudgeReference,
    type BlindAnswer,
    type EvalJudgeEntry,
    type EvalJudgeProgress,
    type EvalJudgeReport,
    type EvalJudgeScale,
    type EvalJudgeTierReport,
} from '../../api/evalJudgeCalibration';
import EvalJudgeTrustBadge from '../evaluation/EvalJudgeTrustBadge';
import './EvalJudgeCalibrationCard.css';

const SOURCE_OPTIONS: { value: CalibrationLabelSource; label: string }[] = [
    { value: 'human', label: 'My labels' },
    { value: 'reference_model', label: 'Reference model labels' },
    { value: 'both', label: 'Both, side by side' },
];

const SCALE_NAMES: Record<EvalJudgeScale, string> = {
    verdict: 'Answer judge (correct / partial / wrong)',
    rubric: 'Plan rubric judge (pass / fail)',
};

const LABEL_HINT: Record<EvalJudgeScale, string> = {
    verdict: 'Correct: states the reference answer\'s key facts and nothing that contradicts them. Partial: some key facts, or a claim the reference does not make. Wrong: contradicts it, answers something else, or is empty.',
    rubric: 'Pass: the plan meets the project\'s rubric. Fail: it misses a critical point or is not adequate overall.',
};

function errorText(err: any, fallback: string): string {
    const detail = err?.response?.data;
    return budgetExceededMessage(err)
        ?? (typeof detail?.message === 'string' ? detail.message : typeof detail?.detail === 'string' ? detail.detail : err?.message)
        ?? fallback;
}

function pct(v: number | null | undefined): string {
    return typeof v === 'number' ? `${Math.round(v * 100)}%` : 'n/a';
}

function rateText(r: CalibrationRate | undefined): string {
    if (!r || r.value == null) return 'n/a';
    const ci = r.low != null && r.high != null ? ` (95% range ${pct(r.low)}-${pct(r.high)})` : '';
    return `${pct(r.value)}${ci}`;
}

export default function EvalJudgeCalibrationCard({ projectId }: { projectId: number }) {
    // Opened by a deep link (the bell, the eval summary's "Calibrate the judge").
    const [open, setOpen] = useState(() => typeof window !== 'undefined' && window.location.hash === '#eval-judge-calibration');
    const [source, setSource] = useState<CalibrationLabelSource>('human');
    const [scale, setScale] = useState<EvalJudgeScale>('verdict');
    const [judge, setJudge] = useState<string | null>(null);
    const [progress, setProgress] = useState<EvalJudgeProgress | null>(null);
    const [report, setReport] = useState<EvalJudgeReport | null>(null);
    const [sampleSize, setSampleSize] = useState('20');
    const [rows, setRows] = useState<BlindAnswer[] | null>(null);
    const [index, setIndex] = useState(0);
    const [labelled, setLabelled] = useState(0);
    const [reason, setReason] = useState('');
    const [busy, setBusy] = useState(false);
    const [refPin, setRefPin] = useState<string | null>(null);
    const [refN, setRefN] = useState('20');

    const refresh = useCallback(async (nextScale: EvalJudgeScale) => {
        try {
            const [p, r] = await Promise.all([getEvalJudgeProgress(projectId), getEvalJudgeReport(projectId, nextScale)]);
            setProgress(p);
            setReport(r);
            const judges = p?.scales?.[nextScale]?.judges ?? [];
            setJudge((cur) => (cur && judges.some((j) => j.judge === cur) ? cur : judges[0]?.judge ?? null));
        } catch (err) {
            toast.error(errorText(err, 'Could not load the eval judge calibration.'), 6000);
        }
    }, [projectId]);

    useEffect(() => {
        if (!open) return;
        let cancelled = false;
        getCalibrationSettings(projectId)
            .then((s) => {
                if (!cancelled) setSource(s.label_source);
            })
            .catch(() => undefined);
        void refresh(scale);
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open, projectId]);

    const changeSource = async (next: CalibrationLabelSource) => {
        setSource(next);
        try {
            await putCalibrationSettings(projectId, next);
        } catch (err) {
            toast.error(errorText(err, 'Could not save the label source.'), 6000);
        }
        await refresh(scale);
    };

    const changeScale = async (next: EvalJudgeScale) => {
        setScale(next);
        setJudge(null);
        setRows(null);
        await refresh(next);
    };

    const startLabelling = async () => {
        setBusy(true);
        try {
            const n = Math.min(100, Math.max(1, Math.floor(Number(sampleSize) || 20)));
            const sample = await getEvalJudgeSample(projectId, scale, judge, n);
            if (sample.length === 0) toast.info('Every judged answer already has your label.', 5000);
            setRows(sample.length ? sample : null);
            setIndex(0);
            setLabelled(0);
            setReason('');
        } catch (err) {
            toast.error(errorText(err, 'Could not load answers to label.'), 6000);
        } finally {
            setBusy(false);
        }
    };

    const advance = () => {
        setReason('');
        const next = index + 1;
        setIndex(next);
        if (rows && next >= rows.length) {
            setRows(null);
            void refresh(scale);
        }
    };

    const label = async (value: string) => {
        if (!rows) return;
        setBusy(true);
        try {
            await postEvalJudgeLabel(projectId, scale, rows[index].content_key, value, judge, reason);
            setLabelled((n) => n + 1);
            advance();
        } catch (err) {
            toast.error(errorText(err, 'Could not save the label.'), 6000);
        } finally {
            setBusy(false);
        }
    };

    const skip = async () => {
        if (!rows) return;
        setBusy(true);
        try {
            await postEvalJudgeSkip(projectId, scale, rows[index].content_key, reason);
            advance();
        } catch (err) {
            toast.error(errorText(err, 'Could not record the skip.'), 6000);
        } finally {
            setBusy(false);
        }
    };

    const startReference = async () => {
        setBusy(true);
        try {
            const job = await startEvalJudgeReference(projectId, {
                backend: refPin, scale, judge, n: Math.min(100, Math.max(1, Math.floor(Number(refN) || 20))),
            });
            const budgetMsg = budgetWarningMessage(job);
            if (budgetMsg) toast.warning(budgetMsg, 8000);
            toast.info(`Reference labelling started with ${refPin ?? 'a local model'} (job #${job.id}). Open the bell for progress.`, 6000);
        } catch (err) {
            toast.error(errorText(err, 'Could not start reference labelling.'), 8000);
        } finally {
            setBusy(false);
        }
    };

    const scaleProgress = progress?.scales?.[scale];
    const judges = scaleProgress?.judges ?? [];
    const hasRubric = (progress?.scales?.rubric?.population ?? 0) > 0;
    const current = rows ? rows[index] : null;
    const human = scaleProgress?.tiers?.human;
    const reference = scaleProgress?.tiers?.reference_model;

    return (
        <div className="eval-judge-calibration" id="eval-judge-calibration" data-testid="eval-judge-calibration-card">
            <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setOpen((v) => !v)}
                data-testid="eval-judge-calibration-toggle"
            >
                {open ? 'Hide eval judge calibration' : 'Calibrate eval judge'}
            </button>
            {open && (
                <div className="eval-judge-calibration__body">
                    <p className="eval-judge-calibration__hint">
                        Is the judge too strict, or too lenient? Grade some answers it judged yourself, without seeing its
                        verdict, then compare. A label covers the same answer in every eval. Nothing is blocked by these numbers.
                    </p>
                    <fieldset className="eval-judge-calibration__group" data-testid="eval-judge-calibration-source">
                        <legend>Compare the judge with</legend>
                        {SOURCE_OPTIONS.map((opt) => (
                            <label key={opt.value} className="eval-judge-calibration__option">
                                <input
                                    type="radio"
                                    name={`eval-judge-calibration-source-${projectId}`}
                                    value={opt.value}
                                    checked={source === opt.value}
                                    onChange={() => void changeSource(opt.value)}
                                    data-testid={`eval-judge-calibration-source-${opt.value}`}
                                />
                                {' '}{opt.label}
                            </label>
                        ))}
                    </fieldset>
                    {hasRubric && (
                        <fieldset className="eval-judge-calibration__group" data-testid="eval-judge-calibration-scale">
                            <legend>Which judge</legend>
                            {(['verdict', 'rubric'] as EvalJudgeScale[]).map((s) => (
                                <label key={s} className="eval-judge-calibration__option">
                                    <input
                                        type="radio"
                                        name={`eval-judge-calibration-scale-${projectId}`}
                                        value={s}
                                        checked={scale === s}
                                        onChange={() => void changeScale(s)}
                                        data-testid={`eval-judge-calibration-scale-${s}`}
                                    />
                                    {' '}{SCALE_NAMES[s]}
                                </label>
                            ))}
                        </fieldset>
                    )}
                    {scaleProgress && (
                        <p className="eval-judge-calibration__hint" data-testid="eval-judge-calibration-progress">
                            {scaleProgress.population} judged answer{scaleProgress.population === 1 ? '' : 's'} stored.
                            {' '}Your labels: {human?.labelled ?? 0}{human?.skipped ? `, ${human.skipped} skipped` : ''}.
                            {' '}Reference model labels: {reference?.labelled ?? 0}{reference?.skipped ? `, ${reference.skipped} skipped` : ''}.
                            {scaleProgress.population === 0 ? ' Run an eval with a judge first; older evals did not keep their answers.' : ''}
                        </p>
                    )}
                    {judges.length > 1 && (
                        <label className="eval-judge-calibration__field">
                            Sample answers judged by{' '}
                            <select
                                value={judge ?? ''}
                                onChange={(e) => setJudge(e.target.value || null)}
                                aria-label="Judge to sample"
                                data-testid="eval-judge-calibration-judge"
                            >
                                {judges.map((j) => (
                                    <option key={j.judge} value={j.judge}>{j.judge} ({j.rows} answers)</option>
                                ))}
                            </select>
                        </label>
                    )}

                    {current ? (
                        <div className="eval-judge-calibration__labeller" data-testid="eval-judge-calibration-labeller">
                            <p className="eval-judge-calibration__hint">
                                Answer {index + 1} of {rows?.length ?? 0} · {labelled} labelled so far. {LABEL_HINT[scale]}
                            </p>
                            <dl className="eval-judge-calibration__fields" data-testid="eval-judge-calibration-row">
                                <div className="eval-judge-calibration__field-row"><dt>Question</dt><dd><pre>{current.question}</pre></dd></div>
                                <div className="eval-judge-calibration__field-row"><dt>Reference answer</dt><dd><pre>{current.reference}</pre></dd></div>
                                <div className="eval-judge-calibration__field-row"><dt>Model's answer</dt><dd><pre>{current.prediction}</pre></dd></div>
                                {(current.passages ?? []).length > 0 && (
                                    <div className="eval-judge-calibration__field-row" data-testid="eval-judge-calibration-passages">
                                        <dt>Reference passages{current.passages_truncated ? ' (cut for length, as the reference model gets them)' : ''}</dt>
                                        <dd>
                                            <pre>{(current.passages ?? []).map((p) => `[${p.id}]${p.source ? ` ${p.source}` : ''} ${p.text}`).join('\n\n')}</pre>
                                        </dd>
                                    </div>
                                )}
                            </dl>
                            <label className="eval-judge-calibration__field">
                                Reason (optional){' '}
                                <input
                                    type="text"
                                    value={reason}
                                    maxLength={500}
                                    onChange={(e) => setReason(e.target.value)}
                                    data-testid="eval-judge-calibration-reason"
                                />
                            </label>
                            <div className="eval-judge-calibration__actions">
                                {SCALE_LABELS[scale].map((value) => (
                                    <button
                                        key={value}
                                        type="button"
                                        className={`btn ${value === 'correct' || value === 'pass' ? 'btn-primary' : value === 'partial' ? 'btn-secondary' : 'btn-danger'}`}
                                        disabled={busy}
                                        onClick={() => void label(value)}
                                        data-testid={`eval-judge-calibration-label-${value}`}
                                    >
                                        {value.charAt(0).toUpperCase() + value.slice(1)}
                                    </button>
                                ))}
                                <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => void skip()} data-testid="eval-judge-calibration-skip">
                                    Skip
                                </button>
                                <button
                                    type="button"
                                    className="btn btn-secondary"
                                    disabled={busy}
                                    onClick={() => { setRows(null); void refresh(scale); }}
                                    data-testid="eval-judge-calibration-stop"
                                >
                                    Stop
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="eval-judge-calibration__actions">
                            <label className="eval-judge-calibration__field">
                                Answers to label{' '}
                                <input
                                    type="number"
                                    min={1}
                                    max={100}
                                    value={sampleSize}
                                    onChange={(e) => setSampleSize(e.target.value)}
                                    data-testid="eval-judge-calibration-sample-size"
                                />
                            </label>
                            <button
                                type="button"
                                className="btn btn-primary"
                                onClick={() => void startLabelling()}
                                disabled={busy || !scaleProgress?.population}
                                data-testid="eval-judge-calibration-label-start"
                            >
                                Label answers blind
                            </button>
                        </div>
                    )}

                    <details className="eval-judge-calibration__reference">
                        <summary>Let a reference model label answers</summary>
                        <p className="eval-judge-calibration__hint">
                            A strong model grades a sample blind, with the judge's own instructions. Its labels are kept apart
                            from yours, and a model is never compared with its own labels.
                        </p>
                        <GenerationModelPicker
                            projectId={projectId}
                            value={refPin}
                            onChange={setRefPin}
                            disabled={busy}
                            testIdPrefix="eval-judge-calibration-ref"
                            label="Reference model"
                        />
                        <label className="eval-judge-calibration__field">
                            Answers{' '}
                            <input
                                type="number"
                                min={1}
                                max={100}
                                value={refN}
                                onChange={(e) => setRefN(e.target.value)}
                                data-testid="eval-judge-calibration-ref-n"
                            />
                        </label>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => void startReference()}
                            disabled={busy}
                            data-testid="eval-judge-calibration-ref-start"
                        >
                            Run reference labelling
                        </button>
                    </details>

                    {report && (
                        <div className="eval-judge-calibration__reports">
                            {report.reports.map((t) => <TierReport key={t.tier} tier={t} report={report} />)}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function tierName(t: EvalJudgeTierReport): string {
    if (t.tier === 'human') return 'vs your labels (human)';
    return `vs reference model${t.labellers.length ? ` (${t.labellers.join(', ')})` : ''}`;
}

function TierReport({ tier, report }: { tier: EvalJudgeTierReport; report: EvalJudgeReport }) {
    return (
        <section className="eval-judge-calibration__tier" data-testid={`eval-judge-calibration-report-${tier.tier}`}>
            <h4 className="eval-judge-calibration__tier-title">
                {tier.tier === 'human' ? 'Human labels' : 'Reference model labels'}: {tier.labelled} labelled answer{tier.labelled === 1 ? '' : 's'}
                {tier.skipped ? `, ${tier.skipped} skipped` : ''}
            </h4>
            {tier.judges.length === 0 ? (
                <p className="eval-judge-calibration__hint">No judged answer is stored yet.</p>
            ) : (
                <div className="eval-judge-calibration__cards">
                    {tier.judges.map((j) => <JudgeCard key={j.judge} judge={j} tier={tier} report={report} />)}
                </div>
            )}
        </section>
    );
}

function JudgeCard({ judge, tier, report }: { judge: EvalJudgeEntry; tier: EvalJudgeTierReport; report: EvalJudgeReport }) {
    const pos = judge.scale === 'rubric' ? 'pass' : 'correct';
    const neg = judge.scale === 'rubric' ? 'fail' : 'wrong';
    const trust = report.trust?.[judge.judge];
    const testId = `eval-judge-calibration-judge-${tier.tier}-${judge.judge}`;
    return (
        <div className="eval-judge-calibration__judge" data-testid={testId}>
            <div className="eval-judge-calibration__judge-head">
                <code>{judge.judge}</code> <span className="eval-judge-calibration__tier-name">{tierName(tier)}</span>
            </div>
            {report.reports.length === 1 && trust && <EvalJudgeTrustBadge trust={trust} testId={`${testId}-trust`} />}
            <dl className="eval-judge-calibration__metrics">
                <div><dt>Agrees on {pos} vs not</dt><dd>{rateText(judge.strict_agreement)}</dd></div>
                <div><dt>Exact agreement (every verdict)</dt><dd>{rateText(judge.exact_agreement)}</dd></div>
                <div><dt>Too strict (says {judge.scale === 'rubric' ? 'fail' : 'partial / wrong'}, label {pos})</dt><dd>{rateText(judge.too_strict)}</dd></div>
                <div><dt>Too lenient (says {pos}, label {judge.scale === 'rubric' ? 'fail' : 'partial / wrong'})</dt><dd>{rateText(judge.too_lenient)}</dd></div>
                <div><dt>Its {pos} verdicts are right</dt><dd>{rateText(judge.pass_precision)}</dd></div>
                <div><dt>Finds the {pos} answers</dt><dd>{rateText(judge.pass_recall)}</dd></div>
                <div><dt>Catches the {neg} answers</dt><dd>{rateText(judge.wrong_recall)}</dd></div>
                <div>
                    <dt>Agreement beyond chance ({judge.kappa_kind === 'linear_weighted' ? 'weighted kappa' : 'kappa'})</dt>
                    <dd>{judge.kappa == null ? 'n/a' : judge.kappa.toFixed(2)}</dd>
                </div>
                <div>
                    <dt>Labelled answers it judged</dt>
                    <dd>
                        {judge.n} ({Object.entries(judge.by_label).map(([k, v]) => `${v} ${k}`).join(', ')}; weighted as {judge.n_eff})
                    </dd>
                </div>
            </dl>
            {judge.insufficient && (
                <p className="eval-judge-calibration__note" data-testid={`${testId}-insufficient`}>
                    Not enough labels to trust these numbers yet: {judge.insufficient_reasons.join('; ')}.
                </p>
            )}
            {judge.skip_note && <p className="eval-judge-calibration__note">Many answers were skipped: {judge.skip_note}.</p>}
            {judge.self ? (
                <p className="eval-judge-calibration__note">This judge is the reference model itself, so there is nothing to compare it with here.</p>
            ) : judge.self_rows > 0 ? (
                <p className="eval-judge-calibration__hint">
                    {judge.self_rows} answer{judge.self_rows === 1 ? '' : 's'} this model labelled itself left out.
                </p>
            ) : null}
        </div>
    );
}
