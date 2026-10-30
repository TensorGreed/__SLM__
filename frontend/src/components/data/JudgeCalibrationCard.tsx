/**
 * "Calibrate judges" (Judge calibration V2): blind labels, then how far each review-assist judge can be trusted.
 *
 * Label a sample of rows blind (no verdict shown), or let a strong reference model do it, then see per judge how often it catches a bad row (fail-recall), how
 * often its passes are right (pass-precision), its agreement beyond chance (kappa) and how often it
 * is unsure — each against ONE named label tier (human / reference model), never pooled. Nothing is
 * gated by these numbers.
 */

import { useCallback, useEffect, useState } from 'react';

import { toast } from '../../stores/toastStore';
import { budgetExceededMessage, budgetWarningMessage } from '../../api/llmSpend';
import GenerationModelPicker from '../shared/GenerationModelPicker';
import {
    getCalibrationProgress,
    getCalibrationReport,
    getCalibrationSample,
    getCalibrationSettings,
    postCalibrationLabel,
    postCalibrationSkip,
    putCalibrationSettings,
    startReferenceLabelling,
    type BlindRow,
    type CalibrationJudge,
    type CalibrationLabel,
    type CalibrationLabelSource,
    type CalibrationProgress,
    type CalibrationRate,
    type CalibrationReport,
    type CalibrationTierReport,
} from '../../api/judgeCalibration';
import './JudgeCalibrationCard.css';

const SOURCE_OPTIONS: { value: CalibrationLabelSource; label: string }[] = [
    { value: 'human', label: 'My labels' },
    { value: 'reference_model', label: 'Reference model labels' },
    { value: 'both', label: 'Both, side by side' },
];

function errorText(err: any, fallback: string): string {
    const detail = err?.response?.data;
    return budgetExceededMessage(err)
        ?? (typeof detail?.message === 'string' ? detail.message : typeof detail?.detail === 'string' ? detail.detail : err?.message)
        ?? fallback;
}

function pct(v: number | null | undefined): string {
    return typeof v === 'number' ? `${Math.round(v * 100)}%` : 'n/a';
}

function rateText(r: CalibrationRate): string {
    if (r.value == null) return 'n/a';
    const ci = r.low != null && r.high != null ? ` (95% range ${pct(r.low)}-${pct(r.high)})` : '';
    return `${pct(r.value)}${ci}`;
}

function fieldText(v: unknown): string {
    if (typeof v === 'string') return v;
    try {
        return JSON.stringify(v, null, 2);
    } catch {
        return String(v);
    }
}

interface Props {
    projectId: number;
}

export default function JudgeCalibrationCard({ projectId }: Props) {
    const [open, setOpen] = useState(false);
    const [source, setSource] = useState<CalibrationLabelSource>('human');
    const [progress, setProgress] = useState<CalibrationProgress | null>(null);
    const [report, setReport] = useState<CalibrationReport | null>(null);
    const [sampleSize, setSampleSize] = useState('20');
    const [rows, setRows] = useState<BlindRow[] | null>(null);
    const [index, setIndex] = useState(0);
    const [labelled, setLabelled] = useState(0);
    const [reason, setReason] = useState('');
    const [busy, setBusy] = useState(false);
    const [refPin, setRefPin] = useState<string | null>(null);
    const [refN, setRefN] = useState('20');

    const refresh = useCallback(async (tier?: CalibrationLabelSource) => {
        try {
            const [p, r] = await Promise.all([getCalibrationProgress(projectId), getCalibrationReport(projectId, tier)]);
            setProgress(p);
            setReport(r);
        } catch (err) {
            toast.error(errorText(err, 'Could not load the judge calibration.'), 6000);
        }
    }, [projectId]);

    useEffect(() => {
        if (!open) return;
        let cancelled = false;
        getCalibrationSettings(projectId)
            .then((s) => {
                if (cancelled) return;
                setSource(s.label_source);
                void refresh(s.label_source);
            })
            .catch((err) => toast.error(errorText(err, 'Could not load the judge calibration.'), 6000));
        return () => {
            cancelled = true;
        };
    }, [open, projectId, refresh]);

    const changeSource = async (next: CalibrationLabelSource) => {
        setSource(next);
        try {
            await putCalibrationSettings(projectId, next);
        } catch (err) {
            toast.error(errorText(err, 'Could not save the label source.'), 6000);
        }
        await refresh(next);
    };

    const startLabelling = async () => {
        setBusy(true);
        try {
            const n = Math.min(100, Math.max(1, Math.floor(Number(sampleSize) || 20)));
            const sample = await getCalibrationSample(projectId, n);
            if (sample.length === 0) toast.info('Every row with a judge verdict already has your label.', 5000);
            setRows(sample.length ? sample : null);
            setIndex(0);
            setLabelled(0);
            setReason('');
        } catch (err) {
            toast.error(errorText(err, 'Could not load rows to label.'), 6000);
        } finally {
            setBusy(false);
        }
    };

    const finishIfLast = (next: number) => {
        if (rows && next >= rows.length) {
            setRows(null);
            void refresh(source);
        }
    };

    const label = async (value: CalibrationLabel) => {
        if (!rows) return;
        const row = rows[index];
        setBusy(true);
        try {
            await postCalibrationLabel(projectId, row.row_id, value, reason);
            setLabelled((n) => n + 1);
            setReason('');
            setIndex(index + 1);
            finishIfLast(index + 1);
        } catch (err) {
            toast.error(errorText(err, 'Could not save the label.'), 6000);
        } finally {
            setBusy(false);
        }
    };

    const skip = async () => {
        if (!rows) return;
        const row = rows[index];
        setBusy(true);
        try {
            // Recorded (not a label): skipped rows are often the hard ones, and the report says so.
            await postCalibrationSkip(projectId, row.row_id, reason);
            setReason('');
            setIndex(index + 1);
            finishIfLast(index + 1);
        } catch (err) {
            toast.error(errorText(err, 'Could not record the skip.'), 6000);
        } finally {
            setBusy(false);
        }
    };

    const startReference = async () => {
        setBusy(true);
        try {
            const job = await startReferenceLabelling(projectId, {
                backend: refPin,
                n: Math.min(100, Math.max(1, Math.floor(Number(refN) || 20))),
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

    const current = rows ? rows[index] : null;
    const human = progress?.tiers?.human;
    const reference = progress?.tiers?.reference_model;

    return (
        <div className="judge-calibration" data-testid="judge-calibration-card">
            <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setOpen((v) => !v)}
                data-testid="judge-calibration-toggle"
            >
                {open ? 'Hide judge calibration' : 'Calibrate judges'}
            </button>
            {open && (
                <div className="judge-calibration__body">
                    <p className="judge-calibration__hint">
                        Can you trust a judge? Label some rows yourself without seeing any verdict, then compare each judge with
                        your labels. Review-queue accepts and rejects never count here. Nothing is blocked by these numbers.
                    </p>
                    <fieldset className="judge-calibration__source" data-testid="judge-calibration-source">
                        <legend>Compare judges with</legend>
                        {SOURCE_OPTIONS.map((opt) => (
                            <label key={opt.value} className="judge-calibration__source-option">
                                <input
                                    type="radio"
                                    name={`judge-calibration-source-${projectId}`}
                                    value={opt.value}
                                    checked={source === opt.value}
                                    onChange={() => void changeSource(opt.value)}
                                    data-testid={`judge-calibration-source-${opt.value}`}
                                />
                                {' '}{opt.label}
                            </label>
                        ))}
                    </fieldset>
                    {progress && (
                        <p className="judge-calibration__progress" data-testid="judge-calibration-progress">
                            {progress.population} row{progress.population === 1 ? '' : 's'} have a judge verdict.
                            {' '}Your labels: {human?.labelled ?? 0} (pass {human?.pass ?? 0}, fail {human?.fail ?? 0})
                            {human?.skipped ? `, ${human.skipped} skipped` : ''}
                            {human?.stale ? `, ${human.stale} out of date` : ''}.
                            {' '}Reference model labels: {reference?.labelled ?? 0}
                            {reference?.skipped ? `, ${reference.skipped} skipped` : ''}
                            {reference?.stale ? `, ${reference.stale} out of date` : ''}.
                        </p>
                    )}

                    {current ? (
                        <BlindLabeller
                            row={current}
                            position={index + 1}
                            total={rows?.length ?? 0}
                            labelled={labelled}
                            reason={reason}
                            busy={busy}
                            onReason={setReason}
                            onLabel={(v) => void label(v)}
                            onSkip={() => void skip()}
                            onStop={() => { setRows(null); void refresh(source); }}
                        />
                    ) : (
                        <div className="judge-calibration__actions">
                            <label className="judge-calibration__field">
                                Rows to label{' '}
                                <input
                                    type="number"
                                    min={1}
                                    max={100}
                                    value={sampleSize}
                                    onChange={(e) => setSampleSize(e.target.value)}
                                    data-testid="judge-calibration-sample-size"
                                />
                            </label>
                            <button
                                type="button"
                                className="btn btn-primary"
                                onClick={() => void startLabelling()}
                                disabled={busy}
                                data-testid="judge-calibration-label-start"
                            >
                                Label rows blind
                            </button>
                        </div>
                    )}

                    <details className="judge-calibration__reference">
                        <summary>Let a reference model label rows</summary>
                        <p className="judge-calibration__hint">
                            A strong model labels a sample blind, with the same instructions the judges get. Its labels are kept
                            apart from yours and are never mixed with them.
                        </p>
                        <GenerationModelPicker
                            projectId={projectId}
                            value={refPin}
                            onChange={setRefPin}
                            disabled={busy}
                            testIdPrefix="judge-calibration-ref"
                            label="Reference model"
                        />
                        <label className="judge-calibration__field">
                            Rows{' '}
                            <input
                                type="number"
                                min={1}
                                max={100}
                                value={refN}
                                onChange={(e) => setRefN(e.target.value)}
                                data-testid="judge-calibration-ref-n"
                            />
                        </label>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => void startReference()}
                            disabled={busy}
                            data-testid="judge-calibration-ref-start"
                        >
                            Run reference labelling
                        </button>
                    </details>

                    {report && (
                        <div className="judge-calibration__reports">
                            {report.reports.map((t) => <TierReport key={t.tier} tier={t} />)}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

interface BlindLabellerProps {
    row: BlindRow;
    position: number;
    total: number;
    labelled: number;
    reason: string;
    busy: boolean;
    onReason: (v: string) => void;
    onLabel: (v: CalibrationLabel) => void;
    onSkip: () => void;
    onStop: () => void;
}

/** One row at a time, as the judges read it: no verdict, no review status. */
function BlindLabeller({ row, position, total, labelled, reason, busy, onReason, onLabel, onSkip, onStop }: BlindLabellerProps) {
    const fields = row.fields ? Object.entries(row.fields) : null;
    return (
        <div className="judge-calibration__labeller" data-testid="judge-calibration-labeller">
            <p className="judge-calibration__progress">
                Row {position} of {total} (row #{row.row_id}, {row.kind}) · {labelled} labelled so far.
                {' '}Is this a good training row?
            </p>
            {fields ? (
                <dl className="judge-calibration__fields" data-testid="judge-calibration-row">
                    {fields.map(([k, v]) => (
                        <div key={k} className="judge-calibration__field-row">
                            <dt>{k}</dt>
                            <dd><pre>{fieldText(v)}</pre></dd>
                        </div>
                    ))}
                </dl>
            ) : (
                <pre className="judge-calibration__raw" data-testid="judge-calibration-row">{row.content}</pre>
            )}
            {row.context_truncated && (
                <p className="judge-calibration__hint">Passages marked [truncated] were cut for length, as the judges saw them.</p>
            )}
            <label className="judge-calibration__field">
                Reason (optional){' '}
                <input
                    type="text"
                    value={reason}
                    maxLength={500}
                    onChange={(e) => onReason(e.target.value)}
                    data-testid="judge-calibration-reason"
                />
            </label>
            <div className="judge-calibration__actions">
                <button type="button" className="btn btn-primary" disabled={busy} onClick={() => onLabel('pass')} data-testid="judge-calibration-pass">
                    Pass
                </button>
                <button type="button" className="btn btn-danger" disabled={busy} onClick={() => onLabel('fail')} data-testid="judge-calibration-fail">
                    Fail
                </button>
                <button type="button" className="btn btn-secondary" disabled={busy} onClick={onSkip} data-testid="judge-calibration-skip">
                    Skip
                </button>
                <button type="button" className="btn btn-secondary" disabled={busy} onClick={onStop} data-testid="judge-calibration-stop">
                    Stop
                </button>
            </div>
        </div>
    );
}

const STRATUM_NAMES: Record<string, string> = {
    single_pass: 'rows one judge passed',
    single_fail: 'rows one judge failed',
    single_unsure: 'rows one judge was unsure about',
    agree_pass: 'rows every judge passed',
    agree_fail: 'rows every judge failed',
    disagree: 'rows the judges disagree on',
};

function strataName(s: string): string {
    return STRATUM_NAMES[s] ?? s;
}

function tierName(t: CalibrationTierReport): string {
    if (t.tier === 'human') return 'vs your labels (human)';
    return `vs reference model${t.labellers.length ? ` (${t.labellers.join(', ')})` : ''}`;
}

function TierReport({ tier }: { tier: CalibrationTierReport }) {
    return (
        <section className="judge-calibration__tier" data-testid={`judge-calibration-report-${tier.tier}`}>
            <h4 className="judge-calibration__tier-title">
                {tier.tier === 'human' ? 'Human labels' : 'Reference model labels'}: {tier.labelled} labelled row{tier.labelled === 1 ? '' : 's'}
                {tier.skipped ? `, ${tier.skipped} skipped` : ''}
                {tier.stale ? ` (${tier.stale} out of date, ignored)` : ''}
            </h4>
            {tier.skip_note && (
                <p className="judge-calibration__note" data-testid={`judge-calibration-skip-note-${tier.tier}`}>
                    Many rows were skipped in {(tier.skip_warning_strata ?? []).map(strataName).join(', ')}: {tier.skip_note}.
                </p>
            )}
            {tier.judges.length === 0 ? (
                <p className="judge-calibration__hint">No labelled row has a judge verdict yet.</p>
            ) : (
                <div className="judge-calibration__cards">
                    {tier.judges.map((j) => <JudgeCard key={j.judge} judge={j} tier={tier} />)}
                </div>
            )}
        </section>
    );
}

function JudgeCard({ judge, tier }: { judge: CalibrationJudge; tier: CalibrationTierReport }) {
    const rubrics = Object.keys(judge.rubric_shas).length;
    return (
        <div className="judge-calibration__judge" data-testid={`judge-calibration-judge-${tier.tier}-${judge.judge}`}>
            <div className="judge-calibration__judge-head">
                <code>{judge.judge}</code> <span className="judge-calibration__tier-name">{tierName(tier)}</span>
            </div>
            <dl className="judge-calibration__metrics">
                <div><dt>Catches bad rows (fail-recall)</dt><dd>{rateText(judge.fail_recall)}</dd></div>
                <div><dt>Its passes are right (pass-precision)</dt><dd>{rateText(judge.pass_precision)}</dd></div>
                <div><dt>Agreement beyond chance (kappa)</dt><dd>{judge.kappa == null ? 'n/a' : judge.kappa.toFixed(2)}</dd></div>
                <div><dt>Unsure</dt><dd>{pct(judge.unsure_rate)}</dd></div>
                <div>
                    <dt>Labelled rows it judged</dt>
                    <dd>{judge.n} ({judge.label_fail} fail, {judge.label_pass} pass; weighted as {judge.n_eff})</dd>
                </div>
            </dl>
            {judge.insufficient && (
                <p className="judge-calibration__note" data-testid={`judge-calibration-insufficient-${tier.tier}-${judge.judge}`}>
                    Not enough labels to trust these numbers yet: {judge.insufficient_reasons.join('; ')}.
                </p>
            )}
            {judge.self ? (
                <p className="judge-calibration__note">This judge is the reference model itself, so there is nothing to compare it with here.</p>
            ) : judge.self_rows > 0 ? (
                <p className="judge-calibration__hint">
                    {judge.self_rows} row{judge.self_rows === 1 ? '' : 's'} this model labelled itself left out.
                </p>
            ) : null}
            {judge.rubric_mixed && (
                <p className="judge-calibration__hint">Its verdicts here came from {rubrics} different rubrics.</p>
            )}
            {judge.unparseable > 0 && (
                <p className="judge-calibration__hint">{judge.unparseable} unreadable verdict{judge.unparseable === 1 ? '' : 's'} left out.</p>
            )}
        </div>
    );
}
