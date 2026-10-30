/**
 * Compare page evidence block (Epic W1): is B really different from A?
 * Sits at the top of Results → Compare. Backend: ``compare_row_evidence_service``
 * pairs the two runs' per-row scores by row key and reports rows better /
 * worse / same, the 95% range of the mean per-row change and a verdict.
 *
 * When the two evals were scored against a different answer key, judge or
 * scoring mode there is no verdict: the block says what differs and offers
 * to re-score the older run (the lift check) on the current answer key and
 * judge. Wording follows the Eval tab's lift card (``liftEvidence``).
 */

import { useState } from 'react';

import {
    rescoreExperiment,
    type CompareProvenance,
    type CompareRowEvidence as RowEvidence,
} from '../../api/experimentCompare';
import { toast } from '../../stores/toastStore';
import { useJobsStore } from '../../stores/jobsStore';
import { metricUnit } from './liftEvidence';


const ANSWER_KEY_NAMES: Record<string, string> = {
    test: 'test examples',
    gold_test: 'answer key (final)',
    gold_dev: 'answer key (practice)',
    validation: 'validation split',
    benchmark: 'external benchmark',
};


/** "test examples, split v5" / "… split v3 (inferred)" / "… file 1a2b3c4d". */
export function answerKeyText(prov: CompareProvenance | null | undefined): string {
    const key = prov?.answer_key;
    if (!key) return 'unknown';
    const kind = ANSWER_KEY_NAMES[key.dataset_type ?? ''] ?? key.dataset_type ?? 'unknown';
    if (key.version !== null && key.version !== undefined) {
        return `${kind}, split v${key.version}${key.version_source === 'inferred' ? ' (inferred from eval time)' : ''}`;
    }
    if (key.sha256) return `${kind}, file ${key.sha256.slice(0, 8)}`;
    return kind;
}


/** Every judge that scored the eval, or "none". */
export function judgeText(prov: CompareProvenance | null | undefined): string {
    const judge = prov?.judge;
    if (!judge) return 'none';
    const parts: string[] = [];
    if (judge.answer) parts.push(judge.answer);
    if (judge.rubric) {
        parts.push(judge.rubric_sha
            ? `plan rubric ${judge.rubric_sha.slice(0, 8)} by ${judge.rubric}`
            : `plan rubric by ${judge.rubric}`);
    }
    if (judge.code_example) parts.push(`code examples by ${judge.code_example}`);
    return parts.length ? parts.join('; ') : 'none';
}


function pts(value: number, digits: number): string {
    const scaled = value * 100;
    const rounded = Number(scaled.toFixed(digits));
    const sign = rounded > 0 ? '+' : rounded < 0 ? '−' : '';
    return `${sign}${Math.abs(rounded).toFixed(digits)}`;
}


const VERDICT_PHRASE: Record<string, string> = {
    better: 'B is better than A',
    worse: 'B is worse than A',
    within_noise: 'B is within noise of A',
    too_few_rows: 'Too few rows to tell',
};

const VERDICT_LABEL: Record<string, string> = {
    better: 'Gain on these rows',
    worse: 'Drop on these rows',
    within_noise: 'Within noise',
    too_few_rows: 'Too few rows to tell',
    not_comparable: 'Not comparable',
};


/** "B is within noise of A: 2 better, 7 worse, 69 same of 78 rows;
 *  change −6.4 pts (95% CI −14 to +1)". */
export function evidenceHeadline(ev: Pick<RowEvidence, 'verdict' | 'n' | 'better' | 'worse' | 'same' | 'mean_diff' | 'ci_low' | 'ci_high'>): string {
    const counts = `${ev.better} better, ${ev.worse} worse, ${ev.same} same of ${ev.n} row${ev.n === 1 ? '' : 's'}`;
    const phrase = VERDICT_PHRASE[ev.verdict] ?? ev.verdict;
    if (ev.mean_diff === null) return `${phrase}: ${counts}.`;
    const change = `change ${pts(ev.mean_diff, 1)} pts`;
    if (ev.ci_low === null || ev.ci_high === null) return `${phrase}: ${counts}; ${change}.`;
    return `${phrase}: ${counts}; ${change} (95% CI ${pts(ev.ci_low, 0)} to ${pts(ev.ci_high, 0)})`;
}


function explanation(ev: RowEvidence): string | null {
    if (ev.verdict === 'within_noise') {
        return "The 95% range includes zero, so this change could be chance. Don't read it as a gain or a loss.";
    }
    if (ev.verdict === 'better' || ev.verdict === 'worse') {
        return 'The 95% range excludes zero: a change beyond row-to-row noise. Each side is one training run, and another seed can move it.';
    }
    if (ev.verdict === 'too_few_rows' && !ev.reason) {
        return `Only ${ev.n} paired row${ev.n === 1 ? '' : 's'}: not enough to tell a gain from noise.`;
    }
    return null;
}


/** "Run A (run #11) · eval #91 · test examples, split v5 · judge: X". */
export function provenanceText(prov: CompareProvenance | null | undefined): string {
    if (!prov) return 'not recorded';
    const scoring = prov.scoring?.scoring_mode ? ` · scoring: ${prov.scoring.scoring_mode}` : '';
    if (prov.eval_result_id === null) return `${prov.experiment_name} (run #${prov.experiment_id}) · no eval yet`;
    return `${prov.experiment_name} (run #${prov.experiment_id}) · eval #${prov.eval_result_id} · `
        + `${answerKeyText(prov)} · judge: ${judgeText(prov)}${scoring}`;
}


export default function CompareRowEvidence({
    projectId,
    evidence,
    a,
    b,
}: {
    projectId: number;
    evidence: RowEvidence;
    a: CompareProvenance | null | undefined;
    b: CompareProvenance | null | undefined;
}) {
    const [starting, setStarting] = useState(false);
    const [started, setStarted] = useState(false);
    const ev = evidence;
    const headline = ev.verdict === 'not_comparable'
        ? (ev.comparability?.certain === false
            ? 'A and B may have been scored differently, so there is no verdict'
            : 'A and B were scored differently, so there is no verdict')
        : ev.reason
            ? `No row-by-row verdict: ${ev.message ?? ev.reason}`
            : evidenceHeadline(ev);
    const why = explanation(ev);
    const offer = ev.rescore;

    const rescore = async () => {
        if (!offer) return;
        const ok = window.confirm(
            `Re-score run #${offer.experiment_id}? This re-evaluates it on the GPU (the base model's result is reused when cached).`,
        );
        if (!ok) return;
        setStarting(true);
        try {
            await rescoreExperiment(projectId, offer.experiment_id);
            setStarted(true);
            toast.info(`Re-scoring run #${offer.experiment_id} — the bell will tell you when it is ready.`, 4000);
            void useJobsStore.getState().refreshAfterLocalChange();
        } catch (err: any) {
            toast.error(err?.response?.data?.detail || err?.message || 'Re-score failed to start.');
        } finally {
            setStarting(false);
        }
    };

    const others = ev.metrics.slice(1);
    const sides = [{ label: 'A', prov: a }, { label: 'B', prov: b }];

    return (
        <section
            className={`eval-compare__evidence eval-compare__evidence--${ev.verdict}`}
            data-testid="eval-compare-evidence"
            data-verdict={ev.verdict}
        >
            <p className="eval-compare__evidence-headline">
                <span className="eval-compare__evidence-label">{VERDICT_LABEL[ev.verdict] ?? ev.verdict}</span>
                <span data-testid="eval-compare-evidence-headline">{headline}</span>
            </p>
            {ev.metric_id && !ev.reason && (
                <p className="eval-compare__evidence-detail" data-testid="eval-compare-evidence-pairing">
                    Per-row {metricUnit(ev.metric_id)}, paired by row: {ev.n} row{ev.n === 1 ? '' : 's'} paired
                    {ev.only_in_a > 0 && `, ${ev.only_in_a} only in A`}
                    {ev.only_in_b > 0 && `, ${ev.only_in_b} only in B`}
                    {ev.unscored > 0 && `, ${ev.unscored} not scored on one side`}.
                </p>
            )}
            {why && <p className="eval-compare__evidence-detail">{why}</p>}
            {ev.verdict === 'not_comparable' && ev.comparability && (
                <ul className="eval-compare__evidence-diffs" data-testid="eval-compare-evidence-diffs">
                    {ev.comparability.differences.map((d) => (
                        <li key={d.message}>{d.message}</li>
                    ))}
                </ul>
            )}
            {ev.verdict === 'not_comparable' && (
                <p className="eval-compare__evidence-detail">
                    The metric deltas below are still shown, but they compare different scoring.
                </p>
            )}
            {ev.comparability?.notes.map((note) => (
                <p key={note} className="eval-compare__evidence-note">{note}</p>
            ))}
            {(ev.comparability?.caveats ?? []).map((caveat) => (
                <p key={caveat} className="eval-compare__evidence-caveat" data-testid="eval-compare-evidence-caveat">{caveat}</p>
            ))}
            {others.length > 0 && (
                <ul className="eval-compare__evidence-metrics" data-testid="eval-compare-evidence-metrics">
                    {others.map((m) => (
                        <li key={m.metric_id} data-testid={`eval-compare-evidence-metric-${m.metric_id}`}>
                            <code>{m.metric_id}</code> {evidenceHeadline(m)}
                        </li>
                    ))}
                </ul>
            )}
            {offer && (
                <div className="eval-compare__evidence-rescore" data-testid="eval-compare-rescore">
                    <p className="eval-compare__evidence-detail">{offer.instruction}</p>
                    {(offer.available || offer.in_progress) && (
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={rescore}
                            disabled={starting || started || !!offer.in_progress}
                            data-testid="eval-compare-rescore-button"
                        >
                            {offer.in_progress
                                ? 'Re-score running'
                                : started
                                    ? 'Re-score queued'
                                    : starting
                                        ? 'Starting…'
                                        : `Re-score ${offer.experiment_name} (run #${offer.experiment_id}) on the current answer key and judge`}
                        </button>
                    )}
                </div>
            )}
            <dl className="eval-compare__provenance" data-testid="eval-compare-provenance">
                {sides.map(({ label, prov }) => (
                    <div key={label} data-testid={`eval-compare-provenance-${label.toLowerCase()}`}>
                        <dt>{label}</dt>
                        <dd>{provenanceText(prov)}</dd>
                    </div>
                ))}
            </dl>
        </section>
    );
}
