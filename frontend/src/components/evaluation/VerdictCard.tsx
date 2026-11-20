/**
 * VerdictCard — the top of Results › Verdict (UI simplification pass, phase 1): one card that
 * answers "is the fine-tune better than the base model?" for the selected run.
 *
 * - Score of record per evaluated set (newest result per set): what it was scored as, the value
 *   with a Wilson 95% interval when it is a per-row pass / fail share, rows scored, and a
 *   "Partial" warning when only some of the set's rows were scored.
 * - Base vs fine-tuned per set, from the existing lift summary (one call per set): base → fine-tuned
 *   on the headline metric, rows better / worse, the noise verdict with McNemar, and "Open side by
 *   side" (Compare with A = the baseline run, B = this run, on that set).
 * - One line for the pass/fail rules: PASS / FAIL / NOT MEASURED (never folded together) and the
 *   safety risk, with a button that opens the details (rule report, scorecard, runtime) below.
 *
 * Every number carries its provenance: run #id, the set label (version, prompt format), result id.
 * Read-only: it shows what is stored and changes nothing.
 */

import { useEffect, useState } from 'react';

import { evalDatasetLabel, evalResultDatasetLabel } from '../../api/evalSets';
import { fetchSftLiftSummary, type SftLiftSummary } from '../../api/sftLift';
import { resultsPath } from '../../utils/workspaceRoutes';
import { isUnproven, metricDisplayName, shortEvidenceText } from './liftEvidence';
import { sliceBreakdownOf } from './SliceBreakdownPanel';
import {
    isStringMatchAside,
    liftKeys,
    newestPerDataset,
    partialRowsText,
    pct,
    scoreOfRecord,
    scoreText,
    unscoredRowsText,
    type VerdictResultLike,
} from './verdictScore';
import './VerdictCard.css';

export type VerdictGateStatus = 'pass' | 'fail' | 'not_measured';

export interface VerdictGate {
    status: VerdictGateStatus;
    /** Measured rules that failed. */
    failed: string[];
    /** Required metrics with no value (not a failure of the model). */
    missing: string[];
    ruleSet?: string | null;
}

interface VerdictCardProps {
    projectId: number;
    experimentId: number | null;
    /** "Run #21 · exp-21" (picked-data provenance). */
    runLabel: string;
    /** The run is a lift-check baseline (the base model). */
    isBaseline?: boolean;
    /** The selected run's results, newest first. */
    results: VerdictResultLike[];
    gate: VerdictGate | null;
    /** The safety scorecard's overall risk, when there is one. */
    safetyRisk?: string | null;
    /** Opens the Verdict details (rule report, scorecard, runtime). */
    onOpenDetails?: () => void;
    /** Where "define slices" points (Failures › Advanced). */
    slicesHref?: string;
}

const GATE_BADGE: Record<VerdictGateStatus, { label: string; className: string }> = {
    pass: { label: 'PASS', className: 'badge-success' },
    fail: { label: 'FAIL', className: 'badge-error' },
    not_measured: { label: 'NOT MEASURED', className: 'badge-warning' },
};

function signed(value: number): string {
    return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;
}

function LiftLine({ projectId, experimentId, liftKey, refreshToken }: { projectId: number; experimentId: number; liftKey: string; refreshToken: unknown }) {
    // The answer for one request (run, set, refresh); a newer request shows "loading" until it lands.
    const token = `${projectId}|${experimentId}|${liftKey}|${String(refreshToken)}`;
    const [loaded, setLoaded] = useState<{ token: string; summary: SftLiftSummary | null; error: boolean } | null>(null);
    const setLabel = liftKey ? evalDatasetLabel(liftKey) : 'the built-in test examples';
    const testKey = liftKey || 'builtin';

    useEffect(() => {
        let alive = true;
        const request = liftKey ? fetchSftLiftSummary(projectId, experimentId, liftKey) : fetchSftLiftSummary(projectId, experimentId);
        request
            .then((res) => { if (alive) setLoaded({ token, summary: res && typeof res === 'object' && res.status ? res : null, error: false }); })
            .catch(() => { if (alive) setLoaded({ token, summary: null, error: true }); });
        return () => { alive = false; };
    }, [projectId, experimentId, liftKey, token]);

    const current = loaded?.token === token ? loaded : null;
    const summary = current?.summary ?? null;
    if (current?.error) {
        return <li className="verdict-card__lift" data-testid={`verdict-lift-${testKey}`}>Base vs fine-tuned on {setLabel}: could not be loaded.</li>;
    }
    if (!summary) {
        return <li className="verdict-card__lift u-meta" data-testid={`verdict-lift-${testKey}-loading`}>Base vs fine-tuned on {setLabel}: loading…</li>;
    }
    if (summary.status !== 'ok' || !Array.isArray(summary.metric_lifts)) {
        return (
            <li className="verdict-card__lift" data-testid={`verdict-lift-${testKey}`} data-status={summary.status}>
                Base vs fine-tuned on {setLabel}: {summary.message || 'no base-model result to compare with yet.'}
            </li>
        );
    }
    const headline = summary.metric_lifts.find((m) => m.is_headline) ?? summary.metric_lifts[0];
    const evidence = headline?.evidence ?? null;
    const tone = !headline || isUnproven(evidence) || headline.direction === 'unchanged'
        ? 'neutral'
        : headline.direction === 'improved' ? 'good' : 'bad';
    const baseId = summary.baseline?.experiment_id;
    const tunedId = summary.trained?.experiment_id ?? experimentId;
    const params = new URLSearchParams();
    if (baseId != null) params.set('a', String(baseId));
    params.set('b', String(tunedId));
    if (liftKey) params.set('set', liftKey);
    const baseModel = summary.baseline?.base_model;
    return (
        <li className="verdict-card__lift" data-testid={`verdict-lift-${testKey}`} data-status="ok">
            <div>
                <strong>Base vs fine-tuned</strong> on {setLabel}
                {summary.cross_variant ? ` (${summary.cross_variant.label})` : ''}
                {headline && (
                    <>
                        {': '}{metricDisplayName(headline.metric_id)}{' '}
                        <code data-testid={`verdict-lift-${testKey}-values`}>
                            {headline.baseline_value.toFixed(3)} → {headline.trained_value.toFixed(3)}
                        </code>{' '}
                        <span className={`verdict-card__tone--${tone}`} data-testid={`verdict-lift-${testKey}-delta`}>
                            ({signed(headline.absolute_delta)})
                        </span>
                    </>
                )}
            </div>
            {evidence && (
                <div className="u-meta" data-testid={`verdict-lift-${testKey}-evidence`} data-verdict={evidence.verdict}>
                    {shortEvidenceText(evidence)}
                </div>
            )}
            <div className="u-meta" data-testid={`verdict-lift-${testKey}-provenance`}>
                Base: run #{baseId ?? '?'}{baseModel ? ` (${baseModel})` : ''}
                {summary.baseline?.eval_result_id != null ? `, result #${summary.baseline.eval_result_id}` : ''}
                {' · '}fine-tuned: run #{tunedId}
                {summary.trained?.eval_result_id != null ? `, result #${summary.trained.eval_result_id}` : ''}
                {baseId != null && (
                    <>
                        {' · '}
                        <a href={resultsPath(projectId, 'compare', params.toString())} data-testid={`verdict-lift-${testKey}-side-by-side`}>
                            Open side by side
                        </a>
                    </>
                )}
            </div>
        </li>
    );
}

export default function VerdictCard({
    projectId,
    experimentId,
    runLabel,
    isBaseline = false,
    results,
    gate,
    safetyRisk = null,
    onOpenDetails,
    slicesHref,
}: VerdictCardProps) {
    const shown = newestPerDataset(results).filter((r) => !isStringMatchAside(r, results));
    const asideCount = newestPerDataset(results).length - shown.length;
    const keys = experimentId != null && !isBaseline ? liftKeys(results) : [];

    return (
        <section className="card verdict-card" data-testid="verdict-card">
            <h3 className="verdict-card__title">Is the fine-tune better than the base model?</h3>
            <p className="u-meta u-m0" data-testid="verdict-card-run">
                {experimentId != null ? runLabel : 'No run selected'}
                {isBaseline ? ' · this run is the base model (a lift-check baseline): pick a fine-tuned run for base vs fine-tuned' : ''}
            </p>

            {experimentId != null && shown.length === 0 && (
                <p className="u-hint" data-testid="verdict-card-not-scored">
                    Not scored yet. Run an evaluation below; the automatic base-vs-fine-tuned check (when it is on) scores the test
                    examples after training.
                </p>
            )}

            {shown.length > 0 && (
                <ul className="verdict-card__scores" data-testid="verdict-card-scores">
                    {shown.map((r) => {
                        const score = scoreOfRecord(r);
                        const partial = partialRowsText(r);
                        const unscored = unscoredRowsText(r);
                        const noSlices = !sliceBreakdownOf(r.metrics ?? null);
                        return (
                            <li key={r.id} className="verdict-card__score" data-testid={`verdict-score-${r.id}`}>
                                <div className="verdict-card__score-head">
                                    <span className="verdict-card__score-value" data-testid={`verdict-score-value-${r.id}`}>
                                        {scoreText(score)}
                                    </span>
                                    <span className="verdict-card__score-name">
                                        {score.label}
                                        {score.pass !== null && score.scored !== null ? ` · ${score.pass} of ${score.scored} rows pass` : score.scored !== null ? ` · ${score.scored} rows scored` : ''}
                                    </span>
                                </div>
                                <div className="u-meta" data-testid={`verdict-score-provenance-${r.id}`}>
                                    {runLabel} · {evalResultDatasetLabel(r as Parameters<typeof evalResultDatasetLabel>[0])} · result #{r.id} · scored as {score.scoredAs}
                                </div>
                                {score.ci === null && score.value !== null && (
                                    <div className="u-meta" data-testid={`verdict-score-no-interval-${r.id}`}>
                                        No interval: this score is not a share of passing rows.
                                    </div>
                                )}
                                {score.other && (
                                    <div className="u-meta" data-testid={`verdict-score-other-${r.id}`}>
                                        Also stored: {score.other.label} {pct(score.other.value)}
                                    </div>
                                )}
                                {partial && (
                                    <div className="verdict-card__warn" role="status" data-testid={`verdict-partial-${r.id}`}>{partial}</div>
                                )}
                                {unscored && (
                                    <div className="verdict-card__warn" role="status" data-testid={`verdict-unscored-${r.id}`}>{unscored}</div>
                                )}
                                {noSlices && (
                                    <div className="u-meta" data-testid={`verdict-slices-hint-${r.id}`}>
                                        No slices: tag rows with <code>slice</code>, or{' '}
                                        {slicesHref ? <a href={slicesHref}>define project slices</a> : 'define project slices'}{' '}
                                        (Failures › Advanced › Pass/fail rules). Slices apply to evals run after they exist.
                                    </div>
                                )}
                            </li>
                        );
                    })}
                </ul>
            )}
            {asideCount > 0 && (
                <p className="u-meta" data-testid="verdict-card-string-match-aside">
                    {asideCount} plain string-match result{asideCount === 1 ? '' : 's'} on this run{' '}
                    {asideCount === 1 ? 'is' : 'are'} listed under Scores › Other stored metrics (string match).
                </p>
            )}

            {keys.length > 0 && experimentId != null && (
                <ul className="verdict-card__lifts" data-testid="verdict-card-lifts">
                    {keys.map((key) => (
                        <LiftLine key={key || 'builtin'} projectId={projectId} experimentId={experimentId} liftKey={key} refreshToken={results.length} />
                    ))}
                </ul>
            )}

            {experimentId != null && (
                <div className="verdict-card__rules" data-testid="verdict-rules">
                    <span>Pass/fail rules{gate?.ruleSet ? ` (${gate.ruleSet})` : ''}:</span>
                    {gate ? (
                        <>
                            <span className={`badge ${GATE_BADGE[gate.status].className}`} data-testid="verdict-rules-status">
                                {GATE_BADGE[gate.status].label}
                            </span>
                            {gate.failed.length > 0 && (
                                <span className="verdict-card__tone--bad" data-testid="verdict-rules-failed">
                                    failed: {gate.failed.join(', ')}
                                </span>
                            )}
                            {gate.missing.length > 0 && (
                                <span className="u-meta" data-testid="verdict-rules-missing">
                                    not measured: {gate.missing.join(', ')} (no value on this run, not a failure of the model)
                                </span>
                            )}
                        </>
                    ) : (
                        <span className="u-meta" data-testid="verdict-rules-status">not checked yet</span>
                    )}
                    {safetyRisk && (
                        <span className="u-meta" data-testid="verdict-safety">· safety risk: {safetyRisk}</span>
                    )}
                    {onOpenDetails && (
                        <button type="button" className="btn btn-link" onClick={onOpenDetails} data-testid="verdict-rules-details">
                            details
                        </button>
                    )}
                </div>
            )}
        </section>
    );
}
