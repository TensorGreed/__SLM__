/**
 * VerdictCard — the top of Results › Verdict (UI simplification pass, phase 1): one card that
 * answers "is the fine-tune better than the base model?" for the selected run.
 *
 * - Score of record per evaluated set (newest result per set): what it was scored as, the value
 *   with a Wilson 95% interval when it is a per-row pass / fail share, rows scored, and a
 *   "Partial" warning when only some of the set's rows were scored.
 * - Base vs fine-tuned per set, from the existing lift summary (one call per set): base → fine-tuned
 *   on the score-of-record metric (the lift row whose metric is the score's key), rows better /
 *   worse, the noise verdict with McNemar, rules that regressed / still fail, and "Open side by
 *   side" (Compare with A = the baseline run, B = this run, pinned to the two cited result ids).
 *   If the summary cites a fine-tuned result another model answered (an explicit model path; servers
 *   before phase 2 could), the line says so instead of passing it off as the fine-tune's lift. When the
 *   two cited results ran with different rows or max new tokens (read from each result's stored
 *   details), the line says "not like for like": the headline subtracts the two results' aggregates.
 * - One line for the pass/fail rules: PASS / FAIL / NOT MEASURED (never folded together) and the
 *   safety risk, with a button that opens the details (rule report, scorecard, runtime) below.
 *
 * Every number carries its provenance: run #id, the set label (version, prompt format), result id.
 * Read-only: it shows what is stored and changes nothing.
 */

import { useEffect, useState } from 'react';

import api from '../../api/client';
import { evalDatasetLabel, evalResultDatasetLabel, evaluatedModelText, isEvalSetName as isEvalSetLabel } from '../../api/evalSets';
import { fetchSftLiftSummary, type SftLiftSummary } from '../../api/sftLift';
import { resultsPath } from '../../utils/workspaceRoutes';
import { evaluatedOnText, isUnproven, metricDisplayName, shortEvidenceText, type GateEvaluatedOn } from './liftEvidence';
import { sliceBreakdownOf } from './SliceBreakdownPanel';
import { resultRunSettings, type ResultRunSettings } from './evalRunPayload';
import {
    excludedRowsText,
    isOtherModelResult,
    isStringMatchAside,
    liftKeys,
    partialRowsText,
    pct,
    scoreOfRecord,
    scoreText,
    unscoredRowsText,
    verdictResultsPerSet,
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
    /** A seed group's rules are measured on one seed run (the gate report's ``evaluated_on``). */
    evaluatedOn?: GateEvaluatedOn | null;
}

/** Same signature as ``fetchSftLiftSummary``; the panel passes a shared one so Details reuses the answer. */
export type LiftFetcher = (projectId: number, experimentId?: number | null, datasetName?: string | null) => Promise<SftLiftSummary>;

/** A run's stored eval results (``GET /evaluation/results/{run}``): reads the base side's rows / tokens. */
export type ResultsFetcher = (projectId: number, experimentId: number) => Promise<VerdictResultLike[]>;

async function fetchRunResults(projectId: number, experimentId: number): Promise<VerdictResultLike[]> {
    const res = await api.get<VerdictResultLike[]>(`/projects/${projectId}/evaluation/results/${experimentId}`);
    return Array.isArray(res?.data) ? res.data : [];
}

/** "100 rows at 256 max new tokens" (the parts that are stored). */
function settingsText(s: ResultRunSettings): string {
    const rows = s.rowsAnswered != null ? `${s.rowsAnswered.toLocaleString('en-US')} rows` : 'rows not stored';
    return s.tokens != null ? `${rows} at ${s.tokens} max new tokens` : rows;
}

/** The two cited results ran with different rows answered or max new tokens (only where both are stored). */
export function settingsDiffer(base: ResultRunSettings, tuned: ResultRunSettings): boolean {
    const rows = base.rowsAnswered != null && tuned.rowsAnswered != null && base.rowsAnswered !== tuned.rowsAnswered;
    const tokens = base.tokens != null && tuned.tokens != null && base.tokens !== tuned.tokens;
    return rows || tokens;
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
    /** The gate report could not be loaded (its error text). */
    gateError?: string | null;
    /** The safety scorecard's overall risk, when there is one. */
    safetyRisk?: string | null;
    /** Opens the Verdict details (rule report, scorecard, runtime). */
    onOpenDetails?: () => void;
    /** Where "define slices" points (Failures › Advanced). */
    slicesHref?: string;
    /** Lift-summary fetch (defaults to ``fetchSftLiftSummary``). */
    fetchLift?: LiftFetcher;
    /** A run's results (defaults to ``GET /evaluation/results/{run}``): the base side's stored rows / tokens. */
    fetchResults?: ResultsFetcher;
}

const GATE_BADGE: Record<VerdictGateStatus, { label: string; className: string }> = {
    pass: { label: 'PASS', className: 'badge-success' },
    fail: { label: 'FAIL', className: 'badge-error' },
    not_measured: { label: 'NOT MEASURED', className: 'badge-warning' },
};

function signed(value: number): string {
    return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;
}

function plural(n: number, one: string, many: string): string {
    return `${n} ${n === 1 ? one : many}`;
}

interface LiftLineProps {
    projectId: number;
    experimentId: number;
    liftKey: string;
    refreshToken: unknown;
    /** The run's own results on this key's set(s): the score of record names the lift row. */
    results: VerdictResultLike[];
    /** Every result of the run (to tell when the summary cites one another model answered). */
    allResults?: VerdictResultLike[];
    fetchLift: LiftFetcher;
    fetchResults: ResultsFetcher;
    onOpenDetails?: () => void;
}

function LiftLine({ projectId, experimentId, liftKey, refreshToken, results, allResults = [], fetchLift, fetchResults, onOpenDetails }: LiftLineProps) {
    // The answer for one request (run, set, refresh); a newer request shows "loading" until it lands.
    const token = `${projectId}|${experimentId}|${liftKey}|${String(refreshToken)}`;
    const [loaded, setLoaded] = useState<{ token: string; summary: SftLiftSummary | null; error: boolean } | null>(null);
    const setLabel = liftKey ? evalDatasetLabel(liftKey) : 'the built-in test examples';
    const testKey = liftKey || 'builtin';

    useEffect(() => {
        let alive = true;
        const request = liftKey ? fetchLift(projectId, experimentId, liftKey) : fetchLift(projectId, experimentId);
        request
            .then((res) => { if (alive) setLoaded({ token, summary: res && typeof res === 'object' && res.status ? res : null, error: false }); })
            .catch(() => { if (alive) setLoaded({ token, summary: null, error: true }); });
        return () => { alive = false; };
    }, [projectId, experimentId, liftKey, token, fetchLift]);

    const current = loaded?.token === token ? loaded : null;
    const summary = current?.summary ?? null;

    // The base side's stored result (filed under the baseline run): its rows / max new tokens.
    const baseRunId = summary?.status === 'ok' ? summary.baseline?.experiment_id ?? null : null;
    const baseResultId = summary?.status === 'ok' ? summary.baseline?.eval_result_id ?? null : null;
    const baseToken = `${projectId}|${baseRunId}|${baseResultId}`;
    const [baseLoaded, setBaseLoaded] = useState<{ token: string; settings: ResultRunSettings | null } | null>(null);
    useEffect(() => {
        if (baseRunId == null || baseResultId == null) return undefined;
        let alive = true;
        fetchResults(projectId, baseRunId)
            .then((rows) => {
                const found = Array.isArray(rows) ? rows.find((r) => r.id === baseResultId) : undefined;
                if (alive) setBaseLoaded({ token: baseToken, settings: found ? resultRunSettings(found) : null });
            })
            .catch(() => { if (alive) setBaseLoaded({ token: baseToken, settings: null }); });
        return () => { alive = false; };
    }, [projectId, baseRunId, baseResultId, baseToken, fetchResults]);
    const baseSettings = baseLoaded?.token === baseToken ? baseLoaded.settings : null;
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
    // The fine-tuned result the summary cites (else this set's newest own result) names the score of record;
    // the lift row is the one on that metric, so the line and the score above it agree (strict included).
    const cited = results.find((r) => r.id === summary.trained?.eval_result_id) ?? results[0] ?? null;
    // The server picks the fine-tuned side from results the run's own weights answered; an older server could
    // cite one another model answered (an explicit model path). Never present that as the fine-tune's lift silently.
    const citedAny = allResults.find((r) => r.id === summary.trained?.eval_result_id) ?? null;
    const citedOther = citedAny && isOtherModelResult(citedAny) ? evaluatedModelText(citedAny.metrics ?? null) || 'another model' : '';
    const score = cited ? scoreOfRecord(cited) : null;
    const scoreRow = score?.key ? summary.metric_lifts.find((m) => m.metric_id === score.key) : undefined;
    const row = scoreRow ?? summary.metric_lifts.find((m) => m.is_headline) ?? summary.metric_lifts[0];
    const rowName = row && scoreRow && score ? score.label : row ? metricDisplayName(row.metric_id) : '';
    const evidence = row?.evidence ?? null;
    const tone = citedOther || !row || isUnproven(evidence) || row.direction === 'unchanged'
        ? 'neutral'
        : row.direction === 'improved' ? 'good' : 'bad';
    const baseId = summary.baseline?.experiment_id;
    const tunedId = summary.trained?.experiment_id ?? experimentId;
    const params = new URLSearchParams();
    if (baseId != null) params.set('a', String(baseId));
    params.set('b', String(tunedId));
    if (liftKey) params.set('set', liftKey);
    // Pin Compare to the two results this line cites, so it cannot open a different base result.
    if (summary.baseline?.eval_result_id != null) params.set('eval_a', String(summary.baseline.eval_result_id));
    if (summary.trained?.eval_result_id != null) params.set('eval_b', String(summary.trained.eval_result_id));
    if (summary.cross_variant) params.set('cross', '1');
    const baseModel = summary.baseline?.base_model;
    const tunedSettings = citedAny ? resultRunSettings(citedAny) : null;
    const notLikeForLike = baseSettings && tunedSettings && settingsDiffer(baseSettings, tunedSettings);
    const gates = Array.isArray(summary.gate_status) ? summary.gate_status : [];
    const regressed = gates.filter((g) => g.status === 'regressed').length;
    const stillFailing = gates.filter((g) => g.status === 'still_failing').length;
    return (
        <li className="verdict-card__lift" data-testid={`verdict-lift-${testKey}`} data-status="ok" data-metric={row?.metric_id}>
            <div>
                <strong>Base vs fine-tuned</strong> on {setLabel}
                {summary.cross_variant ? ` (${summary.cross_variant.label})` : ''}
                {row && (
                    <>
                        {': '}{rowName}{' '}
                        <code data-testid={`verdict-lift-${testKey}-values`}>
                            {row.baseline_value.toFixed(3)} → {row.trained_value.toFixed(3)}
                        </code>{' '}
                        <span className={`verdict-card__tone--${tone}`} data-testid={`verdict-lift-${testKey}-delta`}>
                            ({signed(row.absolute_delta)})
                        </span>
                    </>
                )}
            </div>
            {citedOther && (
                <div className="verdict-card__warn" role="status" data-testid={`verdict-lift-${testKey}-other-model`}>
                    Not this run&apos;s own weights: the fine-tuned side is result #{summary.trained?.eval_result_id}, {citedOther}.
                    This is not the fine-tune&apos;s lift; score the set with the run&apos;s own model (Score this run › Fine-tuned).
                </div>
            )}
            {notLikeForLike && baseSettings && tunedSettings && (
                <div className="verdict-card__warn" role="status" data-testid={`verdict-lift-${testKey}-settings-differ`}>
                    Not like for like: the base result #{summary.baseline?.eval_result_id} scored {settingsText(baseSettings)}, the fine-tuned
                    result #{summary.trained?.eval_result_id} {settingsText(tunedSettings)}. The values above are each result&apos;s own
                    average; score again with Score this run › Both (or match the base&apos;s rows and tokens) for a like-for-like line.
                </div>
            )}
            {row && score?.key && !scoreRow && (
                <div className="u-meta" data-testid={`verdict-lift-${testKey}-other-metric`}>
                    {score.label} is not stored on both results, so this line uses {metricDisplayName(row.metric_id)}.
                </div>
            )}
            {evidence && (
                <div className="u-meta" data-testid={`verdict-lift-${testKey}-evidence`} data-verdict={evidence.verdict}>
                    {shortEvidenceText(evidence)}
                </div>
            )}
            {(regressed > 0 || stillFailing > 0) && (
                <div className="verdict-card__tone--bad" data-testid={`verdict-lift-${testKey}-gates`}>
                    {[
                        regressed > 0 ? `${plural(regressed, 'rule', 'rules')} regressed` : '',
                        stillFailing > 0 ? `${plural(stillFailing, 'rule', 'rules')} still failing` : '',
                    ].filter(Boolean).join(' / ')}
                    {onOpenDetails && (
                        <>
                            {' · '}
                            <button type="button" className="btn btn-link" onClick={onOpenDetails} data-testid={`verdict-lift-${testKey}-gates-details`}>
                                details
                            </button>
                        </>
                    )}
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
    gateError = null,
    safetyRisk = null,
    onOpenDetails,
    slicesHref,
    fetchLift = fetchSftLiftSummary,
    fetchResults = fetchRunResults,
}: VerdictCardProps) {
    const perSet = verdictResultsPerSet(results);
    const shown = perSet.filter((r) => !isStringMatchAside(r, perSet));
    const asideCount = perSet.length - shown.length;
    // A lift line only for a set whose score is shown (never one whose score moved aside).
    const keys = experimentId != null && !isBaseline ? liftKeys(shown) : [];
    const ownShown = shown.filter((r) => !isOtherModelResult(r));

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
                        const excluded = excludedRowsText(r, score);
                        const unscored = unscoredRowsText(r);
                        const noSlices = !sliceBreakdownOf(r.metrics ?? null);
                        const otherModel = isOtherModelResult(r) ? evaluatedModelText(r.metrics ?? null) : '';
                        return (
                            <li key={r.id} className="verdict-card__score" data-testid={`verdict-score-${r.id}`} data-metric={score.key ?? ''}>
                                <div className="verdict-card__score-head">
                                    <span className="verdict-card__score-value" data-testid={`verdict-score-value-${r.id}`}>
                                        {scoreText(score)}
                                    </span>
                                    <span className="verdict-card__score-name" data-testid={`verdict-score-name-${r.id}`}>
                                        {score.label}
                                        {score.pass !== null && score.scored !== null ? ` · ${score.pass} of ${score.scored} rows pass` : score.scored !== null ? ` · ${score.scored} rows scored` : ''}
                                    </span>
                                </div>
                                {otherModel && (
                                    <div className="verdict-card__warn" role="status" data-testid={`verdict-other-model-${r.id}`}>
                                        Not this run's own weights: {otherModel}. Filed under this run; no base vs fine-tuned line for it.
                                    </div>
                                )}
                                <div className="u-meta" data-testid={`verdict-score-provenance-${r.id}`}>
                                    {runLabel} · {evalResultDatasetLabel(r as Parameters<typeof evalResultDatasetLabel>[0])} · result #{r.id} · scored as {score.scoredAs}
                                </div>
                                {score.ci === null && score.value !== null && (
                                    <div className="u-meta" data-testid={`verdict-score-no-interval-${r.id}`}>
                                        {score.share && score.pass === null
                                            ? 'No interval: the passing-row count is not stored.'
                                            : 'No interval: this score is not a share of passing rows.'}
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
                                {excluded && (
                                    <div className="u-meta" data-testid={`verdict-excluded-${r.id}`}>{excluded}</div>
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
                        <LiftLine
                            key={key || 'builtin'}
                            projectId={projectId}
                            experimentId={experimentId}
                            liftKey={key}
                            refreshToken={results.length}
                            results={ownShown.filter((r) => (key ? r.dataset_name === key : !isEvalSetLabel(r.dataset_name)))}
                            allResults={results}
                            fetchLift={fetchLift}
                            fetchResults={fetchResults}
                            onOpenDetails={onOpenDetails}
                        />
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
                            {gate.evaluatedOn && (
                                <span className="u-meta" data-testid="verdict-rules-evaluated-on">
                                    {evaluatedOnText(gate.evaluatedOn)}
                                </span>
                            )}
                        </>
                    ) : gateError ? (
                        <span className="u-meta" data-testid="verdict-rules-status">could not be loaded</span>
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
