/**
 * Results › Verdict card (UI simplification pass, phase 1): score of record per set with its
 * interval and provenance, the partial-scoring warning, base vs fine-tuned per set with "Open side
 * by side", and the rules line that keeps NOT MEASURED apart from FAIL.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import VerdictCard, { type VerdictGate } from './VerdictCard';
import HowItsScoredCard from './HowItsScoredCard';
import {
    PREFERRED_HEADLINE_METRICS,
    SCORE_KEYS,
    excludedRowsText,
    isStringMatchAside,
    liftKeys,
    partialRowsText,
    scoreOfRecord,
    wilsonInterval,
    type VerdictResultLike,
} from './verdictScore';

const RUN = 'Run #21 · exp-21';

const TOOL_CALL: VerdictResultLike = {
    id: 1001,
    dataset_name: 'eval_set:fresh@v1+native',
    eval_type: 'exact_match',
    pass_rate: 0.905,
    metrics: {
        scoring_mode: 'tool_call', total: 600, correct: 560, exact_match: 0.9333,
        tool_call_pass_rate: 0.9333, tool_call_strict_pass_rate: 0.905, prompt_format_version: 2,
        tool_call_fingerprint: 'tool_call/v3+abc+strict_args', evaluated_samples: 600,
    },
    details: { dataset: { rows_available: 600, max_samples: 600, content_sha256: 'abcdef1234', eval_set: { name: 'Fresh', rows: 600 } }, inference: { max_new_tokens: 256, temperature: 0, samples: 600 } },
};

const JSON_RESULT: VerdictResultLike = {
    id: 1002,
    dataset_name: 'test',
    eval_type: 'exact_match',
    pass_rate: 0.6,
    // field_match stores exact_match = rows with every field right, f1 = mean per-field F1.
    metrics: { scoring_mode: 'field_match', total: 7, correct: 3, exact_match: 0.4286, f1: 0.6, field_f1: 0.6, overall_em: 0.4286, json_scoring: { null_as_absent: true } },
    details: { dataset: { rows_available: 40, max_samples: 7 }, inference: { samples: 7 } },
};

const QA: VerdictResultLike = {
    id: 1003,
    dataset_name: 'test',
    eval_type: 'exact_match',
    pass_rate: 0.75,
    // QA stores exact_match and token f1; the backend's lift headline is f1 (``_PREFERRED_HEADLINE_METRICS``).
    metrics: { total: 8, correct: 6, exact_match: 0.75, f1: 0.82 },
    details: { dataset: { rows_available: 8, max_samples: 100 }, inference: { samples: 8 } },
};

const F1: VerdictResultLike = {
    id: 1004,
    dataset_name: 'test',
    eval_type: 'f1',
    pass_rate: 0.62,
    metrics: { total: 8, correct: 3, exact_match: 0, f1: 0.62 },
    details: { dataset: { rows_available: 8 }, inference: { samples: 8 } },
};

const LIFT_OK = {
    status: 'ok',
    project_id: 4,
    dataset_name: 'eval_set:fresh@v1+native',
    baseline: { experiment_id: 214, experiment_name: 'Baseline · Qwen/X', base_model: 'Qwen/X', eval_result_id: 990 },
    trained: { experiment_id: 21, experiment_name: 'exp-21', base_model: 'Qwen/X', eval_result_id: 1001 },
    // As ``_compute_metric_lifts`` writes them: preferred-order metrics first, ``is_headline`` = in
    // ``_PREFERRED_HEADLINE_METRICS`` (the strict rate is not, so it is never marked headline).
    metric_lifts: [
        {
            metric_id: 'tool_call_pass_rate', baseline_value: 0.81, trained_value: 0.9333, absolute_delta: 0.1233,
            relative_delta_pct: 15.2, direction: 'improved', is_headline: true,
            evidence: { n: 600, better: 90, worse: 16, same: 494, mean_diff: 0.1233, ci_low: 0.09, ci_high: 0.15, verdict: 'better', mcnemar_p: 0.0001 },
        },
        {
            metric_id: 'exact_match', baseline_value: 0.81, trained_value: 0.9333, absolute_delta: 0.1233,
            relative_delta_pct: 15.2, direction: 'improved', is_headline: true, evidence: null,
        },
        {
            metric_id: 'tool_call_strict_pass_rate', baseline_value: 0.758, trained_value: 0.905, absolute_delta: 0.147,
            relative_delta_pct: 19.4, direction: 'improved', is_headline: false,
            evidence: { n: 600, better: 111, worse: 23, same: 466, mean_diff: 0.147, ci_low: 0.11, ci_high: 0.18, verdict: 'better', mcnemar_p: 0.0001 },
        },
    ],
    gate_status: [],
};

function renderCard(results: VerdictResultLike[], gate: VerdictGate | null = null, extra: Partial<Parameters<typeof VerdictCard>[0]> = {}) {
    return render(
        <VerdictCard projectId={4} experimentId={21} runLabel={RUN} results={results} gate={gate} slicesHref="/project/4/results?tab=failures&open=failures-tools" {...extra} />,
    );
}

describe('verdictScore helpers', () => {
    it('computes a Wilson 95% interval', () => {
        const [low, high] = wilsonInterval(174, 200)!;
        expect(low).toBeCloseTo(0.816, 3);
        expect(high).toBeCloseTo(0.910, 3);
        expect(wilsonInterval(0, 0)).toBeNull();
    });

    it('picks the strict tool-call rate as the score of record and keeps the default rate beside it', () => {
        const score = scoreOfRecord(TOOL_CALL);
        expect(score.value).toBe(0.905);
        expect(score.pass).toBe(543);
        expect(score.scored).toBe(600);
        expect(score.scoredAs).toBe('tool call (strict)');
        expect(score.other?.value).toBe(0.9333);
    });

    it('gives no interval when the score is not a share of passing rows', () => {
        expect(scoreOfRecord(F1).ci).toBeNull();
        expect(scoreOfRecord(QA).ci).toBeNull();
        expect(scoreOfRecord(TOOL_CALL).ci).not.toBeNull();
    });

    it('flags a capped eval as partial from the rows answered', () => {
        expect(partialRowsText(JSON_RESULT)).toBe('Partial: 7 of 40 rows scored (row limit 7). This score covers those rows only.');
        expect(partialRowsText(QA)).toBeNull();
    });

    it('every per-mode key list follows the backend headline order (strict tool-call rate aside)', () => {
        for (const [shape, keys] of Object.entries(SCORE_KEYS)) {
            const ordered = keys.filter((k) => k !== 'tool_call_strict_pass_rate').map((k) => PREFERRED_HEADLINE_METRICS.indexOf(k));
            expect(ordered.every((i) => i >= 0), shape).toBe(true);
            expect([...ordered].sort((a, b) => a - b), shape).toEqual(ordered);
        }
    });

    it('picks the metric by mode, never by value: EM 0 (base) and EM > 0 (fine-tuned) show the same metric', () => {
        for (const mode of ['span_set', 'finding_set', 'field_match']) {
            const base: VerdictResultLike = { id: 1, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.31, metrics: { scoring_mode: mode, total: 50, correct: 0, exact_match: 0, f1: 0.31 } };
            const tuned: VerdictResultLike = { id: 2, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.12, metrics: { scoring_mode: mode, total: 50, correct: 6, exact_match: 0.12, f1: 0.74 } };
            const a = scoreOfRecord(base);
            const b = scoreOfRecord(tuned);
            expect(a.key, mode).toBe('f1');
            expect(b.key, mode).toBe('f1');
            expect(a.label, mode).toBe(b.label);
            expect(b.value, mode).toBe(0.74);
            expect(b.ci, mode).toBeNull();
        }
        expect(scoreOfRecord({ id: 3, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.7, metrics: { scoring_mode: 'span_set', f1: 0.7, exact_match: 0.1, total: 10, correct: 1 } }).label).toBe('Span F1 (micro)');
        expect(scoreOfRecord({ id: 4, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.7, metrics: { scoring_mode: 'field_match', f1: 0.7, exact_match: 0.1, total: 10, correct: 1 } }).label).toBe('Mean per-field F1');
    });

    it('classification: macro F1 (the lift headline), named as such, not "Exact match"', () => {
        const cls: VerdictResultLike = {
            id: 5, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.8,
            metrics: { accuracy: 0.8, macro_f1: 0.71, exact_match: 0.8, f1: 0.71, total: 50, correct: 40 },
        };
        const score = scoreOfRecord(cls);
        expect(score.key).toBe('f1');
        expect(score.label).toBe('Macro F1');
        expect(score.scoredAs).toBe('classification (label match)');
        expect(score.ci).toBeNull();
    });

    it('structured plan / remediation / multiple options stand on their own pass rate', () => {
        const plan = scoreOfRecord({ id: 6, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.5, metrics: { scoring_mode: 'structured_plan', structured_plan_pass_rate: 0.5, exact_match: 0.5, total: 20, correct: 10 } });
        expect(plan.key).toBe('structured_plan_pass_rate');
        expect(plan.label).toBe('Plan pass rate');
        expect(plan.ci).not.toBeNull();
        const rem = scoreOfRecord({ id: 7, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.25, metrics: { scoring_mode: 'remediation', remediation_pass_rate: 0.25, exact_match: 0.25, total: 20, correct: 5 } });
        expect(rem.label).toBe('Remediation pass rate');
    });

    it('judge_correct leads a QA result that has it, like the lift headline', () => {
        const score = scoreOfRecord({ ...QA, metrics: { ...QA.metrics, judge_correct: 0.66 } });
        expect(score.key).toBe('judge_correct');
        expect(score.ci).toBeNull();
    });

    it('multiple options: verifier-excluded rows are named on their own, never as a row limit', () => {
        const mo: VerdictResultLike = {
            id: 8, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.6,
            metrics: { scoring_mode: 'multi_option', multi_option_pass_rate: 0.6, exact_match: 0.6, total: 35, correct: 21, verifier_error_rows: 5 },
            details: { dataset: { rows_available: 40, max_samples: 100 }, inference: { samples: 40 } },
        };
        expect(partialRowsText(mo)).toBeNull();
        expect(excludedRowsText(mo)).toBe('5 of 40 answered rows are not in this rate: the option verifier could not check them, or they are not multiple-option rows. The rate is over 35 rows.');
    });

    it('puts a plain string-match result aside only next to a task-scored one', () => {
        expect(isStringMatchAside(QA, [QA, TOOL_CALL])).toBe(true);
        expect(isStringMatchAside(QA, [QA])).toBe(false);
        expect(isStringMatchAside(TOOL_CALL, [QA, TOOL_CALL])).toBe(false);
    });

    it('asks for one lift per eval set and one for the built-in sets', () => {
        expect(liftKeys([TOOL_CALL, QA, F1])).toEqual(['eval_set:fresh@v1+native', '']);
    });
});

describe('VerdictCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.get.mockResolvedValue({ data: LIFT_OK });
    });

    it('tool call: strict score of record with interval, rows and full provenance', async () => {
        renderCard([TOOL_CALL]);
        expect(screen.getByTestId('verdict-card-run')).toHaveTextContent(RUN);
        expect(screen.getByTestId('verdict-score-value-1001')).toHaveTextContent('90.5% (95% CI');
        expect(screen.getByTestId('verdict-score-1001')).toHaveTextContent('Strict tool-call pass rate (extra arguments count as wrong) · 543 of 600 rows pass');
        const provenance = screen.getByTestId('verdict-score-provenance-1001');
        expect(provenance).toHaveTextContent(RUN);
        expect(provenance).toHaveTextContent('eval set fresh v1 (native) "Fresh" · 600 rows · content abcdef12 · prompt rendering v2');
        expect(provenance).toHaveTextContent('result #1001 · scored as tool call (strict)');
        expect(screen.getByTestId('verdict-score-other-1001')).toHaveTextContent('default pass rate (ignores arguments the answer key does not have) 93.3%');
        expect(screen.queryByTestId('verdict-partial-1001')).toBeNull();
        await screen.findByTestId('verdict-lift-eval_set:fresh@v1+native');
    });

    it('JSON: scored as JSON fields on mean per-field F1, and a capped eval says "Partial"', () => {
        renderCard([JSON_RESULT]);
        expect(screen.getByTestId('verdict-score-provenance-1002')).toHaveTextContent('scored as JSON fields');
        expect(screen.getByTestId('verdict-score-name-1002')).toHaveTextContent('Mean per-field F1 · 7 rows scored');
        expect(screen.getByTestId('verdict-partial-1002')).toHaveTextContent('Partial: 7 of 40 rows scored');
    });

    it('QA: token F1 (the lift headline) with no interval, named by its key', () => {
        renderCard([QA]);
        expect(screen.getByTestId('verdict-score-value-1003')).toHaveTextContent(/^82\.0%$/);
        expect(screen.getByTestId('verdict-score-provenance-1003')).toHaveTextContent('scored as free text vs answer key (exact match + token F1)');
        expect(screen.getByTestId('verdict-score-1003')).toHaveTextContent('Token F1 · 8 rows scored');
        expect(screen.getByTestId('verdict-score-no-interval-1003')).toHaveTextContent('not a share of passing rows');
    });

    it('a QA result with exact match 0 still shows token F1 (never a different metric by value)', () => {
        renderCard([F1]);
        expect(screen.getByTestId('verdict-score-value-1004')).toHaveTextContent(/^62\.0%$/);
        expect(screen.getByTestId('verdict-score-no-interval-1004')).toBeInTheDocument();
        expect(screen.getByTestId('verdict-score-1004')).toHaveTextContent('Token F1 · 8 rows scored');
    });

    it('span results: base with EM 0 and fine-tuned with EM > 0 show the same metric on the card', () => {
        const base = { id: 2001, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.31, metrics: { scoring_mode: 'span_set', total: 50, correct: 0, exact_match: 0, f1: 0.31 } };
        const tuned = { id: 2002, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.12, metrics: { scoring_mode: 'span_set', total: 50, correct: 6, exact_match: 0.12, f1: 0.74 } };
        const { unmount } = renderCard([base]);
        expect(screen.getByTestId('verdict-score-name-2001')).toHaveTextContent('Span F1 (micro)');
        unmount();
        renderCard([tuned]);
        expect(screen.getByTestId('verdict-score-name-2002')).toHaveTextContent('Span F1 (micro)');
        expect(screen.getByTestId('verdict-score-value-2002')).toHaveTextContent(/^74\.0%$/);
    });

    it('multiple options with verifier-excluded rows: no false "Partial", the excluded rows named on their own', () => {
        renderCard([{
            id: 2003, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.6,
            metrics: { scoring_mode: 'multi_option', multi_option_pass_rate: 0.6, exact_match: 0.6, total: 35, correct: 21, verifier_error_rows: 5 },
            details: { dataset: { rows_available: 40, max_samples: 100 }, inference: { samples: 40 } },
        }]);
        expect(screen.queryByTestId('verdict-partial-2003')).toBeNull();
        expect(screen.getByTestId('verdict-excluded-2003')).toHaveTextContent('5 of 40 answered rows are not in this rate: the option verifier');
        expect(screen.getByTestId('verdict-score-2003')).toHaveTextContent('Multiple-options pass rate · 21 of 35 rows pass');
    });

    it('a result another model answered is labelled, and skipped when the run has its own result on that set', () => {
        const custom = { ...QA, id: 2004, metrics: { ...QA.metrics, evaluated_model: { model_path: '/models/other', kind: 'custom' } } };
        const { unmount } = renderCard([custom]);
        expect(screen.getByTestId('verdict-other-model-2004')).toHaveTextContent('answered by model path /models/other');
        // No lift line for another model's answers.
        expect(screen.queryByTestId('verdict-card-lifts')).toBeNull();
        unmount();
        renderCard([custom, QA]);
        expect(screen.queryByTestId('verdict-score-2004')).toBeNull();
        expect(screen.getByTestId('verdict-score-1003')).toBeInTheDocument();
    });

    it("Fix A: a lift summary citing a result another model answered is flagged, never shown as the fine-tune's lift", async () => {
        // Older servers picked the newest result even when an explicit model path answered it.
        const other = { ...TOOL_CALL, id: 1005, metrics: { ...TOOL_CALL.metrics, evaluated_model: { model_path: '/models/other', kind: 'custom' } } };
        apiMock.get.mockResolvedValue({ data: { ...LIFT_OK, trained: { ...LIFT_OK.trained, eval_result_id: 1005 } } });
        renderCard([other, TOOL_CALL]);
        const key = 'eval_set:fresh@v1+native';
        const warn = await screen.findByTestId(`verdict-lift-${key}-other-model`);
        expect(warn).toHaveTextContent("Not this run's own weights: the fine-tuned side is result #1005, answered by model path /models/other");
        expect(warn).toHaveTextContent("This is not the fine-tune's lift");
        expect(screen.getByTestId(`verdict-lift-${key}-delta`)).toHaveClass('verdict-card__tone--neutral');
    });

    it("Fix A: no warning when the cited result is the run's own", async () => {
        renderCard([TOOL_CALL]);
        const key = 'eval_set:fresh@v1+native';
        await screen.findByTestId(`verdict-lift-${key}-values`);
        expect(screen.queryByTestId(`verdict-lift-${key}-other-model`)).toBeNull();
    });

    it('base vs fine-tuned per set: strict on → the strict lift row (same metric as the score), McNemar, both runs', async () => {
        renderCard([TOOL_CALL]);
        const key = 'eval_set:fresh@v1+native';
        expect(await screen.findByTestId(`verdict-lift-${key}-values`)).toHaveTextContent('0.758 → 0.905');
        expect(screen.getByTestId(`verdict-lift-${key}`)).toHaveAttribute('data-metric', 'tool_call_strict_pass_rate');
        expect(screen.getByTestId(`verdict-lift-${key}`)).toHaveTextContent('Strict tool-call pass rate');
        expect(screen.getByTestId(`verdict-lift-${key}-evidence`)).toHaveTextContent('beyond row noise · 111 rows better, 23 worse (McNemar p < 0.001)');
        expect(screen.getByTestId(`verdict-lift-${key}-provenance`)).toHaveTextContent('Base: run #214 (Qwen/X), result #990 · fine-tuned: run #21, result #1001');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/4/evaluation/sft-lift-summary', { params: { experiment_id: 21, dataset_name: key } });
    });

    it('strict off → the tool-call pass rate row', async () => {
        const plain = { ...TOOL_CALL, metrics: { ...TOOL_CALL.metrics, tool_call_strict_pass_rate: undefined, tool_call_fingerprint: 'tool_call/v3+abc' } };
        renderCard([plain]);
        const key = 'eval_set:fresh@v1+native';
        expect(await screen.findByTestId(`verdict-lift-${key}-values`)).toHaveTextContent('0.810 → 0.933');
        expect(screen.getByTestId(`verdict-lift-${key}`)).toHaveAttribute('data-metric', 'tool_call_pass_rate');
    });

    it('Open side by side carries the two cited result ids (eval_a / eval_b)', async () => {
        renderCard([TOOL_CALL]);
        const key = 'eval_set:fresh@v1+native';
        const link = await screen.findByTestId(`verdict-lift-${key}-side-by-side`);
        expect(link.getAttribute('href')).toBe('/project/4/results?a=214&b=21&set=eval_set%3Afresh%40v1%2Bnative&eval_a=990&eval_b=1001&tab=compare');
    });

    it('rules that regressed / still fail on the lift are counted on the lift line, with details', async () => {
        const gateRow = (id: string, status: string) => ({
            gate_id: id, metric_id: 'f1', threshold: 0.5, operator: 'gte', required: true, baseline_value: 0.6, trained_value: 0.4,
            baseline_passes: true, trained_passes: false, delta_to_threshold: -0.1, status,
        });
        apiMock.get.mockResolvedValue({ data: { ...LIFT_OK, gate_status: [gateRow('a', 'regressed'), gateRow('b', 'regressed'), gateRow('c', 'still_failing'), gateRow('d', 'cleared')] } });
        const onOpen = vi.fn();
        renderCard([TOOL_CALL], null, { onOpenDetails: onOpen });
        const key = 'eval_set:fresh@v1+native';
        expect(await screen.findByTestId(`verdict-lift-${key}-gates`)).toHaveTextContent('2 rules regressed / 1 rule still failing · details');
        await userEvent.setup().click(screen.getByTestId(`verdict-lift-${key}-gates-details`));
        expect(onOpen).toHaveBeenCalled();
    });

    it('a shared lift fetch is used when given', async () => {
        const fetchLift = vi.fn().mockResolvedValue(LIFT_OK);
        renderCard([TOOL_CALL], null, { fetchLift });
        await screen.findByTestId('verdict-lift-eval_set:fresh@v1+native-values');
        expect(fetchLift).toHaveBeenCalledWith(4, 21, 'eval_set:fresh@v1+native');
        expect(apiMock.get).not.toHaveBeenCalled();
    });

    it('a set without a base-model result says so instead of a lift', async () => {
        apiMock.get.mockResolvedValue({ data: { status: 'no_baseline', message: 'No base-model eval on this set yet.', metric_lifts: [], gate_status: [] } });
        renderCard([QA]);
        expect(await screen.findByTestId('verdict-lift-builtin')).toHaveTextContent('Base vs fine-tuned on the built-in test examples: No base-model eval on this set yet.');
        expect(screen.queryByTestId('verdict-lift-builtin-side-by-side')).toBeNull();
    });

    it('a baseline run asks for no lift', () => {
        renderCard([QA], null, { isBaseline: true });
        expect(screen.getByTestId('verdict-card-run')).toHaveTextContent('this run is the base model');
        expect(apiMock.get).not.toHaveBeenCalled();
    });

    it('rules line: NOT MEASURED stays apart from FAIL, and a failing rule is named', async () => {
        const onOpen = vi.fn();
        const { unmount } = renderCard([QA], { status: 'not_measured', failed: [], missing: ['tool_call_pass_rate'], ruleSet: 'General' }, { onOpenDetails: onOpen });
        expect(screen.getByTestId('verdict-rules-status')).toHaveTextContent('NOT MEASURED');
        expect(screen.getByTestId('verdict-rules-missing')).toHaveTextContent('not measured: tool_call_pass_rate (no value on this run, not a failure of the model)');
        expect(screen.queryByTestId('verdict-rules-failed')).toBeNull();
        await userEvent.setup().click(screen.getByTestId('verdict-rules-details'));
        expect(onOpen).toHaveBeenCalled();
        unmount();

        renderCard([QA], { status: 'fail', failed: ['min_f1'], missing: ['judge_correct'] });
        expect(screen.getByTestId('verdict-rules-status')).toHaveTextContent('FAIL');
        expect(screen.getByTestId('verdict-rules-failed')).toHaveTextContent('failed: min_f1');
        expect(screen.getByTestId('verdict-rules-missing')).toHaveTextContent('judge_correct');
    });

    it('no gate report yet reads "not checked yet", never PASS; a failed fetch says it could not be loaded', () => {
        const { unmount } = renderCard([QA]);
        expect(screen.getByTestId('verdict-rules-status')).toHaveTextContent('not checked yet');
        unmount();
        renderCard([QA], null, { gateError: 'Network Error' });
        expect(screen.getByTestId('verdict-rules-status')).toHaveTextContent('could not be loaded');
    });

    it('a seed group names the measured seed run on the rules line', () => {
        renderCard([QA], { status: 'pass', failed: [], missing: [], evaluatedOn: { experiment_id: 166, seed_value: 2, n_seeds: 3, basis: 'median_seed' } });
        expect(screen.getByTestId('verdict-rules-evaluated-on')).toHaveTextContent('Measured on seed run #166 (seed 2), the median of 3 seeds');
    });

    it('slices hint when a result has no breakdown, none when it has one', () => {
        const withSlices = { ...QA, id: 1005, dataset_name: 'validation', metrics: { scoring_mode: 'structured_plan', structured_plan_pass_rate: 0.75, exact_match: 0.75, total: 8, correct: 6, slice_breakdown: { slices: [{ id: 'a', rows: 4, scored: 4, pass: 3 }], overall: { rows: 8, scored: 8, pass: 6, pass_ci: [0.4, 0.93] } } } };
        renderCard([withSlices, { ...TOOL_CALL }]);
        expect(screen.queryByTestId('verdict-slices-hint-1005')).toBeNull();
        expect(screen.getByTestId('verdict-score-value-1005')).toHaveTextContent('75.0% (95% CI 40.0%–93.0%)');
        const hint = screen.getByTestId('verdict-slices-hint-1001');
        expect(hint).toHaveTextContent('No slices: tag rows with slice, or define project slices');
        expect(hint.querySelector('a')?.getAttribute('href')).toBe('/project/4/results?tab=failures&open=failures-tools');
    });

    it('a plain string-match result next to a task-scored one moves aside, with no orphan lift line', async () => {
        renderCard([TOOL_CALL, QA]);
        expect(screen.queryByTestId('verdict-score-1003')).toBeNull();
        expect(screen.getByTestId('verdict-card-string-match-aside')).toHaveTextContent('1 plain string-match result');
        await screen.findByTestId('verdict-lift-eval_set:fresh@v1+native');
        expect(screen.queryByTestId('verdict-lift-builtin')).toBeNull();
        expect(screen.queryByTestId('verdict-lift-builtin-loading')).toBeNull();
    });

    it('nothing scored yet says so', () => {
        renderCard([]);
        expect(screen.getByTestId('verdict-card-not-scored')).toBeInTheDocument();
        expect(screen.queryByTestId('verdict-card-lifts')).toBeNull();
    });

    it('unjudged rows left out of the rate are called out', () => {
        renderCard([{ ...QA, metrics: { ...QA.metrics, pass_rate_unjudged_excluded: 3 } }]);
        expect(screen.getByTestId('verdict-unscored-1003')).toHaveTextContent('3 rows left without a score (unjudged) are not in this rate.');
    });
});

describe('HowItsScoredCard', () => {
    it('reads how each result was scored and links to the settings', async () => {
        render(<HowItsScoredCard projectId={4} runLabel={RUN} results={[TOOL_CALL, JSON_RESULT]} />);
        await waitFor(() => expect(screen.getByTestId('how-scored-1001')).toBeInTheDocument());
        expect(screen.getByTestId('how-scored-1001')).toHaveTextContent('scored as tool call · strict arguments on · prompt rendering v2 · max 256 new tokens, temperature 0');
        expect(screen.getByTestId('how-scored-1002')).toHaveTextContent('scored as JSON fields · null counts as absent');
    });

    it('strict arguments: off only when the result records it, else "not recorded"', () => {
        const off = { ...TOOL_CALL, id: 1010, metrics: { ...TOOL_CALL.metrics, tool_call_strict_pass_rate: undefined, tool_call_fingerprint: 'tool_call/v3+abc' } };
        const old = { ...TOOL_CALL, id: 1011, dataset_name: 'test', metrics: { scoring_mode: 'tool_call', tool_call_pass_rate: 0.9, total: 10 } };
        render(<HowItsScoredCard projectId={4} runLabel={RUN} results={[off, old]} />);
        expect(screen.getByTestId('how-scored-1010')).toHaveTextContent('strict arguments off');
        expect(screen.getByTestId('how-scored-1011')).toHaveTextContent('strict arguments not recorded');
        const links = screen.getByTestId('how-scored-links');
        expect(links.querySelector('a[href="/project/4/train?open=tool-scoring"]')).not.toBeNull();
        expect(links.querySelector('a[href="/project/4/train?open=auto-lift"]')).not.toBeNull();
    });
});
