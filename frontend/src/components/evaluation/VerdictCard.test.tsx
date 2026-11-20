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
import { isStringMatchAside, liftKeys, partialRowsText, scoreOfRecord, wilsonInterval, type VerdictResultLike } from './verdictScore';

const RUN = 'Run #21 · exp-21';

const TOOL_CALL: VerdictResultLike = {
    id: 1001,
    dataset_name: 'eval_set:fresh@v1+native',
    eval_type: 'exact_match',
    pass_rate: 0.905,
    metrics: {
        scoring_mode: 'tool_call', total: 600, correct: 560, exact_match: 0.9333,
        tool_call_pass_rate: 0.9333, tool_call_strict_pass_rate: 0.905, prompt_format_version: 2,
    },
    details: { dataset: { rows_available: 600, content_sha256: 'abcdef1234', eval_set: { name: 'Fresh', rows: 600 } }, inference: { max_new_tokens: 256, temperature: 0 } },
};

const JSON_RESULT: VerdictResultLike = {
    id: 1002,
    dataset_name: 'test',
    eval_type: 'exact_match',
    pass_rate: 0.5,
    metrics: { scoring_mode: 'field_match', total: 7, correct: 3, json_scoring: { null_as_absent: true } },
    details: { dataset: { rows_available: 40 } },
};

const QA: VerdictResultLike = {
    id: 1003,
    dataset_name: 'test',
    eval_type: 'exact_match',
    pass_rate: 0.75,
    metrics: { total: 8, correct: 6, exact_match: 0.75 },
    details: { dataset: { rows_available: 8 } },
};

const F1: VerdictResultLike = {
    id: 1004,
    dataset_name: 'test',
    eval_type: 'f1',
    pass_rate: 0.62,
    metrics: { total: 8, correct: 3, f1: 0.62 },
    details: { dataset: { rows_available: 8 } },
};

const LIFT_OK = {
    status: 'ok',
    project_id: 4,
    dataset_name: 'eval_set:fresh@v1+native',
    baseline: { experiment_id: 214, experiment_name: 'Baseline · Qwen/X', base_model: 'Qwen/X', eval_result_id: 990 },
    trained: { experiment_id: 21, experiment_name: 'exp-21', base_model: 'Qwen/X', eval_result_id: 1001 },
    metric_lifts: [{
        metric_id: 'tool_call_strict_pass_rate', baseline_value: 0.758, trained_value: 0.905, absolute_delta: 0.147,
        relative_delta_pct: 19.4, direction: 'improved', is_headline: true,
        evidence: { n: 600, better: 111, worse: 23, same: 466, mean_diff: 0.147, ci_low: 0.11, ci_high: 0.18, verdict: 'better', mcnemar_p: 0.0001 },
    }],
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
        expect(scoreOfRecord(QA).ci).not.toBeNull();
    });

    it('flags a capped eval as partial', () => {
        expect(partialRowsText(JSON_RESULT)).toBe('Partial: 7 of 40 rows scored (row limit). This score covers those rows only.');
        expect(partialRowsText(QA)).toBeNull();
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

    it('JSON: scored as JSON fields, and a capped eval says "Partial"', () => {
        renderCard([JSON_RESULT]);
        expect(screen.getByTestId('verdict-score-provenance-1002')).toHaveTextContent('scored as JSON fields');
        expect(screen.getByTestId('verdict-partial-1002')).toHaveTextContent('Partial: 7 of 40 rows scored');
    });

    it('QA: exact match with interval; a token-F1 score gets no interval and says why', () => {
        renderCard([QA]);
        expect(screen.getByTestId('verdict-score-value-1003')).toHaveTextContent('75.0% (95% CI');
        expect(screen.getByTestId('verdict-score-provenance-1003')).toHaveTextContent('scored as exact match (string)');
        expect(screen.getByTestId('verdict-score-1003')).toHaveTextContent('Exact match · 6 of 8 rows pass');
    });

    it('a token-F1 result shows no interval', () => {
        renderCard([F1]);
        expect(screen.getByTestId('verdict-score-value-1004')).toHaveTextContent(/^62\.0%$/);
        expect(screen.getByTestId('verdict-score-no-interval-1004')).toBeInTheDocument();
        // The stored pass rate is named by the metric it is (a QA result's headline is token F1).
        expect(screen.getByTestId('verdict-score-1004')).toHaveTextContent('Token F1 · 8 rows scored');
    });

    it('base vs fine-tuned per set: values, rows better / worse with McNemar, both runs, and Open side by side', async () => {
        renderCard([TOOL_CALL]);
        const key = 'eval_set:fresh@v1+native';
        expect(await screen.findByTestId(`verdict-lift-${key}-values`)).toHaveTextContent('0.758 → 0.905');
        expect(screen.getByTestId(`verdict-lift-${key}-evidence`)).toHaveTextContent('beyond row noise · 111 rows better, 23 worse (McNemar p < 0.001)');
        expect(screen.getByTestId(`verdict-lift-${key}-provenance`)).toHaveTextContent('Base: run #214 (Qwen/X), result #990 · fine-tuned: run #21, result #1001');
        const link = screen.getByTestId(`verdict-lift-${key}-side-by-side`);
        expect(link.getAttribute('href')).toBe('/project/4/results?a=214&b=21&set=eval_set%3Afresh%40v1%2Bnative&tab=compare');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/4/evaluation/sft-lift-summary', { params: { experiment_id: 21, dataset_name: key } });
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

    it('no gate report yet reads "not checked yet", never PASS', () => {
        renderCard([QA]);
        expect(screen.getByTestId('verdict-rules-status')).toHaveTextContent('not checked yet');
    });

    it('slices hint when a result has no breakdown, none when it has one', () => {
        const withSlices = { ...QA, id: 1005, dataset_name: 'validation', metrics: { ...QA.metrics, scoring_mode: 'field_match', slice_breakdown: { slices: [{ id: 'a', rows: 4, scored: 4, pass: 3 }], overall: { rows: 8, scored: 8, pass: 6, pass_ci: [0.4, 0.93] } } } };
        renderCard([withSlices, { ...TOOL_CALL }]);
        expect(screen.queryByTestId('verdict-slices-hint-1005')).toBeNull();
        expect(screen.getByTestId('verdict-score-value-1005')).toHaveTextContent('75.0% (95% CI 40.0%–93.0%)');
        const hint = screen.getByTestId('verdict-slices-hint-1001');
        expect(hint).toHaveTextContent('No slices: tag rows with slice, or define project slices');
        expect(hint.querySelector('a')?.getAttribute('href')).toBe('/project/4/results?tab=failures&open=failures-tools');
    });

    it('a plain string-match result next to a task-scored one moves aside', () => {
        renderCard([TOOL_CALL, QA]);
        expect(screen.queryByTestId('verdict-score-1003')).toBeNull();
        expect(screen.getByTestId('verdict-card-string-match-aside')).toHaveTextContent('1 plain string-match result');
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
        const links = screen.getByTestId('how-scored-links');
        expect(links.querySelector('a[href="/project/4/train?open=tool-scoring"]')).not.toBeNull();
        expect(links.querySelector('a[href="/project/4/train?open=auto-lift"]')).not.toBeNull();
    });
});
