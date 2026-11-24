// Phase 7k metric vocabulary: scale, format and name per metric key + the score of record on it.
import { describe, expect, it } from 'vitest';
import { formatMetricDelta, formatMetricValue, metricHigherIsBetter, metricLabel, metricName, metricScale } from './metricVocabulary';
import { howScoredText, metricValue, scoreOfRecord, scoreSummaryText, scoreToneClass, scoreValueText, seedGroupRuns, seedMeanText } from './verdictScore';
import { metricDisplayName, metricUnit, seedEvidenceNote, type SeedEvidence } from './liftEvidence';

describe('metric vocabulary (phase 7k)', () => {
    it('scale per key: shares as %, 0–1 means as 0.184, perplexity / loss as plain numbers', () => {
        expect(metricScale('exact_match')).toBe('share');
        expect(metricScale('tool_call_pass_rate')).toBe('share');
        expect(metricScale('accuracy')).toBe('share');
        expect(metricScale('f1')).toBe('mean01');
        expect(metricScale('f1', 'span_set')).toBe('mean01');
        expect(metricScale('f1', 'safety')).toBe('share');
        expect(metricScale('judge_correct')).toBe('mean01');
        expect(metricScale('perplexity')).toBe('unbounded');
        expect(metricScale('eval_loss')).toBe('unbounded');
        expect(metricScale('rubric_pass_rate')).toBe('share');
        expect(metricHigherIsBetter('perplexity')).toBe(false);
        expect(metricHigherIsBetter('f1')).toBe(true);
    });

    it('formats a value and a difference in the score\'s own unit', () => {
        expect(formatMetricValue('exact_match', 0.857)).toBe('85.7%');
        expect(formatMetricValue('f1', 0.184)).toBe('0.184');
        expect(formatMetricValue('judge_correct', 0.5)).toBe('0.500');
        expect(formatMetricValue('perplexity', 12.3456)).toBe('12.346');
        expect(formatMetricValue('f1', null)).toBe('—');
        expect(formatMetricDelta('exact_match', 0.048)).toBe('+4.8 pts');
        expect(formatMetricDelta('f1', 0.048)).toBe('+0.048');
        expect(formatMetricDelta('f1', -0.048)).toBe('-0.048');
    });

    it('names come from key + shape', () => {
        expect(metricLabel('f1')).toBe('Token F1');
        expect(metricLabel('f1', 'span_set')).toBe('Span F1 (micro)');
        expect(metricLabel('exact_match', 'classification')).toBe('Accuracy');
        expect(metricName('f1')).toBe('F1');
        expect(metricName('f1', 'text')).toBe('token F1');
        expect(metricName('exec_match_rate', 'text')).toBe('SQL execution match');
        expect(metricDisplayName('judge_correct')).toBe('judge score');
        expect(metricUnit('structured_plan_pass_rate')).toBe('plan pass rate');
        expect(metricUnit('')).toBe('score');
    });
});

describe('score of record on the vocabulary (phase 7k)', () => {
    const QA_F1 = { id: 1, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.184, metrics: { exact_match: 0, f1: 0.184, total: 21, correct: 0 } };
    const PLAN = { id: 2, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.857, metrics: { scoring_mode: 'structured_plan', structured_plan_pass_rate: 0.857, total: 21, correct: 18 } };

    it('token F1 is a 0–1 score, a share is a percentage with a colour', () => {
        const f1 = scoreOfRecord(QA_F1);
        expect(f1.scale).toBe('mean01');
        expect(scoreValueText(f1)).toBe('0.184');
        expect(scoreSummaryText(QA_F1)).toBe('Token F1 0.184');
        expect(scoreToneClass(f1)).toBe('');
        const plan = scoreOfRecord(PLAN);
        expect(plan.scale).toBe('share');
        expect(scoreValueText(plan)).toBe('85.7%');
        expect(scoreSummaryText(PLAN)).toBe('Plan pass rate 85.7%');
        expect(scoreToneClass(plan)).toBe('u-text-success');
    });

    it('how it is scored names the judge for a judged result', () => {
        const judged = { ...QA_F1, metrics: { ...QA_F1.metrics, judge_correct: 0.095 } };
        expect(howScoredText(judged)).toContain('scored as free text vs answer key, graded by a judge model');
        expect(howScoredText(QA_F1)).toContain('scored as free text vs answer key (exact match + token F1)');
    });

    it('a seed-group aggregate stands on the mean of its seeds', () => {
        const agg = {
            id: 9, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.75,
            metrics: { exact_match: { mean: 0.75, std: 0.04, min: 0.7, max: 0.8, n: 3 } },
            details: { per_seed: [{ experiment_id: 5 }, { experiment_id: 6 }, { experiment_id: 7 }] },
        };
        expect(metricValue(agg.metrics, 'exact_match')).toBe(0.75);
        expect(scoreOfRecord(agg).key).toBe('exact_match');
        expect(seedMeanText(agg)).toBe('Mean of 3 seeds (#5, #6, #7)');
        expect(seedMeanText(QA_F1)).toBeNull();
        expect(seedGroupRuns({ config: { _seed_group: { child_experiment_ids: [5, 6, 7] } } })).toEqual([5, 6, 7]);
        expect(seedGroupRuns({ config: { num_seeds: 3 } })).toEqual([]);
        expect(seedGroupRuns({ config: { num_epochs: 3 } })).toBeNull();
    });
});

describe('seed evidence wording (phase 7k)', () => {
    const EV: SeedEvidence = {
        kind: 'seeds', n: 3, verdict: 'better', baseline_value: 0.1, values: [0.14, 0.16, 0.15],
        mean: 0.15, std: 0.008, min: 0.14, max: 0.16, ci_low: 0.13, ci_high: 0.17, all_better: true, all_worse: false,
    };

    it('every seed above the base but each within row noise: the agreement is the evidence', () => {
        const note = seedEvidenceNote(EV, 'F1', { all_within_noise: true, rows: 40 });
        expect(note.label).toBe('Holds across seeds');
        expect(note.text).toContain('every seed scored above the base (0.140, 0.160, 0.150)');
        expect(note.text).toContain('no seed alone is beyond row noise on 40 rows — the agreement across seeds is the evidence');
    });

    it('unchanged when a seed is beyond row noise on its own', () => {
        expect(seedEvidenceNote(EV, 'F1', { all_within_noise: false, rows: 40 }).text).toContain('every seed beat the base model');
        expect(seedEvidenceNote(EV, 'F1').text).toContain('every seed beat the base model');
    });
});
