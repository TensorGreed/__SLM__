import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), patch: vi.fn(), put: vi.fn(), delete: vi.fn() },
    toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import EvalJudgeCalibrationCard from './EvalJudgeCalibrationCard';
import { evalJudgeTrustText } from '../../api/evalJudgeCalibration';

const rate = (value: number) => ({ value, low: Math.max(0, value - 0.1), high: Math.min(1, value + 0.1), n: 30, n_eff: 25 });
const entry = (judge: string, extra: Record<string, unknown> = {}) => ({
    judge, scale: 'verdict', n: 30, n_eff: 25, by_label: { correct: 12, partial: 8, wrong: 10 },
    exact_agreement: rate(0.8), strict_agreement: rate(0.9), pass_precision: rate(0.85), pass_recall: rate(0.9),
    wrong_recall: rate(0.7), too_strict: rate(0.06), too_lenient: rate(0.03), kappa: 0.72, kappa_kind: 'linear_weighted',
    insufficient: false, insufficient_reasons: [], population: 80, skipped: 0, skip_note: null, tasks: { answer: 30 },
    self: false, self_rows: 0, ...extra,
});
const trust = {
    status: 'calibrated_ok', judge: 'ollama:judge', scale: 'verdict', tier: 'human', n: 30, agreement: 0.8,
    strict_agreement: 0.9, ci: [0.8, 0.97], too_strict: 0.06, too_lenient: 0.03, flags: [], label_source: 'human',
    floor: 0.7, too_strict_ceiling: 0.15, too_lenient_ceiling: 0.15, min_labels: 20,
};
const progress = (rubricPopulation = 0) => ({
    scales: {
        verdict: {
            population: 80, labels: ['correct', 'partial', 'wrong'],
            judges: [{ judge: 'ollama:judge', rows: 80, by_verdict: { correct: 40, partial: 20, wrong: 20 } },
                { judge: 'cloud:other', rows: 10, by_verdict: { correct: 5, partial: 0, wrong: 5 } }],
            tiers: {
                human: { labelled: 30, skipped: 2, by_label: {}, labellers: ['human'] },
                reference_model: { labelled: 0, skipped: 0, by_label: {}, labellers: [] },
            },
        },
        rubric: {
            population: rubricPopulation, labels: ['pass', 'fail'], judges: [],
            tiers: { human: { labelled: 0, skipped: 0, by_label: {}, labellers: [] }, reference_model: { labelled: 0, skipped: 0, by_label: {}, labellers: [] } },
        },
    },
});

function mockGets(rubricPopulation = 0) {
    apiMock.get.mockImplementation((url: string) => {
        if (url.endsWith('/synthetic/calibration/settings')) {
            return Promise.resolve({ data: { label_source: 'human', default: 'human', options: ['human', 'reference_model', 'both'] } });
        }
        if (url.endsWith('/eval-judge/calibration/labels')) return Promise.resolve({ data: progress(rubricPopulation) });
        if (url.endsWith('/eval-judge/calibration/report')) {
            return Promise.resolve({ data: {
                label_source: 'human', tier: 'human', scale: 'verdict', labels: ['correct', 'partial', 'wrong'],
                reports: [{ tier: 'human', scale: 'verdict', labelled: 30, skipped: 2, labellers: ['human'], population: 80,
                    judges: [entry('ollama:judge'), entry('cloud:other', { n: 4, insufficient: true, insufficient_reasons: ['fewer than 20 labelled answers (4)'] })] }],
                trust: { 'ollama:judge': trust },
            } });
        }
        if (url.endsWith('/eval-judge/calibration/sample')) {
            return Promise.resolve({ data: { scale: 'verdict', tier: 'human', labels: ['correct', 'partial', 'wrong'], rows: [
                { content_key: 'k1', scale: 'verdict', question: 'What is q1?', reference: 'ref one', prediction: 'answer one' },
                { content_key: 'k2', scale: 'verdict', question: 'What is q2?', reference: 'ref two', prediction: 'answer two',
                    passages: [{ id: 'P1', text: 'grounding text' }], passages_truncated: true },
            ] } });
        }
        return Promise.resolve({ data: { providers: [], backends: [] } });
    });
}

describe('EvalJudgeCalibrationCard', () => {
    beforeEach(() => {
        Object.values(apiMock).forEach((fn) => fn.mockReset());
        Object.values(toastMock).forEach((fn) => fn.mockReset());
        apiMock.post.mockResolvedValue({ data: {} });
        window.location.hash = '';
    });

    it('stays collapsed until opened, then shows the per-judge report with its trust badge', async () => {
        mockGets();
        render(<EvalJudgeCalibrationCard projectId={4} />);
        expect(apiMock.get).not.toHaveBeenCalled();
        await userEvent.click(screen.getByTestId('eval-judge-calibration-toggle'));
        const card = await screen.findByTestId('eval-judge-calibration-judge-human-ollama:judge');
        expect(card).toHaveTextContent('Agrees on correct vs not90%');
        expect(card).toHaveTextContent('Exact agreement (every verdict)80%');
        expect(card).toHaveTextContent('Too strict (says partial / wrong, label correct)6%');
        expect(card).toHaveTextContent('weighted kappa');
        expect(screen.getByTestId('eval-judge-calibration-judge-human-ollama:judge-trust')).toHaveTextContent('judge calibrated vs human: agrees on correct vs not 90% (n 30; exact 80%)');
        expect(screen.getByTestId('eval-judge-calibration-judge-human-cloud:other-insufficient')).toHaveTextContent('fewer than 20 labelled answers (4)');
        expect(screen.getByTestId('eval-judge-calibration-progress')).toHaveTextContent('80 judged answers stored. Your labels: 30, 2 skipped.');
        expect(screen.queryByTestId('eval-judge-calibration-scale')).toBeNull(); // no plan rubric answers
    });

    it('labels answers blind (question, reference, answer only) and posts each label with the sampled judge', async () => {
        mockGets();
        render(<EvalJudgeCalibrationCard projectId={4} />);
        await userEvent.click(screen.getByTestId('eval-judge-calibration-toggle'));
        await screen.findByTestId('eval-judge-calibration-progress');
        await userEvent.selectOptions(screen.getByTestId('eval-judge-calibration-judge'), 'cloud:other');
        await userEvent.click(screen.getByTestId('eval-judge-calibration-label-start'));
        const row = await screen.findByTestId('eval-judge-calibration-row');
        expect(row).toHaveTextContent('What is q1?');
        expect(row).toHaveTextContent('ref one');
        expect(row).toHaveTextContent('answer one');
        expect(screen.getByTestId('eval-judge-calibration-labeller')).not.toHaveTextContent('cloud:other');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/4/eval-judge/calibration/sample', { params: { scale: 'verdict', n: 20, tier: 'human', judge: 'cloud:other' } });
        await userEvent.type(screen.getByTestId('eval-judge-calibration-reason'), 'misses a fact');
        await userEvent.click(screen.getByTestId('eval-judge-calibration-label-partial'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/4/eval-judge/calibration/labels', {
            content_key: 'k1', scale: 'verdict', tier: 'human', label: 'partial', judge: 'cloud:other', reason: 'misses a fact',
        }));
        expect(await screen.findByTestId('eval-judge-calibration-row')).toHaveTextContent('What is q2?');
        expect(screen.getByTestId('eval-judge-calibration-passages')).toHaveTextContent('[P1] grounding text');
        await userEvent.click(screen.getByTestId('eval-judge-calibration-skip'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/4/eval-judge/calibration/skips', {
            content_key: 'k2', scale: 'verdict', tier: 'human',
        }));
        await waitFor(() => expect(screen.queryByTestId('eval-judge-calibration-labeller')).toBeNull());
    });

    it('offers the rubric judge when plans were graded, and starts a reference run', async () => {
        mockGets(6);
        apiMock.post.mockResolvedValue({ data: { id: 12, kind: 'eval_judge_calibration_reference' } });
        render(<EvalJudgeCalibrationCard projectId={4} />);
        await userEvent.click(screen.getByTestId('eval-judge-calibration-toggle'));
        expect(await screen.findByTestId('eval-judge-calibration-scale')).toBeInTheDocument();
        await userEvent.click(screen.getByTestId('eval-judge-calibration-ref-start'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/4/eval-judge/calibration/reference-labelling',
            { backend: null, scale: 'verdict', n: 20, judge: 'ollama:judge' },
            { params: { async_job: true } },
        ));
        expect(toastMock.info).toHaveBeenCalled();
    });

    it('names the floor in a below-floor message', () => {
        const text = evalJudgeTrustText({ ...trust, status: 'below_floor', strict_agreement: 0.55, flags: ['low_agreement'] } as never);
        expect(text).toBe('judge below the trust floor vs human (agreement below the 70% floor): agrees on correct vs not 55% (n 30; exact 80%); too strict 6%, too lenient 3%');
    });
});
