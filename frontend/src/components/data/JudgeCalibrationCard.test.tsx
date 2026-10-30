import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import JudgeCalibrationCard from './JudgeCalibrationCard';

const rate = (value: number) => ({ value, low: value - 0.2, high: Math.min(1, value + 0.1), n: 10, n_eff: 8.5 });
const judge = (name: string, extra: Record<string, unknown> = {}) => ({
    judge: name, n: 12, n_eff: 9.4, label_fail: 4, label_pass: 8,
    counts: { tp: 3, fp: 1, fn: 1, tn: 7, unsure_label_fail: 0, unsure_label_pass: 0 },
    fail_recall: rate(0.75), pass_precision: rate(0.88), unsure_rate: 0.05, kappa: 0.61,
    insufficient: true, insufficient_reasons: ['fewer than 5 labelled fails (4)'],
    unparseable: 0, rubric_shas: { abc: 12 }, rubric_mixed: false, self: false, self_rows: 0, ...extra,
});
const tierReport = (tier: string, judges: unknown[], labellers: string[] = [], extra: Record<string, unknown> = {}) => ({
    tier, labelled: 12, stale: 0, skipped: 0, labellers, population: 40, coverage: 1, strata: {},
    skip_warning_strata: [], skip_note: null, judges, ...extra,
});
const progress = {
    population: 40,
    tiers: {
        human: { labelled: 12, stale: 1, skipped: 2, pass: 8, fail: 4, labellers: ['human'], strata: {} },
        reference_model: { labelled: 0, stale: 0, skipped: 0, pass: 0, fail: 0, labellers: [], strata: {} },
    },
};

function mockGets(source = 'human') {
    apiMock.get.mockImplementation((url: string, config?: { params?: Record<string, unknown> }) => {
        if (url.endsWith('/calibration/settings')) {
            return Promise.resolve({ data: { label_source: source, default: 'human', options: ['human', 'reference_model', 'both'] } });
        }
        if (url.endsWith('/calibration/labels')) return Promise.resolve({ data: progress });
        if (url.endsWith('/calibration/report')) {
            const tier = config?.params?.tier ?? source;
            const reports = [];
            if (tier === 'human' || tier === 'both') reports.push(tierReport('human', [judge('ollama:small')]));
            if (tier === 'reference_model' || tier === 'both') {
                reports.push(tierReport('reference_model', [judge('ollama:small'), judge('cloud:big', { self: true, self_rows: 12 })], ['cloud:big'], {
                    skipped: 6, skip_warning_strata: ['disagree'],
                    skip_note: 'estimates may be optimistic; skipped rows are often the hard ones',
                }));
            }
            return Promise.resolve({ data: { label_source: source, tier, min_class_labels: 5, min_total_labels: 20, reports } });
        }
        if (url.endsWith('/calibration/sample')) {
            return Promise.resolve({ data: { tier: 'human', rows: [
                { row_id: 7, kind: 'qa', content: '{"question":"q7","answer":"a7"}', fields: { question: 'q7', answer: 'a7' }, context_truncated: false },
                { row_id: 9, kind: 'qa', content: '{"question":"q9","answer":"a9"}', fields: { question: 'q9', answer: 'a9' }, context_truncated: false },
            ] } });
        }
        // GenerationModelPicker's catalog calls
        return Promise.resolve({ data: { providers: [], backends: [] } });
    });
}

describe('JudgeCalibrationCard', () => {
    beforeEach(() => {
        Object.values(apiMock).forEach((fn) => fn.mockReset());
        Object.values(toastMock).forEach((fn) => fn.mockReset());
        apiMock.post.mockResolvedValue({ data: {} });
        apiMock.put.mockImplementation((_url: string, body: { label_source: string }) => Promise.resolve({ data: { label_source: body.label_source } }));
    });

    it('stays collapsed (no calibration fetch) until opened, then shows the per-judge report for the saved tier', async () => {
        mockGets();
        render(<JudgeCalibrationCard projectId={5} />);
        expect(screen.getByTestId('judge-calibration-toggle')).toHaveTextContent('Calibrate judges');
        expect(apiMock.get).not.toHaveBeenCalled();
        await userEvent.click(screen.getByTestId('judge-calibration-toggle'));
        const card = await screen.findByTestId('judge-calibration-judge-human-ollama:small');
        expect(card).toHaveTextContent('vs your labels (human)');
        expect(card).toHaveTextContent('75%');
        expect(card).toHaveTextContent('88%');
        expect(card).toHaveTextContent('0.61');
        expect(screen.getByTestId('judge-calibration-insufficient-human-ollama:small')).toHaveTextContent('fewer than 5 labelled fails');
        expect(screen.getByTestId('judge-calibration-progress')).toHaveTextContent('Your labels: 12');
        expect(screen.queryByTestId('judge-calibration-report-reference_model')).toBeNull();
    });

    it('the switch saves the label source and shows the chosen tier(s), naming the reference labeller', async () => {
        mockGets();
        render(<JudgeCalibrationCard projectId={5} />);
        await userEvent.click(screen.getByTestId('judge-calibration-toggle'));
        await screen.findByTestId('judge-calibration-report-human');
        await userEvent.click(screen.getByTestId('judge-calibration-source-both'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/5/synthetic/calibration/settings', { label_source: 'both' }));
        await screen.findByTestId('judge-calibration-report-reference_model');
        expect(screen.getByTestId('judge-calibration-report-human')).toBeInTheDocument();
        const ref = screen.getByTestId('judge-calibration-judge-reference_model-cloud:big');
        expect(ref).toHaveTextContent('vs reference model (cloud:big)');
        expect(ref).toHaveTextContent('reference model itself');
        expect(screen.getByTestId('judge-calibration-skip-note-reference_model'))
            .toHaveTextContent('rows the judges disagree on: estimates may be optimistic');
        expect(screen.queryByTestId('judge-calibration-skip-note-human')).toBeNull();

        await userEvent.click(screen.getByTestId('judge-calibration-source-reference_model'));
        await waitFor(() => expect(screen.queryByTestId('judge-calibration-report-human')).toBeNull());
        expect(screen.getByTestId('judge-calibration-report-reference_model')).toBeInTheDocument();
    });

    it('the blind labeller shows one row without verdicts, posts pass / fail labels and records a skip', async () => {
        mockGets();
        render(<JudgeCalibrationCard projectId={5} />);
        await userEvent.click(screen.getByTestId('judge-calibration-toggle'));
        await screen.findByTestId('judge-calibration-report-human');
        await userEvent.click(screen.getByTestId('judge-calibration-label-start'));
        const shown = await screen.findByTestId('judge-calibration-row');
        expect(shown).toHaveTextContent('q7');
        expect(screen.getByTestId('judge-calibration-labeller')).not.toHaveTextContent(/verdict|consensus/i);
        await userEvent.type(screen.getByTestId('judge-calibration-reason'), 'wrong unit');
        await userEvent.click(screen.getByTestId('judge-calibration-fail'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/calibration/labels', { row_id: 7, tier: 'human', label: 'fail', reason: 'wrong unit' },
        ));
        expect(await screen.findByTestId('judge-calibration-row')).toHaveTextContent('q9');
        await userEvent.click(screen.getByTestId('judge-calibration-skip'));
        await waitFor(() => expect(screen.queryByTestId('judge-calibration-labeller')).toBeNull());
        expect(apiMock.post).toHaveBeenCalledTimes(2);
        expect(apiMock.post).toHaveBeenLastCalledWith('/projects/5/synthetic/calibration/skips', { row_id: 9, tier: 'human' });
        expect(screen.getByTestId('judge-calibration-progress')).toHaveTextContent('2 skipped');
    });

    it('starts the reference labelling Job with the picked n', async () => {
        mockGets();
        apiMock.post.mockResolvedValue({ data: { id: 44, kind: 'judge_calibration_reference' } });
        render(<JudgeCalibrationCard projectId={5} />);
        await userEvent.click(screen.getByTestId('judge-calibration-toggle'));
        await screen.findByTestId('judge-calibration-report-human');
        const n = screen.getByTestId('judge-calibration-ref-n');
        await userEvent.clear(n);
        await userEvent.type(n, '30');
        await userEvent.click(screen.getByTestId('judge-calibration-ref-start'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/calibration/reference-labelling', { backend: null, n: 30 }, { params: { async_job: true } },
        ));
        expect(toastMock.info).toHaveBeenCalled();
    });
});
