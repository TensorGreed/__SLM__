import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));
vi.mock('../../stores/jobsStore', () => ({
    useJobsStore: { getState: () => ({ refreshAfterLocalChange: vi.fn() }) },
}));

import CompareRowEvidence, { answerKeyText, evidenceHeadline } from './CompareRowEvidence';
import type { CompareProvenance, CompareRowEvidence as RowEvidence } from '../../api/experimentCompare';


function prov(overrides: Partial<CompareProvenance> = {}): CompareProvenance {
    return {
        experiment_id: 11,
        experiment_name: 'Run A',
        eval_result_id: 91,
        evaluated_at: '2026-10-20T10:00:00Z',
        answer_key: { dataset_id: 3, dataset_type: 'test', version: 5, version_source: 'recorded', sha256: 'b'.repeat(64) },
        judge: { answer: 'judge-x', rubric: null, rubric_sha: null, code_example: null },
        scoring: { eval_type: 'exact_match', scoring_mode: null, fingerprints: {} },
        has_row_scores: true,
        ...overrides,
    };
}

// 78 rows: 2 better, 7 worse, 69 same. mean = -5/78 = -0.0641 (−6.4 pts);
// 95% CI -0.1394 to +0.0112 (−14 to +1 pts rounded).
function withinNoise(overrides: Partial<RowEvidence> = {}): RowEvidence {
    return {
        verdict: 'within_noise',
        reason: null,
        message: null,
        metric_id: 'exact_match',
        score_key: 'correct',
        n: 78,
        better: 2,
        worse: 7,
        same: 69,
        mean_diff: -5 / 78,
        ci_low: -0.13937,
        ci_high: 0.01117,
        only_in_a: 0,
        only_in_b: 1,
        unscored: 0,
        metrics: [],
        comparability: { comparable: true, differences: [], notes: [] },
        rescore: null,
        ...overrides,
    };
}

describe('evidenceHeadline', () => {
    it('reads the 78-row within-noise example', () => {
        expect(evidenceHeadline(withinNoise())).toBe(
            'B is within noise of A: 2 better, 7 worse, 69 same of 78 rows; change −6.4 pts (95% CI −14 to +1)',
        );
    });

    it('reads a gain with a positive interval', () => {
        // 9 better, 1 worse of 20: mean +0.4; CI +0.1 to +0.7 → +10 to +70 pts.
        expect(evidenceHeadline({ verdict: 'better', n: 20, better: 9, worse: 1, same: 10, mean_diff: 0.4, ci_low: 0.1, ci_high: 0.7 }))
            .toBe('B is better than A: 9 better, 1 worse, 10 same of 20 rows; change +40.0 pts (95% CI +10 to +70)');
    });
});

describe('answerKeyText', () => {
    it('marks an inferred split version', () => {
        expect(answerKeyText(prov({
            answer_key: { dataset_id: 3, dataset_type: 'test', version: 3, version_source: 'inferred', sha256: null },
        }))).toBe('test examples, split v3 (inferred from eval time)');
    });
});

describe('CompareRowEvidence', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
        toastMock.info.mockReset();
    });

    it('shows the headline, pairing counts and both sides\' provenance', () => {
        render(<CompareRowEvidence projectId={5} evidence={withinNoise()} a={prov()} b={prov({ experiment_id: 12, experiment_name: 'Run B', eval_result_id: 92 })} />);
        expect(screen.getByTestId('eval-compare-evidence')).toHaveAttribute('data-verdict', 'within_noise');
        expect(screen.getByTestId('eval-compare-evidence-headline').textContent).toContain('2 better, 7 worse, 69 same of 78 rows');
        expect(screen.getByTestId('eval-compare-evidence-pairing').textContent).toContain('78 rows paired, 1 only in B');
        expect(screen.getByTestId('eval-compare-provenance-a').textContent).toContain('Run A (run #11) · eval #91 · test examples, split v5 · judge: judge-x');
        expect(screen.getByTestId('eval-compare-provenance-b').textContent).toContain('eval #92');
        expect(screen.queryByTestId('eval-compare-rescore')).not.toBeInTheDocument();
    });

    it('within noise: adds the detectable-difference line; a clear gain does not', () => {
        // Backend: s_d² = 0.112721 → gap 0.1065 (~11 pts), rows(5 pts) 354 (~360).
        const sizing = {
            basis: 'paired' as const, metric_id: 'exact_match', n: 78, score_kind: 'binary' as const, sd: 0.3357,
            detectable_gap: 0.1065, target_gap: 0.05, rows_needed: 354, enough: false, degenerate: null,
            confidence: 0.95, power: 0.8,
            line: '78 paired rows (exact match): gaps under ~11 pts may go unnoticed; ~360 rows to see a 5-pt change.',
        };
        const { unmount } = render(<CompareRowEvidence projectId={5} evidence={withinNoise({ sizing })} a={prov()} b={prov()} />);
        expect(screen.getByTestId('eval-compare-evidence-sizing')).toHaveTextContent(sizing.line);
        unmount();
        render(<CompareRowEvidence projectId={5} evidence={withinNoise({ sizing, verdict: 'better', ci_low: 0.01 })} a={prov()} b={prov()} />);
        expect(screen.queryByTestId('eval-compare-evidence-sizing')).not.toBeInTheDocument();
    });

    it('not comparable: names the differences, gives no counts, re-scores the older run', async () => {
        apiMock.post.mockResolvedValue({ data: { started: true, job_id: 7 } });
        const ev = withinNoise({
            verdict: 'not_comparable',
            reason: 'not_comparable',
            message: 'different answer key: split v3 vs v5',
            n: 0, better: 0, worse: 0, same: 0, mean_diff: null, ci_low: null, ci_high: null,
            comparability: {
                comparable: false,
                differences: [{ kind: 'answer_key', message: 'different answer key: split v3 vs v5', a: 3, b: 5 }],
                notes: [],
            },
            rescore: {
                side: 'a', experiment_id: 11, experiment_name: 'Run A', why: 'not_comparable',
                available: true, unavailable_reason: null, instruction: 'Re-scores Run #11 on the current test examples.',
            },
        });
        const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
        render(<CompareRowEvidence projectId={5} evidence={ev} a={prov()} b={prov()} />);
        expect(screen.getByTestId('eval-compare-evidence-headline').textContent).toBe(
            'A and B were scored differently, so there is no verdict',
        );
        expect(screen.getByTestId('eval-compare-evidence-diffs').textContent).toContain('different answer key: split v3 vs v5');
        expect(screen.getByTestId('eval-compare-evidence').textContent).not.toMatch(/better,/);
        const button = screen.getByTestId('eval-compare-rescore-button');
        expect(button.textContent).toBe('Re-score Run A (run #11) on the current answer key and judge');
        await userEvent.click(button);
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/evaluation/summary/lift-check', { experiment_id: 11 },
        ));
        expect(confirmSpy.mock.calls[0][0]).toMatch(/Re-score run #11\? This re-evaluates it on the GPU/);
        expect(toastMock.info).toHaveBeenCalled();
        expect(screen.getByTestId('eval-compare-rescore-button')).toBeDisabled();
        confirmSpy.mockRestore();
    });

    const notComparable = (rescore: Partial<NonNullable<RowEvidence['rescore']>> = {}, certain = true) => withinNoise({
        verdict: 'not_comparable',
        reason: 'not_comparable',
        message: 'answer key may differ: split v3 vs v5 (v3 inferred from eval time)',
        n: 0, better: 0, worse: 0, same: 0, mean_diff: null, ci_low: null, ci_high: null,
        comparability: {
            comparable: false,
            certain,
            differences: [{ kind: 'answer_key', message: 'answer key may differ: split v3 vs v5 (v3 inferred from eval time)', a: 3, b: 5, inferred: true }],
            notes: [],
            caveats: ['Decoded differently: repetition_penalty 1.1 vs 1.0. The gap may partly come from decoding, not the model.'],
        },
        rescore: {
            side: 'a', experiment_id: 11, experiment_name: 'Run A', why: 'not_comparable',
            available: true, in_progress: false, job_id: null, unavailable_reason: null, instruction: 'Re-scores Run #11.',
            ...rescore,
        },
    });

    it('declining the confirmation posts nothing', async () => {
        const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
        render(<CompareRowEvidence projectId={5} evidence={notComparable()} a={prov()} b={prov()} />);
        await userEvent.click(screen.getByTestId('eval-compare-rescore-button'));
        expect(apiMock.post).not.toHaveBeenCalled();
        confirmSpy.mockRestore();
    });

    it('an inferred-only difference reads "may" and the decoding caveat shows', () => {
        render(<CompareRowEvidence projectId={5} evidence={notComparable({}, false)} a={prov()} b={prov()} />);
        expect(screen.getByTestId('eval-compare-evidence-headline').textContent).toBe(
            'A and B may have been scored differently, so there is no verdict',
        );
        expect(screen.getByTestId('eval-compare-evidence-diffs').textContent).toContain('(v3 inferred from eval time)');
        expect(screen.getByTestId('eval-compare-evidence-caveat').textContent).toMatch(/^Decoded differently: repetition_penalty 1.1 vs 1.0/);
    });

    it('a re-score already running disables the button', () => {
        render(<CompareRowEvidence projectId={5} evidence={notComparable({ available: false, in_progress: true, job_id: 7 })} a={prov()} b={prov()} />);
        const button = screen.getByTestId('eval-compare-rescore-button');
        expect(button.textContent).toBe('Re-score running');
        expect(button).toBeDisabled();
    });

    it('missing row scores with no re-score path shows the instruction instead of a button', () => {
        const ev = withinNoise({
            verdict: 'too_few_rows',
            reason: 'no_row_scores',
            message: 'A: no per-row scores (evaluated before per-row capture)',
            n: 0, better: 0, worse: 0, same: 0, mean_diff: null, ci_low: null, ci_high: null,
            rescore: {
                side: 'a', experiment_id: 11, experiment_name: 'Run A', why: 'no_row_scores', available: false,
                unavailable_reason: 'simulated_run', instruction: 'Run #11 is a simulated run with no model weights.',
            },
        });
        render(<CompareRowEvidence projectId={5} evidence={ev} a={prov({ has_row_scores: false })} b={prov()} />);
        expect(screen.getByTestId('eval-compare-evidence-headline').textContent)
            .toBe('No row-by-row verdict: A: no per-row scores (evaluated before per-row capture)');
        expect(screen.getByTestId('eval-compare-rescore').textContent).toContain('simulated run');
        expect(screen.queryByTestId('eval-compare-rescore-button')).not.toBeInTheDocument();
    });

    it('lists the other per-row metrics under the headline', () => {
        const ev = withinNoise({
            metrics: [
                { metric_id: 'exact_match', score_key: 'correct', n: 78, better: 2, worse: 7, same: 69, mean_diff: -5 / 78, ci_low: -0.13937, ci_high: 0.01117, verdict: 'within_noise', only_in_a: 0, only_in_b: 0, unscored: 0 },
                { metric_id: 'f1', score_key: 'f1', n: 78, better: 10, worse: 30, same: 38, mean_diff: -0.1, ci_low: -0.15, ci_high: -0.05, verdict: 'worse', only_in_a: 0, only_in_b: 0, unscored: 0 },
            ],
        });
        render(<CompareRowEvidence projectId={5} evidence={ev} a={prov()} b={prov()} />);
        expect(screen.getByTestId('eval-compare-evidence-metric-f1').textContent).toContain(
            'B is worse than A: 10 better, 30 worse, 38 same of 78 rows; change −10.0 pts (95% CI −15 to −5)',
        );
    });
});
