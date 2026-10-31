import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() } }));
vi.mock('../../stores/jobsStore', () => ({
    useJobsStore: { getState: () => ({ refreshAfterLocalChange: vi.fn() }) },
}));

import CheckedDecodingBadge from './CheckedDecodingBadge';
import CheckedDecodingSettings from './CheckedDecodingSettings';
import CompareRowEvidence from './CompareRowEvidence';
import EvalSummaryCard from './EvalSummaryCard';
import { checkedDecodingBadgeText, rowsFixedText, type CheckedDecodingSettings as Settings } from '../../api/checkedDecoding';
import type { CompareProvenance, CompareRowEvidence as RowEvidence } from '../../api/experimentCompare';

const OFF: Settings = { enabled: false, mode: 'resample', max_attempts: 3, verifiers: null, sample_temperature: 0.7, seed: 0 };

function payload(settings: Settings) {
    return {
        data: {
            settings,
            label: settings.enabled ? `Checked decoding · up to ${settings.max_attempts} tries · ${settings.mode}` : 'no checked decoding',
            modes: ['resample', 'repair'],
            verifier_ids: ['code_checks', 'context_grounding', 'json_schema', 'label_set', 'citations'],
            max_attempts_limit: 5,
        },
    };
}

const SUMMARY = {
    policy: { mode: 'resample' as const, max_attempts: 3, verifiers: 'auto' as const, sample_temperature: 0.7, seed: 0 },
    rows_retried: 4,
    rows_fixed_by_retry: 2,
    rows_still_failing: 1,
};

describe('checked decoding text', () => {
    it('labels the policy and the rows fixed', () => {
        expect(checkedDecodingBadgeText(SUMMARY.policy)).toBe('Checked decoding · up to 3 tries · resample');
        expect(checkedDecodingBadgeText(null)).toBeNull();
        expect(rowsFixedText(SUMMARY)).toBe('2 rows fixed by a retry · 1 still failing the checks');
        expect(rowsFixedText({ ...SUMMARY, rows_fixed_by_retry: 1, rows_still_failing: 0 })).toBe('1 row fixed by a retry');
        expect(rowsFixedText({ policy: SUMMARY.policy, skipped: 'engine_unsupported' })).toMatch(/not applied/);
    });
});

describe('CheckedDecodingBadge', () => {
    it('renders the label, the rows fixed and the source eval', () => {
        render(<CheckedDecodingBadge summary={SUMMARY} source="eval #91" />);
        expect(screen.getByTestId('checked-decoding-badge-label')).toHaveTextContent('Checked decoding · up to 3 tries · resample');
        expect(screen.getByTestId('checked-decoding-badge-fixed')).toHaveTextContent('2 rows fixed by a retry');
        expect(screen.getByTestId('checked-decoding-badge')).toHaveTextContent('(eval #91)');
    });

    it('renders nothing for an eval without checked decoding', () => {
        const { container } = render(<CheckedDecodingBadge summary={null} />);
        expect(container).toBeEmptyDOMElement();
    });
});

describe('CheckedDecodingSettings', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
    });

    it('is off by default and saves the toggle, mode and max tries', async () => {
        apiMock.get.mockResolvedValue(payload(OFF));
        apiMock.put.mockImplementation((_url: string, body: Partial<Settings>) => Promise.resolve(payload({ ...OFF, enabled: true, ...body })));
        render(<CheckedDecodingSettings projectId={7} />);
        const toggle = await screen.findByTestId('checked-decoding-toggle');
        expect(toggle).not.toBeChecked();
        expect(screen.queryByTestId('checked-decoding-mode')).toBeNull();
        expect(apiMock.get).toHaveBeenCalledWith('/projects/7/evaluation/checked-decoding');

        await userEvent.click(toggle);
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/7/evaluation/checked-decoding', { enabled: true }));
        const mode = await screen.findByTestId('checked-decoding-mode');
        await userEvent.selectOptions(mode, 'repair');
        await waitFor(() => expect(apiMock.put).toHaveBeenLastCalledWith('/projects/7/evaluation/checked-decoding', { mode: 'repair' }));
        expect(screen.getByTestId('checked-decoding-max')).toHaveAttribute('max', '5');
    });

    it('shows the server message when a save is refused', async () => {
        apiMock.get.mockResolvedValue(payload(OFF));
        apiMock.put.mockImplementation(() => Promise.reject({ response: { data: { detail: 'checked_decoding.mode must be one of: resample, repair' } } }));
        render(<CheckedDecodingSettings projectId={7} />);
        await userEvent.click(await screen.findByTestId('checked-decoding-toggle'));
        expect(await screen.findByTestId('checked-decoding-settings-error')).toHaveTextContent('must be one of');
    });
});

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

describe('Compare with checked decoding', () => {
    it('shows the policy badge per side and the decoding-policy difference', () => {
        const ev: RowEvidence = {
            verdict: 'not_comparable',
            reason: 'not_comparable',
            message: 'different decoding policy: checked decoding off vs up to 3 tries · resample',
            metric_id: null,
            score_key: null,
            n: 0,
            better: 0,
            worse: 0,
            same: 0,
            mean_diff: null,
            ci_low: null,
            ci_high: null,
            only_in_a: 0,
            only_in_b: 0,
            unscored: 0,
            metrics: [],
            comparability: {
                comparable: false,
                differences: [{
                    kind: 'decoding_policy',
                    message: 'different decoding policy: checked decoding off vs up to 3 tries · resample',
                    a: null,
                    b: SUMMARY.policy,
                    inferred: false,
                }],
                notes: [],
            },
            rescore: null,
        } as unknown as RowEvidence;
        render(
            <CompareRowEvidence
                projectId={5}
                evidence={ev}
                a={prov()}
                b={prov({ experiment_id: 12, experiment_name: 'Run B', eval_result_id: 92, checked_decoding: SUMMARY })}
            />,
        );
        expect(screen.getByTestId('eval-compare-evidence-diffs')).toHaveTextContent('different decoding policy');
        expect(screen.queryByTestId('eval-compare-checked-decoding-a')).toBeNull();
        expect(screen.getByTestId('eval-compare-checked-decoding-b-label')).toHaveTextContent('up to 3 tries · resample');
        expect(screen.getByTestId('eval-compare-checked-decoding-b-fixed')).toHaveTextContent('2 rows fixed by a retry');
    });
});

describe('Eval summary / lift card with checked decoding', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    const SUMMARY_CARD = {
        project_id: 7,
        experiment_id: 21,
        verdict: 'better',
        message: null,
        headline: {
            metric_id: 'exact_match', baseline_value: 0.2, trained_value: 0.55, absolute_delta: 0.35,
            relative_delta_pct: 175, direction: 'improved',
        },
        baseline: { experiment_id: 20, base_model: 'SmolLM2-135M' },
        trained: { experiment_id: 21, experiment_name: 'exp-21' },
        eval_result_id: 91,
        eval_type: 'exact_match',
        evaluated_samples: 20,
        failures: [],
        failed_count: 0,
    };

    it('labels a checked eval with its source and warns when the two sides differ', async () => {
        apiMock.get.mockResolvedValue({ data: { ...SUMMARY_CARD, checked_decoding: SUMMARY, checked_decoding_differs: true } });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);
        const badge = await screen.findByTestId('eval-summary-checked-decoding-badge');
        expect(badge).toHaveTextContent('Checked decoding · up to 3 tries · resample');
        expect(badge).toHaveTextContent('2 rows fixed by a retry');
        expect(badge).toHaveTextContent('(eval #91)');
        expect(screen.getByTestId('eval-summary-checked-decoding-differs')).toHaveTextContent('not like for like');
    });

    it('shows nothing extra for an eval without checked decoding', async () => {
        apiMock.get.mockResolvedValue({ data: SUMMARY_CARD });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);
        await screen.findByTestId('eval-summary');
        expect(screen.queryByTestId('eval-summary-checked-decoding')).toBeNull();
        expect(screen.queryByTestId('eval-summary-checked-decoding-differs')).toBeNull();
    });
});
