import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalGapPatchPreviewModal from './EvalGapPatchPreviewModal';

const BASELINE_PREVIEW = {
    project_id: 42,
    signal_id: 'eval_gaps.no_regression_baseline',
    patch_kind: 'regression_baseline_promote_last_green' as const,
    patch_label: 'Promote last-green checkpoint as baseline',
    plain_english:
        'Sets promoted_at on the best checkpoint of your most recent passing run.',
    before: {
        promoted_checkpoint_id: null,
        promoted_experiment_id: null,
        promoted_step: null,
    },
    after: {
        promoted_checkpoint_id: 17,
        promoted_experiment_id: 5,
        promoted_step: 200,
    },
    candidate: {
        experiment_id: 5,
        experiment_name: 'green-run-1',
        checkpoint_id: 17,
        checkpoint_step: 200,
        checkpoint_is_best: true,
        pass_rate: 0.85,
    },
    safe_to_apply: true,
};

const KL_PREVIEW = {
    project_id: 42,
    signal_id: 'eval_gaps.train_eval_label_kl_high',
    patch_kind: 'label_kl_rebalance_eval' as const,
    patch_label: 'Trim GOLD_DEV toward train distribution',
    plain_english:
        'Drops over-represented rows from GOLD_DEV. GOLD_TEST is intentionally untouched.',
    before: { counts: { pos: 27, neg: 3 }, kl_nats: 0.4 },
    after: { counts: { pos: 15, neg: 3 }, kl_nats: 0.05 },
    rows_to_drop: 12,
    gold_dev_path: '/tmp/gold_dev.jsonl',
    safe_to_apply: true,
    skipped_reason: null,
};

const APPLIED = { ...BASELINE_PREVIEW, applied: true as const };

describe('EvalGapPatchPreviewModal', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders the baseline-promote candidate and applies', async () => {
        apiMock.post
            .mockResolvedValueOnce({ data: BASELINE_PREVIEW })
            .mockResolvedValueOnce({ data: APPLIED });
        const onClose = vi.fn();
        const onApplied = vi.fn();
        render(
            <EvalGapPatchPreviewModal
                projectId={42}
                signalId="eval_gaps.no_regression_baseline"
                onClose={onClose}
                onApplied={onApplied}
            />,
        );
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/eval-gaps/patch/preview',
                { signal_id: 'eval_gaps.no_regression_baseline' },
            );
        });
        expect(
            await screen.findByTestId('eval-patch-row-experiment'),
        ).toHaveTextContent('green-run-1');
        expect(
            screen.getByTestId('eval-patch-row-pass-rate'),
        ).toHaveTextContent('85.0%');

        // Apply.
        await userEvent.click(
            screen.getByTestId('eval-patch-apply'),
        );
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/eval-gaps/patch/apply',
                { signal_id: 'eval_gaps.no_regression_baseline' },
            );
        });
        expect(onApplied).toHaveBeenCalledWith(APPLIED);
        expect(onClose).toHaveBeenCalled();
    });

    it('renders the label-KL per-class trim table', async () => {
        apiMock.post.mockResolvedValueOnce({ data: KL_PREVIEW });
        render(
            <EvalGapPatchPreviewModal
                projectId={42}
                signalId="eval_gaps.train_eval_label_kl_high"
                onClose={vi.fn()}
                onApplied={vi.fn()}
            />,
        );
        const klTable = await screen.findByTestId('eval-patch-kl-table');
        expect(klTable).toHaveTextContent('pos');
        expect(klTable).toHaveTextContent('27');
        expect(klTable).toHaveTextContent('15');
        // Summary row shows KL before → after.
        expect(
            screen.getByTestId('eval-patch-kl-row-summary'),
        ).toHaveTextContent('0.400');
        expect(
            screen.getByTestId('eval-patch-kl-row-summary'),
        ).toHaveTextContent('0.050');
    });

    it('Cancel closes without applying', async () => {
        apiMock.post.mockResolvedValueOnce({ data: BASELINE_PREVIEW });
        const onClose = vi.fn();
        const onApplied = vi.fn();
        render(
            <EvalGapPatchPreviewModal
                projectId={42}
                signalId="eval_gaps.no_regression_baseline"
                onClose={onClose}
                onApplied={onApplied}
            />,
        );
        await userEvent.click(
            await screen.findByTestId('eval-patch-cancel'),
        );
        expect(onClose).toHaveBeenCalled();
        expect(onApplied).not.toHaveBeenCalled();
        // Only the preview call should have fired.
        expect(apiMock.post).toHaveBeenCalledTimes(1);
    });
});
