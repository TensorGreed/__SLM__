import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import TrainingConfigPatchPreviewModal from './TrainingConfigPatchPreviewModal';

const PREVIEW = {
    project_id: 42,
    signal_id: 'training_config.eval_cadence_too_sparse',
    patch_kind: 'eval_steps_recommend',
    patch_label: 'Tighten eval cadence',
    plain_english:
        'Bumps eval_steps so the trainer checks itself often enough to draw a learning curve.',
    patch: { eval_steps: 10 },
    before: { eval_steps: 100 },
    after: { eval_steps: 10 },
    safe_to_apply: true,
};

const APPLY_RESULT = {
    ...PREVIEW,
    applied: true,
    overrides_after: { eval_steps: 10 },
};

describe('TrainingConfigPatchPreviewModal', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('loads preview on mount and renders the before → after diff', async () => {
        apiMock.post.mockResolvedValueOnce({ data: PREVIEW });
        const onClose = vi.fn();
        const onApplied = vi.fn();
        render(
            <TrainingConfigPatchPreviewModal
                projectId={42}
                signalId="training_config.eval_cadence_too_sparse"
                onClose={onClose}
                onApplied={onApplied}
            />,
        );
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/training-config-gaps/patch/preview',
                { signal_id: 'training_config.eval_cadence_too_sparse' },
            );
        });
        const row = await screen.findByTestId(
            'training-config-patch-row-eval_steps',
        );
        expect(row).toHaveTextContent('100');
        expect(row).toHaveTextContent('10');
        expect(
            screen.getByTestId('training-config-patch-plain'),
        ).toHaveTextContent(/learning curve/i);
    });

    it('Apply calls the apply endpoint and forwards the result + closes', async () => {
        apiMock.post
            .mockResolvedValueOnce({ data: PREVIEW })
            .mockResolvedValueOnce({ data: APPLY_RESULT });
        const onClose = vi.fn();
        const onApplied = vi.fn();
        render(
            <TrainingConfigPatchPreviewModal
                projectId={42}
                signalId="training_config.eval_cadence_too_sparse"
                onClose={onClose}
                onApplied={onApplied}
            />,
        );
        const applyBtn = await screen.findByTestId('training-config-patch-apply');
        await userEvent.click(applyBtn);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/training-config-gaps/patch/apply',
                { signal_id: 'training_config.eval_cadence_too_sparse' },
            );
        });
        expect(onApplied).toHaveBeenCalledWith(APPLY_RESULT);
        expect(onClose).toHaveBeenCalled();
    });

    it('Cancel button calls onClose without applying', async () => {
        apiMock.post.mockResolvedValueOnce({ data: PREVIEW });
        const onClose = vi.fn();
        const onApplied = vi.fn();
        render(
            <TrainingConfigPatchPreviewModal
                projectId={42}
                signalId="training_config.eval_cadence_too_sparse"
                onClose={onClose}
                onApplied={onApplied}
            />,
        );
        const cancel = await screen.findByTestId('training-config-patch-cancel');
        await userEvent.click(cancel);
        expect(onClose).toHaveBeenCalled();
        expect(onApplied).not.toHaveBeenCalled();
        // Only the preview call should have fired (no apply).
        expect(apiMock.post).toHaveBeenCalledTimes(1);
    });
});
