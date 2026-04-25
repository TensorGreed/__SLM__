import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import PreRunConfirmModal from './PreRunConfirmModal';

const ESTIMATED_COST = {
    gpu_hours: 0.45,
    usd: 0.27,
    co2_kg: 0.054,
    provenance: 'estimated',
    confidence: 0.30,
    confidence_band: 'low',
    calibration: { cohort: 'none', sample_count: 0 },
    pricing: { hourly_rate_usd: 0.85, source: 'cloud_burst_catalog_avg' },
};

const MEASURED_COST = {
    ...ESTIMATED_COST,
    gpu_hours: 0.30,
    usd: 0.26,
    co2_kg: 0.036,
    provenance: 'measured',
    confidence: 0.78,
    confidence_band: 'medium',
    calibration: { cohort: 'mode+model_size', sample_count: 3 },
};

describe('PreRunConfirmModal', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
    });

    it('posts the planned config to the cost estimator and renders the estimated badge', async () => {
        apiMock.post.mockResolvedValueOnce({ data: ESTIMATED_COST });

        render(
            <PreRunConfirmModal
                projectId={5}
                config={{ training_mode: 'sft', num_epochs: 3 }}
                baseModel="microsoft/phi-2"
                targetProfileId="vllm_server"
                onConfirm={() => undefined}
                onCancel={() => undefined}
            />,
        );

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/5/training/plan/estimate-cost',
                {
                    config: { training_mode: 'sft', num_epochs: 3 },
                    base_model: 'microsoft/phi-2',
                    target_profile_id: 'vllm_server',
                },
            );
        });

        // Provenance badge marked estimated.
        const badge = await screen.findByText('estimated');
        expect(badge.className).toMatch(/provenance--estimated/);
        // Cost numbers rendered.
        expect(screen.getByText('$0.27')).toBeInTheDocument();
        expect(screen.getByText('0.054 kg')).toBeInTheDocument();
    });

    it('renders the measured badge with the cohort + sample count', async () => {
        apiMock.post.mockResolvedValueOnce({ data: MEASURED_COST });

        render(
            <PreRunConfirmModal
                projectId={5}
                config={{ training_mode: 'sft' }}
                baseModel="microsoft/phi-2"
                onConfirm={() => undefined}
                onCancel={() => undefined}
            />,
        );

        const badge = await screen.findByText('measured');
        expect(badge.className).toMatch(/provenance--measured/);
        expect(screen.getByText('mode+model_size (n=3)')).toBeInTheDocument();
    });

    it('Confirm button invokes onConfirm and Cancel invokes onCancel', async () => {
        apiMock.post.mockResolvedValueOnce({ data: ESTIMATED_COST });

        const onConfirm = vi.fn();
        const onCancel = vi.fn();

        const user = userEvent.setup();
        render(
            <PreRunConfirmModal
                projectId={5}
                config={{ training_mode: 'sft' }}
                onConfirm={onConfirm}
                onCancel={onCancel}
                confirmLabel="Launch"
            />,
        );

        // Wait for fetch so the Confirm button isn't disabled.
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        await waitFor(() => expect(screen.getByText('estimated')).toBeInTheDocument());

        await user.click(screen.getByRole('button', { name: 'Cancel' }));
        expect(onCancel).toHaveBeenCalledTimes(1);
        expect(onConfirm).not.toHaveBeenCalled();

        await user.click(screen.getByRole('button', { name: 'Launch' }));
        expect(onConfirm).toHaveBeenCalledTimes(1);
    });

    it('surfaces a friendly error when the estimator fails', async () => {
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'estimator_failed' } },
        });

        render(
            <PreRunConfirmModal
                projectId={5}
                config={{}}
                onConfirm={() => undefined}
                onCancel={() => undefined}
            />,
        );

        expect(await screen.findByText('estimator_failed')).toBeInTheDocument();
    });
});
