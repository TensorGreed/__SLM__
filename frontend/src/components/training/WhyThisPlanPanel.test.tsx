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

import WhyThisPlanPanel from './WhyThisPlanPanel';

const ESTIMATED_COST = {
    gpu_hours: 0.45,
    usd: 0.27,
    co2_kg: 0.054,
    provenance: 'estimated',
    confidence: 0.30,
    confidence_band: 'low',
    calibration: {
        cohort: 'none',
        sample_count: 0,
        median_cohort_seconds: null,
        variability_cv: null,
        heuristic_seconds: 1620,
        fallback_used: true,
        planned_total_steps: 300,
        planned_gpu_count: 1,
        planned_size_bucket: 'medium',
        training_mode: 'sft',
        target_profile_id: 'vllm_server',
    },
    pricing: { hourly_rate_usd: 0.85, source: 'cloud_burst_catalog_avg' },
    co2: { power_kw: 0.3, grid_intensity_g_per_kwh: 400, intensity_source: 'global_average' },
};

const MEASURED_COST = {
    ...ESTIMATED_COST,
    gpu_hours: 0.28,
    usd: 0.24,
    co2_kg: 0.034,
    provenance: 'measured',
    confidence: 0.78,
    confidence_band: 'medium',
    calibration: {
        ...ESTIMATED_COST.calibration,
        cohort: 'mode+model_size',
        sample_count: 3,
        median_cohort_seconds: 1000,
        variability_cv: 0.05,
        fallback_used: false,
    },
};

const EXPERIMENT = {
    id: 7,
    name: 'phi-2 sft',
    base_model: 'microsoft/phi-2',
    training_mode: 'sft',
    status: 'pending',
    config: {
        recipe: 'recipe.pipeline.sft_default',
        training_runtime_id: 'builtin.simulate',
        num_epochs: 3,
        batch_size: 4,
        gradient_accumulation_steps: 4,
        max_seq_length: 2048,
        use_lora: true,
        gradient_checkpointing: true,
    },
};

describe('WhyThisPlanPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('posts the planned config to the cost estimator and renders the estimated badge', async () => {
        apiMock.post.mockResolvedValueOnce({ data: ESTIMATED_COST });

        render(<WhyThisPlanPanel projectId={5} experiment={EXPERIMENT} />);

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/5/training/plan/estimate-cost',
                expect.objectContaining({
                    base_model: 'microsoft/phi-2',
                    target_profile_id: 'vllm_server',
                    config: expect.objectContaining({ training_mode: 'sft' }),
                }),
            );
        });

        // Provenance badge present, marked estimated.
        const badge = await screen.findByText('estimated');
        expect(badge.className).toMatch(/provenance--estimated/);

        // Strategy + Memory budget reflect the experiment's config.
        expect(screen.getByText('recipe.pipeline.sft_default')).toBeInTheDocument();
        expect(screen.getByText('builtin.simulate')).toBeInTheDocument();
        expect(screen.getByText('4 × 4')).toBeInTheDocument();
        expect(screen.getByText('2048')).toBeInTheDocument();

        // Cost numbers rendered.
        expect(screen.getByText('$0.27')).toBeInTheDocument();
        expect(screen.getByText('0.054 kg')).toBeInTheDocument();
    });

    it('renders the measured badge with the cohort + sample count when calibration kicks in', async () => {
        apiMock.post.mockResolvedValueOnce({ data: MEASURED_COST });

        render(<WhyThisPlanPanel projectId={5} experiment={EXPERIMENT} />);

        const badge = await screen.findByText('measured');
        expect(badge.className).toMatch(/provenance--measured/);
        // The cohort-meta line shows the cohort + sample size.
        expect(screen.getByText('mode+model_size (n=3)')).toBeInTheDocument();
    });

    it('lazy-fetches the manifest only when the operator expands "Show the manifest"', async () => {
        apiMock.post.mockResolvedValueOnce({ data: ESTIMATED_COST });
        apiMock.get.mockResolvedValueOnce({
            data: { experiment_id: 7, git_sha: 'abc123', env_digest: 'xyz789' },
        });

        const user = userEvent.setup();
        render(<WhyThisPlanPanel projectId={5} experiment={EXPERIMENT} />);
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        // Manifest endpoint not called until the operator toggles expand.
        expect(apiMock.get).not.toHaveBeenCalled();

        await user.click(screen.getByRole('button', { name: /Show the manifest/i }));
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/5/training/runs/7/manifest',
            );
        });
        expect(await screen.findByText(/abc123/)).toBeInTheDocument();
    });

    it('surfaces a friendly error when the manifest is not yet captured', async () => {
        apiMock.post.mockResolvedValueOnce({ data: ESTIMATED_COST });
        apiMock.get.mockRejectedValueOnce({
            response: { data: { detail: 'manifest_not_captured' } },
        });

        const user = userEvent.setup();
        render(<WhyThisPlanPanel projectId={5} experiment={EXPERIMENT} />);
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());

        await user.click(screen.getByRole('button', { name: /Show the manifest/i }));
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalled();
        });
        expect(await screen.findByText('manifest_not_captured')).toBeInTheDocument();
    });
});
