import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import PlanRefinementCard from './PlanRefinementCard';

const REPORT = {
    project_id: 1,
    plan: {
        recipe_id: 'classification',
        task_profile: 'classification',
        base_model_name: 'Qwen/Qwen1.5-1.8B-Chat',
        target_profile_id: 'mobile_cpu',
    },
    cloud_safe_profile: {
        recipe_id: 'classification',
        task_profile: 'classification',
        base_model_name: 'Qwen/Qwen1.5-1.8B-Chat',
        target_profile_id: 'mobile_cpu',
        labelled_row_count: 8,
        label_distribution_shape: {
            num_classes: 3, min_class_count: 1, max_class_count: 6,
            imbalance_ratio: 0.166, classes_below_floor: 2,
        },
        truncation_risk: 'ok',
        tokenizer_oov: 'ok',
        archetype_below_band_features: [],
        forecast_verdict: 'borderline',
    },
    plan_health: {
        verdict: 'attention',
        signals: [
            { id: 'plan.classes_below_floor', severity: 'warn', headline: '2 class(es) sit below the per-class floor — the model will underfit them.', target_tab: 'synthetic' },
            { id: 'plan.forecast_borderline', severity: 'warn', headline: 'Trainability forecast is borderline.', target_tab: 'training-config' },
        ],
    },
    privacy: {
        cloud_sharing: 'aggregate_only',
        note: 'Only the aggregate signals in cloud_safe_profile are ever eligible to be sent to a cloud model (Phase 2). Your ingested rows, document text, gold answers, and label names never leave BrewSLM.',
    },
    cloud_refinement: { available: false, supported_providers: ['anthropic', 'openai', 'deepseek', 'qwen', 'ollama'], reason: 'phase_1_deterministic_only' },
};

describe('PlanRefinementCard', () => {
    beforeEach(() => apiMock.get.mockReset());

    it('renders the plan-fit verdict, signals, aggregate profile, and privacy line', async () => {
        apiMock.get.mockResolvedValueOnce({ data: REPORT });
        render(<PlanRefinementCard projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('plan-refinement-card')).toBeInTheDocument();
        });
        expect(screen.getByText(/Plan needs attention/i)).toBeInTheDocument();
        expect(screen.getByText(/below the per-class floor/i)).toBeInTheDocument();
        // Aggregate profile chips (shape, not names).
        const profile = screen.getByTestId('plan-refinement-profile');
        expect(profile).toHaveTextContent('8');
        expect(profile).toHaveTextContent('3');
        expect(profile).toHaveTextContent(/below floor/i);
        // Privacy transparency line.
        expect(screen.getByTestId('plan-refinement-privacy'))
            .toHaveTextContent(/never leave BrewSLM/i);
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/refine-plan');
    });

    it('renders an error state without crashing', async () => {
        apiMock.get.mockRejectedValueOnce({ response: { data: { detail: 'Project not found' } } });
        render(<PlanRefinementCard projectId={9} />);
        await waitFor(() => {
            expect(screen.getByText(/Project not found/i)).toBeInTheDocument();
        });
    });
});
