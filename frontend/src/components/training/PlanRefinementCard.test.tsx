import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn() },
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
    refinement: null,
    privacy: {
        cloud_sharing: 'aggregate_only',
        note: 'Only the aggregate signals in cloud_safe_profile are ever eligible to be sent to a cloud model (Phase 2). Your ingested rows, document text, gold answers, and label names never leave BrewSLM.',
    },
    cloud_refinement: { available: false, supported_providers: ['anthropic', 'openai', 'deepseek', 'qwen', 'ollama'], reason: 'phase_1_deterministic_only' },
};

const STRATEGY = {
    plan_delta: { rag_first: true, task_profile: 'rag_qa' },
    directional_config: [{ kind: 'num_epochs_recommend', direction: 'down', reason: 'memorization risk on a small set' }],
    data_gaps: [{ kind: 'class_balance', detail: 'minority classes are thin', suggested_count: 30 }],
    rationale: 'Small, imbalanced, retrieval-shaped gold → RAG-first + balance the minority classes.',
    confidence: 0.72,
    provenance: { model: 'deepseek:deepseek-chat', shared: 'cloud_safe_profile' },
    from_cache: false,
};

describe('PlanRefinementCard', () => {
    beforeEach(() => { apiMock.get.mockReset(); apiMock.post.mockReset(); });

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

    it('runs the cloud strategy pass and renders the validated recommendation', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { ...REPORT, cloud_refinement: { ...REPORT.cloud_refinement, available: true } },
        });
        apiMock.post.mockResolvedValueOnce({ data: { available: true, refinement: STRATEGY } });

        const { default: userEvent } = await import('@testing-library/user-event');
        render(<PlanRefinementCard projectId={1} />);
        const btn = await screen.findByTestId('plan-refinement-get-strategy');
        await userEvent.setup().click(btn);

        await waitFor(() => {
            expect(screen.getByTestId('plan-refinement-strategy')).toBeInTheDocument();
        });
        expect(apiMock.post).toHaveBeenCalledWith('/projects/1/refine-plan/cloud');
        expect(screen.getByText('Enable RAG-first retrieval')).toBeInTheDocument();
        expect(screen.getByText(/via deepseek:deepseek-chat/)).toBeInTheDocument();
        // Data-gap remediation deep-links into the synthetic flow.
        expect(screen.getByRole('button', { name: /Generate ~30/ })).toBeInTheDocument();
    });

    it('accepts & applies the recommendation through the apply endpoint', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...REPORT,
                cloud_refinement: { ...REPORT.cloud_refinement, available: true },
                refinement: { ...STRATEGY, from_cache: true, applied: null },
            },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                project_id: 1,
                plan_delta: [{ field: 'rag_first', status: 'applied' }, { field: 'task_profile', status: 'applied' }],
                directional_config: [{ kind: 'num_epochs_recommend', status: 'applied' }],
                applied: { plan_delta: ['rag_first', 'task_profile'], directional: ['num_epochs_recommend'] },
            },
        });

        const { default: userEvent } = await import('@testing-library/user-event');
        render(<PlanRefinementCard projectId={1} />);
        const applyBtn = await screen.findByTestId('plan-refinement-apply');
        await userEvent.setup().click(applyBtn);

        await waitFor(() => {
            expect(screen.getByTestId('plan-refinement-applied')).toBeInTheDocument();
        });
        expect(apiMock.post).toHaveBeenCalledWith('/projects/1/refine-plan/apply', {});
        expect(screen.getByTestId('plan-refinement-applied')).toHaveTextContent(/rag first/i);
        // The Accept button is gone once applied.
        expect(screen.queryByTestId('plan-refinement-apply')).not.toBeInTheDocument();
    });

    it('shows the already-applied state without an Accept button', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...REPORT,
                cloud_refinement: { ...REPORT.cloud_refinement, available: true },
                refinement: { ...STRATEGY, from_cache: true, applied: { plan_delta: ['rag_first'], directional: [] } },
            },
        });
        render(<PlanRefinementCard projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('plan-refinement-applied')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('plan-refinement-apply')).not.toBeInTheDocument();
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('shows a cached refinement from the GET without re-calling cloud', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...REPORT,
                cloud_refinement: { ...REPORT.cloud_refinement, available: true },
                refinement: { ...STRATEGY, from_cache: true },
            },
        });
        render(<PlanRefinementCard projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('plan-refinement-strategy')).toBeInTheDocument();
        });
        // No button (already have a recommendation); no POST fired.
        expect(screen.queryByTestId('plan-refinement-get-strategy')).not.toBeInTheDocument();
        expect(apiMock.post).not.toHaveBeenCalled();
    });
});
