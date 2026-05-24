import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioSyntheticRecommendationsPanel from './DataStudioSyntheticRecommendationsPanel';

const recommendationsPayload = {
    project_id: 1,
    verdict: 'attention',
    read_only: true,
    auto_apply: false,
    source_of_truth: 'deterministic_data_studio_checks',
    domain: {
        id: 'support_faq',
        label: 'Support FAQ',
        confidence: 0.86,
        source: 'sampled_data',
    },
    recipe: {
        id: 'classification',
        name: 'Classification',
        task_profile: 'classification',
        adapter_id: 'classification-label',
    },
    signals: {
        mapping_verdict: 'attention',
        mapping_required_gaps: ['label'],
        gold_trusted_examples: 2,
        gold_review_needed: 0,
        gold_label_field_count: 1,
        synthetic_pending: 2,
        synthetic_accepted: 1,
        compatible_playbook_modes: ['positives_paraphrase', 'hard_negatives'],
        ollama_available: true,
    },
    recommendations: [
        {
            id: 'review_pending_synthetic_before_more_generation',
            title: 'Review pending synthetic rows before generating more',
            strategy: 'review queue',
            priority: 'high',
            target_tab: 'synthetic',
            action_label: 'Review queue',
            rationale: 'Reviewing existing synthetic rows prevents low-quality rows from piling up.',
            domain_reason: 'Support FAQ quality depends on accepted examples matching policy and labels.',
            evidence: ['2 synthetic row(s) are pending review.'],
            confidence: 0.9,
            playbook_mode: null,
            playbook_available: false,
            requires_user_confirmation: true,
            generation_path: {
                backend: 'ollama',
                available: true,
                describe: 'ollama:llama3',
                local_default: true,
                paid_required: false,
            },
        },
        {
            id: 'domain_support_faq_customer_phrasing',
            title: 'Generate customer phrasing variants',
            strategy: 'positive paraphrase',
            priority: 'medium',
            target_tab: 'synthetic',
            action_label: 'Open Synthetic',
            rationale: 'This strategy follows from deterministic domain signals.',
            domain_reason: 'Support assistants need to recognize the same intent across messy customer wording.',
            evidence: [
                'Detected domain: Support FAQ (86% confidence).',
                'Domain keywords: refund, password reset.',
            ],
            confidence: 0.86,
            playbook_mode: 'positives_paraphrase',
            playbook_available: true,
            requires_user_confirmation: true,
            generation_path: {
                backend: 'ollama',
                available: true,
                describe: 'ollama:llama3',
                local_default: true,
                paid_required: false,
            },
        },
    ],
    issues: [
        {
            id: 'synthetic_recommendation_pending_review_queue',
            severity: 'warning',
            title: 'Review pending synthetic rows',
            message: 'Pending synthetic rows are gated out of training until accepted.',
            action_label: 'Review synthetic rows',
            target_tab: 'synthetic',
        },
    ],
    entry_points: [
        {
            label: 'Open Synthetic workflow',
            target_tab: 'synthetic',
            reason: 'Run playbooks in the existing Synthetic tab.',
        },
        {
            label: 'Open Gold Set workflow',
            target_tab: 'goldset',
            reason: 'Improve trusted anchors.',
        },
    ],
    power_details: {},
};

describe('DataStudioSyntheticRecommendationsPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders domain-aware recommendations and routes actions to existing tabs', async () => {
        apiMock.get.mockResolvedValueOnce({ data: recommendationsPayload });
        const onOpenTab = vi.fn();

        render(<DataStudioSyntheticRecommendationsPanel projectId={1} onOpenTab={onOpenTab} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-synth-recommendations')).toBeInTheDocument();
        });

        expect(screen.getByText('Review advice')).toBeInTheDocument();
        expect(screen.getByText('Support FAQ')).toBeInTheDocument();
        expect(screen.getByText('86% domain confidence')).toBeInTheDocument();
        expect(screen.getByText('Generate customer phrasing variants')).toBeInTheDocument();
        expect(screen.getByText('Review pending synthetic rows before generating more')).toBeInTheDocument();
        expect(screen.getAllByText(/ollama:llama3/i).length).toBeGreaterThan(0);
        expect(screen.getByText('Review pending synthetic rows')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: /^Open Synthetic$/i }));
        expect(onOpenTab).toHaveBeenCalledWith('synthetic');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/synthetic-recommendations');
    });

    it('renders setup recommendations when recipe and Ollama are missing', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...recommendationsPayload,
                recipe: null,
                domain: {
                    id: 'generic_domain',
                    label: 'Generic Domain',
                    confidence: 0.25,
                    source: 'runtime_default',
                },
                signals: {
                    ...recommendationsPayload.signals,
                    mapping_required_gaps: [],
                    gold_trusted_examples: 0,
                    synthetic_pending: 0,
                    compatible_playbook_modes: [],
                    ollama_available: false,
                },
                recommendations: [
                    {
                        ...recommendationsPayload.recommendations[0],
                        id: 'setup_recipe_for_synthetic_recommendations',
                        title: 'Choose a recipe before generating synthetic data',
                        strategy: 'setup',
                        priority: 'high',
                        target_tab: 'data',
                        action_label: 'Choose recipe',
                        domain_reason: 'Generic Domain recommendations become more precise after the training recipe is known.',
                        generation_path: {
                            backend: 'ollama',
                            available: false,
                            describe: 'ollama',
                            local_default: true,
                            paid_required: false,
                        },
                    },
                    {
                        ...recommendationsPayload.recommendations[1],
                        id: 'start_local_ollama_for_synthetic_generation',
                        title: 'Start local Ollama for free generation',
                        strategy: 'backend setup',
                        priority: 'medium',
                        target_tab: 'synthetic',
                        action_label: 'Open Synthetic',
                        domain_reason: 'Generic Domain recommendations can be executed locally once an Ollama model is reachable.',
                        generation_path: {
                            backend: 'ollama',
                            available: false,
                            describe: 'ollama',
                            local_default: true,
                            paid_required: false,
                        },
                    },
                ],
                issues: [
                    {
                        id: 'synthetic_recommendation_recipe_missing',
                        severity: 'blocker',
                        title: 'Recipe needed before recommending playbooks',
                        message: 'Pick a recipe so recommendations can target compatible synthetic strategies.',
                        action_label: 'Choose recipe',
                        target_tab: 'data',
                    },
                ],
            },
        });
        const onOpenTab = vi.fn();

        render(<DataStudioSyntheticRecommendationsPanel projectId={1} onOpenTab={onOpenTab} />);

        await waitFor(() => {
            expect(screen.getByText('Choose a recipe before generating synthetic data')).toBeInTheDocument();
        });

        expect(screen.getByText('Generic Domain')).toBeInTheDocument();
        expect(screen.getByText('Ollama setup')).toBeInTheDocument();
        expect(screen.getByText('Start local Ollama for free generation')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: /Choose recipe/i }));
        expect(onOpenTab).toHaveBeenCalledWith('data');
    });
});
