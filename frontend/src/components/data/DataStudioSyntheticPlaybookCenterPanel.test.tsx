import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioSyntheticPlaybookCenterPanel from './DataStudioSyntheticPlaybookCenterPanel';

const syntheticPayload = {
    project_id: 1,
    verdict: 'attention',
    read_only: true,
    recipe: {
        id: 'classification',
        name: 'Classification',
        task_profile: 'classification',
        adapter_id: 'classification-label',
    },
    catalog: {
        total_playbooks: 18,
        compatible_playbooks: 3,
        preview_playbooks: [
            {
                recipe_id: 'classification',
                mode: 'positives_paraphrase',
                label: 'Paraphrase positives',
            },
            {
                recipe_id: 'classification',
                mode: 'hard_negatives',
                label: 'Hard negatives',
            },
        ],
        supported_recipes: ['classification', 'qa-sft'],
        compatible_modes: ['positives_paraphrase', 'hard_negatives'],
    },
    backends: [
        {
            name: 'ollama',
            available: true,
            describe: 'ollama:llama3',
            is_default: true,
            is_local: true,
            paid_required: false,
        },
        {
            name: 'teacher',
            available: false,
            describe: 'teacher',
            is_default: false,
            is_local: false,
            paid_required: false,
        },
    ],
    recommended_backend: {
        name: 'ollama',
        available: true,
        describe: 'ollama:llama3',
        local_default: true,
        paid_required: false,
    },
    prerequisites: [
        {
            id: 'recipe',
            label: 'Recipe selected',
            status: 'met',
            message: 'Classification is active.',
            target_tab: 'data',
        },
        {
            id: 'gold_examples',
            label: 'Gold examples',
            status: 'met',
            message: '2 file-backed gold rows can seed playbook generation.',
            target_tab: 'goldset',
        },
        {
            id: 'local_ollama',
            label: 'Local Ollama',
            status: 'met',
            message: 'Local default backend is ready: ollama:llama3.',
            target_tab: 'synthetic',
        },
    ],
    review_queue: {
        dataset_id: 9,
        total_rows: 3,
        total_pending: 2,
        total_accepted: 1,
        pending_group_count: 1,
        accepted_group_count: 1,
        top_pending_groups: [
            {
                synth_source: 'playbook:classification:positives_paraphrase',
                count: 2,
                truncated: false,
            },
        ],
        top_accepted_groups: [
            {
                synth_source: 'playbook:classification:hard_negatives',
                count: 1,
                truncated: false,
            },
        ],
    },
    issues: [
        {
            id: 'synthetic_rows_pending_review',
            severity: 'warning',
            title: 'Synthetic rows pending review',
            message: '2 synthetic rows must be accepted before they enter dataset prep.',
            action_label: 'Review synthetic rows',
            target_tab: 'synthetic',
        },
    ],
    entry_point: {
        label: 'Open Synthetic workflow',
        target_tab: 'synthetic',
        reason: 'Use the existing Synthetic tab and PlaybookPickerPanel.',
    },
};

describe('DataStudioSyntheticPlaybookCenterPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders playbook readiness and routes into the existing Synthetic workflow', async () => {
        apiMock.get.mockResolvedValueOnce({ data: syntheticPayload });
        const onOpenSynthetic = vi.fn();

        render(<DataStudioSyntheticPlaybookCenterPanel projectId={1} onOpenSynthetic={onOpenSynthetic} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-synth-playbooks')).toBeInTheDocument();
        });

        expect(screen.getByText('Needs setup')).toBeInTheDocument();
        expect(screen.getByText('3 / 18')).toBeInTheDocument();
        expect(screen.getAllByText('ollama:llama3').length).toBeGreaterThan(0);
        expect(screen.getByText('Paraphrase positives')).toBeInTheDocument();
        expect(screen.getByText('Gold examples')).toBeInTheDocument();
        expect(screen.getByText('Synthetic rows pending review')).toBeInTheDocument();
        expect(screen.getAllByText(/playbook:classification:positives_paraphrase/i).length).toBeGreaterThan(0);

        fireEvent.click(screen.getByRole('button', { name: /Open Synthetic workflow/i }));
        expect(onOpenSynthetic).toHaveBeenCalledTimes(1);
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/synthetic-playbooks');
    });

    it('renders no-recipe local Ollama setup guidance', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...syntheticPayload,
                verdict: 'attention',
                recipe: null,
                catalog: {
                    ...syntheticPayload.catalog,
                    compatible_playbooks: 0,
                    preview_playbooks: [],
                },
                recommended_backend: {
                    ...syntheticPayload.recommended_backend,
                    available: false,
                    describe: 'ollama',
                },
                review_queue: {
                    ...syntheticPayload.review_queue,
                    total_pending: 0,
                    total_accepted: 0,
                    top_pending_groups: [],
                },
                prerequisites: [
                    {
                        id: 'recipe',
                        label: 'Recipe selected',
                        status: 'missing',
                        message: 'Pick a recipe so BrewSLM can show compatible synthetic playbooks.',
                        target_tab: 'data',
                    },
                    {
                        id: 'local_ollama',
                        label: 'Local Ollama',
                        status: 'attention',
                        message: 'Ollama is the free local default; start Ollama or pull a local model before generating.',
                        target_tab: 'synthetic',
                    },
                ],
                issues: [
                    {
                        id: 'synthetic_recipe_missing',
                        severity: 'blocker',
                        title: 'Recipe not selected',
                        message: 'Synthetic playbooks are recipe-aware.',
                        action_label: 'Choose recipe',
                        target_tab: 'data',
                    },
                ],
            },
        });

        render(<DataStudioSyntheticPlaybookCenterPanel projectId={1} onOpenSynthetic={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText(/No recipe/i)).toBeInTheDocument();
        });
        expect(screen.getByText('Ollama not ready')).toBeInTheDocument();
        expect(screen.getAllByText('Recipe not selected').length).toBeGreaterThan(0);
        expect(screen.getAllByText(/Pick a recipe/i).length).toBeGreaterThan(0);
    });
});
