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
    domain_libraries: {
        read_only: true,
        local_first: true,
        default_backend: 'ollama',
        ollama_ready: true,
        library_count: 1,
        ready_count: 0,
        attention_count: 1,
        blocked_count: 0,
        detected_domain: {
            id: 'support_faq',
            label: 'Support FAQ',
            confidence: 0.86,
            source: 'sampled_data',
        },
        applied_domain: {
            profile_id: 'generic-domain-v1',
            pack_id: 'generic-pack-v1',
            display_name: 'Generic Domain',
        },
        libraries: [
            {
                id: 'support_faq-detected',
                domain_id: 'support_faq',
                domain_label: 'Support FAQ',
                source: 'detected',
                confidence: 0.86,
                status: 'attention',
                summary: 'Support FAQ library uses local Ollama by default and keeps generated rows behind review.',
                local_first: true,
                active_recipe_id: 'classification',
                active_recipe_label: 'Classification',
                recommended_recipes: ['qa-sft', 'classification'],
                recipe_compatible: true,
                desired_modes: ['hard_negatives', 'positives_paraphrase'],
                compatible_modes: ['positives_paraphrase'],
                missing_modes: ['hard_negatives'],
                review_gates: [
                    'Human review is required before generated rows enter prepared datasets.',
                    'Review account, billing, and cancellation answers for escalation boundaries.',
                ],
                playbooks: [
                    {
                        id: 'support_faq:customer_phrasing',
                        title: 'Generate customer phrasing variants',
                        strategy: 'positive paraphrase',
                        mode: 'positives_paraphrase',
                        mode_label: 'Paraphrase positives',
                        mode_available: true,
                        recipe_id: 'classification',
                        recipe_compatible: true,
                        required_fields: ['text', 'label'],
                        missing_fields: [],
                        expected_output_shape: {
                            format: 'jsonl',
                            recipe_id: 'classification',
                            payload_fields: ['text', 'label', 'synth_source', 'synth_confidence', 'review_status'],
                            review_status: 'pending',
                            notes: [
                                'Rows are generated in the existing Synthetic workflow, not in Data Studio.',
                            ],
                        },
                        prompt_focus: [
                            'Support assistants need to recognize the same intent across messy customer wording.',
                            'Vary customer wording while keeping the answer intent and escalation boundary stable.',
                        ],
                        review_gates: [
                            'Human review is required before generated rows enter prepared datasets.',
                            'Review account, billing, and cancellation answers for escalation boundaries.',
                        ],
                        prerequisites: [
                            {
                                id: 'recipe',
                                label: 'Recipe compatibility',
                                status: 'met',
                                message: 'classification matches this domain library.',
                                target_tab: 'data',
                            },
                            {
                                id: 'playbook_mode',
                                label: 'Playbook mode',
                                status: 'met',
                                message: 'At least one curated playbook mode is registered.',
                                target_tab: 'synthetic',
                            },
                            {
                                id: 'mapping',
                                label: 'Required fields',
                                status: 'met',
                                message: 'Required recipe fields look ready.',
                                target_tab: 'dataprep',
                            },
                            {
                                id: 'gold_examples',
                                label: 'Gold anchors',
                                status: 'met',
                                message: '2 file-backed Gold Set rows can anchor generation.',
                                target_tab: 'goldset',
                            },
                            {
                                id: 'local_ollama',
                                label: 'Local Ollama',
                                status: 'met',
                                message: 'Local Ollama is ready.',
                                target_tab: 'synthetic',
                            },
                            {
                                id: 'review_gate',
                                label: 'Review gate',
                                status: 'attention',
                                message: '2 synthetic rows are already pending review.',
                                target_tab: 'synthetic',
                            },
                        ],
                        readiness: 'attention',
                        readiness_reason: 'The library can be reviewed, but setup or review gates need attention.',
                        generation_path: {
                            backend: 'ollama',
                            available: true,
                            describe: 'ollama:llama3',
                            local_default: true,
                            paid_required: false,
                        },
                        generation_action: {
                            label: 'Open Synthetic workflow',
                            target_tab: 'synthetic',
                            requires_confirmation: true,
                            description: 'Run this library from the existing Synthetic workflow.',
                        },
                    },
                ],
            },
        ],
        entry_point: {
            label: 'Open Synthetic workflow',
            target_tab: 'synthetic',
            reason: 'Run domain-specific playbooks in the existing Synthetic tab.',
            requires_confirmation: true,
            description: 'Run domain-specific playbooks in the existing Synthetic tab.',
        },
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
        expect(screen.getByText('Domain playbook libraries')).toBeInTheDocument();
        expect(screen.getAllByText('Support FAQ').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Generate customer phrasing variants').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Required fields').length).toBeGreaterThan(0);
        expect(screen.getByText('jsonl · text, label, synth_source, synth_confidence, review_status')).toBeInTheDocument();
        expect(screen.getAllByText(/local default/i).length).toBeGreaterThan(0);
        expect(screen.getByText('Gold examples')).toBeInTheDocument();
        expect(screen.getByText('Synthetic rows pending review')).toBeInTheDocument();
        expect(screen.getAllByText(/playbook:classification:positives_paraphrase/i).length).toBeGreaterThan(0);

        fireEvent.click(screen.getAllByRole('button', { name: /Open Synthetic workflow/i })[0]);
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
                domain_libraries: {
                    ...syntheticPayload.domain_libraries,
                    ollama_ready: false,
                    library_count: 1,
                    ready_count: 0,
                    attention_count: 0,
                    blocked_count: 1,
                    detected_domain: {
                        id: 'generic_domain',
                        label: 'Generic Domain',
                        confidence: 0.25,
                        source: 'runtime_default',
                    },
                    libraries: [
                        {
                            ...syntheticPayload.domain_libraries.libraries[0],
                            id: 'generic_domain-fallback',
                            domain_id: 'generic_domain',
                            domain_label: 'Generic Domain',
                            source: 'fallback',
                            confidence: 0.25,
                            status: 'blocked',
                            summary: 'Synthetic rows are safer when the domain and recipe are confirmed first.',
                            active_recipe_id: null,
                            active_recipe_label: 'No recipe',
                            recommended_recipes: [],
                            recipe_compatible: false,
                            compatible_modes: [],
                            playbooks: [
                                {
                                    ...syntheticPayload.domain_libraries.libraries[0].playbooks[0],
                                    id: 'generic_domain:baseline_variants',
                                    readiness: 'blocked',
                                    readiness_reason: 'Recipe, playbook mode, or Gold Set prerequisites need setup first.',
                                    generation_path: {
                                        backend: 'ollama',
                                        available: false,
                                        describe: 'ollama',
                                        local_default: true,
                                        paid_required: false,
                                    },
                                    prerequisites: [
                                        {
                                            id: 'recipe',
                                            label: 'Recipe compatibility',
                                            status: 'missing',
                                            message: 'Choose a recipe before using a domain-specific synthetic library.',
                                            target_tab: 'data',
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
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
            expect(screen.getAllByText(/No recipe/i).length).toBeGreaterThan(0);
        });
        expect(screen.getByText('Ollama not ready')).toBeInTheDocument();
        expect(screen.getAllByText('Recipe not selected').length).toBeGreaterThan(0);
        expect(screen.getAllByText(/Pick a recipe/i).length).toBeGreaterThan(0);
    });

    // ─────────────────────────────────────────────────────────────────
    // Arc C — turn unmet prerequisites into one-click setup
    // affordances. Met items stay as static info rows; the rest become
    // buttons with a specific setup label and route through the
    // optional ``onOpenTarget`` callback.
    // ─────────────────────────────────────────────────────────────────

    const unmetPayload = {
        ...syntheticPayload,
        prerequisites: [
            {
                id: 'recipe',
                label: 'Recipe selected',
                status: 'met',
                message: 'Classification is active.',
                target_tab: 'data',
            },
            {
                id: 'local_ollama',
                label: 'Local Ollama',
                status: 'attention',
                message: 'Ollama is the free local default; start Ollama or pull a local model.',
                target_tab: 'synthetic',
            },
            {
                id: 'mapping',
                label: 'Required fields',
                status: 'missing',
                message: 'Required field "label" has no source mapping.',
                target_tab: 'dataprep',
            },
        ],
    };

    it('renders unmet prerequisites as actionable buttons with specific setup labels', async () => {
        apiMock.get.mockResolvedValueOnce({ data: unmetPayload });
        render(
            <DataStudioSyntheticPlaybookCenterPanel
                projectId={1}
                onOpenSynthetic={vi.fn()}
                onOpenTarget={vi.fn()}
            />,
        );
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-synth-prereq-local_ollama')).toBeInTheDocument();
        });

        // Met prerequisite (recipe) is NOT a button.
        expect(
            screen.queryByTestId('data-studio-synth-prereq-recipe'),
        ).toBeNull();
        // Unmet prereqs render with the curated setup label.
        expect(
            screen.getByTestId('data-studio-synth-prereq-local_ollama'),
        ).toHaveTextContent('Configure Ollama');
        expect(
            screen.getByTestId('data-studio-synth-prereq-mapping'),
        ).toHaveTextContent('Fix mapping');
    });

    it('routes the unmet prerequisite click through onOpenTarget(target_tab)', async () => {
        apiMock.get.mockResolvedValueOnce({ data: unmetPayload });
        const onOpenTarget = vi.fn();
        render(
            <DataStudioSyntheticPlaybookCenterPanel
                projectId={1}
                onOpenSynthetic={vi.fn()}
                onOpenTarget={onOpenTarget}
            />,
        );
        await waitFor(() => {
            expect(screen.getByTestId('data-studio-synth-prereq-mapping')).toBeInTheDocument();
        });

        fireEvent.click(screen.getByTestId('data-studio-synth-prereq-mapping'));
        expect(onOpenTarget).toHaveBeenCalledWith('dataprep');

        fireEvent.click(screen.getByTestId('data-studio-synth-prereq-local_ollama'));
        expect(onOpenTarget).toHaveBeenCalledWith('synthetic');
    });

    it('falls back to static rows when no onOpenTarget is provided (backward compat)', async () => {
        apiMock.get.mockResolvedValueOnce({ data: unmetPayload });
        render(
            <DataStudioSyntheticPlaybookCenterPanel
                projectId={1}
                onOpenSynthetic={vi.fn()}
            />,
        );
        await waitFor(() => {
            expect(screen.getAllByText('Local Ollama').length).toBeGreaterThan(0);
        });
        // No actionable button when the page hasn't wired up the route.
        expect(
            screen.queryByTestId('data-studio-synth-prereq-local_ollama'),
        ).toBeNull();
        expect(
            screen.queryByTestId('data-studio-synth-prereq-mapping'),
        ).toBeNull();
    });
});
