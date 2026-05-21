import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import RecipePicker from './RecipePicker';
import type { Recipe, SniffResponse } from '../../api/recipes';

function makeRecipe(overrides: Partial<Recipe>): Recipe {
    return {
        id: overrides.id ?? 'recipe-id',
        name: overrides.name ?? 'Recipe',
        headline: overrides.headline ?? 'A headline.',
        description: overrides.description ?? '',
        icon: overrides.icon ?? '🧪',
        task_profile: overrides.task_profile ?? 'instruction_sft',
        adapter_id: overrides.adapter_id ?? 'default-canonical',
        scoring_mode: overrides.scoring_mode ?? 'field_match',
        default_input_column: overrides.default_input_column ?? 'input',
        default_output_column: overrides.default_output_column ?? 'output',
        suggested_base_model: overrides.suggested_base_model ?? 'SmolLM2-135M',
        alt_base_models: overrides.alt_base_models ?? [],
        target_profile: overrides.target_profile ?? 'vllm_server',
        training_plan_profile: overrides.training_plan_profile ?? 'balanced',
        eval_pack_id: overrides.eval_pack_id ?? 'evalpack.general.default',
        gold_template: overrides.gold_template ?? {
            shape_label: 'input_output',
            min_rows_recommended: 50,
            fields: [],
            example_row: {},
        },
        sample_eval_prompts: overrides.sample_eval_prompts ?? [],
        data_acquisition_hints: overrides.data_acquisition_hints ?? [],
        shape_signatures: overrides.shape_signatures ?? [],
        catalog_source: 'builtin',
        catalog_version: 'builtin-v1',
        is_builtin: true,
    };
}

const QA_RECIPE = makeRecipe({
    id: 'qa-sft',
    name: 'Q&A Assistant',
    headline: 'Train a Q&A model.',
    icon: '💬',
    task_profile: 'instruction_sft',
});
const CLASSIFICATION_RECIPE = makeRecipe({
    id: 'classification',
    name: 'Text Classifier',
    headline: 'Classify text.',
    icon: '🏷️',
    task_profile: 'classification',
});
const GENERIC_RECIPE = makeRecipe({
    id: 'generic-sft',
    name: 'Generic Instruction SFT',
    headline: 'Catch-all SFT.',
    icon: '🧰',
});

function buildSniff(): SniffResponse {
    return {
        headers: ['question', 'answer'],
        suggestions: [
            {
                recipe_id: 'qa-sft',
                recipe_name: 'Q&A Assistant',
                icon: '💬',
                confidence: 0.92,
                matched_columns: { input: 'question', output: 'answer' },
                signature_index: 0,
            },
            {
                recipe_id: 'generic-sft',
                recipe_name: 'Generic Instruction SFT',
                icon: '🧰',
                confidence: 0.30,
                matched_columns: {},
                signature_index: 0,
                fallback: true,
            },
        ],
        top_recipe_id: 'qa-sft',
    };
}

describe('RecipePicker', () => {
    it('renders the top sniffed recipe with confidence + why callout', async () => {
        const sniff = vi.fn().mockResolvedValue(buildSniff());
        const listAll = vi.fn().mockResolvedValue([QA_RECIPE, GENERIC_RECIPE]);
        const onSelect = vi.fn();
        const onOverride = vi.fn();

        render(
            <RecipePicker
                headers={['question', 'answer']}
                onSelect={onSelect}
                onOverride={onOverride}
                sniff={sniff}
                listAll={listAll}
            />,
        );

        await waitFor(() => {
            expect(screen.getByTestId('recipe-card-qa-sft')).toBeInTheDocument();
        });

        expect(screen.getByText('Q&A Assistant')).toBeInTheDocument();
        expect(screen.getByText('Train a Q&A model.')).toBeInTheDocument();
        expect(screen.getByTestId('recipe-confidence-qa-sft')).toHaveTextContent('92%');
        expect(screen.getByTestId('recipe-recommended-badge')).toBeInTheDocument();

        const why = screen.getByTestId('recipe-why-qa-sft');
        expect(why).toHaveTextContent(/Why this recipe/i);
        expect(why).toHaveTextContent('question');
        expect(why).toHaveTextContent('answer');
        expect(why).toHaveTextContent(/input/);
        expect(why).toHaveTextContent(/output/);
    });

    it('calls onSelect with the matched recipe + suggestion', async () => {
        const sniff = vi.fn().mockResolvedValue(buildSniff());
        const listAll = vi.fn().mockResolvedValue([QA_RECIPE, GENERIC_RECIPE]);
        const onSelect = vi.fn();
        const onOverride = vi.fn();

        render(
            <RecipePicker
                headers={['question', 'answer']}
                onSelect={onSelect}
                onOverride={onOverride}
                sniff={sniff}
                listAll={listAll}
            />,
        );

        await waitFor(() => {
            expect(screen.getByTestId('recipe-card-qa-sft')).toBeInTheDocument();
        });

        const user = userEvent.setup();
        await user.click(screen.getByTestId('recipe-select-qa-sft'));

        expect(onSelect).toHaveBeenCalledTimes(1);
        const [recipeArg, suggestionArg] = onSelect.mock.calls[0];
        expect(recipeArg.id).toBe('qa-sft');
        expect(suggestionArg.recipe_id).toBe('qa-sft');
        expect(suggestionArg.confidence).toBeCloseTo(0.92);
        expect(onOverride).not.toHaveBeenCalled();
    });

    it('renders the override link and calls onOverride when clicked', async () => {
        const sniff = vi.fn().mockResolvedValue(buildSniff());
        const listAll = vi.fn().mockResolvedValue([QA_RECIPE, GENERIC_RECIPE]);
        const onSelect = vi.fn();
        const onOverride = vi.fn();

        render(
            <RecipePicker
                headers={['question', 'answer']}
                onSelect={onSelect}
                onOverride={onOverride}
                sniff={sniff}
                listAll={listAll}
            />,
        );

        await waitFor(() => {
            expect(screen.getByTestId('recipe-picker-override')).toBeInTheDocument();
        });

        const user = userEvent.setup();
        await user.click(screen.getByTestId('recipe-picker-override'));

        expect(onOverride).toHaveBeenCalledTimes(1);
        expect(onSelect).not.toHaveBeenCalled();
    });

    it('shows "show more" toggle when there are more than 3 suggestions', async () => {
        const sniff = vi.fn().mockResolvedValue({
            headers: ['x'],
            suggestions: [
                {
                    recipe_id: 'qa-sft',
                    recipe_name: 'Q&A Assistant',
                    icon: '💬',
                    confidence: 0.92,
                    matched_columns: { input: 'x' },
                    signature_index: 0,
                },
                {
                    recipe_id: 'classification',
                    recipe_name: 'Text Classifier',
                    icon: '🏷️',
                    confidence: 0.81,
                    matched_columns: { input: 'x' },
                    signature_index: 0,
                },
                {
                    recipe_id: 'r3',
                    recipe_name: 'Three',
                    icon: '🧪',
                    confidence: 0.71,
                    matched_columns: {},
                    signature_index: 0,
                },
                {
                    recipe_id: 'r4',
                    recipe_name: 'Four',
                    icon: '🧪',
                    confidence: 0.61,
                    matched_columns: {},
                    signature_index: 0,
                },
            ],
            top_recipe_id: 'qa-sft',
        });
        const listAll = vi
            .fn()
            .mockResolvedValue([
                QA_RECIPE,
                CLASSIFICATION_RECIPE,
                makeRecipe({ id: 'r3', name: 'Three' }),
                makeRecipe({ id: 'r4', name: 'Four' }),
            ]);

        render(
            <RecipePicker
                headers={['x']}
                onSelect={vi.fn()}
                onOverride={vi.fn()}
                sniff={sniff}
                listAll={listAll}
            />,
        );

        await waitFor(() => {
            expect(screen.getByTestId('recipe-card-qa-sft')).toBeInTheDocument();
        });

        // Only top 3 visible by default.
        expect(screen.queryByTestId('recipe-card-r4')).not.toBeInTheDocument();

        const user = userEvent.setup();
        await user.click(screen.getByTestId('recipe-picker-toggle-all'));

        expect(screen.getByTestId('recipe-card-r4')).toBeInTheDocument();
    });

    it('renders the fallback explanation for the generic-sft floor entry', async () => {
        const sniff = vi.fn().mockResolvedValue({
            headers: ['mystery'],
            suggestions: [
                {
                    recipe_id: 'generic-sft',
                    recipe_name: 'Generic Instruction SFT',
                    icon: '🧰',
                    confidence: 0.30,
                    matched_columns: {},
                    signature_index: 0,
                    fallback: true,
                },
            ],
            top_recipe_id: 'generic-sft',
        });
        const listAll = vi.fn().mockResolvedValue([GENERIC_RECIPE]);

        render(
            <RecipePicker
                headers={['mystery']}
                onSelect={vi.fn()}
                onOverride={vi.fn()}
                sniff={sniff}
                listAll={listAll}
            />,
        );

        await waitFor(() => {
            expect(screen.getByTestId('recipe-card-generic-sft')).toBeInTheDocument();
        });

        const why = screen.getByTestId('recipe-why-generic-sft');
        expect(why).toHaveTextContent(/None of the more specific recipes matched/i);
        expect(screen.getByTestId('recipe-confidence-generic-sft')).toHaveTextContent('fallback');
    });

    it('shows an error message when the sniff call fails', async () => {
        const sniff = vi.fn().mockRejectedValue(new Error('Network down'));
        const listAll = vi.fn().mockResolvedValue([QA_RECIPE]);

        render(
            <RecipePicker
                headers={['question', 'answer']}
                onSelect={vi.fn()}
                onOverride={vi.fn()}
                sniff={sniff}
                listAll={listAll}
            />,
        );

        await waitFor(() => {
            expect(screen.getByRole('alert')).toHaveTextContent('Network down');
        });
    });
});
