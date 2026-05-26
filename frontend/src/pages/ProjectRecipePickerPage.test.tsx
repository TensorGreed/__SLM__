import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: {
        success: vi.fn(),
        error: vi.fn(),
        info: vi.fn(),
        warning: vi.fn(),
    },
}));

vi.mock('../api/client', () => ({ default: apiMock }));
vi.mock('../stores/toastStore', () => ({ toast: toastMock }));

// react-router shim — exposes a controllable searchParams and a stub
// outlet context. Each test resets ``contextValue`` and ``searchString``
// before render so cases are independent.
let contextValue: { projectId: number; project: Record<string, unknown> } = {
    projectId: 1,
    project: { id: 1, name: 'Test' },
};
let searchString = '';

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>(
        'react-router-dom',
    );
    return {
        ...actual,
        useOutletContext: () => contextValue,
        useNavigate: () => vi.fn(),
        useSearchParams: () => [new URLSearchParams(searchString), vi.fn()] as const,
    };
});

import ProjectRecipePickerPage from './ProjectRecipePickerPage';


const SAMPLE_CATALOG = {
    catalog_version: 'v1',
    catalog_source: 'builtin',
    recipe_count: 2,
    recipes: [
        {
            id: 'qa-sft',
            name: 'Question & Answer Assistant',
            headline: 'Train a model to answer questions like a domain expert.',
            description: 'For Q&A datasets.',
            icon: '💬',
            task_profile: 'instruction_sft',
            adapter_id: 'qa-pair',
            scoring_mode: 'field_match',
            default_input_column: 'question',
            default_output_column: 'answer',
            suggested_base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
            alt_base_models: [],
            target_profile: 'vllm_server',
            training_plan_profile: 'balanced',
            eval_pack_id: 'evalpack.general.default',
            gold_template: {
                shape_label: 'q_a',
                min_rows_recommended: 50,
                fields: [],
                example_row: {},
            },
            sample_eval_prompts: [],
            data_acquisition_hints: [],
            shape_signatures: [],
            catalog_source: 'builtin',
            catalog_version: 'v1',
            is_builtin: true,
        },
        {
            id: 'classification',
            name: 'Text Classifier',
            headline: 'Train a model to assign each input to a label.',
            description: 'For classification datasets.',
            icon: '🏷️',
            task_profile: 'classification',
            adapter_id: 'label-pair',
            scoring_mode: 'field_match',
            default_input_column: 'text',
            default_output_column: 'label',
            suggested_base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
            alt_base_models: [],
            target_profile: 'vllm_server',
            training_plan_profile: 'balanced',
            eval_pack_id: 'evalpack.general.default',
            gold_template: {
                shape_label: 'text_label',
                min_rows_recommended: 50,
                fields: [],
                example_row: {},
            },
            sample_eval_prompts: [],
            data_acquisition_hints: [],
            shape_signatures: [],
            catalog_source: 'builtin',
            catalog_version: 'v1',
            is_builtin: true,
        },
    ],
};


describe('ProjectRecipePickerPage', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
        toastMock.success.mockReset();
        toastMock.error.mockReset();
        contextValue = { projectId: 1, project: { id: 1, name: 'Test' } };
        searchString = '';
        // Stub the hard navigation so we can assert on it without
        // jsdom yelling about unsupported window.location.assign.
        Object.defineProperty(window, 'location', {
            writable: true,
            value: { ...window.location, assign: vi.fn(), pathname: '/' },
        });
    });

    it('renders the catalog tiles with name + headline + base model', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_CATALOG });
        render(<ProjectRecipePickerPage />);
        await waitFor(() => {
            expect(
                screen.getByTestId('project-recipe-picker-card-qa-sft'),
            ).toBeInTheDocument();
        });
        const card = screen.getByTestId('project-recipe-picker-card-qa-sft');
        expect(card.textContent).toContain('Question & Answer Assistant');
        expect(card.textContent).toContain(
            'Train a model to answer questions like a domain expert.',
        );
        expect(card.textContent).toContain('HuggingFaceTB/SmolLM2-135M-Instruct');
    });

    it('marks the currently-applied recipe and disables its "Use this recipe" button', async () => {
        contextValue = {
            projectId: 1,
            project: {
                id: 1,
                name: 'Test',
                selected_recipe: { recipe_id: 'qa-sft' },
            },
        };
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_CATALOG });
        render(<ProjectRecipePickerPage />);
        await waitFor(() => {
            expect(
                screen.getByTestId('project-recipe-picker-current'),
            ).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('project-recipe-picker-current').textContent,
        ).toContain('qa-sft');
        const applied = screen.getByTestId(
            'project-recipe-picker-apply-qa-sft',
        ) as HTMLButtonElement;
        expect(applied.disabled).toBe(true);
        expect(applied.textContent).toMatch(/Currently applied/);
        // Other recipes stay clickable.
        const other = screen.getByTestId(
            'project-recipe-picker-apply-classification',
        ) as HTMLButtonElement;
        expect(other.disabled).toBe(false);
    });

    it('PUTs to /projects/{id}/recipe on apply + navigates back via return_to', async () => {
        searchString = `?return_to=${encodeURIComponent('/project/1/pipeline/synthetic')}`;
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_CATALOG });
        apiMock.put.mockResolvedValueOnce({ data: { id: 1, selected_recipe: { recipe_id: 'qa-sft' } } });
        render(<ProjectRecipePickerPage />);
        await waitFor(() => {
            expect(
                screen.getByTestId('project-recipe-picker-apply-qa-sft'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('project-recipe-picker-apply-qa-sft'),
        );
        await waitFor(() => {
            expect(apiMock.put).toHaveBeenCalledWith(
                '/projects/1/recipe',
                { recipe_id: 'qa-sft' },
            );
        });
        await waitFor(() => {
            expect(window.location.assign).toHaveBeenCalledWith(
                '/project/1/pipeline/synthetic',
            );
        });
        expect(toastMock.success).toHaveBeenCalledWith(
            expect.stringContaining('Question & Answer Assistant'),
            4000,
        );
    });

    it('ignores absolute / external return_to values (open-redirect guard)', async () => {
        searchString = `?return_to=${encodeURIComponent('https://evil.example.com/x')}`;
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_CATALOG });
        apiMock.put.mockResolvedValueOnce({ data: { id: 1 } });
        render(<ProjectRecipePickerPage />);
        await waitFor(() => {
            expect(
                screen.getByTestId('project-recipe-picker-apply-qa-sft'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('project-recipe-picker-apply-qa-sft'),
        );
        await waitFor(() => {
            expect(window.location.assign).toHaveBeenCalled();
        });
        // External URL was rejected; fell back to the default
        // in-project path.
        expect(window.location.assign).toHaveBeenCalledWith(
            '/project/1/pipeline/data',
        );
    });

    it('toasts on apply error + leaves the button re-clickable', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_CATALOG });
        apiMock.put.mockRejectedValueOnce({
            response: { data: { detail: 'Recipe rejected.' } },
        });
        render(<ProjectRecipePickerPage />);
        await waitFor(() => {
            expect(
                screen.getByTestId('project-recipe-picker-apply-qa-sft'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('project-recipe-picker-apply-qa-sft'),
        );
        await waitFor(() => {
            expect(toastMock.error).toHaveBeenCalledWith('Recipe rejected.');
        });
        // No navigation on failure — user stays on the picker.
        expect(window.location.assign).not.toHaveBeenCalled();
        // Button comes back to its clickable state (not stuck in "Applying…").
        const button = screen.getByTestId(
            'project-recipe-picker-apply-qa-sft',
        ) as HTMLButtonElement;
        expect(button.disabled).toBe(false);
    });

    it('surfaces a catalog-load error inline', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { data: { detail: 'Catalog unavailable' } },
        });
        render(<ProjectRecipePickerPage />);
        await waitFor(() => {
            expect(
                screen.getByTestId('project-recipe-picker-error'),
            ).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('project-recipe-picker-error').textContent,
        ).toMatch(/Catalog unavailable/);
    });
});
