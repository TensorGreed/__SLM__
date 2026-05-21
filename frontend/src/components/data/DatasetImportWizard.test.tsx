/**
 * Phase F — UI wizard contract.
 *
 * Pins the user-facing flow: source picker → introspect call → mapper
 * pick + field-map edit → confidence gate (low-confidence requires
 * explicit "proceed anyway") → preview with bulk-drop UX → run.
 *
 * The dataset-import API is mocked so the suite is hermetic.
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import DatasetImportWizard from './DatasetImportWizard';

const HIGH_CONFIDENCE_INTROSPECTION = {
    source_id: 'jsonl',
    locator: 'jsonl:/tmp/data.jsonl',
    resolved_path: '/tmp/data.jsonl',
    approximate_total_rows: 100,
    columns: ['text', 'label'],
    sample_rows: [{ text: 'hello world', label: 'pos' }],
    column_signatures: [
        {
            name: 'text',
            column_type: 'text_like',
            confidence: 0.85,
            unique_values: [],
            sample_value: 'hello world',
            notes: '',
        },
        {
            name: 'label',
            column_type: 'categorical',
            confidence: 0.9,
            unique_values: ['pos', 'neg', 'neu'],
            sample_value: 'pos',
            notes: '',
        },
    ],
    hypotheses: [
        {
            mapper_id: 'label_to_classification',
            target_task_profile: 'classification',
            field_map: {
                text_field: 'text',
                label_field: 'label',
                allowed_labels: ['pos', 'neg', 'neu'],
            },
            confidence: 0.92,
            rationale:
                "detected text column 'text' + categorical column 'label' with 3 distinct values (pos, neg, neu)",
            warnings: [],
        },
    ],
    proposal: {
        target_task_profile: 'classification',
        mapper_id: 'label_to_classification',
        field_map: {
            text_field: 'text',
            label_field: 'label',
            allowed_labels: ['pos', 'neg', 'neu'],
        },
        confidence: 0.92,
        rationale: 'detected canonical text + label columns',
        warnings: [],
        needs_force: false,
    },
    confidence_threshold: 0.8,
};

const LOW_CONFIDENCE_INTROSPECTION = {
    ...HIGH_CONFIDENCE_INTROSPECTION,
    hypotheses: [
        {
            ...HIGH_CONFIDENCE_INTROSPECTION.hypotheses[0],
            confidence: 0.65,
        },
    ],
    proposal: {
        ...HIGH_CONFIDENCE_INTROSPECTION.proposal!,
        confidence: 0.65,
        needs_force: true,
    },
};

const PREVIEW_RESULT_WITH_REJECTS = {
    accepted_count: 2,
    rejected_count: 3,
    source_id: 'jsonl',
    mapper_id: 'label_to_classification',
    target_task_profile: 'classification',
    locator: 'jsonl:/tmp/data.jsonl',
    written_path: null,
    dry_run: true,
    rejection_counts: { missing_text: 2, missing_label: 1 },
    warnings: [],
    accepted_sample: [
        {
            payload: { text: 'great service', label: 'pos' },
            row_key: null,
            warnings: [],
        },
        {
            payload: { text: 'too bad', label: 'neg' },
            row_key: null,
            warnings: [],
        },
    ],
    rejected_sample: [
        {
            reason: 'missing_text',
            detail: '',
            row_index: 1,
            raw_row: { label: 'pos' },
        },
        {
            reason: 'missing_label',
            detail: '',
            row_index: 2,
            raw_row: { text: 'hi' },
        },
    ],
};

const RUN_RESULT_SUCCESS = {
    ...PREVIEW_RESULT_WITH_REJECTS,
    dry_run: false,
    written_path: '/var/brewslm/projects/77/synthetic/synthetic.jsonl',
};

const RECIPE_CATALOG_STUB = {
    catalog_version: 'recipes.builtin/v1',
    catalog_source: 'builtin',
    recipe_count: 1,
    recipes: [
        {
            id: 'classification',
            name: 'Text Classifier',
            headline: 'Classify text.',
            description: '',
            icon: '🏷️',
            task_profile: 'classification',
            adapter_id: 'classification-label',
            scoring_mode: 'field_match',
            default_input_column: 'text',
            default_output_column: 'label',
            suggested_base_model: 'SmolLM2-135M',
            alt_base_models: [],
            target_profile: 'mobile_cpu',
            training_plan_profile: 'balanced',
            eval_pack_id: 'evalpack.classification.default',
            gold_template: {
                shape_label: 'text_label',
                min_rows_recommended: 100,
                fields: [],
                example_row: {},
            },
            sample_eval_prompts: [],
            data_acquisition_hints: [],
            shape_signatures: [],
            catalog_source: 'builtin',
            catalog_version: 'builtin-v1',
            is_builtin: true,
        },
    ],
};

const RECIPE_SNIFF_STUB = {
    headers: ['text', 'label'],
    suggestions: [
        {
            recipe_id: 'classification',
            recipe_name: 'Text Classifier',
            icon: '🏷️',
            confidence: 0.9,
            matched_columns: { input: 'text', label: 'label' },
            signature_index: 0,
        },
    ],
    top_recipe_id: 'classification',
};

function defaultApiHandlers() {
    apiMock.put.mockImplementation(async (url: string, body?: unknown) => {
        if (url === '/projects/77/recipe') {
            return {
                data: {
                    id: 77,
                    selected_recipe: {
                        recipe_id: (body as { recipe_id?: string })?.recipe_id ?? '',
                    },
                },
            };
        }
        return { data: {} };
    });
    apiMock.get.mockImplementation(async (url: string) => {
        if (url === '/dataset-import/sources') {
            return { data: { sources: ['jsonl', 'csv', 'hf', 'kaggle'] } };
        }
        if (url === '/dataset-import/mappers') {
            return {
                data: {
                    mappers: ['bio_to_spans', 'label_to_classification', 'text_only'],
                },
            };
        }
        if (url === '/recipes') {
            return { data: RECIPE_CATALOG_STUB };
        }
        return { data: {} };
    });
    apiMock.post.mockImplementation(async (url: string) => {
        if (url === '/dataset-import/introspect') {
            return { data: HIGH_CONFIDENCE_INTROSPECTION };
        }
        if (url === '/recipes/sniff') {
            return { data: RECIPE_SNIFF_STUB };
        }
        if (url === '/projects/77/dataset-import/preview') {
            return { data: PREVIEW_RESULT_WITH_REJECTS };
        }
        if (url === '/projects/77/dataset-import/run') {
            return { data: RUN_RESULT_SUCCESS };
        }
        return { data: {} };
    });
}

/**
 * The dataset-import wizard now lands on a "Recipe" step after introspect.
 * Tests that exercise the legacy map → preview → run flow click through
 * the recipe picker's "Override" link to skip it.
 */
async function skipRecipeStep(user: ReturnType<typeof userEvent.setup>) {
    const override = await screen.findByTestId('recipe-picker-override');
    await user.click(override);
}

describe('DatasetImportWizard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.put.mockReset();
        apiMock.delete.mockReset();
        defaultApiHandlers();
    });

    it('walks the happy path: introspect → map → preview → run', async () => {
        const user = userEvent.setup();
        const onSuccess = vi.fn();
        render(
            <DatasetImportWizard
                projectId={77}
                onClose={() => undefined}
                onSuccess={onSuccess}
            />,
        );

        // Source step is visible.
        expect(await screen.findByText('Import dataset')).toBeInTheDocument();

        // Fill the locator and click Introspect.
        const locatorInput = screen.getByTestId('locator-input');
        await user.type(locatorInput, '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));
        await skipRecipeStep(user);

        // Introspection POST happened with the right locator.
        await waitFor(() =>
            expect(apiMock.post).toHaveBeenCalledWith(
                '/dataset-import/introspect',
                expect.objectContaining({ locator: 'jsonl:/tmp/data.jsonl' }),
            ),
        );

        // Map step shows column signatures + the proposed mapper.
        expect(await screen.findByText('Column signatures')).toBeInTheDocument();
        expect(
            within(screen.getByTestId('mapper-select')).getByRole('option', {
                name: /label_to_classification/i,
            }),
        ).toBeInTheDocument();
        // Field-map editor pre-populates from the proposal.
        const fieldMap = screen.getByTestId('field-map-input') as HTMLTextAreaElement;
        expect(fieldMap.value).toContain('text_field');
        expect(fieldMap.value).toContain('label_field');

        // Proceed to preview.
        await user.click(screen.getByTestId('preview-btn'));
        await waitFor(() =>
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/77/dataset-import/preview',
                expect.objectContaining({
                    locator: 'jsonl:/tmp/data.jsonl',
                    mapper_id: 'label_to_classification',
                }),
            ),
        );

        // Preview step shows accepted + rejected breakdown.
        expect(await screen.findByText('Dry-run summary')).toBeInTheDocument();
        expect(screen.getByText('Rejected rows by reason')).toBeInTheDocument();
        expect(screen.getByTestId('reject-row-missing_text')).toBeInTheDocument();

        // Commit.
        await user.click(screen.getByTestId('run-btn'));
        await waitFor(() =>
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/77/dataset-import/run',
                expect.objectContaining({
                    locator: 'jsonl:/tmp/data.jsonl',
                    mapper_id: 'label_to_classification',
                }),
            ),
        );

        // Success banner.
        expect(await screen.findByTestId('run-success-banner')).toBeInTheDocument();
        expect(onSuccess).toHaveBeenCalledTimes(1);
    });

    it('blocks preview below the confidence threshold until force is checked', async () => {
        apiMock.post.mockImplementation(async (url: string) => {
            if (url === '/dataset-import/introspect') {
                return { data: LOW_CONFIDENCE_INTROSPECTION };
            }
            if (url === '/projects/77/dataset-import/preview') {
                return { data: PREVIEW_RESULT_WITH_REJECTS };
            }
            return { data: {} };
        });

        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        await user.type(screen.getByTestId('locator-input'), '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));
        await skipRecipeStep(user);

        // Low-confidence banner is shown; preview button disabled.
        expect(await screen.findByTestId('confidence-warning')).toBeInTheDocument();
        const previewBtn = screen.getByTestId('preview-btn') as HTMLButtonElement;
        expect(previewBtn.disabled).toBe(true);

        // Tick the force checkbox → preview unlocks.
        await user.click(screen.getByTestId('force-checkbox'));
        expect((screen.getByTestId('preview-btn') as HTMLButtonElement).disabled).toBe(
            false,
        );
    });

    it('forwards drop_reasons to /preview when the user bulk-drops a category', async () => {
        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        await user.type(screen.getByTestId('locator-input'), '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));
        await skipRecipeStep(user);
        await screen.findByText('Column signatures');
        await user.click(screen.getByTestId('preview-btn'));
        await screen.findByText('Dry-run summary');

        // Tick missing_text + refresh.
        await user.click(screen.getByTestId('drop-missing_text'));
        await user.click(screen.getByTestId('refresh-preview-btn'));

        // Last /preview call should include drop_reasons: ["missing_text"].
        await waitFor(() => {
            const previewCalls = apiMock.post.mock.calls.filter(
                ([url]) => url === '/projects/77/dataset-import/preview',
            );
            const lastCall = previewCalls[previewCalls.length - 1];
            expect(lastCall[1].drop_reasons).toEqual(['missing_text']);
        });
    });

    it('surfaces an introspect error and stays on the source step', async () => {
        apiMock.post.mockImplementation(async (url: string) => {
            if (url === '/dataset-import/introspect') {
                throw {
                    response: {
                        data: {
                            detail: "JSONL file not found at '/tmp/nope.jsonl'",
                        },
                    },
                };
            }
            return { data: {} };
        });

        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        await user.type(screen.getByTestId('locator-input'), '/tmp/nope.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));
        // No skipRecipeStep here — introspect fails, recipe step never opens.

        const errBanner = await screen.findByTestId('introspect-error');
        expect(errBanner.textContent).toContain('not found');
        // Map step never opened.
        expect(screen.queryByText('Column signatures')).not.toBeInTheDocument();
    });

    it('switching mapper resets the field-map textarea to the new hypothesis', async () => {
        const multiHypothesis = {
            ...HIGH_CONFIDENCE_INTROSPECTION,
            hypotheses: [
                {
                    mapper_id: 'label_to_classification',
                    target_task_profile: 'classification',
                    field_map: { text_field: 'text', label_field: 'label' },
                    confidence: 0.9,
                    rationale: 'r1',
                    warnings: [],
                },
                {
                    mapper_id: 'text_only',
                    target_task_profile: 'language_modeling',
                    field_map: { text_field: 'text' },
                    confidence: 0.85,
                    rationale: 'r2',
                    warnings: [],
                },
            ],
        };
        apiMock.post.mockImplementation(async (url: string) => {
            if (url === '/dataset-import/introspect') {
                return { data: multiHypothesis };
            }
            return { data: {} };
        });

        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );
        await user.type(screen.getByTestId('locator-input'), '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));
        await skipRecipeStep(user);
        await screen.findByText('Column signatures');

        const fieldMap = screen.getByTestId('field-map-input') as HTMLTextAreaElement;
        expect(fieldMap.value).toContain('label_field');

        // Switch to text_only — field map should change.
        await user.selectOptions(screen.getByTestId('mapper-select'), 'text_only');
        await waitFor(() => {
            expect(
                (screen.getByTestId('field-map-input') as HTMLTextAreaElement).value,
            ).not.toContain('label_field');
        });
        expect(
            (screen.getByTestId('field-map-input') as HTMLTextAreaElement).value,
        ).toContain('text_field');
    });

    it('persists the picked recipe to the project (PUT /projects/77/recipe)', async () => {
        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        await user.type(screen.getByTestId('locator-input'), '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));

        // Pick the top sniffed recipe.
        const pick = await screen.findByTestId('recipe-select-classification');
        await user.click(pick);

        await waitFor(() =>
            expect(apiMock.put).toHaveBeenCalledWith(
                '/projects/77/recipe',
                { recipe_id: 'classification' },
            ),
        );

        // Wizard advances to the Map step with a "Change recipe" chip showing.
        expect(await screen.findByTestId('recipe-summary-chip')).toBeInTheDocument();
        expect(screen.queryByTestId('recipe-persist-error')).not.toBeInTheDocument();
    });

    it('surfaces a non-blocking warning if recipe persistence fails', async () => {
        apiMock.put.mockImplementation(async () => {
            throw new Error('500 backend down');
        });

        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        await user.type(screen.getByTestId('locator-input'), '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));

        const pick = await screen.findByTestId('recipe-select-classification');
        await user.click(pick);

        // We still land on the Map step (best-effort persist), but a
        // banner explains the side-effect didn't take.
        expect(await screen.findByTestId('recipe-persist-error')).toHaveTextContent(
            /500 backend down/,
        );
    });

    it('skips persistence and persists nothing when the user clicks Override', async () => {
        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        await user.type(screen.getByTestId('locator-input'), '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));

        const override = await screen.findByTestId('recipe-picker-override');
        await user.click(override);

        // Map step opened, but PUT /recipe was never called.
        expect(await screen.findByText('Column signatures')).toBeInTheDocument();
        expect(apiMock.put).not.toHaveBeenCalled();
    });

    it('renders an auth note for hf source', async () => {
        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        // Pick the hf source.
        await user.selectOptions(
            screen.getByRole('combobox', { name: /source/i }),
            'hf',
        );

        expect(
            screen.getByText(/Gated datasets require HF_TOKEN/i),
        ).toBeInTheDocument();
    });

    it('renders an auth note for kaggle source', async () => {
        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        await user.selectOptions(
            screen.getByRole('combobox', { name: /source/i }),
            'kaggle',
        );

        expect(
            screen.getByText(/KAGGLE_USERNAME \+ KAGGLE_KEY/i),
        ).toBeInTheDocument();
    });

    it('saves the current mapping as a config and notifies the parent', async () => {
        // Override the default mock so we can verify the save POST body.
        apiMock.post.mockImplementation(async (url: string, body?: unknown) => {
            if (url === '/dataset-import/introspect') {
                return { data: HIGH_CONFIDENCE_INTROSPECTION };
            }
            if (url === '/projects/77/dataset-import/preview') {
                return { data: PREVIEW_RESULT_WITH_REJECTS };
            }
            if (url === '/projects/77/dataset-import/configs') {
                return {
                    data: {
                        id: 99,
                        project_id: 77,
                        name: (body as { name: string }).name,
                        description: null,
                        locator: 'jsonl:/tmp/data.jsonl',
                        mapper_id: 'label_to_classification',
                        field_map: {},
                        drop_reasons: [],
                        limit: null,
                        created_at: null,
                        updated_at: null,
                        last_run_at: null,
                        last_run_accepted: null,
                    },
                };
            }
            return { data: {} };
        });

        const user = userEvent.setup();
        const onConfigSaved = vi.fn();
        render(
            <DatasetImportWizard
                projectId={77}
                onClose={() => undefined}
                onConfigSaved={onConfigSaved}
            />,
        );

        await user.type(screen.getByTestId('locator-input'), '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));
        await skipRecipeStep(user);
        await screen.findByText('Column signatures');
        await user.click(screen.getByTestId('preview-btn'));
        await screen.findByText('Dry-run summary');

        // Toggle the save form, type a name, save.
        await user.click(screen.getByTestId('toggle-save-form-btn'));
        await user.type(
            screen.getByTestId('save-name-input'),
            'weekly-sentiment',
        );
        await user.click(screen.getByTestId('save-config-btn'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/77/dataset-import/configs',
                expect.objectContaining({
                    name: 'weekly-sentiment',
                    locator: 'jsonl:/tmp/data.jsonl',
                    mapper_id: 'label_to_classification',
                }),
            );
        });
        expect(
            await screen.findByTestId('saved-config-confirm'),
        ).toBeInTheDocument();
        expect(onConfigSaved).toHaveBeenCalledTimes(1);
    });

    it('surfaces a 409 duplicate-name error inline without closing the form', async () => {
        apiMock.post.mockImplementation(async (url: string) => {
            if (url === '/dataset-import/introspect') {
                return { data: HIGH_CONFIDENCE_INTROSPECTION };
            }
            if (url === '/projects/77/dataset-import/preview') {
                return { data: PREVIEW_RESULT_WITH_REJECTS };
            }
            if (url === '/projects/77/dataset-import/configs') {
                throw {
                    response: {
                        data: {
                            detail:
                                'A saved mapping with that name already exists in this project.',
                        },
                    },
                };
            }
            return { data: {} };
        });

        const user = userEvent.setup();
        render(
            <DatasetImportWizard projectId={77} onClose={() => undefined} />,
        );

        await user.type(screen.getByTestId('locator-input'), '/tmp/data.jsonl');
        await user.click(screen.getByTestId('introspect-btn'));
        await skipRecipeStep(user);
        await screen.findByText('Column signatures');
        await user.click(screen.getByTestId('preview-btn'));
        await screen.findByText('Dry-run summary');

        await user.click(screen.getByTestId('toggle-save-form-btn'));
        await user.type(screen.getByTestId('save-name-input'), 'taken-name');
        await user.click(screen.getByTestId('save-config-btn'));

        const err = await screen.findByTestId('save-error');
        expect(err.textContent).toContain('already exists');
        // Save form still open + Save button re-enabled for a retry.
        expect(screen.getByTestId('save-config-btn')).toBeInTheDocument();
    });
});
