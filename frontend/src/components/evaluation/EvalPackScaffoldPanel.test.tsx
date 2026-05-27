import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        patch: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: {
        success: vi.fn(),
        error: vi.fn(),
        info: vi.fn(),
        warning: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import EvalPackScaffoldPanel from './EvalPackScaffoldPanel';
import type { ScaffoldResponse } from '../../api/evalPackScaffold';


function makeScaffoldResponse(overrides: Partial<ScaffoldResponse> = {}): ScaffoldResponse {
    return {
        project_id: 5,
        recipe_id: 'classification',
        gold_set_summary: { row_count: 200, dataset_types_seen: ['gold_dev', 'gold_test'] },
        draft_pack: {
            pack_id: 'evalpack.project.scaffolded',
            display_name: 'Scaffolded · Classification',
            description: 'Auto-generated for the classification recipe (gold set: 200 rows).',
            version: '1.0.0',
            owner: 'project_scaffold',
            tags: ['scaffolded', 'classification'],
            default_task_profile: 'classification',
            task_specs: [{
                task_profile: 'classification',
                display_name: 'Classification',
                description: '',
                required_metric_ids: ['macro_f1', 'accuracy'],
                gates: [
                    { gate_id: 'min_macro_f1', metric_id: 'macro_f1', operator: 'gte', threshold: 0.65, required: true },
                    { gate_id: 'min_accuracy', metric_id: 'accuracy', operator: 'gte', threshold: 0.70, required: true },
                    { gate_id: 'min_per_class_f1', metric_id: 'min_per_class_f1', operator: 'gte', threshold: 0.50, required: true },
                ],
            }],
            gates: [],
        },
        ...overrides,
    };
}


describe('EvalPackScaffoldPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        toastMock.success.mockReset();
        toastMock.error.mockReset();
    });

    it('renders the draft pack with per-gate threshold + required inputs', async () => {
        apiMock.get.mockResolvedValue({ data: makeScaffoldResponse() });
        render(<EvalPackScaffoldPanel projectId={5} />);
        await waitFor(() => {
            expect(screen.getByTestId('eval-pack-scaffold')).toBeInTheDocument();
        });
        // GET hit the right URL.
        expect(apiMock.get).toHaveBeenCalledWith('/projects/5/evaluation/pack-scaffold');
        // Each gate row carries its own threshold input pre-populated.
        const macroInput = screen.getByTestId('eval-pack-scaffold-gate-min_macro_f1-threshold') as HTMLInputElement;
        expect(macroInput.value).toBe('0.65');
        const accuracyInput = screen.getByTestId('eval-pack-scaffold-gate-min_accuracy-threshold') as HTMLInputElement;
        expect(accuracyInput.value).toBe('0.7');
        // Required checkboxes default to the backend value.
        const macroRequired = screen.getByTestId('eval-pack-scaffold-gate-min_macro_f1-required') as HTMLInputElement;
        expect(macroRequired.checked).toBe(true);
    });

    it('shows a quiet empty state when the project has no recipe (400 recipe_required)', async () => {
        apiMock.get.mockRejectedValue({
            response: { status: 400, data: { detail: 'recipe_required' } },
        });
        render(<EvalPackScaffoldPanel projectId={5} />);
        await waitFor(() => {
            expect(screen.getByTestId('eval-pack-scaffold-empty')).toBeInTheDocument();
        });
        expect(screen.getByTestId('eval-pack-scaffold-empty').textContent)
            .toMatch(/Pick a recipe/);
    });

    it('saves edits via POST and fires onSaved + success toast on success', async () => {
        apiMock.get.mockResolvedValue({ data: makeScaffoldResponse() });
        apiMock.post.mockResolvedValue({
            data: {
                project_id: 5,
                preferred_pack_id: 'evalpack.project.scaffolded',
                scaffolded_pack: makeScaffoldResponse().draft_pack,
            },
        });
        const onSaved = vi.fn();
        render(<EvalPackScaffoldPanel projectId={5} onSaved={onSaved} />);
        await waitFor(() => {
            expect(screen.getByTestId('eval-pack-scaffold')).toBeInTheDocument();
        });

        // Edit the macro_f1 gate threshold from 0.65 to 0.55.
        const input = screen.getByTestId('eval-pack-scaffold-gate-min_macro_f1-threshold');
        await userEvent.clear(input);
        await userEvent.type(input, '0.55');

        await userEvent.click(screen.getByTestId('eval-pack-scaffold-save'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalled();
        });

        // POST body carries the edited threshold.
        const [url, body] = apiMock.post.mock.calls[0];
        expect(url).toBe('/projects/5/evaluation/pack-scaffold');
        const editedGate = body.draft_pack.task_specs[0].gates.find(
            (g: any) => g.gate_id === 'min_macro_f1',
        );
        expect(editedGate.threshold).toBe(0.55);

        // Success toast + onSaved invoked.
        expect(toastMock.success).toHaveBeenCalled();
        expect(onSaved).toHaveBeenCalledWith('evalpack.project.scaffolded');
    });

    it('toggling the Required checkbox flips the gate flag in the POST body', async () => {
        apiMock.get.mockResolvedValue({ data: makeScaffoldResponse() });
        apiMock.post.mockResolvedValue({
            data: {
                project_id: 5,
                preferred_pack_id: 'evalpack.project.scaffolded',
                scaffolded_pack: makeScaffoldResponse().draft_pack,
            },
        });
        render(<EvalPackScaffoldPanel projectId={5} />);
        await waitFor(() => {
            expect(screen.getByTestId('eval-pack-scaffold')).toBeInTheDocument();
        });

        await userEvent.click(screen.getByTestId('eval-pack-scaffold-gate-min_per_class_f1-required'));
        await userEvent.click(screen.getByTestId('eval-pack-scaffold-save'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalled();
        });
        const body = apiMock.post.mock.calls[0][1];
        const flipped = body.draft_pack.task_specs[0].gates.find(
            (g: any) => g.gate_id === 'min_per_class_f1',
        );
        // Started required=true, toggled → now false.
        expect(flipped.required).toBe(false);
    });

    it('discard edits restores the original draft pack', async () => {
        apiMock.get.mockResolvedValue({ data: makeScaffoldResponse() });
        render(<EvalPackScaffoldPanel projectId={5} />);
        await waitFor(() => {
            expect(screen.getByTestId('eval-pack-scaffold')).toBeInTheDocument();
        });
        // Mutate then discard.
        const input = screen.getByTestId('eval-pack-scaffold-gate-min_accuracy-threshold') as HTMLInputElement;
        await userEvent.clear(input);
        await userEvent.type(input, '0.95');
        expect(input.value).toBe('0.95');

        await userEvent.click(screen.getByTestId('eval-pack-scaffold-discard'));
        // After discard, value matches the backend-sourced default.
        const after = screen.getByTestId('eval-pack-scaffold-gate-min_accuracy-threshold') as HTMLInputElement;
        expect(after.value).toBe('0.7');
    });

    it('surfaces an error toast when the save endpoint rejects', async () => {
        apiMock.get.mockResolvedValue({ data: makeScaffoldResponse() });
        apiMock.post.mockRejectedValue({
            response: { status: 400, data: { detail: 'draft_pack_missing_task_specs' } },
        });
        render(<EvalPackScaffoldPanel projectId={5} />);
        await waitFor(() => {
            expect(screen.getByTestId('eval-pack-scaffold')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('eval-pack-scaffold-save'));
        await waitFor(() => {
            expect(toastMock.error).toHaveBeenCalled();
        });
        expect(toastMock.error.mock.calls[0][0]).toMatch(/draft_pack_missing_task_specs/);
    });
});
