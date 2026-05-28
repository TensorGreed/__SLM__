import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import HyperparameterSweepPanel from './HyperparameterSweepPanel';

const SWEEP = {
    sweep_id: 'sweep123',
    cell_count: 4,
    completed_count: 4,
    best_label: 'r16-lr0.0002',
    best_experiment_id: 42,
    pareto: { quality_key: 'quality_score', cost_key: 'lora_r', optimal_labels: ['r8-lr0.0002', 'r16-lr0.0002'] },
    cells: [
        { label: 'r8-lr0.0002', experiment_id: 1, lora_r: 8, learning_rate: 2e-4, base_model: 'm', status: 'completed', final_train_loss: 2.0, final_eval_loss: null, quality_score: 0.333, pareto_optimal: true, dominated_by: [] },
        { label: 'r8-lr0.0003', experiment_id: 2, lora_r: 8, learning_rate: 3e-4, base_model: 'm', status: 'completed', final_train_loss: 2.4, final_eval_loss: null, quality_score: 0.294, pareto_optimal: false, dominated_by: ['r8-lr0.0002'] },
        { label: 'r16-lr0.0002', experiment_id: 3, lora_r: 16, learning_rate: 2e-4, base_model: 'm', status: 'completed', final_train_loss: 1.5, final_eval_loss: null, quality_score: 0.4, pareto_optimal: true, dominated_by: [] },
        { label: 'r16-lr0.0003', experiment_id: 4, lora_r: 16, learning_rate: 3e-4, base_model: 'm', status: 'completed', final_train_loss: 2.6, final_eval_loss: null, quality_score: 0.277, pareto_optimal: false, dominated_by: ['r8-lr0.0002', 'r16-lr0.0002'] },
    ],
};

describe('HyperparameterSweepPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('launches a sweep with parsed ranks/lrs and renders the Pareto cells', async () => {
        apiMock.post.mockResolvedValueOnce({ data: { sweep_id: 'sweep123', dispatched_cells: 4 } });
        apiMock.get.mockResolvedValue({ data: SWEEP });

        const user = userEvent.setup();
        render(<HyperparameterSweepPanel projectId={5} baseModel="HF/SmolLM2-135M" baseConfig={{ num_epochs: 1 }} />);

        await user.click(screen.getByRole('button', { name: /Run sweep/ }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/5/training/sweeps',
                expect.objectContaining({
                    base_model: 'HF/SmolLM2-135M',
                    lora_r_values: [8, 16],
                    learning_rate_values: [2e-4, 3e-4],
                }),
            );
        });
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith('/projects/5/training/sweeps/sweep123'));

        // Frontier cells are classified from the backend annotation.
        await waitFor(() => {
            expect(screen.getByTestId('hp-point-r16-lr0.0002').getAttribute('class')).toMatch(/is-optimal/);
        });
        expect(screen.getByTestId('hp-point-r8-lr0.0003').getAttribute('class')).toMatch(/is-dominated/);
        expect(screen.getByText('4/4 cells complete')).toBeInTheDocument();
        // Best cell surfaced.
        expect(screen.getAllByText('r16-lr0.0002').length).toBeGreaterThan(0);
    });

    it('shows a validation error when no ranks/lrs are given', async () => {
        const user = userEvent.setup();
        render(<HyperparameterSweepPanel projectId={5} baseModel="HF/SmolLM2-135M" />);
        const ranks = screen.getByPlaceholderText('8, 16, 32');
        await user.clear(ranks);
        await user.click(screen.getByRole('button', { name: /Run sweep/ }));
        expect(await screen.findByText(/at least one LoRA rank/i)).toBeInTheDocument();
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('invokes onOpenExperiment when a cell row is clicked', async () => {
        apiMock.post.mockResolvedValueOnce({ data: { sweep_id: 'sweep123', dispatched_cells: 4 } });
        apiMock.get.mockResolvedValue({ data: SWEEP });
        const onOpen = vi.fn();
        const user = userEvent.setup();
        render(<HyperparameterSweepPanel projectId={5} baseModel="m" onOpenExperiment={onOpen} />);
        await user.click(screen.getByRole('button', { name: /Run sweep/ }));
        await waitFor(() => expect(screen.getByTestId('hp-row-r8-lr0.0003')).toBeInTheDocument());
        await user.click(screen.getByTestId('hp-row-r8-lr0.0003'));
        expect(onOpen).toHaveBeenCalledWith(2);
    });
});
